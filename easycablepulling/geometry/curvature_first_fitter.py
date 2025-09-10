"""Curvature-First Reconstruction methodology.

Engineer's "clean-up then snap" approach:
1. Detect where polyline is trying to be straight vs curved
2. Fit proper straights/arcs to each region
3. Snap to inventory constraints (6m straights, 11.25°/22.5° bends)
4. Enforce tangent continuity
"""

import math
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple, cast

import numpy as np

from ..core.models import Bend, Primitive, Section, Straight
from ..inventory import DuctInventory
from .base_fitter import BaseFitter


class SegmentType(Enum):
    """Classification of polyline segments."""

    STRAIGHT = "straight"
    ARC_LIKE = "arc_like"
    NOISE = "noise"


@dataclass
class CurvaturePoint:
    """Curvature information at a polyline point."""

    index: int
    position: Tuple[float, float]
    turn_angle_deg: float  # Signed turn angle at this point
    curvature: float  # 1/radius estimate
    radius_estimate: float  # Estimated radius of curvature


@dataclass
class PolylineSegment:
    """A classified segment of the polyline."""

    start_index: int
    end_index: int
    points: List[Tuple[float, float]]
    segment_type: SegmentType
    avg_curvature: float
    length: float


class CurvatureFirstFitter(BaseFitter):
    """Curvature-first reconstruction with inventory snapping."""

    def __init__(
        self,
        duct_type: str = "200mm",
        window_size: int = 5,
        collinearity_threshold_deg: float = 0.7,
        spike_threshold_m: float = 0.3,
        straight_curvature_threshold: float = 0.05,  # 1/radius threshold (more sensitive)
        min_segment_length_m: float = 2.0,
        radius_tolerance_m: float = 0.25,
        angle_snap_tolerance_deg: float = 3.0,
        **kwargs,
    ) -> None:
        """Initialize curvature-first fitter.

        Args:
            duct_type: Type of duct inventory to use
            window_size: Window size for curvature calculation (3-7 vertices)
            collinearity_threshold_deg: Angle change threshold for merging segments
            spike_threshold_m: Minimum segment length to avoid spikes
            straight_curvature_threshold: Curvature below this = straight
            min_segment_length_m: Minimum segment length for processing
            radius_tolerance_m: Tolerance for R=3.9m matching
            angle_snap_tolerance_deg: Tolerance for angle snapping
        """
        super().__init__(**kwargs)
        self.methodology = "1_curvature_first"
        self.inventory = DuctInventory(duct_type)

        # Algorithm parameters
        self.window_size = max(3, min(7, window_size))
        self.collinearity_threshold = math.radians(collinearity_threshold_deg)
        self.spike_threshold = spike_threshold_m
        self.straight_curvature_threshold = straight_curvature_threshold
        self.min_segment_length = min_segment_length_m
        self.radius_tolerance = radius_tolerance_m
        self.angle_snap_tolerance = math.radians(angle_snap_tolerance_deg)

        # Standard bend radius from inventory
        self.standard_radius = self.inventory.get_bend_radius()

    def fit_section(
        self, section: Section, points: Optional[List[Tuple[float, float]]] = None
    ) -> List[Primitive]:
        """Fit section using curvature-first methodology.

        Args:
            section: Section to fit
            points: Optional polyline points (uses section.original_polyline if None)

        Returns:
            List of fitted primitives respecting inventory constraints
        """
        polyline = points or section.original_polyline
        if not polyline or len(polyline) < 3:
            return self._fallback_simple_straight(polyline)

        # Step 1: Densify and tidy polyline
        cleaned_points = self._densify_and_tidy(polyline)

        # Step 2: Calculate curvature profile
        curvature_points = self._calculate_curvature_profile(cleaned_points)

        # Step 3: Segment classification
        segments = self._classify_segments(curvature_points, cleaned_points)

        # Step 4: Fit primitives to segments
        primitives = []
        for segment in segments:
            segment_primitives = self._fit_segment_primitives(segment)
            primitives.extend(segment_primitives)

        # Step 5: Snap to inventory constraints
        snapped_primitives = self._snap_to_inventory(primitives)

        # Step 6: Enforce G¹ continuity
        continuous_primitives = self._enforce_continuity(snapped_primitives)

        return continuous_primitives

    def _densify_and_tidy(
        self, points: List[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        """Clean up polyline by merging collinear segments and removing spikes.

        Args:
            points: Input polyline points

        Returns:
            Cleaned polyline points
        """
        if len(points) <= 2:
            return points

        cleaned = [points[0]]  # Always keep first point

        for i in range(1, len(points) - 1):
            prev_point = cleaned[-1]
            curr_point = points[i]
            next_point = points[i + 1]

            # Calculate vectors
            v1 = (curr_point[0] - prev_point[0], curr_point[1] - prev_point[1])
            v2 = (next_point[0] - curr_point[0], next_point[1] - curr_point[1])

            # Check segment lengths
            len1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
            len2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)

            # Remove spikes (very short segments)
            if len1 < self.spike_threshold or len2 < self.spike_threshold:
                continue

            # Calculate angle between vectors
            if len1 > 0 and len2 > 0:
                dot_product = v1[0] * v2[0] + v1[1] * v2[1]
                cross_product = v1[0] * v2[1] - v1[1] * v2[0]
                angle_change = abs(math.atan2(cross_product, dot_product))

                # Skip nearly collinear points
                if angle_change < self.collinearity_threshold:
                    continue

            cleaned.append(curr_point)

        cleaned.append(points[-1])  # Always keep last point

        return cleaned

    def _calculate_curvature_profile(
        self, points: List[Tuple[float, float]]
    ) -> List[CurvaturePoint]:
        """Calculate curvature at each point using sliding window.

        Args:
            points: Cleaned polyline points

        Returns:
            List of curvature information
        """
        curvature_points = []
        half_window = self.window_size // 2

        for i in range(len(points)):
            # Define window bounds
            start_idx = max(0, i - half_window)
            end_idx = min(len(points), i + half_window + 1)

            if end_idx - start_idx < 3:
                # Not enough points for curvature calculation
                curvature_points.append(
                    CurvaturePoint(
                        index=i,
                        position=points[i],
                        turn_angle_deg=0.0,
                        curvature=0.0,
                        radius_estimate=float("inf"),
                    )
                )
                continue

            # Calculate turn angle at point i
            if i > 0 and i < len(points) - 1:
                prev_point = points[i - 1]
                curr_point = points[i]
                next_point = points[i + 1]

                # Vectors
                v_in = (curr_point[0] - prev_point[0], curr_point[1] - prev_point[1])
                v_out = (next_point[0] - curr_point[0], next_point[1] - curr_point[1])

                # Normalize vectors
                len_in = math.sqrt(v_in[0] ** 2 + v_in[1] ** 2)
                len_out = math.sqrt(v_out[0] ** 2 + v_out[1] ** 2)

                if len_in > 0.01 and len_out > 0.01:
                    v_in_norm = (v_in[0] / len_in, v_in[1] / len_in)
                    v_out_norm = (v_out[0] / len_out, v_out[1] / len_out)

                    # Signed turn angle
                    dot = v_in_norm[0] * v_out_norm[0] + v_in_norm[1] * v_out_norm[1]
                    cross = v_in_norm[0] * v_out_norm[1] - v_in_norm[1] * v_out_norm[0]

                    turn_angle = math.atan2(cross, dot)
                    turn_angle_deg = math.degrees(turn_angle)

                    # Estimate radius from window
                    radius_est = self._estimate_radius_from_window(
                        points[start_idx:end_idx]
                    )

                    curvature = 1.0 / radius_est if radius_est > 0.1 else 0.0

                else:
                    turn_angle_deg = 0.0
                    radius_est = float("inf")
                    curvature = 0.0
            else:
                turn_angle_deg = 0.0
                radius_est = float("inf")
                curvature = 0.0

            curvature_points.append(
                CurvaturePoint(
                    index=i,
                    position=points[i],
                    turn_angle_deg=turn_angle_deg,
                    curvature=curvature,
                    radius_estimate=radius_est,
                )
            )

        return curvature_points

    def _estimate_radius_from_window(
        self, window_points: List[Tuple[float, float]]
    ) -> float:
        """Estimate radius of curvature from a window of points.

        Args:
            window_points: Points in the window

        Returns:
            Estimated radius in meters
        """
        if len(window_points) < 3:
            return float("inf")

        # Use circle fit through three well-spaced points
        n = len(window_points)
        if n >= 3:
            p1 = window_points[0]
            p2 = window_points[n // 2]
            p3 = window_points[-1]

            # Calculate radius from three points using circumcircle
            radius = self._circumcircle_radius(p1, p2, p3)
            return radius if radius > 0.1 else float("inf")

        return float("inf")

    def _circumcircle_radius(
        self, p1: Tuple[float, float], p2: Tuple[float, float], p3: Tuple[float, float]
    ) -> float:
        """Calculate radius of circumcircle through three points.

        Args:
            p1, p2, p3: Three points

        Returns:
            Radius of circumcircle
        """
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3

        # Calculate side lengths
        a = math.sqrt((x2 - x3) ** 2 + (y2 - y3) ** 2)  # Opposite to p1
        b = math.sqrt((x1 - x3) ** 2 + (y1 - y3) ** 2)  # Opposite to p2
        c = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)  # Opposite to p3

        # Calculate area using cross product
        area = abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)) / 2.0

        if area < 1e-10:  # Nearly collinear
            return float("inf")

        # Radius = (abc)/(4*Area)
        radius = (a * b * c) / (4.0 * area)

        return radius

    def _classify_segments(
        self, curvature_points: List[CurvaturePoint], points: List[Tuple[float, float]]
    ) -> List[PolylineSegment]:
        """Classify polyline into straight and arc-like segments with smoothing.

        Args:
            curvature_points: Curvature information
            points: Polyline points

        Returns:
            List of classified segments
        """
        if not curvature_points:
            return []

        # Step 1: Create raw classifications
        raw_classes = []
        for cp in curvature_points:
            if abs(cp.curvature) < self.straight_curvature_threshold:
                raw_classes.append(SegmentType.STRAIGHT)
            else:
                raw_classes.append(SegmentType.ARC_LIKE)

        # Step 2: Apply smoothing with majority vote
        smooth_window = max(5, self.window_size)  # Use at least 5 points for smoothing
        smoothed_classes = self._apply_classification_smoothing(
            raw_classes, smooth_window
        )

        # Step 3: Create segments from smoothed classifications
        segments: List[PolylineSegment] = []
        if not smoothed_classes:
            return segments

        current_type = smoothed_classes[0]
        segment_start = 0

        for i, class_type in enumerate(smoothed_classes[1:], 1):
            if class_type != current_type:
                # Segment boundary - only add if long enough
                if i - segment_start >= 3:  # Minimum 3 points
                    self._add_segment_if_valid(
                        segments,
                        segment_start,
                        i,
                        points,
                        current_type,
                        curvature_points,
                    )
                current_type = class_type
                segment_start = i

        # Add final segment
        if len(smoothed_classes) - segment_start >= 3:
            self._add_segment_if_valid(
                segments,
                segment_start,
                len(smoothed_classes),
                points,
                current_type,
                curvature_points,
            )

        return segments

    def _apply_classification_smoothing(
        self, raw_classes: List[SegmentType], smooth_window: int
    ) -> List[SegmentType]:
        """Apply smoothing to reduce classification noise.

        Args:
            raw_classes: Raw point-by-point classifications
            smooth_window: Window size for majority voting

        Returns:
            Smoothed classifications
        """
        smoothed_classes = []
        half_window = smooth_window // 2

        for i in range(len(raw_classes)):
            start_idx = max(0, i - half_window)
            end_idx = min(len(raw_classes), i + half_window + 1)

            window_classes = raw_classes[start_idx:end_idx]
            arc_count = sum(1 for c in window_classes if c == SegmentType.ARC_LIKE)
            straight_count = len(window_classes) - arc_count

            if arc_count > straight_count:
                smoothed_classes.append(SegmentType.ARC_LIKE)
            else:
                smoothed_classes.append(SegmentType.STRAIGHT)

        return smoothed_classes

    def _add_segment_if_valid(
        self,
        segments: List[PolylineSegment],
        start_idx: int,
        end_idx: int,
        points: List[Tuple[float, float]],
        segment_type: SegmentType,
        curvature_points: List[CurvaturePoint],
    ) -> None:
        """Add a segment if it meets minimum requirements.

        Args:
            segments: List to add segment to
            start_idx: Start index
            end_idx: End index (exclusive)
            points: Polyline points
            segment_type: Type of segment
            curvature_points: Curvature information
        """
        if end_idx <= start_idx + 1:
            return  # Too short

        segment_points = points[start_idx:end_idx]

        # Calculate segment length
        length = sum(
            math.sqrt(
                (segment_points[i + 1][0] - segment_points[i][0]) ** 2
                + (segment_points[i + 1][1] - segment_points[i][1]) ** 2
            )
            for i in range(len(segment_points) - 1)
        )

        if length < self.min_segment_length:
            return  # Too short

        # Calculate average curvature
        segment_curvatures = [
            curvature_points[i].curvature for i in range(start_idx, end_idx)
        ]
        avg_curvature = (
            sum(segment_curvatures) / len(segment_curvatures)
            if segment_curvatures
            else 0.0
        )

        segment = PolylineSegment(
            start_index=start_idx,
            end_index=end_idx,
            points=segment_points,
            segment_type=segment_type,
            avg_curvature=avg_curvature,
            length=length,
        )

        segments.append(segment)

    def _fit_segment_primitives(self, segment: PolylineSegment) -> List[Primitive]:
        """Fit primitives to a classified segment.

        Args:
            segment: Classified polyline segment

        Returns:
            List of fitted primitives
        """
        if segment.segment_type == SegmentType.STRAIGHT:
            return self._fit_straight_segment(segment)
        elif segment.segment_type == SegmentType.ARC_LIKE:
            return self._fit_arc_segment(segment)
        else:
            # Noise - treat as straight
            return self._fit_straight_segment(segment)

    def _fit_straight_segment(self, segment: PolylineSegment) -> List[Primitive]:
        """Fit straight primitives to a straight segment.

        Args:
            segment: Straight segment

        Returns:
            List of straight primitives
        """
        if len(segment.points) < 2:
            return []

        # IMPORTANT: Follow the actual polyline path, don't create best-fit lines
        # that deviate from the route. We need to maintain the trajectory.

        # Create straights that follow the actual polyline segments
        straights = []

        # Group consecutive points into straight runs
        i = 0
        while i < len(segment.points) - 1:
            # Start a new straight run
            start_point = segment.points[i]

            # Find how far we can extend this straight
            j = i + 1
            while j < len(segment.points):
                # Check if we can extend the straight to include point j
                current_end = segment.points[j]

                # Calculate if intermediate points are close to the line
                all_close = True
                if j > i + 1:
                    # Check intermediate points
                    dx = current_end[0] - start_point[0]
                    dy = current_end[1] - start_point[1]
                    line_length = math.sqrt(dx**2 + dy**2)

                    if line_length > 0:
                        line_dir = (dx / line_length, dy / line_length)

                        for k in range(i + 1, j):
                            point = segment.points[k]
                            # Calculate perpendicular distance to line
                            vec_to_point = (
                                point[0] - start_point[0],
                                point[1] - start_point[1],
                            )
                            proj_length = (
                                vec_to_point[0] * line_dir[0]
                                + vec_to_point[1] * line_dir[1]
                            )

                            if 0 <= proj_length <= line_length:
                                perp_x = point[0] - (
                                    start_point[0] + proj_length * line_dir[0]
                                )
                                perp_y = point[1] - (
                                    start_point[1] + proj_length * line_dir[1]
                                )
                                perp_dist = math.sqrt(perp_x**2 + perp_y**2)

                                if perp_dist > 0.5:  # More than 0.5m off the line
                                    all_close = False
                                    break

                if all_close:
                    j += 1
                else:
                    break

            # Create straight from i to j-1
            end_point = segment.points[j - 1]
            dx = end_point[0] - start_point[0]
            dy = end_point[1] - start_point[1]
            length = math.sqrt(dx**2 + dy**2)

            if length > 0.01:  # Skip tiny segments
                straight = Straight(
                    length_m=length,
                    start_point=cast(Tuple[float, float], tuple(start_point)),
                    end_point=cast(Tuple[float, float], tuple(end_point)),
                )
                straights.append(straight)

            i = j - 1  # Continue from the end of this straight

        return straights if straights else []

    def _fit_arc_segment(self, segment: PolylineSegment) -> List[Primitive]:
        """Fit arc primitive to an arc-like segment.

        Args:
            segment: Arc-like segment

        Returns:
            List with single bend primitive
        """
        if len(segment.points) < 3:
            # Fallback to straight
            return self._fit_straight_segment(segment)

        # Fit circle using least squares
        center, radius = self._fit_circle_least_squares(segment.points)

        if radius < 0.5:  # Too tight, treat as straight
            return self._fit_straight_segment(segment)

        # Calculate start and end angles
        start_point = segment.points[0]
        end_point = segment.points[-1]

        start_angle = math.atan2(start_point[1] - center[1], start_point[0] - center[0])
        end_angle = math.atan2(end_point[1] - center[1], end_point[0] - center[0])

        # Calculate total angle (handle wraparound)
        total_angle = end_angle - start_angle
        if total_angle > math.pi:
            total_angle -= 2 * math.pi
        elif total_angle < -math.pi:
            total_angle += 2 * math.pi

        total_angle_deg = math.degrees(total_angle)

        # Create bend primitive
        bend = Bend(
            radius_m=radius,
            angle_deg=total_angle_deg,
            direction="CCW" if total_angle_deg > 0 else "CW",
            center_point=center,
            bend_type="natural",
            control_points=segment.points,
        )

        return [bend]

    def _fit_circle_least_squares(
        self, points: List[Tuple[float, float]]
    ) -> Tuple[Tuple[float, float], float]:
        """Fit circle to points using least squares.

        Args:
            points: Points to fit

        Returns:
            Tuple of (center, radius)
        """
        if len(points) < 3:
            return (0.0, 0.0), 0.0

        # Use algebraic circle fitting (Pratt method)
        points_array = np.array(points)
        x = points_array[:, 0]
        y = points_array[:, 1]

        # Set up system: x² + y² + Dx + Ey + F = 0
        A = np.column_stack([x, y, np.ones(len(points))])
        b = -(x**2 + y**2)

        try:
            # Solve least squares
            params, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            D, E, F = params

            # Convert to center and radius
            center_x = -D / 2
            center_y = -E / 2
            radius = math.sqrt(D**2 + E**2 - 4 * F) / 2

            return (center_x, center_y), radius

        except np.linalg.LinAlgError:
            # Fallback: use circumcircle of first, middle, last points
            n = len(points)
            p1, p2, p3 = points[0], points[n // 2], points[-1]
            radius = self._circumcircle_radius(p1, p2, p3)

            # Approximate center
            center_x = sum(p[0] for p in points) / len(points)
            center_y = sum(p[1] for p in points) / len(points)

            return (center_x, center_y), radius

    def _snap_to_inventory(self, primitives: List[Primitive]) -> List[Primitive]:
        """Snap primitives to inventory constraints.

        Args:
            primitives: Fitted primitives

        Returns:
            Primitives snapped to inventory
        """
        snapped: List[Primitive] = []

        for primitive in primitives:
            if isinstance(primitive, Straight):
                # Split long straights into 6m segments
                snapped.extend(self._split_straight_to_inventory(primitive))

            elif isinstance(primitive, Bend):
                # Snap bend to standard angles and radius
                snapped_bend = self._snap_bend_to_inventory(primitive)
                if snapped_bend:
                    if isinstance(snapped_bend, list):
                        snapped.extend(snapped_bend)
                    else:
                        snapped.append(snapped_bend)

        return snapped

    def _split_straight_to_inventory(self, straight: Straight) -> List[Primitive]:
        """Split a straight into inventory-constrained segments.

        Args:
            straight: Long straight primitive

        Returns:
            List of 6m straights plus remainder
        """
        max_length = self.inventory.spec.straight_length_m
        total_length = straight.length_m

        if total_length <= max_length:
            return [straight]  # No splitting needed

        # Calculate cuts
        cuts = self.inventory.optimize_straight_cuts(total_length)

        if not cuts:
            return [straight]

        # Create straight segments
        straights = []
        current_pos = np.array(straight.start_point)

        # Calculate direction from start to end points
        dx = straight.end_point[0] - straight.start_point[0]
        dy = straight.end_point[1] - straight.start_point[1]
        length = math.sqrt(dx**2 + dy**2)

        if length > 0:
            direction = np.array([dx / length, dy / length])
        else:
            direction = np.array([1.0, 0.0])

        for length, is_full in cuts:
            next_pos = current_pos + direction * length

            segment = Straight(
                length_m=length,
                start_point=cast(Tuple[float, float], tuple(current_pos)),
                end_point=cast(Tuple[float, float], tuple(next_pos)),
            )

            straights.append(segment)
            current_pos = next_pos

        return straights

    def _snap_bend_to_inventory(self, bend: Bend) -> Optional[List[Bend]]:
        """Snap bend to inventory constraints.

        Args:
            bend: Fitted bend primitive

        Returns:
            List of snapped bends or None
        """
        fitted_radius = bend.radius_m
        fitted_angle = bend.angle_deg

        # Check if radius is close to standard radius
        if abs(fitted_radius - self.standard_radius) <= self.radius_tolerance:
            # Radius is good, snap angle
            snapped_angles = self.inventory.decompose_angle(fitted_angle)

            if not snapped_angles:
                return None  # Angle too small

            # Create snapped bends
            snapped_bends = []
            current_start_angle = bend.start_angle_deg

            for angle in snapped_angles:
                snapped_bend = Bend(
                    radius_m=self.standard_radius,
                    angle_deg=angle,
                    direction=bend.direction,
                    center_point=bend.center_point,
                    bend_type="manufactured",
                )
                snapped_bends.append(snapped_bend)
                current_start_angle += angle

            return snapped_bends

        else:
            # Radius doesn't match - create new bend with standard radius
            # Keep same tangent directions but use standard radius
            snapped_angles = self.inventory.decompose_angle(fitted_angle)

            if not snapped_angles:
                return None

            # Approximate new center (simplified)
            snapped_bends = []
            for angle in snapped_angles:
                snapped_bend = Bend(
                    radius_m=self.standard_radius,
                    angle_deg=angle,
                    direction=bend.direction,
                    center_point=bend.center_point,  # Simplified - should recalculate
                    bend_type="manufactured",
                )
                snapped_bends.append(snapped_bend)

            return snapped_bends

    def _enforce_continuity(self, primitives: List[Primitive]) -> List[Primitive]:
        """Enforce G¹ continuity between primitives.

        Args:
            primitives: Snapped primitives

        Returns:
            Primitives with enforced continuity
        """
        # Simplified continuity enforcement
        # Full implementation would adjust positions and lengths
        return primitives

    def _fallback_simple_straight(
        self, points: Optional[List[Tuple[float, float]]]
    ) -> List[Straight]:
        """Fallback for very short polylines.

        Args:
            points: Polyline points

        Returns:
            Simple straight line primitive
        """
        if not points or len(points) < 2:
            return []

        start = points[0]
        end = points[-1]

        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = math.sqrt(dx**2 + dy**2)
        math.degrees(math.atan2(dy, dx))

        straight = Straight(length_m=length, start_point=start, end_point=end)

        # Split to inventory if needed
        return self._split_straight_to_inventory(straight)

    def get_methodology_name(self) -> str:
        """Get human-readable name."""
        return "Curvature-First Reconstruction (Engineer's Clean-up then Snap)"

    def get_methodology_code(self) -> str:
        """Get methodology code."""
        return "1_curvature_first"
