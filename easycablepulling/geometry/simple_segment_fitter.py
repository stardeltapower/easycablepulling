"""Simple segment-based fitter: polyline segments → straights + 3.9m fillets."""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from ..core.models import Bend, Primitive, Route, Section, Straight


@dataclass
class SegmentFitResult:
    """Result of simple segment fitting."""

    original_length: float
    fitted_length: float
    primitives: List[Primitive]
    length_error_percent: float
    straight_count: int
    bend_count: int


class SimpleSegmentFitter:
    """Simple approach: convert polyline segments to straights with 3.9m fillets at junctions."""

    def __init__(self, standard_radius: float = 3.9):
        """Initialize simple segment fitter.

        Args:
            standard_radius: Standard bend radius (3.9m for 200mm duct)
        """
        self.standard_radius = standard_radius

    def _remove_duplicate_vertices(
        self, polyline: List[Tuple[float, float]], tolerance: float = 0.001
    ) -> List[Tuple[float, float]]:
        """Remove duplicate consecutive vertices from polyline."""
        if len(polyline) < 2:
            return polyline

        cleaned = [polyline[0]]  # Always keep first vertex

        for i in range(1, len(polyline)):
            # Check distance to previous kept vertex
            prev = cleaned[-1]
            curr = polyline[i]
            distance = self._distance(prev, curr)

            if distance > tolerance:  # Keep vertex if far enough from previous
                cleaned.append(curr)

        return cleaned

    def fit_polyline_to_primitives(
        self, polyline: List[Tuple[float, float]]
    ) -> SegmentFitResult:
        """Convert polyline to straights + fillets using simple segment approach.

        Each polyline segment becomes a straight, with fillets at junctions.

        Args:
            polyline: List of (x, y) points

        Returns:
            SegmentFitResult with fitted primitives
        """
        if len(polyline) < 2:
            return SegmentFitResult(
                original_length=0.0,
                fitted_length=0.0,
                primitives=[],
                length_error_percent=0.0,
                straight_count=0,
                bend_count=0,
            )

        # Clean polyline by removing duplicate vertices
        original_length = self._polyline_length(polyline)
        polyline = self._remove_duplicate_vertices(polyline)

        # Convert each segment to straight + fillet
        primitives = self._create_segment_primitives(polyline)

        # Calculate results
        fitted_length = sum(p.length() for p in primitives)
        length_error = (
            abs(fitted_length - original_length) / original_length * 100
            if original_length > 0
            else 0
        )

        straight_count = sum(1 for p in primitives if isinstance(p, Straight))
        bend_count = sum(1 for p in primitives if isinstance(p, Bend))

        return SegmentFitResult(
            original_length=original_length,
            fitted_length=fitted_length,
            primitives=primitives,
            length_error_percent=length_error,
            straight_count=straight_count,
            bend_count=bend_count,
        )

    def fit_section_to_primitives(self, section: Section) -> SegmentFitResult:
        """Convert section to primitives using simple segment approach."""
        return self.fit_polyline_to_primitives(section.original_polyline)

    def _polyline_length(self, polyline: List[Tuple[float, float]]) -> float:
        """Calculate total polyline length."""
        return sum(
            self._distance(polyline[i], polyline[i + 1])
            for i in range(len(polyline) - 1)
        )

    def _create_segment_primitives(
        self, polyline: List[Tuple[float, float]]
    ) -> List[Primitive]:
        """Create primitives from polyline segments with fillets at junctions."""
        if len(polyline) < 2:
            return []

        primitives = []

        # Handle simple case: only 2 points = single straight
        if len(polyline) == 2:
            straight = Straight(
                length_m=self._distance(polyline[0], polyline[1]),
                start_point=polyline[0],
                end_point=polyline[1],
            )
            return [straight]

        # Process each segment with fillets
        current_start = polyline[0]

        for i in range(len(polyline) - 2):  # Stop before last segment
            vertex = polyline[i + 1]
            next_point = polyline[i + 2]

            # Calculate angle at this vertex
            incoming_bearing = self._bearing(current_start, vertex)
            outgoing_bearing = self._bearing(vertex, next_point)
            angle_change = self._angle_difference(incoming_bearing, outgoing_bearing)

            # If angle is significant enough for filleting
            if (
                abs(angle_change) >= 0.1
            ):  # 0.1° minimum for fillet (catch all junctions)
                # Calculate fillet geometry
                fillet_data = self._calculate_simple_fillet(
                    current_start,
                    vertex,
                    next_point,
                    incoming_bearing,
                    outgoing_bearing,
                    angle_change,
                )

                if fillet_data:
                    # Create straight to fillet start
                    straight_length = self._distance(
                        current_start, fillet_data["tangent_start"]
                    )
                    if straight_length > 0.1:  # Minimum 10cm straight
                        straight = Straight(
                            length_m=straight_length,
                            start_point=current_start,
                            end_point=fillet_data["tangent_start"],
                        )
                        primitives.append(straight)

                    # Create fillet bend
                    bend_direction = "CCW" if angle_change > 0 else "CW"
                    bend = Bend(
                        radius_m=self.standard_radius,
                        angle_deg=abs(angle_change),
                        direction=bend_direction,
                        center_point=fillet_data["center"],
                        start_angle_deg=fillet_data["start_angle"],
                        end_angle_deg=fillet_data["end_angle"],
                        bend_type="manufactured",
                    )
                    primitives.append(bend)

                    # Update start for next segment
                    current_start = fillet_data["tangent_end"]
                else:
                    # Fallback: straight to vertex
                    straight_length = self._distance(current_start, vertex)
                    if straight_length > 0.1:
                        straight = Straight(
                            length_m=straight_length,
                            start_point=current_start,
                            end_point=vertex,
                        )
                        primitives.append(straight)
                    current_start = vertex
            else:
                # No significant angle change - straight to vertex
                straight_length = self._distance(current_start, vertex)
                if straight_length > 0.1:
                    straight = Straight(
                        length_m=straight_length,
                        start_point=current_start,
                        end_point=vertex,
                    )
                    primitives.append(straight)
                current_start = vertex

        # Final straight segment
        final_length = self._distance(current_start, polyline[-1])
        if final_length > 0.1:
            final_straight = Straight(
                length_m=final_length, start_point=current_start, end_point=polyline[-1]
            )
            primitives.append(final_straight)

        return primitives

    def _calculate_simple_fillet(
        self,
        start_point: Tuple[float, float],
        vertex_point: Tuple[float, float],
        end_point: Tuple[float, float],
        incoming_bearing: float,
        outgoing_bearing: float,
        angle_change: float,
    ) -> Optional[dict]:
        """Calculate fillet geometry using parallel guide method."""

        if abs(angle_change) < 0.1:  # Skip very small angles
            return None

        # Convert bearings to unit vectors
        incoming_rad = math.radians(incoming_bearing)
        outgoing_rad = math.radians(outgoing_bearing)

        incoming_unit = np.array([math.cos(incoming_rad), math.sin(incoming_rad)])
        outgoing_unit = np.array([math.cos(outgoing_rad), math.sin(outgoing_rad)])

        # Determine turn direction using cross product
        cross_product = (
            incoming_unit[0] * outgoing_unit[1] - incoming_unit[1] * outgoing_unit[0]
        )
        is_left_turn = cross_product > 0

        # Create perpendicular vectors (pointing toward the acute side)
        if is_left_turn:
            # Left turn: perpendiculars point left (CCW rotation)
            perp_incoming = np.array([-incoming_unit[1], incoming_unit[0]])
            perp_outgoing = np.array([-outgoing_unit[1], outgoing_unit[0]])
        else:
            # Right turn: perpendiculars point right (CW rotation)
            perp_incoming = np.array([incoming_unit[1], -incoming_unit[0]])
            perp_outgoing = np.array([outgoing_unit[1], -outgoing_unit[0]])

        # Create parallel guides at distance = standard_radius from each line
        # Incoming line parallel guide: any point on incoming line + radius * perpendicular
        # Outgoing line parallel guide: any point on outgoing line + radius * perpendicular

        vertex = np.array(vertex_point)

        # Points on the parallel guides
        parallel_incoming_point = vertex + self.standard_radius * perp_incoming
        parallel_outgoing_point = vertex + self.standard_radius * perp_outgoing

        # Find intersection of parallel guides
        # Incoming parallel guide: parallel_incoming_point + t * incoming_unit
        # Outgoing parallel guide: parallel_outgoing_point + s * outgoing_unit
        # Solve: parallel_incoming_point + t * incoming_unit = parallel_outgoing_point + s * outgoing_unit

        # Rearrange: t * incoming_unit - s * outgoing_unit = parallel_outgoing_point - parallel_incoming_point
        rhs = parallel_outgoing_point - parallel_incoming_point

        # Solve 2x2 system: [incoming_unit, -outgoing_unit] * [t, s] = rhs
        A = np.column_stack([incoming_unit, -outgoing_unit])

        try:
            params = np.linalg.solve(A, rhs)
            t = params[0]

            # Calculate bend center (intersection of parallel guides)
            center = parallel_incoming_point + t * incoming_unit

        except np.linalg.LinAlgError:
            # Lines are parallel - no intersection
            return None

        # Calculate tangent points: drop perpendiculars from center to original lines
        # Tangent on incoming line
        to_vertex_incoming = vertex - np.array(start_point)
        incoming_line_dir = incoming_unit

        # Project center onto incoming line
        start_to_center = center - np.array(start_point)
        projection_length = np.dot(start_to_center, incoming_line_dir)
        tangent_start = np.array(start_point) + projection_length * incoming_line_dir

        # Tangent on outgoing line
        start_to_center_out = center - vertex
        projection_length_out = np.dot(start_to_center_out, outgoing_unit)
        tangent_end = vertex + projection_length_out * outgoing_unit

        # Verify the center is correct distance from tangent points
        dist_start = np.linalg.norm(center - tangent_start)
        dist_end = np.linalg.norm(center - tangent_end)

        # Check if geometry is reasonable
        if (
            abs(dist_start - self.standard_radius) > 0.1
            or abs(dist_end - self.standard_radius) > 0.1
        ):
            return None

        # Calculate arc angles
        start_vector = tangent_start - center
        end_vector = tangent_end - center

        start_angle = math.degrees(math.atan2(start_vector[1], start_vector[0]))
        end_angle = math.degrees(math.atan2(end_vector[1], end_vector[0]))

        return {
            "tangent_start": tuple(tangent_start),
            "tangent_end": tuple(tangent_end),
            "center": tuple(center),
            "start_angle": start_angle,
            "end_angle": end_angle,
        }

    def _distance(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """Calculate distance between two points."""
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        return math.sqrt(dx**2 + dy**2)

    def _bearing(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """Calculate bearing from p1 to p2 in degrees."""
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        bearing_rad = math.atan2(dy, dx)
        return math.degrees(bearing_rad)

    def _angle_difference(self, angle1: float, angle2: float) -> float:
        """Calculate difference between two angles, handling wrap-around."""
        diff = angle2 - angle1

        # Normalize to [-180, 180]
        while diff > 180:
            diff -= 360
        while diff < -180:
            diff += 360

        return diff
