"""Geometry fitting algorithms for cable routes."""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
from scipy.optimize import least_squares

from ..config import GEOMETRY_TOLERANCES, STANDARD_DUCT_BENDS
from ..core.models import Bend, DuctSpec, PolynomialCurve, Primitive, Straight


@dataclass
class FittingResult:
    """Result of geometry fitting operation."""

    primitives: List[Primitive]
    total_error: float
    max_error: float
    fitted_points: List[Tuple[float, float]]
    success: bool
    message: str = ""


class GeometryFitter:
    """Fit polylines to straights and arcs with diameter-based classification."""

    def __init__(
        self,
        straight_tolerance: Optional[float] = None,
        arc_tolerance: Optional[float] = None,
        min_straight_length: Optional[float] = None,
        min_arc_angle: Optional[float] = None,
        duct_spec: Optional[DuctSpec] = None,
    ):
        """Initialize geometry fitter.

        Args:
            straight_tolerance: Max deviation for straight line fit (meters)
            arc_tolerance: Max deviation for arc fit (meters)
            min_straight_length: Minimum length for straight segment (meters)
            min_arc_angle: Minimum angle for arc segment (degrees)
            duct_spec: Duct specification for diameter-based bend classification
        """
        self.straight_tolerance = (
            straight_tolerance or GEOMETRY_TOLERANCES["straight_tolerance"]
        )
        self.arc_tolerance = arc_tolerance or GEOMETRY_TOLERANCES["arc_tolerance"]
        self.min_straight_length = (
            min_straight_length or GEOMETRY_TOLERANCES["min_straight_length"]
        )
        self.min_arc_angle = min_arc_angle or GEOMETRY_TOLERANCES["min_arc_angle"]
        self.duct_spec = duct_spec

        # Calculate diameter-based bend radius threshold
        if duct_spec:
            # For HDPE: 20-25× diameter, use 22× as middle ground
            self.natural_bend_threshold = (
                duct_spec.inner_diameter / 1000
            ) * 22  # Convert mm to m
        else:
            # Default assumption for 125mm duct
            self.natural_bend_threshold = 0.125 * 22  # 2.75m

    def fit_polyline(self, points: List[Tuple[float, float]]) -> FittingResult:
        """Fit a polyline to straight lines and arcs.

        Preserves existing acceptable geometry (curves within bend radius limits,
        straight segments) rather than re-fitting everything.

        Args:
            points: List of (x, y) coordinate tuples

        Returns:
            FittingResult with fitted primitives
        """
        if len(points) < 2:
            return FittingResult(
                primitives=[],
                total_error=0.0,
                max_error=0.0,
                fitted_points=points,
                success=False,
                message="Insufficient points for fitting",
            )

        # First check if the entire polyline can be preserved as-is
        # Force splitting at standard duct lengths for construction practicality
        max_single_primitive_length = (
            0.0  # Force all sections to split for better path following
        )
        actual_length = sum(
            math.sqrt(
                (points[i + 1][0] - points[i][0]) ** 2
                + (points[i + 1][1] - points[i][1]) ** 2
            )
            for i in range(len(points) - 1)
        )

        if actual_length <= max_single_primitive_length:
            preserved_primitive = self._try_preserve_original(points)
            if preserved_primitive:
                primitives = [preserved_primitive]
            else:
                # Use recursive splitting approach
                primitives = self._recursive_fit(points, 0, len(points) - 1)
        else:
            # Use recursive splitting approach for longer sections
            primitives = self._recursive_fit(points, 0, len(points) - 1)

        # Fallback strategy if poor length preservation (indicates shortcuts)
        if primitives:
            # Calculate length preservation
            actual_length = sum(
                math.sqrt(
                    (points[i + 1][0] - points[i][0]) ** 2
                    + (points[i + 1][1] - points[i][1]) ** 2
                )
                for i in range(len(points) - 1)
            )
            fitted_length = sum(p.length() for p in primitives)
            length_preservation = (
                fitted_length / actual_length if actual_length > 0 else 1.0
            )

            # Use fallback if length preservation < 90% (indicates shortcuts)
            poor_preservation = length_preservation < 0.9
            few_primitives = len(points) > 10 and len(primitives) < 3

            if not primitives or poor_preservation or few_primitives:
                fallback_primitives = self._fallback_fit(points)
                if fallback_primitives:
                    # Check if fallback preserves length better
                    fallback_length = sum(p.length() for p in fallback_primitives)
                    fallback_preservation = fallback_length / actual_length

                    # Use fallback if no primitives exist OR fallback is better
                    if not primitives or fallback_preservation > length_preservation:
                        primitives = fallback_primitives
        else:
            # If no primitives at all, try fallback
            fallback_primitives = self._fallback_fit(points)
            if fallback_primitives:
                primitives = fallback_primitives

        # Post-process: improve areas with poor path following
        if primitives:
            primitives = self._improve_poor_path_following(primitives, points)

        # Post-process: rejoin consecutive straights with same trajectory
        if primitives:
            primitives = self._rejoin_aligned_straights(primitives)

        # Calculate fitting errors
        fitted_points = self._generate_fitted_points(primitives)
        total_error, max_error = self._calculate_fitting_error(points, fitted_points)

        return FittingResult(
            primitives=primitives,
            total_error=total_error,
            max_error=max_error,
            fitted_points=fitted_points,
            success=True,
            message=f"Fitted {len(primitives)} primitives",
        )

    def _recursive_fit(
        self,
        points: List[Tuple[float, float]],
        start_idx: int,
        end_idx: int,
        depth: int = 0,
    ) -> List[Primitive]:
        """Recursively fit segments of the polyline."""
        # Prevent infinite recursion
        if depth > 50 or end_idx - start_idx < 1:
            return []

        # Minimum segment length check
        if end_idx - start_idx < 2:
            segment_points = points[start_idx : end_idx + 1]
            if len(segment_points) >= 2:
                # Create a simple straight line
                p1, p2 = segment_points[0], segment_points[-1]
                length = math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
                if length >= self.min_straight_length:
                    return [Straight(length_m=length, start_point=p1, end_point=p2)]
            return []

        segment_points = points[start_idx : end_idx + 1]

        # PRIORITIZE POLYNOMIAL FITTING for accurate path following
        # Try polynomial fitting first - it follows CAD routes exactly
        if len(segment_points) >= 4:  # Need at least 4 points for cubic polynomial
            poly_fit = self._fit_polynomial(segment_points, degree=3)
            if poly_fit:
                # Check if polynomial stays within curvature limits
                # Allow tighter curves for polynomials
                if poly_fit["min_radius"] >= self.natural_bend_threshold * 0.5:
                    # Check fitting accuracy with adaptive tolerance
                    segment_length = sum(
                        math.sqrt(
                            (segment_points[i + 1][0] - segment_points[i][0]) ** 2
                            + (segment_points[i + 1][1] - segment_points[i][1]) ** 2
                        )
                        for i in range(len(segment_points) - 1)
                    )
                    # Tighter tolerance: 0.3m to 1.0m based on segment length
                    adaptive_tolerance = min(1.0, max(0.3, segment_length * 0.01))

                    if poly_fit["max_error"] <= adaptive_tolerance:
                        # Check if this is actually a gentle curve that should be preserved as a bend
                        # If arc fitting shows this is a curve, preserve it as such
                        arc_fit = self._fit_arc(segment_points)
                        if (
                            arc_fit
                            and abs(arc_fit["angle"]) >= self.min_arc_angle
                            and arc_fit["radius"] >= self.natural_bend_threshold
                        ):
                            # This is a gentle curve - preserve as single bend using
                            # polynomial accuracy
                            return [
                                Bend(
                                    radius_m=arc_fit["radius"],
                                    angle_deg=arc_fit["angle"],
                                    direction="CW" if arc_fit["angle"] > 0 else "CCW",
                                    center_point=arc_fit["center"],
                                    bend_type="natural",
                                    control_points=segment_points,
                                    start_angle_deg=arc_fit.get("start_angle_deg", 0.0),
                                    end_angle_deg=arc_fit.get("end_angle_deg", 0.0),
                                )
                            ]
                        else:
                            # For complex curves, preserve as polynomial curve for
                            # accurate path following
                            return [
                                PolynomialCurve(
                                    coefficients_x=poly_fit["coefficients_x"],
                                    coefficients_y=poly_fit["coefficients_y"],
                                    t_start=poly_fit["t_start"],
                                    t_end=poly_fit["t_end"],
                                    control_points=poly_fit["control_points"],
                                    curve_length=poly_fit["curve_length"],
                                    min_radius=poly_fit["min_radius"],
                                )
                            ]

        # Fallback to traditional arc/straight fitting
        # Try fitting an arc first to preserve curves
        # Use adaptive tolerance: strict for tight curves, relaxed for natural
        # large-radius curves
        arc_fit = self._fit_arc(segment_points)
        if arc_fit:
            # Adaptive tolerance based on radius - larger radius curves get more
            # tolerance
            if arc_fit["radius"] >= self.natural_bend_threshold:
                # Natural curves: use regular tolerance (up to 1m for large sweeping
                # curves)
                arc_tolerance = self.arc_tolerance
            else:
                # Tight curves: use strict 300mm tolerance for CAD accuracy
                arc_tolerance = 0.3

            if arc_fit["max_error"] <= arc_tolerance:
                # Check if arc angle is large enough
                if abs(arc_fit["angle"]) >= self.min_arc_angle:
                    # If arc radius is already acceptable (>= natural threshold),
                    # preserve exactly
                    if arc_fit["radius"] >= self.natural_bend_threshold:
                        # Natural sweeping curve - preserve fitted radius exactly
                        return [
                            Bend(
                                radius_m=arc_fit["radius"],
                                angle_deg=arc_fit["angle"],
                                direction="CW" if arc_fit["angle"] > 0 else "CCW",
                                center_point=arc_fit["center"],
                                bend_type="natural",
                                control_points=segment_points,
                                start_angle_deg=arc_fit.get("start_angle_deg", 0.0),
                                end_angle_deg=arc_fit.get("end_angle_deg", 0.0),
                            )
                        ]
                else:
                    # Only snap to manufactured if significantly different from natural
                    # minimum
                    min_natural_radius_mm = (
                        self.natural_bend_threshold * 1000
                    )  # Convert to mm

                    # If close to natural minimum (within 20%), keep as natural
                    if arc_fit["radius"] * 1000 >= min_natural_radius_mm * 0.8:
                        return [
                            Bend(
                                radius_m=arc_fit["radius"],
                                angle_deg=arc_fit["angle"],
                                direction="CW" if arc_fit["angle"] > 0 else "CCW",
                                center_point=arc_fit["center"],
                                bend_type="natural",
                            )
                        ]
                    else:
                        # True manufactured bend - snap to standard radius
                        standard_radius = self._find_closest_standard_bend(
                            arc_fit["radius"], abs(arc_fit["angle"])
                        )
                        return [
                            Bend(
                                radius_m=standard_radius / 1000,  # Convert mm to m
                                angle_deg=arc_fit["angle"],
                                direction="CW" if arc_fit["angle"] > 0 else "CCW",
                                center_point=arc_fit["center"],
                                bend_type="manufactured",
                            )
                        ]

        # Try fitting a straight line only if arc doesn't work
        # For pavement routes: use strict tolerance (300mm)
        pavement_straight_tolerance = min(0.3, self.straight_tolerance)

        line_fit = self._fit_straight_line(segment_points)
        if line_fit and line_fit["max_error"] <= pavement_straight_tolerance:
            # Check if line is long enough
            if line_fit["length"] >= self.min_straight_length:
                return [
                    Straight(
                        length_m=line_fit["length"],
                        start_point=segment_points[0],
                        end_point=segment_points[-1],
                    )
                ]

        # If neither fits well, prefer splitting at arc error point to preserve curves
        if arc_fit and line_fit:
            # Prefer arc error point for splitting to maintain curve potential
            split_idx = start_idx + arc_fit["max_error_idx"]
        elif arc_fit:
            split_idx = start_idx + arc_fit["max_error_idx"]
        elif line_fit:
            split_idx = start_idx + line_fit["max_error_idx"]
        else:
            # For snaking routes, split at 1/3 point to create more arc-friendly
            # segments
            split_idx = start_idx + (end_idx - start_idx) // 3

        # Ensure split_idx creates meaningful segments to avoid infinite recursion
        # Split index must be: start_idx < split_idx < end_idx
        min_split = start_idx + 1  # At least one point in left segment
        max_split = end_idx - 1  # At least one point in right segment
        split_idx = max(min_split, min(split_idx, max_split))

        # Additional safety: if still invalid, force middle split
        if split_idx <= start_idx or split_idx >= end_idx:
            split_idx = start_idx + (end_idx - start_idx) // 2

        # Recursively fit sub-segments
        left_primitives = self._recursive_fit(points, start_idx, split_idx, depth + 1)
        right_primitives = self._recursive_fit(points, split_idx, end_idx, depth + 1)

        return left_primitives + right_primitives

    def _fit_straight_line(self, points: List[Tuple[float, float]]) -> Optional[dict]:
        """Fit points to a straight line."""
        if len(points) < 2:
            return None

        # Start and end points define the line
        p1 = np.array(points[0])
        p2 = np.array(points[-1])

        # Calculate line parameters
        line_vec = p2 - p1
        line_length = np.linalg.norm(line_vec)

        if line_length < 1e-6:
            return None

        line_unit = line_vec / line_length

        # Calculate distances from all points to the line
        max_error = 0.0
        max_error_idx = 0

        for i, point in enumerate(points[1:-1], 1):
            p = np.array(point)
            # Project point onto line
            t = np.dot(p - p1, line_unit)
            t = np.clip(t, 0, line_length)
            closest = p1 + t * line_unit

            error = np.linalg.norm(p - closest)
            if error > max_error:
                max_error = error
                max_error_idx = i

        return {
            "length": line_length,
            "max_error": max_error,
            "max_error_idx": max_error_idx,
        }

    def _fit_arc(self, points: List[Tuple[float, float]]) -> Optional[dict]:
        """Fit points to a circular arc."""
        if len(points) < 3:
            return None

        # Use least squares circle fitting
        center, radius = self._fit_circle_least_squares(points)

        if radius < 1e-6 or radius > 10000:  # Reject unreasonable radii
            return None

        # Calculate arc angle
        p1 = np.array(points[0])
        p2 = np.array(points[-1])

        v1 = p1 - center
        v2 = p2 - center

        angle1 = math.atan2(v1[1], v1[0])
        angle2 = math.atan2(v2[1], v2[0])

        # Calculate arc angle considering direction
        angle_diff = angle2 - angle1

        # Normalize to [-pi, pi]
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi

        angle_deg = math.degrees(angle_diff)

        # Calculate fitting errors
        max_error = 0.0
        max_error_idx = 0

        for i, point in enumerate(points):
            p = np.array(point)
            error = abs(np.linalg.norm(p - center) - radius)

            if error > max_error:
                max_error = error
                max_error_idx = i

        return {
            "center": tuple(center),
            "radius": radius,
            "angle": angle_deg,
            "max_error": max_error,
            "max_error_idx": max_error_idx,
        }

    def _fit_circle_least_squares(
        self, points: List[Tuple[float, float]]
    ) -> Tuple[np.ndarray, float]:
        """Fit a circle to points using least squares."""
        # Convert to numpy array
        pts = np.array(points)

        # Initial guess: center at centroid
        x_mean = np.mean(pts[:, 0])
        y_mean = np.mean(pts[:, 1])

        # Function to minimize: sum of squared distances from points to circle
        def residuals(params: np.ndarray) -> np.ndarray:
            cx, cy, r = params
            distances = np.sqrt((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2)
            return distances - r

        # Initial radius guess
        r_init = np.mean(np.sqrt((pts[:, 0] - x_mean) ** 2 + (pts[:, 1] - y_mean) ** 2))

        # Optimize
        result = least_squares(residuals, [x_mean, y_mean, r_init])

        if result.success:
            cx, cy, r = result.x
            return np.array([cx, cy]), float(abs(r))
        else:
            # Fallback to simple method
            return np.array([x_mean, y_mean]), float(r_init)

    def _fit_polynomial(
        self, points: List[Tuple[float, float]], degree: int = 3
    ) -> Optional[dict]:
        """Fit points to a polynomial curve with curvature validation.

        Args:
            points: List of (x, y) coordinate tuples
            degree: Polynomial degree (default 3 for cubic)

        Returns:
            Dict with polynomial info or None if fitting fails
        """
        if len(points) < degree + 1:
            return None

        # Convert to numpy
        pts = np.array(points)

        # Create parameter values (cumulative distance along polyline)
        t_values = [0.0]
        for i in range(1, len(points)):
            dist = np.linalg.norm(pts[i] - pts[i - 1])
            t_values.append(t_values[-1] + dist)

        # Normalize t to [0, 1]
        t_max = t_values[-1]
        if t_max < 1e-6:
            return None

        t_normalized = np.array(t_values) / t_max

        # Fit polynomials for x(t) and y(t)
        try:
            coeffs_x = np.polyfit(t_normalized, pts[:, 0], degree)
            coeffs_y = np.polyfit(t_normalized, pts[:, 1], degree)
        except np.linalg.LinAlgError:
            return None

        # Calculate fitting error
        max_error = 0.0
        max_error_idx = 0
        curve_length = 0.0
        min_radius = float("inf")

        # Sample curve densely to check curvature and calculate length
        n_samples = max(50, len(points) * 10)
        prev_point = None

        for i in range(n_samples + 1):
            t = i / n_samples

            # Evaluate polynomial
            x = sum(coef * (t**j) for j, coef in enumerate(reversed(coeffs_x)))
            y = sum(coef * (t**j) for j, coef in enumerate(reversed(coeffs_y)))
            point = np.array([x, y])

            # Calculate length increment
            if prev_point is not None:
                curve_length += np.linalg.norm(point - prev_point)
            prev_point = point

            # Check curvature (skip endpoints where derivatives might be unstable)
            if 0.05 <= t <= 0.95:
                radius = self._evaluate_polynomial_curvature_at_t(
                    coeffs_x, coeffs_y, t, t_max
                )
                min_radius = min(min_radius, radius)

        # Calculate fitting error against original points
        for i, orig_point in enumerate(points):
            t = t_normalized[i]
            x = sum(coef * (t**j) for j, coef in enumerate(reversed(coeffs_x)))
            y = sum(coef * (t**j) for j, coef in enumerate(reversed(coeffs_y)))
            fitted_point = np.array([x, y])

            error = np.linalg.norm(np.array(orig_point) - fitted_point)
            if error > max_error:
                max_error = error
                max_error_idx = i

        return {
            "coefficients_x": coeffs_x.tolist(),
            "coefficients_y": coeffs_y.tolist(),
            "t_start": 0.0,
            "t_end": t_max,
            "curve_length": curve_length,
            "min_radius": min_radius,
            "max_error": max_error,
            "max_error_idx": max_error_idx,
            "control_points": points,
        }

    def _evaluate_polynomial_curvature_at_t(
        self, coeffs_x: np.ndarray, coeffs_y: np.ndarray, t: float, t_scale: float
    ) -> float:
        """Calculate radius of curvature for polynomial at parameter t."""
        # Convert coefficients to derivatives (numpy polyfit gives highest degree first)
        # First derivatives
        if len(coeffs_x) > 1:
            dx_coeffs = np.polyder(coeffs_x)
            dy_coeffs = np.polyder(coeffs_y)
            dx_dt = np.polyval(dx_coeffs, t) / t_scale  # Chain rule for normalized t
            dy_dt = np.polyval(dy_coeffs, t) / t_scale
        else:
            return float("inf")  # Constant polynomial = straight line

        # Second derivatives
        if len(coeffs_x) > 2:
            d2x_coeffs = np.polyder(dx_coeffs)
            d2y_coeffs = np.polyder(dy_coeffs)
            d2x_dt2 = np.polyval(d2x_coeffs, t) / (t_scale**2)  # Chain rule
            d2y_dt2 = np.polyval(d2y_coeffs, t) / (t_scale**2)
        else:
            return float("inf")  # Linear polynomial = straight line

        # Curvature formula: |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)
        numerator = abs(dx_dt * d2y_dt2 - dy_dt * d2x_dt2)
        denominator = (dx_dt**2 + dy_dt**2) ** (3 / 2)

        if denominator < 1e-10:
            return float("inf")  # Infinite radius (straight)

        curvature = numerator / denominator
        return 1.0 / curvature if curvature > 1e-10 else float("inf")

    def _decompose_polynomial_to_primitives(self, poly_fit: dict) -> List[Primitive]:
        """Decompose polynomial curve into straights and bends for engineering calculations.

        Args:
            poly_fit: Dictionary from _fit_polynomial

        Returns:
            List of Straight and Bend primitives
        """
        coeffs_x = np.array(poly_fit["coefficients_x"])
        coeffs_y = np.array(poly_fit["coefficients_y"])
        t_scale = poly_fit["t_end"]

        # Sample polynomial to analyze curvature changes
        n_samples = max(100, int(poly_fit["curve_length"] * 2))  # ~2 samples per meter
        primitives = []

        # Track regions of similar curvature
        current_region_start = 0.0
        current_region_type = None  # 'straight' or 'curve'
        prev_radius = None

        for i in range(n_samples + 1):
            t = i / n_samples

            # Calculate radius at this point
            radius = self._evaluate_polynomial_curvature_at_t(
                coeffs_x, coeffs_y, t, t_scale
            )

            # Classify as straight or curve - use more sensitive threshold for polynomial decomposition
            # Since polynomial already follows path accurately, preserve gentle curves
            is_straight = (
                radius >= self.natural_bend_threshold * 3
            )  # Much higher threshold
            region_type = "straight" if is_straight else "curve"

            # Detect transitions
            if current_region_type is None:
                current_region_type = region_type
                current_region_start = t
            elif current_region_type != region_type:
                # Region transition - create primitive for previous region
                primitive = self._create_primitive_from_polynomial_region(
                    coeffs_x,
                    coeffs_y,
                    t_scale,
                    current_region_start,
                    t,
                    current_region_type,
                )
                if primitive:
                    primitives.append(primitive)

                # Start new region
                current_region_type = region_type
                current_region_start = t

            prev_radius = radius

        # Handle final region
        if current_region_type is not None:
            primitive = self._create_primitive_from_polynomial_region(
                coeffs_x,
                coeffs_y,
                t_scale,
                current_region_start,
                1.0,
                current_region_type,
            )
            if primitive:
                primitives.append(primitive)

        return primitives

    def _create_primitive_from_polynomial_region(
        self,
        coeffs_x: np.ndarray,
        coeffs_y: np.ndarray,
        t_scale: float,
        t_start: float,
        t_end: float,
        region_type: str,
    ) -> Optional[Primitive]:
        """Create a primitive from a polynomial region."""
        if t_end <= t_start:
            return None

        # Get start and end points
        x_start = np.polyval(coeffs_x, t_start)
        y_start = np.polyval(coeffs_y, t_start)
        x_end = np.polyval(coeffs_x, t_end)
        y_end = np.polyval(coeffs_y, t_end)

        start_point = (x_start, y_start)
        end_point = (x_end, y_end)

        if region_type == "straight":
            # Create straight segment
            length = math.sqrt((x_end - x_start) ** 2 + (y_end - y_start) ** 2)
            if length >= self.min_straight_length:
                return Straight(
                    length_m=length, start_point=start_point, end_point=end_point
                )

        elif region_type == "curve":
            # Create bend by fitting arc to this polynomial region
            # Sample points from polynomial for arc fitting
            n_points = max(10, int((t_end - t_start) * 50))  # Dense sampling
            region_points = []

            for i in range(n_points + 1):
                t = t_start + (t_end - t_start) * i / n_points
                x = np.polyval(coeffs_x, t)
                y = np.polyval(coeffs_y, t)
                region_points.append((x, y))

            # Fit arc to these points
            arc_fit = self._fit_arc(region_points)
            if arc_fit and abs(arc_fit["angle"]) >= self.min_arc_angle:
                return Bend(
                    radius_m=arc_fit["radius"],
                    angle_deg=arc_fit["angle"],
                    direction="CW" if arc_fit["angle"] > 0 else "CCW",
                    center_point=arc_fit["center"],
                    bend_type="natural"
                    if arc_fit["radius"] >= self.natural_bend_threshold
                    else "manufactured",
                    control_points=region_points,
                    start_angle_deg=arc_fit.get("start_angle_deg", 0.0),
                    end_angle_deg=arc_fit.get("end_angle_deg", 0.0),
                )

        return None

    def _rejoin_aligned_straights(self, primitives: List[Primitive]) -> List[Primitive]:
        """Rejoin consecutive straight segments with same trajectory.

        This allows duct sections to be joined when they're aligned,
        matching real construction practices.
        """
        if len(primitives) <= 1:
            return primitives

        rejoined: List[Primitive] = []
        current_straight = None

        for primitive in primitives:
            if isinstance(primitive, Straight):
                if current_straight is None:
                    # Start new straight group
                    current_straight = primitive
                else:
                    # Check if this straight is aligned with the current one
                    if self._are_straights_aligned(current_straight, primitive):
                        # Merge with current straight - calculate actual length from
                        # coordinates
                        start = current_straight.start_point
                        end = primitive.end_point
                        actual_length = math.sqrt(
                            (end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2
                        )

                        current_straight = Straight(
                            length_m=actual_length,  # Use calculated length to avoid precision errors
                            start_point=start,
                            end_point=end,
                        )
                    else:
                        # Different trajectory - save current and start new
                        rejoined.append(current_straight)
                        current_straight = primitive
            else:
                # Non-straight primitive - save any current straight and add this
                # primitive
                if current_straight is not None:
                    rejoined.append(current_straight)
                    current_straight = None
                rejoined.append(primitive)

        # Add final straight if exists
        if current_straight is not None:
            rejoined.append(current_straight)

        return rejoined

    def _are_straights_aligned(
        self, straight1: Straight, straight2: Straight, angle_tolerance_deg: float = 2.0
    ) -> bool:
        """Check if two consecutive straights have same trajectory within tolerance."""
        # Check if straight1's end connects to straight2's start (within 1m)
        connection_distance = math.sqrt(
            (straight1.end_point[0] - straight2.start_point[0]) ** 2
            + (straight1.end_point[1] - straight2.start_point[1]) ** 2
        )

        if connection_distance > 1.0:  # 1m connection tolerance
            return False

        # Calculate bearing/direction for each straight
        def calculate_bearing(
            start: Tuple[float, float], end: Tuple[float, float]
        ) -> float:
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            return math.degrees(math.atan2(dy, dx))

        bearing1 = calculate_bearing(straight1.start_point, straight1.end_point)
        bearing2 = calculate_bearing(straight2.start_point, straight2.end_point)

        # Normalize bearings to [0, 360)
        bearing1 = bearing1 % 360
        bearing2 = bearing2 % 360

        # Calculate angular difference
        angle_diff = abs(bearing1 - bearing2)
        if angle_diff > 180:
            angle_diff = 360 - angle_diff

        return angle_diff <= angle_tolerance_deg

    def _improve_poor_path_following(
        self, primitives: List[Primitive], original_points: List[Tuple[float, float]]
    ) -> List[Primitive]:
        """Improve path following for areas with >1m deviation by using smaller segments."""
        if not primitives:
            return primitives

        # Quick check: if any primitive creates >1m deviation, refit with smaller
        # segments
        fitted_points = self._generate_fitted_points(primitives)
        if not fitted_points:
            return primitives

        # Find maximum deviation
        max_deviation = 0.0
        for orig_point in original_points[::2]:  # Sample every 2nd point
            min_dist = min(
                math.sqrt((orig_point[0] - fp[0]) ** 2 + (orig_point[1] - fp[1]) ** 2)
                for fp in fitted_points
            )
            max_deviation = max(max_deviation, min_dist)

        # If deviation is acceptable, keep current primitives
        if max_deviation <= 1.0:
            return primitives

        print(f"  Improving poor path following: {max_deviation:.1f}m deviation")

        # For poor path following, use very small segments with polynomial fitting
        improved_primitives: List[Primitive] = []

        # Use 1m segments for ultra-precise following to achieve <1m deviation
        segment_length = 1.0  # Reduced from 2.0m to 1.0m
        current_distance = 0.0
        current_segment_start = 0

        # Calculate cumulative distances
        distances = [0.0]
        for i in range(len(original_points) - 1):
            dist = math.sqrt(
                (original_points[i + 1][0] - original_points[i][0]) ** 2
                + (original_points[i + 1][1] - original_points[i][1]) ** 2
            )
            distances.append(distances[-1] + dist)

        total_distance = distances[-1]

        # Create segments at regular distance intervals
        current_start_idx = 0

        while current_distance < total_distance - 0.1:  # Small tolerance
            target_distance = current_distance + segment_length

            # Find end index for this segment
            current_end_idx = len(original_points) - 1
            for i, dist in enumerate(distances):
                if dist >= target_distance:
                    current_end_idx = i
                    break

            # Ensure minimum segment size
            if current_end_idx <= current_start_idx + 1:
                current_end_idx = min(current_start_idx + 2, len(original_points) - 1)

            # Extract segment points
            segment_points = original_points[current_start_idx : current_end_idx + 1]

            if len(segment_points) >= 2:
                # Try polynomial fitting for ultra-precise following
                if len(segment_points) >= 4:
                    poly_fit = self._fit_polynomial(segment_points, degree=3)
                    if (
                        poly_fit and poly_fit["max_error"] <= 0.15
                    ):  # Tighter tolerance from 0.3m to 0.15m
                        improved_primitives.append(
                            PolynomialCurve(
                                coefficients_x=poly_fit["coefficients_x"],
                                coefficients_y=poly_fit["coefficients_y"],
                                t_start=poly_fit["t_start"],
                                t_end=poly_fit["t_end"],
                                control_points=poly_fit["control_points"],
                                curve_length=poly_fit["curve_length"],
                                min_radius=poly_fit["min_radius"],
                            )
                        )
                    else:
                        # Fallback to straight
                        length = math.sqrt(
                            (segment_points[-1][0] - segment_points[0][0]) ** 2
                            + (segment_points[-1][1] - segment_points[0][1]) ** 2
                        )
                        improved_primitives.append(
                            Straight(
                                length_m=length,
                                start_point=segment_points[0],
                                end_point=segment_points[-1],
                            )
                        )
                else:
                    # Short segment - use straight
                    length = math.sqrt(
                        (segment_points[-1][0] - segment_points[0][0]) ** 2
                        + (segment_points[-1][1] - segment_points[0][1]) ** 2
                    )
                    if length >= 0.5:  # Minimum 0.5m segment
                        improved_primitives.append(
                            Straight(
                                length_m=length,
                                start_point=segment_points[0],
                                end_point=segment_points[-1],
                            )
                        )

            # Move to next segment
            current_start_idx = current_end_idx
            current_distance = (
                distances[current_end_idx]
                if current_end_idx < len(distances)
                else total_distance
            )

        return improved_primitives if improved_primitives else primitives

    def _generate_fitted_points(
        self, primitives: List[Primitive], points_per_meter: float = 10
    ) -> List[Tuple[float, float]]:
        """Generate points along the fitted primitives."""
        fitted_points = []

        for primitive in primitives:
            if isinstance(primitive, Straight):
                # Generate points along straight line
                n_points = max(2, int(primitive.length_m * points_per_meter))
                for i in range(n_points):
                    t = i / (n_points - 1)
                    x = primitive.start_point[0] + t * (
                        primitive.end_point[0] - primitive.start_point[0]
                    )
                    y = primitive.start_point[1] + t * (
                        primitive.end_point[1] - primitive.start_point[1]
                    )
                    fitted_points.append((x, y))

            elif isinstance(primitive, Bend):
                # FIXED: Generate arc points using actual start/end angles from fitting
                if hasattr(primitive, "control_points") and primitive.control_points:
                    # Use stored control points to generate correct arc path
                    if len(primitive.control_points) >= 2:
                        # Use actual control points for precise path generation
                        for point in primitive.control_points:
                            fitted_points.append(point)
                else:
                    # Fallback: Generate arc using stored start/end angles
                    arc_length = (
                        abs(math.radians(primitive.angle_deg)) * primitive.radius_m
                    )
                    n_points = max(3, int(arc_length * points_per_meter))

                    # Use actual start and end angles if available
                    if hasattr(primitive, "start_angle_deg") and hasattr(
                        primitive, "end_angle_deg"
                    ):
                        start_angle = math.radians(primitive.start_angle_deg)
                        end_angle = math.radians(primitive.end_angle_deg)

                        # Handle angle wrapping
                        angle_diff = end_angle - start_angle
                        if abs(angle_diff) > math.pi:
                            if angle_diff > 0:
                                angle_diff -= 2 * math.pi
                            else:
                                angle_diff += 2 * math.pi

                        # Generate points along actual arc
                        for i in range(n_points):
                            t = i / (n_points - 1) if n_points > 1 else 0
                            angle = start_angle + t * angle_diff
                            x = primitive.center_point[
                                0
                            ] + primitive.radius_m * math.cos(angle)
                            y = primitive.center_point[
                                1
                            ] + primitive.radius_m * math.sin(angle)
                            fitted_points.append((x, y))
                    else:
                        # Ultimate fallback: use center point only (better than wrong
                        # arc)
                        fitted_points.append(primitive.center_point)

            elif isinstance(primitive, PolynomialCurve):
                # Generate points along polynomial curve
                n_points = max(5, int(primitive.curve_length * points_per_meter))
                for i in range(n_points):
                    t = primitive.t_start + (
                        primitive.t_end - primitive.t_start
                    ) * i / (n_points - 1)
                    # Normalize t for polynomial evaluation
                    t_norm = t / primitive.t_end if primitive.t_end > 0 else 0
                    point = primitive.evaluate_at_t(t_norm)
                    fitted_points.append(point)

        return fitted_points

    def _calculate_fitting_error(
        self, original: List[Tuple[float, float]], fitted: List[Tuple[float, float]]
    ) -> Tuple[float, float]:
        """Calculate error between original and fitted points."""
        if not original or not fitted:
            return 0.0, 0.0

        # For each original point, find closest fitted point
        total_error = 0.0
        max_error = 0.0

        for orig_point in original:
            min_dist = float("inf")

            for fit_point in fitted:
                dist = math.sqrt(
                    (orig_point[0] - fit_point[0]) ** 2
                    + (orig_point[1] - fit_point[1]) ** 2
                )
                min_dist = min(min_dist, dist)

            total_error += min_dist
            max_error = max(max_error, min_dist)

        return total_error, max_error

    def _find_closest_standard_bend(
        self, fitted_radius: float, fitted_angle: float
    ) -> int:
        """Find closest standard duct bend to fitted parameters.

        Args:
            fitted_radius: Fitted radius in meters
            fitted_angle: Fitted angle in degrees

        Returns:
            Standard bend radius in mm
        """
        fitted_radius_mm = fitted_radius * 1000  # Convert to mm

        # Find bends with similar angles (within 10 degrees)
        angle_tolerance = 10.0
        candidate_bends = [
            bend
            for bend in STANDARD_DUCT_BENDS
            if abs(bend["angle"] - fitted_angle) <= angle_tolerance
        ]

        if not candidate_bends:
            # If no angle match, use all bends and find closest radius
            candidate_bends = STANDARD_DUCT_BENDS

        # Find closest radius
        best_bend = min(
            candidate_bends, key=lambda b: abs(b["radius"] - fitted_radius_mm)
        )

        return best_bend["radius"]

    def _fallback_fit(self, points: List[Tuple[float, float]]) -> List[Primitive]:
        """Fallback fitting strategy for complex polylines that fail recursive fitting.

        Uses a simpler approach: try to fit the entire polyline as either a straight or arc,
        with very relaxed tolerances.
        """
        if len(points) < 2:
            return []

        # For snaking routes: try multiple arc approach first
        snaking_result = self._fit_snaking_route(points)
        if snaking_result:
            return snaking_result

        # Try fitting entire segment as single arc - use pavement tolerance even
        # for fallback
        arc_fit = self._fit_arc(points)
        pavement_fallback_tolerance = (
            2.0  # 2m tolerance for fallback (still much tighter than 50m)
        )
        if arc_fit and arc_fit["max_error"] <= pavement_fallback_tolerance:
            if abs(arc_fit["angle"]) >= 1.0:  # Very relaxed 1° minimum
                # Classify as natural vs manufactured
                if arc_fit["radius"] >= self.natural_bend_threshold:
                    return [
                        Bend(
                            radius_m=arc_fit["radius"],
                            angle_deg=arc_fit["angle"],
                            direction="CW" if arc_fit["angle"] > 0 else "CCW",
                            center_point=arc_fit["center"],
                            bend_type="natural",
                        )
                    ]
                else:
                    standard_radius = self._find_closest_standard_bend(
                        arc_fit["radius"], abs(arc_fit["angle"])
                    )
                    return [
                        Bend(
                            radius_m=standard_radius / 1000,
                            angle_deg=arc_fit["angle"],
                            direction="CW" if arc_fit["angle"] > 0 else "CCW",
                            center_point=arc_fit["center"],
                            bend_type="manufactured",
                        )
                    ]

        # Try fitting as single straight line - use pavement tolerance
        line_fit = self._fit_straight_line(points)
        if line_fit and line_fit["max_error"] <= pavement_fallback_tolerance:
            if line_fit["length"] >= 1.0:  # 1m minimum
                return [
                    Straight(
                        length_m=line_fit["length"],
                        start_point=points[0],
                        end_point=points[-1],
                    )
                ]

        # Last resort: break into smaller segments
        segment_size = max(3, len(points) // 10)  # Break into ~10 segments
        fallback_primitives: List[Primitive] = []

        for i in range(0, len(points) - 1, segment_size):
            end_idx = min(i + segment_size, len(points) - 1)
            if end_idx > i:
                segment = points[i : end_idx + 1]

                # Force fit as straight line
                if len(segment) >= 2:
                    p1, p2 = segment[0], segment[-1]
                    length = math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
                    if length > 0.1:  # Only add if meaningful length
                        fallback_primitives.append(
                            Straight(length_m=length, start_point=p1, end_point=p2)
                        )

        return fallback_primitives

    def _fit_snaking_route(
        self, points: List[Tuple[float, float]]
    ) -> Optional[List[Primitive]]:
        """Attempt to fit snaking routes as multiple connected arcs.

        For routes that snake (low route efficiency), try to fit as series of
        connected arcs rather than straight shortcuts.
        """
        if len(points) < 6:  # Need enough points for multiple arcs
            return None

        # Calculate route efficiency (straight distance / actual length)
        start_point = np.array(points[0])
        end_point = np.array(points[-1])
        straight_distance = np.linalg.norm(end_point - start_point)

        actual_length = sum(
            np.linalg.norm(np.array(points[i + 1]) - np.array(points[i]))
            for i in range(len(points) - 1)
        )

        route_efficiency = (
            straight_distance / actual_length if actual_length > 0 else 1.0
        )

        # Apply snaking logic if route efficiency < 95% OR has many points
        # (indicates curves)
        if route_efficiency > 0.95 and len(points) < 15:
            return None

        # Try fitting as multiple connected arcs for snaking routes
        primitives: List[Primitive] = []

        # For snaking routes, use smaller segments to capture more curves
        num_segments = min(
            5, max(3, len(points) // 8)
        )  # 3-5 segments based on point count
        segment_size = len(points) // num_segments

        for i in range(num_segments):
            start_idx = i * segment_size
            if i == num_segments - 1:  # Last segment gets remaining points
                end_idx = len(points)
            else:
                end_idx = (
                    i + 1
                ) * segment_size + 2  # Overlap by 2 points for continuity

            if end_idx > len(points):
                end_idx = len(points)

            segment_points = points[start_idx:end_idx]

            if len(segment_points) < 3:
                continue

            # Calculate segment length
            segment_length = sum(
                np.linalg.norm(
                    np.array(segment_points[j + 1]) - np.array(segment_points[j])
                )
                for j in range(len(segment_points) - 1)
            )

            # Try arc first for each segment with strict pavement tolerance
            arc_fit = self._fit_arc(segment_points)
            if arc_fit and abs(arc_fit["angle"]) >= 2.0:  # 2° minimum for snaking
                # For pavement routes: strict 300mm tolerance even for snaking
                max_allowed_error = min(0.3, segment_length * 0.08)

                if arc_fit["max_error"] <= max_allowed_error:
                    if arc_fit["radius"] >= self.natural_bend_threshold:
                        primitives.append(
                            Bend(
                                radius_m=arc_fit["radius"],
                                angle_deg=arc_fit["angle"],
                                direction="CW" if arc_fit["angle"] > 0 else "CCW",
                                center_point=arc_fit["center"],
                                bend_type="natural",
                            )
                        )
                        continue

            # Fallback to straight for this segment
            line_fit = self._fit_straight_line(segment_points)
            if line_fit and line_fit["length"] >= 5.0:  # 5m minimum for snaking
                primitives.append(
                    Straight(
                        length_m=line_fit["length"],
                        start_point=segment_points[0],
                        end_point=segment_points[-1],
                    )
                )

        # Check if this approach preserves more length than simple fitting
        if primitives:
            fitted_length = sum(p.length() for p in primitives)
            length_preservation = fitted_length / actual_length

            # Accept if preserves 96-104% of original length (very tight bounds)
            if 0.96 <= length_preservation <= 1.04:
                return primitives

        return None

    def _try_preserve_original(
        self, points: List[Tuple[float, float]]
    ) -> Optional[Primitive]:
        """Try to preserve the original polyline geometry if it's already acceptable.

        Only preserve as single primitive if path following is excellent (<0.5m deviation).
        Force splitting for sections that would create large deviations.

        Args:
            points: List of (x, y) coordinate tuples

        Returns:
            Primitive if original geometry is acceptable, None otherwise
        """
        if len(points) < 2:
            return None

        # Calculate actual polyline length
        actual_length = sum(
            math.sqrt(
                (points[i + 1][0] - points[i][0]) ** 2
                + (points[i + 1][1] - points[i][1]) ** 2
            )
            for i in range(len(points) - 1)
        )

        # Check if it's essentially straight (within tolerance)
        start_point = np.array(points[0])
        end_point = np.array(points[-1])
        straight_distance = np.linalg.norm(end_point - start_point)

        # If very close to straight line, preserve as straight
        deviation_from_straight = abs(actual_length - straight_distance)
        # Also check that all points are close to the straight line (within 0.3m
        # for pavements)
        max_lateral_deviation = 0.0
        if len(points) > 2:
            p1 = start_point
            p2 = end_point
            line_vec = p2 - p1
            line_length = np.linalg.norm(line_vec)

            if line_length > 1e-6:
                line_unit = line_vec / line_length

                for point in points[1:-1]:
                    p = np.array(point)
                    # Project point onto line
                    ap = p - p1
                    t = np.clip(np.dot(ap, line_unit), 0, line_length)
                    closest = p1 + t * line_unit
                    lateral_dev = np.linalg.norm(p - closest)
                    max_lateral_deviation = max(max_lateral_deviation, lateral_dev)

        # Strict 0.3m lateral tolerance for CAD design accuracy
        design_tolerance = 0.3  # 300mm for CAD alignment

        # ADDITIONAL CHECK: For longer sections, ensure excellent path following before preserving
        # as single primitive. Force splitting if path following would be poor.
        max_allowed_deviation = 0.5  # 500mm max for any single primitive
        if max_lateral_deviation > max_allowed_deviation:
            return None  # Force splitting for better path following

        if (
            deviation_from_straight <= self.straight_tolerance
            and straight_distance >= self.min_straight_length
            and max_lateral_deviation <= design_tolerance
        ):
            return Straight(
                length_m=straight_distance,  # Use straight distance for model validation
                start_point=tuple(points[0]),
                end_point=tuple(points[-1]),
            )

        # Check if it's a good arc fit that should be preserved
        if len(points) >= 3:
            arc_fit = self._fit_arc(points)
            # Use strict 0.3m tolerance for CAD design accuracy
            strict_arc_tolerance = min(0.3, self.arc_tolerance)

            if arc_fit and arc_fit["max_error"] <= strict_arc_tolerance:
                if abs(arc_fit["angle"]) >= self.min_arc_angle:
                    # Check if the arc radius is acceptable for this duct
                    if (
                        arc_fit["radius"] >= self.natural_bend_threshold * 0.9
                    ):  # 10% tolerance
                        # Arc length should match polyline length closely
                        arc_length = (
                            abs(math.radians(arc_fit["angle"])) * arc_fit["radius"]
                        )
                        length_match = abs(arc_length - actual_length) / actual_length

                        if length_match <= 0.02:  # Within 2% length match
                            # CRITICAL: Check path following before preserving large arcs
                            # Calculate max deviation from arc to original points
                            max_arc_deviation = 0.0
                            center = np.array(arc_fit["center"])
                            for point in points[1:-1]:  # Skip endpoints
                                p = np.array(point)
                                dist_to_center = np.linalg.norm(p - center)
                                deviation_from_arc = abs(
                                    dist_to_center - arc_fit["radius"]
                                )
                                max_arc_deviation = max(
                                    max_arc_deviation, deviation_from_arc
                                )

                            # Only preserve arc if path following is excellent
                            if (
                                max_arc_deviation <= 0.3
                            ):  # 300mm max deviation for single arc
                                return Bend(
                                    radius_m=arc_fit["radius"],
                                    angle_deg=arc_fit["angle"],
                                    direction="CW" if arc_fit["angle"] > 0 else "CCW",
                                    center_point=arc_fit["center"],
                                    bend_type="natural",
                                    control_points=points,
                                    start_angle_deg=arc_fit.get("start_angle_deg", 0.0),
                                    end_angle_deg=arc_fit.get("end_angle_deg", 0.0),
                                )
                            # If deviation too large, force splitting for better path
                            # following

        return None
