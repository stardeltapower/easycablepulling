"""Geometry validation for cable routes."""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

from ..config import GEOMETRY_TOLERANCES
from ..core.models import Bend, CableSpec, Primitive, Route, Section, Straight


@dataclass
class ValidationIssue:
    """Represents a validation issue found in geometry."""

    section_id: str
    primitive_index: int
    issue_type: str
    severity: str  # "error", "warning", "info"
    message: str
    value: Optional[float] = None
    limit: Optional[float] = None


@dataclass
class ValidationResult:
    """Result of geometry validation."""

    is_valid: bool
    issues: List[ValidationIssue]
    total_errors: int
    total_warnings: int

    def add_issue(self, issue: ValidationIssue) -> None:
        """Add a validation issue."""
        self.issues.append(issue)
        if issue.severity == "error":
            self.total_errors += 1
            self.is_valid = False
        elif issue.severity == "warning":
            self.total_warnings += 1

    def get_summary(self) -> str:
        """Get validation summary."""
        if self.is_valid:
            return f"Validation passed with {self.total_warnings} warnings"
        else:
            return f"Validation failed: {self.total_errors} errors, {self.total_warnings} warnings"


class GeometryValidator:
    """Validate cable route geometry against specifications."""

    def __init__(
        self,
        lateral_tolerance: Optional[float] = None,
        length_tolerance_percent: Optional[float] = None,
        min_straight_length: Optional[float] = None,
        min_bend_radius: Optional[float] = None,
        max_bend_angle: Optional[float] = None,
    ):
        """Initialize geometry validator.

        Args:
            lateral_tolerance: Max lateral deviation from original (meters)
            length_tolerance_percent: Max length error as percentage
            min_straight_length: Minimum straight segment length (meters)
            min_bend_radius: Minimum bend radius (meters)
            max_bend_angle: Maximum bend angle (degrees)
        """
        self.lateral_tolerance = (
            lateral_tolerance or GEOMETRY_TOLERANCES["lateral_tolerance"]
        )
        self.length_tolerance_percent = (
            length_tolerance_percent or GEOMETRY_TOLERANCES["length_tolerance_percent"]
        )
        self.min_straight_length = (
            min_straight_length or GEOMETRY_TOLERANCES["min_straight_length"]
        )
        self.min_bend_radius = min_bend_radius or GEOMETRY_TOLERANCES["min_bend_radius"]
        self.max_bend_angle = max_bend_angle or GEOMETRY_TOLERANCES["max_bend_angle"]

    def validate_route(
        self, route: Route, cable_spec: Optional[CableSpec] = None
    ) -> ValidationResult:
        """Validate entire route geometry.

        Args:
            route: Route to validate
            cable_spec: Optional cable specification for additional checks

        Returns:
            ValidationResult with all issues found
        """
        result = ValidationResult(
            is_valid=True, issues=[], total_errors=0, total_warnings=0
        )

        # Validate each section
        for section in route.sections:
            self._validate_section(section, result, cable_spec)

        # Validate route continuity
        self._validate_route_continuity(route, result)

        return result

    def _validate_section(
        self,
        section: Section,
        result: ValidationResult,
        cable_spec: Optional[CableSpec] = None,
    ) -> None:
        """Validate a single section."""
        # Check if section has been fitted
        if not section.primitives:
            result.add_issue(
                ValidationIssue(
                    section_id=section.id,
                    primitive_index=-1,
                    issue_type="missing_primitives",
                    severity="warning",
                    message="Section has not been fitted to primitives",
                )
            )
            return

        # Validate fitted length vs original
        fitted_length = sum(self._get_primitive_length(p) for p in section.primitives)
        length_error_percent = (
            abs(fitted_length - section.original_length) / section.original_length * 100
        )

        if length_error_percent > self.length_tolerance_percent:
            result.add_issue(
                ValidationIssue(
                    section_id=section.id,
                    primitive_index=-1,
                    issue_type="length_error",
                    severity="error",
                    message=f"Fitted length error {length_error_percent:.1f}% exceeds tolerance",
                    value=length_error_percent,
                    limit=self.length_tolerance_percent,
                )
            )

        # Validate each primitive
        for i, primitive in enumerate(section.primitives):
            self._validate_primitive(primitive, section.id, i, result, cable_spec)

        # Validate primitive connections
        self._validate_primitive_connections(section, result)

    def _validate_primitive(
        self,
        primitive: Primitive,
        section_id: str,
        index: int,
        result: ValidationResult,
        cable_spec: Optional[CableSpec] = None,
    ) -> None:
        """Validate a single primitive."""
        if isinstance(primitive, Straight):
            # Check minimum length
            if primitive.length_m < self.min_straight_length:
                result.add_issue(
                    ValidationIssue(
                        section_id=section_id,
                        primitive_index=index,
                        issue_type="short_straight",
                        severity="warning",
                        message=f"Straight segment {primitive.length_m:.2f}m is shorter than minimum",
                        value=primitive.length_m,
                        limit=self.min_straight_length,
                    )
                )

        elif isinstance(primitive, Bend):
            # Check minimum radius
            if primitive.radius_m < self.min_bend_radius:
                result.add_issue(
                    ValidationIssue(
                        section_id=section_id,
                        primitive_index=index,
                        issue_type="small_radius",
                        severity="error",
                        message=f"Bend radius {primitive.radius_m:.2f}m is less than minimum",
                        value=primitive.radius_m,
                        limit=self.min_bend_radius,
                    )
                )

            # Check against cable minimum bend radius
            if (
                cable_spec and primitive.radius_m < cable_spec.min_bend_radius / 1000
            ):  # Convert mm to m
                result.add_issue(
                    ValidationIssue(
                        section_id=section_id,
                        primitive_index=index,
                        issue_type="cable_bend_radius",
                        severity="error",
                        message=f"Bend radius {primitive.radius_m:.2f}m is less than cable minimum {cable_spec.min_bend_radius/1000:.2f}m",
                        value=primitive.radius_m,
                        limit=cable_spec.min_bend_radius / 1000,
                    )
                )

            # Validate bend type classification
            if hasattr(primitive, "bend_type"):
                # Check if manufactured bend uses standard radius
                if primitive.bend_type == "manufactured":
                    from ..config import STANDARD_DUCT_BENDS

                    standard_radii = [float(b["radius"]) for b in STANDARD_DUCT_BENDS]
                    bend_radius_mm = primitive.radius_m * 1000

                    # Allow 5% tolerance for standard bend matching
                    tolerance = 0.05
                    is_standard = any(
                        abs(bend_radius_mm - std_radius) / std_radius <= tolerance
                        for std_radius in standard_radii
                    )

                    if not is_standard:
                        result.add_issue(
                            ValidationIssue(
                                section_id=section_id,
                                primitive_index=index,
                                issue_type="non_standard_bend",
                                severity="warning",
                                message=f"Manufactured bend radius {primitive.radius_m:.2f}m doesn't match standard options",
                                value=primitive.radius_m,
                                limit=None,
                            )
                        )

            # Check maximum angle
            if abs(primitive.angle_deg) > self.max_bend_angle:
                result.add_issue(
                    ValidationIssue(
                        section_id=section_id,
                        primitive_index=index,
                        issue_type="large_angle",
                        severity="warning",
                        message=f"Bend angle {abs(primitive.angle_deg):.1f}° exceeds maximum",
                        value=abs(primitive.angle_deg),
                        limit=self.max_bend_angle,
                    )
                )

    def _validate_primitive_connections(
        self, section: Section, result: ValidationResult
    ) -> None:
        """Validate that primitives connect properly."""
        for i in range(len(section.primitives) - 1):
            p1 = section.primitives[i]
            p2 = section.primitives[i + 1]

            # Get end point of p1 and start point of p2
            end_p1 = self._get_primitive_end_point(p1)
            start_p2 = self._get_primitive_start_point(p2)

            # Check if they connect within tolerance
            if end_p1 and start_p2:
                distance = math.sqrt(
                    (end_p1[0] - start_p2[0]) ** 2 + (end_p1[1] - start_p2[1]) ** 2
                )

                if distance > 0.01:  # 1cm tolerance
                    result.add_issue(
                        ValidationIssue(
                            section_id=section.id,
                            primitive_index=i,
                            issue_type="disconnected_primitives",
                            severity="error",
                            message=f"Gap of {distance:.3f}m between primitives {i} and {i+1}",
                            value=distance,
                            limit=0.01,
                        )
                    )

    def _validate_route_continuity(
        self, route: Route, result: ValidationResult
    ) -> None:
        """Validate continuity between sections."""
        for i in range(len(route.sections) - 1):
            s1 = route.sections[i]
            s2 = route.sections[i + 1]

            # Check if sections connect
            if s1.original_polyline and s2.original_polyline:
                end_s1 = s1.original_polyline[-1]
                start_s2 = s2.original_polyline[0]

                distance = math.sqrt(
                    (end_s1[0] - start_s2[0]) ** 2 + (end_s1[1] - start_s2[1]) ** 2
                )

                if distance > 50.0:  # 50m gap indicates missing connection
                    result.add_issue(
                        ValidationIssue(
                            section_id=s1.id,
                            primitive_index=-1,
                            issue_type="section_gap",
                            severity="warning",
                            message=f"Large gap of {distance:.1f}m to next section {s2.id}",
                            value=distance,
                            limit=50.0,
                        )
                    )

    def _get_primitive_length(self, primitive: Primitive) -> float:
        """Get length of a primitive."""
        if isinstance(primitive, Straight):
            return primitive.length_m
        elif isinstance(primitive, Bend):
            # Arc length = radius * angle (in radians)
            return primitive.radius_m * abs(math.radians(primitive.angle_deg))
        return 0.0

    def _get_primitive_start_point(
        self, primitive: Primitive
    ) -> Optional[Tuple[float, float]]:
        """Get start point of a primitive."""
        if isinstance(primitive, Straight):
            return primitive.start_point
        elif isinstance(primitive, Bend):
            # For bend, calculate from center and start angle
            # This is simplified - real implementation needs proper calculation
            return None
        return None

    def _get_primitive_end_point(
        self, primitive: Primitive
    ) -> Optional[Tuple[float, float]]:
        """Get end point of a primitive."""
        if isinstance(primitive, Straight):
            return primitive.end_point
        elif isinstance(primitive, Bend):
            # For bend, calculate from center and end angle
            # This is simplified - real implementation needs proper calculation
            return None
        return None

    def validate_lateral_deviation(
        self,
        original_points: List[Tuple[float, float]],
        fitted_points: List[Tuple[float, float]],
    ) -> Tuple[float, bool]:
        """Validate lateral deviation between original and fitted points.

        Args:
            original_points: Original polyline points
            fitted_points: Fitted geometry points

        Returns:
            Tuple of (max_deviation, is_within_tolerance)
        """
        max_deviation = 0.0

        for orig_point in original_points:
            min_dist = float("inf")

            # Find closest fitted point
            for fit_point in fitted_points:
                dist = math.sqrt(
                    (orig_point[0] - fit_point[0]) ** 2
                    + (orig_point[1] - fit_point[1]) ** 2
                )
                min_dist = min(min_dist, dist)

            max_deviation = max(max_deviation, min_dist)

        is_valid = max_deviation <= self.lateral_tolerance
        return max_deviation, is_valid
