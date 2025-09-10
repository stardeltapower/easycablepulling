"""Minor splitting logic for cable route sections exceeding maximum cable length."""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

from ..core.models import Route, Section


@dataclass
class SplitPoint:
    """Information about a potential split point."""

    polyline_index: int  # Index in original polyline
    position: float  # Position along route (chainage) in meters
    reason: str  # Reason for split ("max_length", "joint_location", "optimal")
    priority: int  # Split priority (1=highest, lower numbers preferred)


@dataclass
class SplittingResult:
    """Result of route splitting operation."""

    original_route: Route
    split_route: Route
    split_points: List[SplitPoint]
    sections_created: int
    success: bool
    message: str = ""


class RouteSplitter:
    """Handles minor splitting of cable route sections."""

    def __init__(
        self,
        max_cable_length: float = 500.0,
        min_section_length: float = 50.0,
        avoid_bend_distance: float = 10.0,
    ) -> None:
        """Initialize route splitter.

        Args:
            max_cable_length: Maximum allowable cable length in meters
            min_section_length: Minimum section length after splitting
            avoid_bend_distance: Distance to avoid splitting near bends (meters)
        """
        self.max_cable_length = max_cable_length
        self.min_section_length = min_section_length
        self.avoid_bend_distance = avoid_bend_distance

    def needs_splitting(self, section: Section) -> bool:
        """Check if section exceeds maximum cable length."""
        return section.original_length > self.max_cable_length

    def find_optimal_split_points(
        self,
        section: Section,
        target_length: Optional[float] = None,
    ) -> List[SplitPoint]:
        """Find optimal points to split a long section.

        Args:
            section: Section to analyze for splitting
            target_length: Target length for each subsection (defaults to max_cable_length)

        Returns:
            List of optimal split points sorted by priority
        """
        if target_length is None:
            target_length = self.max_cable_length

        polyline = section.original_polyline
        if len(polyline) < 3:
            return []  # Cannot split section with < 3 points

        # Calculate cumulative distances along polyline
        distances = [0.0]
        for i in range(1, len(polyline)):
            p1 = polyline[i - 1]
            p2 = polyline[i]
            dist = math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
            distances.append(distances[-1] + dist)

        total_length = distances[-1]

        if total_length <= self.max_cable_length:
            return []  # No splitting needed

        # Calculate number of sections needed
        num_sections = math.ceil(total_length / target_length)
        split_points = []

        # Find split points at regular intervals
        for i in range(1, num_sections):
            target_distance = i * (total_length / num_sections)

            # Find closest polyline point to target distance
            best_idx = 1
            best_distance_diff = float("inf")

            for j in range(1, len(distances) - 1):  # Don't split at endpoints
                distance_diff = abs(distances[j] - target_distance)
                if distance_diff < best_distance_diff:
                    best_distance_diff = distance_diff
                    best_idx = j

            # Check if this split point is valid
            if self._is_valid_split_point(polyline, best_idx, distances):
                split_points.append(
                    SplitPoint(
                        polyline_index=best_idx,
                        position=distances[best_idx],
                        reason="max_length",
                        priority=1,
                    )
                )

        return split_points

    def _is_valid_split_point(
        self,
        polyline: List[Tuple[float, float]],
        index: int,
        distances: List[float],
    ) -> bool:
        """Check if a polyline index is a valid split point.

        Args:
            polyline: Original polyline points
            index: Index to check
            distances: Cumulative distances along polyline

        Returns:
            True if index is a valid split point
        """
        if index <= 0 or index >= len(polyline) - 1:
            return False  # Cannot split at endpoints

        # Check if potential sections would be long enough
        left_length = distances[index]
        right_length = distances[-1] - distances[index]

        if (
            left_length < self.min_section_length
            or right_length < self.min_section_length
        ):
            return False

        # Check for sharp bends near split point (avoid splitting near bends)
        if self._has_sharp_bend_nearby(polyline, index):
            return False

        return True

    def _has_sharp_bend_nearby(
        self,
        polyline: List[Tuple[float, float]],
        index: int,
        angle_threshold: float = 30.0,  # degrees
    ) -> bool:
        """Check if there's a sharp bend near the split point."""
        if index < 2 or index >= len(polyline) - 2:
            return False

        # Check angles at nearby points
        for check_idx in range(max(1, index - 2), min(len(polyline) - 1, index + 3)):
            if check_idx < 1 or check_idx >= len(polyline) - 1:
                continue

            # Calculate angle at this point
            p1 = polyline[check_idx - 1]
            p2 = polyline[check_idx]
            p3 = polyline[check_idx + 1]

            # Vectors
            v1 = (p1[0] - p2[0], p1[1] - p2[1])
            v2 = (p3[0] - p2[0], p3[1] - p2[1])

            # Calculate angle between vectors
            dot_product = v1[0] * v2[0] + v1[1] * v2[1]
            mag1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
            mag2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)

            if mag1 < 1e-6 or mag2 < 1e-6:
                continue

            cos_angle = dot_product / (mag1 * mag2)
            cos_angle = max(-1.0, min(1.0, cos_angle))  # Clamp to valid range
            angle_deg = math.degrees(math.acos(cos_angle))

            # If angle is sharp (deviation from straight), avoid splitting nearby
            deviation_angle = 180.0 - angle_deg
            if deviation_angle > angle_threshold:
                return True

        return False

    def split_section(
        self,
        section: Section,
        split_points: List[SplitPoint],
    ) -> List[Section]:
        """Split a section at the specified points.

        Args:
            section: Section to split
            split_points: Points where to split the section

        Returns:
            List of new sections after splitting
        """
        if not split_points:
            return [section]

        polyline = section.original_polyline

        # Sort split points by polyline index
        sorted_splits = sorted(split_points, key=lambda sp: sp.polyline_index)

        # Create new sections
        new_sections = []
        start_idx = 0

        for i, split_point in enumerate(sorted_splits):
            end_idx = (
                split_point.polyline_index + 1
            )  # Include split point in both sections

            # Create subsection
            subsection_polyline = polyline[start_idx:end_idx]
            subsection_id = f"{section.id}_{i+1:02d}"

            new_section = Section(
                id=subsection_id,
                original_polyline=subsection_polyline,
                primitives=[],  # Will be fitted later
            )
            new_sections.append(new_section)

            start_idx = split_point.polyline_index  # Start next section at split point

        # Create final section from last split point to end
        final_polyline = polyline[start_idx:]
        final_id = f"{section.id}_{len(sorted_splits)+1:02d}"

        final_section = Section(
            id=final_id,
            original_polyline=final_polyline,
            primitives=[],
        )
        new_sections.append(final_section)

        return new_sections

    def split_route(self, route: Route) -> SplittingResult:
        """Split all sections in a route that exceed maximum cable length.

        Args:
            route: Route to split

        Returns:
            SplittingResult with new route and split information
        """
        new_sections = []
        all_split_points = []
        sections_created = 0

        for section in route.sections:
            if self.needs_splitting(section):
                # Find optimal split points
                split_points = self.find_optimal_split_points(section)

                if split_points:
                    # Split the section
                    subsections = self.split_section(section, split_points)
                    new_sections.extend(subsections)
                    all_split_points.extend(split_points)
                    sections_created += (
                        len(subsections) - 1
                    )  # Number of additional sections
                else:
                    # Cannot split effectively, keep original with warning
                    new_sections.append(section)
            else:
                # Section is within limits, keep as-is
                new_sections.append(section)

        # Create new route with split sections
        split_route = Route(
            name=f"{route.name}_split",
            sections=new_sections,
            metadata={
                **route.metadata,
                "split_operation": {
                    "original_sections": len(route.sections),
                    "final_sections": len(new_sections),
                    "sections_added": sections_created,
                    "max_cable_length": self.max_cable_length,
                },
            },
        )

        return SplittingResult(
            original_route=route,
            split_route=split_route,
            split_points=all_split_points,
            sections_created=sections_created,
            success=True,
            message=f"Split {len(route.sections)} sections into {len(new_sections)} sections",
        )

    def find_joint_locations(
        self,
        polyline: List[Tuple[float, float]],
        joint_spacing: float = 6.0,  # Standard joint spacing in meters
    ) -> List[int]:
        """Find potential joint locations for splitting.

        Args:
            polyline: Polyline points
            joint_spacing: Standard joint spacing distance

        Returns:
            List of polyline indices near joint locations
        """
        if len(polyline) < 3:
            return []

        # Calculate cumulative distances
        distances = [0.0]
        for i in range(1, len(polyline)):
            p1 = polyline[i - 1]
            p2 = polyline[i]
            dist = math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
            distances.append(distances[-1] + dist)

        total_length = distances[-1]
        joint_indices = []

        # Find points near multiples of joint spacing
        current_joint_distance = joint_spacing
        while current_joint_distance < total_length - joint_spacing:
            # Find closest polyline point to joint distance
            best_idx = 1
            best_diff = float("inf")

            for i in range(1, len(distances) - 1):
                diff = abs(distances[i] - current_joint_distance)
                if diff < best_diff:
                    best_diff = diff
                    best_idx = i

            # Only add if reasonably close to joint spacing (within 2m)
            if best_diff < 2.0:
                joint_indices.append(best_idx)

            current_joint_distance += joint_spacing

        return joint_indices

    def split_at_joints(
        self,
        section: Section,
        joint_spacing: float = 6.0,
    ) -> List[Section]:
        """Split section at standard joint locations.

        Args:
            section: Section to split at joints
            joint_spacing: Standard joint spacing in meters

        Returns:
            List of sections split at joint locations
        """
        joint_indices = self.find_joint_locations(
            section.original_polyline, joint_spacing
        )

        if not joint_indices:
            return [section]

        # Convert joint indices to split points
        split_points = [
            SplitPoint(
                polyline_index=idx,
                position=0.0,  # Will be calculated if needed
                reason="joint_location",
                priority=2,  # Lower priority than max_length splits
            )
            for idx in joint_indices
        ]

        return self.split_section(section, split_points)
