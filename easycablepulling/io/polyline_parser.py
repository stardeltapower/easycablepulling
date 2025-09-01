"""Polyline parsing and section identification."""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

from ..core.models import Section

logger = logging.getLogger(__name__)


class PolylineParser:
    """Parser for converting polylines into route sections."""

    def __init__(self, joint_detection_distance: float = 50.0):
        """Initialize parser.

        Args:
            joint_detection_distance: Distance threshold for detecting joints/pits
                (meters)
        """
        self.joint_detection_distance = joint_detection_distance

    def identify_sections(
        self, polylines: List[Tuple[str, List[Tuple[float, float]]]]
    ) -> List[Section]:
        """Identify sections from a list of polylines.

        Based on user clarification: expect one layer with one or multiple polylines
        that form a single route when combined.

        Args:
            polylines: List of (layer_name, points) tuples

        Returns:
            List of Section objects
        """
        if not polylines:
            raise ValueError("No polylines provided")

        # If single polyline, split at potential joint locations
        if len(polylines) == 1:
            layer_name, points = polylines[0]
            return self._split_polyline_at_joints(points)

        # Multiple polylines - each becomes a section
        sections = []
        for i, (layer_name, points) in enumerate(polylines):
            section_id = self._generate_section_id(i, len(polylines))

            section = Section(id=section_id, original_polyline=points)
            sections.append(section)
            logger.info(
                f"Created section {section_id} from polyline with {len(points)} points"
            )

        return sections

    def _split_polyline_at_joints(
        self, points: List[Tuple[float, float]]
    ) -> List[Section]:
        """Split a single polyline at potential joint/pit locations.

        Joint detection based on:
        - Significant direction changes (> 45 degrees)
        - Distance-based splitting for very long sections

        Args:
            points: List of (x, y) coordinates

        Returns:
            List of Section objects
        """
        if len(points) < 2:
            raise ValueError("Polyline must have at least 2 points")

        sections = []
        joint_indices = self._find_joint_locations(points)

        # Add start and end indices
        all_indices = [0] + joint_indices + [len(points) - 1]
        all_indices = sorted(set(all_indices))  # Remove duplicates and sort

        # Create sections between joints
        for i in range(len(all_indices) - 1):
            start_idx = all_indices[i]
            end_idx = all_indices[i + 1]

            section_points = points[start_idx : end_idx + 1]
            section_id = self._generate_section_id(i, len(all_indices) - 1)

            section = Section(id=section_id, original_polyline=section_points)
            sections.append(section)
            logger.info(
                f"Created section {section_id} from points {start_idx} to {end_idx}"
            )

        return sections

    def _find_joint_locations(self, points: List[Tuple[float, float]]) -> List[int]:
        """Find potential joint/pit locations in polyline.

        Args:
            points: List of (x, y) coordinates

        Returns:
            List of point indices where joints should be placed
        """
        joint_indices: List[int] = []

        if len(points) < 3:
            return joint_indices

        # Look for significant direction changes
        for i in range(1, len(points) - 1):
            prev_point = points[i - 1]
            curr_point = points[i]
            next_point = points[i + 1]

            # Calculate angle change at this point
            angle_change = self._calculate_angle_change(
                prev_point, curr_point, next_point
            )

            # If angle change is significant (> 45 degrees), consider it a joint
            if abs(angle_change) > 45.0:
                joint_indices.append(i)
                logger.debug(
                    f"Found joint at point {i} with angle change {angle_change:.1f}°"
                )

        # Also split based on distance if sections would be too long
        distance_joints = self._find_distance_based_joints(points)
        joint_indices.extend(distance_joints)

        return sorted(set(joint_indices))

    def _calculate_angle_change(
        self, p1: Tuple[float, float], p2: Tuple[float, float], p3: Tuple[float, float]
    ) -> float:
        """Calculate angle change at point p2 between vectors p1->p2 and p2->p3.

        Args:
            p1, p2, p3: Three consecutive points

        Returns:
            Angle change in degrees (-180 to 180)
        """
        # Vector from p1 to p2
        v1 = (p2[0] - p1[0], p2[1] - p1[1])
        # Vector from p2 to p3
        v2 = (p3[0] - p2[0], p3[1] - p2[1])

        # Calculate angles
        angle1 = math.atan2(v1[1], v1[0])
        angle2 = math.atan2(v2[1], v2[0])

        # Calculate angle change
        angle_change = angle2 - angle1

        # Normalize to [-π, π]
        while angle_change > math.pi:
            angle_change -= 2 * math.pi
        while angle_change < -math.pi:
            angle_change += 2 * math.pi

        return math.degrees(angle_change)

    def _find_distance_based_joints(
        self, points: List[Tuple[float, float]]
    ) -> List[int]:
        """Find joint locations based on distance thresholds.

        Args:
            points: List of (x, y) coordinates

        Returns:
            List of point indices for distance-based joints
        """
        joint_indices: List[int] = []
        cumulative_distance = 0.0

        for i in range(1, len(points)):
            prev_point = points[i - 1]
            curr_point = points[i]

            # Calculate distance
            dx = curr_point[0] - prev_point[0]
            dy = curr_point[1] - prev_point[1]
            distance = math.sqrt(dx**2 + dy**2)
            cumulative_distance += distance

            # If we've traveled the joint detection distance, mark as potential joint
            if cumulative_distance >= self.joint_detection_distance:
                joint_indices.append(i)
                cumulative_distance = 0.0
                logger.debug(f"Distance-based joint at point {i}")

        return joint_indices

    def _generate_section_id(self, index: int, total_sections: int) -> str:
        """Generate section ID based on alphabetical sequence.

        Args:
            index: Section index (0-based)
            total_sections: Total number of sections

        Returns:
            Section ID (e.g., "AB", "BC", "CD", etc.)
        """
        if total_sections <= 26:
            # Simple A-B, B-C, C-D naming
            start_char = chr(ord("A") + index)
            end_char = chr(ord("A") + index + 1)
            return f"{start_char}{end_char}"
        else:
            # Fallback to numbered sections for many sections
            return f"SECT_{index+1:02d}_{index+2:02d}"
