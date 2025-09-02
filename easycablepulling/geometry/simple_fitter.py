"""Simplified geometry fitting for initial implementation."""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, cast

import numpy as np

from ..config import GEOMETRY_TOLERANCES
from ..core.models import Bend, Primitive, Straight


@dataclass
class SimpleFittingResult:
    """Result of simplified geometry fitting."""

    primitives: List[Primitive]
    total_error: float
    max_error: float
    success: bool
    message: str = ""


class SimpleGeometryFitter:
    """Simplified geometry fitter for initial Phase 3 implementation."""

    def __init__(self, straight_tolerance: float = 2.0):
        """Initialize simple fitter.

        Args:
            straight_tolerance: Tolerance for straight line fitting (meters)
        """
        self.straight_tolerance = straight_tolerance

    def fit_polyline_as_straights(
        self, points: List[Tuple[float, float]]
    ) -> SimpleFittingResult:
        """Fit polyline as series of straight segments.

        This is a simplified approach for Phase 3 - fit each pair of consecutive
        points as straight segments.

        Args:
            points: List of (x, y) coordinate tuples

        Returns:
            SimpleFittingResult with straight line primitives
        """
        if len(points) < 2:
            return SimpleFittingResult(
                primitives=[],
                total_error=0.0,
                max_error=0.0,
                success=False,
                message="Insufficient points",
            )

        primitives = []

        # Create straight segments between consecutive points
        for i in range(len(points) - 1):
            start = points[i]
            end = points[i + 1]

            # Calculate length
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            length = math.sqrt(dx * dx + dy * dy)

            # Only add if length is meaningful
            if length > 0.1:  # 10cm minimum
                straight = Straight(length_m=length, start_point=start, end_point=end)
                primitives.append(straight)

        # For simplified fitting, error is minimal since we're using original points
        return SimpleFittingResult(
            primitives=cast(List[Primitive], primitives),
            total_error=0.0,
            max_error=0.0,
            success=True,
            message=f"Fitted {len(primitives)} straight segments",
        )

    def fit_section_simplified(
        self, points: List[Tuple[float, float]]
    ) -> SimpleFittingResult:
        """Fit a section as a single straight line if possible, otherwise as segments.

        Args:
            points: List of (x, y) coordinate tuples

        Returns:
            SimpleFittingResult with primitives
        """
        if len(points) < 2:
            return SimpleFittingResult(
                primitives=[],
                total_error=0.0,
                max_error=0.0,
                success=False,
                message="Insufficient points",
            )

        # Try fitting as single straight line
        start_point = points[0]
        end_point = points[-1]

        # Calculate direct length
        dx = end_point[0] - start_point[0]
        dy = end_point[1] - start_point[1]
        direct_length = math.sqrt(dx * dx + dy * dy)

        # Check maximum deviation from straight line
        max_deviation = self._calculate_max_deviation_from_line(
            points, start_point, end_point
        )

        if max_deviation <= self.straight_tolerance:
            # Fit as single straight line
            return SimpleFittingResult(
                primitives=[
                    Straight(
                        length_m=direct_length,
                        start_point=start_point,
                        end_point=end_point,
                    )
                ],
                total_error=max_deviation,
                max_error=max_deviation,
                success=True,
                message="Fitted as single straight line",
            )
        else:
            # Fit as multiple segments
            return self.fit_polyline_as_straights(points)

    def _calculate_max_deviation_from_line(
        self,
        points: List[Tuple[float, float]],
        start: Tuple[float, float],
        end: Tuple[float, float],
    ) -> float:
        """Calculate maximum perpendicular distance from points to line."""
        if len(points) <= 2:
            return 0.0

        # Line vector
        line_vec = np.array([end[0] - start[0], end[1] - start[1]])
        line_length = np.linalg.norm(line_vec)

        if line_length < 1e-6:
            return 0.0

        line_unit = line_vec / line_length
        line_start = np.array(start)

        max_distance = 0.0

        for point in points[1:-1]:  # Skip start and end points
            p = np.array(point)

            # Project point onto line
            t = np.dot(p - line_start, line_unit)
            t = np.clip(t, 0, line_length)  # Clamp to line segment

            # Find closest point on line
            closest = line_start + t * line_unit

            # Calculate perpendicular distance
            distance = np.linalg.norm(p - closest)
            max_distance = float(max(max_distance, distance))

        return max_distance
