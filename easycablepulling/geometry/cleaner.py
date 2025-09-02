"""Polyline cleaning and preprocessing utilities."""

import math
from typing import List, Optional, Tuple

import numpy as np
from shapely.geometry import LineString, Point


class PolylineCleaner:
    """Clean and simplify polylines for geometry fitting."""

    def __init__(
        self,
        duplicate_tolerance: float = 0.001,
        simplify_tolerance: float = 0.1,
        min_segment_length: float = 0.01,
    ):
        """Initialize polyline cleaner.

        Args:
            duplicate_tolerance: Distance threshold for duplicate points (meters)
            simplify_tolerance: Tolerance for Douglas-Peucker simplification (meters)
            min_segment_length: Minimum segment length to retain (meters)
        """
        self.duplicate_tolerance = duplicate_tolerance
        self.simplify_tolerance = simplify_tolerance
        self.min_segment_length = min_segment_length

    def clean_polyline(
        self, points: List[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        """Clean a polyline by removing duplicates and invalid segments.

        Args:
            points: List of (x, y) coordinate tuples

        Returns:
            Cleaned list of points
        """
        if len(points) < 2:
            return points

        # Remove duplicate points
        cleaned = self._remove_duplicates(points)

        # Remove collinear points
        cleaned = self._remove_collinear_points(cleaned)

        # Remove short segments
        cleaned = self._remove_short_segments(cleaned)

        return cleaned

    def simplify_polyline(
        self, points: List[Tuple[float, float]], tolerance: Optional[float] = None
    ) -> List[Tuple[float, float]]:
        """Simplify polyline using Douglas-Peucker algorithm.

        Args:
            points: List of (x, y) coordinate tuples
            tolerance: Simplification tolerance (uses default if None)

        Returns:
            Simplified list of points
        """
        if len(points) < 3:
            return points

        tolerance = tolerance or self.simplify_tolerance

        # Create LineString and simplify
        line = LineString(points)
        simplified = line.simplify(tolerance, preserve_topology=True)

        # Extract coordinates
        if simplified.geom_type == "LineString":
            return list(simplified.coords)
        else:
            # Handle case where simplification creates MultiLineString
            return list(simplified.geoms[0].coords)

    def _remove_duplicates(
        self, points: List[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        """Remove duplicate or very close points."""
        if not points:
            return points

        cleaned = [points[0]]

        for point in points[1:]:
            last_point = cleaned[-1]
            distance = self._point_distance(last_point, point)

            if distance > self.duplicate_tolerance:
                cleaned.append(point)

        return cleaned

    def _remove_collinear_points(
        self, points: List[Tuple[float, float]], tolerance: float = 0.001
    ) -> List[Tuple[float, float]]:
        """Remove points that are collinear with their neighbors."""
        if len(points) < 3:
            return points

        cleaned = [points[0]]

        for i in range(1, len(points) - 1):
            p1 = points[i - 1]
            p2 = points[i]
            p3 = points[i + 1]

            # Check if p2 is on the line between p1 and p3
            if not self._is_collinear(p1, p2, p3, tolerance):
                cleaned.append(p2)

        cleaned.append(points[-1])
        return cleaned

    def _remove_short_segments(
        self, points: List[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        """Remove segments shorter than minimum length."""
        if len(points) < 2:
            return points

        cleaned = [points[0]]

        for point in points[1:]:
            distance = self._point_distance(cleaned[-1], point)

            if distance >= self.min_segment_length:
                cleaned.append(point)
            elif point == points[-1]:
                # Always keep the last point
                cleaned.append(point)

        return cleaned

    def _point_distance(
        self, p1: Tuple[float, float], p2: Tuple[float, float]
    ) -> float:
        """Calculate Euclidean distance between two points."""
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        return math.sqrt(dx * dx + dy * dy)

    def _is_collinear(
        self,
        p1: Tuple[float, float],
        p2: Tuple[float, float],
        p3: Tuple[float, float],
        tolerance: float,
    ) -> bool:
        """Check if three points are collinear within tolerance."""
        # Calculate cross product (area of triangle)
        cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])

        # Calculate lengths
        len1 = self._point_distance(p1, p2)
        len2 = self._point_distance(p2, p3)
        len3 = self._point_distance(p1, p3)

        # Check if area is small relative to perimeter
        perimeter = len1 + len2 + len3
        if perimeter > 0:
            return abs(cross) / perimeter < tolerance
        else:
            return True

    def smooth_polyline(
        self, points: List[Tuple[float, float]], window_size: int = 3
    ) -> List[Tuple[float, float]]:
        """Apply moving average smoothing to polyline.

        Args:
            points: List of (x, y) coordinate tuples
            window_size: Size of smoothing window (must be odd)

        Returns:
            Smoothed list of points
        """
        if len(points) < window_size or window_size < 3:
            return points

        # Ensure window size is odd
        if window_size % 2 == 0:
            window_size += 1

        # Convert to numpy array
        arr = np.array(points)

        # Apply smoothing
        smoothed = np.copy(arr)
        half_window = window_size // 2

        for i in range(half_window, len(arr) - half_window):
            window = arr[i - half_window : i + half_window + 1]
            smoothed[i] = np.mean(window, axis=0)

        # Keep endpoints fixed
        smoothed[0] = arr[0]
        smoothed[-1] = arr[-1]

        return [tuple(p) for p in smoothed]

    def resample_polyline(
        self, points: List[Tuple[float, float]], target_spacing: float
    ) -> List[Tuple[float, float]]:
        """Resample polyline with uniform point spacing.

        Args:
            points: List of (x, y) coordinate tuples
            target_spacing: Target distance between points (meters)

        Returns:
            Resampled list of points
        """
        if len(points) < 2:
            return points

        resampled = [points[0]]
        accumulated_dist = 0.0

        for i in range(1, len(points)):
            p1 = resampled[-1]
            p2 = points[i]
            segment_dist = self._point_distance(p1, p2)

            if accumulated_dist + segment_dist >= target_spacing:
                # Add interpolated points
                remaining = target_spacing - accumulated_dist

                while remaining < segment_dist:
                    t = remaining / segment_dist
                    x = p1[0] + t * (p2[0] - p1[0])
                    y = p1[1] + t * (p2[1] - p1[1])
                    resampled.append((x, y))

                    remaining += target_spacing

                accumulated_dist = segment_dist - (remaining - target_spacing)
            else:
                accumulated_dist += segment_dist

        # Always include the last point
        if resampled[-1] != points[-1]:
            resampled.append(points[-1])

        return resampled
