"""Accuracy analysis module for route fitting methodologies."""

import math
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from pathlib import Path
import matplotlib.pyplot as plt

from ..core.models import Route, Section, Primitive

# Removed FitterFactory import - not needed


@dataclass
class AccuracyResult:
    """Result of accuracy analysis."""

    section_id: str
    sample_points: int
    max_deviation: float
    min_deviation: float
    avg_deviation: float
    median_deviation: float
    std_deviation: float
    excellent_count: int  # ≤10cm
    good_count: int  # 10-50cm
    acceptable_count: int  # 50cm-1m
    poor_count: int  # >1m


@dataclass
class RouteAccuracyResult:
    """Overall route accuracy analysis result."""

    total_sample_points: int
    global_max_deviation: float
    global_avg_deviation: float
    global_median_deviation: float
    excellent_percentage: float
    good_percentage: float
    acceptable_percentage: float
    poor_percentage: float
    section_results: List[AccuracyResult]


class AccuracyAnalyzer:
    """Analyzer for route fitting accuracy using 25m sampling."""

    def __init__(self, sample_interval: float = 25.0):
        """Initialize accuracy analyzer.

        Args:
            sample_interval: Distance interval for sampling points (meters)
        """
        self.sample_interval = sample_interval
        # Factory not needed in simplified version

    def analyze_route_accuracy(
        self, route: Route, methodology: str = "1_curvature_first"
    ) -> RouteAccuracyResult:
        """Analyze accuracy for complete route using specified methodology.

        Args:
            route: Route to analyze
            methodology: Fitting methodology to use ("direct" uses existing primitives)

        Returns:
            RouteAccuracyResult with comprehensive accuracy statistics
        """
        section_results = []
        all_deviations = []
        total_sample_points = 0

        for section in route.sections:
            try:
                if methodology == "direct":
                    # Use existing primitives in the section
                    primitives = section.primitives
                else:
                    # Apply methodology to section
                    fitter = self.factory.create_fitter(methodology, duct_type="200mm")
                    primitives = fitter.fit_section(section)

                if not primitives:
                    continue

                # Analyze accuracy for this section
                result = self._analyze_section_accuracy(section, primitives)
                section_results.append(result)

                # Collect global statistics
                sample_deviations = self._get_section_deviations(section, primitives)
                all_deviations.extend(sample_deviations)
                total_sample_points += result.sample_points

            except Exception as e:
                print(f"Warning: Section {section.id} failed: {e}")
                continue

        # Calculate global statistics
        if all_deviations:
            global_max = max(all_deviations)
            global_avg = np.mean(all_deviations)
            global_median = np.median(all_deviations)

            # Accuracy categories
            excellent = sum(1 for d in all_deviations if d <= 0.1)
            good = sum(1 for d in all_deviations if 0.1 < d <= 0.5)
            acceptable = sum(1 for d in all_deviations if 0.5 < d <= 1.0)
            poor = sum(1 for d in all_deviations if d > 1.0)

            total = len(all_deviations)

            return RouteAccuracyResult(
                total_sample_points=total_sample_points,
                global_max_deviation=global_max,
                global_avg_deviation=global_avg,
                global_median_deviation=global_median,
                excellent_percentage=excellent / total * 100 if total > 0 else 0,
                good_percentage=good / total * 100 if total > 0 else 0,
                acceptable_percentage=acceptable / total * 100 if total > 0 else 0,
                poor_percentage=poor / total * 100 if total > 0 else 0,
                section_results=section_results,
            )
        else:
            return RouteAccuracyResult(
                total_sample_points=0,
                global_max_deviation=0.0,
                global_avg_deviation=0.0,
                global_median_deviation=0.0,
                excellent_percentage=0.0,
                good_percentage=0.0,
                acceptable_percentage=0.0,
                poor_percentage=0.0,
                section_results=section_results,
            )

    def _analyze_section_accuracy(
        self, section: Section, primitives: List[Primitive]
    ) -> AccuracyResult:
        """Analyze accuracy for a single section."""
        # Sample points along original polyline
        sample_points = self._sample_points_along_polyline(
            section.original_polyline, section.original_length
        )

        if not sample_points:
            return AccuracyResult(
                section_id=section.id,
                sample_points=0,
                max_deviation=0.0,
                min_deviation=0.0,
                avg_deviation=0.0,
                median_deviation=0.0,
                std_deviation=0.0,
                excellent_count=0,
                good_count=0,
                acceptable_count=0,
                poor_count=0,
            )

        # Get fitted polyline
        fitted_polyline = self._get_fitted_polyline_points(primitives)

        # Calculate deviations
        deviations = []
        for _, point in sample_points:
            deviation = self._distance_to_polyline(point, fitted_polyline)
            deviations.append(deviation)

        if deviations:
            # Statistics
            max_dev = max(deviations)
            min_dev = min(deviations)
            avg_dev = np.mean(deviations)
            median_dev = np.median(deviations)
            std_dev = np.std(deviations)

            # Accuracy categories
            excellent = sum(1 for d in deviations if d <= 0.1)
            good = sum(1 for d in deviations if 0.1 < d <= 0.5)
            acceptable = sum(1 for d in deviations if 0.5 < d <= 1.0)
            poor = sum(1 for d in deviations if d > 1.0)

            return AccuracyResult(
                section_id=section.id,
                sample_points=len(sample_points),
                max_deviation=max_dev,
                min_deviation=min_dev,
                avg_deviation=avg_dev,
                median_deviation=median_dev,
                std_deviation=std_dev,
                excellent_count=excellent,
                good_count=good,
                acceptable_count=acceptable,
                poor_count=poor,
            )
        else:
            return AccuracyResult(
                section_id=section.id,
                sample_points=len(sample_points),
                max_deviation=0.0,
                min_deviation=0.0,
                avg_deviation=0.0,
                median_deviation=0.0,
                std_deviation=0.0,
                excellent_count=0,
                good_count=0,
                acceptable_count=0,
                poor_count=0,
            )

    def _get_section_deviations(
        self, section: Section, primitives: List[Primitive]
    ) -> List[float]:
        """Get all deviation measurements for a section."""
        sample_points = self._sample_points_along_polyline(
            section.original_polyline, section.original_length
        )

        if not sample_points:
            return []

        fitted_polyline = self._get_fitted_polyline_points(primitives)
        deviations = []

        for _, point in sample_points:
            deviation = self._distance_to_polyline(point, fitted_polyline)
            deviations.append(deviation)

        return deviations

    def _sample_points_along_polyline(
        self, polyline: List[Tuple[float, float]], total_length: float
    ) -> List[Tuple[float, Tuple[float, float]]]:
        """Sample points every sample_interval meters along polyline."""
        sample_points = []
        distance = 0.0

        while distance <= total_length:
            point = self._interpolate_polyline_at_distance(polyline, distance)
            if point:
                sample_points.append((distance, point))
            distance += self.sample_interval

        return sample_points

    def _interpolate_polyline_at_distance(
        self, polyline: List[Tuple[float, float]], target_distance: float
    ) -> Optional[Tuple[float, float]]:
        """Get point along polyline at specific cumulative distance."""
        if not polyline or target_distance < 0:
            return None

        cumulative_distance = 0.0

        for i in range(len(polyline) - 1):
            p1 = np.array(polyline[i])
            p2 = np.array(polyline[i + 1])

            segment_length = np.linalg.norm(p2 - p1)

            if cumulative_distance + segment_length >= target_distance:
                remaining_distance = target_distance - cumulative_distance
                if segment_length > 1e-6:
                    t = remaining_distance / segment_length
                    interpolated_point = p1 + t * (p2 - p1)
                    return tuple(interpolated_point)
                else:
                    return tuple(p1)

            cumulative_distance += segment_length

        return tuple(polyline[-1]) if polyline else None

    def _get_fitted_polyline_points(
        self, primitives: List[Primitive]
    ) -> List[Tuple[float, float]]:
        """Extract all points from fitted primitives to create continuous polyline."""
        fitted_points = []

        for primitive in primitives:
            if hasattr(primitive, "start_point") and hasattr(primitive, "end_point"):
                if not fitted_points:
                    fitted_points.append(primitive.start_point)
                fitted_points.append(primitive.end_point)

        return fitted_points

    def _distance_to_polyline(
        self, point: Tuple[float, float], polyline: List[Tuple[float, float]]
    ) -> float:
        """Calculate minimum distance from point to polyline."""
        if not polyline:
            return float("inf")

        min_distance = float("inf")
        pt = np.array(point)

        for i in range(len(polyline) - 1):
            p1 = np.array(polyline[i])
            p2 = np.array(polyline[i + 1])

            distance = self._point_to_line_segment_distance(pt, p1, p2)
            min_distance = min(min_distance, distance)

        return min_distance

    def _point_to_line_segment_distance(
        self, point: np.ndarray, line_start: np.ndarray, line_end: np.ndarray
    ) -> float:
        """Calculate distance from point to line segment."""
        line_vec = line_end - line_start
        line_len = np.linalg.norm(line_vec)

        if line_len < 1e-6:
            return np.linalg.norm(point - line_start)

        line_unit = line_vec / line_len
        point_vec = point - line_start
        projection_length = np.dot(point_vec, line_unit)
        projection_length = max(0, min(line_len, projection_length))
        closest_point = line_start + projection_length * line_unit

        return np.linalg.norm(point - closest_point)

    def create_accuracy_visualization(
        self, result: RouteAccuracyResult, output_path: Path, route_name: str = "Route"
    ) -> None:
        """Create visualization plots for accuracy analysis."""
        if not result.section_results:
            print("No data available for visualization")
            return

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        # Plot 1: Deviation along route
        cumulative_distance = 0
        section_distances = []
        section_max_devs = []
        section_median_devs = []

        for section_result in result.section_results:
            section_distances.append(cumulative_distance)
            section_max_devs.append(section_result.max_deviation)
            section_median_devs.append(section_result.median_deviation)

            # Estimate section length (simplified)
            cumulative_distance += section_result.sample_points * self.sample_interval

        ax1.plot(
            section_distances,
            section_max_devs,
            "o-",
            label="Max Deviation",
            color="red",
            linewidth=1.5,
            markersize=4,
        )
        ax1.plot(
            section_distances,
            section_median_devs,
            "o-",
            label="Median Deviation",
            color="blue",
            linewidth=1.5,
            markersize=4,
        )

        # Reference lines
        ax1.axhline(
            y=0.1, color="green", linestyle="--", alpha=0.5, label="10cm (Excellent)"
        )
        ax1.axhline(
            y=0.5, color="orange", linestyle="--", alpha=0.5, label="50cm (Good)"
        )
        ax1.axhline(
            y=1.0, color="red", linestyle="--", alpha=0.5, label="1m (Acceptable)"
        )

        ax1.set_xlabel("Distance Along Route (m)")
        ax1.set_ylabel("Lateral Deviation (m)")
        ax1.set_title(
            f"{route_name} - Lateral Deviation Analysis (Every {self.sample_interval}m)"
        )
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Plot 2: Accuracy category pie chart
        categories = [
            "Excellent (≤10cm)",
            "Good (10-50cm)",
            "Acceptable (50cm-1m)",
            "Poor (>1m)",
        ]
        percentages = [
            result.excellent_percentage,
            result.good_percentage,
            result.acceptable_percentage,
            result.poor_percentage,
        ]
        colors = ["green", "orange", "yellow", "red"]

        # Filter out zero values
        non_zero_data = [
            (cat, pct, col)
            for cat, pct, col in zip(categories, percentages, colors)
            if pct > 0
        ]
        if non_zero_data:
            cats, pcts, cols = zip(*non_zero_data)

            ax2.pie(pcts, labels=cats, colors=cols, autopct="%1.1f%%", startangle=90)
            ax2.set_title(
                f"Accuracy Distribution ({result.total_sample_points} sample points)\n"
                f"Median: {result.global_median_deviation*100:.1f}cm, "
                f"Average: {result.global_avg_deviation*100:.1f}cm"
            )

        plt.tight_layout()

        # Save plot
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

    def print_accuracy_summary(
        self, result: RouteAccuracyResult, route_name: str = "Route"
    ) -> None:
        """Print formatted accuracy summary."""
        print(f"\n📊 ACCURACY ANALYSIS SUMMARY - {route_name}")
        print("=" * 60)

        print(f"📏 Sample interval: {self.sample_interval}m")
        print(f"📏 Total sample points: {result.total_sample_points}")
        print(f"🎯 Global statistics:")
        print(f"   Max deviation: {result.global_max_deviation:.3f}m")
        print(f"   Average deviation: {result.global_avg_deviation:.3f}m")
        print(f"   Median deviation: {result.global_median_deviation:.3f}m")

        print(f"\n🎯 Accuracy Categories:")
        print(f"   Excellent (≤10cm): {result.excellent_percentage:.1f}%")
        print(f"   Good (10-50cm): {result.good_percentage:.1f}%")
        print(f"   Acceptable (50cm-1m): {result.acceptable_percentage:.1f}%")
        print(f"   Poor (>1m): {result.poor_percentage:.1f}%")

        print(f"\n📋 Section Breakdown:")
        print(f"{'Section':<10} {'Samples':<7} {'Max':<6} {'Med':<6} {'Quality'}")
        print(f"{'-'*10} {'-'*7} {'-'*6} {'-'*6} {'-'*10}")

        for section_result in result.section_results:
            if section_result.sample_points > 0:
                quality = (
                    "Excellent"
                    if section_result.median_deviation <= 0.1
                    else (
                        "Good"
                        if section_result.median_deviation <= 0.5
                        else (
                            "Acceptable"
                            if section_result.median_deviation <= 1.0
                            else "Poor"
                        )
                    )
                )

                print(
                    f"{section_result.section_id:<10} {section_result.sample_points:<7} "
                    f"{section_result.max_deviation:<6.2f} {section_result.median_deviation:<6.3f} "
                    f"{quality}"
                )
