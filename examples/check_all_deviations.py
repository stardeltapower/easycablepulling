#!/usr/bin/env python3
"""Check deviation statistics for all fitted routes."""

import json
import math

# Add parent directory to path for imports
import sys
from pathlib import Path
from typing import List, Tuple

sys.path.append(str(Path(__file__).parent.parent))

from easycablepulling.core.models import DuctSpec
from easycablepulling.geometry.fitter import GeometryFitter
from easycablepulling.geometry.processor import GeometryProcessor


def calculate_lateral_deviation(
    fitted_points: List[Tuple[float, float]], original_points: List[Tuple[float, float]]
) -> dict:
    """Calculate lateral deviation statistics."""
    if not fitted_points or not original_points:
        return {
            "max": 0.0,
            "percentile_99": 0.0,
            "percentile_95": 0.0,
            "worst_sections": [],
        }

    deviations = []

    # For each original point, find minimum distance to fitted route
    for orig_point in original_points:
        min_dist = float("inf")

        # Check distance to each fitted segment
        for i in range(len(fitted_points) - 1):
            p1 = fitted_points[i]
            p2 = fitted_points[i + 1]

            # Calculate perpendicular distance to line segment
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]

            segment_length = math.sqrt(dx * dx + dy * dy)
            if segment_length < 1e-6:
                # Degenerate segment
                dist = math.sqrt(
                    (orig_point[0] - p1[0]) ** 2 + (orig_point[1] - p1[1]) ** 2
                )
            else:
                # Project point onto line
                t = ((orig_point[0] - p1[0]) * dx + (orig_point[1] - p1[1]) * dy) / (
                    segment_length**2
                )
                t = max(0, min(1, t))  # Clamp to segment

                # Closest point on segment
                closest_x = p1[0] + t * dx
                closest_y = p1[1] + t * dy

                # Distance to closest point
                dist = math.sqrt(
                    (orig_point[0] - closest_x) ** 2 + (orig_point[1] - closest_y) ** 2
                )

            min_dist = min(min_dist, dist)

        deviations.append(min_dist)

    # Sort deviations for percentile calculation
    deviations.sort()

    # Calculate statistics
    max_dev = max(deviations) if deviations else 0.0
    p95_idx = int(len(deviations) * 0.95)
    p99_idx = int(len(deviations) * 0.99)

    # Find sections with worst deviations
    worst_sections = []
    for i, dev in enumerate(deviations):
        if dev > 1.0:  # Find all points > 1m
            worst_sections.append((i, dev))

    worst_sections.sort(key=lambda x: x[1], reverse=True)
    worst_sections = worst_sections[:10]  # Top 10 worst

    return {
        "max": max_dev,
        "percentile_99": deviations[p99_idx] if p99_idx < len(deviations) else max_dev,
        "percentile_95": deviations[p95_idx] if p95_idx < len(deviations) else max_dev,
        "num_over_1m": sum(1 for d in deviations if d > 1.0),
        "num_over_2m": sum(1 for d in deviations if d > 2.0),
        "total_points": len(deviations),
        "worst_sections": worst_sections,
    }


def main():
    # Load configuration
    with open("input.dxf", "rb") as f:
        dxf_data = f.read()

    # Create processor with 200mm duct spec
    duct_spec = DuctSpec(
        inner_diameter=200.0, type="HDPE", friction_dry=0.5, friction_lubricated=0.2
    )

    processor = GeometryProcessor()

    # Parse DXF and identify sections
    result = processor.identify_cable_sections(dxf_data)

    if not result["success"]:
        print(f"Error: {result['message']}")
        return

    sections = result["sections"]
    print(f"Found {len(sections)} cable sections")
    print("=" * 80)
    print("DEVIATION ANALYSIS FOR ALL SECTIONS")
    print("=" * 80)

    # Create geometry fitter with duct spec
    fitter = GeometryFitter(duct_spec=duct_spec)

    total_over_1m = 0
    total_over_2m = 0
    total_points = 0
    worst_overall = 0.0
    worst_section = None

    for section_name, points in sections.items():
        # Fit geometry to section
        fit_result = fitter.fit_polyline(points)

        if not fit_result.success:
            print(f"\n{section_name}: FAILED - {fit_result.message}")
            continue

        # Calculate deviation stats
        stats = calculate_lateral_deviation(fit_result.fitted_points, points)

        print(f"\n{section_name}:")
        print(f"  Max deviation: {stats['max']:.3f}m")
        print(f"  99th percentile: {stats['percentile_99']:.3f}m")
        print(f"  95th percentile: {stats['percentile_95']:.3f}m")
        print(
            f"  Points >1m: {stats['num_over_1m']}/{stats['total_points']} ({stats['num_over_1m']/stats['total_points']*100:.1f}%)"
        )
        print(
            f"  Points >2m: {stats['num_over_2m']}/{stats['total_points']} ({stats['num_over_2m']/stats['total_points']*100:.1f}%)"
        )

        total_over_1m += stats["num_over_1m"]
        total_over_2m += stats["num_over_2m"]
        total_points += stats["total_points"]

        if stats["max"] > worst_overall:
            worst_overall = stats["max"]
            worst_section = section_name

    print("\n" + "=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)
    print(f"Total points analyzed: {total_points:,}")
    print(f"Points >1m: {total_over_1m:,} ({total_over_1m/total_points*100:.1f}%)")
    print(f"Points >2m: {total_over_2m:,} ({total_over_2m/total_points*100:.1f}%)")
    print(f"Worst deviation: {worst_overall:.3f}m in {worst_section}")

    print("\n" + "=" * 80)
    print("TARGET STATUS")
    print("=" * 80)
    if total_over_1m == 0:
        print("✅ SUCCESS: All deviations within 1m!")
    else:
        print(
            f"❌ {total_over_1m} points ({total_over_1m/total_points*100:.1f}%) still exceed 1m"
        )
        print(f"   Need to improve {total_over_1m} points to achieve target")


if __name__ == "__main__":
    main()
