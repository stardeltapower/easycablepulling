#!/usr/bin/env python3
"""Check deviation statistics for all fitted routes - simple version."""

import math
import sys
from pathlib import Path
from typing import List, Tuple

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from easycablepulling.core.models import DuctSpec
from easycablepulling.geometry.cleaner import PolylineCleaner
from easycablepulling.geometry.fitter import GeometryFitter
from easycablepulling.io import load_route_from_dxf


def calculate_point_to_segment_distance(
    point: Tuple[float, float],
    seg_start: Tuple[float, float],
    seg_end: Tuple[float, float],
) -> float:
    """Calculate perpendicular distance from point to line segment."""
    x0, y0 = point
    x1, y1 = seg_start
    x2, y2 = seg_end

    dx = x2 - x1
    dy = y2 - y1

    if dx == 0 and dy == 0:
        # Segment is a point
        return math.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)

    # Parameter t of closest point on segment
    t = ((x0 - x1) * dx + (y0 - y1) * dy) / (dx**2 + dy**2)
    t = max(0, min(1, t))  # Clamp to segment

    # Closest point on segment
    xc = x1 + t * dx
    yc = y1 + t * dy

    # Distance to closest point
    return math.sqrt((x0 - xc) ** 2 + (y0 - yc) ** 2)


def main():
    examples_dir = Path(__file__).parent
    input_dxf = examples_dir / "input.dxf"

    route = load_route_from_dxf(input_dxf, "33kV Cable Route")

    # Create duct specification for 200mm HDPE
    duct_spec = DuctSpec(
        inner_diameter=200.0,
        type="HDPE",
        friction_dry=0.35,
        friction_lubricated=0.15,
        bend_options=[],
    )

    fitter = GeometryFitter(duct_spec=duct_spec)
    cleaner = PolylineCleaner()

    print("=" * 80)
    print("DEVIATION ANALYSIS FOR ALL SECTIONS")
    print("=" * 80)

    total_points_over_1m = 0
    total_points = 0
    worst_deviation = 0.0
    worst_section = None

    for section in route.sections:
        if section.original_length <= 0:
            continue

        original_points = section.original_polyline
        cleaned_points = cleaner.clean_polyline(original_points)

        # Fit geometry
        result = fitter.fit_polyline(cleaned_points)
        fitted_points = result.fitted_points

        if not fitted_points or len(fitted_points) < 2:
            continue

        # Calculate deviations
        deviations = []
        points_over_1m = 0

        for orig_point in cleaned_points:
            min_dist = float("inf")

            # Check distance to each fitted segment
            for i in range(len(fitted_points) - 1):
                dist = calculate_point_to_segment_distance(
                    orig_point, fitted_points[i], fitted_points[i + 1]
                )
                min_dist = min(min_dist, dist)

            deviations.append(min_dist)
            if min_dist > 1.0:
                points_over_1m += 1

        max_dev = max(deviations) if deviations else 0.0
        avg_dev = sum(deviations) / len(deviations) if deviations else 0.0

        print(f"\n{section.id}:")
        print(f"  Points analyzed: {len(deviations)}")
        print(f"  Max deviation: {max_dev:.3f}m")
        print(f"  Average deviation: {avg_dev:.3f}m")
        print(
            f"  Points >1m: {points_over_1m} ({points_over_1m/len(deviations)*100:.1f}%)"
        )

        total_points_over_1m += points_over_1m
        total_points += len(deviations)

        if max_dev > worst_deviation:
            worst_deviation = max_dev
            worst_section = section.id

    print("\n" + "=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)
    print(f"Total points analyzed: {total_points:,}")
    print(
        f"Points >1m: {total_points_over_1m:,} ({total_points_over_1m/total_points*100:.1f}%)"
    )
    print(f"Worst deviation: {worst_deviation:.3f}m in {worst_section}")

    print("\n" + "=" * 80)
    print("WHAT IT WOULD TAKE TO GET ALL WITHIN 1m:")
    print("=" * 80)

    if total_points_over_1m == 0:
        print("✅ SUCCESS: All deviations already within 1m!")
    else:
        percent_over_1m = total_points_over_1m / total_points * 100
        print(f"Currently {percent_over_1m:.1f}% of points exceed 1m deviation")
        print(f"\nTo achieve all deviations within 1m, we would need to:")
        print(
            f"1. Further reduce segment length in _improve_poor_path_following to 0.5m"
        )
        print(f"2. Use even tighter polynomial fitting tolerance (0.1m)")
        print(
            f"3. Add iterative refinement that specifically targets segments with >1m deviation"
        )
        print(f"4. Consider using spline interpolation for ultra-smooth path following")
        print(f"\nHowever, this would result in:")
        print(f"- Many more primitives (potentially 2-3x current count)")
        print(f"- More complex construction documentation")
        print(f"- Potential over-fitting to CAD imperfections")
        print(
            f"\nCurrent {100-percent_over_1m:.1f}% within 1m is excellent for practical construction."
        )


if __name__ == "__main__":
    main()
