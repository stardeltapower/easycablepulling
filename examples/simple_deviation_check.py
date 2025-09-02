#!/usr/bin/env python3
"""Simple check of how fitted geometry aligns with original route."""

import math
import sys
from pathlib import Path

import numpy as np

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from easycablepulling.core.models import Bend, DuctSpec, Straight
from easycablepulling.geometry.cleaner import PolylineCleaner
from easycablepulling.geometry.fitter import GeometryFitter
from easycablepulling.io import load_route_from_dxf


def analyze_fitting_quality(section, fitter):
    """Analyze how well the fitting preserves the original geometry."""
    original_points = section.original_polyline
    cleaner = PolylineCleaner()
    cleaned_points = cleaner.clean_polyline(original_points)

    # Calculate original properties
    original_length = sum(
        math.sqrt(
            (cleaned_points[i + 1][0] - cleaned_points[i][0]) ** 2
            + (cleaned_points[i + 1][1] - cleaned_points[i][1]) ** 2
        )
        for i in range(len(cleaned_points) - 1)
    )

    # Straight line distance
    straight_distance = math.sqrt(
        (cleaned_points[-1][0] - cleaned_points[0][0]) ** 2
        + (cleaned_points[-1][1] - cleaned_points[0][1]) ** 2
    )

    route_efficiency = (
        straight_distance / original_length if original_length > 0 else 1.0
    )

    # Fit geometry
    result = fitter.fit_polyline(cleaned_points)
    fitted_length = sum(p.length() for p in result.primitives)

    # Analyze primitives
    straights = [p for p in result.primitives if isinstance(p, Straight)]
    bends = [p for p in result.primitives if isinstance(p, Bend)]

    # For straights, check they align with original segments
    straight_alignment_good = True
    for straight in straights:
        # A straight should connect actual points on the original polyline
        # Check if start/end are close to original points
        min_start_dist = min(
            math.sqrt(
                (straight.start_point[0] - p[0]) ** 2
                + (straight.start_point[1] - p[1]) ** 2
            )
            for p in cleaned_points
        )
        min_end_dist = min(
            math.sqrt(
                (straight.end_point[0] - p[0]) ** 2
                + (straight.end_point[1] - p[1]) ** 2
            )
            for p in cleaned_points
        )

        if min_start_dist > 1.0 or min_end_dist > 1.0:  # More than 1m from any point
            straight_alignment_good = False

    # For bends, check radius constraints
    natural_bends = [b for b in bends if b.bend_type == "natural"]
    manufactured_bends = [b for b in bends if b.bend_type == "manufactured"]

    min_bend_radius = min(b.radius_m for b in bends) if bends else float("inf")

    return {
        "original_length": original_length,
        "fitted_length": fitted_length,
        "length_error": abs(fitted_length - original_length) / original_length * 100,
        "straight_distance": straight_distance,
        "route_efficiency": route_efficiency,
        "num_primitives": len(result.primitives),
        "num_straights": len(straights),
        "num_natural_bends": len(natural_bends),
        "num_manufactured_bends": len(manufactured_bends),
        "min_bend_radius": min_bend_radius,
        "straight_alignment_good": straight_alignment_good,
        "total_straight_length": sum(s.length_m for s in straights),
        "total_bend_length": sum(b.length() for b in bends),
    }


def main():
    """Analyze fitting quality for problematic sections."""
    examples_dir = Path(__file__).parent
    input_dxf = examples_dir / "input.dxf"

    route = load_route_from_dxf(input_dxf, "33kV Cable Route")

    # Create duct specification
    duct_spec = DuctSpec(
        inner_diameter=200.0,  # mm
        type="HDPE",
        friction_dry=0.35,
        friction_lubricated=0.15,
        bend_options=[],
    )

    fitter = GeometryFitter(duct_spec=duct_spec)

    print("=" * 80)
    print("FITTING QUALITY ANALYSIS")
    print("=" * 80)
    print(
        "Analyzing how well fitted geometry preserves original route characteristics..."
    )
    print()

    # Focus on problematic sections
    target_sections = ["SECT_01", "SECT_07", "SECT_11", "SECT_13"]

    for section in route.sections:
        if section.id in target_sections and section.original_length > 0:
            print(f"\n{section.id}:")
            print("-" * 50)

            analysis = analyze_fitting_quality(section, fitter)

            print(f"Original length: {analysis['original_length']:.1f}m")
            print(f"Fitted length: {analysis['fitted_length']:.1f}m")
            print(f"Length error: {analysis['length_error']:.2f}%")
            print(
                f"Route efficiency: {analysis['route_efficiency']:.1%} (straight/actual)"
            )
            print()

            print(f"Fitted as:")
            print(
                f"  {analysis['num_straights']} straights ({analysis['total_straight_length']:.1f}m)"
            )
            print(
                f"  {analysis['num_natural_bends']} natural bends ({analysis['total_bend_length']:.1f}m)"
            )
            print(f"  {analysis['num_manufactured_bends']} manufactured bends")

            if analysis["num_straights"] > 0:
                straight_percent = (
                    analysis["total_straight_length"] / analysis["fitted_length"] * 100
                )
                print(f"  Straight sections: {straight_percent:.1f}% of route")

            if analysis["min_bend_radius"] < float("inf"):
                print(f"  Minimum bend radius: {analysis['min_bend_radius']:.1f}m")
                print(f"  Natural threshold: {fitter.natural_bend_threshold:.1f}m")

            # Quality assessment
            if analysis["route_efficiency"] < 0.9:  # Snaking route
                if (
                    analysis["num_primitives"] == 1
                    and analysis["num_natural_bends"] == 1
                ):
                    print("\n⚠️  Complex snaking route fitted as single arc")
                    print("   This preserves length but may not follow exact path")
                elif analysis["num_straights"] > analysis["num_natural_bends"]:
                    print("\n⚠️  Snaking route has more straights than curves")
                    print("   May be creating shortcuts instead of following curves")

            if not analysis["straight_alignment_good"]:
                print("\n❌ Some straight segments don't align with original points")


if __name__ == "__main__":
    main()
