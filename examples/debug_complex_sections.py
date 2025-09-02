#!/usr/bin/env python3
"""Debug why complex sections are failing to fit."""

import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import math

from easycablepulling.core.models import DuctSpec
from easycablepulling.geometry.cleaner import PolylineCleaner
from easycablepulling.geometry.fitter import GeometryFitter
from easycablepulling.io import load_route_from_dxf


def analyze_recursive_fitting(points, fitter, max_depth=5):
    """Analyze recursive fitting behavior."""
    print(f"Analyzing {len(points)} points...")

    # Try fitting the whole segment
    if len(points) < 2:
        print("  Not enough points")
        return

    # Test straight line fit
    line_fit = fitter._fit_straight_line(points)
    if line_fit:
        print(
            f"  Straight fit: length={line_fit['length']:.2f}m, max_error={line_fit['max_error']:.3f}m"
        )
        print(f"  Straight tolerance: {fitter.straight_tolerance:.3f}m")
        print(
            f"  Straight fit acceptable: {line_fit['max_error'] <= fitter.straight_tolerance}"
        )

    # Test arc fit
    arc_fit = fitter._fit_arc(points)
    if arc_fit:
        print(
            f"  Arc fit: radius={arc_fit['radius']:.2f}m, angle={arc_fit['angle']:.1f}°, max_error={arc_fit['max_error']:.3f}m"
        )
        print(f"  Arc tolerance: {fitter.arc_tolerance:.3f}m")
        print(f"  Arc fit acceptable: {arc_fit['max_error'] <= fitter.arc_tolerance}")
        print(f"  Natural bend threshold: {fitter.natural_bend_threshold:.1f}m")
        print(
            f"  Would be natural bend: {arc_fit['radius'] >= fitter.natural_bend_threshold}"
        )

    # Check minimum thresholds
    if line_fit:
        print(f"  Min straight length: {fitter.min_straight_length:.1f}m")
        print(f"  Line long enough: {line_fit['length'] >= fitter.min_straight_length}")

    if arc_fit:
        print(f"  Min arc angle: {fitter.min_arc_angle:.1f}°")
        print(
            f"  Arc angle large enough: {abs(arc_fit['angle']) >= fitter.min_arc_angle}"
        )


def main():
    """Debug complex section fitting."""
    examples_dir = Path(__file__).parent
    input_dxf = examples_dir / "input.dxf"

    print(f"Loading route from {input_dxf}")
    route = load_route_from_dxf(input_dxf, "33kV Cable Route")

    # Create duct specification
    duct_spec = DuctSpec(
        inner_diameter=200.0,  # mm
        type="HDPE",
        friction_dry=0.35,
        friction_lubricated=0.15,
        bend_options=[],
    )

    # Create fitter and cleaner
    fitter = GeometryFitter(duct_spec=duct_spec)
    cleaner = PolylineCleaner()

    print(f"Fitter settings:")
    print(f"  Straight tolerance: {fitter.straight_tolerance:.3f}m")
    print(f"  Arc tolerance: {fitter.arc_tolerance:.3f}m")
    print(f"  Min straight length: {fitter.min_straight_length:.1f}m")
    print(f"  Min arc angle: {fitter.min_arc_angle:.1f}°")
    print(f"  Natural bend threshold: {fitter.natural_bend_threshold:.1f}m")

    # Focus on the problem sections
    problem_sections = ["SECT_03", "SECT_05", "SECT_07", "SECT_11"]

    for section in route.sections:
        if section.id in problem_sections and section.original_length > 0:
            cleaned_points = cleaner.clean_polyline(section.original_polyline)
            analyze_recursive_fitting(cleaned_points, fitter)

            # Try with relaxed tolerances
            print(f"\n  --- Testing with relaxed tolerances ---")
            relaxed_fitter = GeometryFitter(
                straight_tolerance=2.0,  # Much more relaxed
                arc_tolerance=2.0,
                min_straight_length=5.0,
                min_arc_angle=2.0,
                duct_spec=duct_spec,
            )

            result = relaxed_fitter.fit_polyline(cleaned_points)
            print(
                f"  Relaxed fitting: {len(result.primitives)} primitives, success: {result.success}"
            )

            if (
                len(problem_sections) > 2
            ):  # Only analyze first few to avoid too much output
                break


if __name__ == "__main__":
    main()
