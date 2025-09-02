#!/usr/bin/env python3
"""Debug why 6-point segments aren't being fitted as arcs."""

import math
import sys
from pathlib import Path

import numpy as np

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from easycablepulling.core.models import DuctSpec
from easycablepulling.geometry.cleaner import PolylineCleaner
from easycablepulling.geometry.fitter import GeometryFitter
from easycablepulling.io import load_route_from_dxf


def debug_segment_fitting():
    """Debug why 6-point segments create straights instead of arcs."""
    examples_dir = Path(__file__).parent
    input_dxf = examples_dir / "input.dxf"

    route = load_route_from_dxf(input_dxf, "33kV Cable Route")

    # Find SECT_07
    sect07 = None
    for section in route.sections:
        if section.id == "SECT_07":
            sect07 = section
            break

    if not sect07:
        print("SECT_07 not found")
        return

    # Create duct specification
    duct_spec = DuctSpec(
        inner_diameter=200.0,  # mm
        type="HDPE",
        friction_dry=0.35,
        friction_lubricated=0.15,
        bend_options=[],
    )

    fitter = GeometryFitter(duct_spec=duct_spec)

    original_points = sect07.original_polyline
    cleaner = PolylineCleaner()
    cleaned_points = cleaner.clean_polyline(original_points)

    print(f"SECT_07 Debug Analysis:")
    print(f"Natural bend threshold: {fitter.natural_bend_threshold:.1f}m")
    print(f"Arc tolerance: {fitter.arc_tolerance:.3f}m")
    print(f"Strict arc tolerance: {min(0.3, fitter.arc_tolerance):.3f}m")
    print()

    # Test first 7 points (like first straight segment)
    test_points = cleaned_points[0:7]  # 7 points = 6 segments

    print(f"Testing first 7 points:")
    for i, point in enumerate(test_points):
        print(f"  Point {i}: ({point[0]:.1f}, {point[1]:.1f})")

    # Calculate actual segment length
    actual_length = sum(
        math.sqrt(
            (test_points[i + 1][0] - test_points[i][0]) ** 2
            + (test_points[i + 1][1] - test_points[i][1]) ** 2
        )
        for i in range(len(test_points) - 1)
    )

    # Test arc fitting
    arc_fit = fitter._fit_arc(test_points)

    print(f"\nArc fitting results:")
    if arc_fit:
        print(f"  Radius: {arc_fit['radius']:.1f}m")
        print(f"  Angle: {arc_fit['angle']:.2f}°")
        print(f"  Max error: {arc_fit['max_error']:.3f}m")
        print(
            f"  Arc length: {abs(math.radians(arc_fit['angle'])) * arc_fit['radius']:.1f}m"
        )
        print(f"  Original length: {actual_length:.1f}m")

        # Check against thresholds
        print(f"\nThreshold checks:")
        print(
            f"  Radius >= natural threshold ({fitter.natural_bend_threshold:.1f}m): {arc_fit['radius'] >= fitter.natural_bend_threshold}"
        )
        print(
            f"  Angle >= min angle ({fitter.min_arc_angle:.1f}°): {abs(arc_fit['angle']) >= fitter.min_arc_angle}"
        )
        print(f"  Error <= strict tolerance (0.3m): {arc_fit['max_error'] <= 0.3}")
        print(
            f"  Error <= arc tolerance ({fitter.arc_tolerance:.3f}m): {arc_fit['max_error'] <= fitter.arc_tolerance}"
        )

        # Why was it rejected?
        if arc_fit["max_error"] > 0.3:
            print(
                f"\n❌ ARC REJECTED: Error {arc_fit['max_error']:.3f}m > 0.3m strict tolerance"
            )
        elif abs(arc_fit["angle"]) < fitter.min_arc_angle:
            print(
                f"\n❌ ARC REJECTED: Angle {abs(arc_fit['angle']):.2f}° < {fitter.min_arc_angle:.1f}° minimum"
            )
        elif arc_fit["radius"] < fitter.natural_bend_threshold:
            print(
                f"\n❌ ARC REJECTED: Radius {arc_fit['radius']:.1f}m < {fitter.natural_bend_threshold:.1f}m natural threshold"
            )
        else:
            print(f"\n✅ ARC SHOULD BE ACCEPTED - check algorithm logic")

    else:
        print("  Arc fitting failed")

    # Test straight fitting
    line_fit = fitter._fit_straight_line(test_points)

    print(f"\nStraight fitting results:")
    if line_fit:
        print(f"  Length: {line_fit['length']:.1f}m")
        print(f"  Max error: {line_fit['max_error']:.3f}m")
        print(f"  Error <= strict tolerance (0.3m): {line_fit['max_error'] <= 0.3}")
        print(
            f"  Error <= straight tolerance ({fitter.straight_tolerance:.3f}m): {line_fit['max_error'] <= fitter.straight_tolerance}"
        )

        if line_fit["max_error"] <= 0.3:
            print(f"\n✅ STRAIGHT ACCEPTED: Error {line_fit['max_error']:.3f}m <= 0.3m")
        else:
            print(f"\n❌ STRAIGHT REJECTED: Error {line_fit['max_error']:.3f}m > 0.3m")


if __name__ == "__main__":
    debug_segment_fitting()
