#!/usr/bin/env python3
"""Investigate why sections have massive length errors."""

import math
import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from easycablepulling.core.models import DuctSpec
from easycablepulling.geometry.cleaner import PolylineCleaner
from easycablepulling.geometry.fitter import GeometryFitter
from easycablepulling.io import load_route_from_dxf


def calculate_polyline_length(points):
    """Manually calculate polyline length."""
    total = 0.0
    for i in range(len(points) - 1):
        p1, p2 = points[i], points[i + 1]
        dist = math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
        total += dist
    return total


def analyze_section_detailed(section, fitter):
    """Detailed analysis of a problematic section."""
    print(f"\n{'='*50}")
    print(f"DETAILED ANALYSIS: {section.id}")
    print(f"{'='*50}")

    original_points = section.original_polyline
    print(f"Original points: {len(original_points)}")
    print(f"Original reported length: {section.original_length:.2f}m")

    # Manual calculation
    manual_length = calculate_polyline_length(original_points)
    print(f"Manual calculated length: {manual_length:.2f}m")
    print(
        f"Length calculation difference: {abs(manual_length - section.original_length):.2f}m"
    )

    # Clean polyline
    cleaner = PolylineCleaner()
    cleaned_points = cleaner.clean_polyline(original_points)
    cleaned_length = calculate_polyline_length(cleaned_points)
    print(f"Cleaned points: {len(cleaned_points)}")
    print(f"Cleaned length: {cleaned_length:.2f}m")

    # Show first and last few points
    print(f"\nFirst 3 points: {original_points[:3]}")
    print(f"Last 3 points: {original_points[-3:]}")

    # Try simple single-segment fits
    print(f"\n--- Testing single segment fits ---")

    # Test entire segment as straight line
    line_fit = fitter._fit_straight_line(cleaned_points)
    if line_fit:
        print(f"Whole segment as straight:")
        print(f"  Length: {line_fit['length']:.2f}m")
        print(f"  Max error: {line_fit['max_error']:.2f}m")
        print(f"  Error vs length: {line_fit['max_error']/line_fit['length']*100:.1f}%")

    # Test entire segment as arc
    arc_fit = fitter._fit_arc(cleaned_points)
    if arc_fit:
        arc_length = abs(math.radians(arc_fit["angle"])) * arc_fit["radius"]
        print(f"Whole segment as arc:")
        print(f"  Radius: {arc_fit['radius']:.2f}m")
        print(f"  Angle: {arc_fit['angle']:.2f}°")
        print(f"  Arc length: {arc_length:.2f}m")
        print(f"  Max error: {arc_fit['max_error']:.2f}m")
        print(f"  Length vs original: {arc_length/section.original_length*100:.1f}%")

    # Test actual fitting result
    result = fitter.fit_polyline(cleaned_points)
    print(f"\n--- Actual fitting result ---")
    print(f"Success: {result.success}")
    print(f"Primitives: {len(result.primitives)}")

    fitted_total = 0.0
    for i, primitive in enumerate(result.primitives):
        length = primitive.length()
        fitted_total += length
        ptype = type(primitive).__name__
        if hasattr(primitive, "bend_type"):
            ptype += f"({primitive.bend_type})"
        print(f"  {i}: {ptype} = {length:.2f}m")

    print(f"Total fitted length: {fitted_total:.2f}m")
    print(
        f"Length error: {abs(fitted_total - section.original_length)/section.original_length*100:.2f}%"
    )

    return result


def main():
    """Investigate length errors in detail."""
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
    print(f"Natural bend threshold: {fitter.natural_bend_threshold:.1f}m")

    # Focus on worst sections
    worst_sections = ["SECT_01", "SECT_05", "SECT_13"]

    for section in route.sections:
        if section.id in worst_sections and section.original_length > 0:
            analyze_section_detailed(section, fitter)
            print(f"\n{'='*50}")


if __name__ == "__main__":
    main()
