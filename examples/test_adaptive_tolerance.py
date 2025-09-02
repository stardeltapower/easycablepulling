#!/usr/bin/env python3
"""Test the adaptive tolerance calculation."""

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


def test_adaptive_tolerance():
    """Test adaptive tolerance calculation."""
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

    print("Testing adaptive tolerance on different segment sizes:")
    print()

    # Test different segment sizes
    for size in [7, 10, 15, 25, 40, 60]:
        if size > len(cleaned_points):
            break

        segment_points = cleaned_points[:size]

        # Calculate segment length
        segment_length = sum(
            math.sqrt(
                (segment_points[i + 1][0] - segment_points[i][0]) ** 2
                + (segment_points[i + 1][1] - segment_points[i][1]) ** 2
            )
            for i in range(len(segment_points) - 1)
        )

        # Calculate adaptive tolerance (same as in fitter)
        adaptive_tolerance = min(2.0, max(0.5, segment_length * 0.02))

        # Test polynomial fitting
        poly_fit = fitter._fit_polynomial(segment_points, degree=3)
        if poly_fit:
            radius_ok = poly_fit["min_radius"] >= fitter.natural_bend_threshold * 0.5
            error_ok = poly_fit["max_error"] <= adaptive_tolerance

            print(f"Segment size {size:2d} points, length {segment_length:5.1f}m:")
            print(f"  Adaptive tolerance: {adaptive_tolerance:.3f}m")
            print(f"  Polynomial error:   {poly_fit['max_error']:.3f}m")
            print(
                f"  Min radius:         {poly_fit['min_radius']:.1f}m (need >= {fitter.natural_bend_threshold * 0.5:.1f}m)"
            )
            print(
                f"  Would accept: radius={radius_ok}, error={error_ok} -> {radius_ok and error_ok}"
            )
            print()
        else:
            print(f"Segment size {size:2d} points: ❌ Polynomial fitting failed")
            print()


if __name__ == "__main__":
    test_adaptive_tolerance()
