#!/usr/bin/env python3
"""Test polynomial fitting on problematic segments."""

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


def test_polynomial_fitting():
    """Test polynomial fitting on SECT_07 first segment."""
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

    print(f"Testing polynomial fitting on SECT_07 first segment:")
    print(f"Natural bend threshold: {fitter.natural_bend_threshold:.1f}m")
    print()

    # Test first 7 points (like first straight segment)
    test_points = cleaned_points[0:7]

    print(f"Testing {len(test_points)} points:")
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

    print(f"\nOriginal segment length: {actual_length:.1f}m")
    print()

    # Test polynomial fitting
    for degree in [2, 3, 4]:
        print(f"Testing degree {degree} polynomial:")

        poly_fit = fitter._fit_polynomial(test_points, degree)

        if poly_fit:
            print(f"  ✅ Polynomial fit successful")
            print(f"  Max error: {poly_fit['max_error']:.3f}m")
            print(f"  Curve length: {poly_fit['curve_length']:.1f}m")
            print(f"  Min radius: {poly_fit['min_radius']:.1f}m")

            # Check thresholds
            radius_ok = poly_fit["min_radius"] >= fitter.natural_bend_threshold * 0.5
            error_ok = poly_fit["max_error"] <= 0.3

            print(
                f"  Min radius >= threshold ({fitter.natural_bend_threshold * 0.5:.1f}m): {radius_ok}"
            )
            print(f"  Error <= 0.3m: {error_ok}")

            if radius_ok and error_ok:
                print(f"  🎉 POLYNOMIAL ACCEPTABLE - would be used")

                # Test decomposition
                primitives = fitter._decompose_polynomial_to_primitives(poly_fit)
                print(f"  Decomposes to {len(primitives)} primitives:")
                for j, prim in enumerate(primitives):
                    if hasattr(prim, "radius_m"):
                        print(
                            f"    {j+1}. Bend: R={prim.radius_m:.1f}m, {prim.angle_deg:.1f}°"
                        )
                    else:
                        print(f"    {j+1}. Straight: {prim.length_m:.1f}m")

                total_decomp_length = sum(p.length() for p in primitives)
                print(f"  Total decomposed length: {total_decomp_length:.1f}m")

                # Debug decomposition process
                print(
                    f"  Decomposition threshold: {fitter.natural_bend_threshold * 3:.1f}m"
                )
                print(
                    f"  Min radius ({poly_fit['min_radius']:.1f}m) >= threshold: {poly_fit['min_radius'] >= fitter.natural_bend_threshold * 3}"
                )

                # If it's all classified as curve but decomposed to straight, something's wrong
                if (
                    poly_fit["min_radius"] < fitter.natural_bend_threshold * 3
                    and len(primitives) == 1
                    and hasattr(primitives[0], "length_m")
                ):
                    print(
                        f"  🐛 BUG: Curve radius but decomposed to straight - check _create_primitive_from_polynomial_region"
                    )

            else:
                print(f"  ❌ POLYNOMIAL REJECTED")
        else:
            print(f"  ❌ Polynomial fitting failed")

        print()


if __name__ == "__main__":
    test_polynomial_fitting()
