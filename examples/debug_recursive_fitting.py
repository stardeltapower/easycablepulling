#!/usr/bin/env python3
"""Debug the recursive fitting process to see why polynomial fitting isn't working."""

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


def debug_recursive_process():
    """Debug why polynomial fitting isn't being used in recursive process."""
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

    print(f"SECT_07 Recursive Fitting Debug:")
    print(f"Total cleaned points: {len(cleaned_points)}")
    print()

    # Manually test the _recursive_fit method with first segment
    print("Testing _recursive_fit on full section:")
    primitives = fitter._recursive_fit(cleaned_points, 0, len(cleaned_points) - 1, 0)

    print(f"Recursive fit returned {len(primitives)} primitives:")
    for i, prim in enumerate(primitives):
        if hasattr(prim, "radius_m"):
            print(f"  {i+1}. Bend: R={prim.radius_m:.1f}m, {prim.angle_deg:.1f}°")
        else:
            print(f"  {i+1}. Straight: {prim.length_m:.1f}m")

    print()
    print("Let's trace what happens in the first recursive call...")

    # Manually check what happens with the first 7 points in recursive fitting
    test_points = cleaned_points[0:7]
    print(f"\nTesting polynomial fitting on first 7 points manually:")

    if len(test_points) >= 4:
        poly_fit = fitter._fit_polynomial(test_points, degree=3)
        if poly_fit:
            print(
                f"  Polynomial fit: error={poly_fit['max_error']:.3f}m, min_radius={poly_fit['min_radius']:.1f}m"
            )

            # Check conditions
            radius_ok = poly_fit["min_radius"] >= fitter.natural_bend_threshold * 0.5
            error_ok = poly_fit["max_error"] <= 0.5

            print(
                f"  Radius check (>= {fitter.natural_bend_threshold * 0.5:.1f}m): {radius_ok}"
            )
            print(f"  Error check (<= 0.5m): {error_ok}")

            if radius_ok and error_ok:
                print("  ✅ Polynomial should be accepted")

                # Test arc fitting
                arc_fit = fitter._fit_arc(test_points)
                if arc_fit:
                    print(
                        f"  Arc fit: R={arc_fit['radius']:.1f}m, angle={arc_fit['angle']:.1f}°"
                    )

                    arc_valid = (
                        abs(arc_fit["angle"]) >= fitter.min_arc_angle
                        and arc_fit["radius"] >= fitter.natural_bend_threshold
                    )
                    print(f"  Arc valid for curve preservation: {arc_valid}")

                    if arc_valid:
                        print("  🎉 Should return single bend")
                    else:
                        print("  ➡️ Should decompose polynomial")
                else:
                    print("  ❌ Arc fitting failed")
            else:
                print("  ❌ Polynomial rejected")
        else:
            print("  ❌ Polynomial fitting failed")
    else:
        print(f"  ❌ Not enough points ({len(test_points)}) for polynomial")


if __name__ == "__main__":
    debug_recursive_process()
