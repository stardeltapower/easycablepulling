#!/usr/bin/env python3
"""Debug SECT_09's remaining 23.8m deviation issue."""

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


def debug_sect09():
    """Debug why SECT_09 still has 23.8m deviation."""
    examples_dir = Path(__file__).parent
    input_dxf = examples_dir / "input.dxf"

    route = load_route_from_dxf(input_dxf, "33kV Cable Route")

    # Find SECT_09
    sect09 = None
    for section in route.sections:
        if section.id == "SECT_09":
            sect09 = section
            break

    if not sect09:
        print("SECT_09 not found")
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

    original_points = sect09.original_polyline
    cleaner = PolylineCleaner()
    cleaned_points = cleaner.clean_polyline(original_points)

    print(f"SECT_09 Deviation Debug:")
    print(f"Original points: {len(cleaned_points)}")
    print(f"Section length: {sect09.original_length:.1f}m")
    print()

    # Show original points
    print("Original points:")
    for i, point in enumerate(cleaned_points):
        print(f"  {i}: ({point[0]:.1f}, {point[1]:.1f})")
    print()

    # Fit geometry
    result = fitter.fit_polyline(cleaned_points)

    print(f"Fitted as {len(result.primitives)} primitives:")
    for i, prim in enumerate(result.primitives):
        if isinstance(prim, Bend):
            print(f"  {i+1}. Bend: R={prim.radius_m:.1f}m, {prim.angle_deg:.1f}°")
            print(
                f"      Center: ({prim.center_point[0]:.1f}, {prim.center_point[1]:.1f})"
            )
            if hasattr(prim, "control_points") and prim.control_points:
                print(f"      Control points: {len(prim.control_points)}")
        else:
            print(f"  {i+1}. Straight: {prim.length_m:.1f}m")
    print()

    # Generate fitted points and check deviation
    fitted_points = fitter._generate_fitted_points(result.primitives)

    print(f"Generated {len(fitted_points)} fitted points")
    print("First 5 fitted points:")
    for i, point in enumerate(fitted_points[:5]):
        print(f"  {i}: ({point[0]:.1f}, {point[1]:.1f})")

    # Calculate actual deviation
    max_dev = 0.0
    worst_orig = None
    worst_fitted = None

    for orig_point in cleaned_points:
        min_dist = min(
            math.sqrt((orig_point[0] - fp[0]) ** 2 + (orig_point[1] - fp[1]) ** 2)
            for fp in fitted_points
        )
        if min_dist > max_dev:
            max_dev = min_dist
            worst_orig = orig_point
            # Find closest fitted point
            closest_fitted = min(
                fitted_points,
                key=lambda fp: math.sqrt(
                    (orig_point[0] - fp[0]) ** 2 + (orig_point[1] - fp[1]) ** 2
                ),
            )
            worst_fitted = closest_fitted

    print(f"\nActual deviation analysis:")
    print(f"Max deviation: {max_dev:.1f}m")
    if worst_orig and worst_fitted:
        print(f"Worst original point: ({worst_orig[0]:.1f}, {worst_orig[1]:.1f})")
        print(f"Closest fitted point: ({worst_fitted[0]:.1f}, {worst_fitted[1]:.1f})")
        print(
            f"Distance: {math.sqrt((worst_orig[0] - worst_fitted[0])**2 + (worst_orig[1] - worst_fitted[1])**2):.1f}m"
        )


if __name__ == "__main__":
    debug_sect09()
