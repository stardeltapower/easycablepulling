#!/usr/bin/env python3
"""Test the current fitting behavior after fixes."""

import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from easycablepulling.core.models import DuctSpec
from easycablepulling.geometry.cleaner import PolylineCleaner
from easycablepulling.geometry.fitter import GeometryFitter
from easycablepulling.io import load_route_from_dxf


def test_current_fitting():
    """Test current fitting behavior."""
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

    print("SECT_07 Current Fitting Behavior:")
    print(f"Total cleaned points: {len(cleaned_points)}")
    print()

    # Test the full fitting process
    result = fitter.fit_polyline(cleaned_points)

    print(f"Final result: {len(result.primitives)} primitives")
    print(f"Total error: {result.total_error:.3f}m")
    print(f"Max error: {result.max_error:.3f}m")
    print(f"Success: {result.success}")
    print()

    # Count types
    straights = sum(
        1
        for p in result.primitives
        if hasattr(p, "length_m") and not hasattr(p, "radius_m")
    )
    bends = sum(1 for p in result.primitives if hasattr(p, "radius_m"))

    print(f"Summary: {straights} straights + {bends} bends")
    print()

    print("Detailed primitives:")
    for i, prim in enumerate(result.primitives):
        if hasattr(prim, "radius_m"):
            print(f"  {i+1}. Bend: R={prim.radius_m:.1f}m, {prim.angle_deg:.1f}°")
        else:
            print(f"  {i+1}. Straight: {prim.length_m:.1f}m")


if __name__ == "__main__":
    test_current_fitting()
