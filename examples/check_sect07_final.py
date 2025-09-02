#!/usr/bin/env python3
"""Check the final SECT_07 fitting results."""

import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from easycablepulling.core.models import DuctSpec
from easycablepulling.geometry.processor import GeometryProcessor
from easycablepulling.io import load_route_from_dxf


def check_sect07_final():
    """Check final SECT_07 results."""
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

    # Process geometry
    processor = GeometryProcessor()
    processing_result = processor.process_route(route, duct_spec=duct_spec)

    # Find SECT_07
    sect07 = None
    for section in processing_result.route.sections:
        if section.id == "SECT_07":
            sect07 = section
            break

    if not sect07:
        print("SECT_07 not found")
        return

    print("SECT_07 Final Fitting Results:")
    print(f"Original length: {sect07.original_length:.1f}m")

    # Calculate fitted length from primitives
    if hasattr(sect07, "primitives") and sect07.primitives:
        fitted_length = sum(p.length() for p in sect07.primitives)
        print(f"Fitted length: {fitted_length:.1f}m")
        print(
            f"Error: {abs(fitted_length - sect07.original_length)/sect07.original_length*100:.2f}%"
        )
        print()

        print(f"Total primitives: {len(sect07.primitives)}")

        # Count types
        straights = [
            p
            for p in sect07.primitives
            if hasattr(p, "length_m") and not hasattr(p, "radius_m")
        ]
        bends = [p for p in sect07.primitives if hasattr(p, "radius_m")]

        print(f"Straights: {len(straights)}")
        print(f"Bends: {len(bends)}")
        print()

        print("Detailed primitives:")
        for i, prim in enumerate(sect07.primitives):
            if hasattr(prim, "radius_m"):
                print(
                    f"  {i+1}. Bend: R={prim.radius_m:.1f}m, {prim.angle_deg:.1f}°, type={prim.bend_type}"
                )
            else:
                print(f"  {i+1}. Straight: {prim.length_m:.1f}m")
    else:
        print("No primitives found - section not processed yet")

        # Process section directly with fitter
        from easycablepulling.geometry.cleaner import PolylineCleaner
        from easycablepulling.geometry.fitter import GeometryFitter

        fitter = GeometryFitter(duct_spec=duct_spec)
        cleaner = PolylineCleaner()

        cleaned_points = cleaner.clean_polyline(sect07.original_polyline)
        result = fitter.fit_polyline(cleaned_points)

        print(f"Direct fitting result: {len(result.primitives)} primitives")
        print(f"Total error: {result.total_error:.3f}m")
        print(f"Max error: {result.max_error:.3f}m")
        print()

        # Count types
        straights = [
            p
            for p in result.primitives
            if hasattr(p, "length_m") and not hasattr(p, "radius_m")
        ]
        bends = [p for p in result.primitives if hasattr(p, "radius_m")]

        print(f"Straights: {len(straights)}")
        print(f"Bends: {len(bends)}")
        print()

        print("Detailed primitives:")
        for i, prim in enumerate(result.primitives):
            if hasattr(prim, "radius_m"):
                print(f"  {i+1}. Bend: R={prim.radius_m:.1f}m, {prim.angle_deg:.1f}°")
            else:
                print(f"  {i+1}. Straight: {prim.length_m:.1f}m")


if __name__ == "__main__":
    import math

    check_sect07_final()
