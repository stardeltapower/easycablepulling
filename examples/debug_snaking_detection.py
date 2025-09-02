#!/usr/bin/env python3
"""Debug why snaking detection isn't working for SECT_13."""

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


def debug_snaking_detection(section, fitter):
    """Debug snaking detection logic."""
    print(f"\n{'='*60}")
    print(f"SNAKING DETECTION DEBUG: {section.id}")
    print(f"{'='*60}")

    original_points = section.original_polyline

    # Clean polyline
    cleaner = PolylineCleaner()
    cleaned_points = cleaner.clean_polyline(original_points)

    print(f"Original points: {len(original_points)}")
    print(f"Cleaned points: {len(cleaned_points)}")

    # Replicate snaking detection logic
    start_point = np.array(cleaned_points[0])
    end_point = np.array(cleaned_points[-1])
    straight_distance = np.linalg.norm(end_point - start_point)

    actual_length = sum(
        np.linalg.norm(np.array(cleaned_points[i + 1]) - np.array(cleaned_points[i]))
        for i in range(len(cleaned_points) - 1)
    )

    route_efficiency = straight_distance / actual_length if actual_length > 0 else 1.0

    print(f"Start point: {start_point}")
    print(f"End point: {end_point}")
    print(f"Straight distance: {straight_distance:.1f}m")
    print(f"Actual length: {actual_length:.1f}m")
    print(f"Route efficiency: {route_efficiency:.3f} ({route_efficiency*100:.1f}%)")

    # Check snaking criteria
    efficiency_check = route_efficiency <= 0.95
    points_check = len(cleaned_points) >= 15

    print(f"\nSnaking criteria:")
    print(
        f"  Route efficiency <= 95%: {efficiency_check} ({route_efficiency*100:.1f}% <= 95%)"
    )
    print(f"  Points >= 15: {points_check} ({len(cleaned_points)} >= 15)")
    print(f"  Will trigger snaking: {efficiency_check or points_check}")

    # Test the snaking method directly
    snaking_result = fitter._fit_snaking_route(cleaned_points)
    if snaking_result:
        snaking_length = sum(p.length() for p in snaking_result)
        snaking_preservation = snaking_length / actual_length
        print(f"\nSnaking result:")
        print(f"  Primitives: {len(snaking_result)}")
        print(f"  Total length: {snaking_length:.1f}m")
        print(
            f"  Length preservation: {snaking_preservation:.3f} ({snaking_preservation*100:.1f}%)"
        )

        for i, primitive in enumerate(snaking_result):
            ptype = type(primitive).__name__
            if hasattr(primitive, "bend_type"):
                ptype += f"({primitive.bend_type})"
            print(f"    {i}: {ptype} = {primitive.length():.1f}m")
    else:
        print(f"\nSnaking result: None (failed to fit)")


def main():
    """Debug snaking detection."""
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

    # Focus on SECT_13
    for section in route.sections:
        if section.id == "SECT_13" and section.original_length > 0:
            debug_snaking_detection(section, fitter)


if __name__ == "__main__":
    main()
