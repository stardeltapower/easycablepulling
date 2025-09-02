#!/usr/bin/env python3
"""Debug SECT_12's massive 973.8m deviation to understand arc generation issue."""

import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from easycablepulling.core.models import Bend, DuctSpec, Straight
from easycablepulling.geometry.cleaner import PolylineCleaner
from easycablepulling.geometry.fitter import GeometryFitter
from easycablepulling.io import load_route_from_dxf


def debug_sect12_arc():
    """Debug why SECT_12 has 973.8m deviation."""
    examples_dir = Path(__file__).parent
    input_dxf = examples_dir / "input.dxf"

    route = load_route_from_dxf(input_dxf, "33kV Cable Route")

    # Find SECT_12
    sect12 = None
    for section in route.sections:
        if section.id == "SECT_12":
            sect12 = section
            break

    if not sect12:
        print("SECT_12 not found")
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

    original_points = sect12.original_polyline
    cleaner = PolylineCleaner()
    cleaned_points = cleaner.clean_polyline(original_points)

    print(f"SECT_12 Arc Generation Debug:")
    print(f"Original points: {len(cleaned_points)}")
    print(f"Section length: {sect12.original_length:.1f}m")
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
            print(f"      Arc length: {prim.length():.1f}m")
        else:
            print(f"  {i+1}. Straight: {prim.length_m:.1f}m")
    print()

    # Test arc fitting directly
    print("Direct arc fitting test:")
    arc_fit = fitter._fit_arc(cleaned_points)
    if arc_fit:
        print(f"  Radius: {arc_fit['radius']:.1f}m")
        print(f"  Angle: {arc_fit['angle']:.1f}°")
        print(f"  Center: ({arc_fit['center'][0]:.1f}, {arc_fit['center'][1]:.1f})")
        print(f"  Max fitting error: {arc_fit['max_error']:.3f}m")

        # Check if this arc actually passes near the original points
        center = np.array(arc_fit["center"])
        print(f"\nDistance from each point to arc:")
        max_arc_dev = 0.0
        for i, point in enumerate(cleaned_points):
            p = np.array(point)
            dist_to_center = np.linalg.norm(p - center)
            deviation_from_arc = abs(dist_to_center - arc_fit["radius"])
            max_arc_dev = max(max_arc_dev, deviation_from_arc)
            print(f"    Point {i}: {deviation_from_arc:.3f}m from arc")

        print(f"  Max arc deviation: {max_arc_dev:.3f}m")

        # This should explain the 973m lateral deviation issue
        print(f"\n🔍 Analysis:")
        if max_arc_dev > 100:
            print(
                f"    🔴 MASSIVE arc deviation ({max_arc_dev:.1f}m) - arc doesn't fit the points!"
            )
            print(
                f"    🎯 Root cause: Circle fitting algorithm is creating incorrect arc geometry"
            )
        elif arc_fit["max_error"] > 10:
            print(
                f"    🔴 High fitting error ({arc_fit['max_error']:.1f}m) - points don't form good circle"
            )
        else:
            print(f"    🟢 Arc geometry looks reasonable")

    else:
        print("  Arc fitting failed")


if __name__ == "__main__":
    debug_sect12_arc()
