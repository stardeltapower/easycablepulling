#!/usr/bin/env python3
"""Test polynomial decomposition on SECT_07 segments."""

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


def test_polynomial_decomposition():
    """Test polynomial decomposition on different segments."""
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

    print("Testing polynomial decomposition on SECT_07 segments:")
    print()

    # Test different segment sizes that should work with polynomial fitting
    test_segments = [
        (0, 10, "First 10 points"),
        (10, 20, "Points 10-20"),
        (20, 35, "Points 20-35"),
        (35, 50, "Points 35-50"),
        (45, 60, "Last 15 points"),
    ]

    for start, end, desc in test_segments:
        if end > len(cleaned_points):
            end = len(cleaned_points)

        segment_points = cleaned_points[start:end]

        if len(segment_points) < 4:
            continue

        print(f"{desc} ({len(segment_points)} points):")

        # Calculate segment length
        segment_length = sum(
            math.sqrt(
                (segment_points[i + 1][0] - segment_points[i][0]) ** 2
                + (segment_points[i + 1][1] - segment_points[i][1]) ** 2
            )
            for i in range(len(segment_points) - 1)
        )

        # Test polynomial fitting
        poly_fit = fitter._fit_polynomial(segment_points, degree=3)
        if poly_fit:
            print(f"  Segment length: {segment_length:.1f}m")
            print(f"  Polynomial error: {poly_fit['max_error']:.3f}m")
            print(f"  Min radius: {poly_fit['min_radius']:.1f}m")

            # Check conditions
            adaptive_tolerance = min(2.0, max(0.5, segment_length * 0.02))
            radius_ok = poly_fit["min_radius"] >= fitter.natural_bend_threshold * 0.5
            error_ok = poly_fit["max_error"] <= adaptive_tolerance

            print(f"  Adaptive tolerance: {adaptive_tolerance:.3f}m")
            print(
                f"  Radius OK (>= {fitter.natural_bend_threshold * 0.5:.1f}m): {radius_ok}"
            )
            print(f"  Error OK (<= {adaptive_tolerance:.3f}m): {error_ok}")

            if radius_ok and error_ok:
                print("  ✅ Polynomial conditions met")

                # Test arc fitting for curve preservation
                arc_fit = fitter._fit_arc(segment_points)
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
                        print("  🎭 Would preserve as single bend")
                    else:
                        print("  🔧 Would decompose polynomial")

                        # Test decomposition
                        poly_primitives = fitter._decompose_polynomial_to_primitives(
                            poly_fit
                        )
                        print(
                            f"  Decomposition result: {len(poly_primitives)} primitives"
                        )
                        for j, prim in enumerate(poly_primitives):
                            if hasattr(prim, "radius_m"):
                                print(
                                    f"    {j+1}. Bend: R={prim.radius_m:.1f}m, {prim.angle_deg:.1f}°"
                                )
                            else:
                                print(f"    {j+1}. Straight: {prim.length_m:.1f}m")
                else:
                    print("  ❌ Arc fitting failed")
            else:
                print("  ❌ Polynomial conditions not met")
        else:
            print("  ❌ Polynomial fitting failed")

        print()


if __name__ == "__main__":
    test_polynomial_decomposition()
