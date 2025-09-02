#!/usr/bin/env python3
"""Test polynomial fitting on the full 60-point SECT_07."""

import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from easycablepulling.core.models import DuctSpec
from easycablepulling.geometry.cleaner import PolylineCleaner
from easycablepulling.geometry.fitter import GeometryFitter
from easycablepulling.io import load_route_from_dxf


def test_full_section():
    """Test polynomial fitting on full SECT_07."""
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

    print(f"SECT_07 Full Section Test:")
    print(f"Total cleaned points: {len(cleaned_points)}")
    print()

    # Test polynomial fitting on full section
    print("Testing polynomial fitting on full 60-point section:")
    poly_fit = fitter._fit_polynomial(cleaned_points, degree=3)
    if poly_fit:
        print(f"  ✅ Polynomial fit successful:")
        print(f"     Max error: {poly_fit['max_error']:.3f}m")
        print(f"     Min radius: {poly_fit['min_radius']:.1f}m")
        print(f"     Curve length: {poly_fit['curve_length']:.1f}m")

        # Check conditions
        radius_ok = poly_fit["min_radius"] >= fitter.natural_bend_threshold * 0.5
        error_ok = poly_fit["max_error"] <= 0.5

        print(
            f"     Radius check (>= {fitter.natural_bend_threshold * 0.5:.1f}m): {radius_ok}"
        )
        print(f"     Error check (<= 0.5m): {error_ok}")

        if not error_ok:
            print(
                f"     ❌ Error too high! Need <= 0.5m, got {poly_fit['max_error']:.3f}m"
            )

            # Test with higher tolerance
            print()
            print("Testing with relaxed error tolerance (1.0m):")
            if poly_fit["max_error"] <= 1.0:
                print(f"     ✅ Would pass with 1.0m tolerance")

                # Test arc fitting on full section
                arc_fit = fitter._fit_arc(cleaned_points)
                if arc_fit:
                    print(
                        f"     Arc fit: R={arc_fit['radius']:.1f}m, angle={arc_fit['angle']:.1f}°, error={arc_fit['max_error']:.3f}m"
                    )

                    arc_valid = (
                        abs(arc_fit["angle"]) >= fitter.min_arc_angle
                        and arc_fit["radius"] >= fitter.natural_bend_threshold
                    )
                    print(f"     Arc would be valid: {arc_valid}")

                # Test polynomial decomposition
                print()
                print("Testing polynomial decomposition:")
                poly_primitives = fitter._decompose_polynomial_to_primitives(poly_fit)
                print(f"     Decomposition returned {len(poly_primitives)} primitives:")
                for i, prim in enumerate(poly_primitives):
                    if hasattr(prim, "radius_m"):
                        print(
                            f"       {i+1}. Bend: R={prim.radius_m:.1f}m, {prim.angle_deg:.1f}°"
                        )
                    else:
                        print(f"       {i+1}. Straight: {prim.length_m:.1f}m")
    else:
        print("  ❌ Polynomial fitting failed")

    # Now test with different polynomial degrees
    print()
    print("Testing different polynomial degrees:")
    for degree in [2, 4, 5]:
        poly_fit = fitter._fit_polynomial(cleaned_points, degree=degree)
        if poly_fit:
            print(
                f"  Degree {degree}: error={poly_fit['max_error']:.3f}m, min_radius={poly_fit['min_radius']:.1f}m"
            )
        else:
            print(f"  Degree {degree}: ❌ Failed")


if __name__ == "__main__":
    test_full_section()
