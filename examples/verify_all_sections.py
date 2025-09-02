#!/usr/bin/env python3
"""Verify all sections meet the <2% error requirement after snaking improvements."""

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


def main():
    """Verify all sections meet accuracy requirements."""
    examples_dir = Path(__file__).parent
    input_dxf = examples_dir / "input.dxf"

    route = load_route_from_dxf(input_dxf, "33kV Cable Route")

    # Create duct specification for 200mm HDPE
    duct_spec = DuctSpec(
        inner_diameter=200.0,  # mm
        type="HDPE",
        friction_dry=0.35,
        friction_lubricated=0.15,
        bend_options=[],
    )

    fitter = GeometryFitter(duct_spec=duct_spec)
    cleaner = PolylineCleaner()

    print("=" * 80)
    print("COMPLETE ROUTE ACCURACY VERIFICATION")
    print("=" * 80)
    print(f"Target: <2% length error for cable pulling accuracy")
    print(f"Duct: {duct_spec.inner_diameter}mm {duct_spec.type}")
    print(f"Natural bend threshold: {fitter.natural_bend_threshold:.1f}m")
    print()

    total_original = 0.0
    total_fitted = 0.0
    sections_over_2_percent = []
    sections_processed = 0

    for section in route.sections:
        if section.original_length <= 0:
            continue

        sections_processed += 1
        original_points = section.original_polyline
        cleaned_points = cleaner.clean_polyline(original_points)

        # Calculate manual length for verification
        manual_length = sum(
            math.sqrt(
                (cleaned_points[i + 1][0] - cleaned_points[i][0]) ** 2
                + (cleaned_points[i + 1][1] - cleaned_points[i][1]) ** 2
            )
            for i in range(len(cleaned_points) - 1)
        )

        # Fit geometry
        result = fitter.fit_polyline(cleaned_points)
        fitted_length = sum(p.length() for p in result.primitives)

        # Calculate error
        length_error = abs(fitted_length - manual_length) / manual_length * 100

        total_original += manual_length
        total_fitted += fitted_length

        # Status indicator
        status = "✅" if length_error < 2.0 else "❌"

        print(
            f"{status} {section.id:>8}: {fitted_length:>7.1f}m vs {manual_length:>7.1f}m "
            f"(error: {length_error:>5.2f}%) - {len(result.primitives)} primitives"
        )

        if length_error >= 2.0:
            sections_over_2_percent.append((section.id, length_error))

            # Show primitive breakdown for problem sections
            print(f"      Breakdown:", end="")
            for i, primitive in enumerate(result.primitives):
                ptype = type(primitive).__name__[0]  # S or B
                if hasattr(primitive, "bend_type"):
                    bend_type = primitive.bend_type[0].upper()  # N or M
                    print(f" {ptype}{bend_type}({primitive.length():.0f}m)", end="")
                else:
                    print(f" {ptype}({primitive.length():.0f}m)", end="")
            print()

    # Summary
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    overall_error = abs(total_fitted - total_original) / total_original * 100

    print(f"Sections processed: {sections_processed}")
    print(f"Total original length: {total_original:.1f}m")
    print(f"Total fitted length: {total_fitted:.1f}m")
    print(f"Overall length error: {overall_error:.2f}%")
    print()

    if sections_over_2_percent:
        print(f"❌ FAILED: {len(sections_over_2_percent)} sections exceed 2% error:")
        for section_id, error in sections_over_2_percent:
            print(f"    {section_id}: {error:.2f}%")
    else:
        print("✅ SUCCESS: All sections meet <2% error requirement!")

    print()
    print(
        f"Cable pulling accuracy: {'ACCEPTABLE' if overall_error < 2.0 else 'UNACCEPTABLE'}"
    )

    if overall_error >= 2.0:
        print(
            f"⚠️  Route length could vary by ±{overall_error/100 * total_original/1000:.2f}km"
        )

    return sections_over_2_percent


if __name__ == "__main__":
    main()
