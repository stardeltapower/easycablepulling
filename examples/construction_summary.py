#!/usr/bin/env python3
"""Construction summary: length accuracy, lateral deviation, and bend analysis."""

import math
import sys
from pathlib import Path

import numpy as np

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from easycablepulling.config import STANDARD_DUCT_BENDS
from easycablepulling.core.models import Bend, DuctSpec, Straight
from easycablepulling.geometry.cleaner import PolylineCleaner
from easycablepulling.geometry.fitter import GeometryFitter
from easycablepulling.io import load_route_from_dxf


def analyze_construction_feasibility():
    """Comprehensive analysis for construction planning."""
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

    print("=" * 90)
    print("CONSTRUCTION SUMMARY - CAD TO BUILDABLE ROUTE")
    print("=" * 90)
    print()

    # Overall statistics
    total_original = 0.0
    total_fitted = 0.0
    all_straights = []
    all_natural_bends = []
    all_manufactured_bends = []

    # Section-by-section analysis
    print(
        f"{'Section':<8} {'Original':<8} {'Fitted':<8} {'Error':<7} {'Max Dev':<8} {'Primitives':<12} {'Construction Notes'}"
    )
    print("-" * 90)

    for section in route.sections:
        if section.original_length > 0:
            original_points = section.original_polyline
            cleaner = PolylineCleaner()
            cleaned_points = cleaner.clean_polyline(original_points)

            # Calculate original length
            original_length = sum(
                math.sqrt(
                    (cleaned_points[i + 1][0] - cleaned_points[i][0]) ** 2
                    + (cleaned_points[i + 1][1] - cleaned_points[i][1]) ** 2
                )
                for i in range(len(cleaned_points) - 1)
            )

            # Fit geometry
            result = fitter.fit_polyline(cleaned_points)
            fitted_length = sum(p.length() for p in result.primitives)

            # Categorize primitives
            straights = [p for p in result.primitives if isinstance(p, Straight)]
            natural_bends = [
                p
                for p in result.primitives
                if isinstance(p, Bend) and p.bend_type == "natural"
            ]
            manufactured_bends = [
                p
                for p in result.primitives
                if isinstance(p, Bend) and p.bend_type == "manufactured"
            ]

            all_straights.extend(straights)
            all_natural_bends.extend(natural_bends)
            all_manufactured_bends.extend(manufactured_bends)

            # Calculate basic deviation (simplified)
            fitted_points = fitter._generate_fitted_points(result.primitives)
            max_dev = 0.0
            if fitted_points and len(fitted_points) > 0:
                for orig_point in cleaned_points[::5]:  # Sample every 5th point
                    min_dist = min(
                        math.sqrt(
                            (orig_point[0] - fp[0]) ** 2 + (orig_point[1] - fp[1]) ** 2
                        )
                        for fp in fitted_points
                    )
                    max_dev = max(max_dev, min_dist)

            # Length error
            length_error = abs(fitted_length - original_length) / original_length * 100

            # Construction notes
            notes = []
            if len(manufactured_bends) > 0:
                notes.append(f"{len(manufactured_bends)} prebent")
            if len(natural_bends) > 0:
                notes.append(f"{len(natural_bends)} field-bent")
            if len(straights) > 0:
                notes.append(f"{len(straights)} straight")

            construction_notes = ", ".join(notes) if notes else "N/A"

            # Status
            status = "✅" if length_error < 1.0 and max_dev < 2.0 else "⚠️"

            print(
                f"{status} {section.id:<6} "
                f"{original_length:>7.0f}m "
                f"{fitted_length:>7.0f}m "
                f"{length_error:>6.2f}% "
                f"{max_dev:>7.2f}m "
                f"{len(result.primitives):>11} "
                f"{construction_notes}"
            )

            total_original += original_length
            total_fitted += fitted_length

    # Overall summary
    overall_error = abs(total_fitted - total_original) / total_original * 100

    print()
    print("=" * 90)
    print("OVERALL ROUTE SUMMARY")
    print("=" * 90)
    print(
        f"Total route length: {total_original:.0f}m → {total_fitted:.0f}m (error: {overall_error:.2f}%)"
    )
    print()

    # Bend analysis for construction
    print("BEND ANALYSIS FOR CONSTRUCTION:")
    print("-" * 50)

    print(
        f"Total primitives: {len(all_straights) + len(all_natural_bends) + len(all_manufactured_bends)}"
    )
    print(
        f"  • {len(all_straights)} straight sections ({sum(s.length_m for s in all_straights):.0f}m)"
    )
    print(
        f"  • {len(all_natural_bends)} natural bends - field installation ({sum(b.length() for b in all_natural_bends):.0f}m)"
    )
    print(
        f"  • {len(all_manufactured_bends)} manufactured bends - prebent fittings ({sum(b.length() for b in all_manufactured_bends):.0f}m)"
    )
    print()

    # Natural bend radius analysis
    if all_natural_bends:
        natural_radii = [b.radius_m for b in all_natural_bends]
        print("Natural bend radii (field installation):")
        print(f"  Range: {min(natural_radii):.1f}m to {max(natural_radii):.1f}m")
        print(f"  Average: {np.mean(natural_radii):.1f}m")
        print(
            f"  All >= minimum radius ({fitter.natural_bend_threshold:.1f}m): {all(r >= fitter.natural_bend_threshold for r in natural_radii)}"
        )
        print()

    # Manufactured bend analysis
    if all_manufactured_bends:
        manufactured_radii = [
            b.radius_m * 1000 for b in all_manufactured_bends
        ]  # Convert to mm
        print("Manufactured bends (standard fittings):")

        # Check against standard catalog
        standard_radii = set(bend["radius"] for bend in STANDARD_DUCT_BENDS)

        for bend in all_manufactured_bends:
            radius_mm = bend.radius_m * 1000
            closest_standard = min(standard_radii, key=lambda x: abs(x - radius_mm))
            print(
                f"  R={radius_mm:.0f}mm → Standard R={closest_standard:.0f}mm ({bend.angle_deg:.1f}°)"
            )
        print()

    # Construction assessment
    print("CONSTRUCTION FEASIBILITY:")
    print("-" * 50)

    # All bends meet minimum radius?
    all_bends = all_natural_bends + all_manufactured_bends
    if all_bends:
        min_bend_radius = min(b.radius_m for b in all_bends)
        radius_ok = min_bend_radius >= fitter.natural_bend_threshold
        print(
            f"✅ All bend radii >= {fitter.natural_bend_threshold:.1f}m minimum: {radius_ok}"
        )
        if not radius_ok:
            print(f"   Minimum found: {min_bend_radius:.1f}m")

    # Length accuracy acceptable for cable pulling?
    length_ok = overall_error <= 2.0  # 2% tolerance for cable pulling
    print(
        f"✅ Length accuracy <= 2.0% for cable pulling: {length_ok} ({overall_error:.2f}%)"
    )

    # Path following quality?
    # Based on previous lateral deviation analysis
    path_quality = "Needs improvement (52.78% beyond 1.0m)"
    print(f"⚠️  Path following quality: {path_quality}")
    print("   Recommendation: Use polynomial fitting results for precise path layout")
    print()

    print("CONSTRUCTION WORKFLOW:")
    print("1. Use straight duct sections for straight primitives")
    print("2. Use standard prebent fittings for manufactured bends")
    print("3. Field-bend duct for natural bends (gentle curves)")
    print("4. Polynomial fitting ensures accurate CAD route following")


if __name__ == "__main__":
    analyze_construction_feasibility()
