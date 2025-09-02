#!/usr/bin/env python3
"""Identify sections with worst lateral deviations for targeted improvement."""

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


def analyze_worst_deviations():
    """Find sections contributing most to poor path following."""
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

    print("WORST DEVIATION ANALYSIS")
    print("=" * 60)

    section_stats = []

    for section in route.sections:
        if section.original_length > 0:
            original_points = section.original_polyline
            cleaner = PolylineCleaner()
            cleaned_points = cleaner.clean_polyline(original_points)

            # Fit geometry
            result = fitter.fit_polyline(cleaned_points)

            # Quick deviation calculation
            fitted_points = fitter._generate_fitted_points(result.primitives)

            deviations = []
            if fitted_points:
                # Sample every 2nd original point for speed
                for orig_point in cleaned_points[::2]:
                    min_dist = min(
                        math.sqrt(
                            (orig_point[0] - fp[0]) ** 2 + (orig_point[1] - fp[1]) ** 2
                        )
                        for fp in fitted_points
                    )
                    deviations.append(min_dist)

            if deviations:
                max_dev = max(deviations)
                p95_dev = np.percentile(deviations, 95)
                bad_points = sum(1 for d in deviations if d > 1.0)
                bad_pct = (bad_points / len(deviations)) * 100

                section_stats.append(
                    {
                        "id": section.id,
                        "max_dev": max_dev,
                        "p95_dev": p95_dev,
                        "bad_pct": bad_pct,
                        "length": section.original_length,
                        "primitives": len(result.primitives),
                        "natural_bends": len(
                            [
                                p
                                for p in result.primitives
                                if isinstance(p, Bend) and p.bend_type == "natural"
                            ]
                        ),
                        "straights": len(
                            [p for p in result.primitives if isinstance(p, Straight)]
                        ),
                    }
                )

    # Sort by worst deviation percentage
    section_stats.sort(key=lambda x: x["bad_pct"], reverse=True)

    print(
        f"{'Section':<8} {'Max Dev':<8} {'95% Dev':<8} {'>1m %':<7} {'Length':<7} {'Prims':<5} {'Type'}"
    )
    print("-" * 60)

    for stat in section_stats:
        geometry_type = f"{stat['straights']}S+{stat['natural_bends']}B"
        status = (
            "🔴" if stat["bad_pct"] > 70 else "🟡" if stat["bad_pct"] > 30 else "🟢"
        )

        print(
            f"{status} {stat['id']:<6} "
            f"{stat['max_dev']:>7.2f}m "
            f"{stat['p95_dev']:>7.2f}m "
            f"{stat['bad_pct']:>6.1f}% "
            f"{stat['length']:>6.0f}m "
            f"{stat['primitives']:>4} "
            f"{geometry_type}"
        )

    print()
    print("WORST OFFENDERS (>70% points beyond 1m):")
    worst_sections = [s for s in section_stats if s["bad_pct"] > 70]

    for stat in worst_sections:
        print(
            f"\n{stat['id']}: {stat['bad_pct']:.1f}% beyond 1m (max: {stat['max_dev']:.1f}m)"
        )
        print(f"  Length: {stat['length']:.0f}m, Primitives: {stat['primitives']}")
        print(
            f"  Composition: {stat['straights']} straights + {stat['natural_bends']} natural bends"
        )

        # Specific recommendations
        if stat["straights"] > stat["natural_bends"]:
            print(f"  🎯 ISSUE: Too many straights creating shortcuts")
            print(f"     Fix: Force smaller segments to preserve curves")
        elif stat["natural_bends"] > 0:
            print(f"  🎯 ISSUE: Large radius bends not following path closely")
            print(f"     Fix: Use higher-degree polynomials or multiple connected arcs")


if __name__ == "__main__":
    analyze_worst_deviations()
