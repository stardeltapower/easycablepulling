#!/usr/bin/env python3
"""Analyze SECT_07 straight segments to see if they follow original route."""

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


def analyze_sect07_straights():
    """Analyze how SECT_07 straights relate to original points."""
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

    print(f"SECT_07 Analysis:")
    print(f"Original points: {len(cleaned_points)}")
    print(f"Original length: {sect07.original_length:.1f}m")
    print()

    # Fit geometry
    result = fitter.fit_polyline(cleaned_points)

    print(f"Fitted primitives: {len(result.primitives)}")
    print(f"Fitted length: {sum(p.length() for p in result.primitives):.1f}m")
    print()

    # Analyze each straight
    straights = [p for p in result.primitives if isinstance(p, Straight)]

    print(f"Found {len(straights)} straight segments:")
    print()

    for i, straight in enumerate(straights, 1):
        print(f"Straight {i}:")
        print(
            f"  Start: ({straight.start_point[0]:.1f}, {straight.start_point[1]:.1f})"
        )
        print(f"  End: ({straight.end_point[0]:.1f}, {straight.end_point[1]:.1f})")
        print(f"  Length: {straight.length_m:.1f}m")

        # Find closest original points to start and end
        start_distances = [
            math.sqrt(
                (straight.start_point[0] - p[0]) ** 2
                + (straight.start_point[1] - p[1]) ** 2
            )
            for p in cleaned_points
        ]
        end_distances = [
            math.sqrt(
                (straight.end_point[0] - p[0]) ** 2
                + (straight.end_point[1] - p[1]) ** 2
            )
            for p in cleaned_points
        ]

        min_start_dist = min(start_distances)
        min_end_dist = min(end_distances)
        start_idx = start_distances.index(min_start_dist)
        end_idx = end_distances.index(min_end_dist)

        print(f"  Closest to original point {start_idx}: {min_start_dist:.3f}m")
        print(f"  Closest to original point {end_idx}: {min_end_dist:.3f}m")

        # Calculate how much original route this straight skips
        if end_idx > start_idx:
            skipped_points = end_idx - start_idx
            original_segment_length = sum(
                math.sqrt(
                    (cleaned_points[j + 1][0] - cleaned_points[j][0]) ** 2
                    + (cleaned_points[j + 1][1] - cleaned_points[j][1]) ** 2
                )
                for j in range(start_idx, min(end_idx, len(cleaned_points) - 1))
            )
            shortcut_ratio = (
                straight.length_m / original_segment_length
                if original_segment_length > 0
                else 1.0
            )

            print(f"  Skips {skipped_points} original points")
            print(f"  Original segment: {original_segment_length:.1f}m")
            print(
                f"  Shortcut ratio: {shortcut_ratio:.3f} (1.0 = no shortcut, <1.0 = shortcut)"
            )

        print()


if __name__ == "__main__":
    analyze_sect07_straights()
