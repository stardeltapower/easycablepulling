#!/usr/bin/env python3
"""Position-based analysis comparing fitted vs original at corresponding locations."""

import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from easycablepulling.core.models import DuctSpec
from easycablepulling.geometry.cleaner import PolylineCleaner
from easycablepulling.geometry.fitter import GeometryFitter
from easycablepulling.io import load_route_from_dxf


def analyze_positional_accuracy(section, fitter):
    """Analyze how fitted route deviates at corresponding positions."""
    original_points = section.original_polyline
    cleaner = PolylineCleaner()
    cleaned_points = cleaner.clean_polyline(original_points)

    # Calculate cumulative distances along original route
    original_distances = [0.0]
    for i in range(len(cleaned_points) - 1):
        p1 = np.array(cleaned_points[i])
        p2 = np.array(cleaned_points[i + 1])
        dist = np.linalg.norm(p2 - p1)
        original_distances.append(original_distances[-1] + dist)

    total_original_length = original_distances[-1]

    # Fit geometry
    result = fitter.fit_polyline(cleaned_points)
    fitted_length = sum(p.length() for p in result.primitives)

    # Analysis based on geometry type
    geometry_type = "Unknown"
    if len(result.primitives) == 1:
        if hasattr(result.primitives[0], "radius_m"):
            geometry_type = f"Single Arc (R={result.primitives[0].radius_m:.0f}m)"
        else:
            geometry_type = "Single Straight"
    else:
        straights = sum(1 for p in result.primitives if hasattr(p, "length_m"))
        bends = len(result.primitives) - straights
        geometry_type = f"Mixed: {straights} straights + {bends} bends"

    # Simple quality metrics
    length_error = (
        abs(fitted_length - total_original_length) / total_original_length * 100
    )

    # Route complexity
    start_point = np.array(cleaned_points[0])
    end_point = np.array(cleaned_points[-1])
    straight_distance = np.linalg.norm(end_point - start_point)
    route_efficiency = straight_distance / total_original_length

    # Geometric smoothness (how much direction changes)
    direction_changes = 0
    for i in range(1, len(cleaned_points) - 1):
        p1 = np.array(cleaned_points[i - 1])
        p2 = np.array(cleaned_points[i])
        p3 = np.array(cleaned_points[i + 1])

        v1 = p2 - p1
        v2 = p3 - p2

        if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
            # Calculate angle between vectors
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_angle = np.clip(cos_angle, -1, 1)
            angle_deg = math.degrees(math.acos(cos_angle))

            if angle_deg > 5:  # 5 degree threshold for direction change
                direction_changes += 1

    return {
        "section_id": section.id,
        "original_length": total_original_length,
        "fitted_length": fitted_length,
        "length_error": length_error,
        "straight_distance": straight_distance,
        "route_efficiency": route_efficiency,
        "direction_changes": direction_changes,
        "geometry_type": geometry_type,
        "num_primitives": len(result.primitives),
        "points_count": len(cleaned_points),
    }


def main():
    """Analyze positional accuracy for all sections."""
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
    print("POSITIONAL ACCURACY ANALYSIS")
    print("=" * 90)
    print("Analyzing how geometry fitting affects route characteristics...")
    print()

    # Header
    print(
        f"{'Section':<8} {'Original':<8} {'Fitted':<8} {'Error':<7} {'Efficiency':<10} "
        f"{'Changes':<8} {'Primitives':<11} {'Type'}"
    )
    print("-" * 90)

    good_sections = []  # <1% error
    acceptable_sections = []  # 1-2% error
    problem_sections = []  # >2% error

    total_original = 0.0
    total_fitted = 0.0

    for section in route.sections:
        if section.original_length > 0:
            analysis = analyze_positional_accuracy(section, fitter)

            total_original += analysis["original_length"]
            total_fitted += analysis["fitted_length"]

            # Classify section quality
            if analysis["length_error"] < 1.0:
                good_sections.append(analysis)
                status = "✅"
            elif analysis["length_error"] < 2.0:
                acceptable_sections.append(analysis)
                status = "⚠️ "
            else:
                problem_sections.append(analysis)
                status = "❌"

            print(
                f"{status} {analysis['section_id']:<6} "
                f"{analysis['original_length']:>7.0f}m "
                f"{analysis['fitted_length']:>7.0f}m "
                f"{analysis['length_error']:>6.2f}% "
                f"{analysis['route_efficiency']:>9.1%} "
                f"{analysis['direction_changes']:>7} "
                f"{analysis['num_primitives']:>10} "
                f"{analysis['geometry_type']}"
            )

    # Summary
    overall_error = abs(total_fitted - total_original) / total_original * 100

    print()
    print("=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print(
        f"Route sections: {len(good_sections)} excellent, {len(acceptable_sections)} acceptable, "
        f"{len(problem_sections)} problematic"
    )
    print(
        f"Overall length: {total_original:.0f}m → {total_fitted:.0f}m (error: {overall_error:.2f}%)"
    )
    print()

    if problem_sections:
        print("Problematic sections (>2% error):")
        for analysis in problem_sections:
            complexity = "complex" if analysis["direction_changes"] > 10 else "simple"
            efficiency = "snaking" if analysis["route_efficiency"] < 0.9 else "direct"
            print(
                f"  {analysis['section_id']}: {analysis['length_error']:.2f}% error - "
                f"{complexity} {efficiency} route ({analysis['direction_changes']} direction changes)"
            )

    print(
        f"\nRecommendation: Focus on {'complex snaking' if any(s['route_efficiency'] < 0.9 and s['direction_changes'] > 10 for s in problem_sections) else 'large radius'} sections"
    )


if __name__ == "__main__":
    main()
