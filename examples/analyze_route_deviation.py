#!/usr/bin/env python3
"""Analyze perpendicular deviation of fitted route from original polyline."""

import math
import sys
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from easycablepulling.core.models import Bend, DuctSpec, Straight
from easycablepulling.geometry.cleaner import PolylineCleaner
from easycablepulling.geometry.fitter import GeometryFitter
from easycablepulling.io import load_route_from_dxf


def point_to_line_distance(
    point: Tuple[float, float],
    line_start: Tuple[float, float],
    line_end: Tuple[float, float],
) -> float:
    """Calculate perpendicular distance from point to line segment."""
    p = np.array(point)
    a = np.array(line_start)
    b = np.array(line_end)

    # Vector from a to b
    ab = b - a
    # Vector from a to p
    ap = p - a

    # Project ap onto ab
    ab_length_sq = np.dot(ab, ab)
    if ab_length_sq == 0:
        return np.linalg.norm(ap)

    t = np.clip(np.dot(ap, ab) / ab_length_sq, 0, 1)

    # Find closest point on line segment
    closest = a + t * ab

    return np.linalg.norm(p - closest)


def generate_points_along_primitive(
    primitive, num_points=100, prev_point=None, next_point=None
) -> List[Tuple[float, float]]:
    """Generate points along a primitive (straight or bend)."""
    points = []

    if isinstance(primitive, Straight):
        # Linear interpolation
        for i in range(num_points):
            t = i / (num_points - 1)
            x = primitive.start_point[0] + t * (
                primitive.end_point[0] - primitive.start_point[0]
            )
            y = primitive.start_point[1] + t * (
                primitive.end_point[1] - primitive.start_point[1]
            )
            points.append((x, y))

    elif isinstance(primitive, Bend):
        # For now, just sample along the original fitted geometry
        # A proper implementation would calculate the actual arc path
        # This is a limitation of not having the complete geometric context

        # Approximate by creating a smooth curve between connection points
        # This won't be perfect but will give us a reasonable deviation estimate
        if prev_point and next_point:
            # Create a bezier-like curve approximation
            for i in range(num_points):
                t = i / (num_points - 1)
                # Simple quadratic bezier approximation
                p0 = np.array(prev_point)
                p2 = np.array(next_point)
                # Control point at the circle center projected
                p1 = np.array(primitive.center_point)

                # Quadratic bezier formula
                point = (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t**2 * p2
                points.append(tuple(point))
        else:
            # Fallback: just connect start to end
            # This is very approximate but better than nothing
            return [(0, 0)] * num_points  # Will show as max deviation

    return points


def calculate_deviations(
    original_points: List[Tuple[float, float]], fitted_points: List[Tuple[float, float]]
) -> List[float]:
    """Calculate deviation of each fitted point from original polyline."""
    deviations = []

    for fitted_point in fitted_points:
        min_distance = float("inf")

        # Find minimum distance to any segment of original polyline
        for i in range(len(original_points) - 1):
            dist = point_to_line_distance(
                fitted_point, original_points[i], original_points[i + 1]
            )
            min_distance = min(min_distance, dist)

        deviations.append(min_distance)

    return deviations


def analyze_section_deviation(section, fitter, max_allowed_deviation=0.5):
    """Analyze deviation for a single section."""
    original_points = section.original_polyline
    cleaned_points = (
        fitter._polyline_cleaner.clean_polyline(original_points)
        if hasattr(fitter, "_polyline_cleaner")
        else original_points
    )

    # Fit geometry
    result = fitter.fit_polyline(cleaned_points)

    # Use the fitted points that the geometry fitter generates
    # This properly handles arc geometry
    all_fitted_points = result.fitted_points

    if not all_fitted_points:
        return None

    # Calculate deviations from fitted points to original polyline
    deviations = calculate_deviations(cleaned_points, all_fitted_points)

    # Statistics
    max_deviation = max(deviations) if deviations else 0
    avg_deviation = sum(deviations) / len(deviations) if deviations else 0
    percentile_95 = np.percentile(deviations, 95) if deviations else 0

    # Count points exceeding threshold
    exceeding_count = sum(1 for d in deviations if d > max_allowed_deviation)
    exceeding_percent = (exceeding_count / len(deviations) * 100) if deviations else 0

    return {
        "max_deviation": max_deviation,
        "avg_deviation": avg_deviation,
        "percentile_95": percentile_95,
        "exceeding_count": exceeding_count,
        "exceeding_percent": exceeding_percent,
        "total_points": len(deviations),
        "deviations": deviations,
    }


def plot_deviation_histogram(section_id, deviations, max_allowed=0.5):
    """Plot histogram of deviations for a section."""
    plt.figure(figsize=(10, 6))

    # Create bins
    bins = np.linspace(0, max(deviations) * 1.1, 50)

    plt.hist(deviations, bins=bins, alpha=0.7, color="blue", edgecolor="black")
    plt.axvline(
        max_allowed,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Max allowed: {max_allowed}m",
    )
    plt.axvline(
        0.2, color="orange", linestyle="--", linewidth=2, label="Tight tolerance: 0.2m"
    )

    plt.xlabel("Deviation from Original Route (m)")
    plt.ylabel("Number of Points")
    plt.title(f"{section_id} - Deviation Distribution")
    plt.legend()
    plt.grid(True, alpha=0.3)

    return plt.gcf()


def main():
    """Analyze route deviation for all sections."""
    examples_dir = Path(__file__).parent
    input_dxf = examples_dir / "input.dxf"
    output_dir = examples_dir / "output"
    output_dir.mkdir(exist_ok=True)

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
    cleaner = PolylineCleaner()
    fitter._polyline_cleaner = cleaner  # Add cleaner for consistency

    print("=" * 80)
    print("ROUTE DEVIATION ANALYSIS")
    print("=" * 80)
    print("Analyzing perpendicular deviation from original route...")
    print()

    # Focus on problematic sections and some good ones for comparison
    target_sections = ["SECT_01", "SECT_07", "SECT_11", "SECT_13", "SECT_09"]

    for section in route.sections:
        if section.id in target_sections and section.original_length > 0:
            print(f"\n{section.id}:")
            print("-" * 40)

            result = analyze_section_deviation(
                section, fitter, max_allowed_deviation=0.5
            )

            if result:
                print(f"  Max deviation: {result['max_deviation']:.3f}m")
                print(f"  Average deviation: {result['avg_deviation']:.3f}m")
                print(f"  95th percentile: {result['percentile_95']:.3f}m")
                print(
                    f"  Points > 0.5m: {result['exceeding_count']}/{result['total_points']} "
                    f"({result['exceeding_percent']:.1f}%)"
                )

                # Check tighter tolerances
                tight_02m = sum(1 for d in result["deviations"] if d > 0.2)
                tight_02m_percent = tight_02m / len(result["deviations"]) * 100
                print(
                    f"  Points > 0.2m: {tight_02m}/{result['total_points']} "
                    f"({tight_02m_percent:.1f}%)"
                )

                # Visual indicator
                if result["max_deviation"] <= 0.2:
                    status = "✅ Excellent (≤0.2m)"
                elif result["max_deviation"] <= 0.5:
                    status = "⚠️  Good (≤0.5m)"
                else:
                    status = "❌ Poor (>0.5m)"
                print(f"  Status: {status}")

                # Create histogram for problematic sections
                if section.id in ["SECT_07", "SECT_11", "SECT_13"]:
                    fig = plot_deviation_histogram(section.id, result["deviations"])
                    fig.savefig(
                        output_dir / f"deviation_{section.id}.png",
                        dpi=150,
                        bbox_inches="tight",
                    )
                    plt.close(fig)
                    print(f"  Saved deviation plot: deviation_{section.id}.png")


if __name__ == "__main__":
    main()
