#!/usr/bin/env python3
"""Analyze lateral deviation of fitted route from original polyline."""

import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from easycablepulling.core.models import Bend, DuctSpec, Straight
from easycablepulling.geometry.cleaner import PolylineCleaner
from easycablepulling.geometry.fitter import GeometryFitter
from easycablepulling.io import load_route_from_dxf


def point_to_line_segment_distance(
    point: Tuple[float, float],
    line_start: Tuple[float, float],
    line_end: Tuple[float, float],
) -> float:
    """Calculate minimum distance from point to line segment."""
    p = np.array(point)
    a = np.array(line_start)
    b = np.array(line_end)

    # Vector from a to b
    ab = b - a
    # Vector from a to p
    ap = p - a

    # Handle zero-length line
    ab_length_sq = np.dot(ab, ab)
    if ab_length_sq < 1e-10:
        return np.linalg.norm(ap)

    # Project ap onto ab, clamped to segment
    t = np.clip(np.dot(ap, ab) / ab_length_sq, 0, 1)

    # Find closest point on line segment
    closest = a + t * ab

    return np.linalg.norm(p - closest)


def point_to_arc_distance(
    point: Tuple[float, float], center: Tuple[float, float], radius: float
) -> float:
    """Calculate minimum distance from point to arc (simplified - assumes full circle)."""
    p = np.array(point)
    c = np.array(center)

    # Distance from point to center
    dist_to_center = np.linalg.norm(p - c)

    # Distance to arc is absolute difference from radius
    return abs(dist_to_center - radius)


def calculate_point_to_primitive_distance(
    point: Tuple[float, float], primitive, prev_end_point: Tuple[float, float] = None
) -> float:
    """Calculate minimum distance from a point to a primitive."""

    if isinstance(primitive, Straight):
        return point_to_line_segment_distance(
            point, primitive.start_point, primitive.end_point
        )

    elif isinstance(primitive, Bend):
        # Simplified: use distance to arc
        # A full implementation would need to consider the arc's actual start/end angles
        return point_to_arc_distance(point, primitive.center_point, primitive.radius_m)

    return float("inf")


def analyze_route_lateral_deviation(section, fitter):
    """Analyze lateral deviation for a route section."""
    original_points = section.original_polyline
    cleaner = PolylineCleaner()
    cleaned_points = cleaner.clean_polyline(original_points)

    # Fit geometry
    result = fitter.fit_polyline(cleaned_points)

    if not result.primitives:
        return None

    # For each point on the original route, find minimum distance to fitted geometry
    deviations = []

    # Sample more densely along the original polyline
    sampled_points = []
    for i in range(len(cleaned_points) - 1):
        p1 = np.array(cleaned_points[i])
        p2 = np.array(cleaned_points[i + 1])
        segment_length = np.linalg.norm(p2 - p1)

        # Sample every 0.5m along segment
        num_samples = max(2, int(segment_length / 0.5))

        for j in range(num_samples):
            t = j / (num_samples - 1) if num_samples > 1 else 0
            sample_point = p1 + t * (p2 - p1)
            sampled_points.append(tuple(sample_point))

    # Calculate deviation for each sampled point
    for point in sampled_points:
        min_distance = float("inf")

        # Find minimum distance to any primitive
        for primitive in result.primitives:
            dist = calculate_point_to_primitive_distance(point, primitive)
            min_distance = min(min_distance, dist)

        deviations.append(min_distance)

    return deviations


def create_deviation_statistics(deviations: List[float]) -> Dict:
    """Create detailed statistics about deviations."""
    if not deviations:
        return {}

    deviations_array = np.array(deviations)

    # Define deviation bands
    bands = [
        (0, 0.15, "≤0.15m (excellent)"),
        (0.15, 0.5, "0.15-0.5m (good)"),
        (0.5, 1.0, "0.5-1.0m (acceptable)"),
        (1.0, 2.0, "1.0-2.0m (marginal)"),
        (2.0, float("inf"), ">2.0m (poor)"),
    ]

    stats = {
        "count": len(deviations),
        "max": np.max(deviations_array),
        "mean": np.mean(deviations_array),
        "median": np.median(deviations_array),
        "p95": np.percentile(deviations_array, 95),
        "p99": np.percentile(deviations_array, 99),
        "bands": {},
    }

    # Count points in each band
    for lower, upper, label in bands:
        if upper == float("inf"):
            count = np.sum(deviations_array > lower)
        else:
            count = np.sum((deviations_array > lower) & (deviations_array <= upper))

        percentage = (count / len(deviations)) * 100
        stats["bands"][label] = {"count": count, "percentage": percentage}

    # Cumulative percentages
    stats["cumulative"] = {
        "≤0.15m": (np.sum(deviations_array <= 0.15) / len(deviations)) * 100,
        "≤0.5m": (np.sum(deviations_array <= 0.5) / len(deviations)) * 100,
        "≤1.0m": (np.sum(deviations_array <= 1.0) / len(deviations)) * 100,
        "≤2.0m": (np.sum(deviations_array <= 2.0) / len(deviations)) * 100,
    }

    return stats


def plot_deviation_distribution(
    all_deviations: Dict[str, List[float]], output_path: Path
):
    """Create comprehensive deviation distribution plot."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Combine all deviations
    combined_deviations = []
    for deviations in all_deviations.values():
        combined_deviations.extend(deviations)

    if not combined_deviations:
        return

    combined_array = np.array(combined_deviations)

    # Top plot: Histogram with log scale
    bins = np.logspace(-3, 1, 50)  # 0.001m to 10m
    ax1.hist(combined_array, bins=bins, alpha=0.7, color="blue", edgecolor="black")
    ax1.axvline(
        0.15, color="green", linestyle="--", linewidth=2, label="0.15m (excellent)"
    )
    ax1.axvline(0.5, color="orange", linestyle="--", linewidth=2, label="0.5m (good)")
    ax1.axvline(
        1.0, color="red", linestyle="--", linewidth=2, label="1.0m (acceptable)"
    )

    ax1.set_xscale("log")
    ax1.set_xlabel("Lateral Deviation (m)")
    ax1.set_ylabel("Number of Points")
    ax1.set_title("Distribution of Lateral Deviations from Original Route")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_xlim(0.001, 10)

    # Bottom plot: Cumulative distribution
    sorted_deviations = np.sort(combined_array)
    cumulative = np.arange(1, len(sorted_deviations) + 1) / len(sorted_deviations) * 100

    ax2.plot(sorted_deviations, cumulative, "b-", linewidth=2)
    ax2.axvline(0.15, color="green", linestyle="--", linewidth=1, alpha=0.5)
    ax2.axvline(0.5, color="orange", linestyle="--", linewidth=1, alpha=0.5)
    ax2.axvline(1.0, color="red", linestyle="--", linewidth=1, alpha=0.5)
    ax2.axhline(95, color="gray", linestyle=":", linewidth=1, alpha=0.5)
    ax2.axhline(99, color="gray", linestyle=":", linewidth=1, alpha=0.5)

    ax2.set_xscale("log")
    ax2.set_xlabel("Lateral Deviation (m)")
    ax2.set_ylabel("Cumulative Percentage (%)")
    ax2.set_title("Cumulative Distribution of Lateral Deviations")
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0.001, 10)
    ax2.set_ylim(0, 100)

    # Add text annotations for key percentiles
    for pct in [95, 99]:
        dev_at_pct = np.percentile(combined_array, pct)
        ax2.text(
            dev_at_pct * 1.2,
            pct - 2,
            f"{pct}%: {dev_at_pct:.3f}m",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    """Analyze lateral deviation for entire route."""
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

    print("=" * 80)
    print("LATERAL DEVIATION ANALYSIS")
    print("=" * 80)
    print("Analyzing perpendicular distance from fitted route to original...")
    print()

    all_deviations = {}
    total_stats = {
        "total_points": 0,
        "bands": {
            "≤0.15m (excellent)": {"count": 0, "percentage": 0},
            "0.15-0.5m (good)": {"count": 0, "percentage": 0},
            "0.5-1.0m (acceptable)": {"count": 0, "percentage": 0},
            "1.0-2.0m (marginal)": {"count": 0, "percentage": 0},
            ">2.0m (poor)": {"count": 0, "percentage": 0},
        },
    }

    sections_analyzed = 0

    for section in route.sections:
        if section.original_length > 0:
            deviations = analyze_route_lateral_deviation(section, fitter)

            if deviations:
                all_deviations[section.id] = deviations
                stats = create_deviation_statistics(deviations)
                sections_analyzed += 1

                # Update totals
                total_stats["total_points"] += stats["count"]
                for band, data in stats["bands"].items():
                    total_stats["bands"][band]["count"] += data["count"]

                # Print section summary
                print(f"\n{section.id}:")
                print(f"  Points analyzed: {stats['count']}")
                print(f"  Max deviation: {stats['max']:.3f}m")
                print(f"  95th percentile: {stats['p95']:.3f}m")
                print(f"  99th percentile: {stats['p99']:.3f}m")

                # Show cumulative percentages
                print(
                    f"  Cumulative: {stats['cumulative']['≤0.15m']:.1f}% ≤0.15m, "
                    f"{stats['cumulative']['≤0.5m']:.1f}% ≤0.5m, "
                    f"{stats['cumulative']['≤1.0m']:.1f}% ≤1.0m"
                )

    # Calculate overall percentages
    if total_stats["total_points"] > 0:
        for band in total_stats["bands"]:
            count = total_stats["bands"][band]["count"]
            total_stats["bands"][band]["percentage"] = (
                count / total_stats["total_points"]
            ) * 100

    # Print overall summary
    print("\n" + "=" * 80)
    print("OVERALL ROUTE STATISTICS")
    print("=" * 80)
    print(f"Total sections analyzed: {sections_analyzed}")
    print(f"Total points analyzed: {total_stats['total_points']:,}")
    print()
    print("Lateral Deviation Distribution:")

    cumulative = 0
    for band, data in total_stats["bands"].items():
        cumulative += data["percentage"]
        print(f"  {band:25} {data['percentage']:6.2f}% ({data['count']:,} points)")

    # Calculate cumulative percentages
    excellent_pct = total_stats["bands"]["≤0.15m (excellent)"]["percentage"]
    good_pct = excellent_pct + total_stats["bands"]["0.15-0.5m (good)"]["percentage"]
    acceptable_pct = (
        good_pct + total_stats["bands"]["0.5-1.0m (acceptable)"]["percentage"]
    )

    print()
    print("Cumulative Statistics:")
    print(f"  {excellent_pct:6.2f}% within 0.15m (excellent)")
    print(f"  {good_pct:6.2f}% within 0.5m (good or better)")
    print(f"  {acceptable_pct:6.2f}% within 1.0m (acceptable or better)")
    print(f"  {100 - acceptable_pct:6.2f}% beyond 1.0m (needs review)")

    # Create visualization
    if all_deviations:
        plot_path = output_dir / "lateral_deviation_distribution.png"
        plot_deviation_distribution(all_deviations, plot_path)
        print(f"\nSaved deviation distribution plot: {plot_path}")


if __name__ == "__main__":
    main()
