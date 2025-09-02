#!/usr/bin/env python3
"""Investigate how snaking routes with multiple curves are handled."""

import math
import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import matplotlib.pyplot as plt

from easycablepulling.core.models import DuctSpec
from easycablepulling.geometry.cleaner import PolylineCleaner
from easycablepulling.geometry.fitter import GeometryFitter
from easycablepulling.io import load_route_from_dxf


def visualize_points_and_fitting(points, primitives, section_id):
    """Visualize original points vs fitted primitives."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot original points
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]

    ax1.plot(
        x_coords, y_coords, "bo-", markersize=4, linewidth=1, label="Original polyline"
    )
    ax1.set_title(f"{section_id} - Original Points ({len(points)} points)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.axis("equal")

    # Plot fitted primitives
    ax2.plot(x_coords, y_coords, "b-", alpha=0.3, linewidth=1, label="Original")

    colors = ["red", "green", "orange", "purple", "brown", "pink", "gray", "olive"]

    for i, primitive in enumerate(primitives):
        color = colors[i % len(colors)]

        if hasattr(primitive, "length_m"):  # Straight
            # Plot straight line
            start, end = primitive.start_point, primitive.end_point
            ax2.plot(
                [start[0], end[0]],
                [start[1], end[1]],
                color=color,
                linewidth=3,
                label=f"Straight {i}: {primitive.length_m:.1f}m",
            )

        elif hasattr(primitive, "radius_m"):  # Bend
            # Plot arc (simplified representation)
            center = primitive.center_point
            radius = primitive.radius_m

            # Draw circle center
            ax2.plot(center[0], center[1], "x", color=color, markersize=8)

            # Draw arc representation (simplified)
            bend_type = getattr(primitive, "bend_type", "unknown")
            arc_length = primitive.length()
            ax2.plot(
                center[0],
                center[1],
                "o",
                color=color,
                markersize=6,
                label=f"{bend_type.title()} arc {i}: R={radius:.0f}m, L={arc_length:.1f}m",
            )

    ax2.set_title(f"{section_id} - Fitted Primitives ({len(primitives)} primitives)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.axis("equal")

    plt.tight_layout()
    return fig


def analyze_snaking_section(section, fitter):
    """Analyze how a snaking section is being fitted."""
    print(f"\n{'='*60}")
    print(f"SNAKING ROUTE ANALYSIS: {section.id}")
    print(f"{'='*60}")

    original_points = section.original_polyline

    # Calculate manual length
    manual_length = sum(
        math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
        for p1, p2 in zip(original_points[:-1], original_points[1:])
    )

    print(f"Points: {len(original_points)}")
    print(f"Original length: {section.original_length:.2f}m")
    print(f"Manual calculated: {manual_length:.2f}m")

    # Show spatial extent
    x_coords = [p[0] for p in original_points]
    y_coords = [p[1] for p in original_points]
    x_range = max(x_coords) - min(x_coords)
    y_range = max(y_coords) - min(y_coords)
    straight_distance = math.sqrt(
        (original_points[-1][0] - original_points[0][0]) ** 2
        + (original_points[-1][1] - original_points[0][1]) ** 2
    )

    print(f"Spatial extent: {x_range:.1f}m × {y_range:.1f}m")
    print(f"Start-to-end straight distance: {straight_distance:.1f}m")
    print(
        f"Route efficiency: {straight_distance/manual_length*100:.1f}% (lower = more snaking)"
    )

    # Test different fitting approaches
    print(f"\n--- Testing different fitting approaches ---")

    # 1. Single arc fit
    arc_fit = fitter._fit_arc(original_points)
    if arc_fit:
        arc_length = abs(math.radians(arc_fit["angle"])) * arc_fit["radius"]
        print(f"Single arc fit:")
        print(f"  Radius: {arc_fit['radius']:.1f}m, Angle: {arc_fit['angle']:.1f}°")
        print(
            f"  Arc length: {arc_length:.1f}m ({arc_length/manual_length*100:.1f}% of original)"
        )
        print(
            f"  Max error: {arc_fit['max_error']:.1f}m ({arc_fit['max_error']/manual_length*100:.1f}% of length)"
        )

    # 2. Single straight fit
    line_fit = fitter._fit_straight_line(original_points)
    if line_fit:
        print(f"Single straight fit:")
        print(
            f"  Length: {line_fit['length']:.1f}m ({line_fit['length']/manual_length*100:.1f}% of original)"
        )
        print(
            f"  Max error: {line_fit['max_error']:.1f}m ({line_fit['max_error']/manual_length*100:.1f}% of length)"
        )

    # 3. Actual recursive fit
    cleaner = PolylineCleaner()
    cleaned_points = cleaner.clean_polyline(original_points)
    result = fitter.fit_polyline(cleaned_points)

    print(f"\nActual recursive fit:")
    print(f"  Primitives: {len(result.primitives)}")

    total_fitted = 0.0
    arc_count = 0
    straight_count = 0

    for i, primitive in enumerate(result.primitives):
        length = primitive.length()
        total_fitted += length

        if hasattr(primitive, "radius_m"):
            arc_count += 1
            bend_type = getattr(primitive, "bend_type", "unknown")
            print(
                f"    Arc {i}: {bend_type} R={primitive.radius_m:.1f}m, ∠={primitive.angle_deg:.1f}°, L={length:.1f}m"
            )
        else:
            straight_count += 1
            print(f"    Straight {i}: L={length:.1f}m")

    print(
        f"  Total: {arc_count} arcs + {straight_count} straights = {total_fitted:.1f}m"
    )
    print(f"  Length error: {abs(total_fitted - manual_length)/manual_length*100:.2f}%")

    # Create visualization
    fig = visualize_points_and_fitting(cleaned_points, result.primitives, section.id)

    return result, fig


def main():
    """Investigate snaking route fitting."""
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

    # Focus on SECT_13 (15.3% error) and other potentially snaking sections
    target_sections = ["SECT_13", "SECT_09", "SECT_07"]  # Different complexities

    for section in route.sections:
        if section.id in target_sections and section.original_length > 0:
            result, fig = analyze_snaking_section(section, fitter)

            # Save visualization
            plot_path = output_dir / f"snaking_analysis_{section.id}.png"
            fig.savefig(plot_path, dpi=300, bbox_inches="tight")
            print(f"Saved analysis plot: {plot_path}")
            plt.close(fig)


if __name__ == "__main__":
    main()
