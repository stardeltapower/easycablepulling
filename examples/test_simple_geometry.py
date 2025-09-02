#!/usr/bin/env python3
"""Test simplified geometry processing."""

import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from easycablepulling.core.models import CableArrangement, CableSpec, PullingMethod
from easycablepulling.geometry.simple_fitter import SimpleGeometryFitter
from easycablepulling.io import export_route_to_dxf, load_route_from_dxf
from easycablepulling.visualization import StylePlotter


def main():
    """Test simplified geometry processing."""
    examples_dir = Path(__file__).parent
    input_dxf = examples_dir / "input.dxf"
    output_dir = examples_dir / "output"

    # Create output directory
    output_dir.mkdir(exist_ok=True)

    print(f"Loading route from {input_dxf}")

    # Load the route
    route = load_route_from_dxf(input_dxf, "33kV Cable Route")

    print(f"Loaded route with {route.section_count} sections")
    print(
        f"Original total length: {sum(s.original_length for s in route.sections):.1f}m"
    )

    # Process each section with simplified fitting
    fitter = SimpleGeometryFitter(straight_tolerance=1.0)  # 1m tolerance

    total_primitives = 0
    total_straights = 0
    fitted_sections = []

    print("\nProcessing sections:")
    for section in route.sections:
        result = fitter.fit_section_simplified(section.original_polyline)

        if result.success:
            section.primitives = result.primitives
            total_primitives += len(result.primitives)
            total_straights += len(
                result.primitives
            )  # All are straights in simple fitter

            print(
                f"  {section.id}: {len(result.primitives)} segments, max error {result.max_error:.2f}m"
            )
        else:
            print(f"  {section.id}: FAILED - {result.message}")

        fitted_sections.append(section)

    # Update route
    route.sections = fitted_sections

    print(f"\nFitting Results:")
    print(f"  Total primitives fitted: {total_primitives}")
    print(f"  All straight segments: {total_straights}")

    # Calculate fitted vs original length
    fitted_length = sum(s.total_length for s in route.sections)
    original_length = sum(s.original_length for s in route.sections)
    length_diff_percent = abs(fitted_length - original_length) / original_length * 100

    print(f"  Fitted length: {fitted_length:.1f}m")
    print(f"  Original length: {original_length:.1f}m")
    print(f"  Length difference: {length_diff_percent:.2f}%")

    # Export fitted route to DXF
    fitted_dxf_path = output_dir / "simple_fitted_route.dxf"
    print(f"\nExporting fitted route to {fitted_dxf_path}")

    export_route_to_dxf(
        route=route,
        file_path=fitted_dxf_path,
        analysis_results={
            "fitted_primitives": total_primitives,
            "straight_segments": total_straights,
            "bend_segments": 0,
            "length_error_percent": length_diff_percent,
            "fitting_method": "simplified_straight_segments",
        },
        include_annotations=True,
        include_joint_markers=True,
    )

    # Create visualization
    print("\nCreating fitted geometry visualization...")
    plotter = StylePlotter(figsize=(11.69, 8.27))  # A4 landscape

    fig, ax = plotter.plot_cable_route(
        route,
        title="Cable Route - Simplified Geometry Fitting",
        show_legend=True,
        label_all_joints=True,
        units="m",
    )

    # Add processing statistics
    stats_text = (
        f"Simplified Geometry Fitting\n"
        f"{'─' * 25}\n"
        f"Primitives: {total_primitives}\n"
        f"  All Straights: {total_straights}\n"
        f"Length Error: {length_diff_percent:.2f}%\n"
        f"Method: Straight segments"
    )

    props = dict(
        boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.9, edgecolor="black"
    )
    ax.text(
        0.02,
        0.02,
        stats_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="bottom",
        bbox=props,
    )

    plot_path = output_dir / "simple_geometry_result.png"
    plotter.save_plot(fig, plot_path, dpi=300)
    print(f"Saved geometry fitting plot to {plot_path}")

    print(f"\nSimplified geometry processing complete!")

    # Show some section details
    print(f"\nSection Details:")
    for section in route.sections[:5]:  # Show first 5
        if section.primitives:
            print(
                f"  {section.id}: {len(section.primitives)} primitives, {section.total_length:.1f}m fitted"
            )
        else:
            print(f"  {section.id}: No primitives fitted")


if __name__ == "__main__":
    main()
