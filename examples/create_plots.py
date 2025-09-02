#!/usr/bin/env python3
"""Create example plots from the input DXF file."""

import os
import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from easycablepulling.io import export_route_to_dxf, load_route_from_dxf
from easycablepulling.visualization import RoutePlotter, plot_dxf_comparison, plot_route


def main():
    """Create example plots."""
    examples_dir = Path(__file__).parent
    input_dxf = examples_dir / "input.dxf"
    output_dir = examples_dir / "output"

    # Create output directory
    output_dir.mkdir(exist_ok=True)

    print(f"Loading route from {input_dxf}")

    # Load the route
    route = load_route_from_dxf(input_dxf, "Example 33kV Route")

    print(f"Loaded route with {route.section_count} sections")
    print(f"Total length: {sum(s.original_length for s in route.sections):.1f}m")

    # Create route plot (A4 portrait for better fit)
    print("Creating route plot (A4 portrait, 300 DPI)...")
    fig, ax = plot_route(
        route,
        output_path=output_dir / "route_plot.png",
        show_original=True,
        show_fitted=False,  # No fitted primitives yet (Phase 3)
        show_joints=True,
        show_annotations=True,
        figsize=(8.27, 11.69),  # A4 portrait
    )

    print(f"Saved route plot to {output_dir / 'route_plot.png'}")

    # Create DXF layer visualization (A4 portrait)
    print("Creating DXF layer plot (A4 portrait, 300 DPI)...")
    plotter = RoutePlotter(figsize=(8.27, 11.69))  # A4 portrait
    fig2, ax2 = plotter.plot_dxf_layers(input_dxf, title="Original DXF File Layers")
    plotter.save_plot(fig2, output_dir / "dxf_layers.png", dpi=300)

    print(f"Saved DXF layers plot to {output_dir / 'dxf_layers.png'}")

    # Create a sample analysis DXF for comparison
    print("Creating sample analysis DXF...")
    analysis_dxf = output_dir / "analysis_sample.dxf"

    # Export with some sample analysis data
    sample_analysis = {
        "max_tension": 15000.0,
        "max_sidewall_pressure": 3500.0,
        "analysis_date": "2025-09-01",
    }

    sample_warnings = [
        "Section SECT_01: Example validation warning",
        "Section SECT_05: Another example warning",
    ]

    export_route_to_dxf(
        route=route,
        file_path=analysis_dxf,
        analysis_results=sample_analysis,
        warnings=sample_warnings,
        include_annotations=True,
        include_joint_markers=True,
    )

    # Create comparison plot (A4 landscape)
    print("Creating comparison plot (A4 landscape, 300 DPI)...")
    fig3, ax3 = plot_dxf_comparison(
        original_dxf=input_dxf,
        analysis_dxf=analysis_dxf,
        route=route,
        output_path=output_dir / "dxf_comparison.png",
        figsize=(11.69, 8.27),  # A4 landscape
    )

    print(f"Saved comparison plot to {output_dir / 'dxf_comparison.png'}")

    # Create a detailed section plot (first few sections, A4 landscape)
    print("Creating detailed section plot (A4 landscape, 300 DPI)...")
    plotter = RoutePlotter(figsize=(11.69, 8.27))  # A4 landscape
    fig4, ax4 = plotter.create_figure()

    # Plot first 3 sections in detail
    for i, section in enumerate(route.sections[:3]):
        offset_x = i * 100  # Offset sections horizontally for clarity

        # Offset the polyline points
        offset_points = [(x + offset_x, y) for x, y in section.original_polyline]

        plotter.plot_polyline(
            ax4,
            offset_points,
            color=plotter.colors["original"],
            linewidth=3.0,
            label=f"Section {section.id}" if i < 3 else None,
        )

        # Add section label
        if offset_points:
            mid_point = offset_points[len(offset_points) // 2]
            ax4.text(
                mid_point[0],
                mid_point[1] + 50,
                f"{section.id}\n{section.original_length:.0f}m",
                ha="center",
                va="bottom",
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            )

    ax4.set_title(
        "Detailed Section View (First 3 Sections)", fontsize=14, fontweight="bold"
    )
    ax4.legend(loc="upper right", fontsize=10)

    plotter.save_plot(fig4, output_dir / "sections_detail.png", dpi=300)
    print(f"Saved detailed sections plot to {output_dir / 'sections_detail.png'}")

    print("\nPlot creation complete! Generated A4-sized files at 300 DPI:")
    for plot_file in sorted(output_dir.glob("*.png")):
        # Get file size in KB
        size_kb = plot_file.stat().st_size / 1024
        print(f"  - {plot_file.name} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
