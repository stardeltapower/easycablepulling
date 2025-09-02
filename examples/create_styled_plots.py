#!/usr/bin/env python3
"""Create professional styled plots matching input_1200mm.png aesthetic."""

import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from easycablepulling.io import load_route_from_dxf
from easycablepulling.visualization import StylePlotter


def main():
    """Create styled plots."""
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

    # Create styled plot (similar to input_1200mm.png)
    print("Creating professional styled plot...")
    plotter = StylePlotter(figsize=(8.27, 11.69))  # A4 portrait

    fig, ax = plotter.plot_cable_route(
        route,
        title="Cable Route Analysis - Color-Coded Legs with Labeled Endpoints",
        show_legend=True,
        label_all_joints=True,
        units="m",
        convert_to_mm=False,  # Keep in meters for now
    )

    plotter.save_plot(fig, output_dir / "styled_route_plot.png", dpi=300)
    print(f"Saved styled plot to {output_dir / 'styled_route_plot.png'}")

    # Create another version with mm units
    print("Creating mm-based styled plot...")
    fig2, ax2 = plotter.plot_cable_route(
        route,
        title="Cable Route Analysis - Color-Coded Legs with Labeled Endpoints",
        show_legend=True,
        label_all_joints=True,
        units="mm",
        convert_to_mm=True,  # Convert to mm
    )

    plotter.save_plot(fig2, output_dir / "styled_route_plot_mm.png", dpi=300)
    print(f"Saved mm plot to {output_dir / 'styled_route_plot_mm.png'}")

    # Create a cleaner version without input route overlay
    print("Creating clean styled plot...")
    plotter_clean = StylePlotter(figsize=(8.27, 11.69))

    # For cleaner look, we'll modify the route to have fewer sections
    # by combining some adjacent short sections
    fig3, ax3 = plotter_clean.plot_cable_route(
        route,
        title="Cable Route Analysis - Simplified View",
        show_legend=True,
        label_all_joints=True,
        units="m",
        convert_to_mm=False,
    )

    plotter_clean.save_plot(fig3, output_dir / "styled_route_clean.png", dpi=300)
    print(f"Saved clean plot to {output_dir / 'styled_route_clean.png'}")

    print("\nStyled plot creation complete!")
    print("Generated files:")
    for plot_file in sorted(output_dir.glob("styled_*.png")):
        size_kb = plot_file.stat().st_size / 1024
        print(f"  - {plot_file.name} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
