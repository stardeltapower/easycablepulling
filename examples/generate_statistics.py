#!/usr/bin/env python3
"""Generate statistics and reports for cable routes."""

import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import matplotlib.pyplot as plt

from easycablepulling.analysis import calculate_route_statistics
from easycablepulling.io import load_route_from_dxf
from easycablepulling.visualization import StylePlotter


def main():
    """Generate statistics and visualizations."""
    examples_dir = Path(__file__).parent
    input_dxf = examples_dir / "input.dxf"
    output_dir = examples_dir / "output"

    # Create output directory
    output_dir.mkdir(exist_ok=True)

    print(f"Loading route from {input_dxf}")

    # Load the route
    route = load_route_from_dxf(input_dxf, "Example 33kV Route")

    print(f"Loaded route with {route.section_count} sections")

    # Calculate statistics
    print("\nCalculating route statistics...")
    stats = calculate_route_statistics(route)

    # Generate text report
    report = stats.generate_report()
    print("\n" + report)

    # Save report to file
    report_path = output_dir / "route_statistics_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\nSaved text report to {report_path}")

    # Save JSON statistics
    json_path = output_dir / "route_statistics.json"
    stats.save_json(json_path)
    print(f"Saved JSON statistics to {json_path}")

    # Save CSV statistics
    csv_path = output_dir / "section_statistics.csv"
    stats.save_csv(csv_path)
    print(f"Saved CSV statistics to {csv_path}")

    # Create statistical visualizations
    print("\nCreating statistical visualizations...")

    # 1. Section length histogram
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.27, 11.69))  # A4 portrait

    # Histogram of section lengths
    section_lengths = [s.length for s in stats.sections]
    ax1.hist(section_lengths, bins=10, edgecolor="black", alpha=0.7, color="steelblue")
    ax1.set_xlabel("Section Length (m)")
    ax1.set_ylabel("Count")
    ax1.set_title("Distribution of Section Lengths")
    ax1.grid(True, alpha=0.3)

    # Add statistics text
    stats_text = (
        f"Mean: {stats.avg_section_length:.1f}m\n"
        f"Std: {stats.std_section_length:.1f}m\n"
        f"Min: {stats.min_section_length:.1f}m\n"
        f"Max: {stats.max_section_length:.1f}m"
    )
    ax1.text(
        0.95,
        0.95,
        stats_text,
        transform=ax1.transAxes,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    # 2. Section lengths bar chart
    section_ids = [s.section_id for s in stats.sections]
    colors = plt.cm.viridis(
        [
            (l - min(section_lengths)) / (max(section_lengths) - min(section_lengths))
            for l in section_lengths
        ]
    )

    bars = ax2.bar(
        range(len(section_ids)),
        section_lengths,
        color=colors,
        edgecolor="black",
        linewidth=0.5,
    )
    ax2.set_xlabel("Section")
    ax2.set_ylabel("Length (m)")
    ax2.set_title("Section Lengths by ID")
    ax2.set_xticks(range(len(section_ids)))
    ax2.set_xticklabels(section_ids, rotation=45, ha="right")
    ax2.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bar, length in zip(bars, section_lengths):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{length:.0f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()
    stats_plot_path = output_dir / "section_statistics_plot.png"
    plt.savefig(stats_plot_path, dpi=300, bbox_inches="tight")
    print(f"Saved statistics plot to {stats_plot_path}")

    # 3. Create route plot with statistics overlay
    print("\nCreating route plot with statistics...")
    plotter = StylePlotter(figsize=(11.69, 8.27))  # A4 landscape
    fig2, ax = plotter.plot_cable_route(
        route,
        title="Cable Route with Section Statistics",
        show_legend=True,
        label_all_joints=True,
        units="m",
    )

    # Add statistics box
    stats_summary = (
        f"Route Statistics\n"
        f"{'─' * 20}\n"
        f"Total Length: {stats.total_length:,.0f}m\n"
        f"Sections: {stats.section_count}\n"
        f"Joints: {stats.total_joints}\n"
        f"Route Width: {stats.route_width:,.0f}m\n"
        f"Route Height: {stats.route_height:,.0f}m\n"
        f"\nSection Length:\n"
        f"  Min: {stats.min_section_length:,.0f}m\n"
        f"  Max: {stats.max_section_length:,.0f}m\n"
        f"  Avg: {stats.avg_section_length:,.0f}m"
    )

    props = dict(
        boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9, edgecolor="black"
    )
    ax.text(
        0.02,
        0.98,
        stats_summary,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=props,
    )

    route_stats_plot_path = output_dir / "route_with_statistics.png"
    plotter.save_plot(fig2, route_stats_plot_path, dpi=300)
    print(f"Saved route statistics plot to {route_stats_plot_path}")

    print("\nStatistics generation complete!")


if __name__ == "__main__":
    main()
