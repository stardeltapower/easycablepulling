#!/usr/bin/env python3
"""Generate A0 600dpi overlay plot of fitted vs original route."""

import math
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from easycablepulling.core.models import Bend, DuctSpec, Straight
from easycablepulling.geometry.cleaner import PolylineCleaner
from easycablepulling.geometry.fitter import GeometryFitter
from easycablepulling.io import load_route_from_dxf


def generate_a0_overlay():
    """Generate detailed A0 overlay plot at 600dpi."""
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

    # A0 size at 300dpi (more manageable file size)
    # 841mm x 1189mm = 33.11" x 46.81"
    fig_width = 16.5  # Half size for memory management
    fig_height = 23.4

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=300)

    print("Generating A0 300dpi overlay plot...")
    print("This may take a moment due to high resolution...")

    # Collect all points for axis limits
    all_x = []
    all_y = []

    # Colors for sections
    colors = plt.cm.tab20(np.linspace(0, 1, len(route.sections)))

    section_stats = []

    for i, section in enumerate(route.sections):
        if section.original_length > 0:
            original_points = section.original_polyline
            cleaner = PolylineCleaner()
            cleaned_points = cleaner.clean_polyline(original_points)

            # Collect coordinates
            xs = [p[0] for p in cleaned_points]
            ys = [p[1] for p in cleaned_points]
            all_x.extend(xs)
            all_y.extend(ys)

            # Plot original route
            ax.plot(
                xs,
                ys,
                "o-",
                color=colors[i],
                alpha=0.7,
                linewidth=1,
                markersize=0.5,
                label=f"{section.id} Original",
            )

            # Fit geometry
            result = fitter.fit_polyline(cleaned_points)

            # Calculate stats
            original_length = sum(
                math.sqrt(
                    (cleaned_points[j + 1][0] - cleaned_points[j][0]) ** 2
                    + (cleaned_points[j + 1][1] - cleaned_points[j][1]) ** 2
                )
                for j in range(len(cleaned_points) - 1)
            )
            fitted_length = sum(p.length() for p in result.primitives)
            length_error = abs(fitted_length - original_length) / original_length * 100

            straights = [p for p in result.primitives if isinstance(p, Straight)]
            bends = [p for p in result.primitives if isinstance(p, Bend)]

            section_stats.append(
                {
                    "id": section.id,
                    "original_length": original_length,
                    "fitted_length": fitted_length,
                    "length_error": length_error,
                    "primitives": len(result.primitives),
                    "straights": len(straights),
                    "bends": len(bends),
                    "bend_radii": [b.radius_m for b in bends] if bends else [],
                }
            )

            # Plot fitted primitives
            for j, primitive in enumerate(result.primitives):
                if isinstance(primitive, Straight):
                    # Plot straight as thick line
                    ax.plot(
                        [primitive.start_point[0], primitive.end_point[0]],
                        [primitive.start_point[1], primitive.end_point[1]],
                        color="red",
                        linewidth=3,
                        alpha=0.8,
                    )

                elif isinstance(primitive, Bend):
                    # Plot bend as thick curved line (no circles)
                    # Generate points along the actual bend path
                    n_points = max(10, int(primitive.length() * 2))  # Dense sampling
                    angle_start = 0  # Simplified - should be calculated from geometry
                    angle_span = math.radians(primitive.angle_deg)

                    bend_x = []
                    bend_y = []
                    for k in range(n_points + 1):
                        t = k / n_points if n_points > 0 else 0
                        angle = angle_start + t * angle_span
                        x = primitive.center_point[0] + primitive.radius_m * math.cos(
                            angle
                        )
                        y = primitive.center_point[1] + primitive.radius_m * math.sin(
                            angle
                        )
                        bend_x.append(x)
                        bend_y.append(y)

                    ax.plot(bend_x, bend_y, color="blue", linewidth=3, alpha=0.8)

    # Set equal aspect ratio and tight layout
    ax.set_aspect("equal")

    # Add margin around data
    if all_x and all_y:
        x_range = max(all_x) - min(all_x)
        y_range = max(all_y) - min(all_y)
        margin = max(x_range, y_range) * 0.05

        ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
        ax.set_ylim(min(all_y) - margin, max(all_y) + margin)

    # Formatting
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Easting (m)", fontsize=12)
    ax.set_ylabel("Northing (m)", fontsize=12)
    ax.set_title(
        "CAD Route vs Fitted Construction Geometry\n"
        "Original Route (gray dots) vs Fitted Primitives (red=straights, blue=bends)",
        fontsize=14,
        pad=20,
    )

    # Legend
    legend_elements = [
        mpatches.Patch(color="gray", alpha=0.7, label="Original CAD Route"),
        mpatches.Patch(color="red", label="Fitted Straights"),
        mpatches.Patch(color="blue", label="Fitted Bends"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=10)

    # Add statistics text box
    stats_text = []
    stats_text.append("CONSTRUCTION STATISTICS:")
    stats_text.append(
        f"Total route: {sum(s['original_length'] for s in section_stats):.0f}m → {sum(s['fitted_length'] for s in section_stats):.0f}m"
    )
    stats_text.append(
        f"Length error: {abs(sum(s['fitted_length'] for s in section_stats) - sum(s['original_length'] for s in section_stats)) / sum(s['original_length'] for s in section_stats) * 100:.2f}%"
    )
    stats_text.append(
        f"Total primitives: {sum(s['primitives'] for s in section_stats)}"
    )
    stats_text.append(f"Straights: {sum(s['straights'] for s in section_stats)}")
    stats_text.append(f"Bends: {sum(s['bends'] for s in section_stats)}")

    # Add bend radius summary
    all_radii = []
    for s in section_stats:
        all_radii.extend(s["bend_radii"])

    if all_radii:
        stats_text.append(f"Bend radii: {min(all_radii):.1f}m to {max(all_radii):.1f}m")
        stats_text.append(f"All >= 4.4m min: {all(r >= 4.4 for r in all_radii)}")

    # Position text box
    ax.text(
        0.02,
        0.98,
        "\n".join(stats_text),
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()

    # Save high-resolution file
    output_file = output_dir / "route_overlay_clean_A0_300dpi.png"
    plt.savefig(
        output_file, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none"
    )

    print(f"\nA0 overlay plot saved: {output_file}")
    print(f"File size will be large due to high resolution")

    # Print detailed section statistics
    print("\nDETAILED SECTION ANALYSIS:")
    print("=" * 80)
    print(
        f"{'Section':<8} {'Length':<12} {'Error':<8} {'Primitives':<12} {'Bend Details'}"
    )
    print("-" * 80)

    for stat in section_stats:
        length_str = f"{stat['original_length']:.0f}→{stat['fitted_length']:.0f}m"
        error_str = f"{stat['length_error']:.2f}%"
        prim_str = f"{stat['straights']}S+{stat['bends']}B"

        bend_details = ""
        if stat["bend_radii"]:
            if len(stat["bend_radii"]) == 1:
                bend_details = f"R={stat['bend_radii'][0]:.1f}m"
            else:
                min_r = min(stat["bend_radii"])
                max_r = max(stat["bend_radii"])
                bend_details = f"R={min_r:.1f}-{max_r:.1f}m"

        print(
            f"{stat['id']:<8} {length_str:<12} {error_str:<8} {prim_str:<12} {bend_details}"
        )

    plt.close()  # Free memory


if __name__ == "__main__":
    generate_a0_overlay()
