#!/usr/bin/env python3
"""Generate clean overlay plot with worst deviation zoom boxes."""

import math
import sys
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from easycablepulling.core.models import Bend, DuctSpec, Straight
from easycablepulling.geometry.cleaner import PolylineCleaner
from easycablepulling.geometry.fitter import GeometryFitter
from easycablepulling.io import load_route_from_dxf


def calculate_section_deviation(section_id, original_points, fitted_primitives, fitter):
    """Calculate worst deviation for a section and return bounding box."""
    fitted_points = fitter._generate_fitted_points(fitted_primitives)

    if not fitted_points:
        return 0.0, None

    max_deviation = 0.0
    worst_point = None

    # Sample original points and find worst deviation
    for orig_point in original_points[::2]:  # Every 2nd point for speed
        min_dist = min(
            math.sqrt((orig_point[0] - fp[0]) ** 2 + (orig_point[1] - fp[1]) ** 2)
            for fp in fitted_points
        )
        if min_dist > max_deviation:
            max_deviation = min_dist
            worst_point = orig_point

    # Create bounding box around worst point
    if worst_point:
        margin = max(50, max_deviation * 1.2)  # At least 50m margin
        bbox = {
            "x_min": worst_point[0] - margin,
            "x_max": worst_point[0] + margin,
            "y_min": worst_point[1] - margin,
            "y_max": worst_point[1] + margin,
            "center": worst_point,
            "deviation": max_deviation,
        }
        return max_deviation, bbox

    return max_deviation, None


def generate_clean_overlay_with_zoom():
    """Generate clean overlay with single lines and zoom boxes for worst deviations."""
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

    # Create main plot
    fig, (ax_main, ax_zoom) = plt.subplots(1, 2, figsize=(20, 12), dpi=300)

    print("Generating clean overlay plot with worst deviation analysis...")

    # Collect all original points for continuous route
    all_original_x = []
    all_original_y = []
    all_fitted_x = []
    all_fitted_y = []

    # Track worst deviations for zoom boxes
    worst_deviations = []

    for section in route.sections:
        if section.original_length > 0:
            original_points = section.original_polyline
            cleaner = PolylineCleaner()
            cleaned_points = cleaner.clean_polyline(original_points)

            # Add to continuous route
            all_original_x.extend([p[0] for p in cleaned_points])
            all_original_y.extend([p[1] for p in cleaned_points])

            # Fit geometry
            result = fitter.fit_polyline(cleaned_points)

            # Generate fitted points
            fitted_points = fitter._generate_fitted_points(result.primitives)
            if fitted_points:
                all_fitted_x.extend([p[0] for p in fitted_points])
                all_fitted_y.extend([p[1] for p in fitted_points])

            # Calculate worst deviation for this section
            max_dev, bbox = calculate_section_deviation(
                section.id, cleaned_points, result.primitives, fitter
            )

            if bbox and max_dev > 5.0:  # Only track sections with >5m deviation
                worst_deviations.append(
                    {
                        "section_id": section.id,
                        "deviation": max_dev,
                        "bbox": bbox,
                        "original_points": cleaned_points,
                        "fitted_primitives": result.primitives,
                    }
                )

    # Plot main overview
    print("Plotting main route overview...")

    # Original route as fine continuous line
    ax_main.plot(
        all_original_x,
        all_original_y,
        color="black",
        linewidth=1,
        alpha=0.8,
        label="Original CAD Route",
    )

    # Fitted route as dotted red line
    ax_main.plot(
        all_fitted_x,
        all_fitted_y,
        color="red",
        linewidth=2,
        linestyle=":",
        alpha=0.9,
        label="Fitted Construction Route",
    )

    # Draw boxes around worst deviations
    print(f"Found {len(worst_deviations)} areas with >5m deviation...")

    for i, worst in enumerate(worst_deviations):
        bbox = worst["bbox"]

        # Draw box on main plot
        rect = patches.Rectangle(
            (bbox["x_min"], bbox["y_min"]),
            bbox["x_max"] - bbox["x_min"],
            bbox["y_max"] - bbox["y_min"],
            linewidth=2,
            edgecolor="orange",
            facecolor="none",
            linestyle="--",
            alpha=0.8,
        )
        ax_main.add_patch(rect)

        # Label the box
        ax_main.text(
            bbox["center"][0],
            bbox["center"][1],
            f"{worst['section_id']}\n{worst['deviation']:.0f}m",
            ha="center",
            va="center",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
        )

    # Format main plot
    ax_main.set_aspect("equal")
    ax_main.grid(True, alpha=0.3)
    ax_main.set_xlabel("Easting (m)", fontsize=12)
    ax_main.set_ylabel("Northing (m)", fontsize=12)
    ax_main.set_title(
        "Complete Route Overview\nBoxes show worst deviation areas", fontsize=14
    )
    ax_main.legend(fontsize=10)

    # Zoom plot: show worst deviation area
    if worst_deviations:
        print("Creating zoom plot for worst deviation area...")

        # Find absolute worst deviation
        worst = max(worst_deviations, key=lambda x: x["deviation"])
        bbox = worst["bbox"]

        print(
            f"Zooming into {worst['section_id']} with {worst['deviation']:.1f}m deviation..."
        )

        # Plot original points in zoom area
        orig_in_zoom = [
            p
            for p in worst["original_points"]
            if bbox["x_min"] <= p[0] <= bbox["x_max"]
            and bbox["y_min"] <= p[1] <= bbox["y_max"]
        ]

        if orig_in_zoom:
            zoom_orig_x = [p[0] for p in orig_in_zoom]
            zoom_orig_y = [p[1] for p in orig_in_zoom]
            ax_zoom.plot(
                zoom_orig_x,
                zoom_orig_y,
                "o-",
                color="black",
                linewidth=2,
                markersize=3,
                label="Original Points",
            )

        # Plot fitted primitives in zoom area
        for primitive in worst["fitted_primitives"]:
            if isinstance(primitive, Straight):
                # Check if straight intersects zoom box
                start, end = primitive.start_point, primitive.end_point
                if (
                    bbox["x_min"] <= start[0] <= bbox["x_max"]
                    or bbox["x_min"] <= end[0] <= bbox["x_max"]
                ):
                    ax_zoom.plot(
                        [start[0], end[0]],
                        [start[1], end[1]],
                        color="red",
                        linewidth=3,
                        linestyle=":",
                        alpha=0.9,
                    )

            elif isinstance(primitive, Bend):
                # Generate arc points for zoom
                n_points = 50
                angle_start = 0  # Simplified
                angle_span = math.radians(primitive.angle_deg)

                bend_x, bend_y = [], []
                for k in range(n_points + 1):
                    t = k / n_points if n_points > 0 else 0
                    angle = angle_start + t * angle_span
                    x = primitive.center_point[0] + primitive.radius_m * math.cos(angle)
                    y = primitive.center_point[1] + primitive.radius_m * math.sin(angle)

                    # Only include points in zoom area
                    if (
                        bbox["x_min"] <= x <= bbox["x_max"]
                        and bbox["y_min"] <= y <= bbox["y_max"]
                    ):
                        bend_x.append(x)
                        bend_y.append(y)

                if bend_x:
                    ax_zoom.plot(
                        bend_x,
                        bend_y,
                        color="red",
                        linewidth=3,
                        linestyle=":",
                        alpha=0.9,
                    )

        # Format zoom plot
        ax_zoom.set_xlim(bbox["x_min"], bbox["x_max"])
        ax_zoom.set_ylim(bbox["y_min"], bbox["y_max"])
        ax_zoom.set_aspect("equal")
        ax_zoom.grid(True, alpha=0.3)
        ax_zoom.set_xlabel("Easting (m)", fontsize=12)
        ax_zoom.set_ylabel("Northing (m)", fontsize=12)
        ax_zoom.set_title(
            f'Worst Deviation: {worst["section_id"]} ({worst["deviation"]:.1f}m)',
            fontsize=14,
        )
        ax_zoom.legend(fontsize=10)

        # Add deviation annotation
        ax_zoom.text(
            0.05,
            0.95,
            f"Max deviation: {worst['deviation']:.1f}m\nSection: {worst['section_id']}",
            transform=ax_zoom.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.8),
        )

    else:
        ax_zoom.text(
            0.5,
            0.5,
            "No significant deviations found\n(all < 5m)",
            ha="center",
            va="center",
            transform=ax_zoom.transAxes,
            fontsize=12,
        )
        ax_zoom.set_title("Zoom Area - No Major Deviations", fontsize=14)

    plt.tight_layout()

    # Save plot
    output_file = output_dir / "route_overlay_with_zoom_boxes.png"
    plt.savefig(
        output_file, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none"
    )

    print(f"\nClean overlay with zoom boxes saved: {output_file}")

    # Print worst deviation summary
    print("\nWORST DEVIATION AREAS:")
    print("=" * 50)

    if worst_deviations:
        for worst in sorted(
            worst_deviations, key=lambda x: x["deviation"], reverse=True
        ):
            print(f"{worst['section_id']}: {worst['deviation']:.1f}m deviation")
            print(
                f"  Center: ({worst['bbox']['center'][0]:.0f}, {worst['bbox']['center'][1]:.0f})"
            )
            print()
    else:
        print("No areas with >5m deviation found")

    plt.close()  # Free memory


if __name__ == "__main__":
    generate_clean_overlay_with_zoom()
