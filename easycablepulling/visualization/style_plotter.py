"""Professional style plotting matching the input_1200mm.png aesthetic."""

import string
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ..core.models import Route, Section


class StylePlotter:
    """Create professional plots with color-coded sections and labeled nodes."""

    def __init__(self, figsize: Tuple[float, float] = (8.27, 11.69)):
        """Initialize style plotter.

        Args:
            figsize: Figure size in inches (default A4 portrait)
        """
        self.figsize = figsize

        # Professional color palette matching input_1200mm.png
        self.route_colors = [
            "#FF0000",  # Red
            "#0000FF",  # Blue
            "#00FF00",  # Green
            "#FF00FF",  # Magenta
            "#FFA500",  # Orange
            "#8B4513",  # Brown
            "#FF69B4",  # Pink
            "#808080",  # Gray
            "#00CED1",  # Cyan
            "#FFD700",  # Yellow
            "#4B0082",  # Indigo
            "#FF1493",  # Deep Pink
            "#32CD32",  # Lime Green
            "#DC143C",  # Crimson
            "#00BFFF",  # Deep Sky Blue
            "#8A2BE2",  # Blue Violet
            "#FF6347",  # Tomato
            "#40E0D0",  # Turquoise
            "#9370DB",  # Medium Purple
            "#F0E68C",  # Khaki
        ]

    def create_figure(self) -> Tuple[Figure, Axes]:
        """Create clean matplotlib figure matching the style."""
        plt.rcParams["figure.dpi"] = 100
        plt.rcParams["savefig.dpi"] = 300
        plt.rcParams["font.family"] = "DejaVu Sans"
        plt.rcParams["font.size"] = 10

        fig, ax = plt.subplots(figsize=self.figsize)
        ax.set_aspect("equal")

        # Light gray grid
        ax.grid(True, alpha=0.3, linewidth=0.5, color="gray")
        ax.set_axisbelow(True)

        # Set axis labels
        ax.set_xlabel("X Coordinate (mm)", fontsize=11)
        ax.set_ylabel("Y Coordinate (mm)", fontsize=11)

        # Keep all spines visible but thin
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)

        return fig, ax

    def plot_cable_route(
        self,
        route: Route,
        title: str = "Cable Route Analysis - Color-Coded Legs with Labeled Endpoints",
        show_legend: bool = True,
        label_all_joints: bool = True,
        units: str = "m",
        convert_to_mm: bool = False,
    ) -> Tuple[Figure, Axes]:
        """Create professional cable route plot.

        Args:
            route: Route object to plot
            title: Plot title
            show_legend: Whether to show legend
            label_all_joints: Label all joints/nodes
            units: Display units (m or mm)
            convert_to_mm: Convert coordinates from m to mm

        Returns:
            Figure and axes objects
        """
        fig, ax = self.create_figure()

        # Generate node labels (A, B, C, etc.)
        labels = list(string.ascii_uppercase)
        if len(labels) < route.section_count + 1:
            # Extend with A1, B1, etc. if needed
            for i in range(len(labels), route.section_count + 1):
                labels.append(f"{labels[i % 26]}{i // 26 + 1}")

        # Collect all joint points
        joint_points = []
        joint_labels = []

        # Add first point
        if route.sections and route.sections[0].original_polyline:
            first_point = route.sections[0].original_polyline[0]
            joint_points.append(first_point)
            joint_labels.append(labels[0])

        # Plot each section
        legend_entries = []
        for i, section in enumerate(route.sections):
            if not section.original_polyline:
                continue

            # Get coordinates
            x_coords = [p[0] for p in section.original_polyline]
            y_coords = [p[1] for p in section.original_polyline]

            # Convert to mm if requested
            if convert_to_mm:
                x_coords = [x * 1000 for x in x_coords]
                y_coords = [y * 1000 for y in y_coords]

            # Select color
            color = self.route_colors[i % len(self.route_colors)]

            # Create route label
            if i < len(route.sections) - 1:
                route_label = f"Route {labels[i]}{labels[i+1]}"
            else:
                route_label = f"Route {labels[i]}{labels[i+1]}"

            # Plot line
            line = ax.plot(
                x_coords,
                y_coords,
                color=color,
                linewidth=2.5,
                label=route_label,
                solid_capstyle="round",
                solid_joinstyle="round",
            )[0]

            legend_entries.append((line, route_label))

            # Add end point
            end_point = section.original_polyline[-1]
            if convert_to_mm:
                end_point = (end_point[0] * 1000, end_point[1] * 1000)
            joint_points.append(end_point)
            joint_labels.append(labels[i + 1])

        # Plot joint markers
        if label_all_joints:
            for point, label in zip(joint_points, joint_labels):
                # White circle with black border
                # Calculate appropriate radius based on plot scale
                if convert_to_mm:
                    # For mm plots, use a radius that's visible
                    x_range = max(p[0] for p in joint_points) - min(
                        p[0] for p in joint_points
                    )
                    radius = x_range * 0.002  # 0.2% of x range
                else:
                    radius = 8  # Fixed radius for meter plots

                circle = plt.Circle(
                    point,
                    radius=radius,
                    facecolor="white",
                    edgecolor="black",
                    linewidth=1.5,
                    zorder=10,
                )
                ax.add_patch(circle)

                # Add label
                ax.text(
                    point[0],
                    point[1],
                    label,
                    ha="center",
                    va="center",
                    fontsize=11,
                    fontweight="bold",
                    zorder=11,
                )

        # Add input route as first entry if multiple sections
        if route.section_count > 1:
            # Plot full route as thin black line
            all_points = []
            for section in route.sections:
                if section.original_polyline:
                    points = section.original_polyline
                    if convert_to_mm:
                        points = [(p[0] * 1000, p[1] * 1000) for p in points]
                    all_points.extend(points)

            if all_points:
                x_all = [p[0] for p in all_points]
                y_all = [p[1] for p in all_points]
                input_line = ax.plot(
                    x_all,
                    y_all,
                    color="black",
                    linewidth=1.0,
                    alpha=0.5,
                    label="Input route",
                    zorder=1,
                )[0]

                # Insert at beginning of legend
                legend_entries.insert(0, (input_line, "Input route"))

        # Set title
        ax.set_title(title, fontsize=13, pad=15, fontweight="normal")

        # Add legend
        if show_legend and legend_entries:
            # Create custom legend
            lines, labels = zip(*legend_entries)
            ax.legend(
                lines,
                labels,
                loc="lower left",
                frameon=True,
                fancybox=False,
                framealpha=1.0,
                edgecolor="black",
                facecolor="white",
                fontsize=9,
                ncol=1,
            )

        # Update axis labels based on units
        if units == "mm" or convert_to_mm:
            ax.set_xlabel("X Coordinate (mm)", fontsize=11)
            ax.set_ylabel("Y Coordinate (mm)", fontsize=11)
        else:
            ax.set_xlabel("X Coordinate (m)", fontsize=11)
            ax.set_ylabel("Y Coordinate (m)", fontsize=11)

        # Auto-scale with small margin
        ax.autoscale()
        ax.margins(0.03)

        # For mm plots, ensure we're focused on the actual data
        if convert_to_mm and joint_points:
            x_coords = [p[0] for p in joint_points]
            y_coords = [p[1] for p in joint_points]

            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)

            # Add 5% padding
            x_padding = (x_max - x_min) * 0.05
            y_padding = (y_max - y_min) * 0.05

            ax.set_xlim(x_min - x_padding, x_max + x_padding)
            ax.set_ylim(y_min - y_padding, y_max + y_padding)

        # Fix axis formatting for large numbers (mm case)
        if convert_to_mm or units == "mm":
            # Get current limits
            x_min, x_max = ax.get_xlim()
            y_min, y_max = ax.get_ylim()

            # If values are very large (mm), use offset notation
            if x_max > 10000 or y_max > 10000:
                ax.ticklabel_format(style="plain", axis="both")

                # Adjust tick labels to show values nicely
                import matplotlib.ticker as ticker

                ax.xaxis.set_major_formatter(
                    ticker.FuncFormatter(lambda x, p: f"{x:,.0f}")
                )
                ax.yaxis.set_major_formatter(
                    ticker.FuncFormatter(lambda x, p: f"{x:,.0f}")
                )

        plt.tight_layout()
        return fig, ax

    def save_plot(self, fig: Figure, output_path: Path, dpi: int = 300) -> None:
        """Save plot with high quality settings.

        Args:
            fig: Figure to save
            output_path: Output file path
            dpi: Resolution (default 300)
        """
        fig.savefig(
            output_path,
            dpi=dpi,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
            pad_inches=0.1,
        )
