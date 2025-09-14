"""Professional matplotlib visualization with custom styling."""

import math
import string
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.font_manager as fm
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ..core.models import Bend, Route, Section, Straight


class ProfessionalMatplotlibPlotter:
    """Professional visualization using matplotlib with custom engineering styling."""

    def __init__(self) -> None:
        """Initialize with professional engineering styling."""
        # Set professional matplotlib style
        plt.style.use("default")  # Reset to default first

        # Professional color scheme
        self.colors = {
            "primary": "#1f4e79",  # Deep engineering blue
            "secondary": "#2e8b57",  # Sea green
            "accent": "#c7302a",  # Engineering red
            "warning": "#f39c12",  # Amber
            "success": "#27ae60",  # Green
            "neutral": "#7f8c8d",  # Blue gray
            # Section colors (distinct engineering palette)
            "sections": [
                "#1f4e79",
                "#2e8b57",
                "#c7302a",
                "#f39c12",
                "#8e44ad",
                "#16a085",
                "#e74c3c",
                "#f1c40f",
                "#9b59b6",
                "#1abc9c",
                "#e67e22",
                "#3498db",
                "#2c3e50",
                "#95a5a6",
                "#d35400",
            ],
        }

        # Professional typography
        self.fonts = {
            "title": {"family": "DejaVu Sans", "size": 16, "weight": "bold"},
            "label": {"family": "DejaVu Sans", "size": 12, "weight": "normal"},
            "annotation": {"family": "DejaVu Sans", "size": 10, "weight": "normal"},
            "legend": {"family": "DejaVu Sans", "size": 10, "weight": "normal"},
        }

    def setup_professional_style(self, fig: Figure, ax: Axes) -> None:
        """Apply professional styling to figure and axes."""
        # Figure styling
        fig.patch.set_facecolor("white")
        fig.patch.set_edgecolor("none")

        # Axes styling
        ax.set_facecolor("white")
        ax.spines["top"].set_color("#e0e0e0")
        ax.spines["top"].set_linewidth(0.5)
        ax.spines["right"].set_color("#e0e0e0")
        ax.spines["right"].set_linewidth(0.5)
        ax.spines["bottom"].set_color("#2f2f2f")
        ax.spines["bottom"].set_linewidth(1)
        ax.spines["left"].set_color("#2f2f2f")
        ax.spines["left"].set_linewidth(1)

        # Grid styling - subtle engineering grid
        ax.grid(True, which="major", alpha=0.3, linewidth=0.5, color="#d0d0d0")
        ax.grid(True, which="minor", alpha=0.15, linewidth=0.25, color="#e8e8e8")
        ax.minorticks_on()
        ax.set_axisbelow(True)

        # Professional tick styling
        ax.tick_params(
            axis="both",
            which="major",
            labelsize=10,
            direction="out",
            length=4,
            width=1,
            colors="#2f2f2f",
        )
        ax.tick_params(
            axis="both",
            which="minor",
            direction="out",
            length=2,
            width=0.5,
            colors="#2f2f2f",
        )

    def create_figure(
        self, width: float = 12, height: float = 8
    ) -> Tuple[Figure, Axes]:
        """Create professional figure with engineering styling."""
        fig, ax = plt.subplots(figsize=(width, height), dpi=300)
        ax.set_aspect("equal")
        self.setup_professional_style(fig, ax)
        return fig, ax

    def plot_professional_route(
        self,
        route: Route,
        title: Optional[str] = None,
        units: str = "m",
        show_annotations: bool = True,
        show_section_colors: bool = True,
        label_start_index: int = 0,
        show_fitted_geometry: bool = True,
    ) -> Tuple[Figure, Axes]:
        """Create professional route visualization."""
        # Determine orientation based on route bounds
        all_points = []
        for section in route.sections:
            if section.original_polyline:
                all_points.extend(section.original_polyline)
        
        if all_points:
            x_coords = [p[0] for p in all_points]
            y_coords = [p[1] for p in all_points]
            x_range = max(x_coords) - min(x_coords)
            y_range = max(y_coords) - min(y_coords)
            
            # A4 dimensions in inches at 300 DPI
            # A4 = 210mm x 297mm = 8.27" x 11.69"
            if x_range > y_range:
                # Landscape orientation for wider routes
                fig_width = 11.69
                fig_height = 8.27
            else:
                # Portrait orientation for taller routes
                fig_width = 8.27
                fig_height = 11.69
        else:
            # Default to landscape
            fig_width = 11.69
            fig_height = 8.27
            
        fig, ax = self.create_figure(width=fig_width, height=fig_height)

        # Default title
        if title is None:
            title = f"Cable Route Analysis: {route.name}"

        # Unit conversion
        unit_scale = 1000 if units == "mm" else 1
        unit_label = f"Coordinate ({units})"

        # Generate node labels
        labels = list(string.ascii_uppercase)
        if len(labels) < route.section_count + 1:
            for i in range(len(labels), route.section_count + 1):
                labels.append(f"{labels[i % 26]}{i // 26 + 1}")

        # Collect all points for scaling
        all_points = []
        joint_points = []
        joint_labels = []

        # Plot sections with professional styling
        for i, section in enumerate(route.sections):
            if not section.original_polyline:
                continue

            # Convert coordinates
            x_coords = [p[0] * unit_scale for p in section.original_polyline]
            y_coords = [p[1] * unit_scale for p in section.original_polyline]
            all_points.extend(list(zip(x_coords, y_coords)))

            # Section styling with consistent naming
            if show_section_colors:
                color = self.colors["sections"][i % len(self.colors["sections"])]
                # Use actual section ID, removing "SECT_" prefix for cleaner display
                section_name = section.id.replace("SECT_", "").replace("_", "-")
                label = f"Section {section_name}"
            else:
                color = self.colors["primary"]
                label = (
                    f"Section {section.id.replace('SECT_', '').replace('_', '-')}"
                    if i == 0
                    else None
                )

            # Plot original route as thin black line
            ax.plot(
                x_coords,
                y_coords,
                color="black",
                linewidth=1.0,
                linestyle="-",
                alpha=0.8,
                label="Original route" if i == 0 else None,
                solid_capstyle="round",
                solid_joinstyle="round",
                zorder=3,
            )

            # Plot fitted geometry (straights and bends)
            if section.primitives and show_fitted_geometry:
                self._plot_fitted_geometry(ax, section, i, color, unit_scale, labels)

            # Collect joint points
            if i == 0:  # First section
                joint_points.append((x_coords[0], y_coords[0]))
                joint_labels.append(labels[label_start_index])
            joint_points.append((x_coords[-1], y_coords[-1]))
            joint_labels.append(labels[label_start_index + i + 1])

        # Add professional joint markers
        if joint_points:
            # Calculate marker size based on plot scale
            if all_points:
                x_range = max(p[0] for p in all_points) - min(p[0] for p in all_points)
                marker_size = max(150, x_range * 0.0008)  # Dynamic sizing
            else:
                marker_size = 200

            for point, label in zip(joint_points, joint_labels):
                # White circle with dark border (professional CAD style)
                ax.scatter(
                    point[0],
                    point[1],
                    s=marker_size,
                    c="white",
                    edgecolors="#2f2f2f",
                    linewidth=2.5,
                    zorder=10,
                    alpha=1.0,
                )

                # Professional label styling
                ax.annotate(
                    label,
                    point,
                    ha="center",
                    va="center",
                    fontsize=12,
                    fontweight="bold",
                    color="#2f2f2f",
                    zorder=11,
                )

        # Add fitted geometry overlay if available
        self._add_fitted_overlay(ax, route, unit_scale)

        # Professional title styling
        ax.set_title(title, fontsize=16, fontweight="bold", color="#1a1a1a", pad=20)

        # Professional axis labels
        ax.set_xlabel(f"X {unit_label}", **self.fonts["label"])
        ax.set_ylabel(f"Y {unit_label}", **self.fonts["label"])

        # Professional legend
        if show_section_colors:
            legend = ax.legend(
                loc="lower left",
                frameon=True,
                fancybox=False,
                shadow=False,
                framealpha=0.95,
                edgecolor="#2f2f2f",
                facecolor="white",
                fontsize=10,
                ncol=2 if route.section_count > 8 else 1,
            )
            legend.get_frame().set_linewidth(1)

        # Add professional statistics box
        total_length = sum(s.original_length for s in route.sections)
        stats_text = (
            f"Route Statistics\n"
            f"Sections: {route.section_count}\n"
            f"Total Length: {total_length:.1f}m"
        )

        # Professional text box
        props = dict(
            boxstyle="round,pad=0.8",
            facecolor="#f8f9fa",
            edgecolor="#2f2f2f",
            alpha=0.95,
        )
        ax.text(
            0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=props,
            family="DejaVu Sans",
        )

        # Auto-scale with professional margins
        ax.autoscale()
        ax.margins(0.05)

        # Remove excessive whitespace
        plt.tight_layout()

        return fig, ax

    def _plot_fitted_geometry(
        self, ax, section, section_index, color, unit_scale, labels
    ):
        """Plot fitted geometry showing straights and bends separately."""
        from ..core.models import Bend, Straight

        current_x, current_y = None, None
        straight_plotted = False
        bend_plotted = False

        for i, primitive in enumerate(section.primitives):
            if isinstance(primitive, Straight):
                # Plot straight as blue line with 50% opacity
                start_point = primitive.start_point
                end_point = primitive.end_point

                x_coords = [start_point[0] * unit_scale, end_point[0] * unit_scale]
                y_coords = [start_point[1] * unit_scale, end_point[1] * unit_scale]

                ax.plot(
                    x_coords,
                    y_coords,
                    color="blue",
                    linewidth=2.0,
                    alpha=0.5,
                    label="Fitted straights" if not straight_plotted else None,
                    solid_capstyle="round",
                    zorder=4,
                )

                straight_plotted = True
                current_x, current_y = (
                    end_point[0] * unit_scale,
                    end_point[1] * unit_scale,
                )

            elif isinstance(primitive, Bend):
                # Plot bend as red arc with 50% opacity
                center = primitive.center_point
                radius = primitive.radius_m

                # Calculate arc points
                start_angle = primitive.start_angle_deg
                end_angle = primitive.end_angle_deg

                # Generate arc points
                start_rad = np.radians(start_angle)
                end_rad = np.radians(end_angle)

                if primitive.direction == "CW":
                    # For clockwise, ensure we go the short way
                    if end_rad > start_rad:
                        end_rad -= 2 * np.pi
                    angles = np.linspace(start_rad, end_rad, 50)
                else:
                    # For counter-clockwise, ensure we go the short way
                    if end_rad < start_rad:
                        end_rad += 2 * np.pi
                    angles = np.linspace(start_rad, end_rad, 50)

                arc_x = center[0] * unit_scale + radius * unit_scale * np.cos(angles)
                arc_y = center[1] * unit_scale + radius * unit_scale * np.sin(angles)

                ax.plot(
                    arc_x,
                    arc_y,
                    color="red",
                    linewidth=2.0,
                    alpha=0.5,
                    label="Fitted bends" if not bend_plotted else None,
                    solid_capstyle="round",
                    zorder=4,
                )

                bend_plotted = True
                current_x, current_y = arc_x[-1], arc_y[-1]

    def _add_fitted_overlay(self, ax: Axes, route: Route, unit_scale: float) -> None:
        """Add fitted geometry overlay with professional styling."""
        straight_segments = []
        bend_markers = []

        for section in route.sections:
            for primitive in section.primitives:
                if isinstance(primitive, Straight):
                    start = (
                        primitive.start_point[0] * unit_scale,
                        primitive.start_point[1] * unit_scale,
                    )
                    end = (
                        primitive.end_point[0] * unit_scale,
                        primitive.end_point[1] * unit_scale,
                    )
                    straight_segments.append([start, end])

                elif isinstance(primitive, Bend):
                    center = (
                        primitive.center_point[0] * unit_scale,
                        primitive.center_point[1] * unit_scale,
                    )
                    bend_markers.append((center, primitive.radius_m * unit_scale))

        # Plot fitted straights as subtle overlay
        for segment in straight_segments:
            ax.plot(
                [segment[0][0], segment[1][0]],
                [segment[0][1], segment[1][1]],
                color=self.colors["neutral"],
                linewidth=1.5,
                linestyle="--",
                alpha=0.6,
                zorder=3,
            )

        # Bend centers removed per user request

    def plot_tension_analysis(
        self,
        route: Route,
        tension_data: Dict[str, List[float]],
        title: Optional[str] = None,
    ) -> Tuple[Figure, Axes]:
        """Create professional tension analysis plot."""
        # A4 landscape for tension plots (wide format)
        fig, ax = plt.subplots(figsize=(11.69, 8.27), dpi=300)
        self.setup_professional_style(fig, ax)

        if title is None:
            title = f"Cable Tension Analysis: {route.name}"

        # Main tension curve with professional styling
        ax.plot(
            tension_data["chainage"],
            tension_data["tension"],
            color=self.colors["primary"],
            linewidth=3,
            alpha=0.9,
            label="Cable Tension",
            zorder=5,
        )

        # Add subtle fill under curve
        ax.fill_between(
            tension_data["chainage"],
            tension_data["tension"],
            alpha=0.15,
            color=self.colors["primary"],
            zorder=2,
        )

        # Add allowable tension limit
        if "max_allowable" in tension_data:
            ax.axhline(
                y=tension_data["max_allowable"],
                color=self.colors["accent"],
                linewidth=2,
                linestyle="--",
                alpha=0.8,
                label=f'Max Allowable ({tension_data["max_allowable"]:.0f}N)',
                zorder=4,
            )

        # Professional styling
        ax.set_title(title, **self.fonts["title"], pad=20)
        ax.set_xlabel("Chainage (m)", **self.fonts["label"])
        ax.set_ylabel("Cable Tension (N)", **self.fonts["label"])

        # Professional legend
        legend = ax.legend(
            loc="upper right",
            frameon=True,
            fancybox=False,
            edgecolor="#2f2f2f",
            facecolor="white",
            framealpha=0.95,
        )
        legend.get_frame().set_linewidth(1)

        # Set reasonable y-limits
        max_tension = max(tension_data["tension"])
        ax.set_ylim(0, max_tension * 1.1)

        plt.tight_layout()
        return fig, ax

    def plot_pressure_analysis(
        self,
        route: Route,
        pressure_data: Dict[str, List[float]],
        title: Optional[str] = None,
    ) -> Tuple[Figure, Axes]:
        """Create professional pressure analysis plot."""
        # A4 landscape for pressure plots (wide format)
        fig, ax = plt.subplots(figsize=(11.69, 8.27), dpi=300)
        self.setup_professional_style(fig, ax)

        if title is None:
            title = f"Sidewall Pressure Analysis: {route.name}"

        # Color-code pressure points by severity
        pressures = pressure_data["pressure"]
        chainages = pressure_data["chainage"]
        max_allowable = pressure_data.get("max_allowable", max(pressures))

        # Create color array based on pressure ratio
        colors = []
        for p in pressures:
            ratio = p / max_allowable
            if ratio <= 0.7:
                colors.append(self.colors["success"])
            elif ratio <= 0.9:
                colors.append(self.colors["warning"])
            else:
                colors.append(self.colors["accent"])

        # Scatter plot with color coding
        scatter = ax.scatter(
            chainages,
            pressures,
            c=colors,
            s=80,
            alpha=0.8,
            edgecolors="white",
            linewidth=1,
            zorder=6,
        )

        # Connect points with subtle line
        ax.plot(
            chainages,
            pressures,
            color=self.colors["neutral"],
            linewidth=1.5,
            alpha=0.6,
            zorder=3,
        )

        # Add allowable pressure limit
        ax.axhline(
            y=max_allowable,
            color=self.colors["accent"],
            linewidth=2,
            linestyle="--",
            alpha=0.8,
            label=f"Max Allowable ({max_allowable:.1f}N/m)",
            zorder=4,
        )

        # Professional styling
        ax.set_title(title, **self.fonts["title"], pad=20)
        ax.set_xlabel("Chainage (m)", **self.fonts["label"])
        ax.set_ylabel("Sidewall Pressure (N/m)", **self.fonts["label"])

        # Custom legend for pressure levels
        from matplotlib.lines import Line2D

        legend_elements = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=self.colors["success"],
                markersize=8,
                label="Safe (<70%)",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=self.colors["warning"],
                markersize=8,
                label="Caution (70-90%)",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=self.colors["accent"],
                markersize=8,
                label="Critical (>90%)",
            ),
            Line2D(
                [0],
                [0],
                color=self.colors["accent"],
                linewidth=2,
                linestyle="--",
                label=f"Max Allowable ({max_allowable:.1f}N/m)",
            ),
        ]

        legend = ax.legend(
            handles=legend_elements,
            loc="upper right",
            frameon=True,
            fancybox=False,
            edgecolor="#2f2f2f",
            facecolor="white",
            framealpha=0.95,
        )
        legend.get_frame().set_linewidth(1)

        plt.tight_layout()
        return fig, ax

    def create_analysis_dashboard(
        self, route: Route, analysis_data: Dict[str, Any], title: Optional[str] = None
    ) -> Tuple[Figure, List[Axes]]:
        """Create comprehensive analysis dashboard."""
        if title is None:
            title = f"Cable Pulling Analysis Dashboard: {route.name}"

        # Create 2x2 subplot layout - A4 landscape for dashboard
        fig = plt.figure(figsize=(11.69, 8.27), dpi=300)
        fig.suptitle(title, fontsize=18, fontweight="bold", color="#1a1a1a", y=0.95)

        # Configure subplot layout with proper spacing
        gs = fig.add_gridspec(
            2, 2, hspace=0.3, wspace=0.25, left=0.08, right=0.95, top=0.88, bottom=0.08
        )

        # Route overview (top-left)
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.set_aspect("equal")
        self.setup_professional_style(fig, ax1)
        self._plot_route_overview(ax1, route)
        ax1.set_title("Route Overview", **self.fonts["title"], pad=15)

        # Tension analysis (top-right)
        ax2 = fig.add_subplot(gs[0, 1])
        self.setup_professional_style(fig, ax2)
        if "tension" in analysis_data:
            self._plot_tension_subplot(ax2, analysis_data["tension"])
        ax2.set_title("Tension Analysis", **self.fonts["title"], pad=15)

        # Pressure analysis (bottom-left)
        ax3 = fig.add_subplot(gs[1, 0])
        self.setup_professional_style(fig, ax3)
        if "pressure" in analysis_data:
            self._plot_pressure_subplot(ax3, analysis_data["pressure"])
        ax3.set_title("Pressure Analysis", **self.fonts["title"], pad=15)

        # Section summary (bottom-right)
        ax4 = fig.add_subplot(gs[1, 1])
        self.setup_professional_style(fig, ax4)
        self._plot_section_summary(ax4, route, analysis_data)
        ax4.set_title("Section Summary", **self.fonts["title"], pad=15)

        return fig, [ax1, ax2, ax3, ax4]

    def _plot_route_overview(self, ax: Axes, route: Route) -> None:
        """Plot route overview in subplot."""
        labels = list(string.ascii_uppercase)

        for i, section in enumerate(route.sections):
            if not section.original_polyline:
                continue

            x_coords = [p[0] for p in section.original_polyline]
            y_coords = [p[1] for p in section.original_polyline]
            color = self.colors["sections"][i % len(self.colors["sections"])]

            ax.plot(x_coords, y_coords, color=color, linewidth=2, alpha=0.8)

            # Add section labels at midpoints
            mid_idx = len(x_coords) // 2
            ax.annotate(
                f"{labels[i]}-{labels[i+1]}",
                (x_coords[mid_idx], y_coords[mid_idx]),
                fontsize=8,
                ha="center",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            )

        ax.set_xlabel("X (m)", **self.fonts["label"])
        ax.set_ylabel("Y (m)", **self.fonts["label"])

    def _plot_tension_subplot(self, ax: Axes, tension_data: Dict) -> None:
        """Plot tension analysis in subplot."""
        ax.plot(
            tension_data["chainage"],
            tension_data["tension"],
            color=self.colors["primary"],
            linewidth=2,
        )
        ax.set_xlabel("Chainage (m)", **self.fonts["label"])
        ax.set_ylabel("Tension (N)", **self.fonts["label"])

    def _plot_pressure_subplot(self, ax: Axes, pressure_data: Dict) -> None:
        """Plot pressure analysis in subplot."""
        ax.scatter(
            pressure_data["chainage"],
            pressure_data["pressure"],
            c=self.colors["accent"],
            s=40,
            alpha=0.7,
        )
        ax.set_xlabel("Chainage (m)", **self.fonts["label"])
        ax.set_ylabel("Pressure (N/m)", **self.fonts["label"])

    def _plot_section_summary(
        self, ax: Axes, route: Route, analysis_data: Dict
    ) -> None:
        """Plot section summary bar chart."""
        section_names = [f"S{i+1}" for i in range(route.section_count)]
        section_lengths = [s.original_length for s in route.sections]

        bars = ax.bar(
            section_names,
            section_lengths,
            color=self.colors["secondary"],
            alpha=0.8,
            edgecolor="white",
            linewidth=1,
        )

        # Add value labels on bars
        for bar, length in zip(bars, section_lengths):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + max(section_lengths) * 0.01,
                f"{length:.0f}m",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        ax.set_xlabel("Section", **self.fonts["label"])
        ax.set_ylabel("Length (m)", **self.fonts["label"])
        ax.set_ylim(0, max(section_lengths) * 1.1)

    def save_professional_plot(
        self, fig: Figure, output_path: Path, dpi: int = 300, transparent: bool = False
    ) -> None:
        """Save plot with professional quality settings."""
        fig.savefig(
            output_path,
            dpi=dpi,
            bbox_inches="tight",
            facecolor="white" if not transparent else "none",
            edgecolor="none",
            pad_inches=0.1,  # Reduced padding to maximize use of A4 space
            transparent=transparent,
        )


def create_professional_route_plot_matplotlib(
    route: Route,
    output_path: Optional[Path] = None,
    title: Optional[str] = None,
    units: str = "m",
    **kwargs,
) -> Tuple[Figure, Axes]:
    """Create professional route plot using matplotlib.

    Args:
        route: Route object
        output_path: Optional output file path
        title: Plot title
        units: Display units
        **kwargs: Additional plotting options

    Returns:
        Figure and axes objects
    """
    plotter = ProfessionalMatplotlibPlotter()
    fig, ax = plotter.plot_professional_route(route, title=title, units=units, **kwargs)

    if output_path:
        plotter.save_professional_plot(fig, output_path)

    return fig, ax
