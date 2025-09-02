"""Route visualization and plotting functionality."""

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ..core.models import Bend, Route, Section, Straight
from ..io.dxf_reader import DXFReader


class RoutePlotter:
    """Plotter for cable routes and DXF analysis results."""

    def __init__(self, figsize: Tuple[float, float] = (11.69, 8.27)):
        """Initialize plotter.

        Args:
            figsize: Figure size in inches (width, height)
                     Default is A4 landscape (11.69" x 8.27")
        """
        self.figsize = figsize
        self.colors = {
            "original": "#2E86AB",  # Blue
            "fitted": "#A23B72",  # Purple
            "straights": "#F18F01",  # Orange
            "bends": "#C73E1D",  # Red
            "joints": "#592E83",  # Dark purple
            "annotations": "#333333",  # Dark gray
        }

    def create_figure(self) -> Tuple[Figure, Axes]:
        """Create matplotlib figure and axes."""
        # Set DPI for the figure
        plt.rcParams["figure.dpi"] = 100  # Display DPI
        plt.rcParams["savefig.dpi"] = 300  # Save DPI

        fig, ax = plt.subplots(figsize=self.figsize)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("X (m)", fontsize=12)
        ax.set_ylabel("Y (m)", fontsize=12)
        return fig, ax

    def plot_polyline(
        self,
        ax: Axes,
        points: List[Tuple[float, float]],
        color: str = "blue",
        linestyle: str = "-",
        linewidth: float = 2.0,
        label: Optional[str] = None,
        alpha: float = 1.0,
    ) -> None:
        """Plot a polyline on the axes.

        Args:
            ax: Matplotlib axes
            points: List of (x, y) coordinate tuples
            color: Line color
            linestyle: Line style
            linewidth: Line width
            label: Legend label
            alpha: Transparency
        """
        if len(points) < 2:
            return

        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]

        ax.plot(
            x_coords,
            y_coords,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            label=label,
            alpha=alpha,
        )

    def plot_primitives(
        self,
        ax: Axes,
        section: Section,
        show_straights: bool = True,
        show_bends: bool = True,
    ) -> None:
        """Plot fitted primitives for a section.

        Args:
            ax: Matplotlib axes
            section: Section with fitted primitives
            show_straights: Whether to show straight segments
            show_bends: Whether to show bend segments
        """
        for primitive in section.primitives:
            if isinstance(primitive, Straight) and show_straights:
                self.plot_polyline(
                    ax,
                    [primitive.start_point, primitive.end_point],
                    color=self.colors["straights"],
                    linewidth=3.0,
                    linestyle="-",
                    alpha=0.8,
                )

            elif isinstance(primitive, Bend) and show_bends:
                # Plot bend as arc
                center = primitive.center_point
                radius = primitive.radius_m

                # Calculate start and end angles for the arc
                # This is simplified - real implementation needs proper angle calculation
                start_angle = 0
                end_angle = primitive.angle_deg

                if primitive.direction == "CCW":
                    start_angle, end_angle = end_angle, start_angle

                arc = patches.Arc(
                    center,
                    2 * radius,
                    2 * radius,
                    angle=0,
                    theta1=start_angle,
                    theta2=end_angle,
                    color=self.colors["bends"],
                    linewidth=3.0,
                    linestyle="--",
                )
                ax.add_patch(arc)

    def plot_joints(self, ax: Axes, route: Route, marker_size: float = 100) -> None:
        """Plot joint/pit markers between sections.

        Args:
            ax: Matplotlib axes
            route: Route object
            marker_size: Size of joint markers
        """
        joint_points = []

        # Add start of first section
        if route.sections:
            first_section = route.sections[0]
            if first_section.original_polyline:
                joint_points.append(first_section.original_polyline[0])

        # Add end of each section (which is start of next)
        for section in route.sections:
            if section.original_polyline:
                joint_points.append(section.original_polyline[-1])

        if joint_points:
            x_coords = [p[0] for p in joint_points]
            y_coords = [p[1] for p in joint_points]

            ax.scatter(
                x_coords,
                y_coords,
                s=marker_size,
                color=self.colors["joints"],
                marker="o",
                edgecolor="white",
                linewidth=2,
                label="Joints/Pits",
                zorder=10,
            )

    def plot_annotations(self, ax: Axes, route: Route) -> None:
        """Add section annotations to the plot.

        Args:
            ax: Matplotlib axes
            route: Route object
        """
        for section in route.sections:
            if not section.original_polyline:
                continue

            # Place annotation at midpoint of section
            mid_idx = len(section.original_polyline) // 2
            mid_point = section.original_polyline[mid_idx]

            # Offset text slightly to avoid overlap
            text_x = mid_point[0] + 20
            text_y = mid_point[1] + 20

            annotation = f"{section.id}\n{section.original_length:.0f}m"

            ax.annotate(
                annotation,
                xy=mid_point,
                xytext=(text_x, text_y),
                fontsize=8,
                color=self.colors["annotations"],
                ha="left",
                va="bottom",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                arrowprops=dict(
                    arrowstyle="->", color=self.colors["annotations"], alpha=0.5
                ),
            )

    def plot_route(
        self,
        route: Route,
        show_original: bool = True,
        show_fitted: bool = True,
        show_joints: bool = True,
        show_annotations: bool = True,
        title: Optional[str] = None,
    ) -> Tuple[Figure, Axes]:
        """Plot a complete route with all components.

        Args:
            route: Route object to plot
            show_original: Show original polylines
            show_fitted: Show fitted primitives
            show_joints: Show joint markers
            show_annotations: Show section annotations
            title: Plot title (defaults to route name)

        Returns:
            Matplotlib figure and axes
        """
        fig, ax = self.create_figure()

        if title is None:
            title = f"Cable Route: {route.name}"
        ax.set_title(title, fontsize=14, fontweight="bold")

        # Plot original polylines
        if show_original:
            for section in route.sections:
                self.plot_polyline(
                    ax,
                    section.original_polyline,
                    color=self.colors["original"],
                    linewidth=2.0,
                    label="Original Route" if section == route.sections[0] else None,
                    alpha=0.7,
                )

        # Plot fitted primitives
        if show_fitted and any(section.primitives for section in route.sections):
            for section in route.sections:
                self.plot_primitives(ax, section)

        # Add joint markers
        if show_joints:
            self.plot_joints(ax, route)

        # Add annotations
        if show_annotations:
            self.plot_annotations(ax, route)

        # Add legend
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="upper right", fontsize=10)

        # Add route statistics
        total_length = sum(s.original_length for s in route.sections)
        stats_text = (
            f"Sections: {route.section_count}\nTotal Length: {total_length:.1f}m"
        )

        ax.text(
            0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
        )

        plt.tight_layout()
        return fig, ax

    def plot_dxf_layers(
        self,
        dxf_path: Path,
        layers_to_show: Optional[List[str]] = None,
        title: Optional[str] = None,
    ) -> Tuple[Figure, Axes]:
        """Plot polylines from DXF file layers.

        Args:
            dxf_path: Path to DXF file
            layers_to_show: Specific layers to plot (None for all)
            title: Plot title

        Returns:
            Matplotlib figure and axes
        """
        fig, ax = self.create_figure()

        # Load DXF file
        reader = DXFReader(dxf_path)
        reader.load()

        # Get available layers
        all_layers = reader.get_layers()
        if layers_to_show is None:
            layers_to_show = all_layers

        color_cycle = plt.cm.tab10(range(len(layers_to_show)))

        for i, layer in enumerate(layers_to_show):
            if layer not in all_layers:
                continue

            polylines = reader.extract_polylines(layer)
            color = color_cycle[i % len(color_cycle)]

            for j, (layer_name, points) in enumerate(polylines):
                label = f"Layer: {layer_name}" if j == 0 else None
                self.plot_polyline(ax, points, color=color, linewidth=2.0, label=label)

        if title is None:
            title = f"DXF Layers: {dxf_path.name}"
        ax.set_title(title, fontsize=14, fontweight="bold")

        # Add legend if multiple layers
        handles, labels = ax.get_legend_handles_labels()
        if len(handles) > 1:
            ax.legend(loc="upper right", fontsize=10)

        plt.tight_layout()
        return fig, ax

    def plot_comparison(
        self,
        original_dxf: Path,
        analysis_dxf: Path,
        route: Optional[Route] = None,
        title: Optional[str] = None,
    ) -> Tuple[Figure, Axes]:
        """Plot comparison between original and analysis DXF files.

        Args:
            original_dxf: Path to original DXF file
            analysis_dxf: Path to analysis output DXF file
            route: Optional Route object for additional info
            title: Plot title

        Returns:
            Matplotlib figure and axes
        """
        fig, ax = self.create_figure()

        # Load original DXF
        original_reader = DXFReader(original_dxf)
        original_reader.load()
        original_polylines = original_reader.extract_polylines()

        # Load analysis DXF
        analysis_reader = DXFReader(analysis_dxf)
        analysis_reader.load()
        analysis_polylines = analysis_reader.extract_polylines()

        # Plot original route
        for layer_name, points in original_polylines:
            self.plot_polyline(
                ax,
                points,
                color=self.colors["original"],
                linewidth=3.0,
                label="Original Route",
                alpha=0.8,
            )

        # Plot analysis results by layer
        layer_styles = {
            "ROUTE_FITTED": {
                "color": self.colors["fitted"],
                "linestyle": "-",
                "width": 2.5,
            },
            "PRIMITIVES_STRAIGHT": {
                "color": self.colors["straights"],
                "linestyle": "-",
                "width": 2.0,
            },
            "PRIMITIVES_BEND": {
                "color": self.colors["bends"],
                "linestyle": "--",
                "width": 2.0,
            },
            "JOINTS": {"color": self.colors["joints"], "linestyle": ":", "width": 1.5},
        }

        plotted_labels = set()
        for layer_name, points in analysis_polylines:
            style = layer_styles.get(
                layer_name, {"color": "gray", "linestyle": "-", "width": 1.0}
            )

            # Create label for legend (only once per layer type)
            label = None
            if layer_name not in plotted_labels:
                if "FITTED" in layer_name:
                    label = "Fitted Route"
                elif "STRAIGHT" in layer_name:
                    label = "Straight Segments"
                elif "BEND" in layer_name:
                    label = "Bend Segments"
                elif "JOINT" in layer_name:
                    label = "Joints"
                plotted_labels.add(layer_name)

            self.plot_polyline(
                ax,
                points,
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=style["width"],
                label=label,
                alpha=0.9,
            )

        # Add route information if provided
        if route:
            total_length = sum(s.original_length for s in route.sections)
            stats_text = (
                f"Route: {route.name}\n"
                f"Sections: {route.section_count}\n"
                f"Total Length: {total_length:.1f}m"
            )

            ax.text(
                0.02,
                0.98,
                stats_text,
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.9),
            )

        if title is None:
            title = "DXF Analysis Comparison"
        ax.set_title(title, fontsize=14, fontweight="bold")

        # Add legend
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="upper right", fontsize=10)

        plt.tight_layout()
        return fig, ax

    def save_plot(
        self, fig: Figure, output_path: Path, dpi: int = 300, format: str = "png"
    ) -> None:
        """Save plot to file.

        Args:
            fig: Matplotlib figure
            output_path: Output file path
            dpi: Resolution for raster formats
            format: Output format (png, svg, pdf)
        """
        fig.savefig(
            output_path,
            dpi=dpi,
            format=format,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )


def plot_route(
    route: Route,
    output_path: Optional[Path] = None,
    show_original: bool = True,
    show_fitted: bool = True,
    show_joints: bool = True,
    show_annotations: bool = True,
    figsize: Tuple[float, float] = (11.69, 8.27),
) -> Tuple[Figure, Axes]:
    """Convenience function to plot a route.

    Args:
        route: Route object to plot
        output_path: Optional path to save plot
        show_original: Show original polylines
        show_fitted: Show fitted primitives
        show_joints: Show joint markers
        show_annotations: Show section annotations
        figsize: Figure size in inches

    Returns:
        Matplotlib figure and axes
    """
    plotter = RoutePlotter(figsize=figsize)
    fig, ax = plotter.plot_route(
        route,
        show_original=show_original,
        show_fitted=show_fitted,
        show_joints=show_joints,
        show_annotations=show_annotations,
    )

    if output_path:
        plotter.save_plot(fig, output_path)

    return fig, ax


def plot_dxf_comparison(
    original_dxf: Path,
    analysis_dxf: Path,
    route: Optional[Route] = None,
    output_path: Optional[Path] = None,
    figsize: Tuple[float, float] = (11.69, 8.27),
) -> Tuple[Figure, Axes]:
    """Convenience function to plot DXF comparison.

    Args:
        original_dxf: Path to original DXF file
        analysis_dxf: Path to analysis output DXF file
        route: Optional Route object for additional info
        output_path: Optional path to save plot
        figsize: Figure size in inches

    Returns:
        Matplotlib figure and axes
    """
    plotter = RoutePlotter(figsize=figsize)
    fig, ax = plotter.plot_comparison(original_dxf, analysis_dxf, route=route)

    if output_path:
        plotter.save_plot(fig, output_path)

    return fig, ax
