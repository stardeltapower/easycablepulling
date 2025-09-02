"""Professional visualization using Plotly for publication-quality outputs."""

import math
import string
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from ..core.models import Bend, Route, Section, Straight


class ProfessionalPlotter:
    """Create professional, publication-quality visualizations using Plotly."""

    def __init__(self):
        """Initialize professional plotter with engineering-grade styling."""
        # Professional color scheme based on engineering standards
        self.colors = {
            # Primary route colors (engineering blue tones)
            "original": "#1f77b4",  # Engineering blue
            "fitted": "#ff7f0e",  # Engineering orange
            "deviation": "#d62728",  # Engineering red
            # Section colors (distinct but harmonious)
            "sections": [
                "#2e8b57",  # Sea green
                "#4169e1",  # Royal blue
                "#dc143c",  # Crimson
                "#ff8c00",  # Dark orange
                "#9370db",  # Medium purple
                "#008b8b",  # Dark cyan
                "#b22222",  # Fire brick
                "#228b22",  # Forest green
                "#4b0082",  # Indigo
                "#ff1493",  # Deep pink
                "#00ced1",  # Dark turquoise
                "#ff4500",  # Orange red
                "#8b008b",  # Dark magenta
                "#556b2f",  # Dark olive green
                "#800080",  # Purple
            ],
            # Component colors
            "straights": "#2e8b57",  # Sea green
            "bends": "#dc143c",  # Crimson
            "joints": "#4a4a4a",  # Dark gray
            "annotations": "#2f2f2f",  # Charcoal
            # Status colors
            "pass": "#28a745",  # Success green
            "warning": "#ffc107",  # Warning yellow
            "fail": "#dc3545",  # Danger red
        }

        # Professional layout configuration
        self.layout_config = {
            "font": {"family": "Arial, sans-serif", "size": 12, "color": "#2f2f2f"},
            "plot_bgcolor": "white",
            "paper_bgcolor": "white",
            "showlegend": True,
            "legend": {
                "bgcolor": "rgba(255,255,255,0.9)",
                "bordercolor": "#e0e0e0",
                "borderwidth": 1,
                "font": {"size": 11},
            },
            "margin": {"l": 60, "r": 20, "t": 80, "b": 60},
        }

        # Grid configuration for engineering drawings
        self.grid_config = {
            "showgrid": True,
            "gridwidth": 0.5,
            "gridcolor": "#e0e0e0",
            "zeroline": True,
            "zerolinewidth": 1,
            "zerolinecolor": "#c0c0c0",
        }

    def create_base_layout(
        self,
        title: str,
        xaxis_title: str = "X Coordinate (m)",
        yaxis_title: str = "Y Coordinate (m)",
        width: int = 1200,
        height: int = 800,
    ) -> Dict[str, Any]:
        """Create base layout configuration."""
        layout = self.layout_config.copy()
        layout.update(
            {
                "title": {
                    "text": title,
                    "x": 0.5,
                    "xanchor": "center",
                    "font": {"size": 16, "color": "#1a1a1a"},
                },
                "xaxis": {
                    "title": {"text": xaxis_title, "font": {"size": 14}},
                    "scaleanchor": "y",
                    "scaleratio": 1,
                    **self.grid_config,
                },
                "yaxis": {
                    "title": {"text": yaxis_title, "font": {"size": 14}},
                    **self.grid_config,
                },
                "width": width,
                "height": height,
            }
        )
        return layout

    def plot_route_overview(
        self,
        route: Route,
        title: Optional[str] = None,
        show_fitted: bool = True,
        show_annotations: bool = True,
        show_deviation_analysis: bool = False,
        units: str = "m",
    ) -> go.Figure:
        """Create professional route overview plot.

        Args:
            route: Route object to visualize
            title: Plot title
            show_fitted: Whether to show fitted geometry
            show_annotations: Whether to show section labels
            units: Coordinate units (m or mm)

        Returns:
            Plotly figure object
        """
        if title is None:
            title = f"Cable Route Analysis: {route.name}"

        # Convert units for display
        unit_scale = 1000 if units == "mm" else 1
        xaxis_title = f"X Coordinate ({units})"
        yaxis_title = f"Y Coordinate ({units})"

        fig = go.Figure()

        # Generate node labels
        labels = list(string.ascii_uppercase)
        if len(labels) < route.section_count + 1:
            for i in range(len(labels), route.section_count + 1):
                labels.append(f"{labels[i % 26]}{i // 26 + 1}")

        # Collect joint points for markers
        joint_points = []
        joint_labels = []

        # Plot sections with distinct colors
        for i, section in enumerate(route.sections):
            if not section.original_polyline:
                continue

            # Get coordinates and convert units
            x_coords = [p[0] * unit_scale for p in section.original_polyline]
            y_coords = [p[1] * unit_scale for p in section.original_polyline]

            # Section color
            color = self.colors["sections"][i % len(self.colors["sections"])]

            # Create section label
            start_label = labels[i] if i < len(labels) else f"N{i}"
            end_label = labels[i + 1] if (i + 1) < len(labels) else f"N{i + 1}"
            section_label = f"Section {start_label}-{end_label}"

            # Add trace for this section
            fig.add_trace(
                go.Scatter(
                    x=x_coords,
                    y=y_coords,
                    mode="lines",
                    name=section_label,
                    line=dict(color=color, width=3, shape="linear"),
                    hovertemplate=(
                        f"<b>{section_label}</b><br>"
                        f"Length: {section.original_length:.1f}m<br>"
                        f"X: %{{x:.0f}} {units}<br>"
                        f"Y: %{{y:.0f}} {units}<br>"
                        "<extra></extra>"
                    ),
                )
            )

            # Collect joint points
            if i == 0:  # First section - add start point
                joint_points.append((x_coords[0], y_coords[0]))
                joint_labels.append(start_label)

            # Add end point
            joint_points.append((x_coords[-1], y_coords[-1]))
            joint_labels.append(end_label)

        # Add fitted geometry if requested
        if show_fitted:
            self._add_fitted_traces(fig, route, unit_scale, units)

        # Add joint markers
        if joint_points:
            fig.add_trace(
                go.Scatter(
                    x=[p[0] for p in joint_points],
                    y=[p[1] for p in joint_points],
                    mode="markers+text",
                    name="Joints/Pits",
                    marker=dict(
                        size=12,
                        color="white",
                        line=dict(color="#2f2f2f", width=2),
                        symbol="circle",
                    ),
                    text=joint_labels,
                    textposition="middle center",
                    textfont=dict(size=10, color="#2f2f2f", family="Arial Black"),
                    hovertemplate="<b>Joint %{text}</b><br>X: %{x:.0f} "
                    + units
                    + "<br>Y: %{y:.0f} "
                    + units
                    + "<extra></extra>",
                    showlegend=True,
                )
            )

        # Apply professional layout
        layout = self.create_base_layout(title, xaxis_title, yaxis_title)
        fig.update_layout(**layout)

        return fig

    def _add_fitted_traces(
        self, fig: go.Figure, route: Route, unit_scale: float, units: str
    ) -> None:
        """Add fitted geometry traces to the figure."""
        straight_x, straight_y = [], []
        bend_centers = []
        bend_radii = []
        bend_annotations = []

        for section in route.sections:
            for primitive in section.primitives:
                if isinstance(primitive, Straight):
                    # Collect straight segment points
                    start = (
                        primitive.start_point[0] * unit_scale,
                        primitive.start_point[1] * unit_scale,
                    )
                    end = (
                        primitive.end_point[0] * unit_scale,
                        primitive.end_point[1] * unit_scale,
                    )
                    straight_x.extend(
                        [start[0], end[0], None]
                    )  # None creates line breaks
                    straight_y.extend([start[1], end[1], None])

                elif isinstance(primitive, Bend):
                    # Collect bend information for annotation
                    center = (
                        primitive.center_point[0] * unit_scale,
                        primitive.center_point[1] * unit_scale,
                    )
                    bend_centers.append(center)
                    bend_radii.append(primitive.radius_m * unit_scale)
                    bend_annotations.append(
                        f"R={primitive.radius_m:.0f}m<br>θ={primitive.angle_deg:.1f}°"
                    )

        # Add straight segments
        if straight_x:
            fig.add_trace(
                go.Scatter(
                    x=straight_x,
                    y=straight_y,
                    mode="lines",
                    name="Fitted Straights",
                    line=dict(color=self.colors["straights"], width=2, dash="solid"),
                    hoverinfo="skip",
                    opacity=0.8,
                )
            )

        # Add bend markers
        if bend_centers:
            fig.add_trace(
                go.Scatter(
                    x=[c[0] for c in bend_centers],
                    y=[c[1] for c in bend_centers],
                    mode="markers",
                    name="Fitted Bends",
                    marker=dict(
                        size=8,
                        color=self.colors["bends"],
                        symbol="diamond",
                        line=dict(color="white", width=1),
                    ),
                    text=bend_annotations,
                    hovertemplate="<b>Bend</b><br>%{text}<br>X: %{x:.0f} "
                    + units
                    + "<br>Y: %{y:.0f} "
                    + units
                    + "<extra></extra>",
                )
            )

    def plot_tension_analysis(
        self,
        route: Route,
        tension_data: Dict[str, List[float]],
        title: Optional[str] = None,
    ) -> go.Figure:
        """Create professional tension vs chainage plot.

        Args:
            route: Route object
            tension_data: Dictionary with 'chainage' and 'tension' lists
            title: Plot title

        Returns:
            Plotly figure object
        """
        if title is None:
            title = f"Cable Tension Analysis: {route.name}"

        fig = go.Figure()

        # Main tension curve
        fig.add_trace(
            go.Scatter(
                x=tension_data["chainage"],
                y=tension_data["tension"],
                mode="lines",
                name="Cable Tension",
                line=dict(
                    color=self.colors["original"],
                    width=3,
                    shape="spline",
                    smoothing=0.3,
                ),
                fill="tonexty" if "y" in locals() else None,
                fillcolor="rgba(31, 119, 180, 0.1)",
                hovertemplate="<b>Tension Analysis</b><br>Chainage: %{x:.1f}m<br>Tension: %{y:.0f}N<extra></extra>",
            )
        )

        # Add critical points if available
        if "critical_points" in tension_data:
            critical_chainage = tension_data["critical_points"]["chainage"]
            critical_tension = tension_data["critical_points"]["tension"]

            fig.add_trace(
                go.Scatter(
                    x=critical_chainage,
                    y=critical_tension,
                    mode="markers",
                    name="Critical Points",
                    marker=dict(
                        size=10,
                        color=self.colors["fail"],
                        symbol="triangle-up",
                        line=dict(color="white", width=2),
                    ),
                    hovertemplate="<b>Critical Point</b><br>Chainage: %{x:.1f}m<br>Max Tension: %{y:.0f}N<extra></extra>",
                )
            )

        # Add allowable tension limit line
        if "max_allowable" in tension_data:
            max_tension = tension_data["max_allowable"]
            fig.add_hline(
                y=max_tension,
                line_dash="dash",
                line_color=self.colors["fail"],
                annotation_text=f"Max Allowable: {max_tension:.0f}N",
                annotation_position="top right",
            )

        # Professional layout
        layout = self.create_base_layout(
            title, "Chainage (m)", "Cable Tension (N)", width=1400, height=600
        )
        layout["yaxis"]["range"] = [0, max(tension_data["tension"]) * 1.1]
        fig.update_layout(**layout)

        return fig

    def plot_pressure_analysis(
        self,
        route: Route,
        pressure_data: Dict[str, List[float]],
        title: Optional[str] = None,
    ) -> go.Figure:
        """Create professional sidewall pressure visualization.

        Args:
            route: Route object
            pressure_data: Dictionary with pressure analysis data
            title: Plot title

        Returns:
            Plotly figure object
        """
        if title is None:
            title = f"Sidewall Pressure Analysis: {route.name}"

        fig = go.Figure()

        # Color-code pressure levels
        pressures = pressure_data["pressure"]
        max_allowable = pressure_data.get("max_allowable", max(pressures))

        # Create color scale based on percentage of allowable
        color_scale = []
        for p in pressures:
            ratio = p / max_allowable
            if ratio <= 0.7:
                color_scale.append(self.colors["pass"])  # Green
            elif ratio <= 0.9:
                color_scale.append(self.colors["warning"])  # Yellow
            else:
                color_scale.append(self.colors["fail"])  # Red

        # Pressure vs position plot
        fig.add_trace(
            go.Scatter(
                x=pressure_data["chainage"],
                y=pressures,
                mode="markers+lines",
                name="Sidewall Pressure",
                marker=dict(
                    size=8,
                    color=color_scale,
                    line=dict(color="white", width=1),
                    opacity=0.8,
                ),
                line=dict(color=self.colors["original"], width=2),
                hovertemplate="<b>Pressure Point</b><br>Chainage: %{x:.1f}m<br>Pressure: %{y:.1f}N/m<extra></extra>",
            )
        )

        # Add allowable pressure limit
        fig.add_hline(
            y=max_allowable,
            line_dash="dash",
            line_color=self.colors["fail"],
            annotation_text=f"Max Allowable: {max_allowable:.1f}N/m",
            annotation_position="top right",
        )

        # Professional layout
        layout = self.create_base_layout(
            title, "Chainage (m)", "Sidewall Pressure (N/m)", width=1400, height=600
        )
        fig.update_layout(**layout)

        return fig

    def plot_route_comparison(
        self,
        route: Route,
        title: Optional[str] = None,
        show_fitted: bool = True,
        show_deviation_analysis: bool = True,
        units: str = "m",
    ) -> go.Figure:
        """Create professional route comparison with original vs fitted geometry.

        Args:
            route: Route object
            title: Plot title
            show_fitted: Whether to show fitted geometry
            show_deviation_analysis: Whether to show deviation metrics
            units: Display units

        Returns:
            Plotly figure object
        """
        if title is None:
            title = f"Route Geometry Analysis: {route.name}"

        unit_scale = 1000 if units == "mm" else 1
        xaxis_title = f"X Coordinate ({units})"
        yaxis_title = f"Y Coordinate ({units})"

        fig = go.Figure()

        # Plot original route (all sections as continuous line)
        all_x, all_y = [], []
        for section in route.sections:
            if section.original_polyline:
                x_coords = [p[0] * unit_scale for p in section.original_polyline]
                y_coords = [p[1] * unit_scale for p in section.original_polyline]
                all_x.extend(x_coords)
                all_y.extend(y_coords)

        if all_x:
            fig.add_trace(
                go.Scatter(
                    x=all_x,
                    y=all_y,
                    mode="lines",
                    name="Original Route",
                    line=dict(color=self.colors["original"], width=4, opacity=0.7),
                    hovertemplate="<b>Original Route</b><br>X: %{x:.1f} "
                    + units
                    + "<br>Y: %{y:.1f} "
                    + units
                    + "<extra></extra>",
                )
            )

        # Plot fitted geometry if available
        if show_fitted and any(section.primitives for section in route.sections):
            self._add_fitted_geometry(fig, route, unit_scale, units)

        # Add section markers and labels
        self._add_section_markers(fig, route, unit_scale, units)

        # Add deviation analysis if requested
        if show_deviation_analysis:
            self._add_deviation_overlay(fig, route, unit_scale, units)

        # Apply layout
        layout = self.create_base_layout(title, xaxis_title, yaxis_title, 1600, 1000)
        fig.update_layout(**layout)

        return fig

    def _add_fitted_geometry(
        self, fig: go.Figure, route: Route, unit_scale: float, units: str
    ) -> None:
        """Add fitted geometry traces."""
        straight_x, straight_y = [], []

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
                    straight_x.extend([start[0], end[0], None])
                    straight_y.extend([start[1], end[1], None])

        if straight_x:
            fig.add_trace(
                go.Scatter(
                    x=straight_x,
                    y=straight_y,
                    mode="lines",
                    name="Fitted Geometry",
                    line=dict(color=self.colors["fitted"], width=2, dash="solid"),
                    opacity=0.9,
                    hoverinfo="skip",
                )
            )

    def _add_section_markers(
        self, fig: go.Figure, route: Route, unit_scale: float, units: str
    ) -> None:
        """Add professional section markers."""
        labels = list(string.ascii_uppercase)
        joint_points = []
        joint_labels = []

        # Collect joint data
        if route.sections:
            # First point
            first_section = route.sections[0]
            if first_section.original_polyline:
                point = first_section.original_polyline[0]
                joint_points.append((point[0] * unit_scale, point[1] * unit_scale))
                joint_labels.append(labels[0])

        # End points of each section
        for i, section in enumerate(route.sections):
            if section.original_polyline:
                point = section.original_polyline[-1]
                joint_points.append((point[0] * unit_scale, point[1] * unit_scale))
                joint_labels.append(
                    labels[i + 1] if (i + 1) < len(labels) else f"N{i + 1}"
                )

        if joint_points:
            fig.add_trace(
                go.Scatter(
                    x=[p[0] for p in joint_points],
                    y=[p[1] for p in joint_points],
                    mode="markers+text",
                    name="Access Points",
                    marker=dict(
                        size=16,
                        color="white",
                        line=dict(color="#2f2f2f", width=2),
                        symbol="circle",
                    ),
                    text=joint_labels,
                    textposition="middle center",
                    textfont=dict(size=12, color="#2f2f2f", family="Arial Black"),
                    hovertemplate="<b>Access Point %{text}</b><br>X: %{x:.1f} "
                    + units
                    + "<br>Y: %{y:.1f} "
                    + units
                    + "<extra></extra>",
                )
            )

    def _add_deviation_overlay(
        self, fig: go.Figure, route: Route, unit_scale: float, units: str
    ) -> None:
        """Add deviation analysis overlay."""
        # This would need deviation calculation data
        # For now, just add a placeholder for high-deviation areas
        pass

    def save_professional_plot(
        self,
        fig: go.Figure,
        output_path: Path,
        format: str = "png",
        width: Optional[int] = None,
        height: Optional[int] = None,
        scale: float = 2.0,
    ) -> None:
        """Save plot in professional quality.

        Args:
            fig: Plotly figure
            output_path: Output file path
            format: Output format (png, svg, html, pdf)
            width: Override width
            height: Override height
            scale: Scale factor for raster formats
        """
        if format.lower() == "html":
            # Interactive HTML with professional config
            config = {
                "displayModeBar": True,
                "displaylogo": False,
                "modeBarButtonsToRemove": [
                    "pan2d",
                    "lasso2d",
                    "autoScale2d",
                    "resetScale2d",
                    "hoverClosestCartesian",
                    "hoverCompareCartesian",
                    "toggleSpikelines",
                ],
                "toImageButtonOptions": {
                    "format": "png",
                    "filename": output_path.stem,
                    "height": height or 800,
                    "width": width or 1200,
                    "scale": scale,
                },
            }
            fig.write_html(str(output_path), config=config)

        elif format.lower() in ["png", "jpeg", "webp", "svg", "pdf"]:
            # Static high-quality export
            fig.write_image(
                str(output_path),
                format=format,
                width=width or 1200,
                height=height or 800,
                scale=scale,
                engine="kaleido",
            )
        else:
            raise ValueError(f"Unsupported format: {format}")

    def create_dashboard(
        self, route: Route, analysis_data: Dict[str, Any], title: Optional[str] = None
    ) -> go.Figure:
        """Create comprehensive analysis dashboard.

        Args:
            route: Route object
            analysis_data: Complete analysis results
            title: Dashboard title

        Returns:
            Multi-panel Plotly figure
        """
        if title is None:
            title = f"Cable Pulling Analysis Dashboard: {route.name}"

        # Create subplot layout
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=[
                "Route Overview",
                "Tension Analysis",
                "Pressure Analysis",
                "Section Summary",
            ],
            specs=[
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "bar"}],
            ],
            horizontal_spacing=0.08,
            vertical_spacing=0.12,
        )

        # Route overview (top-left)
        self._add_route_subplot(fig, route, row=1, col=1)

        # Tension analysis (top-right)
        if "tension" in analysis_data:
            self._add_tension_subplot(fig, analysis_data["tension"], row=1, col=2)

        # Pressure analysis (bottom-left)
        if "pressure" in analysis_data:
            self._add_pressure_subplot(fig, analysis_data["pressure"], row=2, col=1)

        # Section summary (bottom-right)
        self._add_section_summary_subplot(fig, route, analysis_data, row=2, col=2)

        # Apply professional layout
        fig.update_layout(
            title={
                "text": title,
                "x": 0.5,
                "xanchor": "center",
                "font": {"size": 18, "color": "#1a1a1a"},
            },
            height=1000,
            width=1600,
            showlegend=False,  # Individual subplots handle their own legends
            **{
                k: v
                for k, v in self.layout_config.items()
                if k not in ["title", "width", "height"]
            },
        )

        return fig

    def _add_route_subplot(
        self, fig: go.Figure, route: Route, row: int, col: int
    ) -> None:
        """Add route overview to subplot."""
        for i, section in enumerate(route.sections):
            if not section.original_polyline:
                continue

            x_coords = [p[0] for p in section.original_polyline]
            y_coords = [p[1] for p in section.original_polyline]
            color = self.colors["sections"][i % len(self.colors["sections"])]

            fig.add_trace(
                go.Scatter(
                    x=x_coords,
                    y=y_coords,
                    mode="lines",
                    name=f"Section {i+1}",
                    line=dict(color=color, width=2),
                    showlegend=False,
                ),
                row=row,
                col=col,
            )

    def _add_tension_subplot(
        self, fig: go.Figure, tension_data: Dict, row: int, col: int
    ) -> None:
        """Add tension analysis to subplot."""
        fig.add_trace(
            go.Scatter(
                x=tension_data["chainage"],
                y=tension_data["tension"],
                mode="lines",
                name="Tension",
                line=dict(color=self.colors["original"], width=2),
                showlegend=False,
            ),
            row=row,
            col=col,
        )

    def _add_pressure_subplot(
        self, fig: go.Figure, pressure_data: Dict, row: int, col: int
    ) -> None:
        """Add pressure analysis to subplot."""
        fig.add_trace(
            go.Scatter(
                x=pressure_data["chainage"],
                y=pressure_data["pressure"],
                mode="markers",
                name="Pressure",
                marker=dict(color=self.colors["bends"], size=6),
                showlegend=False,
            ),
            row=row,
            col=col,
        )

    def _add_section_summary_subplot(
        self, fig: go.Figure, route: Route, analysis_data: Dict, row: int, col: int
    ) -> None:
        """Add section summary bar chart."""
        section_names = [f"Sect {i+1}" for i in range(route.section_count)]
        section_lengths = [s.original_length for s in route.sections]

        fig.add_trace(
            go.Bar(
                x=section_names,
                y=section_lengths,
                name="Length",
                marker_color=self.colors["original"],
                showlegend=False,
            ),
            row=row,
            col=col,
        )


def create_professional_route_plot(
    route: Route,
    output_path: Optional[Path] = None,
    format: str = "png",
    units: str = "m",
    **kwargs,
) -> go.Figure:
    """Convenience function for professional route plotting.

    Args:
        route: Route object to visualize
        output_path: Optional output file path
        format: Output format (png, svg, html, pdf)
        units: Display units
        **kwargs: Additional arguments for plot_route_overview

    Returns:
        Plotly figure object
    """
    plotter = ProfessionalPlotter()
    fig = plotter.plot_route_overview(route, units=units, **kwargs)

    if output_path:
        plotter.save_professional_plot(fig, output_path, format=format)

    return fig


def create_analysis_dashboard(
    route: Route,
    analysis_data: Dict[str, Any],
    output_path: Optional[Path] = None,
    format: str = "html",
) -> go.Figure:
    """Create comprehensive analysis dashboard.

    Args:
        route: Route object
        analysis_data: Analysis results dictionary
        output_path: Optional output file path
        format: Output format

    Returns:
        Plotly dashboard figure
    """
    plotter = ProfessionalPlotter()
    fig = plotter.create_dashboard(route, analysis_data)

    if output_path:
        plotter.save_professional_plot(fig, output_path, format=format)

    return fig
