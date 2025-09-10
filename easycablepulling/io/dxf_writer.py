"""DXF file writing functionality."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import ezdxf
from ezdxf.layouts import Modelspace

from ..core.models import Bend, Route, Straight

logger = logging.getLogger(__name__)


class DXFWriter:
    """DXF file writer for cable route geometry and analysis results."""

    def __init__(self, dxf_version: str = "R2010") -> None:
        """Initialize DXF writer.

        Args:
            dxf_version: DXF version to use (default R2010 for compatibility)
        """
        self.doc = ezdxf.new(dxf_version)
        self.msp: Modelspace = self.doc.modelspace()
        self._setup_layers()

    def _setup_layers(self) -> None:
        """Setup standard layers for cable pulling output."""
        layers_config = {
            "ROUTE_ORIGINAL": {"color": 7, "linetype": "CONTINUOUS"},  # White
            "ROUTE_FITTED": {"color": 2, "linetype": "CONTINUOUS"},  # Yellow
            "PRIMITIVES_STRAIGHT": {"color": 3, "linetype": "CONTINUOUS"},  # Green
            "PRIMITIVES_BEND": {"color": 1, "linetype": "DASHED"},  # Red
            "ANNOTATIONS": {"color": 4, "linetype": "CONTINUOUS"},  # Cyan
            "JOINTS": {"color": 5, "linetype": "CONTINUOUS"},  # Blue
            "WARNINGS": {"color": 1, "linetype": "CONTINUOUS"},  # Red
        }

        for layer_name, properties in layers_config.items():
            layer = self.doc.layers.add(layer_name)
            layer.color = properties["color"]
            layer.linetype = properties["linetype"]

        logger.info(f"Created {len(layers_config)} standard layers")

    def write_original_route(
        self, route: Route, layer_name: str = "ROUTE_ORIGINAL"
    ) -> None:
        """Write original polylines from route sections.

        Args:
            route: Route object to write
            layer_name: Layer name for original geometry
        """
        for section in route.sections:
            if len(section.original_polyline) >= 2:
                # Create lightweight polyline
                points_2d = [(x, y) for x, y in section.original_polyline]
                lwpolyline = self.msp.add_lwpolyline(points_2d)
                lwpolyline.dxf.layer = layer_name

        logger.info(
            f"Wrote {len(route.sections)} original polylines to layer {layer_name}"
        )

    def write_fitted_route(
        self, route: Route, layer_name: str = "ROUTE_FITTED"
    ) -> None:
        """Write fitted primitives (straights and bends) from route sections.

        Args:
            route: Route object with fitted primitives
            layer_name: Layer name for fitted geometry
        """
        total_primitives = 0

        for section in route.sections:
            for primitive in section.primitives:
                if isinstance(primitive, Straight):
                    # Draw straight line
                    line = self.msp.add_line(primitive.start_point, primitive.end_point)
                    line.dxf.layer = "PRIMITIVES_STRAIGHT"
                    total_primitives += 1

                elif isinstance(primitive, Bend):
                    # Draw arc
                    arc = self.msp.add_arc(
                        center=primitive.center_point,
                        radius=primitive.radius_m,
                        start_angle=0,  # Will need proper start/end angles
                        end_angle=primitive.angle_deg,
                    )
                    arc.dxf.layer = "PRIMITIVES_BEND"
                    total_primitives += 1

        logger.info(f"Wrote {total_primitives} fitted primitives")

    def write_section_annotations(
        self, route: Route, layer_name: str = "ANNOTATIONS", text_height: float = 5.0
    ) -> None:
        """Write section labels and annotations.

        Args:
            route: Route object to annotate
            layer_name: Layer name for annotations
            text_height: Text height in drawing units
        """
        for section in route.sections:
            if section.original_polyline:
                # Place label at midpoint of section
                start_point = section.original_polyline[0]
                end_point = section.original_polyline[-1]

                midpoint = (
                    (start_point[0] + end_point[0]) / 2,
                    (start_point[1] + end_point[1]) / 2,
                )

                # Add section ID text
                text = self.msp.add_text(
                    section.id, height=text_height, dxfattribs={"layer": layer_name}
                )
                text.set_placement(midpoint)

        logger.info(f"Added annotations for {len(route.sections)} sections")

    def write_joint_markers(
        self, route: Route, layer_name: str = "JOINTS", marker_size: float = 10.0
    ) -> None:
        """Write joint/pit markers at section boundaries.

        Args:
            route: Route object
            layer_name: Layer name for joint markers
            marker_size: Size of joint markers
        """
        joint_points = set()

        # Collect all section endpoints
        for section in route.sections:
            if section.original_polyline:
                start_point = section.original_polyline[0]
                end_point = section.original_polyline[-1]
                joint_points.add(start_point)
                joint_points.add(end_point)

        # Draw circle markers at each joint
        for x, y in joint_points:
            self.msp.add_circle(
                center=(x, y), radius=marker_size, dxfattribs={"layer": layer_name}
            )

        logger.info(f"Added {len(joint_points)} joint markers")

    def write_analysis_results(
        self,
        route: Route,
        analysis_results: Optional[Dict[str, Any]] = None,
        layer_name: str = "ANNOTATIONS",
    ) -> None:
        """Write analysis results as text annotations.

        Args:
            route: Route object
            analysis_results: Dictionary with analysis results
            layer_name: Layer name for result annotations
        """
        if not analysis_results:
            return

        # Find a good location for results (offset from route)
        if route.sections and route.sections[0].original_polyline:
            first_point = route.sections[0].original_polyline[0]
            text_location = (first_point[0] + 100, first_point[1] + 100)
        else:
            text_location = (0, 0)

        # Write summary text
        summary_lines = [
            "Cable Pulling Analysis Results",
            f"Route: {route.name}",
            f"Total Length: {route.total_length:.2f}m",
            f"Sections: {route.section_count}",
        ]

        if analysis_results:
            if "max_tension" in analysis_results:
                summary_lines.append(
                    f"Max Tension: {analysis_results['max_tension']:.0f}N"
                )
            if "max_sidewall_pressure" in analysis_results:
                summary_lines.append(
                    f"Max Sidewall Pressure: "
                    f"{analysis_results['max_sidewall_pressure']:.0f}N/m"
                )

        # Add each line as separate text entity
        for i, line in enumerate(summary_lines):
            text = self.msp.add_text(line, height=8.0, dxfattribs={"layer": layer_name})
            text.set_placement((text_location[0], text_location[1] - i * 15))

        logger.info("Added analysis results annotation")

    def write_warnings(
        self,
        warnings: List[str],
        location: Tuple[float, float],
        layer_name: str = "WARNINGS",
    ) -> None:
        """Write validation warnings as text.

        Args:
            warnings: List of warning messages
            location: Location to place warning text
            layer_name: Layer name for warnings
        """
        if not warnings:
            return

        # Add warning header
        header = self.msp.add_text(
            "WARNINGS:", height=10.0, dxfattribs={"layer": layer_name}
        )
        header.set_placement(location)

        # Add each warning
        for i, warning in enumerate(warnings):
            text = self.msp.add_text(
                f"• {warning}", height=8.0, dxfattribs={"layer": layer_name}
            )
            text.set_placement((location[0], location[1] - (i + 1) * 12))

        logger.info(f"Added {len(warnings)} warnings")

    def save(self, file_path: Path) -> None:
        """Save DXF file to disk.

        Args:
            file_path: Output file path
        """
        try:
            self.doc.saveas(str(file_path))
            logger.info(f"Saved DXF file: {file_path}")
        except Exception as e:
            raise ValueError(f"Failed to save DXF file {file_path}: {e}")


def export_route_to_dxf(
    route: Route,
    file_path: Path,
    include_annotations: bool = True,
    include_joint_markers: bool = True,
    analysis_results: Optional[Dict[str, Any]] = None,
    warnings: Optional[List[str]] = None,
) -> None:
    """Convenience function to export a route to DXF file.

    Args:
        route: Route object to export
        file_path: Output file path
        include_annotations: Whether to include section labels
        include_joint_markers: Whether to include joint markers
        analysis_results: Optional analysis results to include
        warnings: Optional warnings to include
    """
    writer = DXFWriter()

    # Write original geometry
    writer.write_original_route(route)

    # Write fitted geometry if available
    has_primitives = any(section.primitives for section in route.sections)
    if has_primitives:
        writer.write_fitted_route(route)

    # Write annotations
    if include_annotations:
        writer.write_section_annotations(route)

    # Write joint markers
    if include_joint_markers:
        writer.write_joint_markers(route)

    # Write analysis results
    if analysis_results:
        writer.write_analysis_results(route, analysis_results)

    # Write warnings
    if warnings and route.sections:
        # Place warnings near the first section
        first_point = route.sections[0].original_polyline[0]
        warning_location = (first_point[0] - 200, first_point[1] + 200)
        writer.write_warnings(warnings, warning_location)

    # Save file
    writer.save(file_path)
