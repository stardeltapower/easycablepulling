"""DXF file reading functionality."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import ezdxf

from ..core.models import Route, Section

logger = logging.getLogger(__name__)


class DXFReader:
    """DXF file reader for cable route geometry."""

    def __init__(self, file_path: Path) -> None:
        """Initialize DXF reader.

        Args:
            file_path: Path to DXF file
        """
        self.file_path = file_path
        self.doc: Optional[ezdxf.document.Drawing] = None

    def load(self) -> None:
        """Load DXF file."""
        try:
            self.doc = ezdxf.readfile(str(self.file_path))
            logger.info(f"Loaded DXF file: {self.file_path}")
        except Exception as e:
            raise ValueError(f"Failed to load DXF file {self.file_path}: {e}")

    def extract_polylines(
        self, layer_name: Optional[str] = None
    ) -> List[Tuple[str, List[Tuple[float, float]]]]:
        """Extract polylines from DXF file.

        Args:
            layer_name: Specific layer to extract from (None for all layers)

        Returns:
            List of (layer_name, points) tuples
        """
        if not self.doc:
            raise ValueError("DXF file not loaded. Call load() first.")

        polylines = []
        msp = self.doc.modelspace()

        # Extract LWPolylines (lightweight polylines)
        for entity in msp.query("LWPOLYLINE"):
            if layer_name and entity.dxf.layer != layer_name:
                continue

            points = []
            for point in entity.get_points():
                # ezdxf returns (x, y, start_width, end_width, bulge)
                # We only need x, y coordinates
                points.append((float(point[0]), float(point[1])))

            if len(points) >= 2:
                polylines.append((entity.dxf.layer, points))

        # Extract regular Polylines
        for entity in msp.query("POLYLINE"):
            if layer_name and entity.dxf.layer != layer_name:
                continue

            points = []
            for vertex in entity.vertices:
                points.append(
                    (float(vertex.dxf.location.x), float(vertex.dxf.location.y))
                )

            if len(points) >= 2:
                polylines.append((entity.dxf.layer, points))

        logger.info(f"Extracted {len(polylines)} polylines from DXF")
        return polylines

    def get_layers(self) -> List[str]:
        """Get list of layer names in the DXF file.

        Returns:
            List of layer names
        """
        if not self.doc:
            raise ValueError("DXF file not loaded. Call load() first.")

        layers = []
        for layer in self.doc.layers:
            layers.append(layer.dxf.name)

        return layers

    def get_polyline_count(self, layer_name: Optional[str] = None) -> int:
        """Get count of polylines in specified layer or all layers.

        Args:
            layer_name: Layer to count (None for all layers)

        Returns:
            Number of polylines
        """
        if not self.doc:
            raise ValueError("DXF file not loaded. Call load() first.")

        msp = self.doc.modelspace()
        count = 0

        # Count LWPolylines
        for entity in msp.query("LWPOLYLINE"):
            if layer_name is None or entity.dxf.layer == layer_name:
                count += 1

        # Count regular Polylines
        for entity in msp.query("POLYLINE"):
            if layer_name is None or entity.dxf.layer == layer_name:
                count += 1

        return count

    def create_route_from_polylines(
        self,
        route_name: str,
        layer_name: Optional[str] = None,
        section_prefix: str = "SECT",
    ) -> Route:
        """Create a Route object from polylines in the DXF file.

        Args:
            route_name: Name for the route
            layer_name: Specific layer to use (None for first available)
            section_prefix: Prefix for section IDs

        Returns:
            Route object with sections
        """
        polylines = self.extract_polylines(layer_name)

        if not polylines:
            raise ValueError("No polylines found in DXF file")

        route = Route(name=route_name)

        # Create sections from polylines
        for i, (layer, points) in enumerate(polylines):
            section_id = f"{section_prefix}_{i+1:02d}"

            section = Section(id=section_id, original_polyline=points)

            route.add_section(section)
            logger.info(f"Created section {section_id} with {len(points)} points")

        # Add metadata
        route.metadata.update(
            {
                "source_file": str(self.file_path),
                "source_layers": [layer for layer, _ in polylines],
                "polyline_count": len(polylines),
            }
        )

        return route

    def get_route_summary(self) -> Dict[str, Any]:
        """Get summary information about the DXF file.

        Returns:
            Dictionary with file summary information
        """
        if not self.doc:
            raise ValueError("DXF file not loaded. Call load() first.")

        layers = self.get_layers()
        total_polylines = self.get_polyline_count()

        summary = {
            "file_path": str(self.file_path),
            "dxf_version": self.doc.dxfversion,
            "layers": layers,
            "total_polylines": total_polylines,
            "polylines_per_layer": {},
        }

        # Count polylines per layer
        for layer in layers:
            count = self.get_polyline_count(layer)
            summary["polylines_per_layer"][layer] = count

        return summary


def load_route_from_dxf(file_path: Path, route_name: Optional[str] = None) -> Route:
    """Convenience function to load a route from a DXF file.

    Args:
        file_path: Path to DXF file
        route_name: Name for the route (defaults to filename)

    Returns:
        Route object
    """
    if route_name is None:
        route_name = file_path.stem

    reader = DXFReader(file_path)
    reader.load()

    return reader.create_route_from_polylines(route_name)
