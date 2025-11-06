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

        # Find the absolute northernmost point across ALL sections
        northernmost_y = max(pt[1] for _, points in polylines for pt in points)
        northernmost_section_idx = None
        northernmost_at_start = False

        for idx, (layer, points) in enumerate(polylines):
            # Check if first point is northernmost
            if abs(points[0][1] - northernmost_y) < 0.01:
                northernmost_section_idx = idx
                northernmost_at_start = True
                break
            # Check if last point is northernmost
            if abs(points[-1][1] - northernmost_y) < 0.01:
                northernmost_section_idx = idx
                northernmost_at_start = False
                break

        if northernmost_section_idx is None:
            northernmost_section_idx = 0
            northernmost_at_start = True

        logger.info(f"Northernmost section: {northernmost_section_idx} (DXF index), "
                   f"northernmost point at {'START' if northernmost_at_start else 'END'}")

        # Build connectivity chain by following coordinate connections
        # Start from northernmost section and follow connections
        reordered_polylines = []
        used_indices = set()

        # Add first section (with northernmost point)
        current_idx = northernmost_section_idx
        current_layer, current_points = polylines[current_idx]
        reordered_polylines.append((current_layer, current_points))
        used_indices.add(current_idx)

        # Get the endpoint we should connect FROM (opposite end from northernmost point)
        if northernmost_at_start:
            # Northernmost is at start, so we connect from the END
            current_endpoint = current_points[-1]
        else:
            # Northernmost is at end, so we connect from the START
            current_endpoint = current_points[0]

        # Follow the chain by finding sections that connect
        while len(used_indices) < len(polylines):
            next_idx = None
            next_reversed = False

            # Find a section that connects to current_endpoint
            for idx, (layer, points) in enumerate(polylines):
                if idx in used_indices:
                    continue

                # Check if this section's START connects to our current endpoint
                dx = abs(points[0][0] - current_endpoint[0])
                dy = abs(points[0][1] - current_endpoint[1])
                if dx < 0.01 and dy < 0.01:
                    next_idx = idx
                    next_reversed = False
                    current_endpoint = points[-1]  # Next connection from END
                    break

                # Check if this section's END connects to our current endpoint
                dx = abs(points[-1][0] - current_endpoint[0])
                dy = abs(points[-1][1] - current_endpoint[1])
                if dx < 0.01 and dy < 0.01:
                    next_idx = idx
                    next_reversed = True
                    current_endpoint = points[0]  # Next connection from START
                    break

            if next_idx is None:
                # No connection found - append remaining sections
                logger.warning(f"Connection break after {len(used_indices)} sections")
                for idx, (layer, points) in enumerate(polylines):
                    if idx not in used_indices:
                        reordered_polylines.append((layer, points))
                        used_indices.add(idx)
                break

            # Add the connected section
            layer, points = polylines[next_idx]
            # If reversed, flip the points
            if next_reversed:
                points = points[::-1]
            reordered_polylines.append((layer, points))
            used_indices.add(next_idx)

        # Create Route and sections with proper junction labels
        route = Route(name=route_name)
        current_junction = ord('A')

        for i, (layer, points) in enumerate(reordered_polylines):
            section_id = f"{section_prefix}_{i+1:02d}"

            # Assign junctions: A→B, B→C, C→D, etc.
            start_junc = chr(current_junction)
            end_junc = chr(current_junction + 1)

            section = Section(
                id=section_id,
                original_polyline=points,
                start_junction=start_junc,
                end_junction=end_junc
            )

            current_junction += 1

            route.add_section(section)
            logger.info(f"Created section {section_id} ({start_junc}→{end_junc}) with {len(points)} points")

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
