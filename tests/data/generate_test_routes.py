"""Generate synthetic test routes for comprehensive testing."""

import math
from pathlib import Path
from typing import List, Tuple

import ezdxf
import numpy as np


class SyntheticRouteGenerator:
    """Generate synthetic DXF files for testing different route scenarios."""

    def __init__(self, output_dir: Path):
        """Initialize generator.

        Args:
            output_dir: Directory to save generated DXF files
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)

    def generate_straight_route(self, length: float = 1000.0) -> Path:
        """Generate a simple straight route.

        Args:
            length: Route length in mm

        Returns:
            Path to generated DXF file
        """
        doc = ezdxf.new("R2010")
        msp = doc.modelspace()

        # Simple straight line
        points = [(0.0, 0.0), (length, 0.0)]
        msp.add_lwpolyline(points, close=False)

        output_path = self.output_dir / "straight_route.dxf"
        doc.saveas(output_path)
        return output_path

    def generate_simple_s_curve(self) -> Path:
        """Generate an S-curve with two 90-degree bends.

        Returns:
            Path to generated DXF file
        """
        doc = ezdxf.new("R2010")
        msp = doc.modelspace()

        points = [
            (0.0, 0.0),
            (500.0, 0.0),  # Straight section
            (1000.0, 500.0),  # First bend
            (1500.0, 500.0),  # Straight section
            (2000.0, 0.0),  # Second bend (S-curve)
            (2500.0, 0.0),  # Final straight
        ]
        msp.add_lwpolyline(points, close=False)

        output_path = self.output_dir / "s_curve_route.dxf"
        doc.saveas(output_path)
        return output_path

    def generate_circular_arc(
        self, radius: float = 1000.0, angle_deg: float = 90.0
    ) -> Path:
        """Generate a circular arc route.

        Args:
            radius: Arc radius in mm
            angle_deg: Arc angle in degrees

        Returns:
            Path to generated DXF file
        """
        doc = ezdxf.new("R2010")
        msp = doc.modelspace()

        # Generate points along circular arc
        angle_rad = math.radians(angle_deg)
        num_points = max(
            10, int(angle_deg / 5)
        )  # At least 10 points, more for larger angles

        points = []
        for i in range(num_points + 1):
            t = i * angle_rad / num_points
            x = radius * math.sin(t)
            y = radius * (1 - math.cos(t))
            points.append((x, y))

        msp.add_lwpolyline(points, close=False)

        output_path = self.output_dir / f"circular_arc_{angle_deg}deg.dxf"
        doc.saveas(output_path)
        return output_path

    def generate_complex_route(self) -> Path:
        """Generate a complex route with multiple bends and varying radii.

        Returns:
            Path to generated DXF file
        """
        doc = ezdxf.new("R2010")
        msp = doc.modelspace()

        # Complex route with multiple sections
        points = []

        # Section 1: Straight start
        for i in range(11):
            points.append((i * 50.0, 0.0))

        # Section 2: Large radius curve (natural sweep)
        center_x, center_y = 500.0, 2000.0  # Large radius natural curve
        for i in range(20):
            angle = i * math.pi / 40  # 90 degrees over 20 points
            x = center_x + 2000.0 * math.sin(angle)
            y = center_y - 2000.0 * math.cos(angle)
            points.append((x, y))

        # Section 3: Straight middle section
        start_x, start_y = points[-1]
        for i in range(1, 21):
            points.append((start_x + i * 30.0, start_y))

        # Section 4: Sharp manufactured bend
        start_x, start_y = points[-1]
        bend_radius = 600.0  # Standard manufactured bend
        for i in range(10):
            angle = i * math.pi / 18  # 90 degrees over 10 points
            x = start_x + bend_radius * math.sin(angle)
            y = start_y + bend_radius * (1 - math.cos(angle))
            points.append((x, y))

        # Section 5: Final straight
        start_x, start_y = points[-1]
        for i in range(1, 16):
            points.append((start_x, start_y + i * 40.0))

        msp.add_lwpolyline(points, close=False)

        output_path = self.output_dir / "complex_route.dxf"
        doc.saveas(output_path)
        return output_path

    def generate_long_route(self, total_length: float = 10000.0) -> Path:
        """Generate a very long route for performance testing.

        Args:
            total_length: Total route length in mm

        Returns:
            Path to generated DXF file
        """
        doc = ezdxf.new("R2010")
        msp = doc.modelspace()

        points = []
        current_x, current_y = 0.0, 0.0
        segment_length = 100.0  # mm per segment
        num_segments = int(total_length / segment_length)

        # Create a meandering path
        for i in range(num_segments + 1):
            # Add some gentle curves
            offset_y = 50.0 * math.sin(i * 0.1)
            points.append((current_x, current_y + offset_y))
            current_x += segment_length

        msp.add_lwpolyline(points, close=False)

        output_path = self.output_dir / "long_route.dxf"
        doc.saveas(output_path)
        return output_path

    def generate_multiple_sections_route(self) -> Path:
        """Generate a route with multiple disconnected sections.

        Returns:
            Path to generated DXF file
        """
        doc = ezdxf.new("R2010")
        msp = doc.modelspace()

        # Section 1: Straight horizontal
        section1 = [(0.0, 0.0), (500.0, 0.0)]
        msp.add_lwpolyline(section1, close=False)

        # Section 2: L-shaped (disconnected)
        section2 = [(600.0, 100.0), (1100.0, 100.0), (1100.0, 600.0)]
        msp.add_lwpolyline(section2, close=False)

        # Section 3: Curved section (disconnected)
        section3 = []
        for i in range(21):
            angle = i * math.pi / 20  # 180 degrees
            x = 1200.0 + 300.0 * math.cos(angle)
            y = 700.0 + 300.0 * math.sin(angle)
            section3.append((x, y))
        msp.add_lwpolyline(section3, close=False)

        output_path = self.output_dir / "multiple_sections_route.dxf"
        doc.saveas(output_path)
        return output_path

    def generate_edge_case_routes(self) -> List[Path]:
        """Generate various edge case routes for robustness testing.

        Returns:
            List of paths to generated DXF files
        """
        generated_files = []

        # Very short route
        doc = ezdxf.new("R2010")
        msp = doc.modelspace()
        msp.add_lwpolyline([(0.0, 0.0), (10.0, 0.0)], close=False)
        path = self.output_dir / "very_short_route.dxf"
        doc.saveas(path)
        generated_files.append(path)

        # Route with very tight bend (below minimum radius)
        doc = ezdxf.new("R2010")
        msp = doc.modelspace()
        points = []
        # Sharp 90-degree turn with tiny radius
        for i in range(11):
            if i <= 5:
                points.append((i * 100.0, 0.0))
            else:
                angle = (i - 5) * math.pi / 10
                x = 500.0 + 50.0 * math.sin(angle)  # Very small 50mm radius
                y = 50.0 * (1 - math.cos(angle))
                points.append((x, y))
        msp.add_lwpolyline(points, close=False)
        path = self.output_dir / "tight_bend_route.dxf"
        doc.saveas(path)
        generated_files.append(path)

        # Route with many consecutive bends
        doc = ezdxf.new("R2010")
        msp = doc.modelspace()
        points = [(0.0, 0.0)]
        current_x, current_y = 0.0, 0.0
        current_angle = 0.0

        for bend_num in range(8):  # 8 consecutive bends
            # Straight section
            length = 200.0
            current_x += length * math.cos(current_angle)
            current_y += length * math.sin(current_angle)
            points.append((current_x, current_y))

            # Bend section
            bend_angle = math.pi / 4  # 45 degree bend
            radius = 800.0
            center_x = current_x - radius * math.sin(current_angle)
            center_y = current_y + radius * math.cos(current_angle)

            for i in range(1, 10):
                t = i / 9.0
                angle = current_angle + t * bend_angle
                x = center_x + radius * math.sin(angle)
                y = center_y - radius * math.cos(angle)
                points.append((x, y))

            current_angle += bend_angle
            current_x = points[-1][0]
            current_y = points[-1][1]

        msp.add_lwpolyline(points, close=False)
        path = self.output_dir / "many_bends_route.dxf"
        doc.saveas(path)
        generated_files.append(path)

        return generated_files

    def generate_all_test_routes(self) -> List[Path]:
        """Generate all synthetic test routes.

        Returns:
            List of paths to all generated DXF files
        """
        generated_files = []

        generated_files.append(self.generate_straight_route())
        generated_files.append(self.generate_simple_s_curve())
        generated_files.append(self.generate_circular_arc(1000.0, 90.0))
        generated_files.append(self.generate_circular_arc(600.0, 45.0))
        generated_files.append(self.generate_complex_route())
        generated_files.append(self.generate_long_route())
        generated_files.append(self.generate_multiple_sections_route())
        generated_files.extend(self.generate_edge_case_routes())

        return generated_files


if __name__ == "__main__":
    # Generate all test routes
    data_dir = Path(__file__).parent
    generator = SyntheticRouteGenerator(data_dir)
    files = generator.generate_all_test_routes()

    print(f"Generated {len(files)} test DXF files:")
    for file in files:
        print(f"  {file.name}")
