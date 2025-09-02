"""Route and cable pulling statistics calculations."""

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..core.models import Bend, Route, Section, Straight


@dataclass
class SectionStatistics:
    """Statistics for a single route section."""

    section_id: str
    length: float  # meters
    point_count: int
    start_point: Tuple[float, float]
    end_point: Tuple[float, float]

    # Geometry statistics
    total_angle_change: float  # degrees
    max_angle_change: float  # degrees
    straight_count: int
    bend_count: int

    # Distance statistics
    straight_distance: float  # Direct distance from start to end
    sinuosity: float  # Length / straight_distance (measure of curvature)

    # Bend statistics
    min_bend_radius: Optional[float] = None
    max_bend_radius: Optional[float] = None
    avg_bend_radius: Optional[float] = None

    # Elevation statistics (future)
    elevation_change: Optional[float] = None
    max_slope: Optional[float] = None


@dataclass
class RouteStatistics:
    """Complete route statistics."""

    route_name: str
    total_length: float  # meters
    section_count: int
    total_joints: int

    # Section statistics
    sections: List[SectionStatistics]

    # Aggregate statistics
    min_section_length: float
    max_section_length: float
    avg_section_length: float
    std_section_length: float

    # Geometry statistics
    total_straight_count: int
    total_bend_count: int
    total_angle_change: float

    # Route extent
    min_x: float
    max_x: float
    min_y: float
    max_y: float
    route_width: float
    route_height: float

    # Bend statistics (optional)
    min_bend_radius: Optional[float] = None
    max_bend_radius: Optional[float] = None
    avg_bend_radius: Optional[float] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON export."""
        return {
            "route_name": self.route_name,
            "total_length_m": round(self.total_length, 2),
            "section_count": self.section_count,
            "total_joints": self.total_joints,
            "sections": [
                {
                    "id": s.section_id,
                    "length_m": round(s.length, 2),
                    "point_count": s.point_count,
                    "sinuosity": round(s.sinuosity, 3),
                    "angle_change_deg": round(s.total_angle_change, 1),
                    "straight_count": s.straight_count,
                    "bend_count": s.bend_count,
                }
                for s in self.sections
            ],
            "aggregate_statistics": {
                "min_section_length_m": round(self.min_section_length, 2),
                "max_section_length_m": round(self.max_section_length, 2),
                "avg_section_length_m": round(self.avg_section_length, 2),
                "std_section_length_m": round(self.std_section_length, 2),
                "total_straight_count": self.total_straight_count,
                "total_bend_count": self.total_bend_count,
                "total_angle_change_deg": round(self.total_angle_change, 1),
            },
            "route_extent": {
                "min_x": round(self.min_x, 2),
                "max_x": round(self.max_x, 2),
                "min_y": round(self.min_y, 2),
                "max_y": round(self.max_y, 2),
                "width_m": round(self.route_width, 2),
                "height_m": round(self.route_height, 2),
            },
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def save_json(self, file_path: Path) -> None:
        """Save statistics to JSON file."""
        with open(file_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def save_csv(self, file_path: Path) -> None:
        """Save section statistics to CSV file."""
        with open(file_path, "w", newline="") as f:
            writer = csv.writer(f)

            # Write header
            writer.writerow(
                [
                    "Section ID",
                    "Length (m)",
                    "Points",
                    "Start X",
                    "Start Y",
                    "End X",
                    "End Y",
                    "Sinuosity",
                    "Total Angle Change (deg)",
                    "Straight Count",
                    "Bend Count",
                ]
            )

            # Write section data
            for s in self.sections:
                writer.writerow(
                    [
                        s.section_id,
                        round(s.length, 2),
                        s.point_count,
                        round(s.start_point[0], 2),
                        round(s.start_point[1], 2),
                        round(s.end_point[0], 2),
                        round(s.end_point[1], 2),
                        round(s.sinuosity, 3),
                        round(s.total_angle_change, 1),
                        s.straight_count,
                        s.bend_count,
                    ]
                )

    def generate_report(self) -> str:
        """Generate a text report of statistics."""
        lines = []
        lines.append("=" * 60)
        lines.append("CABLE ROUTE STATISTICS REPORT")
        lines.append(f"Route: {self.route_name}")
        lines.append("=" * 60)
        lines.append("")

        # Overview
        lines.append("OVERVIEW")
        lines.append("-" * 30)
        lines.append(f"Total Length: {self.total_length:,.1f} m")
        lines.append(f"Total Sections: {self.section_count}")
        lines.append(f"Total Joints/Pits: {self.total_joints}")
        lines.append("")

        # Route extent
        lines.append("ROUTE EXTENT")
        lines.append("-" * 30)
        lines.append(f"X Range: {self.min_x:,.1f} to {self.max_x:,.1f} m")
        lines.append(f"Y Range: {self.min_y:,.1f} to {self.max_y:,.1f} m")
        lines.append(f"Width: {self.route_width:,.1f} m")
        lines.append(f"Height: {self.route_height:,.1f} m")
        lines.append("")

        # Section statistics
        lines.append("SECTION STATISTICS")
        lines.append("-" * 30)
        lines.append(f"Minimum Length: {self.min_section_length:,.1f} m")
        lines.append(f"Maximum Length: {self.max_section_length:,.1f} m")
        lines.append(f"Average Length: {self.avg_section_length:,.1f} m")
        lines.append(f"Std Deviation: {self.std_section_length:,.1f} m")
        lines.append("")

        # Geometry statistics
        lines.append("GEOMETRY STATISTICS")
        lines.append("-" * 30)
        lines.append(f"Total Straight Segments: {self.total_straight_count}")
        lines.append(f"Total Bend Segments: {self.total_bend_count}")
        lines.append(f"Total Angle Change: {self.total_angle_change:,.1f}°")

        if self.min_bend_radius is not None:
            lines.append(f"Minimum Bend Radius: {self.min_bend_radius:,.1f} m")
            lines.append(f"Maximum Bend Radius: {self.max_bend_radius:,.1f} m")
            lines.append(f"Average Bend Radius: {self.avg_bend_radius:,.1f} m")
        lines.append("")

        # Section details
        lines.append("SECTION DETAILS")
        lines.append("-" * 30)
        lines.append(
            f"{'ID':<10} {'Length(m)':<12} {'Points':<8} {'Sinuosity':<10} {'Angle(°)':<10}"
        )
        lines.append("-" * 50)

        for s in self.sections:
            lines.append(
                f"{s.section_id:<10} {s.length:<12.1f} {s.point_count:<8} "
                f"{s.sinuosity:<10.3f} {s.total_angle_change:<10.1f}"
            )

        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)


def calculate_angle_change(
    p1: Tuple[float, float], p2: Tuple[float, float], p3: Tuple[float, float]
) -> float:
    """Calculate angle change between three points in degrees."""
    # Vector from p1 to p2
    v1 = (p2[0] - p1[0], p2[1] - p1[1])
    # Vector from p2 to p3
    v2 = (p3[0] - p2[0], p3[1] - p2[1])

    # Calculate angle between vectors
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    cross = v1[0] * v2[1] - v1[1] * v2[0]

    angle = math.atan2(cross, dot)
    return math.degrees(angle)


def calculate_section_statistics(section: Section) -> SectionStatistics:
    """Calculate statistics for a single section."""
    points = section.original_polyline

    if len(points) < 2:
        raise ValueError(f"Section {section.id} has insufficient points")

    # Basic statistics
    start_point = points[0]
    end_point = points[-1]

    # Calculate straight-line distance
    dx = end_point[0] - start_point[0]
    dy = end_point[1] - start_point[1]
    straight_distance = math.sqrt(dx * dx + dy * dy)

    # Calculate sinuosity
    sinuosity = (
        section.original_length / straight_distance if straight_distance > 0 else 1.0
    )

    # Calculate angle changes
    angle_changes = []
    if len(points) >= 3:
        for i in range(1, len(points) - 1):
            angle = calculate_angle_change(points[i - 1], points[i], points[i + 1])
            angle_changes.append(abs(angle))

    total_angle_change = sum(angle_changes) if angle_changes else 0.0
    max_angle_change = max(angle_changes) if angle_changes else 0.0

    # Count primitives (if fitted)
    straight_count = sum(1 for p in section.primitives if isinstance(p, Straight))
    bend_count = sum(1 for p in section.primitives if isinstance(p, Bend))

    # Bend radius statistics
    bend_radii = [p.radius_m for p in section.primitives if isinstance(p, Bend)]

    stats = SectionStatistics(
        section_id=section.id,
        length=section.original_length,
        point_count=len(points),
        start_point=start_point,
        end_point=end_point,
        total_angle_change=total_angle_change,
        max_angle_change=max_angle_change,
        straight_count=straight_count,
        bend_count=bend_count,
        straight_distance=straight_distance,
        sinuosity=sinuosity,
    )

    if bend_radii:
        stats.min_bend_radius = min(bend_radii)
        stats.max_bend_radius = max(bend_radii)
        stats.avg_bend_radius = sum(bend_radii) / len(bend_radii)

    return stats


def calculate_route_statistics(route: Route) -> RouteStatistics:
    """Calculate comprehensive statistics for a route."""
    # Calculate section statistics
    section_stats = []
    for section in route.sections:
        section_stats.append(calculate_section_statistics(section))

    # Section lengths
    lengths = [s.length for s in section_stats]

    # Route extent
    all_points = []
    for section in route.sections:
        all_points.extend(section.original_polyline)

    x_coords = [p[0] for p in all_points]
    y_coords = [p[1] for p in all_points]

    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)

    # Aggregate bend statistics
    all_bend_radii = []
    for stats in section_stats:
        if stats.min_bend_radius is not None:
            all_bend_radii.extend(
                [stats.min_bend_radius, stats.max_bend_radius, stats.avg_bend_radius]
            )

    route_stats = RouteStatistics(
        route_name=route.name,
        total_length=sum(lengths),
        section_count=route.section_count,
        total_joints=route.section_count + 1,  # Joints at start/end of each section
        sections=section_stats,
        min_section_length=min(lengths),
        max_section_length=max(lengths),
        avg_section_length=np.mean(lengths),
        std_section_length=np.std(lengths),
        total_straight_count=sum(s.straight_count for s in section_stats),
        total_bend_count=sum(s.bend_count for s in section_stats),
        total_angle_change=sum(s.total_angle_change for s in section_stats),
        min_x=min_x,
        max_x=max_x,
        min_y=min_y,
        max_y=max_y,
        route_width=max_x - min_x,
        route_height=max_y - min_y,
    )

    if all_bend_radii:
        route_stats.min_bend_radius = min(all_bend_radii)
        route_stats.max_bend_radius = max(all_bend_radii)
        route_stats.avg_bend_radius = np.mean(all_bend_radii)

    return route_stats
