"""CSV report generation for cable pulling analysis."""

import csv
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from ..core.models import Bend, Route, Straight


class CSVReporter:
    """Generate professional CSV reports for cable pulling analysis."""

    def __init__(self) -> None:
        """Initialize CSV reporter."""
        self.headers = {
            "route_summary": [
                "Route Name",
                "Total Length (m)",
                "Section Count",
                "Analysis Date",
                "Max Tension (N)",
                "Max Pressure (N/m)",
                "Status",
                "Limiting Factor",
            ],
            "section_details": [
                "Section ID",
                "Length (m)",
                "Start Point X (m)",
                "Start Point Y (m)",
                "End Point X (m)",
                "End Point Y (m)",
                "Straight Count",
                "Bend Count",
                "Max Tension (N)",
                "Max Pressure (N/m)",
                "Lateral Deviation (m)",
                "Length Error (%)",
                "Status",
            ],
            "primitive_details": [
                "Section ID",
                "Primitive Type",
                "Length/Radius (m)",
                "Angle (deg)",
                "Start X (m)",
                "Start Y (m)",
                "End X (m)",
                "End Y (m)",
                "Center X (m)",
                "Center Y (m)",
            ],
            "calculation_results": [
                "Section ID",
                "Chainage (m)",
                "Cable Tension (N)",
                "Sidewall Pressure (N/m)",
                "Bend Radius (m)",
                "Friction Coefficient",
                "Pull Direction",
                "Status",
                "Notes",
            ],
        }

    def generate_route_summary(
        self, route: Route, analysis_results: Dict[str, Any], output_path: Path
    ) -> None:
        """Generate route summary CSV report.

        Args:
            route: Route object
            analysis_results: Analysis calculation results
            output_path: Output CSV file path
        """
        with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)

            # Write header
            writer.writerow(self.headers["route_summary"])

            # Calculate summary metrics
            total_length = sum(s.original_length for s in route.sections)
            max_tension = analysis_results.get("max_tension", 0)
            max_pressure = analysis_results.get("max_pressure", 0)
            status = analysis_results.get("overall_status", "Unknown")
            limiting_factor = analysis_results.get("limiting_factor", "N/A")

            # Write data row
            writer.writerow(
                [
                    route.name,
                    f"{total_length:.2f}",
                    route.section_count,
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    f"{max_tension:.0f}",
                    f"{max_pressure:.2f}",
                    status,
                    limiting_factor,
                ]
            )

    def generate_section_details(
        self, route: Route, analysis_results: Dict[str, Any], output_path: Path
    ) -> None:
        """Generate detailed section analysis CSV report.

        Args:
            route: Route object
            analysis_results: Analysis calculation results
            output_path: Output CSV file path
        """
        with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)

            # Write header
            writer.writerow(self.headers["section_details"])

            # Get section results
            section_results = analysis_results.get("sections", {})

            for section in route.sections:
                if not section.original_polyline:
                    continue

                section_analysis = section_results.get(section.id, {})

                # Count primitives
                straight_count = sum(
                    1 for p in section.primitives if hasattr(p, "length_m")
                )
                bend_count = sum(
                    1 for p in section.primitives if hasattr(p, "radius_m")
                )

                # Get start/end points
                start_point = section.original_polyline[0]
                end_point = section.original_polyline[-1]

                writer.writerow(
                    [
                        section.id,
                        f"{section.original_length:.2f}",
                        f"{start_point[0]:.3f}",
                        f"{start_point[1]:.3f}",
                        f"{end_point[0]:.3f}",
                        f"{end_point[1]:.3f}",
                        straight_count,
                        bend_count,
                        f"{section_analysis.get('max_tension', 0):.0f}",
                        f"{section_analysis.get('max_pressure', 0):.2f}",
                        f"{section_analysis.get('lateral_deviation', 0):.3f}",
                        f"{section_analysis.get('length_error_percent', 0):.2f}",
                        section_analysis.get("status", "Unknown"),
                    ]
                )

    def generate_primitive_details(self, route: Route, output_path: Path) -> None:
        """Generate primitive geometry details CSV report.

        Args:
            route: Route object with fitted primitives
            output_path: Output CSV file path
        """
        with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)

            # Write header
            writer.writerow(self.headers["primitive_details"])

            for section in route.sections:
                for primitive in section.primitives:
                    if isinstance(primitive, Straight):
                        writer.writerow(
                            [
                                section.id,
                                "Straight",
                                f"{primitive.length_m:.3f}",
                                "",  # No angle for straights
                                f"{primitive.start_point[0]:.3f}",
                                f"{primitive.start_point[1]:.3f}",
                                f"{primitive.end_point[0]:.3f}",
                                f"{primitive.end_point[1]:.3f}",
                                "",  # No center for straights
                                "",
                            ]
                        )
                    elif isinstance(primitive, Bend):
                        writer.writerow(
                            [
                                section.id,
                                "Bend",
                                f"{primitive.radius_m:.3f}",
                                f"{primitive.angle_deg:.2f}",
                                "",  # Start/end points complex for bends
                                "",
                                "",
                                "",
                                f"{primitive.center_point[0]:.3f}",
                                f"{primitive.center_point[1]:.3f}",
                            ]
                        )

    def generate_calculation_results(
        self, route: Route, calculation_data: Dict[str, Any], output_path: Path
    ) -> None:
        """Generate detailed calculation results CSV.

        Args:
            route: Route object
            calculation_data: Detailed calculation results
            output_path: Output CSV file path
        """
        with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)

            # Write header
            writer.writerow(self.headers["calculation_results"])

            # Extract calculation data
            tensions = calculation_data.get("tensions", [])
            pressures = calculation_data.get("pressures", [])
            chainages = calculation_data.get("chainages", [])

            for i, chainage in enumerate(chainages):
                tension = tensions[i] if i < len(tensions) else 0
                pressure = pressures[i] if i < len(pressures) else 0

                # Determine which section this chainage belongs to
                section_id = self._get_section_at_chainage(route, chainage)

                writer.writerow(
                    [
                        section_id,
                        f"{chainage:.2f}",
                        f"{tension:.0f}",
                        f"{pressure:.2f}",
                        "",  # Bend radius - would need more detail
                        calculation_data.get("friction_coefficient", 0.3),
                        calculation_data.get("pull_direction", "Forward"),
                        (
                            "Pass"
                            if tension
                            < calculation_data.get(
                                "max_allowable_tension", float("inf")
                            )
                            else "Fail"
                        ),
                        "",
                    ]
                )

    def _get_section_at_chainage(self, route: Route, chainage: float) -> str:
        """Determine which section contains the given chainage."""
        cumulative_length = 0.0

        for section in route.sections:
            if (
                cumulative_length
                <= chainage
                <= cumulative_length + section.original_length
            ):
                return section.id
            cumulative_length += section.original_length

        return route.sections[-1].id if route.sections else "UNKNOWN"

    def generate_comprehensive_report(
        self,
        route: Route,
        analysis_results: Dict[str, Any],
        output_dir: Path,
        base_filename: str = "cable_analysis",
    ) -> List[Path]:
        """Generate all CSV reports for comprehensive analysis.

        Args:
            route: Route object
            analysis_results: Complete analysis results
            output_dir: Output directory
            base_filename: Base filename for reports

        Returns:
            List of generated file paths
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        generated_files = []

        # Route summary
        summary_path = output_dir / f"{base_filename}_summary.csv"
        self.generate_route_summary(route, analysis_results, summary_path)
        generated_files.append(summary_path)

        # Section details
        sections_path = output_dir / f"{base_filename}_sections.csv"
        self.generate_section_details(route, analysis_results, sections_path)
        generated_files.append(sections_path)

        # Primitive details
        primitives_path = output_dir / f"{base_filename}_primitives.csv"
        self.generate_primitive_details(route, primitives_path)
        generated_files.append(primitives_path)

        # Calculation results (if available)
        if "calculations" in analysis_results:
            calc_path = output_dir / f"{base_filename}_calculations.csv"
            self.generate_calculation_results(
                route, analysis_results["calculations"], calc_path
            )
            generated_files.append(calc_path)

        return generated_files


def generate_csv_report(
    route: Route,
    analysis_results: Dict[str, Any],
    output_path: Path,
    report_type: str = "summary",
) -> None:
    """Convenience function to generate CSV reports.

    Args:
        route: Route object
        analysis_results: Analysis results
        output_path: Output file path
        report_type: Type of report (summary, sections, primitives, calculations)
    """
    reporter = CSVReporter()

    if report_type == "summary":
        reporter.generate_route_summary(route, analysis_results, output_path)
    elif report_type == "sections":
        reporter.generate_section_details(route, analysis_results, output_path)
    elif report_type == "primitives":
        reporter.generate_primitive_details(route, output_path)
    elif report_type == "calculations":
        if "calculations" in analysis_results:
            reporter.generate_calculation_results(
                route, analysis_results["calculations"], output_path
            )
        else:
            raise ValueError("No calculation data available for calculations report")
    else:
        raise ValueError(f"Unknown report type: {report_type}")
