"""JSON report generation for cable pulling analysis."""

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from ..core.models import Bend, Route, Straight


class JSONReporter:
    """Generate structured JSON reports for cable pulling analysis."""

    def __init__(self, indent: int = 2) -> None:
        """Initialize JSON reporter.

        Args:
            indent: JSON indentation for readable output
        """
        self.indent = indent

    def generate_route_report(
        self, route: Route, analysis_results: Dict[str, Any], output_path: Path
    ) -> None:
        """Generate comprehensive JSON report.

        Args:
            route: Route object
            analysis_results: Analysis calculation results
            output_path: Output JSON file path
        """
        report = {
            "metadata": {
                "route_name": route.name,
                "analysis_date": datetime.now().isoformat(),
                "generator": "Easy Cable Pulling Library",
                "version": "1.0.0",
            },
            "route_summary": self._build_route_summary(route, analysis_results),
            "sections": self._build_sections_data(route, analysis_results),
            "geometry": self._build_geometry_data(route),
            "calculations": self._build_calculations_data(analysis_results),
            "validation": self._build_validation_data(analysis_results),
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=self.indent, ensure_ascii=False)

    def _build_route_summary(
        self, route: Route, analysis_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build route summary section."""
        total_length = sum(s.original_length for s in route.sections)

        return {
            "total_length_m": round(total_length, 3),
            "section_count": route.section_count,
            "max_tension_n": analysis_results.get("max_tension", 0),
            "max_pressure_n_per_m": analysis_results.get("max_pressure", 0),
            "overall_status": analysis_results.get("overall_status", "Unknown"),
            "limiting_factor": analysis_results.get("limiting_factor", "N/A"),
            "recommended_pull_direction": analysis_results.get(
                "recommended_direction", "Forward"
            ),
            "total_bends": sum(
                len([p for p in s.primitives if hasattr(p, "radius_m")])
                for s in route.sections
            ),
            "total_straights": sum(
                len([p for p in s.primitives if hasattr(p, "length_m")])
                for s in route.sections
            ),
        }

    def _build_sections_data(
        self, route: Route, analysis_results: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Build detailed sections data."""
        sections_data = []
        section_results = analysis_results.get("sections", {})

        for section in route.sections:
            if not section.original_polyline:
                continue

            section_analysis = section_results.get(section.id, {})

            section_data = {
                "id": section.id,
                "length_m": round(section.original_length, 3),
                "start_point": {
                    "x_m": round(section.original_polyline[0][0], 6),
                    "y_m": round(section.original_polyline[0][1], 6),
                },
                "end_point": {
                    "x_m": round(section.original_polyline[-1][0], 6),
                    "y_m": round(section.original_polyline[-1][1], 6),
                },
                "primitive_count": {
                    "straights": len(
                        [p for p in section.primitives if hasattr(p, "length_m")]
                    ),
                    "bends": len(
                        [p for p in section.primitives if hasattr(p, "radius_m")]
                    ),
                },
                "analysis": {
                    "max_tension_n": section_analysis.get("max_tension", 0),
                    "max_pressure_n_per_m": section_analysis.get("max_pressure", 0),
                    "lateral_deviation_m": section_analysis.get("lateral_deviation", 0),
                    "length_error_percent": section_analysis.get(
                        "length_error_percent", 0
                    ),
                    "status": section_analysis.get("status", "Unknown"),
                },
                "original_polyline": [
                    {"x_m": round(p[0], 6), "y_m": round(p[1], 6)}
                    for p in section.original_polyline
                ],
            }

            sections_data.append(section_data)

        return sections_data

    def _build_geometry_data(self, route: Route) -> Dict[str, Any]:
        """Build fitted geometry data."""
        geometry_data = {
            "primitives": [],
            "fitting_statistics": {
                "total_primitives": 0,
                "straight_total_length_m": 0,
                "bend_total_angle_deg": 0,
            },
        }

        total_primitives = 0
        total_straight_length = 0.0
        total_bend_angle = 0.0

        for section in route.sections:
            for primitive in section.primitives:
                primitive_data: Dict[str, Any] = {
                    "section_id": section.id,
                    "type": type(primitive).__name__.lower(),
                }

                if isinstance(primitive, Straight):
                    primitive_data.update(
                        {
                            "length_m": round(primitive.length_m, 3),
                            "start_point": {
                                "x_m": round(primitive.start_point[0], 6),
                                "y_m": round(primitive.start_point[1], 6),
                            },
                            "end_point": {
                                "x_m": round(primitive.end_point[0], 6),
                                "y_m": round(primitive.end_point[1], 6),
                            },
                        }
                    )
                    total_straight_length += float(primitive.length_m)

                elif isinstance(primitive, Bend):
                    primitive_data.update(
                        {
                            "radius_m": round(primitive.radius_m, 3),
                            "angle_deg": round(primitive.angle_deg, 2),
                            "direction": primitive.direction,
                            "center_point": {
                                "x_m": round(primitive.center_point[0], 6),
                                "y_m": round(primitive.center_point[1], 6),
                            },
                            "arc_length_m": round(
                                primitive.radius_m
                                * math.radians(abs(primitive.angle_deg)),
                                3,
                            ),
                        }
                    )
                    total_bend_angle += float(abs(primitive.angle_deg))

                geometry_data["primitives"].append(primitive_data)
                total_primitives += 1

        # Update statistics
        geometry_data["fitting_statistics"].update(
            {
                "total_primitives": total_primitives,
                "straight_total_length_m": round(total_straight_length, 3),
                "bend_total_angle_deg": round(total_bend_angle, 2),
            }
        )

        return geometry_data

    def _build_calculations_data(
        self, analysis_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build calculations data section."""
        calculations = analysis_results.get("calculations", {})

        return {
            "tension_analysis": {
                "max_tension_n": calculations.get("max_tension", 0),
                "critical_section": calculations.get("critical_section", ""),
                "pull_direction": calculations.get("recommended_direction", "Forward"),
                "tension_profile": calculations.get("tension_profile", []),
            },
            "pressure_analysis": {
                "max_pressure_n_per_m": calculations.get("max_pressure", 0),
                "critical_bend": calculations.get("critical_bend", ""),
                "pressure_points": calculations.get("pressure_points", []),
            },
            "parameters": {
                "cable_specification": calculations.get("cable_spec", {}),
                "duct_specification": calculations.get("duct_spec", {}),
                "friction_coefficients": calculations.get("friction_coefficients", {}),
                "environmental_factors": calculations.get("environmental_factors", {}),
            },
        }

    def _build_validation_data(
        self, analysis_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build validation and compliance data."""
        validation = analysis_results.get("validation", {})

        return {
            "geometry_validation": {
                "total_lateral_deviation_m": validation.get(
                    "total_lateral_deviation", 0
                ),
                "max_lateral_deviation_m": validation.get("max_lateral_deviation", 0),
                "length_error_percent": validation.get("length_error_percent", 0),
                "geometry_status": validation.get("geometry_status", "Unknown"),
            },
            "engineering_compliance": {
                "min_bend_radius_check": validation.get("min_bend_radius_ok", True),
                "max_tension_check": validation.get("max_tension_ok", True),
                "max_pressure_check": validation.get("max_pressure_ok", True),
                "jam_ratio_check": validation.get("jam_ratio_ok", True),
            },
            "standards_compliance": {
                "ieee_525": validation.get("ieee_525_compliant", True),
                "iec_standards": validation.get("iec_compliant", True),
                "local_regulations": validation.get("local_compliant", True),
            },
        }

    def generate_machine_readable_export(
        self, route: Route, analysis_results: Dict[str, Any], output_path: Path
    ) -> None:
        """Generate machine-readable JSON for API integration.

        Args:
            route: Route object
            analysis_results: Analysis results
            output_path: Output JSON file path
        """
        # Compact format for machine consumption
        export_data = {
            "route_id": route.name,
            "timestamp": datetime.now().isoformat(),
            "status": analysis_results.get("overall_status", "Unknown"),
            "metrics": {
                "total_length": sum(s.original_length for s in route.sections),
                "section_count": route.section_count,
                "max_tension": analysis_results.get("max_tension", 0),
                "max_pressure": analysis_results.get("max_pressure", 0),
            },
            "sections": [
                {
                    "id": s.id,
                    "length": s.original_length,
                    "start": (
                        list(s.original_polyline[0]) if s.original_polyline else [0, 0]
                    ),
                    "end": (
                        list(s.original_polyline[-1]) if s.original_polyline else [0, 0]
                    ),
                    "primitives": len(s.primitives),
                }
                for s in route.sections
            ],
            "compliance": analysis_results.get("validation", {}),
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, separators=(",", ":"))  # Compact format


def generate_json_report(
    route: Route,
    analysis_results: Dict[str, Any],
    output_path: Path,
    report_format: str = "comprehensive",
) -> None:
    """Convenience function to generate JSON reports.

    Args:
        route: Route object
        analysis_results: Analysis results
        output_path: Output file path
        report_format: Format type (comprehensive, machine_readable)
    """
    reporter = JSONReporter()

    if report_format == "comprehensive":
        reporter.generate_route_report(route, analysis_results, output_path)
    elif report_format == "machine_readable":
        reporter.generate_machine_readable_export(route, analysis_results, output_path)
    else:
        raise ValueError(f"Unknown report format: {report_format}")
