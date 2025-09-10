"""Main cable pulling analysis pipeline implementing the complete workflow."""

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt

from ..analysis.accuracy_analyzer import AccuracyAnalyzer
from ..calculations.pressure import PressureCalculator
from ..calculations.tension import TensionCalculator
from ..core.models import Bend, CableSpec, DuctSpec, Route, Section, Straight
from ..geometry.simple_segment_fitter import SimpleSegmentFitter
from ..geometry.splitter import RouteSplitter
from ..io.dxf_reader import DXFReader
from ..io.dxf_writer import DXFWriter
from ..reporting.csv_reporter import CSVReporter
from ..reporting.json_reporter import JSONReporter
from ..visualization.professional_matplotlib import ProfessionalMatplotlibPlotter


@dataclass
class AnalysisConfig:
    """Configuration for cable analysis pipeline."""

    # Geometry settings
    duct_type: str = "200mm"
    max_section_length_m: float = 1000.0

    # Cable specifications
    cable_diameter_mm: float = 50.0
    cable_weight_kg_m: float = 1.5
    cable_max_tension_n: float = 15000.0
    cable_max_sidewall_pressure_n_m: float = 300.0
    cable_min_bend_radius_mm: float = 500.0
    number_of_cables: int = 1

    # Output settings
    sample_interval_m: float = 25.0
    generate_json: bool = True
    generate_csv: bool = True
    generate_excel: bool = False
    generate_dxf: bool = False
    generate_png: bool = True


@dataclass
class SectionResult:
    """Results for a single section."""

    section_id: str
    length_m: float
    straight_count: int
    bend_count: int

    # Geometry details
    straights: List[Dict[str, float]]  # [{"length_m": 50.5}, ...]
    bends: List[Dict[str, float]]  # [{"angle_deg": 45.0, "radius_m": 3.9}, ...]

    # Pulling calculations
    forward_tension_n: float
    reverse_tension_n: float
    max_sidewall_pressure_n_m: float

    # Cumulative values
    cumulative_forward_n: float
    cumulative_reverse_n: float


@dataclass
class AnalysisResults:
    """Complete analysis results."""

    route_name: str
    total_length_m: float
    section_count: int

    # Geometry summary
    total_straights: int
    total_bends: int

    # Final pulling forces
    final_forward_tension_n: float
    final_reverse_tension_n: float
    max_sidewall_pressure_n_m: float

    # Section details
    sections: List[SectionResult]

    # Accuracy metrics
    excellent_accuracy_percent: float
    median_deviation_cm: float
    max_deviation_cm: float


class CableAnalysisPipeline:
    """Complete cable pulling analysis pipeline."""

    def __init__(self, config: Optional[AnalysisConfig] = None):
        """Initialize analysis pipeline with configuration."""
        self.config = config or AnalysisConfig()

        # Initialize components
        self.fitter = SimpleSegmentFitter(
            standard_radius=self._get_duct_radius(self.config.duct_type)
        )
        self.splitter = RouteSplitter(max_cable_length=self.config.max_section_length_m)
        self.visualizer = ProfessionalMatplotlibPlotter()
        self.analyzer = AccuracyAnalyzer(sample_interval=self.config.sample_interval_m)

        # Create cable and duct specs
        self.cable_spec = self._create_cable_spec()
        self.duct_spec = self._create_duct_spec()

        # Initialize calculators
        self.tension_calc = TensionCalculator()
        self.pressure_calc = PressureCalculator()

    def analyze_dxf(
        self, dxf_path: Union[str, Path], output_dir: Union[str, Path] = "output"
    ) -> AnalysisResults:
        """
        Complete analysis workflow for DXF file.

        Args:
            dxf_path: Path to DXF file
            output_dir: Output directory for results

        Returns:
            Complete analysis results
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Step 1: Digest DXF
        print("1. 📁 Loading DXF file...")
        reader = DXFReader(Path(dxf_path))
        reader.load()
        route = reader.create_route_from_polylines(Path(dxf_path).stem)

        # Step 2: Remove duplicate points (tidy)
        print("2. 🧹 Cleaning duplicate vertices...")
        for section in route.sections:
            section.original_polyline = self.fitter._remove_duplicate_vertices(
                section.original_polyline
            )

        # Step 3: Fillet all changes of direction
        print("3. 🔧 Applying filleting with duct radius...")
        for section in route.sections:
            result = self.fitter.fit_section_to_primitives(section)
            section.primitives = result.primitives

        # Step 4: Split long sections
        print("4. ✂️  Splitting long sections...")
        split_result = self.splitter.split_route(route)
        route = split_result.split_route

        # Re-fit any new sections created during splitting
        for section in route.sections:
            if not section.primitives:  # Section has no fitted geometry
                result = self.fitter.fit_section_to_primitives(section)
                section.primitives = result.primitives

        # Store all sections (including empty ones) for CSV reporting
        all_sections_with_splits = route.sections.copy()
        # Store for later use - avoiding mypy issues by using setattr
        setattr(route, "_all_sections_with_splits", all_sections_with_splits)

        # Step 5: Generate PNG visualizations
        if self.config.generate_png:
            print("5. 🖼️  Generating visualizations...")
            self._generate_visualizations(route, output_path)

        # Calculate total original length before filtering
        original_total_length = sum(s.original_length for s in route.sections)

        # Filter out empty sections
        route.sections = [s for s in route.sections if len(s.primitives) > 0]
        print(f"Filtered to {len(route.sections)} non-empty sections")

        # Store original length on route for reporting - avoiding mypy issues by using setattr
        setattr(route, "_original_total_length", original_total_length)

        # Step 6: Apply pulling calculations
        print("6. 📊 Calculating pulling forces...")
        section_results = self._calculate_pulling_forces(route)

        # Step 7: Generate section reports
        print("7. 📝 Generating section reports...")
        if self.config.generate_json:
            self._export_json_reports(section_results, output_path, route)

        if self.config.generate_csv:
            self._export_csv_reports(section_results, output_path, route)

        # Step 8: Generate summary report
        print("8. 📋 Generating summary report...")
        summary_results = self._create_summary_results(route, section_results)
        self._export_summary_reports(summary_results, output_path)

        # Step 9: Export DXF (optional)
        if self.config.generate_dxf:
            print("9. 📐 Exporting fitted DXF...")
            self._export_fitted_dxf(route, output_path)

        print("✅ Analysis complete!")
        return summary_results

    def _get_duct_radius(self, duct_type: str) -> float:
        """Get bend radius for duct type."""
        from ..inventory.duct_inventory import DUCT_SPECIFICATIONS

        if duct_type in DUCT_SPECIFICATIONS:
            spec = DUCT_SPECIFICATIONS[duct_type]
            return spec.bends[0].radius_m if spec.bends else 3.9
        return 3.9

    def _create_cable_spec(self) -> CableSpec:
        """Create cable specification from config."""
        return CableSpec(
            diameter=self.config.cable_diameter_mm,
            weight_per_meter=self.config.cable_weight_kg_m,
            max_tension=self.config.cable_max_tension_n,
            max_sidewall_pressure=self.config.cable_max_sidewall_pressure_n_m,
            min_bend_radius=self.config.cable_min_bend_radius_mm,
            number_of_cables=self.config.number_of_cables,
        )

    def _create_duct_spec(self) -> DuctSpec:
        """Create duct specification from config."""
        # Simplified - would normally look up from inventory
        return DuctSpec(
            inner_diameter=200,  # mm
            type="HDPE",
            friction_dry=0.5,
            friction_lubricated=0.3,
        )

    def _generate_visualizations(self, route: Route, output_path: Path) -> None:
        """Generate PNG visualizations."""
        vis_path = output_path / "visualizations"
        vis_path.mkdir(exist_ok=True)

        # Overall route
        fig, ax = self.visualizer.plot_professional_route(
            route,
            title=f"Cable Route Analysis: {route.name}",
            show_section_colors=False,  # Simplified route overview
            show_fitted_geometry=False,  # Only original route and markers
        )
        fig.savefig(vis_path / "route_overview.png", dpi=100, bbox_inches="tight")
        plt.close(fig)

        # Individual sections
        sections_path = vis_path / "sections"
        sections_path.mkdir(exist_ok=True)

        # Use all sections including empty ones for visualization
        all_sections = (
            route._all_sections_with_splits
            if hasattr(route, "_all_sections_with_splits")
            else route.sections
        )
        for i, section in enumerate(all_sections):
            # Create a route with just this section for visualization
            single_section_route = Route(name=f"Section {section.id}")
            single_section_route.sections = [section]

            fig, ax = self.visualizer.plot_professional_route(
                single_section_route,
                title=f"{section.id} Detail",
                label_start_index=i,  # Use correct starting letter
            )
            fig.savefig(
                sections_path / f"{section.id}.png", dpi=100, bbox_inches="tight"
            )
            plt.close(fig)

    def _calculate_pulling_forces(self, route: Route) -> List[SectionResult]:
        """Calculate pulling forces for all sections."""
        results = []
        cumulative_forward = 0.0
        cumulative_reverse = 0.0

        for section in route.sections:
            # Calculate section forces
            forward_tension = self.tension_calc.calculate_forward_tension(
                section, self.cable_spec, self.duct_spec
            )
            reverse_tension = self.tension_calc.calculate_reverse_tension(
                section, self.cable_spec, self.duct_spec
            )
            max_pressure = self.pressure_calc.calculate_max_sidewall_pressure(
                section, self.cable_spec, self.duct_spec
            )

            # Update cumulative values
            cumulative_forward += forward_tension
            cumulative_reverse += reverse_tension

            # Extract geometry details with cumulative tensions
            straights = []
            bends = []

            # Get detailed tension calculations for this section
            from ..calculations.tension import analyze_section_tension

            tension_analysis = analyze_section_tension(
                section, self.cable_spec, self.duct_spec
            )

            # Build ordered geometry arrays matching CSV format (interleaved straights/bends)
            # First separate straights and bends
            section_straights = []
            section_bends = []

            for i, primitive in enumerate(section.primitives):
                if hasattr(primitive, "length_m"):  # Straight
                    # Get tension at end of this primitive
                    forward_tension = (
                        tension_analysis.forward_tensions[i].tension
                        if i < len(tension_analysis.forward_tensions)
                        else 0
                    )

                    # For reverse tension, reverse the mapping so first primitive gets highest tension
                    reverse_idx = len(section.primitives) - 1 - i
                    reverse_tension = (
                        tension_analysis.backward_tensions[reverse_idx].tension
                        if reverse_idx < len(tension_analysis.backward_tensions)
                        else 0
                    )

                    section_straights.append(
                        {
                            "length_m": primitive.length_m,
                            "cumulative_forward_tension_n": forward_tension,
                            "cumulative_reverse_tension_n": reverse_tension,
                            "primitive_order": i,  # Track original order
                        }
                    )

                elif isinstance(primitive, Bend):  # Bend
                    # Get tension at end of this primitive
                    forward_tension = (
                        tension_analysis.forward_tensions[i].tension
                        if i < len(tension_analysis.forward_tensions)
                        else 0
                    )

                    # For reverse tension, reverse the mapping so first primitive gets highest tension
                    reverse_idx = len(section.primitives) - 1 - i
                    reverse_tension = (
                        tension_analysis.backward_tensions[reverse_idx].tension
                        if reverse_idx < len(tension_analysis.backward_tensions)
                        else 0
                    )

                    sidewall_pressure = (
                        forward_tension / primitive.radius_m
                        if primitive.radius_m > 0
                        else 0
                    )

                    section_bends.append(
                        {
                            "angle_deg": primitive.angle_deg,
                            "radius_m": primitive.radius_m,
                            "cumulative_forward_tension_n": forward_tension,
                            "cumulative_reverse_tension_n": reverse_tension,
                            "sidewall_pressure_n_m": sidewall_pressure,
                            "primitive_order": i,  # Track original order
                        }
                    )

            # Sort both arrays by original primitive order for JSON output
            straights = sorted(section_straights, key=lambda x: x["primitive_order"])
            bends = sorted(section_bends, key=lambda x: x["primitive_order"])

            # Remove the order tracking field from final output
            for s in straights:
                del s["primitive_order"]
            for b in bends:
                del b["primitive_order"]

            result = SectionResult(
                section_id=section.id,
                length_m=section.total_length,
                straight_count=len(straights),
                bend_count=len(bends),
                straights=straights,
                bends=bends,
                forward_tension_n=forward_tension,
                reverse_tension_n=reverse_tension,
                max_sidewall_pressure_n_m=max_pressure,
                cumulative_forward_n=cumulative_forward,
                cumulative_reverse_n=cumulative_reverse,
            )

            results.append(result)

        return results

    def _create_summary_results(
        self, route: Route, section_results: List[SectionResult]
    ) -> AnalysisResults:
        """Create summary analysis results."""

        # Run accuracy analysis
        accuracy = self.analyzer.analyze_route_accuracy(route, methodology="direct")

        # Count ALL sections including empty subsections
        all_sections = getattr(route, "_all_sections_with_splits", route.sections)
        total_section_count = len(all_sections)

        return AnalysisResults(
            route_name=route.name,
            total_length_m=getattr(route, "_original_total_length", route.total_length),
            section_count=total_section_count,
            total_straights=sum(r.straight_count for r in section_results),
            total_bends=sum(r.bend_count for r in section_results),
            final_forward_tension_n=(
                section_results[-1].cumulative_forward_n if section_results else 0
            ),
            final_reverse_tension_n=(
                section_results[-1].cumulative_reverse_n if section_results else 0
            ),
            max_sidewall_pressure_n_m=(
                max(r.max_sidewall_pressure_n_m for r in section_results)
                if section_results
                else 0
            ),
            sections=section_results,
            excellent_accuracy_percent=accuracy.excellent_percentage,
            median_deviation_cm=accuracy.global_median_deviation * 100,
            max_deviation_cm=accuracy.global_max_deviation * 100,
        )

    def _export_json_reports(
        self, section_results: List[SectionResult], output_path: Path, route: Route
    ) -> None:
        """Export JSON reports."""
        json_path = output_path / "json"
        json_path.mkdir(exist_ok=True)

        # Create lookup for results
        results_by_id = {r.section_id: r for r in section_results}

        # Export ALL sections including empty subsections
        all_sections = getattr(route, "_all_sections_with_splits", route.sections)
        for section in all_sections:
            if section.id in results_by_id:
                # Non-empty section with data
                result = results_by_id[section.id]
                with open(json_path / f"section_{result.section_id}.json", "w") as f:
                    json.dump(asdict(result), f, indent=2)
            else:
                # Empty subsection
                empty_result = {
                    "section_id": section.id,
                    "length_m": section.original_length,
                    "straight_count": 0,
                    "bend_count": 0,
                    "straights": [],
                    "bends": [],
                    "forward_tension_n": 0.0,
                    "reverse_tension_n": 0.0,
                    "max_sidewall_pressure_n_m": 0.0,
                    "cumulative_forward_n": 0.0,
                    "cumulative_reverse_n": 0.0,
                }
                with open(json_path / f"section_{section.id}.json", "w") as f:
                    json.dump(empty_result, f, indent=2)

    def _export_csv_reports(
        self, section_results: List[SectionResult], output_path: Path, route: Route
    ) -> None:
        """Export CSV reports."""
        csv_path = output_path / "csv"
        csv_path.mkdir(exist_ok=True)

        # Sections summary CSV - include ALL sections (even empty split ones)
        with open(csv_path / "sections_summary.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "Section ID",
                    "Length (m)",
                    "Straights",
                    "Bends",
                    "Forward Tension (N)",
                    "Reverse Tension (N)",
                    "Max Sidewall Pressure (N/m)",
                    "Cumulative Forward (N)",
                    "Cumulative Reverse (N)",
                ]
            )

            # Create a lookup for section results
            results_by_id = {r.section_id: r for r in section_results}

            # Include all sections from split result
            all_sections = getattr(route, "_all_sections_with_splits", route.sections)
            for section in all_sections:
                if section.id in results_by_id:
                    result = results_by_id[section.id]
                    writer.writerow(
                        [
                            result.section_id,
                            result.length_m,
                            result.straight_count,
                            result.bend_count,
                            result.forward_tension_n,
                            result.reverse_tension_n,
                            result.max_sidewall_pressure_n_m,
                            result.cumulative_forward_n,
                            result.cumulative_reverse_n,
                        ]
                    )
                else:
                    # Empty split section
                    writer.writerow(
                        [section.id, section.original_length, 0, 0, 0, 0, 0, 0, 0]
                    )

        # Individual section CSV files - ALL sections including empty ones
        all_sections = getattr(route, "_all_sections_with_splits", route.sections)
        for section in all_sections:
            if section.id in results_by_id:
                result = results_by_id[section.id]
                self._export_individual_section_csv(result, csv_path)
            else:
                # Empty subsection CSV
                self._export_empty_section_csv(section, csv_path)

    def _export_individual_section_csv(
        self, result: SectionResult, csv_path: Path
    ) -> None:
        """Export CSV for individual section with detailed geometry and cumulative tensions."""
        filename = csv_path / f"section_{result.section_id}.csv"

        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Section", result.section_id])
            writer.writerow(["Total Length (m)", result.length_m])
            writer.writerow([])

            # Geometry breakdown with cumulative tensions
            writer.writerow(
                [
                    "Type",
                    "Length/Angle",
                    "Radius (m)",
                    "Cumulative Forward (N)",
                    "Cumulative Reverse (N)",
                    "Forward Sidewall (N/m)",
                    "Reverse Sidewall (N/m)",
                ]
            )

            # Create ordered arrays that interleave straights and bends
            geometry_items = []
            straight_idx = 0
            bend_idx = 0

            # Build the interleaved sequence
            for i in range(len(result.straights) + len(result.bends)):
                if i % 2 == 0 and straight_idx < len(result.straights):
                    geometry_items.append(("straight", result.straights[straight_idx]))
                    straight_idx += 1
                elif bend_idx < len(result.bends):
                    geometry_items.append(("bend", result.bends[bend_idx]))
                    bend_idx += 1

            # For reverse tensions: we want to display them in descending order
            # So first CSV row gets highest reverse tension (from end of route)
            # Last CSV row gets lowest reverse tension (from start of route)

            # Write rows using forward geometry order but reverse tensions in descending order
            for i, (item_type, item_data) in enumerate(geometry_items):
                # Use same index - JSON data is already ordered with highest reverse at start
                reverse_item_type, reverse_item_data = geometry_items[i]

                if item_type == "straight":
                    forward_tension = item_data["cumulative_forward_tension_n"]
                    reverse_tension = reverse_item_data["cumulative_reverse_tension_n"]

                    writer.writerow(
                        [
                            "Straight",
                            f"{item_data['length_m']:.1f}m",
                            "",
                            f"{forward_tension:.0f}",
                            f"{reverse_tension:.0f}",
                            "",  # No sidewall pressure for straights
                            "",  # No sidewall pressure for straights
                        ]
                    )

                elif item_type == "bend":
                    forward_tension = item_data["cumulative_forward_tension_n"]
                    reverse_tension = reverse_item_data["cumulative_reverse_tension_n"]

                    # Calculate both forward and reverse sidewall pressures using actual tensions
                    forward_sidewall = forward_tension / item_data["radius_m"]
                    reverse_sidewall = reverse_tension / item_data["radius_m"]

                    writer.writerow(
                        [
                            "Bend",
                            f"{item_data['angle_deg']:.1f}deg",
                            f"{item_data['radius_m']:.1f}",
                            f"{forward_tension:.0f}",
                            f"{reverse_tension:.0f}",
                            f"{forward_sidewall:.0f}",
                            f"{reverse_sidewall:.0f}",
                        ]
                    )

    def _export_empty_section_csv(self, section: Section, csv_path: Path) -> None:
        """Export CSV for empty subsection."""
        filename = csv_path / f"section_{section.id}.csv"

        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Section", section.id])
            writer.writerow(["Total Length (m)", section.original_length])
            writer.writerow(["Status", "Empty subsection (no primitives)"])
            writer.writerow([])
            writer.writerow(
                [
                    "Type",
                    "Length/Angle",
                    "Radius (m)",
                    "Cumulative Forward (N)",
                    "Cumulative Reverse (N)",
                    "Sidewall Pressure (N/m)",
                ]
            )
            writer.writerow(["No geometry data", "", "", "0", "0", ""])

    def _export_summary_reports(
        self, results: AnalysisResults, output_path: Path
    ) -> None:
        """Export summary reports."""

        # JSON summary
        if self.config.generate_json:
            with open(output_path / "analysis_summary.json", "w") as f:
                json.dump(asdict(results), f, indent=2)

        # CSV summary
        if self.config.generate_csv:
            with open(output_path / "analysis_summary.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Metric", "Value", "Unit"])
                writer.writerow(["Route Name", results.route_name, ""])
                writer.writerow(["Total Length", results.total_length_m, "m"])
                writer.writerow(["Section Count", results.section_count, ""])
                writer.writerow(["Total Straights", results.total_straights, ""])
                writer.writerow(["Total Bends", results.total_bends, ""])
                writer.writerow(
                    ["Final Forward Tension", results.final_forward_tension_n, "N"]
                )
                writer.writerow(
                    ["Final Reverse Tension", results.final_reverse_tension_n, "N"]
                )
                writer.writerow(
                    ["Max Sidewall Pressure", results.max_sidewall_pressure_n_m, "N/m"]
                )
                writer.writerow(
                    ["Excellent Accuracy", results.excellent_accuracy_percent, "%"]
                )
                writer.writerow(["Median Deviation", results.median_deviation_cm, "cm"])
                writer.writerow(["Max Deviation", results.max_deviation_cm, "cm"])

    def _export_fitted_dxf(self, route: Route, output_path: Path) -> None:
        """Export fitted route as DXF."""
        from ..io.dxf_writer import export_route_to_dxf

        fitted_dxf_path = output_path / "fitted_route.dxf"
        export_route_to_dxf(route, fitted_dxf_path)


def analyze_cable_route(
    dxf_path: Union[str, Path],
    output_dir: Union[str, Path] = "output",
    config: Optional[AnalysisConfig] = None,
    **kwargs,
) -> AnalysisResults:
    """
    Convenience function to analyze a cable route from DXF file.

    Args:
        dxf_path: Path to DXF file
        output_dir: Output directory for results
        config: Analysis configuration
        **kwargs: Configuration overrides

    Returns:
        Complete analysis results
    """
    if config is None:
        config = AnalysisConfig()

    # Apply any keyword overrides
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    pipeline = CableAnalysisPipeline(config)
    return pipeline.analyze_dxf(dxf_path, output_dir)
