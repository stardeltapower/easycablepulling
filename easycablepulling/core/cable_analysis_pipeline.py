"""Main cable pulling analysis pipeline implementing the complete workflow."""

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt

from ..analysis.accuracy_analyzer import AccuracyAnalyzer
from ..analysis.route_optimizer import RouteOptimizer, PullingDirection
from ..calculations.config import CalculationConfig, CalculationStandard
from ..calculations.pressure import PressureCalculator
from ..calculations.tension import TensionCalculator
from ..core.models import Bend, CableSpec, DuctSpec, Route, Section, Straight
from ..geometry.simple_segment_fitter import SimpleSegmentFitter
from ..geometry.splitter import RouteSplitter
from ..inventory.duct_inventory import DuctInventory
from ..io.dxf_reader import DXFReader
from ..visualization.professional_matplotlib import ProfessionalMatplotlibPlotter


@dataclass
class AnalysisConfig:
    """Configuration for cable analysis pipeline."""

    # Geometry settings
    duct_type: str = "200mm"
    max_section_length_m: float = 1000.0

    # Cable specifications (per individual cable)
    cable_diameter_mm: float = 50.0  # Individual cable diameter
    cable_weight_kg_m: float = 1.5  # Weight per individual cable
    cable_max_tension_n: float = 15000.0
    cable_max_sidewall_pressure_n_m: float = 300.0
    cable_min_bend_radius_mm: float = 500.0
    number_of_cables: int = 1
    cable_arrangement: str = "single"  # "single", "trefoil", or "flat"

    # Calculation method
    calculation_standard: str = "cigre"  # "cigre", "aeic", or "polywater"

    # Splitting method
    splitting_method: str = "simple"  # "simple" (length-based) or "optimizer" (tension/pressure-based)
    target_utilization: float = 0.8  # Target utilization for optimizer (80% = 20% safety margin)

    # Friction and lubrication settings
    lubricated: Union[bool, List[bool]] = False
    friction_override: Optional[Union[float, List[float]]] = None

    # Pulling settings
    initial_tension_n: float = 100.0  # Drum/winch tension (minimum tension at pull start)

    # Output settings
    sample_interval_m: float = 25.0
    generate_json: bool = True
    generate_csv: bool = True
    generate_excel: bool = False
    generate_dxf: bool = False
    generate_png: bool = True
    generate_latex: bool = True


@dataclass
class SectionResult:
    """Results for a single section."""

    section_id: str
    length_m: float
    straight_count: int
    bend_count: int

    # Geometry details - ordered list of primitives in sequence
    primitives: List[Dict[str, float]]  # Ordered list: [{"type": "straight", "length_m": 50.5, "cumulative_forward_tension_n": ...}, ...]

    # Legacy fields for backward compatibility (deprecated)
    straights: List[Dict[str, float]]  # [{"length_m": 50.5}, ...]
    bends: List[Dict[str, float]]  # [{"angle_deg": 45.0, "radius_m": 3.9}, ...]

    # Pulling calculations
    forward_tension_n: float
    reverse_tension_n: float
    max_sidewall_pressure_n_m: float  # Legacy field - max of forward/reverse
    forward_sidewall_pressure_n_m: float  # Forward pulling max sidewall
    reverse_sidewall_pressure_n_m: float  # Reverse pulling max sidewall

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

    def __init__(self, config: Optional[AnalysisConfig] = None) -> None:
        """Initialize analysis pipeline with configuration."""
        self.config = config or AnalysisConfig()

        # Initialize components
        duct_inventory = DuctInventory(self.config.duct_type)
        self.fitter = SimpleSegmentFitter(
            duct_inventory=duct_inventory,
            standard_radius=self._get_duct_radius(self.config.duct_type),
        )
        self.visualizer = ProfessionalMatplotlibPlotter()
        self.analyzer = AccuracyAnalyzer(sample_interval=self.config.sample_interval_m)

        # Create cable and duct specs
        self.cable_spec = self._create_cable_spec()
        self.duct_spec = self._create_duct_spec()

        # Create calculation config from standard string
        standard_map = {
            "cigre": CalculationStandard.CIGRE,
            "aeic": CalculationStandard.AEIC,
            "polywater": CalculationStandard.POLYWATER,
        }
        calc_standard = standard_map.get(
            self.config.calculation_standard.lower(), CalculationStandard.CIGRE
        )
        self.calc_config = CalculationConfig(standard=calc_standard)

        # Initialize calculators with config
        self.tension_calc = TensionCalculator(config=self.calc_config)
        self.pressure_calc = PressureCalculator(config=self.calc_config)

        # Initialize splitting method based on configuration
        splitting_method = self.config.splitting_method.lower()
        if splitting_method == "optimizer":
            # Use smart optimizer that splits based on tension/pressure limits
            self.optimizer = RouteOptimizer(
                cable_spec=self.cable_spec,
                duct_spec=self.duct_spec,
                target_utilization=self.config.target_utilization,
                max_section_length=self.config.max_section_length_m,
                config=self.calc_config,
            )
            self.splitter = None
        else:
            # Use simple length-based splitter (default)
            self.splitter = RouteSplitter(max_cable_length=self.config.max_section_length_m)
            self.optimizer = None

    def analyze_dxf(
        self, dxf_path: Union[str, Path], output_dir: Union[str, Path] = "output", dxf_layer: Optional[str] = None
    ) -> AnalysisResults:
        """
        Complete analysis workflow for DXF file.

        Args:
            dxf_path: Path to DXF file
            output_dir: Output directory for results
            dxf_layer: Specific DXF layer to use (None for default)

        Returns:
            Complete analysis results
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Step 1: Digest DXF
        print("1. [FILE] Loading DXF file...")
        reader = DXFReader(Path(dxf_path))
        reader.load()
        route = reader.create_route_from_polylines(Path(dxf_path).stem, layer_name=dxf_layer)

        # Step 2: Remove duplicate points (tidy)
        print("2. [PROCESSING] Cleaning duplicate vertices...")
        for section in route.sections:
            section.original_polyline = self.fitter._remove_duplicate_vertices(
                section.original_polyline
            )

        # Step 3: Fillet all changes of direction
        print("3. [PROCESSING] Applying filleting with duct radius...")
        for section in route.sections:
            result = self.fitter.fit_section_to_primitives(section)
            section.primitives = result.primitives

        # Step 4: Split long sections
        print("4. [SPLITTING] Splitting long sections...")
        if self.optimizer:
            # Use optimizer for intelligent splitting based on tension/pressure
            print("   Using intelligent optimizer (tension/pressure-based splitting)...")
            opt_result = self.optimizer.optimize_route(
                route=route,
                direction=PullingDirection.FORWARD,
                friction_override=self.config.friction_override if isinstance(self.config.friction_override, float) else None,
            )
            # Convert OptimizationResult back to Route with split sections
            route = self._optimizer_result_to_route(route, opt_result)
            # Sections are now in order from DXF file (A at northernmost point)
        else:
            # Use simple length-based splitter
            print("   Using simple splitter (length-based splitting)...")
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
            print("5. [VIZ] Generating visualizations...")
            self._generate_visualizations(route, output_path)

        # Calculate total original length before filtering
        original_total_length = sum(s.original_length for s in route.sections)

        # Filter out empty sections
        route.sections = [s for s in route.sections if len(s.primitives) > 0]
        print(f"Filtered to {len(route.sections)} non-empty sections")

        # Store original length on route for reporting - avoiding mypy issues by using setattr
        setattr(route, "_original_total_length", original_total_length)

        # Step 6: Apply pulling calculations
        print("6. [CALC] Calculating pulling forces...")
        section_results = self._calculate_pulling_forces(route)

        # Step 7: Generate section reports
        print("7. [REPORT] Generating section reports...")
        if self.config.generate_json:
            self._export_json_reports(section_results, output_path, route)

        if self.config.generate_csv:
            self._export_csv_reports(section_results, output_path, route)
            self._export_coordinates_csv(route, output_path)

        if self.config.generate_latex:
            self._export_latex_reports(section_results, output_path, route)

        # Step 8: Generate summary report
        print("8. [SUMMARY] Generating summary report...")
        summary_results = self._create_summary_results(route, section_results)
        self._export_summary_reports(summary_results, output_path)

        # Step 9: Export DXF (optional)
        if self.config.generate_dxf:
            print("9. [DXF] Exporting fitted DXF...")
            self._export_fitted_dxf(route, output_path)

        print("[PASS] Analysis complete!")
        return summary_results

    def _optimizer_result_to_route(self, original_route: Route, opt_result) -> Route:
        """Convert OptimizationResult back to Route with split sections.

        Args:
            original_route: Original route (for name and metadata)
            opt_result: OptimizationResult from RouteOptimizer

        Returns:
            New Route with optimized sections
        """
        from ..analysis.route_optimizer import OptimizationResult

        new_route = Route(name=original_route.name)

        for opt_section in opt_result.sections:
            # Extract primitives from PrimitiveResult objects
            primitives = [pr.primitive for pr in opt_section.primitives]

            # Reconstruct polyline from primitive coordinates
            # This preserves the actual DXF route geometry for visualization
            import math
            polyline = []
            for i, prim_result in enumerate(primitives):
                # Extract actual primitive from PrimitiveResult wrapper
                prim = prim_result.primitive if hasattr(prim_result, 'primitive') else prim_result

                if isinstance(prim, Straight):
                    # Add start point (only for first primitive)
                    if i == 0:
                        polyline.append(prim.start_point)
                    # Always add end point
                    polyline.append(prim.end_point)
                elif isinstance(prim, Bend):
                    # Calculate bend endpoints from center and angles
                    cx, cy = prim.center_point
                    r = prim.radius_m

                    # Convert angles to radians
                    start_rad = math.radians(prim.start_angle_deg)
                    end_rad = math.radians(prim.end_angle_deg)

                    # Calculate start point (only for first primitive)
                    if i == 0:
                        start_x = cx + r * math.cos(start_rad)
                        start_y = cy + r * math.sin(start_rad)
                        polyline.append((start_x, start_y))

                    # Always add end point
                    end_x = cx + r * math.cos(end_rad)
                    end_y = cy + r * math.sin(end_rad)
                    polyline.append((end_x, end_y))

            # Fallback: if polyline construction failed, use dummy line
            if len(polyline) < 2:
                import math
                section_length = sum(
                    p.length_m if isinstance(p, Straight)
                    else p.radius_m * abs(p.angle_deg) * math.pi / 180
                    if isinstance(p, Bend)
                    else 0
                    for p in primitives
                )
                polyline = [(0.0, 0.0), (section_length, 0.0)]

            # Primitives are now always in geographical order (optimizer fix applied)
            # Polyline constructed from them is therefore also in geographical order
            # No reversal needed

            # Create a new Section with optimized primitives
            section = Section(
                id=opt_section.section_id,
                original_polyline=polyline,
                start_junction=opt_section.start_junction,
                end_junction=opt_section.end_junction,
            )
            section.primitives = primitives

            # Store optimizer's pre-calculated values to avoid incorrect recalculation
            # These values are correct for the chosen pulling direction (F or R)
            setattr(section, "_optimizer_max_tension", opt_section.max_tension)
            setattr(section, "_optimizer_max_sidewall", opt_section.max_sidewall_pressure)
            setattr(section, "_optimizer_forward_tension", opt_section.forward_tension)
            setattr(section, "_optimizer_reverse_tension", opt_section.reverse_tension)
            setattr(section, "_optimizer_forward_sidewall", opt_section.forward_sidewall)
            setattr(section, "_optimizer_reverse_sidewall", opt_section.reverse_sidewall)
            setattr(section, "_optimizer_passes_tension", opt_section.passes_tension)
            setattr(section, "_optimizer_passes_sidewall", opt_section.passes_sidewall)

            new_route.sections.append(section)

        # Reorder sections using junction labels to ensure correct geographical order
        # Junction 'A' is at northernmost point, so this puts northernmost section first
        new_route.sections = self._reorder_sections_by_connection(new_route.sections)

        # Rename sections to match sequential order (SECT_01_01 = A→B, SECT_01_02 = B→C, etc.)
        new_route.sections = self._rename_sections_sequentially(new_route.sections)

        return new_route

    def _reorder_sections_by_connection(self, sections: List[Section]) -> List[Section]:
        """Reorder sections using junction labels to form a continuous chain.

        Uses junction labels (A, B, C, etc.) to determine connectivity.
        Finds the section starting with 'A' (northernmost point) and follows
        the junction chain to build the correct sequence.

        Args:
            sections: List of sections in arbitrary order

        Returns:
            List of sections in connection order (starting from junction 'A')
        """
        if len(sections) <= 1:
            return sections

        # Check if all sections have junction labels
        if not all(hasattr(s, 'start_junction') and hasattr(s, 'end_junction') for s in sections):
            print("[WARNING] Not all sections have junction labels, using original order")
            return sections

        # Build a map of junctions: junction -> [(section, is_start), ...]
        junction_map = {}
        for section in sections:
            if section.start_junction:
                if section.start_junction not in junction_map:
                    junction_map[section.start_junction] = []
                junction_map[section.start_junction].append((section, True))  # True = start

            if section.end_junction:
                if section.end_junction not in junction_map:
                    junction_map[section.end_junction] = []
                junction_map[section.end_junction].append((section, False))  # False = end

        # Find the section that starts with 'A' (northernmost point)
        start_section = None
        for section in sections:
            if section.start_junction == 'A':
                start_section = section
                break

        if start_section is None:
            # Try finding any section with 'A' at its end (might be reversed)
            for section in sections:
                if section.end_junction == 'A':
                    start_section = section
                    break

        if start_section is None:
            print("[WARNING] Could not find section starting with junction 'A', using original order")
            return sections

        # Build ordered chain by following junction connections
        ordered = [start_section]
        remaining = [s for s in sections if s != start_section]

        while remaining:
            current = ordered[-1]
            next_section = None

            # Look for a section whose start_junction matches our current end_junction
            if current.end_junction:
                for candidate in remaining:
                    if candidate.start_junction == current.end_junction:
                        next_section = candidate
                        break

            if next_section:
                ordered.append(next_section)
                remaining.remove(next_section)
            else:
                # No direct connection - might be a branching point or gap
                break

        if len(ordered) != len(sections):
            print(f"[WARNING] Junction-based ordering incomplete: {len(ordered)}/{len(sections)} sections connected")
            # Append unconnected sections at the end
            for section in sections:
                if section not in ordered:
                    ordered.append(section)

        return ordered

    def _rename_sections_sequentially(self, sections: List[Section]) -> List[Section]:
        """Rename sections to match sequential order after reordering.

        After sections are reordered by connection, their IDs may not match
        the sequential order. This method renumbers them so SECT_01_01 is always
        the first section (A→B), SECT_01_02 is second (B→C), etc.

        Args:
            sections: List of sections in correct connection order

        Returns:
            List of sections with sequential IDs
        """
        if not sections:
            return sections

        # Track original section numbers to detect subsections
        # E.g., SECT_07_01 and SECT_07_02 are subsections of section 07
        section_groups = {}
        for section in sections:
            # Extract base section number (e.g., "07" from "SECT_07_01")
            parts = section.id.split("_")
            if len(parts) >= 2:
                base_num = parts[1]  # "01", "02", "07", etc.
                if base_num not in section_groups:
                    section_groups[base_num] = []
                section_groups[base_num].append(section)

        # Renumber sections sequentially
        new_section_num = 1
        for section in sections:
            parts = section.id.split("_")

            if len(parts) == 2:
                # Simple section like "SECT_02"
                new_id = f"SECT_{new_section_num:02d}"
            elif len(parts) >= 3:
                # Subsection like "SECT_07_01"
                subsection_num = parts[2]  # "01", "02", etc.
                new_id = f"SECT_{new_section_num:02d}_{subsection_num}"
            else:
                # Unknown format, keep original
                new_id = section.id

            # Check if next section is from a different group
            # If so, increment section number
            current_idx = sections.index(section)
            if current_idx < len(sections) - 1:
                next_section = sections[current_idx + 1]
                current_base = parts[1] if len(parts) >= 2 else ""
                next_parts = next_section.id.split("_")
                next_base = next_parts[1] if len(next_parts) >= 2 else ""

                # If base section numbers differ, this is the last subsection
                if current_base != next_base:
                    new_section_num += 1

            section.id = new_id

        return sections

    def _get_duct_radius(self, duct_type: str) -> float:
        """Get bend radius for duct type."""
        from ..inventory.duct_inventory import DUCT_SPECIFICATIONS

        if duct_type in DUCT_SPECIFICATIONS:
            spec = DUCT_SPECIFICATIONS[duct_type]
            return spec.bends[0].radius_m if spec.bends else 3.9
        return 3.9

    def _create_cable_spec(self) -> CableSpec:
        """Create cable specification from config."""
        from ..core.models import CableArrangement, PullingMethod
        
        # Map string arrangement to enum
        arrangement_map = {
            "single": CableArrangement.SINGLE,
            "trefoil": CableArrangement.TREFOIL,
            "flat": CableArrangement.FLAT,
        }
        
        arrangement = arrangement_map.get(
            self.config.cable_arrangement.lower(), 
            CableArrangement.SINGLE
        )
        
        # Validate number of cables for arrangement
        if arrangement == CableArrangement.SINGLE and self.config.number_of_cables != 1:
            print(f"Warning: Single arrangement requires 1 cable, got {self.config.number_of_cables}. Setting to 1.")
            self.config.number_of_cables = 1
        elif arrangement == CableArrangement.TREFOIL and self.config.number_of_cables != 3:
            print(f"Note: Trefoil arrangement typically uses 3 cables. Setting to 3.")
            self.config.number_of_cables = 3
        elif arrangement == CableArrangement.FLAT and self.config.number_of_cables < 2:
            print(f"Warning: Flat arrangement requires at least 2 cables. Setting to 2.")
            self.config.number_of_cables = 2
        
        return CableSpec(
            diameter=self.config.cable_diameter_mm,  # Individual cable diameter
            weight_per_meter=self.config.cable_weight_kg_m,  # Per cable weight
            max_tension=self.config.cable_max_tension_n,
            max_sidewall_pressure=self.config.cable_max_sidewall_pressure_n_m,
            min_bend_radius=self.config.cable_min_bend_radius_mm,
            arrangement=arrangement,
            number_of_cables=self.config.number_of_cables,
            pulling_method=PullingMethod.EYE,  # Default
        )

    def _create_duct_spec(self) -> DuctSpec:
        """Create duct specification from config."""
        from ..inventory.duct_inventory import DUCT_SPECIFICATIONS

        # Calculate inner diameter from outer diameter
        # Standard wall thickness assumptions for HDPE SDR 11:
        # - 200mm OD -> ~180mm ID (wall ~10mm)
        # - 225mm OD -> ~215mm ID (wall ~5mm, as specified by user)
        inner_diameter_map = {
            "200mm": 180.0,
            "225mm": 215.0,  # User-specified conservative value
        }

        inner_diameter = inner_diameter_map.get(self.config.duct_type, 200.0)

        # Use friction override if provided, otherwise use defaults
        if self.config.friction_override is not None:
            # friction_override represents the lubricated friction value
            friction_lub = self.config.friction_override if isinstance(self.config.friction_override, float) else self.config.friction_override[0]
            friction_dry = friction_lub * 1.5  # Dry friction typically 50% higher than lubricated
            return DuctSpec(
                inner_diameter=inner_diameter,
                type="HDPE",
                friction_dry=friction_dry,
                friction_lubricated=friction_lub,
            )
        else:
            return DuctSpec(
                inner_diameter=inner_diameter,
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
            # Use new individual section plotting method
            fig, ax = self.visualizer.plot_individual_section(
                section,
                section_index=i,
                title=f"{section.id} Detail",
                show_fitted_geometry=False,  # Only show route geometry
            )
            fig.savefig(
                sections_path / f"{section.id}.png", dpi=100, bbox_inches="tight"
            )
            plt.close(fig)

    def _calculate_pulling_forces(self, route: Route) -> List[SectionResult]:
        """Calculate pulling forces for all sections.

        Forward tensions are calculated left-to-right (0 → route end).
        Reverse tensions are calculated right-to-left (route end → 0).
        """
        from ..calculations.tension import analyze_section_tension

        # PASS 1: Calculate forward tensions (left to right)
        # IMPORTANT: Optimizer-created subsections are INDEPENDENT pulls
        # Each subsection should start from tension = 0
        forward_results = []

        # Track which original section we're in
        previous_original_section = None

        for section in route.sections:
            # Determine if this is a new original section
            # Section names are either "SECT_XX" or "SECT_XX_YY"
            section_name = getattr(section, 'name', str(section))

            # Extract original section ID (the "XX" part)
            if '_' in section_name:
                parts = section_name.split('_')
                if len(parts) >= 2:
                    original_section_id = parts[1]  # "SECT_XX_YY" -> "XX"
                else:
                    original_section_id = section_name
            else:
                original_section_id = section_name

            # Reset cumulative tension at each new original section
            # Subsections within same original section also start from 0
            # (Each subsection is an independent pull)
            cumulative_forward = 0.0

            # Check if this section has optimizer-calculated values
            # If so, use those instead of recalculating (optimizer's values are correct for F/R direction)
            optimizer_forward_tension = getattr(section, '_optimizer_forward_tension', None)

            if optimizer_forward_tension is not None:
                # Use optimizer's pre-calculated forward tension value
                forward_tension = optimizer_forward_tension
                cumulative_forward = forward_tension

                # Still need tension_analysis for primitive-level data in CSV
                # But note: these values will be wrong for reversed sections
                # We'll fix the CSV generation to use optimizer values instead
                friction = self.config.friction_override if isinstance(self.config.friction_override, float) else self.duct_spec.friction_dry
                is_lubricated = friction < 0.4
                tension_analysis = analyze_section_tension(
                    section, self.cable_spec, self.duct_spec, lubricated=is_lubricated,
                    initial_tension_n=self.config.initial_tension_n, config=self.calc_config
                )
            else:
                # No optimizer values - calculate normally
                friction = self.config.friction_override if isinstance(self.config.friction_override, float) else self.duct_spec.friction_dry
                is_lubricated = friction < 0.4

                forward_tension = self.tension_calc.calculate_forward_tension(
                    section, self.cable_spec, self.duct_spec, lubricated=is_lubricated
                )
                cumulative_forward = forward_tension
                tension_analysis = analyze_section_tension(
                    section, self.cable_spec, self.duct_spec, lubricated=is_lubricated,
                    initial_tension_n=self.config.initial_tension_n, config=self.calc_config
                )

            forward_results.append({
                "section": section,
                "forward_tension": forward_tension,
                "cumulative_forward": cumulative_forward,
                "tension_analysis": tension_analysis,
            })

            previous_original_section = original_section_id

        # PASS 2: Calculate reverse tensions (right to left)
        # IMPORTANT: Same as forward - each subsection is an independent pull from tension = 0
        reverse_results = [{} for _ in forward_results]  # Placeholder

        for idx in range(len(route.sections) - 1, -1, -1):
            section = route.sections[idx]

            # Each subsection is independent - starts from 0
            cumulative_reverse = 0.0

            # Check if this section has optimizer-calculated values
            optimizer_reverse_tension = getattr(section, '_optimizer_reverse_tension', None)

            if optimizer_reverse_tension is not None:
                # Use optimizer's pre-calculated reverse tension value
                reverse_tension = optimizer_reverse_tension
                cumulative_reverse = reverse_tension
            else:
                # No optimizer values - calculate normally
                friction = self.config.friction_override if isinstance(self.config.friction_override, float) else self.duct_spec.friction_dry
                is_lubricated = friction < 0.4

                # For reverse pulling, calculate tension as if pulling from end backwards
                reverse_tension = self.tension_calc.calculate_reverse_tension(
                    section, self.cable_spec, self.duct_spec, lubricated=is_lubricated
                )
                cumulative_reverse = reverse_tension

            reverse_results[idx] = {
                "reverse_tension": reverse_tension,
                "cumulative_reverse": cumulative_reverse,
            }

        # PASS 3: Combine results
        results = []

        for idx, section in enumerate(route.sections):
            forward_data = forward_results[idx]
            reverse_data = reverse_results[idx]

            forward_tension = forward_data["forward_tension"]
            reverse_tension = reverse_data["reverse_tension"]
            cumulative_forward = forward_data["cumulative_forward"]
            cumulative_reverse = reverse_data["cumulative_reverse"]
            tension_analysis = forward_data["tension_analysis"]

            # Check if optimizer calculated sidewall pressures
            optimizer_forward_sidewall = getattr(section, '_optimizer_forward_sidewall', None)
            optimizer_reverse_sidewall = getattr(section, '_optimizer_reverse_sidewall', None)

            if optimizer_forward_sidewall is not None and optimizer_reverse_sidewall is not None:
                # Use optimizer's pre-calculated sidewall pressures
                forward_sidewall = optimizer_forward_sidewall
                reverse_sidewall = optimizer_reverse_sidewall
                max_pressure = max(forward_sidewall, reverse_sidewall)
            else:
                # Calculate normally - sidewall pressure is geometry-dependent, not direction-dependent
                # (The max sidewall pressure at bends should be the same for forward and reverse)
                max_pressure = self.pressure_calc.calculate_max_sidewall_pressure(
                    section, self.cable_spec, self.duct_spec
                )
                forward_sidewall = max_pressure
                reverse_sidewall = max_pressure

            # Build ordered geometry array - single list of all primitives in sequence
            primitives_list = []
            section_straights = []  # Legacy - for backward compatibility
            section_bends = []  # Legacy - for backward compatibility

            # Keep track of how many straights and bends we've seen
            straight_count = 0
            bend_count = 0

            for prim_idx, primitive in enumerate(section.primitives):
                # Get tension at end of this primitive (using actual primitive index)
                forward_tension_at_prim = (
                    tension_analysis.forward_tensions[prim_idx].tension
                    if prim_idx < len(tension_analysis.forward_tensions)
                    else 0
                )

                # For reverse tension, reverse the mapping so first primitive gets highest tension
                reverse_prim_idx = len(section.primitives) - 1 - prim_idx
                reverse_tension_at_prim = (
                    tension_analysis.backward_tensions[reverse_prim_idx].tension
                    if reverse_prim_idx < len(tension_analysis.backward_tensions)
                    else 0
                )

                if hasattr(primitive, "length_m"):  # Straight
                    prim_data = {
                        "type": "straight",
                        "length_m": primitive.length_m,
                        "cumulative_forward_tension_n": forward_tension_at_prim,
                        "cumulative_reverse_tension_n": reverse_tension_at_prim,
                    }
                    primitives_list.append(prim_data)

                    # Also add to legacy straights array
                    section_straights.append({
                        "length_m": primitive.length_m,
                        "cumulative_forward_tension_n": forward_tension_at_prim,
                        "cumulative_reverse_tension_n": reverse_tension_at_prim,
                    })
                    straight_count += 1

                elif isinstance(primitive, Bend):  # Bend
                    sidewall_pressure = (
                        forward_tension_at_prim / primitive.radius_m
                        if primitive.radius_m > 0
                        else 0
                    )

                    prim_data = {
                        "type": "bend",
                        "angle_deg": primitive.angle_deg,
                        "radius_m": primitive.radius_m,
                        "cumulative_forward_tension_n": forward_tension_at_prim,
                        "cumulative_reverse_tension_n": reverse_tension_at_prim,
                        "sidewall_pressure_n_m": sidewall_pressure,
                    }
                    primitives_list.append(prim_data)

                    # Also add to legacy bends array
                    section_bends.append({
                        "angle_deg": primitive.angle_deg,
                        "radius_m": primitive.radius_m,
                        "cumulative_forward_tension_n": forward_tension_at_prim,
                        "cumulative_reverse_tension_n": reverse_tension_at_prim,
                        "sidewall_pressure_n_m": sidewall_pressure,
                    })
                    bend_count += 1

            # Legacy arrays (no need to sort - already in order)
            straights = section_straights
            bends = section_bends

            result = SectionResult(
                section_id=section.id,
                length_m=section.total_length,
                straight_count=len(straights),
                bend_count=len(bends),
                primitives=primitives_list,  # New ordered list
                straights=straights,  # Legacy
                bends=bends,  # Legacy
                forward_tension_n=forward_tension,
                reverse_tension_n=reverse_tension,
                max_sidewall_pressure_n_m=max_pressure,
                forward_sidewall_pressure_n_m=forward_sidewall,
                reverse_sidewall_pressure_n_m=reverse_sidewall,
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

            # Use the new primitives list for correct ordering
            for prim_data in result.primitives:
                prim_type = prim_data["type"]
                forward_tension = prim_data["cumulative_forward_tension_n"]
                reverse_tension = prim_data["cumulative_reverse_tension_n"]

                if prim_type == "straight":
                    writer.writerow(
                        [
                            "Straight",
                            f"{prim_data['length_m']:.1f}m",
                            "",
                            f"{forward_tension:.0f}",
                            f"{reverse_tension:.0f}",
                            "",  # No sidewall pressure for straights
                            "",  # No sidewall pressure for straights
                        ]
                    )

                elif prim_type == "bend":
                    # Calculate both forward and reverse sidewall pressures using actual tensions
                    forward_sidewall = forward_tension / prim_data["radius_m"] if prim_data["radius_m"] > 0 else 0
                    reverse_sidewall = reverse_tension / prim_data["radius_m"] if prim_data["radius_m"] > 0 else 0

                    writer.writerow(
                        [
                            "Bend",
                            f"{prim_data['angle_deg']:.1f}deg",
                            f"{prim_data['radius_m']:.1f}",
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

    def _export_coordinates_csv(self, route: Route, output_path: Path) -> None:
        """Export section start/end coordinates to CSV file."""
        filename = output_path / "section_coordinates.csv"

        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Section ID", "Start X", "Start Y", "End X", "End Y", "Length (m)"])

            # Use all sections including empty ones
            all_sections = getattr(route, "_all_sections_with_splits", route.sections)

            for section in all_sections:
                start_coord = section.start_coordinate
                end_coord = section.end_coordinate

                # Format coordinates with 2 decimal places
                if start_coord and end_coord:
                    writer.writerow([
                        section.id,
                        f"{start_coord[0]:.2f}",
                        f"{start_coord[1]:.2f}",
                        f"{end_coord[0]:.2f}",
                        f"{end_coord[1]:.2f}",
                        f"{section.total_length:.2f}"
                    ])
                else:
                    # If coordinates aren't available, leave them blank
                    writer.writerow([
                        section.id,
                        "",
                        "",
                        "",
                        "",
                        f"{section.total_length:.2f}" if hasattr(section, 'total_length') else f"{section.original_length:.2f}"
                    ])

    def _export_latex_reports(
        self, section_results: List[SectionResult], output_path: Path, route: Route
    ) -> None:
        """Export LaTeX reports for pulling calculations."""
        print("  Generating LaTeX tables...")

        # Generate pulling results LaTeX file
        self._export_pulling_results_latex(section_results, output_path, route)

        print(f"  LaTeX file written to: {output_path}")

    def _export_pulling_results_latex(
        self, section_results: List[SectionResult], output_path: Path, route: Route
    ) -> None:
        """Generate pulling_calculation_results.tex file with section-by-section analysis.

        Args:
            section_results: List of section results with pulling calculations
            output_path: Directory path for output files
            route: Route object containing sections with coordinate data
        """
        lines = []
        lines.append("\\section{Pulling Calculation Results}")
        lines.append("")
        lines.append("\\subsection{Section-by-Section Analysis}")
        lines.append("")
        lines.append("The following table presents detailed pulling analysis for each cable section, ")
        lines.append("showing starting coordinates, tension and sidewall pressure analysis for both ")
        lines.append("pulling directions, with the optimal direction highlighted.")
        lines.append("")

        # Create a mapping from section_id to section object for coordinate lookup
        section_map = {s.id: s for s in route.sections}

        # Start longtable with 8 columns (no float, can span pages)
        lines.append("\\begin{longtable}{|l|l|l|l|l|l|l|l|}")
        lines.append("\\caption{Detailed Section-by-Section Pulling Analysis} \\\\")
        lines.append("\\tablelineeight")
        lines.append("\\headercell{Section} & \\headercell{Length} & \\headercell{Dir} & "
                    "\\headercell{Tension} & \\headercell{Utilisation (\\%)} & \\headercell{Sidewall} & "
                    "\\headercell{Utilisation (\\%)} & \\headercell{Status} \\\\")
        lines.append("\\tablelineeight")
        lines.append("\\endfirsthead")
        lines.append("\\multicolumn{8}{c}{\\tablename\\ \\thetable\\ -- continued from previous page} \\\\")
        lines.append("\\tablelineeight")
        lines.append("\\headercell{Section} & \\headercell{Length} & \\headercell{Dir} & "
                    "\\headercell{Tension} & \\headercell{Utilisation (\\%)} & \\headercell{Sidewall} & "
                    "\\headercell{Utilisation (\\%)} & \\headercell{Status} \\\\")
        lines.append("\\tablelineeight")
        lines.append("\\endhead")
        lines.append("\\tablelineeight \\multicolumn{8}{r}{Continued on next page} \\\\")
        lines.append("\\endfoot")
        lines.append("\\tablelineeight")
        lines.append("\\endlastfoot")

        # Iterate through section results
        point_number = 0  # Start at 0 so A=0, B=1, etc.
        for result in section_results:
            # Get section object for coordinates
            section = section_map.get(result.section_id)

            # Convert point number to letter (A, B, C, ...)
            point_letter = chr(ord('A') + point_number)

            # Add starting point coordinate row (merged across all columns)
            if section and section.start_coordinate:
                x, y = section.start_coordinate
                # Use 3 decimal places for coordinate precision (mm if in meters)
                coord_text = f"Point {point_letter}: X {x:.3f}, Y {y:.3f}"
            else:
                coord_text = f"Point {point_letter} (coordinates unavailable)"

            lines.append(f"\\multicolumn{{8}}{{|c|}}{{{coord_text}}} \\\\")
            lines.append("\\tablelineeight")
            point_number += 1

            # Calculate utilizations
            forward_tension_util = (result.forward_tension_n / self.config.cable_max_tension_n) * 100
            reverse_tension_util = (result.reverse_tension_n / self.config.cable_max_tension_n) * 100
            forward_sidewall_util = (result.forward_sidewall_pressure_n_m / self.config.cable_max_sidewall_pressure_n_m) * 100
            reverse_sidewall_util = (result.reverse_sidewall_pressure_n_m / self.config.cable_max_sidewall_pressure_n_m) * 100

            # Determine pass/fail for each direction
            forward_passes = forward_tension_util <= 100 and forward_sidewall_util <= 100
            reverse_passes = reverse_tension_util <= 100 and reverse_sidewall_util <= 100

            # Determine best direction (lowest tension utilization among passing directions)
            if forward_passes and reverse_passes:
                # Both pass - best is the one with lower tension utilization
                best_direction = "forward" if forward_tension_util <= reverse_tension_util else "reverse"
            elif forward_passes:
                # Only forward passes
                best_direction = "forward"
            elif reverse_passes:
                # Only reverse passes
                best_direction = "reverse"
            else:
                # Neither passes - no best direction
                best_direction = None

            # Determine status for each direction
            def get_status(passes, is_best):
                if not passes:
                    return "\\cellcolor{red!25}Fail"
                elif is_best:
                    return "\\cellcolor{green!25}Best"
                else:
                    return "Pass"

            forward_status = get_status(forward_passes, best_direction == "forward")
            reverse_status = get_status(reverse_passes, best_direction == "reverse")

            # Clean section ID for LaTeX (escape underscores)
            section_id_latex = result.section_id.replace("_", "\\_")

            # Generate table rows (multirow for section and length, two direction rows)
            lines.append(f"\\multirow{{2}}{{*}}{{{section_id_latex}}} & "
                        f"\\multirow{{2}}{{*}}{{{result.length_m:.1f}}} & "
                        f"forward & "
                        f"{result.forward_tension_n:.0f} & "
                        f"{forward_tension_util:.1f} \\% & "
                        f"{result.forward_sidewall_pressure_n_m:.0f} & "
                        f"{forward_sidewall_util:.1f} \\% & "
                        f"{forward_status} \\\\\\cline{{3-8}}")

            lines.append(f" & & reverse & "
                        f"{result.reverse_tension_n:.0f} & "
                        f"{reverse_tension_util:.1f} \\% & "
                        f"{result.reverse_sidewall_pressure_n_m:.0f} & "
                        f"{reverse_sidewall_util:.1f} \\% & "
                        f"{reverse_status} \\\\")
            lines.append("\\tablelineeight")

        # Add final endpoint coordinate row
        if section_results:
            last_section = section_map.get(section_results[-1].section_id)
            # point_number is now one more than the last section start, so it's the endpoint
            endpoint_letter = chr(ord('A') + point_number)
            if last_section and last_section.end_coordinate:
                x, y = last_section.end_coordinate
                coord_text = f"Point {endpoint_letter}: X {x:.3f}, Y {y:.3f}"
            else:
                coord_text = f"Point {endpoint_letter} (coordinates unavailable)"

            lines.append(f"\\multicolumn{{8}}{{|c|}}{{{coord_text}}} \\\\")
            lines.append("\\tablelineeight")

        lines.append("\\end{longtable}")
        lines.append("")

        # Add summary statistics
        total_sections = len(section_results)
        sections_with_fails = sum(
            1 for r in section_results
            if (r.forward_tension_n / self.config.cable_max_tension_n * 100 > 100 or
                r.max_sidewall_pressure_n_m / self.config.cable_max_sidewall_pressure_n_m * 100 > 100) and
               (r.reverse_tension_n / self.config.cable_max_tension_n * 100 > 100 or
                r.max_sidewall_pressure_n_m / self.config.cable_max_sidewall_pressure_n_m * 100 > 100)
        )

        lines.append("\\subsection{Summary}")
        lines.append("")
        if sections_with_fails == 0:
            lines.append(f"All {total_sections} cable sections have at least one viable pulling direction ")
            lines.append("within manufacturer limits. The recommended pulling direction for each section ")
            lines.append("is highlighted as 'Best' in the table above.")
        else:
            lines.append(f"Out of {total_sections} cable sections, {sections_with_fails} section(s) exceed ")
            lines.append("manufacturer limits in both pulling directions and require attention.")
        lines.append("")

        # Key limits
        lines.append("\\textbf{Cable Limits:}")
        lines.append("\\begin{itemize}")
        lines.append(f"    \\item Maximum Pulling Tension: {self.config.cable_max_tension_n:.0f} N ({self.config.cable_max_tension_n/1000:.1f} kN)")
        lines.append(f"    \\item Maximum Sidewall Pressure: {self.config.cable_max_sidewall_pressure_n_m:.0f} N/m ({self.config.cable_max_sidewall_pressure_n_m/1000:.1f} kN/m)")
        lines.append("\\end{itemize}")

        # Write to file
        output_file = output_path / "pulling_calculation_results.tex"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

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
    dxf_layer: Optional[str] = None,
    **kwargs,
) -> AnalysisResults:
    """
    Convenience function to analyze a cable route from DXF file.

    Args:
        dxf_path: Path to DXF file
        output_dir: Output directory for results
        config: Analysis configuration
        dxf_layer: Specific DXF layer to use (None for default)
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
    return pipeline.analyze_dxf(dxf_path, output_dir, dxf_layer=dxf_layer)
