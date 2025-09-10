"""Complete analysis pipeline for cable pulling calculations."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..calculations import (
    LimitCheckResult,
    SectionTensionAnalysis,
    analyze_route_with_varying_conditions,
    calculate_pulling_feasibility,
    find_critical_sections,
)
from ..core.models import Bend, CableSpec, DuctSpec, Route, Straight
from ..geometry import GeometryProcessor, ProcessingResult
from ..io import load_route_from_dxf


@dataclass
class PipelineResult:
    """Complete result of cable pulling analysis pipeline."""

    # Input information
    input_file: str
    cable_spec: CableSpec
    duct_specs: List[DuctSpec]

    # Processing results
    original_route: Route
    processed_route: Route
    geometry_result: ProcessingResult

    # Analysis results
    tension_analyses: List[SectionTensionAnalysis]
    limit_results: List[LimitCheckResult]
    feasibility_result: Dict[str, Any]
    critical_sections: Dict[str, List[str]]

    # Overall status
    success: bool
    errors: List[str]
    warnings: List[str]
    summary: Dict[str, Any]


class CablePullingPipeline:
    """Complete cable pulling analysis pipeline."""

    def __init__(
        self,
        geometry_processor: Optional[GeometryProcessor] = None,
        enable_splitting: bool = True,
        max_cable_length: float = 500.0,
        safety_factor: float = 1.5,
    ) -> None:
        """Initialize pipeline.

        Args:
            geometry_processor: Optional custom geometry processor
            enable_splitting: Whether to enable minor splitting
            max_cable_length: Maximum cable length for splitting
            safety_factor: Safety factor for limit checking
        """
        self.geometry_processor = geometry_processor or GeometryProcessor()
        self.enable_splitting = enable_splitting
        self.max_cable_length = max_cable_length
        self.safety_factor = safety_factor

    def _expand_per_section_parameter(
        self,
        parameter: Union[bool, float, List[bool], List[float]],
        num_sections: int,
        parameter_name: str,
    ) -> List[Union[bool, float]]:
        """Expand a parameter to per-section values.

        Args:
            parameter: Single value or list of values
            num_sections: Number of sections in route
            parameter_name: Name of parameter for error messages

        Returns:
            List with one value per section

        Raises:
            ValueError: If list length doesn't match number of sections
        """
        if isinstance(parameter, list):
            if len(parameter) != num_sections:
                raise ValueError(
                    f"{parameter_name} list length ({len(parameter)}) must match "
                    f"number of sections ({num_sections})"
                )
            return parameter
        else:
            # Single value - expand to all sections
            return [parameter] * num_sections

    def run_analysis(
        self,
        dxf_file: str,
        cable_spec: CableSpec,
        duct_spec: DuctSpec,
        lubricated: Union[bool, List[bool]] = False,
        friction_override: Optional[Union[float, List[float]]] = None,
        **kwargs: Any,
    ) -> PipelineResult:
        """Run complete cable pulling analysis pipeline.

        Args:
            dxf_file: Path to DXF input file
            cable_spec: Cable specifications
            duct_spec: Duct specifications (same for all sections)
            lubricated: Whether duct is lubricated - can be bool for all sections
                        or list of bools for per-section control (ignored if friction_override used)
            friction_override: Optional base friction override - can be float for all sections
                              or list of floats for per-section control. When specified,
                              uses as base friction (ignores lubrication) but still applies
                              cable arrangement multipliers (trefoil 1.3x, flat 1.1x)
            **kwargs: Additional parameters for advanced analysis

        Returns:
            Complete pipeline results
        """
        errors = []
        warnings = []

        try:
            # Step 1: Load DXF file
            original_route = load_route_from_dxf(Path(dxf_file))

            # Step 2: Process geometry (including splitting if enabled)
            geometry_result = self.geometry_processor.process_route(
                original_route,
                cable_spec=cable_spec,
                duct_spec=duct_spec,
                enable_splitting=self.enable_splitting,
                max_cable_length=self.max_cable_length,
            )

            if not geometry_result.success:
                errors.append(f"Geometry processing failed: {geometry_result.message}")
                if geometry_result.validation_result:
                    warnings.extend(
                        [
                            f"Issue: {issue.message}"
                            for issue in geometry_result.validation_result.issues
                            if issue.severity == "warning"
                        ]
                    )

            processed_route = geometry_result.route

            # Step 3: Run cable pulling calculations
            # Expand per-section parameters
            num_sections = len(processed_route.sections)
            lubricated_sections = self._expand_per_section_parameter(
                lubricated, num_sections, "lubricated"
            )

            # Create duct specs list with optional friction overrides
            if friction_override is not None:
                friction_overrides = self._expand_per_section_parameter(
                    friction_override, num_sections, "friction_override"
                )
                # Create modified duct specs with overridden friction values
                duct_specs = []
                for i, override_friction in enumerate(friction_overrides):
                    # When friction is overridden, use the base value specified by engineer
                    # but still apply cable arrangement multipliers (trefoil 1.3x, flat 1.1x)
                    from dataclasses import replace

                    modified_spec = replace(
                        duct_spec,
                        friction_dry=override_friction,
                        friction_lubricated=override_friction,  # Same base value - override disregards lubrication
                    )
                    # Note: Cable arrangement multipliers are still applied via DuctSpec.get_friction()
                    duct_specs.append(modified_spec)
            else:
                # Use same spec for all sections (normal lubrication logic applies)
                duct_specs = [duct_spec] * num_sections

            # Perform advanced analysis
            route_analysis = analyze_route_with_varying_conditions(
                processed_route,
                cable_spec,
                duct_specs,
                lubricated_sections,
                **kwargs,
            )

            # Extract results
            tension_analyses = [analysis[0] for analysis in route_analysis]
            limit_results = [analysis[1] for analysis in route_analysis]

            # Step 4: Calculate overall feasibility
            feasibility_result = calculate_pulling_feasibility(
                processed_route,
                cable_spec,
                duct_specs,
                lubricated_sections,
                self.safety_factor,
                **kwargs,
            )

            # Step 5: Identify critical sections
            critical_sections = find_critical_sections(route_analysis)

            # Step 6: Generate summary
            summary = self._generate_summary(
                original_route,
                processed_route,
                geometry_result,
                feasibility_result,
                critical_sections,
            )

            # Check overall success
            overall_success = (
                geometry_result.success
                and feasibility_result["feasible"]
                and len(errors) == 0
            )

            return PipelineResult(
                input_file=dxf_file,
                cable_spec=cable_spec,
                duct_specs=duct_specs,
                original_route=original_route,
                processed_route=processed_route,
                geometry_result=geometry_result,
                tension_analyses=tension_analyses,
                limit_results=limit_results,
                feasibility_result=feasibility_result,
                critical_sections=critical_sections,
                success=overall_success,
                errors=errors,
                warnings=warnings,
                summary=summary,
            )

        except Exception as e:
            errors.append(f"Pipeline failed: {str(e)}")

            # Return minimal result on failure
            return PipelineResult(
                input_file=dxf_file,
                cable_spec=cable_spec,
                duct_specs=[duct_spec],
                original_route=Route(name="failed", sections=[]),
                processed_route=Route(name="failed", sections=[]),
                geometry_result=ProcessingResult(
                    route=Route(name="failed", sections=[]),
                    fitting_results=[],
                    validation_result=None,  # type: ignore
                    splitting_result=None,
                    success=False,
                    message="Pipeline failed",
                ),
                tension_analyses=[],
                limit_results=[],
                feasibility_result={"feasible": False},
                critical_sections={},
                success=False,
                errors=errors,
                warnings=warnings,
                summary={"status": "failed"},
            )

    def _generate_summary(
        self,
        original_route: Route,
        processed_route: Route,
        geometry_result: ProcessingResult,
        feasibility_result: Dict[str, Any],
        critical_sections: Dict[str, List[str]],
    ) -> Dict[str, Any]:
        """Generate analysis summary."""
        # Calculate section statistics
        section_lengths = [
            section.original_length for section in processed_route.sections
        ]
        total_primitives = sum(
            len(section.primitives) for section in processed_route.sections
        )

        # Count primitive types
        straight_count = 0
        bend_count = 0
        curve_count = 0

        for section in processed_route.sections:
            for primitive in section.primitives:
                if hasattr(primitive, "length_m"):  # Straight
                    straight_count += 1
                elif hasattr(primitive, "radius_m"):  # Bend
                    bend_count += 1
                else:  # Curve or other
                    curve_count += 1

        # Splitting information
        splitting_info = {}
        if geometry_result.splitting_result:
            splitting_info = {
                "sections_added": geometry_result.splitting_result.sections_created,
                "split_points": len(geometry_result.splitting_result.split_points),
                "original_sections": len(original_route.sections),
                "final_sections": len(processed_route.sections),
            }

        return {
            "route_name": processed_route.name,
            "total_length_m": processed_route.total_length,
            "section_count": processed_route.section_count,
            "section_statistics": {
                "min_length_m": min(section_lengths) if section_lengths else 0,
                "max_length_m": max(section_lengths) if section_lengths else 0,
                "avg_length_m": (
                    sum(section_lengths) / len(section_lengths)
                    if section_lengths
                    else 0
                ),
            },
            "geometry_summary": {
                "total_primitives": total_primitives,
                "straight_sections": straight_count,
                "bend_sections": bend_count,
                "curve_sections": curve_count,
            },
            "splitting_summary": splitting_info,
            "feasibility": {
                "overall_feasible": feasibility_result["feasible"],
                "max_tension_n": feasibility_result["max_tension_n"],
                "max_pressure_n_per_m": feasibility_result["max_pressure_n_per_m"],
                "safety_factor": feasibility_result["safety_factor_applied"],
            },
            "critical_sections": {
                "tension_limited": len(critical_sections.get("tension", [])),
                "pressure_limited": len(critical_sections.get("pressure", [])),
                "bend_radius_limited": len(critical_sections.get("bend_radius", [])),
            },
            "recommendations": feasibility_result["section_recommendations"],
        }

    def run_geometry_only(
        self,
        dxf_file: str,
        duct_spec: Optional[DuctSpec] = None,
        cable_spec: Optional[CableSpec] = None,
    ) -> ProcessingResult:
        """Run geometry processing only (no calculations).

        Args:
            dxf_file: Path to DXF input file
            duct_spec: Optional duct specification for bend classification
            cable_spec: Optional cable specification for validation

        Returns:
            Geometry processing results
        """
        # Load DXF file
        route = load_route_from_dxf(Path(dxf_file))

        # Process geometry
        return self.geometry_processor.process_route(
            route,
            cable_spec=cable_spec,
            duct_spec=duct_spec,
            enable_splitting=self.enable_splitting,
            max_cable_length=self.max_cable_length,
        )


class AnalysisReporter:
    """Generate reports from pipeline results."""

    @staticmethod
    def generate_text_report(result: PipelineResult) -> str:
        """Generate human-readable text report."""
        lines = []

        # Header
        lines.append("=" * 60)
        lines.append("CABLE PULLING ANALYSIS REPORT")
        lines.append("=" * 60)
        lines.append(f"Route: {result.summary['route_name']}")
        lines.append(f"Input: {result.input_file}")
        lines.append(
            f"Cable: {result.cable_spec.diameter}mm, {result.cable_spec.arrangement.value}"
        )
        lines.append(
            f"Duct: {result.duct_specs[0].inner_diameter}mm {result.duct_specs[0].type}"
        )
        lines.append("")

        # Overall Status
        status = "✓ FEASIBLE" if result.success else "✗ NOT FEASIBLE"
        lines.append(f"OVERALL STATUS: {status}")
        lines.append("")

        # Route Summary
        lines.append("ROUTE SUMMARY:")
        lines.append(f"  Total Length: {result.summary['total_length_m']:.1f}m")
        lines.append(f"  Sections: {result.summary['section_count']}")
        lines.append(
            f"  Primitives: {result.summary['geometry_summary']['total_primitives']}"
        )
        lines.append(
            f"    - Straights: {result.summary['geometry_summary']['straight_sections']}"
        )
        lines.append(
            f"    - Bends: {result.summary['geometry_summary']['bend_sections']}"
        )
        lines.append(
            f"    - Curves: {result.summary['geometry_summary']['curve_sections']}"
        )
        lines.append("")

        # Splitting Results
        if result.geometry_result.splitting_result:
            split_info = result.summary["splitting_summary"]
            lines.append("SPLITTING RESULTS:")
            lines.append(f"  Original sections: {split_info['original_sections']}")
            lines.append(f"  Final sections: {split_info['final_sections']}")
            lines.append(f"  Sections added: {split_info['sections_added']}")
            lines.append(f"  Split points: {split_info['split_points']}")
            lines.append("")

        # Feasibility Analysis
        feasibility = result.summary["feasibility"]
        lines.append("FEASIBILITY ANALYSIS:")
        lines.append(f"  Overall feasible: {feasibility['overall_feasible']}")
        lines.append(f"  Max tension: {feasibility['max_tension_n']:.0f} N")
        lines.append(f"  Max pressure: {feasibility['max_pressure_n_per_m']:.0f} N/m")
        lines.append(f"  Safety factor: {feasibility['safety_factor']:.1f}")
        lines.append("")

        # Critical Sections
        critical = result.summary["critical_sections"]
        if any(critical.values()):
            lines.append("CRITICAL SECTIONS:")
            if critical["tension_limited"]:
                lines.append(
                    f"  Tension limited: {critical['tension_limited']} sections"
                )
            if critical["pressure_limited"]:
                lines.append(
                    f"  Pressure limited: {critical['pressure_limited']} sections"
                )
            if critical["bend_radius_limited"]:
                lines.append(
                    f"  Bend radius limited: {critical['bend_radius_limited']} sections"
                )
            lines.append("")

        # Section Details
        lines.append("SECTION DETAILS:")
        for i, (tension_analysis, limit_result) in enumerate(
            zip(result.tension_analyses, result.limit_results)
        ):
            section = result.processed_route.sections[i]
            lines.append(f"  {section.id}:")
            lines.append(f"    Length: {section.original_length:.1f}m")
            lines.append(f"    Max tension: {limit_result.max_tension:.0f} N")
            lines.append(
                f"    Recommended direction: {limit_result.recommended_direction}"
            )

            if limit_result.limiting_factors:
                lines.append(
                    f"    Limiting factors: {', '.join(limit_result.limiting_factors)}"
                )

            if limit_result.max_pressure > 0:
                lines.append(f"    Max pressure: {limit_result.max_pressure:.0f} N/m")

        # Errors and Warnings
        if result.errors:
            lines.append("")
            lines.append("ERRORS:")
            for error in result.errors:
                lines.append(f"  ✗ {error}")

        if result.warnings:
            lines.append("")
            lines.append("WARNINGS:")
            for warning in result.warnings:
                lines.append(f"  ⚠ {warning}")

        return "\n".join(lines)

    @staticmethod
    def generate_csv_report(result: PipelineResult) -> str:
        """Generate CSV report for section-by-section results."""
        lines = []

        # Header
        lines.append(
            "section_id,length_m,max_tension_n,max_pressure_n_per_m,recommended_direction,passes_all_limits,limiting_factors"
        )

        # Data rows
        for i, (tension_analysis, limit_result) in enumerate(
            zip(result.tension_analyses, result.limit_results)
        ):
            section = result.processed_route.sections[i]

            passes_all = (
                limit_result.passes_tension_limit
                and limit_result.passes_pressure_limit
                and limit_result.passes_bend_radius_limit
            )

            limiting_factors = (
                "|".join(limit_result.limiting_factors)
                if limit_result.limiting_factors
                else "none"
            )

            lines.append(
                f"{section.id},"
                f"{section.original_length:.1f},"
                f"{limit_result.max_tension:.0f},"
                f"{limit_result.max_pressure:.0f},"
                f"{limit_result.recommended_direction},"
                f"{passes_all},"
                f"{limiting_factors}"
            )

        return "\n".join(lines)

    @staticmethod
    def generate_json_summary(result: PipelineResult) -> Dict[str, Any]:
        """Generate JSON summary of analysis results."""
        return {
            "meta": {
                "input_file": result.input_file,
                "analysis_timestamp": "2025-09-02",  # TODO: Use actual timestamp
                "cable_spec": {
                    "diameter_mm": result.cable_spec.diameter,
                    "weight_kg_per_m": result.cable_spec.weight_per_meter,
                    "max_tension_n": result.cable_spec.max_tension,
                    "arrangement": result.cable_spec.arrangement.value,
                    "number_of_cables": result.cable_spec.number_of_cables,
                },
                "duct_spec": {
                    "inner_diameter_mm": result.duct_specs[0].inner_diameter,
                    "type": result.duct_specs[0].type,
                    "friction_dry": result.duct_specs[0].friction_dry,
                },
            },
            "results": {
                "overall_feasible": result.success,
                "total_length_m": result.summary["total_length_m"],
                "max_tension_n": result.feasibility_result["max_tension_n"],
                "max_pressure_n_per_m": result.feasibility_result[
                    "max_pressure_n_per_m"
                ],
                "critical_sections": result.critical_sections,
            },
            "sections": [
                {
                    "id": section.id,
                    "length_m": section.original_length,
                    "primitives": len(section.primitives),
                    "max_tension_n": limit_result.max_tension,
                    "max_pressure_n_per_m": limit_result.max_pressure,
                    "recommended_direction": limit_result.recommended_direction,
                    "passes_limits": (
                        limit_result.passes_tension_limit
                        and limit_result.passes_pressure_limit
                        and limit_result.passes_bend_radius_limit
                    ),
                    "limiting_factors": limit_result.limiting_factors,
                }
                for section, limit_result in zip(
                    result.processed_route.sections, result.limit_results
                )
            ],
            "summary": result.summary,
        }
