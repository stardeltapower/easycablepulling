"""Advanced cable pulling calculations including slope corrections and multi-cable scenarios."""

import math
from typing import Any, Dict, List, Optional, Tuple

from ..core.models import CableArrangement, CableSpec, DuctSpec, Route, Section
from .pressure import LimitCheckResult, check_section_limits
from .tension import SectionTensionAnalysis, analyze_section_tension


def calculate_slope_correction_factor(
    elevation_profile: List[Tuple[float, float]], total_length: float
) -> float:
    """Calculate average slope correction factor for a section.

    Args:
        elevation_profile: List of (distance, elevation) points along section
        total_length: Total length of section in meters

    Returns:
        Average slope factor (dimensionless, positive = uphill)
    """
    if len(elevation_profile) < 2:
        return 0.0  # No slope if insufficient data

    # Calculate total elevation change
    start_elevation = elevation_profile[0][1]
    end_elevation = elevation_profile[-1][1]
    elevation_change = end_elevation - start_elevation

    # Calculate average slope angle
    slope_angle = math.atan(elevation_change / total_length)

    return math.sin(slope_angle)


def calculate_multi_cable_weight_factor(cable_spec: CableSpec) -> float:
    """Calculate weight correction factor for multi-cable arrangements.

    Args:
        cable_spec: Cable specifications

    Returns:
        Weight factor (1.0 for single cable, higher for multi-cable)
    """
    if cable_spec.arrangement == CableArrangement.SINGLE:
        return 1.0
    elif cable_spec.arrangement == CableArrangement.TREFOIL:
        # Trefoil arrangement has slightly higher effective weight due to geometry
        return 1.05
    else:  # FLAT arrangement
        # Flat arrangement has higher friction and weight distribution
        return 1.1


def calculate_temperature_friction_factor(
    base_temperature: float = 20.0,
    actual_temperature: float = 20.0,
    duct_type: str = "PVC",
) -> float:
    """Calculate friction factor adjustment for temperature.

    Args:
        base_temperature: Reference temperature in Celsius
        actual_temperature: Actual temperature in Celsius
        duct_type: Type of duct material

    Returns:
        Temperature correction factor for friction
    """
    temp_diff = actual_temperature - base_temperature

    # Temperature coefficients by duct material
    temp_coefficients = {
        "PVC": 0.001,  # Moderate temperature sensitivity
        "HDPE": 0.0015,  # Higher temperature sensitivity
        "Steel": 0.0005,  # Lower temperature sensitivity
        "Concrete": 0.0008,  # Moderate temperature sensitivity
    }

    coeff = temp_coefficients.get(duct_type, 0.001)

    # Higher temperature typically reduces friction (cables expand, lubrication improved)
    factor = 1.0 - (coeff * temp_diff)

    # Limit factor to reasonable range
    return max(0.5, min(1.5, factor))


def calculate_jam_ratio_iec(cable_spec: CableSpec, duct_spec: DuctSpec) -> float:
    """Calculate jam ratio according to IEC standards.

    IEC 61936-1 provides guidance on cable spacing and jamming in ducts.

    Args:
        cable_spec: Cable specifications
        duct_spec: Duct specifications

    Returns:
        Jam ratio (0.0 = no jamming risk, 1.0+ = high jamming risk)
    """
    if cable_spec.number_of_cables == 1:
        return 0.0

    cable_diameter = cable_spec.diameter  # mm
    duct_diameter = duct_spec.inner_diameter  # mm

    # IEC-based jam factor calculation
    if cable_spec.arrangement == CableArrangement.TREFOIL:
        # For trefoil: bundle factor is approximately 2.15
        bundle_factor = 2.15
        # Trefoil has lower jam risk due to stable geometry
        jam_base = 0.8
    elif cable_spec.arrangement == CableArrangement.FLAT:
        # For flat arrangement: simple linear arrangement
        bundle_factor = cable_spec.number_of_cables
        # Flat has higher jam risk
        jam_base = 1.2
    else:
        # Unknown arrangement, use conservative estimate
        bundle_factor = cable_spec.number_of_cables
        jam_base = 1.0

    # Calculate effective fill ratio
    fill_ratio = (cable_diameter * bundle_factor) / duct_diameter

    # Jam ratio increases exponentially with fill ratio
    if fill_ratio < 0.6:
        jam_ratio = 0.0  # No jamming risk
    elif fill_ratio < 0.8:
        jam_ratio = jam_base * ((fill_ratio - 0.6) / 0.2) ** 2
    else:
        jam_ratio = jam_base * (
            1.0 + (fill_ratio - 0.8) * 5
        )  # High risk above 80% fill

    return jam_ratio


def analyze_multi_cable_clearance(
    cable_spec: CableSpec, duct_spec: DuctSpec
) -> Dict[str, Any]:
    """Comprehensive multi-cable clearance analysis.

    Args:
        cable_spec: Cable specifications
        duct_spec: Duct specifications

    Returns:
        Dictionary with clearance analysis results
    """
    bundle_diameter = cable_spec.bundle_diameter
    duct_diameter = duct_spec.inner_diameter

    results: Dict[str, Any] = {
        "bundle_diameter_mm": bundle_diameter,
        "duct_diameter_mm": duct_diameter,
        "fill_ratio": bundle_diameter / duct_diameter,
        "clearance_mm": duct_diameter - bundle_diameter,
        "jam_ratio_iec": calculate_jam_ratio_iec(cable_spec, duct_spec),
        "weight_factor": calculate_multi_cable_weight_factor(cable_spec),
    }

    # Determine clearance status
    max_fill = 0.9 if cable_spec.number_of_cables == 1 else 0.8
    results["clearance_adequate"] = results["fill_ratio"] <= max_fill
    jam_ratio_val = float(results["jam_ratio_iec"])
    if jam_ratio_val < 0.3:
        results["jam_risk_level"] = "low"
    elif jam_ratio_val < 0.7:
        results["jam_risk_level"] = "medium"
    else:
        results["jam_risk_level"] = "high"

    return results


def analyze_section_with_environment(
    section: Section,
    cable_spec: CableSpec,
    duct_spec: DuctSpec,
    lubricated: bool = False,
    temperature: float = 20.0,
    elevation_profile: Optional[List[Tuple[float, float]]] = None,
) -> Tuple[SectionTensionAnalysis, Dict[str, float]]:
    """Perform advanced section analysis including environmental factors.

    Args:
        section: Section to analyze
        cable_spec: Cable specifications
        duct_spec: Duct specifications
        lubricated: Whether duct is lubricated
        temperature: Ambient temperature in Celsius
        elevation_profile: Optional elevation data as (distance, elevation) pairs

    Returns:
        Tuple of (tension_analysis, environmental_factors)
    """
    # Calculate environmental correction factors
    temp_factor = calculate_temperature_friction_factor(
        20.0, temperature, duct_spec.type
    )
    slope_factor = 0.0

    if elevation_profile:
        slope_factor = calculate_slope_correction_factor(
            elevation_profile, section.total_length
        )

    weight_factor = calculate_multi_cable_weight_factor(cable_spec)

    # Create modified duct spec with environmental corrections
    adjusted_friction_dry = duct_spec.friction_dry * temp_factor
    adjusted_friction_lubricated = duct_spec.friction_lubricated * temp_factor

    adjusted_duct = DuctSpec(
        inner_diameter=duct_spec.inner_diameter,
        type=duct_spec.type,
        friction_dry=adjusted_friction_dry,
        friction_lubricated=adjusted_friction_lubricated,
        bend_options=duct_spec.bend_options,
    )

    # Create modified cable spec with weight corrections
    adjusted_weight = cable_spec.weight_per_meter * weight_factor
    adjusted_cable = CableSpec(
        diameter=cable_spec.diameter,
        weight_per_meter=adjusted_weight,
        max_tension=cable_spec.max_tension,
        max_sidewall_pressure=cable_spec.max_sidewall_pressure,
        min_bend_radius=cable_spec.min_bend_radius,
        pulling_method=cable_spec.pulling_method,
        arrangement=cable_spec.arrangement,
        number_of_cables=cable_spec.number_of_cables,
    )

    # Perform analysis with adjusted parameters
    tension_analysis = analyze_section_tension(
        section, adjusted_cable, adjusted_duct, lubricated
    )

    environmental_factors = {
        "temperature_factor": temp_factor,
        "slope_factor": slope_factor,
        "weight_factor": weight_factor,
        "adjusted_friction_dry": adjusted_friction_dry,
        "adjusted_friction_lubricated": adjusted_friction_lubricated,
        "adjusted_weight_per_meter": adjusted_weight,
    }

    return tension_analysis, environmental_factors


def analyze_route_with_varying_conditions(
    route: Route,
    cable_spec: CableSpec,
    duct_specs: List[DuctSpec],  # One per section
    lubricated_sections: List[bool],  # One per section
    temperatures: Optional[List[float]] = None,  # One per section
    elevation_profiles: Optional[
        List[Optional[List[Tuple[float, float]]]]
    ] = None,  # One per section
) -> List[Tuple[SectionTensionAnalysis, LimitCheckResult, Dict[str, float]]]:
    """Analyze route with varying conditions across sections.

    Args:
        route: Route to analyze
        cable_spec: Cable specifications (same for all sections)
        duct_specs: Duct specifications for each section
        lubricated_sections: Lubrication status for each section
        temperatures: Optional temperatures for each section
        elevation_profiles: Optional elevation data for each section

    Returns:
        List of (tension_analysis, limit_result, environmental_factors) for each section
    """
    if len(duct_specs) != len(route.sections):
        raise ValueError("Must provide duct spec for each section")
    if len(lubricated_sections) != len(route.sections):
        raise ValueError("Must provide lubrication status for each section")

    # Use default values if not provided
    if temperatures is None:
        temperatures = [20.0] * len(route.sections)
    if elevation_profiles is None:
        elevation_profiles = [None] * len(route.sections)

    if len(temperatures) != len(route.sections):
        raise ValueError("Must provide temperature for each section")
    if len(elevation_profiles) != len(route.sections):
        raise ValueError("Must provide elevation profile for each section")

    results = []

    for i, section in enumerate(route.sections):
        # Analyze section with environmental factors
        tension_analysis, env_factors = analyze_section_with_environment(
            section,
            cable_spec,
            duct_specs[i],
            lubricated_sections[i],
            temperatures[i],
            elevation_profiles[i] if elevation_profiles else None,
        )

        # Check limits with environmental adjustments
        limit_result = check_section_limits(
            section,
            cable_spec,  # Use original cable spec for limit checking
            duct_specs[i],
            tension_analysis.forward_tensions,
            tension_analysis.backward_tensions,
            lubricated_sections[i],
        )

        results.append((tension_analysis, limit_result, env_factors))

    return results


def find_critical_sections(
    route_analysis: List[
        Tuple[SectionTensionAnalysis, LimitCheckResult, Dict[str, float]]
    ]
) -> Dict[str, List[str]]:
    """Identify critical sections that limit the cable pull.

    Args:
        route_analysis: Results from analyze_route_with_varying_conditions

    Returns:
        Dictionary mapping limit types to lists of critical section IDs
    """
    critical_sections: Dict[str, List[str]] = {
        "tension": [],
        "pressure": [],
        "bend_radius": [],
        "clearance": [],
    }

    for tension_analysis, limit_result, _ in route_analysis:
        section_id = tension_analysis.section_id

        if not limit_result.passes_tension_limit:
            critical_sections["tension"].append(section_id)

        if not limit_result.passes_pressure_limit:
            critical_sections["pressure"].append(section_id)

        if not limit_result.passes_bend_radius_limit:
            critical_sections["bend_radius"].append(section_id)

    return critical_sections


def calculate_pulling_feasibility(
    route: Route,
    cable_spec: CableSpec,
    duct_specs: List[DuctSpec],
    lubricated_sections: List[bool],
    safety_factor: float = 1.5,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Calculate overall pulling feasibility for a route.

    Args:
        route: Route to analyze
        cable_spec: Cable specifications
        duct_specs: Duct specifications for each section
        lubricated_sections: Lubrication status for each section
        safety_factor: Safety factor to apply to limits
        **kwargs: Additional arguments for analyze_route_with_varying_conditions

    Returns:
        Dictionary with feasibility analysis results
    """
    # Apply safety factor to cable specifications
    safe_cable_spec = CableSpec(
        diameter=cable_spec.diameter,
        weight_per_meter=cable_spec.weight_per_meter,
        max_tension=cable_spec.max_tension / safety_factor,
        max_sidewall_pressure=cable_spec.max_sidewall_pressure / safety_factor,
        min_bend_radius=cable_spec.min_bend_radius * safety_factor,
        pulling_method=cable_spec.pulling_method,
        arrangement=cable_spec.arrangement,
        number_of_cables=cable_spec.number_of_cables,
    )

    # Perform analysis with safety factors
    route_analysis = analyze_route_with_varying_conditions(
        route, safe_cable_spec, duct_specs, lubricated_sections, **kwargs
    )

    # Determine overall feasibility
    all_sections_pass = all(
        limit_result.passes_tension_limit
        and limit_result.passes_pressure_limit
        and limit_result.passes_bend_radius_limit
        for _, limit_result, _ in route_analysis
    )

    # Find critical sections
    critical_sections = find_critical_sections(route_analysis)

    # Calculate statistics
    if route_analysis:
        max_tension = max(
            tension_analysis.max_tension for tension_analysis, _, _ in route_analysis
        )
        pressure_values = [
            limit_result.max_pressure
            for _, limit_result, _ in route_analysis
            if limit_result.max_pressure > 0
        ]
        max_pressure = max(pressure_values) if pressure_values else 0.0
    else:
        max_tension = 0.0
        max_pressure = 0.0

    # Recommend optimal pulling strategy
    section_recommendations = {}
    for tension_analysis, limit_result, _ in route_analysis:
        section_recommendations[tension_analysis.section_id] = {
            "direction": limit_result.recommended_direction,
            "max_tension": limit_result.max_tension,
            "limiting_factors": limit_result.limiting_factors,
        }

    return {
        "feasible": all_sections_pass,
        "safety_factor_applied": safety_factor,
        "max_tension_n": max_tension,
        "max_pressure_n_per_m": max_pressure,
        "critical_sections": critical_sections,
        "section_recommendations": section_recommendations,
        "total_route_length_m": route.total_length,
        "analysis_details": route_analysis,
    }
