"""Cable pulling calculations for tension and sidewall pressure."""

from .advanced import (
    analyze_route_with_varying_conditions,
    analyze_section_with_environment,
    calculate_jam_ratio_iec,
    calculate_multi_cable_weight_factor,
    calculate_pulling_feasibility,
    calculate_slope_correction_factor,
    calculate_temperature_friction_factor,
    find_critical_sections,
)
from .pressure import (
    LimitCheckResult,
    PressureResult,
    analyze_section_pressures,
    calculate_jam_factor,
    calculate_sidewall_pressure,
    check_duct_clearance,
    check_section_limits,
)
from .tension import (
    SectionTensionAnalysis,
    TensionResult,
    analyze_route_tension,
    analyze_section_tension,
    calculate_bend_tension,
    calculate_section_tensions,
    calculate_straight_tension,
    find_optimal_pull_direction,
)

__all__ = [
    # Tension calculations
    "TensionResult",
    "SectionTensionAnalysis",
    "calculate_straight_tension",
    "calculate_bend_tension",
    "calculate_section_tensions",
    "analyze_section_tension",
    "analyze_route_tension",
    "find_optimal_pull_direction",
    # Pressure calculations
    "PressureResult",
    "LimitCheckResult",
    "calculate_sidewall_pressure",
    "analyze_section_pressures",
    "check_section_limits",
    "check_duct_clearance",
    "calculate_jam_factor",
    # Advanced calculations
    "calculate_slope_correction_factor",
    "calculate_multi_cable_weight_factor",
    "calculate_temperature_friction_factor",
    "calculate_jam_ratio_iec",
    "analyze_section_with_environment",
    "analyze_route_with_varying_conditions",
    "calculate_pulling_feasibility",
    "find_critical_sections",
]
