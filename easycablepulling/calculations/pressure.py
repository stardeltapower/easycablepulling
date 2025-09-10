"""Sidewall pressure calculations and limit checking."""

import math
from typing import List, NamedTuple, Optional, Tuple

from ..core.models import Bend, CableSpec, DuctSpec, Primitive, Route, Section
from .tension import TensionResult


class PressureResult(NamedTuple):
    """Result of sidewall pressure calculation."""

    position: float  # Position along route in meters
    tension: float  # Tension at this point in Newtons
    radius: float  # Bend radius in meters (None for straight sections)
    pressure: float  # Sidewall pressure in N/m (None for straight sections)
    primitive_index: int  # Index of primitive in section
    is_critical: bool  # Whether pressure exceeds limits


class LimitCheckResult(NamedTuple):
    """Result of limit checking analysis."""

    section_id: str
    passes_tension_limit: bool
    passes_pressure_limit: bool
    passes_bend_radius_limit: bool
    max_tension: float
    max_pressure: float
    min_bend_radius: float
    limiting_factors: List[str]  # List of what's limiting the pull
    recommended_direction: str  # "forward" or "backward"


class PressureCalculator:
    """Simplified pressure calculator for pipeline interface."""

    def calculate_max_sidewall_pressure(
        self,
        section: Section,
        cable_spec: CableSpec,
        duct_spec: DuctSpec,
        lubricated: bool = False,
    ) -> float:
        """Calculate maximum sidewall pressure for a section."""
        from .tension import analyze_section_tension

        # Get tension analysis
        analysis = analyze_section_tension(section, cable_spec, duct_spec, lubricated)

        # Find maximum pressure across all bends
        max_pressure = 0.0

        # Check forward direction
        for i, result in enumerate(analysis.forward_tensions):
            if i < len(section.primitives):
                primitive = section.primitives[i]
                if isinstance(primitive, Bend):
                    pressure = calculate_sidewall_pressure(
                        result.tension, primitive.radius_m
                    )
                    max_pressure = max(max_pressure, pressure)

        # Check backward direction
        for i, result in enumerate(analysis.backward_tensions):
            if i < len(section.primitives):
                primitive = section.primitives[i]
                if isinstance(primitive, Bend):
                    pressure = calculate_sidewall_pressure(
                        result.tension, primitive.radius_m
                    )
                    max_pressure = max(max_pressure, pressure)

        return max_pressure


def calculate_sidewall_pressure(tension: float, bend_radius: float) -> float:
    """Calculate sidewall pressure in a bend.

    Uses the formula: P = T / r
    where:
    - P: Sidewall pressure (N/m)
    - T: Cable tension (N)
    - r: Bend radius (m)

    Args:
        tension: Cable tension in Newtons
        bend_radius: Bend radius in meters

    Returns:
        Sidewall pressure in N/m
    """
    if tension < 0:
        raise ValueError("Tension cannot be negative")
    if bend_radius <= 0:
        raise ValueError("Bend radius must be positive")

    return tension / bend_radius


def analyze_section_pressures(
    section: Section,
    tension_results: List[TensionResult],
    cable_spec: CableSpec,
) -> List[PressureResult]:
    """Calculate sidewall pressures for all bends in a section.

    Args:
        section: Section being analyzed
        tension_results: Tension results from tension analysis
        cable_spec: Cable specifications for pressure limits

    Returns:
        List of pressure results for each primitive
    """
    pressure_results = []

    for tension_result in tension_results:
        primitive = section.primitives[tension_result.primitive_index]

        if isinstance(primitive, Bend):
            # Calculate sidewall pressure for bend
            pressure = calculate_sidewall_pressure(
                tension_result.tension, primitive.radius_m
            )

            # Check if pressure exceeds limits
            is_critical = pressure > cable_spec.max_sidewall_pressure

            pressure_results.append(
                PressureResult(
                    position=tension_result.position,
                    tension=tension_result.tension,
                    radius=primitive.radius_m,
                    pressure=pressure,
                    primitive_index=tension_result.primitive_index,
                    is_critical=is_critical,
                )
            )
        else:
            # Straight sections don't have sidewall pressure
            pressure_results.append(
                PressureResult(
                    position=tension_result.position,
                    tension=tension_result.tension,
                    radius=0.0,  # No radius for straight
                    pressure=0.0,  # No pressure for straight
                    primitive_index=tension_result.primitive_index,
                    is_critical=False,
                )
            )

    return pressure_results


def check_section_limits(
    section: Section,
    cable_spec: CableSpec,
    duct_spec: DuctSpec,
    forward_tensions: List[TensionResult],
    backward_tensions: List[TensionResult],
    lubricated: bool = False,
) -> LimitCheckResult:
    """Check all limits for a section and determine feasibility.

    Args:
        section: Section to check
        cable_spec: Cable specifications
        duct_spec: Duct specifications
        forward_tensions: Forward pulling tension results
        backward_tensions: Backward pulling tension results
        lubricated: Whether duct is lubricated

    Returns:
        Complete limit check results
    """
    limiting_factors = []

    # Check tension limits for both directions
    max_forward_tension = max(result.tension for result in forward_tensions)
    max_backward_tension = max(result.tension for result in backward_tensions)

    passes_tension_limit = (
        max_forward_tension <= cable_spec.max_tension
        and max_backward_tension <= cable_spec.max_tension
    )

    if not passes_tension_limit:
        limiting_factors.append("tension_limit")

    # Determine better pulling direction
    if max_forward_tension <= max_backward_tension:
        recommended_direction = "forward"
        recommended_tensions = forward_tensions
        max_tension = max_forward_tension
    else:
        recommended_direction = "backward"
        recommended_tensions = backward_tensions
        max_tension = max_backward_tension

    # Check pressure limits using recommended direction
    pressure_results = analyze_section_pressures(
        section, recommended_tensions, cable_spec
    )

    max_pressure = (
        max(
            result.pressure
            for result in pressure_results
            if result.pressure > 0  # Only consider bends
        )
        if any(result.pressure > 0 for result in pressure_results)
        else 0.0
    )

    passes_pressure_limit = max_pressure <= cable_spec.max_sidewall_pressure

    if not passes_pressure_limit:
        limiting_factors.append("sidewall_pressure")

    # Check bend radius limits
    min_radius_required = cable_spec.min_bend_radius / 1000  # Convert mm to m
    min_actual_radius = float("inf")

    for primitive in section.primitives:
        if isinstance(primitive, Bend):
            min_actual_radius = min(min_actual_radius, primitive.radius_m)

    # If no bends, set to infinity
    if min_actual_radius == float("inf"):
        min_actual_radius = float("inf")
        passes_bend_radius_limit = True
    else:
        passes_bend_radius_limit = min_actual_radius >= min_radius_required
        if not passes_bend_radius_limit:
            limiting_factors.append("bend_radius")

    return LimitCheckResult(
        section_id=section.id,
        passes_tension_limit=passes_tension_limit,
        passes_pressure_limit=passes_pressure_limit,
        passes_bend_radius_limit=passes_bend_radius_limit,
        max_tension=max_tension,
        max_pressure=max_pressure,
        min_bend_radius=min_actual_radius,
        limiting_factors=limiting_factors,
        recommended_direction=recommended_direction,
    )


def check_duct_clearance(
    cable_spec: CableSpec, duct_spec: DuctSpec
) -> Tuple[bool, float]:
    """Check if cable bundle fits in duct with adequate clearance.

    Args:
        cable_spec: Cable specifications
        duct_spec: Duct specifications

    Returns:
        Tuple of (fits, clearance_ratio) where clearance_ratio is bundle/duct diameter
    """
    bundle_diameter = cable_spec.bundle_diameter
    duct_diameter = duct_spec.inner_diameter

    clearance_ratio = bundle_diameter / duct_diameter

    # Standard clearance limits:
    # - Single cable: 90% max fill
    # - Multiple cables: 80% max fill (more conservative due to jamming)
    max_fill_ratio = 0.9 if cable_spec.number_of_cables == 1 else 0.8

    fits = clearance_ratio <= max_fill_ratio

    return fits, clearance_ratio


def calculate_jam_factor(cable_spec: CableSpec, duct_spec: DuctSpec) -> float:
    """Calculate jam factor for multi-cable installations.

    Based on IEC standards for cable jamming in ducts.

    Args:
        cable_spec: Cable specifications
        duct_spec: Duct specifications

    Returns:
        Jam factor (dimensionless). Values > 1.0 indicate high jamming risk.
    """
    if cable_spec.number_of_cables == 1:
        return 0.0  # No jamming with single cable

    bundle_diameter = cable_spec.bundle_diameter
    duct_diameter = duct_spec.inner_diameter

    # Simplified jam factor calculation
    # Higher values indicate greater jamming risk
    jam_factor = (bundle_diameter / duct_diameter) ** 2

    # Adjust for arrangement type
    if cable_spec.arrangement.value == "trefoil":
        # Trefoil has lower jamming risk due to stable geometry
        jam_factor *= 0.8
    elif cable_spec.arrangement.value == "flat":
        # Flat arrangement has higher jamming risk
        jam_factor *= 1.2

    return jam_factor
