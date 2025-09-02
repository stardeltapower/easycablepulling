"""Cable tension calculations for pulling analysis."""

import math
from typing import List, NamedTuple, Optional, Tuple

from ..core.models import Bend, CableSpec, DuctSpec, Primitive, Route, Section, Straight


class TensionResult(NamedTuple):
    """Result of tension calculation at a point."""

    position: float  # Position along route in meters
    tension: float  # Tension in Newtons
    primitive_index: int  # Index of primitive in section
    primitive_type: str  # Type of primitive ("straight" or "bend")


class SectionTensionAnalysis(NamedTuple):
    """Complete tension analysis for a section."""

    section_id: str
    forward_tensions: List[TensionResult]  # Pulling from start to end
    backward_tensions: List[TensionResult]  # Pulling from end to start
    max_tension: float  # Maximum tension in section
    max_tension_position: float  # Position of maximum tension
    critical_primitive_index: int  # Index of primitive with max tension


def calculate_straight_tension(
    tension_in: float,
    cable_spec: CableSpec,
    duct_spec: DuctSpec,
    length: float,
    lubricated: bool = False,
    slope_angle: float = 0.0,
) -> float:
    """Calculate tension at end of straight section.

    Uses the formula: T_out = T_in + W * f * L
    where:
    - T_in: Input tension (N)
    - W: Cable weight per unit length including slope correction (N/m)
    - f: Friction coefficient
    - L: Length of straight section (m)

    Args:
        tension_in: Input tension in Newtons
        cable_spec: Cable specifications
        duct_spec: Duct specifications
        length: Length of straight section in meters
        lubricated: Whether duct is lubricated
        slope_angle: Slope angle in degrees (positive = uphill)

    Returns:
        Output tension in Newtons
    """
    if tension_in < 0:
        raise ValueError("Input tension cannot be negative")
    if length < 0:
        raise ValueError("Length cannot be negative")

    # Get friction coefficient for cable arrangement
    friction = duct_spec.get_friction(cable_spec.arrangement, lubricated)

    # Calculate weight per meter with slope correction
    # Positive slope_angle means uphill pulling (adds to tension)
    slope_factor = math.sin(math.radians(slope_angle))
    weight_per_meter = cable_spec.total_weight_per_meter * 9.81  # Convert kg/m to N/m
    effective_weight = weight_per_meter * (friction + slope_factor)

    # Apply tension formula
    tension_out = tension_in + effective_weight * length

    return max(0.0, tension_out)  # Tension cannot be negative


def calculate_bend_tension(
    tension_in: float,
    cable_spec: CableSpec,
    duct_spec: DuctSpec,
    bend_angle: float,
    lubricated: bool = False,
) -> float:
    """Calculate tension at end of bend using capstan equation.

    Uses the formula: T_out = T_in * e^(f * θ)
    where:
    - T_in: Input tension (N)
    - f: Friction coefficient
    - θ: Bend angle in radians (always positive)

    Args:
        tension_in: Input tension in Newtons
        cable_spec: Cable specifications
        duct_spec: Duct specifications
        bend_angle: Bend angle in degrees (sign doesn't matter for tension)
        lubricated: Whether duct is lubricated

    Returns:
        Output tension in Newtons
    """
    if tension_in < 0:
        raise ValueError("Input tension cannot be negative")

    # Get friction coefficient for cable arrangement
    friction = duct_spec.get_friction(cable_spec.arrangement, lubricated)

    # Convert angle to radians and take absolute value
    angle_rad = math.radians(abs(bend_angle))

    # Apply capstan equation
    tension_out = tension_in * math.exp(friction * angle_rad)

    return tension_out


def calculate_section_tensions(
    section: Section,
    cable_spec: CableSpec,
    duct_spec: DuctSpec,
    initial_tension: float = 0.0,
    lubricated: bool = False,
    reverse: bool = False,
) -> List[TensionResult]:
    """Calculate tensions throughout a section.

    Args:
        section: Section to analyze
        cable_spec: Cable specifications
        duct_spec: Duct specifications
        initial_tension: Starting tension in Newtons
        lubricated: Whether duct is lubricated
        reverse: If True, calculate pulling from end to start

    Returns:
        List of tension results at each primitive
    """
    results = []
    current_tension = initial_tension
    current_position = 0.0

    # Get primitives in correct order
    primitives = list(reversed(section.primitives)) if reverse else section.primitives

    for i, primitive in enumerate(primitives):
        primitive_index = len(section.primitives) - 1 - i if reverse else i

        if isinstance(primitive, Straight):
            # Calculate tension through straight section
            tension_out = calculate_straight_tension(
                current_tension,
                cable_spec,
                duct_spec,
                primitive.length(),
                lubricated,
            )
            primitive_type = "straight"

        elif isinstance(primitive, Bend):
            # Calculate tension through bend
            tension_out = calculate_bend_tension(
                current_tension,
                cable_spec,
                duct_spec,
                primitive.angle_deg,
                lubricated,
            )
            primitive_type = "bend"

        else:
            # For polynomial curves, treat as equivalent bend
            # This is a simplification - in practice might need more sophisticated approach
            tension_out = current_tension  # Placeholder
            primitive_type = "curve"

        # Record result
        results.append(
            TensionResult(
                position=current_position + primitive.length(),
                tension=tension_out,
                primitive_index=primitive_index,
                primitive_type=primitive_type,
            )
        )

        # Update for next iteration
        current_tension = tension_out
        current_position += primitive.length()

    return results


def analyze_section_tension(
    section: Section,
    cable_spec: CableSpec,
    duct_spec: DuctSpec,
    lubricated: bool = False,
) -> SectionTensionAnalysis:
    """Perform complete tension analysis for a section.

    Calculates tensions for both forward and backward pulling directions
    and identifies the critical (maximum tension) points.

    Args:
        section: Section to analyze
        cable_spec: Cable specifications
        duct_spec: Duct specifications
        lubricated: Whether duct is lubricated

    Returns:
        Complete tension analysis results
    """
    # Calculate forward tensions (pulling from start to end)
    forward_tensions = calculate_section_tensions(
        section, cable_spec, duct_spec, 0.0, lubricated, False
    )

    # Calculate backward tensions (pulling from end to start)
    backward_tensions = calculate_section_tensions(
        section, cable_spec, duct_spec, 0.0, lubricated, True
    )

    # Find maximum tension and its location
    all_tensions = forward_tensions + backward_tensions
    max_result = max(all_tensions, key=lambda x: x.tension)

    return SectionTensionAnalysis(
        section_id=section.id,
        forward_tensions=forward_tensions,
        backward_tensions=backward_tensions,
        max_tension=max_result.tension,
        max_tension_position=max_result.position,
        critical_primitive_index=max_result.primitive_index,
    )


def analyze_route_tension(
    route: Route,
    cable_spec: CableSpec,
    duct_spec: DuctSpec,
    lubricated: bool = False,
) -> List[SectionTensionAnalysis]:
    """Analyze tension for all sections in a route.

    Args:
        route: Route to analyze
        cable_spec: Cable specifications
        duct_spec: Duct specifications
        lubricated: Whether duct is lubricated

    Returns:
        List of tension analyses for each section
    """
    analyses = []

    for section in route.sections:
        analysis = analyze_section_tension(section, cable_spec, duct_spec, lubricated)
        analyses.append(analysis)

    return analyses


def find_optimal_pull_direction(
    section: Section,
    cable_spec: CableSpec,
    duct_spec: DuctSpec,
    lubricated: bool = False,
) -> Tuple[str, float]:
    """Determine optimal pulling direction for minimum tension.

    Args:
        section: Section to analyze
        cable_spec: Cable specifications
        duct_spec: Duct specifications
        lubricated: Whether duct is lubricated

    Returns:
        Tuple of (direction, max_tension) where direction is "forward" or "backward"
    """
    analysis = analyze_section_tension(section, cable_spec, duct_spec, lubricated)

    # Find maximum tension in each direction
    max_forward = max(result.tension for result in analysis.forward_tensions)
    max_backward = max(result.tension for result in analysis.backward_tensions)

    if max_forward <= max_backward:
        return ("forward", max_forward)
    else:
        return ("backward", max_backward)
