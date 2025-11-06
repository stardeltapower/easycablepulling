"""Weight correction factor calculations for multi-cable installations."""

import math
from typing import Tuple

from ..core.models import CableArrangement, CableSpec, DuctSpec


def calculate_weight_correction_factor(
    cable_diameter: float, duct_inner_diameter: float, arrangement: CableArrangement
) -> float:
    """Calculate weight correction factor based on cable configuration.

    The weight correction factor accounts for the additional normal force
    created when multiple cables are pulled through a duct. Different
    geometric arrangements produce different correction factors.

    Based on formulas from:
    - Polywater Pull-Planner
    - AEIC CG5-2015
    - CIGRE TB-889 Section 4.5.7

    Args:
        cable_diameter: Outside diameter of single cable in mm
        duct_inner_diameter: Inner diameter of duct in mm
        arrangement: Cable arrangement type

    Returns:
        Weight correction factor (dimensionless, >= 1.0)

    Raises:
        ValueError: If diameters are invalid or ratio is out of range
    """
    if cable_diameter <= 0:
        raise ValueError("Cable diameter must be positive")
    if duct_inner_diameter <= 0:
        raise ValueError("Duct inner diameter must be positive")
    if cable_diameter >= duct_inner_diameter:
        raise ValueError("Cable diameter must be less than duct diameter")

    d = cable_diameter
    D = duct_inner_diameter

    # Calculate ratio
    ratio = d / D

    if arrangement == CableArrangement.SINGLE:
        return 1.0

    elif arrangement == CableArrangement.TREFOIL:
        # Polywater formula for triangular configuration
        # WCF = √[1 + (d/D)²]
        return math.sqrt(1 + ratio**2)

    elif arrangement == CableArrangement.FLAT:
        # For flat arrangement with 2+ cables
        # Use cradled formula as it's more conservative for flat
        # WCF = 2 / √[1 - (d/D)²]
        denominator = 1 - ratio**2
        if denominator <= 0:
            raise ValueError(
                f"Invalid diameter ratio {ratio:.3f} for flat arrangement. "
                f"Must be < 1.0"
            )
        return 2.0 / math.sqrt(denominator)

    else:
        # Fallback: return 1.0 (no correction)
        return 1.0


def calculate_weight_correction_factor_cigre(
    cable_diameter: float, duct_inner_diameter: float, arrangement: CableArrangement
) -> float:
    """Calculate weight correction factor using CIGRE TB-889 formulas.

    CIGRE uses (D-d) in the denominator instead of D.
    This is slightly different from Polywater formulas.

    Args:
        cable_diameter: Outside diameter of single cable in mm
        duct_inner_diameter: Inner diameter of duct in mm
        arrangement: Cable arrangement type

    Returns:
        Weight correction factor (dimensionless, >= 1.0)
    """
    if cable_diameter <= 0:
        raise ValueError("Cable diameter must be positive")
    if duct_inner_diameter <= 0:
        raise ValueError("Duct inner diameter must be positive")
    if cable_diameter >= duct_inner_diameter:
        raise ValueError("Cable diameter must be less than duct diameter")

    d = cable_diameter
    D = duct_inner_diameter

    if arrangement == CableArrangement.SINGLE:
        return 1.0

    elif arrangement == CableArrangement.TREFOIL:
        # CIGRE formula for triangular
        # Wc = 1 / √[1 - (d/(D-d))²]
        ratio = d / (D - d)
        denominator = 1 - ratio**2
        if denominator <= 0:
            raise ValueError(
                f"Invalid diameter ratio for triangular arrangement. "
                f"Cable too large for duct."
            )
        return 1.0 / math.sqrt(denominator)

    elif arrangement == CableArrangement.FLAT:
        # CIGRE formula for cradled
        # Wc = 1 + (4/3) × [d/(D-d)]²
        ratio = d / (D - d)
        return 1.0 + (4.0 / 3.0) * ratio**2

    else:
        return 1.0


def get_weight_correction_factor(
    cable_spec: CableSpec, duct_spec: DuctSpec, use_cigre_formula: bool = False
) -> float:
    """Get weight correction factor for a cable/duct combination.

    Convenience wrapper that handles CableSpec and DuctSpec objects.

    Args:
        cable_spec: Cable specifications
        duct_spec: Duct specifications
        use_cigre_formula: If True, use CIGRE formulas; otherwise use Polywater

    Returns:
        Weight correction factor
    """
    if cable_spec.arrangement == CableArrangement.SINGLE:
        return 1.0

    if use_cigre_formula:
        return calculate_weight_correction_factor_cigre(
            cable_spec.diameter, duct_spec.inner_diameter, cable_spec.arrangement
        )
    else:
        return calculate_weight_correction_factor(
            cable_spec.diameter, duct_spec.inner_diameter, cable_spec.arrangement
        )


def check_jam_ratio(
    cable_spec: CableSpec, duct_spec: DuctSpec, safety_margin: bool = True
) -> Tuple[float, str, bool]:
    """Check jam ratio for multi-cable installations.

    The jam ratio (J = D/d) determines whether cables will jam during pulling.
    Critical range is J = 2.7-3.1 where cables can wedge.

    Args:
        cable_spec: Cable specifications
        duct_spec: Duct specifications
        safety_margin: If True, apply 5% safety margin to diameter

    Returns:
        Tuple of (jam_ratio, status_message, is_safe)
    """
    if cable_spec.number_of_cables == 1 or cable_spec.is_bound:
        # Single cable or bound cables cannot jam
        return (0.0, "Not applicable (single/bound cables)", True)

    d = cable_spec.diameter
    D = duct_spec.inner_diameter

    # Apply 5% safety margin if requested
    if safety_margin:
        d = d * 1.05

    jam_ratio = D / d

    # Determine status based on jam ratio
    if jam_ratio < 2.5:
        status = "SAFE - Triangular configuration, no jamming risk"
        is_safe = True
    elif 2.5 <= jam_ratio < 2.7:
        status = "CAUTION - Transitional zone, monitor during pull"
        is_safe = True
    elif 2.7 <= jam_ratio <= 3.1:
        status = "DANGER - JAMMING ZONE! High risk of cable wedging"
        is_safe = False
    elif 3.1 < jam_ratio <= 3.2:
        status = "CAUTION - Transitional zone, monitor during pull"
        is_safe = True
    else:
        status = "SAFE - Cradled configuration, no jamming risk"
        is_safe = True

    return (jam_ratio, status, is_safe)
