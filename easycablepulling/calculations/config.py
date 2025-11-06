"""Configuration for cable pulling calculation methods and standards."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class CalculationStandard(Enum):
    """Industry standards for cable pulling calculations."""

    CIGRE = "cigre"  # CIGRE TB-889 - International, conservative
    AEIC = "aeic"  # AEIC CG5-2015 - North American, with triangular reductions
    POLYWATER = "polywater"  # Polywater formulas - similar to AEIC


@dataclass
class CalculationConfig:
    """Configuration for calculation methods and options.

    Provides control over which industry standard to use and allows
    fine-grained control over specific calculation features.

    Attributes:
        standard: Primary calculation standard to use
        apply_sidewall_reduction: Whether to apply triangular reduction (÷2) to sidewall pressure.
            If None, defaults based on standard (False for CIGRE, True for AEIC/Polywater)
        apply_weight_correction: Whether to apply weight correction factors.
            If None, defaults to True for all standards
        check_jam_ratio: Whether to check and warn about cable jamming.
            If None, defaults to True for all standards
        tension_safety_factor: Safety factor applied to maximum allowable tension (≥1.0)
        pressure_safety_factor: Safety factor applied to maximum sidewall pressure (≥1.0)
    """

    standard: CalculationStandard = CalculationStandard.CIGRE

    # Optional overrides (if None, follows standard defaults)
    apply_sidewall_reduction: Optional[bool] = None
    apply_weight_correction: Optional[bool] = None
    check_jam_ratio: Optional[bool] = None

    # Safety factors (multipliers for limits)
    tension_safety_factor: float = 1.0
    pressure_safety_factor: float = 1.0

    def __post_init__(self) -> None:
        """Set defaults based on selected standard and validate inputs."""
        # Validate safety factors
        if self.tension_safety_factor < 1.0:
            raise ValueError("Tension safety factor must be >= 1.0")
        if self.pressure_safety_factor < 1.0:
            raise ValueError("Pressure safety factor must be >= 1.0")

        # Set standard-specific defaults
        if self.standard == CalculationStandard.CIGRE:
            # CIGRE TB-889: Conservative, no sidewall reduction
            if self.apply_sidewall_reduction is None:
                self.apply_sidewall_reduction = False
            if self.apply_weight_correction is None:
                self.apply_weight_correction = True
            if self.check_jam_ratio is None:
                self.check_jam_ratio = True

        elif self.standard in (CalculationStandard.AEIC, CalculationStandard.POLYWATER):
            # AEIC/Polywater: With triangular reductions
            if self.apply_sidewall_reduction is None:
                self.apply_sidewall_reduction = True
            if self.apply_weight_correction is None:
                self.apply_weight_correction = True
            if self.check_jam_ratio is None:
                self.check_jam_ratio = True

    @property
    def jam_ratio_warning_min(self) -> float:
        """Get minimum jam ratio for warning zone based on standard."""
        if self.standard == CalculationStandard.CIGRE:
            return 2.8  # CIGRE uses 2.8-3.0
        else:
            return 2.7  # AEIC/Polywater use 2.7-3.1

    @property
    def jam_ratio_warning_max(self) -> float:
        """Get maximum jam ratio for warning zone based on standard."""
        if self.standard == CalculationStandard.CIGRE:
            return 3.0  # CIGRE uses 2.8-3.0
        else:
            return 3.1  # AEIC/Polywater use 2.7-3.1

    def description(self) -> str:
        """Get human-readable description of configuration."""
        lines = [f"Calculation Standard: {self.standard.value.upper()}"]

        if self.standard == CalculationStandard.CIGRE:
            lines.append("  - CIGRE TB-889 (International/European)")
            lines.append("  - Conservative approach")
        elif self.standard == CalculationStandard.AEIC:
            lines.append("  - AEIC CG5-2015 (North American)")
            lines.append("  - Association of Edison Illuminating Companies")
        elif self.standard == CalculationStandard.POLYWATER:
            lines.append("  - Polywater Pull-Planner formulas")

        lines.append(f"Sidewall Reduction (÷2 for triangular): {self.apply_sidewall_reduction}")
        lines.append(f"Weight Correction Factors: {self.apply_weight_correction}")
        lines.append(f"Jam Ratio Checks: {self.check_jam_ratio}")

        if self.tension_safety_factor != 1.0:
            lines.append(f"Tension Safety Factor: {self.tension_safety_factor:.2f}")
        if self.pressure_safety_factor != 1.0:
            lines.append(f"Pressure Safety Factor: {self.pressure_safety_factor:.2f}")

        return "\n".join(lines)


# Preset configurations for common scenarios
def cigre_conservative() -> CalculationConfig:
    """Conservative CIGRE configuration with additional safety margins."""
    return CalculationConfig(
        standard=CalculationStandard.CIGRE,
        tension_safety_factor=1.1,  # 10% additional margin
        pressure_safety_factor=1.1,
    )


def aeic_standard() -> CalculationConfig:
    """Standard AEIC configuration (North American practice)."""
    return CalculationConfig(standard=CalculationStandard.AEIC)


def midlands_config() -> CalculationConfig:
    """Configuration for Midlands project (bound trefoil, AEIC standard)."""
    return CalculationConfig(
        standard=CalculationStandard.AEIC,
        apply_sidewall_reduction=True,
        apply_weight_correction=True,
        check_jam_ratio=False,  # Bound cables don't jam
    )
