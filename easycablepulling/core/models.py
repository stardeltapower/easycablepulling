"""Core data models for cable pulling analysis."""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Literal, Optional, Tuple, Union


class CableArrangement(str, Enum):
    """Cable arrangement types."""

    SINGLE = "single"
    TREFOIL = "trefoil"
    FLAT = "flat"


class PullingMethod(str, Enum):
    """Cable pulling methods."""

    EYE = "eye"
    BASKET = "basket"


@dataclass
class CableSpec:
    """Cable specification for pulling calculations."""

    diameter: float  # Outside diameter in mm
    weight_per_meter: float  # Weight in kg/m
    max_tension: float  # Maximum allowable pulling tension in N
    max_sidewall_pressure: float  # Maximum sidewall pressure in N/m
    min_bend_radius: float  # Minimum bend radius in mm
    pulling_method: PullingMethod = PullingMethod.EYE
    arrangement: CableArrangement = CableArrangement.SINGLE
    number_of_cables: int = 1

    def __post_init__(self) -> None:
        """Validate cable specifications."""
        if self.diameter <= 0:
            raise ValueError("Cable diameter must be positive")
        if self.weight_per_meter <= 0:
            raise ValueError("Cable weight must be positive")
        if self.max_tension <= 0:
            raise ValueError("Maximum tension must be positive")
        if self.max_sidewall_pressure <= 0:
            raise ValueError("Maximum sidewall pressure must be positive")
        if self.min_bend_radius <= 0:
            raise ValueError("Minimum bend radius must be positive")
        if self.number_of_cables < 1:
            raise ValueError("Number of cables must be at least 1")

        # Validate arrangement vs number of cables
        if self.arrangement == CableArrangement.SINGLE and self.number_of_cables != 1:
            raise ValueError("Single arrangement requires exactly 1 cable")
        if self.arrangement == CableArrangement.TREFOIL and self.number_of_cables != 3:
            raise ValueError("Trefoil arrangement requires exactly 3 cables")
        if self.arrangement == CableArrangement.FLAT and self.number_of_cables < 2:
            raise ValueError("Flat arrangement requires at least 2 cables")

    @property
    def bundle_diameter(self) -> float:
        """Calculate effective bundle diameter based on arrangement."""
        if self.arrangement == CableArrangement.SINGLE:
            return self.diameter
        elif self.arrangement == CableArrangement.TREFOIL:
            # Trefoil bundle diameter is approximately 2.15 * single cable diameter
            return 2.15 * self.diameter
        else:  # FLAT
            # Flat arrangement width
            return self.diameter * self.number_of_cables

    @property
    def total_weight_per_meter(self) -> float:
        """Calculate total weight per meter for all cables."""
        return self.weight_per_meter * self.number_of_cables


@dataclass
class BendOption:
    """Standard duct bend option."""

    radius: float  # Bend radius in mm
    angle: float  # Bend angle in degrees

    def __post_init__(self) -> None:
        """Validate bend specifications."""
        if self.radius <= 0:
            raise ValueError("Bend radius must be positive")
        if not 0 < self.angle <= 180:
            raise ValueError("Bend angle must be between 0 and 180 degrees")

    @property
    def angle_radians(self) -> float:
        """Convert angle to radians."""
        return math.radians(self.angle)


@dataclass
class DuctSpec:
    """Duct specification for cable pulling."""

    inner_diameter: float  # Inner diameter in mm
    type: Literal["PVC", "HDPE", "Steel", "Concrete"]
    friction_dry: float  # Dry friction coefficient
    friction_lubricated: float  # Lubricated friction coefficient
    bend_options: List[BendOption] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate duct specifications."""
        if self.inner_diameter <= 0:
            raise ValueError("Duct inner diameter must be positive")
        if not 0 < self.friction_dry <= 1:
            raise ValueError("Dry friction coefficient must be between 0 and 1")
        if not 0 < self.friction_lubricated <= 1:
            raise ValueError("Lubricated friction coefficient must be between 0 and 1")
        if self.friction_lubricated >= self.friction_dry:
            raise ValueError("Lubricated friction must be less than dry friction")

    def can_accommodate_cable(self, cable_spec: CableSpec) -> bool:
        """Check if cable bundle can fit in duct."""
        # Basic clearance check - bundle should be smaller than duct
        clearance_factor = 0.9  # 90% fill ratio max
        return cable_spec.bundle_diameter < (self.inner_diameter * clearance_factor)

    def get_friction(
        self, cable_arrangement: CableArrangement, lubricated: bool = False
    ) -> float:
        """Get friction coefficient based on cable arrangement and lubrication."""
        base_friction = self.friction_lubricated if lubricated else self.friction_dry

        # Adjust friction for cable arrangement
        if cable_arrangement == CableArrangement.TREFOIL:
            # Trefoil has 20-40% higher friction, use 30% as default
            return base_friction * 1.3
        elif cable_arrangement == CableArrangement.FLAT:
            # Flat arrangement has slightly higher friction
            return base_friction * 1.1
        else:
            return base_friction


class Primitive(ABC):
    """Abstract base class for route primitives (straights and bends)."""

    @abstractmethod
    def length(self) -> float:
        """Get the length of the primitive in meters."""
        pass

    @abstractmethod
    def validate(self, cable_spec: CableSpec) -> List[str]:
        """Validate primitive against cable specifications.

        Returns list of validation warnings/errors.
        """
        pass


@dataclass
class Straight(Primitive):
    """Straight section of cable route."""

    length_m: float  # Length in meters
    start_point: Tuple[float, float]  # (x, y) coordinates
    end_point: Tuple[float, float]  # (x, y) coordinates

    def __post_init__(self) -> None:
        """Validate straight section."""
        if self.length_m <= 0:
            raise ValueError("Straight section length must be positive")

        # Verify length matches coordinates
        dx = self.end_point[0] - self.start_point[0]
        dy = self.end_point[1] - self.start_point[1]
        calculated_length = math.sqrt(dx**2 + dy**2)

        if abs(calculated_length - self.length_m) > 0.001:  # 1mm tolerance
            raise ValueError(
                f"Length mismatch: specified {self.length_m}m, "
                f"calculated {calculated_length}m from coordinates"
            )

    def length(self) -> float:
        """Get length in meters."""
        return self.length_m

    def validate(self, cable_spec: CableSpec) -> List[str]:
        """Validate straight section."""
        # Straight sections typically don't have cable-specific constraints
        return []


@dataclass
class Bend(Primitive):
    """Bend section of cable route."""

    radius_m: float  # Bend radius in meters
    angle_deg: float  # Bend angle in degrees
    direction: Literal["CW", "CCW"]  # Clockwise or counter-clockwise
    center_point: Tuple[float, float]  # Center of bend arc

    def __post_init__(self) -> None:
        """Validate bend section."""
        if self.radius_m <= 0:
            raise ValueError("Bend radius must be positive")
        if not 0 < self.angle_deg <= 180:
            raise ValueError("Bend angle must be between 0 and 180 degrees")

    @property
    def angle_rad(self) -> float:
        """Get angle in radians."""
        return math.radians(self.angle_deg)

    def length(self) -> float:
        """Get arc length in meters."""
        return self.radius_m * self.angle_rad

    def validate(self, cable_spec: CableSpec) -> List[str]:
        """Validate bend against cable specifications."""
        warnings = []

        # Check minimum bend radius
        min_radius_m = cable_spec.min_bend_radius / 1000  # Convert mm to m
        if self.radius_m < min_radius_m:
            warnings.append(
                f"Bend radius {self.radius_m:.2f}m is less than minimum "
                f"allowed {min_radius_m:.2f}m"
            )

        return warnings


@dataclass
class Section:
    """A section of cable route between joints/pits."""

    id: str  # Section identifier (e.g., "AB", "BC", etc.)
    original_polyline: List[Tuple[float, float]]  # Original (x, y) points
    primitives: List[Primitive] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate section."""
        if not self.id:
            raise ValueError("Section must have an ID")
        if len(self.original_polyline) < 2:
            raise ValueError("Section must have at least 2 points")

    @property
    def total_length(self) -> float:
        """Calculate total section length from primitives."""
        return sum(p.length() for p in self.primitives)

    @property
    def original_length(self) -> float:
        """Calculate length of original polyline."""
        length = 0.0
        for i in range(len(self.original_polyline) - 1):
            p1 = self.original_polyline[i]
            p2 = self.original_polyline[i + 1]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            length += math.sqrt(dx**2 + dy**2)
        return length

    @property
    def length_error_percent(self) -> float:
        """Calculate length error percentage."""
        if self.original_length == 0:
            return 0.0
        return (
            abs(self.total_length - self.original_length) / self.original_length * 100
        )

    def validate_fit(self, max_length_error: float = 0.2) -> List[str]:
        """Validate fitted geometry against original."""
        warnings = []

        # Check length error
        if self.length_error_percent > max_length_error:
            warnings.append(
                f"Section {self.id}: Length error {self.length_error_percent:.2f}% "
                f"exceeds maximum {max_length_error}%"
            )

        # TODO: Add lateral deviation check

        return warnings

    def validate_cable(self, cable_spec: CableSpec) -> List[str]:
        """Validate section against cable specifications."""
        warnings = []

        # Check each primitive
        for i, primitive in enumerate(self.primitives):
            primitive_warnings = primitive.validate(cable_spec)
            for warning in primitive_warnings:
                warnings.append(f"Section {self.id}, Primitive {i+1}: {warning}")

        return warnings


@dataclass
class Route:
    """Complete cable route composed of multiple sections."""

    name: str
    sections: List[Section] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate route."""
        if not self.name:
            raise ValueError("Route must have a name")

    @property
    def total_length(self) -> float:
        """Calculate total route length."""
        return sum(section.total_length for section in self.sections)

    @property
    def section_count(self) -> int:
        """Get number of sections."""
        return len(self.sections)

    def add_section(self, section: Section) -> None:
        """Add a section to the route."""
        self.sections.append(section)

    def validate_cable(self, cable_spec: CableSpec) -> List[str]:
        """Validate entire route against cable specifications."""
        warnings = []

        for section in self.sections:
            warnings.extend(section.validate_cable(cable_spec))

        return warnings

    def validate_fit(self, max_length_error: float = 0.2) -> List[str]:
        """Validate fitted geometry for all sections."""
        warnings = []

        for section in self.sections:
            warnings.extend(section.validate_fit(max_length_error))

        return warnings
