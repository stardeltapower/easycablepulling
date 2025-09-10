"""Duct inventory specifications and management."""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class BendSpecification:
    """Specification for a duct bend."""

    radius_m: float  # Bend radius in meters
    angle_deg: float  # Bend angle in degrees
    chord_length_m: float  # Pre-calculated chord length
    arc_length_m: float  # Pre-calculated arc length

    @classmethod
    def from_radius_angle(
        cls, radius_m: float, angle_deg: float
    ) -> "BendSpecification":
        """Create bend spec with calculated lengths."""
        angle_rad = math.radians(angle_deg)
        chord_length = 2 * radius_m * math.sin(angle_rad / 2)
        arc_length = radius_m * angle_rad
        return cls(radius_m, angle_deg, chord_length, arc_length)


@dataclass
class DuctSpecification:
    """Complete specification for a duct type."""

    name: str
    outer_diameter_mm: int
    straight_length_m: float
    min_cut_length_m: float
    bends: List[BendSpecification]
    max_section_length_m: float = 1000.0
    max_lateral_deviation_m: float = 1.0

    def get_available_angles(self) -> List[float]:
        """Get list of available bend angles."""
        return [bend.angle_deg for bend in self.bends]

    def get_bend_by_angle(self, angle_deg: float) -> Optional[BendSpecification]:
        """Get bend specification by angle."""
        for bend in self.bends:
            if abs(bend.angle_deg - angle_deg) < 0.01:
                return bend
        return None


# Standard duct specifications
DUCT_SPECIFICATIONS: Dict[str, DuctSpecification] = {
    "200mm": DuctSpecification(
        name="200mm",
        outer_diameter_mm=200,
        straight_length_m=6.0,
        min_cut_length_m=0.5,
        bends=[
            BendSpecification.from_radius_angle(3.9, 11.25),
            BendSpecification.from_radius_angle(3.9, 22.5),
        ],
        max_section_length_m=1000.0,
        max_lateral_deviation_m=1.0,
    ),
    # Additional duct sizes can be added here
    # "110mm": DuctSpecification(...),
    # "160mm": DuctSpecification(...),
    # "250mm": DuctSpecification(...),
}


class DuctInventory:
    """Manages duct inventory and constraints for fitting."""

    def __init__(self, duct_type: str = "200mm"):
        """Initialize inventory with specified duct type.

        Args:
            duct_type: Type of duct (e.g., "200mm")

        Raises:
            ValueError: If duct type is not recognized
        """
        if duct_type not in DUCT_SPECIFICATIONS:
            available = ", ".join(DUCT_SPECIFICATIONS.keys())
            raise ValueError(
                f"Unknown duct type: {duct_type}. " f"Available types: {available}"
            )

        self.duct_type = duct_type
        self.spec = DUCT_SPECIFICATIONS[duct_type]

    def get_nearest_bend(
        self, target_angle_deg: float, tolerance_deg: float = 3.0
    ) -> Optional[BendSpecification]:
        """Find the nearest available bend to target angle.

        Args:
            target_angle_deg: Desired bend angle in degrees
            tolerance_deg: Maximum acceptable difference

        Returns:
            Nearest bend specification or None if none within tolerance
        """
        best_bend = None
        min_diff = float("inf")

        for bend in self.spec.bends:
            diff = abs(bend.angle_deg - abs(target_angle_deg))
            if diff < min_diff and diff <= tolerance_deg:
                min_diff = diff
                best_bend = bend

        return best_bend

    def snap_angle_to_inventory(self, angle_deg: float) -> float:
        """Snap an angle to nearest inventory angle.

        Args:
            angle_deg: Input angle in degrees

        Returns:
            Nearest available angle, preserving sign
        """
        sign = 1 if angle_deg >= 0 else -1
        abs_angle = abs(angle_deg)

        # Find nearest available angle
        available_angles = self.get_available_angles()
        if not available_angles:
            return 0.0

        nearest = min(available_angles, key=lambda a: abs(a - abs_angle))
        return sign * nearest

    def get_available_angles(self) -> List[float]:
        """Get list of available bend angles."""
        return self.spec.get_available_angles()

    def optimize_straight_cuts(
        self, required_length_m: float
    ) -> List[Tuple[float, bool]]:
        """Optimize straight duct cuts for required length.

        Args:
            required_length_m: Total straight length needed

        Returns:
            List of (length, is_full_length) tuples
        """
        standard_length = self.spec.straight_length_m
        min_cut = self.spec.min_cut_length_m

        cuts = []
        remaining = required_length_m

        # Use as many full lengths as possible
        full_lengths = int(remaining / standard_length)
        for _ in range(full_lengths):
            cuts.append((standard_length, True))
            remaining -= standard_length

        # Add final cut if needed
        if remaining >= min_cut:
            cuts.append((remaining, False))
        elif remaining > 0 and cuts:
            # Adjust last full length to accommodate remainder
            cuts[-1] = (standard_length - min_cut + remaining, False)

        return cuts

    def calculate_cut_penalty(self, length_m: float) -> float:
        """Calculate penalty for non-standard straight cut.

        Args:
            length_m: Length of straight segment

        Returns:
            Penalty value (0 for full lengths, >0 for cuts)
        """
        if abs(length_m - self.spec.straight_length_m) < 0.01:
            return 0.0
        return 0.1  # Base cut penalty

    def validate_primitive_length(
        self, length_m: float, is_straight: bool = True
    ) -> bool:
        """Check if a primitive length is valid.

        Args:
            length_m: Length to validate
            is_straight: True for straight, False for bend

        Returns:
            True if length is valid
        """
        if is_straight:
            return length_m >= self.spec.min_cut_length_m
        else:
            # Bends have fixed arc lengths
            for bend in self.spec.bends:
                if abs(length_m - bend.arc_length_m) < 0.01:
                    return True
            return False

    def get_bend_radius(self) -> float:
        """Get the standard bend radius for this duct type.

        Returns:
            Bend radius in meters
        """
        if self.spec.bends:
            return self.spec.bends[0].radius_m
        return 0.0

    def decompose_angle(self, total_angle_deg: float) -> List[float]:
        """Decompose a total angle into available bend angles.

        Uses greedy approach to minimize bend count.

        Args:
            total_angle_deg: Total angle to achieve

        Returns:
            List of bend angles that sum to approximately total_angle
        """
        sign = 1 if total_angle_deg >= 0 else -1
        remaining = abs(total_angle_deg)
        result = []

        # Sort available angles in descending order
        available = sorted(self.get_available_angles(), reverse=True)

        # Greedy decomposition
        while remaining > 0.1:  # Small tolerance
            used_any = False
            for angle in available:
                if angle <= remaining + 0.1:
                    result.append(sign * angle)
                    remaining -= angle
                    used_any = True
                    break

            if not used_any:
                # Can't decompose exactly, use nearest
                if available:
                    result.append(sign * available[-1])
                break

        return result
