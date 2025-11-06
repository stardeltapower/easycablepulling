"""Route optimization module for cable pulling analysis.

This module optimizes cable routes by automatically splitting sections
to stay within tension and sidewall pressure limits.

KEY PRINCIPLES:
- Original section boundaries are PRESERVED (no merging across sections)
- Each section can be split into subsections
- User-defined friction is used throughout
- Calculations respect AEIC/CIGRE standards via config
"""

import math
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, NamedTuple
from enum import Enum

from ..core.models import Bend, Straight, Section, Route, CableSpec, DuctSpec
from ..calculations.config import CalculationConfig
from ..calculations.tension import calculate_straight_tension, calculate_bend_tension
from ..calculations.pressure import calculate_sidewall_pressure


class PullingDirection(Enum):
    """Pulling direction for optimization."""
    FORWARD = "forward"
    REVERSE = "reverse"


class PrimitiveResult(NamedTuple):
    """Result of calculating forces for a single primitive."""
    primitive: object  # Straight or Bend
    position: float  # Position along route (m)
    tension_in: float  # Tension at start of primitive (N)
    tension_out: float  # Tension at end of primitive (N)
    sidewall_pressure: float  # Sidewall pressure (N/m) - 0 for straights
    passes_limits: bool  # Whether this primitive passes limits


@dataclass
class OptimizedSection:
    """An optimized section with detailed analysis."""
    section_id: str  # SECT_XX or SECT_XX_YY
    original_section_id: str  # Original DXF section (e.g., "01", "02")
    subsection_number: int  # 0 if not split, else 1, 2, 3, etc.
    start_position: float  # Start position in overall route (m)
    end_position: float  # End position in overall route (m)
    length: float  # Section length (m)
    primitives: List[PrimitiveResult]  # Detailed primitive results

    # Peak values
    max_tension: float  # Maximum tension in section (N)
    max_sidewall_pressure: float  # Maximum sidewall pressure (N/m)

    # Direction-specific values (for displaying both options in reports)
    forward_tension: float  # Forward pulling tension (N)
    reverse_tension: float  # Reverse pulling tension (N)
    forward_sidewall: float  # Forward pulling max sidewall pressure (N/m)
    reverse_sidewall: float  # Reverse pulling max sidewall pressure (N/m)

    # Utilization ratios (0-1)
    tension_utilization: float
    sidewall_utilization: float

    # Pass/fail status
    passes_tension: bool
    passes_sidewall: bool
    overall_pass: bool

    # Junction labels for connectivity (A, B, C, etc.)
    start_junction: Optional[str] = None
    end_junction: Optional[str] = None

    # Warning status
    has_warning: bool = False
    warning_message: str = ""


@dataclass
class OptimizationResult:
    """Result of route optimization."""
    direction: PullingDirection
    original_sections: int
    optimized_sections: int
    total_length: float
    sections: List[OptimizedSection]
    max_tension: float
    max_sidewall_pressure: float
    max_tension_utilization: float
    max_sidewall_utilization: float
    all_sections_pass: bool
    feasible: bool
    target_utilization: float
    max_section_length: float


class RouteOptimizer:
    """Optimizes cable routes for pulling feasibility.

    Preserves original section boundaries - sections can be split but never merged.
    """

    def __init__(
        self,
        cable_spec: CableSpec,
        duct_spec: DuctSpec,
        target_utilization: float = 0.95,
        max_section_length: float = 500.0,
        config: Optional[CalculationConfig] = None,
    ):
        """Initialize route optimizer.

        Args:
            cable_spec: Cable specifications
            duct_spec: Duct specifications
            target_utilization: Target utilization ratio (0-1) for splitting (default 95%)
            max_section_length: Maximum section length in meters
            config: Calculation configuration (for AEIC/CIGRE standards, weight correction)
        """
        self.cable_spec = cable_spec
        self.duct_spec = duct_spec
        self.target_utilization = target_utilization
        self.max_section_length = max_section_length
        self.config = config if config is not None else CalculationConfig()

    def optimize_route(
        self,
        route: Route,
        direction: PullingDirection = PullingDirection.FORWARD,
        friction_override: Optional[float] = None,
    ) -> OptimizationResult:
        """Optimize a route for the specified pulling direction.

        IMPORTANT: Preserves original section boundaries. Each section is optimized
        independently and can be split into subsections, but sections are NEVER merged.

        Args:
            route: Route to optimize
            direction: Pulling direction (forward or reverse)
            friction_override: User-defined friction coefficient (if None, uses duct_spec.friction_dry)

        Returns:
            Optimization result with split sections
        """
        all_optimized_sections = []

        # Process each original section independently
        for section_idx, section in enumerate(route.sections):
            original_section_id = f"{section_idx + 1:02d}"

            # Get primitives for this section only
            section_primitives = list(section.primitives)
            if direction == PullingDirection.REVERSE:
                section_primitives = section_primitives[::-1]

            if not section_primitives:
                continue

            # Optimize this section (may split into subsections)
            # Pass original section to preserve junction labels
            subsections = self._optimize_single_section(
                section_primitives=section_primitives,
                original_section_id=original_section_id,
                friction_override=friction_override,
                original_section=section,
            )

            all_optimized_sections.extend(subsections)

        # Calculate summary statistics
        if not all_optimized_sections:
            return self._empty_result(route, direction)

        max_tension = max(s.max_tension for s in all_optimized_sections)
        max_sidewall = max(s.max_sidewall_pressure for s in all_optimized_sections)
        max_tension_util = max(s.tension_utilization for s in all_optimized_sections)
        max_sidewall_util = max(s.sidewall_utilization for s in all_optimized_sections)

        return OptimizationResult(
            direction=direction,
            original_sections=len(route.sections),
            optimized_sections=len(all_optimized_sections),
            total_length=sum(s.length for s in all_optimized_sections),
            sections=all_optimized_sections,
            max_tension=max_tension,
            max_sidewall_pressure=max_sidewall,
            max_tension_utilization=max_tension_util,
            max_sidewall_utilization=max_sidewall_util,
            all_sections_pass=all(s.overall_pass for s in all_optimized_sections),
            feasible=all(s.overall_pass for s in all_optimized_sections),
            target_utilization=self.target_utilization,
            max_section_length=self.max_section_length,
        )

    def _optimize_single_section(
        self,
        section_primitives: List[object],
        original_section_id: str,
        friction_override: Optional[float] = None,
        original_section: Optional[object] = None,
    ) -> List[OptimizedSection]:
        """Optimize a single section using BOTH methods and choose the best.

        Method 1: Equal splitting (divide into N equal parts, increase N until all pass)
        Method 2: Adaptive splitting (split at 80% threshold)

        Choose whichever gives fewer subsections.

        Args:
            section_primitives: Primitives for this section only
            original_section_id: Original section ID (e.g., "01", "02")
            friction_override: User-defined friction coefficient
            original_section: Original Section object (for junction labels)

        Returns:
            List of optimized subsections (length 1 if no split needed)
        """
        # Calculate tensions for full section
        primitive_results = self._calculate_primitive_results(
            section_primitives, friction_override
        )

        if not primitive_results:
            return []

        # Calculate section length
        section_length = sum(p.length() for p in section_primitives)

        # Check if section needs splitting (either exceeds limits OR exceeds max length)
        exceeds_limits = any(
            not result.passes_limits for result in primitive_results
        )
        exceeds_length = section_length > self.max_section_length
        needs_split = exceeds_limits or exceeds_length

        # Debug output for section analysis
        max_tension = max(r.tension_out for r in primitive_results) if primitive_results else 0
        max_sidewall = max(r.sidewall_pressure for r in primitive_results) if primitive_results else 0
        length_status = f"length={section_length:.0f}m (max={self.max_section_length:.0f}m)" if exceeds_length else f"length={section_length:.0f}m"
        print(f"    Section {original_section_id}: max_tension={max_tension/1000:.2f}kN, max_sidewall={max_sidewall:.0f}N/m, {length_status}, needs_split={needs_split}")

        if not needs_split:
            # Section passes as-is, return single section
            return self._create_section_from_primitives(
                section_primitives=section_primitives,
                original_section_id=original_section_id,
                subsection_number=0,
                friction_override=friction_override,
                original_section=original_section,
                total_subsections=1,
            )

        # Section exceeds limits - try BOTH methods

        # Method 1: Equal splitting
        equal_subsections = self._optimize_equal_splitting(
            section_primitives, original_section_id, friction_override, original_section
        )

        # Method 2: Adaptive splitting
        adaptive_subsections = self._optimize_adaptive_splitting(
            section_primitives, original_section_id, friction_override, original_section
        )

        # Check if subsections respect max_section_length constraint
        def all_subsections_valid(subsections, method_name=""):
            """Check if all subsections are within max_section_length."""
            for i, subsection in enumerate(subsections):
                if subsection.length > self.max_section_length:
                    print(f"    DEBUG: {method_name} subsection {i+1}/{len(subsections)} length={subsection.length:.0f}m exceeds max={self.max_section_length:.0f}m")
                    return False
            return True

        equal_valid = all_subsections_valid(equal_subsections, "Equal")
        adaptive_valid = all_subsections_valid(adaptive_subsections, "Adaptive")

        # Choose method that respects length constraint
        # Prefer valid over invalid
        # If both valid or both invalid, prefer more subsections (better chance of staying within limits)
        if equal_valid and not adaptive_valid:
            print(f"  Section {original_section_id}: Equal splitting wins ({len(equal_subsections)} vs {len(adaptive_subsections)} subsections, adaptive exceeds length limit)")
            return equal_subsections
        elif adaptive_valid and not equal_valid:
            print(f"  Section {original_section_id}: Adaptive splitting wins ({len(adaptive_subsections)} vs {len(equal_subsections)} subsections, equal exceeds length limit)")
            return adaptive_subsections
        elif not equal_valid and not adaptive_valid:
            # Both invalid - choose the one with MORE subsections (closer to satisfying constraint)
            # Equal splitting always tries more subsections when needed
            print(f"  Section {original_section_id}: Equal splitting wins ({len(equal_subsections)} vs {len(adaptive_subsections)} subsections, both exceed limits but equal is closer)")
            return equal_subsections
        elif len(equal_subsections) <= len(adaptive_subsections):
            print(f"  Section {original_section_id}: Equal splitting wins ({len(equal_subsections)} vs {len(adaptive_subsections)} subsections)")
            return equal_subsections
        else:
            print(f"  Section {original_section_id}: Adaptive splitting wins ({len(adaptive_subsections)} vs {len(equal_subsections)} subsections)")
            return adaptive_subsections

    def _optimize_equal_splitting(
        self,
        section_primitives: List[object],
        original_section_id: str,
        friction_override: Optional[float] = None,
        original_section: Optional[object] = None,
    ) -> List[OptimizedSection]:
        """Optimize using equal splitting method with bidirectional testing.

        For each N:
        1. Split into N equal subsections
        2. Test EACH subsection in both forward and reverse
        3. If at least one direction passes for a subsection → viable
        4. When all subsections have a passing direction, choose combination
           that minimizes total pulling tension

        Args:
            section_primitives: Primitives for this section
            original_section_id: Original section ID
            friction_override: User-defined friction coefficient
            original_section: Original Section object (for junction labels)

        Returns:
            List of subsections with optimal pulling directions (minimum N where all pass)
        """
        # Calculate minimum number of subsections needed to satisfy length constraint
        section_length = sum(p.length() for p in section_primitives)
        min_subsections_for_length = max(1, int(math.ceil(section_length / self.max_section_length)))

        # Try N=min_subsections, min_subsections+1, ... until all subsections have a passing direction
        for num_subsections in range(min_subsections_for_length, 21):  # Try up to 20 subsections
            result = self._test_equal_split_bidirectional(
                section_primitives,
                num_subsections,
                original_section_id,
                friction_override,
                original_section,
            )

            if result is not None:
                # Found N where all subsections have at least one passing direction
                return result

        # If even 20 subsections don't work, fall back to single-direction attempt
        return self._split_into_equal_subsections(
            section_primitives,
            20,
            original_section_id,
            friction_override,
            original_section,
        )

    def _test_equal_split_bidirectional(
        self,
        section_primitives: List[object],
        num_subsections: int,
        original_section_id: str,
        friction_override: Optional[float] = None,
        original_section: Optional[object] = None,
    ) -> Optional[List[OptimizedSection]]:
        """Test N equal subsections bidirectionally and choose optimal directions.

        For each subsection:
        - Test forward direction
        - Test reverse direction
        - If at least one passes → viable

        If all subsections have at least one passing direction:
        - Choose direction combination that minimizes total pulling tension
        - Return optimized subsections

        Args:
            section_primitives: Primitives for this section
            num_subsections: Number of subsections to test
            original_section_id: Original section ID
            friction_override: User-defined friction coefficient
            original_section: Original Section object (for junction labels)

        Returns:
            List of subsections with optimal directions, or None if not all subsections viable
        """
        # Split into N equal subsections
        # First get the split indices
        cumulative_lengths = [0.0]
        for primitive in section_primitives:
            prim_length = (
                primitive.length_m if isinstance(primitive, Straight)
                else primitive.radius_m * abs(primitive.angle_deg) * math.pi / 180
            )
            cumulative_lengths.append(cumulative_lengths[-1] + prim_length)

        total_length = cumulative_lengths[-1]
        target_length = total_length / num_subsections

        # Find split points
        split_indices = [0]
        for split_num in range(1, num_subsections):
            target_position = split_num * target_length
            best_idx = min(
                range(1, len(cumulative_lengths)),
                key=lambda idx: abs(cumulative_lengths[idx] - target_position)
            )
            if best_idx not in split_indices:
                split_indices.append(best_idx)

        split_indices = sorted(set(split_indices))
        if split_indices[-1] != len(section_primitives):
            split_indices.append(len(section_primitives))

        # Test each subsection in both directions
        subsection_options = []  # List of (forward_result, reverse_result) for each subsection

        for i in range(len(split_indices) - 1):
            start_idx = split_indices[i]
            end_idx = split_indices[i + 1]
            subsection_primitives = section_primitives[start_idx:end_idx]

            if not subsection_primitives:
                continue

            # Test forward
            forward_results = self._create_section_from_primitives(
                section_primitives=subsection_primitives,
                original_section_id=original_section_id,
                subsection_number=i + 1,
                friction_override=friction_override,
                original_section=original_section,
                total_subsections=num_subsections,
            )
            forward_section = forward_results[0] if forward_results else None

            # Test reverse (reverse the primitives)
            reverse_results = self._create_section_from_primitives(
                section_primitives=subsection_primitives[::-1],
                original_section_id=original_section_id,
                subsection_number=i + 1,
                friction_override=friction_override,
                original_section=original_section,
                total_subsections=num_subsections,
            )
            reverse_section = reverse_results[0] if reverse_results else None

            # Check if at least one direction passes
            forward_passes = forward_section and forward_section.passes_tension and forward_section.passes_sidewall
            reverse_passes = reverse_section and reverse_section.passes_tension and reverse_section.passes_sidewall

            if not forward_passes and not reverse_passes:
                # Neither direction passes for this subsection - N is not enough
                return None

            subsection_options.append((forward_section, reverse_section, forward_passes, reverse_passes))

        # All subsections have at least one passing direction!
        # Now choose the combination that minimizes total pulling tension

        # For each subsection, choose the direction with lower max tension
        # (This minimizes the peak tension across all subsections)
        optimal_subsections = []

        for i, (forward_sec, reverse_sec, fwd_passes, rev_passes) in enumerate(subsection_options):
            # Store both forward and reverse values for comparison
            # (even though we'll choose the optimal direction)
            fwd_tension = forward_sec.max_tension if forward_sec else 0
            rev_tension = reverse_sec.max_tension if reverse_sec else 0
            fwd_sidewall = forward_sec.max_sidewall_pressure if forward_sec else 0
            rev_sidewall = reverse_sec.max_sidewall_pressure if reverse_sec else 0

            if fwd_passes and rev_passes:
                # Both pass - choose one with lower peak tension
                if forward_sec.max_tension <= reverse_sec.max_tension:
                    chosen = forward_sec
                else:
                    # Use reverse section's calculations but forward section's primitives
                    # This keeps primitives in geographical order
                    chosen = reverse_sec
                    chosen.primitives = forward_sec.primitives  # Use geographical order!
            elif fwd_passes:
                chosen = forward_sec
            else:
                # Use reverse section's calculations but forward section's primitives
                # This keeps primitives in geographical order
                chosen = reverse_sec
                chosen.primitives = forward_sec.primitives  # Use geographical order!

            # Store both forward and reverse values for reporting
            chosen.forward_tension = fwd_tension
            chosen.reverse_tension = rev_tension
            chosen.forward_sidewall = fwd_sidewall
            chosen.reverse_sidewall = rev_sidewall

            # Update section ID and subsection number (no direction suffix)
            if num_subsections > 1:
                chosen.section_id = f"SECT_{original_section_id}_{i+1:02d}"
                chosen.subsection_number = i + 1
            else:
                chosen.section_id = f"SECT_{original_section_id}"
                chosen.subsection_number = 0

            optimal_subsections.append(chosen)

        # Subsections are in split order, which follows DXF route order
        # Don't reorder - maintain DXF sequence

        # Renumber subsections in order
        for i, subsection in enumerate(optimal_subsections):
            if num_subsections > 1:
                subsection.section_id = f"SECT_{original_section_id}_{i+1:02d}"
                subsection.subsection_number = i + 1
            else:
                subsection.section_id = f"SECT_{original_section_id}"
                subsection.subsection_number = 0

        return optimal_subsections

    def _split_into_equal_subsections(
        self,
        section_primitives: List[object],
        num_subsections: int,
        original_section_id: str,
        friction_override: Optional[float] = None,
        original_section: Optional[object] = None,
    ) -> List[OptimizedSection]:
        """Split section into N equal-length subsections.

        Uses optimal algorithm: finds primitive boundaries CLOSEST to target lengths
        to minimize subsection length differences (maximize balance).

        Args:
            section_primitives: Primitives for this section
            num_subsections: Number of subsections to create
            original_section_id: Original section ID
            friction_override: User-defined friction coefficient
            original_section: Original Section object (for junction labels)

        Returns:
            List of N subsections with maximally balanced lengths
        """
        # Build cumulative length array at each primitive boundary
        cumulative_lengths = [0.0]
        for primitive in section_primitives:
            prim_length = (
                primitive.length_m if isinstance(primitive, Straight)
                else primitive.radius_m * abs(primitive.angle_deg) * math.pi / 180
            )
            cumulative_lengths.append(cumulative_lengths[-1] + prim_length)

        total_length = cumulative_lengths[-1]
        target_length = total_length / num_subsections

        # Find split points that minimize deviation from equal lengths
        # For each target position, find the primitive boundary CLOSEST to it
        split_indices = [0]

        for split_num in range(1, num_subsections):
            target_position = split_num * target_length

            # Find primitive boundary closest to this target position
            best_idx = min(
                range(1, len(cumulative_lengths)),
                key=lambda idx: abs(cumulative_lengths[idx] - target_position)
            )

            # Ensure we don't duplicate split indices (can happen with very short primitives)
            if best_idx not in split_indices:
                split_indices.append(best_idx)

        # Ensure split_indices is sorted and includes the end
        split_indices = sorted(set(split_indices))
        if split_indices[-1] != len(section_primitives):
            split_indices.append(len(section_primitives))

        # Create subsections
        subsections = []
        for i in range(len(split_indices) - 1):
            start_idx = split_indices[i]
            end_idx = split_indices[i + 1]

            subsection_primitives = section_primitives[start_idx:end_idx]
            if not subsection_primitives:
                continue

            subsection = self._create_section_from_primitives(
                section_primitives=subsection_primitives,
                original_section_id=original_section_id,
                subsection_number=i + 1 if num_subsections > 1 else 0,
                friction_override=friction_override,
                original_section=original_section,
                total_subsections=num_subsections,
            )

            subsections.extend(subsection)

        # Update IDs
        if len(subsections) == 1:
            subsections[0].subsection_number = 0
            subsections[0].section_id = f"SECT_{original_section_id}"
        else:
            for i, subsection in enumerate(subsections):
                subsection.subsection_number = i + 1
                subsection.section_id = f"SECT_{original_section_id}_{i+1:02d}"

        return subsections

    def _optimize_adaptive_splitting(
        self,
        section_primitives: List[object],
        original_section_id: str,
        friction_override: Optional[float] = None,
        original_section: Optional[object] = None,
    ) -> List[OptimizedSection]:
        """Optimize using adaptive splitting method (split at 80% threshold).

        Args:
            section_primitives: Primitives for this section
            original_section_id: Original section ID
            friction_override: User-defined friction coefficient
            original_section: Original Section object (for junction labels)

        Returns:
            List of subsections
        """
        # Find split points based on tension/pressure thresholds
        split_points = self._find_split_points(
            section_primitives, friction_override
        )

        # Create subsections - test both forward and reverse directions for each
        subsections = []
        total_subsections = len(split_points) - 1

        for i in range(total_subsections):
            start_idx = split_points[i]
            end_idx = split_points[i + 1]

            subsection_primitives = section_primitives[start_idx:end_idx]
            if not subsection_primitives:
                continue

            # Test forward direction
            forward_results = self._create_section_from_primitives(
                section_primitives=subsection_primitives,
                original_section_id=original_section_id,
                subsection_number=i + 1,
                friction_override=friction_override,
                original_section=original_section,
                total_subsections=total_subsections,
            )
            forward_section = forward_results[0] if forward_results else None

            # Test reverse direction (reverse the primitives)
            reverse_results = self._create_section_from_primitives(
                section_primitives=subsection_primitives[::-1],
                original_section_id=original_section_id,
                subsection_number=i + 1,
                friction_override=friction_override,
                original_section=original_section,
                total_subsections=total_subsections,
            )
            reverse_section = reverse_results[0] if reverse_results else None

            # Check which directions pass
            forward_passes = forward_section and forward_section.passes_tension and forward_section.passes_sidewall
            reverse_passes = reverse_section and reverse_section.passes_tension and reverse_section.passes_sidewall

            # Store both forward and reverse values for comparison
            fwd_tension = forward_section.max_tension if forward_section else 0
            rev_tension = reverse_section.max_tension if reverse_section else 0
            fwd_sidewall = forward_section.max_sidewall_pressure if forward_section else 0
            rev_sidewall = reverse_section.max_sidewall_pressure if reverse_section else 0

            # Choose the better direction
            if forward_passes and reverse_passes:
                # Both pass - choose one with lower peak tension
                if forward_section.max_tension <= reverse_section.max_tension:
                    chosen = forward_section
                else:
                    # Use reverse section's calculations but forward section's primitives
                    chosen = reverse_section
                    chosen.primitives = forward_section.primitives  # Use geographical order!
            elif forward_passes:
                chosen = forward_section
            elif reverse_passes:
                # Use reverse section's calculations but forward section's primitives
                chosen = reverse_section
                chosen.primitives = forward_section.primitives  # Use geographical order!
            else:
                # Neither passes - use forward section
                chosen = forward_section

            # Store both forward and reverse values for reporting
            chosen.forward_tension = fwd_tension
            chosen.reverse_tension = rev_tension
            chosen.forward_sidewall = fwd_sidewall
            chosen.reverse_sidewall = rev_sidewall

            subsections.append(chosen)

        # Update IDs
        if len(subsections) == 1:
            subsections[0].subsection_number = 0
            subsections[0].section_id = f"SECT_{original_section_id}"
        else:
            for i, subsection in enumerate(subsections):
                subsection.subsection_number = i + 1
                subsection.section_id = f"SECT_{original_section_id}_{i+1:02d}"

        return subsections

    def _find_split_points(
        self,
        section_primitives: List[object],
        friction_override: Optional[float] = None,
    ) -> List[int]:
        """Find split points for a section that exceeds limits.

        Uses target_utilization as preferred split point, but will split earlier
        if 100% limit is exceeded.

        Args:
            section_primitives: Primitives for this section
            friction_override: User-defined friction coefficient

        Returns:
            List of split indices (includes 0 and len(primitives))
        """
        split_points = [0]
        current_start = 0

        while current_start < len(section_primitives):
            # Calculate from current start point
            remaining_primitives = section_primitives[current_start:]
            results = self._calculate_primitive_results(
                remaining_primitives, friction_override
            )

            if not results:
                break

            # Find where to split
            split_idx = None
            preferred_split = None  # Target utilization split
            length_split = None  # Length-based split
            cumulative_length = 0.0

            for i, result in enumerate(results):
                # Track cumulative length
                cumulative_length += remaining_primitives[i].length()

                tension_util = result.tension_out / self.cable_spec.max_tension
                sidewall_util = result.sidewall_pressure / self.cable_spec.max_sidewall_pressure

                # Mark preferred split at target utilization
                if (tension_util >= self.target_utilization or
                    sidewall_util >= self.target_utilization) and preferred_split is None:
                    preferred_split = i

                # Mark split if exceeding length limit
                if cumulative_length > self.max_section_length and length_split is None:
                    length_split = max(1, i)

                # Force split if exceeding 100%
                if tension_util > 1.0 or sidewall_util > 1.0:
                    split_idx = preferred_split if preferred_split is not None else max(1, i)
                    break

            # If we found a preferred split but no forced split, use the preferred split
            if split_idx is None and preferred_split is not None:
                split_idx = preferred_split

            # If no split found but length exceeded, use length split
            if split_idx is None and length_split is not None:
                split_idx = length_split

            if split_idx is None:
                # Rest of section is OK
                break

            # Find nearest straight to split at
            split_idx = self._find_nearest_straight(remaining_primitives, split_idx)
            global_split_idx = current_start + split_idx

            if global_split_idx > current_start:
                split_points.append(global_split_idx)
                current_start = global_split_idx
            else:
                # Can't split, force after first primitive
                split_points.append(current_start + 1)
                current_start += 1

        # Add end point
        if split_points[-1] != len(section_primitives):
            split_points.append(len(section_primitives))

        return split_points

    def _find_nearest_straight(
        self, primitives: List[object], target_idx: int
    ) -> int:
        """Find nearest straight segment to target index.

        Args:
            primitives: List of primitives
            target_idx: Target index to split near

        Returns:
            Index of nearest straight (or target_idx if none found)
        """
        # Look backwards from target
        for i in range(target_idx - 1, -1, -1):
            if isinstance(primitives[i], Straight):
                return i

        # Look forward from target
        for i in range(target_idx, len(primitives)):
            if isinstance(primitives[i], Straight):
                return i

        # No straight found, return target
        return max(1, target_idx)

    def _create_section_from_primitives(
        self,
        section_primitives: List[object],
        original_section_id: str,
        subsection_number: int,
        friction_override: Optional[float] = None,
        original_section: Optional[object] = None,
        total_subsections: int = 1,
    ) -> List[OptimizedSection]:
        """Create optimized section from primitives.

        Args:
            section_primitives: Primitives for this section
            original_section_id: Original section ID
            subsection_number: Subsection number (0 if not split)
            friction_override: User-defined friction coefficient
            original_section: Original Section object (for junction labels)
            total_subsections: Total number of subsections (for junction labeling)

        Returns:
            List containing single OptimizedSection
        """
        # Recalculate tensions from 0
        results = self._calculate_primitive_results(
            section_primitives, friction_override
        )

        if not results:
            return []

        # Calculate metrics
        length = sum(
            p.length_m if isinstance(p, Straight)
            else p.radius_m * abs(p.angle_deg) * math.pi / 180
            for p in section_primitives
        )

        max_tension = max(r.tension_out for r in results)
        max_sidewall = max(r.sidewall_pressure for r in results)

        tension_util = max_tension / self.cable_spec.max_tension
        sidewall_util = max_sidewall / self.cable_spec.max_sidewall_pressure

        passes_tension = max_tension <= self.cable_spec.max_tension
        passes_sidewall = max_sidewall <= self.cable_spec.max_sidewall_pressure

        has_warning = tension_util > 0.9 or sidewall_util > 0.9
        warning_message = ""
        if has_warning:
            warnings = []
            if tension_util > 0.9:
                warnings.append(f"tension {tension_util*100:.1f}%")
            if sidewall_util > 0.9:
                warnings.append(f"sidewall {sidewall_util*100:.1f}%")
            warning_message = f"High utilization: {', '.join(warnings)}"

        # Generate section ID (will be updated by caller if needed)
        if subsection_number == 0:
            section_id = f"SECT_{original_section_id}"
        else:
            section_id = f"SECT_{original_section_id}_{subsection_number:02d}"

        # Compute junction labels
        start_junction = None
        end_junction = None
        if original_section and hasattr(original_section, 'start_junction') and hasattr(original_section, 'end_junction'):
            if total_subsections == 1:
                # Not split - use original junctions
                start_junction = original_section.start_junction
                end_junction = original_section.end_junction
            else:
                # Split into multiple subsections - only use real junctions at section boundaries
                # Subsections in the middle don't get junction labels (will show coordinates only)
                orig_start = original_section.start_junction
                orig_end = original_section.end_junction

                if subsection_number == 1:
                    # First subsection: starts at original start junction
                    start_junction = orig_start
                    end_junction = None  # No junction label for split point
                elif subsection_number == total_subsections:
                    # Last subsection: ends at original end junction
                    start_junction = None  # No junction label for split point
                    end_junction = orig_end
                else:
                    # Middle subsection: no junction labels
                    start_junction = None
                    end_junction = None

        return [OptimizedSection(
            section_id=section_id,
            original_section_id=original_section_id,
            subsection_number=subsection_number,
            start_position=0.0,
            end_position=length,
            length=length,
            primitives=results,
            max_tension=max_tension,
            max_sidewall_pressure=max_sidewall,
            # Initially set to same values - will be updated when comparing forward/reverse
            forward_tension=max_tension,
            reverse_tension=max_tension,
            forward_sidewall=max_sidewall,
            reverse_sidewall=max_sidewall,
            tension_utilization=tension_util,
            sidewall_utilization=sidewall_util,
            passes_tension=passes_tension,
            passes_sidewall=passes_sidewall,
            overall_pass=(passes_tension and passes_sidewall),
            start_junction=start_junction,
            end_junction=end_junction,
            has_warning=has_warning,
            warning_message=warning_message,
        )]

    def _calculate_primitive_results(
        self,
        primitives: List[object],
        friction_override: Optional[float] = None,
    ) -> List[PrimitiveResult]:
        """Calculate tension and pressure for each primitive.

        Uses user-defined friction throughout all calculations.

        Args:
            primitives: List of primitives in order
            friction_override: User-defined friction coefficient

        Returns:
            List of primitive results with tensions and pressures
        """
        results = []
        current_tension = 0.0  # Start with zero tension
        current_position = 0.0

        # Determine friction coefficient to use
        # If friction_override provided, it should already include trefoil adjustment (e.g., 0.39)
        # Otherwise, get it from duct spec with automatic trefoil multiplier
        if friction_override is not None:
            # friction_override is the FINAL friction to use (already includes trefoil 1.3× if applicable)
            # For trefoil with base 0.3: friction_override should be 0.39
            friction = friction_override
        else:
            # Get base friction and let get_friction() apply trefoil multiplier
            friction = self.duct_spec.get_friction(self.cable_spec.arrangement, lubricated=True)

        for primitive in primitives:
            if isinstance(primitive, Straight):
                # Calculate straight section tension using explicit friction
                # FORMULA: T_out = T_in + (μ × w_c × L)
                tension_out = calculate_straight_tension(
                    tension_in=current_tension,
                    cable_spec=self.cable_spec,
                    duct_spec=self.duct_spec,
                    length=primitive.length_m,
                    config=self.config,  # Pass config for weight corrections (WCF)
                    friction_override=friction,  # Use explicit friction (bypasses get_friction)
                )

                sidewall_pressure = 0.0  # No sidewall pressure in straights
                current_position += primitive.length_m

            elif isinstance(primitive, Bend):
                # Calculate bend tension using explicit friction
                # FORMULA: T_out = T_in × e^(μ × α)
                tension_out = calculate_bend_tension(
                    tension_in=current_tension,
                    cable_spec=self.cable_spec,
                    duct_spec=self.duct_spec,
                    bend_angle=primitive.angle_deg,
                    friction_override=friction,  # Use explicit friction (bypasses get_friction)
                )

                # Calculate sidewall pressure using AEIC/CIGRE formula
                # FORMULA (AEIC trefoil): P = (WCF × T_out) / (2 × r)
                sidewall_pressure = calculate_sidewall_pressure(
                    tension=tension_out,  # Uses T_OUT (exit tension) for max pressure
                    bend_radius=primitive.radius_m,
                    cable_spec=self.cable_spec,
                    duct_spec=self.duct_spec,
                    config=self.config,  # Respects AEIC vs CIGRE standards
                )

                # Add bend length to position
                bend_length = primitive.radius_m * math.radians(abs(primitive.angle_deg))
                current_position += bend_length
            else:
                continue

            # Check if within limits
            passes_tension = tension_out <= self.cable_spec.max_tension
            passes_sidewall = sidewall_pressure <= self.cable_spec.max_sidewall_pressure

            results.append(
                PrimitiveResult(
                    primitive=primitive,
                    position=current_position,
                    tension_in=current_tension,
                    tension_out=tension_out,
                    sidewall_pressure=sidewall_pressure,
                    passes_limits=(passes_tension and passes_sidewall),
                )
            )

            # Update current tension for next primitive
            current_tension = tension_out

        return results

    def _empty_result(
        self, route: Route, direction: PullingDirection
    ) -> OptimizationResult:
        """Create an empty optimization result.

        Args:
            route: Original route
            direction: Pulling direction

        Returns:
            Empty optimization result
        """
        return OptimizationResult(
            direction=direction,
            original_sections=len(route.sections) if route.sections else 0,
            optimized_sections=0,
            total_length=0.0,
            sections=[],
            max_tension=0.0,
            max_sidewall_pressure=0.0,
            max_tension_utilization=0.0,
            max_sidewall_utilization=0.0,
            all_sections_pass=False,
            feasible=False,
            target_utilization=self.target_utilization,
            max_section_length=self.max_section_length,
        )


def optimize_cable_route(
    route: Route,
    cable_spec: CableSpec,
    duct_spec: DuctSpec,
    target_utilization: float = 0.8,
    max_section_length: float = 500.0,
    friction_override: Optional[float] = None,
    config: Optional[CalculationConfig] = None,
) -> Tuple[OptimizationResult, OptimizationResult]:
    """Optimize a cable route for both pulling directions.

    Args:
        route: Route to optimize
        cable_spec: Cable specifications
        duct_spec: Duct specifications
        target_utilization: Target utilization ratio (default 80%)
        max_section_length: Maximum section length in meters
        friction_override: User-defined friction coefficient
        config: Calculation configuration (for AEIC/CIGRE standards)

    Returns:
        Tuple of (forward_result, reverse_result)
    """
    optimizer = RouteOptimizer(
        cable_spec=cable_spec,
        duct_spec=duct_spec,
        target_utilization=target_utilization,
        max_section_length=max_section_length,
        config=config,
    )

    # Optimize for forward pulling
    forward_result = optimizer.optimize_route(
        route=route,
        direction=PullingDirection.FORWARD,
        friction_override=friction_override,
    )

    # Optimize for reverse pulling
    reverse_result = optimizer.optimize_route(
        route=route,
        direction=PullingDirection.REVERSE,
        friction_override=friction_override,
    )

    return forward_result, reverse_result
