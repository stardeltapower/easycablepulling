"""Section-by-section optimization for cable pulling analysis.

This module optimizes each route section (polyline) independently by:
1. Testing if the section fits within pulling limits
2. If not, splitting it into equal subsections and testing each
3. Iteratively increasing the number of subsections until all pass
"""

from dataclasses import dataclass
from typing import List, Tuple
import math

from ..core.models import Section, CableSpec, DuctSpec, Route
from ..calculations.tension import calculate_straight_tension, calculate_bend_tension


@dataclass
class SubsectionResult:
    """Result for a single subsection of a route section."""
    section_index: int  # Original section index
    subsection_num: int  # Which subsection (1, 2, 3, ...)
    total_subsections: int  # Total number this was split into
    start_position: float  # Start position in original section
    end_position: float  # End position in original section
    length: float  # Length of subsection
    max_tension: float  # Maximum tension in subsection
    max_sidewall_pressure: float  # Maximum sidewall pressure
    passes_limits: bool  # Whether it passes both limits


class SectionOptimizer:
    """Optimizes each route section independently."""

    def __init__(
        self,
        cable_spec: CableSpec,
        duct_spec: DuctSpec,
        max_tension_limit: float,
        max_sidewall_limit: float,
        max_section_length: float = 500.0,
        friction_override: float = None,
    ):
        """Initialize section optimizer.

        Args:
            cable_spec: Cable specifications
            duct_spec: Duct specifications
            max_tension_limit: Maximum allowed tension (N)
            max_sidewall_limit: Maximum allowed sidewall pressure (N/m)
            max_section_length: Maximum length before requiring split (m)
            friction_override: Optional friction coefficient override
        """
        self.cable_spec = cable_spec
        self.duct_spec = duct_spec
        self.max_tension_limit = max_tension_limit
        self.max_sidewall_limit = max_sidewall_limit
        self.max_section_length = max_section_length
        self.friction = friction_override if friction_override else duct_spec.friction_dry

    def optimize_route(self, route: Route) -> List[SubsectionResult]:
        """Optimize each section of the route independently.

        Args:
            route: Route with independent sections/polylines

        Returns:
            List of all subsection results (passing and failing)
        """
        all_results = []

        for section_idx, section in enumerate(route.sections):
            # Get subsections for this section
            subsections = self._optimize_section(section, section_idx)
            all_results.extend(subsections)

        return all_results

    def calculate_max_straight_length(self) -> Tuple[float, str]:
        """Calculate the maximum length for a straight cable pull.

        Tests how far a cable can be pulled through a straight duct before hitting
        either tension or sidewall pressure limits. This represents the theoretical
        maximum section length in ideal conditions (no bends).

        Returns:
            Tuple of (max_length_m, limiting_factor)
            limiting_factor is either "tension" or "sidewall_pressure"
        """
        from ..core.models import Straight

        # Start with a reasonable test length and increase until we hit a limit
        current_tension = 0.0
        current_length = 0.0
        increment = 10.0  # Test 10m increments
        max_iterations = 10000  # Safety limit (~100km)
        iteration = 0

        while iteration < max_iterations:
            # Test adding another increment
            test_length = current_length + increment

            # Calculate tension for a straight section of this length
            test_tension = calculate_straight_tension(
                current_tension,
                self.cable_spec,
                self.duct_spec,
                increment,
                lubricated=False,
                slope_angle=0.0,
            )

            # Check if we exceed limits
            if test_tension > self.max_tension_limit:
                # Tension limit exceeded - binary search for exact point
                return self._find_max_straight_length_binary(
                    current_tension, current_length, "tension"
                )

            # For straight sections, sidewall is only at bends, so no sidewall limit
            # But we should still check if we've hit tension limit
            current_tension = test_tension
            current_length = test_length
            iteration += 1

        # Reached max iterations without hitting tension limit
        return current_length, "tension"

    def _find_max_straight_length_binary(
        self, start_tension: float, start_length: float, limiting_factor: str
    ) -> Tuple[float, str]:
        """Binary search to find exact maximum straight length.

        Args:
            start_tension: Tension at start of search region
            start_length: Length at start of search region
            limiting_factor: Which limit we're searching for

        Returns:
            Tuple of (max_length, limiting_factor)
        """
        low = start_length
        high = start_length + 5000.0  # Upper bound of 5km
        best_safe_length = start_length

        while high - low > 1.0:  # Search to 1m precision
            mid = (low + high) / 2.0
            test_increment = mid - start_length

            # Calculate tension at this length
            test_tension = calculate_straight_tension(
                start_tension,
                self.cable_spec,
                self.duct_spec,
                test_increment,
                lubricated=False,
                slope_angle=0.0,
            )

            if test_tension <= self.max_tension_limit:
                best_safe_length = mid
                low = mid
            else:
                high = mid

        return best_safe_length, limiting_factor

    def _optimize_section(self, section: Section, section_idx: int) -> List[SubsectionResult]:
        """Optimize a single section by iteratively splitting at violation points.

        Algorithm:
        1. Test section from start - find where limits are first exceeded
        2. Back up to last safe point and create a subsection
        3. Continue with remaining section from that point
        4. Repeat until entire section is covered
        5. Then try to rebalance lengths to be more equal

        Args:
            section: The route section to optimize
            section_idx: Index of this section in the route

        Returns:
            List of subsection results
        """
        # Find split points by iteratively finding safe sections
        split_points = self._find_safe_split_points(section)

        # Create subsections at those split points
        results = self._create_subsections_at_splits(
            section, section_idx, split_points
        )

        # Rebalance subsection lengths to be more equal IF all pass
        if len(results) > 1 and all(r.passes_limits for r in results):
            rebalanced = self._rebalance_subsection_lengths(section, section_idx, results)
            # Only use rebalanced if all subsections still pass
            if all(r.passes_limits for r in rebalanced):
                results = rebalanced

        return results

    def _find_safe_split_points(self, section: Section) -> List[float]:
        """Find split points where subsections are guaranteed to pass limits.

        Iteratively finds the furthest distance we can go while staying within limits,
        then creates a subsection up to that point. Repeats for remaining section.

        Args:
            section: The route section to analyze

        Returns:
            List of positions (in meters) where splits should occur
        """
        split_points = [0.0]  # Always start at beginning
        current_start_pos = 0.0
        section_length = section.original_length

        while current_start_pos < section_length:
            # Find the furthest position we can reach while staying within limits
            safe_end_pos = self._find_furthest_safe_position(
                section, current_start_pos, section_length
            )

            if safe_end_pos is None or safe_end_pos <= current_start_pos:
                # Can't find any safe distance - this section is problematic
                # Include at least one primitive
                prims = self._extract_primitives_in_range(
                    section.primitives, current_start_pos, section_length
                )
                if prims:
                    current_start_pos += prims[0].length()
                    split_points.append(current_start_pos)
                else:
                    break
            else:
                # Split at safe position
                split_points.append(safe_end_pos)
                current_start_pos = safe_end_pos

        # Remove duplicates and sort
        split_points = sorted(set(split_points))

        # Ensure we have end point
        if split_points[-1] != section_length:
            split_points.append(section_length)

        return split_points

    def _find_furthest_safe_position(
        self, section: Section, start_pos: float, end_pos: float
    ) -> float:
        """Find the furthest position from start_pos that stays within limits.

        Does a binary search to find the maximum distance we can go.

        Args:
            section: The route section
            start_pos: Starting position (m)
            end_pos: Maximum position to consider (m)

        Returns:
            Position that stays within limits, or None if no safe distance
        """
        # First check if even a tiny subsection passes
        min_test_length = 10.0  # Minimum 10m to test
        test_prims = self._extract_primitives_in_range(
            section.primitives, start_pos, start_pos + min_test_length
        )
        if test_prims:
            max_t, max_sw = self._calculate_subsection_limits(test_prims)
            if max_t > self.max_tension_limit or max_sw > self.max_sidewall_limit:
                # Even first 10m fails - return None
                return None

        # Binary search for maximum safe distance
        low = start_pos + min_test_length
        high = end_pos
        best_safe = None

        while high - low > 1.0:  # Search to 1m precision
            mid = (low + high) / 2.0

            test_prims = self._extract_primitives_in_range(
                section.primitives, start_pos, mid
            )

            if not test_prims:
                high = mid
                continue

            max_t, max_sw = self._calculate_subsection_limits(test_prims)

            # Check if this passes limits
            if max_t <= self.max_tension_limit and max_sw <= self.max_sidewall_limit:
                # This distance is safe - try going further
                best_safe = mid
                low = mid
            else:
                # This distance exceeded limits - back off
                high = mid

        return best_safe

    def _find_first_violation_position(
        self, primitives, start_pos: float
    ) -> float:
        """Find the position where limits are first violated.

        Walks through primitives from the beginning, accumulating tensions,
        and returns the position where any limit is exceeded.

        Args:
            primitives: List of primitives to test (should be from start of section)
            start_pos: Starting position offset (to add to relative positions)

        Returns:
            Absolute position where violation occurs, or None if no violation
        """
        from ..core.models import Straight, Bend

        current_tension = 0.0
        cumulative_pos = 0.0

        for primitive in primitives:
            # Calculate tension after this primitive
            if isinstance(primitive, Straight):
                current_tension = calculate_straight_tension(
                    current_tension,
                    self.cable_spec,
                    self.duct_spec,
                    primitive.length_m,
                    lubricated=False,
                )
            elif isinstance(primitive, Bend):
                current_tension = calculate_bend_tension(
                    current_tension,
                    self.cable_spec,
                    self.duct_spec,
                    primitive.angle_deg,
                    lubricated=False,
                )

            cumulative_pos += primitive.length()
            absolute_pos = start_pos + cumulative_pos

            # Check tension limit
            if current_tension > self.max_tension_limit:
                return absolute_pos

            # Check sidewall pressure at bends
            if isinstance(primitive, Bend) and primitive.radius_m > 0:
                sidewall = current_tension / primitive.radius_m
                if sidewall > self.max_sidewall_limit:
                    return absolute_pos

        return None

    def _create_subsections_at_splits(
        self, section: Section, section_idx: int, split_points: List[float]
    ) -> List[SubsectionResult]:
        """Create subsection results at specified split points.

        Args:
            section: The route section
            section_idx: Index of this section
            split_points: List of positions where splits occur

        Returns:
            List of subsection results
        """
        results = []

        for i in range(len(split_points) - 1):
            start_pos = split_points[i]
            end_pos = split_points[i + 1]

            # Extract primitives in this range
            sub_prims = self._extract_primitives_in_range(
                section.primitives, start_pos, end_pos
            )

            if not sub_prims:
                continue

            # Calculate limits for this subsection
            max_tension, max_sidewall = self._calculate_subsection_limits(sub_prims)

            # Check if it passes
            passes = (max_tension <= self.max_tension_limit
                     and max_sidewall <= self.max_sidewall_limit)

            results.append(
                SubsectionResult(
                    section_index=section_idx,
                    subsection_num=i + 1,
                    total_subsections=len(split_points) - 1,
                    start_position=start_pos,
                    end_position=end_pos,
                    length=end_pos - start_pos,
                    max_tension=max_tension,
                    max_sidewall_pressure=max_sidewall,
                    passes_limits=passes,
                )
            )

        # Update subsection numbers after creating all
        for i, result in enumerate(results):
            result.subsection_num = i + 1
            result.total_subsections = len(results)

        return results

    def _rebalance_subsection_lengths(
        self, section: Section, section_idx: int, initial_results: List[SubsectionResult]
    ) -> List[SubsectionResult]:
        """Rebalance subsection lengths to be more equal.

        Takes subsections that may have unequal lengths (e.g., 300m, 300m, 300m, 50m)
        and adjusts split points to make them more equal while maintaining that all pass limits.

        Also collapses any subsections smaller than the minimum into adjacent sections.

        Args:
            section: The route section
            section_idx: Index of section
            initial_results: Initial subsection results

        Returns:
            Rebalanced subsection results
        """
        if len(initial_results) < 2:
            return initial_results  # Nothing to rebalance

        # First, merge any subsections that are too small (< 20m)
        merged_results = self._merge_small_subsections(initial_results, min_length=20.0)

        # If merging created failures, keep original
        if not all(r.passes_limits for r in merged_results):
            return initial_results

        if len(merged_results) < 2:
            return merged_results

        # Calculate target length for even distribution
        section_length = section.original_length
        num_subsections = len(merged_results)
        target_length = section_length / num_subsections

        # Create new split points at approximately equal distances
        new_split_points = [
            i * target_length for i in range(num_subsections + 1)
        ]
        new_split_points[0] = 0.0  # Ensure start
        new_split_points[-1] = section_length  # Ensure end

        # Snap split points to nearby primitives
        new_split_points = self._snap_split_points_to_primitives(
            section.primitives, new_split_points
        )

        # Create results with new split points
        rebalanced = self._create_subsections_at_splits(
            section, section_idx, new_split_points
        )

        # If all still pass, use rebalanced version; otherwise keep merged original
        if all(r.passes_limits for r in rebalanced):
            return rebalanced
        else:
            return merged_results

    def _merge_small_subsections(
        self, subsections: List[SubsectionResult], min_length: float
    ) -> List[SubsectionResult]:
        """Merge subsections that are smaller than minimum length with neighbors.

        Args:
            subsections: List of subsection results
            min_length: Minimum acceptable subsection length (m)

        Returns:
            List of subsections with small ones merged
        """
        if len(subsections) <= 1:
            return subsections

        # Identify which are too small
        merged = []
        pending_length = 0.0
        pending_tension = 0.0
        pending_sidewall = 0.0

        for i, sub in enumerate(subsections):
            if sub.length < min_length and i < len(subsections) - 1:
                # Accumulate this subsection into pending
                pending_length += sub.length
                pending_tension = max(pending_tension, sub.max_tension)
                pending_sidewall = max(pending_sidewall, sub.max_sidewall_pressure)
                continue

            # This one is large enough - flush pending + this one
            if pending_length > 0:
                # Merge pending with this subsection
                merged_length = pending_length + sub.length
                merged_tension = max(pending_tension, sub.max_tension)
                merged_sidewall = max(pending_sidewall, sub.max_sidewall_pressure)
                merged_passes = (merged_tension <= self.max_tension_limit
                                and merged_sidewall <= self.max_sidewall_limit)

                merged.append(
                    SubsectionResult(
                        section_index=sub.section_index,
                        subsection_num=len(merged) + 1,
                        total_subsections=0,  # Will be updated below
                        start_position=sub.start_position - pending_length,
                        end_position=sub.end_position,
                        length=merged_length,
                        max_tension=merged_tension,
                        max_sidewall_pressure=merged_sidewall,
                        passes_limits=merged_passes,
                    )
                )
                pending_length = 0.0
                pending_tension = 0.0
                pending_sidewall = 0.0
            else:
                merged.append(sub)

        # Handle any remaining pending (last subsection is too small)
        if pending_length > 0 and merged:
            last = merged[-1]
            merged[-1] = SubsectionResult(
                section_index=last.section_index,
                subsection_num=last.subsection_num,
                total_subsections=0,
                start_position=last.start_position,
                end_position=last.end_position + pending_length,
                length=last.length + pending_length,
                max_tension=max(last.max_tension, pending_tension),
                max_sidewall_pressure=max(last.max_sidewall_pressure, pending_sidewall),
                passes_limits=(max(last.max_tension, pending_tension) <= self.max_tension_limit
                              and max(last.max_sidewall_pressure, pending_sidewall) <= self.max_sidewall_limit),
            )

        # Update total subsections count
        for i, sub in enumerate(merged):
            sub.subsection_num = i + 1
            sub.total_subsections = len(merged)

        return merged

    def _snap_split_points_to_primitives(
        self, primitives, split_points: List[float]
    ) -> List[float]:
        """Snap split points to nearby primitive boundaries.

        Args:
            primitives: List of primitives in section
            split_points: Target split points

        Returns:
            Adjusted split points aligned to primitives
        """
        if not primitives:
            return split_points

        # Build map of primitive positions
        primitive_positions = []
        cumulative = 0.0
        for prim in primitives:
            primitive_positions.append((cumulative, cumulative + prim.length()))
            cumulative += prim.length()

        # Snap each split point to nearest primitive boundary
        snapped = []
        for split_pos in split_points:
            # Find nearest primitive boundary
            best_boundary = 0.0
            best_distance = float("inf")

            for start, end in primitive_positions:
                for boundary in [start, end]:
                    distance = abs(boundary - split_pos)
                    if distance < best_distance:
                        best_distance = distance
                        best_boundary = boundary

            snapped.append(best_boundary)

        # Remove duplicates and sort
        snapped = sorted(set(snapped))
        return snapped

    def _test_subsections(
        self, section: Section, section_idx: int, num_subsections: int
    ) -> List[SubsectionResult]:
        """Test a section split into equal subsections.

        Args:
            section: The route section
            section_idx: Index of section
            num_subsections: Number of equal subsections to test

        Returns:
            List of subsection results
        """
        results = []
        section_length = section.original_length
        subsection_length = section_length / num_subsections

        # Get primitives from the section
        if not section.primitives:
            # No primitives - can't analyze
            return []

        for sub_num in range(1, num_subsections + 1):
            # Calculate start and end positions for this subsection
            start_pos = (sub_num - 1) * subsection_length
            end_pos = sub_num * subsection_length

            # Extract primitives that fall within this subsection
            # (Assuming primitives have cumulative positions within the section)
            sub_primitives = self._extract_primitives_in_range(
                section.primitives, start_pos, end_pos
            )

            if not sub_primitives:
                # No primitives in range - can't analyze
                continue

            # Calculate max tension and sidewall for this subsection
            max_tension, max_sidewall = self._calculate_subsection_limits(sub_primitives)

            # Check if it passes limits
            passes = (max_tension <= self.max_tension_limit and
                     max_sidewall <= self.max_sidewall_limit)

            results.append(
                SubsectionResult(
                    section_index=section_idx,
                    subsection_num=sub_num,
                    total_subsections=num_subsections,
                    start_position=start_pos,
                    end_position=end_pos,
                    length=subsection_length,
                    max_tension=max_tension,
                    max_sidewall_pressure=max_sidewall,
                    passes_limits=passes,
                )
            )

        return results

    def _extract_primitives_in_range(self, primitives, start_pos, end_pos):
        """Extract primitives that fall within a position range.

        Args:
            primitives: List of primitives in the section
            start_pos: Start position in section (m)
            end_pos: End position in section (m)

        Returns:
            List of primitives that fall within the range
        """
        from ..core.models import Straight, Bend

        selected_primitives = []
        cumulative_pos = 0.0

        for primitive in primitives:
            primitive_length = primitive.length()
            primitive_start = cumulative_pos
            primitive_end = cumulative_pos + primitive_length

            # Check if this primitive overlaps with the range [start_pos, end_pos]
            if primitive_end > start_pos and primitive_start < end_pos:
                # This primitive is at least partially in range
                selected_primitives.append(primitive)

            cumulative_pos = primitive_end

        return selected_primitives

    def _calculate_subsection_limits(self, primitives) -> Tuple[float, float]:
        """Calculate max tension and sidewall pressure for primitives.

        Recalculates tensions starting from 0 for this subsection independently.

        Args:
            primitives: List of primitives in the subsection

        Returns:
            Tuple of (max_tension, max_sidewall_pressure)
        """
        from ..core.models import Straight, Bend

        max_tension = 0.0
        max_sidewall = 0.0
        current_tension = 0.0  # Start from zero for independent subsection

        for primitive in primitives:
            if isinstance(primitive, Straight):
                # Calculate tension through straight section
                current_tension = calculate_straight_tension(
                    current_tension,
                    self.cable_spec,
                    self.duct_spec,
                    primitive.length_m,
                    lubricated=False,
                    slope_angle=0.0,
                )
                max_tension = max(max_tension, current_tension)

            elif isinstance(primitive, Bend):
                # Calculate tension through bend
                current_tension = calculate_bend_tension(
                    current_tension,
                    self.cable_spec,
                    self.duct_spec,
                    primitive.angle_deg,
                    lubricated=False,
                )
                max_tension = max(max_tension, current_tension)

                # Calculate sidewall pressure at this bend
                # P = T / R (tension divided by bend radius)
                if primitive.radius_m > 0:
                    sidewall_pressure = current_tension / primitive.radius_m
                    max_sidewall = max(max_sidewall, sidewall_pressure)

        return max_tension, max_sidewall
