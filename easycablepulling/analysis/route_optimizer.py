"""Route optimization module for cable pulling analysis.

This module optimizes cable routes by automatically splitting sections
to stay within tension and sidewall pressure limits, with different
strategies for forward and reverse pulling directions.
"""

import math
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, NamedTuple
from enum import Enum

from ..core.models import Bend, Straight, Section, Route, CableSpec, DuctSpec
from ..calculations.tension import calculate_straight_tension, calculate_bend_tension


class PullingDirection(Enum):
    """Pulling direction for optimization."""
    FORWARD = "forward"
    REVERSE = "reverse"


class PrimitiveResult(NamedTuple):
    """Result for a single primitive (straight or bend)."""
    primitive: object  # Straight or Bend
    position: float  # Cumulative position in meters
    tension_in: float  # Input tension (N)
    tension_out: float  # Output tension (N)
    sidewall_pressure: float  # Sidewall pressure (N/m) for bends, 0 for straights
    passes_limits: bool  # Whether this primitive passes all limits


@dataclass
class OptimizedSection:
    """An optimized section with detailed analysis."""
    section_id: str
    start_position: float  # Start position in overall route (m)
    end_position: float  # End position in overall route (m)
    length: float  # Section length (m)
    primitives: List[PrimitiveResult]  # Detailed primitive results
    
    # Peak values
    max_tension: float  # Maximum tension in section (N)
    max_sidewall_pressure: float  # Maximum sidewall pressure (N/m)
    
    # Utilization ratios (0-1)
    tension_utilization: float
    sidewall_utilization: float
    
    # Pass/fail status
    passes_tension: bool
    passes_sidewall: bool
    overall_pass: bool


@dataclass
class OptimizationResult:
    """Complete optimization result for a route."""
    direction: PullingDirection
    original_sections: int  # Number of sections before optimization
    optimized_sections: int  # Number of sections after optimization
    total_length: float  # Total route length (m)
    
    # Optimized sections
    sections: List[OptimizedSection]
    
    # Summary statistics
    max_tension: float  # Peak tension across all sections (N)
    max_sidewall_pressure: float  # Peak sidewall pressure across all sections (N/m)
    max_tension_utilization: float  # Peak utilization ratio
    max_sidewall_utilization: float  # Peak utilization ratio
    
    # Overall status
    all_sections_pass: bool
    feasible: bool
    
    # Optimization details
    target_utilization: float  # Target utilization (e.g., 0.8 for 80%)
    max_section_length: float  # Maximum allowed section length (m)


class RouteOptimizer:
    """Optimizes cable routes for pulling feasibility."""
    
    def __init__(
        self,
        cable_spec: CableSpec,
        duct_spec: DuctSpec,
        target_utilization: float = 0.8,  # 80% safety margin
        max_section_length: float = 500.0,  # Maximum 500m sections
    ):
        """Initialize route optimizer.
        
        Args:
            cable_spec: Cable specifications
            duct_spec: Duct specifications
            target_utilization: Target utilization ratio (0-1)
            max_section_length: Maximum section length in meters
        """
        self.cable_spec = cable_spec
        self.duct_spec = duct_spec
        self.target_utilization = target_utilization
        self.max_section_length = max_section_length
        
        # Calculate limits with safety margin
        self.tension_limit = cable_spec.max_tension * target_utilization
        self.sidewall_limit = cable_spec.max_sidewall_pressure * target_utilization
        
    def optimize_route(
        self,
        route: Route,
        direction: PullingDirection = PullingDirection.FORWARD,
        friction_override: Optional[float] = None,
    ) -> OptimizationResult:
        """Optimize a route for the specified pulling direction.

        Args:
            route: Route to optimize
            direction: Pulling direction (forward or reverse)
            friction_override: Optional friction coefficient override

        Returns:
            Optimization result with split sections
        """
        # Get all primitives in order
        all_primitives = self._extract_primitives(route, direction)

        # Calculate tensions and pressures for all primitives (for reference)
        primitive_results = self._calculate_primitive_results(
            all_primitives, friction_override
        )

        # Find optimal split points based on cumulative tensions
        split_points = self._find_optimal_splits(primitive_results)

        # Create optimized sections with recalculated tensions for each section
        optimized_sections = self._create_optimized_sections_with_recalc(
            all_primitives, split_points, friction_override
        )

        # Balance section lengths if possible
        optimized_sections = self._balance_sections(optimized_sections, primitive_results)

        # Calculate summary statistics
        max_tension = max(s.max_tension for s in optimized_sections)
        max_sidewall = max(s.max_sidewall_pressure for s in optimized_sections)
        max_tension_util = max(s.tension_utilization for s in optimized_sections)
        max_sidewall_util = max(s.sidewall_utilization for s in optimized_sections)

        return OptimizationResult(
            direction=direction,
            original_sections=len(route.sections),
            optimized_sections=len(optimized_sections),
            total_length=sum(s.length for s in optimized_sections),
            sections=optimized_sections,
            max_tension=max_tension,
            max_sidewall_pressure=max_sidewall,
            max_tension_utilization=max_tension_util,
            max_sidewall_utilization=max_sidewall_util,
            all_sections_pass=all(s.overall_pass for s in optimized_sections),
            feasible=all(s.overall_pass for s in optimized_sections),
            target_utilization=self.target_utilization,
            max_section_length=self.max_section_length,
        )
    
    def _extract_primitives(
        self, route: Route, direction: PullingDirection
    ) -> List[object]:
        """Extract all primitives from route in pulling order.
        
        Args:
            route: Route to extract from
            direction: Pulling direction
            
        Returns:
            List of primitives (Straight and Bend objects) in order
        """
        all_primitives = []
        
        for section in route.sections:
            section_primitives = list(section.primitives)
            
            # Reverse order for reverse pulling
            if direction == PullingDirection.REVERSE:
                section_primitives = section_primitives[::-1]
                
            all_primitives.extend(section_primitives)
        
        # For reverse pulling, reverse the entire list
        if direction == PullingDirection.REVERSE:
            all_primitives = all_primitives[::-1]
            
        return all_primitives
    
    def _calculate_primitive_results(
        self,
        primitives: List[object],
        friction_override: Optional[float] = None,
    ) -> List[PrimitiveResult]:
        """Calculate tension and pressure for each primitive.
        
        Args:
            primitives: List of primitives in order
            friction_override: Optional friction coefficient
            
        Returns:
            List of primitive results with tensions and pressures
        """
        results = []
        current_tension = 0.0  # Start with zero tension
        current_position = 0.0
        
        friction = friction_override if friction_override else self.duct_spec.friction_dry
        
        for primitive in primitives:
            if isinstance(primitive, Straight):
                # Calculate straight section tension
                tension_out = calculate_straight_tension(
                    tension_in=current_tension,
                    cable_spec=self.cable_spec,
                    duct_spec=self.duct_spec,
                    length=primitive.length_m,
                    lubricated=(friction < 0.4),  # Assume lubricated if friction < 0.4
                )
                
                sidewall_pressure = 0.0  # No sidewall pressure in straights
                current_position += primitive.length_m
                
            elif isinstance(primitive, Bend):
                # Calculate bend tension
                tension_out = calculate_bend_tension(
                    tension_in=current_tension,
                    cable_spec=self.cable_spec,
                    duct_spec=self.duct_spec,
                    bend_angle=primitive.angle_deg,
                    lubricated=(friction < 0.4),
                )
                
                # Calculate sidewall pressure (P = T/R)
                # Use average tension for better estimate
                avg_tension = (current_tension + tension_out) / 2
                sidewall_pressure = avg_tension / primitive.radius_m
                
                # Add bend length to position
                bend_length = primitive.radius_m * math.radians(primitive.angle_deg)
                current_position += bend_length
            else:
                continue
            
            # Check if within limits
            passes_tension = tension_out <= self.tension_limit
            passes_sidewall = sidewall_pressure <= self.sidewall_limit
            
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
    
    def _find_optimal_splits(
        self, primitive_results: List[PrimitiveResult]
    ) -> List[int]:
        """Find optimal split points to keep within limits.

        Uses a greedy algorithm that:
        1. Extends each section as far as possible while within limits
        2. When limits are consistently exceeded, compresses those sections to avoid tiny splits
        3. Recalculates tensions from 0 for each new section

        Args:
            primitive_results: List of primitive results from cumulative calculation

        Returns:
            List of indices where sections should be split
        """
        if not primitive_results:
            return [0, 0]

        split_points = [0]  # Always start at beginning
        section_start_idx = 0
        section_start_pos = 0.0

        while section_start_idx < len(primitive_results):
            # Find the furthest primitive we can include while ALL limits are within tolerance
            last_good_idx = section_start_idx - 1
            limit_first_exceeded_at = None

            for i in range(section_start_idx, len(primitive_results)):
                result = primitive_results[i]
                section_length = result.position - section_start_pos

                # Check if limits are satisfied
                tension_ok = result.tension_out <= self.tension_limit
                sidewall_ok = result.sidewall_pressure <= self.sidewall_limit
                length_ok = section_length <= self.max_section_length

                if tension_ok and sidewall_ok and length_ok:
                    last_good_idx = i
                elif limit_first_exceeded_at is None:
                    # Mark where limits first fail
                    limit_first_exceeded_at = i
                    break

            # Decide where to split
            if last_good_idx >= section_start_idx:
                # We have some good primitives - split after the last good one
                final_idx = last_good_idx
                split_points.append(final_idx + 1)
                section_start_pos = primitive_results[final_idx].position
                section_start_idx = final_idx + 1
            else:
                # Even first primitive fails limits
                # This means we're in a region where tension is too high
                # Compress all remaining failed primitives into one section to avoid tiny splits

                # Find how far we can extend before hitting hard limit (1.5x the safety target)
                hard_limit_idx = section_start_idx - 1
                hard_tension_limit = self.tension_limit * 1.5  # Hard stop at 1.5x safety target
                hard_sidewall_limit = self.sidewall_limit * 1.5

                for i in range(section_start_idx, len(primitive_results)):
                    result = primitive_results[i]

                    # Hard stop if way over limits
                    if (result.tension_out > hard_tension_limit or
                        result.sidewall_pressure > hard_sidewall_limit):
                        break
                    hard_limit_idx = i

                if hard_limit_idx >= section_start_idx:
                    # Extend to hard limit
                    section_start_idx = hard_limit_idx + 1
                    section_start_pos = primitive_results[hard_limit_idx].position
                    split_points.append(section_start_idx)
                else:
                    # Even hard limits exceeded immediately - include just one primitive
                    section_start_idx = section_start_idx + 1
                    section_start_pos = primitive_results[section_start_idx - 1].position
                    split_points.append(section_start_idx)

                if section_start_idx >= len(primitive_results):
                    break

        # Add end point if not already there
        if split_points[-1] != len(primitive_results):
            split_points.append(len(primitive_results))

        # Remove duplicates and sort
        split_points = sorted(set(split_points))

        return split_points
    
    def _create_optimized_sections_with_recalc(
        self,
        all_primitives: List[object],
        split_points: List[int],
        friction_override: Optional[float] = None,
    ) -> List[OptimizedSection]:
        """Create optimized sections with recalculated tensions for each section.

        This recalculates tensions for each section starting from zero, ensuring
        that each section has independent tension analysis.

        Args:
            all_primitives: List of all primitives
            split_points: Indices where to split
            friction_override: Optional friction coefficient

        Returns:
            List of optimized sections
        """
        sections = []
        section_count = 0

        for i in range(len(split_points) - 1):
            start_idx = split_points[i]
            end_idx = split_points[i + 1]

            # Get primitives for this section
            section_primitives = all_primitives[start_idx:end_idx]

            if not section_primitives:
                continue

            # Recalculate tensions for this section starting from 0
            section_results = self._calculate_primitive_results(
                section_primitives, friction_override
            )

            if not section_results:
                continue

            # Calculate section properties
            start_pos = section_results[0].position if start_idx > 0 else 0.0
            end_pos = section_results[-1].position
            length = end_pos - start_pos

            # Skip zero-length sections (pure bends with no straight sections)
            # These don't represent actual pulling sections
            if length < 0.1:  # Tolerance for floating point
                continue

            # Find maximum values
            max_tension = max(p.tension_out for p in section_results)
            max_sidewall = max(p.sidewall_pressure for p in section_results)

            # Calculate utilization
            tension_util = max_tension / self.cable_spec.max_tension
            sidewall_util = max_sidewall / self.cable_spec.max_sidewall_pressure

            # Check pass/fail
            passes_tension = max_tension <= self.tension_limit
            passes_sidewall = max_sidewall <= self.sidewall_limit

            section_count += 1
            sections.append(
                OptimizedSection(
                    section_id=f"OPT_{section_count:02d}",
                    start_position=start_pos,
                    end_position=end_pos,
                    length=length,
                    primitives=section_results,
                    max_tension=max_tension,
                    max_sidewall_pressure=max_sidewall,
                    tension_utilization=tension_util,
                    sidewall_utilization=sidewall_util,
                    passes_tension=passes_tension,
                    passes_sidewall=passes_sidewall,
                    overall_pass=(passes_tension and passes_sidewall),
                )
            )

        return sections

    def _create_optimized_sections(
        self,
        primitive_results: List[PrimitiveResult],
        split_points: List[int],
    ) -> List[OptimizedSection]:
        """Create optimized sections from split points.

        Args:
            primitive_results: List of all primitive results
            split_points: Indices where to split

        Returns:
            List of optimized sections
        """
        sections = []

        for i in range(len(split_points) - 1):
            start_idx = split_points[i]
            end_idx = split_points[i + 1]

            # Get primitives for this section
            section_primitives = primitive_results[start_idx:end_idx]

            if not section_primitives:
                continue

            # Calculate section properties
            start_pos = section_primitives[0].position if start_idx > 0 else 0.0
            end_pos = section_primitives[-1].position
            length = end_pos - start_pos

            # Find maximum values
            max_tension = max(p.tension_out for p in section_primitives)
            max_sidewall = max(p.sidewall_pressure for p in section_primitives)

            # Calculate utilization
            tension_util = max_tension / self.cable_spec.max_tension
            sidewall_util = max_sidewall / self.cable_spec.max_sidewall_pressure

            # Check pass/fail
            passes_tension = max_tension <= self.tension_limit
            passes_sidewall = max_sidewall <= self.sidewall_limit

            sections.append(
                OptimizedSection(
                    section_id=f"OPT_{i+1:02d}",
                    start_position=start_pos,
                    end_position=end_pos,
                    length=length,
                    primitives=section_primitives,
                    max_tension=max_tension,
                    max_sidewall_pressure=max_sidewall,
                    tension_utilization=tension_util,
                    sidewall_utilization=sidewall_util,
                    passes_tension=passes_tension,
                    passes_sidewall=passes_sidewall,
                    overall_pass=(passes_tension and passes_sidewall),
                )
            )

        return sections
    
    def _balance_sections(
        self,
        sections: List[OptimizedSection],
        primitive_results: List[PrimitiveResult],
    ) -> List[OptimizedSection]:
        """Balance section lengths for more even distribution.
        
        Args:
            sections: Initial optimized sections
            primitive_results: All primitive results
            
        Returns:
            Balanced sections if possible, otherwise original
        """
        # If all sections pass and we have room to balance, try to equalize lengths
        if all(s.overall_pass for s in sections) and len(sections) > 1:
            total_length = sum(s.length for s in sections)
            target_length = total_length / len(sections)
            
            # Only balance if variation is significant (>20% difference)
            max_length = max(s.length for s in sections)
            min_length = min(s.length for s in sections)
            
            if (max_length - min_length) / target_length > 0.2:
                # Attempt to rebalance
                # This is a simplified approach - could be made more sophisticated
                new_split_points = self._find_balanced_splits(
                    primitive_results, len(sections)
                )
                
                # Recreate sections with new splits
                new_sections = self._create_optimized_sections(
                    primitive_results, new_split_points
                )
                
                # Only use new sections if they all pass
                if all(s.overall_pass for s in new_sections):
                    return new_sections
                    
        return sections
    
    def _find_balanced_splits(
        self,
        primitive_results: List[PrimitiveResult],
        num_sections: int,
    ) -> List[int]:
        """Find split points for balanced section lengths.
        
        Args:
            primitive_results: All primitive results
            num_sections: Target number of sections
            
        Returns:
            List of split indices
        """
        if not primitive_results:
            return [0, 0]
            
        total_length = primitive_results[-1].position
        target_length = total_length / num_sections
        
        split_points = [0]
        current_target = target_length
        
        for i, result in enumerate(primitive_results):
            if result.position >= current_target and i > split_points[-1]:
                split_points.append(i)
                current_target += target_length
                
                if len(split_points) == num_sections:
                    break
                    
        split_points.append(len(primitive_results))
        
        return split_points


def optimize_cable_route(
    route: Route,
    cable_spec: CableSpec,
    duct_spec: DuctSpec,
    target_utilization: float = 0.8,
    max_section_length: float = 500.0,
    friction_override: Optional[float] = None,
) -> Tuple[OptimizationResult, OptimizationResult]:
    """Optimize a cable route for both pulling directions.
    
    Args:
        route: Route to optimize
        cable_spec: Cable specifications
        duct_spec: Duct specifications
        target_utilization: Target utilization ratio (default 80%)
        max_section_length: Maximum section length in meters
        friction_override: Optional friction coefficient
        
    Returns:
        Tuple of (forward_result, reverse_result)
    """
    optimizer = RouteOptimizer(
        cable_spec=cable_spec,
        duct_spec=duct_spec,
        target_utilization=target_utilization,
        max_section_length=max_section_length,
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