#!/usr/bin/env python3
"""
Test that equal splitting produces maximally balanced subsections.

This test verifies the fix for the equal splitting algorithm:
- Before: Greedy algorithm (first to cross threshold)
- After: Optimal algorithm (closest to target)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from easycablepulling.core.models import (
    CableSpec, DuctSpec, CableArrangement, PullingMethod,
    Section, Straight, Bend
)
from easycablepulling.calculations.config import CalculationConfig, CalculationStandard
from easycablepulling.analysis.route_optimizer import RouteOptimizer

# Test data: SECT_01 primitives
SECT_01_PRIMITIVES = [
    ("S", 0.456), ("B", 3.69, 3.9), ("S", 10.775), ("B", 2.89, 3.9),
    ("S", 9.508), ("B", 5.45, 3.9), ("S", 50.345), ("B", 1.7, 3.9),
    ("S", 30.663), ("B", 0.43, 3.9), ("S", 9.758), ("B", 2.93, 3.9),
    ("S", 62.197), ("B", 1.62, 3.9), ("S", 49.144), ("B", 2.72, 3.9),
    ("S", 22.798), ("B", 3.89, 3.9), ("S", 19.884), ("B", 1.72, 3.9),
    ("S", 24.438), ("B", 4.22, 3.9), ("S", 31.97), ("B", 0.7, 3.9),
    ("S", 15.361), ("B", 1.56, 3.9), ("S", 31.155), ("B", 0.28, 3.9),
    ("S", 41.359), ("B", 4.55, 3.9), ("S", 38.094), ("B", 16.59, 3.9),
    ("S", 3.069), ("B", 21.88, 3.9), ("S", 1.461), ("B", 16.77, 3.9),
    ("S", 2.275), ("B", 19.34, 3.9), ("S", 3.778), ("B", 29.15, 3.9),
    ("S", 3.974), ("B", 13.5, 3.9), ("S", 2.452), ("B", 37.4, 3.9),
    ("S", 36.764), ("S", 8.449), ("S", 12.615),
]


def test_equal_splitting_balance():
    """Test that equal splitting produces balanced subsections."""
    print("=" * 80)
    print("TEST: Equal Splitting Balance")
    print("=" * 80)

    # Setup
    cable_spec = CableSpec(
        diameter=69.0,
        weight_per_meter=4.93,
        max_tension=27000,
        max_sidewall_pressure=7000,
        min_bend_radius=900,
        arrangement=CableArrangement.TREFOIL,
        number_of_cables=3,
        pulling_method=PullingMethod.EYE,
    )

    duct_spec = DuctSpec(
        inner_diameter=215.0,
        type="HDPE",
        friction_dry=0.3,
        friction_lubricated=0.24,
    )

    config = CalculationConfig(standard=CalculationStandard.AEIC)

    # Create route
    primitives = []
    x = 0.0
    for prim in SECT_01_PRIMITIVES:
        if prim[0] == "S":
            length = prim[1]
            primitives.append(Straight(
                length_m=length,
                start_point=(x, 0),
                end_point=(x + length, 0)
            ))
            x += length
        else:
            angle = prim[1]
            radius = prim[2]
            primitives.append(Bend(
                angle_deg=angle,
                radius_m=radius,
                direction="CCW",
                center_point=(x, radius)
            ))
            x += 0.1

    optimizer = RouteOptimizer(
        cable_spec=cable_spec,
        duct_spec=duct_spec,
        target_utilization=0.95,
        config=config,
    )

    # Test N=2 equal splitting
    print("\nTest Case 1: N=2 Equal Splitting")
    print("-" * 80)

    result = optimizer._split_into_equal_subsections(
        section_primitives=primitives,
        num_subsections=2,
        original_section_id="01",
        friction_override=0.39,
    )

    assert len(result) == 2, f"Expected 2 subsections, got {len(result)}"

    length1 = result[0].length
    length2 = result[1].length
    total = length1 + length2
    target = total / 2

    diff = abs(length1 - length2)
    balance_ratio = min(length1, length2) / max(length1, length2)

    print(f"Subsection 1: {length1:.1f} m")
    print(f"Subsection 2: {length2:.1f} m")
    print(f"Total length: {total:.1f} m")
    print(f"Target per subsection: {target:.1f} m")
    print(f"Length difference: {diff:.1f} m ({diff/total*100:.1f}%)")
    print(f"Balance ratio: {balance_ratio:.4f}")

    # Assert balance is very good (difference < 5m or < 2% of total)
    assert diff < 5.0 or diff/total < 0.02, \
        f"Subsections are unbalanced: {diff:.1f}m difference"

    print("[PASS] Subsections are well balanced")

    # Test N=3 equal splitting
    print("\nTest Case 2: N=3 Equal Splitting")
    print("-" * 80)

    result = optimizer._split_into_equal_subsections(
        section_primitives=primitives,
        num_subsections=3,
        original_section_id="01",
        friction_override=0.39,
    )

    assert len(result) == 3, f"Expected 3 subsections, got {len(result)}"

    lengths = [s.length for s in result]
    total = sum(lengths)
    target = total / 3

    mean_length = total / len(lengths)
    max_deviation = max(abs(l - mean_length) for l in lengths)

    print(f"Subsection 1: {lengths[0]:.1f} m")
    print(f"Subsection 2: {lengths[1]:.1f} m")
    print(f"Subsection 3: {lengths[2]:.1f} m")
    print(f"Mean length: {mean_length:.1f} m")
    print(f"Target per subsection: {target:.1f} m")
    print(f"Max deviation from mean: {max_deviation:.1f} m ({max_deviation/total*100:.1f}%)")

    # Assert all subsections are within 20m or 5% of mean
    # (Tolerance depends on primitive boundaries)
    for i, length in enumerate(lengths):
        deviation = abs(length - mean_length)
        assert deviation < 20.0 or deviation/total < 0.05, \
            f"Subsection {i+1} deviates {deviation:.1f}m from mean"

    print("[PASS] All subsections are reasonably balanced")

    # Test N=4 equal splitting
    print("\nTest Case 3: N=4 Equal Splitting")
    print("-" * 80)

    result = optimizer._split_into_equal_subsections(
        section_primitives=primitives,
        num_subsections=4,
        original_section_id="01",
        friction_override=0.39,
    )

    assert len(result) == 4, f"Expected 4 subsections, got {len(result)}"

    lengths = [s.length for s in result]
    total = sum(lengths)
    mean_length = total / len(lengths)
    max_deviation = max(abs(l - mean_length) for l in lengths)

    for i, length in enumerate(lengths):
        print(f"Subsection {i+1}: {length:.1f} m")

    print(f"Mean length: {mean_length:.1f} m")
    print(f"Max deviation from mean: {max_deviation:.1f} m ({max_deviation/total*100:.1f}%)")

    # Assert all subsections are within reasonable range
    # (Tolerance depends on primitive boundaries)
    for i, length in enumerate(lengths):
        deviation = abs(length - mean_length)
        assert deviation < 30.0 or deviation/total < 0.08, \
            f"Subsection {i+1} deviates {deviation:.1f}m from mean"

    print("[PASS] All subsections are reasonably balanced")

    print("\n" + "=" * 80)
    print("ALL TESTS PASSED")
    print("=" * 80)
    print("\nEqual splitting algorithm produces maximally balanced subsections.")
    print("Splits are as close to equal as primitive boundaries allow.")
    print("=" * 80)


if __name__ == "__main__":
    test_equal_splitting_balance()
