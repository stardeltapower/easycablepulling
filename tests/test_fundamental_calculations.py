#!/usr/bin/env python3
"""
Unit tests for fundamental cable pulling calculations.
Tests each calculation against exact values from Excel.
"""

import math
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from easycablepulling.core.models import CableSpec, DuctSpec, CableArrangement, PullingMethod
from easycablepulling.calculations.config import CalculationConfig, CalculationStandard
from easycablepulling.calculations.tension import calculate_straight_tension, calculate_bend_tension
from easycablepulling.calculations.pressure import calculate_sidewall_pressure
from easycablepulling.calculations.weight_correction import calculate_weight_correction_factor


# Test parameters from Excel
CABLE_D_MM = 69.0
DUCT_D_MM = 215.0
CABLE_WEIGHT_KG_M = 4.93
NUM_CABLES = 3
BASE_FRICTION = 0.3
FRICTION_MULTIPLIER = 1.3
FRICTION = 0.39

# Expected values from Excel
EXPECTED_WCF = 1.0502
EXPECTED_EFFECTIVE_WEIGHT = 152.3786748  # N/m


def test_weight_correction_factor():
    """Test WCF calculation: sqrt(1 + (d/D)^2)"""
    print("\n" + "="*80)
    print("TEST 1: Weight Correction Factor (WCF)")
    print("="*80)

    # AEIC formula: WCF = sqrt(1 + (d/D)^2)
    calculated = calculate_weight_correction_factor(
        cable_diameter=CABLE_D_MM,
        duct_inner_diameter=DUCT_D_MM,
        arrangement=CableArrangement.TREFOIL
    )

    print(f"Formula: WCF = sqrt(1 + (d/D)^2)")
    print(f"  d (cable diameter): {CABLE_D_MM} mm")
    print(f"  D (duct inner diameter): {DUCT_D_MM} mm")
    print(f"  d/D ratio: {CABLE_D_MM/DUCT_D_MM:.6f}")
    print(f"\nCalculated: {calculated:.6f}")
    print(f"Expected:   {EXPECTED_WCF:.6f}")
    print(f"Difference: {abs(calculated - EXPECTED_WCF):.8f}")

    tolerance = 0.0001
    passed = abs(calculated - EXPECTED_WCF) < tolerance
    print(f"\nResult: {'[PASS]' if passed else '[FAIL]'}")

    return passed


def test_effective_weight():
    """Test effective weight: cable_weight × num_cables × 9.81 × WCF"""
    print("\n" + "="*80)
    print("TEST 2: Effective Weight Calculation")
    print("="*80)

    wc = calculate_weight_correction_factor(CABLE_D_MM, DUCT_D_MM, CableArrangement.TREFOIL)
    calculated = CABLE_WEIGHT_KG_M * NUM_CABLES * 9.81 * wc

    print(f"Formula: w_eff = cable_weight × num_cables × 9.81 × WCF")
    print(f"  Cable weight: {CABLE_WEIGHT_KG_M} kg/m")
    print(f"  Number of cables: {NUM_CABLES}")
    print(f"  Gravity: 9.81 m/s²")
    print(f"  WCF: {wc:.6f}")
    print(f"\nCalculated: {calculated:.7f} N/m")
    print(f"Expected:   {EXPECTED_EFFECTIVE_WEIGHT:.7f} N/m")
    print(f"Difference: {abs(calculated - EXPECTED_EFFECTIVE_WEIGHT):.8f}")

    tolerance = 0.0001
    passed = abs(calculated - EXPECTED_EFFECTIVE_WEIGHT) < tolerance
    print(f"\nResult: {'[PASS]' if passed else '[FAIL]'}")

    return passed


def test_friction_coefficient():
    """Test friction coefficient with trefoil multiplier"""
    print("\n" + "="*80)
    print("TEST 3: Friction Coefficient (Trefoil)")
    print("="*80)

    # Create specs
    duct_spec = DuctSpec(
        inner_diameter=DUCT_D_MM,
        type="HDPE",
        friction_dry=BASE_FRICTION,
        friction_lubricated=BASE_FRICTION * 0.8,
    )

    calculated = duct_spec.get_friction(CableArrangement.TREFOIL, lubricated=False)

    print(f"Formula: u_trefoil = u_base * 1.3")
    print(f"  Base friction: {BASE_FRICTION}")
    print(f"  Trefoil multiplier: {FRICTION_MULTIPLIER}")
    print(f"\nCalculated: {calculated:.2f}")
    print(f"Expected:   {FRICTION:.2f}")
    print(f"Difference: {abs(calculated - FRICTION):.6f}")

    tolerance = 0.01
    passed = abs(calculated - FRICTION) < tolerance
    print(f"\nResult: {'[PASS]' if passed else '[FAIL]'}")

    return passed


def test_straight_section_1():
    """Test straight section tension - Case 1: T_in=0, L=0.456m"""
    print("\n" + "="*80)
    print("TEST 4: Straight Section Tension (Case 1)")
    print("="*80)

    # Excel data: Row 1
    T_in = 0.00
    length = 0.456
    expected_T_out = 27.10

    # Create specs
    cable_spec = CableSpec(
        diameter=CABLE_D_MM,
        weight_per_meter=CABLE_WEIGHT_KG_M,
        max_tension=27000,
        max_sidewall_pressure=7000,
        min_bend_radius=900,
        arrangement=CableArrangement.TREFOIL,
        number_of_cables=NUM_CABLES,
        pulling_method=PullingMethod.EYE,
    )

    duct_spec = DuctSpec(
        inner_diameter=DUCT_D_MM,
        type="HDPE",
        friction_dry=BASE_FRICTION,
        friction_lubricated=BASE_FRICTION * 0.8,
    )

    config = CalculationConfig(standard=CalculationStandard.AEIC)

    calculated = calculate_straight_tension(
        tension_in=T_in,
        cable_spec=cable_spec,
        duct_spec=duct_spec,
        length=length,
        lubricated=False,
        config=config,
        friction_override=FRICTION,
    )

    print(f"Formula: T_out = T_in + (u * w_c * L)")
    print(f"  T_in: {T_in} N")
    print(f"  u (friction): {FRICTION}")
    print(f"  w_c (effective weight): {EXPECTED_EFFECTIVE_WEIGHT:.4f} N/m")
    print(f"  L (length): {length} m")
    print(f"\nCalculation: {T_in} + ({FRICTION} * {EXPECTED_EFFECTIVE_WEIGHT:.4f} * {length})")
    print(f"           = {T_in} + {FRICTION * EXPECTED_EFFECTIVE_WEIGHT * length:.2f}")
    print(f"\nCalculated: {calculated:.2f} N")
    print(f"Expected:   {expected_T_out:.2f} N")
    print(f"Difference: {abs(calculated - expected_T_out):.2f} N")

    tolerance = 0.1
    passed = abs(calculated - expected_T_out) < tolerance
    print(f"\nResult: {'[PASS]' if passed else '[FAIL]'}")

    return passed


def test_straight_section_2():
    """Test straight section tension - Case 2: T_in=27.83, L=10.775m"""
    print("\n" + "="*80)
    print("TEST 5: Straight Section Tension (Case 2)")
    print("="*80)

    # Excel data: Row 3
    T_in = 27.83
    length = 10.775
    expected_T_out = 668.16

    cable_spec = CableSpec(
        diameter=CABLE_D_MM,
        weight_per_meter=CABLE_WEIGHT_KG_M,
        max_tension=27000,
        max_sidewall_pressure=7000,
        min_bend_radius=900,
        arrangement=CableArrangement.TREFOIL,
        number_of_cables=NUM_CABLES,
        pulling_method=PullingMethod.EYE,
    )

    duct_spec = DuctSpec(
        inner_diameter=DUCT_D_MM,
        type="HDPE",
        friction_dry=BASE_FRICTION,
        friction_lubricated=BASE_FRICTION * 0.8,
    )

    config = CalculationConfig(standard=CalculationStandard.AEIC)

    calculated = calculate_straight_tension(
        tension_in=T_in,
        cable_spec=cable_spec,
        duct_spec=duct_spec,
        length=length,
        lubricated=False,
        config=config,
        friction_override=FRICTION,
    )

    print(f"Formula: T_out = T_in + (u * w_c * L)")
    print(f"  T_in: {T_in} N")
    print(f"  u: {FRICTION}")
    print(f"  w_c: {EXPECTED_EFFECTIVE_WEIGHT:.4f} N/m")
    print(f"  L: {length} m")
    print(f"\nCalculation: {T_in} + ({FRICTION} * {EXPECTED_EFFECTIVE_WEIGHT:.4f} * {length})")
    print(f"           = {T_in} + {FRICTION * EXPECTED_EFFECTIVE_WEIGHT * length:.2f}")
    print(f"\nCalculated: {calculated:.2f} N")
    print(f"Expected:   {expected_T_out:.2f} N")
    print(f"Difference: {abs(calculated - expected_T_out):.2f} N")

    tolerance = 0.1
    passed = abs(calculated - expected_T_out) < tolerance
    print(f"\nResult: {'[PASS]' if passed else '[FAIL]'}")

    return passed


def test_bend_section_1():
    """Test bend tension - Case 1: T_in=27.10, angle=3.69°"""
    print("\n" + "="*80)
    print("TEST 6: Bend Section Tension (Case 1)")
    print("="*80)

    # Excel data: Row 2
    T_in = 27.10
    angle_deg = 3.69
    expected_T_out = 27.83

    cable_spec = CableSpec(
        diameter=CABLE_D_MM,
        weight_per_meter=CABLE_WEIGHT_KG_M,
        max_tension=27000,
        max_sidewall_pressure=7000,
        min_bend_radius=900,
        arrangement=CableArrangement.TREFOIL,
        number_of_cables=NUM_CABLES,
        pulling_method=PullingMethod.EYE,
    )

    duct_spec = DuctSpec(
        inner_diameter=DUCT_D_MM,
        type="HDPE",
        friction_dry=BASE_FRICTION,
        friction_lubricated=BASE_FRICTION * 0.8,
    )

    calculated = calculate_bend_tension(
        tension_in=T_in,
        cable_spec=cable_spec,
        duct_spec=duct_spec,
        bend_angle=angle_deg,
        lubricated=False,
        friction_override=FRICTION,
    )

    angle_rad = math.radians(angle_deg)
    print(f"Formula: T_out = T_in * e^(u * alpha)")
    print(f"  T_in: {T_in} N")
    print(f"  u: {FRICTION}")
    print(f"  alpha: {angle_deg}deg = {angle_rad:.6f} radians")
    print(f"\nCalculation: {T_in} * e^({FRICTION} * {angle_rad:.6f})")
    print(f"           = {T_in} * e^{FRICTION * angle_rad:.6f}")
    print(f"           = {T_in} × {math.exp(FRICTION * angle_rad):.6f}")
    print(f"\nCalculated: {calculated:.2f} N")
    print(f"Expected:   {expected_T_out:.2f} N")
    print(f"Difference: {abs(calculated - expected_T_out):.2f} N")

    tolerance = 0.1
    passed = abs(calculated - expected_T_out) < tolerance
    print(f"\nResult: {'[PASS]' if passed else '[FAIL]'}")

    return passed


def test_bend_section_2():
    """Test bend tension - Case 2: T_in=668.16, angle=2.89°"""
    print("\n" + "="*80)
    print("TEST 7: Bend Section Tension (Case 2)")
    print("="*80)

    # Excel data: Row 4
    T_in = 668.16
    angle_deg = 2.89
    expected_T_out = 686.14

    cable_spec = CableSpec(
        diameter=CABLE_D_MM,
        weight_per_meter=CABLE_WEIGHT_KG_M,
        max_tension=27000,
        max_sidewall_pressure=7000,
        min_bend_radius=900,
        arrangement=CableArrangement.TREFOIL,
        number_of_cables=NUM_CABLES,
        pulling_method=PullingMethod.EYE,
    )

    duct_spec = DuctSpec(
        inner_diameter=DUCT_D_MM,
        type="HDPE",
        friction_dry=BASE_FRICTION,
        friction_lubricated=BASE_FRICTION * 0.8,
    )

    calculated = calculate_bend_tension(
        tension_in=T_in,
        cable_spec=cable_spec,
        duct_spec=duct_spec,
        bend_angle=angle_deg,
        lubricated=False,
        friction_override=FRICTION,
    )

    angle_rad = math.radians(angle_deg)
    print(f"Formula: T_out = T_in * e^(u * alpha)")
    print(f"  T_in: {T_in} N")
    print(f"  u: {FRICTION}")
    print(f"  alpha: {angle_deg}deg = {angle_rad:.6f} radians")
    print(f"\nCalculation: {T_in} * e^({FRICTION} * {angle_rad:.6f})")
    print(f"           = {T_in} * e^{FRICTION * angle_rad:.6f}")
    print(f"           = {T_in} × {math.exp(FRICTION * angle_rad):.6f}")
    print(f"\nCalculated: {calculated:.2f} N")
    print(f"Expected:   {expected_T_out:.2f} N")
    print(f"Difference: {abs(calculated - expected_T_out):.2f} N")

    tolerance = 0.1
    passed = abs(calculated - expected_T_out) < tolerance
    print(f"\nResult: {'[PASS]' if passed else '[FAIL]'}")

    return passed


def test_sidewall_pressure():
    """Test sidewall pressure: (w_c × T) / (2 × r)"""
    print("\n" + "="*80)
    print("TEST 8: Sidewall Pressure")
    print("="*80)

    # Excel data: Row 4
    tension = 686.14
    radius = 3.9
    expected_sidewall = 92.39

    cable_spec = CableSpec(
        diameter=CABLE_D_MM,
        weight_per_meter=CABLE_WEIGHT_KG_M,
        max_tension=27000,
        max_sidewall_pressure=7000,
        min_bend_radius=900,
        arrangement=CableArrangement.TREFOIL,
        number_of_cables=NUM_CABLES,
        pulling_method=PullingMethod.EYE,
    )

    duct_spec = DuctSpec(
        inner_diameter=DUCT_D_MM,
        type="HDPE",
        friction_dry=BASE_FRICTION,
        friction_lubricated=BASE_FRICTION * 0.8,
    )

    config = CalculationConfig(standard=CalculationStandard.AEIC)

    calculated = calculate_sidewall_pressure(
        tension=tension,
        bend_radius=radius,
        cable_spec=cable_spec,
        duct_spec=duct_spec,
        config=config,
    )

    print(f"Formula: P = (w_c * T) / (2 * r)  [AEIC with trefoil reduction]")
    print(f"  w_c (WCF): {EXPECTED_WCF:.4f}")
    print(f"  T (tension): {tension} N")
    print(f"  r (radius): {radius} m")
    print(f"\nCalculation: ({EXPECTED_WCF:.4f} * {tension}) / (2 * {radius})")
    print(f"           = {EXPECTED_WCF * tension:.2f} / {2 * radius}")
    print(f"           = {(EXPECTED_WCF * tension) / (2 * radius):.2f}")
    print(f"\nCalculated: {calculated:.2f} N/m")
    print(f"Expected:   {expected_sidewall:.2f} N/m")
    print(f"Difference: {abs(calculated - expected_sidewall):.2f} N/m")

    tolerance = 0.5
    passed = abs(calculated - expected_sidewall) < tolerance
    print(f"\nResult: {'[PASS]' if passed else '[FAIL]'}")

    return passed


def run_all_tests():
    """Run all unit tests"""
    print("\n" + "="*80)
    print("FUNDAMENTAL CALCULATIONS UNIT TESTS")
    print("Testing against Excel values")
    print("="*80)

    tests = [
        ("Weight Correction Factor", test_weight_correction_factor),
        ("Effective Weight", test_effective_weight),
        ("Friction Coefficient", test_friction_coefficient),
        ("Straight Tension (Case 1)", test_straight_section_1),
        ("Straight Tension (Case 2)", test_straight_section_2),
        ("Bend Tension (Case 1)", test_bend_section_1),
        ("Bend Tension (Case 2)", test_bend_section_2),
        ("Sidewall Pressure", test_sidewall_pressure),
    ]

    results = []
    for name, test_func in tests:
        passed = test_func()
        results.append((name, passed))

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    for name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status} {name}")

    total = len(results)
    passed_count = sum(1 for _, p in results if p)

    print(f"\nTotal: {passed_count}/{total} tests passed")

    if passed_count == total:
        print("\n[SUCCESS] All tests passed!")
    else:
        print(f"\n[FAILURE] {total - passed_count} test(s) failed")

    return passed_count == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
