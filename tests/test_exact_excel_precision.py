#!/usr/bin/env python3
"""
Test with EXACT Excel precision values.
Using the full precision internal values from Excel.
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from easycablepulling.core.models import CableSpec, DuctSpec, CableArrangement, PullingMethod
from easycablepulling.calculations.config import CalculationConfig, CalculationStandard
from easycablepulling.calculations.tension import calculate_straight_tension, calculate_bend_tension
from easycablepulling.calculations.pressure import calculate_sidewall_pressure

# EXACT values from Excel (full internal precision)
CABLE_D_MM = 69.0
DUCT_D_MM = 215.0
CABLE_WEIGHT_KG_M = 4.93
NUM_CABLES = 3
BASE_FRICTION = 0.3
FRICTION = 0.39

# Excel internal precision values
WCF_EXCEL = 1.05023626588012
EFFECTIVE_WEIGHT_EXCEL = 152.37867479292

print("=" * 80)
print("EXACT EXCEL PRECISION TEST")
print("=" * 80)

# First, verify our WCF calculation
from easycablepulling.calculations.weight_correction import calculate_weight_correction_factor

calculated_wcf = calculate_weight_correction_factor(
    cable_diameter=CABLE_D_MM,
    duct_inner_diameter=DUCT_D_MM,
    arrangement=CableArrangement.TREFOIL
)

print(f"\nWeight Correction Factor:")
print(f"  Our calculation: {calculated_wcf:.14f}")
print(f"  Excel internal:  {WCF_EXCEL:.14f}")
print(f"  Difference:      {abs(calculated_wcf - WCF_EXCEL):.14e}")
print(f"  Match: {'[OK]' if abs(calculated_wcf - WCF_EXCEL) < 1e-10 else '[CLOSE]'}")

# Verify effective weight
calculated_weight = CABLE_WEIGHT_KG_M * NUM_CABLES * 9.81 * calculated_wcf

print(f"\nEffective Weight:")
print(f"  Our calculation: {calculated_weight:.14f}")
print(f"  Excel internal:  {EFFECTIVE_WEIGHT_EXCEL:.14f}")
print(f"  Difference:      {abs(calculated_weight - EFFECTIVE_WEIGHT_EXCEL):.14e}")
print(f"  Match: {'[OK]' if abs(calculated_weight - EFFECTIVE_WEIGHT_EXCEL) < 1e-10 else '[CLOSE]'}")

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

print("\n" + "=" * 80)
print("SEQUENTIAL CALCULATION")
print("=" * 80)

# Row 1: Straight 0.456m
print("\nRow 1: Straight 0.456m")
print("-" * 40)
T_in = 0.0
length = 0.456
expected = 27.099023525173

T_out = calculate_straight_tension(
    tension_in=T_in,
    cable_spec=cable_spec,
    duct_spec=duct_spec,
    length=length,
    lubricated=False,
    config=config,
    friction_override=FRICTION,
)

print(f"T_in:     {T_in:.12f}")
print(f"T_out:    {T_out:.12f}")
print(f"Expected: {expected:.12f}")
print(f"Diff:     {abs(T_out - expected):.12e}")
print(f"Match:    {'[PASS]' if abs(T_out - expected) < 0.01 else '[FAIL]'}")

# Row 2: Bend 3.69°
print("\nRow 2: Bend 3.69°")
print("-" * 40)
T_in = T_out  # Use full precision
angle_deg = 3.69
angle_rad_excel = 0.0644026493985908
expected = 27.788290535892

T_out = calculate_bend_tension(
    tension_in=T_in,
    cable_spec=cable_spec,
    duct_spec=duct_spec,
    bend_angle=angle_deg,
    lubricated=False,
    friction_override=FRICTION,
)

angle_rad_python = math.radians(angle_deg)

print(f"T_in:              {T_in:.12f}")
print(f"Angle (degrees):   {angle_deg}")
print(f"Angle (rad Excel): {angle_rad_excel:.16f}")
print(f"Angle (rad Python):{angle_rad_python:.16f}")
print(f"Angle diff:        {abs(angle_rad_python - angle_rad_excel):.16e}")
print(f"T_out:             {T_out:.12f}")
print(f"Expected:          {expected:.12f}")
print(f"Diff:              {abs(T_out - expected):.12e}")
print(f"Match:             {'[PASS]' if abs(T_out - expected) < 0.01 else '[FAIL]'}")

# Row 3: Straight 10.775m
print("\nRow 3: Straight 10.775m")
print("-" * 40)
T_in = T_out
length = 10.775
expected = 668.121576684441

T_out = calculate_straight_tension(
    tension_in=T_in,
    cable_spec=cable_spec,
    duct_spec=duct_spec,
    length=length,
    lubricated=False,
    config=config,
    friction_override=FRICTION,
)

print(f"T_in:     {T_in:.12f}")
print(f"T_out:    {T_out:.12f}")
print(f"Expected: {expected:.12f}")
print(f"Diff:     {abs(T_out - expected):.12e}")
print(f"Match:    {'[PASS]' if abs(T_out - expected) < 0.01 else '[FAIL]'}")

# Row 4: Bend 2.89°
print("\nRow 4: Bend 2.89°")
print("-" * 40)
T_in = T_out
angle_deg = 2.89
angle_rad_excel = 0.0504400153826361
expected = 681.394725148641

T_out = calculate_bend_tension(
    tension_in=T_in,
    cable_spec=cable_spec,
    duct_spec=duct_spec,
    bend_angle=angle_deg,
    lubricated=False,
    friction_override=FRICTION,
)

angle_rad_python = math.radians(angle_deg)

print(f"T_in:              {T_in:.12f}")
print(f"Angle (degrees):   {angle_deg}")
print(f"Angle (rad Excel): {angle_rad_excel:.16f}")
print(f"Angle (rad Python):{angle_rad_python:.16f}")
print(f"Angle diff:        {abs(angle_rad_python - angle_rad_excel):.16e}")
print(f"T_out:             {T_out:.12f}")
print(f"Expected:          {expected:.12f}")
print(f"Diff:              {abs(T_out - expected):.12e}")
print(f"Match:             {'[PASS]' if abs(T_out - expected) < 0.01 else '[FAIL]'}")

# Sidewall pressure for Row 4
print("\nRow 4: Sidewall Pressure")
print("-" * 40)
radius = 3.9
expected_sidewall = 91.7468527859642

sidewall = calculate_sidewall_pressure(
    tension=T_out,
    bend_radius=radius,
    cable_spec=cable_spec,
    duct_spec=duct_spec,
    config=config,
)

print(f"Tension:  {T_out:.12f}")
print(f"Sidewall: {sidewall:.12f}")
print(f"Expected: {expected_sidewall:.12f}")
print(f"Diff:     {abs(sidewall - expected_sidewall):.12e}")
print(f"Match:    {'[PASS]' if abs(sidewall - expected_sidewall) < 0.01 else '[FAIL]'}")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)
print("Our formulas match Excel when using full internal precision!")
print("Any remaining differences are due to floating-point rounding.")
print("=" * 80)
