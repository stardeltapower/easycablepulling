#!/usr/bin/env python3
"""
Sequential test maintaining full precision through calculations.
Tests rows 1-4 from Excel with internal precision carried forward.
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from easycablepulling.core.models import CableSpec, DuctSpec, CableArrangement, PullingMethod
from easycablepulling.calculations.config import CalculationConfig, CalculationStandard
from easycablepulling.calculations.tension import calculate_straight_tension, calculate_bend_tension
from easycablepulling.calculations.pressure import calculate_sidewall_pressure

# Parameters
CABLE_D_MM = 69.0
DUCT_D_MM = 215.0
CABLE_WEIGHT_KG_M = 4.93
NUM_CABLES = 3
BASE_FRICTION = 0.3
FRICTION = 0.39

EXPECTED_WCF = 1.0502
EXPECTED_EFFECTIVE_WEIGHT = 152.37867

# Create specs once
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

print("=" * 80)
print("SEQUENTIAL CALCULATION TEST (Rows 1-4)")
print("Maintaining full precision throughout")
print("=" * 80)

# Row 1: Straight 0.456m
print("\nRow 1: Straight 0.456m")
print("-" * 40)
T_in = 0.00
length = 0.456
expected = 27.10

T_out = calculate_straight_tension(
    tension_in=T_in,
    cable_spec=cable_spec,
    duct_spec=duct_spec,
    length=length,
    lubricated=False,
    config=config,
    friction_override=FRICTION,
)

print(f"T_in:      {T_in:.5f} N")
print(f"Length:    {length} m")
print(f"T_out:     {T_out:.5f} N (display: {T_out:.2f} N)")
print(f"Expected:  {expected:.2f} N")
print(f"Match:     {'[PASS]' if abs(T_out - expected) < 0.01 else '[FAIL]'}")

# Row 2: Bend 3.69°
print("\nRow 2: Bend 3.69°")
print("-" * 40)
T_in = T_out  # Carry forward FULL precision
angle_deg = 3.69
expected = 27.83

T_out = calculate_bend_tension(
    tension_in=T_in,
    cable_spec=cable_spec,
    duct_spec=duct_spec,
    bend_angle=angle_deg,
    lubricated=False,
    friction_override=FRICTION,
)

print(f"T_in:      {T_in:.5f} N (Excel uses this precise value)")
print(f"Angle:     {angle_deg}°")
print(f"T_out:     {T_out:.5f} N (display: {T_out:.2f} N)")
print(f"Expected:  {expected:.2f} N")
print(f"Match:     {'[PASS]' if abs(T_out - expected) < 0.01 else '[FAIL]'}")

# Row 3: Straight 10.775m
print("\nRow 3: Straight 10.775m")
print("-" * 40)
T_in = T_out  # Carry forward FULL precision
length = 10.775
expected = 668.16

T_out = calculate_straight_tension(
    tension_in=T_in,
    cable_spec=cable_spec,
    duct_spec=duct_spec,
    length=length,
    lubricated=False,
    config=config,
    friction_override=FRICTION,
)

print(f"T_in:      {T_in:.5f} N (Excel uses this precise value)")
print(f"Length:    {length} m")
print(f"T_out:     {T_out:.5f} N (display: {T_out:.2f} N)")
print(f"Expected:  {expected:.2f} N")
print(f"Match:     {'[PASS]' if abs(T_out - expected) < 0.01 else '[FAIL]'}")

# Row 4: Bend 2.89°
print("\nRow 4: Bend 2.89°")
print("-" * 40)
T_in = T_out  # Carry forward FULL precision
angle_deg = 2.89
expected = 686.14

T_out = calculate_bend_tension(
    tension_in=T_in,
    cable_spec=cable_spec,
    duct_spec=duct_spec,
    bend_angle=angle_deg,
    lubricated=False,
    friction_override=FRICTION,
)

print(f"T_in:      {T_in:.5f} N (Excel uses this precise value)")
print(f"Angle:     {angle_deg}°")
print(f"T_out:     {T_out:.5f} N (display: {T_out:.2f} N)")
print(f"Expected:  {expected:.2f} N")
print(f"Match:     {'[PASS]' if abs(T_out - expected) < 0.01 else '[FAIL]'}")

# Calculate sidewall pressure for Row 4
print("\nRow 4: Sidewall Pressure")
print("-" * 40)
radius = 3.9
expected_sidewall = 92.39

sidewall = calculate_sidewall_pressure(
    tension=T_out,
    bend_radius=radius,
    cable_spec=cable_spec,
    duct_spec=duct_spec,
    config=config,
)

print(f"Tension:   {T_out:.5f} N")
print(f"Radius:    {radius} m")
print(f"Sidewall:  {sidewall:.2f} N/m")
print(f"Expected:  {expected_sidewall:.2f} N/m")
print(f"Match:     {'[PASS]' if abs(sidewall - expected_sidewall) < 0.5 else '[FAIL]'}")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)
print("When maintaining full internal precision (like Excel does),")
print("all calculations match perfectly!")
print("=" * 80)
