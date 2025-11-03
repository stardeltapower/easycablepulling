#!/usr/bin/env python3
"""Debug the section optimizer algorithm."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from easycablepulling.core.models import (
    CableSpec,
    DuctSpec,
    CableArrangement,
    PullingMethod,
)
from easycablepulling.io.dxf_reader import DXFReader
from easycablepulling.analysis.section_optimizer import SectionOptimizer
from easycablepulling.geometry.simple_segment_fitter import SimpleSegmentFitter
from easycablepulling.inventory.duct_inventory import DuctInventory as DuctInv

# Cable specs
cable_spec = CableSpec(
    diameter=69.0,
    weight_per_meter=4.93,
    max_tension=27000.0,
    max_sidewall_pressure=2400.0,
    min_bend_radius=1500.0,
    pulling_method=PullingMethod.EYE,
    arrangement=CableArrangement.TREFOIL,
    number_of_cables=3,
)

# Duct spec
duct_spec = DuctSpec(
    inner_diameter=225.0,
    type="HDPE",
    friction_dry=0.40,
    friction_lubricated=0.18,
)

# Parse DXF
dxf_path = Path(
    r"C:\Users\rsmith\PycharmProjects\easycablepulling\midlands\midlands.dxf"
)
reader = DXFReader(dxf_path)
reader.load()
route = reader.create_route_from_polylines(
    dxf_path.stem,
    layer_name="_FUN_33kV OPT 2 Overview Route",
)

# Fit primitives
fitter = SimpleSegmentFitter(
    duct_inventory=DuctInv("225mm"),
    standard_radius=3.9,
)

for section in route.sections:
    result = fitter.fit_section_to_primitives(section)
    section.primitives = result.primitives

# Create optimizer
optimizer = SectionOptimizer(
    cable_spec=cable_spec,
    duct_spec=duct_spec,
    max_tension_limit=27000.0,
    max_sidewall_limit=2400.0,
    max_section_length=500.0,
    friction_override=None,
)

# Test just section 0
section = route.sections[0]
print(f"[DEBUG] Testing Section 0")
print(f"  Length: {section.original_length:.1f}m")
print(f"  Primitives: {len(section.primitives)}")
print()

# Test safe split finding
print("[DEBUG] Finding safe split points...")
split_points = optimizer._find_safe_split_points(section)
print(f"  Split points: {[f'{p:.1f}' for p in split_points]}")
print(f"  Number of splits: {len(split_points) - 1}")
print()

# Test subsection creation
print("[DEBUG] Creating subsections at split points...")
subsections = optimizer._create_subsections_at_splits(section, 0, split_points)
print(f"  Created {len(subsections)} subsections")
for sub in subsections:
    status = "[PASS]" if sub.passes_limits else "[FAIL]"
    print(f"    {sub.subsection_num}/{sub.total_subsections}: {sub.length:.1f}m, "
          f"T={sub.max_tension:.0f}N, SW={sub.max_sidewall_pressure:.0f}N/m {status}")
print()

# Test rebalancing
if len(subsections) > 1:
    print("[DEBUG] Attempting rebalance...")
    rebalanced = optimizer._rebalance_subsection_lengths(section, 0, subsections)
    print(f"  After rebalance: {len(rebalanced)} subsections")
    for sub in rebalanced:
        status = "[PASS]" if sub.passes_limits else "[FAIL]"
        print(f"    {sub.subsection_num}/{sub.total_subsections}: {sub.length:.1f}m, "
              f"T={sub.max_tension:.0f}N, SW={sub.max_sidewall_pressure:.0f}N/m {status}")
