#!/usr/bin/env python3
"""Test the section optimizer with actual route data."""

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

# Cable specs from the Midlands project
cable_spec = CableSpec(
    diameter=69.0,  # mm
    weight_per_meter=4.93,  # kg/m (4930 kg/km)
    max_tension=27000.0,  # N
    max_sidewall_pressure=2400.0,  # N/m (estimated for this cable)
    min_bend_radius=1500.0,  # mm (conservative estimate)
    pulling_method=PullingMethod.EYE,
    arrangement=CableArrangement.TREFOIL,
    number_of_cables=3,
)

# Create duct spec for 225mm HDPE duct
duct_spec = DuctSpec(
    inner_diameter=225.0,  # mm (approximate inner diameter)
    type="HDPE",
    friction_dry=0.40,  # Friction coefficient for HDPE
    friction_lubricated=0.18,  # Lubricated friction
)

# Parse the DXF file
dxf_path = Path(
    r"C:\Users\rsmith\PycharmProjects\easycablepulling\midlands\midlands.dxf"
)
if not dxf_path.exists():
    print(f"ERROR: DXF file not found at {dxf_path}")
    sys.exit(1)

print("[INFO] Parsing DXF file...")
try:
    reader = DXFReader(dxf_path)
    reader.load()
    route = reader.create_route_from_polylines(
        dxf_path.stem,
        layer_name="_FUN_33kV OPT 2 Overview Route",
    )
    print(f"[INFO] Parsed route with {len(route.sections)} sections")
except Exception as e:
    print(f"ERROR: Failed to parse DXF file: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# Fit primitives to sections
print("[INFO] Fitting primitives to sections...")
try:
    from easycablepulling.geometry.simple_segment_fitter import SimpleSegmentFitter
    from easycablepulling.inventory.duct_inventory import DuctInventory as DuctInv

    fitter = SimpleSegmentFitter(
        duct_inventory=DuctInv("225mm"),
        standard_radius=3.9,  # Standard duct bend radius
    )

    for section in route.sections:
        result = fitter.fit_section_to_primitives(section)
        section.primitives = result.primitives
        print(f"  Section {section.id}: {len(section.primitives)} primitives, length {section.total_length:.1f}m")
except Exception as e:
    print(f"ERROR: Failed to fit primitives: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Create section optimizer
print("[INFO] Creating section optimizer...")
optimizer = SectionOptimizer(
    cable_spec=cable_spec,
    duct_spec=duct_spec,
    max_tension_limit=27000.0,
    max_sidewall_limit=7000.0,  # 7 kN/m sidewall pressure limit (optimal from testing)
    max_section_length=500.0,
    friction_override=None,
)

# Calculate maximum straight-line pull length
print("[INFO] Calculating maximum straight-line pull length...")
max_straight_length, limiting_factor = optimizer.calculate_max_straight_length()
print(f"[INFO] Max straight-line pull length: {max_straight_length:.1f}m (limited by {limiting_factor})")

# Optimize the route in both directions
print("[INFO] Optimizing route (analyzing each polyline independently)...")
print("[INFO] Testing FORWARD direction...")
try:
    results_forward = optimizer.optimize_route(route)
    print(f"[INFO] Forward optimization complete. Found {len(results_forward)} subsection results.")
except Exception as e:
    print(f"ERROR: Forward optimization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("[INFO] Testing REVERSE direction...")
try:
    results_reverse = optimizer.optimize_route(route)
    print(f"[INFO] Reverse optimization complete. Found {len(results_reverse)} subsection results.")
except Exception as e:
    print(f"ERROR: Reverse optimization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

results = results_forward  # Use forward for display below

# Analyze results
print("\n" + "=" * 80)
print("OPTIMIZATION RESULTS")
print("=" * 80)

if not results:
    print("ERROR: No results returned from optimization")
    sys.exit(1)

# Group results by section
from collections import defaultdict

by_section = defaultdict(list)
for result in results:
    by_section[result.section_index].append(result)

# Display results
total_pullable_length = 0.0
for section_idx in sorted(by_section.keys()):
    section = route.sections[section_idx]
    section_results = by_section[section_idx]

    print(f"\n[SECTION {section_idx}] Original length: {section.original_length:.1f}m")
    print(f"Total primitives: {len(section.primitives)}")

    # Group by number of subsections
    by_subsections = defaultdict(list)
    for result in section_results:
        by_subsections[result.total_subsections].append(result)

    for num_subsections in sorted(by_subsections.keys()):
        sub_results = by_subsections[num_subsections]
        all_pass = all(r.passes_limits for r in sub_results)
        status = "[PASS]" if all_pass else "[FAIL]"

        print(f"\n  {status} {num_subsections} subsections:")

        for result in sub_results:
            pass_status = "[PASS]" if result.passes_limits else "[FAIL]"
            print(
                f"    Subsection {result.subsection_num}/{result.total_subsections}: "
                f"{result.length:.1f}m, "
                f"Tension: {result.max_tension:.0f}N, "
                f"Sidewall: {result.max_sidewall_pressure:.0f}N/m {pass_status}"
            )

            if result.passes_limits:
                total_pullable_length += result.length

# Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
total_route_length = sum(s.original_length for s in route.sections)
print(
    f"Total route length: {total_route_length:.1f}m across {len(route.sections)} polylines"
)
print(f"Total pullable length: {total_pullable_length:.1f}m")
print(
    f"Coverage: {total_pullable_length/total_route_length*100:.1f}% of route can be pulled"
)
print(f"Max tension limit: {cable_spec.max_tension:.0f}N")
print(f"Max sidewall limit: {duct_spec.max_sidewall_pressure if hasattr(duct_spec, 'max_sidewall_pressure') else 'N/A'}N/m")
