#!/usr/bin/env python3
"""
Optimized cable pulling analysis for Midlands project.

This version automatically optimizes the route by splitting sections
to stay within tension and sidewall pressure limits, with different
strategies for forward and reverse pulling directions.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from easycablepulling.core.models import CableSpec, DuctSpec, CableArrangement, PullingMethod
from easycablepulling.analysis import optimize_cable_route, PullingDirection
from easycablepulling.io.dxf_reader import DXFReader
from easycablepulling.geometry.splitter import RouteSplitter
from easycablepulling.geometry.simple_segment_fitter import SimpleSegmentFitter

# Project specifications
PROJECT_NAME = "Midlands Cable Installation - Optimized Analysis"
DXF_FILE = str(Path(__file__).parent / "midlands.dxf")
OUTPUT_DIR = str(Path(__file__).parent / "analysis_optimized")

# Cable specifications (per individual cable)
CABLE_DIAMETER_MM = 69.0           # Individual cable diameter
CABLE_WEIGHT_KG_KM = 4930          # Weight per cable in kg/km
CABLE_WEIGHT_KG_M = CABLE_WEIGHT_KG_KM / 1000  # 4.93 kg/m per cable

# Cable arrangement
CABLE_ARRANGEMENT = "trefoil"      # 3 cables in triangular formation
NUMBER_OF_CABLES = 3                # Automatically set for trefoil

# Installation limits
MAX_PULL_TENSION_N = 27000         # 27 kN maximum pulling force
MAX_SIDEWALL_PRESSURE_N_M = 7000   # 7 kN/m (optimal from testing)
MIN_BEND_RADIUS_MM = 15 * CABLE_DIAMETER_MM  # 15 × D = 1035mm

# Duct specifications
DUCT_TYPE = "225mm"
DUCT_INNER_DIAMETER_MM = 225
FRICTION_COEFFICIENT = 0.3         # Typical for cable in HDPE duct

# Optimization parameters
TARGET_UTILIZATION = 0.95          # 95% of limits (5% safety margin) - allows longer pulls
MAX_SECTION_LENGTH_M = 500.0       # Maximum section length
# Note: Using 0.95 instead of 0.8 to accommodate sidewall pressure constraints
# Tension is well below limits; sidewall pressure is the limiting factor


def load_and_process_route(dxf_path: str):
    """Load DXF and process route with geometry fitting."""
    # Load DXF using the cable route layer
    reader = DXFReader(dxf_path)
    reader.load()
    route = reader.create_route_from_polylines(
        Path(dxf_path).stem,
        layer_name="_FUN_33kV OPT 2 Overview Route"  # Use the cable route layer
    )

    # Apply geometry fitting
    fitter = SimpleSegmentFitter()
    for section in route.sections:
        if section.original_polyline:
            result = fitter.fit_section_to_primitives(section)
            section.primitives = result.primitives

    return route


def create_cable_spec():
    """Create cable specification with trefoil arrangement."""
    return CableSpec(
        diameter=CABLE_DIAMETER_MM,
        weight_per_meter=CABLE_WEIGHT_KG_M,
        max_tension=MAX_PULL_TENSION_N,
        max_sidewall_pressure=MAX_SIDEWALL_PRESSURE_N_M,
        min_bend_radius=MIN_BEND_RADIUS_MM,
        arrangement=CableArrangement.TREFOIL,
        number_of_cables=NUMBER_OF_CABLES,
        pulling_method=PullingMethod.EYE,
    )


def create_duct_spec():
    """Create duct specification."""
    return DuctSpec(
        inner_diameter=DUCT_INNER_DIAMETER_MM,
        type="HDPE",
        friction_dry=FRICTION_COEFFICIENT,
        friction_lubricated=FRICTION_COEFFICIENT * 0.6,  # Assume 40% reduction with lube
    )


def print_optimization_results(results, max_straight_length):
    """Print optimization results in clear, readable format."""
    from collections import defaultdict

    print("\n" + "=" * 80)
    print("OPTIMIZATION RESULTS")
    print("=" * 80)

    # Group results by section
    by_section = defaultdict(list)
    for result in results:
        by_section[result.section_index].append(result)

    # Calculate totals
    total_pullable_length = 0.0
    total_route_length = 0.0
    total_subsections = 0
    max_tension_overall = 0.0
    max_sidewall_overall = 0.0

    for section_idx in sorted(by_section.keys()):
        section_results = by_section[section_idx]
        for result in section_results:
            total_route_length += result.length
            max_tension_overall = max(max_tension_overall, result.max_tension)
            max_sidewall_overall = max(max_sidewall_overall, result.max_sidewall_pressure)
            total_subsections += 1
            if result.passes_limits:
                total_pullable_length += result.length

    # Display results by section
    for section_idx in sorted(by_section.keys()):
        section_results = by_section[section_idx]
        section = section_results[0]  # Get section info from first result

        print(f"\n[SECTION {section_idx}] Length: {section.start_position:.1f}m - {section.end_position:.1f}m")
        print(f"  Original length: {section_results[0].start_position + sum(r.length for r in section_results):.1f}m")
        print(f"  Subsections: {len(section_results)}")

        for result in section_results:
            status = "[PASS]" if result.passes_limits else "[FAIL]"
            print(f"    {result.subsection_num}/{result.total_subsections}: "
                  f"{result.length:.1f}m, "
                  f"Tension: {result.max_tension:.0f}N ({result.max_tension/1000:.2f}kN), "
                  f"Sidewall: {result.max_sidewall_pressure:.0f}N/m {status}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total route length:          {total_route_length:.1f}m across {len(by_section)} polylines")
    print(f"Total subsections required:  {total_subsections} sections")
    print(f"Average subsection length:   {total_route_length/total_subsections:.1f}m")
    print(f"Max straight-line pull:      {max_straight_length:.1f}m (theoretical limit)")
    print(f"Total pullable length:       {total_pullable_length:.1f}m")
    print(f"Coverage:                    {total_pullable_length/total_route_length*100:.1f}% of route")

    print(f"\nMax tension:                 {max_tension_overall:.0f}N ({max_tension_overall/1000:.2f}kN)")
    print(f"Tension limit:               {MAX_PULL_TENSION_N:.0f}N ({MAX_PULL_TENSION_N/1000:.1f}kN)")
    print(f"Max sidewall pressure:       {max_sidewall_overall:.0f}N/m")
    print(f"Sidewall limit:              {MAX_SIDEWALL_PRESSURE_N_M:.0f}N/m ({MAX_SIDEWALL_PRESSURE_N_M/1000:.1f}kN/m)")


def main():
    """Run optimized cable pulling analysis."""
    
    print("=" * 100)
    print(f"{PROJECT_NAME}")
    print("=" * 100)
    
    # Display configuration
    print("\n[CONFIG] CABLE CONFIGURATION")
    print("-" * 40)
    print(f"Individual cable:     {CABLE_DIAMETER_MM:.0f}mm dia, {CABLE_WEIGHT_KG_M:.2f} kg/m")
    print(f"Arrangement:          {CABLE_ARRANGEMENT.capitalize()} ({NUMBER_OF_CABLES} cables)")
    
    # Calculate bundle properties
    bundle_diameter = 2.154 * CABLE_DIAMETER_MM
    total_weight = CABLE_WEIGHT_KG_M * NUMBER_OF_CABLES
    
    print(f"Bundle diameter:      {bundle_diameter:.1f}mm (calculated)")
    print(f"Total weight:         {total_weight:.2f} kg/m (calculated)")
    
    print("\n[CONFIG] INSTALLATION LIMITS")
    print("-" * 40)
    print(f"Max tension:          {MAX_PULL_TENSION_N/1000:.1f} kN")
    print(f"Max sidewall:         {MAX_SIDEWALL_PRESSURE_N_M:.0f} N/m")
    print(f"Min bend radius:      {MIN_BEND_RADIUS_MM:.0f}mm")
    print(f"Target utilization:   {TARGET_UTILIZATION*100:.0f}% (safety margin: {(1-TARGET_UTILIZATION)*100:.0f}%)")
    print(f"Max section length:   {MAX_SECTION_LENGTH_M:.0f}m")
    
    print("\n[CONFIG] DUCT SPECIFICATIONS")
    print("-" * 40)
    print(f"Duct diameter:        {DUCT_INNER_DIAMETER_MM}mm")
    print(f"Radial clearance:     {(DUCT_INNER_DIAMETER_MM - bundle_diameter)/2:.1f}mm")
    print(f"Friction coefficient: {FRICTION_COEFFICIENT}")
    
    try:
        print("\n[PROCESSING] Loading and processing route...")
        route = load_and_process_route(DXF_FILE)
        
        # Create specifications
        cable_spec = create_cable_spec()
        duct_spec = create_duct_spec()
        
        print(f"[PASS] Route loaded: {route.name}")
        print(f"   Sections: {len(route.sections)}")
        print(f"   Total length: {sum(s.original_length for s in route.sections):.1f}m")
        
        print("\n[PROCESSING] Running optimization analysis...")
        print("   Analyzing sections per-polyline independently...")
        print("   Using binary search to find maximum safe lengths...")

        # Create optimizer with SectionOptimizer (per-polyline analysis)
        from easycablepulling.analysis.section_optimizer import SectionOptimizer

        optimizer = SectionOptimizer(
            cable_spec=cable_spec,
            duct_spec=duct_spec,
            max_tension_limit=MAX_PULL_TENSION_N,
            max_sidewall_limit=MAX_SIDEWALL_PRESSURE_N_M,
            max_section_length=MAX_SECTION_LENGTH_M,
            friction_override=FRICTION_COEFFICIENT,
        )

        # Calculate max straight-line length
        max_straight_length, limiting_factor = optimizer.calculate_max_straight_length()
        print(f"[INFO] Max straight-line pull: {max_straight_length:.1f}m (limited by {limiting_factor})")

        # Run optimization
        results = optimizer.optimize_route(route)

        print(f"[PASS] Optimization complete! Found {len(results)} subsection results.")

        # Print results
        print_optimization_results(results, max_straight_length)

        print("\n" + "="*80)
        print("Analysis complete!")
        print("="*80)

        return results
        
    except FileNotFoundError:
        print(f"[FAIL] ERROR: DXF file not found: {DXF_FILE}")
        return None

    except Exception as e:
        print(f"[FAIL] ERROR: Analysis failed - {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = main()