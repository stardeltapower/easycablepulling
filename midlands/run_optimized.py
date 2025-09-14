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
DXF_FILE = "midlands/midlands.dxf"
OUTPUT_DIR = "midlands/analysis_optimized"

# Cable specifications (per individual cable)
CABLE_DIAMETER_MM = 60.0           # Individual cable diameter
CABLE_WEIGHT_KG_KM = 3520          # Weight per cable in kg/km
CABLE_WEIGHT_KG_M = CABLE_WEIGHT_KG_KM / 1000  # 3.52 kg/m per cable

# Cable arrangement
CABLE_ARRANGEMENT = "trefoil"      # 3 cables in triangular formation
NUMBER_OF_CABLES = 3                # Automatically set for trefoil

# Installation limits
MAX_PULL_TENSION_N = 17800         # 17.8 kN maximum pulling force
MAX_SIDEWALL_PRESSURE_N_M = 3000   # Typical for MV cables
MIN_BEND_RADIUS_MM = 15 * CABLE_DIAMETER_MM  # 15 × D = 900mm

# Duct specifications
DUCT_TYPE = "200mm"
DUCT_INNER_DIAMETER_MM = 200
FRICTION_COEFFICIENT = 0.3         # Typical for cable in HDPE duct

# Optimization parameters
TARGET_UTILIZATION = 0.8           # 80% of limits (20% safety margin)
MAX_SECTION_LENGTH_M = 500.0       # Maximum section length


def load_and_process_route(dxf_path: str):
    """Load DXF and process route with geometry fitting."""
    # Load DXF
    reader = DXFReader(dxf_path)
    reader.load()
    route = reader.create_route_from_polylines(Path(dxf_path).stem)
    
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


def print_optimization_results(opt_result, direction_name):
    """Print detailed optimization results for a direction."""
    print(f"\n{'='*100}")
    print(f"📐 {direction_name.upper()} PULLING OPTIMIZATION")
    print(f"{'='*100}")
    
    # Summary
    print(f"\n📊 Optimization Summary:")
    print(f"  Original sections:    {opt_result.original_sections}")
    print(f"  Optimized sections:   {opt_result.optimized_sections}")
    print(f"  Total route length:   {opt_result.total_length:.1f}m")
    print(f"  Target utilization:   {opt_result.target_utilization*100:.0f}%")
    print(f"  Max section length:   {opt_result.max_section_length:.0f}m")
    
    # Peak values
    print(f"\n💪 Peak Values:")
    print(f"  Max tension:          {opt_result.max_tension:.0f}N ({opt_result.max_tension/1000:.2f} kN)")
    print(f"  Max sidewall:         {opt_result.max_sidewall_pressure:.0f} N/m")
    print(f"  Tension utilization:  {opt_result.max_tension_utilization*100:.1f}%")
    print(f"  Sidewall utilization: {opt_result.max_sidewall_utilization*100:.1f}%")
    
    # Overall status
    print(f"\n🎯 Overall Status:")
    if opt_result.feasible:
        print(f"  ✅ FEASIBLE - All sections within limits with {(1-opt_result.target_utilization)*100:.0f}% margin")
    else:
        print(f"  ❌ NOT FEASIBLE - Some sections exceed limits even after optimization")
    
    # Detailed section table
    print(f"\n📋 OPTIMIZED SECTIONS - {direction_name} Pulling")
    print("-" * 100)
    print(f"{'Section':<10} {'Start':<8} {'End':<8} {'Length':<8} "
          f"{'Max Ten.':<10} {'Ten.%':<8} {'Max SW':<10} {'SW%':<8} {'Status':<10}")
    print(f"{'ID':<10} {'(m)':<8} {'(m)':<8} {'(m)':<8} "
          f"{'(kN)':<10} {'':<8} {'(N/m)':<10} {'':<8} {'':<10}")
    print("-" * 100)
    
    for section in opt_result.sections:
        # Determine status
        if section.overall_pass:
            if section.tension_utilization > 0.7 or section.sidewall_utilization > 0.7:
                status = "⚠️  WARN"
            else:
                status = "✅ PASS"
        else:
            status = "❌ FAIL"
        
        print(f"{section.section_id:<10} "
              f"{section.start_position:<8.0f} "
              f"{section.end_position:<8.0f} "
              f"{section.length:<8.1f} "
              f"{section.max_tension/1000:<10.2f} "
              f"{section.tension_utilization*100:<8.0f} "
              f"{section.max_sidewall_pressure:<10.0f} "
              f"{section.sidewall_utilization*100:<8.0f} "
              f"{status:<10}")
    
    print("-" * 100)
    
    # Statistics
    avg_length = opt_result.total_length / opt_result.optimized_sections
    max_length = max(s.length for s in opt_result.sections)
    min_length = min(s.length for s in opt_result.sections)
    
    print(f"\n📈 Length Distribution:")
    print(f"  Average: {avg_length:.1f}m")
    print(f"  Maximum: {max_length:.1f}m")
    print(f"  Minimum: {min_length:.1f}m")
    
    # Critical sections
    critical = [s for s in opt_result.sections if not s.overall_pass]
    warning = [s for s in opt_result.sections 
               if s.overall_pass and (s.tension_utilization > 0.7 or s.sidewall_utilization > 0.7)]
    
    if critical:
        print(f"\n❌ Critical Sections (exceeding limits):")
        for s in critical:
            print(f"  {s.section_id}: {s.length:.1f}m - "
                  f"Tension: {s.tension_utilization*100:.0f}%, "
                  f"Sidewall: {s.sidewall_utilization*100:.0f}%")
    
    if warning:
        print(f"\n⚠️  Warning Sections (>70% utilization):")
        for s in warning:
            print(f"  {s.section_id}: {s.length:.1f}m - "
                  f"Tension: {s.tension_utilization*100:.0f}%, "
                  f"Sidewall: {s.sidewall_utilization*100:.0f}%")


def main():
    """Run optimized cable pulling analysis."""
    
    print("=" * 100)
    print(f"{PROJECT_NAME}")
    print("=" * 100)
    
    # Display configuration
    print("\n📋 CABLE CONFIGURATION")
    print("-" * 40)
    print(f"Individual cable:     {CABLE_DIAMETER_MM:.0f}mm dia, {CABLE_WEIGHT_KG_M:.2f} kg/m")
    print(f"Arrangement:          {CABLE_ARRANGEMENT.capitalize()} ({NUMBER_OF_CABLES} cables)")
    
    # Calculate bundle properties
    bundle_diameter = 2.154 * CABLE_DIAMETER_MM
    total_weight = CABLE_WEIGHT_KG_M * NUMBER_OF_CABLES
    
    print(f"Bundle diameter:      {bundle_diameter:.1f}mm (calculated)")
    print(f"Total weight:         {total_weight:.2f} kg/m (calculated)")
    
    print("\n📋 INSTALLATION LIMITS")
    print("-" * 40)
    print(f"Max tension:          {MAX_PULL_TENSION_N/1000:.1f} kN")
    print(f"Max sidewall:         {MAX_SIDEWALL_PRESSURE_N_M:.0f} N/m")
    print(f"Min bend radius:      {MIN_BEND_RADIUS_MM:.0f}mm")
    print(f"Target utilization:   {TARGET_UTILIZATION*100:.0f}% (safety margin: {(1-TARGET_UTILIZATION)*100:.0f}%)")
    print(f"Max section length:   {MAX_SECTION_LENGTH_M:.0f}m")
    
    print("\n📋 DUCT SPECIFICATIONS")
    print("-" * 40)
    print(f"Duct diameter:        {DUCT_INNER_DIAMETER_MM}mm")
    print(f"Radial clearance:     {(DUCT_INNER_DIAMETER_MM - bundle_diameter)/2:.1f}mm")
    print(f"Friction coefficient: {FRICTION_COEFFICIENT}")
    
    try:
        print("\n🔄 Loading and processing route...")
        route = load_and_process_route(DXF_FILE)
        
        # Create specifications
        cable_spec = create_cable_spec()
        duct_spec = create_duct_spec()
        
        print(f"✅ Route loaded: {route.name}")
        print(f"   Sections: {len(route.sections)}")
        print(f"   Total length: {sum(s.original_length for s in route.sections):.1f}m")
        
        print("\n🔄 Running optimization analysis...")
        print("   Analyzing forward and reverse pulling directions...")
        print("   Automatically splitting sections to stay within limits...")
        
        # Run optimization
        forward_result, reverse_result = optimize_cable_route(
            route=route,
            cable_spec=cable_spec,
            duct_spec=duct_spec,
            target_utilization=TARGET_UTILIZATION,
            max_section_length=MAX_SECTION_LENGTH_M,
            friction_override=FRICTION_COEFFICIENT,
        )
        
        print("✅ Optimization complete!")
        
        # Print results for both directions
        print_optimization_results(forward_result, "Forward")
        print_optimization_results(reverse_result, "Reverse")
        
        # Comparison
        print("\n" + "="*100)
        print("🔄 DIRECTION COMPARISON")
        print("="*100)
        
        print(f"\n{'Metric':<30} {'Forward':<20} {'Reverse':<20} {'Better':<15}")
        print("-" * 85)
        
        # Number of sections
        forward_sections = forward_result.optimized_sections
        reverse_sections = reverse_result.optimized_sections
        better_sections = "Forward ↑" if forward_sections <= reverse_sections else "Reverse ↑"
        print(f"{'Required sections:':<30} {forward_sections:<20} {reverse_sections:<20} {better_sections:<15}")
        
        # Max tension
        forward_tension_pct = forward_result.max_tension_utilization * 100
        reverse_tension_pct = reverse_result.max_tension_utilization * 100
        better_tension = "Forward ↑" if forward_tension_pct <= reverse_tension_pct else "Reverse ↑"
        print(f"{'Max tension utilization:':<30} {f'{forward_tension_pct:.1f}%':<20} "
              f"{f'{reverse_tension_pct:.1f}%':<20} {better_tension:<15}")
        
        # Max sidewall
        forward_sidewall_pct = forward_result.max_sidewall_utilization * 100
        reverse_sidewall_pct = reverse_result.max_sidewall_utilization * 100
        better_sidewall = "Forward ↑" if forward_sidewall_pct <= reverse_sidewall_pct else "Reverse ↑"
        print(f"{'Max sidewall utilization:':<30} {f'{forward_sidewall_pct:.1f}%':<20} "
              f"{f'{reverse_sidewall_pct:.1f}%':<20} {better_sidewall:<15}")
        
        # Feasibility
        forward_feasible = "✅ Yes" if forward_result.feasible else "❌ No"
        reverse_feasible = "✅ Yes" if reverse_result.feasible else "❌ No"
        if forward_result.feasible and reverse_result.feasible:
            better_feasible = "Both ✅"
        elif forward_result.feasible:
            better_feasible = "Forward only ↑"
        elif reverse_result.feasible:
            better_feasible = "Reverse only ↑"
        else:
            better_feasible = "Neither ❌"
        print(f"{'Feasible:':<30} {forward_feasible:<20} {reverse_feasible:<20} {better_feasible:<15}")
        
        print("-" * 85)
        
        # Recommendation
        print("\n🎯 RECOMMENDATION:")
        if forward_result.feasible and reverse_result.feasible:
            if forward_sections < reverse_sections:
                print("  ✅ Forward pulling recommended - fewer sections required")
            elif reverse_sections < forward_sections:
                print("  ✅ Reverse pulling recommended - fewer sections required")
            elif forward_sidewall_pct < reverse_sidewall_pct:
                print("  ✅ Forward pulling recommended - lower sidewall pressure")
            elif reverse_sidewall_pct < forward_sidewall_pct:
                print("  ✅ Reverse pulling recommended - lower sidewall pressure")
            else:
                print("  ✅ Either direction feasible - similar performance")
        elif forward_result.feasible:
            print("  ⚠️  Only forward pulling is feasible")
        elif reverse_result.feasible:
            print("  ⚠️  Only reverse pulling is feasible")
        else:
            print("  ❌ Neither direction feasible with current parameters")
            print("     Consider: reducing cable size, using lubricant, or different routing")
        
        print("\n" + "="*100)
        print("Analysis complete!")
        print("="*100)
        
        return forward_result, reverse_result
        
    except FileNotFoundError:
        print(f"❌ ERROR: DXF file not found: {DXF_FILE}")
        return None, None
        
    except Exception as e:
        print(f"❌ ERROR: Analysis failed - {e}")
        import traceback
        traceback.print_exc()
        return None, None


if __name__ == "__main__":
    forward, reverse = main()