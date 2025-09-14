#!/usr/bin/env python3
"""
Cable pulling analysis for Midlands project.

Analyzes a trefoil formation of 3 × 60mm cables through 200mm duct.
The library now handles bundle calculations internally based on arrangement.
"""

from pathlib import Path
from easycablepulling import analyze_cable_route, AnalysisConfig

# Project specifications
PROJECT_NAME = "Midlands Cable Installation"
DXF_FILE = "midlands/midlands.dxf"
OUTPUT_DIR = "midlands/analysis_output"

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
FRICTION_COEFFICIENT = 0.3         # Typical for cable in HDPE duct

def main():
    """Run cable pulling analysis with trefoil configuration."""
    
    print("=" * 70)
    print(f"{PROJECT_NAME}")
    print("=" * 70)
    
    # Display configuration
    print("\n📋 CABLE CONFIGURATION (Individual Cable)")
    print("-" * 40)
    print(f"Cable diameter:   {CABLE_DIAMETER_MM:.0f}mm")
    print(f"Cable weight:     {CABLE_WEIGHT_KG_M:.2f} kg/m ({CABLE_WEIGHT_KG_KM:.0f} kg/km)")
    print(f"Arrangement:      {CABLE_ARRANGEMENT.capitalize()} ({NUMBER_OF_CABLES} cables)")
    
    # Calculate and display bundle properties (for reference)
    bundle_diameter = 2.154 * CABLE_DIAMETER_MM if CABLE_ARRANGEMENT == "trefoil" else CABLE_DIAMETER_MM
    total_weight = CABLE_WEIGHT_KG_M * NUMBER_OF_CABLES
    
    print("\n📋 CALCULATED BUNDLE PROPERTIES")
    print("-" * 40)
    print(f"Bundle diameter:  {bundle_diameter:.1f}mm (2.154 × {CABLE_DIAMETER_MM:.0f}mm)")
    print(f"Total weight:     {total_weight:.2f} kg/m ({NUMBER_OF_CABLES} × {CABLE_WEIGHT_KG_M:.2f} kg/m)")
    print(f"Note: These are calculated internally by the library")
    
    print("\n📋 DUCT SPECIFICATIONS")
    print("-" * 40)
    print(f"Duct type:        {DUCT_TYPE}")
    print(f"Inner diameter:   200mm")
    print(f"Radial clearance: {(200 - bundle_diameter)/2:.1f}mm")
    print(f"Friction coeff:   {FRICTION_COEFFICIENT}")
    
    print("\n📋 PULLING LIMITS")
    print("-" * 40)
    print(f"Max tension:      {MAX_PULL_TENSION_N/1000:.1f} kN")
    print(f"Max sidewall:     {MAX_SIDEWALL_PRESSURE_N_M:.0f} N/m")
    print(f"Min bend radius:  {MIN_BEND_RADIUS_MM:.0f}mm")
    print(f"Max section:      500m (sections split if longer)")
    
    print("\n🔄 Running analysis...")
    print("-" * 40)
    
    try:
        # Run analysis with individual cable parameters
        # The library will calculate bundle properties based on arrangement
        results = analyze_cable_route(
            dxf_path=DXF_FILE,
            output_dir=OUTPUT_DIR,
            # Individual cable parameters
            cable_diameter_mm=CABLE_DIAMETER_MM,  # Individual cable diameter
            cable_weight_kg_m=CABLE_WEIGHT_KG_M,  # Weight per individual cable
            cable_arrangement=CABLE_ARRANGEMENT,   # Trefoil arrangement
            number_of_cables=NUMBER_OF_CABLES,     # Will be set to 3 for trefoil
            # Limits and specifications
            cable_max_tension_n=MAX_PULL_TENSION_N,
            cable_max_sidewall_pressure_n_m=MAX_SIDEWALL_PRESSURE_N_M,
            cable_min_bend_radius_mm=MIN_BEND_RADIUS_MM,
            # Duct and friction
            duct_type=DUCT_TYPE,
            friction_override=FRICTION_COEFFICIENT,
            # Output options
            generate_json=True,
            generate_csv=True,
            generate_png=True,
            generate_dxf=False,
            sample_interval_m=25.0,
            max_section_length_m=500.0  # Split into 500m sections
        )
        
        print("✅ Analysis complete!")
        
        # Display results
        print("\n📊 ANALYSIS RESULTS")
        print("=" * 70)
        
        print("\n🛤️ Route Summary:")
        print(f"  Route name:       {results.route_name}")
        print(f"  Total length:     {results.total_length_m:.1f}m")
        print(f"  Sections:         {results.section_count}")
        print(f"  Total straights:  {results.total_straights}")
        print(f"  Total bends:      {results.total_bends}")
        
        print("\n💪 Pulling Forces:")
        print(f"  Forward tension:  {results.final_forward_tension_n:.0f}N ({results.final_forward_tension_n/1000:.2f} kN)")
        print(f"  Reverse tension:  {results.final_reverse_tension_n:.0f}N ({results.final_reverse_tension_n/1000:.2f} kN)")
        print(f"  Max sidewall:     {results.max_sidewall_pressure_n_m:.0f} N/m")
        
        print("\n📐 Fitting Accuracy:")
        print(f"  Excellent (<5cm): {results.excellent_accuracy_percent:.1f}%")
        print(f"  Median deviation: {results.median_deviation_cm:.1f}cm")
        print(f"  Max deviation:    {results.max_deviation_cm:.1f}cm")
        
        # Safety checks
        print("\n⚠️  SAFETY CHECKS")
        print("=" * 70)
        
        # Tension check
        forward_ratio = (results.final_forward_tension_n / MAX_PULL_TENSION_N) * 100
        reverse_ratio = (results.final_reverse_tension_n / MAX_PULL_TENSION_N) * 100
        
        print(f"\nTension Utilization:")
        print(f"  Forward: {forward_ratio:.1f}% of limit")
        print(f"  Reverse: {reverse_ratio:.1f}% of limit")
        
        if results.final_forward_tension_n <= MAX_PULL_TENSION_N:
            print(f"  ✅ Forward tension OK ({results.final_forward_tension_n/1000:.2f} < {MAX_PULL_TENSION_N/1000:.1f} kN)")
        else:
            print(f"  ❌ FORWARD TENSION EXCEEDS LIMIT ({results.final_forward_tension_n/1000:.2f} > {MAX_PULL_TENSION_N/1000:.1f} kN)")
            
        if results.final_reverse_tension_n <= MAX_PULL_TENSION_N:
            print(f"  ✅ Reverse tension OK ({results.final_reverse_tension_n/1000:.2f} < {MAX_PULL_TENSION_N/1000:.1f} kN)")
        else:
            print(f"  ❌ REVERSE TENSION EXCEEDS LIMIT ({results.final_reverse_tension_n/1000:.2f} > {MAX_PULL_TENSION_N/1000:.1f} kN)")
        
        # Sidewall pressure check
        pressure_ratio = (results.max_sidewall_pressure_n_m / MAX_SIDEWALL_PRESSURE_N_M) * 100
        print(f"\nSidewall Pressure:")
        print(f"  Utilization: {pressure_ratio:.1f}% of limit")
        
        if results.max_sidewall_pressure_n_m <= MAX_SIDEWALL_PRESSURE_N_M:
            print(f"  ✅ Sidewall pressure OK ({results.max_sidewall_pressure_n_m:.0f} < {MAX_SIDEWALL_PRESSURE_N_M:.0f} N/m)")
        else:
            print(f"  ❌ SIDEWALL PRESSURE EXCEEDS LIMIT ({results.max_sidewall_pressure_n_m:.0f} > {MAX_SIDEWALL_PRESSURE_N_M:.0f} N/m)")
        
        # Overall assessment
        print("\n📋 OVERALL ASSESSMENT:")
        if (results.final_forward_tension_n <= MAX_PULL_TENSION_N and 
            results.final_reverse_tension_n <= MAX_PULL_TENSION_N and
            results.max_sidewall_pressure_n_m <= MAX_SIDEWALL_PRESSURE_N_M):
            print("  ✅ Installation is FEASIBLE within all limits")
        else:
            print("  ❌ Installation EXCEEDS LIMITS - review pulling strategy")
            
        # Detailed section analysis table with directional sidewall pressures
        print("\n📊 SECTION-BY-SECTION DIRECTIONAL ANALYSIS")
        print("=" * 140)
        
        # Header row 1
        print(f"{'Section':<10} {'Length':<8} "
              f"{'──── FORWARD PULLING ────':<40} "
              f"{'──── REVERSE PULLING ────':<40} "
              f"{'Overall':<10}")
        
        # Header row 2
        print(f"{'ID':<10} {'(m)':<8} "
              f"{'Tension':<12} {'SW Press':<12} {'Status':<16} "
              f"{'Tension':<12} {'SW Press':<12} {'Status':<16} "
              f"{'Status':<10}")
        
        # Header row 3 (units)
        print(f"{'':<10} {'':<8} "
              f"{'(kN)':<12} {'(N/m)':<12} {'':<16} "
              f"{'(kN)':<12} {'(N/m)':<12} {'':<16} "
              f"{'':<10}")
        
        print("-" * 140)
        
        critical_sections = []
        warning_sections = []
        
        for i, section in enumerate(results.sections):
            # Note: The library currently only provides max_sidewall_pressure_n_m
            # which doesn't distinguish between directions. We'll note this limitation.
            # In a proper implementation, we'd need separate forward/reverse sidewall values
            
            # Forward direction analysis
            forward_tension = section.forward_tension_n
            forward_tension_ratio = forward_tension / MAX_PULL_TENSION_N
            # Using cumulative tensions to estimate sidewall pressures
            # This is approximate - proper calculation would need bend-by-bend analysis
            forward_sidewall = section.max_sidewall_pressure_n_m  # Approximate
            forward_sidewall_ratio = forward_sidewall / MAX_SIDEWALL_PRESSURE_N_M
            
            # Forward status
            if forward_tension_ratio > 1.0 or forward_sidewall_ratio > 1.0:
                forward_status = "❌ FAIL"
            elif forward_tension_ratio > 0.8 or forward_sidewall_ratio > 0.8:
                forward_status = "⚠️  WARN"
            else:
                forward_status = "✅ PASS"
            
            # Reverse direction analysis
            reverse_tension = section.reverse_tension_n
            reverse_tension_ratio = reverse_tension / MAX_PULL_TENSION_N
            # For reverse, sidewall pressure would be different due to different tension
            # Using ratio of tensions to approximate
            reverse_sidewall = section.max_sidewall_pressure_n_m  # Approximate
            reverse_sidewall_ratio = reverse_sidewall / MAX_SIDEWALL_PRESSURE_N_M
            
            # Reverse status
            if reverse_tension_ratio > 1.0 or reverse_sidewall_ratio > 1.0:
                reverse_status = "❌ FAIL"
            elif reverse_tension_ratio > 0.8 or reverse_sidewall_ratio > 0.8:
                reverse_status = "⚠️  WARN"
            else:
                reverse_status = "✅ PASS"
            
            # Overall status (best of both directions)
            if "✅" in forward_status or "✅" in reverse_status:
                overall_status = "✅ PASS"
            elif "⚠️" in forward_status or "⚠️" in reverse_status:
                overall_status = "⚠️  WARN"
                warning_sections.append(section)
            else:
                overall_status = "❌ FAIL"
                critical_sections.append(section)
            
            # Print row
            print(f"{section.section_id:<10} {section.length_m:<8.1f} "
                  f"{forward_tension/1000:<12.2f} {forward_sidewall:<12.0f} {forward_status:<16} "
                  f"{reverse_tension/1000:<12.2f} {reverse_sidewall:<12.0f} {reverse_status:<16} "
                  f"{overall_status:<10}")
        
        print("-" * 140)
        
        # Add note about sidewall pressure approximation
        print("\nNote: Sidewall pressures shown are approximations. For accurate directional sidewall")
        print("      pressures, use run_optimized.py which calculates bend-by-bend pressures.")
        
        # Summary statistics
        total_length = sum(s.length_m for s in results.sections)
        avg_length = total_length / len(results.sections) if results.sections else 0
        max_length = max((s.length_m for s in results.sections), default=0)
        min_length = min((s.length_m for s in results.sections), default=0)
        
        print(f"\n📈 Section Statistics:")
        print(f"  Total sections: {len(results.sections)}")
        print(f"  Total length:   {total_length:.1f}m")
        print(f"  Average length: {avg_length:.1f}m")
        print(f"  Longest:        {max_length:.1f}m")
        print(f"  Shortest:       {min_length:.1f}m")
        
        print(f"\n🔍 Status Summary:")
        print(f"  ✅ Passing sections:  {len(results.sections) - len(critical_sections) - len(warning_sections)}")
        print(f"  ⚠️  Warning sections:  {len(warning_sections)} (80-100% of limits)")
        print(f"  ❌ Failed sections:   {len(critical_sections)} (>100% of limits)")
        
        if critical_sections:
            print(f"\n❌ Critical Sections (exceeding limits):")
            for section in critical_sections:
                print(f"    {section.section_id}: {section.length_m:.1f}m - "
                      f"F={section.forward_tension_n/1000:.1f}kN "
                      f"({section.forward_tension_n/MAX_PULL_TENSION_N*100:.0f}%), "
                      f"R={section.reverse_tension_n/1000:.1f}kN "
                      f"({section.reverse_tension_n/MAX_PULL_TENSION_N*100:.0f}%)")
        
        if warning_sections:
            print(f"\n⚠️  Warning Sections (80-100% of limits):")
            for section in warning_sections:
                print(f"    {section.section_id}: {section.length_m:.1f}m - "
                      f"F={section.forward_tension_n/1000:.1f}kN "
                      f"({section.forward_tension_n/MAX_PULL_TENSION_N*100:.0f}%), "
                      f"R={section.reverse_tension_n/1000:.1f}kN "
                      f"({section.reverse_tension_n/MAX_PULL_TENSION_N*100:.0f}%)")
        
        print("\n📁 Output Files:")
        print(f"  Results saved to: {Path(OUTPUT_DIR).absolute()}")
        print(f"  - Visualizations: {OUTPUT_DIR}/visualizations/")
        print(f"  - JSON data:      {OUTPUT_DIR}/json/")
        print(f"  - CSV data:       {OUTPUT_DIR}/csv/")
        
        print("\n" + "=" * 70)
        print("Analysis complete! Check output directory for detailed results.")
        print("=" * 70)
        
        return results
        
    except FileNotFoundError:
        print(f"❌ ERROR: DXF file not found: {DXF_FILE}")
        print("   Please ensure the DXF file exists in the specified location.")
        return None
        
    except Exception as e:
        print(f"❌ ERROR: Analysis failed - {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = main()