#!/usr/bin/env python3
"""
Midlands Cable Pulling Analysis

Analyzes the Midlands DXF route with optimized splitting using:
- AEIC calculation standard
- Bidirectional equal splitting strategy
- 3 cables in trefoil arrangement
- Generates comprehensive reports and visualizations
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from easycablepulling.core.cable_analysis_pipeline import CableAnalysisPipeline, AnalysisConfig

def main():
    """Run Midlands cable analysis with optimized splitting."""

    print("=" * 80)
    print("MIDLANDS CABLE PULLING ANALYSIS")
    print("=" * 80)

    # Configuration
    config = AnalysisConfig(
        # Duct
        duct_type="225mm",  # 215mm inner diameter

        # Cable specifications (per individual cable)
        cable_diameter_mm=69.0,
        cable_weight_kg_m=4.93,
        cable_max_tension_n=27000.0,        # 27 kN limit
        cable_max_sidewall_pressure_n_m=7000.0,  # 7 kN/m limit
        cable_min_bend_radius_mm=980.0,
        number_of_cables=3,
        cable_arrangement="trefoil",

        # Calculation standard
        calculation_standard="aeic",

        # Optimization
        splitting_method="optimizer",      # Intelligent tension/pressure-based splitting
        target_utilization=0.95,           # Split at 95% utilization
        max_section_length_m=490.0,        # 490m max length (500m drums with overlap)

        # Friction (trefoil-adjusted: 0.3 base × 1.3 multiplier)
        friction_override=0.39,

        # Output generation
        generate_json=True,
        generate_csv=True,
        generate_png=True,  # Generate route and section visualizations
    )

    # Display configuration
    print("\nConfiguration:")
    print(f"  Standard: {config.calculation_standard.upper()}")
    print(f"  Cable: {config.cable_diameter_mm}mm, {config.cable_weight_kg_m} kg/m")
    print(f"  Cables: {config.number_of_cables} in {config.cable_arrangement}")
    print(f"  Duct: {config.duct_type} (215mm inner diameter)")
    print(f"  Friction: {config.friction_override}")
    print(f"  Limits: {config.cable_max_tension_n/1000:.0f} kN tension, "
          f"{config.cable_max_sidewall_pressure_n_m/1000:.0f} kN/m sidewall")
    print(f"  Optimization: {config.splitting_method} with {config.target_utilization*100:.0f}% target")

    # File paths
    dxf_path = Path(__file__).parent / "midlands.dxf"
    output_dir = Path(__file__).parent / "analysis_output"

    if not dxf_path.exists():
        print(f"\n[ERROR] DXF file not found: {dxf_path}")
        sys.exit(1)

    print(f"\nInput: {dxf_path.name}")
    print(f"Output: {output_dir}")

    # Run analysis
    print("\n" + "=" * 80)
    print("RUNNING ANALYSIS")
    print("=" * 80)

    pipeline = CableAnalysisPipeline(config)

    try:
        results = pipeline.analyze_dxf(
            dxf_path=dxf_path,
            output_dir=output_dir,
        )

        # Display results summary
        print("\n" + "=" * 80)
        print("ANALYSIS RESULTS")
        print("=" * 80)

        print(f"\nRoute: {results.route_name}")
        print(f"Total length: {results.total_length_m:.1f} m")
        print(f"Optimized sections: {results.section_count}")

        print(f"\nGeometry:")
        print(f"  Total straights: {results.total_straights}")
        print(f"  Total bends: {results.total_bends}")

        print(f"\nFinal Forces:")
        print(f"  Final forward tension: {results.final_forward_tension_n/1000:.2f} kN")
        print(f"  Final reverse tension: {results.final_reverse_tension_n/1000:.2f} kN")
        print(f"  Max sidewall pressure: {results.max_sidewall_pressure_n_m/1000:.2f} kN/m")

        # Check if all sections pass
        all_pass = all(
            section.forward_tension_n <= config.cable_max_tension_n and
            section.max_sidewall_pressure_n_m <= config.cable_max_sidewall_pressure_n_m
            for section in results.sections
        )

        status = "PASS" if all_pass else "FAIL"
        print(f"\nStatus: {status}")
        if all_pass:
            print("  All sections pass tension and sidewall limits!")
        else:
            failing = [
                s.section_id for s in results.sections
                if s.forward_tension_n > config.cable_max_tension_n or
                   s.max_sidewall_pressure_n_m > config.cable_max_sidewall_pressure_n_m
            ]
            print(f"  {len(failing)} sections exceed limits: {', '.join(failing[:5])}")

        print("\n" + "=" * 80)
        print("OUTPUT FILES")
        print("=" * 80)
        print(f"\nResults saved to: {output_dir}")
        print(f"  - CSV reports: {output_dir}/csv/")
        print(f"  - JSON data: {output_dir}/json/")
        print(f"  - Visualizations: {output_dir}/visualizations/")
        print(f"  - Summary: {output_dir}/analysis_summary.json")

        print("\n" + "=" * 80)
        print("SUCCESS!")
        print("=" * 80)

        return 0

    except Exception as e:
        print(f"\n[ERROR] Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
