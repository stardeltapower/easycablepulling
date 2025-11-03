#!/usr/bin/env python3
"""Debug script to trace section splitting logic."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from easycablepulling.core.models import CableSpec, DuctSpec, CableArrangement, PullingMethod
from easycablepulling.analysis import optimize_cable_route, PullingDirection
from easycablepulling.io.dxf_reader import DXFReader
from easycablepulling.geometry.simple_segment_fitter import SimpleSegmentFitter

# Cable specifications
CABLE_DIAMETER_MM = 69.0
CABLE_WEIGHT_KG_KM = 4930
CABLE_WEIGHT_KG_M = CABLE_WEIGHT_KG_KM / 1000  # 4.93 kg/m per cable

# Cable arrangement
NUMBER_OF_CABLES = 3

# Installation limits
MAX_PULL_TENSION_N = 27000         # 27 kN maximum pulling force
MAX_SIDEWALL_PRESSURE_N_M = 3000   # Typical for MV cables
MIN_BEND_RADIUS_MM = 15 * CABLE_DIAMETER_MM  # 15 × D = 900mm

# Duct specifications
DUCT_INNER_DIAMETER_MM = 225
FRICTION_COEFFICIENT = 0.3

# Optimization parameters
TARGET_UTILIZATION = 0.8
MAX_SECTION_LENGTH_M = 500.0


def load_and_process_route(dxf_path: str):
    """Load DXF and process route with geometry fitting."""
    reader = DXFReader(dxf_path)
    reader.load()
    route = reader.create_route_from_polylines(
        Path(dxf_path).stem,
        layer_name="_FUN_33kV OPT 2 Overview Route"
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
        friction_lubricated=FRICTION_COEFFICIENT * 0.6,
    )


def debug_optimization():
    """Debug the optimization process."""

    print("=" * 100)
    print("DEBUGGING SECTION SPLITTING")
    print("=" * 100)

    # Load route
    route = load_and_process_route("midlands/midlands.dxf")
    cable_spec = create_cable_spec()
    duct_spec = create_duct_spec()

    print(f"\nRoute loaded: {len(route.sections)} sections")
    for i, section in enumerate(route.sections):
        print(f"  Section {i}: {len(section.primitives)} primitives, length={section.original_length:.1f}m")

    # Manually create optimizer to debug
    from easycablepulling.analysis.route_optimizer import RouteOptimizer

    optimizer = RouteOptimizer(
        cable_spec=cable_spec,
        duct_spec=duct_spec,
        target_utilization=TARGET_UTILIZATION,
        max_section_length=MAX_SECTION_LENGTH_M,
    )

    print(f"\nOptimizer limits:")
    print(f"  Tension limit (80% of 27000N): {optimizer.tension_limit:.0f}N")
    print(f"  Sidewall limit (80% of 3000 N/m): {optimizer.sidewall_limit:.0f}N/m")
    print(f"  Max section length: {optimizer.max_section_length:.0f}m")

    # Get primitives
    primitives = optimizer._extract_primitives(route, PullingDirection.FORWARD)
    print(f"\nTotal primitives (forward): {len(primitives)}")

    # Calculate primitive results
    primitive_results = optimizer._calculate_primitive_results(
        primitives, friction_override=FRICTION_COEFFICIENT
    )
    print(f"Primitive results calculated: {len(primitive_results)}")

    # Analyze primitive results
    print(f"\nFirst 20 primitives:")
    print(f"{'Index':<6} {'Type':<8} {'Pos(m)':<8} {'T_in(N)':<12} {'T_out(N)':<12} {'SW(N/m)':<10} {'Passes':<7}")
    print("-" * 70)

    for i, result in enumerate(primitive_results[:20]):
        prim_type = "Bend" if hasattr(result.primitive, 'angle_deg') else "Straight"
        print(f"{i:<6} {prim_type:<8} {result.position:<8.1f} {result.tension_in:<12.0f} "
              f"{result.tension_out:<12.0f} {result.sidewall_pressure:<10.0f} {str(result.passes_limits):<7}")

    # Find first failure
    print(f"\n--- Looking for first limit violation ---")
    for i, result in enumerate(primitive_results):
        if not result.passes_limits:
            print(f"\nFirst failure at primitive {i}:")
            print(f"  Position: {result.position:.1f}m")
            print(f"  Type: {'Bend' if hasattr(result.primitive, 'angle_deg') else 'Straight'}")
            print(f"  Tension out: {result.tension_out:.0f}N (limit: {optimizer.tension_limit:.0f}N)")
            print(f"  Sidewall: {result.sidewall_pressure:.0f}N/m (limit: {optimizer.sidewall_limit:.0f}N/m)")
            if hasattr(result.primitive, 'angle_deg'):
                print(f"  Bend angle: {result.primitive.angle_deg:.2f}°")
                print(f"  Bend radius: {result.primitive.radius_m:.2f}m")
            break

    # Find split points
    split_points = optimizer._find_optimal_splits(primitive_results)
    print(f"\n--- Split points found (first 50): {split_points[:50]}")
    print(f"Total split points: {len(split_points)}")
    print(f"Number of sections: {len(split_points) - 1}")

    # Show where splits happen and why
    print(f"\n--- Split point details (first 15 splits) ---")
    section_start_pos = 0.0
    for j, split_idx in enumerate(split_points[:16]):
        if split_idx < len(primitive_results):
            result = primitive_results[split_idx]
            prim_type = "Bend" if hasattr(result.primitive, 'angle_deg') else "Straight"
            section_len = result.position - section_start_pos
            print(f"  Split {j}: idx={split_idx}, pos={result.position:.1f}m, len_from_start={section_len:.1f}m, "
                  f"type={prim_type}, T={result.tension_out:.0f}N (limit={optimizer.tension_limit:.0f}N), "
                  f"SW={result.sidewall_pressure:.0f}N/m (limit={optimizer.sidewall_limit:.0f}N/m)")
            if j < 15:  # Update for next iteration
                section_start_pos = primitive_results[split_idx - 1].position if split_idx > 0 else 0.0

    # Analyze split points
    print(f"\nSplit analysis:")
    for i in range(len(split_points) - 1):
        start_idx = split_points[i]
        end_idx = split_points[i + 1]
        section_prims = primitive_results[start_idx:end_idx]

        if section_prims:
            start_pos = section_prims[0].tension_in if start_idx == 0 else 0.0  # WRONG
            end_pos = section_prims[-1].position
            length = end_pos - start_pos if start_idx > 0 else end_pos
            max_tension = max(p.tension_out for p in section_prims)

            print(f"  Section {i+1}: primitives [{start_idx}:{end_idx}] "
                  f"pos={end_pos:.1f}m, T_max={max_tension:.0f}N, passes={max_tension <= optimizer.tension_limit}")


if __name__ == "__main__":
    try:
        debug_optimization()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
