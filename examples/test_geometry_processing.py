#!/usr/bin/env python3
"""Test geometry processing with the input route."""

import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from easycablepulling.core.models import CableArrangement, CableSpec, PullingMethod
from easycablepulling.geometry import GeometryProcessor
from easycablepulling.io import export_route_to_dxf, load_route_from_dxf
from easycablepulling.visualization import StylePlotter


def main():
    """Test geometry processing pipeline."""
    examples_dir = Path(__file__).parent
    input_dxf = examples_dir / "input.dxf"
    output_dir = examples_dir / "output"

    # Create output directory
    output_dir.mkdir(exist_ok=True)

    print(f"Loading route from {input_dxf}")

    # Load the route
    route = load_route_from_dxf(input_dxf, "33kV Cable Route")

    print(f"Loaded route with {route.section_count} sections")
    print(
        f"Original total length: {sum(s.original_length for s in route.sections):.1f}m"
    )

    # Create cable specification for validation
    cable_spec = CableSpec(
        diameter=50.0,  # 50mm cable
        weight_per_meter=2.5,  # kg/m
        max_tension=50000.0,  # 50kN
        max_sidewall_pressure=10000.0,  # 10kN/m
        min_bend_radius=1500.0,  # 1.5m minimum bend radius
        pulling_method=PullingMethod.EYE,
        arrangement=CableArrangement.SINGLE,
        number_of_cables=1,
    )

    print(
        f"\nProcessing geometry with cable spec: {cable_spec.diameter}mm, min bend radius {cable_spec.min_bend_radius/1000:.1f}m"
    )

    # Process geometry
    processor = GeometryProcessor()
    result = processor.process_route(route, cable_spec)

    print(f"\nGeometry processing complete: {result.message}")
    print(f"Success: {result.success}")

    # Print validation results
    validation = result.validation_result
    print(f"\nValidation: {validation.get_summary()}")

    if validation.issues:
        print("\nValidation Issues:")
        for issue in validation.issues[:10]:  # Show first 10 issues
            print(f"  {issue.severity.upper()}: {issue.section_id} - {issue.message}")
        if len(validation.issues) > 10:
            print(f"  ... and {len(validation.issues) - 10} more issues")

    # Print fitting statistics
    total_primitives = sum(len(fr.primitives) for fr in result.fitting_results)
    total_straights = sum(
        sum(1 for p in fr.primitives if hasattr(p, "length_m"))
        for fr in result.fitting_results
    )
    total_bends = total_primitives - total_straights

    print(f"\nFitting Results:")
    print(f"  Total primitives fitted: {total_primitives}")
    print(f"  Straight segments: {total_straights}")
    print(f"  Bend segments: {total_bends}")

    # Calculate fitted vs original length
    fitted_length = sum(s.total_length for s in result.route.sections)
    original_length = sum(s.original_length for s in route.sections)
    length_diff_percent = abs(fitted_length - original_length) / original_length * 100

    print(f"  Fitted length: {fitted_length:.1f}m")
    print(f"  Original length: {original_length:.1f}m")
    print(f"  Length difference: {length_diff_percent:.2f}%")

    # Export fitted route to DXF
    fitted_dxf_path = output_dir / "fitted_route.dxf"
    print(f"\nExporting fitted route to {fitted_dxf_path}")

    export_route_to_dxf(
        route=result.route,
        file_path=fitted_dxf_path,
        analysis_results={
            "fitted_primitives": total_primitives,
            "straight_segments": total_straights,
            "bend_segments": total_bends,
            "length_error_percent": length_diff_percent,
        },
        warnings=[
            issue.message for issue in validation.issues if issue.severity == "warning"
        ][:5],
        include_annotations=True,
        include_joint_markers=True,
    )

    # Create visualization showing original vs fitted
    print("\nCreating fitted geometry visualization...")
    plotter = StylePlotter(figsize=(11.69, 8.27))  # A4 landscape

    fig, ax = plotter.plot_cable_route(
        result.route,
        title="Cable Route - Original vs Fitted Geometry",
        show_legend=True,
        label_all_joints=True,
        units="m",
    )

    # TODO: Add fitted primitives overlay once visualization supports it

    # Add processing statistics
    stats_text = (
        f"Geometry Processing Results\n"
        f"{'─' * 25}\n"
        f"Primitives: {total_primitives}\n"
        f"  Straights: {total_straights}\n"
        f"  Bends: {total_bends}\n"
        f"Length Error: {length_diff_percent:.2f}%\n"
        f"Validation: {validation.get_summary()}"
    )

    props = dict(
        boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.9, edgecolor="black"
    )
    ax.text(
        0.02,
        0.02,
        stats_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="bottom",
        bbox=props,
    )

    plot_path = output_dir / "geometry_processing_result.png"
    plotter.save_plot(fig, plot_path, dpi=300)
    print(f"Saved geometry processing plot to {plot_path}")

    print(f"\nGeometry processing complete!")
    return result


if __name__ == "__main__":
    main()
