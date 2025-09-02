#!/usr/bin/env python3
"""Test enhanced geometry processing with diameter-based bend classification."""

import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from easycablepulling.core.models import (
    CableArrangement,
    CableSpec,
    DuctSpec,
    PullingMethod,
)
from easycablepulling.geometry.processor import GeometryProcessor
from easycablepulling.io import export_route_to_dxf, load_route_from_dxf
from easycablepulling.visualization import StylePlotter


def main():
    """Test enhanced geometry processing."""
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

    # Create cable and duct specifications
    cable_spec = CableSpec(
        diameter=50.0,  # mm
        weight_per_meter=5.2,  # kg/m
        max_tension=15000.0,  # N
        max_sidewall_pressure=500.0,  # N/m
        min_bend_radius=1500.0,  # 1.5m minimum bend radius
        pulling_method=PullingMethod.EYE,
        arrangement=CableArrangement.SINGLE,
        number_of_cables=1,
    )

    # Test with 200mm HDPE duct (appropriate for this cable route)
    duct_spec = DuctSpec(
        inner_diameter=200.0,  # mm
        type="HDPE",
        friction_dry=0.35,
        friction_lubricated=0.15,
        bend_options=[],
    )

    print(
        f"\nProcessing geometry with duct spec: {duct_spec.inner_diameter}mm {duct_spec.type}"
    )
    print(f"Natural bend threshold: {(duct_spec.inner_diameter/1000) * 22:.1f}m")

    # Process geometry with enhanced fitter
    processor = GeometryProcessor()
    result = processor.process_route(route, cable_spec, duct_spec)

    print(f"\nProcessing Results:")
    print(f"  Success: {result.success}")
    print(f"  Message: {result.message}")

    # Analyze fitted primitives
    total_primitives = 0
    natural_bends = 0
    manufactured_bends = 0
    straights = 0

    for section in result.route.sections:
        for primitive in section.primitives:
            total_primitives += 1
            if hasattr(primitive, "bend_type"):
                if primitive.bend_type == "natural":
                    natural_bends += 1
                else:
                    manufactured_bends += 1
            else:
                straights += 1

    print(f"\nPrimitive Analysis:")
    print(f"  Total primitives: {total_primitives}")
    print(f"  Straight segments: {straights}")
    print(f"  Natural sweeping bends: {natural_bends}")
    print(f"  Manufactured bends: {manufactured_bends}")

    # Show validation results
    print(f"\nValidation Results:")
    print(f"  Valid: {result.validation_result.is_valid}")
    print(f"  Errors: {result.validation_result.total_errors}")
    print(f"  Warnings: {result.validation_result.total_warnings}")

    # Export enhanced fitted route to DXF
    fitted_dxf_path = output_dir / "enhanced_fitted_route.dxf"
    print(f"\nExporting fitted route to {fitted_dxf_path}")

    export_route_to_dxf(
        route=result.route,
        file_path=fitted_dxf_path,
        analysis_results={
            "fitted_primitives": total_primitives,
            "straight_segments": straights,
            "natural_bends": natural_bends,
            "manufactured_bends": manufactured_bends,
            "natural_bend_threshold_m": (duct_spec.inner_diameter / 1000) * 22,
            "fitting_method": "enhanced_diameter_based",
        },
        include_annotations=True,
        include_joint_markers=True,
    )

    # Create detailed analysis comparison
    print("\nPerforming detailed geometry analysis...")

    # Calculate accuracy metrics
    total_original_length = sum(s.original_length for s in route.sections)
    total_fitted_length = sum(s.total_length for s in result.route.sections)
    length_error_percent = (
        abs(total_fitted_length - total_original_length) / total_original_length * 100
    )

    print(f"\nGeometry Accuracy Analysis:")
    print(f"  Original total length: {total_original_length:.2f}m")
    print(f"  Fitted total length: {total_fitted_length:.2f}m")
    print(f"  Length error: {length_error_percent:.3f}%")

    # Analyze individual sections
    print(f"\nSection-by-Section Analysis:")
    for section in result.route.sections:
        if section.primitives:
            section_error = (
                abs(section.total_length - section.original_length)
                / section.original_length
                * 100
            )
            print(
                f"  {section.id}: {section.total_length:.1f}m vs {section.original_length:.1f}m (error: {section_error:.2f}%)"
            )

    # Create high-resolution A0 visualization
    print("\nCreating high-resolution A0 geometry visualization...")
    plotter = StylePlotter(figsize=(16.5, 11.7))  # A2 landscape to avoid memory issues

    fig, ax = plotter.plot_cable_route(
        result.route,
        title="Cable Route - Enhanced Geometry Fitting",
        show_legend=True,
        label_all_joints=True,
        units="m",
    )

    # Add processing statistics
    stats_text = (
        f"Enhanced Geometry Fitting\n"
        f"{'─' * 25}\n"
        f"Duct: {duct_spec.inner_diameter}mm {duct_spec.type}\n"
        f"Natural threshold: {(duct_spec.inner_diameter/1000) * 22:.1f}m\n"
        f"Primitives: {total_primitives}\n"
        f"  Straights: {straights}\n"
        f"  Natural bends: {natural_bends}\n"
        f"  Manufactured: {manufactured_bends}\n"
        f"Valid: {result.validation_result.is_valid}"
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

    plot_path = output_dir / "enhanced_geometry_A0_600dpi.png"
    plotter.save_plot(fig, plot_path, dpi=600)
    print(f"Saved geometry fitting plot to {plot_path}")

    print(f"\nEnhanced geometry processing complete!")

    # Show section details with bend types
    print(f"\nSection Details:")
    for i, section in enumerate(result.route.sections[:5]):  # Show first 5
        if section.primitives:
            bend_info = []
            for p in section.primitives:
                if hasattr(p, "bend_type"):
                    bend_info.append(f"{p.bend_type}({p.radius_m:.1f}m)")
                else:
                    bend_info.append(f"straight({p.length_m:.1f}m)")
            print(f"  {section.id}: {' → '.join(bend_info)}")
        else:
            print(f"  {section.id}: No primitives fitted")


if __name__ == "__main__":
    main()
