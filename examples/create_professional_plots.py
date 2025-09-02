#!/usr/bin/env python3
"""Example script demonstrating professional visualization capabilities."""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from easycablepulling.core.pipeline import CablePullingPipeline
from easycablepulling.reporting import generate_csv_report, generate_json_report
from easycablepulling.visualization.professional_matplotlib import (
    ProfessionalMatplotlibPlotter,
    create_professional_route_plot_matplotlib,
)


def main():
    """Demonstrate professional visualization and reporting."""
    # Input and output paths
    input_dxf = Path(__file__).parent / "input.dxf"
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    if not input_dxf.exists():
        print(f"Error: Input file {input_dxf} not found")
        return

    print("🔄 Running cable pulling analysis...")

    # Run geometry processing pipeline
    pipeline = CablePullingPipeline()
    geometry_result = pipeline.run_geometry_only(str(input_dxf))

    if not geometry_result.success:
        print(f"Error: Geometry processing failed - {geometry_result.message}")
        return

    route = geometry_result.route

    # Mock analysis results for demonstration
    # In a real scenario, these would come from actual calculations
    analysis_results = {
        "max_tension": 12500,
        "max_pressure": 850.5,
        "overall_status": "PASS",
        "limiting_factor": "Sidewall Pressure at Bend #3",
        "recommended_direction": "Forward",
        "sections": {
            section.id: {
                "max_tension": 8000 + i * 1000,
                "max_pressure": 600 + i * 50,
                "lateral_deviation": 0.05 + i * 0.01,
                "length_error_percent": 0.8 + i * 0.2,
                "status": "PASS" if i < 3 else "WARNING",
            }
            for i, section in enumerate(route.sections)
        },
        "calculations": {
            "tension_profile": [5000 + i * 200 for i in range(100)],
            "chainages": [i * 10 for i in range(100)],
            "pressure_points": [400 + i * 10 for i in range(20)],
            "max_allowable_tension": 15000,
            "max_allowable_pressure": 1000,
        },
    }

    print("📊 Creating professional visualizations...")

    # Create professional plotter
    plotter = ProfessionalMatplotlibPlotter()

    # 1. Professional route overview (PNG)
    print("  ↳ Route overview plot...")
    route_fig, route_ax = create_professional_route_plot_matplotlib(
        route,
        title="Professional Cable Route Analysis",
        units="m",
        show_section_colors=True,
        show_annotations=True,
    )
    plotter.save_professional_plot(
        route_fig, output_dir / "professional_route_overview.png", dpi=300
    )

    # 2. Analysis dashboard
    print("  ↳ Analysis dashboard...")
    dashboard_fig, dashboard_axes = plotter.create_analysis_dashboard(
        route, analysis_results, title="Cable Pulling Analysis Dashboard"
    )
    plotter.save_professional_plot(
        dashboard_fig, output_dir / "analysis_dashboard.png", dpi=300
    )

    # 3. Tension analysis plot
    print("  ↳ Tension analysis plot...")
    tension_fig, tension_ax = plotter.plot_tension_analysis(
        route,
        {
            "chainage": analysis_results["calculations"]["chainages"],
            "tension": analysis_results["calculations"]["tension_profile"],
            "max_allowable": analysis_results["calculations"]["max_allowable_tension"],
        },
    )
    plotter.save_professional_plot(
        tension_fig, output_dir / "tension_analysis_professional.png", dpi=300
    )

    # 4. Pressure analysis plot
    print("  ↳ Pressure analysis plot...")
    pressure_fig, pressure_ax = plotter.plot_pressure_analysis(
        route,
        {
            "chainage": [
                i * 50
                for i in range(len(analysis_results["calculations"]["pressure_points"]))
            ],
            "pressure": analysis_results["calculations"]["pressure_points"],
            "max_allowable": analysis_results["calculations"]["max_allowable_pressure"],
        },
    )
    plotter.save_professional_plot(
        pressure_fig, output_dir / "pressure_analysis_professional.png", dpi=300
    )

    print("📋 Generating professional reports...")

    # 6. CSV reports
    print("  ↳ CSV reports...")
    generate_csv_report(
        route, analysis_results, output_dir / "route_summary.csv", report_type="summary"
    )

    generate_csv_report(
        route,
        analysis_results,
        output_dir / "section_details.csv",
        report_type="sections",
    )

    # 7. JSON reports
    print("  ↳ JSON reports...")
    generate_json_report(
        route,
        analysis_results,
        output_dir / "comprehensive_analysis.json",
        report_format="comprehensive",
    )

    generate_json_report(
        route,
        analysis_results,
        output_dir / "machine_readable_results.json",
        report_format="machine_readable",
    )

    print("✅ Professional visualization and reporting complete!")
    print(f"📁 Output files saved to: {output_dir}")
    print("\nGenerated files:")
    for file_path in sorted(output_dir.glob("*")):
        if file_path.is_file() and file_path.name.startswith(
            (
                "professional_",
                "analysis_",
                "tension_",
                "pressure_",
                "route_",
                "section_",
                "comprehensive_",
                "machine_",
            )
        ):
            print(f"  • {file_path.name}")


if __name__ == "__main__":
    main()
