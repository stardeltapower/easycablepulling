"""Command-line interface for easycablepulling."""

from pathlib import Path
from typing import Optional

import click

from .config import DEFAULT_INPUT_DXF
from .core.models import CableArrangement, CableSpec, DuctSpec
from .core.pipeline import AnalysisReporter, CablePullingPipeline
from .geometry import GeometryProcessor
from .io import load_route_from_dxf


@click.group()
@click.version_option()
def main() -> None:
    """Easy Cable Pulling - Cable pulling calculations and route analysis."""
    pass


@main.command()
@click.argument(
    "dxf_file",
    type=click.Path(exists=True),
    default=str(DEFAULT_INPUT_DXF),
    required=False,
)
@click.option("--cable-diameter", type=float, default=35.0, help="Cable diameter (mm)")
@click.option("--cable-weight", type=float, default=2.5, help="Cable weight (kg/m)")
@click.option("--max-tension", type=float, default=8000.0, help="Max tension (N)")
@click.option(
    "--max-pressure", type=float, default=500.0, help="Max sidewall pressure (N/m)"
)
@click.option(
    "--min-bend-radius", type=float, default=1200.0, help="Min bend radius (mm)"
)
@click.option(
    "--arrangement",
    type=click.Choice(["single", "trefoil", "flat"]),
    default="single",
    help="Cable arrangement",
)
@click.option("--num-cables", type=int, default=1, help="Number of cables")
@click.option(
    "--duct-diameter", type=float, default=100.0, help="Duct inner diameter (mm)"
)
@click.option(
    "--duct-type",
    type=click.Choice(["PVC", "HDPE", "Steel", "Concrete"]),
    default="PVC",
    help="Duct material",
)
@click.option("--lubricated", is_flag=True, help="Use lubricated friction values")
@click.option(
    "--max-length", type=float, default=500.0, help="Maximum cable length for splitting"
)
@click.option(
    "--safety-factor", type=float, default=1.5, help="Safety factor for limits"
)
@click.option("--output", "-o", type=click.Path(), help="Output directory")
@click.option(
    "--format",
    type=click.Choice(["text", "csv", "json"]),
    default="text",
    help="Output format",
)
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def analyze(
    dxf_file: str,
    cable_diameter: float,
    cable_weight: float,
    max_tension: float,
    max_pressure: float,
    min_bend_radius: float,
    arrangement: str,
    num_cables: int,
    duct_diameter: float,
    duct_type: str,
    lubricated: bool,
    max_length: float,
    safety_factor: float,
    output: Optional[str],
    format: str,
    verbose: bool,
) -> None:
    """Analyze cable pulling route from DXF file."""
    try:
        if verbose:
            click.echo(f"Analyzing route from: {dxf_file}")
            click.echo(f"Cable: {cable_diameter}mm {arrangement} ({num_cables} cables)")
            click.echo(f"Duct: {duct_diameter}mm {duct_type}")
            click.echo(f"Lubricated: {lubricated}")
            click.echo(f"Max length: {max_length}m")
            click.echo(f"Safety factor: {safety_factor}")
            click.echo("")

        # Create specifications
        cable_spec = CableSpec(
            diameter=cable_diameter,
            weight_per_meter=cable_weight,
            max_tension=max_tension,
            max_sidewall_pressure=max_pressure,
            min_bend_radius=min_bend_radius,
            arrangement=CableArrangement(arrangement),
            number_of_cables=num_cables,
        )

        duct_spec = DuctSpec(
            inner_diameter=duct_diameter,
            type=duct_type,  # type: ignore  # CLI validates choices
            friction_dry=0.35,  # Default values, should be from config
            friction_lubricated=0.15,
        )

        # Initialize and run pipeline
        pipeline = CablePullingPipeline(
            enable_splitting=True,
            max_cable_length=max_length,
            safety_factor=safety_factor,
        )

        result = pipeline.run_analysis(dxf_file, cable_spec, duct_spec, lubricated)

        # Generate report
        if format == "text":
            report = AnalysisReporter.generate_text_report(result)
            click.echo(report)
        elif format == "csv":
            report = AnalysisReporter.generate_csv_report(result)
            click.echo(report)
        elif format == "json":
            import json

            json_report = AnalysisReporter.generate_json_summary(result)
            click.echo(json.dumps(json_report, indent=2))

        # Save to file if output specified
        if output:
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w") as f:
                if format == "json":
                    import json

                    json.dump(
                        AnalysisReporter.generate_json_summary(result), f, indent=2
                    )
                elif format == "csv":
                    f.write(AnalysisReporter.generate_csv_report(result))
                else:
                    f.write(AnalysisReporter.generate_text_report(result))

            click.echo(f"\n✓ Report saved to: {output_path}")

        # Exit with error code if not feasible
        if not result.success:
            raise click.ClickException(
                "Cable pulling not feasible with current parameters"
            )

    except Exception as e:
        click.echo(f"✗ Analysis failed: {e}")
        raise click.ClickException("Failed to analyze route")


@main.command()
@click.argument("dxf_file", type=click.Path(exists=True))
@click.option("--max-length", type=float, default=500.0, help="Maximum cable length")
@click.option("--output", "-o", type=click.Path(), help="Output DXF file")
def split(dxf_file: str, max_length: float, output: Optional[str]) -> None:
    """Split route sections that exceed maximum cable length."""
    try:
        click.echo(f"Splitting sections in: {dxf_file}")
        click.echo(f"Maximum length: {max_length}m")

        # Read DXF file
        route = load_route_from_dxf(Path(dxf_file))
        click.echo(f"Loaded route with {route.section_count} sections")

        # Initialize processor with splitting
        processor = GeometryProcessor()

        # Perform splitting
        splitting_result = processor.split_route(route, max_cable_length=max_length)

        if splitting_result.success:
            click.echo(f"✓ {splitting_result.message}")
            click.echo(f"  Original sections: {len(route.sections)}")
            click.echo(
                f"  Final sections: {len(splitting_result.split_route.sections)}"
            )
            click.echo(f"  Split points: {len(splitting_result.split_points)}")

            # Show sections that were split
            for split_point in splitting_result.split_points:
                click.echo(
                    f"    Split at {split_point.position:.1f}m ({split_point.reason})"
                )

            # Show detailed results
            for section in splitting_result.split_route.sections:
                length = section.original_length
                marker = (
                    " (split)"
                    if "_" in section.id and section.id.split("_")[-1].isdigit()
                    else ""
                )
                click.echo(f"    {section.id}: {length:.1f}m{marker}")

            # Write output if specified (TODO: Fix DXF writer layer issue)
            if output or True:  # Always skip for now due to DXF writer issue
                click.echo(
                    "  Note: DXF output temporarily disabled due to layer validation issue"
                )
                # Placeholder for when DXF writer is fixed:
                # writer = DXFWriter()
                # writer.write_original_route(splitting_result.split_route, output_path)
                # click.echo(f"✓ Split route saved to: {output_path}")
        else:
            click.echo(f"✗ Splitting failed: {splitting_result.message}")
            raise click.ClickException("Splitting operation failed")

    except Exception as e:
        click.echo(f"✗ Error: {e}")
        raise click.ClickException("Failed to split route")


@main.command()
@click.argument("dxf_file", type=click.Path(exists=True))
@click.option(
    "--duct-diameter",
    type=float,
    default=100.0,
    help="Duct diameter for bend classification (mm)",
)
@click.option("--tolerance", type=float, default=0.1, help="Geometry tolerance (m)")
@click.option(
    "--max-length", type=float, default=500.0, help="Max cable length for splitting"
)
@click.option("--output", "-o", type=click.Path(), help="Output DXF file")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def interpret(
    dxf_file: str,
    duct_diameter: float,
    tolerance: float,
    max_length: float,
    output: Optional[str],
    verbose: bool,
) -> None:
    """Interpret DXF geometry as straights and bends."""
    try:
        if verbose:
            click.echo(f"Interpreting geometry from: {dxf_file}")
            click.echo(f"Duct diameter: {duct_diameter}mm")
            click.echo(f"Tolerance: {tolerance}m")
            click.echo(f"Max length: {max_length}m")
            click.echo("")

        # Create duct spec for bend classification
        duct_spec = DuctSpec(
            inner_diameter=duct_diameter,
            type="PVC",  # Default
            friction_dry=0.35,
            friction_lubricated=0.15,
        )

        # Initialize pipeline for geometry-only processing
        pipeline = CablePullingPipeline(
            enable_splitting=True,
            max_cable_length=max_length,
        )

        # Run geometry processing only
        result = pipeline.run_geometry_only(dxf_file, duct_spec)

        if result.success:
            click.echo(f"✓ {result.message}")

            # Show geometry summary
            total_primitives = sum(
                len(section.primitives) for section in result.route.sections
            )
            click.echo(f"  Route: {result.route.name}")
            click.echo(f"  Sections: {result.route.section_count}")
            click.echo(f"  Total length: {result.route.total_length:.1f}m")
            click.echo(f"  Primitives fitted: {total_primitives}")

            if result.splitting_result:
                click.echo(
                    f"  Splitting: {result.splitting_result.sections_created} sections added"
                )

            # Show primitive breakdown
            if verbose:
                straight_count = bend_count = curve_count = 0
                for section in result.route.sections:
                    for primitive in section.primitives:
                        if hasattr(primitive, "length_m"):  # Straight
                            straight_count += 1
                        elif hasattr(primitive, "radius_m"):  # Bend
                            bend_count += 1
                        else:  # Curve
                            curve_count += 1

                click.echo(f"    - Straights: {straight_count}")
                click.echo(f"    - Bends: {bend_count}")
                click.echo(f"    - Curves: {curve_count}")

            # TODO: Add DXF output when writer is fixed
            if output:
                click.echo(
                    "  Note: DXF output temporarily disabled due to layer validation issue"
                )

        else:
            click.echo(f"✗ Geometry processing failed: {result.message}")
            raise click.ClickException("Geometry interpretation failed")

    except Exception as e:
        click.echo(f"✗ Error: {e}")
        raise click.ClickException("Failed to interpret geometry")


@main.command()
@click.argument("dxf_file", type=click.Path(exists=True))
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def import_dxf(dxf_file: str, verbose: bool) -> None:
    """Import and validate DXF file."""
    try:
        if verbose:
            click.echo(f"Importing DXF file: {dxf_file}")

        # Load route
        route = load_route_from_dxf(Path(dxf_file))

        click.echo(f"✓ Successfully imported: {route.name}")
        click.echo(f"  Sections: {route.section_count}")
        click.echo(
            f"  Total original length: {sum(s.original_length for s in route.sections):.1f}m"
        )

        if verbose:
            click.echo("\n  Section details:")
            for section in route.sections:
                click.echo(
                    f"    {section.id}: {section.original_length:.1f}m ({len(section.original_polyline)} points)"
                )

    except Exception as e:
        click.echo(f"✗ Import failed: {e}")
        raise click.ClickException("Failed to import DXF file")


@main.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option(
    "--format",
    type=click.Choice(["csv", "json", "dxf"]),
    default="csv",
    help="Export format",
)
@click.option("--output", "-o", type=click.Path(), help="Output file")
def export(input_file: str, format: str, output: Optional[str]) -> None:
    """Export analysis results to various formats."""
    try:
        # This would typically load analysis results from a previous run
        # For now, just show the concept

        if not output:
            input_path = Path(input_file)
            output = f"{input_path.stem}_export.{format}"

        click.echo(f"Exporting {format.upper()} from: {input_file}")
        click.echo(f"Output: {output}")

        # TODO: Implement actual export functionality
        # This would load previously computed analysis results and export them
        click.echo("✓ Export functionality ready for implementation")

    except Exception as e:
        click.echo(f"✗ Export failed: {e}")
        raise click.ClickException("Failed to export results")


if __name__ == "__main__":
    main()
