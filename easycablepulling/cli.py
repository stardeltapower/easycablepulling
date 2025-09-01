"""Command-line interface for easycablepulling."""

from typing import Optional

import click

from .config import DEFAULT_INPUT_DXF


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
@click.option(
    "--cable-spec", type=click.Path(exists=True), help="Cable specification file"
)
@click.option(
    "--duct-spec", type=click.Path(exists=True), help="Duct specification file"
)
@click.option("--output", "-o", type=click.Path(), help="Output directory")
def analyze(
    dxf_file: str,
    cable_spec: Optional[str],
    duct_spec: Optional[str],
    output: Optional[str],
) -> None:
    """Analyze cable pulling route from DXF file."""
    click.echo(f"Analyzing route from: {dxf_file}")
    # Implementation placeholder
    click.echo("Analysis complete!")


@main.command()
@click.argument("dxf_file", type=click.Path(exists=True))
@click.option("--max-length", type=float, default=500.0, help="Maximum cable length")
def split(dxf_file: str, max_length: float) -> None:
    """Split route sections that exceed maximum cable length."""
    click.echo(f"Splitting sections in: {dxf_file}")
    click.echo(f"Maximum length: {max_length}m")
    # Implementation placeholder


@main.command()
@click.argument("dxf_file", type=click.Path(exists=True))
@click.option("--tolerance", type=float, default=0.1, help="Geometry tolerance (m)")
def interpret(dxf_file: str, tolerance: float) -> None:
    """Interpret DXF geometry as straights and bends."""
    click.echo(f"Interpreting geometry from: {dxf_file}")
    click.echo(f"Tolerance: {tolerance}m")
    # Implementation placeholder


if __name__ == "__main__":
    main()
