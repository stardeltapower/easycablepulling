# Easy Cable Pulling

Cable pulling calculations and route analysis software for power cable installation.

## Overview

Easy Cable Pulling converts CAD geometry into realistic cable pulling routes and performs tension and sidewall pressure calculations according to IEEE 525 standards.

## Features

- Import DXF files containing cable routes
- Convert polylines to realistic straight and bend segments
- Calculate pulling tensions and sidewall pressures
- Support for multiple cable arrangements (single, trefoil, flat)
- Comprehensive validation of cable and duct specifications
- Generate detailed reports and visualizations
- Export adjusted routes back to DXF

## Installation

### For Users

```bash
pip install easycablepulling
```

### For Developers

```bash
# Clone the repository
git clone https://github.com/yourusername/easycablepulling.git
cd easycablepulling

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

## Quick Start

### Import and Analyze DXF Route

```python
from easycablepulling.io import load_route_from_dxf
from easycablepulling.core import CableSpec, CableArrangement
from pathlib import Path

# Load route from DXF file
route = load_route_from_dxf(Path("examples/input.dxf"), "33kV Route")

print(f"Loaded route: {route.name}")
print(f"Sections: {route.section_count}")
print(f"Total length: {sum(s.original_length for s in route.sections):.1f}m")

# Define cable specifications
cable = CableSpec(
    diameter=50.0,                        # mm
    weight_per_meter=2.5,                # kg/m
    max_tension=5000.0,                  # N
    max_sidewall_pressure=3000.0,        # N/m
    min_bend_radius=600.0,               # mm
    arrangement=CableArrangement.TREFOIL,
    number_of_cables=3
)

# Validate route against cable specs
warnings = route.validate_cable(cable)
if warnings:
    print("Validation warnings:")
    for warning in warnings:
        print(f"  - {warning}")
```

### Export Results to DXF

```python
from easycablepulling.io import export_route_to_dxf

# Export with analysis results
analysis_results = {
    "max_tension": 15000.0,
    "max_sidewall_pressure": 3500.0
}

export_route_to_dxf(
    route=route,
    file_path=Path("output/analyzed_route.dxf"),
    analysis_results=analysis_results,
    warnings=warnings
)
```

### Using the CLI

```bash
# Analyze the default input.dxf file from examples/
easycablepulling analyze

# Or specify a different DXF file
easycablepulling analyze path/to/your/route.dxf
```

### Example Files

The project includes a sample DXF file at `examples/input.dxf` which is used as the default input for testing and development.

## Documentation

Full documentation is available at [docs/](docs/).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute to this project.

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.
