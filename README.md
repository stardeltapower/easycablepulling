# Easy Cable Pulling

Cable pulling calculations and route analysis software for power cable installation.

## Overview

Easy Cable Pulling converts CAD geometry into realistic cable pulling routes and performs tension and sidewall pressure calculations according to IEEE 525 standards.

## Features

- Import DXF files containing cable routes
- Convert polylines to realistic straight and bend segments
- Calculate pulling tensions and sidewall pressures
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

```python
from easycablepulling import analyze_route

# Analyze a cable route
results = analyze_route(
    dxf_file="route.dxf",
    cable_spec="cable_specs.json",
    duct_spec="duct_specs.json"
)

# View results
print(results.summary())
```

## Documentation

Full documentation is available at [docs/](docs/).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute to this project.

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.