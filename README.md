# EasyCablePulling

A professional cable pulling analysis library for electrical engineering applications. This library provides complete workflow automation for analyzing cable routes from DXF files, calculating pulling forces, and generating comprehensive reports.

## Features

- **DXF Processing**: Import and process DXF polylines with automatic cleaning
- **Geometric Filleting**: Apply standard radius bends (3.9m for 200mm duct) with geometrically correct bend center calculation
- **Route Splitting**: Automatically split long sections (>1000m) for practical pulling analysis
- **Realistic Tension Calculations**: Calculate forward/reverse pulling tensions with symmetrical physics and direction-dependent sidewall pressures
- **Professional Visualization**: Generate PNG visualizations showing original vs fitted geometry
- **Multi-format Export**: Export results as JSON, CSV, Excel, and optionally fitted DXF

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from easycablepulling import analyze_cable_route

# Analyze a DXF file with default settings
results = analyze_cable_route("path/to/route.dxf", output_dir="analysis_output")

print(f"Total length: {results.total_length_m:.1f}m")
print(f"Final forward tension: {results.final_forward_tension_n:.0f}N")
print(f"Max sidewall pressure: {results.max_sidewall_pressure_n_m:.0f}N/m")
```

## Complete Workflow

The analysis follows an 11-step workflow:

### 1. Digest DXF
Load DXF file and extract polylines representing cable routes.

### 2. Remove Duplicate Points (Tidy)
Clean polylines by removing duplicate vertices within 0.001m tolerance.

### 3. Fillet All Changes of Direction
Apply standard radius bends at every junction using geometrically correct parallel guide intersection method:
- Uses duct type radius (3.9m for 200mm duct)
- Calculates bend centers by intersecting parallel guides on acute side of turn
- Maintains theoretical maximum deviation compliance

### 4. Split Long Sections
If any section exceeds maximum length (default 1000m), split equally into subsections for practical pulling analysis.

### 5. Generate PNG Visualizations
Create professional visualizations showing:
- Original polylines
- Fitted geometry with bend centers
- Route overview with section details

### 6. Apply Pulling Calculations
Calculate realistic tension and pressure values:
- Forward and reverse pulling tensions using symmetrical physics
- Direction-dependent sidewall pressures (P = T/R) for each pulling direction
- Cumulative forces across sections with proper physics modeling

### 7. Export Section Reports
Generate detailed reports per section with:
- Straights: length data with cumulative tensions
- Bends: angle and radius data with forward/reverse sidewall pressures
- Cumulative pulling forces in both directions
- Direction-dependent sidewall pressure analysis

### 8. Export Summary Report
Create overall analysis summary with:
- Final forward/reverse tensions
- Maximum sidewall pressures
- Geometry statistics
- Accuracy metrics

### 9. Multi-format Results
Export in multiple formats:
- **JSON**: For SaaS integration and programmatic access
- **CSV**: For spreadsheet analysis
- **Excel**: Optional comprehensive reports

### 10. Optional DXF Export
Export fitted geometry as DXF with proper straights and arcs for CAD integration.

### 11. Configurable Pipeline
Run complete workflow with switches for different output formats and settings.

## Configuration

```python
from easycablepulling import CableAnalysisPipeline, AnalysisConfig

# Custom configuration
config = AnalysisConfig(
    duct_type="200mm",
    max_section_length_m=1000.0,
    cable_diameter_mm=50.0,
    cable_weight_kg_m=1.5,
    cable_max_tension_n=15000.0,
    sample_interval_m=25.0,
    generate_json=True,
    generate_csv=True,
    generate_png=True,
    generate_dxf=False
)

pipeline = CableAnalysisPipeline(config)
results = pipeline.analyze_dxf("route.dxf", "output")
```

## API Reference

### Main Functions

#### `analyze_cable_route(dxf_path, output_dir="output", config=None, **kwargs)`
Convenience function for complete analysis workflow.

**Parameters:**
- `dxf_path`: Path to DXF file
- `output_dir`: Output directory for results
- `config`: AnalysisConfig instance
- `**kwargs`: Configuration overrides

**Returns:** `AnalysisResults` with complete analysis data

### Core Classes

#### `CableAnalysisPipeline`
Main pipeline orchestrating the complete workflow.

#### `AnalysisConfig`
Configuration dataclass with all analysis parameters.

#### `AnalysisResults`
Results dataclass containing:
- Route statistics
- Section-by-section results
- Tension and pressure calculations
- Accuracy metrics

## Technical Details

### Geometric Accuracy
- Uses parallel guide intersection method for bend center calculation
- Achieves 97%+ excellent accuracy classification
- Theoretical maximum deviation compliance for 90° corners
- All straights maintain 0mm deviation from original lines

### Calculations
- **Straight sections**: T_out = T_in + W × f × L
- **Bends**: T_out = T_in × e^(f × θ) (capstan equation)
- **Sidewall pressure**: P = T/R (direction-dependent, uses actual tension at each bend)
- **Reverse calculations**: Use same methodology as forward with flipped geometry for realistic results

### File Structure
```
output/
├── visualizations/
│   ├── route_overview.png
│   └── sections/
│       └── section_*.png
├── json/
│   └── section_*.json
├── csv/
│   ├── sections_summary.csv
│   └── section_*.csv (with forward/reverse sidewall pressures)
├── analysis_summary.json
├── analysis_summary.csv
└── fitted_route.dxf (optional)
```

## Requirements

- Python 3.8+
- ezdxf
- matplotlib
- numpy
- pathlib
- dataclasses

## License

MIT License
