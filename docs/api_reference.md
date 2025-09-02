# API Reference

## Core Models

### CableSpec

Cable specification for pulling analysis.

```python
from easycablepulling.core.models import CableSpec, CableArrangement, PullingMethod

cable_spec = CableSpec(
    diameter=35.0,                    # Cable diameter in mm
    weight_per_meter=2.5,            # Weight per meter in kg/m
    max_tension=8000.0,              # Maximum pulling tension in N
    max_sidewall_pressure=500.0,     # Maximum sidewall pressure in N/m
    min_bend_radius=1200.0,          # Minimum bend radius in mm
    pulling_method=PullingMethod.EYE, # Pulling method (EYE, SOCK, KELLEMS)
    arrangement=CableArrangement.SINGLE, # Cable arrangement
    number_of_cables=1               # Number of cables in arrangement
)
```

**Attributes:**
- `diameter`: Cable outer diameter in millimeters
- `weight_per_meter`: Cable weight per meter in kg/m
- `max_tension`: Maximum allowable pulling tension in Newtons
- `max_sidewall_pressure`: Maximum allowable sidewall pressure in N/m
- `min_bend_radius`: Minimum allowable bend radius in millimeters
- `pulling_method`: Pulling method (eye, sock, kellems grip)
- `arrangement`: Cable arrangement (single, trefoil, flat)
- `number_of_cables`: Number of cables in the arrangement

### DuctSpec

Duct specification for friction calculations.

```python
from easycablepulling.core.models import DuctSpec, BendOption

duct_spec = DuctSpec(
    inner_diameter=100.0,            # Inner diameter in mm
    type="PVC",                      # Duct material type
    friction_dry=0.35,               # Dry friction coefficient
    friction_lubricated=0.15,        # Lubricated friction coefficient
    bend_options=[                   # Available bend radii
        BendOption(radius=1000.0, angle=90.0),
        BendOption(radius=1500.0, angle=45.0),
    ]
)
```

**Attributes:**
- `inner_diameter`: Duct inner diameter in millimeters
- `type`: Duct material type (PVC, HDPE, etc.)
- `friction_dry`: Friction coefficient for dry conditions
- `friction_lubricated`: Friction coefficient for lubricated conditions
- `bend_options`: Available manufactured bend options

### Route

Complete cable route with sections and metadata.

```python
from easycablepulling.core.models import Route, Section

# Routes are typically loaded from DXF files
route = Route(
    name="route_name",
    sections=[section1, section2],
    metadata={"source_file": "input.dxf"}
)
```

**Properties:**
- `total_length`: Total route length in meters
- `section_count`: Number of sections in the route

### Section

Individual section of a cable route between joints.

```python
from easycablepulling.core.models import Section, Straight, Bend

section = Section(
    id="SECT_01",
    original_polyline=[(0.0, 0.0), (100.0, 0.0)],
    primitives=[
        Straight(length_m=100.0, start_point=(0.0, 0.0), end_point=(100.0, 0.0))
    ]
)
```

**Properties:**
- `total_length`: Total length from fitted primitives
- `original_length`: Length from original polyline

## Pipeline

### CablePullingPipeline

Main analysis pipeline for cable pulling calculations.

```python
from easycablepulling.core.pipeline import CablePullingPipeline

pipeline = CablePullingPipeline(
    enable_splitting=True,           # Enable route splitting
    max_cable_length=500.0,         # Maximum cable length in meters
    safety_factor=1.5,              # Safety factor for limits
    geometric_tolerance=1.0,        # Geometric fitting tolerance in mm
    angle_tolerance=2.0             # Angle tolerance in degrees
)
```

#### Methods

##### run_analysis(dxf_file, cable_spec, duct_spec, lubricated=False, **kwargs)

Run complete cable pulling analysis.

**Parameters:**
- `dxf_file`: Path to DXF input file (str)
- `cable_spec`: Cable specifications (CableSpec)
- `duct_spec`: Duct specifications (DuctSpec)
- `lubricated`: Whether duct is lubricated (bool, default False)
- `**kwargs`: Additional analysis parameters

**Returns:**
- `PipelineResult`: Complete analysis results

**Example:**
```python
result = pipeline.run_analysis(
    "input.dxf",
    cable_spec,
    duct_spec,
    lubricated=True
)

if result.success:
    print(f"Analysis successful: {result.summary['feasibility']['overall_feasible']}")
    print(f"Max tension: {result.summary['feasibility']['max_tension_n']:.1f}N")
else:
    print(f"Analysis failed: {result.errors}")
```

##### run_geometry_only(dxf_file, duct_spec, cable_spec)

Run geometry processing only (no cable pulling calculations).

**Parameters:**
- `dxf_file`: Path to DXF input file (str)
- `duct_spec`: Duct specifications (DuctSpec)
- `cable_spec`: Cable specifications (CableSpec)

**Returns:**
- `PipelineResult`: Geometry processing results

### PipelineResult

Complete results from pipeline analysis.

**Key Attributes:**
- `success`: Whether analysis completed successfully (bool)
- `summary`: Summary dictionary with key metrics
- `tension_analyses`: Detailed tension analysis per section
- `limit_results`: Limit check results per section
- `critical_sections`: Sections that exceed limits
- `errors`: List of error messages
- `warnings`: List of warning messages

## I/O Operations

### load_route_from_dxf(file_path)

Load cable route from DXF file.

```python
from easycablepulling.io import load_route_from_dxf
from pathlib import Path

route = load_route_from_dxf(Path("input.dxf"))
print(f"Loaded route: {route.name}")
print(f"Sections: {route.section_count}")
```

### DXFWriter

Write analysis results to DXF format.

```python
from easycablepulling.io.dxf_writer import DXFWriter

writer = DXFWriter()
writer.write_fitted_route(processed_route)
writer.write_analysis_annotations(result)
writer.save("output.dxf")
```

## Calculations

### analyze_route_with_varying_conditions(route, cable_spec, duct_specs, lubricated_sections)

Analyze cable pulling for route with varying conditions per section.

**Parameters:**
- `route`: Route to analyze (Route)
- `cable_spec`: Cable specifications (CableSpec)
- `duct_specs`: Duct specifications per section (List[DuctSpec])
- `lubricated_sections`: Lubrication status per section (List[bool])

**Returns:**
- `List[Tuple[SectionTensionAnalysis, LimitCheckResult, Dict]]`: Analysis results per section

### calculate_pulling_feasibility(route, cable_spec, duct_specs, lubricated_sections, safety_factor=1.5)

Calculate overall pulling feasibility.

**Returns:**
- `Dict`: Feasibility analysis with overall status and recommendations

## Geometry Processing

### GeometryProcessor

Process and fit cable route geometry.

```python
from easycablepulling.geometry import GeometryProcessor

processor = GeometryProcessor(
    tolerance_mm=1.0,               # Fitting tolerance in mm
    min_segment_length=50.0,        # Minimum segment length in mm
    max_iterations=100              # Maximum fitting iterations
)

result = processor.process_route(
    route,
    cable_spec=cable_spec,
    duct_spec=duct_spec,
    enable_splitting=True,
    max_cable_length=500.0
)
```

### ProcessingResult

Results from geometry processing.

**Attributes:**
- `success`: Whether processing succeeded (bool)
- `route`: Processed route with fitted primitives
- `fitting_results`: Detailed fitting results per section
- `validation_result`: Geometry validation results
- `splitting_result`: Route splitting results

## Analysis Results

### SectionTensionAnalysis

Detailed tension analysis for a route section.

**Attributes:**
- `section_id`: Section identifier
- `forward_tensions`: Tension results pulling forward
- `backward_tensions`: Tension results pulling backward
- `max_tension`: Maximum tension in section
- `max_tension_position`: Position of maximum tension

### LimitCheckResult

Limit check results for a route section.

**Attributes:**
- `passes_tension_limit`: Whether section passes tension limits
- `passes_pressure_limit`: Whether section passes pressure limits
- `passes_bend_radius_limit`: Whether section passes bend radius limits
- `limiting_factors`: List of factors that limit pulling
- `recommended_direction`: Recommended pulling direction

## Reporting

### AnalysisReporter

Generate reports from analysis results.

```python
from easycablepulling.core.pipeline import AnalysisReporter

# Generate text report
text_report = AnalysisReporter.generate_text_report(result)
print(text_report)

# Generate CSV report
csv_report = AnalysisReporter.generate_csv_report(result)
with open("analysis.csv", "w") as f:
    f.write(csv_report)

# Generate JSON summary
json_summary = AnalysisReporter.generate_json_summary(result)
```

## Visualization

### Professional Plotting

Create professional-quality plots and visualizations.

```python
from easycablepulling.visualization import create_professional_plot

# Create route overview plot
fig = create_professional_plot(
    result,
    plot_type="route_overview",
    title="Cable Route Analysis",
    format="png",
    dpi=300
)
fig.write_image("route_overview.png")

# Create tension analysis plot
fig = create_professional_plot(
    result,
    plot_type="tension_analysis",
    show_critical_sections=True
)
fig.write_image("tension_analysis.png")
```

### Route Plotting

Basic route visualization using matplotlib.

```python
from easycablepulling.visualization.route_plotter import RoutePlotter

plotter = RoutePlotter()
fig, ax = plotter.plot_route_overlay(
    original_route,
    fitted_route,
    show_primitives=True,
    show_annotations=True
)
fig.savefig("route_overlay.png", dpi=300)
```

## Configuration

### Default Configuration

```python
from easycablepulling.config import (
    DEFAULT_INPUT_DXF,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_CABLE_SPEC,
    DEFAULT_DUCT_SPEC
)

# Use default specifications
cable_spec = DEFAULT_CABLE_SPEC
duct_spec = DEFAULT_DUCT_SPEC
```

## Command Line Interface

### Basic Usage

```bash
# Analyze a route
easycablepulling analyze input.dxf --output results.json

# Geometry processing only
easycablepulling analyze-geometry input.dxf --output fitted.dxf

# Batch processing
easycablepulling batch-analyze input_dir/ --output-dir results/
```

### Common Parameters

- `--cable-diameter`: Cable diameter in mm
- `--cable-weight`: Cable weight per meter in kg/m
- `--max-tension`: Maximum tension limit in N
- `--duct-diameter`: Duct inner diameter in mm
- `--friction-dry`: Dry friction coefficient
- `--friction-lubricated`: Lubricated friction coefficient
- `--safety-factor`: Safety factor to apply
- `--enable-splitting`: Enable route splitting
- `--max-cable-length`: Maximum cable length in meters

## Error Handling

### Common Exceptions

- `ValueError`: Invalid parameter values
- `FileNotFoundError`: DXF file not found
- `DXFError`: DXF file parsing errors
- `GeometryError`: Geometry processing errors

### Error Recovery

The system is designed to handle errors gracefully:

```python
result = pipeline.run_analysis(dxf_file, cable_spec, duct_spec)

if not result.success:
    print("Analysis failed:")
    for error in result.errors:
        print(f"  Error: {error}")

    for warning in result.warnings:
        print(f"  Warning: {warning}")
else:
    # Process successful results
    if not result.summary["feasibility"]["overall_feasible"]:
        print("Route is not feasible:")
        for section_id in result.critical_sections["tension"]:
            print(f"  Section {section_id} exceeds tension limits")
```

## Advanced Usage

### Custom Pipeline Configuration

```python
pipeline = CablePullingPipeline(
    enable_splitting=True,
    max_cable_length=400.0,
    safety_factor=2.0,
    geometric_tolerance=0.5,
    angle_tolerance=1.0,
    fitting_method="polynomial",
    max_fitting_iterations=50
)
```

### Multi-Section Analysis

```python
# Different duct specs per section
duct_specs = [duct_spec_1, duct_spec_2, duct_spec_3]
lubricated = [False, True, False]  # Lubrication per section

route_analysis = analyze_route_with_varying_conditions(
    route, cable_spec, duct_specs, lubricated
)
```

### Custom Reporting

```python
# Custom report generation
def generate_custom_report(result):
    report = []
    report.append(f"Route: {result.processed_route.name}")
    report.append(f"Total Length: {result.summary['total_length_m']:.1f}m")

    for section_analysis in result.tension_analyses:
        report.append(f"Section {section_analysis.section_id}:")
        report.append(f"  Max Tension: {section_analysis.max_tension:.1f}N")

    return "\n".join(report)
```

## Performance Considerations

### Memory Usage

- Routes with >10,000 points may require significant memory
- Use `enable_splitting=True` for long routes to manage memory
- Consider processing large routes in smaller sections

### Execution Time

- Simple routes (straight/single bend): <5 seconds
- Complex routes (multiple bends): 5-30 seconds
- Very long routes (>5km): 30-120 seconds
- Performance scales roughly linearly with route complexity

### Optimization Tips

```python
# For batch processing, reuse pipeline instance
pipeline = CablePullingPipeline()
for dxf_file in dxf_files:
    result = pipeline.run_analysis(dxf_file, cable_spec, duct_spec)
    process_result(result)

# Use geometry-only processing when calculations aren't needed
geometry_result = pipeline.run_geometry_only(dxf_file, duct_spec, cable_spec)

# Adjust tolerances for performance vs accuracy trade-off
fast_pipeline = CablePullingPipeline(
    geometric_tolerance=2.0,         # Looser tolerance = faster
    max_fitting_iterations=20        # Fewer iterations = faster
)
```

## Type Hints

The library is fully typed. Import type hints as needed:

```python
from typing import List, Dict, Optional, Tuple
from easycablepulling.core.models import (
    Route, Section, CableSpec, DuctSpec,
    Primitive, Straight, Bend
)
from easycablepulling.core.pipeline import PipelineResult
```

## Compatibility

- Python 3.8+
- NumPy 1.21+
- Matplotlib 3.5+
- Shapely 2.0+
- ezdxf 1.0+
- Plotly 5.17+ (for professional plotting)

## Version Information

Check version information:

```python
import easycablepulling
print(easycablepulling.__version__)
```
