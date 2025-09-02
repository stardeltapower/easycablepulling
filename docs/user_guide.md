# User Guide

## Getting Started

EasyCablePulling is a comprehensive tool for analyzing cable pulling feasibility in underground duct systems. This guide will walk you through the basic usage and common workflows.

### Installation

```bash
pip install easycablepulling
```

For development:
```bash
git clone <repository>
cd easycablepulling
pip install -e ".[dev]"
```

### Quick Start

1. Prepare your route in DXF format
2. Define cable and duct specifications
3. Run analysis
4. Review results and recommendations

## Basic Workflow

### 1. Prepare Input Data

Your cable route should be saved as a DXF file with polylines representing the cable path. The route can include:
- Straight sections
- Curved sections (natural bends)
- Sharp bends (manufactured fittings)
- Multiple disconnected sections

### 2. Define Specifications

```python
from easycablepulling.core.models import CableSpec, DuctSpec, CableArrangement

# Define your cable
cable_spec = CableSpec(
    diameter=35.0,                    # mm
    weight_per_meter=2.5,            # kg/m
    max_tension=8000.0,              # N
    max_sidewall_pressure=500.0,     # N/m
    min_bend_radius=1200.0,          # mm
    arrangement=CableArrangement.SINGLE
)

# Define your duct system
duct_spec = DuctSpec(
    inner_diameter=100.0,            # mm
    type="PVC",
    friction_dry=0.35,
    friction_lubricated=0.15
)
```

### 3. Run Analysis

```python
from easycablepulling.core.pipeline import CablePullingPipeline

pipeline = CablePullingPipeline(
    enable_splitting=True,           # Split long routes automatically
    max_cable_length=500.0,         # Maximum continuous cable length
    safety_factor=1.5               # Apply safety factor to limits
)

result = pipeline.run_analysis(
    "my_route.dxf",
    cable_spec,
    duct_spec,
    lubricated=False
)
```

### 4. Review Results

```python
if result.success:
    print(f"Analysis completed successfully!")
    print(f"Route feasible: {result.summary['feasibility']['overall_feasible']}")
    print(f"Total length: {result.summary['total_length_m']:.1f}m")
    print(f"Max tension: {result.summary['feasibility']['max_tension_n']:.1f}N")
    print(f"Max pressure: {result.summary['feasibility']['max_pressure_n_per_m']:.1f}N/m")
else:
    print("Analysis failed:")
    for error in result.errors:
        print(f"  {error}")
```

## Common Use Cases

### Case 1: Single Cable Installation

```python
# Standard single cable installation
cable_spec = CableSpec(
    diameter=25.0,              # 25mm cable
    weight_per_meter=1.8,       # Lightweight cable
    max_tension=6000.0,         # Conservative tension limit
    max_sidewall_pressure=400.0,
    min_bend_radius=800.0,
    arrangement=CableArrangement.SINGLE
)

duct_spec = DuctSpec(
    inner_diameter=80.0,        # 80mm duct
    type="HDPE",
    friction_dry=0.30,          # Low friction HDPE
    friction_lubricated=0.12
)

# Run with conservative settings
pipeline = CablePullingPipeline(
    max_cable_length=600.0,     # 600m max continuous pull
    safety_factor=1.2           # Moderate safety factor
)

result = pipeline.run_analysis("route.dxf", cable_spec, duct_spec, lubricated=True)
```

### Case 2: Trefoil Cable Installation

```python
# Trefoil arrangement (3 cables)
trefoil_cable = CableSpec(
    diameter=50.0,              # Larger cables
    weight_per_meter=4.0,       # Heavier
    max_tension=12000.0,        # Higher tension capacity
    max_sidewall_pressure=600.0,
    min_bend_radius=1500.0,     # Larger bend radius needed
    arrangement=CableArrangement.TREFOIL,
    number_of_cables=3
)

large_duct = DuctSpec(
    inner_diameter=150.0,       # Larger duct for trefoil
    type="PVC",
    friction_dry=0.35,
    friction_lubricated=0.15
)

result = pipeline.run_analysis("route.dxf", trefoil_cable, large_duct)

# Check if installation is feasible
if result.summary["feasibility"]["overall_feasible"]:
    print("Trefoil installation is feasible!")
else:
    print("Trefoil installation challenges:")
    for section_id in result.critical_sections["tension"]:
        print(f"  Section {section_id}: tension limit exceeded")
    for section_id in result.critical_sections["pressure"]:
        print(f"  Section {section_id}: pressure limit exceeded")
```

### Case 3: Long Route with Splitting

```python
# Long route requiring multiple cable segments
pipeline = CablePullingPipeline(
    enable_splitting=True,
    max_cable_length=400.0,     # Split into 400m segments
    safety_factor=1.5
)

result = pipeline.run_analysis("long_route.dxf", cable_spec, duct_spec)

print(f"Original sections: {len(result.original_route.sections)}")
print(f"Final sections: {len(result.processed_route.sections)}")
print(f"Sections added by splitting: {result.geometry_result.splitting_result.sections_created}")

# Review split points
if result.geometry_result.splitting_result.split_points:
    print("Recommended split points:")
    for point in result.geometry_result.splitting_result.split_points:
        print(f"  Position: {point['position']:.1f}m, Reason: {point['reason']}")
```

## Working with Results

### Feasibility Analysis

```python
if result.success:
    feasibility = result.summary["feasibility"]

    if feasibility["overall_feasible"]:
        print("✓ Installation is feasible")
        print(f"  Max tension: {feasibility['max_tension_n']:.1f}N")
        print(f"  Safety margin: {(cable_spec.max_tension - feasibility['max_tension_n']):.1f}N")
    else:
        print("✗ Installation not feasible with current specifications")

        # Review critical sections
        critical = result.critical_sections
        if critical["tension"]:
            print(f"  Tension limits exceeded in sections: {', '.join(critical['tension'])}")
        if critical["pressure"]:
            print(f"  Pressure limits exceeded in sections: {', '.join(critical['pressure'])}")
        if critical["bend_radius"]:
            print(f"  Bend radius limits exceeded in sections: {', '.join(critical['bend_radius'])}")
```

### Section-by-Section Analysis

```python
print("Section-by-section analysis:")
for i, tension_analysis in enumerate(result.tension_analyses):
    limit_result = result.limit_results[i]

    print(f"\nSection {tension_analysis.section_id}:")
    print(f"  Length: {result.processed_route.sections[i].total_length:.1f}m")
    print(f"  Max tension: {tension_analysis.max_tension:.1f}N")
    print(f"  Max pressure: {limit_result.max_pressure:.1f}N/m")
    print(f"  Recommended direction: {limit_result.recommended_direction}")

    if limit_result.limiting_factors:
        print(f"  Limiting factors: {', '.join(limit_result.limiting_factors)}")
```

### Exporting Results

```python
from easycablepulling.core.pipeline import AnalysisReporter

# Generate comprehensive text report
report = AnalysisReporter.generate_text_report(result)
with open("analysis_report.txt", "w") as f:
    f.write(report)

# Export section data to CSV
csv_data = AnalysisReporter.generate_csv_report(result)
with open("section_analysis.csv", "w") as f:
    f.write(csv_data)

# Export machine-readable summary
import json
json_summary = AnalysisReporter.generate_json_summary(result)
with open("analysis_summary.json", "w") as f:
    json.dump(json_summary, f, indent=2)
```

## Visualization

### Basic Route Visualization

```python
from easycablepulling.visualization.route_plotter import RoutePlotter

plotter = RoutePlotter()

# Plot original vs fitted route
fig, ax = plotter.plot_route_overlay(
    result.original_route,
    result.processed_route,
    show_primitives=True,
    show_annotations=True
)

# Add analysis annotations
plotter.add_tension_annotations(ax, result.tension_analyses)
plotter.add_critical_section_highlights(ax, result.critical_sections)

fig.savefig("route_analysis.png", dpi=300, bbox_inches="tight")
```

### Professional Reports

```python
from easycablepulling.visualization import create_professional_plot

# Create professional route overview
overview_fig = create_professional_plot(
    result,
    plot_type="route_overview",
    title="Cable Installation Feasibility Analysis",
    subtitle=f"Route: {result.processed_route.name}",
    format="png",
    dpi=600
)
overview_fig.write_image("professional_overview.png")

# Create tension analysis plot
tension_fig = create_professional_plot(
    result,
    plot_type="tension_analysis",
    show_critical_sections=True,
    highlight_max_tension=True
)
tension_fig.write_image("tension_analysis.png")

# Create pressure analysis plot
pressure_fig = create_professional_plot(
    result,
    plot_type="pressure_analysis",
    show_bend_radii=True
)
pressure_fig.write_image("pressure_analysis.png")
```

## Command Line Usage

### Basic Analysis

```bash
# Analyze a single route
easycablepulling analyze input.dxf \
    --cable-diameter 35.0 \
    --cable-weight 2.5 \
    --max-tension 8000.0 \
    --duct-diameter 100.0 \
    --friction-dry 0.35 \
    --output analysis.json

# Include lubrication and splitting
easycablepulling analyze input.dxf \
    --cable-diameter 35.0 \
    --duct-diameter 100.0 \
    --lubricated \
    --enable-splitting \
    --max-cable-length 500.0 \
    --safety-factor 1.5 \
    --output results/
```

### Batch Processing

```bash
# Process multiple DXF files
easycablepulling batch-analyze routes/ \
    --output-dir results/ \
    --pattern "*.dxf" \
    --cable-diameter 35.0 \
    --duct-diameter 100.0 \
    --config cable_config.json
```

### Configuration Files

Create a configuration file to avoid repeating parameters:

```json
{
  "cable_spec": {
    "diameter": 35.0,
    "weight_per_meter": 2.5,
    "max_tension": 8000.0,
    "max_sidewall_pressure": 500.0,
    "min_bend_radius": 1200.0,
    "arrangement": "single"
  },
  "duct_spec": {
    "inner_diameter": 100.0,
    "type": "PVC",
    "friction_dry": 0.35,
    "friction_lubricated": 0.15
  },
  "pipeline_options": {
    "enable_splitting": true,
    "max_cable_length": 500.0,
    "safety_factor": 1.5
  }
}
```

Then use it:
```bash
easycablepulling analyze input.dxf --config config.json --output results.json
```

## Troubleshooting

### Common Issues

#### "Route not feasible" Results

**Problem**: Analysis shows route is not feasible for cable installation.

**Solutions**:
1. **Reduce safety factor**: Use `safety_factor=1.0` for preliminary analysis
2. **Enable lubrication**: Set `lubricated=True` to reduce friction
3. **Enable route splitting**: Use `enable_splitting=True` with shorter `max_cable_length`
4. **Use larger duct**: Increase `duct_spec.inner_diameter`
5. **Use lighter cable**: Reduce `cable_spec.weight_per_meter`
6. **Increase cable limits**: Raise `max_tension` or `max_sidewall_pressure` if cable specifications allow

```python
# Try with relaxed constraints
relaxed_pipeline = CablePullingPipeline(
    enable_splitting=True,
    max_cable_length=300.0,          # Shorter segments
    safety_factor=1.0                # No extra safety margin
)

result = relaxed_pipeline.run_analysis(
    "route.dxf",
    cable_spec,
    duct_spec,
    lubricated=True                  # Use lubrication
)
```

#### "Geometry processing failed" Errors

**Problem**: The system cannot fit geometric primitives to the route.

**Solutions**:
1. **Check DXF file**: Ensure polylines are continuous and well-formed
2. **Adjust tolerances**: Increase `geometric_tolerance` for complex routes
3. **Simplify geometry**: Remove unnecessary detail points from DXF
4. **Check scale**: Ensure DXF units are in millimeters

```python
# More tolerant geometry processing
tolerant_pipeline = CablePullingPipeline(
    geometric_tolerance=5.0,         # More tolerant fitting
    angle_tolerance=5.0,             # Allow more angle variation
    max_fitting_iterations=200       # More attempts to fit
)
```

#### High Memory Usage

**Problem**: Analysis uses excessive memory for large routes.

**Solutions**:
1. **Enable splitting**: Break long routes into segments
2. **Reduce DXF detail**: Simplify polylines before analysis
3. **Process sections separately**: Analyze route sections individually

```python
# Memory-efficient processing
efficient_pipeline = CablePullingPipeline(
    enable_splitting=True,
    max_cable_length=200.0,          # Smaller segments
    geometric_tolerance=2.0          # Less detailed fitting
)
```

### Performance Optimization

#### For Large Routes (>2km)

```python
# Optimized for large routes
large_route_pipeline = CablePullingPipeline(
    enable_splitting=True,
    max_cable_length=400.0,
    geometric_tolerance=2.0,         # Faster fitting
    max_fitting_iterations=50        # Fewer iterations
)
```

#### For Batch Processing

```python
# Reuse pipeline instance for multiple analyses
pipeline = CablePullingPipeline()

routes = ["route1.dxf", "route2.dxf", "route3.dxf"]
results = []

for route_file in routes:
    result = pipeline.run_analysis(route_file, cable_spec, duct_spec)
    results.append(result)

    # Log progress
    print(f"Processed {route_file}: {'✓' if result.success else '✗'}")
```

## Advanced Features

### Variable Conditions

Analyze routes where different sections have different duct types or lubrication:

```python
from easycablepulling.calculations import analyze_route_with_varying_conditions

# Different duct types per section
duct_specs = [
    DuctSpec(inner_diameter=100.0, type="PVC", friction_dry=0.35, friction_lubricated=0.15),
    DuctSpec(inner_diameter=120.0, type="HDPE", friction_dry=0.30, friction_lubricated=0.12),
    DuctSpec(inner_diameter=100.0, type="Steel", friction_dry=0.40, friction_lubricated=0.18)
]

# Lubrication varies by section
lubricated_sections = [False, True, False]

# Process geometry first
geometry_result = pipeline.run_geometry_only("route.dxf", duct_specs[0], cable_spec)

if geometry_result.success:
    # Run advanced analysis
    route_analysis = analyze_route_with_varying_conditions(
        geometry_result.route,
        cable_spec,
        duct_specs,
        lubricated_sections
    )

    # Review per-section results
    for tension_analysis, limit_result, conditions in route_analysis:
        print(f"Section {tension_analysis.section_id}:")
        print(f"  Max tension: {tension_analysis.max_tension:.1f}N")
        print(f"  Feasible: {limit_result.passes_all_limits}")
```

### Custom Analysis Parameters

```python
# Environmental factors
result = pipeline.run_analysis(
    "route.dxf",
    cable_spec,
    duct_spec,
    temperature_factor=1.1,          # 10% increase for temperature
    slope_factor=0.05,               # 5% grade
    weight_factor=1.2                # 20% extra weight (ice, etc.)
)
```

### Geometry-Only Processing

For cases where you only need geometry fitting without cable calculations:

```python
geometry_result = pipeline.run_geometry_only(
    "route.dxf",
    duct_spec,
    cable_spec
)

if geometry_result.success:
    print(f"Fitted {sum(len(s.primitives) for s in geometry_result.route.sections)} primitives")

    # Export fitted geometry
    from easycablepulling.io.dxf_writer import DXFWriter
    writer = DXFWriter()
    writer.write_fitted_route(geometry_result.route)
    writer.save("fitted_route.dxf")
```

## Output Formats

### Text Reports

```python
from easycablepulling.core.pipeline import AnalysisReporter

# Comprehensive text report
report = AnalysisReporter.generate_text_report(result)
print(report)

# Save to file
with open("analysis_report.txt", "w") as f:
    f.write(report)
```

### CSV Data Export

```python
# Section-by-section data
csv_data = AnalysisReporter.generate_csv_report(result)

# Load into pandas for further analysis
import pandas as pd
from io import StringIO

df = pd.read_csv(StringIO(csv_data))
print(df.describe())

# Filter critical sections
critical_df = df[~df['passes_all_limits']]
print(f"Critical sections: {len(critical_df)}")
```

### JSON Export

```python
# Machine-readable summary
json_summary = AnalysisReporter.generate_json_summary(result)

# Access specific data
meta = json_summary["meta"]
results = json_summary["results"]
sections = json_summary["sections"]

print(f"Analysis timestamp: {meta['analysis_timestamp']}")
print(f"Overall feasible: {results['overall_feasible']}")

# Find most challenging section
max_tension_section = max(sections, key=lambda s: s['max_tension_n'])
print(f"Highest tension in section: {max_tension_section['id']}")
```

## Best Practices

### Route Preparation

1. **Use consistent units**: Ensure DXF files use millimeters
2. **Simplify geometry**: Remove unnecessary detail points
3. **Check continuity**: Ensure polylines are continuous
4. **Verify scale**: Confirm route dimensions are realistic

### Analysis Setup

1. **Conservative specifications**: Start with manufacturer specifications
2. **Use safety factors**: Apply appropriate safety margins (1.2-2.0)
3. **Consider conditions**: Account for installation environment
4. **Enable splitting**: For routes >500m, enable automatic splitting

### Result Interpretation

1. **Review critical sections**: Focus on sections that exceed limits
2. **Check recommendations**: Follow direction recommendations
3. **Validate results**: Ensure results are physically reasonable
4. **Document assumptions**: Record analysis parameters used

### Workflow Integration

```python
def analyze_cable_route(dxf_file, cable_type="standard", duct_type="pvc_100mm"):
    """Standard workflow for cable route analysis."""

    # Load standard specifications
    cable_specs = {
        "standard": CableSpec(diameter=35.0, weight_per_meter=2.5, max_tension=8000.0,
                             max_sidewall_pressure=500.0, min_bend_radius=1200.0),
        "heavy_duty": CableSpec(diameter=50.0, weight_per_meter=4.0, max_tension=12000.0,
                               max_sidewall_pressure=800.0, min_bend_radius=1500.0)
    }

    duct_specs = {
        "pvc_100mm": DuctSpec(inner_diameter=100.0, type="PVC", friction_dry=0.35, friction_lubricated=0.15),
        "hdpe_120mm": DuctSpec(inner_diameter=120.0, type="HDPE", friction_dry=0.30, friction_lubricated=0.12)
    }

    # Configure pipeline
    pipeline = CablePullingPipeline(
        enable_splitting=True,
        max_cable_length=500.0,
        safety_factor=1.5
    )

    # Run analysis
    result = pipeline.run_analysis(
        dxf_file,
        cable_specs[cable_type],
        duct_specs[duct_type],
        lubricated=True  # Assume lubrication available
    )

    # Generate outputs
    if result.success:
        # Save comprehensive report
        report = AnalysisReporter.generate_text_report(result)
        with open(f"{Path(dxf_file).stem}_report.txt", "w") as f:
            f.write(report)

        # Save data for further analysis
        json_summary = AnalysisReporter.generate_json_summary(result)
        with open(f"{Path(dxf_file).stem}_data.json", "w") as f:
            json.dump(json_summary, f, indent=2)

        print(f"Analysis complete: {'✓ Feasible' if result.summary['feasibility']['overall_feasible'] else '✗ Not feasible'}")
    else:
        print(f"Analysis failed: {result.errors}")

    return result

# Use the workflow
result = analyze_cable_route("my_route.dxf", "standard", "pvc_100mm")
```

## Integration Examples

### CAD Integration

```python
def process_cad_export(cad_file, output_dir):
    """Process route exported from CAD system."""

    pipeline = CablePullingPipeline(enable_splitting=True, safety_factor=1.5)

    # Standard utility specifications
    cable_spec = CableSpec(
        diameter=35.0, weight_per_meter=2.5, max_tension=8000.0,
        max_sidewall_pressure=500.0, min_bend_radius=1200.0
    )

    duct_spec = DuctSpec(
        inner_diameter=100.0, type="PVC",
        friction_dry=0.35, friction_lubricated=0.15
    )

    result = pipeline.run_analysis(cad_file, cable_spec, duct_spec, lubricated=True)

    # Generate standard deliverables
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Technical report
    report = AnalysisReporter.generate_text_report(result)
    (output_path / "technical_report.txt").write_text(report)

    # Data summary
    json_summary = AnalysisReporter.generate_json_summary(result)
    with open(output_path / "analysis_data.json", "w") as f:
        json.dump(json_summary, f, indent=2)

    # Professional plots
    if result.success:
        overview_fig = create_professional_plot(result, "route_overview")
        overview_fig.write_image(output_path / "route_overview.png")

        if not result.summary["feasibility"]["overall_feasible"]:
            # Create problem analysis plots
            tension_fig = create_professional_plot(result, "tension_analysis")
            tension_fig.write_image(output_path / "tension_issues.png")

    return result
```

### Project Management Integration

```python
def generate_project_summary(route_files, project_name):
    """Generate project-level summary for multiple routes."""

    pipeline = CablePullingPipeline(enable_splitting=True)
    project_results = {}

    for route_file in route_files:
        route_name = Path(route_file).stem
        result = pipeline.run_analysis(route_file, cable_spec, duct_spec)
        project_results[route_name] = result

    # Project summary
    total_length = sum(
        r.summary["total_length_m"] for r in project_results.values() if r.success
    )

    feasible_routes = [
        name for name, r in project_results.items()
        if r.success and r.summary["feasibility"]["overall_feasible"]
    ]

    print(f"Project: {project_name}")
    print(f"Total cable length: {total_length:.1f}m")
    print(f"Feasible routes: {len(feasible_routes)}/{len(project_results)}")

    # Critical routes requiring attention
    critical_routes = [
        name for name, r in project_results.items()
        if r.success and not r.summary["feasibility"]["overall_feasible"]
    ]

    if critical_routes:
        print(f"Routes requiring attention: {', '.join(critical_routes)}")

    return project_results
```
