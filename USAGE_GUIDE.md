# EasyCablePulling Usage Guide

This guide provides detailed examples and practical usage scenarios for the EasyCablePulling library.

## Table of Contents

1. [Basic Usage](#basic-usage)
2. [Configuration Options](#configuration-options)
3. [Understanding Results](#understanding-results)
4. [Output Files](#output-files)
5. [Advanced Usage](#advanced-usage)
6. [Troubleshooting](#troubleshooting)

## Basic Usage

### Simple Analysis

The quickest way to analyze a cable route:

```python
from easycablepulling import analyze_cable_route

# Analyze with defaults (200mm duct, 50mm cable, 1000m max sections)
results = analyze_cable_route("my_cable_route.dxf")

print(f"Analysis complete for {results.route_name}")
print(f"Total route length: {results.total_length_m:.1f} meters")
print(f"Number of sections: {results.section_count}")
print(f"Final pulling force (forward): {results.final_forward_tension_n:.0f} N")
print(f"Final pulling force (reverse): {results.final_reverse_tension_n:.0f} N")
```

### Specify Output Directory

```python
results = analyze_cable_route(
    dxf_path="cable_routes/main_trunk.dxf",
    output_dir="analysis_results/main_trunk"
)
```

## Configuration Options

### Standard Configurations

```python
from easycablepulling import CableAnalysisPipeline, AnalysisConfig

# High-voltage cable configuration
hv_config = AnalysisConfig(
    duct_type="200mm",
    cable_diameter_mm=70.0,        # Larger cable
    cable_weight_kg_m=4.2,         # Heavier cable
    cable_max_tension_n=25000.0,   # Higher tension limit
    max_section_length_m=800.0,    # Shorter sections for HV
    sample_interval_m=10.0         # More detailed sampling
)

# Low-voltage multi-cable configuration
lv_config = AnalysisConfig(
    duct_type="100mm",
    cable_diameter_mm=25.0,
    cable_weight_kg_m=1.2,
    number_of_cables=4,            # Multiple cables
    max_section_length_m=1500.0,   # Longer sections OK for LV
    sample_interval_m=50.0
)

# Run analysis with custom config
pipeline = CableAnalysisPipeline(hv_config)
results = pipeline.analyze_dxf("hv_route.dxf", "hv_analysis")
```

### Output Format Control

```python
# JSON only (for API integration)
api_config = AnalysisConfig(
    generate_json=True,
    generate_csv=False,
    generate_png=False,
    generate_dxf=False
)

# Full reporting suite
full_config = AnalysisConfig(
    generate_json=True,
    generate_csv=True,
    generate_excel=True,    # If implemented
    generate_png=True,
    generate_dxf=True       # Export fitted geometry
)
```

### Quick Configuration Overrides

```python
# Override specific parameters without creating full config
results = analyze_cable_route(
    "route.dxf",
    output_dir="analysis",
    cable_diameter_mm=60.0,
    max_section_length_m=750.0,
    generate_png=False
)
```

## Understanding Results

### Analysis Results Structure

```python
results = analyze_cable_route("route.dxf")

# Route-level information
print(f"Route: {results.route_name}")
print(f"Total length: {results.total_length_m:.1f}m")
print(f"Sections: {results.section_count}")

# Geometry summary
print(f"Total straights: {results.total_straights}")
print(f"Total bends: {results.total_bends}")

# Critical pulling forces
print(f"Maximum forward tension: {results.final_forward_tension_n:.0f}N")
print(f"Maximum reverse tension: {results.final_reverse_tension_n:.0f}N")
print(f"Maximum sidewall pressure: {results.max_sidewall_pressure_n_m:.0f}N/m")

# Accuracy metrics
print(f"Excellent accuracy: {results.excellent_accuracy_percent:.1f}%")
print(f"Median deviation: {results.median_deviation_cm:.1f}cm")
print(f"Maximum deviation: {results.max_deviation_cm:.1f}cm")
```

### Section-by-Section Analysis

```python
# Examine individual sections
for i, section in enumerate(results.sections):
    print(f"\n--- Section {section.section_id} ---")
    print(f"Length: {section.length_m:.1f}m")
    print(f"Geometry: {section.straight_count} straights, {section.bend_count} bends")
    print(f"Forward tension: {section.forward_tension_n:.0f}N")
    print(f"Reverse tension: {section.reverse_tension_n:.0f}N")
    print(f"Cumulative forward: {section.cumulative_forward_n:.0f}N")

    # Detailed geometry
    for straight in section.straights:
        print(f"  Straight: {straight['length_m']:.1f}m")

    for bend in section.bends:
        print(f"  Bend: {bend['angle_deg']:.1f}° @ {bend['radius_m']:.1f}m radius")
```

### Tension Safety Analysis

```python
def check_tension_safety(results, max_safe_tension=15000):
    """Check if tensions exceed safe limits."""

    critical_sections = []

    for section in results.sections:
        forward_safe = section.forward_tension_n <= max_safe_tension
        reverse_safe = section.reverse_tension_n <= max_safe_tension

        if not (forward_safe and reverse_safe):
            critical_sections.append({
                'section_id': section.section_id,
                'forward_tension': section.forward_tension_n,
                'reverse_tension': section.reverse_tension_n,
                'forward_safe': forward_safe,
                'reverse_safe': reverse_safe
            })

    if critical_sections:
        print("⚠️  WARNING: High tension sections found:")
        for section in critical_sections:
            print(f"Section {section['section_id']}: "
                  f"F={section['forward_tension']:.0f}N "
                  f"R={section['reverse_tension']:.0f}N")
    else:
        print("✅ All sections within safe tension limits")

    return critical_sections

# Usage
critical = check_tension_safety(results, max_safe_tension=15000)
```

## Output Files

### File Structure Overview

After analysis, you'll find this structure in your output directory:

```
output/
├── visualizations/
│   ├── route_overview.png          # Complete route visualization
│   └── sections/
│       ├── section_1.png           # Individual section details
│       ├── section_2.png
│       └── ...
├── json/
│   ├── section_1.json              # Detailed section data
│   ├── section_2.json
│   └── ...
├── csv/
│   └── sections_summary.csv        # Spreadsheet-friendly summary
├── analysis_summary.json           # Complete results (JSON)
├── analysis_summary.csv            # Summary metrics (CSV)
└── fitted_route.dxf                # Optional: fitted geometry for CAD
```

### Working with JSON Output

```python
import json

# Load detailed section data
with open("output/json/section_1.json", 'r') as f:
    section_data = json.load(f)

print(f"Section {section_data['section_id']}")
print(f"Straights: {len(section_data['straights'])}")
print(f"Bends: {len(section_data['bends'])}")

# Load complete analysis summary
with open("output/analysis_summary.json", 'r') as f:
    summary = json.load(f)

# Extract key metrics
total_length = summary['total_length_m']
max_tension = summary['final_forward_tension_n']
accuracy = summary['excellent_accuracy_percent']
```

### Working with CSV Output

```python
import pandas as pd

# Load sections summary
df = pd.read_csv("output/csv/sections_summary.csv")

# Analyze tension distribution
print("Tension Statistics:")
print(f"Mean forward tension: {df['Forward Tension (N)'].mean():.0f}N")
print(f"Max forward tension: {df['Forward Tension (N)'].max():.0f}N")
print(f"Sections >10kN: {(df['Forward Tension (N)'] > 10000).sum()}")

# Plot tension profile
df.plot(x='Section ID', y=['Forward Tension (N)', 'Reverse Tension (N)'])
```

## Advanced Usage

### Custom Cable Specifications

```python
# Define custom cable properties
custom_config = AnalysisConfig(
    # Large transmission cable
    cable_diameter_mm=95.0,
    cable_weight_kg_m=6.8,
    cable_max_tension_n=50000.0,
    cable_max_sidewall_pressure_n_m=500.0,
    cable_min_bend_radius_mm=1000.0,

    # Specialized duct
    duct_type="250mm",

    # Tight analysis parameters
    max_section_length_m=500.0,
    sample_interval_m=5.0
)
```

### Batch Processing

```python
import os
from pathlib import Path

def batch_analyze_routes(input_dir, output_base_dir):
    """Analyze multiple DXF files in a directory."""

    input_path = Path(input_dir)
    results_summary = []

    for dxf_file in input_path.glob("*.dxf"):
        print(f"Analyzing {dxf_file.name}...")

        # Create output directory for this route
        output_dir = Path(output_base_dir) / dxf_file.stem

        try:
            results = analyze_cable_route(dxf_file, output_dir)

            results_summary.append({
                'file': dxf_file.name,
                'length_m': results.total_length_m,
                'sections': results.section_count,
                'max_forward_tension_n': results.final_forward_tension_n,
                'max_reverse_tension_n': results.final_reverse_tension_n,
                'accuracy_percent': results.excellent_accuracy_percent
            })

            print(f"✅ {dxf_file.name}: {results.total_length_m:.0f}m, "
                  f"{results.section_count} sections")

        except Exception as e:
            print(f"❌ {dxf_file.name}: Error - {e}")
            results_summary.append({
                'file': dxf_file.name,
                'error': str(e)
            })

    return results_summary

# Usage
summary = batch_analyze_routes("input_routes/", "batch_analysis/")
```

### Integration with Other Tools

```python
# Export data for external analysis
def export_for_excel_analysis(results, filename):
    """Export detailed data for Excel analysis."""

    import pandas as pd

    # Create detailed section breakdown
    section_data = []

    for section in results.sections:
        base_data = {
            'Section_ID': section.section_id,
            'Length_m': section.length_m,
            'Straight_Count': section.straight_count,
            'Bend_Count': section.bend_count,
            'Forward_Tension_N': section.forward_tension_n,
            'Reverse_Tension_N': section.reverse_tension_n,
            'Max_Sidewall_Pressure_N_m': section.max_sidewall_pressure_n_m,
            'Cumulative_Forward_N': section.cumulative_forward_n,
            'Cumulative_Reverse_N': section.cumulative_reverse_n
        }

        section_data.append(base_data)

    df = pd.DataFrame(section_data)
    df.to_excel(filename, index=False, sheet_name='Section_Analysis')
    print(f"Excel export saved to {filename}")

# Usage
export_for_excel_analysis(results, "detailed_analysis.xlsx")
```

## Troubleshooting

### Common Issues

**Issue: "No polylines found in DXF"**
```python
# Solution: Check your DXF file contains LWPOLYLINE entities
from easycablepulling.io.dxf_reader import DXFReader

reader = DXFReader("problematic.dxf")
reader.load()
print(f"Found entities: {[ent.dxftype() for ent in reader.doc.modelspace()]}")
```

**Issue: "Section too short after filleting"**
```python
# Solution: Check for very short segments or tight curves
# Increase the minimum segment length or adjust duct radius

config = AnalysisConfig(
    max_section_length_m=2000.0,  # Longer sections
    sample_interval_m=50.0        # Less detailed sampling
)
```

**Issue: "High deviation values"**
```python
# Expected deviations for different angles:
# 90° corner: ~114cm max deviation (3.9m radius)
# 45° corner: ~51cm max deviation
# 30° corner: ~31cm max deviation

# If deviations exceed these, check for:
# - Duplicate vertices in original route
# - Very short segments between direction changes
# - Inappropriate radius for duct type
```

### Performance Optimization

```python
# For large routes, optimize performance:
performance_config = AnalysisConfig(
    sample_interval_m=100.0,      # Larger sampling interval
    generate_png=False,           # Skip visualization for speed
    max_section_length_m=2000.0   # Fewer sections
)
```

### Memory Management

```python
# For very large routes, process sections individually:
def analyze_large_route(dxf_path):
    """Memory-efficient analysis of large routes."""

    pipeline = CableAnalysisPipeline()

    # Load route but don't keep all data in memory
    reader = DXFReader(dxf_path)
    reader.load()
    route = reader.create_route_from_polylines(Path(dxf_path).stem)

    section_results = []

    for i, section in enumerate(route.sections):
        print(f"Processing section {i+1}/{len(route.sections)}")

        # Process individual section
        # ... section processing logic

        # Clear memory after each section if needed
        import gc
        gc.collect()

    return section_results
```

This usage guide covers the most common scenarios and should help you get started with practical cable pulling analysis using the EasyCablePulling library.
