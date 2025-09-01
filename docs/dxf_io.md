# DXF Import/Export Documentation

This document describes the DXF file import and export functionality for cable route analysis.

## Overview

The DXF I/O module handles:

- **Import**: Loading polylines from DXF files and converting them to Route objects
- **Export**: Writing analysis results back to DXF with proper layer organization
- **Section Identification**: Automatically splitting routes into manageable sections
- **Round-trip Compatibility**: Preserving geometry through import/export cycles

## DXF Import

### Basic Usage

```python
from easycablepulling.io import load_route_from_dxf
from pathlib import Path

# Simple import
route = load_route_from_dxf(Path("route.dxf"), "Main Distribution Route")

print(f"Loaded route: {route.name}")
print(f"Sections: {route.section_count}")
print(f"Total length: {sum(s.original_length for s in route.sections):.2f}m")
```

### Advanced Import

```python
from easycablepulling.io import DXFReader, PolylineParser

# Create reader and load file
reader = DXFReader(Path("route.dxf"))
reader.load()

# Get file summary
summary = reader.get_route_summary()
print(f"DXF Version: {summary['dxf_version']}")
print(f"Layers: {summary['layers']}")
print(f"Total polylines: {summary['total_polylines']}")

# Extract polylines from specific layer
polylines = reader.extract_polylines(layer_name="_FUN_33kV OPT 2 Overview Route")

# Create parser and identify sections
parser = PolylineParser(joint_detection_distance=50.0)
sections = parser.identify_sections(polylines)

print(f"Identified {len(sections)} sections")
```

### Section Identification

The parser automatically identifies route sections using:

1. **Multiple Polylines**: Each polyline becomes a separate section
2. **Single Polyline Splitting**: Uses angle changes and distance thresholds
   - Significant direction changes (> 45°) indicate joints/pits
   - Distance-based splitting for very long sections
   - Configurable joint detection distance

### Supported DXF Entities

- **LWPOLYLINE** (Lightweight Polylines) - Primary format
- **POLYLINE** (Legacy Polylines) - Also supported
- **Layers**: Automatically detected, can filter by specific layer

## DXF Export

### Basic Export

```python
from easycablepulling.io import export_route_to_dxf

# Export route with default annotations
export_route_to_dxf(
    route=route,
    file_path=Path("output.dxf"),
    include_annotations=True,
    include_joint_markers=True
)
```

### Export with Analysis Results

```python
# Include analysis results and warnings
analysis_results = {
    "max_tension": 15000.0,
    "max_sidewall_pressure": 3500.0,
    "optimal_direction": "Forward"
}

warnings = [
    "Section AB: Length error 0.3% exceeds maximum 0.2%",
    "Section BC: Bend radius 0.8m is less than minimum 1.0m"
]

export_route_to_dxf(
    route=route,
    file_path=Path("analysis_output.dxf"),
    analysis_results=analysis_results,
    warnings=warnings
)
```

### Advanced Export

```python
from easycablepulling.io import DXFWriter

# Create writer with specific DXF version
writer = DXFWriter(dxf_version="R2018")

# Write different components to different layers
writer.write_original_route(route, layer_name="ROUTE_ORIGINAL")
writer.write_fitted_route(route, layer_name="ROUTE_FITTED")
writer.write_section_annotations(route, text_height=5.0)
writer.write_joint_markers(route, marker_size=10.0)

# Add custom warnings
custom_warnings = ["Custom validation warning"]
writer.write_warnings(custom_warnings, location=(1000, 1000))

# Save file
writer.save(Path("custom_output.dxf"))
```

## Layer Organization

The DXF writer creates organized layers for different types of information:

| Layer Name | Color | Content | Linetype |
|------------|-------|---------|----------|
| ROUTE_ORIGINAL | White (7) | Original polylines from input | CONTINUOUS |
| ROUTE_FITTED | Yellow (2) | Fitted route geometry | CONTINUOUS |
| PRIMITIVES_STRAIGHT | Green (3) | Straight sections | CONTINUOUS |
| PRIMITIVES_BEND | Red (1) | Bend sections | DASHED |
| ANNOTATIONS | Cyan (4) | Section labels and text | CONTINUOUS |
| JOINTS | Blue (5) | Joint/pit markers | CONTINUOUS |
| WARNINGS | Red (1) | Warning messages | CONTINUOUS |

## File Format Support

### Input Requirements

- **DXF Version**: AutoCAD R2010 or newer (recommended)
- **Geometry**: LWPOLYLINE or POLYLINE entities
- **Coordinates**: 2D coordinates (X,Y) - elevation not required
- **Layer Structure**:
  - Single layer preferred
  - Multiple polylines forming one route supported
  - Layer name can be any valid DXF layer name

### Output Format

- **DXF Version**: R2010 (default) for maximum compatibility
- **Entities**:
  - LWPOLYLINE for route geometry
  - LINE for straight primitives
  - ARC for bend primitives
  - TEXT for annotations
  - CIRCLE for joint markers

## Examples with Real Data

Based on the included `examples/input.dxf` file:

```python
from easycablepulling.io import load_route_from_dxf
from pathlib import Path

# Load the example route
route = load_route_from_dxf(Path("examples/input.dxf"))

# File contains:
# - 13 polylines on layer '_FUN_33kV OPT 2 Overview Route'
# - Total length: ~6.1 km
# - Sections range from 17m to 2.2km
# - Complex route with multiple direction changes

print(f"Loaded {route.section_count} sections")
for section in route.sections:
    print(f"  {section.id}: {section.original_length:.0f}m")
```

## Error Handling

### Common Import Issues

```python
try:
    route = load_route_from_dxf(Path("route.dxf"))
except ValueError as e:
    if "Failed to load DXF file" in str(e):
        print("File not found or corrupted")
    elif "No polylines found" in str(e):
        print("DXF file contains no route geometry")
    else:
        print(f"Import error: {e}")
```

### Validation Checks

```python
# Check if route sections are valid
for section in route.sections:
    if len(section.original_polyline) < 2:
        print(f"Warning: Section {section.id} has insufficient points")

    if section.original_length == 0:
        print(f"Warning: Section {section.id} has zero length")
```

## Performance Notes

- **Large Files**: The reader efficiently handles large DXF files
- **Memory Usage**: Polylines are loaded into memory - consider file size
- **Layer Filtering**: Use layer filtering for faster processing of complex files
- **Section Count**: Many small sections may impact performance in later phases

## Integration with Other Modules

The DXF I/O module integrates with:

- **Core Models**: Creates Section and Route objects
- **Geometry Module**: Will provide primitive fitting (Phase 3)
- **Calculations Module**: Analysis results can be exported back to DXF
- **CLI**: Commands will use these functions for file operations

## File Paths and Organization

```
project/
├── examples/
│   └── input.dxf          # Sample input file
├── output/
│   ├── route_original.dxf # Original geometry
│   ├── route_fitted.dxf   # Fitted primitives
│   └── route_analysis.dxf # Complete analysis results
```
