# Data Models Documentation

This document describes the core data models used in the Easy Cable Pulling software.

## Overview

The data models represent the physical components and constraints involved in cable pulling operations:

- **Cable specifications** (diameter, weight, tension limits)
- **Duct specifications** (diameter, friction, bend options)
- **Route geometry** (straights and bends)
- **Sections and routes** (organized geometry data)

## Core Models

### CableSpec

Represents the specifications of a cable or cable bundle.

```python
from easycablepulling.core import CableSpec, CableArrangement, PullingMethod

cable = CableSpec(
    diameter=50.0,                    # mm
    weight_per_meter=2.5,            # kg/m
    max_tension=5000.0,              # N
    max_sidewall_pressure=3000.0,    # N/m
    min_bend_radius=600.0,           # mm
    pulling_method=PullingMethod.EYE,
    arrangement=CableArrangement.SINGLE,
    number_of_cables=1
)
```

#### Cable Arrangements

- **SINGLE**: One cable
- **TREFOIL**: Three cables in triangular formation (bundle diameter ≈ 2.15 × single cable)
- **FLAT**: Multiple cables side by side

#### Properties

- `bundle_diameter`: Effective diameter based on arrangement
- `total_weight_per_meter`: Total weight for all cables

### DuctSpec

Represents the specifications of a duct or conduit.

```python
from easycablepulling.core import DuctSpec, BendOption

duct = DuctSpec(
    inner_diameter=100.0,        # mm
    type="PVC",                  # PVC, HDPE, Steel, Concrete
    friction_dry=0.35,
    friction_lubricated=0.15,
    bend_options=[
        BendOption(radius=1000.0, angle=90.0),
        BendOption(radius=1500.0, angle=45.0)
    ]
)
```

#### Methods

- `can_accommodate_cable(cable_spec)`: Check if cable bundle fits (90% fill ratio max)
- `get_friction(arrangement, lubricated)`: Get friction coefficient adjusted for cable arrangement

### Geometric Primitives

#### Straight

Represents a straight section of cable route.

```python
from easycablepulling.core import Straight

straight = Straight(
    length_m=100.0,
    start_point=(0.0, 0.0),
    end_point=(100.0, 0.0)
)
```

#### Bend

Represents a curved section of cable route.

```python
from easycablepulling.core import Bend

bend = Bend(
    radius_m=1.0,              # meters
    angle_deg=90.0,            # degrees
    direction="CW",            # CW or CCW
    center_point=(0.0, 0.0)
)
```

Properties:
- `angle_rad`: Angle in radians
- `length()`: Arc length in meters

### Section

Represents a section of cable route between joints/pits.

```python
from easycablepulling.core import Section, Straight, Bend

section = Section(
    id="AB",
    original_polyline=[(0, 0), (100, 0), (100, 100)],
    primitives=[
        Straight(length_m=100.0, start_point=(0, 0), end_point=(100, 0)),
        Bend(radius_m=10.0, angle_deg=90.0, direction="CW", center_point=(100, 10))
    ]
)
```

Properties:
- `total_length`: Sum of primitive lengths
- `original_length`: Length of original polyline
- `length_error_percent`: Percentage difference between fitted and original

Methods:
- `validate_fit(max_length_error)`: Check geometry fit tolerance
- `validate_cable(cable_spec)`: Check cable constraints (bend radius, etc.)

### Route

Represents a complete cable route composed of multiple sections.

```python
from easycablepulling.core import Route, Section

route = Route(name="Main Distribution Route")
route.add_section(section_ab)
route.add_section(section_bc)

# Validate entire route
warnings = route.validate_cable(cable_spec)
```

Properties:
- `total_length`: Sum of all section lengths
- `section_count`: Number of sections

Methods:
- `add_section(section)`: Add a section to the route
- `validate_cable(cable_spec)`: Validate all sections against cable specs
- `validate_fit(max_length_error)`: Check geometry fit for all sections

## Validation

The models include built-in validation:

### Cable Validation
- Minimum bend radius enforcement
- Positive values for all measurements
- Arrangement vs. number of cables consistency

### Duct Validation
- Positive dimensions
- Friction coefficients between 0 and 1
- Lubricated friction < dry friction

### Geometry Validation
- Length consistency between coordinates and specified length
- Bend angle between 0 and 180 degrees
- Tangent continuity (in future implementations)

## Usage Example

```python
from easycablepulling.core import (
    CableSpec, DuctSpec, Route, Section,
    Straight, Bend, CableArrangement
)

# Define cable specifications
cable = CableSpec(
    diameter=50.0,
    weight_per_meter=2.5,
    max_tension=5000.0,
    max_sidewall_pressure=3000.0,
    min_bend_radius=600.0,
    arrangement=CableArrangement.TREFOIL,
    number_of_cables=3
)

# Define duct specifications
duct = DuctSpec(
    inner_diameter=150.0,
    type="PVC",
    friction_dry=0.35,
    friction_lubricated=0.15
)

# Check if cable fits in duct
if duct.can_accommodate_cable(cable):
    print(f"Cable bundle ({cable.bundle_diameter:.1f}mm) fits in duct")

# Get appropriate friction coefficient
friction = duct.get_friction(cable.arrangement, lubricated=True)
print(f"Friction coefficient: {friction}")

# Create route and validate
route = Route(name="Substation Feed")
# ... add sections ...

warnings = route.validate_cable(cable)
if warnings:
    print("Validation warnings:")
    for warning in warnings:
        print(f"  - {warning}")
```
