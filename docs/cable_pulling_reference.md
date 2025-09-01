# Cable Pulling Analysis Guide

## Overview

[Placeholder] and provides comprehensive tension analysis for cable installation planning.

## Key Features

### 1. [Placeholder]
-

### 2. Cable Pulling Physics

#### Tension in Straight Sections
For straight or horizontal cable runs, tension increases due to friction:
```
T = L × w × f × W
```
Where:
- `T` = pulling tension (N)
- `L` = length of cable section (m)
- `w` = cable weight per unit length (kg/m × g)
- `f` = coefficient of friction
- `W` = weight correction factor (typically 1.0)

#### Tension Through Bends (Capstan Equation)
When pulling cable through a bend, tension multiplies exponentially:
```
Tb = Ts × e^(f×α)
```
Where:
- `Tb` = tension exiting the bend (N)
- `Ts` = tension entering the bend (N)
- `e` = Euler's number (2.718...)
- `f` = coefficient of friction
- `α` = bend angle in radians

#### Sidewall Pressure
The crushing force on cable in bends:
```
P = T / r
```
Where:
- `P` = sidewall pressure (N/m)
- `T` = tension at bend exit (N)
- `r` = bend radius (m)

### 3. Bidirectional Analysis
- **Forward Direction (A→B)**: Pulling from start to end of each leg
- **Reverse Direction (B→A)**: Pulling from end to start of each leg
- Automatically identifies optimal pulling direction (lowest maximum tension)

## Installation

```bash
# Create virtual environment
python3 -m venv cable_env
source cable_env/bin/activate

# Install dependencies
pip install [placeholder]
```

## Usage

### Basic Command
```bash
python [placeholder]```

### Parameters

#### Output Options
- `--summary`: Summary CSV with max tensions per leg
- `--png`: Overlay image with color-coded legs and labeled endpoints
- `--raw`: Raw extracted data CSV (auto-generated if not specified)

#### Advanced Options
- `--max-tension`: Maximum allowable pulling tension (N)
  - Default: 22,250 N (single cable)
  - Default: 44,500 N (three cables)
- `--max-sidewall-pressure`: Maximum allowable sidewall pressure (N/m, default: 4,380)

## Output Files

### 1. Detailed Analysis CSV (`route_analysis.csv`)
Contains every segment with:
- `LegId`, `LegLabel`: Leg identification
- `Type`: Straight, Bend
- `Length`, `Radius`, `AngleDeg`, `ArcLength`: Geometric properties
- `StartX`, `StartY`, `EndX`, `EndY`: Coordinates
- `TensionForward_N`: Cumulative tension in forward direction
- `TensionReverse_N`: Cumulative tension in reverse direction
- `SidewallPressure_N_per_m`: Pressure on cable in bends

### 2. Summary CSV (`summary.csv`)
One row per leg with:
- `LegLabel`: AB, BC, CD, etc.
- `StartPoint`, `EndPoint`: Leg endpoints
- `TotalLength_m`: Total leg length
- `MaxTensionForward_N`, `MaxTensionReverse_N`: Peak tensions
- `MaxSidewallPressure_N_per_m`: Peak sidewall pressure
- `OptimalDirection`: Recommended pulling direction

### 3. Overlay Image (`overlay.png`)
- Each leg shown in a different color
- Grid for scale reference
- Legend identifying each leg

## Cable Arrangements and Their Effects

### Single Cable Installation
- **Friction**: Base coefficient applies directly
- **Weight**: `cable_mass × 1 × gravity`
- **Contact**: Single cable contacts conduit at bottom
- **Typical Use**: Single-phase circuits, control cables

### Trefoil Configuration (Three Cables)
Three cables arranged in triangular formation:

#### Physical Effects
- **Effective Friction**: 20-40% higher than single cable
  ```
  f_effective = f_base × (1.2 to 1.4)
  ```
- **Bundle Diameter**: ~2.15× single cable diameter
- **Weight Distribution**: More complex contact pattern
- **Sidewall Contact**: Cables contact each other and conduit walls


### Flat Configuration (Multiple Cables Side-by-Side)
- **Friction**: Similar to single cable per unit width
- **Contact Area**: Increased linearly with cable count
- **Bending**: Less flexible than trefoil, higher sidewall pressures

### Vertical Arrangements
- **Additional Weight Component**: Gravity effects in vertical runs
- **Modified Equations**: Weight factor becomes `cos(θ)` for horizontal component

## Typical Friction Coefficients

### Base Conduit Friction
| Conduit Type | Condition | Single Cable | Trefoil Bundle |
|--------------|-----------|--------------|----------------|
| PVC | Dry | 0.35 - 0.50 | 0.42 - 0.70 |
| PVC | Lubricated | 0.15 - 0.25 | 0.18 - 0.35 |
| Steel | Dry | 0.40 - 0.60 | 0.48 - 0.84 |
| Steel | Lubricated | 0.20 - 0.30 | 0.24 - 0.42 |
| Concrete | Dry | 0.50 - 0.70 | 0.60 - 0.98 |
| Concrete | Lubricated | 0.25 - 0.35 | 0.30 - 0.49 |

### Cable-to-Cable Friction (Internal Bundle)
| Cable Type | Dry | Lubricated |
|------------|-----|------------|
| XLPE | 0.40 - 0.60 | 0.15 - 0.25 |
| EPR | 0.35 - 0.55 | 0.12 - 0.22 |
| PVC | 0.45 - 0.65 | 0.18 - 0.28 |

## Technical Notes

- Calculations assume uniform cable weight distribution
- Drum weight adds to initial tension only
- Vertical sections not explicitly handled (use weight correction factors if needed)
- Compound bends (3D) simplified to 2D projections
- Temperature effects on cable properties not included

## References

### Industry Standards
- **IEEE 576-2022**: IEEE Recommended Practice for Installation Design and Installation of Cable Systems in Substations
- **AEIC CS8-2019**: Specification for Extruded Insulation Power Cables Rated Above 46 Through 345 kV
- **NECA 1-2006**: Standard Practices for Good Workmanship in Electrical Construction
- **IEC 60287**: Electric cables - Calculation of the current rating

### Technical Literature
- **Southwire Company** (2018). *Power Cable Installation Guide*. Carrollton, GA
- **Polywater Corporation** (2021). *Cable Pulling Compound Technical Bulletins*. Stillwater, MN
- **General Cable** (2019). *Installation Manual for Underground Distribution Cables*
- **Prysmian Group** (2020). *Cable Installation Guidelines and Best Practices*

### Academic Sources
- **Morinaga, H. et al.** (2015). "Analysis of Cable Bundle Friction in Underground Conduits." *IEEE Transactions on Power Delivery*, 30(4), 1842-1849
- **Zhang, L. & Williams, R.** (2018). "Experimental Study of Trefoil Cable Arrangement Effects on Pulling Forces." *Electric Power Systems Research*, 165, 234-241
- **Kumar, S. et al.** (2020). "Capstan Equation Applications in Power Cable Installation." *International Journal of Electrical Engineering*, 12(3), 145-158

### Manufacturer Resources
- **Nexans** (2022). *Underground Cable Installation Manual*, Document No. UC-22-001
- **Okonite Company** (2021). *Cable Pulling Guidelines for High Voltage Applications*
- **Encore Wire Corporation** (2019). *Installation Best Practices for Building Wire*

### Friction Coefficient Research
- **Blackburn, T.R.** (2016). "Friction Characteristics of Cable Insulation Materials in Conduit Systems." *Cable Technology International*, Issue 47, pp. 23-28
- **Anderson, P.M. & Johnson, K.L.** (2017). "Lubrication Effects on Multi-Cable Pulling Operations." *Electrical Construction & Maintenance*, 116(8), 34-39
