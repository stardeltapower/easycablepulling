"""Configuration settings for easycablepulling."""

from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Default paths
DEFAULT_INPUT_DXF = PROJECT_ROOT / "examples" / "input.dxf"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output"

# Geometry tolerances
DEFAULT_LATERAL_TOLERANCE = 0.15  # meters (150mm)
DEFAULT_LENGTH_TOLERANCE = 0.002  # 0.2% length error

# Cable pulling defaults
DEFAULT_MAX_CABLE_LENGTH = 500.0  # meters

# Friction coefficients for different conduit types and conditions
FRICTION_COEFFICIENTS = {
    "PVC": {
        "dry": {"single": 0.35, "trefoil": 0.42},
        "lubricated": {"single": 0.15, "trefoil": 0.18},
    },
    "HDPE": {  # Similar to PVC
        "dry": {"single": 0.35, "trefoil": 0.42},
        "lubricated": {"single": 0.15, "trefoil": 0.18},
    },
    "Steel": {
        "dry": {"single": 0.40, "trefoil": 0.48},
        "lubricated": {"single": 0.20, "trefoil": 0.24},
    },
    "Concrete": {
        "dry": {"single": 0.50, "trefoil": 0.60},
        "lubricated": {"single": 0.25, "trefoil": 0.30},
    },
}

# Default friction values
DEFAULT_FRICTION_DRY = 0.35  # PVC dry single cable
DEFAULT_FRICTION_LUBRICATED = 0.15  # PVC lubricated single cable

# Standard duct bend options (radius in mm, angle in degrees)
STANDARD_DUCT_BENDS = [
    {"radius": 600, "angle": 90},
    {"radius": 900, "angle": 90},
    {"radius": 1200, "angle": 90},
    {"radius": 1500, "angle": 90},
    {"radius": 1200, "angle": 45},
    {"radius": 1500, "angle": 45},
    {"radius": 2000, "angle": 30},
]
