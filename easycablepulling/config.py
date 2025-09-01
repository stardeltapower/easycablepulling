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
DEFAULT_FRICTION_DRY = 0.35
DEFAULT_FRICTION_LUBRICATED = 0.15

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