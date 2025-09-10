"""Easy Cable Pulling - Cable pulling calculations and route analysis."""

__version__ = "0.1.0"
__author__ = "Your Organization"
__email__ = "contact@example.com"

# Core functionality
from .core.cable_analysis_pipeline import (
    AnalysisConfig,
    AnalysisResults,
    CableAnalysisPipeline,
    SectionResult,
    analyze_cable_route,
)

# Core models
from .core.models import Bend, CableSpec, DuctSpec, Route, Section, Straight

# Main convenience function
__all__ = [
    "analyze_cable_route",
    "CableAnalysisPipeline",
    "AnalysisConfig",
    "AnalysisResults",
    "SectionResult",
    "Route",
    "Section",
    "CableSpec",
    "DuctSpec",
]
