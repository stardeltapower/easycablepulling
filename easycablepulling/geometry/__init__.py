"""Geometry processing for cable route fitting."""

from .cleaner import PolylineCleaner
from .fitter import FittingResult, GeometryFitter
from .processor import GeometryProcessor, ProcessingResult
from .splitter import RouteSplitter, SplitPoint, SplittingResult
from .validator import GeometryValidator, ValidationResult

__all__ = [
    "PolylineCleaner",
    "GeometryFitter",
    "FittingResult",
    "GeometryValidator",
    "ValidationResult",
    "GeometryProcessor",
    "ProcessingResult",
    "RouteSplitter",
    "SplitPoint",
    "SplittingResult",
]
