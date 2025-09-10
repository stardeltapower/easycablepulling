"""Geometry processing for cable route fitting."""

from .processor import GeometryProcessor, ProcessingResult
from .simple_segment_fitter import SimpleSegmentFitter
from .splitter import RouteSplitter

__all__ = [
    "GeometryProcessor",
    "ProcessingResult",
    "SimpleSegmentFitter",
    "RouteSplitter",
]
