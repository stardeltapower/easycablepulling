"""Base interface for route fitting methodologies."""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

from ..core.models import Primitive, Section


class BaseFitter(ABC):
    """Abstract base class for route fitting methodologies."""

    def __init__(self, **kwargs):
        """Initialize fitter with methodology-specific parameters."""
        self.name = self.__class__.__name__
        self.methodology = "unknown"

    @abstractmethod
    def fit_section(
        self, section: Section, points: Optional[List[Tuple[float, float]]] = None
    ) -> List[Primitive]:
        """Fit a section to primitives using the specific methodology.

        Args:
            section: Section to fit
            points: Optional polyline points (uses section.original_polyline if None)

        Returns:
            List of fitted primitives (Straight and Bend objects)
        """
        pass

    @abstractmethod
    def get_methodology_name(self) -> str:
        """Get human-readable name of the fitting methodology."""
        pass

    @abstractmethod
    def get_methodology_code(self) -> str:
        """Get code identifier for the methodology (e.g., '0_best_fit')."""
        pass
