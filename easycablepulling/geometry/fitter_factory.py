"""Factory for creating route fitters based on methodology selection."""

from typing import Dict, List, Optional, Type

from .base_fitter import BaseFitter
from .best_fit_fitter import BestFitFitter
from .curvature_first_fitter import CurvatureFirstFitter
from .curvature_first_inventory_fitter import CurvatureFirstInventoryFitter
from .polynomial_fitter import PolynomialFitter
from .corridor_constrained_fitter import CorridorConstrainedFitter
from .hough_clustering_fitter import HoughClusteringFitter
from ..core.models import DuctSpec


# Registry of available fitting methodologies
FITTER_REGISTRY: Dict[str, Type[BaseFitter]] = {
    "0_best_fit": BestFitFitter,
    "1_curvature_first": CurvatureFirstFitter,
    "1_1_curvature_first": CurvatureFirstInventoryFitter,
    "2_polynomial_detection": PolynomialFitter,
    "3_corridor_constrained": CorridorConstrainedFitter,
    "4_hough_clustering": HoughClusteringFitter,
    # Future methodologies can be added here:
    # "2_angle_quantised_dp": AngleQuantisedDPFitter,
    # "3_ransac_robust": RANSACRobustFitter,
    # "5_lattice_search": LatticeSearchFitter,
}


class FitterFactory:
    """Factory for creating route fitters."""

    @staticmethod
    def create_fitter(methodology: str = "1_curvature_first", **kwargs) -> BaseFitter:
        """Create a fitter instance based on methodology.

        Args:
            methodology: Fitting methodology code (e.g., "0_best_fit")
            **kwargs: Additional parameters passed to fitter constructor

        Returns:
            Fitter instance

        Raises:
            ValueError: If methodology is not recognized
        """
        if methodology not in FITTER_REGISTRY:
            available = ", ".join(FITTER_REGISTRY.keys())
            raise ValueError(
                f"Unknown fitting methodology: {methodology}. "
                f"Available options: {available}"
            )

        fitter_class = FITTER_REGISTRY[methodology]
        return fitter_class(**kwargs)

    @staticmethod
    def get_available_methodologies() -> List[str]:
        """Get list of available methodology codes."""
        return list(FITTER_REGISTRY.keys())

    @staticmethod
    def get_methodology_info(methodology: str) -> Dict[str, str]:
        """Get information about a specific methodology.

        Args:
            methodology: Methodology code

        Returns:
            Dictionary with methodology information

        Raises:
            ValueError: If methodology is not recognized
        """
        if methodology not in FITTER_REGISTRY:
            raise ValueError(f"Unknown methodology: {methodology}")

        fitter_class = FITTER_REGISTRY[methodology]
        # Create temporary instance to get info
        temp_instance = fitter_class()

        return {
            "code": temp_instance.get_methodology_code(),
            "name": temp_instance.get_methodology_name(),
            "class": fitter_class.__name__,
        }

    @staticmethod
    def register_fitter(code: str, fitter_class: Type[BaseFitter]) -> None:
        """Register a new fitter methodology.

        Args:
            code: Methodology code (e.g., "1_inventory_constrained")
            fitter_class: Fitter class implementing BaseFitter
        """
        FITTER_REGISTRY[code] = fitter_class
