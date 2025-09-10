"""Geometry processing orchestrator for cable route analysis."""

from dataclasses import dataclass
from typing import Optional

from ..core.models import CableSpec, DuctSpec, Route
from .simple_segment_fitter import SimpleSegmentFitter
from .splitter import RouteSplitter
from .validator import GeometryValidator, ValidationResult


@dataclass
class ProcessingResult:
    """Result of geometry processing."""

    success: bool
    message: str
    route: Route
    validation_result: Optional[ValidationResult] = None


class GeometryProcessor:
    """Orchestrates geometry processing including fitting and splitting."""

    def __init__(
        self,
        fitter: Optional[SimpleSegmentFitter] = None,
        splitter: Optional[RouteSplitter] = None,
        validator: Optional[GeometryValidator] = None,
    ) -> None:
        """Initialize geometry processor.

        Args:
            fitter: Geometry fitter for converting polylines to primitives
            splitter: Route splitter for managing long sections
            validator: Route validator for checking constraints
        """
        self.fitter = fitter or SimpleSegmentFitter()
        self.splitter = splitter or RouteSplitter()
        self.validator = validator or GeometryValidator()

    def process_route(
        self,
        route: Route,
        cable_spec: CableSpec,
        duct_spec: DuctSpec,
        enable_splitting: bool = True,
        max_cable_length: float = 500.0,
    ) -> ProcessingResult:
        """Process route geometry including fitting and optional splitting.

        Args:
            route: Input route to process
            cable_spec: Cable specifications for validation
            duct_spec: Duct specifications for fitting
            enable_splitting: Whether to enable route splitting
            max_cable_length: Maximum cable length for splitting

        Returns:
            ProcessingResult with processed route
        """
        try:
            processed_route = Route(name=route.name, metadata=route.metadata)

            # Process each section
            for section in route.sections:
                # Fit geometry to primitives
                fit_result = self.fitter.fit_section_to_primitives(section)

                # Update section with fitted primitives
                section.primitives = fit_result.primitives
                processed_route.add_section(section)

            # Apply splitting if enabled
            if enable_splitting:
                # Configure splitter with max cable length
                self.splitter.max_cable_length = max_cable_length
                splitting_result = self.splitter.split_route(processed_route)
                if splitting_result.success:
                    processed_route = splitting_result.split_route
                else:
                    return ProcessingResult(
                        success=False,
                        message=f"Splitting failed: {splitting_result.message}",
                        route=processed_route,
                    )

            # Validate the processed route
            validation_result = self.validator.validate_route(
                processed_route, cable_spec
            )

            return ProcessingResult(
                success=True,
                message=f"Successfully processed route with {len(processed_route.sections)} sections",
                route=processed_route,
                validation_result=validation_result,
            )

        except Exception as e:
            return ProcessingResult(
                success=False,
                message=f"Geometry processing failed: {str(e)}",
                route=route,
            )
