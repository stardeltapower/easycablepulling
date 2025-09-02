"""Main geometry processor that integrates cleaning, fitting, and validation."""

from dataclasses import dataclass
from typing import List, Optional

from ..core.models import CableSpec, DuctSpec, Route, Section
from .cleaner import PolylineCleaner
from .fitter import FittingResult, GeometryFitter
from .validator import GeometryValidator, ValidationResult


@dataclass
class ProcessingResult:
    """Result of complete geometry processing."""

    route: Route
    fitting_results: List[FittingResult]
    validation_result: ValidationResult
    success: bool
    message: str = ""


class GeometryProcessor:
    """Main processor for cable route geometry."""

    def __init__(
        self,
        cleaner: Optional[PolylineCleaner] = None,
        fitter: Optional[GeometryFitter] = None,
        validator: Optional[GeometryValidator] = None,
    ):
        """Initialize geometry processor.

        Args:
            cleaner: Optional custom polyline cleaner
            fitter: Optional custom geometry fitter
            validator: Optional custom geometry validator
        """
        self.cleaner = cleaner or PolylineCleaner()
        self.fitter = fitter or GeometryFitter()
        self.validator = validator or GeometryValidator()

    def process_route(
        self,
        route: Route,
        cable_spec: Optional[CableSpec] = None,
        duct_spec: Optional[DuctSpec] = None,
    ) -> ProcessingResult:
        """Process complete route geometry.

        Args:
            route: Route to process
            cable_spec: Optional cable specification for validation
            duct_spec: Optional duct specification for diameter-based bend classification

        Returns:
            ProcessingResult with fitted primitives and validation
        """
        fitting_results = []
        processed_sections = []

        # Create fitter with duct specification if provided
        fitter = self.fitter
        if duct_spec:
            fitter = GeometryFitter(
                straight_tolerance=self.fitter.straight_tolerance,
                arc_tolerance=self.fitter.arc_tolerance,
                min_straight_length=self.fitter.min_straight_length,
                min_arc_angle=self.fitter.min_arc_angle,
                duct_spec=duct_spec,
            )

        # Process each section
        for section in route.sections:
            # Clean polyline
            cleaned_points = self.cleaner.clean_polyline(section.original_polyline)

            # Fit geometry
            fitting_result = fitter.fit_polyline(cleaned_points)
            fitting_results.append(fitting_result)

            # Update section with fitted primitives
            section.primitives = fitting_result.primitives
            processed_sections.append(section)

        # Create processed route
        processed_route = Route(
            name=route.name, sections=processed_sections, metadata=route.metadata
        )

        # Validate geometry
        validation_result = self.validator.validate_route(processed_route, cable_spec)

        # Check overall success
        success = (
            all(fr.success for fr in fitting_results) and validation_result.is_valid
        )

        # Generate summary message
        total_primitives = sum(len(fr.primitives) for fr in fitting_results)
        message = f"Processed {route.section_count} sections, fitted {total_primitives} primitives"

        if not success:
            message += f". {validation_result.total_errors} errors, {validation_result.total_warnings} warnings"

        return ProcessingResult(
            route=processed_route,
            fitting_results=fitting_results,
            validation_result=validation_result,
            success=success,
            message=message,
        )

    def process_section(
        self, section: Section, cable_spec: Optional[CableSpec] = None
    ) -> FittingResult:
        """Process a single section.

        Args:
            section: Section to process
            cable_spec: Optional cable specification

        Returns:
            FittingResult for the section
        """
        # Clean polyline
        cleaned_points = self.cleaner.clean_polyline(section.original_polyline)

        # Fit geometry
        fitting_result = self.fitter.fit_polyline(cleaned_points)

        # Update section
        section.primitives = fitting_result.primitives

        return fitting_result
