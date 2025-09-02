"""Main geometry processor that integrates cleaning, fitting, and validation."""

from dataclasses import dataclass
from typing import List, Optional

from ..core.models import CableSpec, DuctSpec, Route, Section
from .cleaner import PolylineCleaner
from .fitter import FittingResult, GeometryFitter
from .splitter import RouteSplitter, SplittingResult
from .validator import GeometryValidator, ValidationResult


@dataclass
class ProcessingResult:
    """Result of complete geometry processing."""

    route: Route
    fitting_results: List[FittingResult]
    validation_result: ValidationResult
    splitting_result: Optional[SplittingResult]
    success: bool
    message: str = ""


class GeometryProcessor:
    """Main processor for cable route geometry."""

    def __init__(
        self,
        cleaner: Optional[PolylineCleaner] = None,
        fitter: Optional[GeometryFitter] = None,
        validator: Optional[GeometryValidator] = None,
        splitter: Optional[RouteSplitter] = None,
    ):
        """Initialize geometry processor.

        Args:
            cleaner: Optional custom polyline cleaner
            fitter: Optional custom geometry fitter
            validator: Optional custom geometry validator
            splitter: Optional custom route splitter
        """
        self.cleaner = cleaner or PolylineCleaner()
        self.fitter = fitter or GeometryFitter()
        self.validator = validator or GeometryValidator()
        self.splitter = splitter or RouteSplitter()

    def process_route(
        self,
        route: Route,
        cable_spec: Optional[CableSpec] = None,
        duct_spec: Optional[DuctSpec] = None,
        enable_splitting: bool = True,
        max_cable_length: Optional[float] = None,
    ) -> ProcessingResult:
        """Process complete route geometry.

        Args:
            route: Route to process
            cable_spec: Optional cable specification for validation
            duct_spec: Optional duct specification for diameter-based bend classification
            enable_splitting: Whether to perform minor splitting for long sections
            max_cable_length: Maximum cable length for splitting (overrides splitter default)

        Returns:
            ProcessingResult with fitted primitives and validation
        """
        splitting_result = None
        working_route = route

        # Step 1: Apply minor splitting if enabled
        if enable_splitting:
            # Update splitter max length if provided
            if max_cable_length:
                self.splitter.max_cable_length = max_cable_length

            # Check if any sections need splitting
            sections_needing_split = [
                section
                for section in route.sections
                if self.splitter.needs_splitting(section)
            ]

            if sections_needing_split:
                splitting_result = self.splitter.split_route(route)
                working_route = splitting_result.split_route

        # Step 2: Process geometry (cleaning and fitting)
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
        for section in working_route.sections:
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
            name=working_route.name,
            sections=processed_sections,
            metadata=working_route.metadata,
        )

        # Step 3: Validate geometry
        validation_result = self.validator.validate_route(processed_route, cable_spec)

        # Check overall success
        success = (
            all(fr.success for fr in fitting_results) and validation_result.is_valid
        )

        # Generate summary message
        total_primitives = sum(len(fr.primitives) for fr in fitting_results)
        split_info = (
            f", {splitting_result.sections_created} sections added by splitting"
            if splitting_result
            else ""
        )
        message = f"Processed {working_route.section_count} sections, fitted {total_primitives} primitives{split_info}"

        if not success:
            message += f". {validation_result.total_errors} errors, {validation_result.total_warnings} warnings"

        return ProcessingResult(
            route=processed_route,
            fitting_results=fitting_results,
            validation_result=validation_result,
            splitting_result=splitting_result,
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

    def split_route(
        self,
        route: Route,
        max_cable_length: Optional[float] = None,
    ) -> SplittingResult:
        """Split route sections that exceed maximum cable length.

        Args:
            route: Route to split
            max_cable_length: Maximum cable length (overrides splitter default)

        Returns:
            SplittingResult with split route information
        """
        # Update splitter max length if provided
        if max_cable_length:
            self.splitter.max_cable_length = max_cable_length

        return self.splitter.split_route(route)
