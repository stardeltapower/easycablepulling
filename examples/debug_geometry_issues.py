#!/usr/bin/env python3
"""Debug geometry fitting issues - analyze length discrepancies."""

import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from easycablepulling.core.models import (
    CableArrangement,
    CableSpec,
    DuctSpec,
    PullingMethod,
)
from easycablepulling.geometry.processor import GeometryProcessor
from easycablepulling.io import load_route_from_dxf


def debug_section_fitting(section, processor, duct_spec):
    """Debug fitting for a single section."""
    print(f"\n=== DEBUGGING {section.id} ===")
    print(f"Original polyline: {len(section.original_polyline)} points")
    print(f"Original length: {section.original_length:.2f}m")

    # Clean polyline
    cleaned_points = processor.cleaner.clean_polyline(section.original_polyline)
    print(f"Cleaned polyline: {len(cleaned_points)} points")

    # Calculate cleaned length manually
    cleaned_length = 0.0
    for i in range(len(cleaned_points) - 1):
        p1, p2 = cleaned_points[i], cleaned_points[i + 1]
        seg_len = ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5
        cleaned_length += seg_len
    print(f"Cleaned length: {cleaned_length:.2f}m")

    # Create fitter with duct spec
    from easycablepulling.geometry.fitter import GeometryFitter

    fitter = GeometryFitter(duct_spec=duct_spec)

    # Fit geometry
    fitting_result = fitter.fit_polyline(cleaned_points)
    print(f"Fitting success: {fitting_result.success}")
    print(f"Fitting message: {fitting_result.message}")
    print(f"Fitted primitives: {len(fitting_result.primitives)}")

    # Analyze each primitive
    fitted_total = 0.0
    for i, primitive in enumerate(fitting_result.primitives):
        if hasattr(primitive, "length_m"):
            length = primitive.length_m
            ptype = "straight"
        elif hasattr(primitive, "radius_m"):
            import math

            arc_length = abs(math.radians(primitive.angle_deg)) * primitive.radius_m
            length = arc_length
            ptype = getattr(primitive, "bend_type", "bend")
        else:
            length = 0.0
            ptype = "unknown"

        fitted_total += length
        print(f"  Primitive {i}: {ptype} = {length:.2f}m")

    print(f"Total fitted length: {fitted_total:.2f}m")
    print(
        f"Length error: {abs(fitted_total - section.original_length) / section.original_length * 100:.2f}%"
    )

    return fitting_result


def main():
    """Debug geometry fitting issues."""
    examples_dir = Path(__file__).parent
    input_dxf = examples_dir / "input.dxf"

    print(f"Loading route from {input_dxf}")
    route = load_route_from_dxf(input_dxf, "33kV Cable Route")

    # Create duct specification
    duct_spec = DuctSpec(
        inner_diameter=200.0,  # mm
        type="HDPE",
        friction_dry=0.35,
        friction_lubricated=0.15,
        bend_options=[],
    )

    print(f"Natural bend threshold: {(duct_spec.inner_diameter/1000) * 22:.1f}m")

    # Create processor
    processor = GeometryProcessor()

    # Debug problematic sections
    problem_sections = []
    for section in route.sections:
        if section.original_length > 0:  # Only process sections with actual length
            result = debug_section_fitting(section, processor, duct_spec)
            if not result.success or len(result.primitives) == 0:
                problem_sections.append(section.id)

    print(f"\n=== SUMMARY ===")
    print(f"Problem sections: {problem_sections}")
    print(
        f"Total sections: {len([s for s in route.sections if s.original_length > 0])}"
    )


if __name__ == "__main__":
    main()
