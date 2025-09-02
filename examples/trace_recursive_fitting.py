#!/usr/bin/env python3
"""Trace the exact execution path in recursive fitting."""

import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from easycablepulling.core.models import DuctSpec
from easycablepulling.geometry.cleaner import PolylineCleaner
from easycablepulling.geometry.fitter import GeometryFitter
from easycablepulling.io import load_route_from_dxf


class DebugGeometryFitter(GeometryFitter):
    """Debug version of GeometryFitter with tracing."""

    def _recursive_fit(self, points, start_idx, end_idx, depth=0):
        """Debug version with detailed tracing."""
        indent = "  " * depth
        segment_points = points[start_idx : end_idx + 1]

        print(
            f"{indent}🔍 _recursive_fit: depth={depth}, points={len(segment_points)}, indices={start_idx}-{end_idx}"
        )

        # Prevent infinite recursion
        if depth > 50 or end_idx - start_idx < 1:
            print(f"{indent}❌ Stopping: depth={depth}, span={end_idx - start_idx}")
            return []

        # Minimum segment length check
        if end_idx - start_idx < 2:
            print(f"{indent}📏 Small segment check")
            segment_points = points[start_idx : end_idx + 1]
            if len(segment_points) >= 2:
                # Create a simple straight line
                p1, p2 = segment_points[0], segment_points[-1]
                length = math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
                print(
                    f"{indent}   Length: {length:.3f}m, min_required: {self.min_straight_length}m"
                )
                if length >= self.min_straight_length:
                    print(f"{indent}✅ Returning straight segment")
                    return [Straight(length_m=length, start_point=p1, end_point=p2)]
            print(f"{indent}❌ No valid straight segment")
            return []

        print(
            f"{indent}🧮 Testing polynomial fitting (need >=4 points, have {len(segment_points)})"
        )

        # PRIORITIZE POLYNOMIAL FITTING for accurate path following
        if len(segment_points) >= 4:
            print(f"{indent}  Trying polynomial fitting...")
            poly_fit = self._fit_polynomial(segment_points, degree=3)
            if poly_fit:
                print(f"{indent}  ✅ Polynomial fit successful:")
                print(f"{indent}     Max error: {poly_fit['max_error']:.3f}m")
                print(f"{indent}     Min radius: {poly_fit['min_radius']:.1f}m")

                # Check if polynomial stays within curvature limits
                radius_threshold = self.natural_bend_threshold * 0.5
                print(
                    f"{indent}     Radius check: {poly_fit['min_radius']:.1f} >= {radius_threshold:.1f}"
                )

                if poly_fit["min_radius"] >= radius_threshold:
                    # Check fitting accuracy
                    print(
                        f"{indent}     Error check: {poly_fit['max_error']:.3f} <= 0.5"
                    )
                    if poly_fit["max_error"] <= 0.5:
                        print(
                            f"{indent}  ✅ Polynomial conditions met, checking for curve preservation..."
                        )

                        # Check if this is actually a gentle curve that should be preserved as a bend
                        arc_fit = self._fit_arc(segment_points)
                        if arc_fit:
                            print(
                                f"{indent}     Arc fit: R={arc_fit['radius']:.1f}m, angle={arc_fit['angle']:.1f}°"
                            )
                            print(f"{indent}     Min arc angle: {self.min_arc_angle}°")
                            print(
                                f"{indent}     Natural threshold: {self.natural_bend_threshold:.1f}m"
                            )

                            arc_valid = (
                                abs(arc_fit["angle"]) >= self.min_arc_angle
                                and arc_fit["radius"] >= self.natural_bend_threshold
                            )
                            print(f"{indent}     Arc valid: {arc_valid}")

                            if arc_valid:
                                print(f"{indent}🎉 Returning single natural bend")
                                from easycablepulling.core.models import Bend

                                return [
                                    Bend(
                                        radius_m=arc_fit["radius"],
                                        angle_deg=arc_fit["angle"],
                                        direction=(
                                            "CW" if arc_fit["angle"] > 0 else "CCW"
                                        ),
                                        center_point=arc_fit["center"],
                                        bend_type="natural",
                                    )
                                ]
                            else:
                                print(
                                    f"{indent}➡️ Decomposing polynomial to primitives..."
                                )
                                poly_primitives = (
                                    self._decompose_polynomial_to_primitives(poly_fit)
                                )
                                if poly_primitives and len(poly_primitives) >= 1:
                                    print(
                                        f"{indent}✅ Returning {len(poly_primitives)} polynomial primitives"
                                    )
                                    return poly_primitives
                                else:
                                    print(f"{indent}❌ Polynomial decomposition failed")
                        else:
                            print(
                                f"{indent}❌ Arc fitting failed after polynomial success"
                            )
                    else:
                        print(f"{indent}❌ Polynomial error too high")
                else:
                    print(f"{indent}❌ Polynomial radius too small")
            else:
                print(f"{indent}❌ Polynomial fitting failed")
        else:
            print(f"{indent}❌ Not enough points for polynomial")

        print(f"{indent}⬇️ Falling back to traditional arc/straight fitting...")

        # Continue with original logic...
        return super()._recursive_fit(points, start_idx, end_idx, depth)


def trace_recursive_fitting():
    """Trace recursive fitting execution."""
    examples_dir = Path(__file__).parent
    input_dxf = examples_dir / "input.dxf"

    route = load_route_from_dxf(input_dxf, "33kV Cable Route")

    # Find SECT_07
    sect07 = None
    for section in route.sections:
        if section.id == "SECT_07":
            sect07 = section
            break

    if not sect07:
        print("SECT_07 not found")
        return

    # Create duct specification
    duct_spec = DuctSpec(
        inner_diameter=200.0,  # mm
        type="HDPE",
        friction_dry=0.35,
        friction_lubricated=0.15,
        bend_options=[],
    )

    fitter = DebugGeometryFitter(duct_spec=duct_spec)

    original_points = sect07.original_polyline
    cleaner = PolylineCleaner()
    cleaned_points = cleaner.clean_polyline(original_points)

    print(f"SECT_07 Recursive Fitting Trace:")
    print(f"Total cleaned points: {len(cleaned_points)}")
    print(f"Natural bend threshold: {fitter.natural_bend_threshold:.1f}m")
    print(f"Min arc angle: {fitter.min_arc_angle}°")
    print()

    # Test on first 10 points to see the issue
    test_points = cleaned_points[:10]
    print(f"Testing on first {len(test_points)} points:")
    print()

    primitives = fitter._recursive_fit(test_points, 0, len(test_points) - 1, 0)

    print()
    print(f"Final result: {len(primitives)} primitives")
    for i, prim in enumerate(primitives):
        if hasattr(prim, "radius_m"):
            print(f"  {i+1}. Bend: R={prim.radius_m:.1f}m, {prim.angle_deg:.1f}°")
        else:
            print(f"  {i+1}. Straight: {prim.length_m:.1f}m")


if __name__ == "__main__":
    import math

    from easycablepulling.core.models import Straight

    trace_recursive_fitting()
