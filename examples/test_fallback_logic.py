#!/usr/bin/env python3
"""Test the fallback logic after polynomial rejection."""

import math
import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from easycablepulling.core.models import DuctSpec
from easycablepulling.geometry.cleaner import PolylineCleaner
from easycablepulling.geometry.fitter import GeometryFitter
from easycablepulling.io import load_route_from_dxf


class ExtendedDebugFitter(GeometryFitter):
    """Extended debug version to trace fallback logic."""

    def _recursive_fit(self, points, start_idx, end_idx, depth=0):
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
                from easycablepulling.core.models import Straight

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

        print(f"{indent}🧮 Testing polynomial fitting...")

        # Skip polynomial details, focus on fallback
        if len(segment_points) >= 4:
            poly_fit = self._fit_polynomial(segment_points, degree=3)
            if poly_fit:
                if (
                    poly_fit["min_radius"] >= self.natural_bend_threshold * 0.5
                    and poly_fit["max_error"] <= 0.5
                ):
                    print(f"{indent}✅ Polynomial accepted (not shown for brevity)")
                    # ... polynomial logic would go here
                else:
                    print(
                        f"{indent}❌ Polynomial rejected: error={poly_fit['max_error']:.3f}m"
                    )
            else:
                print(f"{indent}❌ Polynomial fitting failed")
        else:
            print(f"{indent}❌ Not enough points for polynomial")

        print(f"{indent}🏹 Testing arc fitting...")
        arc_fit = self._fit_arc(segment_points)
        if arc_fit:
            print(
                f"{indent}  Arc: R={arc_fit['radius']:.1f}m, angle={arc_fit['angle']:.1f}°, error={arc_fit['max_error']:.3f}m"
            )

            # Adaptive tolerance
            if arc_fit["radius"] >= self.natural_bend_threshold:
                arc_tolerance = self.arc_tolerance
                print(f"{indent}  Using natural curve tolerance: {arc_tolerance}m")
            else:
                arc_tolerance = 0.3
                print(f"{indent}  Using tight curve tolerance: {arc_tolerance}m")

            print(
                f"{indent}  Error check: {arc_fit['max_error']:.3f} <= {arc_tolerance}"
            )
            if arc_fit["max_error"] <= arc_tolerance:
                print(
                    f"{indent}  Angle check: |{arc_fit['angle']:.1f}| >= {self.min_arc_angle}"
                )
                if abs(arc_fit["angle"]) >= self.min_arc_angle:
                    print(f"{indent}✅ Arc accepted - returning bend")
                    from easycablepulling.core.models import Bend

                    return [
                        Bend(
                            radius_m=arc_fit["radius"],
                            angle_deg=arc_fit["angle"],
                            direction="CW" if arc_fit["angle"] > 0 else "CCW",
                            center_point=arc_fit["center"],
                            bend_type="natural"
                            if arc_fit["radius"] >= self.natural_bend_threshold
                            else "manufactured",
                        )
                    ]
                else:
                    print(f"{indent}❌ Arc angle too small")
            else:
                print(f"{indent}❌ Arc error too high")
        else:
            print(f"{indent}❌ Arc fitting failed")

        print(f"{indent}📏 Testing straight line...")
        line_fit = self._fit_straight_line(segment_points)
        pavement_tolerance = min(0.3, self.straight_tolerance)

        if line_fit:
            print(
                f"{indent}  Line: length={line_fit['length']:.1f}m, error={line_fit['max_error']:.3f}m"
            )
            print(
                f"{indent}  Error check: {line_fit['max_error']:.3f} <= {pavement_tolerance}"
            )
            print(
                f"{indent}  Length check: {line_fit['length']:.3f} >= {self.min_straight_length}"
            )

            if line_fit["max_error"] <= pavement_tolerance:
                if line_fit["length"] >= self.min_straight_length:
                    print(f"{indent}✅ Straight accepted - returning straight")
                    from easycablepulling.core.models import Straight

                    return [
                        Straight(
                            length_m=line_fit["length"],
                            start_point=segment_points[0],
                            end_point=segment_points[-1],
                        )
                    ]
                else:
                    print(f"{indent}❌ Straight too short")
            else:
                print(f"{indent}❌ Straight error too high")
        else:
            print(f"{indent}❌ Straight fitting failed")

        # Determine split point
        split_logic = "Unknown"
        if arc_fit and line_fit:
            split_idx = start_idx + arc_fit["max_error_idx"]
            split_logic = f"arc error point (idx {arc_fit['max_error_idx']})"
        elif arc_fit:
            split_idx = start_idx + arc_fit["max_error_idx"]
            split_logic = f"arc error point (idx {arc_fit['max_error_idx']})"
        elif line_fit:
            split_idx = start_idx + line_fit["max_error_idx"]
            split_logic = f"line error point (idx {line_fit['max_error_idx']})"
        else:
            split_idx = start_idx + (end_idx - start_idx) // 3
            split_logic = "1/3 point (no fits available)"

        print(f"{indent}✂️ Splitting at {split_logic}: idx {split_idx}")

        # Recursively fit sub-segments
        print(f"{indent}⬅️ Fitting left segment...")
        left_primitives = self._recursive_fit(points, start_idx, split_idx, depth + 1)
        print(f"{indent}⬅️ Left returned {len(left_primitives)} primitives")

        print(f"{indent}➡️ Fitting right segment...")
        right_primitives = self._recursive_fit(points, split_idx, end_idx, depth + 1)
        print(f"{indent}➡️ Right returned {len(right_primitives)} primitives")

        result = left_primitives + right_primitives
        print(f"{indent}🎯 Total: {len(result)} primitives")
        return result


def test_fallback_logic():
    """Test fallback logic in detail."""
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

    fitter = ExtendedDebugFitter(duct_spec=duct_spec)

    original_points = sect07.original_polyline
    cleaner = PolylineCleaner()
    cleaned_points = cleaner.clean_polyline(original_points)

    print(f"SECT_07 Fallback Logic Test:")
    print(f"Total cleaned points: {len(cleaned_points)}")
    print(f"Natural bend threshold: {fitter.natural_bend_threshold:.1f}m")
    print(f"Arc tolerance: {fitter.arc_tolerance:.1f}m")
    print(f"Straight tolerance: {fitter.straight_tolerance:.1f}m")
    print(f"Min straight length: {fitter.min_straight_length:.1f}m")
    print(f"Min arc angle: {fitter.min_arc_angle}°")
    print()

    # Test on smaller segment first to see the pattern
    test_points = cleaned_points[:15]
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
    test_fallback_logic()
