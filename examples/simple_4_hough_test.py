#!/usr/bin/env python3
"""Simple test of Hough methodology without full imports."""

import sys
from pathlib import Path
import numpy as np
from typing import List, Tuple

# Add parent to path and import directly to avoid module issues
sys.path.append(str(Path(__file__).parent.parent / "easycablepulling"))

# Direct imports
from geometry.hough_clustering_fitter import HoughClusteringFitter
from core.models import Section


def create_simple_test_route() -> List[Tuple[float, float]]:
    """Create a simple test route."""
    points: List[Tuple[float, float]] = []

    # Straight line east
    for x in range(0, 101, 10):
        points.append((float(x), 0.0))

    # 90 degree turn
    points.extend([(100.0, 5.0), (100.0, 10.0), (105.0, 10.0), (110.0, 10.0)])

    # Straight line north
    for y in range(10, 61, 10):
        points.append((110.0, float(y)))

    return points


def main():
    """Test the Hough methodology directly."""
    print("Simple Hough + Turn Clustering Test")
    print("=" * 40)

    # Create test route
    points = create_simple_test_route()
    print(f"Test route with {len(points)} points")

    # Calculate route length
    total_length = sum(
        np.sqrt(
            (points[i + 1][0] - points[i][0]) ** 2
            + (points[i + 1][1] - points[i][1]) ** 2
        )
        for i in range(len(points) - 1)
    )
    print(f"Total length: {total_length:.2f}m")

    # Create section
    section = Section(
        id="test_section", original_polyline=points, original_length=total_length
    )

    # Create fitter directly
    print("\nCreating Hough fitter...")
    fitter = HoughClusteringFitter(
        duct_type="200mm",
        hough_threshold=5,
        corner_zone_radius=6.0,
        min_straight_length=5.0,
    )

    print(f"Methodology: {fitter.get_methodology_name()}")

    # Test fitting
    print("\nTesting fit_section...")
    try:
        primitives = fitter.fit_section(section)
        print(f"Generated {len(primitives)} primitives")

        # Print primitives
        for i, prim in enumerate(primitives):
            prim_type = type(prim).__name__
            if hasattr(prim, "length_m"):
                print(f"  {i+1}: {prim_type} - {prim.length_m:.2f}m")
            elif hasattr(prim, "angle_deg"):
                print(
                    f"  {i+1}: {prim_type} - {prim.angle_deg:.1f}° at R={prim.radius_m:.1f}m"
                )
            else:
                print(f"  {i+1}: {prim_type}")

        print("\n✓ Test successful!")

    except Exception as e:
        print(f"ERROR during fitting: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
