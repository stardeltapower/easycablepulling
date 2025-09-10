#!/usr/bin/env python3
"""Test script for Hough + Turn Clustering methodology."""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple

# Add parent to path
sys.path.append(str(Path(__file__).parent.parent))

from easycablepulling.geometry import FitterFactory
from easycablepulling.core.models import Section


def create_test_route() -> List[Tuple[float, float]]:
    """Create a test route with straights and corners."""
    points: List[Tuple[float, float]] = []

    # Start with a straight section going east
    for i in range(0, 101, 5):
        points.append((float(i), 0.0))

    # Turn 90 degrees north (corner zone)
    center = (100.0, 10.0)
    for angle in range(0, 91, 5):
        rad = np.radians(angle)
        x = 100.0 + 10.0 * np.sin(rad)
        y = 10.0 * (1 - np.cos(rad))
        points.append((x, y))

    # Straight section going north
    for i in range(10, 61, 5):
        points.append((110.0, float(i)))

    # Turn 45 degrees northeast
    center = (110.0, 60.0)
    for angle in range(0, 46, 5):
        rad = np.radians(angle)
        x = 110.0 + 10.0 * np.sin(rad)
        y = 60.0 + 10.0 * (1 - np.cos(rad))
        points.append((x, y))

    # Diagonal straight
    for i in range(0, 51, 5):
        x = 110.0 + 10.0 * np.sin(np.radians(45)) + i * np.cos(np.radians(45))
        y = 60.0 + 10.0 * (1 - np.cos(np.radians(45))) + i * np.sin(np.radians(45))
        points.append((x, y))

    # Small S-curve
    for angle in range(0, 46, 5):
        rad = np.radians(angle)
        x = points[-1][0] + 5.0 * np.sin(rad)
        y = points[-1][1] + 5.0 * (1 - np.cos(rad))
        points.append((x, y))

    for angle in range(0, 46, 5):
        rad = np.radians(angle)
        x = points[-1][0] + 5.0 * np.sin(rad)
        y = points[-1][1] - 5.0 * (1 - np.cos(rad))
        points.append((x, y))

    # Final straight
    last_x, last_y = points[-1]
    for i in range(0, 51, 5):
        points.append((last_x + float(i), last_y))

    return points


def plot_comparison(
    original_points: List[Tuple[float, float]],
    fitted_primitives: List,
    title: str = "Hough + Turn Clustering Result",
) -> None:
    """Plot comparison of original vs fitted route."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot original polyline
    ax1.set_title("Original Polyline")
    xs = [p[0] for p in original_points]
    ys = [p[1] for p in original_points]
    ax1.plot(xs, ys, "b-", linewidth=2, label="Original")
    ax1.plot(xs, ys, "b.", markersize=3)
    ax1.set_aspect("equal")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")

    # Plot fitted route
    ax2.set_title(title)
    ax2.plot(xs, ys, "b-", alpha=0.3, linewidth=1, label="Original")

    # Plot fitted primitives
    from easycablepulling.core.models import Straight, Bend

    current_pos = original_points[0] if original_points else (0, 0)
    current_heading = 0.0

    for i, prim in enumerate(fitted_primitives):
        if isinstance(prim, Straight):
            # Plot straight segment
            x1, y1 = prim.start_point
            x2, y2 = prim.end_point
            ax2.plot([x1, x2], [y1, y2], "r-", linewidth=2)
            ax2.plot([x1, x2], [y1, y2], "ro", markersize=4)

            # Add label
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            ax2.text(
                mid_x,
                mid_y,
                f"S{i+1}\n{prim.length_m:.1f}m",
                ha="center",
                va="center",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.5),
            )

        elif isinstance(prim, Bend):
            # Plot arc
            angles = np.linspace(
                np.radians(prim.start_angle_deg),
                np.radians(prim.start_angle_deg + prim.angle_deg),
                20,
            )
            arc_x = prim.center[0] + prim.radius_m * np.cos(angles)
            arc_y = prim.center[1] + prim.radius_m * np.sin(angles)
            ax2.plot(arc_x, arc_y, "g-", linewidth=2)

            # Add label
            mid_angle = np.radians(prim.start_angle_deg + prim.angle_deg / 2)
            label_x = prim.center[0] + prim.radius_m * np.cos(mid_angle)
            label_y = prim.center[1] + prim.radius_m * np.sin(mid_angle)
            ax2.text(
                label_x,
                label_y,
                f"B{i+1}\n{abs(prim.angle_deg):.1f}°",
                ha="center",
                va="center",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.5),
            )

    ax2.set_aspect("equal")
    ax2.grid(True, alpha=0.3)
    ax2.legend(["Original", "Straights", "Bends"])
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")

    plt.tight_layout()
    plt.savefig("hough_test_result.png", dpi=150, bbox_inches="tight")
    plt.show()


def main():
    """Test the Hough methodology."""
    print("Testing Hough + Turn Clustering Methodology")
    print("=" * 50)

    # Create test route
    print("\nGenerating test route...")
    points = create_test_route()
    print(f"  Created route with {len(points)} points")

    # Calculate route length
    total_length = sum(
        np.sqrt(
            (points[i + 1][0] - points[i][0]) ** 2
            + (points[i + 1][1] - points[i][1]) ** 2
        )
        for i in range(len(points) - 1)
    )
    print(f"  Total length: {total_length:.2f}m")

    # Create section
    section = Section(
        id="test_section", original_polyline=points, original_length=total_length
    )

    # Create Hough fitter
    print("\nCreating Hough + Turn Clustering fitter...")
    try:
        fitter = FitterFactory.create_fitter(
            methodology="4_hough_clustering",
            duct_type="200mm",
            hough_threshold=10,
            corner_zone_radius=8.0,
            min_straight_length=3.0,
        )
        print(f"  Fitter: {fitter.get_methodology_name()}")
    except Exception as e:
        print(f"ERROR creating fitter: {e}")
        return

    # Fit the section
    print("\nFitting route to inventory constraints...")
    try:
        primitives = fitter.fit_section(section)
        print(f"  Generated {len(primitives)} primitives")
    except Exception as e:
        print(f"ERROR during fitting: {e}")
        import traceback

        traceback.print_exc()
        return

    # Analyze results
    from easycablepulling.core.models import Straight, Bend

    num_straights = sum(1 for p in primitives if isinstance(p, Straight))
    num_bends = sum(1 for p in primitives if isinstance(p, Bend))

    print(f"\nResults:")
    print(f"  Straights: {num_straights}")
    print(f"  Bends: {num_bends}")

    if num_bends > 0:
        bend_angles = [p.angle_deg for p in primitives if isinstance(p, Bend)]
        print(f"  Bend angles: {bend_angles}")

        # Count by type
        bends_11_25 = sum(1 for a in bend_angles if abs(abs(a) - 11.25) < 0.1)
        bends_22_5 = sum(1 for a in bend_angles if abs(abs(a) - 22.5) < 0.1)
        print(f"  11.25° bends: {bends_11_25}")
        print(f"  22.5° bends: {bends_22_5}")

    # Calculate fitted length
    fitted_length = sum(
        (
            p.length_m
            if isinstance(p, Straight)
            else p.radius_m * abs(np.radians(p.angle_deg))
        )
        for p in primitives
    )
    print(f"\nLength comparison:")
    print(f"  Original: {total_length:.2f}m")
    print(f"  Fitted: {fitted_length:.2f}m")
    print(
        f"  Difference: {fitted_length - total_length:.2f}m ({(fitted_length/total_length - 1)*100:.1f}%)"
    )

    # Plot comparison
    print("\nGenerating visualization...")
    plot_comparison(points, primitives)
    print("  Saved to hough_test_result.png")

    print("\n✓ Test complete!")


if __name__ == "__main__":
    main()
