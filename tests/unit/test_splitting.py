"""Unit tests for route splitting functionality."""

import math

import pytest

from easycablepulling.core.models import Route, Section
from easycablepulling.geometry.splitter import RouteSplitter, SplitPoint


@pytest.fixture
def long_section():
    """Create a long section that needs splitting."""
    # Create a 600m long straight section (exceeds 500m default)
    points = []
    for i in range(61):  # 61 points for 600m (10m intervals)
        points.append((i * 10.0, 0.0))

    return Section(id="LONG_01", original_polyline=points, primitives=[])


@pytest.fixture
def short_section():
    """Create a short section that doesn't need splitting."""
    # Create a 300m section (within 500m limit)
    points = []
    for i in range(31):  # 31 points for 300m
        points.append((i * 10.0, 0.0))

    return Section(id="SHORT_01", original_polyline=points, primitives=[])


@pytest.fixture
def bent_section():
    """Create a section with bends that needs careful splitting."""
    points = []

    # Straight section 0-200m
    for i in range(21):
        points.append((i * 10.0, 0.0))

    # 90-degree bend 200-250m
    for i in range(1, 11):
        angle = i * math.pi / 2 / 10  # 0 to π/2 over 10 points
        x = 200.0 + 50.0 * math.sin(angle)
        y = 50.0 * (1 - math.cos(angle))
        points.append((x, y))

    # Straight section 250-650m (400m)
    for i in range(1, 41):
        points.append((250.0, 50.0 + i * 10.0))

    return Section(id="BENT_01", original_polyline=points, primitives=[])


class TestRouteSplitter:
    """Test route splitter functionality."""

    def test_splitter_initialization(self):
        """Test splitter initialization with custom parameters."""
        splitter = RouteSplitter(
            max_cable_length=400.0,
            min_section_length=100.0,
            avoid_bend_distance=15.0,
        )

        assert splitter.max_cable_length == 400.0
        assert splitter.min_section_length == 100.0
        assert splitter.avoid_bend_distance == 15.0

    def test_needs_splitting_long_section(self, long_section):
        """Test that long sections are identified for splitting."""
        splitter = RouteSplitter(max_cable_length=500.0)

        assert splitter.needs_splitting(long_section)  # 600m > 500m

    def test_needs_splitting_short_section(self, short_section):
        """Test that short sections are not identified for splitting."""
        splitter = RouteSplitter(max_cable_length=500.0)

        assert not splitter.needs_splitting(short_section)  # 300m < 500m

    def test_find_split_points_long_straight(self, long_section):
        """Test finding split points in long straight section."""
        splitter = RouteSplitter(max_cable_length=500.0)

        split_points = splitter.find_optimal_split_points(long_section)

        # 600m section should be split into 2 sections (~300m each)
        assert len(split_points) == 1
        assert split_points[0].reason == "max_length"
        assert 250.0 < split_points[0].position < 350.0  # Around middle

    def test_find_split_points_very_long(self):
        """Test splitting very long section into multiple parts."""
        # Create 1200m section
        points = [(i * 10.0, 0.0) for i in range(121)]
        section = Section(id="VERY_LONG", original_polyline=points, primitives=[])

        splitter = RouteSplitter(max_cable_length=500.0)
        split_points = splitter.find_optimal_split_points(section)

        # 1200m should split into 3 sections (~400m each) = 2 split points
        assert len(split_points) == 2
        assert all(sp.reason == "max_length" for sp in split_points)

    def test_avoid_splitting_near_bends(self, bent_section):
        """Test that splitter avoids splitting near bends."""
        splitter = RouteSplitter(max_cable_length=500.0, avoid_bend_distance=30.0)

        split_points = splitter.find_optimal_split_points(bent_section)

        # Should find split points but avoid the bend area (around 200-250m)
        for sp in split_points:
            # Split should not be near the bend area
            assert not (180.0 < sp.position < 270.0)  # Avoid bend ±30m

    def test_split_section_basic(self, long_section):
        """Test basic section splitting."""
        splitter = RouteSplitter(max_cable_length=500.0)

        split_points = splitter.find_optimal_split_points(long_section)
        new_sections = splitter.split_section(long_section, split_points)

        assert len(new_sections) == 2  # Original split into 2
        assert new_sections[0].id == "LONG_01_01"
        assert new_sections[1].id == "LONG_01_02"

        # Check that both sections are reasonable length
        assert 250.0 < new_sections[0].original_length < 350.0
        assert 250.0 < new_sections[1].original_length < 350.0

        # Total length should be preserved (approximately)
        total_new_length = sum(s.original_length for s in new_sections)
        assert (
            abs(total_new_length - long_section.original_length) < 1.0
        )  # 1m tolerance

    def test_split_route_mixed_sections(self, long_section, short_section):
        """Test splitting route with mixed section lengths."""
        route = Route(name="MIXED_ROUTE", sections=[short_section, long_section])
        splitter = RouteSplitter(max_cable_length=500.0)

        result = splitter.split_route(route)

        assert result.success
        assert len(result.split_route.sections) == 3  # short + 2 from long split
        assert result.sections_created == 1  # One additional section created

        # Check section IDs
        section_ids = [s.id for s in result.split_route.sections]
        assert "SHORT_01" in section_ids  # Short section unchanged
        assert "LONG_01_01" in section_ids  # First part of split
        assert "LONG_01_02" in section_ids  # Second part of split

    def test_no_splitting_needed(self, short_section):
        """Test route that doesn't need splitting."""
        route = Route(name="SHORT_ROUTE", sections=[short_section])
        splitter = RouteSplitter(max_cable_length=500.0)

        result = splitter.split_route(route)

        assert result.success
        assert len(result.split_route.sections) == 1  # No change
        assert result.sections_created == 0
        assert len(result.split_points) == 0

    def test_joint_location_splitting(self, long_section):
        """Test splitting at standard joint locations."""
        splitter = RouteSplitter()

        joint_sections = splitter.split_at_joints(long_section, joint_spacing=100.0)

        # 600m section with 100m joint spacing should create multiple sections
        assert len(joint_sections) > 1

        # Each section should be around 100m or less
        for section in joint_sections:
            assert (
                section.original_length <= 110.0
            )  # Some tolerance for joint alignment

    def test_minimum_section_length_constraint(self):
        """Test that minimum section length is enforced."""
        # Create section just over limit but would create too-short subsections
        points = [(i * 10.0, 0.0) for i in range(52)]  # 510m
        section = Section(id="MARGINAL", original_polyline=points, primitives=[])

        splitter = RouteSplitter(
            max_cable_length=500.0,
            min_section_length=200.0,  # High minimum
        )

        split_points = splitter.find_optimal_split_points(section)

        # Should not split because resulting sections would be < 200m
        assert len(split_points) == 0

    def test_edge_case_very_short_polyline(self):
        """Test edge case with very short polyline."""
        points = [(0.0, 0.0), (10.0, 0.0)]  # Only 2 points
        section = Section(id="TINY", original_polyline=points, primitives=[])

        splitter = RouteSplitter()
        split_points = splitter.find_optimal_split_points(section)

        assert len(split_points) == 0  # Cannot split 2-point polyline


class TestSplitPointValidation:
    """Test split point validation logic."""

    def test_valid_split_point_straight(self):
        """Test valid split point identification in straight section."""
        points = [(i * 10.0, 0.0) for i in range(61)]  # 600m straight
        distances = [i * 10.0 for i in range(61)]

        splitter = RouteSplitter()

        # Middle point should be valid
        assert splitter._is_valid_split_point(points, 30, distances)

        # Points near ends should be invalid
        assert not splitter._is_valid_split_point(points, 1, distances)
        assert not splitter._is_valid_split_point(points, 59, distances)

    def test_sharp_bend_detection(self):
        """Test detection of sharp bends."""
        # Create polyline with sharp 90-degree turn
        points = [
            (0.0, 0.0),
            (50.0, 0.0),
            (100.0, 0.0),  # Sharp turn point
            (100.0, 50.0),
            (100.0, 100.0),
        ]

        splitter = RouteSplitter()

        # Point 2 (turn point) should have sharp bend nearby
        assert splitter._has_sharp_bend_nearby(points, 2, angle_threshold=30.0)

        # Point 1 (before turn) should also be flagged
        assert splitter._has_sharp_bend_nearby(points, 1, angle_threshold=30.0)

        # Point 0 shouldn't be checked (too close to end)
        assert not splitter._has_sharp_bend_nearby(points, 0, angle_threshold=30.0)


class TestIntegration:
    """Integration tests for splitting with other components."""

    def test_splitting_preserves_polyline_integrity(self, long_section):
        """Test that splitting preserves overall polyline integrity."""
        splitter = RouteSplitter()

        split_points = splitter.find_optimal_split_points(long_section)
        new_sections = splitter.split_section(long_section, split_points)

        # Reconstruct full polyline from split sections
        reconstructed_points = []
        for i, section in enumerate(new_sections):
            if i == 0:
                # First section: add all points
                reconstructed_points.extend(section.original_polyline)
            else:
                # Subsequent sections: skip first point (overlap)
                reconstructed_points.extend(section.original_polyline[1:])

        # Should match original polyline
        assert len(reconstructed_points) == len(long_section.original_polyline)

        for orig, recon in zip(long_section.original_polyline, reconstructed_points):
            assert abs(orig[0] - recon[0]) < 1e-6
            assert abs(orig[1] - recon[1]) < 1e-6
