"""Unit tests for core data models."""

import math

import pytest

from easycablepulling.core.models import (
    Bend,
    BendOption,
    CableArrangement,
    CableSpec,
    DuctSpec,
    PullingMethod,
    Route,
    Section,
    Straight,
)


class TestCableSpec:
    """Test CableSpec model."""

    def test_valid_single_cable(self):
        """Test valid single cable specification."""
        cable = CableSpec(
            diameter=50.0,
            weight_per_meter=2.5,
            max_tension=5000.0,
            max_sidewall_pressure=3000.0,
            min_bend_radius=600.0,
            pulling_method=PullingMethod.EYE,
            arrangement=CableArrangement.SINGLE,
            number_of_cables=1,
        )
        assert cable.bundle_diameter == 50.0
        assert cable.total_weight_per_meter == 2.5

    def test_valid_trefoil_cable(self):
        """Test valid trefoil cable specification."""
        cable = CableSpec(
            diameter=50.0,
            weight_per_meter=2.5,
            max_tension=15000.0,
            max_sidewall_pressure=3000.0,
            min_bend_radius=600.0,
            arrangement=CableArrangement.TREFOIL,
            number_of_cables=3,
        )
        assert cable.bundle_diameter == pytest.approx(107.5, rel=0.01)  # 2.15 * 50
        assert cable.total_weight_per_meter == 7.5  # 2.5 * 3

    def test_invalid_cable_values(self):
        """Test validation of invalid cable values."""
        with pytest.raises(ValueError, match="diameter must be positive"):
            CableSpec(
                diameter=-50.0,
                weight_per_meter=2.5,
                max_tension=5000.0,
                max_sidewall_pressure=3000.0,
                min_bend_radius=600.0,
            )

        with pytest.raises(ValueError, match="weight must be positive"):
            CableSpec(
                diameter=50.0,
                weight_per_meter=0,
                max_tension=5000.0,
                max_sidewall_pressure=3000.0,
                min_bend_radius=600.0,
            )

    def test_arrangement_validation(self):
        """Test cable arrangement validation."""
        # Single arrangement must have 1 cable
        with pytest.raises(
            ValueError, match="Single arrangement requires exactly 1 cable"
        ):
            CableSpec(
                diameter=50.0,
                weight_per_meter=2.5,
                max_tension=5000.0,
                max_sidewall_pressure=3000.0,
                min_bend_radius=600.0,
                arrangement=CableArrangement.SINGLE,
                number_of_cables=3,
            )

        # Trefoil must have 3 cables
        with pytest.raises(
            ValueError, match="Trefoil arrangement requires exactly 3 cables"
        ):
            CableSpec(
                diameter=50.0,
                weight_per_meter=2.5,
                max_tension=5000.0,
                max_sidewall_pressure=3000.0,
                min_bend_radius=600.0,
                arrangement=CableArrangement.TREFOIL,
                number_of_cables=2,
            )


class TestDuctSpec:
    """Test DuctSpec model."""

    def test_valid_duct(self):
        """Test valid duct specification."""
        duct = DuctSpec(
            inner_diameter=100.0,
            type="PVC",
            friction_dry=0.35,
            friction_lubricated=0.15,
            bend_options=[
                BendOption(radius=1000.0, angle=90.0),
                BendOption(radius=1500.0, angle=45.0),
            ],
        )
        assert duct.inner_diameter == 100.0
        assert len(duct.bend_options) == 2

    def test_cable_accommodation(self, sample_cable_spec):
        """Test cable accommodation check."""
        duct = DuctSpec(
            inner_diameter=100.0,
            type="PVC",
            friction_dry=0.35,
            friction_lubricated=0.15,
        )

        # Cable with 50mm diameter should fit in 100mm duct
        assert duct.can_accommodate_cable(sample_cable_spec)

        # Large cable shouldn't fit
        large_cable = CableSpec(
            diameter=95.0,  # Too large for 90% fill ratio
            weight_per_meter=5.0,
            max_tension=5000.0,
            max_sidewall_pressure=3000.0,
            min_bend_radius=1000.0,
        )
        assert not duct.can_accommodate_cable(large_cable)

    def test_friction_coefficients(self):
        """Test friction coefficient calculations."""
        duct = DuctSpec(
            inner_diameter=100.0,
            type="PVC",
            friction_dry=0.35,
            friction_lubricated=0.15,
        )

        # Single cable dry
        assert duct.get_friction(CableArrangement.SINGLE, lubricated=False) == 0.35

        # Single cable lubricated
        assert duct.get_friction(CableArrangement.SINGLE, lubricated=True) == 0.15

        # Trefoil dry (30% higher)
        assert duct.get_friction(
            CableArrangement.TREFOIL, lubricated=False
        ) == pytest.approx(0.455)

        # Flat arrangement (10% higher)
        assert duct.get_friction(
            CableArrangement.FLAT, lubricated=False
        ) == pytest.approx(0.385)

    def test_invalid_friction(self):
        """Test friction validation."""
        with pytest.raises(
            ValueError, match="Lubricated friction must be less than dry"
        ):
            DuctSpec(
                inner_diameter=100.0,
                type="PVC",
                friction_dry=0.35,
                friction_lubricated=0.40,  # Higher than dry
            )


class TestBendOption:
    """Test BendOption model."""

    def test_valid_bend(self):
        """Test valid bend option."""
        bend = BendOption(radius=1000.0, angle=90.0)
        assert bend.radius == 1000.0
        assert bend.angle == 90.0
        assert bend.angle_radians == pytest.approx(math.pi / 2)

    def test_invalid_bend(self):
        """Test bend validation."""
        with pytest.raises(ValueError, match="radius must be positive"):
            BendOption(radius=-1000.0, angle=90.0)

        with pytest.raises(ValueError, match="angle must be between 0 and 180"):
            BendOption(radius=1000.0, angle=200.0)


class TestPrimitives:
    """Test Straight and Bend primitives."""

    def test_straight_section(self):
        """Test straight section creation and validation."""
        straight = Straight(
            length_m=100.0, start_point=(0.0, 0.0), end_point=(100.0, 0.0)
        )
        assert straight.length() == 100.0
        assert (
            straight.validate(None) == []
        )  # No cable-specific validation for straights

    def test_straight_length_mismatch(self):
        """Test straight section with mismatched length."""
        with pytest.raises(ValueError, match="Length mismatch"):
            Straight(
                length_m=100.0,
                start_point=(0.0, 0.0),
                end_point=(50.0, 0.0),  # Only 50m based on coordinates
            )

    def test_bend_section(self):
        """Test bend section creation and validation."""
        bend = Bend(
            radius_m=1.0, angle_deg=90.0, direction="CW", center_point=(0.0, 0.0)
        )
        assert bend.length() == pytest.approx(math.pi / 2)  # Arc length
        assert bend.angle_rad == pytest.approx(math.pi / 2)

    def test_bend_validation(self, sample_cable_spec):
        """Test bend validation against cable spec."""
        # Bend with radius less than minimum
        tight_bend = Bend(
            radius_m=0.5,  # 500mm, less than 600mm minimum
            angle_deg=90.0,
            direction="CW",
            center_point=(0.0, 0.0),
        )

        warnings = tight_bend.validate(sample_cable_spec)
        assert len(warnings) == 1
        assert "less than minimum" in warnings[0]

        # Valid bend
        good_bend = Bend(
            radius_m=1.0,  # 1000mm, more than 600mm minimum
            angle_deg=90.0,
            direction="CW",
            center_point=(0.0, 0.0),
        )
        assert good_bend.validate(sample_cable_spec) == []


class TestSection:
    """Test Section model."""

    def test_section_creation(self):
        """Test section creation with primitives."""
        section = Section(
            id="AB",
            original_polyline=[(0, 0), (100, 0), (100, 100)],
            primitives=[
                Straight(length_m=100.0, start_point=(0, 0), end_point=(100, 0)),
                Bend(
                    radius_m=10.0,
                    angle_deg=90.0,
                    direction="CW",
                    center_point=(100, 10),
                ),
            ],
        )

        assert section.id == "AB"
        assert section.original_length == pytest.approx(200.0)  # 100 + 100
        assert section.total_length == pytest.approx(100.0 + 10.0 * math.pi / 2)

    def test_section_validation(self):
        """Test section validation."""
        # Empty ID
        with pytest.raises(ValueError, match="must have an ID"):
            Section(id="", original_polyline=[(0, 0), (100, 0)])

        # Too few points
        with pytest.raises(ValueError, match="at least 2 points"):
            Section(id="AB", original_polyline=[(0, 0)])

    def test_length_error_calculation(self):
        """Test length error percentage calculation."""
        section = Section(
            id="AB",
            original_polyline=[(0, 0), (100, 0)],
            primitives=[
                Straight(length_m=100.2, start_point=(0, 0), end_point=(100.2, 0))
            ],
        )

        # 0.2m error on 100m = 0.2%
        assert section.length_error_percent == pytest.approx(0.2)

    def test_fit_validation(self):
        """Test geometry fit validation."""
        section = Section(
            id="AB",
            original_polyline=[(0, 0), (100, 0)],
            primitives=[
                Straight(length_m=101.0, start_point=(0, 0), end_point=(101, 0))
            ],
        )

        # 1% error exceeds 0.2% threshold
        warnings = section.validate_fit(max_length_error=0.2)
        assert len(warnings) == 1
        assert "Length error" in warnings[0]


class TestRoute:
    """Test Route model."""

    def test_route_creation(self):
        """Test route creation and section management."""
        route = Route(name="Test Route")
        assert route.name == "Test Route"
        assert route.section_count == 0

        # Add sections
        section1 = Section(
            id="AB",
            original_polyline=[(0, 0), (100, 0)],
            primitives=[
                Straight(length_m=100.0, start_point=(0, 0), end_point=(100, 0))
            ],
        )
        section2 = Section(
            id="BC",
            original_polyline=[(100, 0), (200, 0)],
            primitives=[
                Straight(length_m=100.0, start_point=(100, 0), end_point=(200, 0))
            ],
        )

        route.add_section(section1)
        route.add_section(section2)

        assert route.section_count == 2
        assert route.total_length == 200.0

    def test_route_validation(self, sample_cable_spec):
        """Test route-wide validation."""
        route = Route(name="Test Route")

        # Add section with tight bend
        section = Section(
            id="AB",
            original_polyline=[(0, 0), (100, 0), (100, 100)],
            primitives=[
                Straight(length_m=100.0, start_point=(0, 0), end_point=(100, 0)),
                Bend(
                    radius_m=0.5,
                    angle_deg=90.0,
                    direction="CW",
                    center_point=(100, 0.5),
                ),
            ],
        )
        route.add_section(section)

        warnings = route.validate_cable(sample_cable_spec)
        assert len(warnings) > 0
        assert "less than minimum" in warnings[0]
