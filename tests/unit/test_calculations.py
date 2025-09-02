"""Unit tests for cable pulling calculations."""

import math

import pytest

from easycablepulling.calculations.pressure import (
    LimitCheckResult,
    PressureResult,
    analyze_section_pressures,
    calculate_jam_factor,
    calculate_sidewall_pressure,
    check_duct_clearance,
    check_section_limits,
)
from easycablepulling.calculations.tension import (
    SectionTensionAnalysis,
    TensionResult,
    analyze_route_tension,
    analyze_section_tension,
    calculate_bend_tension,
    calculate_section_tensions,
    calculate_straight_tension,
    find_optimal_pull_direction,
)
from easycablepulling.core.models import (
    Bend,
    CableArrangement,
    CableSpec,
    DuctSpec,
    PullingMethod,
    Route,
    Section,
    Straight,
)


@pytest.fixture
def sample_cable_spec():
    """Sample cable specification for testing."""
    return CableSpec(
        diameter=35.0,  # mm
        weight_per_meter=2.5,  # kg/m
        max_tension=8000.0,  # N
        max_sidewall_pressure=500.0,  # N/m
        min_bend_radius=1200.0,  # mm
        pulling_method=PullingMethod.EYE,
        arrangement=CableArrangement.SINGLE,
        number_of_cables=1,
    )


@pytest.fixture
def sample_trefoil_cable_spec():
    """Sample trefoil cable specification for testing."""
    return CableSpec(
        diameter=35.0,  # mm
        weight_per_meter=2.5,  # kg/m
        max_tension=12000.0,  # N
        max_sidewall_pressure=400.0,  # N/m
        min_bend_radius=1200.0,  # mm
        pulling_method=PullingMethod.BASKET,
        arrangement=CableArrangement.TREFOIL,
        number_of_cables=3,
    )


@pytest.fixture
def sample_duct_spec():
    """Sample duct specification for testing."""
    return DuctSpec(
        inner_diameter=100.0,  # mm
        type="PVC",
        friction_dry=0.35,
        friction_lubricated=0.15,
    )


@pytest.fixture
def sample_section(sample_cable_spec):
    """Sample section with straight and bend primitives."""
    straight = Straight(length_m=50.0, start_point=(0.0, 0.0), end_point=(50.0, 0.0))

    bend = Bend(
        radius_m=1.5,
        angle_deg=90.0,
        direction="CW",
        center_point=(50.0, 1.5),
        bend_type="manufactured",
    )

    section = Section(
        id="TEST_01",
        original_polyline=[(0.0, 0.0), (50.0, 0.0), (51.5, 1.5)],
        primitives=[straight, bend],
    )

    return section


class TestTensionCalculations:
    """Test tension calculation functions."""

    def test_straight_section_tension_basic(self, sample_cable_spec, sample_duct_spec):
        """Test basic tension calculation for straight sections."""
        tension_in = 1000.0  # N
        length = 100.0  # m

        tension_out = calculate_straight_tension(
            tension_in, sample_cable_spec, sample_duct_spec, length
        )

        # T_out = T_in + W * f * L
        # W = 2.5 kg/m * 9.81 m/s² = 24.525 N/m
        # f = 0.35 (dry PVC single cable)
        # Expected: 1000 + 24.525 * 0.35 * 100 = 1858.375 N
        assert abs(tension_out - 1858.375) < 0.01

    def test_straight_section_tension_with_slope(
        self, sample_cable_spec, sample_duct_spec
    ):
        """Test tension calculation with slope correction."""
        tension_in = 1000.0  # N
        length = 100.0  # m
        slope_angle = 10.0  # degrees uphill

        tension_out = calculate_straight_tension(
            tension_in, sample_cable_spec, sample_duct_spec, length, False, slope_angle
        )

        # With slope: effective_weight = weight * (friction + sin(slope))
        weight = 2.5 * 9.81  # N/m
        friction = 0.35
        slope_factor = math.sin(math.radians(10.0))
        effective_weight = weight * (friction + slope_factor)
        expected = 1000.0 + effective_weight * 100.0

        assert abs(tension_out - expected) < 0.01

    def test_straight_section_tension_lubricated(
        self, sample_cable_spec, sample_duct_spec
    ):
        """Test tension calculation with lubrication."""
        tension_in = 1000.0  # N
        length = 100.0  # m

        tension_out = calculate_straight_tension(
            tension_in, sample_cable_spec, sample_duct_spec, length, True
        )

        # With lubrication: f = 0.15
        weight = 2.5 * 9.81  # N/m
        expected = 1000.0 + weight * 0.15 * 100.0

        assert abs(tension_out - expected) < 0.01

    def test_bend_section_tension_basic(self, sample_cable_spec, sample_duct_spec):
        """Test basic tension calculation for bend sections."""
        tension_in = 1000.0  # N
        angle_deg = 90.0

        tension_out = calculate_bend_tension(
            tension_in, sample_cable_spec, sample_duct_spec, angle_deg
        )

        # T_out = T_in * e^(f * θ)
        # f = 0.35, θ = π/2 radians
        angle_rad = math.radians(90.0)
        expected = 1000.0 * math.exp(0.35 * angle_rad)

        assert abs(tension_out - expected) < 0.01

    def test_bend_tension_trefoil(self, sample_trefoil_cable_spec, sample_duct_spec):
        """Test bend tension with trefoil arrangement."""
        tension_in = 1000.0  # N
        angle_deg = 90.0

        tension_out = calculate_bend_tension(
            tension_in, sample_trefoil_cable_spec, sample_duct_spec, angle_deg
        )

        # Trefoil has 30% higher friction: 0.35 * 1.3 = 0.455
        angle_rad = math.radians(90.0)
        expected = 1000.0 * math.exp(0.455 * angle_rad)

        assert abs(tension_out - expected) < 0.01

    def test_section_tension_analysis(
        self, sample_section, sample_cable_spec, sample_duct_spec
    ):
        """Test complete section tension analysis."""
        analysis = analyze_section_tension(
            sample_section, sample_cable_spec, sample_duct_spec
        )

        assert analysis.section_id == "TEST_01"
        assert len(analysis.forward_tensions) == 2  # straight + bend
        assert len(analysis.backward_tensions) == 2  # bend + straight (reversed)
        assert analysis.max_tension > 0
        assert analysis.max_tension_position >= 0

    def test_optimal_pull_direction(
        self, sample_section, sample_cable_spec, sample_duct_spec
    ):
        """Test optimal pull direction determination."""
        direction, max_tension = find_optimal_pull_direction(
            sample_section, sample_cable_spec, sample_duct_spec
        )

        assert direction in ["forward", "backward"]
        assert max_tension > 0


class TestPressureCalculations:
    """Test pressure calculation functions."""

    def test_sidewall_pressure_calculation(self):
        """Test basic sidewall pressure calculation."""
        tension = 2000.0  # N
        radius = 1.0  # m

        pressure = calculate_sidewall_pressure(tension, radius)

        # P = T / r = 2000 / 1.0 = 2000 N/m
        assert pressure == 2000.0

    def test_sidewall_pressure_smaller_radius(self):
        """Test pressure increases with smaller radius."""
        tension = 1000.0  # N

        pressure_1m = calculate_sidewall_pressure(tension, 1.0)
        pressure_05m = calculate_sidewall_pressure(tension, 0.5)

        assert pressure_05m == 2 * pressure_1m

    def test_section_pressure_analysis(self, sample_section, sample_cable_spec):
        """Test pressure analysis for a section."""
        # Create some mock tension results
        tension_results = [
            TensionResult(25.0, 1000.0, 0, "straight"),
            TensionResult(
                52.36, 1500.0, 1, "bend"
            ),  # Approximate arc length for 90° at 1.5m radius
        ]

        pressure_results = analyze_section_pressures(
            sample_section, tension_results, sample_cable_spec
        )

        assert len(pressure_results) == 2

        # Straight section should have no pressure
        assert pressure_results[0].pressure == 0.0
        assert not pressure_results[0].is_critical

        # Bend section should have pressure
        assert pressure_results[1].pressure > 0.0
        # Check if critical (1500 N / 1.5 m = 1000 N/m > 500 N/m limit)
        assert pressure_results[1].is_critical

    def test_duct_clearance_single_cable(self, sample_cable_spec, sample_duct_spec):
        """Test duct clearance check for single cable."""
        fits, clearance_ratio = check_duct_clearance(
            sample_cable_spec, sample_duct_spec
        )

        # 35mm cable in 100mm duct = 0.35 ratio
        assert clearance_ratio == 0.35
        assert fits  # Should fit (< 0.9 limit)

    def test_duct_clearance_trefoil(self, sample_trefoil_cable_spec, sample_duct_spec):
        """Test duct clearance check for trefoil arrangement."""
        fits, clearance_ratio = check_duct_clearance(
            sample_trefoil_cable_spec, sample_duct_spec
        )

        # Trefoil bundle: 35mm * 2.15 = 75.25mm in 100mm duct = 0.7525 ratio
        expected_ratio = (35.0 * 2.15) / 100.0
        assert abs(clearance_ratio - expected_ratio) < 0.01
        assert fits  # Should fit (< 0.8 limit for multi-cable)

    def test_jam_factor_single_cable(self, sample_cable_spec, sample_duct_spec):
        """Test jam factor for single cable."""
        jam_factor = calculate_jam_factor(sample_cable_spec, sample_duct_spec)

        # Single cable should have no jamming
        assert jam_factor == 0.0

    def test_jam_factor_trefoil(self, sample_trefoil_cable_spec, sample_duct_spec):
        """Test jam factor for trefoil arrangement."""
        jam_factor = calculate_jam_factor(sample_trefoil_cable_spec, sample_duct_spec)

        # Should be positive but reduced by trefoil factor (0.8)
        assert jam_factor > 0.0

        # Compare with flat arrangement
        flat_spec = CableSpec(
            diameter=35.0,
            weight_per_meter=2.5,
            max_tension=12000.0,
            max_sidewall_pressure=400.0,
            min_bend_radius=1200.0,
            arrangement=CableArrangement.FLAT,
            number_of_cables=3,
        )

        flat_jam_factor = calculate_jam_factor(flat_spec, sample_duct_spec)
        assert jam_factor < flat_jam_factor  # Trefoil should have lower jamming


class TestLimitChecking:
    """Test limit checking functions."""

    def test_section_limits_pass(
        self, sample_section, sample_cable_spec, sample_duct_spec
    ):
        """Test limit checking when all limits pass."""
        # Create mock tension results that should pass
        forward_tensions = [
            TensionResult(25.0, 1000.0, 0, "straight"),
            TensionResult(52.36, 2000.0, 1, "bend"),
        ]
        backward_tensions = [
            TensionResult(2.36, 1800.0, 1, "bend"),
            TensionResult(52.36, 2200.0, 0, "straight"),
        ]

        result = check_section_limits(
            sample_section,
            sample_cable_spec,
            sample_duct_spec,
            forward_tensions,
            backward_tensions,
        )

        assert result.section_id == "TEST_01"
        assert result.passes_tension_limit  # Max 2200 < 8000 limit
        assert result.passes_bend_radius_limit  # 1.5m > 1.2m limit
        assert result.recommended_direction in ["forward", "backward"]

    def test_section_limits_tension_fail(
        self, sample_section, sample_cable_spec, sample_duct_spec
    ):
        """Test limit checking when tension limit fails."""
        # Create mock tension results that exceed limits
        forward_tensions = [
            TensionResult(25.0, 7000.0, 0, "straight"),
            TensionResult(52.36, 9000.0, 1, "bend"),  # Exceeds 8000 N limit
        ]
        backward_tensions = [
            TensionResult(2.36, 8500.0, 1, "bend"),  # Exceeds 8000 N limit
            TensionResult(52.36, 10000.0, 0, "straight"),
        ]

        result = check_section_limits(
            sample_section,
            sample_cable_spec,
            sample_duct_spec,
            forward_tensions,
            backward_tensions,
        )

        assert not result.passes_tension_limit
        assert "tension_limit" in result.limiting_factors

    def test_section_limits_bend_radius_fail(self, sample_cable_spec, sample_duct_spec):
        """Test limit checking when bend radius fails."""
        # Create section with tight bend
        tight_bend = Bend(
            radius_m=1.0,  # 1000mm < 1200mm minimum
            angle_deg=90.0,
            direction="CW",
            center_point=(50.0, 1.0),
        )

        section = Section(
            id="TIGHT_01",
            original_polyline=[(0.0, 0.0), (50.0, 0.0), (51.0, 1.0)],
            primitives=[tight_bend],
        )

        forward_tensions = [TensionResult(2.36, 1000.0, 0, "bend")]
        backward_tensions = [TensionResult(2.36, 1000.0, 0, "bend")]

        result = check_section_limits(
            section,
            sample_cable_spec,
            sample_duct_spec,
            forward_tensions,
            backward_tensions,
        )

        assert not result.passes_bend_radius_limit
        assert "bend_radius" in result.limiting_factors


class TestValidation:
    """Test validation functions."""

    def test_input_validation_negative_tension(
        self, sample_cable_spec, sample_duct_spec
    ):
        """Test that negative tension raises error."""
        with pytest.raises(ValueError, match="Input tension cannot be negative"):
            calculate_straight_tension(
                -100.0, sample_cable_spec, sample_duct_spec, 10.0
            )

        with pytest.raises(ValueError, match="Input tension cannot be negative"):
            calculate_bend_tension(-100.0, sample_cable_spec, sample_duct_spec, 90.0)

    def test_input_validation_negative_length(
        self, sample_cable_spec, sample_duct_spec
    ):
        """Test that negative length raises error."""
        with pytest.raises(ValueError, match="Length cannot be negative"):
            calculate_straight_tension(
                1000.0, sample_cable_spec, sample_duct_spec, -10.0
            )

    def test_input_validation_pressure(self):
        """Test pressure calculation input validation."""
        with pytest.raises(ValueError, match="Tension cannot be negative"):
            calculate_sidewall_pressure(-100.0, 1.0)

        with pytest.raises(ValueError, match="Bend radius must be positive"):
            calculate_sidewall_pressure(1000.0, 0.0)


class TestIntegration:
    """Integration tests for calculation workflow."""

    def test_complete_section_analysis(
        self, sample_section, sample_cable_spec, sample_duct_spec
    ):
        """Test complete analysis workflow for a section."""
        # Run tension analysis
        tension_analysis = analyze_section_tension(
            sample_section, sample_cable_spec, sample_duct_spec
        )

        # Run pressure analysis using forward tensions
        pressure_results = analyze_section_pressures(
            sample_section, tension_analysis.forward_tensions, sample_cable_spec
        )

        # Run limit checking
        limit_result = check_section_limits(
            sample_section,
            sample_cable_spec,
            sample_duct_spec,
            tension_analysis.forward_tensions,
            tension_analysis.backward_tensions,
        )

        # Verify results are consistent
        assert len(pressure_results) == len(tension_analysis.forward_tensions)
        assert limit_result.section_id == tension_analysis.section_id
        assert limit_result.max_tension == tension_analysis.max_tension

    def test_route_analysis(self, sample_cable_spec, sample_duct_spec):
        """Test route-level tension analysis."""
        # Create a simple route with two sections
        section1 = Section(
            id="SECT_01",
            original_polyline=[(0.0, 0.0), (100.0, 0.0)],
            primitives=[Straight(100.0, (0.0, 0.0), (100.0, 0.0))],
        )

        section2 = Section(
            id="SECT_02",
            original_polyline=[(100.0, 0.0), (200.0, 0.0)],
            primitives=[Straight(100.0, (100.0, 0.0), (200.0, 0.0))],
        )

        route = Route(name="TEST_ROUTE", sections=[section1, section2])

        analyses = analyze_route_tension(route, sample_cable_spec, sample_duct_spec)

        assert len(analyses) == 2
        assert analyses[0].section_id == "SECT_01"
        assert analyses[1].section_id == "SECT_02"
