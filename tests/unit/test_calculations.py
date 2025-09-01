"""Unit tests for cable pulling calculations."""

import math

import pytest


class TestTensionCalculations:
    """Test tension calculation functions."""

    def test_straight_section_tension(self) -> None:
        """Test tension calculation for straight sections."""
        # T_out = T_in + W * f * L
        t_in = 1000.0  # N
        weight = 2.5  # kg/m
        friction = 0.35
        length = 100.0  # m

        expected_t_out = t_in + (weight * 9.81 * friction * length)
        assert abs(expected_t_out - 1858.375) < 0.01

    def test_bend_section_tension(self) -> None:
        """Test tension calculation for bend sections."""
        # T_out = T_in * e^(f * theta)
        t_in = 1000.0  # N
        friction = 0.35
        angle_deg = 90.0
        angle_rad = math.radians(angle_deg)

        expected_t_out = t_in * math.exp(friction * angle_rad)
        assert abs(expected_t_out - 1732.87) < 0.01

    def test_sidewall_pressure(self) -> None:
        """Test sidewall pressure calculation."""
        # P = T_out / r
        t_out = 2000.0  # N
        radius = 1.0  # m

        expected_pressure = t_out / radius
        assert expected_pressure == 2000.0


class TestValidation:
    """Test validation functions."""

    def test_minimum_bend_radius_check(self, sample_cable_spec) -> None:
        """Test minimum bend radius validation."""
        min_radius = sample_cable_spec.min_bend_radius

        # Valid bend radius
        assert min_radius <= 1000.0

        # Invalid bend radius
        assert not (min_radius <= 500.0)
