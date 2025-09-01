"""Pytest configuration and fixtures."""

from pathlib import Path

import pytest

from easycablepulling.config import DEFAULT_INPUT_DXF


@pytest.fixture
def test_data_dir() -> Path:
    """Return the path to test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture
def input_dxf_path() -> Path:
    """Return the path to the default input.dxf file."""
    return DEFAULT_INPUT_DXF


@pytest.fixture
def sample_cable_spec():
    """Return a sample cable specification."""
    from easycablepulling.core.models import CableSpec, PullingMethod

    return CableSpec(
        diameter=50.0,  # mm
        weight_per_meter=2.5,  # kg/m
        max_tension=5000.0,  # N
        max_sidewall_pressure=3000.0,  # N/m
        min_bend_radius=600.0,  # mm
        pulling_method=PullingMethod.EYE,
    )


@pytest.fixture
def sample_duct_spec():
    """Return a sample duct specification."""
    from easycablepulling.core.models import BendOption, DuctSpec

    return DuctSpec(
        inner_diameter=100.0,  # mm
        type="PVC",
        friction_dry=0.35,
        friction_lubricated=0.15,
        bend_options=[
            BendOption(radius=1000.0, angle=90.0),
            BendOption(radius=1500.0, angle=45.0),
            BendOption(radius=2000.0, angle=30.0),
        ],
    )
