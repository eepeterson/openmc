"""Tests for DepletionManager construction and configuration."""

import numpy as np
import pytest

import openmc
from openmc.deplete.depletion_manager import DepletionManager, _TIMESTEP_UNITS
from openmc.deplete.integration_schemes import SCHEMES


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_model(tmp_path):
    """Minimal model with one depletable material (no transport)."""
    mat = openmc.Material()
    mat.add_nuclide('U235', 1.0)
    mat.set_density('g/cm3', 10.0)
    mat.depletable = True
    mat.volume = 100.0

    sph = openmc.Sphere(r=10.0, boundary_type='vacuum')
    cell = openmc.Cell(fill=mat, region=-sph)
    geometry = openmc.Geometry([cell])

    settings = openmc.Settings()
    settings.particles = 100
    settings.batches = 5
    settings.inactive = 2

    return openmc.Model(geometry, openmc.Materials([mat]), settings)


@pytest.fixture
def chain_file():
    """Path to a depletion chain file for testing.

    Uses the chain pointed to by OPENMC_CHAIN_FILE or a known location.
    """
    import os
    cf = os.environ.get('OPENMC_CHAIN_FILE')
    if cf:
        return cf
    # Fallback: look for a commonly installed chain
    from pathlib import Path
    candidates = [
        Path.home() / 'nuclear_data' / 'chain_endfb80_pwr.xml',
        Path.home() / 'nndc_hdf5' / 'chain_simple.xml',
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    pytest.skip('No chain file available')


# ---------------------------------------------------------------------------
# Constructor validation tests
# ---------------------------------------------------------------------------

def test_constructor_validates_source_rate_type(simple_model, chain_file):
    with pytest.raises(ValueError, match='source_rate_type'):
        DepletionManager(simple_model, chain_file, [1.0], 1e6,
                         source_rate_type='invalid')


def test_constructor_validates_scheme_name(simple_model, chain_file):
    with pytest.raises(ValueError, match='scheme'):
        DepletionManager(simple_model, chain_file, [1.0], 1e6,
                         scheme='nonexistent_scheme')


def test_constructor_validates_solver_order(simple_model, chain_file):
    with pytest.raises(ValueError, match='solver_order'):
        DepletionManager(simple_model, chain_file, [1.0], 1e6,
                         solver_order=32)


def test_constructor_validates_source_rate_length(simple_model, chain_file):
    with pytest.raises(ValueError, match='Length of source_rates'):
        DepletionManager(simple_model, chain_file, [1.0, 2.0], [1e6])


def test_constructor_validates_scheme_type(simple_model, chain_file):
    with pytest.raises(TypeError, match='scheme'):
        DepletionManager(simple_model, chain_file, [1.0], 1e6,
                         scheme=42)


def test_constructor_accepts_scheme_string(simple_model, chain_file):
    mgr = DepletionManager(simple_model, chain_file, [1.0], 1e6,
                           scheme='cecm')
    assert mgr._scheme_name == 'cecm'


def test_constructor_scalar_source_rate(simple_model, chain_file):
    mgr = DepletionManager(simple_model, chain_file, [1.0, 2.0], 1e6)
    np.testing.assert_array_equal(mgr._source_rates, [1e6, 1e6])


def test_constructor_timestep_units(simple_model, chain_file):
    mgr = DepletionManager(simple_model, chain_file, [1.0], 1e6,
                           timestep_units='d')
    np.testing.assert_allclose(mgr._timesteps_s, [86400.0])


# ---------------------------------------------------------------------------
# transport_schedule tests
# ---------------------------------------------------------------------------

def test_transport_schedule_every(simple_model, chain_file):
    mgr = DepletionManager(simple_model, chain_file, [1.0, 2.0], 1e6,
                           transport_schedule='every')
    assert mgr._transport_mask == [True, True]


def test_transport_schedule_first(simple_model, chain_file):
    mgr = DepletionManager(simple_model, chain_file, [1.0, 2.0, 3.0], 1e6,
                           transport_schedule='first')
    assert mgr._transport_mask == [True, False, False]


def test_transport_schedule_bool_list(simple_model, chain_file):
    mgr = DepletionManager(simple_model, chain_file, [1.0, 2.0, 3.0], 1e6,
                           transport_schedule=[True, False, True])
    assert mgr._transport_mask == [True, False, True]


def test_transport_schedule_length_mismatch(simple_model, chain_file):
    with pytest.raises(ValueError, match='transport_schedule'):
        DepletionManager(simple_model, chain_file, [1.0, 2.0], 1e6,
                         transport_schedule=[True])


def test_transport_schedule_invalid_string(simple_model, chain_file):
    with pytest.raises(ValueError, match='transport_schedule'):
        DepletionManager(simple_model, chain_file, [1.0], 1e6,
                         transport_schedule='invalid')


# ---------------------------------------------------------------------------
# prev_results validation tests
# ---------------------------------------------------------------------------

def test_prev_results_invalid_type(simple_model, chain_file):
    with pytest.raises(TypeError, match='prev_results'):
        DepletionManager(simple_model, chain_file, [1.0], 1e6,
                         prev_results=42)


def test_prev_results_missing_file(simple_model, chain_file):
    with pytest.raises(Exception):
        DepletionManager(simple_model, chain_file, [1.0], 1e6,
                         prev_results='/nonexistent/results.h5')
