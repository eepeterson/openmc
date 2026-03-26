"""Tests for DepletionDriver construction and scheme interpreter helpers."""

import numpy as np
import pytest
from scipy.sparse import csc_array, eye as speye

import openmc
from openmc.deplete.driver import DepletionDriver, _TIMESTEP_UNITS
from openmc.deplete.integration_schemes import (
    BOS, PREV_STEP, PREV_ITER,
    Transport, Expm, MatrixTerm, AverageMatrix, Iterate,
    IntegrationScheme, SCHEMES,
    predictor, cecm,
)


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
        DepletionDriver(simple_model, chain_file, [1.0], 1e6,
                         source_rate_type='invalid')


def test_constructor_validates_scheme_name(simple_model, chain_file):
    with pytest.raises(ValueError, match='scheme'):
        DepletionDriver(simple_model, chain_file, [1.0], 1e6,
                         scheme='nonexistent_scheme')


def test_constructor_validates_solver_order(simple_model, chain_file):
    with pytest.raises(ValueError, match='solver_order'):
        DepletionDriver(simple_model, chain_file, [1.0], 1e6,
                         solver_order=32)


def test_constructor_validates_source_rate_length(simple_model, chain_file):
    with pytest.raises(ValueError, match='Length of source_rates'):
        DepletionDriver(simple_model, chain_file, [1.0, 2.0], [1e6])


def test_constructor_accepts_scheme_instance(simple_model, chain_file):
    driver = DepletionDriver(simple_model, chain_file, [1.0], 1e6,
                              scheme=predictor)
    assert driver._scheme is predictor


def test_constructor_accepts_scheme_string(simple_model, chain_file):
    driver = DepletionDriver(simple_model, chain_file, [1.0], 1e6,
                              scheme='cecm')
    assert driver._scheme.name == 'cecm'


def test_constructor_scalar_source_rate(simple_model, chain_file):
    driver = DepletionDriver(simple_model, chain_file, [1.0, 2.0], 1e6)
    np.testing.assert_array_equal(driver._source_rates, [1e6, 1e6])


def test_constructor_timestep_units(simple_model, chain_file):
    driver = DepletionDriver(simple_model, chain_file, [1.0], 1e6,
                              timestep_units='d')
    np.testing.assert_allclose(driver._timesteps_s, [86400.0])


# ---------------------------------------------------------------------------
# Helper method tests (static/classmethod)
# ---------------------------------------------------------------------------

def test_resolve_density_bos():
    densities = {BOS: 'bos_data'}
    assert DepletionDriver._resolve_density(BOS, densities) == 'bos_data'


def test_resolve_density_expm_node():
    node = Expm((MatrixTerm(0.5, Transport(BOS)),), BOS)
    densities = {node: 'expm_result'}
    assert DepletionDriver._resolve_density(node, densities) == 'expm_result'


def test_resolve_density_missing():
    with pytest.raises(KeyError):
        DepletionDriver._resolve_density('missing', {})


def test_find_last_expm_simple():
    t = Transport(BOS)
    e1 = Expm((MatrixTerm(1.0, t),), BOS)
    e2 = Expm((MatrixTerm(1.0, t),), e1)
    assert DepletionDriver._find_last_expm([t, e1, e2]) is e2


def test_find_last_expm_in_iterate():
    t = Transport(BOS)
    e_outer = Expm((MatrixTerm(1.0, t),), BOS)
    e_inner = Expm((MatrixTerm(1.0, t),), BOS)
    iterate = Iterate(n_iterations=3, body=(Transport(PREV_ITER), e_inner))
    assert DepletionDriver._find_last_expm([t, e_outer, iterate]) is e_inner


def test_find_last_expm_none():
    t = Transport(BOS)
    assert DepletionDriver._find_last_expm([t]) is None
