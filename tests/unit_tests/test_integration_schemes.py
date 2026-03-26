"""Tests for the integration scheme registry."""

from openmc.deplete.integration_schemes import SCHEMES, FALLBACK_SCHEMES


def test_schemes_is_frozenset():
    assert isinstance(SCHEMES, frozenset)


def test_expected_schemes_present():
    expected = {
        'predictor', 'cecm', 'celi', 'cf4',
        'epc_rk4', 'leqi', 'si_celi', 'si_leqi',
    }
    assert SCHEMES == expected


def test_fallback_schemes():
    assert FALLBACK_SCHEMES == {'leqi': 'celi', 'si_leqi': 'si_celi'}
    for fallback in FALLBACK_SCHEMES.values():
        assert fallback in SCHEMES
