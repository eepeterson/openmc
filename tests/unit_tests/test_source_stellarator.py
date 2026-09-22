import numpy as np
import pytest

import openmc
import openmc.stats

from tests.unit_tests import assert_sample_mean


def make_source(**kwargs):
    """Return a circular-torus StellaratorSource with optional VMEC field data."""
    rho = np.linspace(0.0, 1.0, 5)
    # The third mode has zero geometry coefficient but carries a genuinely 3-D
    # lambda harmonic in polarized tests.
    mode_m = np.array([0, 1, 1])
    mode_n = np.array([0, 0, 1])
    rmnc = np.zeros((len(rho), len(mode_m)))
    zmns = np.zeros_like(rmnc)
    rmnc[:, 0] = 600.0
    rmnc[:, 1] = 150.0 * rho
    zmns[:, 1] = 150.0 * rho
    params = dict(
        rho=rho,
        emission_density=1.0 - 0.5 * rho**2,
        mode_m=mode_m,
        mode_n=mode_n,
        rmnc=rmnc,
        zmns=zmns,
        num_field_periods=2,
        energy=openmc.stats.delta_function(14.07e6),
    )
    params.update(kwargs)
    return openmc.StellaratorSource(**params)


def polarized_field_data():
    field_rho = np.array([0.1, 0.4, 0.7, 0.95])
    iota = 0.35 + 0.1 * field_rho
    lmns = np.zeros((len(field_rho), 3))
    lmns[:, 2] = 0.04 + 0.06 * field_rho
    return field_rho, iota, lmns


def test_stellarator_source_unpolarized_xml_unchanged():
    elem = make_source().to_xml_element()
    for name in ('polarization', 'field_rho', 'iota', 'lmns', 'lmnc'):
        assert elem.find(name) is None


def test_stellarator_source_polarization_roundtrip():
    field_rho, iota, lmns = polarized_field_data()
    src = make_source(
        polarization=(0.5, 0.3, 0.2), field_rho=field_rho, iota=iota,
        lmns=lmns)

    new = openmc.SourceBase.from_xml_element(src.to_xml_element())
    assert isinstance(new, openmc.StellaratorSource)
    assert new.polarization == pytest.approx(src.polarization)
    np.testing.assert_allclose(new.field_rho, field_rho)
    np.testing.assert_allclose(new.iota, iota)
    np.testing.assert_allclose(new.lmns, lmns)
    assert new.lmnc is None


def test_stellarator_source_from_vmec_uses_native_half_mesh(monkeypatch):
    """Classic wout lambda/iota row zero is padding, not an axis value."""
    ns = 5
    rho = np.sqrt(np.linspace(0.0, 1.0, ns))
    rmnc = np.zeros((ns, 2))
    zmns = np.zeros_like(rmnc)
    rmnc[:, 0] = 6.0
    rmnc[:, 1] = 1.5 * rho
    zmns[:, 1] = 1.5 * rho
    lmns = np.zeros((ns, 2))
    lmns[1:, 1] = [0.01, 0.02, 0.03, 0.04]
    data = dict(
        rmnc=rmnc, zmns=zmns, xm=np.array([0, 1]),
        xn=np.array([0, 0]), nfp=np.asarray(1),
        lasym__logical__=np.asarray(0), lmns=lmns,
        iotas=np.array([0.0, 0.31, 0.32, 0.33, 0.34]),
        iotaf=np.linspace(0.3, 0.35, ns),
    )
    monkeypatch.setattr(openmc.StellaratorSource, '_read_wout',
                        staticmethod(lambda path: data))

    src = openmc.StellaratorSource.from_vmec(
        'unused.nc', emission_density=np.ones(ns),
        energy=openmc.stats.delta_function(14.07e6),
        polarization=(1, 0, 0))
    expected_rho = np.sqrt((np.arange(1, ns) - 0.5) / (ns - 1))
    np.testing.assert_allclose(src.field_rho, expected_rho)
    np.testing.assert_allclose(src.iota, data['iotas'][1:])
    np.testing.assert_allclose(src.lmns, lmns[1:])


@pytest.mark.parametrize("kwargs, match", [
    (dict(polarization=(1, 0, 0)), "requires field_rho"),
    (dict(field_rho=[0.1, 0.9], iota=[0.4, 0.4],
          lmns=np.zeros((2, 3))), "only used with polarization"),
    (dict(polarization=(1, 0, 0), field_rho=[0.1, 0.9],
          iota=[0.4], lmns=np.zeros((2, 3))),
     "iota and field_rho"),
    (dict(polarization=(1, 0, 0), field_rho=[0.1, 0.9],
          iota=[0.4, 0.4], lmns=np.zeros((2, 2))), "lmns must have shape"),
])
def test_stellarator_source_invalid_field_data(kwargs, match):
    with pytest.raises(ValueError, match=match):
        make_source(**kwargs)


def test_stellarator_from_desc_lambda_iota_helpers(run_in_tmpdir):
    """DESC lambda/iota extraction helpers exist and read the documented HDF5 layout.

    The full field-direction convention is validated against DESC's own
    compute('B') in spf_validation/validate_desc_bhat.py (needs the desc package,
    so it is not part of this unit suite); here we exercise the HDF5 iota reader,
    which must evaluate a stored PowerSeriesProfile as a plain polynomial in rho.
    """
    import h5py
    assert hasattr(openmc.StellaratorSource, '_desc_lambda_data')
    assert hasattr(openmc.StellaratorSource, '_desc_iota')
    # synthetic DESC-style HDF5 with an even-power PowerSeriesProfile iota, i.e.
    # iota(rho) = 0.4 + 0.1 rho**2 - 0.05 rho**4 (powers from the basis modes,
    # not 0,1,2 -- this is what the real DESC layout uses).
    path = 'fake_desc.h5'
    with h5py.File(path, 'w') as f:
        g = f.create_group('_iota')
        g.create_dataset('_params', data=np.array([0.4, 0.1, -0.05]))
        b = g.create_group('_basis')
        b.create_dataset('_modes', data=np.array([[0, 0, 0], [2, 0, 0], [4, 0, 0]]))
    rho = np.array([0.0, 0.5, 1.0])
    iota = openmc.StellaratorSource._desc_iota(path, rho)
    np.testing.assert_allclose(iota, 0.4 + 0.1 * rho**2 - 0.05 * rho**4)


def _interpolate_clamped(x, xp, values):
    """Linear interpolation with the same endpoint clamping as the C++ path."""
    return np.interp(x, xp, values)


@pytest.mark.parametrize(("polarization", "expected_cos2"), [
    ((1.0, 0.0, 0.0), 1.0 / 5.0),
    ((1.0, 1.0, 1.0), 1.0 / 3.0),
    ((0.0, 1.0, 0.0), 7.0 / 15.0),
])
def test_stellarator_source_polarized_direction_sampling(
    run_in_tmpdir, polarization, expected_cos2
):
    """Check the compiled SPF sampler about an independently rebuilt B-hat."""
    R0, a, nfp = 600.0, 150.0, 2
    field_rho, iota_grid, lmns = polarized_field_data()
    src = make_source(
        polarization=polarization, field_rho=field_rho, iota=iota_grid,
        lmns=lmns)

    sphere = openmc.Sphere(r=2000.0, boundary_type='vacuum')
    model = openmc.Model(
        geometry=openmc.Geometry([openmc.Cell(region=-sphere)]),
        settings=openmc.Settings(
            particles=100, batches=1, run_mode='fixed source', source=src),
    )
    sites = model.sample_external_source(30_000)
    xyz = np.array([site.r for site in sites])
    directions = np.array([site.u for site in sites])

    zeta = np.arctan2(xyz[:, 1], xyz[:, 0])
    major_r = np.hypot(xyz[:, 0], xyz[:, 1])
    theta = np.arctan2(xyz[:, 2], major_r - R0)
    rho = np.hypot(major_r - R0, xyz[:, 2]) / a

    iota = _interpolate_clamped(rho, field_rho, iota_grid)
    lambda_coeff = _interpolate_clamped(rho, field_rho, lmns[:, 2])
    phase = theta - nfp * zeta
    lambda_theta = lambda_coeff * np.cos(phase)
    lambda_zeta = -nfp * lambda_coeff * np.cos(phase)
    b_theta = iota - lambda_zeta
    b_zeta = 1.0 + lambda_theta

    # Circular-torus coordinate basis, evaluated independently of the C++
    # Fourier derivative implementation.
    e_r = np.column_stack((np.cos(zeta), np.sin(zeta), np.zeros_like(zeta)))
    e_phi = np.column_stack((-np.sin(zeta), np.cos(zeta), np.zeros_like(zeta)))
    e_z = np.column_stack((np.zeros_like(zeta), np.zeros_like(zeta),
                           np.ones_like(zeta)))
    e_theta = (-a * rho * np.sin(theta))[:, None] * e_r \
        + (a * rho * np.cos(theta))[:, None] * e_z
    e_zeta = major_r[:, None] * e_phi
    field = b_theta[:, None] * e_theta + b_zeta[:, None] * e_zeta
    field /= np.linalg.norm(field, axis=1)[:, None]

    cos2 = np.einsum('ij,ij->i', directions, field)**2
    assert_sample_mean(cos2, expected_cos2)
    np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1.0,
                               rtol=0.0, atol=2e-14)
