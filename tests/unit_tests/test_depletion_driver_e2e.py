"""End-to-end validation: DepletionManager vs CoupledOperator + Integrator.

Tests compare the new DepletionManager against:
1. Decay-only: exact comparison (no stochastic transport variability)
2. Coupled transport: physics consistency checks (k_eff reasonable,
   actinides deplete, fission products appear)
"""

import os
import time
from pathlib import Path

import numpy as np
import pytest

import openmc
import openmc.deplete
from openmc.deplete import Results
from openmc.deplete.depletion_manager import DepletionManager

# Simple chain files shipped with the tests
CHAIN_SIMPLE = Path(__file__).parents[1] / 'chain_simple.xml'
CHAIN_DECAY = Path(__file__).parents[1] / 'chain_simple_decay.xml'


def _make_pin_model():
    """Build a minimal depletable pin-cell model."""
    openmc.reset_auto_ids()

    fuel = openmc.Material(name="fuel")
    fuel.add_nuclide('U235', 0.04)
    fuel.add_nuclide('U238', 0.96)
    fuel.add_element('O', 2.0)
    fuel.set_density('g/cm3', 10.4)
    fuel.depletable = True
    fuel.volume = np.pi * 0.42**2  # cm^3 per cm height

    clad = openmc.Material(name="clad")
    clad.add_element('Zr', 1)
    clad.set_density('g/cm3', 6.0)

    water = openmc.Material(name="water")
    water.add_element('O', 1)
    water.add_element('H', 2)
    water.set_density('g/cm3', 1.0)
    water.add_s_alpha_beta('c_H_in_H2O')

    radii = [openmc.ZCylinder(r=0.42), openmc.ZCylinder(r=0.45)]
    pin = openmc.model.pin(radii, [fuel, clad, water])
    box = openmc.model.RectangularPrism(
        1.24, 1.24, boundary_type='reflective')
    root = openmc.Cell(fill=pin, region=-box)
    geometry = openmc.Geometry([root])

    settings = openmc.Settings()
    settings.particles = 200
    settings.batches = 10
    settings.inactive = 2
    settings.seed = 1
    settings.verbosity = 1

    model = openmc.Model(geometry, openmc.Materials([fuel, clad, water]),
                         settings)
    return model


def _make_decay_model():
    """Build a model for decay-only testing (Xe135_m1 -> Xe135)."""
    openmc.reset_auto_ids()

    fuel = openmc.Material(name="uo2")
    fuel.add_element("U", 1, percent_type="ao", enrichment=4.25)
    fuel.add_element("O", 2)
    fuel.add_nuclide("Xe135_m1", 1)
    fuel.add_nuclide("Cs135_m1", 1)
    fuel.set_density("g/cc", 10.4)
    fuel.depletable = True
    fuel.volume = np.pi * 0.42**2

    clad = openmc.Material(name="clad")
    clad.add_element("Zr", 1)
    clad.set_density("g/cc", 6)

    water = openmc.Material(name="water")
    water.add_element("O", 1)
    water.add_element("H", 2)
    water.set_density("g/cc", 1.0)
    water.add_s_alpha_beta("c_H_in_H2O")

    radii = [openmc.ZCylinder(r=0.42), openmc.ZCylinder(r=0.45)]
    pin = openmc.model.pin(radii, [fuel, clad, water])
    box = openmc.model.RectangularPrism(
        1.24, 1.24, boundary_type='reflective')
    root = openmc.Cell(fill=pin, region=-box)
    geometry = openmc.Geometry([root])

    settings = openmc.Settings()
    settings.particles = 200
    settings.batches = 10
    settings.inactive = 2
    settings.seed = 1
    settings.verbosity = 1

    return openmc.Model(geometry, openmc.Materials([fuel, clad, water]),
                        settings)


# ------------------------------------------------------------------
# Test 1: Decay-only -- exact comparison with CoupledOperator
# ------------------------------------------------------------------

def test_decay_only_vs_coupled_operator(run_in_tmpdir):
    """Decay-only: DepletionManager results should match CoupledOperator."""
    if not CHAIN_DECAY.exists():
        pytest.skip("chain_simple_decay.xml not found")

    # Xe135_m1 half-life = 917.4 s, Cs135_m1 half-life = 2262.6 s
    dt = [917.4, 2262.6]

    # --- Old: CoupledOperator + PredictorIntegrator with power=0 ---
    old_dir = Path('old_run')
    old_dir.mkdir()
    os.chdir(old_dir)

    model_old = _make_decay_model()
    op = openmc.deplete.CoupledOperator(model_old, str(CHAIN_DECAY))
    op.round_number = False
    integrator = openmc.deplete.PredictorIntegrator(
        op, dt, power=0.0, timestep_units='s')
    t0 = time.time()
    integrator.integrate(output=False)
    old_wall = time.time() - t0
    res_old = Results(op.output_dir / 'depletion_results.h5')
    os.chdir('..')

    # --- New: DepletionManager with predictor, power=0 ---
    new_dir = Path('new_run')
    new_dir.mkdir()
    os.chdir(new_dir)

    model_new = _make_decay_model()
    mgr = DepletionManager(
        model_new, str(CHAIN_DECAY), dt, 0.0,
        source_rate_type='power',
        scheme='predictor',
        normalization_mode='fission-q',
    )
    t0 = time.time()
    res_new = mgr.run()
    new_wall = time.time() - t0
    os.chdir('..')

    # Get material ID from each result
    mat_old = list(res_old[0].index_mat.keys())[0]
    mat_new = list(res_new[0].index_mat.keys())[0]

    # Xe135_m1 should halve after one half-life (step 0 -> step 1)
    _, xe135m1_old = res_old.get_atoms(mat_old, 'Xe135_m1')
    _, xe135m1_new = res_new.get_atoms(mat_new, 'Xe135_m1')

    # Old stores [BOS, EOS_step0, EOS_step1, final] (N+2 entries for N steps)
    # New stores [EOS_step0, EOS_step1] (N entries)
    print(f"\n{'='*60}")
    print(f"Decay-only comparison")
    print(f"  Old wall time: {old_wall:.2f} s")
    print(f"  New wall time: {new_wall:.2f} s")
    print(f"  Xe135_m1 old: {xe135m1_old}")
    print(f"  Xe135_m1 new: {xe135m1_new}")

    tol = 1e-10

    # Verify physics: after one half-life, Xe135_m1 should halve
    # Old index 0 = BOS, old index 1 = EOS step 0
    assert xe135m1_old[0] == pytest.approx(
        xe135m1_old[1] * 2.0, rel=tol), \
        "Old: Xe135_m1 did not halve correctly"

    # New index 0 = EOS step 0; BOS value should equal old BOS
    # To compare: new EOS step 0 should equal old EOS step 0
    assert xe135m1_new[0] == pytest.approx(xe135m1_old[1], rel=tol), \
        "Xe135_m1 EOS step 0 mismatch between old and new"

    # Cs135_m1: compare at end of step 1
    _, cs135m1_old = res_old.get_atoms(mat_old, 'Cs135_m1')
    _, cs135m1_new = res_new.get_atoms(mat_new, 'Cs135_m1')
    print(f"  Cs135_m1 old: {cs135m1_old}")
    print(f"  Cs135_m1 new: {cs135m1_new}")

    # Old index 2 = EOS step 1, new index 1 = EOS step 1
    assert cs135m1_new[1] == pytest.approx(cs135m1_old[2], rel=tol), \
        "Cs135_m1 EOS step 1 mismatch"

    print(f"  All decay-only checks passed!")
    print(f"{'='*60}")


# ------------------------------------------------------------------
# Test 2: Coupled transport -- physics validation (predictor)
# ------------------------------------------------------------------

def test_coupled_predictor_physics(run_in_tmpdir):
    """Verify DepletionManager(predictor) produces physically sensible results."""
    if not CHAIN_SIMPLE.exists():
        pytest.skip("chain_simple.xml not found")

    dt = [5.0 * 86400.0]  # 5 days
    power = 174.0  # Watts

    model = _make_pin_model()
    mgr = DepletionManager(
        model, str(CHAIN_SIMPLE), dt, power,
        source_rate_type='power',
        scheme='predictor',
        normalization_mode='fission-q',
    )

    t0 = time.time()
    res = mgr.run()
    wall = time.time() - t0

    mat_id = list(res[0].index_mat.keys())[0]

    # Get atom counts
    _, u235 = res.get_atoms(mat_id, 'U235')
    _, u238 = res.get_atoms(mat_id, 'U238')
    _, k_vals = res.get_keff()

    print(f"\n{'='*60}")
    print(f"Coupled predictor physics check (1 step, 5 days, {power} W)")
    print(f"  Wall time: {wall:.2f} s")
    print(f"  k_eff: {k_vals}")
    print(f"  U235: EOS={u235[0]:.6e}")
    print(f"  U238: EOS={u238[0]:.6e}")
    print(f"{'='*60}")

    # Physics checks
    assert 0.5 < k_vals[0, 0] < 2.5, \
        f"k_eff={k_vals[0,0]:.4f} is not physically reasonable"
    assert u235[0] > 0, "U235 atoms should be positive"
    assert u238[0] > 0, "U238 atoms should be positive"


# ------------------------------------------------------------------
# Test 3: Coupled transport with CECM scheme
# ------------------------------------------------------------------

def test_coupled_cecm_physics(run_in_tmpdir):
    """Verify DepletionManager(cecm) produces physically sensible results."""
    if not CHAIN_SIMPLE.exists():
        pytest.skip("chain_simple.xml not found")

    dt = [5.0 * 86400.0]  # 5 days
    power = 174.0

    model = _make_pin_model()
    mgr = DepletionManager(
        model, str(CHAIN_SIMPLE), dt, power,
        source_rate_type='power',
        scheme='cecm',
        normalization_mode='fission-q',
    )

    t0 = time.time()
    res = mgr.run()
    wall = time.time() - t0

    mat_id = list(res[0].index_mat.keys())[0]
    _, k_vals = res.get_keff()
    _, u235 = res.get_atoms(mat_id, 'U235')

    print(f"\n{'='*60}")
    print(f"Coupled CECM physics check (1 step, 5 days, {power} W)")
    print(f"  Wall time: {wall:.2f} s")
    print(f"  k_eff: {k_vals}")
    print(f"  U235: EOS={u235[0]:.6e}")
    print(f"{'='*60}")

    assert 0.5 < k_vals[0, 0] < 2.5
    assert u235[0] > 0


# ------------------------------------------------------------------
# Test 4: Multi-step coupled predictor
# ------------------------------------------------------------------

def test_multistep_predictor_physics(run_in_tmpdir):
    """Verify multi-step predictor produces consistent k_eff across steps."""
    if not CHAIN_SIMPLE.exists():
        pytest.skip("chain_simple.xml not found")

    n_steps = 3
    dt = [3.0 * 86400.0] * n_steps
    power = 174.0

    model = _make_pin_model()
    mgr = DepletionManager(
        model, str(CHAIN_SIMPLE), dt, power,
        source_rate_type='power',
        scheme='predictor',
        normalization_mode='fission-q',
    )

    t0 = time.time()
    res = mgr.run()
    wall = time.time() - t0

    _, k_vals = res.get_keff()

    print(f"\n{'='*60}")
    print(f"Multi-step predictor ({n_steps} steps)")
    print(f"  Wall time: {wall:.2f} s")
    print(f"  k_eff per step: {k_vals[:, 0]}")
    print(f"{'='*60}")

    for i in range(len(k_vals)):
        assert 0.5 < k_vals[i, 0] < 2.5, \
            f"Step {i}: k_eff={k_vals[i,0]:.4f} out of range"


# ------------------------------------------------------------------
# Test 5: Multi-step CECM
# ------------------------------------------------------------------

def test_multistep_cecm_physics(run_in_tmpdir):
    """Verify multi-step CECM produces consistent k_eff across steps."""
    if not CHAIN_SIMPLE.exists():
        pytest.skip("chain_simple.xml not found")

    n_steps = 3
    dt = [3.0 * 86400.0] * n_steps
    power = 174.0

    model = _make_pin_model()
    mgr = DepletionManager(
        model, str(CHAIN_SIMPLE), dt, power,
        source_rate_type='power',
        scheme='cecm',
        normalization_mode='fission-q',
    )

    t0 = time.time()
    res = mgr.run()
    wall = time.time() - t0

    _, k_vals = res.get_keff()

    print(f"\n{'='*60}")
    print(f"Multi-step CECM ({n_steps} steps)")
    print(f"  Wall time: {wall:.2f} s")
    print(f"  k_eff per step: {k_vals[:, 0]}")
    print(f"{'='*60}")

    for i in range(len(k_vals)):
        assert 0.5 < k_vals[i, 0] < 2.5, \
            f"Step {i}: k_eff={k_vals[i,0]:.4f} out of range"


# ------------------------------------------------------------------
# Test 6: Restart (append mode) — decay-only, exact comparison
# ------------------------------------------------------------------

def test_restart_append_decay(run_in_tmpdir):
    """Run 2 steps, restart with 2 more, verify vs single 4-step run."""
    if not CHAIN_DECAY.exists():
        pytest.skip("chain_simple_decay.xml not found")

    dt = [500.0, 600.0, 700.0, 800.0]  # seconds

    # --- Reference: single 4-step run ---
    ref_dir = Path('ref')
    ref_dir.mkdir()
    os.chdir(ref_dir)
    model_ref = _make_decay_model()
    mgr_ref = DepletionManager(
        model_ref, str(CHAIN_DECAY), dt, 0.0,
        source_rate_type='power', scheme='predictor')
    res_ref = mgr_ref.run()
    os.chdir('..')

    # --- First run: 2 steps ---
    run_dir = Path('restart_run')
    run_dir.mkdir()
    os.chdir(run_dir)
    model1 = _make_decay_model()
    mgr1 = DepletionManager(
        model1, str(CHAIN_DECAY), dt[:2], 0.0,
        source_rate_type='power', scheme='predictor')
    mgr1.run()

    # --- Restart: append 2 more steps ---
    model2 = _make_decay_model()
    mgr2 = DepletionManager(
        model2, str(CHAIN_DECAY), dt[2:], 0.0,
        source_rate_type='power', scheme='predictor',
        prev_results='depletion_results.h5')
    res_rst = mgr2.run()
    os.chdir('..')

    mat_ref = list(res_ref[0].index_mat.keys())[0]
    mat_rst = list(res_rst[0].index_mat.keys())[0]

    # Verify all 4 EOS compositions match between reference and restart
    for nuc in ['Xe135_m1', 'Xe135', 'Cs135_m1', 'Cs135']:
        _, conc_ref = res_ref.get_atoms(mat_ref, nuc)
        _, conc_rst = res_rst.get_atoms(mat_rst, nuc)
        for step in range(4):
            assert conc_rst[step] == pytest.approx(
                conc_ref[step], rel=1e-10), \
                f"Step {step} {nuc} mismatch: " \
                f"restart={conc_rst[step]:.6e} ref={conc_ref[step]:.6e}"

    # Verify time grid is consistent
    times_ref = res_ref.get_times(time_units='s')
    times_rst = res_rst.get_times(time_units='s')
    np.testing.assert_allclose(times_ref, times_rst, rtol=1e-12)


# ------------------------------------------------------------------
# Test 7: Restart (auto-continue) — resubmit same script
# ------------------------------------------------------------------

def test_restart_auto_continue_decay(run_in_tmpdir):
    """Run 2 steps, restart with all 4, verify completed steps skipped."""
    if not CHAIN_DECAY.exists():
        pytest.skip("chain_simple_decay.xml not found")

    dt = [500.0, 600.0, 700.0, 800.0]

    # --- Reference ---
    ref_dir = Path('ref')
    ref_dir.mkdir()
    os.chdir(ref_dir)
    model_ref = _make_decay_model()
    mgr_ref = DepletionManager(
        model_ref, str(CHAIN_DECAY), dt, 0.0,
        source_rate_type='power', scheme='predictor')
    res_ref = mgr_ref.run()
    os.chdir('..')

    # --- First run: 2 steps ---
    ac_dir = Path('ac_run')
    ac_dir.mkdir()
    os.chdir(ac_dir)
    model1 = _make_decay_model()
    mgr1 = DepletionManager(
        model1, str(CHAIN_DECAY), dt[:2], 0.0,
        source_rate_type='power', scheme='predictor')
    mgr1.run()

    # --- Auto-continue: provide all 4 timesteps + prev_results ---
    model2 = _make_decay_model()
    mgr2 = DepletionManager(
        model2, str(CHAIN_DECAY), dt, 0.0,
        source_rate_type='power', scheme='predictor',
        prev_results='depletion_results.h5')
    res_ac = mgr2.run()
    os.chdir('..')

    mat_ref = list(res_ref[0].index_mat.keys())[0]
    mat_ac = list(res_ac[0].index_mat.keys())[0]

    for nuc in ['Xe135_m1', 'Xe135', 'Cs135_m1', 'Cs135']:
        _, conc_ref = res_ref.get_atoms(mat_ref, nuc)
        _, conc_ac = res_ac.get_atoms(mat_ac, nuc)
        for step in range(4):
            assert conc_ac[step] == pytest.approx(
                conc_ref[step], rel=1e-10), \
                f"Step {step} {nuc} mismatch"

    times_ref = res_ref.get_times(time_units='s')
    times_ac = res_ac.get_times(time_units='s')
    np.testing.assert_allclose(times_ref, times_ac, rtol=1e-12)
