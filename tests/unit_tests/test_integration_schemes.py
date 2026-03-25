import pytest

from openmc.deplete.integration_schemes import (
    DepletionStage, IntegratorScheme, SCHEMES,
    predictor, cecm, celi, cf4, epc_rk4, leqi, si_celi, si_leqi,
    _leqi_f1, _leqi_f2, _leqi_f3, _leqi_f4,
)


def test_schemes_registry():
    """All pre-built schemes appear in the SCHEMES registry."""
    assert len(SCHEMES) == 8
    for name in ('predictor', 'cecm', 'celi', 'cf4', 'epc_rk4',
                 'leqi', 'si_celi', 'si_leqi'):
        assert name in SCHEMES


def test_predictor_structure():
    assert len(predictor.stages) == 1
    assert predictor.num_evaluations == 0
    assert predictor.corrector_start is None
    assert predictor.n_iterations == 1


def test_predictor_weights():
    assert predictor.stages[0].weights == pytest.approx((1.0,))
    assert predictor.stages[0].source == 0
    assert predictor.stages[0].evaluate is False


def test_cecm_structure():
    assert len(cecm.stages) == 2
    assert cecm.num_evaluations == 1
    assert cecm.corrector_start == 1


def test_cecm_weights():
    assert cecm.stages[0].weights == pytest.approx((0.5,))
    assert cecm.stages[0].evaluate is True
    assert cecm.stages[1].weights == pytest.approx((0.0, 1.0))
    assert cecm.stages[1].evaluate is False


def test_celi_structure():
    assert len(celi.stages) == 3
    assert celi.num_evaluations == 1
    assert celi.corrector_start == 1


def test_celi_weights():
    # Forward Euler predictor
    assert celi.stages[0].weights == pytest.approx((1.0,))
    assert celi.stages[0].evaluate is True
    # celi_f1: 5/12 * A(rate_0) + 1/12 * A(rate_1)
    assert celi.stages[1].weights == pytest.approx((5/12, 1/12))
    assert celi.stages[1].source == 0
    # celi_f2: 1/12 * A(rate_0) + 5/12 * A(rate_1)
    assert celi.stages[2].weights == pytest.approx((1/12, 5/12))
    assert celi.stages[2].source == 2  # chains from stage 1


def test_celi_corrector_weights_sum():
    """Each corrector exponential integrates half the step."""
    assert sum(celi.stages[1].weights) == pytest.approx(0.5)
    assert sum(celi.stages[2].weights) == pytest.approx(0.5)


def test_cf4_structure():
    assert len(cf4.stages) == 5
    assert cf4.num_evaluations == 3
    assert cf4.corrector_start == 3


def test_cf4_sources():
    assert cf4.stages[0].source == 0
    assert cf4.stages[1].source == 0
    assert cf4.stages[2].source == 1  # chains from stage 0 output
    assert cf4.stages[3].source == 0
    assert cf4.stages[4].source == 4  # chains from stage 3 output


def test_cf4_weights():
    assert cf4.stages[0].weights == pytest.approx((0.5,))
    assert cf4.stages[1].weights == pytest.approx((0.0, 0.5))
    assert cf4.stages[2].weights == pytest.approx((-0.5, 0.0, 1.0))
    assert cf4.stages[3].weights == pytest.approx((1/4, 1/6, 1/6, -1/12))
    assert cf4.stages[4].weights == pytest.approx((-1/12, 1/6, 1/6, 1/4))


def test_cf4_final_weights_sum():
    """cf4_f3 + cf4_f4 weights sum to 1.0 (full step)."""
    total = sum(cf4.stages[3].weights) + sum(cf4.stages[4].weights)
    assert total == pytest.approx(1.0)


def test_epc_rk4_structure():
    assert len(epc_rk4.stages) == 4
    assert epc_rk4.num_evaluations == 3
    assert epc_rk4.corrector_start == 3


def test_epc_rk4_weights():
    assert epc_rk4.stages[0].weights == pytest.approx((0.5,))
    assert epc_rk4.stages[1].weights == pytest.approx((0.0, 0.5))
    assert epc_rk4.stages[2].weights == pytest.approx((0.0, 0.0, 1.0))
    # Classical RK4 weights: 1/6, 1/3, 1/3, 1/6
    assert epc_rk4.stages[3].weights == pytest.approx((1/6, 1/3, 1/3, 1/6))


def test_epc_rk4_weights_sum():
    assert sum(epc_rk4.stages[3].weights) == pytest.approx(1.0)


def test_leqi_structure():
    assert len(leqi.stages) == 4
    assert leqi.num_evaluations == 1
    assert leqi.uses_prev_rates is True
    assert leqi.corrector_start == 2


def test_leqi_weights_are_callable():
    for stage in leqi.stages:
        assert callable(stage.weights)


def test_leqi_f1_equal_dt():
    w = _leqi_f1(1.0, 1.0)
    assert len(w) == 2
    assert w[0] == pytest.approx(-1/12)
    assert w[1] == pytest.approx(7/12)
    assert sum(w) == pytest.approx(0.5)


def test_leqi_f2_equal_dt():
    w = _leqi_f2(1.0, 1.0)
    assert w[0] == pytest.approx(-5/12)
    assert w[1] == pytest.approx(11/12)
    assert sum(w) == pytest.approx(0.5)


def test_leqi_f3_equal_dt():
    w = _leqi_f3(1.0, 1.0)
    assert len(w) == 3
    assert w[0] == pytest.approx(-1/24)
    assert w[1] == pytest.approx(12/24)
    assert w[2] == pytest.approx(1/24)
    assert sum(w) == pytest.approx(0.5)


def test_leqi_f4_equal_dt():
    w = _leqi_f4(1.0, 1.0)
    assert len(w) == 3
    assert w[0] == pytest.approx(-1/24)
    assert w[1] == pytest.approx(4/24)
    assert w[2] == pytest.approx(9/24)
    assert sum(w) == pytest.approx(0.5)


def test_leqi_predictor_weights_sum():
    """LE predictor exponentials should each sum to 0.5 for equal dt."""
    for func in (_leqi_f1, _leqi_f2):
        assert sum(func(1.0, 1.0)) == pytest.approx(0.5)


def test_leqi_corrector_weights_sum():
    """QI corrector exponentials should each sum to 0.5 for equal dt."""
    for func in (_leqi_f3, _leqi_f4):
        assert sum(func(1.0, 1.0)) == pytest.approx(0.5)


def test_si_celi_shares_stages():
    assert si_celi.stages == celi.stages


def test_si_celi_params():
    assert si_celi.n_iterations == 10
    assert si_celi.corrector_start == 1
    assert si_celi.uses_prev_rates is False


def test_si_leqi_shares_stages():
    assert si_leqi.stages == leqi.stages


def test_si_leqi_params():
    assert si_leqi.n_iterations == 10
    assert si_leqi.corrector_start == 2
    assert si_leqi.uses_prev_rates is True


def test_si_requires_corrector():
    """Predictor has no corrector stages -- can't use SI."""
    with pytest.raises(ValueError, match="no corrector"):
        IntegratorScheme(
            name='si_predictor',
            stages=predictor.stages,
            n_iterations=10,
        )


def test_predictor_does_not_use_prev_rates():
    """Schemes with only constant weights don't need previous rates."""
    assert predictor.uses_prev_rates is False
    assert cecm.uses_prev_rates is False
    assert celi.uses_prev_rates is False
    assert cf4.uses_prev_rates is False
    assert epc_rk4.uses_prev_rates is False


def test_frozen():
    """Scheme instances are immutable."""
    with pytest.raises(AttributeError):
        predictor.name = 'something_else'
    with pytest.raises(AttributeError):
        predictor.stages[0].source = 99
