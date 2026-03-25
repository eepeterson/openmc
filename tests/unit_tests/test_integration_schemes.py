import pytest

from openmc.deplete.integration_schemes import (
    BOS, PREV_STEP, PREV_ITER,
    Transport, Expm, MatrixTerm, AverageMatrix, Iterate,
    IntegrationScheme, SCHEMES,
    predictor, cecm, celi, cf4, epc_rk4, leqi, si_celi, si_leqi,
    _leqi_w1_prev, _leqi_w1_bos, _leqi_w2_prev, _leqi_w2_bos,
    _leqi_w3_prev, _leqi_w3_bos, _leqi_w3_eos,
    _leqi_w4_prev, _leqi_w4_bos, _leqi_w4_eos,
)


# ---------------------------------------------------------------------------
# Sentinel tests
# ---------------------------------------------------------------------------

def test_sentinels_repr():
    assert repr(BOS) == 'BOS'
    assert repr(PREV_STEP) == 'PREV_STEP'
    assert repr(PREV_ITER) == 'PREV_ITER'


# ---------------------------------------------------------------------------
# Node type tests
# ---------------------------------------------------------------------------

def test_transport_rmul():
    """``0.5 * Transport(BOS)`` produces a MatrixTerm."""
    t = Transport(BOS)
    term = 0.5 * t
    assert isinstance(term, MatrixTerm)
    assert term.weight == 0.5
    assert term.matrix is t


def test_average_matrix_rmul():
    """``0.5 * AverageMatrix(...)`` produces a MatrixTerm."""
    t = Transport(PREV_ITER)
    avg = AverageMatrix(t)
    term = 0.5 * avg
    assert isinstance(term, MatrixTerm)
    assert term.weight == 0.5
    assert term.matrix is avg


def test_transport_density_ref():
    t = Transport(BOS)
    assert t.density is BOS
    e = Expm((1.0 * t,), BOS)
    t2 = Transport(e)
    assert t2.density is e


def test_expm_terms_and_density():
    t = Transport(BOS)
    e = Expm((0.5 * t, 0.3 * t), BOS)
    assert len(e.terms) == 2
    assert e.terms[0].weight == pytest.approx(0.5)
    assert e.terms[1].weight == pytest.approx(0.3)
    assert e.density is BOS


def test_iterate_structure():
    t = Transport(PREV_ITER)
    avg = AverageMatrix(t)
    e = Expm((1.0 * avg,), BOS)
    it = Iterate(n_iterations=11, body=(t, e))
    assert it.n_iterations == 11
    assert len(it.body) == 2
    assert isinstance(it.body[0], Transport)
    assert isinstance(it.body[1], Expm)


def test_callable_weight_in_matrix_term():
    f = lambda h_l, h: -h / (12 * h_l)
    term = MatrixTerm(f, PREV_STEP)
    assert callable(term.weight)
    assert term.weight(1.0, 1.0) == pytest.approx(-1/12)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_schemes_registry():
    assert len(SCHEMES) == 8
    for name in ('predictor', 'cecm', 'celi', 'cf4', 'epc_rk4',
                 'leqi', 'si_celi', 'si_leqi'):
        assert name in SCHEMES


# ---------------------------------------------------------------------------
# Helper to collect nodes by type
# ---------------------------------------------------------------------------

def _collect(steps, cls):
    """Collect all nodes of a given type from a steps tuple (non-recursive)."""
    return [op for op in steps if isinstance(op, cls)]


def _collect_all_expm(steps):
    """Collect Expm nodes from steps including inside Iterate bodies."""
    result = []
    for op in steps:
        if isinstance(op, Expm):
            result.append(op)
        elif isinstance(op, Iterate):
            result.extend(e for e in op.body if isinstance(e, Expm))
    return result


def _collect_all_transport(steps):
    """Collect Transport nodes from steps including inside Iterate bodies."""
    result = []
    for op in steps:
        if isinstance(op, Transport):
            result.append(op)
        elif isinstance(op, Iterate):
            result.extend(t for t in op.body if isinstance(t, Transport))
    return result


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------

def test_predictor_structure():
    assert len(_collect(predictor.steps, Transport)) == 1
    assert len(_collect(predictor.steps, Expm)) == 1
    assert predictor.fallback is None


def test_predictor_graph():
    A_0 = _collect(predictor.steps, Transport)[0]
    n_1 = _collect(predictor.steps, Expm)[0]
    assert A_0.density is BOS
    assert n_1.density is BOS
    assert len(n_1.terms) == 1
    assert n_1.terms[0].weight == pytest.approx(1.0)
    assert n_1.terms[0].matrix is A_0


# ---------------------------------------------------------------------------
# CE/CM
# ---------------------------------------------------------------------------

def test_cecm_structure():
    assert len(_collect(cecm.steps, Transport)) == 2
    assert len(_collect(cecm.steps, Expm)) == 2


def test_cecm_graph():
    transports = _collect(cecm.steps, Transport)
    expms = _collect(cecm.steps, Expm)
    A_0, A_1 = transports
    n_half, n_1 = expms

    assert A_0.density is BOS
    # n_half = exp(0.5 * A_0) * n_bos
    assert n_half.density is BOS
    assert len(n_half.terms) == 1
    assert n_half.terms[0].weight == pytest.approx(0.5)
    assert n_half.terms[0].matrix is A_0
    # A_1 from n_half
    assert A_1.density is n_half
    # n_1 = exp(1.0 * A_1) * n_bos
    assert n_1.density is BOS
    assert len(n_1.terms) == 1
    assert n_1.terms[0].weight == pytest.approx(1.0)
    assert n_1.terms[0].matrix is A_1


# ---------------------------------------------------------------------------
# CE/LI
# ---------------------------------------------------------------------------

def test_celi_structure():
    assert len(_collect(celi.steps, Transport)) == 2
    assert len(_collect(celi.steps, Expm)) == 3


def test_celi_graph():
    transports = _collect(celi.steps, Transport)
    expms = _collect(celi.steps, Expm)
    A_0, A_1 = transports
    n_pred, n_inter, n_1 = expms

    # Predictor: exp(A_0) * n_bos
    assert n_pred.terms[0].weight == pytest.approx(1.0)
    assert n_pred.terms[0].matrix is A_0
    assert n_pred.density is BOS
    # A_1 from n_pred
    assert A_1.density is n_pred
    # Corrector 1: exp(5/12*A_0 + 1/12*A_1) * n_bos
    assert n_inter.density is BOS
    assert n_inter.terms[0].weight == pytest.approx(5/12)
    assert n_inter.terms[0].matrix is A_0
    assert n_inter.terms[1].weight == pytest.approx(1/12)
    assert n_inter.terms[1].matrix is A_1
    # Corrector 2: exp(1/12*A_0 + 5/12*A_1) * n_inter
    assert n_1.density is n_inter
    assert n_1.terms[0].weight == pytest.approx(1/12)
    assert n_1.terms[0].matrix is A_0
    assert n_1.terms[1].weight == pytest.approx(5/12)
    assert n_1.terms[1].matrix is A_1


def test_celi_corrector_weights_sum():
    """Each corrector expm integrates half the step."""
    expms = _collect(celi.steps, Expm)
    n_inter, n_1 = expms[1], expms[2]
    assert sum(t.weight for t in n_inter.terms) == pytest.approx(0.5)
    assert sum(t.weight for t in n_1.terms) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# CF4
# ---------------------------------------------------------------------------

def test_cf4_structure():
    assert len(_collect(cf4.steps, Transport)) == 4
    assert len(_collect(cf4.steps, Expm)) == 5


def test_cf4_graph():
    transports = _collect(cf4.steps, Transport)
    expms = _collect(cf4.steps, Expm)
    A_0, A_1, A_2, A_3 = transports

    # All transports chained correctly
    assert A_0.density is BOS
    assert A_1.density is expms[0]   # n_hat1
    assert A_2.density is expms[1]   # n_hat2
    assert A_3.density is expms[2]   # n_hat3

    # n_hat1 = exp(1/2 * A_0) * n_bos
    assert expms[0].terms[0].weight == pytest.approx(0.5)
    assert expms[0].terms[0].matrix is A_0
    assert expms[0].density is BOS

    # n_hat2 = exp(1/2 * A_1) * n_bos
    assert expms[1].terms[0].weight == pytest.approx(0.5)
    assert expms[1].terms[0].matrix is A_1
    assert expms[1].density is BOS

    # n_hat3 = exp(-1/2*A_0 + A_2) * n_hat1
    assert expms[2].density is expms[0]  # applied to n_hat1
    assert expms[2].terms[0].weight == pytest.approx(-0.5)
    assert expms[2].terms[0].matrix is A_0
    assert expms[2].terms[1].weight == pytest.approx(1.0)
    assert expms[2].terms[1].matrix is A_2

    # n_inter = exp(-1/12*A_0 + 1/6*A_1 + 1/6*A_2 + 1/4*A_3) * n_bos
    assert expms[3].density is BOS
    assert expms[3].terms[0].matrix is A_0
    assert expms[3].terms[1].matrix is A_1
    assert expms[3].terms[2].matrix is A_2
    assert expms[3].terms[3].matrix is A_3

    # n_1 = exp(1/4*A_0 + 1/6*A_1 + 1/6*A_2 - 1/12*A_3) * n_inter
    assert expms[4].density is expms[3]  # applied to n_inter


def test_cf4_matrix_reuse():
    """A_0 appears in 4 expm calls, A_1 in 3 — verifying reuse."""
    transports = _collect(cf4.steps, Transport)
    expms = _collect(cf4.steps, Expm)
    A_0 = transports[0]
    A_1 = transports[1]
    a0_count = sum(
        1 for e in expms for t in e.terms if t.matrix is A_0)
    a1_count = sum(
        1 for e in expms for t in e.terms if t.matrix is A_1)
    assert a0_count == 4
    assert a1_count == 3


def test_cf4_final_weights():
    expms = _collect(cf4.steps, Expm)
    n_inter, n_1 = expms[3], expms[4]
    w_inter = [t.weight for t in n_inter.terms]
    w_final = [t.weight for t in n_1.terms]
    assert w_inter == pytest.approx([-1/12, 1/6, 1/6, 1/4])
    assert w_final == pytest.approx([1/4, 1/6, 1/6, -1/12])
    # Total weight across both expm calls sums to 1.0
    assert sum(w_inter) + sum(w_final) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# EPC-RK4
# ---------------------------------------------------------------------------

def test_epc_rk4_structure():
    assert len(_collect(epc_rk4.steps, Transport)) == 4
    assert len(_collect(epc_rk4.steps, Expm)) == 4


def test_epc_rk4_graph():
    transports = _collect(epc_rk4.steps, Transport)
    expms = _collect(epc_rk4.steps, Expm)
    A_0, A_1, A_2, A_3 = transports

    assert A_0.density is BOS
    assert A_1.density is expms[0]
    assert A_2.density is expms[1]
    assert A_3.density is expms[2]

    # All intermediate expms applied to n_bos
    for e in expms[:3]:
        assert e.density is BOS

    # Final expm: 1/6*A_0 + 1/3*A_1 + 1/3*A_2 + 1/6*A_3
    fw = [t.weight for t in expms[3].terms]
    assert fw == pytest.approx([1/6, 1/3, 1/3, 1/6])
    assert expms[3].density is BOS


def test_epc_rk4_weights_sum():
    expms = _collect(epc_rk4.steps, Expm)
    assert sum(t.weight for t in expms[3].terms) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# LE/QI
# ---------------------------------------------------------------------------

def test_leqi_structure():
    assert len(_collect(leqi.steps, Transport)) == 2
    assert len(_collect(leqi.steps, Expm)) == 4
    assert leqi.fallback is celi


def test_leqi_all_expm_reference_prev_step():
    """Every Expm in LE/QI has at least one PREV_STEP term."""
    expms = _collect(leqi.steps, Expm)
    for e in expms:
        refs = [t.matrix for t in e.terms]
        assert PREV_STEP in refs


def test_leqi_all_weights_callable():
    """All matrix weights in LE/QI are callable."""
    expms = _collect(leqi.steps, Expm)
    for e in expms:
        for t in e.terms:
            assert callable(t.weight)


def test_leqi_weight_functions_equal_dt():
    """Check weight values for equal timesteps."""
    assert _leqi_w1_prev(1.0, 1.0) == pytest.approx(-1/12)
    assert _leqi_w1_bos(1.0, 1.0) == pytest.approx(7/12)
    assert _leqi_w2_prev(1.0, 1.0) == pytest.approx(-5/12)
    assert _leqi_w2_bos(1.0, 1.0) == pytest.approx(11/12)
    assert _leqi_w3_prev(1.0, 1.0) == pytest.approx(-1/24)
    assert _leqi_w3_bos(1.0, 1.0) == pytest.approx(12/24)
    assert _leqi_w3_eos(1.0, 1.0) == pytest.approx(1/24)
    assert _leqi_w4_prev(1.0, 1.0) == pytest.approx(-1/24)
    assert _leqi_w4_bos(1.0, 1.0) == pytest.approx(4/24)
    assert _leqi_w4_eos(1.0, 1.0) == pytest.approx(9/24)


def test_leqi_predictor_weights_sum():
    """LE predictor: each expm's weights sum to 0.5 for equal dt."""
    assert (_leqi_w1_prev(1.0, 1.0) + _leqi_w1_bos(1.0, 1.0)
            == pytest.approx(0.5))
    assert (_leqi_w2_prev(1.0, 1.0) + _leqi_w2_bos(1.0, 1.0)
            == pytest.approx(0.5))


def test_leqi_corrector_weights_sum():
    """QI corrector: each expm's weights sum to 0.5 for equal dt."""
    assert (_leqi_w3_prev(1.0, 1.0) + _leqi_w3_bos(1.0, 1.0)
            + _leqi_w3_eos(1.0, 1.0) == pytest.approx(0.5))
    assert (_leqi_w4_prev(1.0, 1.0) + _leqi_w4_bos(1.0, 1.0)
            + _leqi_w4_eos(1.0, 1.0) == pytest.approx(0.5))


def test_leqi_graph_density_chaining():
    """LE predictor chains: n_inter -> n_pred; QI: n_inter2 -> n_1."""
    expms = _collect(leqi.steps, Expm)
    n_inter, n_pred, n_inter2, n_1 = expms
    assert n_inter.density is BOS
    assert n_pred.density is n_inter
    assert n_inter2.density is BOS
    assert n_1.density is n_inter2


# ---------------------------------------------------------------------------
# SI-CE/LI
# ---------------------------------------------------------------------------

def test_si_celi_structure():
    assert len(_collect(si_celi.steps, Transport)) == 1  # only A_0 at top level
    assert len(_collect(si_celi.steps, Expm)) == 1        # only n_pred at top level
    assert any(isinstance(op, Iterate) for op in si_celi.steps)
    assert si_celi.fallback is None


def test_si_celi_iterate():
    iterates = [op for op in si_celi.steps if isinstance(op, Iterate)]
    assert len(iterates) == 1
    it = iterates[0]
    assert it.n_iterations == 11

    # Body has: Transport(PREV_ITER), Expm, Expm
    assert len(it.body) == 3
    A_iter = it.body[0]
    n_corr1 = it.body[1]
    n_corr2 = it.body[2]
    assert isinstance(A_iter, Transport)
    assert A_iter.density is PREV_ITER
    assert isinstance(n_corr1, Expm)
    assert isinstance(n_corr2, Expm)


def test_si_celi_iterate_uses_average():
    """Expm nodes inside SI iterate reference AverageMatrix."""
    it = [op for op in si_celi.steps if isinstance(op, Iterate)][0]
    n_corr1, n_corr2 = it.body[1], it.body[2]
    for e in (n_corr1, n_corr2):
        avg_terms = [t for t in e.terms if isinstance(t.matrix, AverageMatrix)]
        assert len(avg_terms) == 1


def test_si_celi_corrector_weights():
    """SI-CE/LI uses same 5/12, 1/12 weights as CE/LI corrector."""
    top_transport = _collect(si_celi.steps, Transport)[0]  # A_0
    it = [op for op in si_celi.steps if isinstance(op, Iterate)][0]
    n_corr1, n_corr2 = it.body[1], it.body[2]

    # n_corr1: 5/12 * A_0 + 1/12 * A_avg
    assert n_corr1.terms[0].weight == pytest.approx(5/12)
    assert n_corr1.terms[0].matrix is top_transport
    assert n_corr1.terms[1].weight == pytest.approx(1/12)
    assert n_corr1.density is BOS

    # n_corr2: 1/12 * A_0 + 5/12 * A_avg, applied to n_corr1
    assert n_corr2.terms[0].weight == pytest.approx(1/12)
    assert n_corr2.terms[0].matrix is top_transport
    assert n_corr2.terms[1].weight == pytest.approx(5/12)
    assert n_corr2.density is n_corr1


def test_si_celi_corrector_weights_sum():
    it = [op for op in si_celi.steps if isinstance(op, Iterate)][0]
    for e in it.body[1:]:
        assert sum(t.weight for t in e.terms) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# SI-LE/QI
# ---------------------------------------------------------------------------

def test_si_leqi_structure():
    assert any(isinstance(op, Iterate) for op in si_leqi.steps)
    assert si_leqi.fallback is si_celi


def test_si_leqi_iterate():
    iterates = [op for op in si_leqi.steps if isinstance(op, Iterate)]
    assert len(iterates) == 1
    it = iterates[0]
    assert it.n_iterations == 11
    assert len(it.body) == 3
    assert isinstance(it.body[0], Transport)
    assert it.body[0].density is PREV_ITER


def test_si_leqi_iterate_uses_average():
    it = [op for op in si_leqi.steps if isinstance(op, Iterate)][0]
    for e in it.body[1:]:
        avg_terms = [t for t in e.terms if isinstance(t.matrix, AverageMatrix)]
        assert len(avg_terms) == 1


def test_si_leqi_le_predictor():
    """SI-LE/QI has two Expm before the Iterate (the LE predictor)."""
    expms = _collect(si_leqi.steps, Expm)
    assert len(expms) == 2  # n_inter, n_pred at top level
    for e in expms:
        # All top-level Expm have callable weights (LE part)
        for t in e.terms:
            assert callable(t.weight)


# ---------------------------------------------------------------------------
# IntegrationScheme properties
# ---------------------------------------------------------------------------

def test_fallback():
    assert leqi.fallback is celi
    assert si_leqi.fallback is si_celi
    for s in (predictor, cecm, celi, cf4, epc_rk4, si_celi):
        assert s.fallback is None


def test_frozen_scheme():
    """IntegrationScheme is frozen (immutable)."""
    with pytest.raises(AttributeError):
        predictor.name = 'something_else'
