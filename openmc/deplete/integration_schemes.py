"""Declarative integration scheme definitions for depletion.

Each scheme is a directed graph of :class:`Transport` and :class:`Expm` nodes
describing the mathematical structure of a time-integration algorithm.  The
depletion driver interprets these nodes sequentially to execute the algorithm.

Nodes reference each other via Python object identity:

- **Transport(density)** — run transport on a density vector to produce a
  burnup matrix :math:`A_k`.  The ``Transport`` object itself serves as the
  handle that later :class:`Expm` nodes use to reference that matrix.

- **Expm(terms, density)** — compute
  :math:`\\exp\\bigl(\\sum_i w_i A_i \\cdot dt\\bigr)\\, \\mathbf{n}` where
  each term is a :class:`MatrixTerm` pairing a weight with a matrix reference.
  The ``Expm`` object serves as the handle for the resulting density vector.

- **Iterate(n_iterations, body)** — repeat *body* with running-average matrix
  semantics for stochastic-implicit (SI) algorithms.

- **AverageMatrix(source)** — a matrix reference that resolves to the
  running average of a ``Transport`` node's matrix across SI iterations.

Schemes are purely declarative data with no simulation logic.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Union

__all__ = [
    "BOS", "PREV_STEP", "PREV_ITER",
    "Transport", "Expm", "MatrixTerm", "AverageMatrix", "Iterate",
    "IntegrationScheme", "SCHEMES",
    "predictor", "cecm", "celi", "cf4", "epc_rk4", "leqi",
    "si_celi", "si_leqi",
]

# ---- Sentinel types for special density/matrix references ----

class _Sentinel:
    """Named singleton sentinel."""
    def __init__(self, name):
        self._name = name

    def __repr__(self):
        return self._name

#: Beginning-of-step density vector.
BOS = _Sentinel('BOS')

#: Burnup matrix from the previous depletion step (for LE/QI schemes).
PREV_STEP = _Sentinel('PREV_STEP')

#: Density vector from the last Expm in the previous SI iteration.
PREV_ITER = _Sentinel('PREV_ITER')

# Type alias for density references accepted by Transport and Expm.
DensityRef = Union[_Sentinel, 'Expm']

# Type alias for matrix references accepted by MatrixTerm.
MatrixRef = Union['Transport', 'AverageMatrix', _Sentinel]


# ---- Node types ----

class Transport:
    """Run transport on a density vector to produce a burnup matrix.

    The ``Transport`` instance itself is the handle used by :class:`MatrixTerm`
    to reference the resulting matrix.

    Parameters
    ----------
    density : DensityRef
        Density vector to transport.  One of :data:`BOS`, :data:`PREV_ITER`,
        or an :class:`Expm` node whose output is used.

    """
    __slots__ = ('density',)

    def __init__(self, density: DensityRef):
        self.density = density

    def __repr__(self):
        return f'Transport({self.density!r})'

    def __rmul__(self, weight):
        """Shorthand: ``0.5 * A_0`` returns ``MatrixTerm(0.5, A_0)``."""
        return MatrixTerm(weight, self)


class MatrixTerm:
    """A weighted matrix reference: ``weight * A``.

    Parameters
    ----------
    weight : float or callable
        Scalar weight.  For LE/QI schemes this is a callable
        ``(prev_dt, dt) -> float`` whose value depends on timestep geometry.
    matrix : MatrixRef
        A :class:`Transport` node, :class:`AverageMatrix` reference, or
        :data:`PREV_STEP` sentinel.

    """
    __slots__ = ('weight', 'matrix')

    def __init__(self, weight: float | Callable[[float, float], float],
                 matrix: MatrixRef):
        self.weight = weight
        self.matrix = matrix

    def __repr__(self):
        return f'MatrixTerm({self.weight!r}, {self.matrix!r})'


class AverageMatrix:
    """Reference to the running average of a Transport's matrix across SI
    iterations.

    Parameters
    ----------
    source : Transport
        The ``Transport`` node whose matrix is averaged.

    """
    __slots__ = ('source',)

    def __init__(self, source: Transport):
        self.source = source

    def __repr__(self):
        return f'AverageMatrix({self.source!r})'

    def __rmul__(self, weight):
        """Shorthand: ``0.5 * A_avg`` returns ``MatrixTerm(0.5, A_avg)``."""
        return MatrixTerm(weight, self)


class Expm:
    """Apply a matrix exponential to a density vector.

    Computes :math:`\\exp\\bigl(\\sum_i w_i A_i \\cdot dt\\bigr)\\,\\mathbf{n}`.

    The ``Expm`` instance itself is the handle used by other nodes to
    reference the resulting density vector.

    Parameters
    ----------
    terms : tuple of MatrixTerm
        Weighted matrix terms whose sum defines the exponent.
    density : DensityRef
        Density vector to which the exponential is applied.

    """
    __slots__ = ('terms', 'density')

    def __init__(self, terms: tuple[MatrixTerm, ...], density: DensityRef):
        self.terms = terms
        self.density = density

    def __repr__(self):
        return f'Expm({self.terms!r}, {self.density!r})'


class Iterate:
    """Repeat a body of operations with running-average matrix semantics.

    On each iteration *j*:

    - ``Transport`` nodes inside *body* are executed normally.
    - ``AverageMatrix`` references resolve to:
      - Copy of the new matrix when *j* ≤ 1,
      - Incremental mean ``(1/j)*A_new + (1 - 1/j)*A_bar`` when *j* ≥ 2.
    - The density :data:`PREV_ITER` is seeded from the last ``Expm`` before
      this ``Iterate``, and updated to the last ``Expm`` in *body* after each
      iteration.

    Parameters
    ----------
    n_iterations : int
        Number of SI iterations to perform.
    body : tuple
        Sequence of ``Transport`` and ``Expm`` nodes to repeat.

    """
    __slots__ = ('n_iterations', 'body')

    def __init__(self, n_iterations: int, body: tuple):
        self.n_iterations = n_iterations
        self.body = body

    def __repr__(self):
        return f'Iterate({self.n_iterations}, {self.body!r})'


# ---- Scheme container ----

@dataclass(frozen=True)
class IntegrationScheme:
    """Declarative description of a depletion time-integration algorithm.

    Parameters
    ----------
    name : str
        Identifier for the scheme (e.g. ``'cecm'``, ``'cf4'``).
    steps : tuple
        Ordered sequence of :class:`Transport`, :class:`Expm`, and
        :class:`Iterate` nodes that define the algorithm.
    fallback : IntegrationScheme or None
        Scheme to use when this scheme requires previous-step data
        (i.e. references :data:`PREV_STEP`) that is not yet available,
        such as on the first depletion step.

    """

    name: str
    steps: tuple
    fallback: IntegrationScheme | None = None

    @property
    def num_transports(self) -> int:
        """Number of ``Transport`` operations at the top level."""
        return sum(1 for op in self.steps if isinstance(op, Transport))

    @property
    def num_expm(self) -> int:
        """Number of ``Expm`` operations at the top level."""
        return sum(1 for op in self.steps if isinstance(op, Expm))

    @property
    def uses_prev_step(self) -> bool:
        """Whether this scheme references :data:`PREV_STEP`."""
        return self._has_ref(PREV_STEP)

    @property
    def is_si(self) -> bool:
        """Whether this scheme contains an :class:`Iterate` block."""
        return any(isinstance(op, Iterate) for op in self.steps)

    @property
    def first_transport(self) -> Transport | None:
        """The first ``Transport`` node, or ``None``."""
        for op in self.steps:
            if isinstance(op, Transport):
                return op
        return None

    def _has_ref(self, sentinel) -> bool:
        """Check if any MatrixTerm in the scheme references the sentinel."""
        def _check_terms(steps):
            for op in steps:
                if isinstance(op, Expm):
                    for term in op.terms:
                        if term.matrix is sentinel:
                            return True
                elif isinstance(op, Iterate):
                    if _check_terms(op.body):
                        return True
            return False
        return _check_terms(self.steps)


# ---------------------------------------------------------------------------
# LE/QI weight functions  (prev_dt, dt) -> float
# ---------------------------------------------------------------------------

def _leqi_w1_prev(prev_dt, dt):
    return -dt / (12 * prev_dt)

def _leqi_w1_bos(prev_dt, dt):
    return (dt + 6 * prev_dt) / (12 * prev_dt)

def _leqi_w2_prev(prev_dt, dt):
    return -5 * dt / (12 * prev_dt)

def _leqi_w2_bos(prev_dt, dt):
    return (5 * dt + 6 * prev_dt) / (12 * prev_dt)

def _leqi_w3_prev(prev_dt, dt):
    d = 12 * prev_dt * (dt + prev_dt)
    return -dt**2 / d

def _leqi_w3_bos(prev_dt, dt):
    d = 12 * prev_dt * (dt + prev_dt)
    return (dt**2 + 6 * dt * prev_dt + 5 * prev_dt**2) / d

def _leqi_w3_eos(prev_dt, dt):
    return prev_dt / (12 * (dt + prev_dt))

def _leqi_w4_prev(prev_dt, dt):
    d = 12 * prev_dt * (dt + prev_dt)
    return -dt**2 / d

def _leqi_w4_bos(prev_dt, dt):
    d = 12 * prev_dt * (dt + prev_dt)
    return (dt**2 + 2 * dt * prev_dt + prev_dt**2) / d

def _leqi_w4_eos(prev_dt, dt):
    return (4 * dt + 5 * prev_dt) / (12 * (dt + prev_dt))


# ---------------------------------------------------------------------------
# Scheme definitions
# ---------------------------------------------------------------------------

# ---- Predictor ----
def _build_predictor():
    A_0 = Transport(BOS)
    n_1 = Expm((1.0 * A_0,), BOS)
    return IntegrationScheme('predictor', steps=(A_0, n_1))

# ---- CE/CM ----
def _build_cecm():
    A_0 = Transport(BOS)
    n_half = Expm((0.5 * A_0,), BOS)
    A_1 = Transport(n_half)
    n_1 = Expm((1.0 * A_1,), BOS)
    return IntegrationScheme('cecm', steps=(A_0, n_half, A_1, n_1))

# ---- CE/LI ----
def _build_celi():
    A_0 = Transport(BOS)
    n_pred = Expm((1.0 * A_0,), BOS)
    A_1 = Transport(n_pred)
    n_inter = Expm((5/12 * A_0, 1/12 * A_1), BOS)
    n_1 = Expm((1/12 * A_0, 5/12 * A_1), n_inter)
    return IntegrationScheme('celi', steps=(A_0, n_pred, A_1, n_inter, n_1))

# ---- CF4 ----
def _build_cf4():
    A_0 = Transport(BOS)
    n_hat1 = Expm((1/2 * A_0,), BOS)
    A_1 = Transport(n_hat1)
    n_hat2 = Expm((1/2 * A_1,), BOS)
    A_2 = Transport(n_hat2)
    n_hat3 = Expm((-1/2 * A_0, 1.0 * A_2), n_hat1)
    A_3 = Transport(n_hat3)
    n_inter = Expm(
        (-1/12 * A_0, 1/6 * A_1, 1/6 * A_2, 1/4 * A_3), BOS)
    n_1 = Expm(
        (1/4 * A_0, 1/6 * A_1, 1/6 * A_2, -1/12 * A_3), n_inter)
    return IntegrationScheme('cf4', steps=(
        A_0, n_hat1, A_1, n_hat2, A_2, n_hat3, A_3, n_inter, n_1))

# ---- EPC-RK4 ----
def _build_epc_rk4():
    A_0 = Transport(BOS)
    n_hat1 = Expm((1/2 * A_0,), BOS)
    A_1 = Transport(n_hat1)
    n_hat2 = Expm((1/2 * A_1,), BOS)
    A_2 = Transport(n_hat2)
    n_hat3 = Expm((1.0 * A_2,), BOS)
    A_3 = Transport(n_hat3)
    n_1 = Expm(
        (1/6 * A_0, 1/3 * A_1, 1/3 * A_2, 1/6 * A_3), BOS)
    return IntegrationScheme('epc_rk4', steps=(
        A_0, n_hat1, A_1, n_hat2, A_2, n_hat3, A_3, n_1))

# ---- LE/QI ----
def _build_leqi(celi_scheme):
    A_0 = Transport(BOS)
    # LE predictor
    n_inter = Expm((
        MatrixTerm(_leqi_w1_prev, PREV_STEP),
        MatrixTerm(_leqi_w1_bos, A_0),
    ), BOS)
    n_pred = Expm((
        MatrixTerm(_leqi_w2_prev, PREV_STEP),
        MatrixTerm(_leqi_w2_bos, A_0),
    ), n_inter)
    # QI corrector
    A_1 = Transport(n_pred)
    n_inter2 = Expm((
        MatrixTerm(_leqi_w3_prev, PREV_STEP),
        MatrixTerm(_leqi_w3_bos, A_0),
        MatrixTerm(_leqi_w3_eos, A_1),
    ), BOS)
    n_1 = Expm((
        MatrixTerm(_leqi_w4_prev, PREV_STEP),
        MatrixTerm(_leqi_w4_bos, A_0),
        MatrixTerm(_leqi_w4_eos, A_1),
    ), n_inter2)
    return IntegrationScheme('leqi',
        steps=(A_0, n_inter, n_pred, A_1, n_inter2, n_1),
        fallback=celi_scheme)

# ---- SI-CE/LI ----
def _build_si_celi():
    A_0 = Transport(BOS)
    n_pred = Expm((1.0 * A_0,), BOS)
    A_iter = Transport(PREV_ITER)
    A_avg = AverageMatrix(A_iter)
    n_corr1 = Expm((5/12 * A_0, 1/12 * A_avg), BOS)
    n_corr2 = Expm((1/12 * A_0, 5/12 * A_avg), n_corr1)
    return IntegrationScheme('si_celi', steps=(
        A_0, n_pred,
        Iterate(n_iterations=11, body=(A_iter, n_corr1, n_corr2)),
    ))

# ---- SI-LE/QI ----
def _build_si_leqi(si_celi_scheme):
    A_0 = Transport(BOS)
    # LE predictor
    n_inter = Expm((
        MatrixTerm(_leqi_w1_prev, PREV_STEP),
        MatrixTerm(_leqi_w1_bos, A_0),
    ), BOS)
    n_pred = Expm((
        MatrixTerm(_leqi_w2_prev, PREV_STEP),
        MatrixTerm(_leqi_w2_bos, A_0),
    ), n_inter)
    # SI iterate with QI corrector
    A_iter = Transport(PREV_ITER)
    A_avg = AverageMatrix(A_iter)
    n_corr1 = Expm((
        MatrixTerm(_leqi_w3_prev, PREV_STEP),
        MatrixTerm(_leqi_w3_bos, A_0),
        MatrixTerm(_leqi_w3_eos, A_avg),
    ), BOS)
    n_corr2 = Expm((
        MatrixTerm(_leqi_w4_prev, PREV_STEP),
        MatrixTerm(_leqi_w4_bos, A_0),
        MatrixTerm(_leqi_w4_eos, A_avg),
    ), n_corr1)
    return IntegrationScheme('si_leqi',
        steps=(A_0, n_inter, n_pred,
               Iterate(n_iterations=11, body=(A_iter, n_corr1, n_corr2))),
        fallback=si_celi_scheme)


# Build all scheme instances
predictor = _build_predictor()
cecm = _build_cecm()
celi = _build_celi()
cf4 = _build_cf4()
epc_rk4 = _build_epc_rk4()
leqi = _build_leqi(celi)
si_celi = _build_si_celi()
si_leqi = _build_si_leqi(si_celi)

#: Registry mapping scheme names to :class:`IntegrationScheme` instances.
SCHEMES: dict[str, IntegrationScheme] = {
    s.name: s for s in (
        predictor, cecm, celi, cf4, epc_rk4, leqi, si_celi, si_leqi
    )
}
