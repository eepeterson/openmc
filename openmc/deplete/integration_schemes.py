"""Declarative integration scheme definitions for depletion.

Each scheme describes the mathematical structure of a time-integration algorithm
as a sequence of :class:`DepletionStage` objects.  The :class:`IntegratorScheme`
dataclass is consumed by the depletion driver, which interprets the stages to
perform transport evaluations and matrix-exponential solves.

Schemes are purely declarative data — they contain no simulation logic.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Union

# A Weight is either a fixed tuple of coefficients (one per rate in the
# available-rates array) or a callable ``(prev_dt, dt) -> coefficients``
# for schemes whose weights depend on the ratio of adjacent time steps.
Weight = Union[tuple[float, ...], Callable[[float, float], tuple[float, ...]]]

__all__ = [
    "DepletionStage", "IntegratorScheme", "Weight", "SCHEMES",
    "predictor", "cecm", "celi", "cf4", "epc_rk4", "leqi",
    "si_celi", "si_leqi",
]


@dataclass(frozen=True)
class DepletionStage:
    """A single sub-step within an integration scheme.

    Parameters
    ----------
    source : int
        Index of the concentration vector to use as input.  ``0`` is the
        beginning-of-step (BOS) concentrations; ``N`` for *N* > 0 is the
        output of stage *N* - 1.
    weights : Weight
        Coefficients for forming the depletion matrix as a linear combination
        of rate matrices.  A tuple of floats is positionally mapped to the
        rates array; a callable receives ``(prev_dt, dt)`` and returns the
        tuple.

        The rates array is ``[bos_rates, eval_0_rates, eval_1_rates, ...]``.
        When weights are callable (indicating the scheme needs previous-step
        rates), the rates array is prepended with ``prev_step_rates``.
    evaluate : bool
        If ``True``, a transport solve is performed at the resulting
        composition to produce new reaction rates.

    """

    source: int
    weights: Weight
    evaluate: bool = False


@dataclass(frozen=True)
class IntegratorScheme:
    """Declarative description of a depletion time-integration algorithm.

    Parameters
    ----------
    name : str
        Identifier for the scheme (e.g. ``'cecm'``, ``'cf4'``).
    stages : tuple of DepletionStage
        Ordered depletion sub-steps.
    n_iterations : int
        Stochastic-implicit (SI) iteration count.  ``1`` (default) means
        no SI.  Values > 1 cause the corrector stages to be iterated with
        running-averaged rates.

    """

    name: str
    stages: tuple[DepletionStage, ...]
    n_iterations: int = 1

    def __post_init__(self):
        if self.n_iterations > 1 and self.corrector_start is None:
            raise ValueError(
                f"Scheme '{self.name}' has no corrector stages; "
                "n_iterations > 1 requires stages after the last evaluation"
            )

    @property
    def corrector_start(self) -> int | None:
        """Index of the first corrector stage, or ``None`` if none exists.

        The corrector consists of all stages after the last ``evaluate=True``
        stage.  Returns ``None`` when the scheme has no stages after the last
        evaluation (i.e. no corrector is possible).
        """
        for i in reversed(range(len(self.stages))):
            if self.stages[i].evaluate:
                return i + 1 if i + 1 < len(self.stages) else None
        return None

    @property
    def uses_prev_rates(self) -> bool:
        """Whether any stage has callable weights requiring previous rates."""
        return any(callable(s.weights) for s in self.stages)

    @property
    def num_evaluations(self) -> int:
        """Number of intermediate transport evaluations in the scheme."""
        return sum(1 for s in self.stages if s.evaluate)


# ---------------------------------------------------------------------------
# LE/QI weight functions (depend on prev_dt and dt)
# ---------------------------------------------------------------------------

def _leqi_f1(prev_dt: float, dt: float) -> tuple[float, float]:
    """LE predictor, first exponential."""
    return (
        -dt / (12 * prev_dt),
        (dt + 6 * prev_dt) / (12 * prev_dt),
    )


def _leqi_f2(prev_dt: float, dt: float) -> tuple[float, float]:
    """LE predictor, second exponential."""
    return (
        -5 * dt / (12 * prev_dt),
        (5 * dt + 6 * prev_dt) / (12 * prev_dt),
    )


def _leqi_f3(prev_dt: float, dt: float) -> tuple[float, float, float]:
    """QI corrector, first exponential."""
    d = 12 * prev_dt * (dt + prev_dt)
    return (
        -dt**2 / d,
        (dt**2 + 6 * dt * prev_dt + 5 * prev_dt**2) / d,
        prev_dt / (12 * (dt + prev_dt)),
    )


def _leqi_f4(prev_dt: float, dt: float) -> tuple[float, float, float]:
    """QI corrector, second exponential."""
    d = 12 * prev_dt * (dt + prev_dt)
    return (
        -dt**2 / d,
        (dt**2 + 2 * dt * prev_dt + prev_dt**2) / d,
        (4 * dt + 5 * prev_dt) / (12 * (dt + prev_dt)),
    )


# ---------------------------------------------------------------------------
# Scheme definitions
# ---------------------------------------------------------------------------

# Shared stage tuples for schemes reused with different n_iterations.

_celi_stages = (
    DepletionStage(source=0, weights=(1.0,), evaluate=True),
    DepletionStage(source=0, weights=(5/12, 1/12)),
    DepletionStage(source=2, weights=(1/12, 5/12)),
)

_leqi_stages = (
    DepletionStage(source=0, weights=_leqi_f1),
    DepletionStage(source=1, weights=_leqi_f2, evaluate=True),
    DepletionStage(source=0, weights=_leqi_f3),
    DepletionStage(source=3, weights=_leqi_f4),
)

#: First-order predictor (forward Euler).
predictor = IntegratorScheme(
    name='predictor',
    stages=(
        DepletionStage(source=0, weights=(1.0,)),
    ),
)

#: CE/CM: constant extrapolation predictor, constant midpoint corrector.
cecm = IntegratorScheme(
    name='cecm',
    stages=(
        DepletionStage(source=0, weights=(0.5,), evaluate=True),
        DepletionStage(source=0, weights=(0.0, 1.0)),
    ),
)

#: CE/LI CFQ4: constant extrapolation predictor, linear interpolation
#: corrector.
celi = IntegratorScheme(name='celi', stages=_celi_stages)

#: CF4: fourth-order commutator-free Lie algorithm.
cf4 = IntegratorScheme(
    name='cf4',
    stages=(
        DepletionStage(source=0, weights=(0.5,), evaluate=True),
        DepletionStage(source=0, weights=(0.0, 0.5), evaluate=True),
        DepletionStage(source=1, weights=(-0.5, 0.0, 1.0), evaluate=True),
        DepletionStage(source=0, weights=(1/4, 1/6, 1/6, -1/12)),
        DepletionStage(source=4, weights=(-1/12, 1/6, 1/6, 1/4)),
    ),
)

#: EPC-RK4: embedded predictor-corrector with classical Runge-Kutta 4.
epc_rk4 = IntegratorScheme(
    name='epc_rk4',
    stages=(
        DepletionStage(source=0, weights=(0.5,), evaluate=True),
        DepletionStage(source=0, weights=(0.0, 0.5), evaluate=True),
        DepletionStage(source=0, weights=(0.0, 0.0, 1.0), evaluate=True),
        DepletionStage(source=0, weights=(1/6, 1/3, 1/3, 1/6)),
    ),
)

#: LE/QI CFQ4: linear extrapolation predictor, quadratic interpolation
#: corrector.  On the first step (no previous rates), the driver falls
#: back to a constant-weight scheme such as CE/LI.
leqi = IntegratorScheme(
    name='leqi',
    stages=_leqi_stages,
)

#: SI-CE/LI: stochastic implicit variant of CE/LI.
si_celi = IntegratorScheme(
    name='si_celi',
    stages=_celi_stages,
    n_iterations=10,
)

#: SI-LE/QI: stochastic implicit variant of LE/QI.
si_leqi = IntegratorScheme(
    name='si_leqi',
    stages=_leqi_stages,
    n_iterations=10,
)

#: Registry mapping scheme names to :class:`IntegratorScheme` instances.
SCHEMES: dict[str, IntegratorScheme] = {
    s.name: s for s in (
        predictor, cecm, celi, cf4, epc_rk4, leqi, si_celi, si_leqi
    )
}
