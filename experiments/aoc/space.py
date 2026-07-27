"""Deterministic, data-independent S-PACE algorithm primitives.

The objects in this module operate on already parsed, ordered contrasts.  They
do not read Run 6 data and deliberately do not decide whether separate QEC
round roles share one adaptive state.  A caller that wants role-specific
learning must instantiate one witness per role.

Implementation conventions
--------------------------

* Sparse and spectral witnesses start at zero and always score the current
  contrast before learning from it.
* An EWMA with half-life ``h`` uses
  ``alpha = 1 - 2**(-1 / h)`` and
  ``state <- (1 - alpha) * state + alpha * observation``.
* Top-k ties are resolved by increasing original coordinate index.  A zero
  coordinate has positive sign in the static capped-simplex extreme, while an
  uninformed adaptive witness remains the zero witness.
* A spectral update stride of ``m`` recomputes the effect after updates
  ``m, 2m, ...``.  The new effect is first used on the following observation.
* The online logistic learner treats ``monitor`` as the positive member of a
  pair, uses no intercept, and applies one pairwise-logistic SGD update after
  scoring.
* The proper-prior bank uses a uniform prior on start times ``1, ..., H``.
  Both sequential banks store their component recursions in the log domain.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]

_LOG_FLOAT_MAX = float(np.log(np.finfo(np.float64).max))

__all__ = [
    "BankUpdate",
    "EWMASpectralWitness",
    "EWMATopKWitness",
    "LogisticWitnessUpdate",
    "MixtureSRBank",
    "PairwiseOnlineLogistic",
    "ProperUniformStartEProcessBank",
    "SparseWitnessUpdate",
    "SpectralWitnessUpdate",
    "capped_simplex_top_k_extreme",
    "linear_bet_factors",
    "top_k_signed_extreme",
    "validate_bounded_score",
]


def _positive_integer(value: int, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, np.integer),
    ):
        raise TypeError(f"{name} must be a positive integer.")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return result


def _finite_vector(values: ArrayLike, *, name: str) -> FloatArray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or len(vector) == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional vector.")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values.")
    return vector


def _bounded_vector(
    values: ArrayLike,
    *,
    dimension: int,
    name: str,
    lower: float,
    upper: float,
    tolerance: float,
) -> FloatArray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (dimension,):
        raise ValueError(f"{name} must have shape ({dimension},).")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values.")
    if np.any(vector < lower - tolerance) or np.any(vector > upper + tolerance):
        raise ValueError(f"{name} must lie in [{lower}, {upper}].")
    return np.clip(vector, lower, upper)


def _mixture_weights(
    num_components: int,
    values: ArrayLike | None,
) -> FloatArray:
    if values is None:
        return np.full(num_components, 1.0 / num_components, dtype=np.float64)
    weights = np.asarray(values, dtype=np.float64)
    if weights.shape != (num_components,):
        raise ValueError(f"component_weights must have shape ({num_components},).")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("component_weights must be finite and nonnegative.")
    if not np.isclose(weights.sum(), 1.0, atol=1e-12, rtol=1e-12):
        raise ValueError("component_weights must sum to one.")
    return weights / weights.sum()


def _log_nonnegative(values: ArrayLike, *, size: int) -> FloatArray:
    factors = np.asarray(values, dtype=np.float64)
    if factors.shape != (size,):
        raise ValueError(f"factors must have shape ({size},).")
    if not np.all(np.isfinite(factors)) or np.any(factors < 0.0):
        raise ValueError("factors must be finite and nonnegative.")
    logged = np.full(size, -np.inf, dtype=np.float64)
    positive = factors > 0.0
    logged[positive] = np.log(factors[positive])
    return logged


def _logsumexp(values: FloatArray) -> float:
    return float(np.logaddexp.reduce(values))


def _finite_exp(log_value: float) -> float:
    if log_value == -np.inf:
        return 0.0
    if log_value >= _LOG_FLOAT_MAX:
        return float(np.finfo(np.float64).max)
    return float(np.exp(log_value))


def validate_bounded_score(
    score: float,
    *,
    tolerance: float = 1e-12,
) -> float:
    """Validate and return a scalar score in ``[-1, 1]``.

    Values only outside the interval by at most ``tolerance`` are clipped.
    Larger violations, non-scalars and non-finite values are rejected.
    """

    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance must be finite and nonnegative.")
    raw = np.asarray(score, dtype=np.float64)
    if raw.ndim != 0:
        raise ValueError("score must be a scalar.")
    value = float(raw)
    if not np.isfinite(value):
        raise ValueError("score must be finite.")
    if value < -1.0 - tolerance or value > 1.0 + tolerance:
        raise ValueError("score must lie in [-1, 1].")
    return float(np.clip(value, -1.0, 1.0))


def top_k_signed_extreme(direction: ArrayLike, k: int) -> FloatArray:
    """Return the deterministic signed top-k coefficient extreme.

    The result has exactly ``k`` nonzero entries of magnitude ``1/k`` and
    maximizes ``w @ direction`` over signed top-k extremes.  Absolute-value
    ties are resolved by increasing coordinate index; exact zeros receive
    positive sign.
    """

    vector = _finite_vector(direction, name="direction")
    count = _positive_integer(k, name="k")
    if count > len(vector):
        raise ValueError("k cannot exceed the direction dimension.")
    indices = np.lexsort((np.arange(len(vector)), -np.abs(vector)))[:count]
    signs = np.where(vector[indices] < 0.0, -1.0, 1.0)
    result = np.zeros_like(vector)
    result[indices] = signs / count
    return result


def capped_simplex_top_k_extreme(
    direction: ArrayLike,
    k: int,
) -> FloatArray:
    """Return the associated extreme in the ``2p`` signed capped simplex.

    Coordinates ``[:p]`` multiply ``direction`` and coordinates ``[p:]``
    multiply ``-direction``.  The returned vector is nonnegative, sums to one,
    and has cap ``1/k``.
    """

    vector = _finite_vector(direction, name="direction")
    coefficients = top_k_signed_extreme(vector, k)
    dimension = len(vector)
    weights = np.zeros(2 * dimension, dtype=np.float64)
    selected = np.flatnonzero(coefficients)
    positive = selected[coefficients[selected] > 0.0]
    negative = selected[coefficients[selected] < 0.0]
    weights[positive] = np.abs(coefficients[positive])
    weights[dimension + negative] = np.abs(coefficients[negative])
    return weights


def linear_bet_factors(
    score: float,
    bet_fractions: ArrayLike,
    *,
    two_sided: bool = False,
    tolerance: float = 1e-12,
) -> FloatArray:
    """Return factors ``1 + beta * score`` for fixed bet magnitudes.

    In two-sided mode, component order is
    ``(+beta_0, -beta_0, +beta_1, -beta_1, ...)``.
    """

    bounded = validate_bounded_score(score, tolerance=tolerance)
    bets = _finite_vector(bet_fractions, name="bet_fractions")
    if np.any(bets <= 0.0) or np.any(bets > 1.0):
        raise ValueError("bet_fractions must lie in (0, 1].")
    coefficients = np.column_stack((bets, -bets)).reshape(-1) if two_sided else bets
    factors = 1.0 + coefficients * bounded
    return np.clip(factors, 0.0, None)


@dataclass(frozen=True)
class SparseWitnessUpdate:
    """One score-before-update sparse witness record."""

    time: int
    score: float
    witness: FloatArray
    ewma_before: FloatArray
    ewma_after: FloatArray


class EWMATopKWitness:
    """Predictable signed top-k witness learned from an EWMA contrast."""

    def __init__(
        self,
        dimension: int,
        *,
        half_life: float,
        k: int,
        tolerance: float = 1e-12,
    ) -> None:
        self.dimension = _positive_integer(dimension, name="dimension")
        self.k = _positive_integer(k, name="k")
        if self.k > self.dimension:
            raise ValueError("k cannot exceed dimension.")
        if not np.isfinite(half_life) or half_life <= 0.0:
            raise ValueError("half_life must be finite and positive.")
        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("tolerance must be finite and nonnegative.")
        self.half_life = float(half_life)
        self.alpha = float(1.0 - np.exp2(-1.0 / self.half_life))
        self.decay = float(1.0 - self.alpha)
        self.tolerance = float(tolerance)
        self.reset()

    def reset(self) -> None:
        """Restore the deterministic, uninformed initial state."""

        self.time = 0
        self.ewma = np.zeros(self.dimension, dtype=np.float64)
        self.witness = np.zeros(self.dimension, dtype=np.float64)

    def _fit_witness(self) -> None:
        if np.max(np.abs(self.ewma)) <= self.tolerance:
            self.witness = np.zeros(self.dimension, dtype=np.float64)
        else:
            self.witness = top_k_signed_extreme(self.ewma, self.k)

    def update(self, difference: ArrayLike) -> SparseWitnessUpdate:
        """Score ``difference`` with the past witness, then update the EWMA."""

        current = _bounded_vector(
            difference,
            dimension=self.dimension,
            name="difference",
            lower=-1.0,
            upper=1.0,
            tolerance=self.tolerance,
        )
        used_witness = self.witness.copy()
        before = self.ewma.copy()
        score = validate_bounded_score(
            float(used_witness @ current),
            tolerance=max(self.tolerance, 1e-12),
        )
        self.ewma = self.decay * self.ewma + self.alpha * current
        self.time += 1
        self._fit_witness()
        return SparseWitnessUpdate(
            time=self.time,
            score=score,
            witness=used_witness,
            ewma_before=before,
            ewma_after=self.ewma.copy(),
        )


def _operator_contrast(
    values: ArrayLike,
    *,
    dimension: int,
    tolerance: float,
) -> ComplexArray:
    raw = np.asarray(values, dtype=np.complex128)
    if raw.shape != (dimension, dimension):
        raise ValueError(
            f"operator contrast must have shape ({dimension}, {dimension})."
        )
    if not np.all(np.isfinite(raw.real)) or not np.all(np.isfinite(raw.imag)):
        raise ValueError("operator contrast must contain only finite values.")
    hermitian = 0.5 * (raw + raw.conj().T)
    scale = max(1.0, float(np.linalg.norm(hermitian, ord="fro")))
    if float(np.linalg.norm(raw - raw.conj().T, ord="fro")) > tolerance * scale:
        raise ValueError("operator contrast must be Hermitian.")
    trace = complex(np.trace(hermitian))
    if abs(trace) > tolerance * scale:
        raise ValueError("operator contrast must have zero trace.")
    eigenvalues = np.linalg.eigvalsh(hermitian)
    trace_norm = float(np.abs(eigenvalues).sum())
    if trace_norm > 2.0 + tolerance * max(1.0, trace_norm):
        raise ValueError("operator contrast must have trace norm at most two.")
    return hermitian


def _canonical_rank_one_projector(
    eigenvalues: FloatArray,
    eigenvectors: ComplexArray,
    *,
    tolerance: float,
) -> ComplexArray:
    maximum = float(eigenvalues[-1])
    dimension = len(eigenvalues)
    if maximum <= tolerance:
        return np.zeros((dimension, dimension), dtype=np.complex128)
    tied = eigenvalues >= maximum - tolerance
    top_space = eigenvectors[:, tied]
    projector = top_space @ top_space.conj().T
    diagonal = np.clip(np.real(np.diag(projector)), 0.0, None)
    largest = float(diagonal.max())
    anchors = np.flatnonzero(diagonal >= largest - tolerance)
    anchor = int(anchors[0])
    vector = projector[:, anchor] / np.sqrt(diagonal[anchor])
    effect = np.outer(vector, vector.conj())
    return 0.5 * (effect + effect.conj().T)


@dataclass(frozen=True)
class SpectralWitnessUpdate:
    """One score-before-update spectral witness record."""

    time: int
    score: float
    effect: ComplexArray
    ewma_before: ComplexArray
    ewma_after: ComplexArray


class EWMASpectralWitness:
    """Predictable Jordan or rank-one effect learned from operator contrasts."""

    def __init__(
        self,
        dimension: int,
        *,
        half_life: float,
        rank: int | str = "positive",
        update_stride: int = 1,
        eigenvalue_tolerance: float = 1e-10,
        validation_tolerance: float = 1e-9,
    ) -> None:
        self.dimension = _positive_integer(dimension, name="dimension")
        if rank not in {1, "positive"}:
            raise ValueError("rank must be 1 or 'positive'.")
        if not np.isfinite(half_life) or half_life <= 0.0:
            raise ValueError("half_life must be finite and positive.")
        self.update_stride = _positive_integer(
            update_stride,
            name="update_stride",
        )
        if not np.isfinite(eigenvalue_tolerance) or eigenvalue_tolerance < 0.0:
            raise ValueError("eigenvalue_tolerance must be finite and nonnegative.")
        if not np.isfinite(validation_tolerance) or validation_tolerance < 0.0:
            raise ValueError("validation_tolerance must be finite and nonnegative.")
        self.rank = rank
        self.half_life = float(half_life)
        self.alpha = float(1.0 - np.exp2(-1.0 / self.half_life))
        self.decay = float(1.0 - self.alpha)
        self.eigenvalue_tolerance = float(eigenvalue_tolerance)
        self.validation_tolerance = float(validation_tolerance)
        self.reset()

    def reset(self) -> None:
        """Restore the deterministic, uninformed initial state."""

        shape = (self.dimension, self.dimension)
        self.time = 0
        self.ewma = np.zeros(shape, dtype=np.complex128)
        self.effect = np.zeros(shape, dtype=np.complex128)

    def _fit_effect(self) -> None:
        eigenvalues, eigenvectors = np.linalg.eigh(self.ewma)
        if self.rank == "positive":
            positive = eigenvalues > self.eigenvalue_tolerance
            if not np.any(positive):
                self.effect = np.zeros_like(self.ewma)
            else:
                basis = eigenvectors[:, positive]
                effect = basis @ basis.conj().T
                self.effect = 0.5 * (effect + effect.conj().T)
        else:
            self.effect = _canonical_rank_one_projector(
                eigenvalues,
                eigenvectors,
                tolerance=self.eigenvalue_tolerance,
            )

    def update(self, difference: ArrayLike) -> SpectralWitnessUpdate:
        """Score an operator contrast with the past effect, then learn from it."""

        current = _operator_contrast(
            difference,
            dimension=self.dimension,
            tolerance=self.validation_tolerance,
        )
        used_effect = self.effect.copy()
        before = self.ewma.copy()
        raw_score = float(np.real(np.trace(used_effect @ current)))
        score = validate_bounded_score(
            raw_score,
            tolerance=max(self.validation_tolerance, 1e-12),
        )
        self.ewma = self.decay * self.ewma + self.alpha * current
        self.ewma = 0.5 * (self.ewma + self.ewma.conj().T)
        self.time += 1
        if self.time % self.update_stride == 0:
            self._fit_effect()
        return SpectralWitnessUpdate(
            time=self.time,
            score=score,
            effect=used_effect,
            ewma_before=before,
            ewma_after=self.ewma.copy(),
        )


@dataclass(frozen=True)
class LogisticWitnessUpdate:
    """One score-before-update pairwise logistic record."""

    time: int
    score: float
    margin: float
    loss: float
    weights: FloatArray
    difference: FloatArray


class PairwiseOnlineLogistic:
    """One-step-SGD pairwise logistic witness with a bounded antisymmetric score."""

    def __init__(
        self,
        dimension: int,
        *,
        learning_rate: float,
        l2: float = 0.0,
        tolerance: float = 1e-12,
    ) -> None:
        self.dimension = _positive_integer(dimension, name="dimension")
        if not np.isfinite(learning_rate) or learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and positive.")
        if not np.isfinite(l2) or l2 < 0.0:
            raise ValueError("l2 must be finite and nonnegative.")
        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("tolerance must be finite and nonnegative.")
        self.learning_rate = float(learning_rate)
        self.l2 = float(l2)
        self.tolerance = float(tolerance)
        self.reset()

    def reset(self) -> None:
        """Restore zero weights and time."""

        self.time = 0
        self.weights = np.zeros(self.dimension, dtype=np.float64)

    @staticmethod
    def _negative_margin_probability(margin: float) -> float:
        if margin >= 0.0:
            exponential = float(np.exp(-margin))
            return exponential / (1.0 + exponential)
        exponential = float(np.exp(margin))
        return 1.0 / (1.0 + exponential)

    def _difference(
        self,
        reference: ArrayLike,
        monitor: ArrayLike,
    ) -> FloatArray:
        left = _bounded_vector(
            reference,
            dimension=self.dimension,
            name="reference",
            lower=0.0,
            upper=1.0,
            tolerance=self.tolerance,
        )
        right = _bounded_vector(
            monitor,
            dimension=self.dimension,
            name="monitor",
            lower=0.0,
            upper=1.0,
            tolerance=self.tolerance,
        )
        return right - left

    def score(
        self,
        reference: ArrayLike,
        monitor: ArrayLike,
    ) -> float:
        """Score an ordered pair without modifying the learner."""

        difference = self._difference(reference, monitor)
        margin = float(self.weights @ difference)
        denominator = max(1.0, float(np.abs(self.weights).sum()))
        return validate_bounded_score(
            float(np.tanh(margin / denominator)),
            tolerance=max(self.tolerance, 1e-12),
        )

    def update(
        self,
        reference: ArrayLike,
        monitor: ArrayLike,
    ) -> LogisticWitnessUpdate:
        """Score the ordered pair, then treat ``monitor`` as the positive item."""

        difference = self._difference(reference, monitor)
        used_weights = self.weights.copy()
        margin = float(used_weights @ difference)
        denominator = max(1.0, float(np.abs(used_weights).sum()))
        score = validate_bounded_score(
            float(np.tanh(margin / denominator)),
            tolerance=max(self.tolerance, 1e-12),
        )
        loss = float(
            np.logaddexp(0.0, -margin) + 0.5 * self.l2 * (used_weights @ used_weights)
        )
        miss_probability = self._negative_margin_probability(margin)
        gradient = -miss_probability * difference + self.l2 * used_weights
        self.weights = used_weights - self.learning_rate * gradient
        if not np.all(np.isfinite(self.weights)):
            raise FloatingPointError("logistic update produced non-finite weights.")
        self.time += 1
        return LogisticWitnessUpdate(
            time=self.time,
            score=score,
            margin=margin,
            loss=loss,
            weights=used_weights,
            difference=difference.copy(),
        )


@dataclass(frozen=True)
class BankUpdate:
    """One update of a log-domain component bank."""

    time: int
    log_statistic: float
    statistic: float
    alarm: bool
    alarm_time: int | None
    log_components: FloatArray


class ProperUniformStartEProcessBank:
    """Proper uniform-start-prior e-process mixed across fixed components.

    For horizon ``H`` and component factor ``L[t, j]``, this implements

    ``A[t,j] = L[t,j] * (A[t-1,j] + 1/H)``

    and

    ``E[t] = (H-t)/H + sum_j pi[j] * A[t,j]``.
    """

    def __init__(
        self,
        num_components: int,
        *,
        horizon: int,
        alpha: float = 0.01,
        component_weights: ArrayLike | None = None,
    ) -> None:
        self.num_components = _positive_integer(
            num_components,
            name="num_components",
        )
        self.horizon = _positive_integer(horizon, name="horizon")
        if not np.isfinite(alpha) or not 0.0 < alpha < 1.0:
            raise ValueError("alpha must lie in (0, 1).")
        self.alpha = float(alpha)
        self.threshold = float(1.0 / self.alpha)
        self.log_threshold = float(np.log(self.threshold))
        self.component_weights = _mixture_weights(
            self.num_components,
            component_weights,
        )
        self._log_weights = np.full(self.num_components, -np.inf)
        positive = self.component_weights > 0.0
        self._log_weights[positive] = np.log(self.component_weights[positive])
        self._log_start_mass = float(-np.log(self.horizon))
        self.reset()

    def reset(self) -> None:
        """Restore ``E_0=1`` and empty started mass."""

        self.time = 0
        self.log_components = np.full(
            self.num_components,
            -np.inf,
            dtype=np.float64,
        )
        self._alarm_components = np.zeros(
            self.num_components,
            dtype=np.float64,
        )
        self.log_statistic = 0.0
        self.alarm_time: int | None = None

    @property
    def statistic(self) -> float:
        return _finite_exp(self.log_statistic)

    def update(self, factors: ArrayLike) -> BankUpdate:
        """Update all fixed components with nonnegative e-factors."""

        if self.time >= self.horizon:
            raise RuntimeError("uniform-start e-process horizon is exhausted.")
        log_factors = _log_nonnegative(
            factors,
            size=self.num_components,
        )
        raw_factors = np.asarray(factors, dtype=np.float64)
        self.log_components = log_factors + np.logaddexp(
            self.log_components,
            self._log_start_mass,
        )
        self.time += 1
        started = _logsumexp(self._log_weights + self.log_components)
        remaining = self.horizon - self.time
        if remaining > 0:
            log_tail = float(np.log(remaining / self.horizon))
            self.log_statistic = float(np.logaddexp(started, log_tail))
        else:
            self.log_statistic = started
        if self.alarm_time is None:
            with np.errstate(over="ignore", invalid="ignore"):
                self._alarm_components = raw_factors * (
                    self._alarm_components + 1.0 / self.horizon
                )
                positive = self.component_weights > 0.0
                started_linear = float(
                    np.dot(
                        self.component_weights[positive],
                        self._alarm_components[positive],
                    )
                )
            tail_linear = (self.horizon - self.time) / self.horizon
            alarm_statistic = tail_linear + started_linear
            if alarm_statistic >= self.threshold:
                self.alarm_time = self.time
        return BankUpdate(
            time=self.time,
            log_statistic=self.log_statistic,
            statistic=self.statistic,
            alarm=self.alarm_time is not None,
            alarm_time=self.alarm_time,
            log_components=self.log_components.copy(),
        )


class MixtureSRBank:
    """Log-domain mixture Shiryaev--Roberts bank for fixed components."""

    def __init__(
        self,
        num_components: int,
        *,
        gamma: float,
        component_weights: ArrayLike | None = None,
    ) -> None:
        self.num_components = _positive_integer(
            num_components,
            name="num_components",
        )
        if not np.isfinite(gamma) or gamma <= 0.0:
            raise ValueError("gamma must be finite and positive.")
        self.gamma = float(gamma)
        self.log_threshold = float(np.log(self.gamma))
        self.component_weights = _mixture_weights(
            self.num_components,
            component_weights,
        )
        self._log_weights = np.full(self.num_components, -np.inf)
        positive = self.component_weights > 0.0
        self._log_weights[positive] = np.log(self.component_weights[positive])
        self.reset()

    def reset(self) -> None:
        """Restore ``R_0=0`` for every component."""

        self.time = 0
        self.log_components = np.full(
            self.num_components,
            -np.inf,
            dtype=np.float64,
        )
        self._alarm_components = np.zeros(
            self.num_components,
            dtype=np.float64,
        )
        self.log_statistic = -np.inf
        self.alarm_time: int | None = None

    @property
    def statistic(self) -> float:
        return _finite_exp(self.log_statistic)

    def update(self, factors: ArrayLike) -> BankUpdate:
        """Apply ``R[t,j]=(R[t-1,j]+1)*L[t,j]`` in the log domain."""

        log_factors = _log_nonnegative(
            factors,
            size=self.num_components,
        )
        raw_factors = np.asarray(factors, dtype=np.float64)
        self.log_components = log_factors + np.logaddexp(
            self.log_components,
            0.0,
        )
        self.time += 1
        self.log_statistic = _logsumexp(self._log_weights + self.log_components)
        if self.alarm_time is None:
            with np.errstate(over="ignore", invalid="ignore"):
                self._alarm_components = (self._alarm_components + 1.0) * raw_factors
                positive = self.component_weights > 0.0
                alarm_statistic = float(
                    np.dot(
                        self.component_weights[positive],
                        self._alarm_components[positive],
                    )
                )
            if alarm_statistic >= self.gamma:
                self.alarm_time = self.time
        return BankUpdate(
            time=self.time,
            log_statistic=self.log_statistic,
            statistic=self.statistic,
            alarm=self.alarm_time is not None,
            alarm_time=self.alarm_time,
            log_components=self.log_components.copy(),
        )
