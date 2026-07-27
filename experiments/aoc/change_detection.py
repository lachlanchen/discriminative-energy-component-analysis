"""Predictable accessible witnesses and sequential change detectors.

The routines in this module separate two choices that are often conflated:

1. which observable/feature algebra is accessible; and
2. which sequential statistic is applied to the resulting bounded score.

For translation-twirled correlation states, the accessible states are
probability vectors in the Fourier basis.  An effect is therefore a vector in
``[0, 1]^d`` and its centered expectation is automatically in ``[-1, 1]``.
This makes it possible to update the effect from past observations and feed
the next centered score into an average-run-length-valid
Shiryaev--Roberts-style e-detector.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike


def _as_probability_vector(
    values: ArrayLike,
    *,
    name: str,
    tolerance: float = 1e-9,
) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or len(vector) == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional vector.")
    if np.any(vector < -tolerance):
        raise ValueError(f"{name} must be nonnegative.")
    if not np.isclose(vector.sum(), 1.0, atol=tolerance):
        raise ValueError(f"{name} must sum to one.")
    vector = np.clip(vector, 0.0, None)
    return vector / vector.sum()


@dataclass(frozen=True)
class WitnessUpdate:
    """One predictable-witness update."""

    time: int
    score: float
    effect: np.ndarray
    live_mean: np.ndarray
    contrast: float


class PredictableSimplexWitness:
    """Learn an accessible diagonal effect from a trailing simplex window.

    ``mode="raw"`` implements the positive support of the empirical
    coordinate-wise difference.  This is the Helstrom/Jordan effect for
    commuting (Fourier-diagonal) states.

    ``mode="variance"`` computes the regularized Fisher/Hotelling direction

    ``(V0 + ridge I)^(-1) (live_mean - reference)``

    and affinely maps it into ``[0, 1]``.  Adding a constant to every
    coordinate does not change a centered simplex expectation, and dividing
    by the range makes the score bounded.  Fisher/Hotelling whitening is
    established prior art; this class only packages it as a predictable
    accessible effect.
    """

    def __init__(
        self,
        reference: ArrayLike,
        *,
        mode: str = "raw",
        null_covariance: ArrayLike | None = None,
        ridge: float = 1e-6,
        window: int = 64,
        update_interval: int = 1,
        tolerance: float = 1e-12,
    ) -> None:
        self.reference = _as_probability_vector(reference, name="reference")
        self.dimension = len(self.reference)
        if mode not in {"raw", "variance"}:
            raise ValueError("mode must be 'raw' or 'variance'.")
        if window < 1:
            raise ValueError("window must be positive.")
        if update_interval < 1:
            raise ValueError("update_interval must be positive.")
        if ridge < 0:
            raise ValueError("ridge must be nonnegative.")
        self.mode = mode
        self.window = int(window)
        self.update_interval = int(update_interval)
        self.ridge = float(ridge)
        self.tolerance = float(tolerance)
        if null_covariance is None:
            covariance = np.eye(self.dimension, dtype=np.float64)
        else:
            covariance = np.asarray(null_covariance, dtype=np.float64)
            if covariance.shape != (self.dimension, self.dimension):
                raise ValueError("null_covariance has the wrong shape.")
            covariance = 0.5 * (covariance + covariance.T)
        self.null_covariance = covariance
        regularized = covariance + self.ridge * np.eye(self.dimension)
        self._precision = np.linalg.pinv(regularized, hermitian=True)
        self._window: deque[np.ndarray] = deque(maxlen=self.window)
        self._window_sum = np.zeros(self.dimension, dtype=np.float64)
        self.effect = np.zeros(self.dimension, dtype=np.float64)
        self.live_mean = self.reference.copy()
        self.contrast = 0.0
        self.time = 0

    def _append(self, feature: np.ndarray) -> None:
        if len(self._window) == self.window:
            self._window_sum -= self._window[0]
        self._window.append(feature.copy())
        self._window_sum += feature
        self.live_mean = self._window_sum / len(self._window)

    def _fit_effect(self) -> None:
        delta = self.live_mean - self.reference
        if self.mode == "raw":
            self.effect = (delta > self.tolerance).astype(np.float64)
        else:
            direction = self._precision @ delta
            span = float(np.ptp(direction))
            if span <= self.tolerance:
                self.effect = np.zeros(self.dimension, dtype=np.float64)
            else:
                self.effect = (direction - direction.min()) / span
        self.contrast = float(self.effect @ delta)

    def update(self, feature: ArrayLike) -> WitnessUpdate:
        """Score the current feature with the past effect, then learn from it."""

        current = _as_probability_vector(feature, name="feature")
        if len(current) != self.dimension:
            raise ValueError("feature dimension does not match reference.")
        used_effect = self.effect.copy()
        used_mean = self.live_mean.copy()
        used_contrast = self.contrast
        score = float(used_effect @ (current - self.reference))
        if score < -1.0 - 1e-9 or score > 1.0 + 1e-9:
            raise RuntimeError("Simplex effect produced an invalid score bound.")

        self.time += 1
        self._append(current)
        if self.time % self.update_interval == 0:
            self._fit_effect()
        return WitnessUpdate(
            time=self.time,
            score=float(np.clip(score, -1.0, 1.0)),
            effect=used_effect,
            live_mean=used_mean,
            contrast=used_contrast,
        )


class StaticSimplexWitness:
    """A fixed accessible effect used for matched/oracle controls."""

    def __init__(self, reference: ArrayLike, effect: ArrayLike) -> None:
        self.reference = _as_probability_vector(reference, name="reference")
        candidate = np.asarray(effect, dtype=np.float64)
        if candidate.shape != self.reference.shape:
            raise ValueError("effect dimension does not match reference.")
        if np.any(candidate < 0.0) or np.any(candidate > 1.0):
            raise ValueError("effect entries must lie in [0, 1].")
        self.effect = candidate

    def score(self, feature: ArrayLike) -> float:
        current = _as_probability_vector(feature, name="feature")
        if current.shape != self.reference.shape:
            raise ValueError("feature dimension does not match reference.")
        return float(self.effect @ (current - self.reference))


class PredictableBoxWitness:
    """Predictable raw or variance-normalized witness for box-bounded features.

    The learned direction is divided by

    ``sum_i |w_i| (upper_i - lower_i)``,

    which guarantees a centered score in ``[-1, 1]``.  This corresponds to a
    norm-bounded linear observable dictionary; unlike
    :class:`PredictableSimplexWitness`, it is not the unrestricted positive
    support of a commuting density-matrix difference.
    """

    def __init__(
        self,
        reference: ArrayLike,
        lower: ArrayLike,
        upper: ArrayLike,
        *,
        mode: str = "raw",
        null_covariance: ArrayLike | None = None,
        ridge: float = 1e-6,
        window: int = 64,
        update_interval: int = 1,
        tolerance: float = 1e-12,
    ) -> None:
        self.reference = np.asarray(reference, dtype=np.float64)
        self.lower = np.broadcast_to(
            np.asarray(lower, dtype=np.float64),
            self.reference.shape,
        ).copy()
        self.upper = np.broadcast_to(
            np.asarray(upper, dtype=np.float64),
            self.reference.shape,
        ).copy()
        if self.reference.ndim != 1 or len(self.reference) == 0:
            raise ValueError("reference must be a nonempty vector.")
        if np.any(self.upper <= self.lower):
            raise ValueError("Every upper bound must exceed its lower bound.")
        if np.any(self.reference < self.lower) or np.any(self.reference > self.upper):
            raise ValueError("reference must lie inside the feature box.")
        if mode not in {"raw", "variance"}:
            raise ValueError("mode must be 'raw' or 'variance'.")
        if window < 1 or update_interval < 1:
            raise ValueError("window and update_interval must be positive.")
        if ridge < 0:
            raise ValueError("ridge must be nonnegative.")
        self.mode = mode
        self.window = int(window)
        self.update_interval = int(update_interval)
        self.tolerance = float(tolerance)
        dimension = len(self.reference)
        if null_covariance is None:
            covariance = np.eye(dimension)
        else:
            covariance = np.asarray(null_covariance, dtype=np.float64)
            if covariance.shape != (dimension, dimension):
                raise ValueError("null_covariance has the wrong shape.")
            covariance = 0.5 * (covariance + covariance.T)
        self.null_covariance = covariance
        self._precision = np.linalg.pinv(
            covariance + float(ridge) * np.eye(dimension),
            hermitian=True,
        )
        self._window: deque[np.ndarray] = deque(maxlen=self.window)
        self._window_sum = np.zeros(dimension, dtype=np.float64)
        self.live_mean = self.reference.copy()
        self.coefficients = np.zeros(dimension, dtype=np.float64)
        self.contrast = 0.0
        self.time = 0

    def _validate(self, feature: ArrayLike) -> np.ndarray:
        vector = np.asarray(feature, dtype=np.float64)
        if vector.shape != self.reference.shape:
            raise ValueError("feature dimension does not match reference.")
        if np.any(vector < self.lower - 1e-9) or np.any(
            vector > self.upper + 1e-9
        ):
            raise ValueError("feature lies outside the declared bounds.")
        return np.clip(vector, self.lower, self.upper)

    def _append(self, feature: np.ndarray) -> None:
        if len(self._window) == self.window:
            self._window_sum -= self._window[0]
        self._window.append(feature.copy())
        self._window_sum += feature
        self.live_mean = self._window_sum / len(self._window)

    def _fit(self) -> None:
        delta = self.live_mean - self.reference
        direction = delta if self.mode == "raw" else self._precision @ delta
        bound = float(np.abs(direction) @ (self.upper - self.lower))
        if bound <= self.tolerance:
            self.coefficients = np.zeros_like(direction)
        else:
            self.coefficients = direction / bound
        self.contrast = float(self.coefficients @ delta)

    def update(self, feature: ArrayLike) -> WitnessUpdate:
        current = self._validate(feature)
        used_coefficients = self.coefficients.copy()
        used_mean = self.live_mean.copy()
        used_contrast = self.contrast
        score = float(used_coefficients @ (current - self.reference))
        if score < -1.0 - 1e-9 or score > 1.0 + 1e-9:
            raise RuntimeError("Box witness produced an invalid score bound.")
        self.time += 1
        self._append(current)
        if self.time % self.update_interval == 0:
            self._fit()
        return WitnessUpdate(
            time=self.time,
            score=float(np.clip(score, -1.0, 1.0)),
            effect=used_coefficients,
            live_mean=used_mean,
            contrast=used_contrast,
        )


class StaticBoxWitness:
    """Fixed direction normalized for a declared feature box."""

    def __init__(
        self,
        reference: ArrayLike,
        direction: ArrayLike,
        lower: ArrayLike,
        upper: ArrayLike,
    ) -> None:
        self.reference = np.asarray(reference, dtype=np.float64)
        vector = np.asarray(direction, dtype=np.float64)
        self.lower = np.broadcast_to(
            np.asarray(lower, dtype=np.float64),
            self.reference.shape,
        ).copy()
        self.upper = np.broadcast_to(
            np.asarray(upper, dtype=np.float64),
            self.reference.shape,
        ).copy()
        if vector.shape != self.reference.shape:
            raise ValueError("direction dimension does not match reference.")
        bound = float(np.abs(vector) @ (self.upper - self.lower))
        self.coefficients = np.zeros_like(vector) if bound == 0.0 else vector / bound

    def score(self, feature: ArrayLike) -> float:
        current = np.asarray(feature, dtype=np.float64)
        if current.shape != self.reference.shape:
            raise ValueError("feature dimension does not match reference.")
        if np.any(current < self.lower - 1e-9) or np.any(
            current > self.upper + 1e-9
        ):
            raise ValueError("feature lies outside the declared bounds.")
        return float(self.coefficients @ (current - self.reference))


def effect_from_direction(
    direction: ArrayLike,
    *,
    tolerance: float = 1e-12,
) -> np.ndarray:
    """Affinely map a simplex score direction into an equivalent effect."""

    vector = np.asarray(direction, dtype=np.float64)
    if vector.ndim != 1 or len(vector) == 0:
        raise ValueError("direction must be a nonempty vector.")
    span = float(np.ptp(vector))
    if span <= tolerance:
        return np.zeros_like(vector)
    return (vector - vector.min()) / span


@dataclass(frozen=True)
class SequentialUpdate:
    """One update of an average-run-length detector."""

    time: int
    statistic: float
    alarm: bool
    alarm_time: int | None


class BoundedScoreSR:
    """Mixture Shiryaev--Roberts detector for predictable bounded scores.

    If ``score_t`` lies in ``[-1, 1]`` and has conditional null mean zero,
    each factor ``1 + beta * score_t`` is nonnegative with conditional mean
    one.  The recursion

    ``R_t(beta) = (R_{t-1}(beta) + 1) * (1 + beta * score_t)``

    has null expectation ``t``.  A convex mixture thresholded at ``gamma``
    therefore has average run length at least ``gamma`` (with the usual
    infinite-run convention).  This is an e-detector/ARL statement, not a
    claim that the statistic is a test martingale or an oracle quickest-change
    rule.
    """

    def __init__(
        self,
        *,
        threshold: float = 1000.0,
        bet_fractions: ArrayLike = (0.02, 0.05, 0.1, 0.2, 0.4, 0.8),
        mixture_weights: ArrayLike | None = None,
    ) -> None:
        if threshold <= 0:
            raise ValueError("threshold must be positive.")
        bets = np.asarray(bet_fractions, dtype=np.float64)
        if bets.ndim != 1 or len(bets) == 0:
            raise ValueError("bet_fractions must be a nonempty vector.")
        if np.any(bets <= 0.0) or np.any(bets > 1.0):
            raise ValueError("bet fractions must lie in (0, 1].")
        if mixture_weights is None:
            weights = np.full(len(bets), 1.0 / len(bets))
        else:
            weights = np.asarray(mixture_weights, dtype=np.float64)
            if weights.shape != bets.shape or np.any(weights < 0):
                raise ValueError("mixture_weights are invalid.")
            if not np.isclose(weights.sum(), 1.0):
                raise ValueError("mixture_weights must sum to one.")
        self.threshold = float(threshold)
        self.bet_fractions = bets
        self.mixture_weights = weights
        self.components = np.zeros(len(bets), dtype=np.float64)
        self.time = 0
        self.alarm_time: int | None = None

    @property
    def statistic(self) -> float:
        return float(self.mixture_weights @ self.components)

    def update(self, score: float) -> SequentialUpdate:
        if not np.isfinite(score) or score < -1.0 - 1e-9 or score > 1.0 + 1e-9:
            raise ValueError("score must be finite and lie in [-1, 1].")
        bounded = float(np.clip(score, -1.0, 1.0))
        if self.alarm_time is not None:
            self.time += 1
            return SequentialUpdate(
                time=self.time,
                statistic=self.statistic,
                alarm=True,
                alarm_time=self.alarm_time,
            )
        factors = 1.0 + self.bet_fractions * bounded
        self.components = (self.components + 1.0) * factors
        self.time += 1
        if (
            self.alarm_time is None
            and self.statistic >= self.threshold * (1.0 - 1e-12)
        ):
            self.alarm_time = self.time
        return SequentialUpdate(
            time=self.time,
            statistic=self.statistic,
            alarm=self.alarm_time is not None,
            alarm_time=self.alarm_time,
        )


class LikelihoodRatioSR:
    """Mixture SR detector for nonnegative likelihood-ratio increments."""

    def __init__(
        self,
        num_alternatives: int,
        *,
        threshold: float = 1000.0,
        mixture_weights: ArrayLike | None = None,
    ) -> None:
        if num_alternatives < 1:
            raise ValueError("num_alternatives must be positive.")
        if threshold <= 0:
            raise ValueError("threshold must be positive.")
        if mixture_weights is None:
            weights = np.full(num_alternatives, 1.0 / num_alternatives)
        else:
            weights = np.asarray(mixture_weights, dtype=np.float64)
            if weights.shape != (num_alternatives,) or np.any(weights < 0):
                raise ValueError("mixture_weights are invalid.")
            if not np.isclose(weights.sum(), 1.0):
                raise ValueError("mixture_weights must sum to one.")
        self.threshold = float(threshold)
        self.mixture_weights = weights
        self.components = np.zeros(num_alternatives, dtype=np.float64)
        self.time = 0
        self.alarm_time: int | None = None

    @property
    def statistic(self) -> float:
        return float(self.mixture_weights @ self.components)

    def update(self, likelihood_ratios: ArrayLike) -> SequentialUpdate:
        ratios = np.asarray(likelihood_ratios, dtype=np.float64)
        if ratios.shape != self.components.shape:
            raise ValueError("likelihood_ratios have the wrong shape.")
        if np.any(ratios < 0.0) or not np.all(np.isfinite(ratios)):
            raise ValueError("likelihood ratios must be finite and nonnegative.")
        if self.alarm_time is not None:
            self.time += 1
            return SequentialUpdate(
                time=self.time,
                statistic=self.statistic,
                alarm=True,
                alarm_time=self.alarm_time,
            )
        self.components = (self.components + 1.0) * ratios
        self.time += 1
        if (
            self.alarm_time is None
            and self.statistic >= self.threshold * (1.0 - 1e-12)
        ):
            self.alarm_time = self.time
        return SequentialUpdate(
            time=self.time,
            statistic=self.statistic,
            alarm=self.alarm_time is not None,
            alarm_time=self.alarm_time,
        )


class HiddenMarkovBlockSR:
    """Exact SR recursion for a post-change HMM at fixed block boundaries.

    Every update consumes two conditionally independent emissions. Candidate
    changepoints are allowed at the first cycle of each block. ``states[k, h]``
    stores the summed likelihood-ratio mass of every prior candidate for
    alternative ``k`` ending in hidden state ``h``. Injecting ``stationary``
    before the first emission creates the new candidate with its own latent
    prior; this is the step that a generic scalar ``(R + 1) * L`` recursion
    cannot reproduce for an HMM.
    """

    def __init__(
        self,
        transitions: ArrayLike,
        stationary: ArrayLike,
        *,
        threshold: float = 1000.0,
        mixture_weights: ArrayLike | None = None,
    ) -> None:
        matrices = np.asarray(transitions, dtype=np.float64)
        prior = np.asarray(stationary, dtype=np.float64)
        if matrices.ndim != 3 or matrices.shape[1] != matrices.shape[2]:
            raise ValueError("transitions must have shape (alternatives, s, s).")
        if prior.shape != (matrices.shape[1],):
            raise ValueError("stationary has the wrong shape.")
        if np.any(matrices < 0.0) or not np.allclose(
            matrices.sum(axis=2),
            1.0,
        ):
            raise ValueError("Every transition row must be a probability vector.")
        if np.any(prior < 0.0) or not np.isclose(prior.sum(), 1.0):
            raise ValueError("stationary must be a probability vector.")
        if not np.allclose(
            np.einsum("h,khj->kj", prior, matrices),
            prior[None, :],
        ):
            raise ValueError("stationary must be invariant for every alternative.")
        if threshold <= 0:
            raise ValueError("threshold must be positive.")
        alternatives = len(matrices)
        if mixture_weights is None:
            weights = np.full(alternatives, 1.0 / alternatives)
        else:
            weights = np.asarray(mixture_weights, dtype=np.float64)
            if weights.shape != (alternatives,) or np.any(weights < 0.0):
                raise ValueError("mixture_weights are invalid.")
            if not np.isclose(weights.sum(), 1.0):
                raise ValueError("mixture_weights must sum to one.")
        self.transitions = matrices
        self.stationary = prior
        self.threshold = float(threshold)
        self.mixture_weights = weights
        self.states = np.zeros(
            (alternatives, matrices.shape[1]),
            dtype=np.float64,
        )
        self.time = 0
        self.alarm_time: int | None = None

    @property
    def components(self) -> np.ndarray:
        return self.states.sum(axis=1)

    @property
    def statistic(self) -> float:
        return float(self.mixture_weights @ self.components)

    def update(
        self,
        first_emission_ratios: ArrayLike,
        second_emission_ratios: ArrayLike,
    ) -> SequentialUpdate:
        first = np.asarray(first_emission_ratios, dtype=np.float64)
        second = np.asarray(second_emission_ratios, dtype=np.float64)
        expected_shape = (self.states.shape[1],)
        if first.shape != expected_shape or second.shape != expected_shape:
            raise ValueError("emission ratios have the wrong shape.")
        if (
            np.any(first < 0.0)
            or np.any(second < 0.0)
            or not np.all(np.isfinite(first))
            or not np.all(np.isfinite(second))
        ):
            raise ValueError("emission ratios must be finite and nonnegative.")
        if self.alarm_time is not None:
            self.time += 1
            return SequentialUpdate(
                time=self.time,
                statistic=self.statistic,
                alarm=True,
                alarm_time=self.alarm_time,
            )

        existing_prediction = np.einsum(
            "kh,khj->kj",
            self.states,
            self.transitions,
        )
        after_first = (existing_prediction + self.stationary) * first[None, :]
        second_prediction = np.einsum(
            "kh,khj->kj",
            after_first,
            self.transitions,
        )
        self.states = second_prediction * second[None, :]
        self.time += 1
        if (
            self.alarm_time is None
            and self.statistic >= self.threshold * (1.0 - 1e-12)
        ):
            self.alarm_time = self.time
        return SequentialUpdate(
            time=self.time,
            statistic=self.statistic,
            alarm=self.alarm_time is not None,
            alarm_time=self.alarm_time,
        )


def paired_bootstrap_mean_difference(
    first: ArrayLike,
    second: ArrayLike,
    *,
    confidence: float = 0.95,
    resamples: int = 10_000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Return paired mean difference and percentile confidence interval."""

    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 1 or len(left) == 0:
        raise ValueError("paired samples must be nonempty vectors of equal shape.")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0, 1).")
    if resamples < 1:
        raise ValueError("resamples must be positive.")
    differences = left - right
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(differences), size=(resamples, len(differences)))
    boot = differences[indices].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(boot, (tail, 1.0 - tail))
    return float(differences.mean()), float(low), float(high)
