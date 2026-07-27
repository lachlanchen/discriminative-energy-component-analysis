"""Data-independent QEC scoring primitives for the locked Run 6 design.

This module contains no file I/O and does not inspect any Run 6 detector or
decoder values.  It fixes the 24-check Google feature map, fixed-model
controls, role-isolated adaptive witnesses, empirical thresholding, and the
within-shot Page--CUSUM control described by the method lock.

The adaptive M4/M5 wrappers deliberately inherit :mod:`aoc.space`'s
uncorrected EWMA convention.  If ``v_t > 0`` is the usual finite-time EWMA
normalizer, the bias-corrected direction is the stored direction divided by
the positive scalar ``v_t``.  Consequently top-k support/signs and positive
spectral eigenspaces are unchanged in exact arithmetic.  The two conventions
can differ only at an absolute zero/eigenvalue tolerance, where the stored
uncorrected scale is intentionally authoritative.

Every paired contrast is oriented ``monitor - reference``.  Adaptive methods
score the current pair before learning from it, and every round role owns
independent state.
"""

from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass
from itertools import combinations
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.covariance import LedoitWolf

from .space import (
    EWMASpectralWitness,
    EWMATopKWitness,
    PairwiseOnlineLogistic,
    linear_bet_factors,
    validate_bounded_score,
)

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]
BoolArray: TypeAlias = NDArray[np.bool_]
ComplexArray: TypeAlias = NDArray[np.complex128]

CHECK_COUNT = 24
FEATURE_DIM = 300
ROLE_COUNT = 51
PAIR_FEATURE_ORDER = tuple(combinations(range(CHECK_COUNT), 2))
_PAIR_FEATURE_LEFT = np.asarray(
    [pair[0] for pair in PAIR_FEATURE_ORDER],
    dtype=np.int64,
)
_PAIR_FEATURE_RIGHT = np.asarray(
    [pair[1] for pair in PAIR_FEATURE_ORDER],
    dtype=np.int64,
)
_PAIR_FEATURE_LEFT.setflags(write=False)
_PAIR_FEATURE_RIGHT.setflags(write=False)

SIGNED_BET_MAGNITUDES = (0.1, 0.3, 0.6, 0.9)
LOGISTIC_LEARNING_RATES = (0.001, 0.01, 0.1)
SPARSE_HALF_LIVES = (4.0, 16.0, 64.0, 256.0)
SPARSE_K_VALUES = (1, 4, 16, 64)
SPECTRAL_HALF_LIVES = (4.0, 16.0, 64.0)
SPECTRAL_RANKS = (1, "positive")
PAGE_CUSUM_KAPPAS = (0.01, 0.05, 0.1)

_SCORE_TOLERANCE = 1e-12
_EIGENVALUE_TOLERANCE = 1e-10
_SPECTRAL_VALIDATION_TOLERANCE = 1e-9
_M2_NEGATIVE_TOLERANCE = 1e-12

if len(PAIR_FEATURE_ORDER) != 276 or CHECK_COUNT + len(PAIR_FEATURE_ORDER) != (
    FEATURE_DIM
):
    raise RuntimeError("The immutable QEC feature ordering must contain 300 entries.")

__all__ = [
    "CHECK_COUNT",
    "FEATURE_DIM",
    "LOGISTIC_LEARNING_RATES",
    "PAGE_CUSUM_KAPPAS",
    "PAIR_FEATURE_ORDER",
    "ROLE_COUNT",
    "SIGNED_BET_MAGNITUDES",
    "SPARSE_HALF_LIVES",
    "SPARSE_K_VALUES",
    "SPECTRAL_HALF_LIVES",
    "SPECTRAL_RANKS",
    "ComponentPrior",
    "DiagonalLikelihoodModel",
    "EmpiricalCycleScores",
    "ExactComponentPriors",
    "FactorBank",
    "PageCUSUMShotResult",
    "PairedQECContrasts",
    "QECCycleUpdate",
    "ResourceCounts",
    "RoleHotellingModel",
    "RoleIsolatedQECBank",
    "RoleStateTimes",
    "StrictShotAlerts",
    "ThresholdSelection",
    "apply_strict_shot_threshold",
    "combine_space_factor_bank",
    "detector_firing_rate",
    "exact_component_priors",
    "paired_page_cusum_shot",
    "paired_qec_contrasts",
    "paired_resource_counts",
    "qec_density",
    "qec_features",
    "select_role_fit_indices",
    "select_strict_shot_threshold",
]


def _readonly(values: ArrayLike, *, dtype: np.dtype | type = np.float64) -> NDArray:
    result = np.array(values, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


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


def _nonnegative_integer(value: int, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, np.integer),
    ):
        raise TypeError(f"{name} must be a nonnegative integer.")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be a nonnegative integer.")
    return result


def _binary_array(
    values: ArrayLike,
    *,
    shape: tuple[int, ...],
    name: str,
) -> FloatArray:
    raw = np.asarray(values)
    if raw.shape != shape:
        raise ValueError(f"{name} must have shape {shape}.")
    try:
        numeric = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric binary data.") from error
    if not np.all(np.isfinite(numeric)):
        raise ValueError(f"{name} must contain only finite values.")
    if not np.all((numeric == 0.0) | (numeric == 1.0)):
        raise ValueError(f"{name} must contain only binary values.")
    return numeric


def _binary_checks(values: ArrayLike, *, name: str) -> FloatArray:
    return _binary_array(values, shape=(CHECK_COUNT,), name=name)


def _binary_pair_tensor(values: ArrayLike, *, name: str) -> FloatArray:
    raw = np.asarray(values)
    if raw.ndim != 3 or raw.shape[2] != CHECK_COUNT:
        raise ValueError(f"{name} must have shape (pairs, roles, {CHECK_COUNT}).")
    if raw.shape[0] < 1 or raw.shape[1] < 1:
        raise ValueError(f"{name} must contain at least one pair and one role.")
    return _binary_array(values, shape=raw.shape, name=name)


def _binary_shot(values: ArrayLike, *, name: str) -> FloatArray:
    raw = np.asarray(values)
    if raw.ndim != 2 or raw.shape[1] != CHECK_COUNT or raw.shape[0] < 1:
        raise ValueError(f"{name} must have shape (roles, {CHECK_COUNT}).")
    return _binary_array(values, shape=raw.shape, name=name)


def _role_index(role: int, *, role_count: int) -> int:
    if isinstance(role, (bool, np.bool_)) or not isinstance(
        role,
        (int, np.integer),
    ):
        raise TypeError("role must be an integer.")
    result = int(role)
    if result < 0 or result >= role_count:
        raise IndexError(f"role must lie in [0, {role_count}).")
    return result


def qec_features(bits: ArrayLike) -> FloatArray:
    """Return the immutable-order 300-vector ``[e_i; 1{e_i=e_j}]``."""

    checks = _binary_checks(bits, name="bits")
    result = np.empty(FEATURE_DIM, dtype=np.float64)
    result[:CHECK_COUNT] = checks
    result[CHECK_COUNT:] = checks[_PAIR_FEATURE_LEFT] == checks[_PAIR_FEATURE_RIGHT]
    return result


def qec_density(bits: ArrayLike) -> FloatArray:
    """Return ``R(e) = z z.T / 24`` with ``z = 1 - 2e``."""

    checks = _binary_checks(bits, name="bits")
    spins = 1.0 - 2.0 * checks
    return np.outer(spins, spins) / CHECK_COUNT


def detector_firing_rate(bits: ArrayLike) -> float:
    """Return the detector firing rate ``mean(e)``."""

    return float(np.mean(_binary_checks(bits, name="bits")))


@dataclass(frozen=True)
class PairedQECContrasts:
    """One monitor-minus-reference QEC feature record."""

    reference_features: FloatArray
    monitor_features: FloatArray
    feature_difference: FloatArray
    reference_density: FloatArray
    monitor_density: FloatArray
    density_difference: FloatArray
    firing_rate_difference: float


def paired_qec_contrasts(
    reference: ArrayLike,
    monitor: ArrayLike,
) -> PairedQECContrasts:
    """Construct all locked sparse, spectral, and global contrasts."""

    reference_checks = _binary_checks(reference, name="reference")
    monitor_checks = _binary_checks(monitor, name="monitor")
    reference_features = qec_features(reference_checks)
    monitor_features = qec_features(monitor_checks)
    reference_density = qec_density(reference_checks)
    monitor_density = qec_density(monitor_checks)
    return PairedQECContrasts(
        reference_features=_readonly(reference_features),
        monitor_features=_readonly(monitor_features),
        feature_difference=_readonly(monitor_features - reference_features),
        reference_density=_readonly(reference_density),
        monitor_density=_readonly(monitor_density),
        density_difference=_readonly(monitor_density - reference_density),
        firing_rate_difference=float(
            np.mean(monitor_checks) - np.mean(reference_checks)
        ),
    )


@dataclass(frozen=True)
class DiagonalLikelihoodModel:
    """Frozen role/check Bernoulli model with bounded antisymmetric score."""

    probabilities: FloatArray
    epsilon: float = 1e-4

    def __post_init__(self) -> None:
        probabilities = np.asarray(self.probabilities, dtype=np.float64)
        if probabilities.ndim != 2 or probabilities.shape[1] != CHECK_COUNT:
            raise ValueError(f"probabilities must have shape (roles, {CHECK_COUNT}).")
        if not np.isfinite(self.epsilon) or not 0.0 < self.epsilon < 0.5:
            raise ValueError("epsilon must lie in (0, 0.5).")
        if not np.all(np.isfinite(probabilities)):
            raise ValueError("probabilities must be finite.")
        if np.any(probabilities < self.epsilon) or np.any(
            probabilities > 1.0 - self.epsilon
        ):
            raise ValueError("probabilities violate the declared clipping interval.")
        object.__setattr__(self, "probabilities", _readonly(probabilities))
        object.__setattr__(self, "epsilon", float(self.epsilon))

    @property
    def role_count(self) -> int:
        return int(self.probabilities.shape[0])

    @property
    def normalization(self) -> float:
        return float(np.log((1.0 - self.epsilon) / self.epsilon))

    @classmethod
    def fit(
        cls,
        reference: ArrayLike,
        monitor: ArrayLike,
        *,
        epsilon: float = 1e-4,
    ) -> DiagonalLikelihoodModel:
        """Fit Jeffreys-smoothed probabilities by pooling both pair sides."""

        left = _binary_pair_tensor(reference, name="reference")
        right = _binary_pair_tensor(monitor, name="monitor")
        if left.shape != right.shape:
            raise ValueError("reference and monitor must have identical shape.")
        if not np.isfinite(epsilon) or not 0.0 < epsilon < 0.5:
            raise ValueError("epsilon must lie in (0, 0.5).")
        pair_count = left.shape[0]
        counts = np.sum(left + right, axis=0, dtype=np.float64)
        probabilities = (counts + 0.5) / (2 * pair_count + 1)
        probabilities = np.clip(probabilities, epsilon, 1.0 - epsilon)
        return cls(probabilities=probabilities, epsilon=float(epsilon))

    def negative_log_likelihood(self, role: int, bits: ArrayLike) -> float:
        """Return the average diagonal negative log likelihood for one role."""

        role_index = _role_index(role, role_count=self.role_count)
        checks = _binary_checks(bits, name="bits")
        probabilities = self.probabilities[role_index]
        terms = checks * np.log(probabilities) + (1.0 - checks) * np.log1p(
            -probabilities
        )
        return float(-np.mean(terms))

    def score(
        self,
        role: int,
        reference: ArrayLike,
        monitor: ArrayLike,
    ) -> float:
        """Return bounded ``(NLL_monitor - NLL_reference) / C_epsilon``."""

        difference = self.negative_log_likelihood(
            role,
            monitor,
        ) - self.negative_log_likelihood(role, reference)
        return validate_bounded_score(
            difference / self.normalization,
            tolerance=_SCORE_TOLERANCE,
        )


def select_role_fit_indices(
    *,
    num_pairs: int = 5000,
    num_roles: int = ROLE_COUNT,
    sample_size: int = 20000,
    seed: int = 610601,
) -> IntArray:
    """Return deterministic, role-stratified ``(pair, role)`` fit indices."""

    pair_count = _positive_integer(num_pairs, name="num_pairs")
    role_count = _positive_integer(num_roles, name="num_roles")
    selected_count = _positive_integer(sample_size, name="sample_size")
    if selected_count < role_count:
        raise ValueError("sample_size must select at least one pair per role.")
    if selected_count > pair_count * role_count:
        raise ValueError("sample_size exceeds the distinct role-pair population.")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(
        seed,
        (int, np.integer),
    ):
        raise TypeError("seed must be an integer.")

    quotient, remainder = divmod(selected_count, role_count)
    if quotient + int(remainder > 0) > pair_count:
        raise ValueError("a role would require more distinct pairs than available.")
    generator = np.random.Generator(np.random.PCG64(int(seed)))
    rows: list[tuple[int, int]] = []
    for role in range(role_count):
        role_size = quotient + int(role < remainder)
        selected_pairs = np.sort(
            generator.choice(pair_count, size=role_size, replace=False)
        )
        rows.extend((int(pair), role) for pair in selected_pairs)
    return _readonly(rows, dtype=np.int64)  # type: ignore[return-value]


@dataclass(frozen=True)
class RoleHotellingModel:
    """Frozen role-centered Ledoit--Wolf precision and empirical M2 score."""

    role_means: FloatArray
    precision: FloatArray
    selected_indices: IntArray

    def __post_init__(self) -> None:
        means = np.asarray(self.role_means, dtype=np.float64)
        precision = np.asarray(self.precision, dtype=np.float64)
        selected = np.asarray(self.selected_indices, dtype=np.int64)
        if means.ndim != 2 or means.shape[1] != FEATURE_DIM:
            raise ValueError(f"role_means must have shape (roles, {FEATURE_DIM}).")
        if precision.shape != (FEATURE_DIM, FEATURE_DIM):
            raise ValueError(
                f"precision must have shape ({FEATURE_DIM}, {FEATURE_DIM})."
            )
        if selected.ndim != 2 or selected.shape[1] != 2 or len(selected) < 1:
            raise ValueError("selected_indices must have shape (observations, 2).")
        if not np.all(np.isfinite(means)) or not np.all(np.isfinite(precision)):
            raise ValueError("M2 fitted arrays must be finite.")
        if not np.allclose(precision, precision.T, atol=1e-12, rtol=1e-12):
            raise ValueError("precision must be symmetric.")
        object.__setattr__(self, "role_means", _readonly(means))
        object.__setattr__(self, "precision", _readonly(precision))
        object.__setattr__(
            self,
            "selected_indices",
            _readonly(selected, dtype=np.int64),
        )

    @property
    def role_count(self) -> int:
        return int(self.role_means.shape[0])

    @classmethod
    def fit(
        cls,
        differences: ArrayLike,
        *,
        sample_size: int = 20000,
        seed: int = 610601,
    ) -> RoleHotellingModel:
        """Fit role means and one centered Ledoit--Wolf precision in float64."""

        values = np.asarray(differences, dtype=np.float64)
        if (
            values.ndim != 3
            or values.shape[0] < 1
            or values.shape[1] < 1
            or values.shape[2] != FEATURE_DIM
        ):
            raise ValueError(
                f"differences must have shape (pairs, roles, {FEATURE_DIM})."
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("differences must contain only finite values.")
        if np.any(values < -1.0 - _SCORE_TOLERANCE) or np.any(
            values > 1.0 + _SCORE_TOLERANCE
        ):
            raise ValueError("differences must lie in [-1, 1].")
        values = np.clip(values, -1.0, 1.0)
        pair_count, role_count, _ = values.shape
        selection = select_role_fit_indices(
            num_pairs=pair_count,
            num_roles=role_count,
            sample_size=sample_size,
            seed=seed,
        )

        means = np.empty((role_count, FEATURE_DIM), dtype=np.float64)
        residuals = np.empty((len(selection), FEATURE_DIM), dtype=np.float64)
        cursor = 0
        for role in range(role_count):
            pair_indices = selection[selection[:, 1] == role, 0]
            role_values = values[pair_indices, role]
            means[role] = np.mean(role_values, axis=0)
            next_cursor = cursor + len(pair_indices)
            residuals[cursor:next_cursor] = role_values - means[role]
            cursor = next_cursor
        if cursor != len(selection):
            raise RuntimeError("M2 selection accounting is inconsistent.")

        estimator = LedoitWolf(store_precision=True, assume_centered=True)
        estimator.fit(np.asarray(residuals, dtype=np.float64))
        precision = np.asarray(estimator.precision_, dtype=np.float64)
        precision = 0.5 * (precision + precision.T)
        return cls(
            role_means=means,
            precision=precision,
            selected_indices=selection,
        )

    def score(self, role: int, difference: ArrayLike) -> float:
        """Return the nonnegative role-centered Hotelling quadratic."""

        role_index = _role_index(role, role_count=self.role_count)
        values = np.asarray(difference, dtype=np.float64)
        if values.shape != (FEATURE_DIM,):
            raise ValueError(f"difference must have shape ({FEATURE_DIM},).")
        if not np.all(np.isfinite(values)):
            raise ValueError("difference must contain only finite values.")
        if np.any(values < -1.0 - _SCORE_TOLERANCE) or np.any(
            values > 1.0 + _SCORE_TOLERANCE
        ):
            raise ValueError("difference must lie in [-1, 1].")
        centered = np.clip(values, -1.0, 1.0) - self.role_means[role_index]
        quadratic = float(centered @ self.precision @ centered)
        if quadratic < -_M2_NEGATIVE_TOLERANCE:
            raise FloatingPointError("M2 quadratic is materially negative.")
        return max(0.0, quadratic)


@dataclass(frozen=True)
class ComponentPrior:
    """Immutable component identifiers and their exact prior weights."""

    component_ids: tuple[tuple[object, ...], ...]
    weights: FloatArray

    def __post_init__(self) -> None:
        weights = np.asarray(self.weights, dtype=np.float64)
        if weights.shape != (len(self.component_ids),):
            raise ValueError("weights must match component_ids.")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError("weights must be finite and nonnegative.")
        if not np.isclose(np.sum(weights), 1.0, atol=1e-15, rtol=1e-15):
            raise ValueError("component weights must sum to one.")
        object.__setattr__(self, "weights", _readonly(weights))


ExactComponentPriors: TypeAlias = dict[str, ComponentPrior]


def _uniform_prior(component_ids: tuple[tuple[object, ...], ...]) -> ComponentPrior:
    return ComponentPrior(
        component_ids=component_ids,
        weights=np.full(len(component_ids), 1.0 / len(component_ids)),
    )


def exact_component_priors() -> ExactComponentPriors:
    """Return the locked M0/M1/M3/M4/M5 and 50/50 S-PACE priors."""

    signed_bets = tuple(
        (sign * bet,) for bet in SIGNED_BET_MAGNITUDES for sign in (1.0, -1.0)
    )
    m0 = _uniform_prior(signed_bets)
    m1 = _uniform_prior(signed_bets)
    m3_ids = tuple(
        (learning_rate, bet)
        for learning_rate in LOGISTIC_LEARNING_RATES
        for bet in SIGNED_BET_MAGNITUDES
    )
    m4_ids = tuple(
        (half_life, k, bet)
        for half_life in SPARSE_HALF_LIVES
        for k in SPARSE_K_VALUES
        for bet in SIGNED_BET_MAGNITUDES
    )
    m5_ids = tuple(
        (half_life, rank, bet)
        for half_life in SPECTRAL_HALF_LIVES
        for rank in SPECTRAL_RANKS
        for bet in SIGNED_BET_MAGNITUDES
    )
    m3 = _uniform_prior(m3_ids)
    m4 = _uniform_prior(m4_ids)
    m5 = _uniform_prior(m5_ids)
    space_ids = tuple(("m4", *identifier) for identifier in m4_ids) + tuple(
        ("m5", *identifier) for identifier in m5_ids
    )
    space_weights = np.concatenate((0.5 * m4.weights, 0.5 * m5.weights))
    space = ComponentPrior(component_ids=space_ids, weights=space_weights)
    return {
        "m0": m0,
        "m1": m1,
        "m3": m3,
        "m4": m4,
        "m5": m5,
        "space": space,
    }


@dataclass(frozen=True)
class FactorBank:
    """One ordered factor vector and matching immutable prior."""

    factors: FloatArray
    weights: FloatArray

    def __post_init__(self) -> None:
        factors = np.asarray(self.factors, dtype=np.float64)
        weights = np.asarray(self.weights, dtype=np.float64)
        if factors.ndim != 1 or factors.shape != weights.shape:
            raise ValueError("factors and weights must be equal-length vectors.")
        if not np.all(np.isfinite(factors)) or np.any(factors < 0.0):
            raise ValueError("factors must be finite and nonnegative.")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError("weights must be finite and nonnegative.")
        if not np.isclose(np.sum(weights), 1.0, atol=1e-15, rtol=1e-15):
            raise ValueError("weights must sum to one.")
        object.__setattr__(self, "factors", _readonly(factors))
        object.__setattr__(self, "weights", _readonly(weights))


def combine_space_factor_bank(
    m4_factors: ArrayLike,
    m5_factors: ArrayLike,
) -> FactorBank:
    """Concatenate M4/M5 factors with exactly one half prior mass on each."""

    m4 = np.asarray(m4_factors, dtype=np.float64)
    m5 = np.asarray(m5_factors, dtype=np.float64)
    if m4.shape != (64,):
        raise ValueError("m4_factors must have shape (64,).")
    if m5.shape != (24,):
        raise ValueError("m5_factors must have shape (24,).")
    if (
        not np.all(np.isfinite(m4))
        or not np.all(np.isfinite(m5))
        or np.any(m4 < 0.0)
        or np.any(m5 < 0.0)
    ):
        raise ValueError("M4/M5 factors must be finite and nonnegative.")
    priors = exact_component_priors()["space"].weights
    return FactorBank(factors=np.concatenate((m4, m5)), weights=priors)


@dataclass(frozen=True)
class EmpiricalCycleScores:
    """Locked empirical cycle scores and deterministic expert attribution."""

    m0: float
    m1: float | None
    m2: float | None
    m3: float
    m4: float
    m5: float
    space: float
    m3_component: tuple[float]
    m4_component: tuple[float, int]
    m5_component: tuple[float, int | str]
    space_component: str


@dataclass(frozen=True)
class RoleStateTimes:
    """Per-expert update times for one adaptive round role."""

    m3: tuple[int, ...]
    m4: tuple[int, ...]
    m5: tuple[int, ...]


@dataclass(frozen=True)
class QECCycleUpdate:
    """One complete score-before-learning role update."""

    role: int
    time: int
    contrasts: PairedQECContrasts
    empirical: EmpiricalCycleScores
    m0_score: float
    m1_score: float | None
    m2_score: float | None
    m3_scores: FloatArray
    m4_scores: FloatArray
    m5_scores: FloatArray
    m0_factors: FloatArray
    m1_factors: FloatArray | None
    m3_factors: FloatArray
    m4_factors: FloatArray
    m5_factors: FloatArray
    space_factors: FloatArray
    space_weights: FloatArray


class _SharedSparseHalfLife:
    """One EWMA shared by all four locked top-k witnesses."""

    def __init__(self, *, half_life: float) -> None:
        self.state = EWMATopKWitness(
            FEATURE_DIM,
            half_life=half_life,
            k=1,
            tolerance=_SCORE_TOLERANCE,
        )

    @property
    def time(self) -> int:
        return self.state.time

    @property
    def ewma(self) -> FloatArray:
        return self.state.ewma

    def reset(self) -> None:
        self.state.reset()

    def witnesses(self) -> FloatArray:
        """Derive all four witnesses with one deterministic ordering."""

        result = np.zeros(
            (len(SPARSE_K_VALUES), FEATURE_DIM),
            dtype=np.float64,
        )
        direction = self.state.ewma
        if np.max(np.abs(direction)) <= _SCORE_TOLERANCE:
            return result
        order = np.lexsort((np.arange(FEATURE_DIM, dtype=np.int64), -np.abs(direction)))
        signs = np.where(direction[order] < 0.0, -1.0, 1.0)
        for row, k in enumerate(SPARSE_K_VALUES):
            result[row, order[:k]] = signs[:k] / k
        return result

    def update(self, difference: FloatArray) -> FloatArray:
        """Score all k values from one past EWMA, then update that EWMA."""

        witnesses = self.witnesses()
        scores = witnesses @ difference
        scores = np.asarray(
            [
                validate_bounded_score(
                    float(score),
                    tolerance=_SCORE_TOLERANCE,
                )
                for score in scores
            ],
            dtype=np.float64,
        )
        self.state.ewma = (
            self.state.decay * self.state.ewma + self.state.alpha * difference
        )
        self.state.time += 1
        return scores


def _stable_rank_one_projector(
    eigenvalues: FloatArray,
    eigenvectors: ComplexArray,
) -> ComplexArray:
    """Match :mod:`aoc.space`'s stable largest-diagonal anchor rule."""

    maximum = float(eigenvalues[-1])
    if maximum <= _EIGENVALUE_TOLERANCE:
        return np.zeros((CHECK_COUNT, CHECK_COUNT), dtype=np.complex128)
    tied = eigenvalues >= maximum - _EIGENVALUE_TOLERANCE
    top_space = eigenvectors[:, tied]
    projector = top_space @ top_space.conj().T
    diagonal = np.clip(np.real(np.diag(projector)), 0.0, None)
    largest = float(np.max(diagonal))
    anchors = np.flatnonzero(diagonal >= largest - _EIGENVALUE_TOLERANCE)
    anchor = int(anchors[0])
    vector = projector[:, anchor] / np.sqrt(diagonal[anchor])
    effect = np.outer(vector, vector.conj())
    return 0.5 * (effect + effect.conj().T)


class _SharedSpectralHalfLife:
    """One spectral EWMA and one eigendecomposition shared by both ranks."""

    _DISABLED_PRIMITIVE_STRIDE = 2**62

    def __init__(self, *, half_life: float) -> None:
        # Reuse the primitive's validated constants/state layout, while this
        # wrapper performs the one shared EWMA update and decomposition for
        # both locked effects. Calling the primitive update here would perform
        # an unnecessary third trace score.
        self.state = EWMASpectralWitness(
            CHECK_COUNT,
            half_life=half_life,
            rank="positive",
            update_stride=self._DISABLED_PRIMITIVE_STRIDE,
            eigenvalue_tolerance=_EIGENVALUE_TOLERANCE,
            validation_tolerance=_SPECTRAL_VALIDATION_TOLERANCE,
        )
        self.reset()

    @property
    def time(self) -> int:
        return self.state.time

    @property
    def ewma(self) -> ComplexArray:
        return self.state.ewma

    def reset(self) -> None:
        self.state.reset()
        shape = (CHECK_COUNT, CHECK_COUNT)
        self.rank_one_effect = np.zeros(shape, dtype=np.complex128)
        self.positive_effect = np.zeros(shape, dtype=np.complex128)

    def effects(self) -> ComplexArray:
        """Return effects in locked rank-one-before-positive order."""

        return np.stack((self.rank_one_effect, self.positive_effect))

    def _fit_effects(self) -> None:
        hermitian = 0.5 * (self.state.ewma + self.state.ewma.conj().T)
        eigenvalues, eigenvectors = np.linalg.eigh(hermitian)
        positive = eigenvalues > _EIGENVALUE_TOLERANCE
        if np.any(positive):
            basis = eigenvectors[:, positive]
            effect = basis @ basis.conj().T
            self.positive_effect = 0.5 * (effect + effect.conj().T)
        else:
            self.positive_effect = np.zeros_like(hermitian)
        self.rank_one_effect = _stable_rank_one_projector(
            eigenvalues,
            eigenvectors,
        )

    def update(self, difference: FloatArray) -> FloatArray:
        """Score both past effects, update once, then refit once if due."""

        effects = self.effects()
        scores = np.asarray(
            [
                validate_bounded_score(
                    float(np.real(np.trace(effect @ difference))),
                    tolerance=_SPECTRAL_VALIDATION_TOLERANCE,
                )
                for effect in effects
            ],
            dtype=np.float64,
        )
        self.state.ewma = (
            self.state.decay * self.state.ewma + self.state.alpha * difference
        )
        self.state.time += 1
        if self.state.time % 8 == 0:
            self._fit_effects()
        return scores


class RoleIsolatedQECBank:
    """M0--M5 cycle scorer with independent M3/M4/M5 state per role.

    M4 and M5 are composed directly from :class:`EWMATopKWitness` and
    :class:`EWMASpectralWitness`.  They therefore use the uncorrected EWMA
    convention documented at module level.
    """

    def __init__(
        self,
        *,
        role_count: int = ROLE_COUNT,
        diagonal_model: DiagonalLikelihoodModel | None = None,
        hotelling_model: RoleHotellingModel | None = None,
    ) -> None:
        self.role_count = _positive_integer(role_count, name="role_count")
        if diagonal_model is not None and diagonal_model.role_count != self.role_count:
            raise ValueError("diagonal_model role count does not match role_count.")
        if (
            hotelling_model is not None
            and hotelling_model.role_count != self.role_count
        ):
            raise ValueError("hotelling_model role count does not match role_count.")
        self.diagonal_model = diagonal_model
        self.hotelling_model = hotelling_model
        self._priors = exact_component_priors()
        self._build_states()

    def _build_states(self) -> None:
        self._m3 = [
            [
                PairwiseOnlineLogistic(
                    FEATURE_DIM,
                    learning_rate=learning_rate,
                    l2=1e-4,
                    tolerance=_SCORE_TOLERANCE,
                )
                for learning_rate in LOGISTIC_LEARNING_RATES
            ]
            for _ in range(self.role_count)
        ]
        self._m4 = [
            [
                _SharedSparseHalfLife(half_life=half_life)
                for half_life in SPARSE_HALF_LIVES
            ]
            for _ in range(self.role_count)
        ]
        self._m5 = [
            [
                _SharedSpectralHalfLife(half_life=half_life)
                for half_life in SPECTRAL_HALF_LIVES
            ]
            for _ in range(self.role_count)
        ]
        self._role_updates = np.zeros(self.role_count, dtype=np.int64)

    def reset(self) -> None:
        """Reset every adaptive witness without changing fixed M1/M2 fits."""

        for role in range(self.role_count):
            for learner in self._m3[role]:
                learner.reset()
            for state in self._m4[role]:
                state.reset()
            for state in self._m5[role]:
                state.reset()
        self._role_updates.fill(0)

    def role_state_times(self, role: int) -> RoleStateTimes:
        """Return all adaptive component times for one role."""

        role_index = _role_index(role, role_count=self.role_count)
        return RoleStateTimes(
            m3=tuple(state.time for state in self._m3[role_index]),
            m4=tuple(state.time for state in self._m4[role_index]),
            m5=tuple(state.time for state in self._m5[role_index]),
        )

    @property
    def role_update_counts(self) -> IntArray:
        return np.array(self._role_updates, copy=True)

    def clone(self) -> RoleIsolatedQECBank:
        """Return an independent in-memory checkpoint clone without pickle."""

        return copy.deepcopy(self)

    def export_numeric_state(self) -> dict[str, NDArray]:
        """Export a canonical-keyed copy of all fixed and mutable numeric state."""

        exported: dict[str, NDArray] = {
            "config.role_count": np.asarray([self.role_count], dtype=np.int64),
            "config.logistic_learning_rates": np.asarray(
                LOGISTIC_LEARNING_RATES,
                dtype=np.float64,
            ),
            "config.sparse_half_lives": np.asarray(
                SPARSE_HALF_LIVES,
                dtype=np.float64,
            ),
            "config.sparse_k_values": np.asarray(
                SPARSE_K_VALUES,
                dtype=np.int64,
            ),
            "config.spectral_half_lives": np.asarray(
                SPECTRAL_HALF_LIVES,
                dtype=np.float64,
            ),
            "mutable.role_updates": np.asarray(
                self._role_updates,
                dtype=np.int64,
            ),
            "mutable.m3.times": np.asarray(
                [[state.time for state in role_states] for role_states in self._m3],
                dtype=np.int64,
            ),
            "mutable.m3.weights": np.asarray(
                [[state.weights for state in role_states] for role_states in self._m3],
                dtype=np.float64,
            ),
            "mutable.m4.times": np.asarray(
                [[state.time for state in role_states] for role_states in self._m4],
                dtype=np.int64,
            ),
            "mutable.m4.ewma": np.asarray(
                [[state.ewma for state in role_states] for role_states in self._m4],
                dtype=np.float64,
            ),
            "mutable.m4.witnesses": np.asarray(
                [
                    [state.witnesses() for state in role_states]
                    for role_states in self._m4
                ],
                dtype=np.float64,
            ),
            "mutable.m5.times": np.asarray(
                [[state.time for state in role_states] for role_states in self._m5],
                dtype=np.int64,
            ),
            "mutable.m5.ewma": np.asarray(
                [[state.ewma for state in role_states] for role_states in self._m5],
                dtype=np.complex128,
            ),
            "mutable.m5.effects": np.asarray(
                [
                    [state.effects() for state in role_states]
                    for role_states in self._m5
                ],
                dtype=np.complex128,
            ),
        }
        if self.diagonal_model is None:
            exported["fixed.m1.present"] = np.asarray([0], dtype=np.int64)
            exported["fixed.m1.probabilities"] = np.empty(
                (0, CHECK_COUNT),
                dtype=np.float64,
            )
            exported["fixed.m1.epsilon"] = np.empty(0, dtype=np.float64)
        else:
            exported["fixed.m1.present"] = np.asarray([1], dtype=np.int64)
            exported["fixed.m1.probabilities"] = np.asarray(
                self.diagonal_model.probabilities,
                dtype=np.float64,
            )
            exported["fixed.m1.epsilon"] = np.asarray(
                [self.diagonal_model.epsilon],
                dtype=np.float64,
            )
        if self.hotelling_model is None:
            exported["fixed.m2.present"] = np.asarray([0], dtype=np.int64)
            exported["fixed.m2.role_means"] = np.empty(
                (0, FEATURE_DIM),
                dtype=np.float64,
            )
            exported["fixed.m2.precision"] = np.empty(
                (0, FEATURE_DIM),
                dtype=np.float64,
            )
            exported["fixed.m2.selected_indices"] = np.empty(
                (0, 2),
                dtype=np.int64,
            )
        else:
            exported["fixed.m2.present"] = np.asarray([1], dtype=np.int64)
            exported["fixed.m2.role_means"] = np.asarray(
                self.hotelling_model.role_means,
                dtype=np.float64,
            )
            exported["fixed.m2.precision"] = np.asarray(
                self.hotelling_model.precision,
                dtype=np.float64,
            )
            exported["fixed.m2.selected_indices"] = np.asarray(
                self.hotelling_model.selected_indices,
                dtype=np.int64,
            )
        return {
            name: _readonly(values, dtype=np.asarray(values).dtype)
            for name, values in exported.items()
        }

    def state_digest(self) -> str:
        """Return a platform-stable SHA-256 digest of canonical numeric state."""

        digest = hashlib.sha256()
        for name, values in sorted(self.export_numeric_state().items()):
            array = np.asarray(values)
            canonical_dtype = array.dtype.newbyteorder("<")
            canonical = np.ascontiguousarray(array.astype(canonical_dtype, copy=False))
            metadata = repr((name, canonical.dtype.str, tuple(canonical.shape))).encode(
                "utf-8"
            )
            digest.update(len(metadata).to_bytes(8, "little"))
            digest.update(metadata)
            payload = canonical.tobytes(order="C")
            digest.update(len(payload).to_bytes(8, "little"))
            digest.update(payload)
        return digest.hexdigest()

    def update(
        self,
        role: int,
        reference: ArrayLike,
        monitor: ArrayLike,
    ) -> QECCycleUpdate:
        """Score one role with past state, then update only that role."""

        role_index = _role_index(role, role_count=self.role_count)
        reference_checks = _binary_checks(reference, name="reference")
        monitor_checks = _binary_checks(monitor, name="monitor")
        contrasts = paired_qec_contrasts(reference_checks, monitor_checks)

        m0_score = validate_bounded_score(
            contrasts.firing_rate_difference,
            tolerance=_SCORE_TOLERANCE,
        )
        m0_factors = linear_bet_factors(
            m0_score,
            SIGNED_BET_MAGNITUDES,
            two_sided=True,
        )

        m1_score: float | None = None
        m1_factors: FloatArray | None = None
        if self.diagonal_model is not None:
            m1_score = self.diagonal_model.score(
                role_index,
                reference_checks,
                monitor_checks,
            )
            m1_factors = linear_bet_factors(
                m1_score,
                SIGNED_BET_MAGNITUDES,
                two_sided=True,
            )

        m2_score = (
            None
            if self.hotelling_model is None
            else self.hotelling_model.score(
                role_index,
                contrasts.feature_difference,
            )
        )

        m3_scores = np.asarray(
            [
                learner.update(
                    contrasts.reference_features,
                    contrasts.monitor_features,
                ).score
                for learner in self._m3[role_index]
            ],
            dtype=np.float64,
        )
        m4_scores = np.asarray(
            [
                state.update(contrasts.feature_difference)
                for state in self._m4[role_index]
            ],
            dtype=np.float64,
        )
        m5_scores = np.asarray(
            [
                state.update(contrasts.density_difference)
                for state in self._m5[role_index]
            ],
            dtype=np.float64,
        )

        m3_factors = np.concatenate(
            [linear_bet_factors(score, SIGNED_BET_MAGNITUDES) for score in m3_scores]
        )
        m4_factors = np.concatenate(
            [
                linear_bet_factors(score, SIGNED_BET_MAGNITUDES)
                for score in m4_scores.reshape(-1)
            ]
        )
        m5_factors = np.concatenate(
            [
                linear_bet_factors(score, SIGNED_BET_MAGNITUDES)
                for score in m5_scores.reshape(-1)
            ]
        )
        space_bank = combine_space_factor_bank(m4_factors, m5_factors)

        m3_index = int(np.argmax(m3_scores))
        m4_flat_index = int(np.argmax(m4_scores))
        m4_index = np.unravel_index(m4_flat_index, m4_scores.shape)
        m5_flat_index = int(np.argmax(m5_scores))
        m5_index = np.unravel_index(m5_flat_index, m5_scores.shape)
        m3_empirical = float(m3_scores[m3_index])
        m4_empirical = float(m4_scores[m4_index])
        m5_empirical = float(m5_scores[m5_index])
        if m4_empirical >= m5_empirical:
            space_empirical = m4_empirical
            space_component = "m4"
        else:
            space_empirical = m5_empirical
            space_component = "m5"
        empirical = EmpiricalCycleScores(
            m0=abs(m0_score),
            m1=None if m1_score is None else abs(m1_score),
            m2=m2_score,
            m3=m3_empirical,
            m4=m4_empirical,
            m5=m5_empirical,
            space=space_empirical,
            m3_component=(LOGISTIC_LEARNING_RATES[m3_index],),
            m4_component=(
                SPARSE_HALF_LIVES[m4_index[0]],
                SPARSE_K_VALUES[m4_index[1]],
            ),
            m5_component=(
                SPECTRAL_HALF_LIVES[m5_index[0]],
                SPECTRAL_RANKS[m5_index[1]],
            ),
            space_component=space_component,
        )

        self._role_updates[role_index] += 1
        return QECCycleUpdate(
            role=role_index,
            time=int(self._role_updates[role_index]),
            contrasts=contrasts,
            empirical=empirical,
            m0_score=m0_score,
            m1_score=m1_score,
            m2_score=m2_score,
            m3_scores=_readonly(m3_scores),
            m4_scores=_readonly(m4_scores.reshape(-1)),
            m5_scores=_readonly(m5_scores.reshape(-1)),
            m0_factors=_readonly(m0_factors),
            m1_factors=None if m1_factors is None else _readonly(m1_factors),
            m3_factors=_readonly(m3_factors),
            m4_factors=_readonly(m4_factors),
            m5_factors=_readonly(m5_factors),
            space_factors=space_bank.factors,
            space_weights=space_bank.weights,
        )


@dataclass(frozen=True)
class PageCUSUMShotResult:
    """Within-shot two-sided global/per-check Page--CUSUM histories."""

    cycle_scores: FloatArray
    positive: FloatArray
    negative: FloatArray
    positive_history: FloatArray
    negative_history: FloatArray


def paired_page_cusum_shot(
    reference: ArrayLike,
    monitor: ArrayLike,
) -> PageCUSUMShotResult:
    """Run the fixed empirical M0C recursion, resetting state for this shot.

    The history arrays have shape ``(roles, 3, 25)``: three fixed kappas and
    one global plus 24 per-check channels.  Calling this function for another
    shot always starts from zero.
    """

    left = _binary_shot(reference, name="reference")
    right = _binary_shot(monitor, name="monitor")
    if left.shape != right.shape:
        raise ValueError("reference and monitor must have identical shape.")
    role_count = left.shape[0]
    channel_count = CHECK_COUNT + 1
    positive_state = np.zeros(
        (len(PAGE_CUSUM_KAPPAS), channel_count),
        dtype=np.float64,
    )
    negative_state = np.zeros_like(positive_state)
    positive_history = np.empty(
        (role_count, len(PAGE_CUSUM_KAPPAS), channel_count),
        dtype=np.float64,
    )
    negative_history = np.empty_like(positive_history)
    cycle_scores = np.empty(role_count, dtype=np.float64)
    kappas = np.asarray(PAGE_CUSUM_KAPPAS, dtype=np.float64)[:, None]

    for role in range(role_count):
        check_difference = right[role] - left[role]
        channels = np.concatenate(
            ([float(np.mean(check_difference))], check_difference)
        )
        positive_state = np.maximum(
            0.0,
            positive_state + channels[None, :] - kappas,
        )
        negative_state = np.maximum(
            0.0,
            negative_state - channels[None, :] - kappas,
        )
        positive_history[role] = positive_state
        negative_history[role] = negative_state
        cycle_scores[role] = max(
            float(np.max(positive_state)),
            float(np.max(negative_state)),
        )
    return PageCUSUMShotResult(
        cycle_scores=_readonly(cycle_scores),
        positive=_readonly(positive_state),
        negative=_readonly(negative_state),
        positive_history=_readonly(positive_history),
        negative_history=_readonly(negative_history),
    )


def _empirical_score_matrix(scores: ArrayLike) -> FloatArray:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 1:
        raise ValueError("scores must have shape (shots, roles).")
    if not np.all(np.isfinite(values)):
        raise ValueError("scores must contain only finite values.")
    return values


@dataclass(frozen=True)
class StrictShotAlerts:
    """One-alert-per-shot result for a strict empirical threshold."""

    threshold: float
    alert_count: int
    shot_scores: FloatArray
    shot_score_roles: IntArray
    shot_alerts: BoolArray
    first_crossing_roles: IntArray
    notifications: IntArray


def apply_strict_shot_threshold(
    scores: ArrayLike,
    threshold: float,
) -> StrictShotAlerts:
    """Apply strict ``>`` with one notification per shot.

    ``scores`` must already have been produced for every role, so notification
    suppression cannot suppress adaptive model updates.
    """

    values = _empirical_score_matrix(scores)
    threshold_array = np.asarray(threshold, dtype=np.float64)
    if threshold_array.ndim != 0 or np.isnan(float(threshold_array)):
        raise ValueError("threshold must be a scalar and not NaN.")
    threshold_value = float(threshold_array)
    crossings = values > threshold_value
    shot_alerts = np.any(crossings, axis=1)
    first_crossing_roles = np.full(values.shape[0], -1, dtype=np.int64)
    first_crossing_roles[shot_alerts] = np.argmax(
        crossings[shot_alerts],
        axis=1,
    )
    alert_rows = np.flatnonzero(shot_alerts)
    notifications = np.column_stack(
        (alert_rows, first_crossing_roles[alert_rows])
    ).astype(np.int64, copy=False)
    shot_score_roles = np.argmax(values, axis=1).astype(np.int64, copy=False)
    shot_scores = values[np.arange(values.shape[0]), shot_score_roles]
    return StrictShotAlerts(
        threshold=threshold_value,
        alert_count=int(np.sum(shot_alerts)),
        shot_scores=_readonly(shot_scores),
        shot_score_roles=_readonly(shot_score_roles, dtype=np.int64),
        shot_alerts=_readonly(shot_alerts, dtype=np.bool_),
        first_crossing_roles=_readonly(first_crossing_roles, dtype=np.int64),
        notifications=_readonly(notifications, dtype=np.int64),
    )


@dataclass(frozen=True)
class ThresholdSelection:
    """Smallest candidate satisfying a maximum shot-alert budget."""

    threshold: float
    max_alerts: int
    alert_count: int
    candidates: FloatArray
    shot_scores: FloatArray
    shot_score_roles: IntArray
    shot_alerts: BoolArray
    first_crossing_roles: IntArray
    notifications: IntArray


def select_strict_shot_threshold(
    scores: ArrayLike,
    max_alerts: int,
) -> ThresholdSelection:
    """Select the smallest ``{-inf, scores, +inf}`` strict threshold in budget.

    The alert count depends only on complete-shot maxima.  The selected finite
    threshold is therefore the ``(max_alerts + 1)``-th largest shot maximum
    (with zero-based index ``max_alerts``), avoiding a quadratic scan over all
    cycle-score candidates.  One final state-machine application constructs
    the notification table.
    """

    values = _empirical_score_matrix(scores)
    permitted = _nonnegative_integer(max_alerts, name="max_alerts")
    if permitted > values.shape[0]:
        raise ValueError("max_alerts cannot exceed the number of shots.")
    candidates = np.concatenate(
        (
            np.asarray([-np.inf]),
            np.unique(values),
            np.asarray([np.inf]),
        )
    )
    if permitted == values.shape[0]:
        threshold = -np.inf
    else:
        shot_maxima = np.max(values, axis=1)
        threshold = float(np.sort(shot_maxima)[::-1][permitted])
    chosen = apply_strict_shot_threshold(values, threshold)
    if chosen.alert_count > permitted:
        raise RuntimeError("analytical strict threshold violated its alert budget.")
    return ThresholdSelection(
        threshold=chosen.threshold,
        max_alerts=permitted,
        alert_count=chosen.alert_count,
        candidates=_readonly(candidates),
        shot_scores=chosen.shot_scores,
        shot_score_roles=chosen.shot_score_roles,
        shot_alerts=chosen.shot_alerts,
        first_crossing_roles=chosen.first_crossing_roles,
        notifications=chosen.notifications,
    )


@dataclass(frozen=True)
class ResourceCounts:
    """Exact paired-record exposure counts for one replay phase."""

    paired_shots: int
    physical_shots: int
    paired_role_updates: int
    detector_bits_exposed: int


def paired_resource_counts(
    num_pairs: int,
    *,
    num_roles: int = ROLE_COUNT,
    num_checks: int = CHECK_COUNT,
) -> ResourceCounts:
    """Return physical-shot, role-update, and detector-bit exposure counts."""

    pair_count = _nonnegative_integer(num_pairs, name="num_pairs")
    role_count = _positive_integer(num_roles, name="num_roles")
    check_count = _positive_integer(num_checks, name="num_checks")
    role_updates = pair_count * role_count
    return ResourceCounts(
        paired_shots=pair_count,
        physical_shots=2 * pair_count,
        paired_role_updates=role_updates,
        detector_bits_exposed=role_updates * 2 * check_count,
    )
