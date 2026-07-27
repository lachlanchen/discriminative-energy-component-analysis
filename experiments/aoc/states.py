"""Positive operator states and exact additive streaming statistics."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

ComplexMatrix = NDArray[np.complex128]


def _hermitian_part(matrix: ArrayLike) -> ComplexMatrix:
    array = np.asarray(matrix, dtype=np.complex128)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError("A state must be a square matrix.")
    return 0.5 * (array + array.conj().T)


def as_density_matrix(
    matrix: ArrayLike,
    *,
    tolerance: float = 1e-9,
    normalize: bool = True,
) -> ComplexMatrix:
    """Validate a positive operator and optionally normalize its trace to one."""

    raw = np.asarray(matrix, dtype=np.complex128)
    hermitian = _hermitian_part(raw)
    hermitian_error = np.linalg.norm(raw - raw.conj().T, ord="fro")
    scale = max(1.0, np.linalg.norm(hermitian, ord="fro"))
    if hermitian_error > tolerance * scale:
        raise ValueError("State is not Hermitian within tolerance.")
    eigenvalues = np.linalg.eigvalsh(hermitian)
    if float(eigenvalues[0]) < -tolerance * scale:
        raise ValueError("State is not positive semidefinite.")
    trace = float(np.real(np.trace(hermitian)))
    if not np.isfinite(trace) or trace <= tolerance:
        raise ValueError("State must have finite positive trace.")
    if normalize:
        hermitian = hermitian / trace
    return hermitian


def pure_state_density(vector: ArrayLike) -> ComplexMatrix:
    """Return ``|x><x|`` after normalizing a nonzero real or complex vector."""

    array = np.asarray(vector, dtype=np.complex128)
    if array.ndim != 1:
        raise ValueError("A pure-state vector must be one-dimensional.")
    norm = float(np.linalg.norm(array))
    if not np.isfinite(norm) or norm <= 1e-15:
        raise ValueError("A pure-state vector must have nonzero finite norm.")
    state = array / norm
    return np.outer(state, state.conj())


def _coerce_state(sample: ArrayLike, dimension: int | None = None) -> ComplexMatrix:
    array = np.asarray(sample)
    state = pure_state_density(array) if array.ndim == 1 else as_density_matrix(array)
    if dimension is not None and state.shape != (dimension, dimension):
        raise ValueError(
            f"Expected a {dimension}x{dimension} state, received {state.shape}."
        )
    return state


@dataclass
class AdditiveState:
    """Mergeable sufficient statistic ``(mass, sum of positive operators)``.

    Each incoming sample is normalized to trace one. The accumulator itself is
    deliberately left unnormalized so additions, removals, merges, and decay
    are exact up to floating-point arithmetic.
    """

    dimension: int
    total_weight: float = 0.0
    accumulator: ComplexMatrix = field(init=False, repr=False)
    samples_seen: int = 0

    def __post_init__(self) -> None:
        if self.dimension <= 0:
            raise ValueError("dimension must be positive.")
        self.accumulator = np.zeros(
            (self.dimension, self.dimension), dtype=np.complex128
        )

    def add(self, sample: ArrayLike, weight: float = 1.0) -> AdditiveState:
        if not np.isfinite(weight) or weight <= 0:
            raise ValueError("weight must be finite and positive.")
        state = _coerce_state(sample, self.dimension)
        self.accumulator += float(weight) * state
        self.total_weight += float(weight)
        self.samples_seen += 1
        return self

    def remove(
        self,
        sample: ArrayLike,
        weight: float = 1.0,
        *,
        tolerance: float = 1e-10,
    ) -> AdditiveState:
        if not np.isfinite(weight) or weight <= 0:
            raise ValueError("weight must be finite and positive.")
        if weight > self.total_weight + tolerance:
            raise ValueError("Cannot remove more mass than is stored.")
        state = _coerce_state(sample, self.dimension)
        candidate = self.accumulator - float(weight) * state
        candidate_mass = self.total_weight - float(weight)
        if candidate_mass <= tolerance:
            self.accumulator.fill(0.0)
            self.total_weight = 0.0
        else:
            minimum = float(np.min(np.linalg.eigvalsh(_hermitian_part(candidate))))
            if minimum < -tolerance * max(1.0, candidate_mass):
                raise ValueError("Removal would make the accumulator nonpositive.")
            self.accumulator = _hermitian_part(candidate)
            self.total_weight = candidate_mass
        return self

    def decay(self, factor: float) -> AdditiveState:
        """Multiply both mass and accumulator by a forgetting factor."""

        if not np.isfinite(factor) or not 0 < factor <= 1:
            raise ValueError("factor must lie in (0, 1].")
        self.accumulator *= float(factor)
        self.total_weight *= float(factor)
        return self

    def merge(self, other: AdditiveState) -> AdditiveState:
        if other.dimension != self.dimension:
            raise ValueError("Only states of equal dimension can be merged.")
        self.accumulator += other.accumulator
        self.total_weight += other.total_weight
        self.samples_seen += other.samples_seen
        return self

    def copy(self) -> AdditiveState:
        duplicate = AdditiveState(self.dimension)
        duplicate.total_weight = self.total_weight
        duplicate.accumulator = self.accumulator.copy()
        duplicate.samples_seen = self.samples_seen
        return duplicate

    @property
    def density(self) -> ComplexMatrix:
        if self.total_weight <= 0:
            raise ValueError("An empty additive state has no density.")
        return _hermitian_part(self.accumulator / self.total_weight)

    @classmethod
    def from_samples(
        cls,
        samples: Iterable[ArrayLike],
        weights: Iterable[float] | None = None,
    ) -> AdditiveState:
        sample_list = list(samples)
        if not sample_list:
            raise ValueError("At least one sample is required.")
        first = _coerce_state(sample_list[0])
        result = cls(first.shape[0])
        if weights is None:
            weight_list = np.ones(len(sample_list), dtype=np.float64)
        else:
            weight_list = np.asarray(list(weights), dtype=np.float64)
            if weight_list.shape != (len(sample_list),):
                raise ValueError("weights must match the number of samples.")
        for sample, weight in zip(sample_list, weight_list):
            result.add(sample, float(weight))
        return result


class SlidingState:
    """Exact fixed-size sliding window backed by :class:`AdditiveState`."""

    def __init__(self, dimension: int, window_size: int) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be positive.")
        self.window_size = int(window_size)
        self.statistic = AdditiveState(dimension)
        self._items: deque[tuple[ComplexMatrix, float]] = deque()

    def add(self, sample: ArrayLike, weight: float = 1.0) -> SlidingState:
        state = _coerce_state(sample, self.statistic.dimension)
        if len(self._items) == self.window_size:
            old_state, old_weight = self._items.popleft()
            self.statistic.remove(old_state, old_weight)
        self._items.append((state, float(weight)))
        self.statistic.add(state, weight)
        return self

    @property
    def density(self) -> ComplexMatrix:
        return self.statistic.density

    @property
    def full(self) -> bool:
        return len(self._items) == self.window_size

    def __len__(self) -> int:
        return len(self._items)
