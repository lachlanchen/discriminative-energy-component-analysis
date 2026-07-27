"""Predictable-witness streaming contrast and anytime evidence."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import logsumexp

from .contrast import effect_expectation, maximum_observable_contrast
from .states import SlidingState, as_density_matrix, pure_state_density


@dataclass(frozen=True)
class SequentialRecord:
    time: int
    score: float
    contrast: float
    e_value: float
    anytime_p_value: float
    alarm: bool
    witness_rank: int


class PredictableContrastEProcess:
    """Learn a contrast effect from the past and test it on the next state.

    Under an i.i.d. null with *known* reference density ``rho0``, the current
    effect is predictable and ``Tr(E_t R_t) - Tr(E_t rho0)`` has conditional
    mean zero and lies in ``[-1, 1]``. For every fixed bet fraction
    ``lambda`` in ``(0, 1)``,

    ``product(1 + lambda * score_t)``

    is a test martingale. Unlike a worst-case Hoeffding bound, this bounded
    betting process automatically benefits when window-level scores have low
    variance. The implementation mixes bet fractions and candidate start
    times with summable weights. If ``rho0`` is estimated from finite
    calibration data, the returned value is a useful diagnostic but does not
    by itself retain the stated exact type-I guarantee.
    """

    def __init__(
        self,
        reference: ArrayLike,
        *,
        adaptation_window: int = 64,
        bet_fractions: ArrayLike | None = None,
        witness_rank: int | None = None,
        alpha: float = 0.01,
        tolerance: float = 1e-12,
    ) -> None:
        self.reference = as_density_matrix(reference)
        self.dimension = self.reference.shape[0]
        self.live = SlidingState(self.dimension, adaptation_window)
        if bet_fractions is None:
            bet_fractions = np.geomspace(0.01, 0.99, 20)
        self.bet_fractions = np.asarray(bet_fractions, dtype=np.float64)
        if (
            self.bet_fractions.ndim != 1
            or len(self.bet_fractions) == 0
            or np.any(self.bet_fractions <= 0)
            or np.any(self.bet_fractions >= 1)
        ):
            raise ValueError("bet_fractions must lie strictly between 0 and 1.")
        if not 0 < alpha < 1:
            raise ValueError("alpha must lie in (0, 1).")
        self.alpha = float(alpha)
        self.tolerance = float(tolerance)
        if witness_rank is not None and witness_rank < 0:
            raise ValueError("witness_rank must be nonnegative.")
        self.witness_rank = witness_rank
        self.time = 0
        self._candidate_logs = np.empty((0, len(self.bet_fractions)))
        self._start_weights = np.empty(0, dtype=np.float64)
        self._bet_log_weights = np.full(
            len(self.bet_fractions),
            -np.log(len(self.bet_fractions)),
        )
        self.max_e_value = 1.0
        self.records: list[SequentialRecord] = []
        self.last_effect = np.zeros_like(self.reference)

    @staticmethod
    def _start_weight(index: int) -> float:
        return float(6.0 / (np.pi**2 * index**2))

    def _current_effect(self):
        if len(self.live) == 0:
            return np.zeros_like(self.reference), 0.0, 0
        result = maximum_observable_contrast(
            self.live.density,
            self.reference,
            rank=self.witness_rank,
            tolerance=self.tolerance,
        )
        return result.effect, result.positive_gap, result.rank

    def update(self, sample: ArrayLike) -> SequentialRecord:
        raw = np.asarray(sample)
        state = pure_state_density(raw) if raw.ndim == 1 else as_density_matrix(raw)
        if state.shape != self.reference.shape:
            raise ValueError("Sample dimension does not match the reference.")

        effect, contrast, witness_rank = self._current_effect()
        self.last_effect = effect
        observed = effect_expectation(effect, state)
        expected = effect_expectation(effect, self.reference)
        score = float(observed - expected)

        self.time += 1
        increment = np.log1p(self.bet_fractions * score)
        if len(self._candidate_logs):
            self._candidate_logs += increment[None, :]
        self._candidate_logs = np.vstack([self._candidate_logs, increment[None, :]])
        start_weight = self._start_weight(self.time)
        self._start_weights = np.append(self._start_weights, start_weight)

        per_start_log = logsumexp(
            self._candidate_logs + self._bet_log_weights[None, :],
            axis=1,
        )
        started_mass = float(self._start_weights.sum())
        tail_mass = max(0.0, 1.0 - started_mass)
        terms = np.log(self._start_weights) + per_start_log
        if tail_mass > 0:
            total_log = float(logsumexp(np.append(terms, np.log(tail_mass))))
        else:
            total_log = float(logsumexp(terms))
        e_value = float(np.exp(min(total_log, 700.0)))
        self.max_e_value = max(self.max_e_value, e_value)
        anytime_p = min(1.0, 1.0 / self.max_e_value)
        alarm = self.max_e_value >= 1.0 / self.alpha

        self.live.add(state)
        record = SequentialRecord(
            time=self.time,
            score=score,
            contrast=contrast,
            e_value=e_value,
            anytime_p_value=anytime_p,
            alarm=alarm,
            witness_rank=witness_rank,
        )
        self.records.append(record)
        return record
