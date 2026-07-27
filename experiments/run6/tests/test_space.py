from __future__ import annotations

import numpy as np
import pytest
from aoc.space import (
    EWMASpectralWitness,
    EWMATopKWitness,
    MixtureSRBank,
    PairwiseOnlineLogistic,
    ProperUniformStartEProcessBank,
    capped_simplex_top_k_extreme,
    linear_bet_factors,
    top_k_signed_extreme,
    validate_bounded_score,
)


def diagonal(*values: float) -> np.ndarray:
    return np.diag(np.asarray(values, dtype=np.float64))


def test_top_k_signed_extreme_has_lexicographic_tie_rule() -> None:
    direction = np.asarray([2.0, -2.0, 2.0, -1.0])
    witness = top_k_signed_extreme(direction, 2)
    np.testing.assert_array_equal(witness, [0.5, -0.5, 0.0, 0.0])
    assert np.isclose(witness @ direction, 2.0)
    assert np.isclose(np.abs(witness).sum(), 1.0)


def test_capped_simplex_extreme_matches_signed_coefficients() -> None:
    direction = np.asarray([3.0, -2.0, 1.0])
    weights = capped_simplex_top_k_extreme(direction, 2)
    assert weights.shape == (6,)
    assert np.all(weights >= 0.0)
    assert np.isclose(weights.sum(), 1.0)
    assert np.max(weights) <= 0.5
    signed = weights[:3] - weights[3:]
    np.testing.assert_array_equal(
        signed,
        top_k_signed_extreme(direction, 2),
    )
    assert np.isclose(weights @ np.concatenate([direction, -direction]), 2.5)


def test_zero_direction_static_extreme_uses_positive_low_indices() -> None:
    witness = top_k_signed_extreme(np.zeros(5), 3)
    np.testing.assert_array_equal(witness, [1 / 3, 1 / 3, 1 / 3, 0.0, 0.0])


def test_top_k_extreme_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        top_k_signed_extreme([1.0, 2.0], 0)
    with pytest.raises(ValueError):
        top_k_signed_extreme([1.0, 2.0], 3)
    with pytest.raises(ValueError):
        top_k_signed_extreme([1.0, np.nan], 1)


def test_ewma_top_k_uses_declared_half_life_and_scores_before_update() -> None:
    witness = EWMATopKWitness(2, half_life=1.0, k=1)
    assert witness.alpha == 0.5
    first = witness.update([1.0, 0.0])
    assert first.score == 0.0
    np.testing.assert_array_equal(first.witness, [0.0, 0.0])
    np.testing.assert_allclose(first.ewma_after, [0.5, 0.0])

    second = witness.update([-1.0, 0.5])
    np.testing.assert_array_equal(second.witness, [1.0, 0.0])
    assert second.score == -1.0
    np.testing.assert_allclose(second.ewma_before, [0.5, 0.0])
    np.testing.assert_allclose(second.ewma_after, [-0.25, 0.25])


def test_ewma_top_k_current_value_cannot_change_used_witness() -> None:
    first = EWMATopKWitness(3, half_life=4.0, k=1)
    second = EWMATopKWitness(3, half_life=4.0, k=1)
    for observed in ([0.0, 0.8, 0.1], [0.0, 0.4, -0.2]):
        first.update(observed)
        second.update(observed)
    left = first.update([1.0, 0.0, 0.0])
    right = second.update([-1.0, 0.0, 0.0])
    np.testing.assert_array_equal(left.witness, right.witness)
    np.testing.assert_array_equal(left.ewma_before, right.ewma_before)


def test_ewma_top_k_reset_and_replay_are_deterministic() -> None:
    sequence = np.asarray(
        [
            [1.0, 0.0, -0.5],
            [0.0, 1.0, 0.2],
            [-0.4, 0.3, 0.7],
        ]
    )
    witness = EWMATopKWitness(3, half_life=2.0, k=2)
    first = [witness.update(value).score for value in sequence]
    ending = witness.ewma.copy()
    witness.reset()
    second = [witness.update(value).score for value in sequence]
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(witness.ewma, ending)
    assert witness.time == len(sequence)


def test_ewma_top_k_scores_remain_bounded() -> None:
    rng = np.random.default_rng(610601)
    witness = EWMATopKWitness(17, half_life=16.0, k=4)
    scores = [witness.update(rng.uniform(-1.0, 1.0, 17)).score for _ in range(1000)]
    assert min(scores) >= -1.0
    assert max(scores) <= 1.0


def test_spectral_jordan_witness_scores_before_update() -> None:
    witness = EWMASpectralWitness(
        2,
        half_life=1.0,
        rank="positive",
    )
    first = witness.update(diagonal(1.0, -1.0))
    assert first.score == 0.0
    np.testing.assert_array_equal(first.effect, np.zeros((2, 2)))
    np.testing.assert_allclose(witness.effect, diagonal(1.0, 0.0))

    second = witness.update(diagonal(-1.0, 1.0))
    np.testing.assert_allclose(second.effect, diagonal(1.0, 0.0))
    assert second.score == -1.0


def test_spectral_rank_one_degeneracy_uses_lexicographic_anchor() -> None:
    contrast = diagonal(0.5, 0.5, -1.0)
    rank_one = EWMASpectralWitness(3, half_life=2.0, rank=1)
    positive = EWMASpectralWitness(3, half_life=2.0, rank="positive")
    rank_one.update(contrast)
    positive.update(contrast)
    np.testing.assert_allclose(rank_one.effect, diagonal(1.0, 0.0, 0.0))
    np.testing.assert_allclose(positive.effect, diagonal(1.0, 1.0, 0.0))


def test_spectral_stride_effect_starts_on_following_observation() -> None:
    contrast = diagonal(1.0, -1.0)
    witness = EWMASpectralWitness(
        2,
        half_life=4.0,
        rank="positive",
        update_stride=2,
    )
    first = witness.update(contrast)
    second = witness.update(contrast)
    third = witness.update(contrast)
    assert first.score == 0.0
    assert second.score == 0.0
    np.testing.assert_allclose(third.effect, diagonal(1.0, 0.0))
    assert third.score == 1.0


def test_spectral_witness_reset_and_replay_are_deterministic() -> None:
    sequence = [
        diagonal(0.5, 0.5, -1.0),
        diagonal(-0.3, 0.8, -0.5),
        diagonal(0.7, -0.2, -0.5),
    ]
    witness = EWMASpectralWitness(
        3,
        half_life=4.0,
        rank=1,
        update_stride=1,
    )
    first = [witness.update(value).score for value in sequence]
    effect = witness.effect.copy()
    witness.reset()
    second = [witness.update(value).score for value in sequence]
    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(witness.effect, effect)


def test_spectral_operator_validation_rejects_invalid_contrasts() -> None:
    witness = EWMASpectralWitness(2, half_life=4.0)
    with pytest.raises(ValueError, match="Hermitian"):
        witness.update([[0.0, 1.0], [0.0, 0.0]])
    with pytest.raises(ValueError, match="zero trace"):
        witness.update(diagonal(1.0, 0.0))
    with pytest.raises(ValueError, match="trace norm"):
        witness.update(diagonal(2.0, -2.0))


def test_spectral_scores_remain_bounded_on_density_differences() -> None:
    rng = np.random.default_rng(610602)
    witness = EWMASpectralWitness(5, half_life=16.0, rank="positive")
    scores = []
    for _ in range(300):
        left = rng.normal(size=5)
        left /= np.linalg.norm(left)
        right = rng.normal(size=5)
        right /= np.linalg.norm(right)
        difference = np.outer(right, right) - np.outer(left, left)
        scores.append(witness.update(difference).score)
    assert min(scores) >= -1.0
    assert max(scores) <= 1.0


def test_pairwise_logistic_scores_before_sgd_update() -> None:
    learner = PairwiseOnlineLogistic(
        2,
        learning_rate=0.1,
        l2=0.0,
    )
    first = learner.update([0.0, 0.0], [1.0, 0.0])
    assert first.score == 0.0
    np.testing.assert_array_equal(first.weights, [0.0, 0.0])
    np.testing.assert_allclose(learner.weights, [0.05, 0.0])

    second = learner.update([0.0, 0.0], [1.0, 0.0])
    assert second.score > 0.0
    np.testing.assert_allclose(second.weights, [0.05, 0.0])


def test_pairwise_logistic_score_is_antisymmetric_without_update() -> None:
    learner = PairwiseOnlineLogistic(3, learning_rate=0.1, l2=0.01)
    learner.update([0.0, 0.2, 0.1], [1.0, 0.4, 0.7])
    left = np.asarray([0.1, 0.8, 0.2])
    right = np.asarray([0.9, 0.3, 0.7])
    assert np.isclose(learner.score(left, right), -learner.score(right, left))


def test_pairwise_logistic_reset_determinism_and_bounds() -> None:
    rng = np.random.default_rng(610603)
    pairs = [(rng.uniform(size=6), rng.uniform(size=6)) for _ in range(200)]
    learner = PairwiseOnlineLogistic(6, learning_rate=0.1, l2=1e-4)
    first = [learner.update(left, right).score for left, right in pairs]
    ending = learner.weights.copy()
    learner.reset()
    second = [learner.update(left, right).score for left, right in pairs]
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(learner.weights, ending)
    assert min(first) >= -1.0
    assert max(first) <= 1.0


def test_bounded_score_and_linear_bet_validation() -> None:
    assert validate_bounded_score(1.0 + 5e-13) == 1.0
    assert validate_bounded_score(-1.0 - 5e-13) == -1.0
    with pytest.raises(ValueError):
        validate_bounded_score(1.01)
    with pytest.raises(ValueError):
        validate_bounded_score(np.nan)
    with pytest.raises(ValueError):
        validate_bounded_score([0.0, 0.1])  # type: ignore[arg-type]

    factors = linear_bet_factors(0.5, [0.2, 0.8], two_sided=True)
    np.testing.assert_allclose(factors, [1.1, 0.9, 1.4, 0.6])
    with pytest.raises(ValueError):
        linear_bet_factors(0.0, [0.0])


def test_uniform_start_bank_is_one_for_unit_factors() -> None:
    bank = ProperUniformStartEProcessBank(
        3,
        horizon=5,
        alpha=0.01,
    )
    for time in range(1, 6):
        update = bank.update(np.ones(3))
        assert update.time == time
        assert np.isclose(update.statistic, 1.0)
        assert np.isclose(update.log_statistic, 0.0, atol=1e-15)
        assert not update.alarm
    with pytest.raises(RuntimeError, match="horizon"):
        bank.update(np.ones(3))


def test_uniform_start_bank_matches_closed_form_and_resets() -> None:
    bank = ProperUniformStartEProcessBank(1, horizon=2, alpha=0.1)
    first = bank.update([2.0])
    second = bank.update([2.0])
    assert np.isclose(first.statistic, 1.5)
    assert np.isclose(second.statistic, 3.0)
    assert second.alarm_time is None
    bank.reset()
    assert bank.time == 0
    assert bank.statistic == 1.0
    np.testing.assert_array_equal(bank.log_components, [-np.inf])


def test_uniform_start_bank_matches_explicit_start_and_component_mixture() -> None:
    horizon = 4
    weights = np.asarray([0.25, 0.75])
    factors = np.asarray(
        [
            [1.2, 0.8],
            [0.9, 1.5],
            [1.1, 0.7],
        ]
    )
    bank = ProperUniformStartEProcessBank(
        2,
        horizon=horizon,
        component_weights=weights,
    )
    for time, current in enumerate(factors, start=1):
        update = bank.update(current)
        component_values = []
        for component in range(2):
            started = sum(
                np.prod(factors[start:time, component]) for start in range(time)
            )
            component_values.append(started / horizon)
        expected = (horizon - time) / horizon + weights @ component_values
        assert np.isclose(update.statistic, expected)


def test_uniform_start_bank_remains_finite_in_log_domain() -> None:
    bank = ProperUniformStartEProcessBank(1, horizon=3)
    bank.update([1e300])
    update = bank.update([1e300])
    assert np.isfinite(update.log_statistic)
    assert np.isfinite(update.statistic)
    assert update.statistic == np.finfo(np.float64).max


def test_mixture_sr_unit_factors_are_a_clock_and_alarm_is_sticky() -> None:
    bank = MixtureSRBank(4, gamma=3.0)
    for time in range(1, 4):
        update = bank.update(np.ones(4))
        assert np.isclose(update.statistic, time)
    assert update.alarm
    assert update.alarm_time == 3
    continued = bank.update(np.ones(4))
    assert continued.alarm
    assert continued.alarm_time == 3
    assert np.isclose(continued.statistic, 4.0)


def test_formal_threshold_comparison_is_not_tolerance_relaxed() -> None:
    e_bank = ProperUniformStartEProcessBank(1, horizon=1, alpha=0.01)
    e_update = e_bank.update([np.nextafter(100.0, 0.0)])
    assert e_update.log_statistic < np.log(100.0)
    assert e_update.alarm is False
    assert (
        ProperUniformStartEProcessBank(
            1,
            horizon=1,
            alpha=0.01,
        )
        .update([100.0])
        .alarm
    )
    assert (
        ProperUniformStartEProcessBank(
            1,
            horizon=1,
            alpha=0.01,
        )
        .update([np.nextafter(100.0, np.inf)])
        .alarm
    )

    sr_bank = MixtureSRBank(1, gamma=2.0)
    sr_update = sr_bank.update([np.nextafter(2.0, 0.0)])
    assert sr_update.log_statistic < np.log(2.0)
    assert sr_update.alarm is False
    assert MixtureSRBank(1, gamma=2.0).update([2.0]).alarm
    assert MixtureSRBank(1, gamma=2.0).update([np.nextafter(2.0, np.inf)]).alarm


def test_mixture_sr_matches_component_recursion_and_resets() -> None:
    bank = MixtureSRBank(1, gamma=100.0)
    assert np.isclose(bank.update([2.0]).statistic, 2.0)
    assert np.isclose(bank.update([0.5]).statistic, 1.5)
    bank.reset()
    assert bank.time == 0
    assert bank.statistic == 0.0
    assert bank.alarm_time is None


def test_mixture_sr_matches_explicit_multi_component_recursion() -> None:
    weights = np.asarray([0.4, 0.6])
    factors = np.asarray(
        [
            [1.2, 0.8],
            [0.5, 1.4],
            [1.1, 0.9],
        ]
    )
    bank = MixtureSRBank(
        2,
        gamma=100.0,
        component_weights=weights,
    )
    components = np.zeros(2)
    for current in factors:
        components = (components + 1.0) * current
        update = bank.update(current)
        assert np.isclose(update.statistic, weights @ components)


def test_component_banks_are_deterministic_and_validate_factors() -> None:
    sequence = [
        np.asarray([1.2, 0.8]),
        np.asarray([0.7, 1.4]),
        np.asarray([1.1, 0.9]),
    ]
    first = MixtureSRBank(2, gamma=100.0, component_weights=[0.3, 0.7])
    second = MixtureSRBank(2, gamma=100.0, component_weights=[0.3, 0.7])
    observed_first = [first.update(value).log_statistic for value in sequence]
    observed_second = [second.update(value).log_statistic for value in sequence]
    np.testing.assert_array_equal(observed_first, observed_second)

    with pytest.raises(ValueError):
        first.update([-0.1, 1.0])
    with pytest.raises(ValueError):
        first.update([np.inf, 1.0])
    with pytest.raises(ValueError):
        MixtureSRBank(2, gamma=10.0, component_weights=[0.2, 0.2])
