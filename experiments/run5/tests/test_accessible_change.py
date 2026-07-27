from __future__ import annotations

import itertools

import numpy as np
import pytest
from aoc.change_detection import (
    BoundedScoreSR,
    HiddenMarkovBlockSR,
    LikelihoodRatioSR,
    PredictableBoxWitness,
    PredictableSimplexWitness,
    StaticBoxWitness,
    StaticSimplexWitness,
    effect_from_direction,
    paired_bootstrap_mean_difference,
)
from aoc.surface_code import PeriodicSurfaceSyndromeModel


def test_raw_simplex_effect_is_positive_support_and_predictable() -> None:
    reference = np.asarray([0.4, 0.35, 0.25])
    witness = PredictableSimplexWitness(reference, mode="raw", window=2)
    first = np.asarray([0.2, 0.7, 0.1])
    record1 = witness.update(first)
    assert record1.score == 0.0
    assert np.allclose(record1.effect, 0.0)

    second = np.asarray([0.1, 0.8, 0.1])
    record2 = witness.update(second)
    assert np.array_equal(record2.effect, np.asarray([0.0, 1.0, 0.0]))
    assert np.isclose(record2.score, second[1] - reference[1])


def test_variance_effect_matches_regularized_direction_up_to_affine_map() -> None:
    reference = np.asarray([0.5, 0.3, 0.2])
    covariance = np.diag([0.2, 0.05, 0.1])
    ridge = 0.01
    live = np.asarray([0.4, 0.5, 0.1])
    witness = PredictableSimplexWitness(
        reference,
        mode="variance",
        null_covariance=covariance,
        ridge=ridge,
    )
    witness.update(live)
    direction = np.linalg.solve(
        covariance + ridge * np.eye(3),
        live - reference,
    )
    assert np.allclose(witness.effect, effect_from_direction(direction))
    assert np.all((witness.effect >= 0.0) & (witness.effect <= 1.0))


def test_simplex_effect_scores_are_bounded() -> None:
    rng = np.random.default_rng(5001)
    reference = rng.dirichlet(np.ones(11))
    effect = rng.uniform(size=11)
    witness = StaticSimplexWitness(reference, effect)
    scores = [witness.score(rng.dirichlet(np.ones(11))) for _ in range(1000)]
    assert min(scores) >= -1.0
    assert max(scores) <= 1.0


def test_box_witness_is_bounded_and_predictable() -> None:
    reference = np.asarray([0.2, -0.1, 0.4])
    covariance = np.diag([0.3, 0.1, 0.5])
    witness = PredictableBoxWitness(
        reference,
        -1.0,
        1.0,
        mode="variance",
        null_covariance=covariance,
        ridge=0.01,
    )
    first = witness.update([0.7, 0.4, -0.2])
    assert first.score == 0.0
    rng = np.random.default_rng(99)
    scores = []
    for _ in range(500):
        scores.append(witness.update(rng.uniform(-1.0, 1.0, 3)).score)
    assert min(scores) >= -1.0
    assert max(scores) <= 1.0

    static = StaticBoxWitness(reference, [2.0, -3.0, 1.0], -1.0, 1.0)
    static_scores = [
        static.score(rng.uniform(-1.0, 1.0, 3)) for _ in range(500)
    ]
    assert min(static_scores) >= -1.0
    assert max(static_scores) <= 1.0


def test_zero_score_sr_statistic_equals_time_and_alarms_at_threshold() -> None:
    detector = BoundedScoreSR(threshold=25.0)
    for time in range(1, 25):
        update = detector.update(0.0)
        assert np.isclose(update.statistic, time)
        assert not update.alarm
    update = detector.update(0.0)
    assert np.isclose(update.statistic, 25.0)
    assert update.alarm_time == 25


def test_likelihood_sr_null_mean_identity_by_exact_two_point_average() -> None:
    detector_a = LikelihoodRatioSR(1, threshold=100.0)
    detector_b = LikelihoodRatioSR(1, threshold=100.0)
    # Under a fair null, L is 0.5 or 1.5 with equal probability and E[L]=1.
    value_a = detector_a.update([0.5]).statistic
    value_b = detector_b.update([1.5]).statistic
    assert np.isclose((value_a + value_b) / 2.0, 1.0)


def test_hidden_markov_block_sr_injects_stationary_candidate() -> None:
    model = PeriodicSurfaceSyndromeModel(
        size=3,
        event_probability=0.4,
        readout_error=0.08,
        allow_small_for_test=True,
    )
    q = 0.37
    kappa = 0.62
    observed = model.sample_temporal(2, q=q, kappa=kappa, seed=991)[0]
    null_logs = model.emission_log_likelihoods(observed, q)
    emission_ratios = np.stack(
        [
            np.exp(
                model.conditional_length_emission_log_likelihoods(
                    observed,
                    length,
                )
                - null_logs
            )
            for length in (1, 2)
        ],
        axis=1,
    )
    detector = HiddenMarkovBlockSR(
        model.length_transition_matrix(q, kappa)[None, :, :],
        [1.0 - q, q],
        threshold=100.0,
    )
    update = detector.update(emission_ratios[0], emission_ratios[1])
    expected = np.exp(
        model.nonoverlapping_pair_log_likelihood(
            observed[0],
            observed[1],
            q=q,
            kappa=kappa,
        )
        - model.nonoverlapping_pair_log_likelihood(
            observed[0],
            observed[1],
            q=q,
            kappa=0.0,
        )
    )
    assert np.isclose(update.statistic, expected)


def test_hidden_markov_block_sr_has_null_expectation_equal_to_block_time() -> None:
    transition = np.asarray([[0.8, 0.2], [0.3, 0.7]])
    stationary = np.asarray([0.6, 0.4])
    emission_ratio = {
        0: np.asarray([1.5, 0.5]),
        1: np.asarray([0.5, 1.5]),
    }
    statistics = []
    for observations in itertools.product((0, 1), repeat=4):
        detector = HiddenMarkovBlockSR(
            transition[None, :, :],
            stationary,
            threshold=100.0,
        )
        detector.update(
            emission_ratio[observations[0]],
            emission_ratio[observations[1]],
        )
        statistics.append(
            detector.update(
                emission_ratio[observations[2]],
                emission_ratio[observations[3]],
            ).statistic
        )
    assert np.isclose(np.mean(statistics), 2.0)


def test_invalid_score_and_probability_vector_are_rejected() -> None:
    detector = BoundedScoreSR()
    with pytest.raises(ValueError):
        detector.update(1.1)
    with pytest.raises(ValueError):
        PredictableSimplexWitness([0.4, 0.4])


def test_paired_bootstrap_uses_paired_differences() -> None:
    first = np.asarray([3.0, 4.0, 5.0, 6.0])
    second = first - 2.0
    estimate, low, high = paired_bootstrap_mean_difference(
        first,
        second,
        resamples=500,
        seed=9,
    )
    assert estimate == 2.0
    assert low == 2.0
    assert high == 2.0
