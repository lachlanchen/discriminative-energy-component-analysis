"""Synthetic API and invariant tests for the Run 6 QEC method layer.

These tests use constructed binary records only.  They do not read, infer, or
depend on any held Run 6 detector values.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass

import numpy as np
import pytest
from aoc.space import EWMASpectralWitness, EWMATopKWitness
from aoc.space_qec import (
    CHECK_COUNT,
    FEATURE_DIM,
    PAIR_FEATURE_ORDER,
    ROLE_COUNT,
    DiagonalLikelihoodModel,
    RoleHotellingModel,
    RoleIsolatedQECBank,
    apply_strict_shot_threshold,
    combine_space_factor_bank,
    detector_firing_rate,
    exact_component_priors,
    paired_page_cusum_shot,
    paired_qec_contrasts,
    paired_resource_counts,
    qec_density,
    qec_features,
    select_role_fit_indices,
    select_strict_shot_threshold,
)


def _contrast_array(record: object, *names: str) -> np.ndarray:
    """Read one semantically named contrast without constraining its dataclass."""

    for name in names:
        if hasattr(record, name):
            return np.asarray(getattr(record, name), dtype=np.float64)
    raise AssertionError(f"Contrast record is missing all of {names!r}.")


def _flatten_numeric(value: object) -> np.ndarray:
    if isinstance(value, Mapping):
        pieces = [_flatten_numeric(item) for item in value.values()]
        return np.concatenate(pieces) if pieces else np.empty(0)
    if isinstance(value, (tuple, list)):
        pieces = [_flatten_numeric(item) for item in value]
        return np.concatenate(pieces) if pieces else np.empty(0)
    if is_dataclass(value):
        pieces = [
            _flatten_numeric(getattr(value, field.name)) for field in fields(value)
        ]
        return np.concatenate(pieces) if pieces else np.empty(0)
    return np.asarray(value, dtype=np.float64).reshape(-1)


def test_locked_dimensions_and_pair_feature_order() -> None:
    assert CHECK_COUNT == 24
    assert FEATURE_DIM == 300
    assert ROLE_COUNT == 51
    assert len(PAIR_FEATURE_ORDER) == 276
    assert tuple(PAIR_FEATURE_ORDER[:4]) == (
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 4),
    )
    assert tuple(PAIR_FEATURE_ORDER[-3:]) == (
        (21, 22),
        (21, 23),
        (22, 23),
    )
    assert tuple(PAIR_FEATURE_ORDER) == tuple(
        (left, right)
        for left in range(CHECK_COUNT)
        for right in range(left + 1, CHECK_COUNT)
    )


def test_qec_features_density_rate_and_paired_contrasts_are_exact() -> None:
    reference = np.zeros(CHECK_COUNT, dtype=np.uint8)
    monitor = reference.copy()
    monitor[[0, 2, 23]] = 1

    features = qec_features(monitor)
    assert features.shape == (FEATURE_DIM,)
    assert np.all((features >= 0.0) & (features <= 1.0))
    np.testing.assert_array_equal(features[:CHECK_COUNT], monitor)
    expected_equalities = np.asarray(
        [monitor[left] == monitor[right] for left, right in PAIR_FEATURE_ORDER],
        dtype=np.float64,
    )
    np.testing.assert_array_equal(features[CHECK_COUNT:], expected_equalities)

    signs = 1.0 - 2.0 * monitor
    density = qec_density(monitor)
    np.testing.assert_allclose(density, np.outer(signs, signs) / CHECK_COUNT)
    np.testing.assert_allclose(density, density.T)
    assert np.isclose(np.trace(density), 1.0)
    assert np.linalg.eigvalsh(density).min() >= -1e-12
    assert detector_firing_rate(monitor) == 3.0 / CHECK_COUNT

    contrasts = paired_qec_contrasts(reference, monitor)
    np.testing.assert_array_equal(
        contrasts.reference_features,
        qec_features(reference),
    )
    np.testing.assert_array_equal(
        contrasts.monitor_features,
        qec_features(monitor),
    )
    np.testing.assert_array_equal(
        contrasts.reference_density,
        qec_density(reference),
    )
    np.testing.assert_array_equal(
        contrasts.monitor_density,
        qec_density(monitor),
    )
    feature_difference = _contrast_array(
        contrasts,
        "feature_difference",
        "sparse_difference",
        "sparse",
        "difference",
    )
    density_difference = _contrast_array(
        contrasts,
        "density_difference",
        "spectral_difference",
        "spectral",
        "delta",
    )
    rate_difference = _contrast_array(
        contrasts,
        "firing_rate_difference",
        "rate_difference",
        "global_difference",
        "global_rate",
    )
    np.testing.assert_allclose(
        feature_difference,
        qec_features(monitor) - qec_features(reference),
    )
    np.testing.assert_allclose(
        density_difference,
        qec_density(monitor) - qec_density(reference),
    )
    assert float(rate_difference) == detector_firing_rate(
        monitor
    ) - detector_firing_rate(reference)
    assert np.linalg.norm(density_difference, ord="nuc") <= 2.0 + 1e-12

    swapped = paired_qec_contrasts(monitor, reference)
    np.testing.assert_allclose(
        _contrast_array(
            swapped,
            "feature_difference",
            "sparse_difference",
            "sparse",
            "difference",
        ),
        -feature_difference,
    )
    np.testing.assert_allclose(
        _contrast_array(
            swapped,
            "density_difference",
            "spectral_difference",
            "spectral",
            "delta",
        ),
        -density_difference,
    )


def test_qec_feature_functions_reject_nonbinary_or_wrong_shape() -> None:
    with pytest.raises(ValueError):
        qec_features(np.zeros(CHECK_COUNT - 1))
    with pytest.raises(ValueError):
        qec_density(np.full(CHECK_COUNT, 0.5))
    with pytest.raises(ValueError):
        detector_firing_rate(np.full(CHECK_COUNT, np.nan))
    with pytest.raises(ValueError):
        paired_qec_contrasts(
            np.zeros(CHECK_COUNT),
            np.zeros((1, CHECK_COUNT)),
        )


def test_diagonal_likelihood_fit_and_score_match_closed_form() -> None:
    pairs = 4
    roles = 3
    reference = np.zeros((pairs, roles, CHECK_COUNT), dtype=np.uint8)
    monitor = np.zeros_like(reference)
    model = DiagonalLikelihoodModel.fit(reference, monitor, epsilon=1e-4)

    left = np.zeros(CHECK_COUNT, dtype=np.uint8)
    right = left.copy()
    right[0] = 1
    fitted_probability = 0.5 / (2 * pairs + 1)
    expected = (
        np.log((1.0 - fitted_probability) / fitted_probability)
        / CHECK_COUNT
        / np.log((1.0 - 1e-4) / 1e-4)
    )
    assert np.isclose(model.score(1, left, right), expected)
    assert np.isclose(model.score(1, right, left), -expected)
    assert model.score(1, left, left) == 0.0
    assert -1.0 <= model.score(1, left, right) <= 1.0


def test_role_fit_selection_is_exact_stratified_and_deterministic() -> None:
    selected = select_role_fit_indices(
        num_pairs=5000,
        num_roles=51,
        sample_size=20000,
        seed=610601,
    )
    replay = select_role_fit_indices(
        num_pairs=5000,
        num_roles=51,
        sample_size=20000,
        seed=610601,
    )
    changed_seed = select_role_fit_indices(
        num_pairs=5000,
        num_roles=51,
        sample_size=20000,
        seed=610602,
    )

    assert selected.shape == (20000, 2)
    np.testing.assert_array_equal(selected, replay)
    assert len(np.unique(selected, axis=0)) == 20000
    for role in range(51):
        current = selected[selected[:, 1] == role, 0]
        assert len(current) == (393 if role < 8 else 392)
        assert np.array_equal(current, np.sort(current))
        assert len(np.unique(current)) == len(current)
        assert current.min() >= 0
        assert current.max() < 5000
    assert not np.array_equal(selected, changed_seed)


def test_role_hotelling_fit_is_deterministic_role_centered_and_nonnegative() -> None:
    rng = np.random.default_rng(610603)
    differences = rng.uniform(-0.4, 0.4, size=(30, 3, FEATURE_DIM))
    differences[:, 0] = np.linspace(-0.2, 0.2, FEATURE_DIM)
    differences[:, 1] = np.clip(differences[:, 1] - 0.2, -1.0, 1.0)
    first = RoleHotellingModel.fit(differences, sample_size=60, seed=41)
    second = RoleHotellingModel.fit(differences, sample_size=60, seed=41)

    probes = (
        differences[0, 0],
        differences[:, 1].mean(axis=0) + 0.2,
        differences[:, 2].mean(axis=0) - 0.1,
    )
    first_scores = np.asarray(
        [first.score(role, probe) for role, probe in enumerate(probes)]
    )
    second_scores = np.asarray(
        [second.score(role, probe) for role, probe in enumerate(probes)]
    )
    assert np.all(np.isfinite(first_scores))
    assert np.all(first_scores >= 0.0)
    np.testing.assert_array_equal(first_scores, second_scores)
    assert np.isclose(first_scores[0], 0.0, atol=1e-12)


def test_exact_component_priors_and_space_composition() -> None:
    priors = exact_component_priors()
    expected_sizes = {
        "m0": 8,
        "m1": 8,
        "m3": 12,
        "m4": 64,
        "m5": 24,
        "space": 88,
    }
    for name, size in expected_sizes.items():
        prior = priors[name]
        weights = np.asarray(prior.weights, dtype=np.float64)
        assert len(prior.component_ids) == size
        assert weights.shape == (size,)
        assert np.all(weights > 0.0)
        assert np.isclose(weights.sum(), 1.0, atol=1e-15)
    np.testing.assert_allclose(priors["m0"].weights, 1.0 / 8.0)
    np.testing.assert_allclose(priors["m1"].weights, 1.0 / 8.0)
    np.testing.assert_allclose(priors["m3"].weights, 1.0 / 12.0)
    np.testing.assert_allclose(priors["m4"].weights, 1.0 / 64.0)
    np.testing.assert_allclose(priors["m5"].weights, 1.0 / 24.0)
    np.testing.assert_allclose(
        np.asarray(priors["space"].weights)[:64],
        1.0 / 128.0,
    )
    np.testing.assert_allclose(
        np.asarray(priors["space"].weights)[64:],
        1.0 / 48.0,
    )

    m4 = np.linspace(0.5, 1.5, 64)
    m5 = np.linspace(0.75, 1.25, 24)
    combined = combine_space_factor_bank(m4, m5)
    np.testing.assert_array_equal(
        combined.factors,
        np.concatenate((m4, m5)),
    )
    np.testing.assert_allclose(combined.weights, priors["space"].weights)


@pytest.fixture(scope="module")
def fitted_qec_models() -> tuple[DiagonalLikelihoodModel, RoleHotellingModel]:
    rng = np.random.default_rng(610604)
    roles = 3
    reference = rng.integers(
        0,
        2,
        size=(8, roles, CHECK_COUNT),
        dtype=np.uint8,
    )
    monitor = rng.integers(
        0,
        2,
        size=(8, roles, CHECK_COUNT),
        dtype=np.uint8,
    )
    diagonal = DiagonalLikelihoodModel.fit(reference, monitor, epsilon=1e-4)
    differences = rng.normal(scale=0.2, size=(16, roles, FEATURE_DIM))
    hotelling = RoleHotellingModel.fit(
        differences,
        sample_size=30,
        seed=610605,
    )
    return diagonal, hotelling


def test_role_bank_record_bounds_causality_and_role_isolation(
    fitted_qec_models: tuple[DiagonalLikelihoodModel, RoleHotellingModel],
) -> None:
    diagonal, hotelling = fitted_qec_models
    left = RoleIsolatedQECBank(
        role_count=3,
        diagonal_model=diagonal,
        hotelling_model=hotelling,
    )
    right = RoleIsolatedQECBank(
        role_count=3,
        diagonal_model=diagonal,
        hotelling_model=hotelling,
    )
    zeros = np.zeros(CHECK_COUNT, dtype=np.uint8)
    changed = zeros.copy()
    changed[[0, 5, 11]] = 1

    before_role_zero = _flatten_numeric(left.role_state_times(0))
    before_role_one = _flatten_numeric(left.role_state_times(1))
    before_role_two = _flatten_numeric(left.role_state_times(2))
    assert np.all(before_role_zero == 0)
    assert np.all(before_role_one == 0)
    assert np.all(before_role_two == 0)

    unchanged_record = left.update(1, zeros, zeros)
    changed_record = right.update(1, zeros, changed)
    required_fields = (
        "contrasts",
        "empirical",
        "m0_factors",
        "m1_factors",
        "m3_factors",
        "m4_factors",
        "m5_factors",
        "space_factors",
        "space_weights",
        "m3_scores",
        "m4_scores",
        "m5_scores",
    )
    for field in required_fields:
        assert hasattr(changed_record, field)

    # Every adaptive witness is uninformed on its first observation.  Changing
    # that current observation therefore cannot alter the score used at t=1.
    np.testing.assert_array_equal(
        unchanged_record.m3_scores,
        changed_record.m3_scores,
    )
    np.testing.assert_array_equal(
        unchanged_record.m4_scores,
        changed_record.m4_scores,
    )
    np.testing.assert_array_equal(
        unchanged_record.m5_scores,
        changed_record.m5_scores,
    )
    np.testing.assert_array_equal(changed_record.m3_scores, np.zeros(3))
    np.testing.assert_array_equal(changed_record.m4_scores, np.zeros(16))
    np.testing.assert_array_equal(changed_record.m5_scores, np.zeros(6))

    expected_shapes = {
        "m0_factors": (8,),
        "m1_factors": (8,),
        "m3_factors": (12,),
        "m4_factors": (64,),
        "m5_factors": (24,),
        "space_factors": (88,),
        "space_weights": (88,),
    }
    for field, shape in expected_shapes.items():
        values = np.asarray(getattr(changed_record, field), dtype=np.float64)
        assert values.shape == shape
        assert np.all(np.isfinite(values))
        assert np.all(values >= 0.0)
    assert np.isclose(np.asarray(changed_record.space_weights).sum(), 1.0)
    empirical = changed_record.empirical
    empirical_values = np.asarray(
        [
            empirical.m0,
            empirical.m1,
            empirical.m2,
            empirical.m3,
            empirical.m4,
            empirical.m5,
            empirical.space,
        ],
        dtype=np.float64,
    )
    assert np.all(np.isfinite(empirical_values))

    after_role_zero = _flatten_numeric(left.role_state_times(0))
    after_role_one = _flatten_numeric(left.role_state_times(1))
    after_role_two = _flatten_numeric(left.role_state_times(2))
    np.testing.assert_array_equal(after_role_zero, before_role_zero)
    np.testing.assert_array_equal(after_role_two, before_role_two)
    assert np.all(after_role_one == 1)

    left.reset()
    for role in range(3):
        assert np.all(_flatten_numeric(left.role_state_times(role)) == 0)


def test_shared_m4_m5_states_match_independent_space_primitives() -> None:
    bank = RoleIsolatedQECBank(role_count=1)
    sparse = [
        EWMATopKWitness(
            FEATURE_DIM,
            half_life=half_life,
            k=k,
        )
        for half_life in (4.0, 16.0, 64.0, 256.0)
        for k in (1, 4, 16, 64)
    ]
    spectral = [
        EWMASpectralWitness(
            CHECK_COUNT,
            half_life=half_life,
            rank=rank,
            update_stride=8,
            eigenvalue_tolerance=1e-10,
            validation_tolerance=1e-9,
        )
        for half_life in (4.0, 16.0, 64.0)
        for rank in (1, "positive")
    ]
    rng = np.random.default_rng(610606)
    for _ in range(12):
        reference = rng.integers(0, 2, CHECK_COUNT, dtype=np.uint8)
        monitor = rng.integers(0, 2, CHECK_COUNT, dtype=np.uint8)
        contrasts = paired_qec_contrasts(reference, monitor)
        actual = bank.update(0, reference, monitor)
        expected_sparse = np.asarray(
            [state.update(contrasts.feature_difference).score for state in sparse]
        )
        expected_spectral = np.asarray(
            [state.update(contrasts.density_difference).score for state in spectral]
        )
        np.testing.assert_allclose(actual.m4_scores, expected_sparse)
        np.testing.assert_allclose(actual.m5_scores, expected_spectral)

    state = bank.export_numeric_state()
    assert state["mutable.m4.ewma"].shape == (1, 4, FEATURE_DIM)
    assert state["mutable.m4.witnesses"].shape == (1, 4, 4, FEATURE_DIM)
    assert state["mutable.m5.ewma"].shape == (
        1,
        3,
        CHECK_COUNT,
        CHECK_COUNT,
    )
    assert state["mutable.m5.effects"].shape == (
        1,
        3,
        2,
        CHECK_COUNT,
        CHECK_COUNT,
    )


def test_role_bank_clone_digest_is_identical_then_diverges() -> None:
    bank = RoleIsolatedQECBank(role_count=2)
    zeros = np.zeros(CHECK_COUNT, dtype=np.uint8)
    ones = np.ones(CHECK_COUNT, dtype=np.uint8)
    bank.update(0, zeros, ones)
    clone = bank.clone()
    assert clone is not bank
    assert clone.state_digest() == bank.state_digest()
    for name, values in bank.export_numeric_state().items():
        np.testing.assert_array_equal(values, clone.export_numeric_state()[name])

    clone.update(1, ones, zeros)
    assert clone.state_digest() != bank.state_digest()
    assert bank.role_state_times(1).m3 == (0, 0, 0)
    assert clone.role_state_times(1).m3 == (1, 1, 1)


def test_within_shot_page_cusum_matches_hand_recursion_and_resets() -> None:
    reference = np.zeros((ROLE_COUNT, CHECK_COUNT), dtype=np.uint8)
    monitor = np.zeros_like(reference)
    monitor[0, 0] = 1

    first = paired_page_cusum_shot(reference, monitor)
    second = paired_page_cusum_shot(reference, monitor)
    first_scores = np.asarray(first.cycle_scores, dtype=np.float64)
    second_scores = np.asarray(second.cycle_scores, dtype=np.float64)
    assert first_scores.shape == (ROLE_COUNT,)
    assert np.asarray(first.positive).shape == (3, CHECK_COUNT + 1)
    assert np.asarray(first.negative).shape == (3, CHECK_COUNT + 1)
    assert np.asarray(first.positive_history).shape == (
        ROLE_COUNT,
        3,
        CHECK_COUNT + 1,
    )
    assert np.asarray(first.negative_history).shape == (
        ROLE_COUNT,
        3,
        CHECK_COUNT + 1,
    )
    # The kappa=.01 positive check channel starts at .99 and loses .01 on
    # every subsequent zero-difference role.
    np.testing.assert_allclose(
        first_scores[:5],
        [0.99, 0.98, 0.97, 0.96, 0.95],
    )
    np.testing.assert_array_equal(first_scores, second_scores)
    np.testing.assert_array_equal(first.positive, second.positive)
    np.testing.assert_array_equal(first.negative, second.negative)
    assert np.all(first_scores >= 0.0)


def test_strict_shot_threshold_uses_ties_and_at_most_one_alert_per_shot() -> None:
    scores = np.asarray(
        [
            [5.0, 0.0, 0.0],
            [4.0, 4.0, 0.0],
            [4.0, 1.0, 0.0],
        ]
    )
    selection = select_strict_shot_threshold(scores, max_alerts=1)
    assert selection.threshold == 4.0
    assert selection.alert_count == 1
    assert np.asarray(selection.candidates).size >= np.unique(scores).size
    applied = apply_strict_shot_threshold(scores, selection.threshold)
    assert np.asarray(applied.shot_alerts).shape == (3,)
    np.testing.assert_array_equal(
        applied.shot_alerts,
        [True, False, False],
    )
    np.testing.assert_array_equal(applied.first_crossing_roles, [0, -1, -1])
    assert applied.alert_count == 1
    assert len(applied.notifications) == 1

    tied = np.full((2, 4), 3.0)
    tied_selection = select_strict_shot_threshold(tied, max_alerts=1)
    assert tied_selection.threshold == 3.0
    assert tied_selection.alert_count == 0
    assert not np.any(tied_selection.shot_alerts)


def test_fast_threshold_selection_matches_brute_force() -> None:
    rng = np.random.default_rng(610607)
    for shot_count in range(1, 9):
        scores = rng.integers(0, 6, size=(shot_count, 5)).astype(np.float64)
        candidates = np.concatenate(([-np.inf], np.unique(scores), [np.inf]))
        for budget in range(shot_count + 1):
            expected = next(
                float(candidate)
                for candidate in candidates
                if np.count_nonzero(np.max(scores, axis=1) > candidate) <= budget
            )
            selected = select_strict_shot_threshold(scores, budget)
            assert selected.threshold == expected
            assert selected.alert_count <= budget


def test_paired_resource_counts_match_closed_form() -> None:
    counts = paired_resource_counts(num_pairs=7, num_roles=3, num_checks=2)
    assert counts.paired_shots == 7
    assert counts.physical_shots == 14
    assert counts.paired_role_updates == 21
    assert counts.detector_bits_exposed == 84
