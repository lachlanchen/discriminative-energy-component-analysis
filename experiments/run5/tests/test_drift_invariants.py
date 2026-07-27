from __future__ import annotations

from itertools import product

import numpy as np
from aoc.surface_code import (
    PeriodicSurfaceSyndromeModel,
    PeriodicSyndromeModel,
    logical_loop_access_no_go,
    periodic_boundary_syndrome,
    toggle_closed_logical_loop,
)


def exact_binary_distribution(model: PeriodicSyndromeModel):
    syndromes = np.asarray(list(product((0, 1), repeat=model.size)), dtype=np.uint8)
    probabilities = model.emission_likelihoods(syndromes)
    return syndromes, probabilities


def all_surface_observations(model: PeriodicSurfaceSyndromeModel) -> np.ndarray:
    return np.asarray(
        list(product((0, 1), repeat=model.num_detectors)),
        dtype=np.uint8,
    )


def test_surface_mixture_preserves_full_count_law_and_detector_marginals() -> None:
    model = PeriodicSurfaceSyndromeModel(
        size=3,
        event_probability=0.38,
        readout_error=0.07,
        allow_small_for_test=True,
    )
    observations = all_surface_observations(model)
    analytic_count = model.count_pmf
    analytic_marginal = model.detector_marginal
    assert np.isclose(analytic_count.sum(), 1.0, atol=2e-15)

    enumerated_count_laws = []
    enumerated_marginals = []
    for q in (0.0, 0.29, 1.0):
        probabilities = model.emission_likelihoods(observations, q)
        assert np.isclose(probabilities.sum(), 1.0, atol=3e-15)
        count_law = np.bincount(
            observations.sum(axis=1),
            weights=probabilities,
            minlength=model.num_detectors + 1,
        )
        marginal = probabilities @ observations
        enumerated_count_laws.append(count_law)
        enumerated_marginals.append(marginal)
        np.testing.assert_allclose(count_law, analytic_count, atol=2e-15)
        np.testing.assert_allclose(marginal, analytic_marginal, atol=2e-15)
    for count_law in enumerated_count_laws[1:]:
        np.testing.assert_allclose(count_law, enumerated_count_laws[0], atol=2e-15)
    for marginal in enumerated_marginals[1:]:
        np.testing.assert_allclose(marginal, enumerated_marginals[0], atol=2e-15)


def test_surface_mixture_changes_full_distribution_and_fourier_spectrum() -> None:
    model = PeriodicSurfaceSyndromeModel(
        size=5,
        event_probability=0.44,
        readout_error=0.06,
    )
    length_one = model.template_table(1)[0]
    assert not np.isclose(
        model.emission_likelihood(length_one, q=0.0),
        model.emission_likelihood(length_one, q=1.0),
    )
    spectrum_one = model.expected_fourier_spectrum(q=0.0)
    spectrum_two = model.expected_fourier_spectrum(q=1.0)
    assert not np.allclose(spectrum_one, spectrum_two)
    np.testing.assert_allclose(spectrum_one.sum(), 1.0, atol=2e-15)
    np.testing.assert_allclose(spectrum_two.sum(), 1.0, atol=2e-15)

    batch = model.sample_spatial(128, q=0.63, seed=20261)
    features = model.fourier_power_features(batch)
    assert features.shape == (128, model.num_detectors)
    np.testing.assert_allclose(features.sum(axis=1), 1.0, atol=2e-15)
    translated = np.roll(
        np.roll(length_one.reshape(5, 5), 2, axis=0),
        3,
        axis=1,
    )
    np.testing.assert_allclose(
        model.fourier_power_features(length_one),
        model.fourier_power_features(translated),
        atol=2e-15,
    )


def test_surface_exact_conditional_emissions_and_expected_spectrum_normalize() -> None:
    model = PeriodicSurfaceSyndromeModel(
        size=3,
        event_probability=0.41,
        readout_error=0.12,
        allow_small_for_test=True,
    )
    observations = all_surface_observations(model)
    for length in (1, 2):
        conditional = model.conditional_length_emission_likelihoods(
            observations,
            length,
        )
        assert np.isclose(conditional.sum(), 1.0, atol=3e-15)
    for q in (0.0, 0.37, 1.0):
        probabilities = model.emission_likelihoods(observations, q)
        exact_spectrum = probabilities @ model.fourier_power_features(observations)
        np.testing.assert_allclose(
            exact_spectrum,
            model.expected_fourier_spectrum(q),
            atol=3e-15,
        )

    first = np.repeat(observations, len(observations), axis=0)
    second = np.tile(observations, (len(observations), 1))
    pair_mass = model.nonoverlapping_pair_likelihoods(
        first,
        second,
        q=0.37,
        kappa=0.72,
    ).sum()
    assert np.isclose(pair_mass, 1.0, atol=3e-15)


def test_surface_markov_one_cycle_is_kappa_invariant_but_pairs_are_not() -> None:
    model = PeriodicSurfaceSyndromeModel(
        size=5,
        event_probability=0.48,
        readout_error=0.025,
    )
    q = 0.43
    stationary = np.asarray([1.0 - q, q])
    for kappa in (0.0, 0.4, 0.91, 1.0):
        transition = model.length_transition_matrix(q, kappa)
        np.testing.assert_allclose(transition.sum(axis=1), 1.0, atol=2e-16)
        np.testing.assert_allclose(stationary @ transition, stationary, atol=2e-16)

    observation = model.template_table(1)[0]
    one_cycle = model.emission_likelihood(observation, q)
    assert one_cycle > 0.0
    independent_pair = model.nonoverlapping_pair_likelihoods(
        observation,
        observation,
        q=q,
        kappa=0.0,
    )
    persistent_pair = model.nonoverlapping_pair_likelihoods(
        observation,
        observation,
        q=q,
        kappa=0.91,
    )
    np.testing.assert_allclose(independent_pair, one_cycle**2, rtol=1e-13)
    assert persistent_pair > independent_pair

    first, first_lengths = model.sample_temporal(
        80,
        streams=4,
        q=q,
        kappa=0.86,
        seed=20991,
        return_lengths=True,
    )
    second, second_lengths = model.sample_temporal(
        80,
        streams=4,
        q=q,
        kappa=0.86,
        seed=20991,
        return_lengths=True,
    )
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first_lengths, second_lengths)


def test_surface_translation_statistics_have_exact_means_and_pair_lr() -> None:
    model = PeriodicSurfaceSyndromeModel(
        size=5,
        event_probability=0.65,
        readout_error=0.03,
    )
    q = 0.35
    samples = model.sample_spatial(50_000, q, seed=713)
    empirical = model.translation_pair_features(samples).mean(axis=0)
    np.testing.assert_allclose(
        empirical,
        model.expected_translation_pair_features(q),
        atol=4e-3,
    )

    first = samples[:1000]
    second = samples[1000:2000]
    kappa = 0.75
    score_first = model.posterior_standardized_length_score(first, q)
    score_second = model.posterior_standardized_length_score(second, q)
    closed_form = 1.0 + kappa * score_first * score_second
    direct = model.nonoverlapping_pair_likelihoods(
        first,
        second,
        q=q,
        kappa=kappa,
    ) / (
        model.emission_likelihoods(first, q)
        * model.emission_likelihoods(second, q)
    )
    np.testing.assert_allclose(closed_form, direct, atol=2e-12)


def test_surface_translation_gap_has_inverse_detector_count_scaling() -> None:
    scaled_gaps = []
    for size in (5, 7, 9):
        model = PeriodicSurfaceSyndromeModel(
            size=size,
            event_probability=0.65,
            readout_error=0.03,
        )
        gap = np.abs(
            model.expected_translation_pair_features(0.55)
            - model.expected_translation_pair_features(0.35)
        ).sum()
        scaled_gaps.append(gap * model.num_detectors)
    np.testing.assert_allclose(scaled_gaps, scaled_gaps[0], atol=2e-14)


def test_chain_length_drift_preserves_full_count_pmf_and_all_marginals() -> None:
    adjacent = PeriodicSyndromeModel(
        size=9,
        event_probability=0.37,
        chain_length=1,
    )
    separated = PeriodicSyndromeModel(
        size=9,
        event_probability=0.37,
        chain_length=3,
    )
    np.testing.assert_allclose(adjacent.count_pmf, separated.count_pmf, atol=0.0)
    np.testing.assert_allclose(
        adjacent.detector_marginal,
        separated.detector_marginal,
        atol=1e-16,
    )
    expected_count_pmf = np.zeros(10)
    expected_count_pmf[0] = 0.63
    expected_count_pmf[2] = 0.37
    np.testing.assert_allclose(adjacent.count_pmf, expected_count_pmf)
    np.testing.assert_allclose(adjacent.detector_marginal, 2 * 0.37 / 9)

    syndromes, adjacent_probability = exact_binary_distribution(adjacent)
    _, separated_probability = exact_binary_distribution(separated)
    adjacent_count = np.bincount(
        syndromes.sum(axis=1),
        weights=adjacent_probability,
        minlength=adjacent.size + 1,
    )
    separated_count = np.bincount(
        syndromes.sum(axis=1),
        weights=separated_probability,
        minlength=separated.size + 1,
    )
    adjacent_marginal = adjacent_probability @ syndromes
    separated_marginal = separated_probability @ syndromes
    np.testing.assert_allclose(adjacent_count, separated_count, atol=1e-16)
    np.testing.assert_allclose(adjacent_marginal, separated_marginal, atol=1e-16)

    # The invariant Fourier-power observer still distinguishes the topology
    # of the open chain while remaining unchanged under cyclic translations.
    assert not np.allclose(
        adjacent.null_fourier_spectrum,
        separated.null_fourier_spectrum,
    )
    example = separated.syndrome_from_start(2)
    shifted = np.roll(example, 4)
    np.testing.assert_allclose(
        separated.fourier_power_features(example),
        separated.fourier_power_features(shifted),
        atol=1e-15,
    )
    assert np.isclose(separated.null_fourier_spectrum.sum(), 2 * 0.37)


def test_exact_emission_likelihoods_normalize_for_small_rings() -> None:
    for size in range(3, 7):
        for chain_length in range(1, size):
            model = PeriodicSyndromeModel(
                size=size,
                event_probability=0.41,
                chain_length=chain_length,
            )
            _, probabilities = exact_binary_distribution(model)
            assert np.all(probabilities >= 0.0)
            assert np.isclose(probabilities.sum(), 1.0, atol=2e-15)

    # At half circumference, two oriented starts map to the same unordered
    # endpoint pair.  Exact likelihoods must retain that multiplicity.
    diametric = PeriodicSyndromeModel(
        size=6,
        event_probability=0.42,
        chain_length=3,
    )
    syndrome = diametric.syndrome_from_start(0)
    assert np.isclose(diametric.emission_probability(syndrome), 2 * 0.42 / 6)
    assert np.isclose(
        diametric.emission_probability(np.zeros(6, dtype=np.uint8)),
        0.58,
    )

    temporal = PeriodicSyndromeModel(
        size=3,
        event_probability=0.39,
        chain_length=1,
        persistence=0.72,
    )
    all_syndromes = np.asarray(list(product((0, 1), repeat=3)), dtype=np.uint8)
    sequence_mass = sum(
        temporal.temporal_likelihood(np.stack([first, second]))
        for first in all_syndromes
        for second in all_syndromes
    )
    assert np.isclose(sequence_mass, 1.0, atol=2e-15)


def test_refresh_repeat_markov_model_has_identical_one_cycle_law() -> None:
    model = PeriodicSyndromeModel(
        size=7,
        event_probability=0.31,
        chain_length=2,
        persistence=0.93,
    )
    stationary = model.stationary_latent_distribution
    for persistence in (0.0, 0.35, 0.93, 1.0):
        transition = model.transition_matrix(persistence)
        np.testing.assert_allclose(transition.sum(axis=1), 1.0, atol=2e-16)
        np.testing.assert_allclose(stationary @ transition, stationary, atol=2e-16)
        for latent_syndrome in model.emission_table:
            one_cycle = model.temporal_likelihood(
                latent_syndrome[None, :],
                persistence=persistence,
            )
            assert np.isclose(
                one_cycle,
                model.emission_probability(latent_syndrome),
            )

    # Persistence changes a genuine two-cycle probability despite preserving
    # every one-cycle likelihood.
    event = model.syndrome_from_start(0)
    repeated = np.stack([event, event])
    independent = model.temporal_likelihood(repeated, persistence=0.0)
    persistent = model.temporal_likelihood(repeated, persistence=0.93)
    assert persistent > independent


def test_closed_logical_loop_is_pathwise_invisible_to_syndromes() -> None:
    rng = np.random.default_rng(20260727)
    paths = rng.integers(0, 2, size=(64, 11), dtype=np.uint8)
    loop_shifted = toggle_closed_logical_loop(paths)
    assert np.all(np.bitwise_xor(paths, loop_shifted) == 1)
    np.testing.assert_array_equal(
        periodic_boundary_syndrome(paths),
        periodic_boundary_syndrome(loop_shifted),
    )
    certificate = logical_loop_access_no_go(paths)
    assert certificate.paths_checked == 64
    assert certificate.ring_size == 11
    assert certificate.pathwise_syndrome_equal
    assert certificate.maximum_syndrome_difference == 0
    assert certificate.syndrome_total_variation == 0.0
    assert certificate.optimal_equal_prior_syndrome_success == 0.5


def test_seeded_spatial_and_temporal_sampling_is_deterministic() -> None:
    model = PeriodicSyndromeModel(
        size=8,
        event_probability=0.46,
        chain_length=3,
        persistence=0.81,
    )
    first_spatial, first_latents = model.sample_spatial(
        200,
        seed=1701,
        return_latents=True,
    )
    second_spatial, second_latents = model.sample_spatial(
        200,
        seed=1701,
        return_latents=True,
    )
    np.testing.assert_array_equal(first_spatial, second_spatial)
    np.testing.assert_array_equal(first_latents, second_latents)

    first_temporal, first_temporal_latents = model.sample_temporal(
        150,
        streams=5,
        seed=1907,
        return_latents=True,
    )
    second_temporal, second_temporal_latents = model.sample_temporal(
        150,
        streams=5,
        seed=1907,
        return_latents=True,
    )
    assert first_temporal.shape == (5, 150, 8)
    np.testing.assert_array_equal(first_temporal, second_temporal)
    np.testing.assert_array_equal(first_temporal_latents, second_temporal_latents)
    assert not np.array_equal(
        first_temporal,
        model.sample_temporal(150, streams=5, seed=1908),
    )
