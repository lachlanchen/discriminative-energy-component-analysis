from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
from aoc.surface_code import PeriodicSurfaceSyndromeModel

SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "run_offline_diagnostic_audit.py"
)
SPEC = importlib.util.spec_from_file_location("run5_offline_diagnostic", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def small_model() -> PeriodicSurfaceSyndromeModel:
    return PeriodicSurfaceSyndromeModel(
        size=3,
        event_probability=0.55,
        readout_error=0.04,
        allow_small_for_test=True,
    )


def test_pair_lift_mean_has_declared_outer_product() -> None:
    features = np.asarray(
        [
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
                [7.0, 8.0],
            ]
        ]
    )
    expected = 0.5 * (
        np.outer(features[0, 0], features[0, 1])
        + np.outer(features[0, 2], features[0, 3])
    )
    assert np.allclose(MODULE.pair_lift_mean(features)[0], expected.reshape(-1))


def test_full_hmm_batch_score_matches_exact_single_stream_method() -> None:
    model = small_model()
    observed = model.sample_temporal(
        6,
        q=0.37,
        kappa=0.64,
        streams=4,
        seed=1981,
    )
    batch = MODULE.temporal_full_hmm_llr(
        model,
        observed,
        q=0.37,
        kappa=0.64,
        batch_windows=2,
    )
    direct = np.asarray(
        [
            model.temporal_hmm_log_likelihood_ratio_increments(
                stream,
                q=0.37,
                kappa=0.64,
            ).sum()
            for stream in observed
        ]
    )
    assert np.allclose(batch, direct, atol=1e-12)


def test_feature_families_have_expected_spatial_and_temporal_shapes() -> None:
    model = small_model()
    orbits = MODULE.d4_frequency_orbits(model.size)
    spatial = model.sample_spatial(20, q=0.51, seed=1982).reshape(5, 4, -1)
    temporal = model.sample_temporal(
        4,
        q=0.37,
        kappa=0.6,
        streams=5,
        seed=1983,
    )
    spatial_features, _ = MODULE.extract_window_features(
        model,
        spatial,
        scenario="spatial",
        orbits=orbits,
    )
    temporal_features, _ = MODULE.extract_window_features(
        model,
        temporal,
        scenario="temporal",
        orbits=orbits,
    )
    assert spatial_features["dfr_count_sequence"].shape == (5, 4)
    assert spatial_features["detector_first_moment"].shape == (5, 9)
    assert spatial_features["translation"].shape == (5, 3)
    assert spatial_features["symmetry_fourier"].shape == (5, len(orbits))
    assert temporal_features["translation"].shape == (5, 9)
    assert temporal_features["symmetry_fourier"].shape == (
        5,
        len(orbits) ** 2,
    )


def test_simplex_projector_is_indicator_not_magnitude_weighted() -> None:
    simplex_features = np.asarray(
        [
            [0.70, 0.20, 0.10],
            [0.60, 0.25, 0.15],
            [0.40, 0.35, 0.25],
            [0.30, 0.40, 0.30],
        ]
    )
    labels = np.asarray([0, 0, 1, 1], dtype=np.uint8)
    projector = MODULE.positive_support_projector(simplex_features, labels)
    assert np.array_equal(projector, np.asarray([0.0, 1.0, 1.0]))


def test_non_simplex_positive_part_direction_has_no_projector_semantics() -> None:
    features = np.asarray(
        [
            [0.0, 2.0],
            [0.1, 2.2],
            [2.0, 3.0],
            [1.2, 2.9],
        ]
    )
    labels = np.asarray([0, 0, 1, 1], dtype=np.uint8)
    direction = MODULE.positive_part_mean_direction(features, labels)
    hotelling, ridge = MODULE.regularized_hotelling_direction(
        features,
        labels,
        ridge_fraction=0.05,
        absolute_floor=1e-8,
    )
    threshold, balanced = MODULE.select_validation_threshold(
        labels,
        features @ direction,
    )
    assert np.all(direction >= 0.0)
    assert np.isclose(direction.sum(), 1.0)
    assert not np.all(np.isin(direction, (0.0, 1.0)))
    assert ridge > 0.0
    assert np.isclose(np.linalg.norm(hotelling), 1.0)
    assert np.isfinite(threshold)
    assert balanced >= 0.5


def test_fixed_prototype_hamming_mmd_points_toward_alternative() -> None:
    null = np.zeros((12, 2, 5), dtype=np.uint8)
    alternative = np.ones((12, 2, 5), dtype=np.uint8)
    witness = MODULE.fit_hamming_mmd_witness(
        null,
        alternative,
        prototypes_per_class=6,
        query_batch_size=4,
        seed=1984,
    )
    scores = MODULE.score_hamming_mmd(
        witness,
        np.concatenate([null[:3], alternative[:3]]),
    )
    assert np.all(scores[:3] < 0.0)
    assert np.all(scores[3:] > 0.0)
