#!/usr/bin/env python3
"""Equal-budget sequential drift tests on the exact periodic syndrome model."""

from __future__ import annotations

import argparse
import json
import time
import tracemalloc
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from aoc.change_detection import (
    BoundedScoreSR,
    HiddenMarkovBlockSR,
    LikelihoodRatioSR,
    PredictableSimplexWitness,
    StaticSimplexWitness,
    effect_from_direction,
    paired_bootstrap_mean_difference,
)
from aoc.repro import write_manifest
from aoc.surface_code import PeriodicSurfaceSyndromeModel
from sklearn.linear_model import LogisticRegression

RUN_ROOT = Path(__file__).resolve().parents[1]
VALIDATION_BASELINE_METHOD = "validation-trained linear logistic effect"
PROPOSED_METHOD = "vAOC / same-feature Hotelling"


@dataclass(frozen=True)
class DetectorInputs:
    """Features and likelihood ratios for one sequential stream."""

    features: np.ndarray
    matched_effect: np.ndarray
    likelihood_ratios: np.ndarray
    oracle_likelihood_ratio: np.ndarray
    likelihood_method: str
    oracle_method: str
    hmm_first_emission_ratios: np.ndarray | None = None
    hmm_second_emission_ratios: np.ndarray | None = None
    hmm_transitions: np.ndarray | None = None
    hmm_oracle_transition: np.ndarray | None = None
    hmm_stationary: np.ndarray | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Committed pilot or paper JSON configuration.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    """Load and enforce the cycle-fair locked/pilot protocol."""

    config = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "name",
        "locked",
        "publication_grade",
        "design_provenance",
        "units",
        "claim_boundary",
        "model",
        "spatial_alternatives",
        "temporal_persistence",
        "calibration_cycles",
        "witness_window_updates",
        "witness_update_interval_updates",
        "ridge_grid",
        "bet_fractions",
        "target_arl_cycles",
        "detector_threshold_updates",
        "change_time_cycles",
        "post_change_horizon_cycles",
        "null_horizon_cycles",
        "validation_streams",
        "ridge_validation_horizon_cycles",
        "null_test_streams",
        "changed_test_streams",
        "validation_trained_baseline",
        "primary_comparison_hypotheses",
        "bootstrap_resamples",
        "seeds",
        "seed_partition_note",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Configuration is missing keys: {missing}")
    if config["publication_grade"] and config["locked"] is not True:
        raise ValueError("A publication-grade sequential design must be locked.")
    if (
        not isinstance(config["design_provenance"], str)
        or not config["design_provenance"].strip()
    ):
        raise ValueError("design_provenance must be a nonempty string.")
    if not isinstance(config["claim_boundary"], list) or not config["claim_boundary"]:
        raise ValueError("claim_boundary must be a nonempty list.")
    if (
        not isinstance(config["spatial_alternatives"], list)
        or not config["spatial_alternatives"]
        or not all(0.0 < float(value) < 1.0 for value in config["spatial_alternatives"])
    ):
        raise ValueError("spatial_alternatives must be probabilities in (0,1).")
    if (
        not isinstance(config["temporal_persistence"], list)
        or not config["temporal_persistence"]
        or not all(
            0.0 <= float(value) < 1.0 for value in config["temporal_persistence"]
        )
    ):
        raise ValueError("temporal_persistence must lie in [0,1).")

    expected_unit_keys = {
        "calibration_and_horizons",
        "spatial_detector_update",
        "temporal_detector_update",
        "target_arl",
        "detector_thresholds",
        "witness_window_and_update_interval",
    }
    if set(config["units"]) != expected_unit_keys:
        raise ValueError("The configuration must declare every protocol unit.")

    cycle_fields = (
        "calibration_cycles",
        "change_time_cycles",
        "post_change_horizon_cycles",
        "null_horizon_cycles",
        "ridge_validation_horizon_cycles",
    )
    for field in cycle_fields:
        value = config[field]
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"{field} must be a positive integer number of cycles.")
        if value % 2:
            raise ValueError(
                f"{field} must be even so the temporal arm uses complete pairs."
            )
    for field in (
        "witness_window_updates",
        "witness_update_interval_updates",
        "validation_streams",
        "null_test_streams",
        "changed_test_streams",
        "bootstrap_resamples",
    ):
        if not isinstance(config[field], int) or config[field] <= 0:
            raise ValueError(f"{field} must be a positive integer.")

    target_cycles = int(config["target_arl_cycles"])
    if target_cycles <= 0 or target_cycles % 2:
        raise ValueError("target_arl_cycles must be a positive even integer.")
    thresholds = config["detector_threshold_updates"]
    if set(thresholds) != {"spatial", "temporal"}:
        raise ValueError("Thresholds must be declared for both families.")
    if int(thresholds["spatial"]) != target_cycles:
        raise ValueError("The spatial threshold must equal target ARL cycles.")
    if 2 * int(thresholds["temporal"]) != target_cycles:
        raise ValueError(
            "The temporal pair-update threshold must be half target ARL cycles."
        )

    baseline = config["validation_trained_baseline"]
    expected_baseline_keys = {
        "method",
        "training_streams_per_class",
        "cycles_per_stream",
        "total_physical_cycles_per_class",
        "spatial_training_examples_per_class",
        "temporal_pair_training_examples_per_class",
        "logistic_c",
        "logistic_max_iter",
        "training_effects",
        "budget_note",
    }
    if set(baseline) != expected_baseline_keys:
        raise ValueError("Unexpected validation-trained baseline specification.")
    if baseline["method"] != VALIDATION_BASELINE_METHOD:
        raise ValueError("The locked baseline method name changed.")
    streams = int(baseline["training_streams_per_class"])
    cycles_per_stream = int(baseline["cycles_per_stream"])
    if streams <= 0 or cycles_per_stream <= 0 or cycles_per_stream % 2:
        raise ValueError("Baseline streams and even cycles/stream must be positive.")
    total_cycles = streams * cycles_per_stream
    if total_cycles != int(baseline["total_physical_cycles_per_class"]):
        raise ValueError("Baseline physical-cycle budget is inconsistent.")
    if int(baseline["spatial_training_examples_per_class"]) != total_cycles:
        raise ValueError("Spatial baseline example count is inconsistent.")
    if int(baseline["temporal_pair_training_examples_per_class"]) != total_cycles // 2:
        raise ValueError("Temporal baseline pair count is inconsistent.")
    if streams != int(config["validation_streams"]):
        raise ValueError(
            "Logistic and ridge validation must use the same number of "
            "independent streams."
        )
    if cycles_per_stream != int(config["ridge_validation_horizon_cycles"]):
        raise ValueError(
            "Logistic and ridge validation streams must have equal cycle length."
        )
    ridge_validation_cycles = int(config["validation_streams"]) * int(
        config["ridge_validation_horizon_cycles"]
    )
    if ridge_validation_cycles != int(config["calibration_cycles"]):
        raise ValueError(
            "Ridge validation must consume the same physical-cycle budget "
            "as null calibration."
        )
    if ridge_validation_cycles != total_cycles:
        raise ValueError(
            "Ridge validation and logistic training must consume the same "
            "physical-cycle budget per class."
        )
    middle_q = float(
        config["spatial_alternatives"][len(config["spatial_alternatives"]) // 2]
    )
    middle_kappa = float(
        config["temporal_persistence"][len(config["temporal_persistence"]) // 2]
    )
    if float(baseline["training_effects"]["spatial_q"]) != middle_q:
        raise ValueError("Baseline spatial training effect must be the middle effect.")
    if float(baseline["training_effects"]["temporal_kappa"]) != middle_kappa:
        raise ValueError("Baseline temporal training effect must be the middle effect.")

    hypotheses = config["primary_comparison_hypotheses"]
    if hypotheses["proposed_method"] != PROPOSED_METHOD:
        raise ValueError("The primary proposed method changed.")
    if hypotheses["baseline_method"] != VALIDATION_BASELINE_METHOD:
        raise ValueError("The primary baseline changed.")
    if float(hypotheses["spatial_effect"]) != middle_q:
        raise ValueError("The spatial primary hypothesis must use the middle effect.")
    if float(hypotheses["temporal_effect"]) != middle_kappa:
        raise ValueError("The temporal primary hypothesis must use the middle effect.")
    alpha = float(hypotheses["familywise_alpha"])
    if not 0.0 < alpha < 1.0:
        raise ValueError("familywise_alpha must lie in (0, 1).")

    if config["publication_grade"]:
        for field in ("scaling_sizes", "scaling_repetitions"):
            if field not in config:
                raise ValueError(f"Publication configuration is missing {field}.")
    assert_disjoint_sampling_seed_partitions(config)
    return config


def cycles_to_updates(cycles: int, family: str) -> int:
    """Convert a physical-cycle budget to detector updates without rounding."""

    value = int(cycles)
    if value <= 0:
        raise ValueError("cycles must be positive.")
    if family == "spatial":
        return value
    if family == "temporal":
        if value % 2:
            raise ValueError("Temporal physical-cycle budgets must be even.")
        return value // 2
    raise ValueError("family must be spatial or temporal.")


def family_time_unit_cycles(family: str) -> int:
    if family == "spatial":
        return 1
    if family == "temporal":
        return 2
    raise ValueError("family must be spatial or temporal.")


def sampling_seed_intervals(
    config: dict[str, Any],
) -> list[tuple[str, int, int]]:
    """Return every sampling-seed interval consumed by this script."""

    seeds = config["seeds"]
    intervals: list[tuple[str, int, int]] = []

    def point(name: str, value: int) -> None:
        intervals.append((name, int(value), int(value)))

    for family, value in seeds["calibration"].items():
        point(f"calibration:{family}", value)
    for name, value in seeds["baseline_training"].items():
        point(f"baseline_training:{name}", value)

    validation_count = int(config["validation_streams"])
    for family, start in seeds["ridge_validation"].items():
        first = int(start)
        intervals.append(
            (f"ridge_validation:{family}", first, first + validation_count - 1)
        )

    null_count = int(config["null_test_streams"])
    changed_count = int(config["changed_test_streams"])
    for family, effects in (
        ("spatial", config["spatial_alternatives"]),
        ("temporal", config["temporal_persistence"]),
    ):
        base = int(seeds["locked_test"][f"{family}_start"])
        for effect_index, _ in enumerate(effects):
            effect_base = base + effect_index * 1_000_000
            intervals.append(
                (
                    f"locked_test:{family}:effect{effect_index}:no_change",
                    effect_base,
                    effect_base + null_count - 1,
                )
            )
            changed_start = effect_base + 100_000
            intervals.append(
                (
                    f"locked_test:{family}:effect{effect_index}:changed",
                    changed_start,
                    changed_start + changed_count - 1,
                )
            )

    if config["publication_grade"]:
        scaling_count = int(config["scaling_repetitions"])
        for size in config["scaling_sizes"]:
            first = int(seeds["scaling_start"]) + int(size) * 100_000
            intervals.append(
                (
                    f"scaling:L{size}",
                    first,
                    first + scaling_count - 1,
                )
            )
            point(
                f"scaling_calibration:L{size}",
                int(seeds["scaling_start"]) + int(size) * 100_000 + 10_000,
            )
            point(
                f"timing:L{size}",
                int(seeds["timing_start"]) + int(size),
            )
    return intervals


def assert_disjoint_sampling_seed_partitions(config: dict[str, Any]) -> None:
    """Reject any overlap between validation, test, or other sampling seeds."""

    intervals = sorted(sampling_seed_intervals(config), key=lambda item: item[1])
    for (left_name, left_start, left_end), (
        right_name,
        right_start,
        right_end,
    ) in pairwise(intervals):
        if right_start <= left_end:
            raise ValueError(
                "Sampling seed partitions overlap: "
                f"{left_name}=[{left_start},{left_end}] and "
                f"{right_name}=[{right_start},{right_end}]."
            )


def d4_frequency_orbits(size: int) -> tuple[np.ndarray, ...]:
    """Partition 2-D Fourier modes into square-lattice D4 orbits."""

    unseen = {(row, column) for row in range(size) for column in range(size)}
    orbits: list[np.ndarray] = []
    while unseen:
        row, column = min(unseen)
        orbit = {
            (row % size, column % size),
            ((-row) % size, column % size),
            (row % size, (-column) % size),
            ((-row) % size, (-column) % size),
            (column % size, row % size),
            ((-column) % size, row % size),
            (column % size, (-row) % size),
            ((-column) % size, (-row) % size),
        }
        unseen -= orbit
        indices = np.asarray(
            sorted(first * size + second for first, second in orbit),
            dtype=np.int64,
        )
        orbits.append(indices)
    return tuple(orbits)


def aggregate_orbits(
    power: np.ndarray,
    orbits: tuple[np.ndarray, ...],
) -> np.ndarray:
    """Sum simplex Fourier power over disjoint symmetry orbits."""

    values = np.asarray(power, dtype=np.float64)
    return np.stack([values[..., indices].sum(axis=-1) for indices in orbits], axis=-1)


def expected_pair_feature(
    model: PeriodicSurfaceSyndromeModel,
    *,
    q: float,
    kappa: float,
    orbits: tuple[np.ndarray, ...],
) -> np.ndarray:
    """Exact expectation of an orbit-power outer product for two cycles."""

    conditional = np.stack(
        [
            aggregate_orbits(model.expected_fourier_spectrum(0.0), orbits),
            aggregate_orbits(model.expected_fourier_spectrum(1.0), orbits),
        ]
    )
    stationary = np.asarray([1.0 - q, q])
    transition = model.length_transition_matrix(q, kappa)
    expected = np.zeros((len(orbits), len(orbits)), dtype=np.float64)
    for previous in range(2):
        for current in range(2):
            expected += (
                stationary[previous]
                * transition[previous, current]
                * np.outer(conditional[previous], conditional[current])
            )
    return expected.reshape(-1)


def pair_features(features: np.ndarray) -> np.ndarray:
    """Lift nonoverlapping consecutive simplex features to pair states."""

    if len(features) % 2:
        raise ValueError("Pair lifting requires an even number of cycles.")
    first = features[0::2]
    second = features[1::2]
    lifted = first[:, :, None] * second[:, None, :]
    return lifted.reshape(len(first), -1)


def covariance_with_ridge_samples(features: np.ndarray) -> np.ndarray:
    covariance = np.cov(np.asarray(features, dtype=np.float64), rowvar=False)
    covariance = np.atleast_2d(covariance)
    return 0.5 * (covariance + covariance.T)


def fit_linear_logistic_effect(
    null_features: np.ndarray,
    alternative_features: np.ndarray,
    *,
    c_value: float,
    max_iter: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Fit a fixed linear logistic direction and map it to a simplex effect."""

    null = np.asarray(null_features, dtype=np.float64)
    alternative = np.asarray(alternative_features, dtype=np.float64)
    if (
        null.ndim != 2
        or alternative.ndim != 2
        or null.shape[1] != alternative.shape[1]
        or len(null) == 0
        or len(alternative) == 0
    ):
        raise ValueError("Baseline feature matrices must be compatible and nonempty.")
    if np.any(null < -1e-10) or np.any(alternative < -1e-10):
        raise ValueError("Baseline features must be nonnegative simplex vectors.")
    if not np.allclose(null.sum(axis=1), 1.0, atol=1e-9) or not np.allclose(
        alternative.sum(axis=1),
        1.0,
        atol=1e-9,
    ):
        raise ValueError("Baseline features must lie on the probability simplex.")
    if c_value <= 0.0 or max_iter <= 0:
        raise ValueError("Logistic hyperparameters must be positive.")

    features = np.concatenate([null, alternative], axis=0)
    labels = np.concatenate(
        [
            np.zeros(len(null), dtype=np.uint8),
            np.ones(len(alternative), dtype=np.uint8),
        ]
    )
    classifier = LogisticRegression(
        C=float(c_value),
        max_iter=int(max_iter),
        solver="lbfgs",
        fit_intercept=True,
    )
    classifier.fit(features, labels)
    coefficients = np.asarray(classifier.coef_[0], dtype=np.float64)
    effect = effect_from_direction(coefficients)
    accuracy = float(classifier.score(features, labels))
    return effect, coefficients, accuracy


def fit_validation_trained_baselines(
    model: PeriodicSurfaceSyndromeModel,
    *,
    config: dict[str, Any],
    q0: float,
    spatial_effect: float,
    temporal_effect: float,
    orbits: tuple[np.ndarray, ...],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Fit both locked model-agnostic effects on independent labeled streams."""

    specification = config["validation_trained_baseline"]
    seeds = config["seeds"]["baseline_training"]
    streams = int(specification["training_streams_per_class"])
    cycles_per_stream = int(specification["cycles_per_stream"])
    total_cycles = streams * cycles_per_stream

    spatial_null_observed = model.sample_spatial(
        total_cycles,
        q0,
        seed=int(seeds["spatial_null"]),
    )
    spatial_alternative_observed = model.sample_spatial(
        total_cycles,
        spatial_effect,
        seed=int(seeds["spatial_alternative"]),
    )
    spatial_null = aggregate_orbits(
        model.fourier_power_features(spatial_null_observed),
        orbits,
    )
    spatial_alternative = aggregate_orbits(
        model.fourier_power_features(spatial_alternative_observed),
        orbits,
    )

    temporal_null_observed = model.sample_temporal(
        cycles_per_stream,
        q=q0,
        kappa=0.0,
        streams=streams,
        seed=int(seeds["temporal_null"]),
    )
    temporal_alternative_observed = model.sample_temporal(
        cycles_per_stream,
        q=q0,
        kappa=temporal_effect,
        streams=streams,
        seed=int(seeds["temporal_alternative"]),
    )
    temporal_null_cycle = aggregate_orbits(
        model.fourier_power_features(
            temporal_null_observed.reshape(-1, model.num_detectors)
        ),
        orbits,
    )
    temporal_alternative_cycle = aggregate_orbits(
        model.fourier_power_features(
            temporal_alternative_observed.reshape(-1, model.num_detectors)
        ),
        orbits,
    )
    temporal_null = pair_features(temporal_null_cycle)
    temporal_alternative = pair_features(temporal_alternative_cycle)

    effects: dict[str, np.ndarray] = {}
    records: dict[str, Any] = {}
    for family, null, alternative in (
        ("spatial", spatial_null, spatial_alternative),
        ("temporal", temporal_null, temporal_alternative),
    ):
        effect, coefficients, training_accuracy = fit_linear_logistic_effect(
            null,
            alternative,
            c_value=float(specification["logistic_c"]),
            max_iter=int(specification["logistic_max_iter"]),
        )
        effects[family] = effect
        records[family] = {
            "method": VALIDATION_BASELINE_METHOD,
            "physical_cycles_per_class": total_cycles,
            "independent_streams_per_class": streams,
            "cycles_per_stream": cycles_per_stream,
            "training_examples_per_class": len(null),
            "feature_dimension": null.shape[1],
            "null_seed": int(seeds[f"{family}_null"]),
            "alternative_seed": int(seeds[f"{family}_alternative"]),
            "training_effect": (
                spatial_effect if family == "spatial" else temporal_effect
            ),
            "coefficient": coefficients.tolist(),
            "effect": effect.tolist(),
            "effect_min": float(effect.min()),
            "effect_max": float(effect.max()),
            "descriptive_training_accuracy": training_accuracy,
            "fit_note": (
                "Unstandardized fixed linear logistic coefficient, affinely "
                "mapped into [0,1] and centered at the analytic null mean. "
                "Training performance is descriptive and does not enter the "
                "locked decision rule."
            ),
        }
    metadata = {
        "method": VALIDATION_BASELINE_METHOD,
        "status": "frozen before locked test",
        "budget_unit": "physical syndrome cycles",
        "cross_family_sample_efficiency_claim": False,
        "specification": specification,
        "fits": records,
    }
    return effects, metadata


def detector_alarm_times(
    inputs: DetectorInputs,
    *,
    reference: np.ndarray,
    covariance: np.ndarray,
    ridge: float,
    validation_effect: np.ndarray | None,
    window: int,
    update_interval: int,
    threshold: float,
    bet_fractions: np.ndarray,
) -> dict[str, int | None]:
    """Run all equal-threshold detectors on one feature stream."""

    raw_witness = PredictableSimplexWitness(
        reference,
        mode="raw",
        window=window,
        update_interval=update_interval,
    )
    variance_witness = PredictableSimplexWitness(
        reference,
        mode="variance",
        null_covariance=covariance,
        ridge=ridge,
        window=window,
        update_interval=update_interval,
    )
    matched_witness = StaticSimplexWitness(reference, inputs.matched_effect)
    validation_witness = (
        None
        if validation_effect is None
        else StaticSimplexWitness(reference, validation_effect)
    )
    raw_detector = BoundedScoreSR(
        threshold=threshold,
        bet_fractions=bet_fractions,
    )
    variance_detector = BoundedScoreSR(
        threshold=threshold,
        bet_fractions=bet_fractions,
    )
    matched_detector = BoundedScoreSR(
        threshold=threshold,
        bet_fractions=bet_fractions,
    )
    validation_detector = (
        None
        if validation_witness is None
        else BoundedScoreSR(
            threshold=threshold,
            bet_fractions=bet_fractions,
        )
    )
    blind_detector = BoundedScoreSR(
        threshold=threshold,
        bet_fractions=bet_fractions,
    )
    grid_detector = LikelihoodRatioSR(
        inputs.likelihood_ratios.shape[1],
        threshold=threshold,
    )
    oracle_detector = LikelihoodRatioSR(1, threshold=threshold)
    hmm_grid_detector = None
    hmm_oracle_detector = None
    if inputs.hmm_transitions is not None:
        if inputs.hmm_stationary is None:
            raise ValueError("HMM transitions require a stationary distribution.")
        hmm_grid_detector = HiddenMarkovBlockSR(
            inputs.hmm_transitions,
            inputs.hmm_stationary,
            threshold=threshold,
        )
        if inputs.hmm_oracle_transition is None:
            raise ValueError("HMM transitions require an oracle transition.")
        hmm_oracle_detector = HiddenMarkovBlockSR(
            inputs.hmm_oracle_transition[None, :, :],
            inputs.hmm_stationary,
            threshold=threshold,
        )
    alarm_times: dict[str, int | None] = {
        "DFR/count exact pushforward": None,
        "raw symmetry-resolved AOC": None,
        "vAOC / same-feature Hotelling": None,
        "matched correlation witness": None,
        inputs.likelihood_method: None,
        inputs.oracle_method: None,
    }
    if validation_witness is not None:
        alarm_times[VALIDATION_BASELINE_METHOD] = None
    if hmm_grid_detector is not None:
        alarm_times["full-HMM block-boundary grid SR"] = None
    if hmm_oracle_detector is not None:
        alarm_times["known-post full-HMM block-boundary SR"] = None
    for index, feature in enumerate(inputs.features):
        raw_score = raw_witness.update(feature).score
        variance_score = variance_witness.update(feature).score
        matched_score = matched_witness.score(feature)
        updates = {
            "DFR/count exact pushforward": blind_detector.update(0.0),
            "raw symmetry-resolved AOC": raw_detector.update(raw_score),
            "vAOC / same-feature Hotelling": variance_detector.update(variance_score),
            "matched correlation witness": matched_detector.update(matched_score),
            inputs.likelihood_method: grid_detector.update(
                inputs.likelihood_ratios[index]
            ),
            inputs.oracle_method: oracle_detector.update(
                [inputs.oracle_likelihood_ratio[index]]
            ),
        }
        if validation_witness is not None and validation_detector is not None:
            updates[VALIDATION_BASELINE_METHOD] = validation_detector.update(
                validation_witness.score(feature)
            )
        if hmm_grid_detector is not None:
            assert inputs.hmm_first_emission_ratios is not None
            assert inputs.hmm_second_emission_ratios is not None
            updates["full-HMM block-boundary grid SR"] = hmm_grid_detector.update(
                inputs.hmm_first_emission_ratios[index],
                inputs.hmm_second_emission_ratios[index],
            )
        if hmm_oracle_detector is not None:
            assert inputs.hmm_first_emission_ratios is not None
            assert inputs.hmm_second_emission_ratios is not None
            updates["known-post full-HMM block-boundary SR"] = (
                hmm_oracle_detector.update(
                    inputs.hmm_first_emission_ratios[index],
                    inputs.hmm_second_emission_ratios[index],
                )
            )
        for method, update in updates.items():
            if alarm_times[method] is None and update.alarm_time is not None:
                alarm_times[method] = update.alarm_time
        if all(value is not None for value in alarm_times.values()):
            break
    return alarm_times


def spatial_inputs(
    model: PeriodicSurfaceSyndromeModel,
    observed: np.ndarray,
    *,
    q0: float,
    q1: float,
    q_grid: np.ndarray,
    orbits: tuple[np.ndarray, ...],
) -> DetectorInputs:
    power = aggregate_orbits(model.fourier_power_features(observed), orbits)
    reference = aggregate_orbits(model.expected_fourier_spectrum(q0), orbits)
    alternative = aggregate_orbits(model.expected_fourier_spectrum(q1), orbits)
    matched_effect = effect_from_direction(alternative - reference)
    null_log = model.emission_log_likelihoods(observed, q0)
    grid_log = np.stack(
        [model.emission_log_likelihoods(observed, value) for value in q_grid],
        axis=1,
    )
    ratios = np.exp(np.clip(grid_log - null_log[:, None], -700.0, 700.0))
    oracle = np.exp(
        np.clip(
            model.emission_log_likelihoods(observed, q1) - null_log,
            -700.0,
            700.0,
        )
    )
    return DetectorInputs(
        features=power,
        matched_effect=matched_effect,
        likelihood_ratios=ratios,
        oracle_likelihood_ratio=oracle,
        likelihood_method="exact one-cycle likelihood grid SR",
        oracle_method="known-post one-cycle likelihood SR",
    )


def temporal_inputs(
    model: PeriodicSurfaceSyndromeModel,
    observed: np.ndarray,
    *,
    q0: float,
    kappa1: float,
    kappa_grid: np.ndarray,
    orbits: tuple[np.ndarray, ...],
) -> DetectorInputs:
    cycle_power = aggregate_orbits(model.fourier_power_features(observed), orbits)
    lifted = pair_features(cycle_power)
    reference_cycle = aggregate_orbits(model.expected_fourier_spectrum(q0), orbits)
    reference = np.outer(reference_cycle, reference_cycle).reshape(-1)
    alternative = expected_pair_feature(
        model,
        q=q0,
        kappa=kappa1,
        orbits=orbits,
    )
    matched_effect = effect_from_direction(alternative - reference)
    first = observed[0::2]
    second = observed[1::2]
    null_log = model.nonoverlapping_pair_log_likelihoods(
        first,
        second,
        q=q0,
        kappa=0.0,
    )
    grid_log = np.stack(
        [
            model.nonoverlapping_pair_log_likelihoods(
                first,
                second,
                q=q0,
                kappa=value,
            )
            for value in kappa_grid
        ],
        axis=1,
    )
    ratios = np.exp(np.clip(grid_log - null_log[:, None], -700.0, 700.0))
    oracle = np.exp(
        np.clip(
            model.nonoverlapping_pair_log_likelihoods(
                first,
                second,
                q=q0,
                kappa=kappa1,
            )
            - null_log,
            -700.0,
            700.0,
        )
    )
    first_conditional_log = np.stack(
        [
            model.conditional_length_emission_log_likelihoods(first, 1),
            model.conditional_length_emission_log_likelihoods(first, 2),
        ],
        axis=1,
    )
    second_conditional_log = np.stack(
        [
            model.conditional_length_emission_log_likelihoods(second, 1),
            model.conditional_length_emission_log_likelihoods(second, 2),
        ],
        axis=1,
    )
    first_null_log = model.emission_log_likelihoods(first, q0)
    second_null_log = model.emission_log_likelihoods(second, q0)
    hmm_transitions = np.stack(
        [model.length_transition_matrix(q0, value) for value in kappa_grid]
    )
    return DetectorInputs(
        features=lifted,
        matched_effect=matched_effect,
        likelihood_ratios=ratios,
        oracle_likelihood_ratio=oracle,
        likelihood_method="exact pair-restricted likelihood grid SR",
        oracle_method="known-post pair-restricted likelihood SR",
        hmm_first_emission_ratios=np.exp(
            first_conditional_log - first_null_log[:, None]
        ),
        hmm_second_emission_ratios=np.exp(
            second_conditional_log - second_null_log[:, None]
        ),
        hmm_transitions=hmm_transitions,
        hmm_oracle_transition=model.length_transition_matrix(q0, kappa1),
        hmm_stationary=np.asarray([1.0 - q0, q0]),
    )


def metric_rows(
    alarm_times: dict[str, int | None],
    *,
    family: str,
    effect: float,
    replicate: int,
    seed: int,
    stream_type: str,
    evaluation: str,
    horizon: int,
    change_time: int | None,
    time_unit_cycles: int,
    threshold: float,
) -> list[dict[str, Any]]:
    """Convert alarm times without treating pre-change alarms as misses.

    ``surveillance`` delays are conditional on the detector having survived
    through the changepoint.  ``restart_at_change`` starts every detector from
    zero on the same post-change segment and therefore removes the age of the
    SR statistic from the detection-delay comparison.  No-change rows retain
    the ordinary right-censored run-length audit.
    """

    allowed_evaluations = {"null_arl", "surveillance", "restart_at_change"}
    if evaluation not in allowed_evaluations:
        raise ValueError(f"Unknown evaluation: {evaluation}")
    if evaluation == "null_arl" and change_time is not None:
        raise ValueError("Null ARL rows cannot have a changepoint.")
    if evaluation != "null_arl" and change_time is None:
        raise ValueError("Changed-stream rows require a changepoint.")

    rows = []
    for method, alarm_time in alarm_times.items():
        if change_time is None:
            false_alarm = alarm_time is not None
            censored = alarm_time is None
            run_length = horizon if alarm_time is None else alarm_time
            delay = np.nan
            conditional_eligible = True
            post_change_detected = False
        else:
            false_alarm = alarm_time is not None and alarm_time <= change_time
            conditional_eligible = not false_alarm
            post_change_detected = (
                conditional_eligible
                and alarm_time is not None
                and alarm_time > change_time
            )
            censored = conditional_eligible and not post_change_detected
            run_length = change_time + horizon if alarm_time is None else alarm_time
            if not conditional_eligible:
                delay = np.nan
            elif post_change_detected:
                delay = min(horizon, alarm_time - change_time)
            else:
                delay = horizon
        rows.append(
            {
                "family": family,
                "effect": effect,
                "replicate": replicate,
                "seed": seed,
                "stream_type": stream_type,
                "evaluation": evaluation,
                "method": method,
                "alarm_time": alarm_time,
                "false_alarm": false_alarm,
                "conditional_eligible": conditional_eligible,
                "post_change_detected": post_change_detected,
                "censored": censored,
                "run_length_units": run_length,
                "run_length_cycles": run_length * time_unit_cycles,
                "restricted_delay_units": delay,
                "time_unit_cycles": time_unit_cycles,
                "restricted_delay_cycles": (
                    np.nan if np.isnan(delay) else delay * time_unit_cycles
                ),
                "restriction_horizon_units": horizon,
                "restriction_horizon_cycles": horizon * time_unit_cycles,
                "threshold": threshold,
            }
        )
    return rows


def run_spatial_stream(
    model: PeriodicSurfaceSyndromeModel,
    *,
    q0: float,
    q1: float,
    q_grid: np.ndarray,
    orbits: tuple[np.ndarray, ...],
    reference: np.ndarray,
    covariance: np.ndarray,
    ridge: float,
    validation_effect: np.ndarray | None,
    config: dict[str, Any],
    seed: int,
    null: bool,
    restart_at_change: bool = False,
) -> dict[str, int | None]:
    rng = np.random.default_rng(seed)
    if null:
        if restart_at_change:
            raise ValueError("A null stream cannot restart at a changepoint.")
        observed = model.sample_spatial(
            int(config["null_horizon_cycles"]),
            q0,
            rng=rng,
        )
    else:
        before = model.sample_spatial(
            int(config["change_time_cycles"]),
            q0,
            rng=rng,
        )
        after = model.sample_spatial(
            int(config["post_change_horizon_cycles"]),
            q1,
            rng=rng,
        )
        observed = after if restart_at_change else np.concatenate([before, after])
    inputs = spatial_inputs(
        model,
        observed,
        q0=q0,
        q1=q1,
        q_grid=q_grid,
        orbits=orbits,
    )
    return detector_alarm_times(
        inputs,
        reference=reference,
        covariance=covariance,
        ridge=ridge,
        validation_effect=validation_effect,
        window=int(config["witness_window_updates"]),
        update_interval=int(config["witness_update_interval_updates"]),
        threshold=float(config["detector_threshold_updates"]["spatial"]),
        bet_fractions=np.asarray(config["bet_fractions"]),
    )


def run_temporal_stream(
    model: PeriodicSurfaceSyndromeModel,
    *,
    q0: float,
    kappa1: float,
    kappa_grid: np.ndarray,
    orbits: tuple[np.ndarray, ...],
    reference: np.ndarray,
    covariance: np.ndarray,
    ridge: float,
    validation_effect: np.ndarray | None,
    config: dict[str, Any],
    seed: int,
    null: bool,
    restart_at_change: bool = False,
) -> dict[str, int | None]:
    rng = np.random.default_rng(seed)
    if null:
        if restart_at_change:
            raise ValueError("A null stream cannot restart at a changepoint.")
        observed = model.sample_temporal(
            int(config["null_horizon_cycles"]),
            q=q0,
            kappa=0.0,
            rng=rng,
        )[0]
    else:
        before = model.sample_temporal(
            int(config["change_time_cycles"]),
            q=q0,
            kappa=0.0,
            rng=rng,
        )[0]
        after = model.sample_temporal(
            int(config["post_change_horizon_cycles"]),
            q=q0,
            kappa=kappa1,
            rng=rng,
        )[0]
        observed = after if restart_at_change else np.concatenate([before, after])
    inputs = temporal_inputs(
        model,
        observed,
        q0=q0,
        kappa1=kappa1,
        kappa_grid=kappa_grid,
        orbits=orbits,
    )
    return detector_alarm_times(
        inputs,
        reference=reference,
        covariance=covariance,
        ridge=ridge,
        validation_effect=validation_effect,
        window=int(config["witness_window_updates"]),
        update_interval=int(config["witness_update_interval_updates"]),
        threshold=float(config["detector_threshold_updates"]["temporal"]),
        bet_fractions=np.asarray(config["bet_fractions"]),
    )


def select_ridge(
    family: str,
    *,
    candidates: list[float],
    stream_runner: Any,
    effect: float,
    validation_streams: int,
    seed_start: int,
    horizon_updates: int,
    time_unit_cycles: int,
) -> tuple[float, list[dict[str, Any]]]:
    """Select a ridge on independent, fresh post-change validation streams."""

    rows = []
    for ridge in candidates:
        delays = []
        for replicate in range(validation_streams):
            alarms = stream_runner(
                ridge=ridge,
                seed=seed_start + replicate,
                null=False,
                restart_at_change=True,
            )
            alarm = alarms["vAOC / same-feature Hotelling"]
            delay = horizon_updates if alarm is None else min(horizon_updates, alarm)
            delays.append(delay)
        rows.append(
            {
                "family": family,
                "effect": effect,
                "ridge": ridge,
                "validation_streams": validation_streams,
                "evaluation": "restart_at_change",
                "time_unit_cycles": time_unit_cycles,
                "restriction_horizon_updates": horizon_updates,
                "restriction_horizon_cycles": (horizon_updates * time_unit_cycles),
                "mean_restricted_delay_from_restart_updates": float(np.mean(delays)),
                "mean_restricted_delay_from_restart_cycles": float(
                    np.mean(delays) * time_unit_cycles
                ),
            }
        )
    selected = min(
        rows,
        key=lambda row: (
            row["mean_restricted_delay_from_restart_cycles"],
            row["ridge"],
        ),
    )["ridge"]
    return float(selected), rows


def aggregate_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate separate null, surveillance, and restart estimands."""

    required_evaluations = {"null_arl", "surveillance", "restart_at_change"}
    missing = required_evaluations - set(frame["evaluation"].unique())
    if missing:
        raise ValueError(f"Metric frame is missing evaluations: {sorted(missing)}")

    keys = ["family", "effect", "method"]
    restart = frame[frame.evaluation == "restart_at_change"]
    surveillance = frame[frame.evaluation == "surveillance"]
    null = frame[frame.evaluation == "null_arl"]
    restart_summary = restart.groupby(keys, as_index=False).agg(
        mean_restart_delay_cycles=("restricted_delay_cycles", "mean"),
        median_restart_delay_cycles=("restricted_delay_cycles", "median"),
        restart_delay_q1_cycles=(
            "restricted_delay_cycles",
            lambda values: values.quantile(0.25),
        ),
        restart_delay_q3_cycles=(
            "restricted_delay_cycles",
            lambda values: values.quantile(0.75),
        ),
        restart_detection_within_64_cycles_fraction=(
            "restricted_delay_cycles",
            lambda values: (values <= 64).mean(),
        ),
        restart_detection_within_128_cycles_fraction=(
            "restricted_delay_cycles",
            lambda values: (values <= 128).mean(),
        ),
        restart_miss_fraction=("censored", "mean"),
        restart_replicates=("replicate", "count"),
        restart_horizon_cycles=("restriction_horizon_cycles", "first"),
    )
    surveillance_rates = surveillance.groupby(keys, as_index=False).agg(
        surveillance_prechange_false_alarm_fraction=("false_alarm", "mean"),
        surveillance_replicates=("replicate", "count"),
    )
    surveillance_survivors = surveillance[surveillance.conditional_eligible]
    surveillance_summary = surveillance_survivors.groupby(keys, as_index=False).agg(
        surveillance_conditional_mean_delay_cycles=(
            "restricted_delay_cycles",
            "mean",
        ),
        surveillance_conditional_median_delay_cycles=(
            "restricted_delay_cycles",
            "median",
        ),
        surveillance_miss_fraction_among_survivors=("censored", "mean"),
        surveillance_survivors=("replicate", "count"),
    )
    null_summary = null.groupby(keys, as_index=False).agg(
        restricted_mean_run_length_cycles=("run_length_cycles", "mean"),
        null_censor_fraction=("censored", "mean"),
        null_replicates=("replicate", "count"),
    )
    aggregate = (
        restart_summary.merge(surveillance_rates, on=keys, how="left")
        .merge(surveillance_summary, on=keys, how="left")
        .merge(null_summary, on=keys, how="left")
    )
    null_lookup = {key: group for key, group in null.groupby(keys, sort=False)}
    no_change_rmst = []
    for row in aggregate.itertuples(index=False):
        key = (row.family, row.effect, row.method)
        group = null_lookup[key]
        horizon_cycles = float(row.restart_horizon_cycles)
        run_lengths = group["alarm_time"].fillna(np.inf).to_numpy(dtype=float)
        run_lengths *= group["time_unit_cycles"].to_numpy(dtype=float)
        no_change_rmst.append(float(np.minimum(run_lengths, horizon_cycles).mean()))
    aggregate["no_change_rmst_at_restart_horizon_cycles"] = no_change_rmst
    aggregate["restart_delay_reduction_vs_no_change_cycles"] = (
        aggregate["no_change_rmst_at_restart_horizon_cycles"]
        - aggregate["mean_restart_delay_cycles"]
    )
    denominator = aggregate["no_change_rmst_at_restart_horizon_cycles"]
    aggregate["restart_delay_ratio_to_no_change"] = np.divide(
        aggregate["mean_restart_delay_cycles"],
        denominator,
        out=np.full(len(aggregate), np.nan),
        where=denominator > 0,
    )
    return aggregate


def bootstrap_comparisons(
    frame: pd.DataFrame,
    *,
    resamples: int,
) -> pd.DataFrame:
    rows = []
    changed = frame[frame.evaluation == "restart_at_change"]
    for (family, effect), subset in changed.groupby(["family", "effect"]):
        pivot = subset.pivot(
            index="replicate",
            columns="method",
            values="restricted_delay_cycles",
        )
        variance = pivot["vAOC / same-feature Hotelling"].to_numpy()
        baselines = [
            method
            for method in pivot.columns
            if method
            not in {
                "DFR/count exact pushforward",
                "vAOC / same-feature Hotelling",
            }
        ]
        for baseline in baselines:
            estimate, low, high = paired_bootstrap_mean_difference(
                variance,
                pivot[baseline].to_numpy(),
                resamples=resamples,
                seed=880000 + round(100 * effect),
            )
            rows.append(
                {
                    "family": family,
                    "effect": effect,
                    "evaluation": "restart_at_change",
                    "comparison": f"vAOC minus {baseline}",
                    "mean_delay_difference_cycles": estimate,
                    "ci95_low": low,
                    "ci95_high": high,
                }
            )
    return pd.DataFrame(rows)


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    """Holm-adjust a finite vector of valid p-values."""

    values = np.asarray(p_values, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("p_values must be a nonempty vector.")
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("p_values must lie in [0,1].")
    order = np.argsort(values, kind="stable")
    adjusted = np.empty_like(values)
    running = 0.0
    hypotheses = len(values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (hypotheses - rank) * values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def bounded_delay_hoeffding_inference(
    differences: np.ndarray,
    *,
    horizon_cycles: float,
    one_sided_alpha: float,
) -> tuple[float, float]:
    """Return a valid one-sided p-value and UCB for bounded paired delays.

    For independent paired differences ``D_i`` in ``[-H, H]``, this tests
    ``H0: E[D] >= 0`` against a negative mean and returns the Hoeffding upper
    confidence bound

    ``mean(D) + H sqrt(2 log(1/alpha) / n)``.
    """

    values = np.asarray(differences, dtype=np.float64)
    horizon = float(horizon_cycles)
    alpha = float(one_sided_alpha)
    if values.ndim != 1 or len(values) == 0 or not np.all(np.isfinite(values)):
        raise ValueError("differences must be a finite nonempty vector.")
    if horizon <= 0.0:
        raise ValueError("horizon_cycles must be positive.")
    if not 0.0 < alpha < 1.0:
        raise ValueError("one_sided_alpha must lie in (0,1).")
    if np.any(values < -horizon - 1e-9) or np.any(values > horizon + 1e-9):
        raise ValueError("Paired delay differences must lie in [-H,H].")
    mean = float(values.mean())
    negative_deviation = max(0.0, -mean)
    p_value = min(
        1.0,
        float(np.exp(-len(values) * negative_deviation**2 / (2.0 * horizon**2))),
    )
    upper = mean + horizon * np.sqrt(2.0 * np.log(1.0 / alpha) / len(values))
    return p_value, float(upper)


def paired_primary_comparator_audit(
    frame: pd.DataFrame,
    *,
    hypotheses: dict[str, Any],
    bootstrap_resamples: int,
    bootstrap_seed: int,
    arl_requirement_met: bool,
) -> pd.DataFrame:
    """Audit vAOC versus the named comparator on two locked paired tasks."""

    proposed = str(hypotheses["proposed_method"])
    baseline = str(hypotheses["baseline_method"])
    alpha = float(hypotheses["familywise_alpha"])
    targets = (
        ("spatial", float(hypotheses["spatial_effect"])),
        ("temporal", float(hypotheses["temporal_effect"])),
    )
    rows: list[dict[str, Any]] = []
    raw_p_values: list[float] = []
    changed = frame[frame["evaluation"] == "restart_at_change"]
    simultaneous_alpha = alpha / len(targets)

    for hypothesis_index, (family, effect) in enumerate(targets):
        subset = changed[
            (changed["family"] == family)
            & np.isclose(changed["effect"], effect)
            & changed["method"].isin([proposed, baseline])
        ]
        pivot = subset.pivot(
            index=["replicate", "seed"],
            columns="method",
            values="restricted_delay_cycles",
        )
        if proposed not in pivot or baseline not in pivot or pivot.isna().any().any():
            raise ValueError(
                f"Primary paired data are incomplete for {family} effect {effect}."
            )
        differences = pivot[proposed].to_numpy(dtype=np.float64) - pivot[
            baseline
        ].to_numpy(dtype=np.float64)
        if len(differences) == 0:
            raise ValueError("A primary comparator hypothesis has no paired data.")
        horizons = subset["restriction_horizon_cycles"].unique()
        if len(horizons) != 1:
            raise ValueError("A primary hypothesis must have one cycle horizon.")
        horizon_cycles = float(horizons[0])

        bootstrap_rng = np.random.default_rng(bootstrap_seed + hypothesis_index)
        bootstrap_indices = bootstrap_rng.integers(
            0,
            len(differences),
            size=(bootstrap_resamples, len(differences)),
        )
        bootstrap_means = differences[bootstrap_indices].mean(axis=1)
        pointwise_low, pointwise_high = np.quantile(
            bootstrap_means,
            (alpha / 2.0, 1.0 - alpha / 2.0),
        )
        observed = float(differences.mean())
        raw_p, simultaneous_upper = bounded_delay_hoeffding_inference(
            differences,
            horizon_cycles=horizon_cycles,
            one_sided_alpha=simultaneous_alpha,
        )
        raw_p_values.append(float(raw_p))
        rows.append(
            {
                "hypothesis": f"{family}@{effect:g}",
                "family": family,
                "effect": effect,
                "evaluation": "restart_at_change",
                "proposed_method": proposed,
                "baseline_method": baseline,
                "paired_replicates": len(differences),
                "estimand": "mean delay(proposed)-mean delay(baseline), cycles",
                "support_direction": "negative",
                "mean_delay_difference_cycles": observed,
                "descriptive_bootstrap_ci95_low_cycles": float(pointwise_low),
                "descriptive_bootstrap_ci95_high_cycles": float(pointwise_high),
                "descriptive_bootstrap_inferential_status": (
                    "Descriptive only; not used for the comparator decision."
                ),
                "simultaneous_method": (
                    "Bonferroni-Hoeffding one-sided upper confidence bounds"
                ),
                "simultaneous_familywise_confidence": 1.0 - alpha,
                "simultaneous_upper_cycles": simultaneous_upper,
                "delay_bound_horizon_cycles": horizon_cycles,
                "raw_one_sided_hoeffding_p": raw_p,
                "inference_assumptions": (
                    "Conditional on the frozen auxiliary fits, paired test "
                    "streams are independent across replicates; both "
                    "restricted delays lie in [0,H], so their difference "
                    "lies in [-H,H]. No symmetry or parametric assumption."
                ),
                "bootstrap_resamples": bootstrap_resamples,
                "arl_requirement_met_by_design": bool(arl_requirement_met),
                "arl_basis": (
                    "Both effects are fixed/predictable simplex effects "
                    "centered at the analytic null; the family threshold in "
                    "detector updates maps exactly to target ARL cycles."
                ),
            }
        )

    adjusted = holm_adjust(np.asarray(raw_p_values))
    for row, adjusted_p in zip(rows, adjusted, strict=True):
        row["holm_adjusted_one_sided_p"] = float(adjusted_p)
        row["statistical_criteria_pass"] = bool(
            row["simultaneous_upper_cycles"] < 0.0 and adjusted_p < alpha
        )
        row["comparison_supported_for_hypothesis"] = bool(
            row["statistical_criteria_pass"] and row["arl_requirement_met_by_design"]
        )
    overall = bool(
        len(rows) == 2
        and all(row["comparison_supported_for_hypothesis"] for row in rows)
    )
    for row in rows:
        row["overall_two_hypothesis_comparison_supported"] = overall
        row["scope_statement"] = (
            "Support is limited to vAOC versus the named predeclared "
            "logistic comparator on these two controlled tasks."
        )
    return pd.DataFrame(rows)


def run_spatial_scaling_arm(
    config: dict[str, Any],
    *,
    q0: float,
    q1: float,
    ridge: float,
) -> pd.DataFrame:
    """Run the locked fresh-start spatial scaling and engineering timing arm."""

    if "scaling_sizes" not in config or "scaling_repetitions" not in config:
        raise ValueError(
            "Publication scaling requires scaling_sizes and scaling_repetitions."
        )
    model_config = config["model"]
    methods = (
        "DFR/count exact pushforward",
        "raw symmetry-resolved AOC",
        "vAOC / same-feature Hotelling",
        "matched correlation witness",
        "known-post one-cycle likelihood SR",
    )
    rows: list[dict[str, Any]] = []
    timing_samples = 10_000
    for size in [int(value) for value in config["scaling_sizes"]]:
        model = PeriodicSurfaceSyndromeModel(
            size=size,
            event_probability=float(model_config["event_probability"]),
            readout_error=float(model_config["readout_error"]),
        )
        orbits = d4_frequency_orbits(size)
        calibration_rng = np.random.default_rng(
            int(config["seeds"]["scaling_start"]) + 100_000 * size + 10_000
        )
        calibration = model.sample_spatial(
            int(config["calibration_cycles"]),
            q0,
            rng=calibration_rng,
        )
        calibration_features = aggregate_orbits(
            model.fourier_power_features(calibration),
            orbits,
        )
        reference = aggregate_orbits(
            model.expected_fourier_spectrum(q0),
            orbits,
        )
        covariance = covariance_with_ridge_samples(calibration_features)

        delays: dict[str, list[float]] = {method: [] for method in methods}
        misses: dict[str, list[bool]] = {method: [] for method in methods}
        for replicate in range(int(config["scaling_repetitions"])):
            seed = int(config["seeds"]["scaling_start"]) + 100_000 * size + replicate
            alarms = run_spatial_stream(
                model,
                q0=q0,
                q1=q1,
                q_grid=np.asarray([q1], dtype=np.float64),
                orbits=orbits,
                reference=reference,
                covariance=covariance,
                ridge=ridge,
                validation_effect=None,
                config=config,
                seed=seed,
                null=False,
                restart_at_change=True,
            )
            for method in methods:
                alarm = alarms[method]
                missed = alarm is None
                misses[method].append(missed)
                delays[method].append(
                    float(config["post_change_horizon_cycles"] if missed else alarm)
                )

        timing_rng = np.random.default_rng(int(config["seeds"]["timing_start"]) + size)
        timing_observed = model.sample_spatial(
            timing_samples,
            q1,
            rng=timing_rng,
        )
        model.translation_pair_features(timing_observed[:10])
        model.emission_log_likelihoods(timing_observed[:10], q0)
        feature_started = time.perf_counter()
        model.translation_pair_features(timing_observed)
        feature_seconds = time.perf_counter() - feature_started
        tracemalloc.start()
        model.translation_pair_features(timing_observed)
        _, feature_peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        likelihood_started = time.perf_counter()
        null_log = model.emission_log_likelihoods(timing_observed, q0)
        post_log = model.emission_log_likelihoods(timing_observed, q1)
        _ = post_log - null_log
        likelihood_seconds = time.perf_counter() - likelihood_started
        tracemalloc.start()
        null_log = model.emission_log_likelihoods(timing_observed, q0)
        post_log = model.emission_log_likelihoods(timing_observed, q1)
        _ = post_log - null_log
        _, likelihood_peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        analytic_gap = np.abs(
            model.expected_translation_pair_features(q1)
            - model.expected_translation_pair_features(q0)
        )
        for method in methods:
            method_delays = np.asarray(delays[method], dtype=np.float64)
            rows.append(
                {
                    "size": size,
                    "detectors": model.num_detectors,
                    "q0": q0,
                    "q1": q1,
                    "method": method,
                    "evaluation": "restart_at_change",
                    "repetitions": int(config["scaling_repetitions"]),
                    "horizon_cycles": int(config["post_change_horizon_cycles"]),
                    "threshold_updates": float(
                        config["detector_threshold_updates"]["spatial"]
                    ),
                    "target_arl_cycles": int(config["target_arl_cycles"]),
                    "ridge_frozen_from_size5": ridge,
                    "mean_restricted_delay_cycles": float(method_delays.mean()),
                    "median_restricted_delay_cycles": float(np.median(method_delays)),
                    "miss_fraction": float(np.mean(misses[method])),
                    "analytic_translation_gap_l1": float(analytic_gap.sum()),
                    "analytic_translation_gap_l2": float(np.linalg.norm(analytic_gap)),
                    "analytic_translation_gap_max": float(analytic_gap.max()),
                    "translation_feature_seconds_per_10k": feature_seconds,
                    "exact_llr_seconds_per_10k": likelihood_seconds,
                    "translation_feature_peak_tracemalloc_bytes_per_10k": int(
                        feature_peak_bytes
                    ),
                    "exact_llr_peak_tracemalloc_bytes_per_10k": int(
                        likelihood_peak_bytes
                    ),
                    "timing_samples": timing_samples,
                    "timing_note": (
                        "Single-process wall-clock engineering measurement "
                        "after a 10-sample warm-up; not a complexity theorem."
                    ),
                    "memory_note": (
                        "Peak Python/NumPy allocation traced by tracemalloc "
                        "during a separate 10,000-sample call; not total "
                        "process RSS or a complexity theorem."
                    ),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    started = time.time()
    result_name = "paper" if config["publication_grade"] else "pilot"
    results = RUN_ROOT / "results" / f"syndrome_drift_{result_name}"
    results.mkdir(parents=True, exist_ok=True)
    model_config = config["model"]
    model = PeriodicSurfaceSyndromeModel(
        size=int(model_config["size"]),
        event_probability=float(model_config["event_probability"]),
        readout_error=float(model_config["readout_error"]),
    )
    q0 = float(model_config["null_chain2_probability"])
    q_grid = np.asarray(config["spatial_alternatives"], dtype=np.float64)
    kappa_grid = np.asarray(config["temporal_persistence"], dtype=np.float64)
    orbits = d4_frequency_orbits(model.size)

    calibration_rng = np.random.default_rng(
        int(config["seeds"]["calibration"]["spatial"])
    )
    calibration = model.sample_spatial(
        int(config["calibration_cycles"]),
        q0,
        rng=calibration_rng,
    )
    calibration_cycle_features = aggregate_orbits(
        model.fourier_power_features(calibration),
        orbits,
    )
    spatial_reference = aggregate_orbits(
        model.expected_fourier_spectrum(q0),
        orbits,
    )
    spatial_covariance = covariance_with_ridge_samples(calibration_cycle_features)

    temporal_calibration = model.sample_spatial(
        int(config["calibration_cycles"]),
        q0,
        seed=int(config["seeds"]["calibration"]["temporal"]),
    )
    temporal_cycle_features = aggregate_orbits(
        model.fourier_power_features(temporal_calibration),
        orbits,
    )
    temporal_features = pair_features(temporal_cycle_features)
    temporal_reference = np.outer(spatial_reference, spatial_reference).reshape(-1)
    temporal_covariance = covariance_with_ridge_samples(temporal_features)

    validation_rows: list[dict[str, Any]] = []
    middle_q = float(q_grid[len(q_grid) // 2])
    middle_kappa = float(kappa_grid[len(kappa_grid) // 2])
    validation_effects, baseline_metadata = fit_validation_trained_baselines(
        model,
        config=config,
        q0=q0,
        spatial_effect=middle_q,
        temporal_effect=middle_kappa,
        orbits=orbits,
    )
    baseline_path = results / "validation_trained_baseline.json"
    baseline_path.write_text(
        json.dumps(baseline_metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    ridge_validation_config = {
        **config,
        "post_change_horizon_cycles": int(config["ridge_validation_horizon_cycles"]),
    }
    spatial_runner = lambda **kwargs: run_spatial_stream(
        model,
        q0=q0,
        q1=middle_q,
        q_grid=q_grid,
        orbits=orbits,
        reference=spatial_reference,
        covariance=spatial_covariance,
        validation_effect=validation_effects["spatial"],
        config=ridge_validation_config,
        **kwargs,
    )
    temporal_runner = lambda **kwargs: run_temporal_stream(
        model,
        q0=q0,
        kappa1=middle_kappa,
        kappa_grid=kappa_grid,
        orbits=orbits,
        reference=temporal_reference,
        covariance=temporal_covariance,
        validation_effect=validation_effects["temporal"],
        config=ridge_validation_config,
        **kwargs,
    )
    selected_spatial_ridge, spatial_validation = select_ridge(
        "spatial",
        candidates=[float(value) for value in config["ridge_grid"]],
        stream_runner=spatial_runner,
        effect=middle_q,
        validation_streams=int(config["validation_streams"]),
        seed_start=int(config["seeds"]["ridge_validation"]["spatial_start"]),
        horizon_updates=cycles_to_updates(
            int(config["ridge_validation_horizon_cycles"]),
            "spatial",
        ),
        time_unit_cycles=family_time_unit_cycles("spatial"),
    )
    selected_temporal_ridge, temporal_validation = select_ridge(
        "temporal",
        candidates=[float(value) for value in config["ridge_grid"]],
        stream_runner=temporal_runner,
        effect=middle_kappa,
        validation_streams=int(config["validation_streams"]),
        seed_start=int(config["seeds"]["ridge_validation"]["temporal_start"]),
        horizon_updates=cycles_to_updates(
            int(config["ridge_validation_horizon_cycles"]),
            "temporal",
        ),
        time_unit_cycles=family_time_unit_cycles("temporal"),
    )
    validation_rows.extend(spatial_validation)
    validation_rows.extend(temporal_validation)
    validation_frame = pd.DataFrame(validation_rows)
    validation_path = results / "ridge_selection.csv"
    validation_frame.to_csv(validation_path, index=False)

    metric_data: list[dict[str, Any]] = []
    spatial_test_seed = int(config["seeds"]["locked_test"]["spatial_start"])
    for effect_index, q1 in enumerate(q_grid):
        for replicate in range(int(config["null_test_streams"])):
            seed = spatial_test_seed + effect_index * 1_000_000 + replicate
            alarms = run_spatial_stream(
                model,
                q0=q0,
                q1=float(q1),
                q_grid=q_grid,
                orbits=orbits,
                reference=spatial_reference,
                covariance=spatial_covariance,
                ridge=selected_spatial_ridge,
                validation_effect=validation_effects["spatial"],
                config=config,
                seed=seed,
                null=True,
            )
            metric_data.extend(
                metric_rows(
                    alarms,
                    family="spatial",
                    effect=float(q1),
                    replicate=replicate,
                    seed=seed,
                    stream_type="no_change",
                    evaluation="null_arl",
                    horizon=cycles_to_updates(
                        int(config["null_horizon_cycles"]),
                        "spatial",
                    ),
                    change_time=None,
                    time_unit_cycles=1,
                    threshold=float(config["detector_threshold_updates"]["spatial"]),
                )
            )
        for replicate in range(int(config["changed_test_streams"])):
            seed = spatial_test_seed + effect_index * 1_000_000 + 100_000 + replicate
            alarms = run_spatial_stream(
                model,
                q0=q0,
                q1=float(q1),
                q_grid=q_grid,
                orbits=orbits,
                reference=spatial_reference,
                covariance=spatial_covariance,
                ridge=selected_spatial_ridge,
                validation_effect=validation_effects["spatial"],
                config=config,
                seed=seed,
                null=False,
            )
            metric_data.extend(
                metric_rows(
                    alarms,
                    family="spatial",
                    effect=float(q1),
                    replicate=replicate,
                    seed=seed,
                    stream_type="changed",
                    evaluation="surveillance",
                    horizon=cycles_to_updates(
                        int(config["post_change_horizon_cycles"]),
                        "spatial",
                    ),
                    change_time=cycles_to_updates(
                        int(config["change_time_cycles"]),
                        "spatial",
                    ),
                    time_unit_cycles=1,
                    threshold=float(config["detector_threshold_updates"]["spatial"]),
                )
            )
            restart_alarms = run_spatial_stream(
                model,
                q0=q0,
                q1=float(q1),
                q_grid=q_grid,
                orbits=orbits,
                reference=spatial_reference,
                covariance=spatial_covariance,
                ridge=selected_spatial_ridge,
                validation_effect=validation_effects["spatial"],
                config=config,
                seed=seed,
                null=False,
                restart_at_change=True,
            )
            metric_data.extend(
                metric_rows(
                    restart_alarms,
                    family="spatial",
                    effect=float(q1),
                    replicate=replicate,
                    seed=seed,
                    stream_type="changed",
                    evaluation="restart_at_change",
                    horizon=cycles_to_updates(
                        int(config["post_change_horizon_cycles"]),
                        "spatial",
                    ),
                    change_time=0,
                    time_unit_cycles=1,
                    threshold=float(config["detector_threshold_updates"]["spatial"]),
                )
            )

    temporal_test_seed = int(config["seeds"]["locked_test"]["temporal_start"])
    for effect_index, kappa1 in enumerate(kappa_grid):
        for replicate in range(int(config["null_test_streams"])):
            seed = temporal_test_seed + effect_index * 1_000_000 + replicate
            alarms = run_temporal_stream(
                model,
                q0=q0,
                kappa1=float(kappa1),
                kappa_grid=kappa_grid,
                orbits=orbits,
                reference=temporal_reference,
                covariance=temporal_covariance,
                ridge=selected_temporal_ridge,
                validation_effect=validation_effects["temporal"],
                config=config,
                seed=seed,
                null=True,
            )
            metric_data.extend(
                metric_rows(
                    alarms,
                    family="temporal",
                    effect=float(kappa1),
                    replicate=replicate,
                    seed=seed,
                    stream_type="no_change",
                    evaluation="null_arl",
                    horizon=cycles_to_updates(
                        int(config["null_horizon_cycles"]),
                        "temporal",
                    ),
                    change_time=None,
                    time_unit_cycles=2,
                    threshold=float(config["detector_threshold_updates"]["temporal"]),
                )
            )
        for replicate in range(int(config["changed_test_streams"])):
            seed = temporal_test_seed + effect_index * 1_000_000 + 100_000 + replicate
            alarms = run_temporal_stream(
                model,
                q0=q0,
                kappa1=float(kappa1),
                kappa_grid=kappa_grid,
                orbits=orbits,
                reference=temporal_reference,
                covariance=temporal_covariance,
                ridge=selected_temporal_ridge,
                validation_effect=validation_effects["temporal"],
                config=config,
                seed=seed,
                null=False,
            )
            metric_data.extend(
                metric_rows(
                    alarms,
                    family="temporal",
                    effect=float(kappa1),
                    replicate=replicate,
                    seed=seed,
                    stream_type="changed",
                    evaluation="surveillance",
                    horizon=cycles_to_updates(
                        int(config["post_change_horizon_cycles"]),
                        "temporal",
                    ),
                    change_time=cycles_to_updates(
                        int(config["change_time_cycles"]),
                        "temporal",
                    ),
                    time_unit_cycles=2,
                    threshold=float(config["detector_threshold_updates"]["temporal"]),
                )
            )
            restart_alarms = run_temporal_stream(
                model,
                q0=q0,
                kappa1=float(kappa1),
                kappa_grid=kappa_grid,
                orbits=orbits,
                reference=temporal_reference,
                covariance=temporal_covariance,
                ridge=selected_temporal_ridge,
                validation_effect=validation_effects["temporal"],
                config=config,
                seed=seed,
                null=False,
                restart_at_change=True,
            )
            metric_data.extend(
                metric_rows(
                    restart_alarms,
                    family="temporal",
                    effect=float(kappa1),
                    replicate=replicate,
                    seed=seed,
                    stream_type="changed",
                    evaluation="restart_at_change",
                    horizon=cycles_to_updates(
                        int(config["post_change_horizon_cycles"]),
                        "temporal",
                    ),
                    change_time=0,
                    time_unit_cycles=2,
                    threshold=float(config["detector_threshold_updates"]["temporal"]),
                )
            )

    metrics = pd.DataFrame(metric_data)
    metrics_path = results / "replicate_metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    aggregate = aggregate_metrics(metrics)
    aggregate_path = results / "aggregate_metrics.csv"
    aggregate.to_csv(aggregate_path, index=False)
    comparisons = bootstrap_comparisons(
        metrics,
        resamples=int(config["bootstrap_resamples"]),
    )
    comparisons_path = results / "paired_bootstrap.csv"
    comparisons.to_csv(comparisons_path, index=False)
    comparator_audit = paired_primary_comparator_audit(
        metrics,
        hypotheses=config["primary_comparison_hypotheses"],
        bootstrap_resamples=int(config["bootstrap_resamples"]),
        bootstrap_seed=int(config["seeds"]["analysis"]["bootstrap"]),
        arl_requirement_met=True,
    )
    comparator_path = results / "primary_named_comparator_audit.csv"
    comparator_audit.to_csv(comparator_path, index=False)
    scaling_path: Path | None = None
    scaling = pd.DataFrame()
    if config["publication_grade"]:
        scaling = run_spatial_scaling_arm(
            config,
            q0=q0,
            q1=middle_q,
            ridge=selected_spatial_ridge,
        )
        scaling_path = results / "spatial_scaling.csv"
        scaling.to_csv(scaling_path, index=False)

    sns.set_theme(style="whitegrid", context="paper")
    figure, axes = plt.subplots(1, 2, figsize=(10.2, 3.8), sharey=False)
    for axis, family in zip(axes, ("spatial", "temporal"), strict=True):
        subset = aggregate[aggregate.family == family]
        sns.lineplot(
            data=subset,
            x="effect",
            y="mean_restart_delay_cycles",
            hue="method",
            marker="o",
            ax=axis,
        )
        axis.set_title(f"{family} drift")
        axis.set_xlabel("post-change parameter")
        axis.set_ylabel("restart-at-change restricted mean delay (cycles)")
        axis.legend(fontsize=6)
    figure.tight_layout()
    pdf_path = results / "syndrome_drift.pdf"
    png_path = results / "syndrome_drift.png"
    figure.savefig(pdf_path)
    figure.savefig(png_path, dpi=220)
    plt.close(figure)

    target_spatial = aggregate[
        (aggregate.family == "spatial") & (aggregate.effect == middle_q)
    ].set_index("method")
    target_temporal = aggregate[
        (aggregate.family == "temporal") & (aggregate.effect == middle_kappa)
    ].set_index("method")
    summary = {
        "configuration": config["name"],
        "locked": bool(config["locked"]),
        "publication_grade": bool(config["publication_grade"]),
        "design_provenance": config["design_provenance"],
        "units": config["units"],
        "model": {
            "size": model.size,
            "detectors": model.num_detectors,
            "event_probability": model.event_probability,
            "readout_error": model.readout_error,
            "q0": q0,
        },
        "cycle_fair_budgets": {
            "calibration_cycles_per_family": int(config["calibration_cycles"]),
            "change_time_cycles": int(config["change_time_cycles"]),
            "post_change_horizon_cycles": int(config["post_change_horizon_cycles"]),
            "null_horizon_cycles": int(config["null_horizon_cycles"]),
            "ridge_validation_streams": int(config["validation_streams"]),
            "ridge_validation_horizon_cycles": int(
                config["ridge_validation_horizon_cycles"]
            ),
            "ridge_validation_total_alternative_cycles": (
                int(config["validation_streams"])
                * int(config["ridge_validation_horizon_cycles"])
            ),
            "target_arl_cycles": int(config["target_arl_cycles"]),
            "detector_threshold_updates": config["detector_threshold_updates"],
        },
        "validation_trained_baseline": {
            "path": baseline_path.name,
            "method": VALIDATION_BASELINE_METHOD,
            "physical_cycles_per_class_per_family": int(
                config["validation_trained_baseline"]["total_physical_cycles_per_class"]
            ),
            "cross_family_sample_efficiency_claim": False,
        },
        "selected_ridge": {
            "spatial": selected_spatial_ridge,
            "temporal": selected_temporal_ridge,
        },
        "target_spatial_q": middle_q,
        "target_temporal_kappa": middle_kappa,
        "target_spatial_restart_mean_delay_cycles": target_spatial[
            "mean_restart_delay_cycles"
        ].to_dict(),
        "target_temporal_restart_mean_delay_cycles": target_temporal[
            "mean_restart_delay_cycles"
        ].to_dict(),
        "target_spatial_delay_reduction_vs_no_change_cycles": target_spatial[
            "restart_delay_reduction_vs_no_change_cycles"
        ].to_dict(),
        "target_temporal_delay_reduction_vs_no_change_cycles": target_temporal[
            "restart_delay_reduction_vs_no_change_cycles"
        ].to_dict(),
        "primary_named_comparator_audit": {
            "path": comparator_path.name,
            "hypotheses": comparator_audit.to_dict(orient="records"),
            "overall_two_hypothesis_comparison_supported": bool(
                comparator_audit["overall_two_hypothesis_comparison_supported"].iloc[0]
            ),
            "scope": (
                "Only vAOC versus the named predeclared logistic comparator "
                "on the two declared controlled tasks."
            ),
        },
        "evaluation_note": (
            "Primary delays restart detectors at the changepoint on the same "
            "post-change segment. Surveillance delays are reported separately "
            "conditional on no pre-change alarm; null rows retain the ARL and "
            "censoring audit."
        ),
        "scaling_arm": (
            None
            if scaling_path is None
            else {
                "path": scaling_path.name,
                "sizes": [int(value) for value in config["scaling_sizes"]],
                "repetitions": int(config["scaling_repetitions"]),
                "q1": middle_q,
                "ridge_frozen_from_size5": selected_spatial_ridge,
                "timing_samples": 10_000,
            }
        ),
        "claim_boundary": config["claim_boundary"],
    }
    summary_path = results / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    outputs = [
        baseline_path,
        validation_path,
        metrics_path,
        aggregate_path,
        comparisons_path,
        comparator_path,
        pdf_path,
        png_path,
        summary_path,
    ]
    if scaling_path is not None:
        outputs.append(scaling_path)
    write_manifest(
        results / "manifest.json",
        experiment=f"run5-syndrome-drift-{result_name}",
        started_at=started,
        config=config,
        outputs=outputs,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
