#!/usr/bin/env python3
"""Locked offline diagnostic audit on the exact periodic syndrome model."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from aoc.repro import sha256_file, write_manifest
from aoc.surface_code import PeriodicSurfaceSyndromeModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

RUN_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = RUN_ROOT / "configs" / "offline_diagnostic_locked.json"
SPLITS = ("train", "validation", "test")
FEATURE_LABELS = {
    "dfr_count_sequence": "DFR/count sequence",
    "detector_first_moment": "Detector first moment",
    "translation": "Translation statistic / pair lift",
    "symmetry_fourier": "D4 Fourier / pair lift",
}
ESTIMATOR_LABELS = {
    "linear_logistic": "Linear logistic",
    "rbf_svm": "RBF SVM",
    "simplex_positive_support_projector": (
        "Simplex indicator positive-support projector"
    ),
    "positive_part_mean_direction": "Positive-part mean direction",
    "regularized_hotelling": "Regularized Hotelling",
}
POPCOUNT = np.unpackbits(
    np.arange(256, dtype=np.uint8)[:, None],
    axis=1,
).sum(axis=1)


@dataclass(frozen=True)
class HammingMmdWitness:
    """Fixed-prototype RBF-Hamming MMD witness."""

    null_prototypes: np.ndarray
    alternative_prototypes: np.ndarray
    original_bits: int
    gamma: float
    median_hamming_fraction: float
    query_batch_size: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the locked offline Run 5 diagnostic audit."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Committed locked configuration.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    """Load and validate the locked design."""

    config = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "name",
        "locked",
        "publication_grade",
        "output_directory",
        "design_provenance",
        "objective_correction",
        "model",
        "spatial",
        "temporal",
        "split_windows_per_class",
        "sampling_seeds",
        "fixed_method_parameters",
        "metrics",
        "claim_boundary",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Configuration is missing keys: {missing}")
    if config["locked"] is not True:
        raise ValueError("The offline diagnostic design must be locked.")
    correction = config["objective_correction"]
    if correction["simplex_family"] != "symmetry_fourier":
        raise ValueError("Only the D4 Fourier family has simplex semantics.")
    if (
        correction["simplex_estimator"] != "simplex_positive_support_projector"
        or correction["non_simplex_estimator"] != "positive_part_mean_direction"
    ):
        raise ValueError("The locked objective correction changed.")
    model = config["model"]
    if int(model["size"]) != 5:
        raise ValueError("The locked offline audit requires L=5.")
    if float(model["null_chain2_probability"]) != 0.35:
        raise ValueError("The locked null mixture must be q0=0.35.")
    if tuple(config["spatial"]["alternatives"]) != (0.45, 0.55, 0.65):
        raise ValueError("Unexpected spatial alternatives.")
    if tuple(config["temporal"]["alternatives"]) != (0.5, 0.75, 0.9):
        raise ValueError("Unexpected temporal alternatives.")
    if int(config["spatial"]["window_cycles"]) != 64:
        raise ValueError("Spatial windows must contain 64 cycles.")
    if (
        int(config["temporal"]["window_pairs"]) != 64
        or int(config["temporal"]["window_cycles"]) != 128
    ):
        raise ValueError("Temporal windows must contain 64 pairs / 128 cycles.")
    for split in SPLITS:
        if int(config["split_windows_per_class"][split]) <= 0:
            raise ValueError(f"{split} windows must be positive.")
    sampling_seed_values: list[int] = []
    for scenario in ("spatial", "temporal"):
        seed_specification = config["sampling_seeds"][scenario]
        sampling_seed_values.extend(
            int(seed_specification["null"][split]) for split in SPLITS
        )
        for alternatives in seed_specification["alternative"].values():
            sampling_seed_values.extend(int(alternatives[split]) for split in SPLITS)
    if len(sampling_seed_values) != len(set(sampling_seed_values)):
        raise ValueError(
            "Offline train/validation/test sampling seeds must be disjoint."
        )
    return config


def d4_frequency_orbits(size: int) -> tuple[np.ndarray, ...]:
    """Partition square-lattice Fourier modes into D4 symmetry orbits."""

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
        orbits.append(
            np.asarray(
                sorted(first * size + second for first, second in orbit),
                dtype=np.int64,
            )
        )
    return tuple(orbits)


def aggregate_orbits(
    power: np.ndarray,
    orbits: tuple[np.ndarray, ...],
) -> np.ndarray:
    """Sum a Fourier simplex over disjoint D4 orbits."""

    values = np.asarray(power, dtype=np.float64)
    return np.stack(
        [values[..., indices].sum(axis=-1) for indices in orbits],
        axis=-1,
    )


def pair_lift_mean(features: np.ndarray) -> np.ndarray:
    """Average outer products across nonoverlapping adjacent cycles."""

    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 3 or values.shape[1] % 2:
        raise ValueError("Pair lifting requires (windows, even cycles, features).")
    first = values[:, 0::2]
    second = values[:, 1::2]
    lifted = np.einsum("npi,npj->npij", first, second, optimize=True)
    return lifted.mean(axis=1).reshape(len(values), -1)


def extract_window_features(
    model: PeriodicSurfaceSyndromeModel,
    windows: np.ndarray,
    *,
    scenario: str,
    orbits: tuple[np.ndarray, ...],
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Extract every declared feature family from identical windows."""

    observed = np.asarray(windows, dtype=np.uint8)
    if observed.ndim != 3 or observed.shape[2] != model.num_detectors:
        raise ValueError("windows must have shape (windows, cycles, detectors).")
    if scenario not in {"spatial", "temporal"}:
        raise ValueError("scenario must be spatial or temporal.")
    feature_sets: dict[str, np.ndarray] = {}
    runtimes: dict[str, float] = {}

    started = time.perf_counter()
    feature_sets["dfr_count_sequence"] = (
        observed.sum(axis=2, dtype=np.float64) / model.num_detectors
    )
    runtimes["dfr_count_sequence"] = time.perf_counter() - started

    started = time.perf_counter()
    feature_sets["detector_first_moment"] = observed.mean(axis=1)
    runtimes["detector_first_moment"] = time.perf_counter() - started

    flat = observed.reshape(-1, model.num_detectors)
    started = time.perf_counter()
    translation_cycle = model.translation_pair_features(flat).reshape(
        observed.shape[0],
        observed.shape[1],
        3,
    )
    feature_sets["translation"] = (
        translation_cycle.mean(axis=1)
        if scenario == "spatial"
        else pair_lift_mean(translation_cycle)
    )
    runtimes["translation"] = time.perf_counter() - started

    started = time.perf_counter()
    fourier_cycle = aggregate_orbits(
        model.fourier_power_features(flat),
        orbits,
    ).reshape(observed.shape[0], observed.shape[1], len(orbits))
    feature_sets["symmetry_fourier"] = (
        fourier_cycle.mean(axis=1)
        if scenario == "spatial"
        else pair_lift_mean(fourier_cycle)
    )
    runtimes["symmetry_fourier"] = time.perf_counter() - started
    return feature_sets, runtimes


def positive_support_projector(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    tolerance: float = 1e-12,
) -> np.ndarray:
    """Fit the indicator positive-support projector for simplex states."""

    values = np.asarray(features, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.uint8)
    if values.ndim != 2 or len(values) != len(targets):
        raise ValueError("features and labels must have compatible rows.")
    if np.any(values < -tolerance) or not np.allclose(
        values.sum(axis=1),
        1.0,
        atol=1e-9,
    ):
        raise ValueError("The projector is defined here only for simplex states.")
    difference = values[targets == 1].mean(axis=0) - values[targets == 0].mean(axis=0)
    return (difference > tolerance).astype(np.float64)


def positive_part_mean_direction(
    features: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """Fit an L1-normalized positive-part direction on generic features."""

    values = np.asarray(features, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.uint8)
    if values.ndim != 2 or len(values) != len(targets):
        raise ValueError("features and labels must have compatible rows.")
    difference = values[targets == 1].mean(axis=0) - values[targets == 0].mean(axis=0)
    direction = np.maximum(difference, 0.0)
    normalizer = float(direction.sum())
    return direction / normalizer if normalizer > 0.0 else np.zeros_like(direction)


def regularized_hotelling_direction(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    ridge_fraction: float,
    absolute_floor: float,
) -> tuple[np.ndarray, float]:
    """Fit a pooled-covariance Fisher/Hotelling direction."""

    values = np.asarray(features, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.uint8)
    first = values[targets == 0]
    second = values[targets == 1]
    if len(first) < 2 or len(second) < 2:
        raise ValueError("Hotelling fitting needs two samples in each class.")
    delta = second.mean(axis=0) - first.mean(axis=0)
    covariance_first = np.atleast_2d(np.cov(first, rowvar=False))
    covariance_second = np.atleast_2d(np.cov(second, rowvar=False))
    pooled = (
        (len(first) - 1) * covariance_first + (len(second) - 1) * covariance_second
    ) / (len(first) + len(second) - 2)
    average_variance = float(np.trace(pooled) / pooled.shape[0])
    ridge = max(float(absolute_floor), float(ridge_fraction) * average_variance)
    direction = np.linalg.solve(
        pooled + ridge * np.eye(pooled.shape[0]),
        delta,
    )
    norm = float(np.linalg.norm(direction))
    if norm:
        direction = direction / norm
    return np.asarray(direction, dtype=np.float64), ridge


def select_validation_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
) -> tuple[float, float]:
    """Choose a finite threshold maximizing validation balanced accuracy."""

    targets = np.asarray(labels, dtype=np.uint8)
    values = np.asarray(scores, dtype=np.float64)
    if targets.shape != values.shape or targets.ndim != 1:
        raise ValueError("labels and scores must be equal nonempty vectors.")
    if len(values) == 0 or set(np.unique(targets)) != {0, 1}:
        raise ValueError("Both validation classes must be present.")
    if not np.all(np.isfinite(values)):
        raise ValueError("Validation scores must be finite.")
    false_positive, true_positive, thresholds = roc_curve(
        targets,
        values,
        drop_intermediate=False,
    )
    balanced = 0.5 * (true_positive + 1.0 - false_positive)
    best = np.flatnonzero(np.isclose(balanced, balanced.max(), atol=1e-15))
    finite = best[np.isfinite(thresholds[best])]
    chosen = int(finite[0] if len(finite) else best[0])
    threshold = float(
        thresholds[chosen] if np.isfinite(thresholds[chosen]) else values.max()
    )
    return threshold, float(balanced[chosen])


def spatial_window_llr(
    model: PeriodicSurfaceSyndromeModel,
    windows: np.ndarray,
    *,
    q0: float,
    q1: float,
    batch_windows: int,
) -> np.ndarray:
    """Exact known-q window LLR for spatial iid cycles."""

    observed = np.asarray(windows, dtype=np.uint8)
    scores = np.empty(len(observed), dtype=np.float64)
    for start in range(0, len(observed), batch_windows):
        stop = min(start + batch_windows, len(observed))
        batch = observed[start:stop]
        flat = batch.reshape(-1, model.num_detectors)
        increment = model.emission_log_likelihoods(
            flat,
            q1,
        ) - model.emission_log_likelihoods(flat, q0)
        scores[start:stop] = increment.reshape(len(batch), -1).sum(axis=1)
    return scores


def temporal_full_hmm_llr(
    model: PeriodicSurfaceSyndromeModel,
    windows: np.ndarray,
    *,
    q: float,
    kappa: float,
    batch_windows: int,
) -> np.ndarray:
    """Exact full-path HMM LLR against the iid-length null."""

    observed = np.asarray(windows, dtype=np.uint8)
    if observed.ndim != 3 or observed.shape[2] != model.num_detectors:
        raise ValueError("windows must have shape (windows, cycles, detectors).")
    stationary = np.asarray([1.0 - q, q], dtype=np.float64)
    transition = model.length_transition_matrix(q, kappa)
    log_stationary = np.log(stationary)
    log_transition = np.log(transition)
    scores = np.empty(len(observed), dtype=np.float64)
    for start in range(0, len(observed), batch_windows):
        stop = min(start + batch_windows, len(observed))
        batch = observed[start:stop]
        flat = batch.reshape(-1, model.num_detectors)
        emission = np.stack(
            [
                model.conditional_length_emission_log_likelihoods(flat, 1),
                model.conditional_length_emission_log_likelihoods(flat, 2),
            ],
            axis=1,
        ).reshape(len(batch), batch.shape[1], 2)
        null_log = model.emission_log_likelihoods(flat, q).reshape(
            len(batch),
            batch.shape[1],
        )
        joint = emission[:, 0] + log_stationary
        cycle_log = np.logaddexp(joint[:, 0], joint[:, 1])
        log_filter = joint - cycle_log[:, None]
        alternative_log = cycle_log.copy()
        for cycle in range(1, batch.shape[1]):
            predicted_first = np.logaddexp(
                log_filter[:, 0] + log_transition[0, 0],
                log_filter[:, 1] + log_transition[1, 0],
            )
            predicted_second = np.logaddexp(
                log_filter[:, 0] + log_transition[0, 1],
                log_filter[:, 1] + log_transition[1, 1],
            )
            joint = emission[:, cycle] + np.stack(
                [predicted_first, predicted_second],
                axis=1,
            )
            cycle_log = np.logaddexp(joint[:, 0], joint[:, 1])
            log_filter = joint - cycle_log[:, None]
            alternative_log += cycle_log
        scores[start:stop] = alternative_log - null_log.sum(axis=1)
    return scores


def normalized_hamming_matrix(
    packed_queries: np.ndarray,
    packed_references: np.ndarray,
    *,
    original_bits: int,
    batch_size: int,
) -> np.ndarray:
    """Compute normalized Hamming distances between packed binary rows."""

    queries = np.asarray(packed_queries, dtype=np.uint8)
    references = np.asarray(packed_references, dtype=np.uint8)
    if queries.ndim != 2 or references.ndim != 2:
        raise ValueError("Packed observations must be row matrices.")
    if queries.shape[1] != references.shape[1] or original_bits <= 0:
        raise ValueError("Packed dimensions or original bit count are invalid.")
    distances = np.empty((len(queries), len(references)), dtype=np.float64)
    for start in range(0, len(queries), batch_size):
        stop = min(start + batch_size, len(queries))
        xor = np.bitwise_xor(
            queries[start:stop, None, :],
            references[None, :, :],
        )
        distances[start:stop] = (
            POPCOUNT[xor].sum(
                axis=2,
                dtype=np.int32,
            )
            / original_bits
        )
    return distances


def fit_hamming_mmd_witness(
    null_windows: np.ndarray,
    alternative_windows: np.ndarray,
    *,
    prototypes_per_class: int,
    query_batch_size: int,
    seed: int,
) -> HammingMmdWitness:
    """Fit a fixed-prototype RBF-Hamming MMD witness on training windows."""

    null_values = np.asarray(null_windows, dtype=np.uint8)
    alternative_values = np.asarray(alternative_windows, dtype=np.uint8)
    if null_values.shape[1:] != alternative_values.shape[1:]:
        raise ValueError("MMD training windows must have equal observation shapes.")
    if prototypes_per_class > min(len(null_values), len(alternative_values)):
        raise ValueError("The prototype budget exceeds a class size.")
    rng = np.random.default_rng(seed)
    null_indices = rng.choice(
        len(null_values),
        size=prototypes_per_class,
        replace=False,
    )
    alternative_indices = rng.choice(
        len(alternative_values),
        size=prototypes_per_class,
        replace=False,
    )
    original_bits = int(np.prod(null_values.shape[1:]))
    null_packed = np.packbits(
        null_values[null_indices].reshape(prototypes_per_class, -1),
        axis=1,
    )
    alternative_packed = np.packbits(
        alternative_values[alternative_indices].reshape(
            prototypes_per_class,
            -1,
        ),
        axis=1,
    )
    combined = np.concatenate([null_packed, alternative_packed])
    distances = normalized_hamming_matrix(
        combined,
        combined,
        original_bits=original_bits,
        batch_size=query_batch_size,
    )
    upper = distances[np.triu_indices(len(combined), k=1)]
    positive = upper[upper > 0.0]
    median = float(np.median(positive)) if len(positive) else 1.0
    gamma = float(np.log(2.0) / median)
    return HammingMmdWitness(
        null_prototypes=null_packed,
        alternative_prototypes=alternative_packed,
        original_bits=original_bits,
        gamma=gamma,
        median_hamming_fraction=median,
        query_batch_size=query_batch_size,
    )


def score_hamming_mmd(
    witness: HammingMmdWitness,
    windows: np.ndarray,
) -> np.ndarray:
    """Evaluate the alternative-minus-null kernel mean witness."""

    values = np.asarray(windows, dtype=np.uint8)
    packed = np.packbits(values.reshape(len(values), -1), axis=1)
    null_distance = normalized_hamming_matrix(
        packed,
        witness.null_prototypes,
        original_bits=witness.original_bits,
        batch_size=witness.query_batch_size,
    )
    alternative_distance = normalized_hamming_matrix(
        packed,
        witness.alternative_prototypes,
        original_bits=witness.original_bits,
        batch_size=witness.query_batch_size,
    )
    return np.exp(-witness.gamma * alternative_distance).mean(axis=1) - np.exp(
        -witness.gamma * null_distance
    ).mean(axis=1)


def sample_windows(
    model: PeriodicSurfaceSyndromeModel,
    *,
    scenario: str,
    effect: float,
    windows: int,
    cycles: int,
    q0: float,
    seed: int,
) -> np.ndarray:
    """Sample independent windows for one class and split."""

    if scenario == "spatial":
        return model.sample_spatial(
            windows * cycles,
            effect,
            seed=seed,
        ).reshape(windows, cycles, model.num_detectors)
    if scenario == "temporal":
        return model.sample_temporal(
            cycles,
            q=q0,
            kappa=effect,
            streams=windows,
            seed=seed,
        )
    raise ValueError("scenario must be spatial or temporal.")


def evaluate_scores(
    validation_labels: np.ndarray,
    validation_scores: np.ndarray,
    test_labels: np.ndarray,
    test_scores: np.ndarray,
) -> dict[str, float | int]:
    """Evaluate held-out scores with a validation-selected threshold."""

    threshold, validation_balanced = select_validation_threshold(
        validation_labels,
        validation_scores,
    )
    predictions = np.asarray(test_scores >= threshold, dtype=np.uint8)
    true_negative, false_positive, false_negative, true_positive = confusion_matrix(
        test_labels, predictions, labels=[0, 1]
    ).ravel()
    return {
        "validation_threshold": threshold,
        "validation_roc_auc": float(
            roc_auc_score(validation_labels, validation_scores)
        ),
        "validation_balanced_accuracy": validation_balanced,
        "test_roc_auc": float(roc_auc_score(test_labels, test_scores)),
        "test_balanced_accuracy": float(
            balanced_accuracy_score(test_labels, predictions)
        ),
        "test_true_negative": int(true_negative),
        "test_false_positive": int(false_positive),
        "test_false_negative": int(false_negative),
        "test_true_positive": int(true_positive),
        "test_true_negative_rate": float(
            true_negative / (true_negative + false_positive)
        ),
        "test_true_positive_rate": float(
            true_positive / (true_positive + false_negative)
        ),
    }


def record_method_result(
    *,
    metric_rows: list[dict[str, Any]],
    score_frames: list[pd.DataFrame],
    scenario: str,
    effect: float,
    labels: dict[str, np.ndarray],
    sample_ids: dict[str, np.ndarray],
    family: str,
    estimator: str,
    method_label: str,
    parameters: dict[str, Any],
    fit_seconds: float,
    validation_seconds: float,
    test_seconds: float,
    validation_scores: np.ndarray,
    test_scores: np.ndarray,
    feature_dimension: int,
) -> None:
    """Record aggregate metrics and paired per-window test scores."""

    evaluation = evaluate_scores(
        labels["validation"],
        validation_scores,
        labels["test"],
        test_scores,
    )
    method_id = f"{family}::{estimator}"
    metric_rows.append(
        {
            "scenario": scenario,
            "effect": effect,
            "method_id": method_id,
            "method_label": method_label,
            "feature_family": family,
            "estimator": estimator,
            "feature_dimension": feature_dimension,
            "train_windows": len(labels["train"]),
            "validation_windows": len(labels["validation"]),
            "test_windows": len(labels["test"]),
            "fixed_parameters": json.dumps(parameters, sort_keys=True),
            "fit_seconds": fit_seconds,
            "validation_score_seconds": validation_seconds,
            "test_score_seconds": test_seconds,
            **evaluation,
        }
    )
    predictions = np.asarray(
        test_scores >= float(evaluation["validation_threshold"]),
        dtype=np.uint8,
    )
    score_frames.append(
        pd.DataFrame(
            {
                "scenario": scenario,
                "effect": effect,
                "sample_id": sample_ids["test"],
                "label": labels["test"],
                "method_id": method_id,
                "score": test_scores,
                "validation_threshold": evaluation["validation_threshold"],
                "prediction": predictions,
            }
        )
    )


def make_figure(
    metrics: pd.DataFrame,
    *,
    png_path: Path,
    pdf_path: Path,
) -> None:
    """Plot complete AUC and balanced-accuracy heatmaps."""

    sns.set_theme(style="white", context="paper")
    method_order = list(dict.fromkeys(metrics["method_label"]))
    figure, axes = plt.subplots(2, 2, figsize=(12.8, 15.5))
    for column, scenario in enumerate(("spatial", "temporal")):
        subset = metrics[metrics["scenario"] == scenario]
        for row, (metric, title) in enumerate(
            (
                ("test_roc_auc", "Test ROC AUC"),
                (
                    "test_balanced_accuracy",
                    "Test balanced accuracy at validation threshold",
                ),
            )
        ):
            table = subset.pivot(
                index="method_label",
                columns="effect",
                values=metric,
            ).reindex(method_order)
            table = table.dropna(how="all")
            sns.heatmap(
                table,
                ax=axes[row, column],
                annot=True,
                fmt=".3f",
                cmap="viridis",
                vmin=0.45,
                vmax=1.0,
                linewidths=0.3,
                cbar_kws={"label": metric.replace("_", " ")},
                annot_kws={"fontsize": 7},
            )
            axes[row, column].set_title(
                f"{scenario.capitalize()}: {title}",
                fontsize=11,
            )
            axes[row, column].set_xlabel(
                "q alternative" if scenario == "spatial" else "κ alternative"
            )
            axes[row, column].set_ylabel("")
    figure.suptitle(
        "Run 5 locked offline diagnostic audit",
        fontsize=14,
        y=0.995,
    )
    figure.tight_layout()
    figure.savefig(png_path, dpi=240, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    started = time.time()
    config_path = args.config.resolve()
    config = load_config(config_path)
    results = RUN_ROOT / "results" / str(config["output_directory"])
    results.mkdir(parents=True, exist_ok=True)
    model_config = config["model"]
    model = PeriodicSurfaceSyndromeModel(
        size=int(model_config["size"]),
        event_probability=float(model_config["event_probability"]),
        readout_error=float(model_config["readout_error"]),
    )
    q0 = float(model_config["null_chain2_probability"])
    split_sizes = {
        split: int(config["split_windows_per_class"][split]) for split in SPLITS
    }
    fixed = config["fixed_method_parameters"]
    orbits = d4_frequency_orbits(model.size)
    metric_rows: list[dict[str, Any]] = []
    feature_runtime_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    score_frames: list[pd.DataFrame] = []

    for scenario_index, scenario in enumerate(("spatial", "temporal")):
        scenario_config = config[scenario]
        cycles = int(scenario_config["window_cycles"])
        null_effect = q0 if scenario == "spatial" else 0.0
        null_windows: dict[str, np.ndarray] = {}
        for split in SPLITS:
            seed = int(config["sampling_seeds"][scenario]["null"][split])
            null_windows[split] = sample_windows(
                model,
                scenario=scenario,
                effect=null_effect,
                windows=split_sizes[split],
                cycles=cycles,
                q0=q0,
                seed=seed,
            )
            sample_rows.append(
                {
                    "scenario": scenario,
                    "effect": null_effect,
                    "class": "null",
                    "split": split,
                    "windows": split_sizes[split],
                    "cycles_per_window": cycles,
                    "sampling_seed": seed,
                    "reused_across_alternatives": True,
                }
            )

        for effect_raw in scenario_config["alternatives"]:
            effect = float(effect_raw)
            effect_key = str(effect_raw)
            alternative_windows: dict[str, np.ndarray] = {}
            combined_windows: dict[str, np.ndarray] = {}
            labels: dict[str, np.ndarray] = {}
            sample_ids: dict[str, np.ndarray] = {}
            for split in SPLITS:
                seed = int(
                    config["sampling_seeds"][scenario]["alternative"][effect_key][split]
                )
                alternative_windows[split] = sample_windows(
                    model,
                    scenario=scenario,
                    effect=effect,
                    windows=split_sizes[split],
                    cycles=cycles,
                    q0=q0,
                    seed=seed,
                )
                combined_windows[split] = np.concatenate(
                    [null_windows[split], alternative_windows[split]],
                    axis=0,
                )
                labels[split] = np.concatenate(
                    [
                        np.zeros(split_sizes[split], dtype=np.uint8),
                        np.ones(split_sizes[split], dtype=np.uint8),
                    ]
                )
                sample_ids[split] = np.asarray(
                    [
                        f"{scenario}:{effect_key}:{split}:null:{index}"
                        for index in range(split_sizes[split])
                    ]
                    + [
                        f"{scenario}:{effect_key}:{split}:alternative:{index}"
                        for index in range(split_sizes[split])
                    ],
                    dtype=object,
                )
                sample_rows.append(
                    {
                        "scenario": scenario,
                        "effect": effect,
                        "class": "alternative",
                        "split": split,
                        "windows": split_sizes[split],
                        "cycles_per_window": cycles,
                        "sampling_seed": seed,
                        "reused_across_alternatives": False,
                    }
                )

            features: dict[str, dict[str, np.ndarray]] = {}
            for split in SPLITS:
                split_features, split_runtimes = extract_window_features(
                    model,
                    combined_windows[split],
                    scenario=scenario,
                    orbits=orbits,
                )
                features[split] = split_features
                for family, seconds in split_runtimes.items():
                    feature_runtime_rows.append(
                        {
                            "scenario": scenario,
                            "effect": effect,
                            "split": split,
                            "feature_family": family,
                            "feature_label": FEATURE_LABELS[family],
                            "windows": len(combined_windows[split]),
                            "cycles_per_window": cycles,
                            "feature_dimension": split_features[family].shape[1],
                            "extraction_seconds": seconds,
                            "seconds_per_1000_windows": (
                                seconds * 1000.0 / len(combined_windows[split])
                            ),
                        }
                    )

            record_method = partial(
                record_method_result,
                metric_rows=metric_rows,
                score_frames=score_frames,
                scenario=scenario,
                effect=effect,
                labels=labels,
                sample_ids=sample_ids,
            )

            for family, feature_label in FEATURE_LABELS.items():
                train_features = features["train"][family]
                validation_features = features["validation"][family]
                test_features = features["test"][family]
                dimension = train_features.shape[1]

                fit_started = time.perf_counter()
                logistic = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(
                        C=float(fixed["logistic_c"]),
                        max_iter=int(fixed["logistic_max_iter"]),
                        solver="lbfgs",
                    ),
                )
                logistic.fit(train_features, labels["train"])
                fit_seconds = time.perf_counter() - fit_started
                score_started = time.perf_counter()
                validation_scores = logistic.decision_function(validation_features)
                validation_seconds = time.perf_counter() - score_started
                score_started = time.perf_counter()
                test_scores = logistic.decision_function(test_features)
                test_seconds = time.perf_counter() - score_started
                record_method(
                    family=family,
                    estimator="linear_logistic",
                    method_label=(
                        f"{feature_label} · {ESTIMATOR_LABELS['linear_logistic']}"
                    ),
                    parameters={
                        "C": float(fixed["logistic_c"]),
                        "standardized": True,
                    },
                    fit_seconds=fit_seconds,
                    validation_seconds=validation_seconds,
                    test_seconds=test_seconds,
                    validation_scores=validation_scores,
                    test_scores=test_scores,
                    feature_dimension=dimension,
                )

                fit_started = time.perf_counter()
                rbf = make_pipeline(
                    StandardScaler(),
                    SVC(
                        C=float(fixed["rbf_svm_c"]),
                        gamma=fixed["rbf_svm_gamma"],
                        kernel="rbf",
                        cache_size=512,
                    ),
                )
                rbf.fit(train_features, labels["train"])
                fit_seconds = time.perf_counter() - fit_started
                score_started = time.perf_counter()
                validation_scores = rbf.decision_function(validation_features)
                validation_seconds = time.perf_counter() - score_started
                score_started = time.perf_counter()
                test_scores = rbf.decision_function(test_features)
                test_seconds = time.perf_counter() - score_started
                record_method(
                    family=family,
                    estimator="rbf_svm",
                    method_label=(f"{feature_label} · {ESTIMATOR_LABELS['rbf_svm']}"),
                    parameters={
                        "C": float(fixed["rbf_svm_c"]),
                        "gamma": fixed["rbf_svm_gamma"],
                        "standardized": True,
                    },
                    fit_seconds=fit_seconds,
                    validation_seconds=validation_seconds,
                    test_seconds=test_seconds,
                    validation_scores=validation_scores,
                    test_scores=test_scores,
                    feature_dimension=dimension,
                )

                fit_started = time.perf_counter()
                if family == "symmetry_fourier":
                    contrast_estimator = "simplex_positive_support_projector"
                    contrast_direction = positive_support_projector(
                        train_features,
                        labels["train"],
                    )
                    contrast_parameters = {
                        "positive_support_size": int(
                            np.count_nonzero(contrast_direction)
                        ),
                        "indicator_effect": True,
                        "simplex_operator_interpretation": True,
                    }
                else:
                    contrast_estimator = "positive_part_mean_direction"
                    contrast_direction = positive_part_mean_direction(
                        train_features,
                        labels["train"],
                    )
                    contrast_parameters = {
                        "positive_support_size": int(
                            np.count_nonzero(contrast_direction)
                        ),
                        "l1_normalized": True,
                        "simplex_operator_interpretation": False,
                    }
                fit_seconds = time.perf_counter() - fit_started
                score_started = time.perf_counter()
                validation_scores = validation_features @ contrast_direction
                validation_seconds = time.perf_counter() - score_started
                score_started = time.perf_counter()
                test_scores = test_features @ contrast_direction
                test_seconds = time.perf_counter() - score_started
                record_method(
                    family=family,
                    estimator=contrast_estimator,
                    method_label=(
                        f"{feature_label} · {ESTIMATOR_LABELS[contrast_estimator]}"
                    ),
                    parameters=contrast_parameters,
                    fit_seconds=fit_seconds,
                    validation_seconds=validation_seconds,
                    test_seconds=test_seconds,
                    validation_scores=validation_scores,
                    test_scores=test_scores,
                    feature_dimension=dimension,
                )

                fit_started = time.perf_counter()
                hotelling_direction, ridge = regularized_hotelling_direction(
                    train_features,
                    labels["train"],
                    ridge_fraction=float(fixed["hotelling_ridge_fraction"]),
                    absolute_floor=float(fixed["hotelling_absolute_floor"]),
                )
                fit_seconds = time.perf_counter() - fit_started
                score_started = time.perf_counter()
                validation_scores = validation_features @ hotelling_direction
                validation_seconds = time.perf_counter() - score_started
                score_started = time.perf_counter()
                test_scores = test_features @ hotelling_direction
                test_seconds = time.perf_counter() - score_started
                record_method(
                    family=family,
                    estimator="regularized_hotelling",
                    method_label=(
                        f"{feature_label} · {ESTIMATOR_LABELS['regularized_hotelling']}"
                    ),
                    parameters={
                        "ridge_fraction": float(fixed["hotelling_ridge_fraction"]),
                        "absolute_floor": float(fixed["hotelling_absolute_floor"]),
                        "realized_ridge": ridge,
                        "l2_normalized": True,
                    },
                    fit_seconds=fit_seconds,
                    validation_seconds=validation_seconds,
                    test_seconds=test_seconds,
                    validation_scores=validation_scores,
                    test_scores=test_scores,
                    feature_dimension=dimension,
                )

            mmd_seed = int(
                fixed["mmd_seed_base"] + 10_000 * scenario_index + round(1000 * effect)
            )
            fit_started = time.perf_counter()
            witness = fit_hamming_mmd_witness(
                null_windows["train"],
                alternative_windows["train"],
                prototypes_per_class=int(fixed["mmd_prototypes_per_class"]),
                query_batch_size=int(fixed["mmd_query_batch_size"]),
                seed=mmd_seed,
            )
            fit_seconds = time.perf_counter() - fit_started
            score_started = time.perf_counter()
            validation_scores = score_hamming_mmd(
                witness,
                combined_windows["validation"],
            )
            validation_seconds = time.perf_counter() - score_started
            score_started = time.perf_counter()
            test_scores = score_hamming_mmd(
                witness,
                combined_windows["test"],
            )
            test_seconds = time.perf_counter() - score_started
            record_method(
                family="raw_binary_window",
                estimator="rbf_hamming_mmd_witness",
                method_label="Raw binary window · RBF-Hamming MMD witness",
                parameters={
                    "prototypes_per_class": int(fixed["mmd_prototypes_per_class"]),
                    "seed": mmd_seed,
                    "median_hamming_fraction": (witness.median_hamming_fraction),
                    "gamma": witness.gamma,
                },
                fit_seconds=fit_seconds,
                validation_seconds=validation_seconds,
                test_seconds=test_seconds,
                validation_scores=validation_scores,
                test_scores=test_scores,
                feature_dimension=int(np.prod(combined_windows["train"].shape[1:])),
            )

            likelihood_batch = int(fixed["likelihood_batch_windows"])
            score_started = time.perf_counter()
            if scenario == "spatial":
                validation_scores = spatial_window_llr(
                    model,
                    combined_windows["validation"],
                    q0=q0,
                    q1=effect,
                    batch_windows=likelihood_batch,
                )
            else:
                validation_scores = temporal_full_hmm_llr(
                    model,
                    combined_windows["validation"],
                    q=q0,
                    kappa=effect,
                    batch_windows=likelihood_batch,
                )
            validation_seconds = time.perf_counter() - score_started
            score_started = time.perf_counter()
            if scenario == "spatial":
                test_scores = spatial_window_llr(
                    model,
                    combined_windows["test"],
                    q0=q0,
                    q1=effect,
                    batch_windows=likelihood_batch,
                )
                likelihood_label = "Exact spatial window LLR ceiling"
            else:
                test_scores = temporal_full_hmm_llr(
                    model,
                    combined_windows["test"],
                    q=q0,
                    kappa=effect,
                    batch_windows=likelihood_batch,
                )
                likelihood_label = "Exact full-HMM window LLR ceiling"
            test_seconds = time.perf_counter() - score_started
            record_method(
                family="full_observation",
                estimator="exact_llr_ceiling",
                method_label=likelihood_label,
                parameters={
                    "known_null": null_effect,
                    "known_alternative": effect,
                    "full_hmm": scenario == "temporal",
                    "batch_windows": likelihood_batch,
                },
                fit_seconds=0.0,
                validation_seconds=validation_seconds,
                test_seconds=test_seconds,
                validation_scores=validation_scores,
                test_scores=test_scores,
                feature_dimension=int(np.prod(combined_windows["test"].shape[1:])),
            )

    metrics = pd.DataFrame(metric_rows)
    feature_runtime = pd.DataFrame(feature_runtime_rows)
    sample_manifest = pd.DataFrame(sample_rows)
    test_scores = pd.concat(score_frames, ignore_index=True)
    metrics_path = results / "metrics.csv"
    feature_path = results / "feature_runtime.csv"
    sample_path = results / "sample_manifest.csv"
    scores_path = results / "test_window_scores.csv.gz"
    png_path = results / "offline_diagnostic.png"
    pdf_path = results / "offline_diagnostic.pdf"
    summary_path = results / "summary.json"
    manifest_path = results / "manifest.json"
    metrics.to_csv(metrics_path, index=False, float_format="%.12g")
    feature_runtime.to_csv(feature_path, index=False, float_format="%.12g")
    sample_manifest.to_csv(sample_path, index=False)
    test_scores.to_csv(
        scores_path,
        index=False,
        float_format="%.12g",
        compression={"method": "gzip", "compresslevel": 9, "mtime": 0},
    )
    make_figure(metrics, png_path=png_path, pdf_path=pdf_path)

    exact = metrics[metrics["estimator"] == "exact_llr_ceiling"]
    non_ceiling = metrics[metrics["estimator"] != "exact_llr_ceiling"]
    validation_selected_rows = (
        non_ceiling.loc[
            non_ceiling.groupby(["scenario", "effect"])["validation_roc_auc"].idxmax()
        ]
        .sort_values(["scenario", "effect"])
        .to_dict(orient="records")
    )
    validation_selected_summary = [
        {
            "scenario": row["scenario"],
            "effect": float(row["effect"]),
            "method": row["method_label"],
            "selection_rule": "largest validation ROC AUC among non-ceilings",
            "validation_roc_auc": float(row["validation_roc_auc"]),
            "test_roc_auc": float(row["test_roc_auc"]),
            "test_balanced_accuracy": float(row["test_balanced_accuracy"]),
        }
        for row in validation_selected_rows
    ]
    exact_summary = [
        {
            "scenario": row["scenario"],
            "effect": float(row["effect"]),
            "method": row["method_label"],
            "test_roc_auc": float(row["test_roc_auc"]),
            "test_balanced_accuracy": float(row["test_balanced_accuracy"]),
        }
        for row in exact.sort_values(["scenario", "effect"]).to_dict(orient="records")
    ]
    count_rows = metrics[metrics["feature_family"] == "dfr_count_sequence"]
    summary = {
        "configuration": config["name"],
        "config_sha256": sha256_file(config_path),
        "design_status": "locked offline diagnostic",
        "design_provenance": config["design_provenance"],
        "objective_correction": config["objective_correction"],
        "publication_grade": bool(config["publication_grade"]),
        "model": config["model"],
        "window_design": {
            "spatial": config["spatial"],
            "temporal": config["temporal"],
            "split_windows_per_class": config["split_windows_per_class"],
        },
        "methods_per_task": int(metrics.groupby(["scenario", "effect"]).size().iloc[0]),
        "tasks": int(metrics.groupby(["scenario", "effect"]).ngroups),
        "mmd_included": True,
        "exact_llr_ceiling": exact_summary,
        "validation_selected_non_ceiling": validation_selected_summary,
        "count_sequence_auc_range": {
            "minimum": float(count_rows["test_roc_auc"].min()),
            "maximum": float(count_rows["test_roc_auc"].max()),
            "interpretation": (
                "These are finite-sample offline results for an exact "
                "count-pushforward no-go feature family."
            ),
        },
        "claim_boundary": config["claim_boundary"],
        "sequential_guarantees_tested": False,
        "test_tuning_performed": False,
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    outputs = [
        metrics_path,
        feature_path,
        sample_path,
        scores_path,
        png_path,
        pdf_path,
        summary_path,
    ]
    write_manifest(
        manifest_path,
        experiment=config["name"],
        started_at=started,
        config=config,
        outputs=outputs,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
