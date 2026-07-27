#!/usr/bin/env python3
"""Locked Stim/PyMatching validation of decoder utility under pair drift."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from statistics import NormalDist
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from aoc.repro import sha256_file, write_manifest
from aoc.surface_code_stim import (
    audit_detector_marginals,
    audit_graphlike_dem,
    build_rotated_memory_z_circuit,
    decode_stale_matched_correlated,
    final_pair_joint_probability,
    sample_detector_observables,
    select_safe_disjoint_data_pairs,
)

RUN_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = RUN_ROOT / "configs" / "circuit_level_locked.json"
DECODER_LABELS = {
    "static_null": "Static null DEM",
    "ordinary_post": "Ordinary post DEM",
    "correlation_aware": "Correlation-aware post DEM",
}
DECODER_ORDER = tuple(DECODER_LABELS)
REGIME_ORDER = ("reference", "correlated")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Run the locked circuit-level surface-code decoder validation.")
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Committed JSON design (default: circuit_level_locked.json).",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    """Load and validate the declared experimental design."""

    config = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "name",
        "locked",
        "publication_grade",
        "design_provenance",
        "output_directory",
        "generator",
        "distances",
        "shots_per_regime",
        "regimes",
        "confidence",
        "paired_bootstrap_resamples",
        "marginal_audit_tolerance",
        "noise",
        "sampling_seeds",
        "paired_bootstrap_seed_base",
        "decoders",
        "primary_endpoint",
        "claim_boundary",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Configuration is missing keys: {missing}")
    if config["locked"] is not True:
        raise ValueError("Circuit-level publication runs require a locked design.")
    if config["generator"].get("name") != "surface_code:rotated_memory_z":
        raise ValueError("Only the declared rotated-memory-Z generator is supported.")
    if config["generator"].get("rounds") != "distance":
        raise ValueError("This locked design requires rounds equal to distance.")
    distances = tuple(int(value) for value in config["distances"])
    if distances != (3, 5, 7):
        raise ValueError("The locked design requires distances [3, 5, 7].")
    if int(config["shots_per_regime"]) <= 0:
        raise ValueError("shots_per_regime must be positive.")
    if not 0.0 < float(config["confidence"]) < 1.0:
        raise ValueError("confidence must lie in (0, 1).")
    if int(config["paired_bootstrap_resamples"]) < 1:
        raise ValueError("paired_bootstrap_resamples must be positive.")
    decoder_ids = tuple(item.get("id") for item in config["decoders"])
    if decoder_ids != DECODER_ORDER:
        raise ValueError(f"Decoder order must be {DECODER_ORDER}.")
    for distance in distances:
        declared = config["sampling_seeds"].get(str(distance))
        if set(declared or ()) != set(REGIME_ORDER):
            raise ValueError(f"Sampling seeds are incomplete for distance {distance}.")
    return config


def wilson_interval(
    errors: int,
    shots: int,
    confidence: float,
) -> tuple[float, float]:
    """Two-sided Wilson score interval for a binomial error rate."""

    if not 0 <= errors <= shots or shots <= 0:
        raise ValueError("Binomial counts are invalid.")
    z_value = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    rate = errors / shots
    denominator = 1.0 + z_value**2 / shots
    center = (rate + z_value**2 / (2.0 * shots)) / denominator
    radius = (
        z_value
        * np.sqrt(rate * (1.0 - rate) / shots + z_value**2 / (4.0 * shots**2))
        / denominator
    )
    return float(center - radius), float(center + radius)


def logical_error_bits(
    predictions: np.ndarray,
    observables: np.ndarray,
) -> np.ndarray:
    """Return one paired logical-failure indicator per shot."""

    predicted = np.asarray(predictions, dtype=np.uint8)
    actual = np.asarray(observables, dtype=np.uint8)
    if predicted.shape != actual.shape or predicted.ndim != 2:
        raise ValueError("Predictions and observables must be equal row matrices.")
    return np.any(predicted != actual, axis=1)


def paired_bootstrap_difference(
    first: np.ndarray,
    second: np.ndarray,
    *,
    confidence: float,
    resamples: int,
    seed: int,
) -> dict[str, int | float]:
    """Efficient paired bootstrap from the four discordance categories.

    The estimand is ``mean(first) - mean(second)``. Positive values therefore
    mean the first decoder has a higher logical-error rate.
    """

    first_bits = np.asarray(first, dtype=np.bool_)
    second_bits = np.asarray(second, dtype=np.bool_)
    if first_bits.shape != second_bits.shape or first_bits.ndim != 1:
        raise ValueError("Paired logical-error samples must be equal vectors.")
    if len(first_bits) == 0:
        raise ValueError("Paired logical-error samples cannot be empty.")
    categories = 2 * first_bits.astype(np.uint8) + second_bits.astype(np.uint8)
    counts = np.bincount(categories, minlength=4).astype(np.int64)
    probabilities = counts / len(first_bits)
    rng = np.random.default_rng(int(seed))
    bootstrap_counts = rng.multinomial(
        len(first_bits),
        probabilities,
        size=int(resamples),
    )
    bootstrap = (bootstrap_counts[:, 2] - bootstrap_counts[:, 1]) / len(first_bits)
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(bootstrap, (tail, 1.0 - tail))
    return {
        "difference": float((counts[2] - counts[1]) / len(first_bits)),
        "ci_low": float(low),
        "ci_high": float(high),
        "both_correct": int(counts[0]),
        "first_correct_second_error": int(counts[1]),
        "first_error_second_correct": int(counts[2]),
        "both_error": int(counts[3]),
    }


def text_sha256(value: Any) -> str:
    """Hash a Stim circuit or detector error model in its canonical text form."""

    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def decoder_payload(comparison: Any) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Map declared decoder names to paired predictions and solution weights."""

    return {
        "static_null": (
            comparison.stale_predictions,
            comparison.stale_weights,
        ),
        "ordinary_post": (
            comparison.matched_predictions,
            comparison.matched_weights,
        ),
        "correlation_aware": (
            comparison.correlated_predictions,
            comparison.correlated_weights,
        ),
    }


def bootstrap_seed(
    base: int,
    *,
    distance: int,
    regime_index: int,
    comparison_index: int,
) -> int:
    """Deterministically assign a distinct seed to every paired comparison."""

    return int(base + 1000 * distance + 100 * regime_index + comparison_index)


def make_figure(
    logical_rates: pd.DataFrame,
    paired_differences: pd.DataFrame,
    dem_audit: pd.DataFrame,
    *,
    confidence: float,
    tolerance: float,
    png_path: Path,
    pdf_path: Path,
) -> None:
    """Render the predeclared decoder and marginal-audit summary."""

    sns.set_theme(style="whitegrid", context="paper")
    colors = {
        "static_null": "#4C78A8",
        "ordinary_post": "#F58518",
        "correlation_aware": "#54A24B",
    }
    figure, axes = plt.subplots(2, 2, figsize=(10.8, 7.6))
    for axis, regime, title in zip(
        axes[0],
        REGIME_ORDER,
        ("Reference independent channel", "Correlated post-change channel"),
        strict=True,
    ):
        subset = logical_rates[logical_rates["regime"] == regime]
        for decoder in DECODER_ORDER:
            values = subset[subset["decoder"] == decoder].sort_values("distance")
            rates = values["logical_error_rate"].to_numpy()
            axis.errorbar(
                values["distance"],
                rates,
                yerr=np.vstack(
                    [
                        rates - values["ci_low"].to_numpy(),
                        values["ci_high"].to_numpy() - rates,
                    ]
                ),
                marker="o",
                linewidth=1.8,
                capsize=3,
                color=colors[decoder],
                label=DECODER_LABELS[decoder],
            )
        axis.set_title(title)
        axis.set_xlabel("Code distance (rounds = distance)")
        axis.set_ylabel("Logical error rate")
        axis.set_xticks(sorted(subset["distance"].unique()))
        axis.set_ylim(bottom=0.0)
    axes[0, 0].legend(frameon=True, fontsize=8)

    post = paired_differences[
        (paired_differences["regime"] == "correlated")
        & (paired_differences["second_decoder"] == "correlation_aware")
    ]
    difference_axis = axes[1, 0]
    for decoder in ("static_null", "ordinary_post"):
        values = post[post["first_decoder"] == decoder].sort_values("distance")
        differences = values["difference"].to_numpy()
        difference_axis.errorbar(
            values["distance"],
            differences,
            yerr=np.vstack(
                [
                    differences - values["ci_low"].to_numpy(),
                    values["ci_high"].to_numpy() - differences,
                ]
            ),
            marker="o",
            linewidth=1.8,
            capsize=3,
            color=colors[decoder],
            label=f"{DECODER_LABELS[decoder]} − correlation-aware",
        )
    difference_axis.axhline(0.0, color="black", linewidth=1.0)
    difference_axis.set_title(f"Paired excess error ({confidence:.0%} bootstrap CI)")
    difference_axis.set_xlabel("Code distance")
    difference_axis.set_ylabel("Logical-error-rate difference")
    difference_axis.set_xticks(sorted(post["distance"].unique()))
    difference_axis.legend(frameon=True, fontsize=8)

    audit_axis = axes[1, 1]
    audit_values = dem_audit.sort_values("distance")
    gaps = np.maximum(
        audit_values["max_exact_detector_marginal_gap"].to_numpy(),
        np.finfo(float).tiny,
    )
    audit_axis.plot(
        audit_values["distance"],
        gaps,
        marker="o",
        linewidth=1.8,
        color="#B279A2",
        label="Exact DEM gap",
    )
    audit_axis.axhline(
        tolerance,
        color="#E45756",
        linestyle="--",
        label="Declared tolerance",
    )
    audit_axis.set_yscale("log")
    audit_axis.set_title("Marginal-preservation certificate")
    audit_axis.set_xlabel("Code distance")
    audit_axis.set_ylabel("Maximum detector-marginal gap")
    audit_axis.set_xticks(audit_values["distance"])
    audit_axis.legend(frameon=True, fontsize=8)

    figure.suptitle(
        "Run 5 controlled Stim/PyMatching decoder validation",
        fontsize=13,
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

    shots = int(config["shots_per_regime"])
    confidence = float(config["confidence"])
    resamples = int(config["paired_bootstrap_resamples"])
    tolerance = float(config["marginal_audit_tolerance"])
    noise = config["noise"]
    marginal = float(noise["marginal_data_x_error"])
    common = float(noise["common_pair_x_error"])
    common_kwargs = {
        "marginal_data_error": marginal,
        "after_clifford_depolarization": float(noise["after_clifford_depolarization"]),
        "before_measure_flip_probability": float(
            noise["before_measure_flip_probability"]
        ),
        "after_reset_flip_probability": float(noise["after_reset_flip_probability"]),
    }

    logical_rows: list[dict[str, Any]] = []
    difference_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    sampling_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []

    for distance_raw in config["distances"]:
        distance = int(distance_raw)
        rounds = distance
        selection = select_safe_disjoint_data_pairs(distance)
        if not selection.audit.valid:
            raise RuntimeError(f"Unsafe pair selection at distance {distance}.")
        reference = build_rotated_memory_z_circuit(
            distance=distance,
            rounds=rounds,
            common_pair_error=0.0,
            correlated_pairs=selection.pairs,
            **common_kwargs,
        )
        candidate = build_rotated_memory_z_circuit(
            distance=distance,
            rounds=rounds,
            common_pair_error=common,
            correlated_pairs=selection.pairs,
            **common_kwargs,
        )
        reference_dem = reference.circuit.detector_error_model(decompose_errors=True)
        candidate_dem = candidate.circuit.detector_error_model(decompose_errors=True)
        marginal_audit = audit_detector_marginals(
            reference_dem,
            candidate_dem,
            tolerance=tolerance,
        )
        reference_graph = audit_graphlike_dem(reference_dem)
        candidate_graph = audit_graphlike_dem(candidate_dem)
        if not marginal_audit.within_tolerance:
            raise RuntimeError(
                f"Detector-marginal audit failed at distance {distance}: "
                f"{marginal_audit.max_absolute_gap:.3e}."
            )
        if (
            reference_graph.undecomposed_error_count
            or candidate_graph.undecomposed_error_count
        ):
            raise RuntimeError(f"Non-graphlike DEM component at distance {distance}.")

        pair_rows.append(
            {
                "distance": distance,
                "rounds": rounds,
                "data_qubit_count": len(selection.detector_supports),
                "pair_count": len(selection.pairs),
                "unpaired_qubit_count": len(selection.unpaired_qubits),
                "pairs": json.dumps(selection.pairs),
                "unpaired_qubits": json.dumps(selection.unpaired_qubits),
                "pair_support_audit_valid": selection.audit.valid,
            }
        )
        audit_row: dict[str, Any] = {
            "distance": distance,
            "rounds": rounds,
            "num_detectors": int(reference_dem.num_detectors),
            "num_observables": int(reference_dem.num_observables),
            "marginal_data_x_error": marginal,
            "common_pair_x_error": common,
            "residual_independent_x_error": (candidate.residual_error_probability),
            "independent_pair_joint_error": marginal**2,
            "correlated_pair_joint_error": final_pair_joint_probability(
                candidate.residual_error_probability,
                common,
            ),
            "max_exact_detector_marginal_gap": (marginal_audit.max_absolute_gap),
            "marginal_tolerance": tolerance,
            "marginal_audit_passed": marginal_audit.within_tolerance,
            "reference_max_detectors_per_component": (
                reference_graph.max_detectors_per_component
            ),
            "candidate_max_detectors_per_component": (
                candidate_graph.max_detectors_per_component
            ),
            "reference_separator_error_count": (reference_graph.separator_error_count),
            "candidate_separator_error_count": (candidate_graph.separator_error_count),
            "reference_undecomposed_error_count": (
                reference_graph.undecomposed_error_count
            ),
            "candidate_undecomposed_error_count": (
                candidate_graph.undecomposed_error_count
            ),
            "reference_circuit_sha256": text_sha256(reference.circuit),
            "candidate_circuit_sha256": text_sha256(candidate.circuit),
            "reference_dem_sha256": text_sha256(reference_dem),
            "candidate_dem_sha256": text_sha256(candidate_dem),
        }
        empirical_marginals: dict[str, np.ndarray] = {}
        circuits = {
            "reference": reference.circuit,
            "correlated": candidate.circuit,
        }
        exact_marginals = {
            "reference": marginal_audit.reference,
            "correlated": marginal_audit.candidate,
        }
        for regime_index, regime in enumerate(REGIME_ORDER):
            seed = int(config["sampling_seeds"][str(distance)][regime])
            sampled = sample_detector_observables(
                circuits[regime],
                shots=shots,
                seed=seed,
            )
            empirical = sampled.detectors.mean(axis=0)
            empirical_marginals[regime] = empirical
            sampling_rows.append(
                {
                    "distance": distance,
                    "rounds": rounds,
                    "regime": regime,
                    "shots": shots,
                    "sampling_seed": seed,
                    "detector_count": sampled.detectors.shape[1],
                    "observable_count": sampled.observables.shape[1],
                    "mean_detector_event_rate": float(sampled.detectors.mean()),
                    "raw_observable_flip_rate": float(
                        np.mean(np.any(sampled.observables, axis=1))
                    ),
                    "max_empirical_vs_exact_marginal_gap": float(
                        np.max(
                            np.abs(empirical - exact_marginals[regime]),
                            initial=0.0,
                        )
                    ),
                }
            )
            comparison = decode_stale_matched_correlated(
                sampled.detectors,
                sampled.observables,
                reference_model=reference_dem,
                candidate_model=candidate_dem,
            )
            payload = decoder_payload(comparison)
            error_bits: dict[str, np.ndarray] = {}
            for decoder in DECODER_ORDER:
                predictions, weights = payload[decoder]
                errors = logical_error_bits(
                    predictions,
                    sampled.observables,
                )
                error_bits[decoder] = errors
                error_count = int(errors.sum())
                ci_low, ci_high = wilson_interval(
                    error_count,
                    shots,
                    confidence,
                )
                logical_rows.append(
                    {
                        "distance": distance,
                        "rounds": rounds,
                        "regime": regime,
                        "decoder": decoder,
                        "decoder_label": DECODER_LABELS[decoder],
                        "shots": shots,
                        "logical_errors": error_count,
                        "logical_error_rate": error_count / shots,
                        "confidence": confidence,
                        "ci_method": "Wilson score",
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                        "mean_matching_weight": float(np.mean(weights)),
                    }
                )

            comparisons = (
                ("static_null", "ordinary_post"),
                ("static_null", "correlation_aware"),
                ("ordinary_post", "correlation_aware"),
            )
            for comparison_index, (first, second) in enumerate(comparisons):
                assigned_seed = bootstrap_seed(
                    int(config["paired_bootstrap_seed_base"]),
                    distance=distance,
                    regime_index=regime_index,
                    comparison_index=comparison_index,
                )
                uncertainty = paired_bootstrap_difference(
                    error_bits[first],
                    error_bits[second],
                    confidence=confidence,
                    resamples=resamples,
                    seed=assigned_seed,
                )
                difference_rows.append(
                    {
                        "distance": distance,
                        "rounds": rounds,
                        "regime": regime,
                        "first_decoder": first,
                        "second_decoder": second,
                        "estimand": (
                            "logical_error_rate(first)-logical_error_rate(second)"
                        ),
                        "shots": shots,
                        "confidence": confidence,
                        "ci_method": "paired multinomial bootstrap",
                        "bootstrap_resamples": resamples,
                        "bootstrap_seed": assigned_seed,
                        **uncertainty,
                    }
                )
        audit_row["max_empirical_reference_vs_correlated_marginal_gap"] = float(
            np.max(
                np.abs(
                    empirical_marginals["reference"] - empirical_marginals["correlated"]
                ),
                initial=0.0,
            )
        )
        audit_rows.append(audit_row)

    logical_rates = pd.DataFrame(logical_rows)
    paired_differences = pd.DataFrame(difference_rows)
    dem_audit = pd.DataFrame(audit_rows)
    sampling_audit = pd.DataFrame(sampling_rows)
    pair_assignments = pd.DataFrame(pair_rows)

    logical_path = results / "logical_error_rates.csv"
    differences_path = results / "paired_decoder_differences.csv"
    dem_path = results / "dem_marginal_audit.csv"
    sampling_path = results / "sampling_audit.csv"
    pairs_path = results / "pair_assignments.csv"
    png_path = results / "decoder_validation.png"
    pdf_path = results / "decoder_validation.pdf"
    summary_path = results / "summary.json"
    manifest_path = results / "manifest.json"
    logical_rates.to_csv(logical_path, index=False, float_format="%.12g")
    paired_differences.to_csv(
        differences_path,
        index=False,
        float_format="%.12g",
    )
    dem_audit.to_csv(dem_path, index=False, float_format="%.12g")
    sampling_audit.to_csv(sampling_path, index=False, float_format="%.12g")
    pair_assignments.to_csv(pairs_path, index=False)
    make_figure(
        logical_rates,
        paired_differences,
        dem_audit,
        confidence=confidence,
        tolerance=tolerance,
        png_path=png_path,
        pdf_path=pdf_path,
    )

    endpoint = config["primary_endpoint"]
    primary = paired_differences[
        (paired_differences["distance"] == int(endpoint["distance"]))
        & (paired_differences["regime"] == endpoint["regime"])
        & (paired_differences["first_decoder"] == endpoint["first_decoder"])
        & (paired_differences["second_decoder"] == endpoint["second_decoder"])
    ]
    if len(primary) != 1:
        raise RuntimeError("The declared primary endpoint was not produced once.")
    primary_row = primary.iloc[0]
    lower_error_supported = bool(primary_row["ci_low"] > 0.0)

    def regime_rate_table(regime: str) -> list[dict[str, float | int]]:
        regime_rows = logical_rates[logical_rates["regime"] == regime]
        table: list[dict[str, float | int]] = []
        for distance in config["distances"]:
            subset = regime_rows[regime_rows["distance"] == int(distance)]
            distance_row: dict[str, float | int] = {"distance": int(distance)}
            distance_row.update(
                {
                    decoder: float(
                        subset[subset["decoder"] == decoder]["logical_error_rate"].iloc[
                            0
                        ]
                    )
                    for decoder in DECODER_ORDER
                }
            )
            table.append(distance_row)
        return table

    primary_rates = logical_rates[
        (logical_rates["distance"] == int(endpoint["distance"]))
        & (logical_rates["regime"] == endpoint["regime"])
    ]
    first_rate = float(
        primary_rates[primary_rates["decoder"] == endpoint["first_decoder"]][
            "logical_error_rate"
        ].iloc[0]
    )
    second_rate = float(
        primary_rates[primary_rates["decoder"] == endpoint["second_decoder"]][
            "logical_error_rate"
        ].iloc[0]
    )
    summary = {
        "configuration": config["name"],
        "config_sha256": sha256_file(config_path),
        "design_status": "locked controlled simulation",
        "design_provenance": config["design_provenance"],
        "publication_grade": bool(config["publication_grade"]),
        "shots_per_regime_per_distance": shots,
        "total_sampled_shots": shots * len(config["distances"]) * len(REGIME_ORDER),
        "distances": config["distances"],
        "rounds_rule": "rounds = distance",
        "sampled_regimes": config["regimes"],
        "generator": config["generator"],
        "noise": noise,
        "decoders": config["decoders"],
        "exact_dem_marginal_audit": {
            "declared_tolerance": tolerance,
            "all_passed": bool(dem_audit["marginal_audit_passed"].all()),
            "maximum_gap": float(dem_audit["max_exact_detector_marginal_gap"].max()),
        },
        "primary_endpoint": {
            **endpoint,
            "difference": float(primary_row["difference"]),
            "confidence": confidence,
            "ci_method": primary_row["ci_method"],
            "ci_low": float(primary_row["ci_low"]),
            "ci_high": float(primary_row["ci_high"]),
            "first_decoder_error_rate": first_rate,
            "second_decoder_error_rate": second_rate,
            "relative_error_reduction": float((first_rate - second_rate) / first_rate),
            "lower_error_supported_at_declared_interval": lower_error_supported,
        },
        "logical_error_rates_by_regime": {
            regime: regime_rate_table(regime) for regime in REGIME_ORDER
        },
        "claim_boundary": config["claim_boundary"],
        "detector_advantage_tested": False,
        "practical_implication": (
            "The correlation-aware post model improved decoding after the "
            "injected drift but degraded decoding when applied to reference "
            "shots. An operational system therefore needs calibrated model "
            "selection or change detection; that detector is not evaluated "
            "in this circuit-level experiment."
        ),
        "interpretation": (
            "At the predeclared d=7 endpoint, correlation-aware PyMatching "
            "had a lower logical-error rate than ordinary post-DEM matching "
            "within this known-channel controlled simulation."
            if lower_error_supported
            else "The predeclared d=7 endpoint did not resolve a lower "
            "logical-error rate for correlation-aware PyMatching at the "
            "declared paired interval."
        ),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    outputs = [
        logical_path,
        differences_path,
        dem_path,
        sampling_path,
        pairs_path,
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
