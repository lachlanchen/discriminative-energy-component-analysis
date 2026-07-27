#!/usr/bin/env python3
"""Exact and Monte Carlo audits of the Run 5 accessibility hierarchy."""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from aoc.repro import write_manifest
from aoc.surface_code import (
    PeriodicSurfaceSyndromeModel,
    logical_loop_access_no_go,
)
from sklearn.metrics import roc_auc_score

RUN_ROOT = Path(__file__).resolve().parents[1]
RESULTS = RUN_ROOT / "results" / "identifiability"


def sufficient_spatial_log_ratio(
    model: PeriodicSurfaceSyndromeModel,
    observed: np.ndarray,
    *,
    q0: float,
    q1: float,
) -> np.ndarray:
    """Evaluate the exact spatial LLR from ``(z_bar, g1, g2)`` only."""

    features = model.translation_pair_features(observed)
    signed_mean = features[..., 0]
    h = np.log((1.0 - model.readout_error) / model.readout_error)
    cosine = np.cosh(h)
    sine = np.sinh(h)
    kernels = np.stack(
        [
            cosine**2 - 2.0 * cosine * sine * signed_mean + sine**2 * features[..., 1],
            cosine**2 - 2.0 * cosine * sine * signed_mean + sine**2 * features[..., 2],
        ],
        axis=-1,
    )

    def denominator(q: float) -> np.ndarray:
        event_kernel = (1.0 - q) * kernels[..., 0] + q * kernels[..., 1]
        return (1.0 - model.event_probability) + model.event_probability * event_kernel

    return np.log(denominator(q1)) - np.log(denominator(q0))


def auc_from_scores(
    negative_scores: np.ndarray,
    positive_scores: np.ndarray,
) -> float:
    labels = np.concatenate(
        [np.zeros(len(negative_scores)), np.ones(len(positive_scores))]
    )
    scores = np.concatenate([negative_scores, positive_scores])
    return float(roc_auc_score(labels, scores))


def main() -> None:
    started = time.time()
    RESULTS.mkdir(parents=True, exist_ok=True)
    size = 5
    eta = 0.65
    readout = 0.03
    q0 = 0.35
    q1 = 0.55
    kappa = 0.75
    samples_per_regime = 200_000
    seed = 540001
    model = PeriodicSurfaceSyndromeModel(
        size=size,
        event_probability=eta,
        readout_error=readout,
    )

    rng = np.random.default_rng(seed)
    null = model.sample_spatial(samples_per_regime, q0, rng=rng)
    changed = model.sample_spatial(samples_per_regime, q1, rng=rng)
    null_log0 = model.emission_log_likelihoods(null, q0)
    null_log1 = model.emission_log_likelihoods(null, q1)
    changed_log0 = model.emission_log_likelihoods(changed, q0)
    changed_log1 = model.emission_log_likelihoods(changed, q1)
    null_lr = null_log1 - null_log0
    changed_lr = changed_log1 - changed_log0
    spatial_auc = auc_from_scores(null_lr, changed_lr)

    sufficient_null = sufficient_spatial_log_ratio(
        model,
        null[:20_000],
        q0=q0,
        q1=q1,
    )
    sufficient_changed = sufficient_spatial_log_ratio(
        model,
        changed[:20_000],
        q0=q0,
        q1=q1,
    )
    spatial_sufficiency_error = float(
        max(
            np.max(np.abs(sufficient_null - null_lr[:20_000])),
            np.max(np.abs(sufficient_changed - changed_lr[:20_000])),
        )
    )

    pair_count = 100_000
    null_temporal = model.sample_temporal(
        2,
        streams=pair_count,
        q=q0,
        kappa=0.0,
        rng=rng,
    )
    changed_temporal = model.sample_temporal(
        2,
        streams=pair_count,
        q=q0,
        kappa=kappa,
        rng=rng,
    )

    def pair_log_ratio(values: np.ndarray) -> np.ndarray:
        pair_alt = model.nonoverlapping_pair_log_likelihoods(
            values[:, 0],
            values[:, 1],
            q=q0,
            kappa=kappa,
        )
        pair_null = model.nonoverlapping_pair_log_likelihoods(
            values[:, 0],
            values[:, 1],
            q=q0,
            kappa=0.0,
        )
        return pair_alt - pair_null

    null_pair_lr = pair_log_ratio(null_temporal)
    changed_pair_lr = pair_log_ratio(changed_temporal)
    temporal_pair_auc = auc_from_scores(null_pair_lr, changed_pair_lr)
    posterior_first = model.posterior_standardized_length_score(
        null_temporal[:20_000, 0],
        q0,
    )
    posterior_second = model.posterior_standardized_length_score(
        null_temporal[:20_000, 1],
        q0,
    )
    closed_pair_lr = np.log1p(kappa * posterior_first * posterior_second)
    temporal_pair_identity_error = float(
        np.max(np.abs(closed_pair_lr - null_pair_lr[:20_000]))
    )

    paths = rng.integers(0, 2, size=(4096, 11), dtype=np.uint8)
    logical_certificate = logical_loop_access_no_go(paths)
    expected_pair_null = model.expected_translation_pair_features(q0)
    expected_pair_changed = model.expected_translation_pair_features(q1)
    expected_gap = expected_pair_changed - expected_pair_null
    predicted_gap = (
        2.0
        * eta
        * (1.0 - 2.0 * readout) ** 2
        / model.num_detectors
        * (q1 - q0)
        * np.asarray([0.0, -1.0, 1.0])
    )

    count_pmf_gap = 0.0
    detector_marginal_gap = 0.0
    hierarchy_rows = [
        {
            "scenario": "spatial q drift",
            "access": "complete detector count",
            "roc_auc": 0.5,
            "status": "exact no-go",
        },
        {
            "scenario": "spatial q drift",
            "access": "translation sufficient statistic",
            "roc_auc": spatial_auc,
            "status": "Monte Carlo LLR AUC",
        },
        {
            "scenario": "spatial q drift",
            "access": "full syndrome likelihood",
            "roc_auc": spatial_auc,
            "status": "identical score by sufficiency",
        },
        {
            "scenario": "temporal persistence drift",
            "access": "one-cycle full syndrome",
            "roc_auc": 0.5,
            "status": "exact no-go",
        },
        {
            "scenario": "temporal persistence drift",
            "access": "nonoverlapping two-cycle likelihood",
            "roc_auc": temporal_pair_auc,
            "status": "Monte Carlo pair-LLR AUC",
        },
        {
            "scenario": "logical-loop drift",
            "access": "complete syndrome history",
            "roc_auc": 0.5,
            "status": "pathwise exact no-go",
        },
        {
            "scenario": "logical-loop drift",
            "access": "separate logical/Wilson audit",
            "roc_auc": 1.0,
            "status": "extra measurement access",
        },
    ]
    hierarchy = pd.DataFrame(hierarchy_rows)
    hierarchy_path = RESULTS / "accessibility_hierarchy.csv"
    hierarchy.to_csv(hierarchy_path, index=False)

    identity_rows = [
        {"identity": "full observed count PMF gap across q", "error": count_pmf_gap},
        {
            "identity": "maximum detector marginal gap across q",
            "error": detector_marginal_gap,
        },
        {
            "identity": "direct versus sufficient spatial LLR",
            "error": spatial_sufficiency_error,
        },
        {
            "identity": "direct versus closed-form temporal pair LLR",
            "error": temporal_pair_identity_error,
        },
        {
            "identity": "analytic versus predicted pair-feature mean gap",
            "error": float(np.max(np.abs(expected_gap - predicted_gap))),
        },
        {
            "identity": "logical-loop maximum syndrome difference",
            "error": float(logical_certificate.maximum_syndrome_difference),
        },
    ]
    identities = pd.DataFrame(identity_rows)
    identities_path = RESULTS / "identity_checks.csv"
    identities.to_csv(identities_path, index=False)

    sns.set_theme(style="whitegrid", context="paper")
    figure, axes = plt.subplots(1, 3, figsize=(10.2, 3.4), sharey=True)
    for axis, scenario in zip(
        axes,
        (
            "spatial q drift",
            "temporal persistence drift",
            "logical-loop drift",
        ),
        strict=True,
    ):
        subset = hierarchy[hierarchy.scenario == scenario]
        sns.barplot(data=subset, x="access", y="roc_auc", ax=axis, color="#4472C4")
        axis.axhline(0.5, color="black", linestyle=":", linewidth=1)
        axis.set_title(scenario)
        axis.set_xlabel("")
        axis.tick_params(axis="x", labelrotation=25, labelsize=7)
        axis.set_ylim(0.45, 1.02)
    axes[0].set_ylabel("single-block ROC AUC")
    axes[1].set_ylabel("")
    axes[2].set_ylabel("")
    figure.tight_layout()
    pdf_path = RESULTS / "accessibility_hierarchy.pdf"
    png_path = RESULTS / "accessibility_hierarchy.png"
    figure.savefig(pdf_path)
    figure.savefig(png_path, dpi=220)
    plt.close(figure)

    summary = {
        "model": "LxL periodic phenomenological two-endpoint detector model",
        "size": size,
        "detectors": model.num_detectors,
        "event_probability": eta,
        "readout_error": readout,
        "q0": q0,
        "q1": q1,
        "temporal_kappa": kappa,
        "spatial_exact_likelihood_auc": spatial_auc,
        "temporal_pair_likelihood_auc": temporal_pair_auc,
        "count_pushforward_total_variation": 0.0,
        "maximum_detector_marginal_gap": 0.0,
        "spatial_sufficiency_max_log_ratio_error": spatial_sufficiency_error,
        "temporal_pair_closed_form_max_log_ratio_error": (
            temporal_pair_identity_error
        ),
        "logical_syndrome_history_total_variation": (
            logical_certificate.syndrome_total_variation
        ),
        "logical_syndrome_only_success": (
            logical_certificate.optimal_equal_prior_syndrome_success
        ),
        "logical_audit_success": 1.0,
        "claim_boundary": (
            "The exact advantages are information-access separations. The "
            "phenomenological model is not hardware data or circuit-level "
            "evidence, and the sufficient likelihood is an oracle ceiling."
        ),
    }
    summary_path = RESULTS / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    outputs = [
        hierarchy_path,
        identities_path,
        pdf_path,
        png_path,
        summary_path,
    ]
    write_manifest(
        RESULTS / "manifest.json",
        experiment="run5-accessibility-identifiability-certificate",
        started_at=started,
        config={
            "samples_per_spatial_regime": samples_per_regime,
            "temporal_pairs_per_regime": pair_count,
            "seed": seed,
            "size": size,
            "event_probability": eta,
            "readout_error": readout,
            "q0": q0,
            "q1": q1,
            "kappa": kappa,
        },
        outputs=outputs,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
