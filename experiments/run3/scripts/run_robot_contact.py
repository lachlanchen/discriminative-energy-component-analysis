#!/usr/bin/env python3
"""Learn a contact wrench mode from zero-mean robot residual windows."""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from aoc import AdditiveState, maximum_observable_contrast
from aoc.repro import write_manifest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

RUN_ROOT = Path(__file__).resolve().parents[1]
RESULTS = RUN_ROOT / "results" / "robot_contact"


def contact_screw(normal: np.ndarray, lever: np.ndarray) -> np.ndarray:
    wrench = np.concatenate([normal, np.cross(lever, normal)])
    return wrench / np.linalg.norm(wrench)


def paired_windows(
    covariance: np.ndarray,
    windows: int,
    length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if windows % 2:
        raise ValueError("windows must be even.")
    half = rng.multivariate_normal(
        np.zeros(6),
        covariance,
        size=(windows // 2, length),
    )
    values = np.concatenate([half, -half], axis=0)
    return values[rng.permutation(windows)]


def state_from_samples(samples: np.ndarray) -> np.ndarray:
    normalized = samples / np.linalg.norm(samples, axis=1, keepdims=True)
    return AdditiveState.from_samples(normalized).density


def score_windows(effect: np.ndarray, windows: np.ndarray) -> np.ndarray:
    normalized = windows / np.linalg.norm(windows, axis=2, keepdims=True)
    return np.real(np.einsum("bwi,ij,bwj->bw", normalized, effect, normalized)).mean(
        axis=1
    )


def scalar_accuracy(train, train_y, test, test_y):
    model = LogisticRegression(C=1e3).fit(train.reshape(-1, 1), train_y)
    return accuracy_score(test_y, model.predict(test.reshape(-1, 1)))


def one_repetition(repetition: int, window_length: int):
    rng = np.random.default_rng(730000 + 1000 * repetition + window_length)
    normal = np.array([0.31, -0.42, 0.852])
    normal /= np.linalg.norm(normal)
    lever = np.array([0.08, -0.04, 0.12])
    mode = contact_screw(normal, lever)
    baseline = np.diag([1.0, 1.1, 0.9, 0.55, 0.60, 0.50])
    contact = baseline + 5.0 * np.outer(mode, mode)
    calibration0 = paired_windows(baseline, 300, 48, rng).reshape(-1, 6)
    calibration1 = paired_windows(contact, 300, 48, rng).reshape(-1, 6)
    reference = state_from_samples(calibration0)
    alternative = state_from_samples(calibration1)
    result = maximum_observable_contrast(alternative, reference, rank=1)

    train0 = paired_windows(baseline, 240, window_length, rng)
    train1 = paired_windows(contact, 240, window_length, rng)
    test0 = paired_windows(baseline, 500, window_length, rng)
    test1 = paired_windows(contact, 500, window_length, rng)
    train = np.concatenate([train0, train1])
    test = np.concatenate([test0, test1])
    train_y = np.concatenate([np.zeros(len(train0)), np.ones(len(train1))])
    test_y = np.concatenate([np.zeros(len(test0)), np.ones(len(test1))])
    train_aoc = score_windows(result.effect, train)
    test_aoc = score_windows(result.effect, test)
    train_oracle = np.mean((train @ mode) ** 2, axis=1)
    test_oracle = np.mean((test @ mode) ** 2, axis=1)
    mean_model = LogisticRegression(max_iter=3000).fit(
        train.mean(axis=1),
        train_y,
    )
    mean_probability = mean_model.predict_proba(test.mean(axis=1))[:, 1]
    rows = [
        {
            "repetition": repetition,
            "window_length": window_length,
            "method": "Rank-1 AOC contact witness",
            "accuracy": scalar_accuracy(
                train_aoc,
                train_y,
                test_aoc,
                test_y,
            ),
            "roc_auc": roc_auc_score(test_y, test_aoc),
        },
        {
            "repetition": repetition,
            "window_length": window_length,
            "method": "Oracle contact screw",
            "accuracy": scalar_accuracy(
                train_oracle,
                train_y,
                test_oracle,
                test_y,
            ),
            "roc_auc": roc_auc_score(test_y, test_oracle),
        },
        {
            "repetition": repetition,
            "window_length": window_length,
            "method": "Window-mean logistic",
            "accuracy": accuracy_score(
                test_y,
                mean_model.predict(test.mean(axis=1)),
            ),
            "roc_auc": roc_auc_score(test_y, mean_probability),
        },
    ]
    diagnostic = {
        "repetition": repetition,
        "window_length": window_length,
        "learned_screw_overlap": float(
            abs(np.vdot(result.eigenvectors[:, 0], mode)) ** 2
        ),
        "trace_distance": result.trace_norm / 2.0,
    }
    return rows, diagnostic


def main():
    started = time.time()
    RESULTS.mkdir(parents=True, exist_ok=True)
    rows = []
    diagnostics = []
    for repetition in range(12):
        for window_length in (1, 4, 16, 64):
            run_rows, diagnostic = one_repetition(
                repetition,
                window_length,
            )
            rows.extend(run_rows)
            diagnostics.append(diagnostic)
    benchmark = pd.DataFrame(rows)
    diagnostic_frame = pd.DataFrame(diagnostics)
    benchmark_path = RESULTS / "contact_detection.csv"
    diagnostic_path = RESULTS / "wrench_localization.csv"
    benchmark.to_csv(benchmark_path, index=False)
    diagnostic_frame.to_csv(diagnostic_path, index=False)

    sns.set_theme(style="whitegrid", context="paper")
    figure, axes = plt.subplots(1, 2, figsize=(8.3, 3.2))
    sns.lineplot(
        data=benchmark,
        x="window_length",
        y="roc_auc",
        hue="method",
        marker="o",
        errorbar=("ci", 95),
        ax=axes[0],
    )
    axes[0].set_xscale("log", base=2)
    axes[0].set_ylim(0.45, 1.02)
    axes[0].set_ylabel("ROC AUC")
    axes[0].legend(fontsize=7)
    sns.lineplot(
        data=diagnostic_frame,
        x="window_length",
        y="learned_screw_overlap",
        marker="o",
        errorbar=("ci", 95),
        color="#b44b35",
        ax=axes[1],
    )
    axes[1].set_xscale("log", base=2)
    axes[1].set_ylim(0.8, 1.005)
    axes[1].set_ylabel("learned/oracle screw overlap")
    figure.tight_layout()
    pdf_path = RESULTS / "robot_contact_detection.pdf"
    png_path = RESULTS / "robot_contact_detection.png"
    figure.savefig(pdf_path)
    figure.savefig(png_path, dpi=220)
    plt.close(figure)

    selected = benchmark[benchmark.window_length == 64]
    summary = {
        "model": "six-axis force/torque residual with rank-one contact mode",
        "window_length_for_summary": 64,
        "mean_roc_auc": (selected.groupby("method").roc_auc.mean().to_dict()),
        "mean_mode_overlap": float(diagnostic_frame.learned_screw_overlap.mean()),
        "claim_boundary": (
            "This is a physically structured simulation, not evidence from "
            "a deployed robot or a public contact dataset."
        ),
    }
    summary_path = RESULTS / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    outputs = [
        benchmark_path,
        diagnostic_path,
        pdf_path,
        png_path,
        summary_path,
    ]
    write_manifest(
        RESULTS / "manifest.json",
        experiment="run3-robot-contact-wrench",
        started_at=started,
        config={
            "repetitions": 12,
            "window_lengths": [1, 4, 16, 64],
            "calibration_windows_per_class": 300,
        },
        outputs=outputs,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
