#!/usr/bin/env python3
"""Thermal mass-spring damage detection and online witness localization."""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from aoc import (
    AdditiveState,
    PredictableContrastEProcess,
    maximum_observable_contrast,
)
from aoc.physics import (
    sample_directional_gaussian,
    thermal_chain_model,
)
from aoc.repro import write_manifest
from scipy.stats import beta
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

RUN_ROOT = Path(__file__).resolve().parents[1]
RESULTS = RUN_ROOT / "results" / "structural"


def window_samples(covariance, windows, length, rng):
    samples = sample_directional_gaussian(
        covariance,
        windows * length,
        rng,
    )
    return samples.reshape(windows, length, covariance.shape[0])


def window_density(windows):
    return np.einsum("bwi,bwj->bij", windows, windows) / windows.shape[1]


def scalar_accuracy(train_score, train_y, test_score, test_y):
    model = LogisticRegression(C=1e3).fit(train_score.reshape(-1, 1), train_y)
    return accuracy_score(test_y, model.predict(test_score.reshape(-1, 1)))


def offline_benchmark(repetitions: int = 8):
    rows = []
    diagnostics = []
    masses = 12
    damage_levels = [0.10, 0.20, 0.35, 0.50, 0.70]
    window_lengths = [1, 8, 32, 128]
    for repetition in range(repetitions):
        for damage in damage_levels:
            rng = np.random.default_rng(700001 + repetition * 1000 + int(damage * 100))
            baseline, damaged, mode = thermal_chain_model(
                masses,
                damaged_spring=6,
                damage_fraction=damage,
            )
            calibration_count = 24000
            reference = AdditiveState.from_samples(
                sample_directional_gaussian(baseline, calibration_count, rng)
            ).density
            alternative = AdditiveState.from_samples(
                sample_directional_gaussian(damaged, calibration_count, rng)
            ).density
            result = maximum_observable_contrast(
                alternative,
                reference,
            )
            rank_one_result = maximum_observable_contrast(
                alternative,
                reference,
                rank=1,
            )
            overlap = float(abs(np.vdot(result.eigenvectors[:, 0], mode)) ** 2)
            diagnostics.append(
                {
                    "repetition": repetition,
                    "damage_fraction": damage,
                    "trace_distance": result.trace_norm / 2,
                    "effect_rank": result.rank,
                    "damage_mode_overlap": overlap,
                }
            )
            for window_length in window_lengths:
                train0 = window_samples(baseline, 300, window_length, rng)
                train1 = window_samples(damaged, 300, window_length, rng)
                test0 = window_samples(baseline, 600, window_length, rng)
                test1 = window_samples(damaged, 600, window_length, rng)
                train = np.concatenate([train0, train1])
                test = np.concatenate([test0, test1])
                train_y = np.concatenate([np.zeros(300), np.ones(300)])
                test_y = np.concatenate([np.zeros(600), np.ones(600)])

                train_aoc = np.real(
                    np.einsum(
                        "bwi,ij,bwj->bw",
                        train,
                        result.effect,
                        train,
                    ).mean(axis=1)
                )
                test_aoc = np.real(
                    np.einsum(
                        "bwi,ij,bwj->bw",
                        test,
                        result.effect,
                        test,
                    ).mean(axis=1)
                )
                train_aoc_rank_one = np.real(
                    np.einsum(
                        "bwi,ij,bwj->bw",
                        train,
                        rank_one_result.effect,
                        train,
                    ).mean(axis=1)
                )
                test_aoc_rank_one = np.real(
                    np.einsum(
                        "bwi,ij,bwj->bw",
                        test,
                        rank_one_result.effect,
                        test,
                    ).mean(axis=1)
                )
                train_oracle = np.mean(np.abs(train @ mode) ** 2, axis=1)
                test_oracle = np.mean(np.abs(test @ mode) ** 2, axis=1)
                train_states = window_density(train)
                test_states = window_density(test)
                train_nearest = np.linalg.norm(
                    train_states - reference[None, :, :],
                    axis=(1, 2),
                ) - np.linalg.norm(
                    train_states - alternative[None, :, :],
                    axis=(1, 2),
                )
                test_nearest = np.linalg.norm(
                    test_states - reference[None, :, :],
                    axis=(1, 2),
                ) - np.linalg.norm(
                    test_states - alternative[None, :, :],
                    axis=(1, 2),
                )
                mean_model = LogisticRegression(max_iter=2000).fit(
                    train.mean(axis=1),
                    train_y,
                )
                method_scores = {
                    "AOC full effect": (train_aoc, test_aoc),
                    "AOC rank-1 witness": (
                        train_aoc_rank_one,
                        test_aoc_rank_one,
                    ),
                    "Oracle damage mode": (train_oracle, test_oracle),
                    "Nearest covariance centroid": (
                        train_nearest,
                        test_nearest,
                    ),
                }
                for method, (train_score, test_score) in method_scores.items():
                    rows.append(
                        {
                            "repetition": repetition,
                            "damage_fraction": damage,
                            "window_length": window_length,
                            "method": method,
                            "roc_auc": roc_auc_score(test_y, test_score),
                            "accuracy": scalar_accuracy(
                                train_score,
                                train_y,
                                test_score,
                                test_y,
                            ),
                        }
                    )
                mean_prediction = mean_model.predict(test.mean(axis=1))
                mean_probability = mean_model.predict_proba(test.mean(axis=1))[:, 1]
                rows.append(
                    {
                        "repetition": repetition,
                        "damage_fraction": damage,
                        "window_length": window_length,
                        "method": "Window-mean logistic",
                        "roc_auc": roc_auc_score(test_y, mean_probability),
                        "accuracy": accuracy_score(test_y, mean_prediction),
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(diagnostics)


def online_benchmark(
    null_streams: int = 120,
    change_streams: int = 120,
):
    masses = 12
    baseline, damaged, _ = thermal_chain_model(
        masses,
        damaged_spring=6,
        damage_fraction=0.70,
    )
    reference = np.eye(masses) / masses
    horizon = 250
    change_time = 50
    window_length = 64
    rows = []
    trace_rows = []
    for regime, streams in (("null", null_streams), ("change", change_streams)):
        for stream in range(streams):
            rng = np.random.default_rng(
                880000 + stream + (10000 if regime == "change" else 0)
            )
            detector = PredictableContrastEProcess(
                reference,
                adaptation_window=16,
                witness_rank=1,
                alpha=0.01,
            )
            alarm_time = None
            for index in range(horizon):
                covariance = (
                    damaged if regime == "change" and index >= change_time else baseline
                )
                samples = sample_directional_gaussian(covariance, window_length, rng)
                state = samples.T @ samples / window_length
                record = detector.update(state)
                if stream == 0:
                    trace_rows.append(
                        {
                            "regime": regime,
                            **record.__dict__,
                        }
                    )
                if alarm_time is None and record.alarm:
                    alarm_time = index + 1
            rows.append(
                {
                    "regime": regime,
                    "stream": stream,
                    "alarm": alarm_time is not None,
                    "alarm_time": alarm_time,
                    "delay": (
                        alarm_time - change_time
                        if regime == "change" and alarm_time is not None
                        else np.nan
                    ),
                    "max_e_value": detector.max_e_value,
                }
            )
    return (
        pd.DataFrame(rows),
        pd.DataFrame(trace_rows),
        {
            "change_time": change_time,
            "horizon": horizon,
            "window_length": window_length,
            "damage_fraction": 0.70,
        },
    )


def main():
    started = time.time()
    RESULTS.mkdir(parents=True, exist_ok=True)
    offline, diagnostics = offline_benchmark()
    online, traces, online_config = online_benchmark()
    offline_path = RESULTS / "offline_benchmark.csv"
    diagnostic_path = RESULTS / "mode_localization.csv"
    online_path = RESULTS / "online_runs.csv"
    trace_path = RESULTS / "online_example_trace.csv"
    offline.to_csv(offline_path, index=False)
    diagnostics.to_csv(diagnostic_path, index=False)
    online.to_csv(online_path, index=False)
    traces.to_csv(trace_path, index=False)

    sns.set_theme(style="whitegrid", context="paper")
    figure, axes = plt.subplots(1, 2, figsize=(8.3, 3.25))
    selected = offline[offline.window_length == 128]
    sns.lineplot(
        data=selected,
        x="damage_fraction",
        y="roc_auc",
        hue="method",
        marker="o",
        errorbar=("ci", 95),
        ax=axes[0],
    )
    axes[0].set_ylim(0.45, 1.02)
    axes[0].set_ylabel("ROC AUC (128-sample window)")
    axes[0].legend(frameon=False, fontsize=7)
    for regime, group in traces.groupby("regime"):
        axes[1].plot(
            group.time,
            np.log10(np.maximum(group.e_value, 1e-12)),
            label=regime,
        )
    axes[1].axhline(
        2,
        color="black",
        linestyle="--",
        linewidth=1,
        label=r"$1/\alpha=100$",
    )
    axes[1].axvline(
        online_config["change_time"],
        color="gray",
        linestyle=":",
        linewidth=1,
    )
    axes[1].set_xlabel("stream window")
    axes[1].set_ylabel(r"$\log_{10}$ anytime evidence")
    axes[1].legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure_paths = [
        RESULTS / "structural_monitoring.pdf",
        RESULTS / "structural_monitoring.png",
    ]
    figure.savefig(figure_paths[0])
    figure.savefig(figure_paths[1], dpi=220)
    plt.close(figure)

    null = online[online.regime == "null"]
    changed = online[online.regime == "change"]
    false_alarms = int(null.alarm.sum())
    false_alarm_upper = float(
        beta.ppf(
            0.975,
            false_alarms + 1,
            len(null) - false_alarms,
        )
    )
    focal = offline[(offline.damage_fraction == 0.35) & (offline.window_length == 128)]
    summary = {
        "focal_damage_fraction": 0.35,
        "focal_window_length": 128,
        "focal_metrics": focal.groupby("method")[["roc_auc", "accuracy"]]
        .mean()
        .to_dict(orient="index"),
        "damage_mode_overlap_at_035": {
            "mean": float(
                diagnostics.loc[
                    diagnostics.damage_fraction == 0.35,
                    "damage_mode_overlap",
                ].mean()
            ),
            "minimum": float(
                diagnostics.loc[
                    diagnostics.damage_fraction == 0.35,
                    "damage_mode_overlap",
                ].min()
            ),
        },
        "online": {
            **online_config,
            "null_streams": len(null),
            "false_alarms": false_alarms,
            "false_alarm_rate": false_alarms / len(null),
            "false_alarm_rate_95pct_upper": false_alarm_upper,
            "change_streams": len(changed),
            "detection_rate": float(changed.alarm.mean()),
            "median_detected_delay": float(changed.delay.median()),
        },
    }
    summary_path = RESULTS / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_manifest(
        RESULTS / "manifest.json",
        experiment="run2_structural_monitoring",
        started_at=started,
        config={
            "masses": 12,
            "damaged_spring": 6,
            "offline_repetitions": 8,
            "damage_levels": [0.10, 0.20, 0.35, 0.50, 0.70],
            "window_lengths": [1, 8, 32, 128],
            **online_config,
        },
        outputs=[
            offline_path,
            diagnostic_path,
            online_path,
            trace_path,
            *figure_paths,
            summary_path,
        ],
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
