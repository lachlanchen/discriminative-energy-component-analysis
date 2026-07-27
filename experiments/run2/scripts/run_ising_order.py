#!/usr/bin/env python3
"""Discover the symmetry-blind Ising order mode from additive states."""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from aoc import AdditiveState, maximum_observable_contrast
from aoc.physics import ising_energy, sample_ising
from aoc.repro import write_manifest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

RUN_ROOT = Path(__file__).resolve().parents[1]
RESULTS = RUN_ROOT / "results" / "ising"


def paired_configurations(
    *,
    size: int,
    temperature: float,
    samples: int,
    seed: int,
    steps_between: int,
):
    if samples % 2:
        raise ValueError("samples must be even for exact Z2 pairing.")
    half = sample_ising(
        size=size,
        temperature=temperature,
        samples=samples // 2,
        seed=seed,
        burn_in=450,
        steps_between=steps_between,
        random_global_flip=False,
    )
    paired = np.concatenate([half, -half], axis=0)
    rng = np.random.default_rng(seed + 7919)
    return paired[rng.permutation(samples)]


def scalar_accuracy(train_scores, train_labels, test_scores, test_labels):
    classifier = LogisticRegression(C=1e3).fit(
        np.asarray(train_scores).reshape(-1, 1),
        train_labels,
    )
    return accuracy_score(
        test_labels,
        classifier.predict(np.asarray(test_scores).reshape(-1, 1)),
    )


def benchmark_repetition(repetition: int, size: int = 12):
    train_per_class = 240
    test_per_class = 480
    cold_temperature = 1.6
    hot_temperature = 3.5
    seeds = np.arange(4) + 10000 * repetition + 31
    cold_train = paired_configurations(
        size=size,
        temperature=cold_temperature,
        samples=train_per_class,
        seed=int(seeds[0]),
        steps_between=4,
    )
    hot_train = paired_configurations(
        size=size,
        temperature=hot_temperature,
        samples=train_per_class,
        seed=int(seeds[1]),
        steps_between=8,
    )
    cold_test = paired_configurations(
        size=size,
        temperature=cold_temperature,
        samples=test_per_class,
        seed=int(seeds[2]),
        steps_between=4,
    )
    hot_test = paired_configurations(
        size=size,
        temperature=hot_temperature,
        samples=test_per_class,
        seed=int(seeds[3]),
        steps_between=8,
    )
    scale = np.sqrt(size * size)
    train = (
        np.concatenate([cold_train, hot_train]).reshape(2 * train_per_class, -1) / scale
    )
    test = np.concatenate([cold_test, hot_test]).reshape(2 * test_per_class, -1) / scale
    y_train = np.concatenate([np.ones(train_per_class), np.zeros(train_per_class)])
    y_test = np.concatenate([np.ones(test_per_class), np.zeros(test_per_class)])
    cold_state = AdditiveState.from_samples(train[y_train == 1]).density
    hot_state = AdditiveState.from_samples(train[y_train == 0]).density
    contrast = maximum_observable_contrast(cold_state, hot_state)
    aoc_train = np.real(np.einsum("bi,ij,bj->b", train, contrast.effect, train))
    aoc_test = np.real(np.einsum("bi,ij,bj->b", test, contrast.effect, test))
    uniform = np.ones(size * size) / scale
    magnetization_train = np.abs(train @ uniform)
    magnetization_test = np.abs(test @ uniform)
    energy_train = np.concatenate(
        [
            [ising_energy(item) for item in cold_train],
            [ising_energy(item) for item in hot_train],
        ]
    )
    energy_test = np.concatenate(
        [
            [ising_energy(item) for item in cold_test],
            [ising_energy(item) for item in hot_test],
        ]
    )
    methods = {
        "AOC learned observable": scalar_accuracy(aoc_train, y_train, aoc_test, y_test),
        "Oracle |magnetization|": scalar_accuracy(
            magnetization_train,
            y_train,
            magnetization_test,
            y_test,
        ),
        "Oracle nearest-neighbor energy": scalar_accuracy(
            energy_train, y_train, energy_test, y_test
        ),
    }
    classifiers = {
        "Linear logistic": LogisticRegression(max_iter=4000),
        "Polynomial SVM degree 2": SVC(kernel="poly", degree=2, C=1.0, gamma="scale"),
        "RBF SVM": make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", C=10.0, gamma="scale"),
        ),
    }
    for name, classifier in classifiers.items():
        classifier.fit(train, y_train)
        methods[name] = accuracy_score(y_test, classifier.predict(test))
    rows = [
        {
            "repetition": repetition,
            "method": name,
            "accuracy": value,
        }
        for name, value in methods.items()
    ]
    diagnostic = {
        "repetition": repetition,
        "effect_rank": contrast.rank,
        "trace_distance": contrast.trace_norm / 2.0,
        "leading_uniform_overlap": float(
            abs(np.vdot(contrast.eigenvectors[:, 0], uniform)) ** 2
        ),
    }
    return rows, diagnostic


def temperature_scan(size: int, seed: int):
    cold = paired_configurations(
        size=size,
        temperature=1.5,
        samples=300,
        seed=seed,
        steps_between=4,
    )
    hot = paired_configurations(
        size=size,
        temperature=3.5,
        samples=300,
        seed=seed + 1,
        steps_between=8,
    )
    scale = np.sqrt(size * size)
    cold_state = AdditiveState.from_samples(cold.reshape(len(cold), -1) / scale).density
    hot_state = AdditiveState.from_samples(hot.reshape(len(hot), -1) / scale).density
    effect = maximum_observable_contrast(cold_state, hot_state).effect
    rows = []
    for index, temperature in enumerate(np.linspace(1.5, 3.5, 17)):
        configurations = paired_configurations(
            size=size,
            temperature=float(temperature),
            samples=400,
            seed=seed + 100 + index,
            steps_between=6,
        )
        states = configurations.reshape(len(configurations), -1) / scale
        scores = np.real(np.einsum("bi,ij,bj->b", states, effect, states))
        rows.append(
            {
                "size": size,
                "temperature": temperature,
                "mean_score": float(scores.mean()),
                "standard_error": float(scores.std(ddof=1) / np.sqrt(len(scores))),
            }
        )
    return rows


def main():
    started = time.time()
    RESULTS.mkdir(parents=True, exist_ok=True)
    benchmark_rows = []
    diagnostic_rows = []
    for repetition in range(10):
        rows, diagnostic = benchmark_repetition(repetition)
        benchmark_rows.extend(rows)
        diagnostic_rows.append(diagnostic)
    benchmark = pd.DataFrame(benchmark_rows)
    diagnostics = pd.DataFrame(diagnostic_rows)
    scan = pd.DataFrame(
        [
            row
            for size, seed in ((8, 801), (12, 1201), (16, 1601))
            for row in temperature_scan(size, seed)
        ]
    )
    benchmark_path = RESULTS / "classification.csv"
    diagnostic_path = RESULTS / "order_mode.csv"
    scan_path = RESULTS / "temperature_scan.csv"
    benchmark.to_csv(benchmark_path, index=False)
    diagnostics.to_csv(diagnostic_path, index=False)
    scan.to_csv(scan_path, index=False)

    sns.set_theme(style="whitegrid", context="paper")
    figure, axes = plt.subplots(1, 2, figsize=(8.4, 3.25))
    order = benchmark.groupby("method").accuracy.mean().sort_values().index
    sns.barplot(
        data=benchmark,
        y="method",
        x="accuracy",
        order=order,
        errorbar=("ci", 95),
        ax=axes[0],
        color="#3567a8",
    )
    axes[0].set_xlim(0.45, 1.02)
    axes[0].set_xlabel("held-out accuracy")
    axes[0].set_ylabel("")
    for size, group in scan.groupby("size"):
        axes[1].errorbar(
            group.temperature,
            group.mean_score,
            yerr=group.standard_error,
            marker="o",
            markersize=3,
            linewidth=1,
            label=f"L={size}",
        )
    exact_tc = 2.0 / np.log(1.0 + np.sqrt(2.0))
    axes[1].axvline(
        exact_tc,
        color="black",
        linestyle="--",
        linewidth=1,
        label=r"$T_c$ exact",
    )
    axes[1].set_xlabel("temperature")
    axes[1].set_ylabel("learned observable expectation")
    axes[1].legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure_paths = [
        RESULTS / "ising_order.pdf",
        RESULTS / "ising_order.png",
    ]
    figure.savefig(figure_paths[0])
    figure.savefig(figure_paths[1], dpi=220)
    plt.close(figure)

    grouped = benchmark.groupby("method").accuracy.agg(["mean", "std"])
    slopes = []
    for size, group in scan.groupby("size"):
        derivative = np.abs(np.gradient(group.mean_score, group.temperature))
        slopes.append(
            {
                "size": int(size),
                "steepest_temperature_grid_point": float(
                    group.temperature.iloc[int(np.argmax(derivative))]
                ),
            }
        )
    summary = {
        "accuracy": grouped.to_dict(orient="index"),
        "leading_uniform_overlap_mean": float(
            diagnostics.leading_uniform_overlap.mean()
        ),
        "leading_uniform_overlap_min": float(diagnostics.leading_uniform_overlap.min()),
        "exact_critical_temperature": exact_tc,
        "finite_grid_steepest_points": slopes,
    }
    summary_path = RESULTS / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_manifest(
        RESULTS / "manifest.json",
        experiment="run2_ising_order",
        started_at=started,
        config={
            "lattice_sizes": [8, 12, 16],
            "benchmark_repetitions": 10,
            "z2_pairing": True,
            "cold_temperature": 1.6,
            "hot_temperature": 3.5,
        },
        outputs=[
            benchmark_path,
            diagnostic_path,
            scan_path,
            *figure_paths,
            summary_path,
        ],
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
