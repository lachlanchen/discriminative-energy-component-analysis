#!/usr/bin/env python3
"""Sample efficiency under an exactly known cyclic-translation nuisance."""

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
from aoc.states import pure_state_density
from aoc.symmetry import translation_power_state
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

RUN_ROOT = Path(__file__).resolve().parents[1]
RESULTS = RUN_ROOT / "results" / "translation"


def signal_pair(
    frequency: int,
    *,
    length: int,
    noise: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return an exactly sign-paired signal with random phase and shift."""

    grid = np.arange(length)
    phase = rng.uniform(0.0, 2.0 * np.pi)
    shift = rng.integers(length)
    signal = np.cos(2.0 * np.pi * frequency * (grid - shift) / length + phase)
    signal += noise * rng.normal(size=length)
    signal /= np.linalg.norm(signal)
    return np.stack([signal, -signal])


def dataset(
    pairs_per_class: int,
    *,
    length: int,
    noise: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    classes = []
    labels = []
    for label, frequency in enumerate((5, 17)):
        samples = np.concatenate(
            [
                signal_pair(
                    frequency,
                    length=length,
                    noise=noise,
                    rng=rng,
                )
                for _ in range(pairs_per_class)
            ]
        )
        classes.append(samples)
        labels.append(np.full(len(samples), label))
    values = np.concatenate(classes)
    targets = np.concatenate(labels)
    order = rng.permutation(len(values))
    return values[order], targets[order]


def effect_scores(effect: np.ndarray, signals: np.ndarray) -> np.ndarray:
    return np.real(np.einsum("bi,ij,bj->b", signals, effect, signals))


def scalar_classifier(train, train_y, test):
    model = LogisticRegression(C=1e3).fit(train.reshape(-1, 1), train_y)
    return model.predict(test.reshape(-1, 1))


def one_repetition(repetition: int, train_pairs: int):
    length = 128
    noise = 0.8
    rng = np.random.default_rng(910000 + 1000 * repetition + train_pairs)
    train_x, train_y = dataset(
        train_pairs,
        length=length,
        noise=noise,
        rng=rng,
    )
    test_x, test_y = dataset(
        300,
        length=length,
        noise=noise,
        rng=rng,
    )

    raw_states = [
        AdditiveState.from_samples(
            [pure_state_density(item) for item in train_x[train_y == label]]
        ).density
        for label in (0, 1)
    ]
    raw = maximum_observable_contrast(raw_states[1], raw_states[0], rank=2)
    raw_train = effect_scores(raw.effect, train_x)
    raw_test = effect_scores(raw.effect, test_x)

    train_power = np.asarray(
        [np.real(np.diag(translation_power_state(item))) for item in train_x]
    )
    test_power = np.asarray(
        [np.real(np.diag(translation_power_state(item))) for item in test_x]
    )
    invariant_states = [
        AdditiveState.from_samples(
            [np.diag(item) for item in train_power[train_y == label]]
        ).density
        for label in (0, 1)
    ]
    invariant = maximum_observable_contrast(
        invariant_states[1],
        invariant_states[0],
        rank=2,
    )
    diagonal = np.real(np.diag(invariant.effect))
    invariant_train = train_power @ diagonal
    invariant_test = test_power @ diagonal

    models = {
        "Raw linear logistic": LogisticRegression(max_iter=3000),
        "Fourier-power logistic": LogisticRegression(
            max_iter=3000,
            C=10.0,
        ),
    }
    predictions = {
        "Untwirled rank-2 AOC": scalar_classifier(
            raw_train,
            train_y,
            raw_test,
        ),
        "Translation-invariant rank-2 AOC": scalar_classifier(
            invariant_train,
            train_y,
            invariant_test,
        ),
    }
    models["Raw linear logistic"].fit(train_x, train_y)
    predictions["Raw linear logistic"] = models["Raw linear logistic"].predict(test_x)
    models["Fourier-power logistic"].fit(train_power, train_y)
    predictions["Fourier-power logistic"] = models["Fourier-power logistic"].predict(
        test_power
    )
    rows = [
        {
            "repetition": repetition,
            "train_pairs_per_class": train_pairs,
            "train_samples_per_class": 2 * train_pairs,
            "method": name,
            "accuracy": accuracy_score(test_y, prediction),
        }
        for name, prediction in predictions.items()
    ]
    diagnostic = {
        "repetition": repetition,
        "train_pairs_per_class": train_pairs,
        "raw_trace_distance": raw.trace_norm / 2.0,
        "invariant_trace_distance": invariant.trace_norm / 2.0,
        "selected_fourier_bins": json.dumps(np.flatnonzero(diagonal > 0.5).tolist()),
    }
    return rows, diagnostic


def main():
    started = time.time()
    RESULTS.mkdir(parents=True, exist_ok=True)
    rows = []
    diagnostics = []
    for repetition in range(12):
        for train_pairs in (1, 2, 4, 8, 16):
            repetition_rows, diagnostic = one_repetition(
                repetition,
                train_pairs,
            )
            rows.extend(repetition_rows)
            diagnostics.append(diagnostic)
    benchmark = pd.DataFrame(rows)
    diagnostic_frame = pd.DataFrame(diagnostics)
    benchmark_path = RESULTS / "sample_efficiency.csv"
    diagnostic_path = RESULTS / "selected_modes.csv"
    benchmark.to_csv(benchmark_path, index=False)
    diagnostic_frame.to_csv(diagnostic_path, index=False)

    sns.set_theme(style="whitegrid", context="paper")
    figure, axis = plt.subplots(figsize=(6.4, 3.5))
    sns.lineplot(
        data=benchmark,
        x="train_samples_per_class",
        y="accuracy",
        hue="method",
        marker="o",
        errorbar=("ci", 95),
        ax=axis,
    )
    axis.axhline(0.5, color="black", linestyle=":", linewidth=1)
    axis.set_ylim(0.45, 1.02)
    axis.set_xscale("log", base=2)
    axis.set_xlabel("training signals per class")
    axis.set_ylabel("held-out accuracy")
    axis.legend(fontsize=7, loc="lower right")
    figure.tight_layout()
    pdf_path = RESULTS / "translation_sample_efficiency.pdf"
    png_path = RESULTS / "translation_sample_efficiency.png"
    figure.savefig(pdf_path)
    figure.savefig(png_path, dpi=220)
    plt.close(figure)

    summary = {
        "signal_length": 128,
        "frequencies": [5, 17],
        "noise_standard_deviation": 0.8,
        "test_pairs_per_class": 300,
        "repetitions": 12,
        "mean_accuracy_at_two_samples": (
            benchmark[benchmark.train_samples_per_class == 2]
            .groupby("method")
            .accuracy.mean()
            .to_dict()
        ),
        "interpretation": (
            "The invariant AOC and a correctly specified Fourier-power "
            "classifier solve the task with one sign-pair per class; the "
            "result isolates symmetry prior rather than generic superiority."
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
        experiment="run3-cyclic-translation-vision",
        started_at=started,
        config={
            "train_pairs_per_class": [1, 2, 4, 8, 16],
            "repetitions": 12,
            "test_pairs_per_class": 300,
            "rank": 2,
        },
        outputs=outputs,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
