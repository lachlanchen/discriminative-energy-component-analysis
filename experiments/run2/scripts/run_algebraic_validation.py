#!/usr/bin/env python3
"""Validate analytical identities, additivity, and sequential calibration."""

from __future__ import annotations

import json
import time
from pathlib import Path

import cvxpy as cp
import numpy as np
import pandas as pd
from aoc import (
    AdditiveState,
    PredictableContrastEProcess,
    maximum_observable_contrast,
    projective_mmd_squared,
)
from aoc.repro import write_manifest
from scipy.stats import beta

RUN_ROOT = Path(__file__).resolve().parents[1]
RESULTS = RUN_ROOT / "results" / "algebra"


def random_density(rng: np.random.Generator, dimension: int):
    matrix = rng.normal(size=(dimension, dimension))
    density = matrix @ matrix.T
    return density / np.trace(density)


def additive_audit(seed: int = 1207, trials: int = 100):
    rng = np.random.default_rng(seed)
    rows = []
    for trial in range(trials):
        dimension = int(rng.integers(2, 18))
        count = int(rng.integers(5, 80))
        vectors = rng.normal(size=(count, dimension))
        weights = rng.uniform(0.05, 3.0, size=count)
        batch = AdditiveState.from_samples(vectors, weights)
        cut1, cut2 = sorted(rng.choice(np.arange(1, count), size=2, replace=False))
        pieces = [
            AdditiveState.from_samples(vectors[:cut1], weights[:cut1]),
            AdditiveState.from_samples(vectors[cut1:cut2], weights[cut1:cut2]),
            AdditiveState.from_samples(vectors[cut2:], weights[cut2:]),
        ]
        merged = pieces[2].copy().merge(pieces[0]).merge(pieces[1])
        rows.append(
            {
                "trial": trial,
                "dimension": dimension,
                "samples": count,
                "accumulator_error": np.linalg.norm(
                    batch.accumulator - merged.accumulator,
                    ord="fro",
                ),
                "density_error": np.linalg.norm(
                    batch.density - merged.density,
                    ord="fro",
                ),
                "mass_error": abs(batch.total_weight - merged.total_weight),
            }
        )
    return pd.DataFrame(rows)


def jordan_sdp_audit(seed: int = 992, trials: int = 24):
    rng = np.random.default_rng(seed)
    rows = []
    for trial in range(trials):
        dimension = int(rng.integers(2, 9))
        first = random_density(rng, dimension)
        second = random_density(rng, dimension)
        analytic = maximum_observable_contrast(first, second)
        effect = cp.Variable((dimension, dimension), symmetric=True)
        problem = cp.Problem(
            cp.Maximize(cp.trace(effect @ (first - second))),
            [effect >> 0, np.eye(dimension) - effect >> 0],
        )
        problem.solve(solver="CLARABEL")
        rows.append(
            {
                "trial": trial,
                "dimension": dimension,
                "analytic_gap": analytic.positive_gap,
                "sdp_gap": problem.value,
                "absolute_error": abs(analytic.positive_gap - problem.value),
                "trace_distance_identity_error": abs(
                    analytic.positive_gap - analytic.trace_norm / 2
                ),
            }
        )
    return pd.DataFrame(rows)


def kernel_audit(seed: int = 411, trials: int = 40):
    rng = np.random.default_rng(seed)
    rows = []
    for trial in range(trials):
        dimension = int(rng.integers(2, 14))
        first = rng.normal(size=(int(rng.integers(8, 30)), dimension))
        second = rng.normal(size=(int(rng.integers(8, 30)), dimension))
        rho_first = AdditiveState.from_samples(first).density
        rho_second = AdditiveState.from_samples(second).density
        operator_value = np.linalg.norm(rho_first - rho_second, ord="fro") ** 2
        kernel_value = projective_mmd_squared(first, second)
        rows.append(
            {
                "trial": trial,
                "dimension": dimension,
                "operator_value": operator_value,
                "kernel_value": kernel_value,
                "absolute_error": abs(operator_value - kernel_value),
            }
        )
    return pd.DataFrame(rows)


def categorical_state(rng, probabilities):
    index = int(rng.choice(len(probabilities), p=probabilities))
    state = np.zeros(len(probabilities))
    state[index] = 1.0
    return state


def sequential_audit(
    seed: int = 551,
    null_streams: int = 500,
    change_streams: int = 200,
    horizon: int = 160,
    change_time: int = 50,
):
    rng = np.random.default_rng(seed)
    baseline = np.array([0.70, 0.10, 0.10, 0.10])
    changed = np.array([0.10, 0.70, 0.10, 0.10])
    reference = np.diag(baseline)
    rows = []
    for regime, streams in (("null", null_streams), ("change", change_streams)):
        for stream in range(streams):
            detector = PredictableContrastEProcess(
                reference,
                adaptation_window=24,
                alpha=0.01,
            )
            alarm_time = None
            maximum_e = 1.0
            for index in range(horizon):
                probabilities = (
                    changed if regime == "change" and index >= change_time else baseline
                )
                record = detector.update(categorical_state(rng, probabilities))
                maximum_e = max(maximum_e, record.e_value)
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
                    "max_e_value": maximum_e,
                }
            )
    frame = pd.DataFrame(rows)
    false_alarms = int(frame.loc[frame.regime == "null", "alarm"].sum())
    upper = float(beta.ppf(0.975, false_alarms + 1, null_streams - false_alarms))
    return frame, {
        "null_streams": null_streams,
        "false_alarms": false_alarms,
        "false_alarm_rate": false_alarms / null_streams,
        "false_alarm_rate_95pct_upper": upper,
        "change_streams": change_streams,
        "detection_rate": float(frame.loc[frame.regime == "change", "alarm"].mean()),
        "median_detected_delay": float(
            frame.loc[frame.regime == "change", "delay"].median()
        ),
    }


def main():
    started = time.time()
    RESULTS.mkdir(parents=True, exist_ok=True)
    additive = additive_audit()
    jordan = jordan_sdp_audit()
    kernel = kernel_audit()
    sequential, sequential_summary = sequential_audit()
    paths = [
        RESULTS / "additive_exactness.csv",
        RESULTS / "jordan_sdp_exactness.csv",
        RESULTS / "kernel_operator_exactness.csv",
        RESULTS / "sequential_calibration.csv",
    ]
    for frame, path in zip(
        (additive, jordan, kernel, sequential),
        paths,
    ):
        frame.to_csv(path, index=False)
    summary = {
        "max_additive_density_error": float(additive.density_error.max()),
        "max_merge_mass_error": float(additive.mass_error.max()),
        "max_jordan_sdp_error": float(jordan.absolute_error.max()),
        "max_trace_distance_identity_error": float(
            jordan.trace_distance_identity_error.max()
        ),
        "max_kernel_operator_error": float(kernel.absolute_error.max()),
        "sequential": sequential_summary,
    }
    summary_path = RESULTS / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_manifest(
        RESULTS / "manifest.json",
        experiment="run2_algebraic_validation",
        started_at=started,
        config={
            "additive_trials": 100,
            "sdp_trials": 24,
            "kernel_trials": 40,
            "null_streams": 500,
            "change_streams": 200,
            "alpha": 0.01,
        },
        outputs=[*paths, summary_path],
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
