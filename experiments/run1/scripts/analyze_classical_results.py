#!/usr/bin/env python3
"""Dataset-level statistical analysis of the frozen classical benchmark."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "classical"


def holm_adjust(p_values):
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    count = len(values)
    for rank, index in enumerate(order):
        candidate = (count - rank) * values[index]
        running = max(running, candidate)
        adjusted[index] = min(1.0, running)
    return adjusted


def main():
    folds = pd.read_csv(RESULTS / "benchmark_folds.csv")
    metadata = pd.read_csv(RESULTS / "datasets.csv")
    public = metadata.loc[
        metadata["kind"].str.startswith("public"), "dataset"
    ].tolist()
    means = (
        folds[folds["dataset"].isin(public)]
        .groupby(["dataset", "method"])["accuracy"]
        .mean()
        .unstack()
    )
    statistic, friedman_p = friedmanchisquare(
        *[means[column].to_numpy() for column in means.columns]
    )

    comparisons = [
        ("PGM-affine", "PVM-DECA-amplitude"),
        ("PGM-affine", "Spectral-DECA-affine"),
        ("PGM-affine", "SVM-RBF"),
        ("PVM-DECA-amplitude", "Spectral-DECA-amplitude"),
    ]
    records = []
    for left, right in comparisons:
        differences = means[left] - means[right]
        result = wilcoxon(
            differences,
            alternative="two-sided",
            zero_method="wilcox",
            method="auto",
        )
        records.append(
            {
                "left": left,
                "right": right,
                "datasets": len(differences),
                "mean_accuracy_difference": differences.mean(),
                "median_accuracy_difference": differences.median(),
                "wins_left": int((differences > 0).sum()),
                "ties": int((differences == 0).sum()),
                "wins_right": int((differences < 0).sum()),
                "wilcoxon_statistic": float(result.statistic),
                "p_value": float(result.pvalue),
            }
        )
    comparison_frame = pd.DataFrame(records)
    comparison_frame["holm_p_value"] = holm_adjust(
        comparison_frame["p_value"]
    )
    comparison_frame.to_csv(
        RESULTS / "paired_dataset_comparisons.csv", index=False
    )

    resource = (
        folds.groupby(["dataset", "method"], as_index=False)
        .agg(
            accuracy=("accuracy", "mean"),
            fit_seconds=("fit_seconds", "median"),
            predict_microseconds=(
                "predict_microseconds_per_sample",
                "median",
            ),
            jacobi_sweeps=("jacobi_sweeps", "median"),
            jacobi_objective_gain=("jacobi_objective_gain", "median"),
            offdiagonal_residual=("offdiagonal_residual", "median"),
        )
    )
    resource.to_csv(RESULTS / "resource_summary.csv", index=False)

    report = {
        "public_datasets": public,
        "friedman": {
            "methods": len(means.columns),
            "datasets": len(means),
            "statistic": float(statistic),
            "p_value": float(friedman_p),
        },
        "interpretation": (
            "Dataset means are the independent blocks. Outer folds are not "
            "treated as independent samples. Pairwise Wilcoxon tests are "
            "exploratory because only eight public datasets are available."
        ),
    }
    (RESULTS / "statistical_summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    print(comparison_frame.to_string(index=False))


if __name__ == "__main__":
    main()
