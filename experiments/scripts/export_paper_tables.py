#!/usr/bin/env python3
"""Export LaTeX tables directly from frozen experiment CSV files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments" / "results" / "classical"
OUTPUT = ROOT / "publication" / "generated"


DATASET_LABELS = {
    "synthetic_covariance": "Covariance signal",
    "synthetic_mean": "Mean signal",
    "iris": "Iris",
    "wine": "Wine",
    "breast_cancer": "Breast Cancer",
    "digits": "Digits",
    "banknote": "Banknote",
    "spambase": "Spambase",
    "dry_bean": "Dry Bean",
    "letter": "Letter",
}

METHOD_LABELS = {
    "Logistic": "Logistic",
    "LDA-shrinkage": "LDA",
    "QDA": "QDA",
    "SVM-polynomial-2": "Poly-SVM",
    "SVM-RBF": "RBF-SVM",
    "Random-forest": "RF",
    "Hist-gradient-boosting": "HGB",
    "PVM-DECA-amplitude": "PVM-A",
    "Spectral-DECA-amplitude": "Spec-A",
    "Spectral-DECA-affine": "Spec-F",
    "PGM-affine": "PGM-F",
}


def accuracy_cell(row):
    return f"{row.accuracy_mean:.3f} $\\pm$ {row.accuracy_std:.3f}"


def write_table(path, lines):
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main_accuracy(summary):
    methods = [
        "Logistic",
        "QDA",
        "SVM-RBF",
        "Random-forest",
        "PVM-DECA-amplitude",
        "PGM-affine",
    ]
    lines = [
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        (
            r"Dataset & Logistic & QDA & RBF-SVM & RF & "
            r"PVM-A & PGM-F \\"
        ),
        r"\midrule",
    ]
    for dataset in DATASET_LABELS:
        row = [DATASET_LABELS[dataset]]
        for method in methods:
            match = summary[
                summary.dataset.eq(dataset) & summary.method.eq(method)
            ].iloc[0]
            row.append(accuracy_cell(match))
        lines.append(" & ".join(row) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    write_table(OUTPUT / "main_accuracy_table.tex", lines)


def full_accuracy(summary):
    methods = list(METHOD_LABELS)
    lines = [
        r"\begin{tabular}{l" + "c" * len(methods) + "}",
        r"\toprule",
        "Dataset & "
        + " & ".join(METHOD_LABELS[method] for method in methods)
        + r" \\",
        r"\midrule",
    ]
    for dataset in DATASET_LABELS:
        row = [DATASET_LABELS[dataset]]
        for method in methods:
            match = summary[
                summary.dataset.eq(dataset) & summary.method.eq(method)
            ].iloc[0]
            row.append(f"{match.accuracy_mean:.3f}")
        lines.append(" & ".join(row) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    write_table(OUTPUT / "full_accuracy_table.tex", lines)


def resource_table(resources):
    datasets = ["digits", "spambase", "dry_bean", "letter"]
    methods = [
        "SVM-RBF",
        "PVM-DECA-amplitude",
        "PGM-affine",
    ]
    lines = [
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        (
            r"Dataset & Method & Accuracy & Fit (s) & "
            r"Predict ($\mu$s/sample) \\"
        ),
        r"\midrule",
    ]
    for dataset in datasets:
        for index, method in enumerate(methods):
            row = resources[
                resources.dataset.eq(dataset)
                & resources.method.eq(method)
            ].iloc[0]
            dataset_label = (
                DATASET_LABELS[dataset] if index == 0 else ""
            )
            lines.append(
                f"{dataset_label} & {METHOD_LABELS[method]} & "
                f"{row.accuracy:.3f} & {row.fit_seconds:.4g} & "
                f"{row.predict_microseconds:.4g}"
                + r" \\"
            )
        if dataset != datasets[-1]:
            lines.append(r"\addlinespace")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    write_table(OUTPUT / "resource_table.tex", lines)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(RESULTS / "benchmark_summary.csv")
    resources = pd.read_csv(RESULTS / "resource_summary.csv")
    main_accuracy(summary)
    full_accuracy(summary)
    resource_table(resources)
    print(f"Wrote paper tables to {OUTPUT}")


if __name__ == "__main__":
    main()
