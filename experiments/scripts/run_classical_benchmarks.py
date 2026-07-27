#!/usr/bin/env python3
"""Repeated stratified benchmarks for DECA and classical baselines."""

from __future__ import annotations

import argparse
import json
import pickle
import platform
import time
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import qiskit
import scipy
import seaborn as sns
import sklearn
from sklearn.base import clone
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from deca.classifiers import DECAClassifier
from deca.datasets import benchmark_datasets


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "classical"
DATA_HOME = ROOT / "data" / "uci"


def model_suite(dimension: int, quick: bool):
    scale = float(np.sqrt(dimension))
    trees = 100 if quick else 300
    if quick:
        deca_starts, deca_sweeps = 0, 5
    elif dimension >= 32:
        deca_starts, deca_sweeps = 0, 10
    elif dimension >= 12:
        deca_starts, deca_sweeps = 0, 20
    else:
        deca_starts, deca_sweeps = 2, 30
    common = dict(
        priors="empirical",
        random_starts=deca_starts,
        max_sweeps=deca_sweeps,
        random_state=90210,
    )
    return {
        "Logistic": make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, max_iter=3000, random_state=90210),
        ),
        "LDA-shrinkage": make_pipeline(
            StandardScaler(),
            LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
        ),
        "QDA": make_pipeline(
            StandardScaler(),
            QuadraticDiscriminantAnalysis(reg_param=0.1),
        ),
        "SVM-polynomial-2": make_pipeline(
            StandardScaler(),
            SVC(C=1.0, kernel="poly", degree=2, gamma="scale", coef0=1.0),
        ),
        "SVM-RBF": make_pipeline(
            StandardScaler(),
            SVC(C=10.0, kernel="rbf", gamma="scale"),
        ),
        "Random-forest": RandomForestClassifier(
            n_estimators=trees,
            min_samples_leaf=1,
            n_jobs=-1,
            random_state=90210,
        ),
        "Hist-gradient-boosting": HistGradientBoostingClassifier(
            learning_rate=0.1,
            max_iter=30 if quick else 50,
            max_leaf_nodes=15,
            l2_regularization=0.1,
            random_state=90210,
        ),
        "PVM-DECA-amplitude": make_pipeline(
            StandardScaler(),
            DECAClassifier(
                measurement="auto",
                encoding="amplitude",
                decision_rule="measurement",
                **common,
            ),
        ),
        "Spectral-DECA-amplitude": make_pipeline(
            StandardScaler(),
            DECAClassifier(
                measurement="auto",
                encoding="amplitude",
                decision_rule="spectral",
                **common,
            ),
        ),
        "Spectral-DECA-affine": make_pipeline(
            StandardScaler(),
            DECAClassifier(
                measurement="auto",
                encoding="affine",
                encoding_scale=scale,
                decision_rule="spectral",
                **common,
            ),
        ),
        "PGM-affine": make_pipeline(
            StandardScaler(),
            DECAClassifier(
                measurement="pgm",
                encoding="affine",
                encoding_scale=scale,
                **common,
            ),
        ),
    }


def fitted_deca(model):
    if hasattr(model, "named_steps"):
        for step in model.named_steps.values():
            if isinstance(step, DECAClassifier):
                return step
    return None


def parameter_storage(model) -> int:
    return len(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL))


def evaluate(args):
    datasets = benchmark_datasets(
        DATA_HOME, include_large=not args.skip_large
    )
    selected = args.datasets or list(datasets)
    missing = sorted(set(selected) - set(datasets))
    if missing:
        raise ValueError(f"Unknown datasets: {', '.join(missing)}")

    splitter = RepeatedStratifiedKFold(
        n_splits=args.folds,
        n_repeats=args.repeats,
        random_state=20260727,
    )
    records = []
    metadata = []
    for dataset_name in selected:
        dataset = datasets[dataset_name]
        X, y = dataset.X, dataset.y
        models = model_suite(X.shape[1], args.quick)
        metadata.append(
            {
                "dataset": dataset.name,
                "kind": dataset.kind,
                "samples": len(X),
                "features": X.shape[1],
                "classes": len(np.unique(y)),
                "source": dataset.source,
            }
        )
        for split_index, (train, test) in enumerate(splitter.split(X, y)):
            repeat = split_index // args.folds
            fold = split_index % args.folds
            for method, prototype in models.items():
                model = clone(prototype)
                started = time.perf_counter()
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore", category=ConvergenceWarning
                    )
                    warnings.filterwarnings(
                        "ignore", message="The covariance matrix"
                    )
                    model.fit(X[train], y[train])
                fit_seconds = time.perf_counter() - started
                started = time.perf_counter()
                predicted = model.predict(X[test])
                prediction_seconds = time.perf_counter() - started
                deca = fitted_deca(model)
                diagnostics = (
                    deca.solution_.diagnostics if deca is not None else {}
                )
                single_shot = (
                    deca.expected_single_shot_accuracy(
                        model[:-1].transform(X[test]), y[test]
                    )
                    if deca is not None
                    else np.nan
                )
                records.append(
                    {
                        "dataset": dataset.name,
                        "kind": dataset.kind,
                        "repeat": repeat,
                        "fold": fold,
                        "method": method,
                        "accuracy": accuracy_score(y[test], predicted),
                        "balanced_accuracy": balanced_accuracy_score(
                            y[test], predicted
                        ),
                        "macro_f1": f1_score(
                            y[test], predicted, average="macro"
                        ),
                        "single_shot_accuracy": single_shot,
                        "fit_seconds": fit_seconds,
                        "predict_microseconds_per_sample": (
                            1e6 * prediction_seconds / len(test)
                        ),
                        "serialized_bytes": parameter_storage(model),
                        "jacobi_sweeps": diagnostics.get(
                            "sweeps", np.nan
                        ),
                        "jacobi_objective_gain": (
                            diagnostics["objective_history"][-1]
                            - diagnostics["objective_history"][0]
                            if "objective_history" in diagnostics
                            else np.nan
                        ),
                        "offdiagonal_residual": diagnostics.get(
                            "offdiagonal_residual", np.nan
                        ),
                        "train_samples": len(train),
                        "test_samples": len(test),
                    }
                )
                print(
                    f"{dataset.name:22s} r{repeat} f{fold} "
                    f"{method:22s} acc={records[-1]['accuracy']:.4f} "
                    f"fit={fit_seconds:.3f}s",
                    flush=True,
                )
    return pd.DataFrame(records), pd.DataFrame(metadata)


def summarize(records):
    summary = (
        records.groupby(["dataset", "kind", "method"], as_index=False)
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            macro_f1_mean=("macro_f1", "mean"),
            single_shot_accuracy_mean=("single_shot_accuracy", "mean"),
            fit_seconds_median=("fit_seconds", "median"),
            predict_microseconds_median=(
                "predict_microseconds_per_sample",
                "median",
            ),
            serialized_bytes_median=("serialized_bytes", "median"),
        )
    )
    ranks = records.groupby(
        ["dataset", "repeat", "fold"], group_keys=False
    ).apply(
        lambda group: group.assign(
            accuracy_rank=group["accuracy"].rank(
                method="average", ascending=False
            )
        ),
        include_groups=False,
    )
    method_ranks = (
        ranks.groupby("method", as_index=False)
        .agg(
            mean_accuracy_rank=("accuracy_rank", "mean"),
            mean_accuracy=("accuracy", "mean"),
        )
        .sort_values("mean_accuracy_rank")
    )
    return summary, method_ranks


def plot_summary(summary, method_ranks):
    order = method_ranks["method"].tolist()
    accuracy = summary.pivot(
        index="dataset", columns="method", values="accuracy_mean"
    ).reindex(columns=order)
    figure, axis = plt.subplots(
        figsize=(10.5, max(4.0, 0.48 * len(accuracy)))
    )
    sns.heatmap(
        accuracy,
        annot=True,
        fmt=".3f",
        cmap="viridis",
        vmin=max(0.0, float(accuracy.min().min()) - 0.05),
        vmax=1.0,
        cbar_kws={"label": "mean outer-fold accuracy"},
        ax=axis,
    )
    axis.set_xlabel("")
    axis.set_ylabel("")
    figure.tight_layout()
    figure.savefig(RESULTS / "benchmark_accuracy.png", dpi=220)
    figure.savefig(RESULTS / "benchmark_accuracy.pdf")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.0, 4.2))
    sns.barplot(
        data=method_ranks,
        x="mean_accuracy_rank",
        y="method",
        order=order,
        color="#4472C4",
        ax=axis,
    )
    axis.set_xlabel("mean rank (lower is better)")
    axis.set_ylabel("")
    figure.tight_layout()
    figure.savefig(RESULTS / "benchmark_mean_rank.png", dpi=220)
    figure.savefig(RESULTS / "benchmark_mean_rank.pdf")
    plt.close(figure)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--datasets", nargs="*")
    parser.add_argument("--skip-large", action="store_true")
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.folds < 2 or args.repeats < 1:
        raise ValueError("folds must be >=2 and repeats must be >=1.")
    RESULTS.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    records, metadata = evaluate(args)
    summary, method_ranks = summarize(records)
    records.to_csv(RESULTS / "benchmark_folds.csv", index=False)
    metadata.to_csv(RESULTS / "datasets.csv", index=False)
    summary.to_csv(RESULTS / "benchmark_summary.csv", index=False)
    method_ranks.to_csv(RESULTS / "benchmark_method_ranks.csv", index=False)
    plot_summary(summary, method_ranks)
    report = {
        "folds": args.folds,
        "repeats": args.repeats,
        "datasets": metadata["dataset"].tolist(),
        "methods": sorted(records["method"].unique().tolist()),
        "outer_fits": len(records),
        "runtime_seconds": time.perf_counter() - started,
        "best_mean_rank_method": method_ranks.iloc[0]["method"],
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "qiskit": qiskit.__version__,
        },
    }
    (RESULTS / "summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
