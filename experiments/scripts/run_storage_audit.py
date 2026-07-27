#!/usr/bin/env python3
"""Audit fitted model storage without retaining training-set predictions."""

from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold

from deca.datasets import benchmark_datasets
from run_classical_benchmarks import model_suite


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "classical"
DATA_HOME = ROOT / "data" / "uci"
METHODS = [
    "Logistic",
    "SVM-RBF",
    "PVM-DECA-amplitude",
    "Spectral-DECA-affine",
    "PGM-affine",
]


def main():
    records = []
    for dataset in benchmark_datasets(DATA_HOME).values():
        train, _ = next(
            StratifiedKFold(
                n_splits=5, shuffle=True, random_state=20260727
            ).split(dataset.X, dataset.y)
        )
        suite = model_suite(dataset.X.shape[1], quick=False)
        for method in METHODS:
            model = clone(suite[method]).fit(
                dataset.X[train], dataset.y[train]
            )
            payload = pickle.dumps(
                model, protocol=pickle.HIGHEST_PROTOCOL
            )
            records.append(
                {
                    "dataset": dataset.name,
                    "method": method,
                    "train_samples": len(train),
                    "model_storage_bytes": len(payload),
                }
            )
            print(
                dataset.name,
                method,
                len(payload),
                flush=True,
            )
    pd.DataFrame(records).to_csv(
        RESULTS / "model_storage_audit.csv", index=False
    )


if __name__ == "__main__":
    main()
