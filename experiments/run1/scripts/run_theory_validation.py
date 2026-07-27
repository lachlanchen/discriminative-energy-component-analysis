#!/usr/bin/env python3
"""Numerically validate the DECA theorems and oracle-gap behavior."""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import cvxpy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import qiskit
import scipy
from scipy.linalg import expm

from deca.jacobi import jacobi_deca
from deca.operators import commutator_measure
from deca.solvers import (
    binary_helstrom,
    optimal_povm,
    pretty_good_measurement,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "theory"


def random_density(dimension, rng, complex_data=False):
    raw = rng.normal(size=(dimension, dimension))
    if complex_data:
        raw = raw + 1j * rng.normal(size=(dimension, dimension))
    density = raw @ raw.conj().T
    return density / np.trace(density)


def random_orthogonal(dimension, rng):
    q_matrix, r_matrix = np.linalg.qr(
        rng.normal(size=(dimension, dimension))
    )
    signs = np.sign(np.diag(r_matrix))
    signs[signs == 0] = 1.0
    return q_matrix * signs


def binary_trials():
    records = []
    for dimension in (2, 4, 8):
        for seed in range(10):
            rng = np.random.default_rng(1000 + 100 * dimension + seed)
            prior = rng.uniform(0.15, 0.85)
            operators = [
                prior * random_density(dimension, rng),
                (1.0 - prior) * random_density(dimension, rng),
            ]
            started = time.perf_counter()
            closed = binary_helstrom(operators)
            closed_time = time.perf_counter() - started
            started = time.perf_counter()
            oracle = optimal_povm(operators)
            sdp_time = time.perf_counter() - started
            records.append(
                {
                    "family": "binary_random",
                    "dimension": dimension,
                    "seed": seed,
                    "prior_class_0": prior,
                    "deca_success": closed.success,
                    "oracle_success": oracle.success,
                    "absolute_gap": abs(closed.success - oracle.success),
                    "closed_form_error": closed.diagnostics[
                        "closed_form_error"
                    ],
                    "deca_fit_seconds": closed_time,
                    "sdp_fit_seconds": sdp_time,
                }
            )
    return records


def commuting_trials():
    records = []
    for dimension in (4, 8):
        for seed in range(8):
            rng = np.random.default_rng(2000 + 100 * dimension + seed)
            basis = random_orthogonal(dimension, rng)
            priors = rng.dirichlet(np.ones(3))
            operators = []
            for prior in priors:
                spectrum = rng.dirichlet(np.ones(dimension))
                density = (basis * spectrum) @ basis.T
                operators.append(prior * density)
            deca = jacobi_deca(
                operators,
                random_starts=3,
                max_sweeps=60,
                random_state=seed,
            )
            oracle = optimal_povm(operators)
            records.append(
                {
                    "family": "commuting_multiclass",
                    "dimension": dimension,
                    "seed": seed,
                    "commutator": commutator_measure(operators),
                    "offdiagonal_residual": deca.diagnostics[
                        "offdiagonal_residual"
                    ],
                    "deca_success": deca.success,
                    "oracle_success": oracle.success,
                    "absolute_gap": abs(oracle.success - deca.success),
                    "gap_bound": deca.diagnostics[
                        "oracle_gap_upper_bound"
                    ],
                }
            )
    return records


def noncommuting_sweep():
    records = []
    strengths = np.linspace(0.0, 1.0, 9)
    dimension = 4
    num_classes = 3
    for seed in range(8):
        rng = np.random.default_rng(3000 + seed)
        common_basis = random_orthogonal(dimension, rng)
        spectra = [
            rng.dirichlet(0.8 * np.ones(dimension))
            for _ in range(num_classes)
        ]
        generators = []
        for _ in range(num_classes):
            raw = rng.normal(size=(dimension, dimension))
            generator = raw - raw.T
            generator /= max(1e-12, np.linalg.norm(generator, ord="fro"))
            generators.append(generator)
        for strength in strengths:
            operators = []
            for spectrum, generator in zip(spectra, generators):
                rotation = common_basis @ expm(3.0 * strength * generator)
                density = (rotation * spectrum) @ rotation.T
                operators.append(density / num_classes)
            deca = jacobi_deca(
                operators,
                random_starts=6,
                max_sweeps=80,
                random_state=seed,
            )
            pgm = pretty_good_measurement(operators)
            oracle = optimal_povm(operators)
            gap = oracle.success - deca.success
            records.append(
                {
                    "family": "noncommuting_sweep",
                    "dimension": dimension,
                    "seed": seed,
                    "strength": strength,
                    "commutator": commutator_measure(operators),
                    "offdiagonal_residual": deca.diagnostics[
                        "offdiagonal_residual"
                    ],
                    "deca_success": deca.success,
                    "pgm_success": pgm.success,
                    "oracle_success": oracle.success,
                    "oracle_gap": gap,
                    "gap_bound": deca.diagnostics[
                        "oracle_gap_upper_bound"
                    ],
                    "bound_slack": deca.diagnostics[
                        "oracle_gap_upper_bound"
                    ]
                    - gap,
                    "jacobi_sweeps": deca.diagnostics["sweeps"],
                }
            )
    return records


def trine_trial():
    phases = 2.0 * np.pi * np.arange(3) / 3.0
    states = np.column_stack(
        [
            np.full(3, 1.0 / np.sqrt(2.0), dtype=np.complex128),
            np.exp(1j * phases) / np.sqrt(2.0),
        ]
    )
    operators = [
        np.outer(state, state.conj()) / 3.0 for state in states
    ]
    deca = jacobi_deca(
        operators, random_starts=24, max_sweeps=120, random_state=41
    )
    pgm = pretty_good_measurement(operators)
    oracle = optimal_povm(operators, solver="SCS")
    return {
        "family": "trine",
        "dimension": 2,
        "deca_success": deca.success,
        "pgm_success": pgm.success,
        "oracle_success": oracle.success,
        "oracle_gap": oracle.success - deca.success,
        "deca_component_counts": deca.diagnostics[
            "class_component_counts"
        ],
        "deca_ancillas": 0,
        "general_povm_minimum_outcome_qubits": 2,
    }


def plot_noncommuting(frame):
    summary = (
        frame.groupby("strength")
        .agg(
            commutator_mean=("commutator", "mean"),
            commutator_std=("commutator", "std"),
            gap_mean=("oracle_gap", "mean"),
            gap_std=("oracle_gap", "std"),
            bound_mean=("gap_bound", "mean"),
            residual_mean=("offdiagonal_residual", "mean"),
        )
        .reset_index()
    )
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.7))
    axes[0].errorbar(
        summary["strength"],
        summary["commutator_mean"],
        yerr=summary["commutator_std"],
        marker="o",
        capsize=3,
        label="commutator",
    )
    axes[0].plot(
        summary["strength"],
        summary["residual_mean"],
        marker="s",
        label="DECA off-diagonal residual",
    )
    axes[0].set_xlabel("noncommuting rotation strength")
    axes[0].set_ylabel("operator mismatch")
    axes[0].legend(frameon=False)

    axes[1].errorbar(
        summary["strength"],
        summary["gap_mean"],
        yerr=summary["gap_std"],
        marker="o",
        capsize=3,
        label="POVM − DECA",
    )
    axes[1].plot(
        summary["strength"],
        summary["bound_mean"],
        linestyle="--",
        label=r"$\sqrt{d}R_{\mathrm{off}}$ bound",
    )
    axes[1].set_xlabel("noncommuting rotation strength")
    axes[1].set_ylabel("single-shot success gap")
    axes[1].legend(frameon=False)
    figure.tight_layout()
    figure.savefig(RESULTS / "noncommutativity_gap.png", dpi=220)
    figure.savefig(RESULTS / "noncommutativity_gap.pdf")
    plt.close(figure)
    return summary


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    binary = pd.DataFrame(binary_trials())
    commuting = pd.DataFrame(commuting_trials())
    noncommuting = pd.DataFrame(noncommuting_sweep())
    trine = trine_trial()

    binary.to_csv(RESULTS / "binary_exactness.csv", index=False)
    commuting.to_csv(RESULTS / "commuting_exactness.csv", index=False)
    noncommuting.to_csv(
        RESULTS / "noncommutativity_sweep.csv", index=False
    )
    noncommuting_summary = plot_noncommuting(noncommuting)
    noncommuting_summary.to_csv(
        RESULTS / "noncommutativity_summary.csv", index=False
    )

    summary = {
        "binary_trials": len(binary),
        "binary_max_closed_form_error": float(
            binary["closed_form_error"].max()
        ),
        "binary_max_sdp_gap": float(binary["absolute_gap"].max()),
        "commuting_trials": len(commuting),
        "commuting_max_sdp_gap": float(commuting["absolute_gap"].max()),
        "commuting_max_residual": float(
            commuting["offdiagonal_residual"].max()
        ),
        "noncommuting_trials": len(noncommuting),
        "noncommuting_bound_violations": int(
            np.sum(noncommuting["bound_slack"] < -1e-7)
        ),
        "noncommuting_gap_commutator_pearson": float(
            noncommuting["oracle_gap"].corr(noncommuting["commutator"])
        ),
        "trine": trine,
        "runtime_seconds": time.perf_counter() - started,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "cvxpy": cvxpy.__version__,
            "qiskit": qiskit.__version__,
        },
    }
    (RESULTS / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
