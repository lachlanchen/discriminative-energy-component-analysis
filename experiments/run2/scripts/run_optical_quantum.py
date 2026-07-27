#!/usr/bin/env python3
"""Optimal polarization contrast and finite-shot Qiskit verification."""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from aoc import maximum_observable_contrast
from aoc.quantum import simulate_density_effect
from aoc.repro import write_manifest

RUN_ROOT = Path(__file__).resolve().parents[1]
RESULTS = RUN_ROOT / "results" / "optics"


def main():
    started = time.time()
    RESULTS.mkdir(parents=True, exist_ok=True)
    horizontal = np.array([1.0, 0.0])
    diagonal = np.array([1.0, 1.0]) / np.sqrt(2.0)
    antidiagonal = np.array([1.0, -1.0]) / np.sqrt(2.0)
    identity = np.eye(2)
    visibility = 0.90
    rho_d = (
        visibility * np.outer(diagonal, diagonal) + (1.0 - visibility) * identity / 2
    )
    rho_a = (
        visibility * np.outer(antidiagonal, antidiagonal)
        + (1.0 - visibility) * identity / 2
    )
    result = maximum_observable_contrast(rho_d, rho_a)
    optimal_effect = result.effect
    horizontal_effect = np.outer(horizontal, horizontal)

    rows = []
    circuit_text = None
    for seed in range(20):
        for label, density in (("D", rho_d), ("A", rho_a)):
            simulation = simulate_density_effect(
                density,
                optimal_effect,
                shots=16384,
                seed=seed,
            )
            rows.append(
                {
                    "seed": seed,
                    "state": label,
                    "shots": simulation.shots,
                    "shot_probability": simulation.shot_probability,
                    "analytic_probability": simulation.analytic_probability,
                    "absolute_error": simulation.absolute_error,
                    "qubits": simulation.qubits,
                    "circuit_depth": simulation.circuit_depth,
                    "transpiled_depth": simulation.transpiled_depth,
                    "transpiled_size": simulation.transpiled_size,
                }
            )
            if circuit_text is None:
                circuit_text = str(simulation.circuit.draw(output="text"))
    records = pd.DataFrame(rows)
    records_path = RESULTS / "quantum_shots.csv"
    records.to_csv(records_path, index=False)
    circuit_path = RESULTS / "circuit.txt"
    clean_circuit = "\n".join(line.rstrip() for line in circuit_text.splitlines())
    circuit_path.write_text(clean_circuit + "\n", encoding="utf-8")

    analytic_optimal_success = 0.5 * (
        np.trace(optimal_effect @ rho_d).real
        + 1.0
        - np.trace(optimal_effect @ rho_a).real
    )
    analytic_horizontal_success = 0.5 * (
        np.trace(horizontal_effect @ rho_d).real
        + 1.0
        - np.trace(horizontal_effect @ rho_a).real
    )
    shot_success = []
    for seed, group in records.groupby("seed"):
        probabilities = group.set_index("state").shot_probability
        shot_success.append(
            {
                "seed": int(seed),
                "success": 0.5 * (probabilities["D"] + 1.0 - probabilities["A"]),
            }
        )
    shot_success = pd.DataFrame(shot_success)
    shot_success_path = RESULTS / "shot_success.csv"
    shot_success.to_csv(shot_success_path, index=False)

    pauli_x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    pauli_y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    pauli_z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    bloch = [
        float(np.trace(optimal_effect @ pauli).real)
        for pauli in (pauli_x, pauli_y, pauli_z)
    ]

    sns.set_theme(style="whitegrid", context="paper")
    figure, axes = plt.subplots(1, 2, figsize=(7.6, 3.0))
    sns.boxplot(
        data=records,
        x="state",
        y="shot_probability",
        ax=axes[0],
        color="#6c8ebf",
    )
    axes[0].set_ylabel(r"measured $\Pr(E^\star)$")
    axes[0].set_xlabel("polarization state")
    comparison = pd.DataFrame(
        {
            "analyzer": ["fixed H/V", "learned Helstrom"],
            "success": [
                analytic_horizontal_success,
                analytic_optimal_success,
            ],
        }
    )
    sns.barplot(
        data=comparison,
        x="analyzer",
        y="success",
        ax=axes[1],
        color="#3567a8",
    )
    axes[1].set_ylim(0.45, 1.0)
    axes[1].set_ylabel("single-copy discrimination success")
    axes[1].set_xlabel("")
    axes[1].tick_params(axis="x", rotation=12)
    figure.tight_layout()
    figure_paths = [
        RESULTS / "optical_quantum.pdf",
        RESULTS / "optical_quantum.png",
    ]
    figure.savefig(figure_paths[0])
    figure.savefig(figure_paths[1], dpi=220)
    plt.close(figure)

    summary = {
        "visibility": visibility,
        "trace_distance": result.trace_norm / 2,
        "helstrom_success": analytic_optimal_success,
        "fixed_horizontal_success": analytic_horizontal_success,
        "finite_shot_success_mean": float(shot_success.success.mean()),
        "finite_shot_success_std": float(shot_success.success.std(ddof=1)),
        "maximum_probability_absolute_error": float(records.absolute_error.max()),
        "optimal_analyzer_bloch_vector": bloch,
        "qubits": int(records.qubits.max()),
        "transpiled_depth": int(records.transpiled_depth.max()),
    }
    summary_path = RESULTS / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_manifest(
        RESULTS / "manifest.json",
        experiment="run2_optical_quantum",
        started_at=started,
        config={
            "visibility": visibility,
            "shots_per_state": 16384,
            "seeds": 20,
            "simulator": "qiskit_aer_density_matrix_instruction",
        },
        outputs=[
            records_path,
            shot_success_path,
            circuit_path,
            *figure_paths,
            summary_path,
        ],
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
