#!/usr/bin/env python3
"""Run ancilla-free DECA and Naimark POVM circuits on Qiskit Aer."""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import qiskit
import qiskit_aer

from deca.jacobi import jacobi_deca
from deca.quantum import (
    simulate_povm_measurement,
    simulate_projective_measurement,
)
from deca.solvers import binary_helstrom, optimal_povm


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "quantum"


def binary_problem():
    angles_class_0 = np.array([-0.28, -0.08, 0.10, 0.25])
    angles_class_1 = np.array([1.02, 1.20, 1.36, 1.54])
    class_0 = np.column_stack(
        [np.cos(angles_class_0), np.sin(angles_class_0)]
    )
    class_1 = np.column_stack(
        [np.cos(angles_class_1), np.sin(angles_class_1)]
    )
    rho_0 = sum(np.outer(x, x) for x in class_0) / len(class_0)
    rho_1 = sum(np.outer(x, x) for x in class_1) / len(class_1)
    solution = binary_helstrom([0.5 * rho_0, 0.5 * rho_1])
    test_states = np.vstack([class_0, class_1])
    true_classes = np.array([0] * len(class_0) + [1] * len(class_1))
    return solution, test_states, true_classes


def trine_problem():
    phases = 2.0 * np.pi * np.arange(3) / 3.0
    states = np.column_stack(
        [
            np.full(3, 1.0 / np.sqrt(2.0), dtype=np.complex128),
            np.exp(1j * phases) / np.sqrt(2.0),
        ]
    )
    operators = [
        np.outer(state, state.conj()) / 3.0
        for state in states
    ]
    deca = jacobi_deca(
        operators, random_starts=24, max_sweeps=100, random_state=31
    )
    oracle = optimal_povm(operators, solver="SCS")
    return states, operators, deca, oracle


def binary_circuits(shots):
    solution, states, true_classes = binary_problem()
    records = []
    for index, (state, true_class) in enumerate(zip(states, true_classes)):
        result = simulate_projective_measurement(
            state,
            solution.basis,
            solution.assignment,
            shots=shots,
            seed=100 + index,
            optimization_level=1,
        )
        record = {
            "problem": "binary_helstrom",
            "state_index": index,
            "true_class": int(true_class),
            "measurement": "ancilla_free_pvm",
            "shots": shots,
            "system_qubits": result.num_system_qubits,
            "ancilla_qubits": result.num_ancilla_qubits,
            "analytic_true_probability": result.analytic_probabilities[
                true_class
            ],
            "shot_true_probability": result.shot_probabilities[true_class],
            "total_variation_error": result.total_variation_error,
            "transpiled_depth": result.transpiled_depth,
            "transpiled_size": result.transpiled_size,
            "operation_counts": json.dumps(result.operation_counts),
        }
        records.append(record)
    return records, solution


def trine_circuits(shots):
    states, operators, deca, oracle = trine_problem()
    records = []
    for index, state in enumerate(states):
        pvm_result = simulate_projective_measurement(
            state,
            deca.basis,
            deca.assignment,
            shots=shots,
            seed=200 + index,
            optimization_level=1,
        )
        povm_result = simulate_povm_measurement(
            state,
            oracle.effects,
            shots=shots,
            seed=300 + index,
            optimization_level=1,
        )
        for label, result in (
            ("ancilla_free_pvm", pvm_result),
            ("naimark_optimal_povm", povm_result),
        ):
            records.append(
                {
                    "problem": "trine",
                    "state_index": index,
                    "true_class": index,
                    "measurement": label,
                    "shots": shots,
                    "system_qubits": result.num_system_qubits,
                    "ancilla_qubits": result.num_ancilla_qubits,
                    "analytic_true_probability": result.analytic_probabilities[
                        index
                    ],
                    "shot_true_probability": result.shot_probabilities[index],
                    "total_variation_error": result.total_variation_error,
                    "transpiled_depth": result.transpiled_depth,
                    "transpiled_size": result.transpiled_size,
                    "operation_counts": json.dumps(result.operation_counts),
                }
            )
    return records, deca, oracle


def make_plot(frame):
    grouped = (
        frame.groupby(["problem", "measurement"])
        .agg(
            analytic_success=("analytic_true_probability", "mean"),
            shot_success=("shot_true_probability", "mean"),
            tv_error=("total_variation_error", "mean"),
            ancillas=("ancilla_qubits", "max"),
            depth=("transpiled_depth", "max"),
        )
        .reset_index()
    )
    positions = np.arange(len(grouped))
    width = 0.36
    figure, axis = plt.subplots(figsize=(8.2, 3.8))
    axis.bar(
        positions - width / 2,
        grouped["analytic_success"],
        width,
        label="analytic",
    )
    axis.bar(
        positions + width / 2,
        grouped["shot_success"],
        width,
        label="Aer shots",
    )
    axis.set_xticks(
        positions,
        [
            f"{row.problem}\n{row.measurement}"
            for row in grouped.itertuples()
        ],
        rotation=12,
        ha="right",
    )
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("mean true-outcome probability")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(RESULTS / "quantum_simulation_success.png", dpi=220)
    figure.savefig(RESULTS / "quantum_simulation_success.pdf")
    plt.close(figure)
    return grouped


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    shots = 16_384
    started = time.perf_counter()
    binary_records, binary_solution = binary_circuits(shots)
    trine_records, trine_deca, trine_oracle = trine_circuits(shots)
    frame = pd.DataFrame(binary_records + trine_records)
    frame.to_csv(RESULTS / "quantum_simulation_records.csv", index=False)
    summary_frame = make_plot(frame)
    summary_frame.to_csv(
        RESULTS / "quantum_simulation_summary.csv", index=False
    )

    summary = {
        "shots_per_state": shots,
        "max_total_variation_error": float(
            frame["total_variation_error"].max()
        ),
        "binary_analytic_training_success": binary_solution.success,
        "trine_deca_training_success": trine_deca.success,
        "trine_optimal_povm_training_success": trine_oracle.success,
        "trine_povm_advantage": trine_oracle.success - trine_deca.success,
        "runtime_seconds": time.perf_counter() - started,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "qiskit": qiskit.__version__,
            "qiskit_aer": qiskit_aer.__version__,
        },
    }
    (RESULTS / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
