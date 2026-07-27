#!/usr/bin/env python3
"""Exact local-blindness and Wilson-flux benchmark on a 3x3 torus."""

from __future__ import annotations

import json
import math
import time
from itertools import combinations, product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from aoc import (
    ToricCodeLattice,
    binary_measurement_success,
    maximum_observable_contrast,
    pauli_string_expectation,
    reduced_density_on_qubits,
    toric_code_ground_state,
    z_parity_projectors,
)
from aoc.repro import write_manifest
from aoc.states import pure_state_density
from aoc.symmetry import invariant_observable_contrast, symmetry_sector_contrasts

RUN_ROOT = Path(__file__).resolve().parents[1]
RESULTS = RUN_ROOT / "results" / "topological_flux"
TOLERANCE = 1e-10


def pauli_masks(paulis: dict[int, str]) -> tuple[int, int]:
    x_mask = 0
    z_mask = 0
    for qubit, operator in paulis.items():
        if operator in {"X", "Y"}:
            x_mask |= 1 << qubit
        if operator in {"Z", "Y"}:
            z_mask |= 1 << qubit
    return x_mask, z_mask


def commutes_with_stabilizers(
    paulis: dict[int, str],
    lattice: ToricCodeLattice,
) -> bool:
    x_mask, z_mask = pauli_masks(paulis)
    return all(
        not (z_mask & sum(1 << edge for edge in star)).bit_count() % 2
        for star in lattice.all_stars()
    ) and all(
        not (x_mask & sum(1 << edge for edge in plaquette)).bit_count() % 2
        for plaquette in lattice.all_plaquettes()
    )


def scan_pauli_weights(
    lattice: ToricCodeLattice,
    positive: np.ndarray,
    negative: np.ndarray,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    positive_support = np.flatnonzero(positive)
    negative_support = np.flatnonzero(negative)
    summary_rows = []
    nonzero_rows: list[dict[str, object]] = []
    for weight in (1, 2, 3):
        count = math.comb(lattice.num_qubits, weight) * 3**weight
        nonzero_count = 0
        centralizer_count = 0
        max_gap = 0.0
        first_maximizer = ""
        for qubits in combinations(range(lattice.num_qubits), weight):
            for operators in product("XYZ", repeat=weight):
                paulis = dict(zip(qubits, operators, strict=True))
                if commutes_with_stabilizers(paulis, lattice):
                    centralizer_count += 1
                first = pauli_string_expectation(
                    positive,
                    paulis,
                    support=positive_support,
                )
                second = pauli_string_expectation(
                    negative,
                    paulis,
                    support=negative_support,
                )
                gap = float(abs(first - second))
                label = " ".join(
                    f"{operator}{qubit}"
                    for qubit, operator in zip(qubits, operators, strict=True)
                )
                if gap > max_gap + TOLERANCE:
                    max_gap = gap
                    first_maximizer = label
                if gap > TOLERANCE:
                    nonzero_count += 1
                    nonzero_rows.append(
                        {
                            "weight": weight,
                            "pauli": label,
                            "positive_expectation": float(np.real(first)),
                            "negative_expectation": float(np.real(second)),
                            "absolute_gap": gap,
                        }
                    )
        summary_rows.append(
            {
                "weight": weight,
                "pauli_count": count,
                "stabilizer_centralizer_count": centralizer_count,
                "nonzero_class_gap_count": nonzero_count,
                "max_absolute_expectation_gap": max_gap,
                "first_maximizer": first_maximizer,
            }
        )
    return pd.DataFrame(summary_rows), nonzero_rows


def trace_distance(first: np.ndarray, second: np.ndarray) -> float:
    values = np.linalg.eigvalsh(0.5 * (first - second + (first - second).conj().T))
    return float(np.sum(np.abs(values)) / 2.0)


def scan_subsets(
    lattice: ToricCodeLattice,
    positive: np.ndarray,
    negative: np.ndarray,
) -> pd.DataFrame:
    target_loops = {
        tuple(sorted(lattice.logical_z_x_edges(row=row))) for row in range(lattice.size)
    }
    rows = []
    for size in (1, 2, 3):
        for region in combinations(range(lattice.num_qubits), size):
            first = reduced_density_on_qubits(positive, region)
            second = reduced_density_on_qubits(negative, region)
            rows.append(
                {
                    "support_size": size,
                    "qubits": json.dumps(region),
                    "trace_distance": trace_distance(first, second),
                    "target_homology_loop": region in target_loops,
                }
            )
    return pd.DataFrame(rows)


def permutation_unitary(permutation: list[int]) -> np.ndarray:
    identity = np.eye(len(permutation), dtype=np.complex128)
    return identity[permutation]


def logical_measurement_benchmarks() -> tuple[pd.DataFrame, dict[str, float]]:
    identity = np.eye(4, dtype=np.complex128)
    nuisance_flip = permutation_unitary([1, 0, 3, 2])
    label_flip = permutation_unitary([2, 3, 0, 1])
    positive_vector = identity[:, 0]
    negative_vector = identity[:, 2]
    positive_sample = pure_state_density(positive_vector)
    negative_sample = pure_state_density(negative_vector)
    positive_test = np.diag([0.5, 0.5, 0.0, 0.0]).astype(np.complex128)
    negative_test = np.diag([0.0, 0.0, 0.5, 0.5]).astype(np.complex128)

    untwirled = maximum_observable_contrast(positive_sample, negative_sample)
    stabilizer_only = invariant_observable_contrast(
        positive_sample,
        negative_sample,
        (identity,),
    )
    correct = invariant_observable_contrast(
        positive_sample,
        negative_sample,
        (identity, nuisance_flip),
    )
    wrong = invariant_observable_contrast(
        positive_sample,
        negative_sample,
        (identity, label_flip),
    )
    logical_z = np.diag([1.0, 1.0, -1.0, -1.0])
    wilson_effect = (identity + logical_z) / 2.0
    methods = [
        (
            "Best fixed <=2-link / weight<=2 observer",
            0.5,
            "certified local blindness",
        ),
        (
            "Untwirled AOC, one representative per class",
            binary_measurement_success(
                untwirled.effect,
                positive_test,
                negative_test,
            ),
            "unknown nuisance logical sector at test time",
        ),
        (
            "Stabilizer-only twirl + AOC",
            binary_measurement_success(
                stabilizer_only.contrast.effect,
                positive_test,
                negative_test,
            ),
            "ground representatives are already stabilizer invariant",
        ),
        (
            "Correct label-preserving nuisance twirl + AOC",
            binary_measurement_success(
                correct.contrast.effect,
                positive_test,
                negative_test,
            ),
            "learned Wilson-equivalent sector projector",
        ),
        (
            "Wrong label-flipping twirl + AOC",
            binary_measurement_success(
                wrong.contrast.effect,
                positive_test,
                negative_test,
            ),
            "negative control: the twirl erases the label",
        ),
        (
            "Supplied Wilson-loop parity threshold",
            binary_measurement_success(
                wilson_effect,
                positive_test,
                negative_test,
            ),
            "physics oracle / equally informed scalar baseline",
        ),
        (
            "Full Helstrom oracle",
            1.0,
            "unrestricted equal-prior upper bound",
        ),
    ]
    rows = [
        {"method": name, "success_probability": value, "interpretation": note}
        for name, value, note in methods
    ]
    diagnostics = {
        "correct_twirl_invariance_error": correct.invariance_error,
        "wrong_twirl_trace_norm": wrong.contrast.trace_norm,
        "learned_sign_vs_logical_z_frobenius_error": float(
            np.linalg.norm(correct.contrast.sign_observable - logical_z, ord="fro")
        ),
        "learned_effect_vs_wilson_projector_frobenius_error": float(
            np.linalg.norm(correct.contrast.effect - wilson_effect, ord="fro")
        ),
    }
    return pd.DataFrame(rows), diagnostics


def robustness_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    logical_rows = []
    for mixing in np.linspace(0.0, 0.5, 11):
        positive = np.diag(
            [
                (1.0 - mixing) / 2.0,
                (1.0 - mixing) / 2.0,
                mixing / 2.0,
                mixing / 2.0,
            ]
        )
        negative = np.diag(
            [
                mixing / 2.0,
                mixing / 2.0,
                (1.0 - mixing) / 2.0,
                (1.0 - mixing) / 2.0,
            ]
        )
        result = maximum_observable_contrast(positive, negative)
        logical_rows.append(
            {
                "logical_sector_flip_probability": mixing,
                "numerical_trace_distance": result.trace_norm / 2.0,
                "analytic_trace_distance": 1.0 - 2.0 * mixing,
                "numerical_optimal_success": binary_measurement_success(
                    result.effect,
                    positive,
                    negative,
                ),
                "analytic_optimal_success": 1.0 - mixing,
            }
        )

    readout_rows = []
    for error in np.linspace(0.0, 0.2, 21):
        single = (1.0 + (1.0 - 2.0 * error) ** 3) / 2.0
        majority = 3.0 * single**2 - 2.0 * single**3
        readout_rows.append(
            {
                "independent_edge_readout_flip_probability": error,
                "single_length3_loop_success": single,
                "three_homologous_loops_majority_success": majority,
            }
        )
    return pd.DataFrame(logical_rows), pd.DataFrame(readout_rows)


def make_figure(
    logical_mixing: pd.DataFrame,
    readout: pd.DataFrame,
    path_pdf: Path,
    path_png: Path,
) -> None:
    sns.set_theme(style="whitegrid", context="paper")
    figure, axes = plt.subplots(1, 2, figsize=(8.2, 3.35))
    axes[0].plot(
        logical_mixing.logical_sector_flip_probability,
        logical_mixing.numerical_optimal_success,
        marker="o",
        label="AOC / Helstrom",
    )
    axes[0].plot(
        logical_mixing.logical_sector_flip_probability,
        logical_mixing.analytic_optimal_success,
        linestyle="--",
        label=r"analytic $1-p$",
    )
    axes[0].axhline(0.5, color="black", linestyle=":", linewidth=1)
    axes[0].set(
        xlabel="logical sector-flip probability",
        ylabel="equal-prior success",
        ylim=(0.48, 1.02),
    )
    axes[0].legend(fontsize=7)

    axes[1].plot(
        readout.independent_edge_readout_flip_probability,
        readout.single_length3_loop_success,
        label="one Wilson loop",
    )
    axes[1].plot(
        readout.independent_edge_readout_flip_probability,
        readout.three_homologous_loops_majority_success,
        label="three-loop majority",
    )
    axes[1].axhline(0.5, color="black", linestyle=":", linewidth=1)
    axes[1].set(
        xlabel="per-edge readout flip probability",
        ylabel="parity classification success",
        ylim=(0.48, 1.02),
    )
    axes[1].legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(path_pdf)
    figure.savefig(path_png, dpi=220)
    plt.close(figure)


def main() -> None:
    started = time.time()
    RESULTS.mkdir(parents=True, exist_ok=True)
    lattice = ToricCodeLattice(3)
    positive = toric_code_ground_state(lattice, logical_z_x=1, logical_z_y=1)
    negative = toric_code_ground_state(lattice, logical_z_x=-1, logical_z_y=1)

    all_states = [
        toric_code_ground_state(
            lattice,
            logical_z_x=logical_x,
            logical_z_y=logical_y,
        )
        for logical_x, logical_y in product((1, -1), repeat=2)
    ]
    overlaps = np.asarray(
        [[np.vdot(first, second) for second in all_states] for first in all_states]
    )
    maximum_stabilizer_error = 0.0
    for state in all_states:
        for edges in lattice.all_stars():
            value = pauli_string_expectation(
                state,
                {edge: "X" for edge in edges},
            )
            maximum_stabilizer_error = max(
                maximum_stabilizer_error,
                float(abs(value - 1.0)),
            )
        for edges in lattice.all_plaquettes():
            value = pauli_string_expectation(
                state,
                {edge: "Z" for edge in edges},
            )
            maximum_stabilizer_error = max(
                maximum_stabilizer_error,
                float(abs(value - 1.0)),
            )

    subset_frame = scan_subsets(lattice, positive, negative)
    subset_path = RESULTS / "subset_trace_distances.csv"
    subset_frame.to_csv(subset_path, index=False)

    pauli_frame, nonzero_paulis = scan_pauli_weights(
        lattice,
        positive,
        negative,
    )
    pauli_path = RESULTS / "pauli_weight_scan.csv"
    nonzero_path = RESULTS / "nonzero_pauli_witnesses.csv"
    pauli_frame.to_csv(pauli_path, index=False)
    pd.DataFrame(nonzero_paulis).to_csv(nonzero_path, index=False)

    loop = lattice.logical_z_x_edges()
    positive_loop = reduced_density_on_qubits(positive, loop)
    negative_loop = reduced_density_on_qubits(negative, loop)
    loop_result = maximum_observable_contrast(positive_loop, negative_loop)
    even, odd = z_parity_projectors(lattice.size)
    sectors = symmetry_sector_contrasts(
        loop_result.contrast,
        {"W=+1": even, "W=-1": odd},
    )

    benchmarks, witness_diagnostics = logical_measurement_benchmarks()
    benchmark_path = RESULTS / "measurement_benchmarks.csv"
    benchmarks.to_csv(benchmark_path, index=False)

    logical_mixing, readout = robustness_tables()
    logical_mixing_path = RESULTS / "logical_sector_mixing.csv"
    readout_path = RESULTS / "readout_robustness.csv"
    logical_mixing.to_csv(logical_mixing_path, index=False)
    readout.to_csv(readout_path, index=False)
    figure_pdf = RESULTS / "robustness.pdf"
    figure_png = RESULTS / "robustness.png"
    make_figure(logical_mixing, readout, figure_pdf, figure_png)

    local_summary = (
        subset_frame.groupby("support_size")
        .trace_distance.agg(["count", "max"])
        .reset_index()
    )
    below_distance = subset_frame[subset_frame.support_size < lattice.size]
    at_distance = subset_frame[subset_frame.support_size == lattice.size]
    summary = {
        "model": "L=3 toric-code / D(Z2) fixed point on a torus",
        "num_link_qubits": lattice.num_qubits,
        "hilbert_dimension": len(positive),
        "ground_state_support": int(np.count_nonzero(positive)),
        "ground_sector_orthogonality_error": float(
            np.linalg.norm(overlaps - np.eye(4), ord="fro")
        ),
        "maximum_stabilizer_expectation_error": maximum_stabilizer_error,
        "pauli_features_below_distance": int(
            pauli_frame.loc[
                pauli_frame.weight < lattice.size,
                "pauli_count",
            ].sum()
        ),
        "nonzero_pauli_gaps_below_distance": int(
            pauli_frame.loc[
                pauli_frame.weight < lattice.size,
                "nonzero_class_gap_count",
            ].sum()
        ),
        "weight3_pauli_count": int(
            pauli_frame.loc[pauli_frame.weight == lattice.size, "pauli_count"].iloc[0]
        ),
        "weight3_centralizer_count": int(
            pauli_frame.loc[
                pauli_frame.weight == lattice.size,
                "stabilizer_centralizer_count",
            ].iloc[0]
        ),
        "weight3_nonzero_class_gap_count": int(
            pauli_frame.loc[
                pauli_frame.weight == lattice.size,
                "nonzero_class_gap_count",
            ].iloc[0]
        ),
        "max_trace_distance_below_distance": float(below_distance.trace_distance.max()),
        "distinguishable_weight3_subsets": int(
            (at_distance.trace_distance > TOLERANCE).sum()
        ),
        "loop_trace_distance": loop_result.trace_norm / 2.0,
        "loop_effect_vs_parity_projector_error": float(
            np.linalg.norm(loop_result.effect - even, ord="fro")
        ),
        "sector_trace_norms": {sector.name: sector.trace_norm for sector in sectors},
        "subset_trace_distance_by_size": local_summary.to_dict(orient="records"),
        "measurement_success": dict(
            zip(
                benchmarks.method,
                benchmarks.success_probability,
                strict=True,
            )
        ),
        "witness_diagnostics": witness_diagnostics,
        "logical_mixing_max_analytic_error": float(
            np.max(
                np.abs(
                    logical_mixing.numerical_optimal_success
                    - logical_mixing.analytic_optimal_success
                )
            )
        ),
        "readout_example_r_0_05": readout.loc[
            np.isclose(
                readout.independent_edge_readout_flip_probability,
                0.05,
            ),
            [
                "single_length3_loop_success",
                "three_homologous_loops_majority_success",
            ],
        ]
        .iloc[0]
        .to_dict(),
        "advantage_conclusion": (
            "The run proves an observable-access and symmetry-prior advantage "
            "over every fixed sub-distance observer and over untwirled "
            "one-representative learning. It does not show superiority over "
            "Helstrom, a supplied Wilson-loop threshold, or another method "
            "given the same topological feature; all correctly informed "
            "methods attain success 1 at the fixed point."
        ),
        "factorization_convention": (
            "Reduced states use the tensor product of link qubits "
            "(extended-link/electric-center prescription)."
        ),
    }
    summary_path = RESULTS / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    outputs = [
        subset_path,
        pauli_path,
        nonzero_path,
        benchmark_path,
        logical_mixing_path,
        readout_path,
        figure_pdf,
        figure_png,
        summary_path,
    ]
    write_manifest(
        RESULTS / "manifest.json",
        experiment="run4-toric-code-topological-flux",
        started_at=started,
        config={
            "lattice_size": lattice.size,
            "num_link_qubits": lattice.num_qubits,
            "logical_z_y": 1,
            "compared_logical_z_x": [1, -1],
            "maximum_pauli_weight": 3,
            "maximum_subset_size": 3,
            "logical_mixing_grid": [0.0, 0.5, 11],
            "readout_error_grid": [0.0, 0.2, 21],
            "tolerance": TOLERANCE,
        },
        outputs=outputs,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
