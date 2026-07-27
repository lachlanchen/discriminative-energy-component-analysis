#!/usr/bin/env python3
"""Matched-copy native-Z versus genuine local-Pauli shadow measurement audit.

The quantum state in this controlled audit is diagonal in the computational
basis.  Each state copy is sampled from the exact periodic syndrome model.
Native measurement reads every qubit in Z.  The shadow policy independently
chooses X, Y, or Z on every qubit, which is the standard random local-Clifford
Pauli measurement ensemble.  X/Y outcomes of a computational-basis component
are independent fair signs; Z outcomes retain the component's eigenvalue.

For a declared pair observable Z_i Z_j, the local inverse channel gives the
single-copy estimator

    9 * 1{basis_i = basis_j = Z} * outcome_i * outcome_j.

The factor nine is three per supported qubit.  Native all-Z instead uses
``outcome_i * outcome_j`` on every copy.  The comparison measures the cost of
measurement universality for a known diagonal target bank.  It is not an
AOC-superiority result and does not reproduce classical-shadow sequential
change detection.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from aoc.repro import sha256_file, write_manifest
from aoc.surface_code import PeriodicSurfaceSyndromeModel
from numpy.typing import ArrayLike, NDArray

RUN_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = RUN_ROOT / "configs" / "shadow_measurement_locked.json"
BASIS_LABELS = np.asarray(["X", "Y", "Z"])
LOCAL_INVERSE_FACTOR = 3
ZZ_INVERSE_FACTOR = LOCAL_INVERSE_FACTOR**2


@dataclass(frozen=True)
class PairBank:
    indices: NDArray[np.int64]
    metadata: pd.DataFrame


@dataclass(frozen=True)
class MeasurementCopies:
    native_z: NDArray[np.int8]
    shadow_bases: NDArray[np.int8]
    shadow_outcomes: NDArray[np.int8]
    syndromes: NDArray[np.uint8]


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        config = json.load(stream)
    if config.get("single_qubit_inverse_channel_factor") != LOCAL_INVERSE_FACTOR:
        raise ValueError("The local-Pauli inverse-channel factor must be 3.")
    if config.get("zz_inverse_channel_factor") != ZZ_INVERSE_FACTOR:
        raise ValueError("The two-qubit ZZ inverse-channel factor must be 9.")
    probabilities = config.get("local_pauli_basis_probabilities", {})
    if set(probabilities) != {"X", "Y", "Z"} or not all(
        np.isclose(probabilities[label], 1.0 / 3.0)
        for label in ("X", "Y", "Z")
    ):
        raise ValueError("The locked audit requires uniform local X/Y/Z bases.")
    budgets = np.asarray(config["copy_budgets"], dtype=np.int64)
    if np.any(budgets <= 0) or np.any(np.diff(budgets) <= 0):
        raise ValueError("copy_budgets must be strictly increasing and positive.")
    if int(config["repetitions"]) <= 1:
        raise ValueError("At least two repetitions are required.")
    return config


def declared_pair_bank(
    size: int,
    displacements: list[list[int]],
) -> PairBank:
    """Return translated directed pairs for the declared positive displacements."""

    pairs: list[tuple[int, int]] = []
    rows: list[dict[str, int | str]] = []
    for displacement_index, (delta_row, delta_column) in enumerate(displacements):
        if (delta_row, delta_column) == (0, 0):
            raise ValueError("Pair displacement must be nonzero.")
        for row in range(size):
            for column in range(size):
                first = row * size + column
                second = (
                    (row + int(delta_row)) % size * size
                    + (column + int(delta_column)) % size
                )
                pairs.append((first, second))
                rows.append(
                    {
                        "observable": f"ZZ_d{displacement_index}_r{row}_c{column}",
                        "displacement_index": displacement_index,
                        "delta_row": int(delta_row),
                        "delta_column": int(delta_column),
                        "start_row": row,
                        "start_column": column,
                        "first_detector": first,
                        "second_detector": second,
                    }
                )
    indices = np.asarray(pairs, dtype=np.int64)
    if np.any(indices[:, 0] == indices[:, 1]):
        raise ValueError("Every declared ZZ observable must use two detectors.")
    return PairBank(indices=indices, metadata=pd.DataFrame(rows))


def exact_zz_expectations(
    model: PeriodicSurfaceSyndromeModel,
    q: float,
    pairs: ArrayLike,
) -> NDArray[np.float64]:
    """Return exact post-BSC expectations for the declared distinct ZZ pairs."""

    pair_indices = np.asarray(pairs, dtype=np.int64)
    if pair_indices.ndim != 2 or pair_indices.shape[1] != 2:
        raise ValueError("pairs must have shape (num_pairs, 2).")
    if np.any(pair_indices < 0) or np.any(pair_indices >= model.num_detectors):
        raise ValueError("pair index lies outside the detector lattice.")
    if np.any(pair_indices[:, 0] == pair_indices[:, 1]):
        raise ValueError("exact_zz_expectations requires distinct detector pairs.")
    mixture = model._validate_mixture(q)
    clean_components = []
    for length in (1, 2):
        templates = model.template_table(length)
        signed = 1.0 - 2.0 * templates.astype(np.float64)
        clean_components.append(
            (
                signed[:, pair_indices[:, 0]]
                * signed[:, pair_indices[:, 1]]
            ).mean(axis=0)
        )
    clean = (
        (1.0 - model.event_probability)
        + model.event_probability
        * ((1.0 - mixture) * clean_components[0] + mixture * clean_components[1])
    )
    return (1.0 - 2.0 * model.readout_error) ** 2 * clean


def simulate_measurement_copies(
    model: PeriodicSurfaceSyndromeModel,
    q: float,
    copies: int,
    *,
    rng: np.random.Generator,
) -> MeasurementCopies:
    """Simulate native-Z and uniform local-Pauli measurements on matched copies."""

    if copies < 0:
        raise ValueError("copies must be nonnegative.")
    syndromes = model.sample_spatial(copies, q, rng=rng)
    native_z = (1 - 2 * syndromes.astype(np.int8)).astype(np.int8)
    bases = rng.integers(
        0,
        3,
        size=(copies, model.num_detectors),
        dtype=np.int8,
    )
    fair_signs = (
        1
        - 2
        * rng.integers(
            0,
            2,
            size=(copies, model.num_detectors),
            dtype=np.int8,
        )
    ).astype(np.int8)
    outcomes = np.where(bases == 2, native_z, fair_signs).astype(np.int8)
    return MeasurementCopies(
        native_z=native_z,
        shadow_bases=bases,
        shadow_outcomes=outcomes,
        syndromes=syndromes,
    )


def native_zz_estimators(
    z_outcomes: ArrayLike,
    pairs: ArrayLike,
) -> NDArray[np.float64]:
    """Single-copy native all-Z estimators for a bank of ZZ observables."""

    outcomes = np.asarray(z_outcomes, dtype=np.int8)
    pair_indices = np.asarray(pairs, dtype=np.int64)
    return (
        outcomes[:, pair_indices[:, 0]]
        * outcomes[:, pair_indices[:, 1]]
    ).astype(np.float64)


def shadow_zz_estimators(
    bases: ArrayLike,
    outcomes: ArrayLike,
    pairs: ArrayLike,
) -> NDArray[np.float64]:
    """Single-copy local-Pauli shadow estimators with inverse factor nine."""

    basis_array = np.asarray(bases, dtype=np.int8)
    outcome_array = np.asarray(outcomes, dtype=np.int8)
    pair_indices = np.asarray(pairs, dtype=np.int64)
    if basis_array.shape != outcome_array.shape or basis_array.ndim != 2:
        raise ValueError("bases and outcomes must have the same matrix shape.")
    selected = (basis_array[:, pair_indices[:, 0]] == 2) & (
        basis_array[:, pair_indices[:, 1]] == 2
    )
    products = (
        outcome_array[:, pair_indices[:, 0]]
        * outcome_array[:, pair_indices[:, 1]]
    )
    return (ZZ_INVERSE_FACTOR * selected * products).astype(np.float64)


def snapshot_audit_rows(
    copies: MeasurementCopies,
    pairs: NDArray[np.int64],
    count: int,
) -> list[dict[str, Any]]:
    rows = []
    for index in range(min(count, len(copies.syndromes))):
        bases = copies.shadow_bases[index]
        outcomes = copies.shadow_outcomes[index]
        measured = (bases[pairs[:, 0]] == 2) & (bases[pairs[:, 1]] == 2)
        rows.append(
            {
                "copy": index,
                "syndrome_count": int(copies.syndromes[index].sum()),
                "native_z_bitstring": "".join(
                    "0" if value == 1 else "1" for value in copies.native_z[index]
                ),
                "local_pauli_bases": "".join(BASIS_LABELS[bases]),
                "local_pauli_outcomes": "".join(
                    "+" if value == 1 else "-" for value in outcomes
                ),
                "declared_zz_observables_sampled": int(measured.sum()),
                "declared_zz_observables": len(pairs),
            }
        )
    return rows


def run_audit(config: dict[str, Any]):
    model_config = config["model"]
    model = PeriodicSurfaceSyndromeModel(
        size=int(model_config["size"]),
        event_probability=float(model_config["event_probability"]),
        readout_error=float(model_config["readout_error"]),
    )
    pair_bank = declared_pair_bank(
        model.size,
        config["target_displacements"],
    )
    budgets = [int(value) for value in config["copy_budgets"]]
    repetitions = int(config["repetitions"])
    base_seed = int(config["seed"])
    replicate_rows = []
    observable_accumulators: dict[
        tuple[str, int, str],
        dict[str, NDArray[np.float64]],
    ] = {}
    snapshot_rows: list[dict[str, Any]] = []
    basis_counts = np.zeros(3, dtype=np.int64)

    for state_index, state_config in enumerate(config["states"]):
        state_name = str(state_config["name"])
        q = float(state_config["chain2_probability"])
        exact = exact_zz_expectations(model, q, pair_bank.indices)
        native_single_variance = 1.0 - exact**2
        shadow_single_variance = ZZ_INVERSE_FACTOR - exact**2
        for repetition in range(repetitions):
            seed = base_seed + state_index * 1_000_000 + repetition
            rng = np.random.default_rng(seed)
            copies = simulate_measurement_copies(
                model,
                q,
                budgets[-1],
                rng=rng,
            )
            basis_counts += np.bincount(
                copies.shadow_bases.ravel(),
                minlength=3,
            )
            native = native_zz_estimators(copies.native_z, pair_bank.indices)
            shadow = shadow_zz_estimators(
                copies.shadow_bases,
                copies.shadow_outcomes,
                pair_bank.indices,
            )
            native_cumulative = np.cumsum(native, axis=0)
            shadow_cumulative = np.cumsum(shadow, axis=0)
            if state_index == 0 and repetition == 0:
                snapshot_rows = snapshot_audit_rows(
                    copies,
                    pair_bank.indices,
                    int(config["snapshot_audit_copies"]),
                )
            for budget in budgets:
                estimates_by_method = {
                    "native_all_z": native_cumulative[budget - 1] / budget,
                    "local_pauli_shadow": shadow_cumulative[budget - 1] / budget,
                }
                for method, estimate in estimates_by_method.items():
                    error = estimate - exact
                    replicate_rows.append(
                        {
                            "state": state_name,
                            "q": q,
                            "repetition": repetition,
                            "seed": seed,
                            "copies": budget,
                            "method": method,
                            "pair_observables": len(pair_bank.indices),
                            "rmse": float(np.sqrt(np.mean(error**2))),
                            "mae": float(np.mean(np.abs(error))),
                            "mean_bias": float(np.mean(error)),
                            "maximum_absolute_error": float(np.max(np.abs(error))),
                            "mean_exact_expectation": float(exact.mean()),
                            "mean_estimate": float(estimate.mean()),
                        }
                    )
                    key = (state_name, budget, method)
                    if key not in observable_accumulators:
                        observable_accumulators[key] = {
                            "sum_estimate": np.zeros_like(exact),
                            "sum_squared_error": np.zeros_like(exact),
                            "sum_error": np.zeros_like(exact),
                        }
                    accumulator = observable_accumulators[key]
                    accumulator["sum_estimate"] += estimate
                    accumulator["sum_squared_error"] += error**2
                    accumulator["sum_error"] += error
        for budget in budgets:
            for method, variance in (
                ("native_all_z", native_single_variance),
                ("local_pauli_shadow", shadow_single_variance),
            ):
                observable_accumulators[(state_name, budget, method)][
                    "theoretical_estimator_variance"
                ] = variance / budget

    replicate_frame = pd.DataFrame(replicate_rows)
    observable_rows = []
    metadata = pair_bank.metadata
    state_lookup = {
        str(item["name"]): float(item["chain2_probability"])
        for item in config["states"]
    }
    exact_lookup = {
        state: exact_zz_expectations(model, q, pair_bank.indices)
        for state, q in state_lookup.items()
    }
    for (state, budget, method), accumulator in observable_accumulators.items():
        exact = exact_lookup[state]
        for observable_index, metadata_row in metadata.iterrows():
            observable_rows.append(
                {
                    **metadata_row.to_dict(),
                    "state": state,
                    "q": state_lookup[state],
                    "copies": budget,
                    "method": method,
                    "exact_expectation": exact[observable_index],
                    "mean_estimate": (
                        accumulator["sum_estimate"][observable_index] / repetitions
                    ),
                    "bias": accumulator["sum_error"][observable_index] / repetitions,
                    "rmse": np.sqrt(
                        accumulator["sum_squared_error"][observable_index] / repetitions
                    ),
                    "theoretical_estimator_standard_error": np.sqrt(
                        accumulator["theoretical_estimator_variance"][
                            observable_index
                        ]
                    ),
                }
            )
    observable_frame = pd.DataFrame(observable_rows)
    aggregate = (
        replicate_frame.groupby(["state", "q", "copies", "method"], as_index=False)
        .agg(
            mean_rmse=("rmse", "mean"),
            standard_deviation_rmse=("rmse", "std"),
            mean_mae=("mae", "mean"),
            mean_maximum_absolute_error=("maximum_absolute_error", "mean"),
            mean_bias=("mean_bias", "mean"),
        )
        .sort_values(["state", "copies", "method"])
    )
    aggregate["rmse_ci95_half_width"] = (
        1.96 * aggregate.standard_deviation_rmse / np.sqrt(repetitions)
    )
    aggregate["copies_per_declared_observable"] = aggregate.copies
    aggregate["expected_shadow_zz_hits_per_observable"] = aggregate.copies / 9.0
    return (
        model,
        pair_bank,
        replicate_frame,
        observable_frame,
        aggregate,
        pd.DataFrame(snapshot_rows),
        basis_counts,
    )


def make_figure(aggregate: pd.DataFrame, output_pdf: Path, output_png: Path) -> None:
    sns.set_theme(style="whitegrid", context="paper")
    figure, axes = plt.subplots(1, 2, figsize=(8.4, 3.35))
    palette = {
        "native_all_z": "#3567a8",
        "local_pauli_shadow": "#c66b32",
    }
    combined = (
        aggregate.groupby(["copies", "method"], as_index=False)
        .agg(
            mean_rmse=("mean_rmse", "mean"),
            ci=("rmse_ci95_half_width", "mean"),
        )
    )
    for method, group in combined.groupby("method"):
        group = group.sort_values("copies")
        axes[0].plot(
            group.copies,
            group.mean_rmse,
            marker="o",
            label=method.replace("_", " "),
            color=palette[method],
        )
        axes[0].fill_between(
            group.copies,
            group.mean_rmse - group.ci,
            group.mean_rmse + group.ci,
            color=palette[method],
            alpha=0.18,
        )
    axes[0].set_xscale("log", base=2)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("state copies")
    axes[0].set_ylabel("pair-bank RMSE")
    axes[0].legend(frameon=False, fontsize=8)

    largest = aggregate[aggregate.copies == aggregate.copies.max()].copy()
    largest["measurement"] = largest.method.map(
        {
            "native_all_z": "native all-Z",
            "local_pauli_shadow": "local-Pauli shadow",
        }
    )
    largest["state_label"] = largest.q.map(lambda value: f"q={value:.2f}")
    sns.barplot(
        data=largest,
        x="state_label",
        y="mean_rmse",
        hue="measurement",
        palette={
            "native all-Z": palette["native_all_z"],
            "local-Pauli shadow": palette["local_pauli_shadow"],
        },
        ax=axes[1],
    )
    axes[1].set_xlabel("")
    axes[1].set_ylabel(f"RMSE at {int(aggregate.copies.max())} copies")
    axes[1].legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(output_pdf)
    figure.savefig(output_png, dpi=220)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    arguments = parser.parse_args()
    config_path = arguments.config.resolve()
    config = load_config(config_path)
    started = time.time()
    output = RUN_ROOT / "results" / str(config["output_directory"])
    output.mkdir(parents=True, exist_ok=True)
    (
        model,
        pair_bank,
        replicate_frame,
        observable_frame,
        aggregate,
        snapshot_frame,
        basis_counts,
    ) = run_audit(config)

    replicate_path = output / "replicate_metrics.csv"
    observable_path = output / "observable_metrics.csv"
    aggregate_path = output / "aggregate_metrics.csv"
    snapshot_path = output / "snapshot_audit.csv"
    pair_path = output / "declared_observables.csv"
    replicate_frame.to_csv(replicate_path, index=False)
    observable_frame.to_csv(observable_path, index=False)
    aggregate.to_csv(aggregate_path, index=False)
    snapshot_frame.to_csv(snapshot_path, index=False)
    pair_bank.metadata.to_csv(pair_path, index=False)

    figure_pdf = output / "shadow_measurement_audit.pdf"
    figure_png = output / "shadow_measurement_audit.png"
    make_figure(aggregate, figure_pdf, figure_png)

    largest_budget = int(max(config["copy_budgets"]))
    largest = aggregate[aggregate.copies == largest_budget]
    comparison = largest.pivot(
        index="state",
        columns="method",
        values="mean_rmse",
    )
    ratios = (
        comparison.local_pauli_shadow / comparison.native_all_z
    ).to_dict()
    total_basis = int(basis_counts.sum())
    basis_fractions = {
        str(label): float(basis_counts[index] / total_basis)
        for index, label in enumerate(BASIS_LABELS)
    }
    summary = {
        "experiment": "genuine local-Pauli classical-shadow measurement audit",
        "locked_config": str(config_path.relative_to(RUN_ROOT.parent.parent)),
        "locked_config_sha256": sha256_file(config_path),
        "model": {
            "size": model.size,
            "detectors": model.num_detectors,
            "event_probability": model.event_probability,
            "readout_error": model.readout_error,
            "states": config["states"],
        },
        "measurement_policies": {
            "native_all_z": (
                "Every copy is measured in Z and contributes to every declared ZZ."
            ),
            "local_pauli_shadow": (
                "Each qubit independently uses uniform X/Y/Z local-Clifford "
                "measurement; ZZ uses estimator 9 I[Z,Z] b_i b_j."
            ),
        },
        "single_qubit_inverse_channel_factor": LOCAL_INVERSE_FACTOR,
        "zz_inverse_channel_factor": ZZ_INVERSE_FACTOR,
        "native_zz_contribution_probability_per_copy": 1.0,
        "shadow_zz_contribution_probability_per_copy": 1.0 / 9.0,
        "declared_pair_observables": len(pair_bank.indices),
        "copy_budgets": config["copy_budgets"],
        "repetitions": int(config["repetitions"]),
        "seed": int(config["seed"]),
        "empirical_local_basis_fractions": basis_fractions,
        "largest_copy_budget": largest_budget,
        "largest_budget_mean_rmse": largest.set_index(["state", "method"])
        .mean_rmse.to_dict(),
        "shadow_to_native_rmse_ratio": ratios,
        "interpretation": (
            "Native all-Z is the task-matched policy for declared diagonal ZZ "
            "observables and is expected to be more copy-efficient. Any observed "
            "shadow penalty is the cost of a universal measurement policy, not "
            "evidence that AOC outperforms classical shadows."
        ),
        "claim_boundary": (
            "This is a standalone measurement-estimation audit on controlled "
            "diagonal states. It is not an implementation or comparison of eSCD "
            "and establishes no quantum or sequential-detection advantage."
        ),
    }
    # JSON object keys cannot be tuples.
    summary["largest_budget_mean_rmse"] = {
        f"{state}/{method}": float(value)
        for (state, method), value in summary["largest_budget_mean_rmse"].items()
    }
    summary["shadow_to_native_rmse_ratio"] = {
        str(state): float(value) for state, value in ratios.items()
    }
    summary_path = output / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    outputs = [
        replicate_path,
        observable_path,
        aggregate_path,
        snapshot_path,
        pair_path,
        figure_pdf,
        figure_png,
        summary_path,
    ]
    write_manifest(
        output / "manifest.json",
        experiment="run5-local-pauli-shadow-measurement-audit",
        started_at=started,
        config={
            **config,
            "locked_config_path": str(
                config_path.relative_to(RUN_ROOT.parent.parent)
            ),
            "locked_config_sha256": sha256_file(config_path),
        },
        outputs=outputs,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
