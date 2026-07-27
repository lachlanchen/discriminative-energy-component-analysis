#!/usr/bin/env python3
"""Symmetry-resolved reduced-state response across the TFIM transition."""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from aoc import maximum_observable_contrast
from aoc.manybody import ground_state, pauli_x_tensor, reduced_density_matrix
from aoc.repro import write_manifest
from aoc.symmetry import symmetry_sector_contrasts

RUN_ROOT = Path(__file__).resolve().parents[1]
RESULTS = RUN_ROOT / "results" / "quantum_phase"


def main():
    started = time.time()
    RESULTS.mkdir(parents=True, exist_ok=True)
    qubits = 10
    kept = 5
    step = 0.025
    fields = np.arange(0.35, 1.6501, step)
    reduced = []
    energies = []
    for field in fields:
        energy, state = ground_state(qubits, float(field))
        energies.append(energy)
        reduced.append(reduced_density_matrix(state, kept))

    parity = pauli_x_tensor(kept)
    identity = np.eye(1 << kept)
    sectors = {
        "even": (identity + parity) / 2.0,
        "odd": (identity - parity) / 2.0,
    }
    rows = []
    for index in range(len(fields) - 1):
        left = reduced[index]
        right = reduced[index + 1]
        contrast = maximum_observable_contrast(right, left)
        midpoint = float((fields[index] + fields[index + 1]) / 2.0)
        resolved = symmetry_sector_contrasts(
            contrast.contrast,
            sectors,
            tolerance=1e-6,
        )
        rows.append(
            {
                "field_midpoint": midpoint,
                "ground_energy_left": energies[index],
                "trace_distance": contrast.trace_norm / 2.0,
                "trace_distance_susceptibility": (contrast.trace_norm / (2.0 * step)),
                "even_trace_norm": next(
                    item.trace_norm for item in resolved if item.name == "even"
                ),
                "odd_trace_norm": next(
                    item.trace_norm for item in resolved if item.name == "odd"
                ),
                "parity_commutator_error": float(
                    np.linalg.norm(
                        parity @ contrast.contrast - contrast.contrast @ parity,
                        ord="fro",
                    )
                ),
            }
        )
    frame = pd.DataFrame(rows)
    result_path = RESULTS / "field_scan.csv"
    frame.to_csv(result_path, index=False)

    sns.set_theme(style="whitegrid", context="paper")
    figure, axes = plt.subplots(1, 2, figsize=(8.3, 3.2))
    axes[0].plot(
        frame.field_midpoint,
        frame.trace_distance_susceptibility,
        color="#7a2d98",
    )
    axes[0].axvline(1.0, color="black", linestyle=":", label=r"$g_c=1$")
    axes[0].set_xlabel(r"transverse field $g/J$")
    axes[0].set_ylabel(r"$D(\rho_{g+\delta},\rho_g)/\delta$")
    axes[0].legend(fontsize=8)
    total = frame.even_trace_norm + frame.odd_trace_norm
    axes[1].stackplot(
        frame.field_midpoint,
        frame.even_trace_norm / total,
        frame.odd_trace_norm / total,
        labels=["even parity", "odd parity"],
        colors=["#3567a8", "#d07635"],
    )
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_xlabel(r"transverse field $g/J$")
    axes[1].set_ylabel("fraction of trace-norm response")
    axes[1].legend(fontsize=8)
    figure.tight_layout()
    pdf_path = RESULTS / "symmetry_resolved_transition.pdf"
    png_path = RESULTS / "symmetry_resolved_transition.png"
    figure.savefig(pdf_path)
    figure.savefig(png_path, dpi=220)
    plt.close(figure)

    peak = frame.iloc[frame.trace_distance_susceptibility.argmax()]
    summary = {
        "model": "periodic transverse-field Ising chain",
        "qubits": qubits,
        "reduced_qubits": kept,
        "field_step": step,
        "thermodynamic_critical_field": 1.0,
        "finite_size_peak_field": float(peak.field_midpoint),
        "peak_trace_distance_susceptibility": float(peak.trace_distance_susceptibility),
        "maximum_parity_commutator_error": float(frame.parity_commutator_error.max()),
        "claim_boundary": (
            "This is an exact small-system diagnostic of a known transition, "
            "not a new solution of the Ising model."
        ),
    }
    summary_path = RESULTS / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    outputs = [result_path, pdf_path, png_path, summary_path]
    write_manifest(
        RESULTS / "manifest.json",
        experiment="run3-symmetry-resolved-tfim",
        started_at=started,
        config={
            "qubits": qubits,
            "kept_qubits": kept,
            "field_min": float(fields[0]),
            "field_max": float(fields[-1]),
            "field_step": step,
        },
        outputs=outputs,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
