#!/usr/bin/env python3
"""Natural difference orbitals in a controlled dimerized Hückel ring."""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from aoc import maximum_observable_contrast
from aoc.chemistry import huckel_ring_hamiltonian, occupied_one_particle_density
from aoc.repro import write_manifest
from aoc.symmetry import finite_group_twirl

RUN_ROOT = Path(__file__).resolve().parents[1]
RESULTS = RUN_ROOT / "results" / "chemistry"


def two_site_translation_unitaries(sites: int):
    identity = np.eye(sites)
    return tuple(np.roll(identity, 2 * shift, axis=0) for shift in range(sites // 2))


def main():
    started = time.time()
    RESULTS.mkdir(parents=True, exist_ok=True)
    sites = 10
    occupied = sites // 2
    distortions = np.linspace(0.0, 0.7, 36)
    reference = occupied_one_particle_density(
        huckel_ring_hamiltonian(np.ones(sites)),
        occupied,
    )
    unitaries = two_site_translation_unitaries(sites)
    rows = []
    final_result = None
    for distortion in distortions:
        hoppings = 1.0 + distortion * (-1.0) ** np.arange(sites)
        state = occupied_one_particle_density(
            huckel_ring_hamiltonian(hoppings),
            occupied,
        )
        state = np.real(finite_group_twirl(state, unitaries))
        result = maximum_observable_contrast(state, reference)
        final_result = result
        rows.append(
            {
                "bond_alternation": distortion,
                "one_body_trace_distance": result.trace_norm / 2.0,
                "positive_attachment_weight": result.positive_gap,
                "negative_detachment_weight": -result.negative_gap,
                "positive_orbital_count": int(np.sum(result.eigenvalues > 1e-10)),
            }
        )
    frame = pd.DataFrame(rows)
    scan_path = RESULTS / "distortion_scan.csv"
    frame.to_csv(scan_path, index=False)

    assert final_result is not None
    orbital_rows = []
    for index, eigenvalue in enumerate(final_result.eigenvalues):
        for site, amplitude in enumerate(final_result.eigenvectors[:, index]):
            orbital_rows.append(
                {
                    "orbital": index,
                    "difference_occupation": eigenvalue,
                    "site": site,
                    "probability": float(abs(amplitude) ** 2),
                    "amplitude": float(np.real(amplitude)),
                }
            )
    orbitals = pd.DataFrame(orbital_rows)
    orbital_path = RESULTS / "difference_orbitals.csv"
    orbitals.to_csv(orbital_path, index=False)

    sns.set_theme(style="whitegrid", context="paper")
    figure, axes = plt.subplots(1, 2, figsize=(8.3, 3.2))
    axes[0].plot(
        frame.bond_alternation,
        frame.one_body_trace_distance,
        color="#34765b",
    )
    axes[0].set_xlabel("bond alternation")
    axes[0].set_ylabel("one-body trace distance")
    leading = orbitals[orbitals.orbital < 4]
    sns.barplot(
        data=leading,
        x="site",
        y="probability",
        hue="orbital",
        ax=axes[1],
        palette="viridis",
    )
    axes[1].set_ylabel(r"$|\phi_i|^2$")
    axes[1].set_xlabel("ring site")
    axes[1].legend(title="difference orbital", fontsize=7)
    figure.tight_layout()
    pdf_path = RESULTS / "huckel_difference_density.pdf"
    png_path = RESULTS / "huckel_difference_density.png"
    figure.savefig(pdf_path)
    figure.savefig(png_path, dpi=220)
    plt.close(figure)

    summary = {
        "model": "half-filled ten-site dimerized Hückel ring",
        "maximum_bond_alternation": float(distortions[-1]),
        "final_one_body_trace_distance": float(frame.iloc[-1].one_body_trace_distance),
        "attachment_detachment_balance_error": float(
            abs(final_result.positive_gap + final_result.negative_gap)
        ),
        "interpretation": (
            "Positive and negative eigenvectors of the one-particle "
            "difference density are natural attachment and detachment modes."
        ),
        "claim_boundary": (
            "The calculation is a controlled tight-binding illustration; "
            "predictive quantum chemistry requires an electronic-structure "
            "backend and validation against molecular data."
        ),
    }
    summary_path = RESULTS / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    outputs = [
        scan_path,
        orbital_path,
        pdf_path,
        png_path,
        summary_path,
    ]
    write_manifest(
        RESULTS / "manifest.json",
        experiment="run3-huckel-difference-density",
        started_at=started,
        config={
            "sites": sites,
            "occupied_orbitals": occupied,
            "distortion_points": len(distortions),
            "maximum_distortion": float(distortions[-1]),
        },
        outputs=outputs,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
