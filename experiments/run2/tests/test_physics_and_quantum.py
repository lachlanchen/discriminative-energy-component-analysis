import numpy as np
from aoc.contrast import maximum_observable_contrast
from aoc.physics import thermal_chain_model
from aoc.quantum import simulate_density_effect


def test_thermal_chain_damage_is_rank_one_before_trace_normalization():
    baseline, damaged, mode = thermal_chain_model(
        8,
        damaged_spring=4,
        damage_fraction=0.18,
    )
    np.testing.assert_allclose(baseline, np.eye(8), atol=2e-13)
    difference = damaged - baseline
    eigenvalues, eigenvectors = np.linalg.eigh(difference)
    assert np.sum(eigenvalues > 1e-10) == 1
    learned = eigenvectors[:, -1]
    assert abs(np.vdot(learned, mode)) ** 2 > 1.0 - 1e-12


def test_qiskit_density_effect_matches_born_probability():
    diagonal = np.array([1.0, 1.0]) / np.sqrt(2.0)
    antidiagonal = np.array([1.0, -1.0]) / np.sqrt(2.0)
    rho_d = 0.9 * np.outer(diagonal, diagonal) + 0.1 * np.eye(2) / 2
    rho_a = 0.9 * np.outer(antidiagonal, antidiagonal) + 0.1 * np.eye(2) / 2
    effect = maximum_observable_contrast(rho_d, rho_a).effect
    result = simulate_density_effect(
        rho_d,
        effect,
        shots=8192,
        seed=91,
    )
    assert result.analytic_probability > 0.94
    assert result.absolute_error < 0.02
