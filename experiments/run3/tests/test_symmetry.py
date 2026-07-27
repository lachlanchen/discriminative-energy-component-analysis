import cvxpy as cp
import numpy as np
from aoc import (
    cyclic_translation_twirl,
    finite_group_twirl,
    invariant_observable_contrast,
    maximum_observable_contrast,
    pure_state_density,
    symmetry_sector_contrasts,
    translation_power_state,
)
from aoc.symmetry import cyclic_translation_unitaries


def test_fft_cyclic_twirl_matches_explicit_group_average():
    rng = np.random.default_rng(8)
    vector = rng.normal(size=9) + 1j * rng.normal(size=9)
    density = pure_state_density(vector)
    explicit = finite_group_twirl(density, cyclic_translation_unitaries(9))
    fast = cyclic_translation_twirl(density)
    np.testing.assert_allclose(fast, explicit, atol=2e-14)

    fourier = np.fft.fft(np.eye(9), axis=0) / 3
    power_basis_state = translation_power_state(vector)
    np.testing.assert_allclose(
        fourier @ fast @ fourier.conj().T,
        power_basis_state,
        atol=2e-14,
    )


def test_invariant_witness_matches_symmetry_constrained_sdp():
    rng = np.random.default_rng(29)
    first_raw = rng.normal(size=(4, 4))
    second_raw = rng.normal(size=(4, 4))
    first = first_raw @ first_raw.T
    second = second_raw @ second_raw.T
    first /= np.trace(first)
    second /= np.trace(second)
    symmetry = np.diag([1.0, 1.0, -1.0, -1.0])
    group = [np.eye(4), symmetry]
    result = invariant_observable_contrast(first, second, group)

    effect = cp.Variable((4, 4), symmetric=True)
    delta = first - second
    problem = cp.Problem(
        cp.Maximize(cp.trace(effect @ delta)),
        [
            effect >> 0,
            np.eye(4) - effect >> 0,
            effect @ symmetry == symmetry @ effect,
        ],
    )
    problem.solve(solver="CLARABEL")
    assert abs(problem.value - result.contrast.positive_gap) < 2e-7
    assert result.invariance_error < 1e-12


def test_sector_contributions_sum_for_block_diagonal_contrast():
    delta = np.diag([0.4, -0.1, 0.2, -0.5])
    plus = np.diag([1.0, 1.0, 0.0, 0.0])
    minus = np.eye(4) - plus
    sectors = symmetry_sector_contrasts(delta, {"plus": plus, "minus": minus})
    assert abs(sum(item.trace_norm for item in sectors) - 1.2) < 1e-14
    assert abs(sum(item.positive_gap for item in sectors) - 0.6) < 1e-14


def test_twirling_cannot_increase_trace_distance():
    rng = np.random.default_rng(103)
    first = pure_state_density(rng.normal(size=6))
    second = pure_state_density(rng.normal(size=6))
    group = cyclic_translation_unitaries(6)
    original = maximum_observable_contrast(first, second).trace_norm
    twirled = maximum_observable_contrast(
        finite_group_twirl(first, group),
        finite_group_twirl(second, group),
    ).trace_norm
    assert twirled <= original + 1e-12
