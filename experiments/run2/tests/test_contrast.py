import cvxpy as cp
import numpy as np
from aoc import (
    AdditiveState,
    maximum_observable_contrast,
    projective_mmd_squared,
)


def _random_density(rng, dimension):
    matrix = rng.normal(size=(dimension, dimension))
    density = matrix @ matrix.T
    return density / np.trace(density)


def test_jordan_effect_matches_sdp():
    rng = np.random.default_rng(9)
    for dimension in (2, 4, 7):
        first = _random_density(rng, dimension)
        second = _random_density(rng, dimension)
        result = maximum_observable_contrast(first, second)

        effect = cp.Variable((dimension, dimension), symmetric=True)
        delta = first - second
        problem = cp.Problem(
            cp.Maximize(cp.trace(effect @ delta)),
            [effect >> 0, np.eye(dimension) - effect >> 0],
        )
        problem.solve(solver="CLARABEL")

        assert abs(problem.value - result.positive_gap) < 2e-7
        assert abs(result.positive_gap - result.trace_norm / 2.0) < 2e-12
        np.testing.assert_allclose(
            result.effect @ result.effect,
            result.effect,
            atol=2e-12,
        )


def test_rank_constraint_is_ky_fan_positive_sum():
    first = np.diag([0.55, 0.25, 0.15, 0.05])
    second = np.diag([0.05, 0.15, 0.25, 0.55])
    full = maximum_observable_contrast(first, second)
    rank_one = maximum_observable_contrast(first, second, rank=1)
    assert full.rank == 2
    assert rank_one.rank == 1
    assert abs(rank_one.positive_gap - 0.5) < 1e-14
    assert abs(full.positive_gap - 0.6) < 1e-14


def test_projective_kernel_mmd_equals_density_frobenius_distance():
    rng = np.random.default_rng(31)
    first = rng.normal(size=(17, 6))
    second = rng.normal(size=(13, 6))
    rho_first = AdditiveState.from_samples(first).density
    rho_second = AdditiveState.from_samples(second).density
    expected = np.linalg.norm(rho_first - rho_second, ord="fro") ** 2
    observed = projective_mmd_squared(first, second)
    assert abs(observed - expected) < 2e-14
