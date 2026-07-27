"""Minimum-error multiclass observable design as a finite-dimensional POVM."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import cvxpy as cp
import numpy as np
from numpy.typing import ArrayLike, NDArray

from .states import as_density_matrix

ComplexMatrix = NDArray[np.complex128]


@dataclass(frozen=True)
class MulticlassObservableResult:
    """Optimal class effects and their prior-weighted discrimination success."""

    effects: tuple[ComplexMatrix, ...]
    priors: NDArray[np.float64]
    success_probability: float
    guessing_baseline: float
    advantage: float
    solver: str
    status: str
    completeness_error: float
    minimum_effect_eigenvalue: float


def minimum_error_observables(
    states: Sequence[ArrayLike],
    *,
    priors: ArrayLike | None = None,
    solver: str = "CLARABEL",
) -> MulticlassObservableResult:
    r"""Solve the minimum-error state-discrimination/POVM problem.

    The primal program is

    .. math::

       \max_{\{E_c\}}\sum_c\pi_c\operatorname{Tr}(E_c\rho_c),
       \quad E_c\succeq0,\quad\sum_cE_c=I.

    This is the operational multiclass extension of binary maximum observable
    contrast. It is a known quantum state-discrimination SDP.
    """

    if len(states) < 2:
        raise ValueError("At least two class states are required.")
    densities = tuple(as_density_matrix(state) for state in states)
    dimension = densities[0].shape[0]
    if any(state.shape != (dimension, dimension) for state in densities):
        raise ValueError("All class states must have the same dimension.")
    if priors is None:
        probabilities = np.full(len(densities), 1.0 / len(densities))
    else:
        probabilities = np.asarray(priors, dtype=np.float64)
        if probabilities.shape != (len(densities),):
            raise ValueError("priors must match the number of states.")
        if np.any(probabilities < 0) or probabilities.sum() <= 0:
            raise ValueError("priors must be nonnegative with positive sum.")
        probabilities = probabilities / probabilities.sum()

    variables = [cp.Variable((dimension, dimension), hermitian=True) for _ in densities]
    constraints = [effect >> 0 for effect in variables]
    constraints.append(sum(variables) == np.eye(dimension))
    objective = cp.Maximize(
        cp.real(
            sum(
                probability * cp.trace(effect @ density)
                for probability, effect, density in zip(
                    probabilities,
                    variables,
                    densities,
                )
            )
        )
    )
    problem = cp.Problem(objective, constraints)
    try:
        problem.solve(solver=solver)
        used_solver = solver
    except cp.error.SolverError:
        problem.solve(solver="SCS", eps=1e-8, max_iters=100000)
        used_solver = "SCS"
    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        raise RuntimeError(f"POVM optimization failed with {problem.status}.")
    effects = tuple(
        0.5
        * (
            np.asarray(variable.value, dtype=np.complex128)
            + np.asarray(variable.value, dtype=np.complex128).conj().T
        )
        for variable in variables
    )
    completeness_error = float(
        np.linalg.norm(sum(effects) - np.eye(dimension), ord="fro")
    )
    minimum_eigenvalue = float(
        min(np.linalg.eigvalsh(effect).min() for effect in effects)
    )
    success = float(
        np.real(
            sum(
                probability * np.trace(effect @ density)
                for probability, effect, density in zip(
                    probabilities,
                    effects,
                    densities,
                )
            )
        )
    )
    baseline = float(np.max(probabilities))
    return MulticlassObservableResult(
        effects=effects,
        priors=probabilities,
        success_probability=success,
        guessing_baseline=baseline,
        advantage=success - baseline,
        solver=used_solver,
        status=str(problem.status),
        completeness_error=completeness_error,
        minimum_effect_eigenvalue=minimum_eigenvalue,
    )
