"""Analytical and convex measurement solvers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .operators import (
    effects_from_basis_assignment,
    measurement_success,
    validate_povm,
)


ComplexArray = NDArray[np.complex128]


@dataclass
class MeasurementSolution:
    """A fitted quantum measurement and its training-ensemble diagnostics."""

    effects: tuple[ComplexArray, ...]
    success: float
    method: str
    basis: ComplexArray | None = None
    assignment: NDArray[np.int64] | None = None
    eigenvalues: NDArray[np.float64] | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


def _weighted_operators(
    operators: Sequence[ArrayLike],
) -> list[ComplexArray]:
    matrices = [np.asarray(operator, dtype=np.complex128) for operator in operators]
    if len(matrices) < 2:
        raise ValueError("At least two weighted class operators are required.")
    dimension = matrices[0].shape[0]
    for matrix in matrices:
        if matrix.shape != (dimension, dimension):
            raise ValueError("All operators must have the same square shape.")
        if not np.allclose(matrix, matrix.conj().T, atol=1e-9):
            raise ValueError("Class operators must be Hermitian.")
        if np.min(np.linalg.eigvalsh(matrix)) < -1e-9:
            raise ValueError("Weighted class operators must be positive semidefinite.")
    total_trace = float(sum(np.real(np.trace(matrix)) for matrix in matrices))
    if not np.isclose(total_trace, 1.0, atol=1e-7):
        raise ValueError(
            "Weighted class operators must have total trace one; "
            f"received {total_trace:.12g}."
        )
    return matrices


def binary_helstrom(
    operators: Sequence[ArrayLike], zero_tolerance: float = 1e-10
) -> MeasurementSolution:
    """Return the closed-form binary DECA/Helstrom measurement."""

    matrices = _weighted_operators(operators)
    if len(matrices) != 2:
        raise ValueError("binary_helstrom requires exactly two class operators.")
    difference = matrices[0] - matrices[1]
    eigenvalues, eigenvectors = np.linalg.eigh(difference)
    order = np.argsort(-np.abs(eigenvalues), kind="stable")
    eigenvalues = np.real(eigenvalues[order])
    basis = eigenvectors[:, order]
    assignment = np.where(eigenvalues >= 0.0, 0, 1).astype(np.int64)

    # A zero-eigenvalue direction contributes equally to both classes. Choose
    # the class with the larger numerical diagonal contribution for stability.
    zero_indices = np.flatnonzero(np.abs(eigenvalues) <= zero_tolerance)
    for index in zero_indices:
        vector = basis[:, index]
        contributions = [
            float(np.real(vector.conj().T @ operator @ vector))
            for operator in matrices
        ]
        assignment[index] = int(np.argmax(contributions))

    effects = tuple(effects_from_basis_assignment(basis, assignment, 2))
    success = measurement_success(matrices, effects)
    trace_formula = 0.5 * (
        sum(float(np.real(np.trace(matrix))) for matrix in matrices)
        + float(np.sum(np.abs(eigenvalues)))
    )
    return MeasurementSolution(
        effects=effects,
        success=success,
        method="binary_helstrom",
        basis=basis,
        assignment=assignment,
        eigenvalues=eigenvalues,
        diagnostics={
            "trace_norm": float(np.sum(np.abs(eigenvalues))),
            "closed_form_success": trace_formula,
            "closed_form_error": abs(success - trace_formula),
            "zero_eigenvalues": int(len(zero_indices)),
            "povm": validate_povm(effects),
        },
    )

def _spectral_inverse_sqrt(
    matrix: ComplexArray, tolerance: float
) -> tuple[ComplexArray, ComplexArray]:
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (matrix + matrix.conj().T))
    threshold = tolerance * max(1.0, float(np.max(np.abs(eigenvalues))))
    support = eigenvalues > threshold
    inverse_values = np.zeros_like(eigenvalues)
    inverse_values[support] = 1.0 / np.sqrt(eigenvalues[support])
    inverse_sqrt = (eigenvectors * inverse_values) @ eigenvectors.conj().T
    kernel_vectors = eigenvectors[:, ~support]
    kernel_projector = kernel_vectors @ kernel_vectors.conj().T
    return inverse_sqrt, kernel_projector


def pretty_good_measurement(
    operators: Sequence[ArrayLike], tolerance: float = 1e-10
) -> MeasurementSolution:
    """Return the square-root/Pretty Good Measurement."""

    matrices = _weighted_operators(operators)
    average = sum(matrices)
    inverse_sqrt, kernel_projector = _spectral_inverse_sqrt(
        average, tolerance
    )
    traces = np.array(
        [float(np.real(np.trace(matrix))) for matrix in matrices],
        dtype=np.float64,
    )
    traces /= traces.sum()
    effects = []
    for weight, matrix in zip(traces, matrices):
        effect = inverse_sqrt @ matrix @ inverse_sqrt
        effect = effect + weight * kernel_projector
        effects.append(0.5 * (effect + effect.conj().T))
    effect_tuple = tuple(effects)
    return MeasurementSolution(
        effects=effect_tuple,
        success=measurement_success(matrices, effect_tuple),
        method="pretty_good_measurement",
        diagnostics={
            "kernel_dimension": int(
                round(float(np.real(np.trace(kernel_projector))))
            ),
            "povm": validate_povm(effect_tuple),
        },
    )


def _repair_povm(
    raw_effects: Sequence[ArrayLike], tolerance: float
) -> tuple[ComplexArray, ...]:
    projected: list[ComplexArray] = []
    for raw in raw_effects:
        matrix = np.asarray(raw, dtype=np.complex128)
        matrix = 0.5 * (matrix + matrix.conj().T)
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        eigenvalues = np.clip(eigenvalues, 0.0, None)
        projected.append((eigenvectors * eigenvalues) @ eigenvectors.conj().T)
    total = sum(projected)
    inverse_sqrt, kernel = _spectral_inverse_sqrt(total, tolerance)
    if np.linalg.norm(kernel, ord="fro") > tolerance:
        raise RuntimeError("Numerical POVM repair encountered a singular sum.")
    repaired = [
        inverse_sqrt @ effect @ inverse_sqrt for effect in projected
    ]
    return tuple(0.5 * (effect + effect.conj().T) for effect in repaired)


def optimal_povm(
    operators: Sequence[ArrayLike],
    solver: str | None = None,
    tolerance: float = 1e-8,
    max_iterations: int = 50_000,
) -> MeasurementSolution:
    """Solve the minimum-error multi-class POVM SDP."""

    matrices = _weighted_operators(operators)
    try:
        import cvxpy as cp
    except ImportError as error:
        raise ImportError(
            "optimal_povm requires cvxpy. Install the experiment package."
        ) from error

    dimension = matrices[0].shape[0]
    variables = [
        cp.Variable((dimension, dimension), hermitian=True)
        for _ in matrices
    ]
    constraints = [variable >> 0 for variable in variables]
    constraints.append(sum(variables) == np.eye(dimension))
    objective = cp.Maximize(
        sum(
            cp.real(cp.trace(matrix @ variable))
            for matrix, variable in zip(matrices, variables)
        )
    )
    problem = cp.Problem(objective, constraints)

    installed = set(cp.installed_solvers())
    selected_solver = solver
    if selected_solver is None:
        if "CLARABEL" in installed:
            selected_solver = "CLARABEL"
        elif "SCS" in installed:
            selected_solver = "SCS"
        else:
            raise RuntimeError(
                "No supported conic solver found. Install CLARABEL or SCS."
            )

    solve_options: dict[str, Any] = {"verbose": False}
    if selected_solver.upper() == "SCS":
        solve_options.update({"eps": tolerance, "max_iters": max_iterations})
    problem.solve(solver=selected_solver, **solve_options)
    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        raise RuntimeError(f"POVM SDP failed with status {problem.status}.")
    if any(variable.value is None for variable in variables):
        raise RuntimeError("POVM SDP returned no measurement matrices.")

    effects = _repair_povm(
        [variable.value for variable in variables], tolerance=tolerance
    )
    success = measurement_success(matrices, effects)
    return MeasurementSolution(
        effects=effects,
        success=success,
        method="optimal_povm",
        diagnostics={
            "solver": selected_solver,
            "status": problem.status,
            "reported_objective": float(problem.value),
            "repaired_objective": success,
            "repair_delta": abs(success - float(problem.value)),
            "solve_time_seconds": getattr(
                problem.solver_stats, "solve_time", None
            ),
            "num_iterations": getattr(
                problem.solver_stats, "num_iters", None
            ),
            "povm": validate_povm(effects, tolerance=1e-6),
        },
    )
