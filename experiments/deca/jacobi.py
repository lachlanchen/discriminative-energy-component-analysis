"""Monotone Jacobi optimization for commuting multi-class measurements."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .operators import (
    effects_from_basis_assignment,
    measurement_success,
    offdiagonal_residual,
    validate_povm,
)
from .solvers import MeasurementSolution


ComplexArray = NDArray[np.complex128]


def _validate_operators(
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
    total_trace = sum(float(np.real(np.trace(matrix))) for matrix in matrices)
    if not np.isclose(total_trace, 1.0, atol=1e-7):
        raise ValueError("Weighted operators must have total trace one.")
    return matrices


def _component_scores(
    basis: ComplexArray, operators: Sequence[ComplexArray]
) -> NDArray[np.float64]:
    return np.column_stack(
        [
            np.real(np.diag(basis.conj().T @ operator @ basis))
            for operator in operators
        ]
    )


def _assignment_objective(
    basis: ComplexArray, operators: Sequence[ComplexArray]
) -> tuple[float, NDArray[np.int64], NDArray[np.float64]]:
    scores = _component_scores(basis, operators)
    assignment = np.argmax(scores, axis=1).astype(np.int64)
    objective = float(np.sum(scores[np.arange(basis.shape[1]), assignment]))
    return objective, assignment, scores


def _haar_unitary(
    dimension: int, random_generator: np.random.Generator, complex_data: bool
) -> ComplexArray:
    if complex_data:
        raw = random_generator.normal(size=(dimension, dimension))
        raw = raw + 1j * random_generator.normal(size=(dimension, dimension))
    else:
        raw = random_generator.normal(size=(dimension, dimension))
    q_matrix, r_matrix = np.linalg.qr(raw)
    diagonal = np.diag(r_matrix)
    phases = np.ones_like(diagonal, dtype=np.complex128)
    nonzero = np.abs(diagonal) > 1e-15
    phases[nonzero] = diagonal[nonzero] / np.abs(diagonal[nonzero])
    return np.asarray(q_matrix * phases.conj(), dtype=np.complex128)


def _initial_bases(
    operators: Sequence[ComplexArray],
    random_generator: np.random.Generator,
    random_starts: int,
) -> list[tuple[str, ComplexArray]]:
    dimension = operators[0].shape[0]
    complex_data = any(
        np.max(np.abs(np.imag(operator))) > 1e-12 for operator in operators
    )
    candidates: list[tuple[str, ComplexArray]] = [
        ("identity", np.eye(dimension, dtype=np.complex128))
    ]

    pooled = sum(operators)
    _, pooled_basis = np.linalg.eigh(pooled)
    candidates.append(("pooled", pooled_basis))

    coefficients = np.linspace(1.0, 2.0, len(operators))
    tagged = sum(
        coefficient * operator
        for coefficient, operator in zip(coefficients, operators)
    )
    _, tagged_basis = np.linalg.eigh(tagged)
    candidates.append(("tagged_sum", tagged_basis))

    contrast = np.zeros_like(operators[0])
    for left_index, left in enumerate(operators):
        for right in operators[left_index + 1 :]:
            difference = left - right
            contrast = contrast + difference @ difference
    contrast = 0.5 * (contrast + contrast.conj().T)
    _, contrast_basis = np.linalg.eigh(contrast)
    candidates.append(("pairwise_contrast", contrast_basis))

    for index in range(random_starts):
        candidates.append(
            (
                f"random_{index}",
                _haar_unitary(dimension, random_generator, complex_data),
            )
        )
    return candidates


def _orthogonal_complement(vector: ComplexArray) -> ComplexArray:
    if vector.shape != (2,):
        raise ValueError("Expected a two-dimensional vector.")
    return np.array(
        [-np.conj(vector[1]), np.conj(vector[0])],
        dtype=np.complex128,
    )


def _run_single_start(
    operators: Sequence[ComplexArray],
    initial_basis: ComplexArray,
    max_sweeps: int,
    tolerance: float,
) -> tuple[
    ComplexArray,
    NDArray[np.int64],
    list[float],
    int,
]:
    basis = np.asarray(initial_basis, dtype=np.complex128).copy()
    objective, assignment, _ = _assignment_objective(basis, operators)
    history = [objective]
    dimension = basis.shape[1]

    for sweep in range(max_sweeps):
        fixed_assignment = assignment.copy()
        for left_index in range(dimension - 1):
            for right_index in range(left_index + 1, dimension):
                left_class = int(fixed_assignment[left_index])
                right_class = int(fixed_assignment[right_index])
                if left_class == right_class:
                    continue
                pair = basis[:, [left_index, right_index]]
                difference = (
                    operators[left_class] - operators[right_class]
                )
                local = pair.conj().T @ difference @ pair
                local = 0.5 * (local + local.conj().T)
                _, local_vectors = np.linalg.eigh(local)
                leading = local_vectors[:, -1]
                rotation = np.column_stack(
                    [leading, _orthogonal_complement(leading)]
                )
                basis[:, [left_index, right_index]] = pair @ rotation

        new_objective, assignment, _ = _assignment_objective(
            basis, operators
        )
        if new_objective + 1e-12 < objective:
            raise RuntimeError(
                "Jacobi-DECA objective decreased beyond numerical tolerance."
            )
        history.append(new_objective)
        improvement = new_objective - objective
        objective = new_objective
        if improvement <= tolerance * max(1.0, abs(objective)):
            return basis, assignment, history, sweep + 1
    return basis, assignment, history, max_sweeps


def jacobi_deca(
    operators: Sequence[ArrayLike],
    random_starts: int = 8,
    max_sweeps: int = 100,
    tolerance: float = 1e-10,
    random_state: int = 0,
) -> MeasurementSolution:
    """Optimize an ancilla-free commuting projective measurement.

    Each basis outcome is assigned to exactly one class. For a fixed
    assignment, every two-component update has an analytical 2x2
    eigensolution.
    """

    matrices = _validate_operators(operators)
    if len(matrices) == 2:
        from .solvers import binary_helstrom

        solution = binary_helstrom(matrices)
        solution.method = "jacobi_deca_binary_closed_form"
        solution.diagnostics["delegated_to_binary_closed_form"] = True
        return solution
    if random_starts < 0:
        raise ValueError("random_starts must be nonnegative.")
    random_generator = np.random.default_rng(random_state)

    best: tuple[
        float,
        str,
        ComplexArray,
        NDArray[np.int64],
        list[float],
        int,
    ] | None = None
    start_summaries = []
    for name, initial_basis in _initial_bases(
        matrices, random_generator, random_starts
    ):
        basis, assignment, history, sweeps = _run_single_start(
            matrices, initial_basis, max_sweeps, tolerance
        )
        objective = history[-1]
        start_summaries.append(
            {
                "name": name,
                "initial_objective": history[0],
                "final_objective": objective,
                "sweeps": sweeps,
            }
        )
        if best is None or objective > best[0]:
            best = (
                objective,
                name,
                basis.copy(),
                assignment.copy(),
                history,
                sweeps,
            )

    if best is None:
        raise RuntimeError("Jacobi-DECA produced no candidate solution.")
    objective, name, basis, assignment, history, sweeps = best
    effects = tuple(
        effects_from_basis_assignment(basis, assignment, len(matrices))
    )
    success = measurement_success(matrices, effects)
    residual = offdiagonal_residual(matrices, basis)
    return MeasurementSolution(
        effects=effects,
        success=success,
        method="jacobi_deca",
        basis=basis,
        assignment=assignment,
        diagnostics={
            "selected_start": name,
            "sweeps": sweeps,
            "objective_history": history,
            "start_summaries": start_summaries,
            "objective_consistency_error": abs(success - objective),
            "offdiagonal_residual": residual,
            "oracle_gap_upper_bound": np.sqrt(basis.shape[0]) * residual,
            "class_component_counts": np.bincount(
                assignment, minlength=len(matrices)
            ).tolist(),
            "povm": validate_povm(effects),
        },
    )
