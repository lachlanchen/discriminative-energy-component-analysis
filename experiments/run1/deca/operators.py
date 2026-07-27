"""Density-operator estimation and measurement diagnostics."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


ComplexArray = NDArray[np.complex128]


def _as_state_matrix(states: ArrayLike) -> ComplexArray:
    array = np.asarray(states, dtype=np.complex128)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2:
        raise ValueError("States must be a matrix with one state per row.")
    norms = np.linalg.norm(array, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-8):
        raise ValueError("Every state vector must have unit Euclidean norm.")
    return array


def density_operator(
    states: ArrayLike, sample_weight: ArrayLike | None = None
) -> ComplexArray:
    """Return the weighted mean of rank-one state projectors."""

    array = _as_state_matrix(states)
    if sample_weight is None:
        weights = np.full(array.shape[0], 1.0 / array.shape[0])
    else:
        weights = np.asarray(sample_weight, dtype=np.float64)
        if weights.shape != (array.shape[0],):
            raise ValueError("sample_weight must have one value per state.")
        if np.any(weights < 0) or not np.isfinite(weights).all():
            raise ValueError("sample_weight must be finite and nonnegative.")
        total = weights.sum()
        if total <= 0:
            raise ValueError("sample_weight must have positive total mass.")
        weights = weights / total
    rho = (array.conj().T * weights) @ array
    return 0.5 * (rho + rho.conj().T)


def class_density_operators(
    states: ArrayLike,
    labels: ArrayLike,
    priors: str | Sequence[float] = "empirical",
) -> tuple[NDArray, list[ComplexArray], NDArray[np.float64]]:
    """Estimate one density operator and one prior per class."""

    array = _as_state_matrix(states)
    y = np.asarray(labels)
    if y.ndim != 1 or y.shape[0] != array.shape[0]:
        raise ValueError("labels must be a one-dimensional array matching states.")
    classes, inverse, counts = np.unique(y, return_inverse=True, return_counts=True)
    rhos = [density_operator(array[inverse == index]) for index in range(len(classes))]
    if isinstance(priors, str):
        normalized = priors.lower()
        if normalized == "empirical":
            prior_values = counts.astype(np.float64) / counts.sum()
        elif normalized == "balanced":
            prior_values = np.full(len(classes), 1.0 / len(classes))
        else:
            raise ValueError("priors must be 'empirical', 'balanced', or an array.")
    else:
        prior_values = np.asarray(priors, dtype=np.float64)
        if prior_values.shape != (len(classes),):
            raise ValueError("Explicit priors must have one value per class.")
        if np.any(prior_values <= 0) or not np.isfinite(prior_values).all():
            raise ValueError("Explicit priors must be finite and positive.")
        prior_values = prior_values / prior_values.sum()
    return classes, rhos, prior_values


def weighted_class_operators(
    rhos: Sequence[ArrayLike], priors: ArrayLike
) -> list[ComplexArray]:
    prior_values = np.asarray(priors, dtype=np.float64)
    if prior_values.shape != (len(rhos),):
        raise ValueError("priors must have one value per density operator.")
    return [
        float(prior) * np.asarray(rho, dtype=np.complex128)
        for prior, rho in zip(prior_values, rhos)
    ]


def measurement_probabilities(
    states: ArrayLike, effects: Sequence[ArrayLike]
) -> NDArray[np.float64]:
    """Evaluate Born probabilities for every state and effect."""

    array = _as_state_matrix(states)
    matrices = [np.asarray(effect, dtype=np.complex128) for effect in effects]
    probabilities = np.column_stack(
        [
            np.real(np.einsum("bi,ij,bj->b", array.conj(), effect, array))
            for effect in matrices
        ]
    )
    probabilities[np.abs(probabilities) < 1e-12] = 0.0
    probabilities = np.clip(probabilities, 0.0, None)
    totals = probabilities.sum(axis=1, keepdims=True)
    if np.any(totals <= 0):
        raise ValueError("Measurement produced zero total probability.")
    return probabilities / totals


def measurement_success(
    weighted_operators: Sequence[ArrayLike], effects: Sequence[ArrayLike]
) -> float:
    if len(weighted_operators) != len(effects):
        raise ValueError("One effect is required per weighted class operator.")
    value = sum(
        np.real(
            np.trace(
                np.asarray(operator, dtype=np.complex128)
                @ np.asarray(effect, dtype=np.complex128)
            )
        )
        for operator, effect in zip(weighted_operators, effects)
    )
    return float(value)


def validate_povm(
    effects: Sequence[ArrayLike], tolerance: float = 1e-7
) -> dict[str, float | bool]:
    matrices = [np.asarray(effect, dtype=np.complex128) for effect in effects]
    if not matrices:
        raise ValueError("At least one effect is required.")
    dimension = matrices[0].shape[0]
    if any(matrix.shape != (dimension, dimension) for matrix in matrices):
        raise ValueError("All effects must be square matrices of equal size.")
    hermitian_error = max(
        np.linalg.norm(matrix - matrix.conj().T, ord="fro")
        for matrix in matrices
    )
    minimum_eigenvalue = min(
        float(np.min(np.linalg.eigvalsh(0.5 * (matrix + matrix.conj().T))))
        for matrix in matrices
    )
    completeness_error = float(
        np.linalg.norm(sum(matrices) - np.eye(dimension), ord="fro")
    )
    valid = (
        hermitian_error <= tolerance
        and minimum_eigenvalue >= -tolerance
        and completeness_error <= tolerance
    )
    return {
        "valid": valid,
        "hermitian_error": float(hermitian_error),
        "minimum_eigenvalue": minimum_eigenvalue,
        "completeness_error": completeness_error,
    }


def effects_from_basis_assignment(
    basis: ArrayLike, assignment: ArrayLike, num_classes: int
) -> list[ComplexArray]:
    matrix = np.asarray(basis, dtype=np.complex128)
    labels = np.asarray(assignment, dtype=np.int64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("basis must be a square unitary matrix.")
    if labels.shape != (matrix.shape[1],):
        raise ValueError("assignment must have one class index per basis vector.")
    effects: list[ComplexArray] = []
    for class_index in range(num_classes):
        selected = matrix[:, labels == class_index]
        effects.append(selected @ selected.conj().T)
    return effects


def offdiagonal_residual(
    weighted_operators: Sequence[ArrayLike], basis: ArrayLike
) -> float:
    matrix = np.asarray(basis, dtype=np.complex128)
    squared = 0.0
    for operator in weighted_operators:
        transformed = matrix.conj().T @ np.asarray(operator) @ matrix
        offdiagonal = transformed - np.diag(np.diag(transformed))
        squared += np.linalg.norm(offdiagonal, ord="fro") ** 2
    return float(np.sqrt(squared))

def commutator_measure(operators: Iterable[ArrayLike]) -> float:
    matrices = [np.asarray(operator, dtype=np.complex128) for operator in operators]
    squared = 0.0
    for left_index, left in enumerate(matrices):
        for right in matrices[left_index + 1 :]:
            commutator = left @ right - right @ left
            squared += np.linalg.norm(commutator, ord="fro") ** 2
    return float(np.sqrt(squared))
