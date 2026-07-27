"""Analytical witnesses for maximum expectation contrast."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .states import as_density_matrix, pure_state_density

ComplexMatrix = NDArray[np.complex128]


@dataclass(frozen=True)
class ContrastResult:
    """Spectral Jordan decomposition of a Hermitian contrast."""

    contrast: ComplexMatrix
    eigenvalues: NDArray[np.float64]
    eigenvectors: ComplexMatrix
    effect: ComplexMatrix
    sign_observable: ComplexMatrix
    positive_gap: float
    negative_gap: float
    trace_norm: float
    rank: int


def _as_hermitian(matrix: ArrayLike, tolerance: float) -> ComplexMatrix:
    array = np.asarray(matrix, dtype=np.complex128)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError("Contrast inputs must be square matrices.")
    hermitian = 0.5 * (array + array.conj().T)
    error = np.linalg.norm(array - array.conj().T, ord="fro")
    if error > tolerance * max(1.0, np.linalg.norm(hermitian, ord="fro")):
        raise ValueError("Contrast input is not Hermitian within tolerance.")
    return hermitian


def maximum_observable_contrast(
    positive_state: ArrayLike,
    negative_state: ArrayLike,
    *,
    positive_weight: float = 1.0,
    negative_weight: float = 1.0,
    rank: int | None = None,
    tolerance: float = 1e-12,
) -> ContrastResult:
    """Return the effect maximizing ``Tr(E Delta)`` over ``0 <= E <= I``.

    If ``rank`` is supplied, the effect is additionally restricted to have at
    most that rank. The solution selects the largest positive eigenvalues
    (Ky Fan maximum principle).
    """

    if positive_weight < 0 or negative_weight < 0:
        raise ValueError("Contrast weights must be nonnegative.")
    first = _as_hermitian(positive_state, tolerance)
    second = _as_hermitian(negative_state, tolerance)
    if first.shape != second.shape:
        raise ValueError("Contrast states must have the same dimension.")
    delta = positive_weight * first - negative_weight * second
    delta = 0.5 * (delta + delta.conj().T)
    eigenvalues, eigenvectors = np.linalg.eigh(delta)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.real(eigenvalues[order])
    eigenvectors = eigenvectors[:, order]
    positive_indices = np.flatnonzero(eigenvalues > tolerance)
    if rank is not None:
        if rank < 0:
            raise ValueError("rank must be nonnegative.")
        positive_indices = positive_indices[: int(rank)]
    selected = eigenvectors[:, positive_indices]
    effect = selected @ selected.conj().T if selected.size else np.zeros_like(delta)
    signs = np.sign(eigenvalues)
    signs[np.abs(eigenvalues) <= tolerance] = 0.0
    sign_observable = (eigenvectors * signs) @ eigenvectors.conj().T
    positive_gap = float(np.sum(eigenvalues[positive_indices]))
    negative_gap = float(np.sum(np.minimum(eigenvalues, 0.0)))
    trace_norm = float(np.sum(np.abs(eigenvalues)))
    return ContrastResult(
        contrast=delta,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        effect=0.5 * (effect + effect.conj().T),
        sign_observable=0.5 * (sign_observable + sign_observable.conj().T),
        positive_gap=positive_gap,
        negative_gap=negative_gap,
        trace_norm=trace_norm,
        rank=len(positive_indices),
    )


def effect_expectation(effect: ArrayLike, state: ArrayLike) -> float:
    """Evaluate a Hermitian effect on a vector or positive operator state."""

    effect_matrix = _as_hermitian(effect, 1e-10)
    sample = np.asarray(state)
    density = (
        pure_state_density(sample) if sample.ndim == 1 else as_density_matrix(sample)
    )
    if effect_matrix.shape != density.shape:
        raise ValueError("Effect and state dimensions do not match.")
    return float(np.real(np.trace(effect_matrix @ density)))


def projective_mmd_squared(
    positive_samples: ArrayLike,
    negative_samples: ArrayLike,
) -> float:
    """Biased MMD squared for ``k(x,z)=|<x,z>|^2``.

    Every row is normalized before evaluation. The result equals the squared
    Hilbert--Schmidt norm of the difference between empirical density states.
    """

    first = np.asarray(positive_samples, dtype=np.complex128)
    second = np.asarray(negative_samples, dtype=np.complex128)
    if first.ndim != 2 or second.ndim != 2 or first.shape[1] != second.shape[1]:
        raise ValueError("Samples must be row matrices of equal feature size.")
    first = first / np.linalg.norm(first, axis=1, keepdims=True)
    second = second / np.linalg.norm(second, axis=1, keepdims=True)
    k_xx = np.abs(first @ first.conj().T) ** 2
    k_yy = np.abs(second @ second.conj().T) ** 2
    k_xy = np.abs(first @ second.conj().T) ** 2
    return float(k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean())
