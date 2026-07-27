"""Symmetry-restricted states, witnesses, and sector diagnostics."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .contrast import ContrastResult, maximum_observable_contrast
from .states import as_density_matrix, pure_state_density

ComplexMatrix = NDArray[np.complex128]


def finite_group_twirl(
    state: ArrayLike,
    unitaries: Sequence[ArrayLike],
    *,
    tolerance: float = 1e-9,
) -> ComplexMatrix:
    """Average ``U rho U^dagger`` over a supplied finite unitary group."""

    density = as_density_matrix(state)
    if not unitaries:
        raise ValueError("At least one group element is required.")
    total = np.zeros_like(density)
    for item in unitaries:
        unitary = np.asarray(item, dtype=np.complex128)
        if unitary.shape != density.shape:
            raise ValueError("Every unitary must match the state dimension.")
        if not np.allclose(
            unitary.conj().T @ unitary,
            np.eye(len(unitary)),
            atol=tolerance,
        ):
            raise ValueError("A supplied group action is not unitary.")
        total += unitary @ density @ unitary.conj().T
    return 0.5 * (total + total.conj().T) / len(unitaries)


def cyclic_translation_unitaries(dimension: int) -> tuple[ComplexMatrix, ...]:
    """Regular representation of the cyclic translation group."""

    identity = np.eye(dimension, dtype=np.complex128)
    return tuple(np.roll(identity, shift, axis=0) for shift in range(dimension))


def cyclic_translation_twirl(state: ArrayLike) -> ComplexMatrix:
    """Exact cyclic twirl, computed through the Fourier diagonal."""

    density = as_density_matrix(state)
    dimension = density.shape[0]
    fourier = np.fft.fft(np.eye(dimension), axis=0) / np.sqrt(dimension)
    transformed = fourier @ density @ fourier.conj().T
    diagonal = np.real(np.diag(transformed))
    twirled = fourier.conj().T @ np.diag(diagonal) @ fourier
    return 0.5 * (twirled + twirled.conj().T)


def translation_power_state(signal: ArrayLike) -> ComplexMatrix:
    """Cyclically invariant pure-signal state from its normalized DFT power."""

    vector = np.asarray(signal, dtype=np.complex128)
    if vector.ndim != 1:
        raise ValueError("signal must be one-dimensional.")
    norm = np.linalg.norm(vector)
    if norm <= 1e-15:
        raise ValueError("signal must be nonzero.")
    spectrum = np.fft.fft(vector / norm) / np.sqrt(len(vector))
    power = np.abs(spectrum) ** 2
    power /= power.sum()
    return np.diag(power.astype(np.complex128))


@dataclass(frozen=True)
class InvariantContrastResult:
    positive_state: ComplexMatrix
    negative_state: ComplexMatrix
    contrast: ContrastResult
    invariance_error: float


def invariant_observable_contrast(
    positive_state: ArrayLike,
    negative_state: ArrayLike,
    unitaries: Sequence[ArrayLike],
    *,
    rank: int | None = None,
) -> InvariantContrastResult:
    """Analytical optimum among effects commuting with a finite group."""

    first = finite_group_twirl(positive_state, unitaries)
    second = finite_group_twirl(negative_state, unitaries)
    result = maximum_observable_contrast(first, second, rank=rank)
    error = max(
        float(
            np.linalg.norm(
                np.asarray(unitary) @ result.effect
                - result.effect @ np.asarray(unitary),
                ord="fro",
            )
        )
        for unitary in unitaries
    )
    return InvariantContrastResult(first, second, result, error)


@dataclass(frozen=True)
class SectorContrast:
    name: str
    dimension: int
    trace: float
    trace_norm: float
    positive_gap: float


def symmetry_sector_contrasts(
    contrast: ArrayLike,
    sectors: dict[str, ArrayLike],
    *,
    tolerance: float = 1e-8,
) -> tuple[SectorContrast, ...]:
    """Resolve a block-diagonal contrast into orthogonal sector projectors."""

    delta = np.asarray(contrast, dtype=np.complex128)
    delta = 0.5 * (delta + delta.conj().T)
    dimension = delta.shape[0]
    projectors = {
        name: np.asarray(projector, dtype=np.complex128)
        for name, projector in sectors.items()
    }
    total = sum(projectors.values())
    if not np.allclose(total, np.eye(dimension), atol=tolerance):
        raise ValueError("Sector projectors must sum to identity.")
    output = []
    names = list(projectors)
    for index, name in enumerate(names):
        projector = projectors[name]
        if not np.allclose(projector @ projector, projector, atol=tolerance):
            raise ValueError(f"Sector {name} is not projective.")
        if np.linalg.norm(projector @ delta - delta @ projector, ord="fro") > tolerance:
            raise ValueError("Contrast is not block diagonal in the sectors.")
        for other_name in names[index + 1 :]:
            if (
                np.linalg.norm(projector @ projectors[other_name], ord="fro")
                > tolerance
            ):
                raise ValueError("Sector projectors are not orthogonal.")
        basis_values, basis_vectors = np.linalg.eigh(projector)
        basis = basis_vectors[:, basis_values > 0.5]
        block = basis.conj().T @ delta @ basis
        values = np.linalg.eigvalsh(0.5 * (block + block.conj().T))
        output.append(
            SectorContrast(
                name=name,
                dimension=basis.shape[1],
                trace=float(np.real(np.trace(block))),
                trace_norm=float(np.sum(np.abs(values))),
                positive_gap=float(np.sum(np.maximum(values, 0.0))),
            )
        )
    return tuple(output)


def twirled_pure_state(
    vector: ArrayLike,
    unitaries: Sequence[ArrayLike],
) -> ComplexMatrix:
    return finite_group_twirl(pure_state_density(vector), unitaries)
