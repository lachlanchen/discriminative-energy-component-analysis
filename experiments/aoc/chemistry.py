"""Controlled one-electron density examples from Hückel theory."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def huckel_ring_hamiltonian(
    hoppings: ArrayLike,
) -> NDArray[np.float64]:
    """Nearest-neighbor cyclic tight-binding/Hückel Hamiltonian."""

    values = np.asarray(hoppings, dtype=np.float64)
    sites = len(values)
    if sites < 3 or np.any(values <= 0):
        raise ValueError("A ring needs at least three positive hoppings.")
    hamiltonian = np.zeros((sites, sites), dtype=np.float64)
    for site, hopping in enumerate(values):
        neighbor = (site + 1) % sites
        hamiltonian[site, neighbor] = -hopping
        hamiltonian[neighbor, site] = -hopping
    return hamiltonian


def occupied_one_particle_density(
    hamiltonian: ArrayLike,
    occupied_orbitals: int,
) -> NDArray[np.float64]:
    """Normalized projector onto the lowest one-electron orbitals."""

    matrix = np.asarray(hamiltonian, dtype=np.float64)
    values, vectors = np.linalg.eigh(matrix)
    if not 0 < occupied_orbitals < len(values):
        raise ValueError("occupied_orbitals must be a proper subspace.")
    occupied = vectors[:, :occupied_orbitals]
    density = occupied @ occupied.T
    return density / np.trace(density)
