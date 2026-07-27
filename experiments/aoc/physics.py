"""Small, reproducible physical systems used by the research runs."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def ising_energy(spins: NDArray[np.int8]) -> float:
    """Energy per site of a periodic square-lattice Ising configuration."""

    array = np.asarray(spins, dtype=np.int8)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError("spins must be a square two-dimensional lattice.")
    bonds = np.sum(array * np.roll(array, -1, axis=0))
    bonds += np.sum(array * np.roll(array, -1, axis=1))
    return float(-bonds / array.size)


def _wolff_step(
    spins: NDArray[np.int8],
    temperature: float,
    rng: np.random.Generator,
) -> int:
    size = spins.shape[0]
    probability = 1.0 - np.exp(-2.0 / temperature)
    seed = (int(rng.integers(size)), int(rng.integers(size)))
    target = int(spins[seed])
    cluster = np.zeros_like(spins, dtype=bool)
    cluster[seed] = True
    stack = [seed]
    count = 1
    while stack:
        row, column = stack.pop()
        neighbors = (
            ((row + 1) % size, column),
            ((row - 1) % size, column),
            (row, (column + 1) % size),
            (row, (column - 1) % size),
        )
        for neighbor in neighbors:
            if (
                not cluster[neighbor]
                and spins[neighbor] == target
                and rng.random() < probability
            ):
                cluster[neighbor] = True
                stack.append(neighbor)
                count += 1
    spins[cluster] *= -1
    return count


def sample_ising(
    *,
    size: int,
    temperature: float,
    samples: int,
    seed: int,
    burn_in: int = 400,
    steps_between: int = 5,
    random_global_flip: bool = True,
) -> NDArray[np.int8]:
    """Draw Ising configurations with Wolff cluster updates."""

    if size < 2 or temperature <= 0 or samples <= 0:
        raise ValueError("Invalid Ising simulation parameters.")
    rng = np.random.default_rng(seed)
    spins = rng.choice(np.array([-1, 1], dtype=np.int8), size=(size, size))
    for _ in range(burn_in):
        _wolff_step(spins, temperature, rng)
    configurations = np.empty((samples, size, size), dtype=np.int8)
    for index in range(samples):
        for _ in range(steps_between):
            _wolff_step(spins, temperature, rng)
        configurations[index] = spins
        if random_global_flip and rng.random() < 0.5:
            configurations[index] *= -1
    return configurations


def fixed_end_chain_stiffness(
    masses: int,
    *,
    spring_constant: float = 1.0,
    damaged_spring: int | None = None,
    damage_fraction: float = 0.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64] | None]:
    """Stiffness of a chain with a spring to ground at both ends.

    Spring indices run from zero to ``masses``. Interior spring ``j`` connects
    masses ``j-1`` and ``j``. The returned damage incidence vector is useful
    as the analytical rank-one perturbation direction.
    """

    if masses <= 1 or spring_constant <= 0:
        raise ValueError("The chain needs at least two masses and positive k.")
    if not 0 <= damage_fraction < 1:
        raise ValueError("damage_fraction must lie in [0, 1).")
    incidence = []
    left = np.zeros(masses)
    left[0] = 1.0
    incidence.append(left)
    for index in range(1, masses):
        vector = np.zeros(masses)
        vector[index - 1] = -1.0
        vector[index] = 1.0
        incidence.append(vector)
    right = np.zeros(masses)
    right[-1] = 1.0
    incidence.append(right)

    stiffness = np.zeros((masses, masses), dtype=np.float64)
    damaged_vector = None
    for index, vector in enumerate(incidence):
        coefficient = spring_constant
        if damaged_spring is not None and index == damaged_spring:
            coefficient *= 1.0 - damage_fraction
            damaged_vector = vector.copy()
        stiffness += coefficient * np.outer(vector, vector)
    if damaged_spring is not None and damaged_vector is None:
        raise ValueError("damaged_spring index is outside [0, masses].")
    return stiffness, damaged_vector


def thermal_chain_model(
    masses: int,
    *,
    damaged_spring: int,
    damage_fraction: float,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Return whitened intact/damaged thermal covariances and damage mode."""

    intact_k, _ = fixed_end_chain_stiffness(masses)
    damaged_k, incidence = fixed_end_chain_stiffness(
        masses,
        damaged_spring=damaged_spring,
        damage_fraction=damage_fraction,
    )
    intact_covariance = np.linalg.inv(intact_k)
    damaged_covariance = np.linalg.inv(damaged_k)
    values, vectors = np.linalg.eigh(intact_covariance)
    square_root = (vectors * np.sqrt(values)) @ vectors.T
    inverse_square_root = (vectors * (1.0 / np.sqrt(values))) @ vectors.T
    baseline = inverse_square_root @ intact_covariance @ inverse_square_root
    damaged = inverse_square_root @ damaged_covariance @ inverse_square_root
    mode = square_root @ incidence
    mode = mode / np.linalg.norm(mode)
    return baseline, damaged, mode


def sample_directional_gaussian(
    covariance: NDArray[np.float64],
    samples: int,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Draw Gaussian samples and retain only their projective directions."""

    vectors = rng.multivariate_normal(
        np.zeros(covariance.shape[0]),
        covariance,
        size=samples,
    )
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
