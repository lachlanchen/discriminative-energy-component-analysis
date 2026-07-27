"""Exact small-system quantum many-body models for physical validation."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

ComplexVector = NDArray[np.complex128]
ComplexMatrix = NDArray[np.complex128]


def transverse_field_ising_hamiltonian(
    qubits: int,
    field: float,
    *,
    coupling: float = 1.0,
    periodic: bool = True,
) -> csr_matrix:
    """Sparse ``-J sum Z_i Z_j - g sum X_i`` Hamiltonian."""

    if qubits < 2 or field < 0 or coupling <= 0:
        raise ValueError("Invalid transverse-field Ising parameters.")
    dimension = 1 << qubits
    rows = []
    columns = []
    values = []
    bonds = qubits if periodic else qubits - 1
    for basis in range(dimension):
        diagonal = 0.0
        for site in range(bonds):
            neighbor = (site + 1) % qubits
            spin_site = 1 - 2 * ((basis >> site) & 1)
            spin_neighbor = 1 - 2 * ((basis >> neighbor) & 1)
            diagonal -= coupling * spin_site * spin_neighbor
        rows.append(basis)
        columns.append(basis)
        values.append(diagonal)
        for site in range(qubits):
            rows.append(basis ^ (1 << site))
            columns.append(basis)
            values.append(-field)
    return csr_matrix((values, (rows, columns)), shape=(dimension, dimension))


def ground_state(
    qubits: int,
    field: float,
) -> tuple[float, ComplexVector]:
    hamiltonian = transverse_field_ising_hamiltonian(qubits, field)
    values, vectors = eigsh(
        hamiltonian,
        k=1,
        which="SA",
        tol=1e-11,
        maxiter=50000,
    )
    vector = vectors[:, 0].astype(np.complex128)
    phase_index = int(np.argmax(np.abs(vector)))
    vector *= np.exp(-1j * np.angle(vector[phase_index]))
    return float(values[0]), vector


def reduced_density_matrix(
    state: NDArray[np.complex128],
    kept_qubits: int,
) -> ComplexMatrix:
    """Trace out the high-index qubits of a pure state."""

    vector = np.asarray(state, dtype=np.complex128)
    total_qubits = int(np.log2(vector.size))
    if (1 << total_qubits) != vector.size:
        raise ValueError("State dimension must be a power of two.")
    if not 0 < kept_qubits < total_qubits:
        raise ValueError("kept_qubits must define a proper subsystem.")
    kept_dimension = 1 << kept_qubits
    environment_dimension = 1 << (total_qubits - kept_qubits)
    reshaped = vector.reshape(environment_dimension, kept_dimension)
    density = reshaped.conj().T @ reshaped
    return 0.5 * (density + density.conj().T)


def pauli_x_tensor(qubits: int) -> ComplexMatrix:
    pauli_x = np.array([[0.0, 1.0], [1.0, 0.0]])
    result = np.array([[1.0]])
    for _ in range(qubits):
        result = np.kron(result, pauli_x)
    return result.astype(np.complex128)
