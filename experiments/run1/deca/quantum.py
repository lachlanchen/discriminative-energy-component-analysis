"""Qiskit simulation of DECA projective and general POVM measurements."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import null_space

from .operators import measurement_probabilities, validate_povm


ComplexArray = NDArray[np.complex128]


@dataclass
class QuantumSimulationResult:
    method: str
    shots: int
    class_counts: dict[int, int]
    shot_probabilities: NDArray[np.float64]
    analytic_probabilities: NDArray[np.float64]
    total_variation_error: float
    num_system_qubits: int
    num_ancilla_qubits: int
    circuit_depth: int
    transpiled_depth: int
    transpiled_size: int
    operation_counts: dict[str, int]
    circuit: Any
    transpiled_circuit: Any


def _require_qiskit():
    try:
        from qiskit import QuantumCircuit, transpile
        from qiskit.circuit.library import StatePreparation
        from qiskit_aer import AerSimulator
    except ImportError as error:
        raise ImportError(
            "Quantum simulation requires qiskit and qiskit-aer. "
            "Install './experiments[quantum]'."
        ) from error
    return QuantumCircuit, transpile, StatePreparation, AerSimulator


def _next_power_of_two(value: int) -> int:
    if value < 1:
        raise ValueError("Dimension must be positive.")
    return 1 << (value - 1).bit_length()


def _normalized_state(state: ArrayLike) -> ComplexArray:
    vector = np.asarray(state, dtype=np.complex128).reshape(-1)
    norm = np.linalg.norm(vector)
    if norm <= 1e-14:
        raise ValueError("Quantum state cannot have zero norm.")
    return vector / norm


def embed_projective_measurement(
    state: ArrayLike,
    basis: ArrayLike,
    assignment: ArrayLike,
    padding_class: int = 0,
) -> tuple[ComplexArray, ComplexArray, NDArray[np.int64]]:
    """Pad a state and PVM basis to a power-of-two dimension."""

    vector = _normalized_state(state)
    matrix = np.asarray(basis, dtype=np.complex128)
    labels = np.asarray(assignment, dtype=np.int64)
    dimension = vector.size
    if matrix.shape != (dimension, dimension):
        raise ValueError("basis shape must match the state dimension.")
    if labels.shape != (dimension,):
        raise ValueError("assignment must have one label per basis vector.")
    if not np.allclose(matrix.conj().T @ matrix, np.eye(dimension), atol=1e-8):
        raise ValueError("basis must be unitary.")

    padded_dimension = _next_power_of_two(dimension)
    padded_state = np.zeros(padded_dimension, dtype=np.complex128)
    padded_state[:dimension] = vector
    padded_basis = np.eye(padded_dimension, dtype=np.complex128)
    padded_basis[:dimension, :dimension] = matrix
    padded_assignment = np.full(
        padded_dimension, int(padding_class), dtype=np.int64
    )
    padded_assignment[:dimension] = labels
    return padded_state, padded_basis, padded_assignment


def _counts_to_class_probabilities(
    counts: dict[str, int], assignment: NDArray[np.int64], shots: int
) -> tuple[dict[int, int], NDArray[np.float64]]:
    num_classes = int(np.max(assignment)) + 1
    class_counts = {class_index: 0 for class_index in range(num_classes)}
    for bitstring, count in counts.items():
        outcome = int(bitstring.replace(" ", ""), 2)
        class_index = int(assignment[outcome])
        class_counts[class_index] += int(count)
    probabilities = np.array(
        [class_counts[index] / shots for index in range(num_classes)],
        dtype=np.float64,
    )
    return class_counts, probabilities


def simulate_projective_measurement(
    state: ArrayLike,
    basis: ArrayLike,
    assignment: ArrayLike,
    shots: int = 8192,
    seed: int = 0,
    optimization_level: int = 1,
    padding_class: int = 0,
) -> QuantumSimulationResult:
    """Run an ancilla-free DECA/PVM circuit on Qiskit Aer."""

    QuantumCircuit, transpile, StatePreparation, AerSimulator = _require_qiskit()
    padded_state, padded_basis, padded_assignment = embed_projective_measurement(
        state, basis, assignment, padding_class=padding_class
    )
    num_qubits = int(np.log2(padded_state.size))
    circuit = QuantumCircuit(num_qubits, num_qubits)
    circuit.append(StatePreparation(padded_state), range(num_qubits))
    circuit.unitary(
        padded_basis.conj().T,
        range(num_qubits),
        label="P_dagger",
    )
    circuit.measure(range(num_qubits), range(num_qubits))

    backend = AerSimulator()
    compiled = transpile(
        circuit,
        backend,
        optimization_level=optimization_level,
        seed_transpiler=seed,
    )
    result = backend.run(
        compiled, shots=shots, seed_simulator=seed
    ).result()
    counts = result.get_counts()
    class_counts, shot_probabilities = _counts_to_class_probabilities(
        counts, padded_assignment, shots
    )

    transformed = padded_basis.conj().T @ padded_state
    outcome_probabilities = np.abs(transformed) ** 2
    analytic_probabilities = np.bincount(
        padded_assignment,
        weights=outcome_probabilities,
        minlength=len(shot_probabilities),
    )
    total_variation = 0.5 * float(
        np.sum(np.abs(shot_probabilities - analytic_probabilities))
    )
    return QuantumSimulationResult(
        method="projective",
        shots=shots,
        class_counts=class_counts,
        shot_probabilities=shot_probabilities,
        analytic_probabilities=analytic_probabilities,
        total_variation_error=total_variation,
        num_system_qubits=num_qubits,
        num_ancilla_qubits=0,
        circuit_depth=circuit.depth(),
        transpiled_depth=compiled.depth(),
        transpiled_size=compiled.size(),
        operation_counts={
            str(name): int(count)
            for name, count in compiled.count_ops().items()
        },
        circuit=circuit,
        transpiled_circuit=compiled,
    )

def _matrix_sqrt_psd(matrix: ComplexArray) -> ComplexArray:
    hermitian = 0.5 * (matrix + matrix.conj().T)
    eigenvalues, eigenvectors = np.linalg.eigh(hermitian)
    if np.min(eigenvalues) < -1e-7:
        raise ValueError("POVM effect is not positive semidefinite.")
    roots = np.sqrt(np.clip(eigenvalues, 0.0, None))
    return (eigenvectors * roots) @ eigenvectors.conj().T


def embed_povm(
    state: ArrayLike, effects: Sequence[ArrayLike]
) -> tuple[ComplexArray, tuple[ComplexArray, ...]]:
    """Pad a state and its POVM effects to a qubit-compatible dimension."""

    vector = _normalized_state(state)
    matrices = tuple(
        np.asarray(effect, dtype=np.complex128) for effect in effects
    )
    dimension = vector.size
    if any(matrix.shape != (dimension, dimension) for matrix in matrices):
        raise ValueError("Every effect must match the state dimension.")
    padded_dimension = _next_power_of_two(dimension)
    if padded_dimension == dimension:
        return vector, matrices
    padded_state = np.zeros(padded_dimension, dtype=np.complex128)
    padded_state[:dimension] = vector
    padded_effects = []
    for matrix in matrices:
        effect = np.zeros(
            (padded_dimension, padded_dimension), dtype=np.complex128
        )
        effect[:dimension, :dimension] = matrix
        padded_effects.append(effect)
    # The original effects sum to identity only on the data subspace. Assign
    # the padded complement to the first outcome; padded input states have no
    # support there, so physical probabilities are unchanged.
    complement = np.eye(padded_dimension, dtype=np.complex128)
    complement[:dimension, :dimension] = 0.0
    padded_effects[0] = padded_effects[0] + complement
    return padded_state, tuple(padded_effects)


def naimark_unitary(
    effects: Sequence[ArrayLike], tolerance: float = 1e-7
) -> tuple[ComplexArray, int]:
    """Construct a unitary completion of the canonical POVM isometry."""

    matrices = tuple(
        np.asarray(effect, dtype=np.complex128) for effect in effects
    )
    diagnostics = validate_povm(matrices, tolerance=tolerance)
    if not diagnostics["valid"]:
        raise ValueError(f"Invalid POVM: {diagnostics}")
    dimension = matrices[0].shape[0]
    outcome_dimension = _next_power_of_two(len(matrices))
    blocks = [_matrix_sqrt_psd(matrix) for matrix in matrices]
    blocks.extend(
        np.zeros((dimension, dimension), dtype=np.complex128)
        for _ in range(outcome_dimension - len(blocks))
    )
    isometry = np.vstack(blocks)
    if not np.allclose(
        isometry.conj().T @ isometry,
        np.eye(dimension),
        atol=tolerance,
    ):
        raise RuntimeError("Canonical POVM isometry is not norm preserving.")
    complement = null_space(isometry.conj().T)
    unitary = np.column_stack([isometry, complement])
    if not np.allclose(
        unitary.conj().T @ unitary,
        np.eye(unitary.shape[0]),
        atol=5 * tolerance,
    ):
        raise RuntimeError("Naimark unitary completion failed.")
    return unitary, outcome_dimension


def simulate_povm_measurement(
    state: ArrayLike,
    effects: Sequence[ArrayLike],
    shots: int = 8192,
    seed: int = 0,
    optimization_level: int = 1,
) -> QuantumSimulationResult:
    """Run a Naimark-dilated general POVM circuit on Qiskit Aer."""

    QuantumCircuit, transpile, StatePreparation, AerSimulator = _require_qiskit()
    padded_state, padded_effects = embed_povm(state, effects)
    unitary, outcome_dimension = naimark_unitary(padded_effects)
    num_system_qubits = int(np.log2(padded_state.size))
    num_ancilla_qubits = int(np.log2(outcome_dimension))
    total_qubits = num_system_qubits + num_ancilla_qubits
    circuit = QuantumCircuit(total_qubits, num_ancilla_qubits)
    system_qubits = list(range(num_system_qubits))
    ancilla_qubits = list(range(num_system_qubits, total_qubits))
    circuit.append(StatePreparation(padded_state), system_qubits)
    circuit.unitary(unitary, range(total_qubits), label="Naimark")
    circuit.measure(ancilla_qubits, range(num_ancilla_qubits))

    backend = AerSimulator()
    compiled = transpile(
        circuit,
        backend,
        optimization_level=optimization_level,
        seed_transpiler=seed,
    )
    result = backend.run(
        compiled, shots=shots, seed_simulator=seed
    ).result()
    counts = result.get_counts()
    class_counts = {index: 0 for index in range(len(padded_effects))}
    for bitstring, count in counts.items():
        outcome = int(bitstring.replace(" ", ""), 2)
        if outcome < len(padded_effects):
            class_counts[outcome] += int(count)
    shot_probabilities = np.array(
        [class_counts[index] / shots for index in range(len(padded_effects))],
        dtype=np.float64,
    )
    analytic_probabilities = measurement_probabilities(
        padded_state.reshape(1, -1), padded_effects
    )[0]
    total_variation = 0.5 * float(
        np.sum(np.abs(shot_probabilities - analytic_probabilities))
    )
    return QuantumSimulationResult(
        method="naimark_povm",
        shots=shots,
        class_counts=class_counts,
        shot_probabilities=shot_probabilities,
        analytic_probabilities=analytic_probabilities,
        total_variation_error=total_variation,
        num_system_qubits=num_system_qubits,
        num_ancilla_qubits=num_ancilla_qubits,
        circuit_depth=circuit.depth(),
        transpiled_depth=compiled.depth(),
        transpiled_size=compiled.size(),
        operation_counts={
            str(name): int(count)
            for name, count in compiled.count_ops().items()
        },
        circuit=circuit,
        transpiled_circuit=compiled,
    )
