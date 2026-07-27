"""Qiskit Aer implementation of a learned projective contrast effect."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from .states import as_density_matrix


@dataclass(frozen=True)
class EffectSimulationResult:
    shots: int
    effect_counts: tuple[int, int]
    shot_probability: float
    analytic_probability: float
    absolute_error: float
    qubits: int
    circuit_depth: int
    transpiled_depth: int
    transpiled_size: int
    operation_counts: dict[str, int]
    circuit: Any
    transpiled_circuit: Any


def simulate_density_effect(
    state: ArrayLike,
    effect: ArrayLike,
    *,
    shots: int = 8192,
    seed: int = 0,
    tolerance: float = 1e-8,
) -> EffectSimulationResult:
    """Measure a projective effect on a mixed state using Qiskit Aer."""

    try:
        from qiskit import QuantumCircuit, transpile
        from qiskit_aer import AerSimulator
        from qiskit_aer.library import SetDensityMatrix
    except ImportError as error:
        raise ImportError(
            "Install './experiments[quantum]' for circuit simulation."
        ) from error

    density = as_density_matrix(state)
    projector = np.asarray(effect, dtype=np.complex128)
    projector = 0.5 * (projector + projector.conj().T)
    if projector.shape != density.shape:
        raise ValueError("state and effect dimensions do not match.")
    if not np.allclose(projector @ projector, projector, atol=tolerance):
        raise ValueError("effect must be projective for this circuit.")

    dimension = density.shape[0]
    padded_dimension = 1 << (dimension - 1).bit_length()
    padded_density = np.zeros((padded_dimension, padded_dimension), dtype=np.complex128)
    padded_density[:dimension, :dimension] = density
    padded_effect = np.zeros_like(padded_density)
    padded_effect[:dimension, :dimension] = projector

    values, vectors = np.linalg.eigh(padded_effect)
    positive = values > 0.5
    assignment = positive.astype(np.int64)
    qubits = int(np.log2(padded_dimension))
    circuit = QuantumCircuit(qubits, qubits)
    circuit.append(SetDensityMatrix(padded_density), range(qubits))
    circuit.unitary(vectors.conj().T, range(qubits), label="E_basis")
    circuit.measure(range(qubits), range(qubits))

    backend = AerSimulator()
    compiled = transpile(
        circuit,
        backend,
        optimization_level=1,
        seed_transpiler=seed,
    )
    result = backend.run(
        compiled,
        shots=shots,
        seed_simulator=seed,
    ).result()
    raw_counts = result.get_counts()
    success = 0
    for bitstring, count in raw_counts.items():
        outcome = int(bitstring.replace(" ", ""), 2)
        success += int(count) * int(assignment[outcome])
    shot_probability = success / shots
    analytic = float(np.real(np.trace(projector @ density)))
    return EffectSimulationResult(
        shots=shots,
        effect_counts=(shots - success, success),
        shot_probability=shot_probability,
        analytic_probability=analytic,
        absolute_error=abs(shot_probability - analytic),
        qubits=qubits,
        circuit_depth=circuit.depth(),
        transpiled_depth=compiled.depth(),
        transpiled_size=compiled.size(),
        operation_counts={
            str(name): int(count) for name, count in compiled.count_ops().items()
        },
        circuit=circuit,
        transpiled_circuit=compiled,
    )
