import numpy as np
import pytest

pytest.importorskip("qiskit")
pytest.importorskip("qiskit_aer")

from deca.quantum import (
    simulate_povm_measurement,
    simulate_projective_measurement,
)
from deca.solvers import pretty_good_measurement


def trine_states_and_operators():
    phases = 2.0 * np.pi * np.arange(3) / 3.0
    states = np.column_stack(
        [
            np.full(3, 1.0 / np.sqrt(2.0), dtype=np.complex128),
            np.exp(1j * phases) / np.sqrt(2.0),
        ]
    )
    operators = [
        np.outer(state, state.conj()) / 3.0
        for state in states
    ]
    return states, operators


def test_ancilla_free_projective_simulator_matches_born_rule():
    state = np.array([1.0, 1.0]) / np.sqrt(2.0)
    basis = np.eye(2)
    assignment = np.array([0, 1])
    result = simulate_projective_measurement(
        state, basis, assignment, shots=8192, seed=13
    )
    assert result.num_ancilla_qubits == 0
    assert result.num_system_qubits == 1
    assert result.total_variation_error < 0.025
    np.testing.assert_allclose(result.analytic_probabilities, [0.5, 0.5])


def test_naimark_simulator_matches_trine_povm():
    states, operators = trine_states_and_operators()
    pgm = pretty_good_measurement(operators)
    result = simulate_povm_measurement(
        states[0], pgm.effects, shots=16384, seed=17
    )
    assert result.num_system_qubits == 1
    assert result.num_ancilla_qubits == 2
    assert result.total_variation_error < 0.025
    np.testing.assert_allclose(
        result.analytic_probabilities.sum(), 1.0, atol=1e-10
    )
