import numpy as np
from aoc.chemistry import (
    huckel_ring_hamiltonian,
    occupied_one_particle_density,
)
from aoc.manybody import (
    ground_state,
    pauli_x_tensor,
    reduced_density_matrix,
)


def test_reduced_tfim_state_respects_subsystem_z2_parity():
    _, state = ground_state(8, field=0.8)
    reduced = reduced_density_matrix(state, kept_qubits=4)
    parity = pauli_x_tensor(4)
    assert np.linalg.norm(parity @ reduced - reduced @ parity) < 1e-8
    assert abs(np.trace(reduced) - 1.0) < 1e-12


def test_huckel_density_is_positive_and_difference_is_traceless():
    uniform = huckel_ring_hamiltonian(np.ones(6))
    dimerized = huckel_ring_hamiltonian(np.array([1.35, 0.65, 1.35, 0.65, 1.35, 0.65]))
    first = occupied_one_particle_density(uniform, 3)
    second = occupied_one_particle_density(dimerized, 3)
    assert np.min(np.linalg.eigvalsh(first)) > -1e-12
    assert np.min(np.linalg.eigvalsh(second)) > -1e-12
    assert abs(np.trace(first) - 1.0) < 1e-12
    assert abs(np.trace(second) - 1.0) < 1e-12
    assert abs(np.trace(second - first)) < 1e-12
