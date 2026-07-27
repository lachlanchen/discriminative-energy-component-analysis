from __future__ import annotations

from itertools import combinations, product

import numpy as np
from aoc import (
    ToricCodeLattice,
    binary_measurement_success,
    even_parity_flip_unitaries,
    maximum_observable_contrast,
    parity_sector_state,
    pauli_string_expectation,
    reduced_density_on_qubits,
    toric_code_ground_state,
    z_parity_projectors,
)
from aoc.states import pure_state_density
from aoc.symmetry import finite_group_twirl, invariant_observable_contrast


def test_four_ground_sectors_and_stabilizers() -> None:
    lattice = ToricCodeLattice(3)
    labels = tuple(product((1, -1), repeat=2))
    states = [
        toric_code_ground_state(
            lattice,
            logical_z_x=logical_x,
            logical_z_y=logical_y,
        )
        for logical_x, logical_y in labels
    ]
    overlaps = np.asarray(
        [[np.vdot(first, second) for second in states] for first in states]
    )
    assert np.allclose(overlaps, np.eye(4))
    assert all(np.count_nonzero(state) == 256 for state in states)
    assert all(
        np.allclose(np.abs(state[np.flatnonzero(state)]), 1.0 / 16.0)
        for state in states
    )
    for (logical_x, logical_y), state in zip(labels, states, strict=True):
        for edges in lattice.all_stars():
            value = pauli_string_expectation(
                state,
                {edge: "X" for edge in edges},
            )
            assert np.isclose(value, 1.0)
        for edges in lattice.all_plaquettes():
            value = pauli_string_expectation(
                state,
                {edge: "Z" for edge in edges},
            )
            assert np.isclose(value, 1.0)
        assert np.isclose(
            pauli_string_expectation(
                state,
                {edge: "Z" for edge in lattice.logical_z_x_edges()},
            ),
            logical_x,
        )
        assert np.isclose(
            pauli_string_expectation(
                state,
                {edge: "Z" for edge in lattice.logical_z_y_edges()},
            ),
            logical_y,
        )


def test_subdistance_regions_are_blind_and_loop_is_perfect() -> None:
    lattice = ToricCodeLattice(3)
    positive = toric_code_ground_state(lattice, logical_z_x=1)
    negative = toric_code_ground_state(lattice, logical_z_x=-1)
    for size in (1, 2):
        for region in combinations(range(lattice.num_qubits), size):
            first = reduced_density_on_qubits(positive, region)
            second = reduced_density_on_qubits(negative, region)
            assert np.allclose(first, second)
    loop = lattice.logical_z_x_edges()
    first = reduced_density_on_qubits(positive, loop)
    second = reduced_density_on_qubits(negative, loop)
    result = maximum_observable_contrast(first, second)
    even, _ = z_parity_projectors(lattice.size)
    assert np.isclose(result.trace_norm / 2.0, 1.0)
    assert np.allclose(result.effect, even)


def test_parity_twirl_recovers_each_loop_sector_from_a_representative() -> None:
    num_qubits = 3
    group = even_parity_flip_unitaries(num_qubits)
    for basis_index, eigenvalue in ((0, 1), (1, -1)):
        vector = np.eye(1 << num_qubits, dtype=np.complex128)[:, basis_index]
        twirled = finite_group_twirl(pure_state_density(vector), group)
        assert np.allclose(
            twirled,
            parity_sector_state(num_qubits, eigenvalue),
        )


def test_correct_and_wrong_logical_nuisance_twirls() -> None:
    identity = np.eye(4, dtype=np.complex128)
    nuisance_flip = identity[[1, 0, 3, 2]]
    label_flip = identity[[2, 3, 0, 1]]
    positive_vector = identity[:, 0]
    negative_vector = identity[:, 2]
    positive_test = np.diag([0.5, 0.5, 0.0, 0.0]).astype(np.complex128)
    negative_test = np.diag([0.0, 0.0, 0.5, 0.5]).astype(np.complex128)

    correct = invariant_observable_contrast(
        pure_state_density(positive_vector),
        pure_state_density(negative_vector),
        (identity, nuisance_flip),
    )
    wrong = invariant_observable_contrast(
        pure_state_density(positive_vector),
        pure_state_density(negative_vector),
        (identity, label_flip),
    )
    logical_z = np.diag([1.0, 1.0, -1.0, -1.0])
    assert np.allclose(correct.contrast.sign_observable, logical_z)
    assert np.isclose(
        binary_measurement_success(
            correct.contrast.effect,
            positive_test,
            negative_test,
        ),
        1.0,
    )
    assert np.isclose(wrong.contrast.trace_norm, 0.0)
    assert np.isclose(
        binary_measurement_success(
            wrong.contrast.effect,
            positive_test,
            negative_test,
        ),
        0.5,
    )


def test_complex_partial_trace_and_pauli_y_phase() -> None:
    state = np.asarray([1.0, 0.0, 0.0, 1j]) / np.sqrt(2.0)
    assert np.allclose(
        reduced_density_on_qubits(state, (0,)),
        np.eye(2) / 2.0,
    )
    plus_y = np.asarray([1.0, 1j]) / np.sqrt(2.0)
    assert np.isclose(pauli_string_expectation(plus_y, {0: "Y"}), 1.0)
