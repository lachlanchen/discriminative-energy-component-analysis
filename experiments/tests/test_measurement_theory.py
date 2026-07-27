import numpy as np

from deca.jacobi import jacobi_deca
from deca.operators import (
    commutator_measure,
    measurement_success,
    validate_povm,
)
from deca.solvers import (
    binary_helstrom,
    optimal_povm,
    pretty_good_measurement,
)


def random_density(dimension, rng, complex_data=False):
    raw = rng.normal(size=(dimension, dimension))
    if complex_data:
        raw = raw + 1j * rng.normal(size=(dimension, dimension))
    rho = raw @ raw.conj().T
    return rho / np.trace(rho)


def trine_weighted_operators():
    phases = 2.0 * np.pi * np.arange(3) / 3.0
    states = np.column_stack(
        [
            np.full(3, 1.0 / np.sqrt(2.0), dtype=np.complex128),
            np.exp(1j * phases) / np.sqrt(2.0),
        ]
    )
    return [
        np.outer(state, state.conj()) / 3.0
        for state in states
    ]


def test_binary_closed_form_matches_sdp():
    rng = np.random.default_rng(4)
    rhos = [random_density(4, rng) for _ in range(2)]
    operators = [0.35 * rhos[0], 0.65 * rhos[1]]
    closed = binary_helstrom(operators)
    oracle = optimal_povm(operators)
    assert closed.diagnostics["povm"]["valid"]
    assert oracle.diagnostics["povm"]["valid"]
    assert abs(closed.success - closed.diagnostics["closed_form_success"]) < 1e-10
    assert abs(closed.success - oracle.success) < 2e-6


def test_pgm_is_a_valid_measurement():
    rng = np.random.default_rng(10)
    operators = [
        random_density(3, rng) / 3.0
        for _ in range(3)
    ]
    solution = pretty_good_measurement(operators)
    assert solution.diagnostics["povm"]["valid"]
    assert 1.0 / 3.0 <= solution.success <= 1.0


def test_commuting_multiclass_deca_matches_sdp():
    rng = np.random.default_rng(12)
    dimension = 5
    raw = rng.normal(size=(dimension, dimension))
    common_basis, _ = np.linalg.qr(raw)
    operators = []
    for _ in range(3):
        spectrum = rng.uniform(0.1, 1.0, size=dimension)
        spectrum /= spectrum.sum()
        rho = (common_basis * spectrum) @ common_basis.T
        operators.append(rho / 3.0)
    assert commutator_measure(operators) < 1e-12

    deca = jacobi_deca(
        operators, random_starts=2, max_sweeps=50, random_state=3
    )
    oracle = optimal_povm(operators)
    assert abs(deca.success - oracle.success) < 2e-6
    assert deca.diagnostics["offdiagonal_residual"] < 1e-8


def test_jacobi_objective_is_monotone_and_gap_bound_holds():
    rng = np.random.default_rng(22)
    priors = np.array([0.2, 0.3, 0.5])
    operators = [
        prior * random_density(4, rng, complex_data=True)
        for prior in priors
    ]
    deca = jacobi_deca(
        operators, random_starts=3, max_sweeps=60, random_state=7
    )
    oracle = optimal_povm(operators, solver="SCS")
    history = np.asarray(deca.diagnostics["objective_history"])
    assert np.all(np.diff(history) >= -1e-11)
    gap = oracle.success - deca.success
    assert gap >= -3e-6
    assert gap <= deca.diagnostics["oracle_gap_upper_bound"] + 3e-6
    assert validate_povm(deca.effects)["valid"]
    assert abs(
        measurement_success(operators, deca.effects) - deca.success
    ) < 1e-10


def test_trine_states_require_a_general_povm_for_optimal_success():
    operators = trine_weighted_operators()
    deca = jacobi_deca(
        operators, random_starts=12, max_sweeps=100, random_state=9
    )
    pgm = pretty_good_measurement(operators)
    oracle = optimal_povm(operators, solver="SCS")
    assert abs(pgm.success - 2.0 / 3.0) < 1e-7
    assert abs(oracle.success - 2.0 / 3.0) < 2e-5
    assert oracle.success - deca.success > 0.02
