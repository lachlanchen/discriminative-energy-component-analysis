import numpy as np
from aoc import maximum_observable_contrast, minimum_error_observables


def test_orthogonal_multiclass_states_are_perfectly_distinguished():
    states = [np.diag(np.eye(3)[index]) for index in range(3)]
    result = minimum_error_observables(states)
    assert abs(result.success_probability - 1.0) < 1e-7
    assert result.completeness_error < 1e-7
    assert result.minimum_effect_eigenvalue > -1e-7


def test_identical_states_cannot_beat_prior_guessing():
    state = np.eye(3) / 3
    result = minimum_error_observables(
        [state, state, state],
        priors=[0.6, 0.3, 0.1],
    )
    assert abs(result.success_probability - 0.6) < 1e-7
    assert abs(result.advantage) < 1e-7


def test_binary_povm_matches_helstrom_success():
    first = np.diag([0.8, 0.2])
    second = np.diag([0.1, 0.9])
    binary = maximum_observable_contrast(first, second)
    expected = 0.5 + 0.25 * binary.trace_norm
    multiclass = minimum_error_observables([first, second])
    assert abs(multiclass.success_probability - expected) < 1e-7
