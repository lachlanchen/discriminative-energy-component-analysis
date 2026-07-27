import numpy as np

from deca.encodings import (
    AffineAmplitudeEncoder,
    AmplitudeEncoder,
    StereographicEncoder,
    pad_states_to_power_of_two,
)
from deca.operators import (
    class_density_operators,
    measurement_probabilities,
)


def test_all_encodings_produce_unit_states():
    X = np.array([[1.0, 2.0, -1.0], [-2.0, 0.5, 3.0]])
    encoders = [
        AmplitudeEncoder(),
        AffineAmplitudeEncoder(scale=0.7),
        StereographicEncoder(scale=1.3),
    ]
    expected_dimensions = [3, 4, 4]
    for encoder, dimension in zip(encoders, expected_dimensions):
        states = encoder.fit_transform(X)
        assert states.shape == (2, dimension)
        np.testing.assert_allclose(
            np.linalg.norm(states, axis=1), np.ones(2), atol=1e-12
        )


def test_affine_encoding_is_not_sign_invariant():
    encoder = AffineAmplitudeEncoder(scale=1.0)
    states = encoder.fit_transform([[1.0, 2.0], [-1.0, -2.0]])
    projectors = [np.outer(state, state) for state in states]
    assert not np.allclose(projectors[0], projectors[1])


def test_class_density_operators_are_trace_one():
    X = AmplitudeEncoder().fit_transform(
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]]
    )
    classes, rhos, priors = class_density_operators(
        X, [0, 0, 1, 1], priors="balanced"
    )
    np.testing.assert_array_equal(classes, [0, 1])
    np.testing.assert_allclose(priors, [0.5, 0.5])
    for rho in rhos:
        np.testing.assert_allclose(np.trace(rho), 1.0, atol=1e-12)
        assert np.min(np.linalg.eigvalsh(rho)) >= -1e-12


def test_measurement_probabilities_are_normalized():
    states = AmplitudeEncoder().fit_transform([[1.0, 1.0], [1.0, -1.0]])
    effects = [
        np.diag([1.0, 0.0]),
        np.diag([0.0, 1.0]),
    ]
    probabilities = measurement_probabilities(states, effects)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    np.testing.assert_allclose(probabilities, 0.5)


def test_padding_preserves_state_and_uses_power_of_two():
    states = AmplitudeEncoder().fit_transform([[1.0, 2.0, 3.0]])
    padded, dimension = pad_states_to_power_of_two(states)
    assert dimension == 4
    np.testing.assert_allclose(padded[0, :3], states[0])
    np.testing.assert_allclose(padded[0, 3], 0.0)
