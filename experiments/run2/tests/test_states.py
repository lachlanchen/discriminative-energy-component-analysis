import numpy as np
from aoc import AdditiveState, SlidingState, pure_state_density


def test_add_merge_and_remove_are_batch_exact():
    rng = np.random.default_rng(17)
    vectors = rng.normal(size=(24, 7))
    weights = rng.uniform(0.2, 2.0, size=len(vectors))

    batch = AdditiveState.from_samples(vectors, weights)
    left = AdditiveState.from_samples(vectors[:9], weights[:9])
    right = AdditiveState.from_samples(vectors[9:], weights[9:])
    left.merge(right)

    np.testing.assert_allclose(left.accumulator, batch.accumulator, atol=2e-14)
    np.testing.assert_allclose(left.density, batch.density, atol=2e-14)
    assert abs(left.total_weight - weights.sum()) < 1e-13

    left.remove(vectors[-1], weights[-1])
    expected = AdditiveState.from_samples(vectors[:-1], weights[:-1])
    np.testing.assert_allclose(left.density, expected.density, atol=2e-14)


def test_sliding_state_matches_explicit_last_window():
    rng = np.random.default_rng(5)
    vectors = rng.normal(size=(13, 5))
    sliding = SlidingState(5, window_size=4)
    for index, vector in enumerate(vectors):
        sliding.add(vector)
        start = max(0, index - 3)
        expected = AdditiveState.from_samples(vectors[start : index + 1])
        np.testing.assert_allclose(sliding.density, expected.density, atol=2e-14)


def test_pure_state_is_phase_and_sign_invariant():
    vector = np.array([1.0, -2.0, 0.5])
    state = pure_state_density(vector)
    np.testing.assert_allclose(state, pure_state_density(-vector))
    np.testing.assert_allclose(
        state,
        pure_state_density(np.exp(0.37j) * vector),
        atol=1e-14,
    )
    assert abs(np.trace(state) - 1.0) < 1e-14
