import numpy as np

from deca.datasets import covariance_signal, mean_signal


def test_covariance_signal_has_equal_means_and_different_variances():
    dataset = covariance_signal(samples_per_class=20_000, random_state=17)
    first = dataset.X[dataset.y == 0]
    second = dataset.X[dataset.y == 1]
    assert np.linalg.norm(first.mean(axis=0) - second.mean(axis=0)) < 0.12
    assert first[:, 0].var() > 8.0 * second[:, 0].var()
    assert second[:, 1].var() > 8.0 * first[:, 1].var()


def test_mean_signal_has_opposite_class_means():
    dataset = mean_signal(samples_per_class=10_000, random_state=19)
    first = dataset.X[dataset.y == 0].mean(axis=0)
    second = dataset.X[dataset.y == 1].mean(axis=0)
    assert first[0] < -0.9
    assert second[0] > 0.9
    assert np.linalg.norm(first[1:] - second[1:]) < 0.12
