import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from deca.classifiers import DECAClassifier


def test_binary_classifier_api_and_probabilities():
    X, y = make_classification(
        n_samples=240,
        n_features=4,
        n_informative=4,
        n_redundant=0,
        class_sep=1.4,
        random_state=2,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=3
    )
    model = make_pipeline(
        StandardScaler(),
        DECAClassifier(
            measurement="helstrom",
            encoding="affine",
            encoding_scale=1.0,
        ),
    )
    model.fit(X_train, y_train)
    assert not hasattr(model[-1], "training_probabilities_")
    probabilities = model.predict_proba(X_test)
    predictions = model.predict(X_test)
    assert probabilities.shape == (len(X_test), 2)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    assert np.mean(predictions == y_test) > 0.65


def test_multiclass_jacobi_classifier_api():
    X, y = make_classification(
        n_samples=300,
        n_features=5,
        n_informative=5,
        n_redundant=0,
        n_classes=3,
        n_clusters_per_class=1,
        class_sep=1.5,
        random_state=8,
    )
    model = make_pipeline(
        StandardScaler(),
        DECAClassifier(
            measurement="jacobi",
            encoding="affine",
            random_starts=2,
            max_sweeps=30,
            random_state=5,
        ),
    )
    model.fit(X, y)
    probabilities = model.predict_proba(X[:20])
    assert probabilities.shape == (20, 3)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-9)


def test_binary_spectral_rule_matches_class_operator_discriminant():
    rng = np.random.default_rng(71)
    X = rng.normal(size=(80, 5))
    y = np.repeat([0, 1], 40)
    X[y == 1, 0] += 1.2
    model = DECAClassifier(
        measurement="auto",
        encoding="affine",
        decision_rule="spectral",
        priors="empirical",
        retain_training_operators=True,
    ).fit(X, y)
    states = model.encoder_.transform(X)
    full_scores = np.column_stack(
        [
            np.real(
                np.einsum("bi,ij,bj->b", states, operator, states)
            )
            for operator in model.weighted_operators_
        ]
    )
    assert np.array_equal(
        np.argmax(model.decision_function(X), axis=1),
        np.argmax(full_scores, axis=1),
    )
    np.testing.assert_allclose(
        model.predict_proba(X).sum(axis=1), 1.0
    )
