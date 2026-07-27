"""Scikit-learn-compatible wrappers for DECA measurements."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

from .encodings import make_encoder
from .jacobi import jacobi_deca
from .operators import (
    class_density_operators,
    measurement_probabilities,
    weighted_class_operators,
)
from .solvers import (
    MeasurementSolution,
    binary_helstrom,
    optimal_povm,
    pretty_good_measurement,
)


class DECAClassifier(ClassifierMixin, BaseEstimator):
    """Classification through class-density measurement effects.

    Parameters
    ----------
    measurement:
        ``"auto"`` selects the binary closed form for two classes and
        Jacobi-DECA otherwise. Other choices are ``"helstrom"``, ``"jacobi"``,
        ``"pgm"``, and ``"sdp"``.
    encoding:
        ``"amplitude"``, ``"affine"``, or ``"stereographic"``.
    encoding_scale:
        Positive scale used by affine or stereographic encoding.
    priors:
        ``"empirical"``, ``"balanced"``, or an explicit prior sequence.
    decision_rule:
        ``"measurement"`` predicts from POVM outcome probabilities.
        ``"spectral"`` retains class-operator eigenvalue magnitudes and
        predicts from common-basis quadratic affinities.
    retain_training_operators:
        Keep the empirical class and weighted operators after fitting for
        research inspection. Disabled by default to reduce serialized model
        storage; it is not needed for prediction.
    """

    def __init__(
        self,
        measurement: str = "auto",
        encoding: str = "amplitude",
        encoding_scale: float = 1.0,
        priors: str | Sequence[float] = "balanced",
        decision_rule: str = "measurement",
        retain_training_operators: bool = False,
        random_starts: int = 8,
        max_sweeps: int = 100,
        tolerance: float = 1e-9,
        random_state: int = 0,
        sdp_solver: str | None = None,
    ) -> None:
        self.measurement = measurement
        self.encoding = encoding
        self.encoding_scale = encoding_scale
        self.priors = priors
        self.decision_rule = decision_rule
        self.retain_training_operators = retain_training_operators
        self.random_starts = random_starts
        self.max_sweeps = max_sweeps
        self.tolerance = tolerance
        self.random_state = random_state
        self.sdp_solver = sdp_solver

    def _solve(self, weighted_operators) -> MeasurementSolution:
        method = self.measurement.strip().lower()
        if method == "auto":
            method = "helstrom" if len(weighted_operators) == 2 else "jacobi"
        if method in {"helstrom", "binary"}:
            return binary_helstrom(
                weighted_operators, zero_tolerance=self.tolerance
            )
        if method in {"jacobi", "commuting", "deca"}:
            return jacobi_deca(
                weighted_operators,
                random_starts=self.random_starts,
                max_sweeps=self.max_sweeps,
                tolerance=self.tolerance,
                random_state=self.random_state,
            )
        if method in {"pgm", "pretty_good"}:
            return pretty_good_measurement(
                weighted_operators, tolerance=self.tolerance
            )
        if method in {"sdp", "povm", "optimal_povm"}:
            return optimal_povm(
                weighted_operators,
                solver=self.sdp_solver,
                tolerance=self.tolerance,
            )
        raise ValueError(
            "measurement must be auto, helstrom, jacobi, pgm, or sdp."
        )

    def fit(self, X: ArrayLike, y: ArrayLike):
        normalized_rule = self.decision_rule.strip().lower()
        if normalized_rule not in {"measurement", "spectral"}:
            raise ValueError(
                "decision_rule must be 'measurement' or 'spectral'."
            )
        self.decision_rule_ = normalized_rule
        self.encoder_ = make_encoder(self.encoding, self.encoding_scale)
        states = self.encoder_.fit_transform(X, y)
        self.classes_, class_operators, self.class_priors_ = (
            class_density_operators(states, y, priors=self.priors)
        )
        weighted_operators = weighted_class_operators(
            class_operators, self.class_priors_
        )
        self.solution_ = self._solve(weighted_operators)
        self.effects_ = self.solution_.effects
        self.score_operators_ = (
            self._spectral_score_operators(weighted_operators)
            if self.decision_rule_ == "spectral"
            else None
        )
        if self.retain_training_operators:
            self.class_operators_ = class_operators
            self.weighted_operators_ = weighted_operators
        self.n_features_in_ = np.asarray(X).shape[1]
        self.state_dimension_ = states.shape[1]
        self.training_single_shot_success_ = self.solution_.success
        return self

    def _spectral_score_operators(
        self, weighted_operators
    ) -> tuple[NDArray, ...] | None:
        if self.solution_.basis is None:
            return None
        basis = self.solution_.basis
        operators = []
        for weighted in weighted_operators:
            diagonal = np.real(
                np.diag(basis.conj().T @ weighted @ basis)
            )
            operators.append((basis * diagonal) @ basis.conj().T)
        return tuple(operators)

    def measurement_probabilities(
        self, X: ArrayLike
    ) -> NDArray[np.float64]:
        """Return actual POVM outcome probabilities."""

        check_is_fitted(self, ["encoder_", "effects_", "classes_"])
        states = self.encoder_.transform(X)
        return measurement_probabilities(states, self.effects_)

    def decision_function(self, X: ArrayLike) -> NDArray[np.float64]:
        """Return measurement probabilities or spectral affinities."""

        check_is_fitted(
            self,
            ["encoder_", "effects_", "classes_", "decision_rule_"],
        )
        states = self.encoder_.transform(X)
        if self.decision_rule_ == "measurement":
            return measurement_probabilities(states, self.effects_)
        if self.score_operators_ is None:
            raise ValueError(
                "Spectral prediction requires a common-basis solution."
            )
        scores = np.column_stack(
            [
                np.real(
                    np.einsum(
                        "bi,ij,bj->b",
                        states.conj(),
                        operator,
                        states,
                    )
                )
                for operator in self.score_operators_
            ]
        )
        scores[np.abs(scores) < 1e-12] = 0.0
        return np.clip(scores, 0.0, None)

    def predict_proba(self, X: ArrayLike) -> NDArray[np.float64]:
        scores = self.decision_function(X)
        totals = scores.sum(axis=1, keepdims=True)
        if np.any(totals <= 0):
            raise ValueError("Class scores have zero total mass.")
        return scores / totals

    def predict(self, X: ArrayLike) -> NDArray:
        scores = self.decision_function(X)
        return self.classes_[np.argmax(scores, axis=1)]

    def transform(self, X: ArrayLike) -> NDArray[np.float64]:
        """Return per-component Born energies when a basis solution exists."""

        check_is_fitted(self, ["encoder_", "solution_"])
        if self.solution_.basis is None:
            raise ValueError(
                "This measurement is not represented by one projective basis."
            )
        states = self.encoder_.transform(X)
        amplitudes = states @ self.solution_.basis.conj()
        return np.abs(amplitudes) ** 2

    def expected_single_shot_accuracy(
        self, X: ArrayLike, y: ArrayLike
    ) -> float:
        probabilities = self.measurement_probabilities(X)
        labels = np.asarray(y)
        lookup = {label: index for index, label in enumerate(self.classes_)}
        try:
            indices = np.array([lookup[label] for label in labels], dtype=int)
        except KeyError as error:
            raise ValueError("y contains a class not seen during fit.") from error
        return float(np.mean(probabilities[np.arange(len(indices)), indices]))
