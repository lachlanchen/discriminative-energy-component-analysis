"""Classical-data encodings into normalized real or complex state vectors."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


def _as_2d_float(X: ArrayLike) -> FloatArray:
    array = np.asarray(X, dtype=np.float64)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D array, received shape {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError("Input contains NaN or infinite values.")
    return array


def _normalize_rows(X: FloatArray, zero_tolerance: float) -> FloatArray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    if np.any(norms <= zero_tolerance):
        raise ValueError(
            "Zero-norm samples cannot be amplitude encoded. "
            "Use affine encoding or remove the zero samples."
        )
    return X / norms


@dataclass(frozen=True)
class AmplitudeEncoder:
    """Map ``x`` to ``x / ||x||``."""

    zero_tolerance: float = 1e-12

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> "AmplitudeEncoder":
        _as_2d_float(X)
        return self

    def transform(self, X: ArrayLike) -> FloatArray:
        return _normalize_rows(_as_2d_float(X), self.zero_tolerance)

    def fit_transform(
        self, X: ArrayLike, y: ArrayLike | None = None
    ) -> FloatArray:
        return self.fit(X, y).transform(X)


@dataclass(frozen=True)
class AffineAmplitudeEncoder:
    """Map ``x`` to normalized ``[x, scale]``.

    The extra coordinate lets quadratic measurement scores contain linear and
    bias-like terms in the original features.
    """

    scale: float = 1.0
    zero_tolerance: float = 1e-12

    def fit(
        self, X: ArrayLike, y: ArrayLike | None = None
    ) -> "AffineAmplitudeEncoder":
        _as_2d_float(X)
        if not np.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("Affine scale must be finite and strictly positive.")
        return self

    def transform(self, X: ArrayLike) -> FloatArray:
        array = _as_2d_float(X)
        lifted = np.column_stack(
            [array, np.full(array.shape[0], self.scale, dtype=np.float64)]
        )
        return _normalize_rows(lifted, self.zero_tolerance)

    def fit_transform(
        self, X: ArrayLike, y: ArrayLike | None = None
    ) -> FloatArray:
        return self.fit(X, y).transform(X)


@dataclass(frozen=True)
class StereographicEncoder:
    """Inverse stereographic encoding used by prior HQC literature."""

    scale: float = 1.0

    def fit(
        self, X: ArrayLike, y: ArrayLike | None = None
    ) -> "StereographicEncoder":
        _as_2d_float(X)
        if not np.isfinite(self.scale) or self.scale <= 0:
            raise ValueError(
                "Stereographic scale must be finite and strictly positive."
            )
        return self

    def transform(self, X: ArrayLike) -> FloatArray:
        array = _as_2d_float(X)
        squared_norm = np.sum(array * array, axis=1, keepdims=True)
        scale_squared = self.scale * self.scale
        denominator = squared_norm + scale_squared
        first = 2.0 * self.scale * array / denominator
        last = (squared_norm - scale_squared) / denominator
        encoded = np.column_stack([first, last[:, 0]])
        # The formula is analytically normalized; renormalize to remove
        # floating-point drift.
        return encoded / np.linalg.norm(encoded, axis=1, keepdims=True)

    def fit_transform(
        self, X: ArrayLike, y: ArrayLike | None = None
    ) -> FloatArray:
        return self.fit(X, y).transform(X)


def make_encoder(name: str, scale: float = 1.0):
    normalized = name.strip().lower().replace("-", "_")
    if normalized == "amplitude":
        return AmplitudeEncoder()
    if normalized in {"affine", "affine_amplitude"}:
        return AffineAmplitudeEncoder(scale=scale)
    if normalized in {"stereographic", "stereo"}:
        return StereographicEncoder(scale=scale)
    raise ValueError(
        "Unknown encoding. Expected 'amplitude', 'affine', or 'stereographic'."
    )


def pad_states_to_power_of_two(
    states: ArrayLike,
) -> tuple[NDArray[np.complex128], int]:
    """Zero-pad normalized states to the next qubit-compatible dimension."""

    array = np.asarray(states, dtype=np.complex128)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2:
        raise ValueError("States must be a matrix with one state per row.")
    dimension = array.shape[1]
    padded_dimension = 1 << max(0, (dimension - 1).bit_length())
    if padded_dimension == dimension:
        return array.copy(), padded_dimension
    padded = np.zeros((array.shape[0], padded_dimension), dtype=np.complex128)
    padded[:, :dimension] = array
    return padded, padded_dimension
