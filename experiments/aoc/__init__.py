"""Additive and symmetry-resolved maximum-observable contrast."""

from .contrast import (
    ContrastResult,
    effect_expectation,
    maximum_observable_contrast,
    projective_mmd_squared,
)
from .multiclass import MulticlassObservableResult, minimum_error_observables
from .states import (
    AdditiveState,
    SlidingState,
    as_density_matrix,
    pure_state_density,
)
from .streaming import PredictableContrastEProcess, SequentialRecord
from .symmetry import (
    InvariantContrastResult,
    SectorContrast,
    cyclic_translation_twirl,
    finite_group_twirl,
    invariant_observable_contrast,
    symmetry_sector_contrasts,
    translation_power_state,
)

__all__ = [
    "AdditiveState",
    "ContrastResult",
    "InvariantContrastResult",
    "MulticlassObservableResult",
    "PredictableContrastEProcess",
    "SectorContrast",
    "SequentialRecord",
    "SlidingState",
    "as_density_matrix",
    "cyclic_translation_twirl",
    "effect_expectation",
    "finite_group_twirl",
    "invariant_observable_contrast",
    "maximum_observable_contrast",
    "minimum_error_observables",
    "projective_mmd_squared",
    "pure_state_density",
    "symmetry_sector_contrasts",
    "translation_power_state",
]

__version__ = "3.0.0"
