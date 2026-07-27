"""Additive and symmetry-resolved maximum-observable contrast."""

from .contrast import (
    ContrastResult,
    effect_expectation,
    maximum_observable_contrast,
    projective_mmd_squared,
)
from .gauge import (
    ToricCodeLattice,
    binary_measurement_success,
    even_parity_flip_unitaries,
    parity_sector_state,
    pauli_string_expectation,
    reduced_density_on_qubits,
    toric_code_ground_state,
    z_parity_projectors,
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
    "ToricCodeLattice",
    "as_density_matrix",
    "binary_measurement_success",
    "cyclic_translation_twirl",
    "effect_expectation",
    "even_parity_flip_unitaries",
    "finite_group_twirl",
    "invariant_observable_contrast",
    "maximum_observable_contrast",
    "minimum_error_observables",
    "parity_sector_state",
    "pauli_string_expectation",
    "projective_mmd_squared",
    "pure_state_density",
    "reduced_density_on_qubits",
    "symmetry_sector_contrasts",
    "toric_code_ground_state",
    "translation_power_state",
    "z_parity_projectors",
]

__version__ = "4.0.0"
