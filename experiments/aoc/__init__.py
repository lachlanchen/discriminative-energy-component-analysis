"""Additive and symmetry-resolved maximum-observable contrast."""

from .change_detection import (
    BoundedScoreSR,
    HiddenMarkovBlockSR,
    LikelihoodRatioSR,
    PredictableBoxWitness,
    PredictableSimplexWitness,
    StaticBoxWitness,
    StaticSimplexWitness,
    effect_from_direction,
)
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
from .surface_code import (
    PeriodicSurfaceSyndromeModel,
    PeriodicSyndromeModel,
    logical_loop_access_no_go,
    periodic_boundary_syndrome,
    toggle_closed_logical_loop,
)
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
    "BoundedScoreSR",
    "ContrastResult",
    "HiddenMarkovBlockSR",
    "InvariantContrastResult",
    "LikelihoodRatioSR",
    "MulticlassObservableResult",
    "PeriodicSurfaceSyndromeModel",
    "PeriodicSyndromeModel",
    "PredictableBoxWitness",
    "PredictableContrastEProcess",
    "PredictableSimplexWitness",
    "SectorContrast",
    "SequentialRecord",
    "SlidingState",
    "StaticBoxWitness",
    "StaticSimplexWitness",
    "ToricCodeLattice",
    "as_density_matrix",
    "binary_measurement_success",
    "cyclic_translation_twirl",
    "effect_expectation",
    "effect_from_direction",
    "even_parity_flip_unitaries",
    "finite_group_twirl",
    "invariant_observable_contrast",
    "logical_loop_access_no_go",
    "maximum_observable_contrast",
    "minimum_error_observables",
    "parity_sector_state",
    "pauli_string_expectation",
    "periodic_boundary_syndrome",
    "projective_mmd_squared",
    "pure_state_density",
    "reduced_density_on_qubits",
    "symmetry_sector_contrasts",
    "toggle_closed_logical_loop",
    "toric_code_ground_state",
    "translation_power_state",
    "z_parity_projectors",
]

__version__ = "6.0.0"
