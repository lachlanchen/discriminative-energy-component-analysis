"""Discriminative Energy Component Analysis research implementation."""

__version__ = "1.0.0"

from .classifiers import DECAClassifier
from .encodings import (
    AffineAmplitudeEncoder,
    AmplitudeEncoder,
    StereographicEncoder,
    make_encoder,
    pad_states_to_power_of_two,
)
from .jacobi import jacobi_deca
from .operators import (
    class_density_operators,
    commutator_measure,
    measurement_probabilities,
    measurement_success,
    offdiagonal_residual,
    validate_povm,
)
from .solvers import (
    MeasurementSolution,
    binary_helstrom,
    optimal_povm,
    pretty_good_measurement,
)

__all__ = [
    "AffineAmplitudeEncoder",
    "AmplitudeEncoder",
    "DECAClassifier",
    "MeasurementSolution",
    "StereographicEncoder",
    "binary_helstrom",
    "class_density_operators",
    "commutator_measure",
    "jacobi_deca",
    "make_encoder",
    "measurement_probabilities",
    "measurement_success",
    "offdiagonal_residual",
    "optimal_povm",
    "pad_states_to_power_of_two",
    "pretty_good_measurement",
    "validate_povm",
    "__version__",
]
