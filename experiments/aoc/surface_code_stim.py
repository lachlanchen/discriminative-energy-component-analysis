"""Optional Stim/PyMatching adapter for controlled surface-code drift tests.

The primary Run 5 simulator has an exact, analytically controlled syndrome
model.  This module supplies a smaller circuit-level validation without making
Stim or PyMatching mandatory dependencies of :mod:`aoc`.

The injected post-change channel correlates pairs of physical ``X`` faults
while preserving the marginal fault probability on every data qubit.  It is
intended for ``surface_code:rotated_memory_z`` circuits, where ``X`` faults can
flip the stored logical-Z observable.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from typing import Any

import numpy as np
from numpy.typing import NDArray

DEFAULT_SENTINEL_PROBABILITY = 0.123456789
_PROBE_ERROR_PROBABILITY = 0.125

BoolMatrix = NDArray[np.bool_]
FloatVector = NDArray[np.float64]


def _require_stim() -> Any:
    try:
        import stim
    except ImportError as error:
        raise ImportError(
            "Stim is required for circuit-level surface-code validation. "
            "Install the optional dependency with `pip install stim==1.16.0`."
        ) from error
    return stim


def _require_pymatching() -> Any:
    try:
        import pymatching
    except ImportError as error:
        raise ImportError(
            "PyMatching is required for surface-code decoder comparisons. "
            "Install it with `pip install pymatching==2.4.0`."
        ) from error
    return pymatching


@dataclass(frozen=True)
class PairSupportAudit:
    """Whether proposed data-qubit pairs have disjoint detector supports."""

    valid: bool
    reused_qubits: tuple[int, ...]
    missing_qubits: tuple[int, ...]
    overlapping_detectors: tuple[tuple[int, int, tuple[int, ...]], ...]


@dataclass(frozen=True)
class SafePairSelection:
    """Greedy physically short pairing with a detector-disjoint certificate."""

    pairs: tuple[tuple[int, int], ...]
    detector_supports: Mapping[int, frozenset[int]]
    unpaired_qubits: tuple[int, ...]
    audit: PairSupportAudit


@dataclass(frozen=True)
class StimCircuitBuild:
    """A generated circuit together with the injected-channel metadata."""

    circuit: Any
    correlated_pairs: tuple[tuple[int, int], ...]
    residual_error_probability: float
    rounds_rewritten: int


@dataclass(frozen=True)
class DetectorSamples:
    """Detector events and logical-observable flips from identical shots."""

    detectors: BoolMatrix
    observables: BoolMatrix


@dataclass(frozen=True)
class DemGraphlikeAudit:
    """Audit of the graphlike components needed by PyMatching."""

    max_detectors_per_component: int
    separator_error_count: int
    undecomposed_error_count: int


@dataclass(frozen=True)
class DetectorMarginalAudit:
    """Exact detector marginals implied by two independent-error DEMs."""

    reference: FloatVector
    candidate: FloatVector
    max_absolute_gap: float
    within_tolerance: bool


@dataclass(frozen=True)
class DecoderComparison:
    """Paired stale, matched, and correlation-aware MWPM results."""

    stale_predictions: NDArray[np.uint8]
    matched_predictions: NDArray[np.uint8]
    correlated_predictions: NDArray[np.uint8]
    stale_weights: FloatVector
    matched_weights: FloatVector
    correlated_weights: FloatVector
    stale_logical_error_rate: float
    matched_logical_error_rate: float
    correlated_logical_error_rate: float


def marginal_preserving_residual_probability(
    marginal_probability: float,
    common_probability: float,
) -> float:
    """Return the independent residual rate after a common pair flip.

    Let ``C ~ Bernoulli(common_probability)`` be a common flip and
    ``A ~ Bernoulli(residual)`` an independent residual.  The final physical
    fault is ``C xor A``.  Choosing

    ``residual = (marginal - common) / (1 - 2 common)``

    makes its marginal probability exactly ``marginal``.
    """

    marginal = float(marginal_probability)
    common = float(common_probability)
    if not 0.0 <= marginal < 0.5:
        raise ValueError("marginal_probability must lie in [0, 0.5).")
    if not 0.0 <= common <= marginal:
        raise ValueError(
            "common_probability must lie between zero and the marginal rate."
        )
    denominator = 1.0 - 2.0 * common
    residual = (marginal - common) / denominator
    if not 0.0 <= residual <= 1.0:
        raise ValueError("The requested channel does not define a probability.")
    return float(residual)


def final_pair_joint_probability(
    residual_probability: float,
    common_probability: float,
) -> float:
    """Return ``P(E_a=1, E_b=1)`` for the common-plus-residual channel."""

    residual = float(residual_probability)
    common = float(common_probability)
    if not 0.0 <= residual <= 1.0 or not 0.0 <= common <= 1.0:
        raise ValueError("Channel arguments must be probabilities.")
    return float(common * (1.0 - residual) ** 2 + (1.0 - common) * residual**2)


def _is_sentinel_instruction(
    instruction: Any,
    sentinel_probability: float,
) -> bool:
    if instruction.name != "DEPOLARIZE1":
        return False
    arguments = instruction.gate_args_copy()
    return len(arguments) == 1 and math.isclose(
        arguments[0],
        sentinel_probability,
        rel_tol=0.0,
        abs_tol=1e-15,
    )


def _sentinel_data_qubits(
    circuit: Any,
    sentinel_probability: float,
) -> tuple[tuple[int, ...], int]:
    expected: tuple[int, ...] | None = None
    count = 0
    for instruction in circuit:
        if not _is_sentinel_instruction(instruction, sentinel_probability):
            continue
        current = tuple(target.qubit_value for target in instruction.targets_copy())
        if expected is None:
            expected = current
        elif current != expected:
            raise ValueError("Sentinel round markers target inconsistent data qubits.")
        count += 1
    if expected is None or count == 0:
        raise ValueError("No sentinel round markers were found in the circuit.")
    return expected, count


def audit_disjoint_pair_supports(
    pairs: Sequence[tuple[int, int]],
    detector_supports: Mapping[int, frozenset[int] | set[int]],
) -> PairSupportAudit:
    """Audit qubit uniqueness and detector-support disjointness."""

    seen: set[int] = set()
    reused: set[int] = set()
    missing: set[int] = set()
    overlaps: list[tuple[int, int, tuple[int, ...]]] = []
    for first_raw, second_raw in pairs:
        first = int(first_raw)
        second = int(second_raw)
        if first == second:
            reused.add(first)
        for qubit in (first, second):
            if qubit in seen:
                reused.add(qubit)
            seen.add(qubit)
            if qubit not in detector_supports:
                missing.add(qubit)
        if first not in detector_supports or second not in detector_supports:
            continue
        intersection = tuple(
            sorted(set(detector_supports[first]) & set(detector_supports[second]))
        )
        if intersection:
            overlaps.append((first, second, intersection))
    return PairSupportAudit(
        valid=not reused and not missing and not overlaps,
        reused_qubits=tuple(sorted(reused)),
        missing_qubits=tuple(sorted(missing)),
        overlapping_detectors=tuple(overlaps),
    )


def _error_detector_support(detector_error_model: Any) -> frozenset[int]:
    support: set[int] = set()
    for instruction in detector_error_model.flattened():
        if instruction.type != "error":
            continue
        for target in instruction.targets_copy():
            if target.is_relative_detector_id():
                support.add(int(target.val))
    return frozenset(support)


@cache
def _probe_fault_supports(
    distance: int,
    probe_rounds: int,
    sentinel_probability: float,
) -> tuple[
    tuple[int, ...],
    tuple[tuple[int, tuple[int, ...]], ...],
    tuple[tuple[int, tuple[float, ...]], ...],
]:
    stim = _require_stim()
    if distance < 2 or probe_rounds < 3:
        raise ValueError("Use distance >= 2 and at least three probe rounds.")
    base = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=distance,
        rounds=probe_rounds,
        before_round_data_depolarization=sentinel_probability,
    ).flattened()
    data_qubits, marker_count = _sentinel_data_qubits(
        base,
        sentinel_probability,
    )
    if marker_count != probe_rounds:
        raise RuntimeError(
            f"Expected {probe_rounds} round markers, found {marker_count}."
        )
    probe_round = probe_rounds // 2
    supports: list[tuple[int, tuple[int, ...]]] = []
    for qubit in data_qubits:
        probe = stim.Circuit()
        round_index = 0
        for instruction in base:
            if _is_sentinel_instruction(instruction, sentinel_probability):
                if round_index == probe_round:
                    probe.append(
                        "X_ERROR",
                        [qubit],
                        _PROBE_ERROR_PROBABILITY,
                    )
                round_index += 1
            else:
                probe.append(instruction)
        if round_index != probe_rounds:
            raise RuntimeError("Probe rewrite did not visit every round.")
        model = probe.detector_error_model(decompose_errors=False)
        supports.append(
            (qubit, tuple(sorted(_error_detector_support(model)))),
        )
    coordinates = base.get_final_qubit_coordinates()
    coordinate_items = tuple(
        (qubit, tuple(float(value) for value in coordinates.get(qubit, ())))
        for qubit in data_qubits
    )
    return data_qubits, tuple(supports), coordinate_items


def select_safe_disjoint_data_pairs(
    distance: int,
    *,
    probe_rounds: int = 3,
    sentinel_probability: float = DEFAULT_SENTINEL_PROBABILITY,
) -> SafePairSelection:
    """Greedily select short pairs whose single-fault symptoms do not overlap."""

    data, support_items, coordinate_items = _probe_fault_supports(
        int(distance),
        int(probe_rounds),
        float(sentinel_probability),
    )
    supports = {qubit: frozenset(detectors) for qubit, detectors in support_items}
    coordinates = dict(coordinate_items)

    def squared_distance(first: int, second: int) -> float:
        first_coordinates = coordinates.get(first, ())
        second_coordinates = coordinates.get(second, ())
        if len(first_coordinates) < 2 or len(second_coordinates) < 2:
            return float(abs(first - second))
        return float(
            sum(
                (first_coordinates[index] - second_coordinates[index]) ** 2
                for index in range(2)
            )
        )

    candidates: list[tuple[int, float, int, int]] = []
    for first_index, first in enumerate(data):
        if not supports[first]:
            continue
        for second in data[first_index + 1 :]:
            if not supports[second] or supports[first] & supports[second]:
                continue
            detector_count = len(supports[first]) + len(supports[second])
            graphlike_penalty = int(detector_count <= 2)
            candidates.append(
                (
                    graphlike_penalty,
                    squared_distance(first, second),
                    first,
                    second,
                )
            )
    candidates.sort()
    used: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for _, _, first, second in candidates:
        if first in used or second in used:
            continue
        pairs.append((first, second))
        used.update((first, second))
    audit = audit_disjoint_pair_supports(pairs, supports)
    if not audit.valid:
        raise RuntimeError("Internal safe-pair selection failed its own audit.")
    return SafePairSelection(
        pairs=tuple(pairs),
        detector_supports=supports,
        unpaired_qubits=tuple(qubit for qubit in data if qubit not in used),
        audit=audit,
    )


def _validate_pairs_against_data(
    pairs: Sequence[tuple[int, int]],
    data_qubits: Sequence[int],
) -> tuple[tuple[int, int], ...]:
    normalized = tuple((int(first), int(second)) for first, second in pairs)
    data_set = set(data_qubits)
    used: set[int] = set()
    for first, second in normalized:
        if first == second:
            raise ValueError("A correlated pair must contain distinct qubits.")
        if first not in data_set or second not in data_set:
            raise ValueError("Correlated-pair targets must be data qubits.")
        if first in used or second in used:
            raise ValueError("Each data qubit may occur in at most one pair.")
        used.update((first, second))
    return normalized


def _append_round_fault_channel(
    circuit: Any,
    *,
    data_qubits: tuple[int, ...],
    pairs: tuple[tuple[int, int], ...],
    marginal_probability: float,
    common_probability: float,
    correlated: bool,
) -> None:
    if marginal_probability == 0.0:
        return
    if not correlated or common_probability == 0.0:
        circuit.append("X_ERROR", list(data_qubits), marginal_probability)
        return
    residual = marginal_preserving_residual_probability(
        marginal_probability,
        common_probability,
    )
    paired = {qubit for pair in pairs for qubit in pair}
    unpaired = [qubit for qubit in data_qubits if qubit not in paired]
    if unpaired:
        circuit.append("X_ERROR", unpaired, marginal_probability)
    for first, second in pairs:
        circuit.append(
            "CORRELATED_ERROR",
            [circuit_target_x(first), circuit_target_x(second)],
            common_probability,
        )
        if residual:
            circuit.append("X_ERROR", [first, second], residual)


def circuit_target_x(qubit: int) -> Any:
    """Create an X Pauli target without importing Stim at module import time."""

    return _require_stim().target_x(int(qubit))


def build_rotated_memory_z_circuit(
    *,
    distance: int,
    rounds: int,
    marginal_data_error: float = 0.0,
    common_pair_error: float = 0.0,
    correlated_pairs: Sequence[tuple[int, int]] | None = None,
    change_round: int | None = None,
    after_clifford_depolarization: float = 0.0,
    before_measure_flip_probability: float = 0.0,
    after_reset_flip_probability: float = 0.0,
    sentinel_probability: float = DEFAULT_SENTINEL_PROBABILITY,
    audit_pairs: bool = True,
) -> StimCircuitBuild:
    """Build a rotated-memory-Z circuit with an optional correlation drift.

    Rounds before ``change_round`` receive independent ``X`` faults.  Rounds at
    or after it receive a common-plus-residual pair channel with the same
    one-qubit marginal.  If ``change_round`` is omitted, every round is in the
    correlated regime.  Setting ``common_pair_error=0`` produces the independent
    reference circuit.
    """

    stim = _require_stim()
    distance = int(distance)
    rounds = int(rounds)
    if distance < 2 or rounds < 1:
        raise ValueError("distance must be >= 2 and rounds must be positive.")
    marginal = float(marginal_data_error)
    common = float(common_pair_error)
    residual = marginal_preserving_residual_probability(marginal, common)
    sentinel = float(sentinel_probability)
    if not 0.0 < sentinel < 1.0:
        raise ValueError("sentinel_probability must lie strictly between 0 and 1.")
    if math.isclose(
        sentinel,
        float(after_clifford_depolarization),
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValueError("The sentinel must differ from after_clifford_depolarization.")
    if change_round is None:
        first_correlated_round = 0
    else:
        first_correlated_round = int(change_round)
        if not 0 <= first_correlated_round <= rounds:
            raise ValueError("change_round must lie between zero and rounds.")

    base = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=float(after_clifford_depolarization),
        before_round_data_depolarization=sentinel,
        before_measure_flip_probability=float(before_measure_flip_probability),
        after_reset_flip_probability=float(after_reset_flip_probability),
    ).flattened()
    data_qubits, marker_count = _sentinel_data_qubits(base, sentinel)
    if marker_count != rounds:
        raise RuntimeError(f"Expected {rounds} markers, found {marker_count}.")

    if common > 0.0:
        if correlated_pairs is None:
            selection = select_safe_disjoint_data_pairs(
                distance,
                sentinel_probability=sentinel,
            )
            pairs = selection.pairs
            if not pairs:
                raise ValueError("No detector-disjoint data-qubit pairs were found.")
        else:
            pairs = _validate_pairs_against_data(
                correlated_pairs,
                data_qubits,
            )
        if audit_pairs:
            _, support_items, _ = _probe_fault_supports(
                distance,
                3,
                sentinel,
            )
            support_map = {
                qubit: frozenset(support) for qubit, support in support_items
            }
            pair_audit = audit_disjoint_pair_supports(pairs, support_map)
            if not pair_audit.valid:
                raise ValueError(
                    "Correlated pairs do not have disjoint detector supports: "
                    f"{pair_audit}"
                )
    else:
        pairs = (
            ()
            if correlated_pairs is None
            else _validate_pairs_against_data(
                correlated_pairs,
                data_qubits,
            )
        )

    result = stim.Circuit()
    round_index = 0
    for instruction in base:
        if _is_sentinel_instruction(instruction, sentinel):
            _append_round_fault_channel(
                result,
                data_qubits=data_qubits,
                pairs=pairs,
                marginal_probability=marginal,
                common_probability=common,
                correlated=round_index >= first_correlated_round,
            )
            round_index += 1
        else:
            result.append(instruction)
    if round_index != rounds:
        raise RuntimeError("Round-fault rewrite did not visit every marker.")
    if any(_is_sentinel_instruction(instruction, sentinel) for instruction in result):
        raise RuntimeError("A sentinel noise instruction survived the rewrite.")
    return StimCircuitBuild(
        circuit=result,
        correlated_pairs=pairs,
        residual_error_probability=residual,
        rounds_rewritten=round_index,
    )


def sample_detector_observables(
    circuit: Any,
    *,
    shots: int,
    seed: int | None = None,
) -> DetectorSamples:
    """Sample detector events and logical flips from the same circuit shots."""

    if shots <= 0:
        raise ValueError("shots must be positive.")
    sampler = circuit.compile_detector_sampler(seed=seed)
    detectors, observables = sampler.sample(
        shots=int(shots),
        separate_observables=True,
    )
    return DetectorSamples(
        detectors=np.asarray(detectors, dtype=np.bool_),
        observables=np.asarray(observables, dtype=np.bool_),
    )


def audit_graphlike_dem(detector_error_model: Any) -> DemGraphlikeAudit:
    """Check that every separated DEM component is graphlike."""

    maximum = 0
    separators = 0
    undecomposed = 0
    for instruction in detector_error_model.flattened():
        if instruction.type != "error":
            continue
        targets = instruction.targets_copy()
        if any(target.is_separator() for target in targets):
            separators += 1
        for group in instruction.target_groups():
            detector_count = sum(target.is_relative_detector_id() for target in group)
            maximum = max(maximum, detector_count)
            if detector_count > 2:
                undecomposed += 1
    return DemGraphlikeAudit(
        max_detectors_per_component=maximum,
        separator_error_count=separators,
        undecomposed_error_count=undecomposed,
    )


def detector_marginals_from_dem(detector_error_model: Any) -> FloatVector:
    """Compute exact detector marginals for an independent-mechanism DEM.

    A DEM error instruction is one Bernoulli mechanism even when ``^`` separates
    its graphlike components.  Detector targets repeated within one instruction
    cancel modulo two.
    """

    factors = np.ones(
        int(detector_error_model.num_detectors),
        dtype=np.float64,
    )
    for instruction in detector_error_model.flattened():
        if instruction.type != "error":
            continue
        arguments = instruction.args_copy()
        if len(arguments) != 1:
            raise ValueError("A DEM error instruction must have one probability.")
        probability = float(arguments[0])
        if not 0.0 <= probability <= 1.0:
            raise ValueError("DEM error probabilities must lie in [0, 1].")
        toggled: set[int] = set()
        for target in instruction.targets_copy():
            if not target.is_relative_detector_id():
                continue
            detector = int(target.val)
            if detector in toggled:
                toggled.remove(detector)
            else:
                toggled.add(detector)
        if toggled:
            indices = np.fromiter(sorted(toggled), dtype=np.int64)
            factors[indices] *= 1.0 - 2.0 * probability
    return (1.0 - factors) / 2.0


def audit_detector_marginals(
    reference_model: Any,
    candidate_model: Any,
    *,
    tolerance: float = 1e-12,
) -> DetectorMarginalAudit:
    """Compare exact detector marginals from two DEMs."""

    reference = detector_marginals_from_dem(reference_model)
    candidate = detector_marginals_from_dem(candidate_model)
    if reference.shape != candidate.shape:
        raise ValueError("Detector error models have different detector counts.")
    gap = float(np.max(np.abs(reference - candidate), initial=0.0))
    return DetectorMarginalAudit(
        reference=reference,
        candidate=candidate,
        max_absolute_gap=gap,
        within_tolerance=gap <= tolerance,
    )


def _logical_error_rate(
    predictions: NDArray[np.uint8],
    actual_observables: NDArray[np.uint8],
) -> float:
    if predictions.shape != actual_observables.shape:
        raise ValueError("Predictions and logical observables have different shapes.")
    if len(predictions) == 0:
        raise ValueError("At least one decoded shot is required.")
    return float(np.mean(np.any(predictions != actual_observables, axis=1)))


def decode_stale_matched_correlated(
    detectors: NDArray[np.bool_] | NDArray[np.uint8],
    actual_observables: NDArray[np.bool_] | NDArray[np.uint8],
    *,
    reference_model: Any,
    candidate_model: Any,
) -> DecoderComparison:
    """Decode identical shots using stale, matched, and correlated MWPM."""

    pymatching = _require_pymatching()
    detector_data = np.ascontiguousarray(detectors, dtype=np.uint8)
    observables = np.ascontiguousarray(actual_observables, dtype=np.uint8)
    if detector_data.ndim != 2 or observables.ndim != 2:
        raise ValueError("Detector and observable samples must be row matrices.")
    if len(detector_data) != len(observables):
        raise ValueError("Detector and observable shot counts differ.")
    if reference_model.num_detectors != candidate_model.num_detectors:
        raise ValueError("Reference and candidate DEM detector counts differ.")
    if detector_data.shape[1] != candidate_model.num_detectors:
        raise ValueError("Detector sample width does not match the DEM.")

    reference_audit = audit_graphlike_dem(reference_model)
    candidate_audit = audit_graphlike_dem(candidate_model)
    if reference_audit.undecomposed_error_count:
        raise ValueError("Reference DEM contains non-graphlike error components.")
    if candidate_audit.undecomposed_error_count:
        raise ValueError("Candidate DEM contains non-graphlike error components.")

    stale = pymatching.Matching.from_detector_error_model(reference_model)
    matched = pymatching.Matching.from_detector_error_model(candidate_model)
    correlated = pymatching.Matching.from_detector_error_model(
        candidate_model,
        enable_correlations=True,
    )
    stale_predictions, stale_weights = stale.decode_batch(
        detector_data,
        return_weights=True,
    )
    matched_predictions, matched_weights = matched.decode_batch(
        detector_data,
        return_weights=True,
    )
    correlated_predictions, correlated_weights = correlated.decode_batch(
        detector_data,
        return_weights=True,
        enable_correlations=True,
    )
    stale_predictions = np.asarray(stale_predictions, dtype=np.uint8)
    matched_predictions = np.asarray(matched_predictions, dtype=np.uint8)
    correlated_predictions = np.asarray(
        correlated_predictions,
        dtype=np.uint8,
    )
    return DecoderComparison(
        stale_predictions=stale_predictions,
        matched_predictions=matched_predictions,
        correlated_predictions=correlated_predictions,
        stale_weights=np.asarray(stale_weights, dtype=np.float64),
        matched_weights=np.asarray(matched_weights, dtype=np.float64),
        correlated_weights=np.asarray(
            correlated_weights,
            dtype=np.float64,
        ),
        stale_logical_error_rate=_logical_error_rate(
            stale_predictions,
            observables,
        ),
        matched_logical_error_rate=_logical_error_rate(
            matched_predictions,
            observables,
        ),
        correlated_logical_error_rate=_logical_error_rate(
            correlated_predictions,
            observables,
        ),
    )
