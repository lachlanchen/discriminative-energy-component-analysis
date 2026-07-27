from __future__ import annotations

import numpy as np
import pytest
from aoc.surface_code_stim import (
    audit_detector_marginals,
    audit_disjoint_pair_supports,
    audit_graphlike_dem,
    build_rotated_memory_z_circuit,
    decode_stale_matched_correlated,
    final_pair_joint_probability,
    marginal_preserving_residual_probability,
    sample_detector_observables,
    select_safe_disjoint_data_pairs,
)


def test_common_plus_residual_channel_preserves_each_marginal() -> None:
    marginal = 0.08
    common = 0.03
    residual = marginal_preserving_residual_probability(marginal, common)
    reconstructed = common + residual - 2.0 * common * residual
    assert np.isclose(reconstructed, marginal)
    assert final_pair_joint_probability(residual, common) > marginal**2


@pytest.mark.parametrize(
    ("marginal", "common"),
    [(-0.1, 0.0), (0.5, 0.0), (0.1, -0.01), (0.1, 0.11)],
)
def test_invalid_pair_channel_parameters_fail(
    marginal: float,
    common: float,
) -> None:
    with pytest.raises(ValueError):
        marginal_preserving_residual_probability(marginal, common)


def test_pair_support_audit_detects_reuse_and_overlap() -> None:
    supports = {
        0: frozenset({1, 2}),
        1: frozenset({3}),
        2: frozenset({3, 4}),
    }
    valid = audit_disjoint_pair_supports(((0, 1),), supports)
    invalid = audit_disjoint_pair_supports(((0, 1), (1, 2)), supports)
    overlap = audit_disjoint_pair_supports(((1, 2),), supports)
    assert valid.valid
    assert not invalid.valid
    assert invalid.reused_qubits == (1,)
    assert not overlap.valid
    assert overlap.overlapping_detectors == ((1, 2, (3,)),)


def test_noiseless_distance_three_circuit_samples_zero_events() -> None:
    pytest.importorskip("stim")
    build = build_rotated_memory_z_circuit(
        distance=3,
        rounds=3,
        marginal_data_error=0.0,
        common_pair_error=0.0,
    )
    samples = sample_detector_observables(
        build.circuit,
        shots=32,
        seed=501,
    )
    assert build.rounds_rewritten == 3
    assert not np.any(samples.detectors)
    assert not np.any(samples.observables)


def test_distance_three_pair_channel_preserves_dem_marginals_and_decodes() -> None:
    pytest.importorskip("stim")
    pytest.importorskip("pymatching")
    selection = select_safe_disjoint_data_pairs(3)
    assert selection.pairs
    assert selection.audit.valid

    reference = build_rotated_memory_z_circuit(
        distance=3,
        rounds=3,
        marginal_data_error=0.02,
        common_pair_error=0.0,
    )
    candidate = build_rotated_memory_z_circuit(
        distance=3,
        rounds=3,
        marginal_data_error=0.02,
        common_pair_error=0.01,
        correlated_pairs=selection.pairs,
    )
    reference_model = reference.circuit.detector_error_model(
        decompose_errors=True,
    )
    candidate_model = candidate.circuit.detector_error_model(
        decompose_errors=True,
    )
    marginal_audit = audit_detector_marginals(
        reference_model,
        candidate_model,
        tolerance=1e-12,
    )
    graph_audit = audit_graphlike_dem(candidate_model)
    assert marginal_audit.within_tolerance
    assert marginal_audit.max_absolute_gap < 1e-12
    assert graph_audit.separator_error_count > 0
    assert graph_audit.undecomposed_error_count == 0

    samples = sample_detector_observables(
        candidate.circuit,
        shots=256,
        seed=502,
    )
    comparison = decode_stale_matched_correlated(
        samples.detectors,
        samples.observables,
        reference_model=reference_model,
        candidate_model=candidate_model,
    )
    assert comparison.stale_predictions.shape == samples.observables.shape
    assert comparison.matched_predictions.shape == samples.observables.shape
    assert comparison.correlated_predictions.shape == samples.observables.shape
    assert comparison.stale_weights.shape == (256,)
    assert 0.0 <= comparison.stale_logical_error_rate <= 1.0
    assert 0.0 <= comparison.matched_logical_error_rate <= 1.0
    assert 0.0 <= comparison.correlated_logical_error_rate <= 1.0
