"""Synthetic tests for the frozen Pittsburgh snapshot runner."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_ROOT = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_ROOT))

from run_pnnl_snapshot import (
    METHOD_ORDER,
    Cohort,
    DimensionAdaptedBank,
    _batched_spectral_effects,
    _stable_rank_one_effect,
    aggregate_results,
    bootstrap_threshold_scalar,
    bootstrap_threshold_vectorized,
    circular_block_indices,
    dimension_adapted_density,
    dimension_adapted_features,
    eligible_sparse_k,
    load_pnnl_config,
    load_snapshot_events,
    method_component_weights,
    randomization_audit,
    replay_actual,
    run_real,
    shot_component_weights,
    stream_metrics,
    synthetic_dry_run,
    validate_pnnl_config,
)


def _cohort(
    cohort_id: str,
    *,
    rounds: int = 2,
    m: int = 4,
    calibration_pair: str = "early--late",
) -> Cohort:
    return Cohort(
        cohort_id=cohort_id,
        distance=3,
        rounds=rounds,
        basis="X",
        register_suffix="path",
        data_qubits=(1, 3, 5),
        syndrome_qubits=(2, 4),
        oriented_path=(1, 2, 3, 4, 5),
        early_snapshot_id=f"{cohort_id}_early",
        late_snapshot_id=f"{cohort_id}_late",
        m=m,
        claim_label="circuit_and_hardware_domain_shift",
        calibration_pair_id=calibration_pair,
    )


def test_dimension_adapted_feature_density_and_priors() -> None:
    bits = np.asarray([0, 1], dtype=np.uint8)
    assert np.array_equal(
        dimension_adapted_features(bits),
        np.asarray([0.0, 1.0, 0.0]),
    )
    assert np.array_equal(
        dimension_adapted_density(bits),
        np.asarray([[0.5, -0.5], [-0.5, 0.5]]),
    )
    assert eligible_sparse_k(2) == (1,)
    assert eligible_sparse_k(4) == (1, 4)
    weights = method_component_weights(2)
    assert {method: len(value) for method, value in weights.items()} == {
        "dfr": 8,
        "online_logistic": 12,
        "space_sparse": 16,
        "space_spectral": 24,
        "space_composite": 40,
    }
    assert all(np.sum(value) == pytest.approx(1.0) for value in weights.values())

    shot_weights = shot_component_weights(2, 3)
    for method, value in shot_weights.items():
        base = len(weights[method])
        assert value.shape == (3 * base,)
        assert np.sum(value) == pytest.approx(1.0)
        for role in range(3):
            assert np.sum(value[role * base : (role + 1) * base]) == pytest.approx(
                1.0 / 3.0
            )


def test_bank_is_score_before_update_role_isolated_and_cloneable() -> None:
    bank = DimensionAdaptedBank(q=2, role_count=2)
    reference = np.asarray([0, 0], dtype=np.uint8)
    monitor = np.asarray([1, 0], dtype=np.uint8)
    first = bank.update_all(0, reference, monitor)
    for method in (
        "online_logistic",
        "space_sparse",
        "space_spectral",
        "space_composite",
    ):
        assert np.all(first[method] == 1.0)

    untouched_role = bank.update_all(1, reference, monitor)
    for method in (
        "online_logistic",
        "space_sparse",
        "space_spectral",
        "space_composite",
    ):
        assert np.all(untouched_role[method] == 1.0)

    informed = bank.update_all(0, reference, monitor)
    assert np.any(informed["online_logistic"] != 1.0)
    assert np.any(informed["space_sparse"] != 1.0)
    assert np.array_equal(bank.role_updates, np.asarray([2, 1]))
    clone = bank.clone()
    assert clone.state_digest() == bank.state_digest()
    clone.update_all(0, monitor, reference)
    assert clone.state_digest() != bank.state_digest()


def test_circular_block_indices_match_literal_replicate_loop() -> None:
    seed = 611004
    expected_rng = np.random.Generator(np.random.PCG64(seed))
    expected: list[np.ndarray] = []
    for _ in range(5):
        starts = expected_rng.integers(
            0,
            7,
            size=3,
            endpoint=False,
            dtype=np.int64,
        )
        expected.append(((starts[:, None] + np.arange(4)) % 7).reshape(-1)[:9])
    observed = circular_block_indices(
        rng=np.random.Generator(np.random.PCG64(seed)),
        replicates=5,
        source_shots=7,
        horizon_shots=9,
        block_length=4,
    )
    assert np.array_equal(observed, np.stack(expected))


@pytest.mark.parametrize("method", METHOD_ORDER)
def test_vectorized_bootstrap_matches_scalar_reference(method: str) -> None:
    rng = np.random.Generator(np.random.PCG64(614000))
    reference = rng.integers(0, 2, size=(8, 2, 4), dtype=np.uint8)
    monitor = rng.integers(0, 2, size=(8, 2, 4), dtype=np.uint8)
    bank = DimensionAdaptedBank(q=4, role_count=2)
    for shot in range(8):
        for role in range(2):
            bank.update_all(role, reference[shot, role], monitor[shot, role])
    scalar_threshold, scalar_maxima = bootstrap_threshold_scalar(
        fit_bank=bank,
        method=method,
        fit_reference=reference,
        fit_monitor=monitor,
        seed=614001,
        replicates=7,
        horizon_shots=10,
        block_length=3,
        alpha=0.2,
    )
    vector_threshold, vector_maxima, _ = bootstrap_threshold_vectorized(
        fit_bank=bank,
        method=method,
        fit_reference=reference,
        fit_monitor=monitor,
        seed=614001,
        replicates=7,
        horizon_shots=10,
        block_length=3,
        alpha=0.2,
    )
    assert np.allclose(vector_maxima, scalar_maxima, atol=1e-13, rtol=1e-13)
    assert vector_threshold == pytest.approx(
        scalar_threshold,
        abs=1e-13,
        rel=1e-13,
    )


def test_batched_spectral_tie_rule_matches_scalar_anchor() -> None:
    matrices = np.stack(
        (
            np.eye(4),
            np.diag([2.0, 2.0, -1.0, -3.0]),
            np.zeros((4, 4)),
        )
    )
    rank_one, positive = _batched_spectral_effects(matrices)
    for index, matrix in enumerate(matrices):
        assert np.allclose(rank_one[index], _stable_rank_one_effect(matrix))
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        selected = eigenvectors[:, eigenvalues > 1e-10]
        expected_positive = selected @ selected.T
        assert np.allclose(positive[index], expected_positive)
    assert np.array_equal(rank_one[0], np.diag([1.0, 0.0, 0.0, 0.0]))
    assert np.array_equal(rank_one[2], np.zeros((4, 4)))


def test_formal_trace_has_one_update_per_shared_orientation_shot() -> None:
    rng = np.random.Generator(np.random.PCG64(614010))
    pre_reference = rng.integers(0, 2, size=(3, 3, 2), dtype=np.uint8)
    pre_monitor = rng.integers(0, 2, size=(3, 3, 2), dtype=np.uint8)
    post_reference = rng.integers(0, 2, size=(4, 3, 2), dtype=np.uint8)
    post_monitor = rng.integers(0, 2, size=(4, 3, 2), dtype=np.uint8)
    result = replay_actual(
        DimensionAdaptedBank(q=2, role_count=3),
        pre_reference,
        pre_monitor,
        post_reference,
        post_monitor,
    )
    for method in METHOD_ORDER:
        assert result[method]["log_e"].shape == (7,)


def test_stream_metrics_use_shot_boundary_and_restricted_delay() -> None:
    trace = np.asarray([0.0, 0.1, 0.2, 0.3, 2.0, 2.1])
    result = stream_metrics(
        trace=trace,
        threshold=1.0,
        pre_surveillance_shots=2,
        post_shots=4,
        roles=7,
    )
    assert result == {
        "first_alarm_update": 4,
        "pre_false_alarm": 0,
        "miss": 0,
        "post_alarm_shot": 2,
        "post_alarm_role": None,
        "restricted_post_delay_fraction": 0.75,
    }

    pre_alarm = stream_metrics(
        trace=trace,
        threshold=0.05,
        pre_surveillance_shots=2,
        post_shots=4,
        roles=7,
    )
    assert pre_alarm["pre_false_alarm"] == 1
    assert pre_alarm["restricted_post_delay_fraction"] == 1.0


def test_aggregate_applies_equal_state_and_equal_cohort_retention() -> None:
    cohorts = (
        _cohort("c0", calibration_pair="a--b"),
        _cohort("c1", calibration_pair="c--d"),
    )
    delays = {
        "dfr": 0.6,
        "online_logistic": 0.5,
        "space_sparse": 0.4,
        "space_spectral": 0.3,
        "space_composite": 0.2,
    }
    rows = [
        {
            "cohort_id": cohort.cohort_id,
            "logical_state": state,
            "method": method,
            "pre_false_alarm": 0,
            "miss": 0,
            "restricted_post_delay_fraction": delay,
        }
        for cohort in cohorts
        for state in (0, 1)
        for method, delay in delays.items()
    ]
    result = aggregate_results(rows, cohorts, bootstrap_replicates=100)
    assert result["retention_pass"] is True
    for comparator in ("dfr", "online_logistic"):
        assert result["comparisons"][comparator]["retention_condition_pass"] is True


def test_strict_payload_parser_uses_syndrome_differences_without_terminal(
    tmp_path: Path,
) -> None:
    syndrome = [
        [0, 1, 1, 1],
        [1, 0, 0, 0],
        [1, 1, 1, 0],
    ]
    payload = [
        {
            "metadata": {
                "logical_state": state,
                "n_syndrome_rounds": 2,
                "basis": "X",
            },
            "per_shot_cregs": {
                "c_data_path": [[state, 0, 1] for _ in range(3)],
                "c_syndrome_path": syndrome,
            },
        }
        for state in (0, 1)
    ]
    path = tmp_path / "bitstrings.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    snapshot = {"metadata": [3, 2, "X", 3, 1, "date", "utc"]}
    events = load_snapshot_events(
        path,
        snapshot=snapshot,
        register_labels=("path",),
    )
    expected = np.asarray(
        [
            [[0, 1], [1, 0]],
            [[1, 0], [1, 0]],
            [[1, 1], [0, 1]],
        ],
        dtype=np.uint8,
    )
    assert np.array_equal(events[(0, "path")], expected)
    assert events[(0, "path")].shape == (3, 2, 2)
    assert np.array_equal(events[(0, "path")], events[(1, "path")])

    payload[0]["per_shot_cregs"]["c_syndrome_path"][0][0] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="integer 0/1"):
        load_snapshot_events(
            path,
            snapshot=snapshot,
            register_labels=("path",),
        )


def test_valid_final_data_is_invariant_through_the_locked_detection_pipeline(
    tmp_path: Path,
) -> None:
    """Final-data values are validated but cannot enter any detector endpoint."""

    shots = 12
    rng = np.random.Generator(np.random.PCG64(614015))
    syndrome_by_state = {
        state: rng.integers(0, 2, size=(shots, 4), dtype=np.uint8).tolist()
        for state in (0, 1)
    }
    data_a = {
        state: [
            [(shot + column + state) % 2 for column in range(3)]
            for shot in range(shots)
        ]
        for state in (0, 1)
    }
    data_b = {
        state: [[1 - value for value in row] for row in data_a[state]]
        for state in (0, 1)
    }

    def payload(data_by_state: dict[int, list[list[int]]]) -> list[dict[str, object]]:
        return [
            {
                "metadata": {
                    "logical_state": state,
                    "n_syndrome_rounds": 2,
                    "basis": "X",
                },
                "per_shot_cregs": {
                    "c_data_path": data_by_state[state],
                    "c_syndrome_path": syndrome_by_state[state],
                },
            }
            for state in (0, 1)
        ]

    path_a = tmp_path / "bitstrings_data_a.json"
    path_b = tmp_path / "bitstrings_data_b.json"
    path_a.write_text(json.dumps(payload(data_a)), encoding="utf-8")
    path_b.write_text(json.dumps(payload(data_b)), encoding="utf-8")
    snapshot = {"metadata": [3, 2, "X", shots, 1, "date", "utc"]}
    events_a = load_snapshot_events(
        path_a,
        snapshot=snapshot,
        register_labels=("path",),
    )
    events_b = load_snapshot_events(
        path_b,
        snapshot=snapshot,
        register_labels=("path",),
    )
    assert set(events_a) == set(events_b) == {(0, "path"), (1, "path")}
    for key in events_a:
        np.testing.assert_array_equal(events_a[key], events_b[key])

    cohort = _cohort("c_data_invariance", rounds=2, m=4)

    def pipeline(
        event_map: dict[tuple[int, str], np.ndarray],
    ) -> tuple[
        dict[int, dict[str, dict[str, object]]],
        list[dict[str, object]],
        dict[str, object],
    ]:
        state_results: dict[int, dict[str, dict[str, object]]] = {}
        retention_inputs: list[dict[str, object]] = []
        for logical_state in (0, 1):
            early = event_map[(logical_state, "path")]
            shifted = event_map[(1 - logical_state, "path")]
            pre_reference = early[:4]
            pre_monitor = early[4:8]
            post_reference = early[8:12]
            post_monitor = shifted[:4]

            bank = DimensionAdaptedBank(q=cohort.q, role_count=cohort.rounds)
            fit_factors = {method: [] for method in METHOD_ORDER}
            for shot in range(cohort.fit_shots):
                for role in range(cohort.rounds):
                    update = bank.update_all(
                        role,
                        pre_reference[shot, role],
                        pre_monitor[shot, role],
                    )
                    for method in METHOD_ORDER:
                        fit_factors[method].append(update[method].copy())

            actual = replay_actual(
                bank.clone(),
                pre_reference[cohort.fit_shots :],
                pre_monitor[cohort.fit_shots :],
                post_reference,
                post_monitor,
            )
            method_results: dict[str, dict[str, object]] = {}
            for method_index, method in enumerate(METHOD_ORDER):
                threshold, maxima, _ = bootstrap_threshold_vectorized(
                    fit_bank=bank,
                    method=method,
                    fit_reference=pre_reference[: cohort.fit_shots],
                    fit_monitor=pre_monitor[: cohort.fit_shots],
                    seed=614100 + 10 * logical_state + method_index,
                    replicates=5,
                    horizon_shots=cohort.surveillance_shots,
                    block_length=2,
                    alpha=0.25,
                )
                trace = actual[method]["log_e"]
                metrics = stream_metrics(
                    trace=trace,
                    threshold=threshold,
                    pre_surveillance_shots=cohort.pre_surveillance_shots,
                    post_shots=cohort.m,
                    roles=cohort.rounds,
                )
                method_results[method] = {
                    "fit_component_factors": np.stack(fit_factors[method]),
                    "fit_checkpoint_sha256": bank.state_digest(),
                    "threshold": threshold,
                    "bootstrap_maxima": maxima,
                    "log_e": trace,
                    "metrics": metrics,
                }
                retention_inputs.append(
                    {
                        "cohort_id": cohort.cohort_id,
                        "logical_state": logical_state,
                        "method": method,
                        "pre_false_alarm": metrics["pre_false_alarm"],
                        "miss": metrics["miss"],
                        "restricted_post_delay_fraction": metrics[
                            "restricted_post_delay_fraction"
                        ],
                    }
                )
            state_results[logical_state] = method_results
        aggregate = aggregate_results(
            retention_inputs,
            (cohort,),
            bootstrap_replicates=32,
        )
        return state_results, retention_inputs, aggregate

    results_a, retention_a, aggregate_a = pipeline(events_a)
    results_b, retention_b, aggregate_b = pipeline(events_b)
    for logical_state in (0, 1):
        for method in METHOD_ORDER:
            left = results_a[logical_state][method]
            right = results_b[logical_state][method]
            np.testing.assert_array_equal(
                left["fit_component_factors"],
                right["fit_component_factors"],
            )
            assert left["fit_checkpoint_sha256"] == right["fit_checkpoint_sha256"]
            assert left["threshold"] == right["threshold"]
            np.testing.assert_array_equal(
                left["bootstrap_maxima"],
                right["bootstrap_maxima"],
            )
            np.testing.assert_array_equal(left["log_e"], right["log_e"])
            assert left["metrics"] == right["metrics"]
    assert retention_a == retention_b
    assert aggregate_a == aggregate_b

    invalid_payload = payload(data_b)
    invalid_payload[0]["per_shot_cregs"]["c_data_path"][3][1] = 2
    invalid_path = tmp_path / "bitstrings_invalid_c_data.json"
    invalid_path.write_text(json.dumps(invalid_payload), encoding="utf-8")
    with pytest.raises(ValueError, match=r"state0\.c_data_path.*integer 0/1"):
        load_snapshot_events(
            invalid_path,
            snapshot=snapshot,
            register_labels=("path",),
        )


def test_randomization_audit_uses_shared_complete_shot_masks() -> None:
    cohort = _cohort("c0", rounds=2, m=4)
    rng = np.random.Generator(np.random.PCG64(614020))
    early = rng.integers(0, 2, size=(12, 2, 2), dtype=np.uint8)
    late = rng.integers(0, 2, size=(4, 2, 2), dtype=np.uint8)
    cache = {
        cohort.early_snapshot_id: {
            (0, "path"): early,
            (1, "path"): early ^ 1,
        },
        cohort.late_snapshot_id: {
            (0, "path"): late,
            (1, "path"): late ^ 1,
        },
    }
    summary, counts, maxima, _ = randomization_audit(
        cohorts=(cohort,),
        event_cache=cache,
        seeds=(610700, 610701, 610702),
    )
    assert counts.shape == (3, len(METHOD_ORDER))
    assert maxima.shape == counts.shape
    assert len(summary["path_state_method_rows"]) == 2 * len(METHOD_ORDER)
    assert summary["claim_scope"].endswith("not a natural hardware null")


def test_real_runner_cannot_reach_unblinding_before_freeze_gate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = REPO_ROOT / "experiments/run6/configs/pnnl_snapshot_locked.json"
    manifest = REPO_ROOT / "experiments/run6/configs/pnnl_pittsburgh_locked.json"
    ratification = tmp_path / "bad-ratification.json"
    repair_ratification = tmp_path / "bad-repair-ratification.json"
    ratification.write_text("{}", encoding="utf-8")
    repair_ratification.write_text("{}", encoding="utf-8")
    reached_unblinding = False

    class Validation:
        snapshots = 20
        cohorts = 11
        held_payloads_statted = 20

    monkeypatch.setattr(
        "run_pnnl_snapshot.validate_lock",
        lambda *args, **kwargs: Validation(),
    )

    def reject_freeze(*args: object, **kwargs: object) -> dict[str, object]:
        raise ValueError("freeze gate rejected")

    monkeypatch.setattr(
        "run_pnnl_snapshot.verify_freeze_ratification",
        reject_freeze,
    )

    def forbidden_unblinding(*args: object, **kwargs: object) -> None:
        nonlocal reached_unblinding
        reached_unblinding = True
        raise AssertionError("unblinding reached before freeze")

    monkeypatch.setattr(
        "run_pnnl_snapshot._first_unblinding_record",
        forbidden_unblinding,
    )
    args = argparse.Namespace(
        dry_run=False,
        config=config,
        manifest=manifest,
        freeze_ratification=ratification,
        repair_ratification=repair_ratification,
        output=tmp_path / "out",
    )
    with pytest.raises(ValueError, match="freeze gate rejected"):
        run_real(args)
    assert reached_unblinding is False


def test_repository_config_and_synthetic_dry_run() -> None:
    config = load_pnnl_config(
        REPO_ROOT / "experiments/run6/configs/pnnl_snapshot_locked.json"
    )
    assert config["protocol_id"] == "run6-pnnl-snapshot-v2"
    result = synthetic_dry_run()
    assert result["status"] == "synthetic_dry_run_passed"
    assert result["raw_run6_values_opened"] is False

    changed = copy.deepcopy(config)
    changed["cohort_filter"]["minimum_shots_per_state_per_cohort"] = 1
    with pytest.raises(ValueError, match="cohort filter"):
        validate_pnnl_config(changed)
