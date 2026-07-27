"""Synthetic tests for the locked post-freeze Google outcome audit."""

from __future__ import annotations

import copy
import csv
import hashlib
import importlib.util
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
from aoc.run6_protocol import (
    canonical_json_bytes,
    environment_fingerprint,
    load_google_lock,
    sha256_file,
)

ROOT = Path(__file__).resolve().parents[3]
GOOGLE_LOCK = ROOT / "experiments/run6/configs/google2022_locked.json"
SCRIPT = ROOT / "experiments/run6/scripts/run_google2022_outcomes.py"
SPEC = importlib.util.spec_from_file_location("run6_google_outcomes", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = audit
SPEC.loader.exec_module(audit)


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _write_json(path: Path, value: object) -> None:
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def _artifact(path: Path) -> dict[str, object]:
    return {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _valid_method_input_parity_evidence(
    *,
    shot_count: int = 4,
    role_count: int = 2,
) -> dict[str, object]:
    cycles = {
        method: np.zeros((shot_count, role_count), dtype=np.float64)
        for method in audit.METHOD_IDS
    }
    labels = {
        label_id: np.zeros(shot_count, dtype=np.uint8) for label_id in audit.LABEL_IDS
    }
    archive_shots = np.arange(40_000, 40_000 + shot_count, dtype=np.int64)
    return audit.derive_method_input_parity_evidence(
        cycles,
        labels,
        archive_shots,
        expected_shot_count=shot_count,
        expected_role_count=role_count,
    )


def _resources() -> dict[str, object]:
    exposure = {
        "fit_warmup": {
            "paired_shots": 5_000,
            "physical_shots": 10_000,
            "paired_role_updates": 255_000,
            "detector_bits_exposed": 12_240_000,
        },
        "threshold": {
            "paired_shots": 5_000,
            "physical_shots": 10_000,
            "paired_role_updates": 255_000,
            "detector_bits_exposed": 12_240_000,
        },
        "held": {
            "paired_shots": 20_000,
            "physical_shots": 40_000,
            "paired_role_updates": 1_020_000,
            "detector_bits_exposed": 48_960_000,
        },
    }
    operation_keys = {
        "paired_shots",
        "paired_role_updates",
        "m0_scores",
        "m1_diagonal_scores",
        "m2_quadratic_scores",
        "m3_pairwise_logistic_scores",
        "m3_sgd_updates",
        "m4_shared_ewma_updates",
        "m4_sort_operations",
        "m4_sparse_dot_products",
        "m5_shared_ewma_updates",
        "m5_trace_scores",
        "m5_shared_eigendecompositions",
        "within_shot_page_cusum_channel_steps",
    }
    empty_operation = dict.fromkeys(operation_keys, 0)
    return {
        "record_exposure": exposure,
        "high_level_operations": {
            "fit_warmup": empty_operation,
            "threshold": empty_operation,
            "held": empty_operation,
            "m2_covariance_fits": 1,
            "m2_precision_matrix_constructions": 1,
            "m2_fit_observations_used": 20_000,
            "held_joint_replay_repetitions": 3,
            "extra_timing_replay_role_updates": 2_040_000,
        },
        "warm_checkpoint_storage": {
            "total_numeric_state_bytes": 0,
            "bytes_by_prefix": {},
            "array_inventory": {},
        },
        "held_final_checkpoint_storage": {
            "total_numeric_state_bytes": 0,
            "bytes_by_prefix": {},
            "array_inventory": {},
        },
        "formal_accumulator": {
            "time_unit": "complete_paired_shot",
            "held_updates": 20_000,
            "role_prior": "uniform_1_over_51",
            "within_shot_factor_compounding": False,
        },
        "output_bytes_before_manifest": 0,
    }


def _performance() -> dict[str, object]:
    return {
        "canonical_joint_pipeline_only": True,
        "not_a_per_method_speed_comparison": True,
        "integrity_and_layout_seconds": 0.0,
        "validation_read_seconds": 0.0,
        "warm_fit_replay_seconds": 0.0,
        "threshold_replay_seconds": 0.0,
        "threshold_serialization_seconds": 0.0,
        "held_read_seconds": 0.0,
        "held_replay_seconds": 0.0,
        "held_joint_replay_all_three_seconds": [0.0, 0.0, 0.0],
        "held_joint_replay_median_seconds": 0.0,
        "held_joint_replay_digests": ["0" * 64, "0" * 64, "0" * 64],
        "held_serialization_seconds": 0.0,
        "elapsed_before_manifest_seconds": 0.0,
        "peak_rss_kib_linux_ru_maxrss": 0,
        "relative_method_speed_claim_authorized": False,
    }


def _detector_manifest_fixture(tmp_path: Path) -> tuple[Path, Path]:
    config = load_google_lock(GOOGLE_LOCK)
    ratification = tmp_path / "freeze_ratification.json"
    _write_json(ratification, {"synthetic": True})
    required = audit.expected_detector_artifact_names()
    artifact_paths: list[Path] = []
    for index, name in enumerate(sorted(required)):
        path = tmp_path / name
        path.write_bytes(f"synthetic-{index}".encode())
        artifact_paths.append(path)
    threshold = tmp_path / "thresholds.json"
    threshold_stage = tmp_path / "threshold_stage_manifest.json"
    _write_json(
        threshold_stage,
        {
            "schema_version": "run6-google-threshold-stage-v1",
            "protocol_id": config["protocol_id"],
            "held_values_decoded_or_scored": False,
            "config_sha256": sha256_file(GOOGLE_LOCK),
            "method_spec_sha256": sha256_file(
                ROOT / config["normative_method_spec"]["path"]
            ),
            "detector_script_sha256": sha256_file(
                ROOT / "experiments/run6/scripts/run_google2022_detector.py"
            ),
            "warm_checkpoint_sha256": "2" * 64,
            "threshold_final_checkpoint_sha256": "3" * 64,
            "threshold_table_sha256": sha256_file(threshold),
            "threshold_artifacts": [
                _artifact(path)
                for path in artifact_paths
                if path.name in {"thresholds.json", "threshold_shots.csv"}
                or path.name.startswith("threshold__")
            ],
        },
    )
    manifest = {
        "schema_version": "run6-google-detector-freeze-v1",
        "protocol_id": config["protocol_id"],
        "detector_only": True,
        "outcome_accessed": False,
        "outcome_join_authorized": False,
        "git_commit": _git_commit(),
        "config_sha256": sha256_file(GOOGLE_LOCK),
        "method_spec_sha256": sha256_file(
            ROOT / config["normative_method_spec"]["path"]
        ),
        "detector_script_sha256": sha256_file(
            ROOT / "experiments/run6/scripts/run_google2022_detector.py"
        ),
        "freeze_ratification_sha256": sha256_file(ratification),
        "deviation_ledger": {
            "path": config["deviation_ledger"],
            "sha256": sha256_file(ROOT / config["deviation_ledger"]),
        },
        "circuit_sha256": "0" * 64,
        "detector_layout_index_sha256": "1" * 64,
        "warm_checkpoint_sha256": "2" * 64,
        "threshold_checkpoint_sha256": "3" * 64,
        "held_final_checkpoint_sha256": "4" * 64,
        "source_archive_sha256": config["source"]["sha256"],
        "source_archive_bytes": config["source"]["archive_bytes"],
        "verified_zip_member_sha256": {
            "circuit_ideal.stim": "5" * 64,
            config["source"]["detection_event_file"]: "6" * 64,
        },
        "detection_file_bytes": config["source"]["detection_event_file_bytes"],
        "threshold_table_sha256": sha256_file(threshold),
        "artifacts": [_artifact(path) for path in artifact_paths],
        "resources": _resources(),
        "performance": _performance(),
        "environment": environment_fingerprint(),
        "command": ["synthetic"],
        "started_unix": 0.0,
        "finished_unix": 1.0,
    }
    path = tmp_path / "detector_freeze_manifest.json"
    _write_json(path, manifest)
    return path, ratification


def _scores(shot_count: int) -> dict[str, np.ndarray]:
    base = np.linspace(0.0, 1.0, shot_count, dtype=np.float64)
    return {
        method: np.roll(base, index) for index, method in enumerate(audit.METHOD_IDS)
    }


def test_frozen_ranking_uses_exact_score_then_lower_archive_shot() -> None:
    scores = np.asarray([0.2, 0.8, 0.8, 0.1], dtype=np.float64)
    archive = np.asarray([40003, 40002, 40001, 40000], dtype=np.int64)
    ranking = audit.frozen_ranking(scores, archive)
    np.testing.assert_array_equal(ranking, [2, 1, 0, 3])


def test_primary_and_secondary_labels_are_xor_and_ranking_is_label_blind() -> None:
    shot_count = 10
    archive = np.arange(40_000, 40_000 + shot_count, dtype=np.int64)
    scores = _scores(shot_count)
    actual = np.asarray([0, 1, 1, 0, 1, 0, 0, 1, 0, 1], dtype=np.uint8)
    correlated = np.asarray([0, 0, 1, 1, 1, 0, 1, 1, 0, 0], dtype=np.uint8)
    primary = actual ^ correlated
    result = audit.one_label_risk_metrics(
        scores,
        primary,
        archive,
        budgets=(2, 4, 6),
    )
    ranking = audit.frozen_ranking(scores["space"], archive)
    expected = int(np.sum(primary[ranking[:2]], dtype=np.int64))
    assert result["space"]["budgets"]["2"]["captured_mismatches"] == expected
    assert (
        result["space"]["budgets"]["2"]["selected_archive_shots"]
        == archive[ranking[:2]].tolist()
    )

    changed_labels = 1 - primary
    changed = audit.one_label_risk_metrics(
        scores,
        changed_labels,
        archive,
        budgets=(2, 4, 6),
    )
    assert (
        changed["space"]["budgets"]["2"]["selected_archive_shots"]
        == result["space"]["budgets"]["2"]["selected_archive_shots"]
    )


def test_circular_bootstrap_is_exact_pcg64_draw_and_wrap() -> None:
    observed = audit.circular_block_bootstrap_indices(
        611000,
        shot_count=10,
        block_length=4,
    )
    starts = np.random.Generator(np.random.PCG64(611000)).integers(
        0,
        10,
        size=3,
    )
    expected = np.concatenate([np.arange(start, start + 4) % 10 for start in starts])[
        :10
    ]
    np.testing.assert_array_equal(observed, expected)


def test_count_based_bootstrap_capture_matches_explicit_resampled_ranking() -> None:
    shot_count = 17
    archive = np.arange(40_000, 40_000 + shot_count, dtype=np.int64)
    scores = np.asarray(
        [0.5, 0.7, 0.7, 0.1, 0.4, 0.3, 0.9, 0.2, 0.8] * 2,
        dtype=np.float64,
    )[:shot_count]
    labels = np.asarray(
        [0, 1, 0, 1, 1, 0, 1, 0, 1] * 2,
        dtype=np.uint8,
    )[:shot_count]
    indices = audit.circular_block_bootstrap_indices(
        611007,
        shot_count=shot_count,
        block_length=5,
    )
    counts = np.bincount(indices, minlength=shot_count)
    ranking = audit.frozen_ranking(scores, archive)
    captures = audit._top_budget_captures_from_counts(
        ranking,
        counts,
        labels,
        (2, 7, 12),
    )
    explicit_order = np.lexsort((archive[indices], -scores[indices]))
    for budget in (2, 7, 12):
        expected = int(np.sum(labels[indices[explicit_order[:budget]]], dtype=np.int64))
        assert captures[budget] == expected


def test_bootstrap_is_deterministic_and_reports_locked_comparator_differences() -> None:
    shot_count = 24
    archive = np.arange(40_000, 40_000 + shot_count, dtype=np.int64)
    scores = _scores(shot_count)
    labels = {
        "correlated_matching_mismatch": np.arange(shot_count, dtype=np.uint8) % 2,
        "pymatching_mismatch": (np.arange(shot_count, dtype=np.uint8) + 1) % 2,
    }
    first = audit.bootstrap_risk_uncertainty(
        scores,
        labels,
        archive,
        budgets=(2, 4, 8),
        replicates=10,
        seed_start=611000,
        block_length=6,
    )
    second = audit.bootstrap_risk_uncertainty(
        scores,
        labels,
        archive,
        budgets=(2, 4, 8),
        replicates=10,
        seed_start=611000,
        block_length=6,
    )
    assert canonical_json_bytes(first) == canonical_json_bytes(second)
    differences = first["space_comparator_difference_intervals"][
        "correlated_matching_mismatch"
    ]
    assert set(differences) == {"space_minus_m0", "space_minus_m3"}


def test_decision_requires_strict_top20_improvement_but_pnnl_remains_not_run() -> None:
    config = load_google_lock(GOOGLE_LOCK)
    event = {
        method: {
            "pre_event_alert_count": 0,
            "windows": {
                "primary": {"detected": method == "space"},
                "narrow": {"detected": False},
                "wide": {"detected": False},
            },
        }
        for method in audit.METHOD_IDS
    }
    risk = {
        method: {
            "budgets": {
                "20": {
                    "captured_mismatches": (
                        4 if method == "space" else 3 if method in {"m0", "m3"} else 0
                    )
                }
            }
        }
        for method in audit.METHOD_IDS
    }
    decision = audit.build_decision_summary(
        config=config,
        event_summary=event,
        primary_risk=risk,
        uncertainty={"replicates": 2_000},
        detector_freeze_verified=True,
        method_input_parity_evidence=_valid_method_input_parity_evidence(),
        repair_ratification_path=audit.REPAIR_RATIFICATION_RELATIVE,
        repair_ratification_sha256="f" * 64,
    )
    assert decision["google_primary_pass"] is True
    assert decision["pnnl_retention_pass"] == "not_run"
    assert decision["overall_run6_advantage"] is False

    risk["space"]["budgets"]["20"]["captured_mismatches"] = 3
    tied = audit.build_decision_summary(
        config=config,
        event_summary=event,
        primary_risk=risk,
        uncertainty={"replicates": 2_000},
        detector_freeze_verified=True,
        method_input_parity_evidence=_valid_method_input_parity_evidence(),
        repair_ratification_path=audit.REPAIR_RATIFICATION_RELATIVE,
        repair_ratification_sha256="f" * 64,
    )
    assert tied["google_primary_pass"] is False


def test_extra_detector_records_fail_derived_method_input_parity_predicate() -> None:
    shot_count = 4
    role_count = 2
    cycles = {
        method: np.zeros((shot_count, role_count), dtype=np.float64)
        for method in audit.METHOD_IDS
    }
    cycles["space"] = np.zeros((shot_count, role_count + 1), dtype=np.float64)
    labels = {
        label_id: np.asarray([0, 1, 0, 1], dtype=np.uint8)
        for label_id in audit.LABEL_IDS
    }
    archive_shots = np.arange(40_000, 40_000 + shot_count, dtype=np.int64)
    evidence = audit.derive_method_input_parity_evidence(
        cycles,
        labels,
        archive_shots,
        expected_shot_count=shot_count,
        expected_role_count=role_count,
    )
    valid_evidence = _valid_method_input_parity_evidence(
        shot_count=shot_count,
        role_count=role_count,
    )
    assert evidence["held_detector_score_inputs"]["space"][
        "record_count"
    ] == shot_count * (role_count + 1)
    assert evidence["all_methods_have_locked_detector_record_shape_and_count"] is False
    assert (
        evidence["shared_outcome_label_bundle"]["single_shared_bundle_for_all_methods"]
        is True
    )
    assert isinstance(
        evidence["shared_outcome_label_bundle"]["sha256"],
        str,
    )
    assert (
        evidence["no_method_received_extra_detector_records_or_outcome_labels"] is False
    )

    config = load_google_lock(GOOGLE_LOCK)
    event = {
        method: {
            "pre_event_alert_count": 0,
            "windows": {
                "primary": {"detected": method == "space"},
                "narrow": {"detected": False},
                "wide": {"detected": False},
            },
        }
        for method in audit.METHOD_IDS
    }
    risk = {
        method: {
            "budgets": {
                "20": {
                    "captured_mismatches": (
                        4 if method == "space" else 3 if method in {"m0", "m3"} else 0
                    )
                }
            }
        }
        for method in audit.METHOD_IDS
    }
    decision = audit.build_decision_summary(
        config=config,
        event_summary=event,
        primary_risk=risk,
        uncertainty={"replicates": 2_000},
        detector_freeze_verified=True,
        method_input_parity_evidence=evidence,
        repair_ratification_path=audit.REPAIR_RATIFICATION_RELATIVE,
        repair_ratification_sha256="f" * 64,
    )
    assert (
        decision["atomic_predicates"][
            "no_method_received_extra_detector_records_or_outcome_labels"
        ]
        is False
    )
    assert decision["google_primary_pass"] is False
    assert (
        valid_evidence["no_method_received_extra_detector_records_or_outcome_labels"]
        is True
    )


def test_final_aggregation_requires_both_manifests_and_uses_locked_conjunction(
    tmp_path: Path,
) -> None:
    randomization_manifest = tmp_path / "randomization_manifest.json"
    pnnl_manifest = tmp_path / "pnnl_results_manifest.json"
    randomization_manifest.write_text("randomization\n", encoding="utf-8")
    pnnl_manifest.write_text("pnnl\n", encoding="utf-8")
    assert audit.require_paired_final_inputs(None, None) is False
    assert (
        audit.require_paired_final_inputs(
            randomization_manifest,
            pnnl_manifest,
        )
        is True
    )
    with pytest.raises(ValueError, match="requires both"):
        audit.require_paired_final_inputs(randomization_manifest, None)

    partial = {
        "summary_scope": "google_detector_and_outcome_only_not_full_run6_summary",
        "google_primary_pass": True,
        "randomization_audit": "not_run_in_this_command",
        "pnnl_retention_pass": "not_run",
        "overall_run6_advantage": False,
        "negative_result_reasons": ["pnnl_retention_not_run"],
    }
    randomization_result = {
        "crossing_counts_at_100": {"space": 3},
        "replicate_count": 256,
        "space_crossing_clopper_pearson_95": {
            "lower": 0.0,
            "upper": 0.1,
        },
    }
    completed = audit.integrate_full_run6_decision(
        partial,
        randomization_manifest_path=randomization_manifest,
        randomization_result=randomization_result,
        pnnl_manifest_path=pnnl_manifest,
        pnnl_manifest={"retention_pass": True},
    )
    assert completed["summary_scope"] == "full_run6_locked_decision"
    assert completed["overall_run6_advantage"] is True
    assert completed["negative_result_reasons"] == []
    assert completed["randomization_audit"]["changes_primary_boolean"] is False

    failed = audit.integrate_full_run6_decision(
        partial,
        randomization_manifest_path=randomization_manifest,
        randomization_result=randomization_result,
        pnnl_manifest_path=pnnl_manifest,
        pnnl_manifest={"retention_pass": False},
    )
    assert failed["overall_run6_advantage"] is False
    assert "pnnl_retention_failed" in failed["negative_result_reasons"]


def test_randomization_manifest_is_strictly_hash_verified(
    tmp_path: Path,
) -> None:
    config = load_google_lock(GOOGLE_LOCK)
    ratification = tmp_path / "freeze_ratification.json"
    repair_ratification = tmp_path / "repair_ratification.json"
    ratification.write_text("synthetic freeze\n", encoding="utf-8")
    repair_ratification.write_text("synthetic repair\n", encoding="utf-8")
    detector_path = tmp_path / "detector_freeze_manifest.json"
    detector_path.write_text("synthetic detector\n", encoding="utf-8")
    commit = _git_commit()
    detector = {
        "git_commit": commit,
        "warm_checkpoint_sha256": "a" * 64,
        "verified_zip_member_sha256": {
            "circuit_ideal.stim": "b" * 64,
            config["source"]["detection_event_file"]: "c" * 64,
        },
    }
    methods = ("m0", "m1", "m3", "m4", "m5", "space")
    formal_experts = {
        "m0": 408,
        "m1": 408,
        "m3": 612,
        "m4": 3264,
        "m5": 1224,
        "space": 4488,
    }
    rows = [
        {
            "replicate_index": index,
            "seed": 610700 + index,
            "swap_sha256": f"{index:064x}",
            "swapped_shot_count": 2_500,
            "checkpoint_restored": True,
            "crossed_100": {method: False for method in methods},
            "first_crossing_shot_number_one_based": {
                method: None for method in methods
            },
            "maximum_log_e": {method: 0.0 for method in methods},
            "final_log_e": {method: 0.0 for method in methods},
            "familywide_any_crossed_600": False,
            "formal_eprocess_updates": 5_000,
            "role_score_updates": 255_000,
            "formal_experts": formal_experts,
        }
        for index in range(256)
    ]
    interval = audit.clopper_pearson_interval(0, 256)
    result = {
        "schema_version": "run6-google-randomization-result-v1",
        "primary_method": "space",
        "primary_statistic": "ever_proper_prior_eprocess_ge_100",
        "replicate_count": 256,
        "seed_start": 610700,
        "seed_stop_exclusive": 610956,
        "rng": "numpy.random.Generator(PCG64)",
        "one_orientation_draw_per_replicate": True,
        "complete_shot_swap_shared_across_roles": True,
        "horizon_paired_shots": 5_000,
        "role_score_updates_per_replicate": 255_000,
        "formal_expert_index": "role_then_locked_base_component",
        "formal_role_prior": "uniform_1_over_role_count",
        "warm_checkpoint_sha256": "a" * 64,
        "crossing_counts_at_100": {method: 0 for method in methods},
        "space_crossing_fraction": 0.0,
        "space_crossing_clopper_pearson_95": {
            "lower": interval[0],
            "upper": interval[1],
        },
        "familywide_any_crossing_count_at_600": 0,
        "interpretation": "exact_design_based_implementation_diagnostic",
        "replicates": rows,
    }
    result_path = tmp_path / "randomization_result.json"
    _write_json(result_path, result)
    threshold = {
        "schema_version": "run6-google-threshold-bootstrap-v1",
        "status": "descriptive_only_does_not_replace_frozen_threshold",
        "unit": "complete_paired_shot",
        "block_length_shots": 128,
        "replicates": 2_000,
        "seed_start": 613_000,
        "seed_stop_exclusive": 615_000,
        "rng": "numpy.random.Generator(PCG64)",
        "blocks_per_replicate": 40,
        "primary_maximum_alerts": 2,
        "secondary_maximum_alerts": 0,
        "summaries": {},
        "replicate_results": [{} for _ in range(2_000)],
    }
    threshold_path = tmp_path / "threshold_bootstrap.json"
    _write_json(threshold_path, threshold)
    manifest = {
        "schema_version": "run6-google-randomization-manifest-v1",
        "protocol_id": config["protocol_id"],
        "config_sha256": sha256_file(GOOGLE_LOCK),
        "method_spec_sha256": sha256_file(
            ROOT / config["normative_method_spec"]["path"]
        ),
        "freeze_ratification_sha256": sha256_file(ratification),
        "repair_ratification_path": audit.REPAIR_RATIFICATION_RELATIVE,
        "repair_ratification_sha256": sha256_file(repair_ratification),
        "detector_manifest_sha256": sha256_file(detector_path),
        "detector_manifest_git_commit": commit,
        "script_sha256": sha256_file(
            ROOT / "experiments/run6/scripts/run_google2022_randomization.py"
        ),
        "git_commit": commit,
        "outcome_accessed": False,
        "source_archive_sha256": config["source"]["sha256"],
        "verified_zip_member_sha256": detector["verified_zip_member_sha256"],
        "warm_checkpoint_sha256": detector["warm_checkpoint_sha256"],
        "rng": {
            "algorithm": "numpy.random.Generator(PCG64)",
            "randomization_seed_start": 610700,
            "randomization_seed_stop_exclusive": 610956,
            "threshold_bootstrap_seed_start": 613000,
            "threshold_bootstrap_seed_stop_exclusive": 615000,
        },
        "execution_mode": "deterministic_gap_free_shard_merge",
        "merge_evidence": {
            "input_shard_count": 1,
            "input_shards": [
                {
                    "replicate_start": 0,
                    "replicate_stop_exclusive": 256,
                    "manifest_sha256": "d" * 64,
                    "resources": {
                        "replicate_count": 256,
                        "formal_eprocess_shot_updates": 256 * 5_000,
                        "role_score_updates": 256 * 255_000,
                        "wall_seconds": 1.0,
                        "peak_rss_kib_linux_ru_maxrss": 1,
                        "worker_process_count": 1,
                        "external_concurrency_not_inferred": True,
                        "output_bytes_excluding_manifest": 1,
                        "output_bytes_including_manifest": 2,
                    },
                }
            ],
            "replicate_index_range": [0, 256],
            "seed_range": [610700, 610956],
            "replicate_indices_sha256": hashlib.sha256(
                np.arange(256, dtype="<i8").tobytes()
            ).hexdigest(),
            "seeds_sha256": hashlib.sha256(
                np.arange(610700, 610956, dtype="<i8").tobytes()
            ).hexdigest(),
            "every_replicate_index_exactly_once": True,
            "every_seed_exactly_once": True,
            "shared_warm_checkpoint_sha256": detector["warm_checkpoint_sha256"],
            "canonical_result_sha256": sha256_file(result_path),
            "canonical_result_independent_of_shard_layout": True,
        },
        "environment": environment_fingerprint(),
        "command": ["synthetic"],
        "artifacts": [_artifact(result_path), _artifact(threshold_path)],
        "resources": {
            "replicate_count": 256,
            "formal_eprocess_shot_updates": 256 * 5_000,
            "role_score_updates": 256 * 255_000,
            "wall_seconds": 1.0,
            "peak_rss_kib_linux_ru_maxrss": 1,
            "worker_process_count": 1,
            "external_concurrency_not_inferred": True,
            "output_bytes_excluding_manifest": (
                result_path.stat().st_size + threshold_path.stat().st_size
            ),
            "output_bytes_including_manifest": 0,
        },
    }
    manifest_path = tmp_path / "randomization_manifest.json"
    for _ in range(16):
        _write_json(manifest_path, manifest)
        expected_total = (
            manifest["resources"]["output_bytes_excluding_manifest"]
            + manifest_path.stat().st_size
        )
        if manifest["resources"]["output_bytes_including_manifest"] == expected_total:
            break
        manifest["resources"]["output_bytes_including_manifest"] = expected_total
    _write_json(manifest_path, manifest)
    _, observed = audit.verify_randomization_result(
        manifest_path,
        repo_root=ROOT,
        config_path=GOOGLE_LOCK,
        config=config,
        ratification_path=ratification,
        repair_ratification_path=repair_ratification,
        detector_manifest_path=detector_path,
        detector_manifest=detector,
    )
    assert observed["replicate_count"] == 256

    result_path.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash/size"):
        audit.verify_randomization_result(
            manifest_path,
            repo_root=ROOT,
            config_path=GOOGLE_LOCK,
            config=config,
            ratification_path=ratification,
            repair_ratification_path=repair_ratification,
            detector_manifest_path=detector_path,
            detector_manifest=detector,
        )


def test_detector_manifest_is_required_and_must_match_ratification(
    tmp_path: Path,
) -> None:
    config = load_google_lock(GOOGLE_LOCK)
    with pytest.raises(FileNotFoundError):
        audit.verify_detector_manifest(
            tmp_path / "absent.json",
            config_path=GOOGLE_LOCK,
            config=config,
            ratification_path=tmp_path / "freeze.json",
            repo_root=ROOT,
        )

    manifest_path, ratification_path = _detector_manifest_fixture(tmp_path)
    audit.verify_detector_manifest(
        manifest_path,
        config_path=GOOGLE_LOCK,
        config=config,
        ratification_path=ratification_path,
        repo_root=ROOT,
    )
    ratification_path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="freeze_ratification_sha256"):
        audit.verify_detector_manifest(
            manifest_path,
            config_path=GOOGLE_LOCK,
            config=config,
            ratification_path=ratification_path,
            repo_root=ROOT,
        )


def test_outcome_freeze_gate_uses_post_detector_repair_helper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = copy.deepcopy(load_google_lock(GOOGLE_LOCK))
    config["status"] = "frozen_before_held_value_access"
    config["normative_method_spec"]["sha256"] = sha256_file(
        ROOT / config["normative_method_spec"]["path"]
    )
    calls: list[dict[str, object]] = []

    def fake_verify(**kwargs: object) -> dict[str, object]:
        calls.append(dict(kwargs))
        return {"verified": True}

    monkeypatch.setattr(audit, "verify_post_detector_repair_chain", fake_verify)
    ratification = tmp_path / "ratification.json"
    repair_ratification = tmp_path / "repair_ratification.json"
    detector_manifest = tmp_path / "detector_manifest.json"
    observed = audit.verify_freeze_ratification(
        ratification,
        repair_ratification_path=repair_ratification,
        repo_root=ROOT,
        config_path=GOOGLE_LOCK,
        config=config,
        detector_manifest_path=detector_manifest,
    )
    assert observed == {"verified": True}
    assert calls[0]["original_ratification_path"] == ratification
    assert calls[0]["repair_ratification_path"] == repair_ratification
    assert calls[0]["detector_manifest_path"] == detector_manifest
    assert calls[0]["expected_environment"] == environment_fingerprint()
    assert (
        calls[0]["expected_thread_environment"]
        == config["numeric_policy"]["thread_environment"]
    )


def test_outcome_files_must_match_exact_verified_zip_members(
    tmp_path: Path,
) -> None:
    member_root = "synthetic_experiment"
    names = (
        "obs_flips_actual.01",
        "obs_flips_predicted_by_correlated_matching.01",
        "obs_flips_predicted_by_pymatching.01",
    )
    paths: dict[str, Path] = {}
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as target:
        for index, name in enumerate(names):
            payload = (f"{index % 2}\n" * 5).encode("ascii")
            target.writestr(f"{member_root}/{name}", payload)
            path = tmp_path / name
            path.write_bytes(payload)
            paths[name] = path
    hashes = audit.verify_outcome_zip_members(
        archive,
        paths,
        archive_member_root=member_root,
    )
    assert set(hashes) == set(names)

    paths[names[0]].write_text("tampered\n", encoding="ascii")
    with pytest.raises(ValueError, match="verified ZIP"):
        audit.verify_outcome_zip_members(
            archive,
            paths,
            archive_member_root=member_root,
        )


def test_little_endian_cycle_arrays_are_required(tmp_path: Path) -> None:
    artifacts: dict[str, Path] = {}
    for index, method in enumerate(audit.METHOD_IDS):
        path = tmp_path / f"held__{method}__empirical_cycle_score.npy"
        values = np.full((3, 2), index / 10.0, dtype="<f8")
        np.save(path, values, allow_pickle=False)
        artifacts[path.name] = path
    cycles, scores = audit.load_frozen_shot_scores(
        artifacts,
        shot_count=3,
        role_count=2,
    )
    assert cycles["m5"].dtype == np.float64
    np.testing.assert_allclose(scores["m5"], 0.6)


def test_threshold_table_is_recomputed_from_frozen_cycle_scores(
    tmp_path: Path,
) -> None:
    artifacts: dict[str, Path] = {}
    table: dict[str, object] = {}
    for index, method in enumerate(audit.METHOD_IDS):
        values = np.asarray(
            [
                [0.1 + index, 0.2 + index],
                [0.3 + index, 0.4 + index],
                [0.5 + index, 0.6 + index],
                [0.7 + index, 0.8 + index],
                [0.9 + index, 1.0 + index],
            ],
            dtype="<f8",
        )
        path = tmp_path / f"threshold__{method}__empirical_cycle_score.npy"
        np.save(path, values, allow_pickle=False)
        artifacts[path.name] = path
        selected = audit.select_strict_shot_threshold(values, 2)
        table[method] = {
            "threshold": selected.threshold,
            "validation_alert_count": selected.alert_count,
            "max_validation_alerts": 2,
            "secondary_zero_alert_threshold": float(np.max(selected.shot_scores)),
            "secondary_validation_alert_count": 0,
        }
    table_path = tmp_path / "thresholds.json"
    _write_json(table_path, table)
    artifacts[table_path.name] = table_path
    observed = audit.load_and_validate_thresholds(
        artifacts,
        shot_count=5,
        role_count=2,
    )
    assert set(observed) == set(audit.METHOD_IDS)


def test_extended_held_shot_table_is_recomputed_exactly(tmp_path: Path) -> None:
    cycles = {
        method: np.asarray(
            [[0.1, 0.7], [0.4, 0.2], [0.8, 0.8]],
            dtype=np.float64,
        )
        + 0.01 * index
        for index, method in enumerate(audit.METHOD_IDS)
    }
    thresholds = {method: {"threshold": 0.5} for method in audit.METHOD_IDS}
    windows = {
        "primary": [40_001, 40_002],
        "narrow": [40_002, 40_003],
        "wide": [40_000, 40_003],
    }
    path = tmp_path / "held_shots.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=audit.SHOT_TABLE_FIELDS)
        writer.writeheader()
        for method in audit.METHOD_IDS:
            applied = audit.apply_strict_shot_threshold(
                cycles[method],
                0.5,
            )
            archive = np.arange(40_000, 40_003)
            order = np.lexsort((archive, -applied.shot_scores))
            ranks = np.empty(3, dtype=np.int64)
            ranks[order] = np.arange(1, 4)
            cumulative = np.cumsum(applied.shot_alerts)
            for pair_index in range(3):
                monitor_shot = 40_000 + pair_index
                writer.writerow(
                    {
                        "phase": "held",
                        "method": method,
                        "pair_index": pair_index,
                        "reference_archive_shot": 20_000 + pair_index,
                        "monitor_archive_shot": monitor_shot,
                        "shot_score": format(
                            float(applied.shot_scores[pair_index]),
                            ".17g",
                        ),
                        "argmax_role": int(applied.shot_score_roles[pair_index]),
                        "first_crossing_role": int(
                            applied.first_crossing_roles[pair_index]
                        ),
                        "shot_alert": int(applied.shot_alerts[pair_index]),
                        "cumulative_alert_count": int(cumulative[pair_index]),
                        "rank": int(ranks[pair_index]),
                        "rank_tie_archive_shot": monitor_shot,
                        "in_primary_window": int(
                            windows["primary"][0]
                            <= monitor_shot
                            < windows["primary"][1]
                        ),
                        "in_narrow_window": int(
                            windows["narrow"][0] <= monitor_shot < windows["narrow"][1]
                        ),
                        "in_wide_window": int(
                            windows["wide"][0] <= monitor_shot < windows["wide"][1]
                        ),
                    }
                )
    audit.validate_frozen_shot_table(
        path,
        cycles,
        thresholds,
        windows=windows,
    )

    threshold_path = tmp_path / "threshold_shots.csv"
    with threshold_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=audit.SHOT_TABLE_FIELDS)
        writer.writeheader()
        for method in audit.METHOD_IDS:
            applied = audit.apply_strict_shot_threshold(cycles[method], 0.5)
            archive = np.arange(15_000, 15_003)
            order = np.lexsort((archive, -applied.shot_scores))
            ranks = np.empty(3, dtype=np.int64)
            ranks[order] = np.arange(1, 4)
            cumulative = np.cumsum(applied.shot_alerts)
            for local_index in range(3):
                monitor_shot = 15_000 + local_index
                writer.writerow(
                    {
                        "phase": "threshold",
                        "method": method,
                        "pair_index": 5_000 + local_index,
                        "reference_archive_shot": 5_000 + local_index,
                        "monitor_archive_shot": monitor_shot,
                        "shot_score": format(
                            float(applied.shot_scores[local_index]),
                            ".17g",
                        ),
                        "argmax_role": int(applied.shot_score_roles[local_index]),
                        "first_crossing_role": int(
                            applied.first_crossing_roles[local_index]
                        ),
                        "shot_alert": int(applied.shot_alerts[local_index]),
                        "cumulative_alert_count": int(cumulative[local_index]),
                        "rank": int(ranks[local_index]),
                        "rank_tie_archive_shot": monitor_shot,
                        "in_primary_window": 0,
                        "in_narrow_window": 0,
                        "in_wide_window": 0,
                    }
                )
    audit.validate_frozen_shot_table(
        threshold_path,
        cycles,
        thresholds,
        phase="threshold",
        pair_index_start=5_000,
        reference_start=5_000,
        monitor_start=15_000,
        windows=None,
    )


def test_synthetic_dry_run_opens_no_run6_values() -> None:
    result = audit.synthetic_dry_run()
    assert result["status"] == "synthetic_dry_run_passed"
    assert result["raw_run6_values_opened"] is False
    assert result["primary_label_equals_actual_xor_correlated"] is True
    assert result["secondary_label_equals_actual_xor_pymatching"] is True


def test_streaming_outcome_slice_does_not_decode_future_values(
    tmp_path: Path,
) -> None:
    path = tmp_path / "synthetic.01"
    path.write_bytes(b"x\n0\n1\n\xff\nnot-a-bit\n")
    observed = audit.parse_dot01_outcome_slice(
        path,
        expected_count=5,
        start=1,
        stop=3,
    )
    np.testing.assert_array_equal(observed, np.asarray([0, 1], dtype=np.uint8))


def test_streaming_outcome_slice_rejects_selected_nonbinary_and_wrong_count(
    tmp_path: Path,
) -> None:
    path = tmp_path / "synthetic.01"
    path.write_bytes(b"future\n0\nx\nfuture\n")
    with pytest.raises(ValueError, match="record 2 is not binary"):
        audit.parse_dot01_outcome_slice(
            path,
            expected_count=4,
            start=1,
            stop=3,
        )
    with pytest.raises(ValueError, match="Expected 5 outcome lines"):
        audit.parse_dot01_outcome_slice(
            path,
            expected_count=5,
            start=1,
            stop=2,
        )


def test_artifact_records_require_canonical_relative_paths(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"frozen")
    valid = {
        "path": artifact.name,
        "bytes": artifact.stat().st_size,
        "sha256": sha256_file(artifact),
    }
    assert audit._resolve_artifact(valid, manifest) == artifact.resolve()

    absolute = {**valid, "path": artifact.resolve().as_posix()}
    with pytest.raises(ValueError, match="canonical relative"):
        audit._resolve_artifact(absolute, manifest)
    traversal = {**valid, "path": "../artifact.bin"}
    with pytest.raises(ValueError, match="canonical relative"):
        audit._resolve_artifact(traversal, manifest)


def test_pnnl_retention_is_recomputed_from_exact_state_rows(tmp_path: Path) -> None:
    cohorts = tuple(f"cohort-{index}" for index in range(11))
    delays = {
        "dfr": 0.4,
        "online_logistic": 0.3,
        "space_sparse": 0.5,
        "space_spectral": 0.5,
        "space_composite": 0.2,
    }
    path = tmp_path / "state_rows.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=audit.PNNL_STATE_FIELDS)
        writer.writeheader()
        for cohort in cohorts:
            for state in (0, 1):
                for method in audit.PNNL_METHOD_IDS:
                    writer.writerow(
                        {
                            "cohort_id": cohort,
                            "logical_state": state,
                            "method": method,
                            "threshold_seed": 1,
                            "threshold_log_e": 0.0,
                            "first_alarm_update": "",
                            "pre_false_alarm": int(
                                method in {"dfr", "online_logistic"} and state == 0
                            ),
                            "miss": 0,
                            "post_alarm_shot": 1,
                            "post_alarm_role": "",
                            "restricted_post_delay_fraction": delays[method],
                        }
                    )
    observed = audit.recompute_pnnl_retention_from_state_rows(
        path,
        cohort_ids=cohorts,
    )
    assert observed["retention_pass"] is True
    assert observed["pre_false_alarm_state_count"]["space_composite"] == 0
    assert observed["pre_false_alarm_state_count"]["dfr"] == 11
    assert observed["comparisons"]["dfr"]["strictly_lower_macro_delay"] is True


def test_pnnl_portable_artifact_contract_is_exact_and_shape_checked(
    tmp_path: Path,
) -> None:
    cohort_ids = [f"cohort-{index}" for index in range(11)]
    pittsburgh = {
        "cohort_row_schema": ["cohort_id", "m"],
        "cohort_order": cohort_ids,
        "cohort_pairs": [[cohort_id, 2] for cohort_id in cohort_ids],
    }
    scalar_names = (
        "first_unblinding_record.json",
        "path_state_method_results.csv",
        "aggregate_results.json",
        "randomization_audit.json",
    )
    artifacts: dict[str, Path] = {}
    for name in scalar_names:
        path = tmp_path / name
        path.write_text("{}\n", encoding="utf-8")
        artifacts[name] = path
    counts_path = tmp_path / "randomization_alarm_counts.npy"
    maxima_path = tmp_path / "randomization_maximum_log_e.npy"
    np.save(counts_path, np.zeros((256, 5), dtype="<i8"), allow_pickle=False)
    np.save(maxima_path, np.zeros((256, 5), dtype="<f8"), allow_pickle=False)
    artifacts[counts_path.name] = counts_path
    artifacts[maxima_path.name] = maxima_path
    for cohort_index in range(11):
        for state in (0, 1):
            for method in audit.PNNL_METHOD_IDS:
                stem = f"{cohort_index:02d}_s{state}_{method}"
                trace = tmp_path / f"{stem}_log_e.npy"
                bootstrap = tmp_path / f"{stem}_bootstrap_maxima.npy"
                np.save(trace, np.zeros(3, dtype="<f8"), allow_pickle=False)
                np.save(
                    bootstrap,
                    np.zeros(4_096, dtype="<f8"),
                    allow_pickle=False,
                )
                artifacts[trace.name] = trace
                artifacts[bootstrap.name] = bootstrap
    manifest = {
        "first_unblinding_record": {"path": "first_unblinding_record.json"},
        "state_rows": {"path": "path_state_method_results.csv"},
        "aggregate_results": {"path": "aggregate_results.json"},
        "randomization_audit": {"path": "randomization_audit.json"},
        "randomization_alarm_counts": {"path": "randomization_alarm_counts.npy"},
        "randomization_maximum_log_e": {"path": "randomization_maximum_log_e.npy"},
    }
    audit.validate_pnnl_artifact_contract(
        manifest,
        artifacts,
        pittsburgh=pittsburgh,
    )

    unknown = tmp_path / "unknown.npy"
    np.save(unknown, np.zeros(1, dtype="<f8"), allow_pickle=False)
    artifacts[unknown.name] = unknown
    with pytest.raises(ValueError, match="artifact contract mismatch"):
        audit.validate_pnnl_artifact_contract(
            manifest,
            artifacts,
            pittsburgh=pittsburgh,
        )


def test_formal_detector_artifacts_enforce_nonformal_nan_scope(
    tmp_path: Path,
) -> None:
    shot_count = 2
    role_count = 1
    protocol_id = "run6-google2022-v2"
    thresholds = {method: {"threshold": 0.5} for method in audit.METHOD_IDS}
    artifacts: dict[str, Path] = {}
    traces: dict[str, dict[str, np.ndarray]] = {}
    for method in audit.METHOD_IDS:
        traces[method] = {}
        for array_id in (
            "log_eprocess",
            "log_sr",
            "first_e_crossing",
            "first_sr_crossing",
        ):
            if array_id.startswith("log_"):
                values = (
                    np.asarray([0.0, 0.1], dtype="<f8")
                    if method in audit.EXACT_METHOD_IDS
                    else np.asarray([np.nan, np.nan], dtype="<f8")
                )
            else:
                values = np.zeros(shot_count, dtype=np.bool_)
            data_path = tmp_path / f"held__{method}__{array_id}.npy"
            np.save(data_path, values, allow_pickle=False)
            sidecar_path = data_path.with_suffix(".json")
            _write_json(
                sidecar_path,
                {
                    "schema_version": "run6-cycle-array-v1",
                    "protocol_id": protocol_id,
                    "run_id": "google2022-canonical-detector",
                    "phase": "held",
                    "method_id": method,
                    "array_id": array_id,
                    "data_file": data_path.name,
                    "data_sha256": sha256_file(data_path),
                    "shape": [shot_count],
                    "dtype": values.dtype.str,
                    "flatten_order": ["paired_shot"],
                    "pair_index_range": [0, shot_count],
                    "reference_archive_start": 20_000,
                    "monitor_archive_start": 40_000,
                    "threshold": 0.5,
                    "formal_claim_scope": (
                        "not_applicable_no_formal_accumulator"
                        if method in audit.NONFORMAL_METHOD_IDS
                        else (
                            "diagnostic_only_on_natural_hardware; "
                            "no exchangeable hardware null asserted"
                        )
                    ),
                    "cooldown_semantics": "synthetic",
                    "checkpoint_and_code_hashes": {},
                },
            )
            artifacts[data_path.name] = data_path
            artifacts[sidecar_path.name] = sidecar_path
            traces[method][array_id] = values

    priors = audit.exact_component_priors()
    summary: dict[str, object] = {
        "schema_version": "run6-formal-component-summary-v1",
        "held_trace_interpretation": "synthetic",
        "proper_prior": {},
        "shiryaev_roberts": {},
        "expert_metadata": {},
    }
    for method in audit.EXACT_METHOD_IDS:
        base_weights = priors[method].weights
        raw_component_weights = np.tile(
            base_weights / role_count,
            role_count,
        )
        component_weights = (
            raw_component_weights / raw_component_weights.sum()
        ).tolist()
        base_count = len(base_weights)
        expert_count = role_count * base_count
        summary["proper_prior"][method] = {
            "component_weights": component_weights,
            "role_count": role_count,
            "base_component_count": base_count,
            "expert_flatten_order": ["role", "base_component"],
            "expert_id_rule": (
                "(role, *base_component_id), role-major then base-component-major"
            ),
            "final_log_components": [0.0] * expert_count,
            "final_log_statistic": 0.1,
            "first_crossing_update": None,
            "threshold": 100.0,
        }
        summary["shiryaev_roberts"][method] = {
            "component_weights": component_weights,
            "role_count": role_count,
            "base_component_count": base_count,
            "expert_flatten_order": ["role", "base_component"],
            "final_log_components": [0.0] * expert_count,
            "final_log_statistic": 0.1,
            "first_crossing_update": None,
            "threshold": 1_000_000.0,
        }
        summary["expert_metadata"][method] = {
            "expert_flatten_order": ["role", "base_component"],
            "role_prior": 1.0,
            "within_shot_factor_compounding": False,
            "base_component_ids": [
                list(identifier) for identifier in priors[method].component_ids
            ],
            "base_component_weights": base_weights.tolist(),
            "expert_count": expert_count,
            "observed_factor_minimum": 0.1,
            "observed_factor_maximum": 1.9,
            "all_factors_finite_and_nonnegative": True,
            "declared_factor_bounds": [0.1, 1.9],
            "factor_bounds_satisfied": True,
            "base_prior_sum": float(base_weights.sum()),
            "full_role_component_prior_sum": float(
                np.sum(component_weights),
            ),
        }
    summary_path = tmp_path / "formal_component_summary.json"
    _write_json(summary_path, summary)
    artifacts[summary_path.name] = summary_path
    audit.validate_formal_detector_artifacts(
        artifacts,
        thresholds,
        protocol_id=protocol_id,
        shot_count=shot_count,
        role_count=role_count,
    )

    nonformal = artifacts["held__m0c__log_eprocess.npy"]
    np.save(nonformal, np.zeros(shot_count, dtype="<f8"), allow_pickle=False)
    with pytest.raises(ValueError, match="array semantics"):
        audit.validate_formal_detector_artifacts(
            artifacts,
            thresholds,
            protocol_id=protocol_id,
            shot_count=shot_count,
            role_count=role_count,
        )

    np.save(
        nonformal,
        np.full(shot_count, np.nan, dtype="<f8"),
        allow_pickle=False,
    )
    summary["expert_metadata"]["space"]["factor_bounds_satisfied"] = False
    _write_json(summary_path, summary)
    with pytest.raises(ValueError, match="component/prior accounting"):
        audit.validate_formal_detector_artifacts(
            artifacts,
            thresholds,
            protocol_id=protocol_id,
            shot_count=shot_count,
            role_count=role_count,
        )
