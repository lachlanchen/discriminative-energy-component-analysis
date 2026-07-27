from __future__ import annotations

import csv
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import extract_results as target

DIGEST = "0" * 64
COMMIT = "1" * 40
COMMON_STDERR_DIGEST = target.FAILED_ATTEMPT_STDERR_SHA256
FIXTURE_EVIDENCE_PATHS = {
    "detector_manifest": "fixture/detector/detector_freeze_manifest.json",
    "freeze_ratification": "fixture/freeze/freeze_ratification.json",
    "repair_manifest": "fixture/repair/repair_manifest.json",
    "repair_ratification": "fixture/repair/repair_ratification.json",
    "randomization_manifest": "fixture/randomization/randomization_manifest.json",
    "pnnl_manifest": "fixture/pnnl/results_manifest.json",
    "pittsburgh_manifest": "fixture/pittsburgh/pnnl_pittsburgh_locked.json",
    "outcome_manifest": "fixture/outcome/outcome_manifest.json",
}


def write_json(path: Path, value: Any) -> None:
    path.write_bytes(target.canonical_json_bytes(value) + b"\n")


def record(path: Path, base: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(base).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": target.sha256_file(path),
    }


def write_pnnl_manifest_with_self_size(path: Path, manifest: dict[str, Any]) -> None:
    artifact_bytes = sum(
        artifact.stat().st_size
        for artifact in path.parent.iterdir()
        if artifact.is_file() and artifact != path
    )
    resources = manifest["resource_ledger"]
    resources["output_bytes_excluding_results_manifest"] = artifact_bytes
    resources["output_bytes_including_results_manifest"] = 0
    for _ in range(16):
        encoded = target.canonical_json_bytes(manifest) + b"\n"
        including = artifact_bytes + len(encoded)
        if resources["output_bytes_including_results_manifest"] == including:
            break
        resources["output_bytes_including_results_manifest"] = including
    else:
        raise AssertionError("fixture manifest self-size did not converge")
    path.write_bytes(encoded)


def make_event_summary() -> dict[str, Any]:
    return {
        method: {
            "pre_event_alert_count": 1 + int(index % 3 == 0),
            "pre_event_alert_shots": (
                [40_100 + 100 * index, 56_900 + index]
                if index % 3 == 0
                else [40_100 + 100 * index]
            ),
            "windows": {
                "primary": {
                    "detected": True,
                    "first_alert_shot": 57_775,
                    "first_alert_role": 7,
                },
                "narrow": {
                    "detected": True,
                    "first_alert_shot": 57_775,
                    "first_alert_role": 7,
                },
                "wide": {
                    "detected": True,
                    "first_alert_shot": 57_775,
                    "first_alert_role": 7,
                },
            },
        }
        for index, method in enumerate(target.METHOD_IDS)
    }


def make_original_ratification(root: Path) -> Path:
    root.mkdir()
    path = root / "freeze_ratification.json"
    write_json(
        path,
        {
            "schema_version": "run6-freeze-ratification-v1",
            "status": "frozen_before_held_value_access",
            "freeze_commit": target.ORIGINAL_FREEZE_COMMIT,
            "hashes": {
                "experiments/run6/freeze_manifest.json": DIGEST,
                "experiments/run6/configs/python_environment_lock.txt": (
                    target.REPAIR_PYTHON_LOCK_SHA256
                ),
            },
            "environment": {},
            "thread_environment": {
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
            },
            "held_value_access_before_ratification": False,
        },
    )
    return path


def make_detector(root: Path, freeze_ratification_path: Path) -> Path:
    root.mkdir()
    event = make_event_summary()
    thresholds = {
        method: {
            "threshold": 0.50 + 0.01 * index,
            "validation_alert_count": index % 3,
            "max_validation_alerts": 2,
            "secondary_zero_alert_threshold": 0.75 + 0.01 * index,
            "secondary_validation_alert_count": 0,
        }
        for index, method in enumerate(target.METHOD_IDS)
    }
    for name in target.expected_detector_artifact_names():
        path = root / name
        if name in {
            "event_summary_detector_only.json",
            "secondary_event_summary_detector_only.json",
        }:
            write_json(path, event)
        elif name == "thresholds.json":
            write_json(path, thresholds)
        elif name.endswith(".json"):
            write_json(path, {})
        else:
            path.write_bytes(b"derived fixture\n")
    artifacts = [
        record(root / name, root)
        for name in sorted(target.expected_detector_artifact_names())
    ]
    manifest = {
        "schema_version": "run6-google-detector-freeze-v1",
        "protocol_id": target.GOOGLE_PROTOCOL,
        "detector_only": True,
        "outcome_accessed": False,
        "outcome_join_authorized": False,
        "git_commit": COMMIT,
        "config_sha256": DIGEST,
        "method_spec_sha256": DIGEST,
        "detector_script_sha256": DIGEST,
        "freeze_ratification_sha256": target.sha256_file(freeze_ratification_path),
        "deviation_ledger": {
            "path": "experiments/run6/deviations.json",
            "sha256": DIGEST,
        },
        "circuit_sha256": DIGEST,
        "detector_layout_index_sha256": DIGEST,
        "warm_checkpoint_sha256": DIGEST,
        "threshold_checkpoint_sha256": DIGEST,
        "held_final_checkpoint_sha256": DIGEST,
        "source_archive_sha256": DIGEST,
        "source_archive_bytes": 1,
        "verified_zip_member_sha256": {"detection_events.b8": DIGEST},
        "detection_file_bytes": 1,
        "threshold_table_sha256": target.sha256_file(root / "thresholds.json"),
        "artifacts": artifacts,
        "resources": {
            "record_exposure": {
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
            },
            "high_level_operations": {},
            "warm_checkpoint_storage": {},
            "held_final_checkpoint_storage": {},
            "formal_accumulator": {
                "time_unit": "complete_paired_shot",
                "held_updates": 20_000,
                "role_prior": "uniform_1_over_51",
                "within_shot_factor_compounding": False,
            },
            "output_bytes_before_manifest": sum(row["bytes"] for row in artifacts),
        },
        "performance": {
            "canonical_joint_pipeline_only": True,
            "not_a_per_method_speed_comparison": True,
            "integrity_and_layout_seconds": 1.0,
            "validation_read_seconds": 1.0,
            "warm_fit_replay_seconds": 2.0,
            "threshold_replay_seconds": 3.0,
            "threshold_serialization_seconds": 1.0,
            "held_read_seconds": 2.0,
            "held_replay_seconds": 4.0,
            "held_joint_replay_all_three_seconds": [4.0, 4.5, 5.0],
            "held_joint_replay_median_seconds": 4.5,
            "held_joint_replay_digests": [DIGEST, DIGEST, DIGEST],
            "held_serialization_seconds": 1.0,
            "elapsed_before_manifest_seconds": 20.0,
            "peak_rss_kib_linux_ru_maxrss": 102_400,
            "relative_method_speed_claim_authorized": False,
        },
        "environment": {},
        "command": ["fixture"],
        "started_unix": 1.0,
        "finished_unix": 2.0,
    }
    path = root / "detector_freeze_manifest.json"
    write_json(path, manifest)
    return path


def make_repair_provenance(
    root: Path,
    *,
    freeze_ratification_path: Path,
    detector_path: Path,
) -> tuple[Path, Path]:
    root.mkdir()
    detector = target.load_json(detector_path)
    detector_registry = {
        f"experiments/run6/results/google_detector/{row['path']}": {
            "bytes": row["bytes"],
            "sha256": row["sha256"],
        }
        for row in detector["artifacts"]
    }
    repair_hashes = {
        path: fixture_digest(2_000 + index)
        for index, path in enumerate(target.REPAIR_DIFF_STATUS)
    }
    failure_files = {
        f"experiments/run6/results/google_randomization/.attempt_{index:03d}/stderr.log": {
            "bytes": 891,
            "sha256": COMMON_STDERR_DIGEST,
        }
        for index in range(32)
    }
    failure_files.update(
        {
            f"experiments/run6/results/google_randomization/.attempt_{index:03d}/stdout.log": {
                "bytes": 0,
                "sha256": target.hashlib.sha256(b"").hexdigest(),
            }
            for index in range(32)
        }
    )
    environment = target.load_json(freeze_ratification_path)["environment"]
    threads = {
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }
    manifest = {
        "schema_version": "run6-post-detector-repair-manifest-v1",
        "status": "post_detector_pre_outcome_repair_implementation_frozen",
        "incident_commit": target.INCIDENT_COMMIT,
        "original_ratification_commit": target.ORIGINAL_RATIFICATION_COMMIT,
        "implementation_commit": target.REPAIR_IMPLEMENTATION_COMMIT,
        "repair_diff": dict(target.REPAIR_DIFF_STATUS),
        "hashes": repair_hashes,
        "original_freeze": {
            "implementation_commit": target.ORIGINAL_IMPLEMENTATION_COMMIT,
            "freeze_commit": target.ORIGINAL_FREEZE_COMMIT,
            "freeze_manifest_sha256": DIGEST,
            "ratification_commit": target.ORIGINAL_RATIFICATION_COMMIT,
            "ratification_path": target.ORIGINAL_RATIFICATION_PATH,
            "ratification_sha256": target.sha256_file(freeze_ratification_path),
        },
        "detector_evidence": {
            "manifest_path": (
                "experiments/run6/results/google_detector/detector_freeze_manifest.json"
            ),
            "manifest_bytes": detector_path.stat().st_size,
            "manifest_sha256": target.sha256_file(detector_path),
            "detector_only": True,
            "outcome_accessed": False,
            "outcome_join_authorized": False,
            "held_joint_replay_digest_count": 3,
            "held_joint_replay_all_identical": True,
            "artifact_count": 231,
            "artifacts": detector_registry,
        },
        "failed_attempt_evidence": {
            "root": "experiments/run6/results/google_randomization",
            "attempt_count": 32,
            "attempt_shard_ranges": [[8 * i, 8 * (i + 1)] for i in range(32)],
            "file_count": 64,
            "files": failure_files,
            "common_stderr_sha256": COMMON_STDERR_DIGEST,
            "all_stderr_logs_byte_identical": True,
            "all_stdout_logs_empty": True,
            "shard_manifest_count": 0,
            "completed_randomization_replicates": 0,
            "empty_result_directory_count": 32,
            "empty_result_directories": [
                f"experiments/run6/results/google_randomization/.attempt_{i:03d}/result"
                for i in range(32)
            ],
        },
        "access_record": dict(target.REPAIR_ACCESS_RECORD),
        "environment": environment,
        "thread_environment": threads,
        "python_environment_lock_sha256": target.REPAIR_PYTHON_LOCK_SHA256,
        "runtime_module_origins": dict(target.REPAIR_RUNTIME_MODULE_ORIGINS),
    }
    manifest_path = root / "repair_manifest.json"
    write_json(manifest_path, manifest)
    ratification = {
        "schema_version": "run6-post-detector-repair-ratification-v1",
        "status": "post_detector_pre_outcome_repair_ratified",
        "repair_manifest_commit": target.REPAIR_MANIFEST_COMMIT,
        "hashes": {
            **repair_hashes,
            target.REPAIR_MANIFEST_PATH: target.sha256_file(manifest_path),
        },
        "original_ratification_sha256": target.sha256_file(freeze_ratification_path),
        "detector_manifest_sha256": target.sha256_file(detector_path),
        "access_record": dict(target.REPAIR_ACCESS_RECORD),
        "environment": environment,
        "thread_environment": threads,
        "python_environment_lock_sha256": target.REPAIR_PYTHON_LOCK_SHA256,
    }
    ratification_path = root / "repair_ratification.json"
    write_json(ratification_path, ratification)
    return manifest_path, ratification_path


def make_randomization(
    root: Path,
    detector_path: Path,
    repair_ratification_path: Path,
) -> Path:
    root.mkdir()
    replicates = []
    for index in range(256):
        crossings = {
            method: bool(method == "space" and index == 0)
            for method in target.FORMAL_METHOD_IDS
        }
        replicates.append(
            {
                "replicate_index": index,
                "seed": 610_700 + index,
                "swap_sha256": DIGEST,
                "swapped_shot_count": 2_500,
                "checkpoint_restored": True,
                "crossed_100": crossings,
                "first_crossing_shot_number_one_based": {
                    method: (100 if crossed else None)
                    for method, crossed in crossings.items()
                },
                "maximum_log_e": {
                    method: (5.0 if crossed else 0.0)
                    for method, crossed in crossings.items()
                },
                "final_log_e": {
                    method: (4.8 if crossed else -0.1)
                    for method, crossed in crossings.items()
                },
                "familywide_any_crossed_600": False,
                "formal_eprocess_updates": 5_000,
                "role_score_updates": 255_000,
                "formal_experts": {method: 408 for method in target.FORMAL_METHOD_IDS},
            }
        )
    result = {
        "schema_version": "run6-google-randomization-result-v1",
        "primary_method": "space",
        "primary_statistic": "ever_proper_prior_eprocess_ge_100",
        "replicate_count": 256,
        "seed_start": 610_700,
        "seed_stop_exclusive": 610_956,
        "rng": "numpy.random.Generator(PCG64)",
        "one_orientation_draw_per_replicate": True,
        "complete_shot_swap_shared_across_roles": True,
        "horizon_paired_shots": 5_000,
        "role_score_updates_per_replicate": 255_000,
        "formal_expert_index": "role_then_locked_base_component",
        "formal_role_prior": "uniform_1_over_role_count",
        "warm_checkpoint_sha256": DIGEST,
        "crossing_counts_at_100": {
            method: (1 if method == "space" else 0)
            for method in target.FORMAL_METHOD_IDS
        },
        "space_crossing_fraction": 1 / 256,
        "space_crossing_clopper_pearson_95": {"lower": 0.0, "upper": 0.02},
        "familywide_any_crossing_count_at_600": 0,
        "interpretation": "exact_design_based_implementation_diagnostic",
        "replicates": replicates,
    }
    result_path = root / "randomization_result.json"
    threshold_path = root / "threshold_bootstrap.json"
    write_json(result_path, result)
    detector_thresholds = target.load_json(detector_path.parent / "thresholds.json")
    bootstrap_vectors: dict[str, dict[str, list[float | int]]] = {
        method: {
            "selected_threshold": [],
            "selected_count": [],
            "frozen_count": [],
            "zero_threshold": [],
        }
        for method in target.METHOD_IDS
    }
    bootstrap_rows = []
    for index in range(target.GOOGLE_THRESHOLD_BOOTSTRAP_REPLICATES):
        methods = {}
        for method_index, method in enumerate(target.METHOD_IDS):
            selected = 0.40 + 0.01 * method_index + 0.000001 * index
            selected_count = (index + method_index) % 3
            frozen_count = (2 * index + method_index) % 7
            selected_zero = 0.90 + 0.01 * method_index + 0.000001 * index
            methods[method] = {
                "selected_primary_threshold": selected,
                "selected_primary_alert_count": selected_count,
                "alert_count_at_frozen_threshold": frozen_count,
                "selected_zero_alert_threshold": selected_zero,
                "selected_zero_alert_count": 0,
            }
            bootstrap_vectors[method]["selected_threshold"].append(selected)
            bootstrap_vectors[method]["selected_count"].append(selected_count)
            bootstrap_vectors[method]["frozen_count"].append(frozen_count)
            bootstrap_vectors[method]["zero_threshold"].append(selected_zero)
        bootstrap_rows.append(
            {
                "replicate_index": index,
                "seed": 613_000 + index,
                "methods": methods,
            }
        )
    bootstrap_summaries = {}
    for method in target.METHOD_IDS:
        vectors = bootstrap_vectors[method]
        bootstrap_summaries[method] = {
            "frozen_threshold": detector_thresholds[method]["threshold"],
            "selected_primary_threshold_percentiles": target.percentile_summary(
                vectors["selected_threshold"]
            ),
            "selected_primary_alert_count_frequency": target.integer_histogram(
                np.asarray(vectors["selected_count"], dtype=np.int64)
            ),
            "alert_count_at_frozen_threshold_percentiles": (
                target.percentile_summary(vectors["frozen_count"])
            ),
            "alert_count_at_frozen_threshold_frequency": target.integer_histogram(
                np.asarray(vectors["frozen_count"], dtype=np.int64)
            ),
            "selected_zero_alert_threshold_percentiles": target.percentile_summary(
                vectors["zero_threshold"]
            ),
        }
    threshold_bootstrap = {
        "schema_version": "run6-google-threshold-bootstrap-v1",
        "status": "descriptive_only_does_not_replace_frozen_threshold",
        "unit": "complete_paired_shot",
        "block_length_shots": 128,
        "replicates": target.GOOGLE_THRESHOLD_BOOTSTRAP_REPLICATES,
        "seed_start": 613_000,
        "seed_stop_exclusive": 615_000,
        "rng": "numpy.random.Generator(PCG64)",
        "blocks_per_replicate": 40,
        "primary_maximum_alerts": 2,
        "secondary_maximum_alerts": 0,
        "summaries": bootstrap_summaries,
        "replicate_results": bootstrap_rows,
    }
    write_json(threshold_path, threshold_bootstrap)
    detector = target.load_json(detector_path)
    manifest = {
        "schema_version": "run6-google-randomization-manifest-v1",
        "protocol_id": target.GOOGLE_PROTOCOL,
        "config_sha256": DIGEST,
        "method_spec_sha256": DIGEST,
        "freeze_ratification_sha256": detector["freeze_ratification_sha256"],
        "repair_ratification_path": target.REPAIR_RATIFICATION_PATH,
        "repair_ratification_sha256": target.sha256_file(repair_ratification_path),
        "detector_manifest_sha256": target.sha256_file(detector_path),
        "detector_manifest_git_commit": detector["git_commit"],
        "script_sha256": DIGEST,
        "git_commit": COMMIT,
        "outcome_accessed": False,
        "source_archive_sha256": DIGEST,
        "verified_zip_member_sha256": detector["verified_zip_member_sha256"],
        "warm_checkpoint_sha256": DIGEST,
        "rng": {
            "algorithm": "numpy.random.Generator(PCG64)",
            "randomization_seed_start": 610_700,
            "randomization_seed_stop_exclusive": 610_956,
            "threshold_bootstrap_seed_start": 613_000,
            "threshold_bootstrap_seed_stop_exclusive": 615_000,
        },
        "execution_mode": "deterministic_gap_free_shard_merge",
        "merge_evidence": {},
        "environment": {},
        "command": ["fixture"],
        "artifacts": [record(result_path, root), record(threshold_path, root)],
        "resources": {},
    }
    path = root / "randomization_manifest.json"
    write_json(path, manifest)
    return path


def fixture_digest(index: int) -> str:
    return f"{index:064x}"


def make_pittsburgh(root: Path) -> Path:
    root.mkdir()
    snapshots: dict[str, Any] = {}
    for index in range(20):
        snapshot_id = f"snapshot_{index:02d}"
        snapshots[snapshot_id] = {
            "relative_job_dir": f"d3_r5/job_{index:02d}",
            "metadata": [
                3,
                5,
                "Z",
                2_048,
                10,
                f"2026-01-{index + 1:02d} 00:00:00+00:00",
                f"2026-01-{index + 1:02d}T00:00:00Z",
            ],
            "info": [100, fixture_digest(100 + index), fixture_digest(200 + index)],
            "calibration": [
                100,
                fixture_digest(300 + index),
                fixture_digest(400 + index),
            ],
            "qasm_state0": [
                100,
                fixture_digest(500 + index),
                fixture_digest(600 + index),
            ],
            "qasm_state1": [
                100,
                fixture_digest(700 + index),
                fixture_digest(800 + index),
            ],
            "qasm_pair": [
                fixture_digest(900 + index),
                fixture_digest(1_000 + index),
            ],
            "held_bitstrings": [1_000, False, None],
        }
    cohorts = [f"cohort_{index:02d}" for index in range(11)]
    pairs = []
    for index, cohort in enumerate(cohorts):
        early_id = f"snapshot_{index:02d}"
        late_id = f"snapshot_{index + 9:02d}"
        early_hash = snapshots[early_id]["calibration"][2]
        late_hash = snapshots[late_id]["calibration"][2]
        pairs.append(
            [
                cohort,
                3,
                5,
                "Z",
                "0_1_2_3_4",
                [0, 2, 4],
                [1, 3],
                [0, 1, 2, 3, 4],
                early_id,
                late_id,
                682,
                False,
                False,
                "circuit_and_hardware_domain_shift",
                f"{early_hash}--{late_hash}",
            ]
        )
    manifest = {
        "manifest_id": "run6-pnnl-pittsburgh-metadata-lock-v2",
        "status": "frozen_before_held_value_access",
        "source": {
            "backend": "ibm_pittsburgh",
            "logical_states": [0, 1],
        },
        "artifact_tuple_schemas": {
            key: list(value)
            for key, value in target.PNNL_ARTIFACT_TUPLE_SCHEMAS.items()
        },
        "snapshots": snapshots,
        "cohort_row_schema": list(target.PNNL_COHORT_FIELDS),
        "cohort_order": cohorts,
        "cohort_pairs": pairs,
    }
    path = root / "pnnl_pittsburgh_locked.json"
    write_json(path, manifest)
    return path


def make_pnnl(
    root: Path,
    *,
    retention: bool,
    pittsburgh_path: Path,
    freeze_ratification_path: Path,
    repair_ratification_path: Path,
) -> Path:
    root.mkdir()
    pittsburgh = target.validate_pittsburgh_manifest(pittsburgh_path)
    cohort_metadata = pittsburgh["cohorts"]
    cohorts = [row["cohort_id"] for row in cohort_metadata]
    if retention:
        delays = {
            "dfr": 0.40,
            "online_logistic": 0.30,
            "space_sparse": 0.25,
            "space_spectral": 0.22,
            "space_composite": 0.20,
        }
    else:
        delays = {
            "dfr": 0.40,
            "online_logistic": 0.30,
            "space_sparse": 0.45,
            "space_spectral": 0.48,
            "space_composite": 0.50,
        }
    state_path = root / "path_state_method_results.csv"
    with state_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=target.PNNL_STATE_FIELDS)
        writer.writeheader()
        for cohort in cohorts:
            for state in (0, 1):
                for method_index, method in enumerate(target.PNNL_METHOD_IDS):
                    writer.writerow(
                        {
                            "cohort_id": cohort,
                            "logical_state": state,
                            "method": method,
                            "threshold_seed": 612_000 + method_index,
                            "threshold_log_e": 4.0,
                            "first_alarm_update": 10,
                            "pre_false_alarm": 0,
                            "miss": 0,
                            "post_alarm_shot": 1,
                            "post_alarm_role": "",
                            "restricted_post_delay_fraction": delays[method],
                        }
                    )
    cohort_rows = [
        {
            "cohort_id": cohort["cohort_id"],
            "basis": cohort["basis"],
            "distance": cohort["distance"],
            "rounds": cohort["rounds"],
            "calibration_pair_id": cohort["calibration_pair_id"],
            "method": method,
            "pre_false_alarm_mean": 0.0,
            "restricted_post_delay_fraction": delays[method],
            "miss_mean": 0.0,
        }
        for cohort in cohort_metadata
        for method in target.PNNL_METHOD_IDS
    ]
    macro = {
        method: {
            "pre_false_alarm_state_count": 0,
            "miss_state_count": 0,
            "macro_restricted_post_delay_fraction": delay,
        }
        for method, delay in delays.items()
    }
    comparisons: dict[str, Any] = {}
    for comparator in ("dfr", "online_logistic"):
        effect = delays["space_composite"] - delays[comparator]
        passed = effect < 0
        comparisons[comparator] = {
            "cohort_delay_differences": [effect] * 11,
            "macro_delay_difference": effect,
            "primary_95_percentile_interval": [effect - 0.01, effect + 0.01],
            "calibration_pair_95_percentile_sensitivity": [
                effect - 0.02,
                effect + 0.02,
            ],
            "exact_sign_flip_two_sided_p": 0.01,
            "no_worse_pre_false_alarm": True,
            "strictly_lower_macro_delay": passed,
            "retention_condition_pass": passed,
        }
    aggregate = {
        "cohort_rows": cohort_rows,
        "macro_by_method": macro,
        "comparisons": comparisons,
        "retention_pass": retention,
    }
    aggregate_path = root / "aggregate_results.json"
    write_json(aggregate_path, aggregate)
    first_path = root / "first_unblinding_record.json"
    random_path = root / "randomization_audit.json"
    counts_path = root / "randomization_alarm_counts.npy"
    maxima_path = root / "randomization_maximum_log_e.npy"
    payload_hashes = {snapshot_id: DIGEST for snapshot_id in pittsburgh["snapshot_ids"]}
    pittsburgh_sha256 = target.sha256_file(pittsburgh_path)
    write_json(
        first_path,
        {
            "schema_version": "run6-pnnl-first-unblinding-v1",
            "utc": "2026-01-01T00:00:00+00:00",
            "git_commit": COMMIT,
            "config_sha256": DIGEST,
            "manifest_sha256": pittsburgh_sha256,
            "freeze_ratification_sha256": target.sha256_file(freeze_ratification_path),
            "repair_ratification_path": target.REPAIR_RATIFICATION_PATH,
            "repair_ratification_sha256": target.sha256_file(repair_ratification_path),
            "package_lock": {
                "path": "experiments/run6/configs/python_environment_lock.txt",
                "bytes": 1,
                "sha256": DIGEST,
            },
            "package_environment": {},
            "held_payloads": [
                {
                    "snapshot_id": snapshot_id,
                    "path": f"metadata-only/{snapshot_id}/payload.json",
                    "bytes": 1,
                    "sha256": digest,
                }
                for snapshot_id, digest in payload_hashes.items()
            ],
            "scores_computed_before_record": False,
        },
    )
    alarm_counts = np.zeros(
        (
            target.PNNL_RANDOMIZATION_REPLICATES,
            len(target.PNNL_METHOD_IDS),
        ),
        dtype="<i8",
    )
    maximum_log_e = np.empty_like(alarm_counts, dtype="<f8")
    for method_index, _method in enumerate(target.PNNL_METHOD_IDS):
        alarm_counts[:, method_index] = method_index
        if method_index == 0:
            maximum_log_e[:, method_index] = (
                1.0 + np.arange(target.PNNL_RANDOMIZATION_REPLICATES) / 10_000
            )
        else:
            maximum_log_e[:, method_index] = (
                5.0
                + method_index
                + np.arange(target.PNNL_RANDOMIZATION_REPLICATES) / 1_000
            )
    np.save(counts_path, alarm_counts, allow_pickle=False)
    np.save(maxima_path, maximum_log_e, allow_pickle=False)
    random_rows = []
    for cohort_index, cohort in enumerate(cohort_metadata):
        for logical_state in (0, 1):
            episode_index = 2 * cohort_index + logical_state
            for method_index, method in enumerate(target.PNNL_METHOD_IDS):
                alarm_fraction = 1.0 if episode_index < method_index else 0.0
                global_maximum = float(np.max(maximum_log_e[:, method_index]))
                row_maximum = (
                    global_maximum
                    if episode_index < method_index
                    or (method_index == 0 and episode_index == 0)
                    else 1.0 + 0.01 * method_index + 0.001 * episode_index
                )
                random_rows.append(
                    {
                        "cohort_index": cohort_index,
                        "cohort_id": cohort["cohort_id"],
                        "logical_state": logical_state,
                        "method": method,
                        "alarm_fraction": alarm_fraction,
                        "maximum_log_e_over_replicates": row_maximum,
                    }
                )
    random_summary = {
        "schema_version": "run6-pnnl-randomization-audit-v1",
        "seeds": list(
            range(
                610_700,
                610_700 + target.PNNL_RANDOMIZATION_REPLICATES,
            )
        ),
        "method_order": list(target.PNNL_METHOD_IDS),
        "path_state_method_rows": random_rows,
        "overall_episode_alarm_fraction": {
            method: float(
                np.sum(alarm_counts[:, method_index])
                / (
                    target.PNNL_RANDOMIZATION_REPLICATES
                    * target.PNNL_PATH_STATE_EPISODES
                )
            )
            for method_index, method in enumerate(target.PNNL_METHOD_IDS)
        },
        "alarmed_episode_count_histogram": {
            method: target.integer_histogram(alarm_counts[:, method_index])
            for method_index, method in enumerate(target.PNNL_METHOD_IDS)
        },
        "maximum_log_e_summary": {
            method: {
                "minimum": float(np.min(maximum_log_e[:, method_index])),
                "median": float(np.median(maximum_log_e[:, method_index])),
                "maximum": float(np.max(maximum_log_e[:, method_index])),
            }
            for method_index, method in enumerate(target.PNNL_METHOD_IDS)
        },
        "claim_scope": (
            "implementation and exact randomized paired design only; "
            "not a natural hardware null"
        ),
    }
    write_json(random_path, random_summary)
    trace_records = []
    bootstrap_records = []
    for index in range(110):
        trace = root / f"fixture_{index:03d}_log_e.npy"
        bootstrap = root / f"fixture_{index:03d}_bootstrap_maxima.npy"
        trace.write_bytes(b"derived trace\n")
        bootstrap.write_bytes(b"derived bootstrap\n")
        trace_records.append(record(trace, root))
        bootstrap_records.append(record(bootstrap, root))
    manifest = {
        "schema_version": "run6-pnnl-snapshot-results-v1",
        "protocol_id": target.PNNL_PROTOCOL,
        "claim_label": "constructed circuit-and-hardware domain shift; not temporal drift",
        "formal_alarm_unit": "one update per complete paired shot",
        "within_shot_roles": "fixed experts under a uniform role prior",
        "git_commit": COMMIT,
        "config_sha256": DIGEST,
        "pittsburgh_manifest_sha256": pittsburgh_sha256,
        "freeze_ratification_sha256": target.sha256_file(freeze_ratification_path),
        "repair_ratification_path": target.REPAIR_RATIFICATION_PATH,
        "repair_ratification_sha256": target.sha256_file(repair_ratification_path),
        "package_lock_sha256": DIGEST,
        "first_unblinding_record": record(first_path, root),
        "metadata_validation": {
            "snapshots": 20,
            "cohorts": 11,
            "held_payloads_statted": 20,
        },
        "held_payload_sha256": payload_hashes,
        "state_rows": record(state_path, root),
        "aggregate_results": record(aggregate_path, root),
        "randomization_audit": record(random_path, root),
        "randomization_alarm_counts": record(counts_path, root),
        "randomization_maximum_log_e": record(maxima_path, root),
        "trace_artifacts": trace_records,
        "bootstrap_artifacts": bootstrap_records,
        "resource_ledger": {
            **target.pnnl_expected_resource_counts(cohort_metadata),
            "adaptive_state_ledger": [
                {
                    "cohort_id": cohort["cohort_id"],
                    "logical_state": logical_state,
                    "q": cohort["distance"] - 1,
                    "roles": cohort["rounds"],
                    "adaptive_bank_numeric_bytes": (
                        target.pnnl_adaptive_bank_numeric_bytes(
                            cohort["distance"] - 1,
                            cohort["rounds"],
                        )
                    ),
                    "formal_accumulator_components": (
                        target.pnnl_formal_component_counts(
                            cohort["distance"] - 1,
                            cohort["rounds"],
                        )
                    ),
                    "formal_accumulator_numeric_bytes": (
                        3
                        * 8
                        * sum(
                            target.pnnl_formal_component_counts(
                                cohort["distance"] - 1,
                                cohort["rounds"],
                            ).values()
                        )
                    ),
                }
                for cohort in cohort_metadata
                for logical_state in (0, 1)
            ],
            "wall_seconds": 30.0,
            "held_value_processing_wall_seconds": 25.0,
            "peak_rss_kib": 204_800,
            "output_bytes_excluding_results_manifest": 0,
            "output_bytes_including_results_manifest": 0,
        },
        "retention_pass": retention,
        "environment": {},
        "command": ["fixture"],
        "started_unix": 1.0,
        "held_value_processing_started_unix": 6.0,
        "finished_unix": 31.0,
    }
    path = root / "results_manifest.json"
    write_pnnl_manifest_with_self_size(path, manifest)
    return path


def make_parity_evidence() -> dict[str, Any]:
    return {
        "schema_version": "run6-google-method-input-parity-evidence-v1",
        "expected_method_ids": list(target.METHOD_IDS),
        "observed_method_ids": list(target.METHOD_IDS),
        "expected_held_score_shape": [20_000, 51],
        "held_detector_score_inputs": {
            method: {
                "shape": [20_000, 51],
                "record_count": 1_020_000,
                "matches_locked_shape_and_count": True,
            }
            for method in target.METHOD_IDS
        },
        "all_methods_have_locked_detector_record_shape_and_count": True,
        "shared_outcome_label_bundle": {
            "serialization": "fixture",
            "sha256": DIGEST,
            "label_ids": list(target.LABEL_IDS),
            "label_record_counts": {label: 20_000 for label in target.LABEL_IDS},
            "archive_shot_count": 20_000,
            "consumer_method_ids": list(target.METHOD_IDS),
            "single_shared_bundle_for_all_methods": True,
        },
        "no_method_received_extra_detector_records_or_outcome_labels": True,
    }


def risk_budget_row(budget: int, captured: int, total: int = 100) -> dict[str, Any]:
    return {
        "alert_budget_shots": budget,
        "alert_fraction": budget / 20_000,
        "captured_mismatches": captured,
        "total_mismatches": total,
        "mismatch_recall": captured / total,
        "alert_precision": captured / budget,
        "retained_mismatch_rate": (total - captured) / (20_000 - budget),
        "coverage": 1 - budget / 20_000,
        "selected_archive_shots": list(range(40_000, 40_000 + budget)),
    }


def make_risk(detector_path: Path) -> dict[str, Any]:
    top20 = {
        "m0": 2,
        "m0c": 2,
        "m1": 2,
        "m2": 2,
        "m3": 3,
        "m4": 4,
        "m5": 4,
        "space": 5,
    }
    point_estimates = {}
    for label in target.LABEL_IDS:
        point_estimates[label] = {}
        for method in target.METHOD_IDS:
            budget_rows = {
                "2": risk_budget_row(2, 1),
                "20": risk_budget_row(20, top20[method]),
                "200": risk_budget_row(200, 10),
            }
            point_estimates[label][method] = {
                "budgets": budget_rows,
                "partial_trapezoidal_recall_area": float(
                    np.trapezoid(
                        [
                            budget_rows[str(budget)]["mismatch_recall"]
                            for budget in target.RISK_BUDGETS
                        ],
                        [budget / 20_000 for budget in target.RISK_BUDGETS],
                    )
                ),
            }
    interval = {"lower": 0.0, "upper": 1.0, "valid_replicates": 2_000}
    method_intervals = {
        label: {
            method: {
                str(budget): {
                    metric: dict(interval)
                    for metric in (
                        "captured_mismatches",
                        "mismatch_recall",
                        "alert_precision",
                        "retained_mismatch_rate",
                    )
                }
                for budget in target.RISK_BUDGETS
            }
            for method in target.METHOD_IDS
        }
        for label in target.LABEL_IDS
    }
    difference_intervals = {
        label: {
            comparator: {
                str(budget): {
                    metric: dict(interval)
                    for metric in (
                        "captured_mismatches",
                        "mismatch_recall",
                        "alert_precision",
                        "retained_mismatch_rate",
                    )
                }
                for budget in target.RISK_BUDGETS
            }
            for comparator in ("space_minus_m0", "space_minus_m3")
        }
        for label in target.LABEL_IDS
    }
    return {
        "schema_version": "run6-google-risk-summary-v1",
        "primary_label": "actual_xor_correlated_matching_prediction",
        "secondary_label": "actual_xor_pymatching_prediction",
        "ranking": "descending_frozen_shot_score_then_ascending_archive_shot",
        "budgets_shots": list(target.RISK_BUDGETS),
        "outcome_table_sha256": "",
        "detector_manifest_sha256": target.sha256_file(detector_path),
        "point_estimates": point_estimates,
        "uncertainty": {
            "kind": "paired_circular_moving_complete_shot_blocks",
            "block_length_shots": 128,
            "replicates": 2_000,
            "seed_start": 611_000,
            "seed_stop_exclusive": 613_000,
            "rng": "numpy.random.Generator(PCG64)",
            "blocks_per_replicate": 157,
            "percentile_interval": [2.5, 97.5],
            "percentile_method": "linear",
            "method_intervals": method_intervals,
            "space_comparator_difference_intervals": difference_intervals,
        },
        "interpretation": "retrospective_veto_or_triage_not_a_decoder",
    }


def make_outcome(
    root: Path,
    *,
    detector_path: Path,
    randomization_path: Path,
    pnnl_path: Path,
    repair_ratification_path: Path,
    retention: bool,
) -> Path:
    root.mkdir()
    outcome_table = root / "outcomes.csv"
    outcome_table.write_text("derived outcome fixture\n", encoding="utf-8")
    risk = make_risk(detector_path)
    risk["outcome_table_sha256"] = target.sha256_file(outcome_table)
    risk_path = root / "risk_summary.json"
    write_json(risk_path, risk)
    evidence = make_parity_evidence()
    capture = {
        method: risk["point_estimates"]["correlated_matching_mismatch"][method][
            "budgets"
        ]["20"]["captured_mismatches"]
        for method in target.METHOD_IDS
    }
    randomization_result = target.load_json(
        randomization_path.parent / "randomization_result.json"
    )
    decision = {
        "schema_version": "run6-google-decision-v1",
        "repair_ratification_path": target.REPAIR_RATIFICATION_PATH,
        "repair_ratification_sha256": target.sha256_file(repair_ratification_path),
        "summary_scope": "full_run6_locked_decision",
        "primary_label": "actual_xor_correlated_matching_prediction",
        "primary_budget_shots": 20,
        "top20_capture": capture,
        "atomic_predicates": {key: True for key in target.ATOMIC_PREDICATES},
        "method_input_parity_evidence": evidence,
        "mandatory_contextual_controls_reported": True,
        "google_primary_pass": True,
        "randomization_audit": {
            "status": "completed_and_hash_verified",
            "manifest_sha256": target.sha256_file(randomization_path),
            "space_crossing_count_at_100": 1,
            "replicates": 256,
            "clopper_pearson_95": randomization_result[
                "space_crossing_clopper_pearson_95"
            ],
            "changes_primary_boolean": False,
        },
        "pnnl_retention_pass": retention,
        "pnnl_results_manifest_sha256": target.sha256_file(pnnl_path),
        "overall_run6_advantage": retention,
        "negative_result_reasons": [] if retention else ["pnnl_retention_failed"],
        "bootstrap_changes_primary_boolean": False,
    }
    decision_path = root / "decision_summary.json"
    write_json(decision_path, decision)
    detector = target.load_json(detector_path)
    manifest = {
        "schema_version": "run6-google-outcome-manifest-v1",
        "protocol_id": target.GOOGLE_PROTOCOL,
        "config_sha256": DIGEST,
        "method_spec_sha256": DIGEST,
        "freeze_ratification_sha256": detector["freeze_ratification_sha256"],
        "repair_ratification_path": target.REPAIR_RATIFICATION_PATH,
        "repair_ratification_sha256": target.sha256_file(repair_ratification_path),
        "detector_manifest_sha256": target.sha256_file(detector_path),
        "detector_manifest_git_commit": detector["git_commit"],
        "script_sha256": DIGEST,
        "git_commit": COMMIT,
        "outcome_accessed_after_detector_freeze": True,
        "outcome_source_hashes": {},
        "verified_outcome_zip_member_sha256": {},
        "primary_label": "actual_xor_correlated_matching_prediction",
        "secondary_label": "actual_xor_pymatching_prediction",
        "method_input_parity_evidence_sha256": target.hashlib.sha256(
            target.canonical_json_bytes(evidence)
        ).hexdigest(),
        "shared_outcome_label_bundle_sha256": DIGEST,
        "final_aggregation_inputs": {
            "status": "completed_and_hash_verified",
            "randomization_manifest_sha256": target.sha256_file(randomization_path),
            "pnnl_results_manifest_sha256": target.sha256_file(pnnl_path),
        },
        "rng": {},
        "environment": {},
        "command": ["fixture"],
        "artifacts": [
            record(outcome_table, root),
            record(risk_path, root),
            record(decision_path, root),
        ],
    }
    path = root / "outcome_manifest.json"
    write_json(path, manifest)
    return path


def make_fixture(root: Path, *, retention: bool = True) -> dict[str, Any]:
    root.mkdir(parents=True)
    freeze_ratification = make_original_ratification(root / "freeze")
    detector = make_detector(root / "detector", freeze_ratification)
    repair_manifest, repair_ratification = make_repair_provenance(
        root / "repair",
        freeze_ratification_path=freeze_ratification,
        detector_path=detector,
    )
    randomization = make_randomization(
        root / "randomization",
        detector,
        repair_ratification,
    )
    pittsburgh = make_pittsburgh(root / "pittsburgh")
    pnnl = make_pnnl(
        root / "pnnl",
        retention=retention,
        pittsburgh_path=pittsburgh,
        freeze_ratification_path=freeze_ratification,
        repair_ratification_path=repair_ratification,
    )
    outcome = make_outcome(
        root / "outcome",
        detector_path=detector,
        randomization_path=randomization,
        pnnl_path=pnnl,
        repair_ratification_path=repair_ratification,
        retention=retention,
    )
    paths: dict[str, Any] = {
        "detector": detector,
        "freeze_ratification": freeze_ratification,
        "repair_manifest": repair_manifest,
        "repair_ratification": repair_ratification,
        "randomization": randomization,
        "pittsburgh": pittsburgh,
        "pnnl": pnnl,
        "outcome": outcome,
    }
    repair = target.load_json(repair_manifest)
    paths["_validation_profile"] = target._make_internal_validation_profile(
        original_ratification_bytes=freeze_ratification.read_bytes(),
        repair_manifest_bytes=repair_manifest.read_bytes(),
        repair_ratification_bytes=repair_ratification.read_bytes(),
        implementation_hashes=repair["hashes"],
        evidence_paths=FIXTURE_EVIDENCE_PATHS,
    )
    return paths


def run_fixture(paths: dict[str, Any], output: Path) -> None:
    target._run_with_validation_profile(
        [
            "--detector-manifest",
            str(paths["detector"]),
            "--freeze-ratification",
            str(paths["freeze_ratification"]),
            "--repair-manifest",
            str(paths["repair_manifest"]),
            "--repair-ratification",
            str(paths["repair_ratification"]),
            "--randomization-manifest",
            str(paths["randomization"]),
            "--pnnl-manifest",
            str(paths["pnnl"]),
            "--pittsburgh-manifest",
            str(paths["pittsburgh"]),
            "--outcome-manifest",
            str(paths["outcome"]),
            "--output-dir",
            str(output),
        ],
        validation_profile=paths["_validation_profile"],
    )


def refresh_randomization_artifact(
    paths: dict[str, Any],
    artifact_name: str,
) -> None:
    manifest = target.load_json(paths["randomization"])
    artifact_path = paths["randomization"].parent / artifact_name
    for row in manifest["artifacts"]:
        if row["path"] == artifact_name:
            row.update(record(artifact_path, paths["randomization"].parent))
            break
    else:
        raise AssertionError(f"missing randomization fixture artifact {artifact_name}")
    write_json(paths["randomization"], manifest)


def refresh_pnnl_artifact(
    paths: dict[str, Any],
    manifest_key: str,
    artifact_name: str,
) -> None:
    manifest = target.load_json(paths["pnnl"])
    artifact_path = paths["pnnl"].parent / artifact_name
    manifest[manifest_key] = record(artifact_path, paths["pnnl"].parent)
    write_pnnl_manifest_with_self_size(paths["pnnl"], manifest)


def test_completed_positive_bundle_is_cross_checked_and_generated(
    tmp_path: Path,
) -> None:
    paths = make_fixture(tmp_path / "input")
    output = tmp_path / "generated"
    run_fixture(paths, output)
    expected = {
        "gate_decision_table.tex",
        "google_event_table.tex",
        "google_threshold_frontier_table.tex",
        "google_threshold_bootstrap_table.tex",
        "google_risk_budget_table.tex",
        "google_uncertainty_table.tex",
        "pnnl_macro_table.tex",
        "pnnl_state_results_table.tex",
        "pnnl_cohort_results_table.tex",
        "pnnl_comparison_table.tex",
        "pnnl_cohort_control_table.tex",
        "pnnl_randomization_audit_table.tex",
        "randomization_table.tex",
        "resource_ledger_table.tex",
        "claim_sentence.tex",
        "manuscript_artifact_contract.json",
        "manuscript_artifact_contract.tex",
        "google_event_alerts.pdf",
        "google_risk_coverage.pdf",
        "google_randomization_proper_prior.pdf",
        "pnnl_delay_forest.pdf",
        "publication_bundle_manifest.json",
    }
    assert {path.name for path in output.iterdir()} == expected
    manifest = target.load_json(output / "publication_bundle_manifest.json")
    assert manifest["schema_version"] == "run6-publication-bundle-v5"
    input_paths = {
        "detector_manifest": paths["detector"],
        "freeze_ratification": paths["freeze_ratification"],
        "repair_manifest": paths["repair_manifest"],
        "repair_ratification": paths["repair_ratification"],
        "randomization_manifest": paths["randomization"],
        "pnnl_manifest": paths["pnnl"],
        "pittsburgh_manifest": paths["pittsburgh"],
        "outcome_manifest": paths["outcome"],
    }
    for name, path in input_paths.items():
        assert manifest["evidence_inputs"][name] == {
            "path": FIXTURE_EVIDENCE_PATHS[name],
            "sha256": target.sha256_file(path),
        }
    assert (
        manifest["dual_provenance"]["post_detector_repair_chain"]["access_record"]
        == target.REPAIR_ACCESS_RECORD
    )
    assert manifest["decision"]["overall_run6_advantage"] is True
    positive_claim = manifest["decision"]["claim_sentence"]
    assert "both satisfied" in positive_claim
    assert "does not establish superiority" in positive_claim
    assert "same-feature or same-parity logistic or threshold classes" in positive_claim
    assert "correct likelihood or oracle rule" in positive_claim
    assert "general algorithmic advantage" in positive_claim
    assert (
        "supports a dataset-specific empirical algorithmic advantage"
        not in positive_claim
    )
    by_name = {row["path"]: row for row in manifest["artifacts"]}
    assert set(by_name) == expected - {"publication_bundle_manifest.json"}
    for name, row in by_name.items():
        artifact = output / name
        assert row["bytes"] == artifact.stat().st_size
        assert row["sha256"] == target.sha256_file(artifact)
    contract = target.load_json(output / "manuscript_artifact_contract.json")
    assert contract["schema_version"] == "run6-manuscript-artifact-contract-v2"
    assert manifest["manuscript_artifact_contract"] == {
        "path": "manuscript_artifact_contract.json",
        "sha256": target.sha256_file(output / "manuscript_artifact_contract.json"),
    }
    assert set(contract["unsupported_manuscript_fields"]) == {
        "RANDOMIZATION_SR_SUMMARY",
        "S_PACE_WALL_TIME_SECONDS",
        "S_PACE_PEAK_MEMORY_MIB",
        "PNNL_PATHS_RETAINED",
        "GOOGLE_S_PACE_EVENT_SCORE_PERCENTILE",
        "GOOGLE_CONTEXTUAL_BEST_METHOD",
        "POSITIVE_ALGORITHMIC_SUPERIORITY",
    }
    assert contract["figures"]["google_event"]["path"] == "google_event_alerts.pdf"
    assert (
        contract["figures"]["google_randomization"]["path"]
        == "google_randomization_proper_prior.pdf"
    )
    assert (
        "budgets 2, 20, and 200"
        in contract["figures"]["google_risk"]["caption_contract"]
    )
    assert (
        "descriptive only"
        in contract["tables"]["google_threshold_bootstrap"]["caption_contract"]
    )
    assert (
        "not a natural-hardware null"
        in contract["tables"]["pnnl_randomization"]["caption_contract"]
    )
    assert (
        "neither empirical gate nor the overall conjunction"
        in contract["tables"]["pnnl_randomization"]["caption_contract"]
    )
    assert (
        "All 22 ordered path-state rows"
        in contract["tables"]["pnnl_state_results"]["caption_contract"]
    )
    assert (
        "All 11 ordered state-averaged cohorts"
        in contract["tables"]["pnnl_cohort_results"]["caption_contract"]
    )
    assert (
        "no equal-QASM controlled cohort"
        in contract["figures"]["pnnl"]["caption_contract"]
    )
    assert (
        "Eleven path-level effects" in contract["figures"]["pnnl"]["caption_contract"]
    )
    assert (
        "absent from every locked Boolean"
        in contract["validated_descriptive_not_gate_fields"][
            "partial_trapezoidal_recall_area"
        ]
    )
    risk_table = (output / "google_risk_budget_table.tex").read_text(encoding="utf-8")
    assert "Primary: correlated matching" in risk_table
    assert "Secondary: PyMatching" in risk_table
    assert all(f" {budget} &" in risk_table for budget in target.RISK_BUDGETS)
    risk_data_lines = [
        line for line in risk_table.splitlines() if line.rstrip().endswith(r"\\")
    ]
    assert len(risk_data_lines) == 49
    label_names = {
        "correlated_matching_mismatch": "Primary: correlated matching",
        "pymatching_mismatch": "Secondary: PyMatching",
    }
    for label in target.LABEL_IDS:
        for budget in target.RISK_BUDGETS:
            for method in target.METHOD_IDS:
                prefix = " & ".join(
                    target.tex_escape(value)
                    for value in (
                        label_names[label],
                        budget,
                        target.METHOD_METADATA[method][0],
                    )
                )
                assert (
                    sum(line.startswith(prefix + " &") for line in risk_data_lines) == 1
                )
    for method in ("m0c", "m1", "m2", "m4", "m5"):
        assert target.METHOD_METADATA[method][0] in risk_table

    uncertainty_table = (output / "google_uncertainty_table.tex").read_text(
        encoding="utf-8"
    )
    uncertainty_lines = [
        line for line in uncertainty_table.splitlines() if line.rstrip().endswith(r"\\")
    ]
    assert len(uncertainty_lines) == 13
    assert "Recall difference [95" in uncertainty_table
    assert "Precision difference [95" in uncertainty_table
    assert uncertainty_table.count("Gate capture contrast") == 2

    event_table = (output / "google_event_table.tex").read_text(encoding="utf-8")
    assert "Pre-event shots" in event_table
    assert "40100, 56900" in event_table

    assert "no equal-QASM control" in (
        output / "pnnl_cohort_control_table.tex"
    ).read_text(encoding="utf-8")
    assert "Selected-alert histogram" in (
        output / "google_threshold_bootstrap_table.tex"
    ).read_text(encoding="utf-8")
    pnnl_randomization_table = (
        output / "pnnl_randomization_audit_table.tex"
    ).read_text(encoding="utf-8")
    assert "Alarmed episodes/replicate histogram" in pnnl_randomization_table
    randomization_lines = [
        line
        for line in pnnl_randomization_table.splitlines()
        if line.rstrip().endswith(r"\\")
    ]
    assert len(randomization_lines) == 29
    randomization_episode_lines = [
        line for line in randomization_lines if line.startswith("C")
    ]
    assert len(randomization_episode_lines) == 22
    assert all(len(line.split(" & ")) == 8 for line in randomization_episode_lines)
    for cohort_index in range(11):
        for logical_state in (0, 1):
            prefix = (
                f"C{cohort_index + 1:02d} & "
                f"cohort\\_{cohort_index:02d} & {logical_state} &"
            )
            assert (
                sum(line.startswith(prefix) for line in randomization_episode_lines)
                == 1
            )
    assert "C01 & cohort\\_00 & 0 & 0.000000 & 1.000000" in (pnnl_randomization_table)
    assert (
        "C11 & cohort\\_10 & 1 & 0.000000 & 0.000000 & 0.000000 & 0.000000 & 0.000000"
    ) in pnnl_randomization_table

    pnnl_state_table = (output / "pnnl_state_results_table.tex").read_text(
        encoding="utf-8"
    )
    state_lines = [
        line for line in pnnl_state_table.splitlines() if line.rstrip().endswith(r"\\")
    ]
    assert len(state_lines) == 23
    state_data_lines = [line for line in state_lines if line.startswith("C")]
    assert len(state_data_lines) == 22
    assert all(len(line.split(" & ")) == 8 for line in state_data_lines)
    assert "C01 & cohort\\_00 & 0 & 0/0/0.4000" in pnnl_state_table
    assert "C11 & cohort\\_10 & 1" in pnnl_state_table

    pnnl_cohort_table = (output / "pnnl_cohort_results_table.tex").read_text(
        encoding="utf-8"
    )
    cohort_result_lines = [
        line for line in pnnl_cohort_table.splitlines() if line.rstrip().endswith(r"\\")
    ]
    assert len(cohort_result_lines) == 12
    cohort_data_lines = [line for line in cohort_result_lines if line.startswith("C")]
    assert len(cohort_data_lines) == 11
    assert all(len(line.split(" & ")) == 7 for line in cohort_data_lines)
    assert "C01 & cohort\\_00 & 0.000/0.000/0.4000" in pnnl_cohort_table
    assert "C11 & cohort\\_10" in pnnl_cohort_table

    comparison_table = (output / "pnnl_comparison_table.tex").read_text(
        encoding="utf-8"
    )
    assert "Calibration-pair sensitivity 95" in comparison_table
    assert "Exact sign-flip p" in comparison_table
    assert "0.010000" in comparison_table

    resource_table = (output / "resource_ledger_table.tex").read_text(encoding="utf-8")
    assert "Joint-pipeline runs 4.000/4.500/5.000 s; median 4.500 s" in resource_table
    assert "Whole-run 30.000 s; held processing 25.000 s" in resource_table
    assert "PNNL threshold-bootstrap surrogates" in resource_table
    assert "PNNL paired-swap surrogates" in resource_table
    pittsburgh = target.validate_pittsburgh_manifest(paths["pittsburgh"])
    expected_resources = target.pnnl_expected_resource_counts(pittsburgh["cohorts"])
    for key in (
        "fit_eigendecompositions",
        "actual_surveillance_eigendecompositions",
        "bootstrap_eigendecompositions",
        "randomization_eigendecompositions",
    ):
        assert f"eigendecompositions {expected_resources[key]:,}" in resource_table


def test_negative_retention_emits_exact_no_advantage_sentence(tmp_path: Path) -> None:
    paths = make_fixture(tmp_path / "input", retention=False)
    output = tmp_path / "generated"
    run_fixture(paths, output)
    manifest = target.load_json(output / "publication_bundle_manifest.json")
    assert manifest["decision"]["overall_run6_advantage"] is False
    assert (
        manifest["decision"]["claim_sentence"]
        == "No demonstrated S-PACE algorithmic advantage."
    )


def test_positive_claim_denies_class_or_oracle_superiority() -> None:
    claim = target.conclusion_sentence(True)
    assert claim.startswith(
        "The locked Google and PNNL empirical gates were both satisfied"
    )
    assert "does not establish superiority" in claim
    assert "same-feature or same-parity logistic or threshold classes" in claim
    assert "a correct likelihood or oracle rule" in claim
    assert "a general algorithmic advantage" in claim
    assert "supports an advantage" not in claim
    assert "outperforms" not in claim

    manuscript = Path(__file__).parents[1] / "main.tex"
    source = manuscript.read_text(encoding="utf-8")
    assert r"\subsection{Locked advantage rule}" not in source
    assert "empirical, dataset-specific algorithmic advantage" not in source
    assert "passing does not establish superiority" in source


def test_production_profile_is_grounded_in_exact_git_blobs() -> None:
    profile = target.load_production_validation_profile()
    original = target.load_json_bytes(
        profile.original_ratification_bytes,
        context="test original ratification",
    )
    manifest = target.load_json_bytes(
        profile.repair_manifest_bytes,
        context="test repair manifest",
    )
    ratification = target.load_json_bytes(
        profile.repair_ratification_bytes,
        context="test repair ratification",
    )

    assert profile.evidence_path_map() == target.PRODUCTION_EVIDENCE_PATHS
    assert manifest["hashes"] == profile.implementation_hash_map()
    assert ratification["hashes"][target.REPAIR_MANIFEST_PATH] == (
        target.sha256_bytes(profile.repair_manifest_bytes)
    )
    assert {
        path: ratification["hashes"][path] for path in target.REPAIR_DIFF_STATUS
    } == profile.implementation_hash_map()

    failures = manifest["failed_attempt_evidence"]
    assert failures["root"] == "experiments/run6/results/google_randomization"
    assert failures["attempt_shard_ranges"] == [
        [start, start + 8] for start in range(0, 256, 8)
    ]
    empty_directories = failures["empty_result_directories"]
    assert len(empty_directories) == 32
    attempt_directories = [path.removesuffix("/result") for path in empty_directories]
    expected_files = {
        f"{directory}/{stream}"
        for directory in attempt_directories
        for stream in ("stderr.log", "stdout.log")
    }
    assert set(failures["files"]) == expected_files
    assert failures["common_stderr_sha256"] == COMMON_STDERR_DIGEST
    for name, row in failures["files"].items():
        if name.endswith("/stderr.log"):
            assert row == {"bytes": 891, "sha256": COMMON_STDERR_DIGEST}
        else:
            assert row == {
                "bytes": 0,
                "sha256": target.sha256_bytes(b""),
            }

    threads = {
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }
    assert manifest["thread_environment"] == threads
    assert manifest["environment"] == original["environment"]
    assert (
        manifest["python_environment_lock_sha256"]
        == original["hashes"]["experiments/run6/configs/python_environment_lock.txt"]
    )
    assert manifest["runtime_module_origins"] == {
        "aoc": "experiments/aoc/__init__.py",
        "aoc.qec_real": "experiments/aoc/qec_real.py",
        "aoc.run6_protocol": "experiments/aoc/run6_protocol.py",
        "aoc.run6_repair": "experiments/aoc/run6_repair.py",
        "aoc.space": "experiments/aoc/space.py",
        "aoc.space_qec": "experiments/aoc/space_qec.py",
    }


def test_incomplete_decision_is_refused(tmp_path: Path) -> None:
    paths = make_fixture(tmp_path / "input")
    decision_path = paths["outcome"].parent / "decision_summary.json"
    decision = target.load_json(decision_path)
    decision["summary_scope"] = "google_detector_and_outcome_only_not_full_run6_summary"
    write_json(decision_path, decision)
    manifest = target.load_json(paths["outcome"])
    for row in manifest["artifacts"]:
        if row["path"] == "decision_summary.json":
            row.update(record(decision_path, paths["outcome"].parent))
    write_json(paths["outcome"], manifest)
    with pytest.raises(target.PublicationDataError, match="completed locked run"):
        run_fixture(paths, tmp_path / "generated")


def test_repair_access_record_cannot_claim_detector_blindness(
    tmp_path: Path,
) -> None:
    paths = make_fixture(tmp_path / "input")
    ratification = target.load_json(paths["repair_ratification"])
    ratification["access_record"]["detector_values_accessed_before_repair"] = False
    write_json(paths["repair_ratification"], ratification)
    with pytest.raises(
        target.PublicationDataError,
        match="recursively bind the repair evidence",
    ):
        run_fixture(paths, tmp_path / "generated")


def test_randomization_must_bind_repair_ratification(tmp_path: Path) -> None:
    paths = make_fixture(tmp_path / "input")
    randomization = target.load_json(paths["randomization"])
    randomization["repair_ratification_sha256"] = DIGEST
    write_json(paths["randomization"], randomization)
    with pytest.raises(
        target.PublicationDataError,
        match="Randomization manifest bindings are inconsistent",
    ):
        run_fixture(paths, tmp_path / "generated")


def test_google_threshold_bootstrap_replicate_tamper_is_refused(
    tmp_path: Path,
) -> None:
    paths = make_fixture(tmp_path / "input")
    bootstrap_path = paths["randomization"].parent / "threshold_bootstrap.json"
    bootstrap = target.load_json(bootstrap_path)
    bootstrap["replicate_results"][0]["methods"]["m0"][
        "selected_primary_threshold"
    ] += 0.125
    write_json(bootstrap_path, bootstrap)
    refresh_randomization_artifact(paths, "threshold_bootstrap.json")
    with pytest.raises(
        target.PublicationDataError,
        match="disagrees with replicate values",
    ):
        run_fixture(paths, tmp_path / "generated")


def test_google_threshold_bootstrap_summary_tamper_is_refused(
    tmp_path: Path,
) -> None:
    paths = make_fixture(tmp_path / "input")
    bootstrap_path = paths["randomization"].parent / "threshold_bootstrap.json"
    bootstrap = target.load_json(bootstrap_path)
    bootstrap["summaries"]["space"]["alert_count_at_frozen_threshold_frequency"] = {
        "0": 2_000
    }
    write_json(bootstrap_path, bootstrap)
    refresh_randomization_artifact(paths, "threshold_bootstrap.json")
    with pytest.raises(
        target.PublicationDataError,
        match="disagrees with replicate values",
    ):
        run_fixture(paths, tmp_path / "generated")


def test_google_partial_recall_area_tamper_is_refused(tmp_path: Path) -> None:
    paths = make_fixture(tmp_path / "input")
    risk_path = paths["outcome"].parent / "risk_summary.json"
    risk = target.load_json(risk_path)
    risk["point_estimates"]["correlated_matching_mismatch"]["m0"][
        "partial_trapezoidal_recall_area"
    ] += 0.01
    write_json(risk_path, risk)
    outcome = target.load_json(paths["outcome"])
    for row in outcome["artifacts"]:
        if row["path"] == "risk_summary.json":
            row.update(record(risk_path, paths["outcome"].parent))
    write_json(paths["outcome"], outcome)
    with pytest.raises(
        target.PublicationDataError,
        match="Partial trapezoidal recall area is inconsistent",
    ):
        run_fixture(paths, tmp_path / "generated")


@pytest.mark.parametrize("tamper", ("seed", "method_schema", "frozen_threshold"))
def test_google_threshold_bootstrap_schema_and_lock_tamper_is_refused(
    tmp_path: Path,
    tamper: str,
) -> None:
    paths = make_fixture(tmp_path / "input")
    bootstrap_path = paths["randomization"].parent / "threshold_bootstrap.json"
    bootstrap = target.load_json(bootstrap_path)
    if tamper == "seed":
        bootstrap["replicate_results"][17]["seed"] += 1
    elif tamper == "method_schema":
        bootstrap["replicate_results"][0]["methods"]["m0"]["extra"] = 0
    else:
        bootstrap["summaries"]["m0"]["frozen_threshold"] += 0.01
    write_json(bootstrap_path, bootstrap)
    refresh_randomization_artifact(paths, "threshold_bootstrap.json")
    with pytest.raises(target.PublicationDataError):
        run_fixture(paths, tmp_path / "generated")


def test_pnnl_must_bind_repair_ratification(tmp_path: Path) -> None:
    paths = make_fixture(tmp_path / "input")
    pnnl = target.load_json(paths["pnnl"])
    pnnl["repair_ratification_sha256"] = DIGEST
    write_json(paths["pnnl"], pnnl)
    with pytest.raises(
        target.PublicationDataError,
        match="PNNL results manifest differs from the lock",
    ):
        run_fixture(paths, tmp_path / "generated")


@pytest.mark.parametrize(
    "tamper",
    (
        "mapping_instead_of_rows",
        "row_count",
        "extra_row",
        "row_schema",
        "row_order",
        "cohort_id",
        "logical_state",
        "q",
        "roles",
        "component_schema_missing",
        "component_schema_extra",
        "component_dfr",
        "component_online_logistic",
        "component_space_sparse",
        "component_space_spectral",
        "component_space_composite",
        "formal_bytes",
        "adaptive_bytes",
    ),
)
def test_pnnl_adaptive_state_ledger_tamper_is_refused(
    tmp_path: Path,
    tamper: str,
) -> None:
    paths = make_fixture(tmp_path / "input")
    pnnl = target.load_json(paths["pnnl"])
    ledger = pnnl["resource_ledger"]["adaptive_state_ledger"]
    if tamper == "mapping_instead_of_rows":
        pnnl["resource_ledger"]["adaptive_state_ledger"] = {}
    elif tamper == "row_count":
        ledger.pop()
    elif tamper == "extra_row":
        ledger.append(dict(ledger[-1]))
    elif tamper == "row_schema":
        ledger[0]["extra"] = 0
    elif tamper == "row_order":
        ledger[0], ledger[1] = ledger[1], ledger[0]
    elif tamper == "cohort_id":
        ledger[0]["cohort_id"] += "_wrong"
    elif tamper == "logical_state":
        ledger[0]["logical_state"] = 1
    elif tamper == "q":
        ledger[0]["q"] += 1
    elif tamper == "roles":
        ledger[0]["roles"] += 1
    elif tamper == "component_schema_missing":
        del ledger[0]["formal_accumulator_components"]["dfr"]
    elif tamper == "component_schema_extra":
        ledger[0]["formal_accumulator_components"]["extra"] = 1
    elif tamper.startswith("component_"):
        method = tamper.removeprefix("component_")
        ledger[0]["formal_accumulator_components"][method] += 1
    elif tamper == "formal_bytes":
        ledger[0]["formal_accumulator_numeric_bytes"] += 8
    else:
        ledger[0]["adaptive_bank_numeric_bytes"] += 8
    write_json(paths["pnnl"], pnnl)
    with pytest.raises(
        target.PublicationDataError,
        match="PNNL adaptive-state ledger",
    ):
        run_fixture(paths, tmp_path / "generated")


@pytest.mark.parametrize(
    "key",
    (
        "path_groups",
        "path_state_streams",
        "paired_shots_per_pre_or_post_phase",
        "paired_cycle_updates_per_pre_or_post_phase",
        "fit_paired_shot_pairs",
        "fit_physical_circuit_shots",
        "fit_role_score_updates",
        "fit_detector_event_bits_consumed",
        "surveillance_paired_shot_pairs",
        "surveillance_physical_circuit_shots",
        "surveillance_formal_eprocess_shot_updates",
        "surveillance_role_score_updates",
        "surveillance_detector_event_bits_consumed",
        "bootstrap_replicates_per_path_state_method",
        "bootstrap_surrogate_shot_updates",
        "bootstrap_surrogate_role_score_updates",
        "randomization_replicates",
        "randomization_surrogate_shot_updates",
        "randomization_surrogate_role_score_updates",
        "fit_eigendecompositions",
        "actual_surveillance_eigendecompositions",
        "bootstrap_eigendecompositions",
        "randomization_eigendecompositions",
    ),
)
def test_pnnl_cohort_derived_resource_count_tamper_is_refused(
    tmp_path: Path,
    key: str,
) -> None:
    paths = make_fixture(tmp_path / "input")
    pnnl = target.load_json(paths["pnnl"])
    pnnl["resource_ledger"][key] += 1
    write_json(paths["pnnl"], pnnl)
    with pytest.raises(
        target.PublicationDataError,
        match=rf"PNNL resource {key} disagrees with the locked cohorts",
    ):
        run_fixture(paths, tmp_path / "generated")


@pytest.mark.parametrize(
    "key",
    (
        "output_bytes_excluding_results_manifest",
        "output_bytes_including_results_manifest",
    ),
)
def test_pnnl_output_byte_ledger_tamper_is_refused(
    tmp_path: Path,
    key: str,
) -> None:
    paths = make_fixture(tmp_path / "input")
    pnnl = target.load_json(paths["pnnl"])
    pnnl["resource_ledger"][key] += 1
    write_json(paths["pnnl"], pnnl)
    with pytest.raises(
        target.PublicationDataError,
        match="PNNL output-byte ledger disagrees",
    ):
        run_fixture(paths, tmp_path / "generated")


@pytest.mark.parametrize(
    ("container", "key"),
    (
        ("resource_ledger", "wall_seconds"),
        ("resource_ledger", "held_value_processing_wall_seconds"),
        ("manifest", "started_unix"),
        ("manifest", "held_value_processing_started_unix"),
        ("manifest", "finished_unix"),
    ),
)
def test_pnnl_wall_time_identity_tamper_is_refused(
    tmp_path: Path,
    container: str,
    key: str,
) -> None:
    paths = make_fixture(tmp_path / "input")
    pnnl = target.load_json(paths["pnnl"])
    target_container = pnnl if container == "manifest" else pnnl[container]
    target_container[key] += 1.0
    write_json(paths["pnnl"], pnnl)
    with pytest.raises(
        target.PublicationDataError,
        match="PNNL wall-time ledger disagrees",
    ):
        run_fixture(paths, tmp_path / "generated")


def test_pnnl_publication_resource_formulas_match_producer_objects() -> None:
    experiments = Path(__file__).resolve().parents[3] / "experiments"
    scripts = experiments / "run6" / "scripts"
    sys.path.insert(0, str(experiments))
    sys.path.insert(0, str(scripts))
    try:
        from run_pnnl_snapshot import (
            DimensionAdaptedBank,
            shot_component_weights,
        )

        for q in (1, 2, 3, 5, 6, 10, 11):
            for roles in (1, 5):
                observed_components = {
                    method: len(weights)
                    for method, weights in shot_component_weights(q, roles).items()
                }
                assert target.pnnl_formal_component_counts(q, roles) == (
                    observed_components
                )
                assert target.pnnl_adaptive_bank_numeric_bytes(q, roles) == (
                    DimensionAdaptedBank(q=q, role_count=roles).state_nbytes()
                )
    finally:
        sys.path.remove(str(scripts))
        sys.path.remove(str(experiments))


def test_pnnl_randomization_counts_array_tamper_is_refused(
    tmp_path: Path,
) -> None:
    paths = make_fixture(tmp_path / "input")
    counts_path = paths["pnnl"].parent / "randomization_alarm_counts.npy"
    counts = np.load(counts_path, allow_pickle=False)
    counts[0, 1] = 2
    np.save(counts_path, counts.astype("<i8"), allow_pickle=False)
    refresh_pnnl_artifact(
        paths,
        "randomization_alarm_counts",
        "randomization_alarm_counts.npy",
    )
    with pytest.raises(
        target.PublicationDataError,
        match="path-state alarm fractions disagree with counts NPY",
    ):
        run_fixture(paths, tmp_path / "generated")


def test_pnnl_randomization_maxima_array_tamper_is_refused(
    tmp_path: Path,
) -> None:
    paths = make_fixture(tmp_path / "input")
    maxima_path = paths["pnnl"].parent / "randomization_maximum_log_e.npy"
    maxima = np.load(maxima_path, allow_pickle=False)
    maxima[0, 1] = 20.0
    np.save(maxima_path, maxima.astype("<f8"), allow_pickle=False)
    refresh_pnnl_artifact(
        paths,
        "randomization_maximum_log_e",
        "randomization_maximum_log_e.npy",
    )
    with pytest.raises(
        target.PublicationDataError,
        match="maximum log-e summary disagrees with maxima NPY",
    ):
        run_fixture(paths, tmp_path / "generated")


@pytest.mark.parametrize("tamper", ("dtype", "shape", "nonfinite"))
def test_pnnl_randomization_array_schema_tamper_is_refused(
    tmp_path: Path,
    tamper: str,
) -> None:
    paths = make_fixture(tmp_path / "input")
    if tamper == "dtype":
        artifact_name = "randomization_alarm_counts.npy"
        manifest_key = "randomization_alarm_counts"
        path = paths["pnnl"].parent / artifact_name
        values = np.load(path, allow_pickle=False).astype("<i4")
    else:
        artifact_name = "randomization_maximum_log_e.npy"
        manifest_key = "randomization_maximum_log_e"
        path = paths["pnnl"].parent / artifact_name
        values = np.load(path, allow_pickle=False)
        if tamper == "shape":
            values = values[:-1]
        else:
            values[0, 0] = np.nan
    np.save(path, values, allow_pickle=False)
    refresh_pnnl_artifact(paths, manifest_key, artifact_name)
    with pytest.raises(target.PublicationDataError):
        run_fixture(paths, tmp_path / "generated")


@pytest.mark.parametrize("tamper", ("seed", "method_order", "row_order", "row_schema"))
def test_pnnl_randomization_json_schema_order_tamper_is_refused(
    tmp_path: Path,
    tamper: str,
) -> None:
    paths = make_fixture(tmp_path / "input")
    audit_path = paths["pnnl"].parent / "randomization_audit.json"
    audit = target.load_json(audit_path)
    if tamper == "seed":
        audit["seeds"][0] += 1
    elif tamper == "method_order":
        audit["method_order"][0], audit["method_order"][1] = (
            audit["method_order"][1],
            audit["method_order"][0],
        )
    elif tamper == "row_order":
        audit["path_state_method_rows"][0], audit["path_state_method_rows"][1] = (
            audit["path_state_method_rows"][1],
            audit["path_state_method_rows"][0],
        )
    else:
        audit["path_state_method_rows"][0]["extra"] = 0
    write_json(audit_path, audit)
    refresh_pnnl_artifact(paths, "randomization_audit", "randomization_audit.json")
    with pytest.raises(target.PublicationDataError):
        run_fixture(paths, tmp_path / "generated")


def test_outcome_must_bind_repair_ratification(tmp_path: Path) -> None:
    paths = make_fixture(tmp_path / "input")
    outcome = target.load_json(paths["outcome"])
    outcome["repair_ratification_sha256"] = DIGEST
    write_json(paths["outcome"], outcome)
    with pytest.raises(
        target.PublicationDataError,
        match="Outcome manifest bindings are inconsistent",
    ):
        run_fixture(paths, tmp_path / "generated")


def test_decision_must_bind_repair_ratification(tmp_path: Path) -> None:
    paths = make_fixture(tmp_path / "input")
    decision_path = paths["outcome"].parent / "decision_summary.json"
    decision = target.load_json(decision_path)
    decision["repair_ratification_sha256"] = DIGEST
    write_json(decision_path, decision)
    outcome = target.load_json(paths["outcome"])
    for row in outcome["artifacts"]:
        if row["path"] == "decision_summary.json":
            row.update(record(decision_path, paths["outcome"].parent))
    write_json(paths["outcome"], outcome)
    with pytest.raises(
        target.PublicationDataError,
        match="completed locked run",
    ):
        run_fixture(paths, tmp_path / "generated")


def test_repair_manifest_cannot_claim_detector_rerun(tmp_path: Path) -> None:
    paths = make_fixture(tmp_path / "input")
    manifest = target.load_json(paths["repair_manifest"])
    manifest["access_record"]["detector_rerun_performed_for_repair"] = True
    write_json(paths["repair_manifest"], manifest)
    ratification = target.load_json(paths["repair_ratification"])
    ratification["hashes"][target.REPAIR_MANIFEST_PATH] = target.sha256_file(
        paths["repair_manifest"]
    )
    ratification["access_record"] = manifest["access_record"]
    write_json(paths["repair_ratification"], ratification)
    with pytest.raises(
        target.PublicationDataError,
        match="exact disclosed post-detector amendment",
    ):
        run_fixture(paths, tmp_path / "generated")


def test_coordinated_repair_provenance_fabrication_is_rejected(
    tmp_path: Path,
) -> None:
    paths = make_fixture(tmp_path / "input")
    manifest = target.load_json(paths["repair_manifest"])
    manifest["failed_attempt_evidence"]["root"] = (
        "experiments/run6/results/fabricated_failed_attempts"
    )
    manifest["failed_attempt_evidence"]["attempt_shard_ranges"][0] = [0, 16]
    manifest["environment"] = {"fabricated": True}
    manifest["python_environment_lock_sha256"] = fixture_digest(99_001)
    manifest["runtime_module_origins"] = {
        "aoc.run6_repair": "fabricated/run6_repair.py"
    }
    write_json(paths["repair_manifest"], manifest)

    ratification = target.load_json(paths["repair_ratification"])
    ratification["hashes"][target.REPAIR_MANIFEST_PATH] = target.sha256_file(
        paths["repair_manifest"]
    )
    ratification["environment"] = manifest["environment"]
    ratification["python_environment_lock_sha256"] = manifest[
        "python_environment_lock_sha256"
    ]
    write_json(paths["repair_ratification"], ratification)

    with pytest.raises(
        target.PublicationDataError,
        match="failure chronology|immutable Git-committed anchor",
    ):
        run_fixture(paths, tmp_path / "generated")


def test_tampered_declared_artifact_is_refused(tmp_path: Path) -> None:
    paths = make_fixture(tmp_path / "input")
    event_path = paths["detector"].parent / "event_summary_detector_only.json"
    event_path.write_bytes(event_path.read_bytes() + b" ")
    with pytest.raises(target.PublicationDataError, match="byte count changed"):
        run_fixture(paths, tmp_path / "generated")


def test_pittsburgh_manifest_digest_mismatch_is_refused(tmp_path: Path) -> None:
    paths = make_fixture(tmp_path / "input")
    paths["pittsburgh"].write_bytes(paths["pittsburgh"].read_bytes() + b" ")
    with pytest.raises(target.PublicationDataError, match="differs from the lock"):
        run_fixture(paths, tmp_path / "generated")


def test_pnnl_cohort_metadata_must_match_pittsburgh_lock(tmp_path: Path) -> None:
    paths = make_fixture(tmp_path / "input")
    aggregate_path = paths["pnnl"].parent / "aggregate_results.json"
    aggregate = target.load_json(aggregate_path)
    aggregate["cohort_rows"][0]["calibration_pair_id"] = "tampered"
    write_json(aggregate_path, aggregate)
    pnnl = target.load_json(paths["pnnl"])
    pnnl["aggregate_results"] = record(aggregate_path, paths["pnnl"].parent)
    write_pnnl_manifest_with_self_size(paths["pnnl"], pnnl)
    with pytest.raises(
        target.PublicationDataError,
        match="metadata disagree with the Pittsburgh lock",
    ):
        run_fixture(paths, tmp_path / "generated")


def test_raw_payload_filename_is_rejected_before_resolution(tmp_path: Path) -> None:
    manifest_path = tmp_path / "results_manifest.json"
    manifest_path.write_text("{}\n", encoding="utf-8")
    raw_record = {"path": "bitstrings.json", "bytes": 1, "sha256": DIGEST}
    with pytest.raises(target.PublicationDataError, match="forbidden raw-payload"):
        target.resolve_artifact(
            raw_record,
            manifest_path,
            context="synthetic forbidden record",
        )


def test_generated_latex_fragments_compile_when_pdflatex_is_available(
    tmp_path: Path,
) -> None:
    executable = shutil.which("pdflatex")
    if executable is None:
        pytest.skip("pdflatex is unavailable")
    paths = make_fixture(tmp_path / "input")
    output = tmp_path / "generated"
    run_fixture(paths, output)
    source = r"""\documentclass{article}
\usepackage[margin=0.2in]{geometry}
\usepackage{booktabs}
\usepackage{graphicx}
\newcommand{\RunSixGeneratedDir}{.}
\input{manuscript_artifact_contract.tex}
\begin{document}
\RunSixVerifiedClaim\par
\RunSixVerifiedGateTable\par
\RunSixVerifiedGoogleEventTable\par
\RunSixVerifiedGoogleThresholdFrontierTable\par
\RunSixVerifiedGoogleRiskBudgetTable\par
\RunSixVerifiedGoogleUncertaintyTable\par
\RunSixVerifiedGoogleThresholdBootstrapTable\par
\RunSixVerifiedGoogleRandomizationTable\par
\RunSixVerifiedPNNLMacroTable\par
\RunSixVerifiedPNNLStateResultsTable\par
\RunSixVerifiedPNNLCohortResultsTable\par
\RunSixVerifiedPNNLComparisonTable\par
\RunSixVerifiedPNNLCohortControlTable\par
\RunSixVerifiedPNNLRandomizationTable\par
\RunSixVerifiedResourceLedgerTable\par
\RunSixVerifiedGoogleEventFigure\par
\RunSixVerifiedGoogleRiskFigure\par
\RunSixVerifiedGoogleRandomizationFigure\par
\RunSixVerifiedPNNLFigure\par
\end{document}
"""
    (output / "fragment_test.tex").write_text(source, encoding="utf-8")
    completed = subprocess.run(
        [
            executable,
            "-interaction=nonstopmode",
            "-halt-on-error",
            "fragment_test.tex",
        ],
        cwd=output,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout[-4_000:]
