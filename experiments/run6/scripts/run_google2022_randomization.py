#!/usr/bin/env python3
"""Locked complete-pair orientation audit for the Google 2022 Run 6 arm.

This command reads detector records but never decoder/outcome records.  It is
allowed to open ``detection_events.b8`` only after validating the committed
freeze ratification and the detector-only result manifest.  Each replicate
restores the identical post-warm-up checkpoint and applies one PCG64-drawn
orientation bit to all 51 roles of a complete paired shot.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import resource
import subprocess
import sys
import time
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
from aoc.qec_real import parse_stim_detector_layout, read_b8_detector_shots
from aoc.run6_protocol import (
    assert_no_outcome_paths,
    canonical_json_bytes,
    environment_fingerprint,
    load_google_lock,
    load_strict_json,
    require_exact_keys,
    require_thread_environment,
    sha256_file,
)
from aoc.run6_repair import verify_post_detector_repair_chain
from aoc.space import ProperUniformStartEProcessBank
from aoc.space_qec import (
    CHECK_COUNT,
    FEATURE_DIM,
    ROLE_COUNT,
    DiagonalLikelihoodModel,
    RoleHotellingModel,
    RoleIsolatedQECBank,
    apply_strict_shot_threshold,
    exact_component_priors,
    paired_qec_contrasts,
    select_role_fit_indices,
)
from scipy.stats import beta
from sklearn.covariance import LedoitWolf

METHOD_IDS = ("m0", "m0c", "m1", "m2", "m3", "m4", "m5", "space")
EXACT_METHOD_IDS = ("m0", "m1", "m3", "m4", "m5", "space")
NONFORMAL_METHOD_IDS = ("m0c", "m2")
REPAIR_RATIFICATION_RELATIVE = "experiments/run6/repair_ratification.json"
DETECTOR_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "protocol_id",
        "detector_only",
        "outcome_accessed",
        "outcome_join_authorized",
        "git_commit",
        "config_sha256",
        "method_spec_sha256",
        "detector_script_sha256",
        "freeze_ratification_sha256",
        "deviation_ledger",
        "circuit_sha256",
        "detector_layout_index_sha256",
        "warm_checkpoint_sha256",
        "threshold_checkpoint_sha256",
        "held_final_checkpoint_sha256",
        "source_archive_sha256",
        "source_archive_bytes",
        "verified_zip_member_sha256",
        "detection_file_bytes",
        "threshold_table_sha256",
        "artifacts",
        "resources",
        "performance",
        "environment",
        "command",
        "started_unix",
        "finished_unix",
    }
)
ARTIFACT_KEYS = frozenset({"path", "bytes", "sha256"})
_LOG_SIX_HUNDRED = float(np.log(600.0))
SHOT_TABLE_FIELDS = (
    "phase",
    "method",
    "pair_index",
    "reference_archive_shot",
    "monitor_archive_shot",
    "shot_score",
    "argmax_role",
    "first_crossing_role",
    "shot_alert",
    "cumulative_alert_count",
    "rank",
    "rank_tie_archive_shot",
    "in_primary_window",
    "in_narrow_window",
    "in_wide_window",
)
RANDOMIZATION_SHARD_KEYS = frozenset(
    {
        "schema_version",
        "protocol_id",
        "replicate_start",
        "replicate_stop_exclusive",
        "seed_start",
        "seed_stop_exclusive",
        "horizon_paired_shots",
        "role_score_updates_per_replicate",
        "warm_checkpoint_sha256",
        "replicates",
    }
)
RANDOMIZATION_SHARD_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "protocol_id",
        "config_sha256",
        "method_spec_sha256",
        "freeze_ratification_sha256",
        "repair_ratification_path",
        "repair_ratification_sha256",
        "detector_manifest_sha256",
        "detector_manifest_git_commit",
        "script_sha256",
        "git_commit",
        "outcome_accessed",
        "source_archive_sha256",
        "verified_zip_member_sha256",
        "warm_checkpoint_sha256",
        "rng",
        "environment",
        "command",
        "artifact",
        "resources",
    }
)


def expected_detector_artifact_names() -> set[str]:
    """Return the exact detector-output artifact contract, excluding manifest."""

    threshold_array_ids = (
        "empirical_cycle_score",
        "above_threshold",
        "notification_emitted",
        "cooldown_active",
    )
    held_array_ids = (
        *threshold_array_ids,
        "log_eprocess",
        "log_sr",
        "first_e_crossing",
        "first_sr_crossing",
    )
    names = {
        "thresholds.json",
        "threshold_shots.csv",
        "threshold_stage_manifest.json",
        "held_shots.csv",
        "event_summary_detector_only.json",
        "secondary_event_summary_detector_only.json",
        "formal_component_summary.json",
    }
    for method in METHOD_IDS:
        for array_id in threshold_array_ids:
            stem = f"threshold__{method}__{array_id}"
            names.update({f"{stem}.npy", f"{stem}.json"})
        for array_id in held_array_ids:
            stem = f"held__{method}__{array_id}"
            names.update({f"{stem}.npy", f"{stem}.json"})
        for array_id in ("candidate_threshold", "shot_alert_count"):
            stem = f"threshold__{method}__frontier_{array_id}"
            names.update({f"{stem}.npy", f"{stem}.json"})
    return names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--freeze-ratification", type=Path)
    parser.add_argument("--repair-ratification", type=Path)
    parser.add_argument("--detector-manifest", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--replicate-start", type=int)
    parser.add_argument("--replicate-stop", type=int)
    parser.add_argument(
        "--merge-shard-manifest",
        type=Path,
        action="append",
        default=[],
    )
    return parser.parse_args()


def _git_commit(repo_root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _require_ancestor(repo_root: Path, ancestor: str, descendant: str) -> None:
    completed = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise ValueError(f"Commit {ancestor!r} is not an ancestor of HEAD.")


def _write_canonical_json(path: Path, value: Any) -> None:
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def _artifact_record(path: Path) -> dict[str, Any]:
    return {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _randomization_resources(
    replicates: Sequence[Mapping[str, Any]],
    *,
    started_unix: float,
    output: Path,
) -> dict[str, Any]:
    """Account for shard/merge work without making a speed comparison."""

    return {
        "replicate_count": len(replicates),
        "formal_eprocess_shot_updates": sum(
            int(row["formal_eprocess_updates"]) for row in replicates
        ),
        "role_score_updates": sum(int(row["role_score_updates"]) for row in replicates),
        "wall_seconds": max(0.0, time.time() - started_unix),
        "peak_rss_kib_linux_ru_maxrss": int(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        ),
        "worker_process_count": 1,
        "external_concurrency_not_inferred": True,
        "output_bytes_excluding_manifest": sum(
            path.stat().st_size for path in output.iterdir() if path.is_file()
        ),
        "output_bytes_including_manifest": 0,
    }


def _write_self_accounting_manifest(path: Path, manifest: dict[str, Any]) -> None:
    """Write a canonical manifest whose byte ledger includes the manifest."""

    resources = manifest["resources"]
    for _ in range(16):
        encoded = canonical_json_bytes(manifest) + b"\n"
        expected_total = int(resources["output_bytes_excluding_manifest"]) + len(
            encoded
        )
        if resources["output_bytes_including_manifest"] == expected_total:
            path.write_bytes(encoded)
            return
        resources["output_bytes_including_manifest"] = expected_total
    raise RuntimeError("Randomization manifest byte ledger did not converge.")


def validate_randomization_replicate_row(
    row: Mapping[str, Any],
    *,
    replicate_index: int,
    horizon_shots: int,
    role_score_updates: int,
) -> None:
    """Validate one deterministic shard row before any merge."""

    require_exact_keys(
        row,
        {
            "replicate_index",
            "seed",
            "swap_sha256",
            "swapped_shot_count",
            "checkpoint_restored",
            "crossed_100",
            "first_crossing_shot_number_one_based",
            "maximum_log_e",
            "final_log_e",
            "familywide_any_crossed_600",
            "formal_eprocess_updates",
            "role_score_updates",
            "formal_experts",
        },
        context=f"randomization replicate {replicate_index}",
    )
    expected_experts = {
        "m0": 408,
        "m1": 408,
        "m3": 612,
        "m4": 3264,
        "m5": 1224,
        "space": 4488,
    }
    if (
        row["replicate_index"] != replicate_index
        or row["seed"] != 610700 + replicate_index
        or row["checkpoint_restored"] is not True
        or row["formal_eprocess_updates"] != horizon_shots
        or row["role_score_updates"] != role_score_updates
        or row["formal_experts"] != expected_experts
    ):
        raise ValueError(
            f"Randomization replicate accounting changed: {replicate_index}"
        )
    if (
        not isinstance(row["swap_sha256"], str)
        or len(row["swap_sha256"]) != 64
        or isinstance(row["swapped_shot_count"], bool)
        or not isinstance(row["swapped_shot_count"], int)
        or not 0 <= row["swapped_shot_count"] <= horizon_shots
        or not isinstance(row["familywide_any_crossed_600"], bool)
    ):
        raise ValueError(f"Randomization replicate scalar changed: {replicate_index}")
    for key in (
        "crossed_100",
        "first_crossing_shot_number_one_based",
        "maximum_log_e",
        "final_log_e",
    ):
        value = row[key]
        if not isinstance(value, Mapping):
            raise TypeError(f"Randomization replicate {key} must be an object.")
        require_exact_keys(
            value,
            EXACT_METHOD_IDS,
            context=f"randomization replicate {replicate_index}.{key}",
        )


def merge_randomization_rows(
    shard_rows: Sequence[Sequence[Mapping[str, Any]]],
    *,
    expected_replicates: int,
    horizon_shots: int,
    role_score_updates: int,
) -> list[dict[str, Any]]:
    """Merge deterministic shards with exact, gap-free index/seed coverage."""

    by_index: dict[int, dict[str, Any]] = {}
    for rows in shard_rows:
        for raw_row in rows:
            if not isinstance(raw_row, Mapping):
                raise TypeError("Randomization shard rows must be objects.")
            index = raw_row.get("replicate_index")
            if isinstance(index, bool) or not isinstance(index, int):
                raise TypeError("Randomization replicate_index must be an integer.")
            if index in by_index:
                raise ValueError(f"Duplicate randomization replicate index: {index}")
            validate_randomization_replicate_row(
                raw_row,
                replicate_index=index,
                horizon_shots=horizon_shots,
                role_score_updates=role_score_updates,
            )
            by_index[index] = dict(raw_row)
    expected_indices = set(range(expected_replicates))
    if set(by_index) != expected_indices:
        raise ValueError(
            "Randomization shard coverage is incomplete; "
            f"missing={sorted(expected_indices - by_index.keys())}, "
            f"unknown={sorted(by_index.keys() - expected_indices)}."
        )
    rows = [by_index[index] for index in range(expected_replicates)]
    seeds = [int(row["seed"]) for row in rows]
    if seeds != list(range(610700, 610700 + expected_replicates)):
        raise ValueError("Randomization seeds are not exact and gap-free.")
    return rows


def _sha256_zip_member(archive: Path, member: str) -> str:
    digest = hashlib.sha256()
    with zipfile.ZipFile(archive) as source:
        matches = [name for name in source.namelist() if name == member]
        if matches != [member]:
            raise ValueError(f"Expected exactly one ZIP member {member!r}.")
        with source.open(member, "r") as handle:
            while block := handle.read(1 << 20):
                digest.update(block)
    return digest.hexdigest()


def verify_freeze_ratification(
    path: Path,
    *,
    repair_ratification_path: Path,
    repo_root: Path,
    config_path: Path,
    config: Mapping[str, Any],
    detector_manifest_path: Path,
) -> dict[str, Any]:
    """Verify the original freeze plus committed post-detector repair chain."""

    expected_config_path = (
        repo_root / "experiments/run6/configs/google2022_locked.json"
    ).resolve()
    if config_path.resolve() != expected_config_path:
        raise ValueError("Real replay requires the canonical Google lock path.")
    if config["status"] != "frozen_before_held_value_access":
        raise ValueError("Google configuration has not reached frozen status.")
    spec_relative = config["normative_method_spec"]["path"]
    spec_path = repo_root / spec_relative
    if config["normative_method_spec"]["sha256"] != sha256_file(spec_path):
        raise ValueError("Normative method-spec hash embedded in config changed.")
    expected_threads = config["numeric_policy"]["thread_environment"]
    return verify_post_detector_repair_chain(
        original_ratification_path=path,
        repair_ratification_path=repair_ratification_path,
        repo_root=repo_root,
        detector_manifest_path=detector_manifest_path,
        expected_environment=environment_fingerprint(),
        expected_thread_environment=expected_threads,
    )


def _resolve_artifact(record: Mapping[str, Any], manifest_path: Path) -> Path:
    require_exact_keys(record, ARTIFACT_KEYS, context="detector artifact")
    raw_path = record["path"]
    if not isinstance(raw_path, str):
        raise TypeError("Detector artifact path must be a string.")
    relative = PurePosixPath(raw_path)
    if (
        not raw_path
        or relative.is_absolute()
        or relative.as_posix() != raw_path
        or any(part in {"", ".", ".."} for part in relative.parts)
        or "\\" in raw_path
    ):
        raise ValueError("Artifact paths must be canonical relative POSIX paths.")
    candidate = (manifest_path.parent / relative).resolve()
    if not candidate.is_relative_to(manifest_path.parent.resolve()):
        raise ValueError("Detector artifacts must remain beside their manifest.")
    if not candidate.is_file():
        raise FileNotFoundError(f"Detector artifact is missing: {candidate}")
    if (
        isinstance(record["bytes"], bool)
        or not isinstance(record["bytes"], int)
        or candidate.stat().st_size != record["bytes"]
        or sha256_file(candidate) != record["sha256"]
    ):
        raise ValueError(f"Detector artifact hash/size mismatch: {candidate.name}")
    return candidate


def _validate_resource_ledger(resources: Any) -> None:
    if not isinstance(resources, Mapping):
        raise TypeError("Detector resources must be an object.")
    require_exact_keys(
        resources,
        {
            "record_exposure",
            "high_level_operations",
            "warm_checkpoint_storage",
            "held_final_checkpoint_storage",
            "formal_accumulator",
            "output_bytes_before_manifest",
        },
        context="detector resources",
    )
    expected = {
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
    exposure = resources["record_exposure"]
    if not isinstance(exposure, Mapping):
        raise TypeError("Detector record_exposure must be an object.")
    require_exact_keys(exposure, expected, context="detector record_exposure")
    for phase, expected_row in expected.items():
        row = exposure[phase]
        if not isinstance(row, Mapping):
            raise TypeError(f"Detector resource row {phase} must be an object.")
        require_exact_keys(row, expected_row, context=f"detector resources.{phase}")
        if dict(row) != expected_row:
            raise ValueError(f"Detector resource row changed: {phase}")
    formal = resources["formal_accumulator"]
    if formal != {
        "time_unit": "complete_paired_shot",
        "held_updates": 20_000,
        "role_prior": "uniform_1_over_51",
        "within_shot_factor_compounding": False,
    }:
        raise ValueError("Detector formal-accumulator resources changed.")
    high_level = resources["high_level_operations"]
    if not isinstance(high_level, Mapping):
        raise TypeError("Detector high_level_operations must be an object.")
    require_exact_keys(
        high_level,
        {
            "fit_warmup",
            "threshold",
            "held",
            "m2_covariance_fits",
            "m2_precision_matrix_constructions",
            "m2_fit_observations_used",
            "held_joint_replay_repetitions",
            "extra_timing_replay_role_updates",
        },
        context="detector high_level_operations",
    )
    if (
        high_level["m2_covariance_fits"] != 1
        or high_level["m2_precision_matrix_constructions"] != 1
        or high_level["m2_fit_observations_used"] != 20_000
        or high_level["held_joint_replay_repetitions"] != 3
        or high_level["extra_timing_replay_role_updates"] != 2_040_000
    ):
        raise ValueError("Detector M2 operation ledger changed.")
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
    for phase in ("fit_warmup", "threshold", "held"):
        row = high_level[phase]
        if not isinstance(row, Mapping):
            raise TypeError(f"Operation ledger {phase} must be an object.")
        require_exact_keys(row, operation_keys, context=f"operations.{phase}")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in row.values()
        ):
            raise ValueError(f"Operation ledger {phase} has invalid counts.")
    for name in ("warm_checkpoint_storage", "held_final_checkpoint_storage"):
        storage = resources[name]
        if not isinstance(storage, Mapping):
            raise TypeError(f"{name} must be an object.")
        require_exact_keys(
            storage,
            {"total_numeric_state_bytes", "bytes_by_prefix", "array_inventory"},
            context=name,
        )
    output_bytes = resources["output_bytes_before_manifest"]
    if isinstance(output_bytes, bool) or not isinstance(output_bytes, int):
        raise TypeError("output_bytes_before_manifest must be an integer.")


def _validate_performance(performance: Any) -> None:
    if not isinstance(performance, Mapping):
        raise TypeError("Detector performance must be an object.")
    timing_keys = {
        "integrity_and_layout_seconds",
        "validation_read_seconds",
        "warm_fit_replay_seconds",
        "threshold_replay_seconds",
        "threshold_serialization_seconds",
        "held_read_seconds",
        "held_replay_seconds",
        "held_serialization_seconds",
        "elapsed_before_manifest_seconds",
    }
    require_exact_keys(
        performance,
        {
            "canonical_joint_pipeline_only",
            "not_a_per_method_speed_comparison",
            *timing_keys,
            "held_joint_replay_all_three_seconds",
            "held_joint_replay_median_seconds",
            "held_joint_replay_digests",
            "peak_rss_kib_linux_ru_maxrss",
            "relative_method_speed_claim_authorized",
        },
        context="detector performance",
    )
    if (
        performance["canonical_joint_pipeline_only"] is not True
        or performance["not_a_per_method_speed_comparison"] is not True
        or performance["relative_method_speed_claim_authorized"] is not False
        or any(
            isinstance(performance[key], bool)
            or not isinstance(performance[key], (int, float))
            or not np.isfinite(float(performance[key]))
            or float(performance[key]) < 0.0
            for key in timing_keys
        )
    ):
        raise ValueError("Detector performance semantics changed.")
    all_three = performance["held_joint_replay_all_three_seconds"]
    digests = performance["held_joint_replay_digests"]
    if (
        not isinstance(all_three, list)
        or len(all_three) != 3
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(float(value))
            or float(value) < 0.0
            for value in all_three
        )
        or performance["held_replay_seconds"] != all_three[0]
        or performance["held_joint_replay_median_seconds"]
        != float(np.median(np.asarray(all_three, dtype=np.float64)))
        or not isinstance(digests, list)
        or len(digests) != 3
        or len(set(digests)) != 1
    ):
        raise ValueError("Detector repeated timing replay ledger changed.")


def _validate_threshold_stage(
    artifacts: Mapping[str, Path],
    manifest: Mapping[str, Any],
    config: Mapping[str, Any],
) -> None:
    """Verify the pre-held threshold freeze and its exact artifact boundary."""

    stage_path = artifacts["threshold_stage_manifest.json"]
    stage = load_strict_json(stage_path)
    require_exact_keys(
        stage,
        {
            "schema_version",
            "protocol_id",
            "held_values_decoded_or_scored",
            "config_sha256",
            "method_spec_sha256",
            "detector_script_sha256",
            "warm_checkpoint_sha256",
            "threshold_final_checkpoint_sha256",
            "threshold_table_sha256",
            "threshold_artifacts",
        },
        context="threshold-stage manifest",
    )
    if (
        stage["schema_version"] != "run6-google-threshold-stage-v1"
        or stage["protocol_id"] != config["protocol_id"]
        or stage["held_values_decoded_or_scored"] is not False
        or stage["config_sha256"] != manifest["config_sha256"]
        or stage["method_spec_sha256"] != manifest["method_spec_sha256"]
        or stage["detector_script_sha256"] != manifest["detector_script_sha256"]
        or stage["warm_checkpoint_sha256"] != manifest["warm_checkpoint_sha256"]
        or stage["threshold_final_checkpoint_sha256"]
        != manifest["threshold_checkpoint_sha256"]
        or stage["threshold_table_sha256"] != manifest["threshold_table_sha256"]
    ):
        raise ValueError("Threshold-stage freeze binding changed.")
    records = stage["threshold_artifacts"]
    if not isinstance(records, list) or not records:
        raise TypeError("Threshold-stage artifacts must be a nonempty list.")
    stage_names: set[str] = set()
    for record in records:
        if not isinstance(record, Mapping):
            raise TypeError("Threshold-stage artifact record must be an object.")
        artifact = _resolve_artifact(record, stage_path)
        if artifact.name in stage_names or artifact.name not in artifacts:
            raise ValueError("Threshold-stage artifact is duplicate or unbound.")
        if artifact != artifacts[artifact.name] or artifact.name.startswith("held__"):
            raise ValueError("Threshold stage contains held or mismatched artifacts.")
        stage_names.add(artifact.name)
    expected_stage_names = {
        name
        for name in expected_detector_artifact_names()
        if name in {"thresholds.json", "threshold_shots.csv"}
        or name.startswith("threshold__")
    }
    if stage_names != expected_stage_names:
        raise ValueError(
            "Threshold-stage artifact contract mismatch; "
            f"missing={sorted(expected_stage_names - stage_names)}, "
            f"unknown={sorted(stage_names - expected_stage_names)}."
        )


def verify_detector_manifest(
    path: Path,
    *,
    config_path: Path,
    config: Mapping[str, Any],
    ratification_path: Path,
    repo_root: Path,
) -> tuple[dict[str, Any], dict[str, Path]]:
    """Hash-check every detector artifact and all cross-manifest bindings."""

    manifest = load_strict_json(path)
    require_exact_keys(
        manifest,
        DETECTOR_MANIFEST_KEYS,
        context="detector freeze manifest",
    )
    if (
        manifest["schema_version"] != "run6-google-detector-freeze-v1"
        or manifest["protocol_id"] != config["protocol_id"]
        or manifest["detector_only"] is not True
        or manifest["outcome_accessed"] is not False
        or manifest["outcome_join_authorized"] is not False
    ):
        raise ValueError("Detector manifest has invalid freeze/embargo semantics.")
    expected = {
        "config_sha256": sha256_file(config_path),
        "method_spec_sha256": sha256_file(
            repo_root / config["normative_method_spec"]["path"]
        ),
        "detector_script_sha256": sha256_file(
            repo_root / "experiments/run6/scripts/run_google2022_detector.py"
        ),
        "freeze_ratification_sha256": sha256_file(ratification_path),
        "source_archive_sha256": config["source"]["sha256"],
        "source_archive_bytes": config["source"]["archive_bytes"],
        "detection_file_bytes": config["source"]["detection_event_file_bytes"],
    }
    for key, value in expected.items():
        if manifest[key] != value:
            raise ValueError(f"Detector manifest binding changed: {key}")
    member_hashes = manifest["verified_zip_member_sha256"]
    expected_member_names = {
        "circuit_ideal.stim",
        config["source"]["detection_event_file"],
    }
    if not isinstance(member_hashes, Mapping):
        raise TypeError("verified_zip_member_sha256 must be an object.")
    require_exact_keys(
        member_hashes,
        expected_member_names,
        context="verified ZIP members",
    )
    if not isinstance(manifest["git_commit"], str):
        raise TypeError("Detector manifest git_commit must be a string.")
    _require_ancestor(repo_root, manifest["git_commit"], _git_commit(repo_root))
    if manifest["environment"] != environment_fingerprint():
        raise ValueError("Detector replay environment differs from the current one.")
    deviation = manifest["deviation_ledger"]
    if not isinstance(deviation, Mapping):
        raise TypeError("Detector deviation_ledger must be an object.")
    require_exact_keys(
        deviation,
        {"path", "sha256"},
        context="detector deviation ledger",
    )
    if deviation["path"] != config["deviation_ledger"] or deviation[
        "sha256"
    ] != sha256_file(repo_root / config["deviation_ledger"]):
        raise ValueError("Detector deviation ledger binding changed.")
    _validate_resource_ledger(manifest["resources"])
    _validate_performance(manifest["performance"])

    records = manifest["artifacts"]
    if not isinstance(records, list) or not records:
        raise TypeError("Detector manifest artifacts must be a nonempty list.")
    observed: dict[str, Path] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise TypeError("Each detector artifact record must be an object.")
        artifact = _resolve_artifact(record, path)
        if artifact.name in observed:
            raise ValueError(f"Duplicate detector artifact: {artifact.name}")
        observed[artifact.name] = artifact
    expected_artifacts = expected_detector_artifact_names()
    if set(observed) != expected_artifacts:
        raise ValueError(
            "Detector artifact contract mismatch; "
            f"missing={sorted(expected_artifacts - observed.keys())}, "
            f"unknown={sorted(observed.keys() - expected_artifacts)}."
        )
    threshold_path = observed["thresholds.json"]
    if manifest["threshold_table_sha256"] != sha256_file(threshold_path):
        raise ValueError("Detector threshold-table binding changed.")
    _validate_threshold_stage(observed, manifest, config)
    return manifest, observed


def verify_randomization_shard_manifest(
    path: Path,
    *,
    repo_root: Path,
    config_path: Path,
    config: Mapping[str, Any],
    ratification_path: Path,
    repair_ratification_path: Path,
    detector_manifest_path: Path,
    detector_manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify one resumable shard and every row before the final merge."""

    manifest = load_strict_json(path)
    require_exact_keys(
        manifest,
        RANDOMIZATION_SHARD_MANIFEST_KEYS,
        context="randomization shard manifest",
    )
    expected = {
        "schema_version": "run6-google-randomization-shard-manifest-v1",
        "protocol_id": config["protocol_id"],
        "config_sha256": sha256_file(config_path),
        "method_spec_sha256": sha256_file(
            repo_root / config["normative_method_spec"]["path"]
        ),
        "freeze_ratification_sha256": sha256_file(ratification_path),
        "repair_ratification_path": REPAIR_RATIFICATION_RELATIVE,
        "repair_ratification_sha256": sha256_file(repair_ratification_path),
        "detector_manifest_sha256": sha256_file(detector_manifest_path),
        "detector_manifest_git_commit": detector_manifest["git_commit"],
        "script_sha256": sha256_file(__file__),
        "outcome_accessed": False,
        "source_archive_sha256": config["source"]["sha256"],
        "verified_zip_member_sha256": detector_manifest["verified_zip_member_sha256"],
        "warm_checkpoint_sha256": detector_manifest["warm_checkpoint_sha256"],
        "rng": {
            "algorithm": "numpy.random.Generator(PCG64)",
            "randomization_seed_start": 610700,
            "randomization_seed_stop_exclusive": 610956,
        },
    }
    for key, value in expected.items():
        if manifest[key] != value:
            raise ValueError(f"Randomization shard binding changed: {key}")
    if manifest["environment"] != environment_fingerprint():
        raise ValueError("Randomization shard environment changed.")
    if not isinstance(manifest["git_commit"], str):
        raise TypeError("Randomization shard git_commit must be a string.")
    _require_ancestor(repo_root, manifest["git_commit"], _git_commit(repo_root))

    shard_path = _resolve_artifact(manifest["artifact"], path)
    shard = load_strict_json(shard_path)
    require_exact_keys(
        shard,
        RANDOMIZATION_SHARD_KEYS,
        context="randomization shard",
    )
    start = shard["replicate_start"]
    stop = shard["replicate_stop_exclusive"]
    if (
        shard["schema_version"] != "run6-google-randomization-shard-v1"
        or shard["protocol_id"] != config["protocol_id"]
        or isinstance(start, bool)
        or not isinstance(start, int)
        or isinstance(stop, bool)
        or not isinstance(stop, int)
        or not 0 <= start < stop <= 256
        or shard["seed_start"] != 610700 + start
        or shard["seed_stop_exclusive"] != 610700 + stop
        or shard["horizon_paired_shots"] != 5_000
        or shard["role_score_updates_per_replicate"] != 255_000
        or shard["warm_checkpoint_sha256"]
        != detector_manifest["warm_checkpoint_sha256"]
    ):
        raise ValueError("Randomization shard dimensions or range changed.")
    rows = shard["replicates"]
    if not isinstance(rows, list) or len(rows) != stop - start:
        raise ValueError("Randomization shard row count changed.")
    for offset, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise TypeError("Randomization shard row must be an object.")
        validate_randomization_replicate_row(
            row,
            replicate_index=start + offset,
            horizon_shots=5_000,
            role_score_updates=255_000,
        )

    resources = manifest["resources"]
    if not isinstance(resources, Mapping):
        raise TypeError("Randomization shard resources must be an object.")
    require_exact_keys(
        resources,
        {
            "replicate_count",
            "formal_eprocess_shot_updates",
            "role_score_updates",
            "wall_seconds",
            "peak_rss_kib_linux_ru_maxrss",
            "worker_process_count",
            "external_concurrency_not_inferred",
            "output_bytes_excluding_manifest",
            "output_bytes_including_manifest",
        },
        context="randomization shard resources",
    )
    if (
        resources["replicate_count"] != len(rows)
        or resources["formal_eprocess_shot_updates"] != len(rows) * 5_000
        or resources["role_score_updates"] != len(rows) * 255_000
        or resources["worker_process_count"] != 1
        or resources["external_concurrency_not_inferred"] is not True
        or resources["output_bytes_excluding_manifest"] != shard_path.stat().st_size
        or resources["output_bytes_including_manifest"]
        != shard_path.stat().st_size + path.stat().st_size
    ):
        raise ValueError("Randomization shard resource accounting changed.")
    for key in ("wall_seconds", "peak_rss_kib_linux_ru_maxrss"):
        value = resources[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise ValueError(f"Randomization shard resource changed: {key}")
    return manifest, shard


def draw_complete_shot_swaps(seed: int, *, shot_count: int) -> np.ndarray:
    """Make the one and only locked PCG64 orientation draw for a replicate."""

    if shot_count < 1:
        raise ValueError("shot_count must be positive.")
    generator = np.random.Generator(np.random.PCG64(int(seed)))
    return generator.integers(0, 2, size=shot_count, dtype=np.uint8)


def apply_complete_shot_swaps(
    reference: np.ndarray,
    monitor: np.ndarray,
    swaps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply one orientation bit to every role of each complete paired shot."""

    left = np.asarray(reference, dtype=np.uint8)
    right = np.asarray(monitor, dtype=np.uint8)
    orientation = np.asarray(swaps, dtype=np.uint8)
    if left.shape != right.shape or left.ndim != 3:
        raise ValueError("reference/monitor must have equal (shot,role,check) shape.")
    if orientation.shape != (left.shape[0],) or np.any(orientation > 1):
        raise ValueError("swaps must be one binary value per complete shot.")
    mask = orientation.astype(bool)[:, None, None]
    return np.where(mask, right, left), np.where(mask, left, right)


def clopper_pearson_interval(
    successes: int,
    trials: int,
    *,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Return the exact equal-tailed binomial Clopper--Pearson interval."""

    if trials < 1 or not 0 <= successes <= trials:
        raise ValueError("Require trials >= 1 and 0 <= successes <= trials.")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0,1).")
    tail = (1.0 - confidence) / 2.0
    lower = (
        0.0
        if successes == 0
        else float(beta.ppf(tail, successes, trials - successes + 1))
    )
    upper = (
        1.0
        if successes == trials
        else float(beta.ppf(1.0 - tail, successes + 1, trials - successes))
    )
    return lower, upper


def load_locked_threshold_scores(
    artifacts: Mapping[str, Path],
    *,
    shot_count: int = 5_000,
    role_count: int = ROLE_COUNT,
) -> dict[str, np.ndarray]:
    """Load every hash-verified threshold-block empirical score array."""

    result: dict[str, np.ndarray] = {}
    for method in METHOD_IDS:
        name = f"threshold__{method}__empirical_cycle_score.npy"
        if name not in artifacts:
            raise KeyError(f"Missing frozen threshold artifact: {name}")
        values = np.load(artifacts[name], allow_pickle=False)
        if values.dtype.str != "<f8":
            raise TypeError(f"{name} must be persisted little-endian float64.")
        if values.shape != (shot_count, role_count) or not np.all(np.isfinite(values)):
            raise ValueError(f"{name} has invalid shape or non-finite values.")
        result[method] = np.asarray(values, dtype=np.float64)
    return result


def load_locked_held_scores(
    artifacts: Mapping[str, Path],
    *,
    shot_count: int = 20_000,
    role_count: int = ROLE_COUNT,
) -> dict[str, np.ndarray]:
    """Load all hash-verified held empirical scores for summary validation."""

    result: dict[str, np.ndarray] = {}
    for method in METHOD_IDS:
        name = f"held__{method}__empirical_cycle_score.npy"
        values = np.load(artifacts[name], allow_pickle=False)
        if (
            values.dtype.str != "<f8"
            or values.shape != (shot_count, role_count)
            or not np.all(np.isfinite(values))
        ):
            raise ValueError(f"Frozen held score array changed: {name}")
        result[method] = np.asarray(values, dtype=np.float64)
    return result


def recompute_event_summary(
    cycles: Mapping[str, np.ndarray],
    thresholds: Mapping[str, Mapping[str, Any]],
    *,
    monitor_start: int,
    windows: Mapping[str, Sequence[int]],
    threshold_key: str,
) -> dict[str, Any]:
    """Recompute one detector-only event summary from exact cycle scores."""

    result: dict[str, Any] = {}
    primary_start = int(windows["primary"][0])
    for method in METHOD_IDS:
        applied = apply_strict_shot_threshold(
            cycles[method],
            float(thresholds[method][threshold_key]),
        )
        alert_shots = monitor_start + np.flatnonzero(applied.shot_alerts)
        method_result: dict[str, Any] = {
            "pre_event_alert_count": int(np.sum(alert_shots < primary_start)),
            "pre_event_alert_shots": alert_shots[alert_shots < primary_start].tolist(),
            "windows": {},
        }
        for window_name in ("primary", "narrow", "wide"):
            start, stop = (int(value) for value in windows[window_name])
            inside = alert_shots[(alert_shots >= start) & (alert_shots < stop)]
            if len(inside):
                first_shot = int(inside[0])
                first_role = int(
                    applied.first_crossing_roles[first_shot - monitor_start]
                )
            else:
                first_shot = None
                first_role = None
            method_result["windows"][window_name] = {
                "detected": bool(len(inside)),
                "first_alert_shot": first_shot,
                "first_alert_role": first_role,
            }
        result[method] = method_result
    return result


def validate_detector_event_summaries(
    artifacts: Mapping[str, Path],
    thresholds: Mapping[str, Mapping[str, Any]],
    *,
    windows: Mapping[str, Sequence[int]],
) -> None:
    """Validate both primary and zero-alert held event summaries."""

    cycles = load_locked_held_scores(artifacts)
    for name, threshold_key in (
        ("event_summary_detector_only.json", "threshold"),
        (
            "secondary_event_summary_detector_only.json",
            "secondary_zero_alert_threshold",
        ),
    ):
        observed = load_strict_json(artifacts[name])
        expected = recompute_event_summary(
            cycles,
            thresholds,
            monitor_start=40_000,
            windows=windows,
            threshold_key=threshold_key,
        )
        if canonical_json_bytes(observed) != canonical_json_bytes(expected):
            raise ValueError(f"Detector event summary changed: {name}")


def validate_locked_threshold_frontiers(
    artifacts: Mapping[str, Path],
    scores: Mapping[str, np.ndarray],
    *,
    protocol_id: str,
    expected_common_hashes: Mapping[str, str],
) -> None:
    """Recompute every complete candidate/count frontier exactly."""

    for method in METHOD_IDS:
        values = np.asarray(scores[method], dtype=np.float64)
        expected_candidates = np.concatenate(
            (
                np.asarray([-np.inf], dtype="<f8"),
                np.unique(values).astype("<f8", copy=False),
                np.asarray([np.inf], dtype="<f8"),
            )
        )
        sorted_shot_maxima = np.sort(np.max(values, axis=1))
        expected_counts = (
            values.shape[0]
            - np.searchsorted(
                sorted_shot_maxima,
                expected_candidates,
                side="right",
            )
        ).astype("<i8", copy=False)
        for array_id, expected, dtype in (
            ("candidate_threshold", expected_candidates, "<f8"),
            ("shot_alert_count", expected_counts, "<i8"),
        ):
            name = f"threshold__{method}__frontier_{array_id}.npy"
            observed = np.load(artifacts[name], allow_pickle=False)
            if observed.dtype.str != dtype or not np.array_equal(observed, expected):
                raise ValueError(f"Frozen threshold frontier changed: {name}")
            sidecar = load_strict_json(artifacts[name].with_suffix(".json"))
            require_exact_keys(
                sidecar,
                {
                    "schema_version",
                    "protocol_id",
                    "method_id",
                    "array_id",
                    "data_file",
                    "data_sha256",
                    "shape",
                    "dtype",
                    "candidate_rule",
                    "count_rule",
                    "pair_index_range",
                    "checkpoint_and_code_hashes",
                },
                context=f"threshold frontier sidecar {name}",
            )
            if (
                sidecar["schema_version"] != "run6-threshold-frontier-array-v1"
                or sidecar["protocol_id"] != protocol_id
                or sidecar["method_id"] != method
                or sidecar["array_id"] != array_id
                or sidecar["data_file"] != name
                or sidecar["data_sha256"] != sha256_file(artifacts[name])
                or sidecar["shape"] != list(observed.shape)
                or sidecar["dtype"] != observed.dtype.str
                or sidecar["candidate_rule"]
                != "[-inf] + sorted_unique_cycle_scores + [+inf]"
                or sidecar["count_rule"]
                != "strict_greater_than_with_at_most_one_notification_per_shot"
                or sidecar["pair_index_range"] != [5_000, 5_000 + values.shape[0]]
            ):
                raise ValueError(f"Threshold frontier sidecar changed: {name}")
            hashes = sidecar["checkpoint_and_code_hashes"]
            if not isinstance(hashes, Mapping):
                raise TypeError("Frontier checkpoint/code hashes must be an object.")
            require_exact_keys(
                hashes,
                {
                    "config_sha256",
                    "method_spec_sha256",
                    "detector_script_sha256",
                    "warm_checkpoint_sha256",
                    "threshold_final_checkpoint_sha256",
                    "freeze_ratification_sha256",
                    "deviation_ledger_sha256",
                    "python_environment_lock_sha256",
                    "freeze_manifest_sha256",
                },
                context=f"threshold frontier hashes {name}",
            )
            if dict(hashes) != dict(expected_common_hashes):
                raise ValueError(f"Threshold frontier hash bindings changed: {name}")


def load_locked_threshold_table(path: Path) -> dict[str, dict[str, Any]]:
    """Load the exact primary empirical thresholds without defaults."""

    payload = load_strict_json(path)
    require_exact_keys(payload, METHOD_IDS, context="locked threshold table")
    result: dict[str, dict[str, Any]] = {}
    for method in METHOD_IDS:
        row = payload[method]
        if not isinstance(row, Mapping):
            raise TypeError(f"Threshold row {method} must be an object.")
        require_exact_keys(
            row,
            {
                "threshold",
                "validation_alert_count",
                "max_validation_alerts",
                "secondary_zero_alert_threshold",
                "secondary_validation_alert_count",
            },
            context=f"threshold table.{method}",
        )
        threshold = row["threshold"]
        count = row["validation_alert_count"]
        threshold_is_valid = (
            isinstance(threshold, (int, float))
            and not isinstance(threshold, bool)
            and np.isfinite(float(threshold))
        ) or threshold == "+inf"
        if (
            not threshold_is_valid
            or isinstance(count, bool)
            or not isinstance(count, int)
            or not 0 <= count <= 2
            or row["max_validation_alerts"] != 2
            or isinstance(row["secondary_zero_alert_threshold"], bool)
            or not isinstance(
                row["secondary_zero_alert_threshold"],
                (int, float),
            )
            or not np.isfinite(float(row["secondary_zero_alert_threshold"]))
            or row["secondary_validation_alert_count"] != 0
        ):
            raise ValueError(f"Threshold row {method} violates the lock.")
        result[method] = dict(row)
    return result


def validate_formal_detector_artifacts(
    artifacts: Mapping[str, Path],
    thresholds: Mapping[str, Mapping[str, Any]],
    *,
    protocol_id: str,
    shot_count: int = 20_000,
    role_count: int = ROLE_COUNT,
) -> None:
    """Validate formal traces, scopes, crossings, and role/base prior metadata."""

    traces: dict[str, dict[str, np.ndarray]] = {}
    for method in METHOD_IDS:
        traces[method] = {}
        for array_id in (
            "log_eprocess",
            "log_sr",
            "first_e_crossing",
            "first_sr_crossing",
        ):
            name = f"held__{method}__{array_id}.npy"
            values = np.load(artifacts[name], allow_pickle=False)
            expected_dtype = "<f8" if array_id.startswith("log_") else "|b1"
            if values.dtype.str != expected_dtype or values.shape != (shot_count,):
                raise ValueError(f"Formal detector array shape/dtype changed: {name}")
            if array_id.startswith("log_"):
                if method in EXACT_METHOD_IDS and not np.all(np.isfinite(values)):
                    raise ValueError(f"Exact formal trace is not finite: {name}")
                if method in NONFORMAL_METHOD_IDS and not np.all(np.isnan(values)):
                    raise ValueError(f"Nonformal trace is not all-NaN: {name}")
            elif method in NONFORMAL_METHOD_IDS and np.any(values):
                raise ValueError(f"Nonformal crossing mask is not all-false: {name}")
            traces[method][array_id] = values

            sidecar = load_strict_json(artifacts[name].with_suffix(".json"))
            require_exact_keys(
                sidecar,
                {
                    "schema_version",
                    "protocol_id",
                    "run_id",
                    "phase",
                    "method_id",
                    "array_id",
                    "data_file",
                    "data_sha256",
                    "shape",
                    "dtype",
                    "flatten_order",
                    "pair_index_range",
                    "reference_archive_start",
                    "monitor_archive_start",
                    "threshold",
                    "formal_claim_scope",
                    "cooldown_semantics",
                    "checkpoint_and_code_hashes",
                },
                context=f"formal cycle sidecar {name}",
            )
            expected_scope = (
                "not_applicable_no_formal_accumulator"
                if method in NONFORMAL_METHOD_IDS
                else (
                    "diagnostic_only_on_natural_hardware; "
                    "no exchangeable hardware null asserted"
                )
            )
            if (
                sidecar["schema_version"] != "run6-cycle-array-v1"
                or sidecar["protocol_id"] != protocol_id
                or sidecar["run_id"] != "google2022-canonical-detector"
                or sidecar["phase"] != "held"
                or sidecar["method_id"] != method
                or sidecar["array_id"] != array_id
                or sidecar["data_file"] != name
                or sidecar["data_sha256"] != sha256_file(artifacts[name])
                or sidecar["shape"] != [shot_count]
                or sidecar["dtype"] != expected_dtype
                or sidecar["flatten_order"] != ["paired_shot"]
                or sidecar["pair_index_range"] != [0, shot_count]
                or sidecar["reference_archive_start"] != 20_000
                or sidecar["monitor_archive_start"] != 40_000
                or sidecar["threshold"] != thresholds[method]["threshold"]
                or sidecar["formal_claim_scope"] != expected_scope
            ):
                raise ValueError(f"Formal detector sidecar changed: {name}")

    summary = load_strict_json(artifacts["formal_component_summary.json"])
    require_exact_keys(
        summary,
        {
            "schema_version",
            "held_trace_interpretation",
            "proper_prior",
            "shiryaev_roberts",
            "expert_metadata",
        },
        context="formal component summary",
    )
    if (
        summary["schema_version"] != "run6-formal-component-summary-v1"
        or set(summary["proper_prior"]) != set(EXACT_METHOD_IDS)
        or set(summary["shiryaev_roberts"]) != set(EXACT_METHOD_IDS)
        or set(summary["expert_metadata"]) != set(EXACT_METHOD_IDS)
    ):
        raise ValueError("Formal component summary method coverage changed.")
    priors = exact_component_priors()
    for method in EXACT_METHOD_IDS:
        base_weights = np.asarray(priors[method].weights, dtype=np.float64)
        base_count = len(base_weights)
        expert_count = role_count * base_count
        raw_expected_component_weights = np.tile(
            base_weights / role_count,
            role_count,
        )
        expected_component_weights = (
            raw_expected_component_weights / raw_expected_component_weights.sum()
        )
        proper = summary["proper_prior"][method]
        sr = summary["shiryaev_roberts"][method]
        metadata = summary["expert_metadata"][method]
        if (
            not isinstance(proper, Mapping)
            or not isinstance(sr, Mapping)
            or not isinstance(metadata, Mapping)
        ):
            raise TypeError("Formal component summary rows must be objects.")
        proper_accumulator_keys = {
            "component_weights",
            "role_count",
            "base_component_count",
            "expert_flatten_order",
            "expert_id_rule",
            "final_log_components",
            "final_log_statistic",
            "first_crossing_update",
            "threshold",
        }
        sr_accumulator_keys = proper_accumulator_keys - {"expert_id_rule"}
        require_exact_keys(
            proper,
            proper_accumulator_keys,
            context=f"formal proper-prior summary {method}",
        )
        require_exact_keys(
            sr,
            sr_accumulator_keys,
            context=f"formal SR summary {method}",
        )
        require_exact_keys(
            metadata,
            {
                "expert_flatten_order",
                "role_prior",
                "within_shot_factor_compounding",
                "base_component_ids",
                "base_component_weights",
                "expert_count",
                "observed_factor_minimum",
                "observed_factor_maximum",
                "all_factors_finite_and_nonnegative",
                "declared_factor_bounds",
                "factor_bounds_satisfied",
                "base_prior_sum",
                "full_role_component_prior_sum",
            },
            context=f"formal expert metadata {method}",
        )
        proper_weights = np.asarray(proper["component_weights"], dtype=np.float64)
        sr_weights = np.asarray(sr["component_weights"], dtype=np.float64)
        metadata_base_weights = np.asarray(
            metadata["base_component_weights"],
            dtype=np.float64,
        )
        observed_minimum = metadata["observed_factor_minimum"]
        observed_maximum = metadata["observed_factor_maximum"]
        observed_bounds_valid = (
            isinstance(observed_minimum, (int, float))
            and not isinstance(observed_minimum, bool)
            and isinstance(observed_maximum, (int, float))
            and not isinstance(observed_maximum, bool)
            and np.isfinite(float(observed_minimum))
            and np.isfinite(float(observed_maximum))
            and 0.1 - 1e-12
            <= float(observed_minimum)
            <= float(observed_maximum)
            <= 1.9 + 1e-12
        )
        if (
            proper["role_count"] != role_count
            or proper["base_component_count"] != base_count
            or sr["role_count"] != role_count
            or sr["base_component_count"] != base_count
            or proper["expert_flatten_order"] != ["role", "base_component"]
            or sr["expert_flatten_order"] != ["role", "base_component"]
            or metadata["expert_flatten_order"] != ["role", "base_component"]
            or metadata["role_prior"] != 1.0 / role_count
            or metadata["within_shot_factor_compounding"] is not False
            or metadata["expert_count"] != expert_count
            or metadata["base_component_ids"]
            != [list(identifier) for identifier in priors[method].component_ids]
            or not np.array_equal(metadata_base_weights, base_weights)
            or not np.array_equal(proper_weights, expected_component_weights)
            or not np.array_equal(sr_weights, expected_component_weights)
            or not np.isclose(
                float(metadata["base_prior_sum"]),
                1.0,
                rtol=0.0,
                atol=1e-15,
            )
            or not np.isclose(
                float(metadata["full_role_component_prior_sum"]),
                1.0,
                rtol=0.0,
                atol=1e-15,
            )
            or not np.isclose(np.sum(proper_weights), 1.0, rtol=0.0, atol=1e-15)
            or not np.isclose(np.sum(sr_weights), 1.0, rtol=0.0, atol=1e-15)
            or proper["expert_id_rule"]
            != "(role, *base_component_id), role-major then base-component-major"
            or proper["threshold"] != 100.0
            or sr["threshold"] != 1_000_000.0
            or metadata["declared_factor_bounds"] != [0.1, 1.9]
            or metadata["all_factors_finite_and_nonnegative"] is not True
            or metadata["factor_bounds_satisfied"] is not True
            or not observed_bounds_valid
            or len(proper["final_log_components"]) != expert_count
            or len(sr["final_log_components"]) != expert_count
            or proper["final_log_statistic"]
            != float(traces[method]["log_eprocess"][-1])
            or sr["final_log_statistic"] != float(traces[method]["log_sr"][-1])
        ):
            raise ValueError(f"Formal component/prior accounting changed: {method}")
        for family, trace_id, crossing_id in (
            (proper, "log_eprocess", "first_e_crossing"),
            (sr, "log_sr", "first_sr_crossing"),
        ):
            crossing = family["first_crossing_update"]
            expected_crossing = np.zeros(shot_count, dtype=np.bool_)
            if crossing is not None:
                if (
                    isinstance(crossing, bool)
                    or not isinstance(crossing, int)
                    or not 1 <= crossing <= shot_count
                ):
                    raise ValueError("Formal first crossing is out of range.")
                expected_crossing[crossing - 1] = True
            if not np.array_equal(
                traces[method][crossing_id],
                expected_crossing,
            ):
                raise ValueError(f"Formal crossing mask changed: {method}/{trace_id}")


def validate_frozen_shot_table(
    path: Path,
    cycles: Mapping[str, np.ndarray],
    thresholds: Mapping[str, Mapping[str, Any]],
    *,
    phase: str,
    pair_index_start: int,
    reference_start: int,
    monitor_start: int,
    windows: Mapping[str, Sequence[int]] | None,
) -> None:
    """Cross-check a canonical detector CSV against every frozen score array."""

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != SHOT_TABLE_FIELDS:
            raise ValueError("Frozen shot-table columns changed.")
        rows = iter(reader)
        for method in METHOD_IDS:
            method_cycles = cycles[method]
            applied = apply_strict_shot_threshold(
                method_cycles,
                float(thresholds[method]["threshold"]),
            )
            archive_shots = monitor_start + np.arange(method_cycles.shape[0])
            order = np.lexsort((archive_shots, -applied.shot_scores))
            ranks = np.empty(len(order), dtype=np.int64)
            ranks[order] = np.arange(1, len(order) + 1, dtype=np.int64)
            cumulative_alerts = np.cumsum(
                applied.shot_alerts,
                dtype=np.int64,
            )
            for local_index in range(method_cycles.shape[0]):
                try:
                    row = next(rows)
                except StopIteration as error:
                    raise ValueError("Frozen shot table ended early.") from error
                monitor_shot = monitor_start + local_index

                def in_window(name: str, *, shot: int = monitor_shot) -> int:
                    if windows is None:
                        return 0
                    start, stop = (int(value) for value in windows[name])
                    return int(start <= shot < stop)

                if (
                    row["phase"] != phase
                    or row["method"] != method
                    or int(row["pair_index"]) != pair_index_start + local_index
                    or int(row["reference_archive_shot"])
                    != reference_start + local_index
                    or int(row["monitor_archive_shot"]) != monitor_shot
                    or float(row["shot_score"])
                    != float(applied.shot_scores[local_index])
                    or int(row["argmax_role"])
                    != int(applied.shot_score_roles[local_index])
                    or int(row["first_crossing_role"])
                    != int(applied.first_crossing_roles[local_index])
                    or int(row["shot_alert"]) != int(applied.shot_alerts[local_index])
                    or int(row["cumulative_alert_count"])
                    != int(cumulative_alerts[local_index])
                    or int(row["rank"]) != int(ranks[local_index])
                    or int(row["rank_tie_archive_shot"]) != monitor_shot
                    or int(row["in_primary_window"]) != in_window("primary")
                    or int(row["in_narrow_window"]) != in_window("narrow")
                    or int(row["in_wide_window"]) != in_window("wide")
                ):
                    raise ValueError(
                        f"Frozen shot-table mismatch at {method}/{local_index}."
                    )
        try:
            next(rows)
        except StopIteration:
            return
        raise ValueError("Frozen shot table contains extra rows.")


def draw_threshold_bootstrap_indices(
    seed: int,
    *,
    shot_count: int,
    block_length: int = 128,
) -> np.ndarray:
    """Draw one locked circular complete-shot threshold bootstrap."""

    if shot_count < 1 or block_length < 1:
        raise ValueError("shot_count and block_length must be positive.")
    block_count = math.ceil(shot_count / block_length)
    generator = np.random.Generator(np.random.PCG64(int(seed)))
    starts = generator.integers(0, shot_count, size=block_count)
    offsets = np.arange(block_length, dtype=np.int64)
    return ((starts[:, None] + offsets[None, :]) % shot_count).reshape(-1)[:shot_count]


def _percentile_summary(values: Sequence[float]) -> dict[str, float]:
    numeric = np.asarray(values, dtype=np.float64)
    lower, median, upper = np.percentile(
        numeric,
        [2.5, 50.0, 97.5],
        method="linear",
    )
    return {
        "lower_2_5": float(lower),
        "median": float(median),
        "upper_97_5": float(upper),
    }


def threshold_bootstrap(
    scores: Mapping[str, np.ndarray],
    locked_table: Mapping[str, Mapping[str, Any]],
    *,
    replicates: int = 2_000,
    seed_start: int = 613_000,
    block_length: int = 128,
) -> dict[str, Any]:
    """Run the locked descriptive threshold/count block bootstrap."""

    if replicates < 1:
        raise ValueError("replicates must be positive.")
    first = np.asarray(scores[METHOD_IDS[0]], dtype=np.float64)
    if first.ndim != 2 or first.shape[0] < 3:
        raise ValueError("Threshold scores must have at least three complete shots.")
    shot_count, role_count = first.shape
    for method in METHOD_IDS:
        values = np.asarray(scores[method], dtype=np.float64)
        if values.shape != (shot_count, role_count) or not np.all(np.isfinite(values)):
            raise ValueError(f"Threshold score matrix changed for {method}.")
        if method not in locked_table:
            raise KeyError(f"Missing locked threshold row for {method}.")

    rows: list[dict[str, Any]] = []
    selected_thresholds = {
        method: np.empty(replicates, dtype=np.float64) for method in METHOD_IDS
    }
    selected_counts = {
        method: np.empty(replicates, dtype=np.int64) for method in METHOD_IDS
    }
    locked_counts = {
        method: np.empty(replicates, dtype=np.int64) for method in METHOD_IDS
    }
    secondary_thresholds = {
        method: np.empty(replicates, dtype=np.float64) for method in METHOD_IDS
    }
    for replicate in range(replicates):
        seed = seed_start + replicate
        indices = draw_threshold_bootstrap_indices(
            seed,
            shot_count=shot_count,
            block_length=block_length,
        )
        method_rows: dict[str, Any] = {}
        for method in METHOD_IDS:
            shot_maxima = np.max(scores[method][indices], axis=1)
            selected = float(np.partition(shot_maxima, shot_count - 3)[shot_count - 3])
            selected_count = int(np.sum(shot_maxima > selected))
            if selected_count > 2:
                raise RuntimeError("Bootstrap primary threshold exceeded its budget.")
            locked_threshold = float(locked_table[method]["threshold"])
            locked_count = int(np.sum(shot_maxima > locked_threshold))
            secondary = float(np.max(shot_maxima))
            selected_thresholds[method][replicate] = selected
            selected_counts[method][replicate] = selected_count
            locked_counts[method][replicate] = locked_count
            secondary_thresholds[method][replicate] = secondary
            method_rows[method] = {
                "selected_primary_threshold": selected,
                "selected_primary_alert_count": selected_count,
                "alert_count_at_frozen_threshold": locked_count,
                "selected_zero_alert_threshold": secondary,
                "selected_zero_alert_count": 0,
            }
        rows.append(
            {
                "replicate_index": replicate,
                "seed": seed,
                "methods": method_rows,
            }
        )
    summaries: dict[str, Any] = {}
    for method in METHOD_IDS:
        unique_selected, selected_frequency = np.unique(
            selected_counts[method],
            return_counts=True,
        )
        unique_locked, locked_frequency = np.unique(
            locked_counts[method],
            return_counts=True,
        )
        summaries[method] = {
            "frozen_threshold": locked_table[method]["threshold"],
            "selected_primary_threshold_percentiles": _percentile_summary(
                selected_thresholds[method]
            ),
            "selected_primary_alert_count_frequency": {
                str(int(value)): int(count)
                for value, count in zip(
                    unique_selected,
                    selected_frequency,
                    strict=True,
                )
            },
            "alert_count_at_frozen_threshold_percentiles": _percentile_summary(
                locked_counts[method]
            ),
            "alert_count_at_frozen_threshold_frequency": {
                str(int(value)): int(count)
                for value, count in zip(
                    unique_locked,
                    locked_frequency,
                    strict=True,
                )
            },
            "selected_zero_alert_threshold_percentiles": _percentile_summary(
                secondary_thresholds[method]
            ),
        }
    return {
        "schema_version": "run6-google-threshold-bootstrap-v1",
        "status": "descriptive_only_does_not_replace_frozen_threshold",
        "unit": "complete_paired_shot",
        "block_length_shots": block_length,
        "replicates": replicates,
        "seed_start": seed_start,
        "seed_stop_exclusive": seed_start + replicates,
        "rng": "numpy.random.Generator(PCG64)",
        "blocks_per_replicate": math.ceil(shot_count / block_length),
        "primary_maximum_alerts": 2,
        "secondary_maximum_alerts": 0,
        "summaries": summaries,
        "replicate_results": rows,
    }


def _fit_locked_hotelling(
    reference: np.ndarray,
    monitor: np.ndarray,
) -> RoleHotellingModel:
    """Fit the exact role-stratified M2 state included in the warm digest."""

    if reference.shape != monitor.shape or reference.shape != (
        5_000,
        ROLE_COUNT,
        CHECK_COUNT,
    ):
        raise ValueError("Locked M2 inputs must have shape (5000,51,24).")
    selection = select_role_fit_indices()
    selected = np.empty((len(selection), FEATURE_DIM), dtype=np.float64)
    role_means = np.empty((ROLE_COUNT, FEATURE_DIM), dtype=np.float64)
    cursor = 0
    for role in range(ROLE_COUNT):
        pair_indices = selection[selection[:, 1] == role, 0]
        role_values = np.empty((len(pair_indices), FEATURE_DIM), dtype=np.float64)
        for local_index, pair_index in enumerate(pair_indices):
            role_values[local_index] = paired_qec_contrasts(
                reference[pair_index, role],
                monitor[pair_index, role],
            ).feature_difference
        role_means[role] = np.mean(role_values, axis=0)
        next_cursor = cursor + len(pair_indices)
        selected[cursor:next_cursor] = role_values - role_means[role]
        cursor = next_cursor
    if cursor != 20_000:
        raise RuntimeError("Locked M2 observation accounting changed.")
    estimator = LedoitWolf(store_precision=True, assume_centered=True)
    estimator.fit(selected)
    precision = np.asarray(estimator.precision_, dtype=np.float64)
    return RoleHotellingModel(
        role_means=role_means,
        precision=0.5 * (precision + precision.T),
        selected_indices=selection,
    )


def fit_and_warm_locked_bank(
    reference: np.ndarray,
    monitor: np.ndarray,
) -> RoleIsolatedQECBank:
    """Fit M1/M2 and causally warm M3--M5 on the locked 5,000 pairs."""

    if reference.shape != monitor.shape or reference.shape != (
        5_000,
        ROLE_COUNT,
        CHECK_COUNT,
    ):
        raise ValueError("Warm-up arrays must have shape (5000,51,24).")
    bank = RoleIsolatedQECBank(
        role_count=ROLE_COUNT,
        diagonal_model=DiagonalLikelihoodModel.fit(reference, monitor),
        hotelling_model=_fit_locked_hotelling(reference, monitor),
    )
    for shot in range(5_000):
        for role in range(ROLE_COUNT):
            bank.update(role, reference[shot, role], monitor[shot, role])
    return bank


def _update_factors(update: Any) -> dict[str, np.ndarray]:
    if update.m1_factors is None:
        raise RuntimeError("The locked randomization replay requires fitted M1.")
    return {
        "m0": update.m0_factors,
        "m1": update.m1_factors,
        "m3": update.m3_factors,
        "m4": update.m4_factors,
        "m5": update.m5_factors,
        "space": update.space_factors,
    }


def run_randomization_replicate(
    warm_checkpoint: RoleIsolatedQECBank,
    reference: np.ndarray,
    monitor: np.ndarray,
    swaps: np.ndarray,
    *,
    seed: int,
    expected_checkpoint_sha256: str | None = None,
) -> dict[str, Any]:
    """Replay one stream with one formal e-update per complete paired shot.

    A shot has one shared random orientation.  Consequently its 51 roles are
    simultaneous experts, not 51 sequential exchangeability opportunities.
    Experts are indexed ``(role, base_component)`` with prior
    ``(1 / role_count) * base_component_prior``.
    """

    left, right = apply_complete_shot_swaps(reference, monitor, swaps)
    role_count = int(left.shape[1])
    if left.shape[2] != CHECK_COUNT or warm_checkpoint.role_count != role_count:
        raise ValueError("Randomization array dimensions do not match the checkpoint.")
    bank = warm_checkpoint.clone()
    initial_digest = bank.state_digest()
    if (
        expected_checkpoint_sha256 is not None
        and initial_digest != expected_checkpoint_sha256
    ):
        raise ValueError("Randomization replicate did not restore the checkpoint.")
    horizon_shots = left.shape[0]
    priors = exact_component_priors()
    processes = {
        method: ProperUniformStartEProcessBank(
            role_count * len(priors[method].weights),
            horizon=horizon_shots,
            alpha=0.01,
            component_weights=np.tile(
                priors[method].weights / role_count,
                role_count,
            ),
        )
        for method in EXACT_METHOD_IDS
    }
    maximum_log_e = {method: 0.0 for method in EXACT_METHOD_IDS}
    familywide_crossing = False
    for shot in range(left.shape[0]):
        shot_factors: dict[str, list[np.ndarray]] = {
            method: [] for method in EXACT_METHOD_IDS
        }
        for role in range(role_count):
            update = bank.update(role, left[shot, role], right[shot, role])
            for method, factors in _update_factors(update).items():
                shot_factors[method].append(factors)
        for method in EXACT_METHOD_IDS:
            factors = np.concatenate(shot_factors[method])
            process_update = processes[method].update(factors)
            maximum_log_e[method] = max(
                maximum_log_e[method],
                process_update.log_statistic,
            )
            if process_update.log_statistic >= _LOG_SIX_HUNDRED:
                familywide_crossing = True
    return {
        "replicate_index": int(seed - 610700),
        "seed": int(seed),
        "swap_sha256": hashlib.sha256(
            np.ascontiguousarray(swaps, dtype=np.uint8).tobytes()
        ).hexdigest(),
        "swapped_shot_count": int(np.sum(swaps, dtype=np.int64)),
        "checkpoint_restored": initial_digest
        == (expected_checkpoint_sha256 or initial_digest),
        "crossed_100": {
            method: processes[method].alarm_time is not None
            for method in EXACT_METHOD_IDS
        },
        "first_crossing_shot_number_one_based": {
            method: processes[method].alarm_time for method in EXACT_METHOD_IDS
        },
        "maximum_log_e": {
            method: float(maximum_log_e[method]) for method in EXACT_METHOD_IDS
        },
        "final_log_e": {
            method: float(processes[method].log_statistic)
            for method in EXACT_METHOD_IDS
        },
        "familywide_any_crossed_600": familywide_crossing,
        "formal_eprocess_updates": horizon_shots,
        "role_score_updates": horizon_shots * role_count,
        "formal_experts": {
            method: role_count * len(priors[method].weights)
            for method in EXACT_METHOD_IDS
        },
    }


def summarize_replicates(
    replicates: list[Mapping[str, Any]],
    *,
    checkpoint_sha256: str,
) -> dict[str, Any]:
    """Build the fixed primary and descriptive randomization summary."""

    if len(replicates) < 1:
        raise ValueError("At least one randomization replicate is required.")
    counts = {
        method: sum(bool(row["crossed_100"][method]) for row in replicates)
        for method in EXACT_METHOD_IDS
    }
    primary_count = counts["space"]
    interval = clopper_pearson_interval(primary_count, len(replicates))
    return {
        "schema_version": "run6-google-randomization-result-v1",
        "primary_method": "space",
        "primary_statistic": "ever_proper_prior_eprocess_ge_100",
        "replicate_count": len(replicates),
        "seed_start": 610700,
        "seed_stop_exclusive": 610700 + len(replicates),
        "rng": "numpy.random.Generator(PCG64)",
        "one_orientation_draw_per_replicate": True,
        "complete_shot_swap_shared_across_roles": True,
        "horizon_paired_shots": int(replicates[0]["formal_eprocess_updates"]),
        "role_score_updates_per_replicate": int(replicates[0]["role_score_updates"]),
        "formal_expert_index": "role_then_locked_base_component",
        "formal_role_prior": "uniform_1_over_role_count",
        "warm_checkpoint_sha256": checkpoint_sha256,
        "crossing_counts_at_100": counts,
        "space_crossing_fraction": primary_count / len(replicates),
        "space_crossing_clopper_pearson_95": {
            "lower": interval[0],
            "upper": interval[1],
        },
        "familywide_any_crossing_count_at_600": sum(
            bool(row["familywide_any_crossed_600"]) for row in replicates
        ),
        "interpretation": "exact_design_based_implementation_diagnostic",
        "replicates": replicates,
    }


def synthetic_dry_run() -> dict[str, Any]:
    """Exercise swap/reset/e-process logic using constructed binary fixtures."""

    generator = np.random.Generator(np.random.PCG64(610699))
    warm_left = generator.integers(0, 2, size=(3, 2, CHECK_COUNT), dtype=np.uint8)
    warm_right = generator.integers(0, 2, size=(3, 2, CHECK_COUNT), dtype=np.uint8)
    bank = RoleIsolatedQECBank(
        role_count=2,
        diagonal_model=DiagonalLikelihoodModel.fit(warm_left, warm_right),
    )
    for shot in range(3):
        for role in range(2):
            bank.update(role, warm_left[shot, role], warm_right[shot, role])
    checkpoint = bank.state_digest()
    left = generator.integers(0, 2, size=(4, 2, CHECK_COUNT), dtype=np.uint8)
    right = generator.integers(0, 2, size=(4, 2, CHECK_COUNT), dtype=np.uint8)
    swaps = draw_complete_shot_swaps(610700, shot_count=4)
    first = run_randomization_replicate(
        bank,
        left,
        right,
        swaps,
        seed=610700,
        expected_checkpoint_sha256=checkpoint,
    )
    second = run_randomization_replicate(
        bank,
        left,
        right,
        swaps,
        seed=610700,
        expected_checkpoint_sha256=checkpoint,
    )
    if canonical_json_bytes(first) != canonical_json_bytes(second):
        raise RuntimeError("Synthetic randomization replay is not deterministic.")
    threshold_scores = {
        method: generator.random((12, 2), dtype=np.float64) for method in METHOD_IDS
    }
    locked_table = {
        method: {
            "threshold": float(np.sort(np.max(values, axis=1))[::-1][2]),
            "validation_alert_count": 2,
            "max_validation_alerts": 2,
            "secondary_zero_alert_threshold": float(np.max(values)),
            "secondary_validation_alert_count": 0,
        }
        for method, values in threshold_scores.items()
    }
    threshold_result = threshold_bootstrap(
        threshold_scores,
        locked_table,
        replicates=8,
        seed_start=613000,
        block_length=4,
    )
    return {
        "status": "synthetic_dry_run_passed",
        "raw_run6_values_opened": False,
        "checkpoint_sha256": checkpoint,
        "replicate": first,
        "threshold_bootstrap": threshold_result,
    }


def run_shard_real(args: argparse.Namespace) -> None:
    started_unix = time.time()
    required = {
        "--config": args.config,
        "--freeze-ratification": args.freeze_ratification,
        "--repair-ratification": args.repair_ratification,
        "--detector-manifest": args.detector_manifest,
        "--data-root": args.data_root,
        "--output": args.output,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError(f"Missing randomization-shard arguments: {missing}")
    if (
        args.replicate_start is None
        or args.replicate_stop is None
        or not 0 <= args.replicate_start < args.replicate_stop <= 256
    ):
        raise ValueError(
            "Shard mode requires 0 <= --replicate-start < --replicate-stop <= 256."
        )

    repo_root = Path(__file__).resolve().parents[3]
    config_path = args.config.resolve()
    ratification_path = args.freeze_ratification.resolve()
    repair_ratification_path = args.repair_ratification.resolve()
    detector_manifest_path = args.detector_manifest.resolve()
    data_root = args.data_root.resolve()
    output = args.output.resolve()
    config = load_google_lock(config_path)
    verify_freeze_ratification(
        ratification_path,
        repair_ratification_path=repair_ratification_path,
        repo_root=repo_root,
        config_path=config_path,
        config=config,
        detector_manifest_path=detector_manifest_path,
    )
    detector_manifest, detector_artifacts = verify_detector_manifest(
        detector_manifest_path,
        config_path=config_path,
        config=config,
        ratification_path=ratification_path,
        repo_root=repo_root,
    )
    expected_threads = config["numeric_policy"]["thread_environment"]
    require_thread_environment(expected_threads)
    circuit = data_root / "circuit_ideal.stim"
    detection_events = data_root / config["source"]["detection_event_file"]
    assert_no_outcome_paths([circuit, detection_events, output])
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Randomization output directory is not empty.")
    output.mkdir(parents=True, exist_ok=True)

    threshold_scores = load_locked_threshold_scores(detector_artifacts)
    validate_locked_threshold_frontiers(
        detector_artifacts,
        threshold_scores,
        protocol_id=config["protocol_id"],
        expected_common_hashes={
            "config_sha256": detector_manifest["config_sha256"],
            "method_spec_sha256": detector_manifest["method_spec_sha256"],
            "detector_script_sha256": detector_manifest["detector_script_sha256"],
            "warm_checkpoint_sha256": detector_manifest["warm_checkpoint_sha256"],
            "threshold_final_checkpoint_sha256": detector_manifest[
                "threshold_checkpoint_sha256"
            ],
            "freeze_ratification_sha256": detector_manifest[
                "freeze_ratification_sha256"
            ],
            "deviation_ledger_sha256": detector_manifest["deviation_ledger"]["sha256"],
            "python_environment_lock_sha256": sha256_file(
                repo_root / "experiments/run6/configs/python_environment_lock.txt"
            ),
            "freeze_manifest_sha256": sha256_file(
                repo_root / "experiments/run6/freeze_manifest.json"
            ),
        },
    )
    locked_thresholds = load_locked_threshold_table(
        detector_artifacts["thresholds.json"]
    )
    validate_formal_detector_artifacts(
        detector_artifacts,
        locked_thresholds,
        protocol_id=config["protocol_id"],
    )
    validate_detector_event_summaries(
        detector_artifacts,
        locked_thresholds,
        windows=config["event_windows"],
    )
    validate_frozen_shot_table(
        detector_artifacts["threshold_shots.csv"],
        threshold_scores,
        locked_thresholds,
        phase="threshold",
        pair_index_start=5_000,
        reference_start=5_000,
        monitor_start=15_000,
        windows=None,
    )

    archive = (
        repo_root
        / "experiments/data/run6/google_2022/google_qec3v5_experiment_data.zip"
    )
    if (
        archive.stat().st_size != config["source"]["archive_bytes"]
        or sha256_file(archive) != config["source"]["sha256"]
    ):
        raise ValueError("Google source archive differs from the lock.")
    archive_member_root = config["source"]["experiment"]
    expected_member_hashes = {
        filename: _sha256_zip_member(
            archive,
            f"{archive_member_root}/{filename}",
        )
        for filename in (
            "circuit_ideal.stim",
            config["source"]["detection_event_file"],
        )
    }
    if expected_member_hashes != detector_manifest["verified_zip_member_sha256"]:
        raise ValueError("Detector manifest ZIP-member hashes changed.")
    for filename, expected_hash in expected_member_hashes.items():
        if sha256_file(data_root / filename) != expected_hash:
            raise ValueError(
                f"Extracted {filename} differs from the verified ZIP member."
            )
    if sha256_file(circuit) != detector_manifest["circuit_sha256"]:
        raise ValueError("Circuit differs from the detector replay.")
    layout = parse_stim_detector_layout(
        circuit,
        expected_roles=ROLE_COUNT,
        expected_checks_per_role=CHECK_COUNT,
    )
    layout_hash = hashlib.sha256(
        np.ascontiguousarray(
            layout.ordered_declaration_indices,
            dtype="<i8",
        ).tobytes()
    ).hexdigest()
    if layout_hash != detector_manifest["detector_layout_index_sha256"]:
        raise ValueError("Detector coordinate permutation changed.")

    # First detector-value access occurs only after all gates above.
    validation_a = read_b8_detector_shots(
        detection_events,
        layout,
        start=0,
        stop=10_000,
        total_shots=500_000,
    )
    validation_b = read_b8_detector_shots(
        detection_events,
        layout,
        start=10_000,
        stop=20_000,
        total_shots=500_000,
    )
    warm = fit_and_warm_locked_bank(validation_a[:5_000], validation_b[:5_000])
    checkpoint = warm.state_digest()
    if checkpoint != detector_manifest["warm_checkpoint_sha256"]:
        raise ValueError("Reconstructed warm checkpoint differs from detector replay.")

    reference = validation_a[5_000:10_000]
    monitor = validation_b[5_000:10_000]
    replicates: list[dict[str, Any]] = []
    for replicate_index in range(args.replicate_start, args.replicate_stop):
        seed = 610700 + replicate_index
        swaps = draw_complete_shot_swaps(seed, shot_count=5_000)
        row = run_randomization_replicate(
            warm,
            reference,
            monitor,
            swaps,
            seed=seed,
            expected_checkpoint_sha256=checkpoint,
        )
        if (
            row["replicate_index"] != replicate_index
            or row["formal_eprocess_updates"] != 5_000
            or row["role_score_updates"] != 255_000
        ):
            raise RuntimeError("Randomization replicate accounting changed.")
        replicates.append(row)

    shard = {
        "schema_version": "run6-google-randomization-shard-v1",
        "protocol_id": config["protocol_id"],
        "replicate_start": args.replicate_start,
        "replicate_stop_exclusive": args.replicate_stop,
        "seed_start": 610700 + args.replicate_start,
        "seed_stop_exclusive": 610700 + args.replicate_stop,
        "horizon_paired_shots": 5_000,
        "role_score_updates_per_replicate": 255_000,
        "warm_checkpoint_sha256": checkpoint,
        "replicates": replicates,
    }
    shard_path = output / "randomization_shard.json"
    _write_canonical_json(shard_path, shard)
    manifest = {
        "schema_version": "run6-google-randomization-shard-manifest-v1",
        "protocol_id": config["protocol_id"],
        "config_sha256": sha256_file(config_path),
        "method_spec_sha256": sha256_file(
            repo_root / config["normative_method_spec"]["path"]
        ),
        "freeze_ratification_sha256": sha256_file(ratification_path),
        "repair_ratification_path": REPAIR_RATIFICATION_RELATIVE,
        "repair_ratification_sha256": sha256_file(repair_ratification_path),
        "detector_manifest_sha256": sha256_file(detector_manifest_path),
        "detector_manifest_git_commit": detector_manifest["git_commit"],
        "script_sha256": sha256_file(__file__),
        "git_commit": _git_commit(repo_root),
        "outcome_accessed": False,
        "source_archive_sha256": config["source"]["sha256"],
        "verified_zip_member_sha256": expected_member_hashes,
        "warm_checkpoint_sha256": checkpoint,
        "rng": {
            "algorithm": "numpy.random.Generator(PCG64)",
            "randomization_seed_start": 610700,
            "randomization_seed_stop_exclusive": 610956,
        },
        "environment": environment_fingerprint(),
        "command": sys.argv,
        "artifact": _artifact_record(shard_path),
        "resources": _randomization_resources(
            replicates,
            started_unix=started_unix,
            output=output,
        ),
    }
    manifest_path = output / "randomization_shard_manifest.json"
    _write_self_accounting_manifest(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


def run_merge_real(args: argparse.Namespace) -> None:
    """Merge gap-free shards and run the one deterministic threshold bootstrap."""

    started_unix = time.time()
    required = {
        "--config": args.config,
        "--freeze-ratification": args.freeze_ratification,
        "--repair-ratification": args.repair_ratification,
        "--detector-manifest": args.detector_manifest,
        "--output": args.output,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError(f"Missing randomization-merge arguments: {missing}")
    if not args.merge_shard_manifest:
        raise ValueError("Merge mode requires at least one --merge-shard-manifest.")
    if args.replicate_start is not None or args.replicate_stop is not None:
        raise ValueError("Merge mode does not accept replicate range arguments.")

    repo_root = Path(__file__).resolve().parents[3]
    config_path = args.config.resolve()
    ratification_path = args.freeze_ratification.resolve()
    repair_ratification_path = args.repair_ratification.resolve()
    detector_manifest_path = args.detector_manifest.resolve()
    output = args.output.resolve()
    shard_paths = [path.resolve() for path in args.merge_shard_manifest]
    if len(set(shard_paths)) != len(shard_paths):
        raise ValueError("Duplicate shard manifest path.")
    assert_no_outcome_paths([*shard_paths, output])
    config = load_google_lock(config_path)
    verify_freeze_ratification(
        ratification_path,
        repair_ratification_path=repair_ratification_path,
        repo_root=repo_root,
        config_path=config_path,
        config=config,
        detector_manifest_path=detector_manifest_path,
    )
    detector_manifest, detector_artifacts = verify_detector_manifest(
        detector_manifest_path,
        config_path=config_path,
        config=config,
        ratification_path=ratification_path,
        repo_root=repo_root,
    )
    require_thread_environment(config["numeric_policy"]["thread_environment"])
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Randomization merge output directory is not empty.")
    output.mkdir(parents=True, exist_ok=True)

    threshold_scores = load_locked_threshold_scores(detector_artifacts)
    validate_locked_threshold_frontiers(
        detector_artifacts,
        threshold_scores,
        protocol_id=config["protocol_id"],
        expected_common_hashes={
            "config_sha256": detector_manifest["config_sha256"],
            "method_spec_sha256": detector_manifest["method_spec_sha256"],
            "detector_script_sha256": detector_manifest["detector_script_sha256"],
            "warm_checkpoint_sha256": detector_manifest["warm_checkpoint_sha256"],
            "threshold_final_checkpoint_sha256": detector_manifest[
                "threshold_checkpoint_sha256"
            ],
            "freeze_ratification_sha256": detector_manifest[
                "freeze_ratification_sha256"
            ],
            "deviation_ledger_sha256": detector_manifest["deviation_ledger"]["sha256"],
            "python_environment_lock_sha256": sha256_file(
                repo_root / "experiments/run6/configs/python_environment_lock.txt"
            ),
            "freeze_manifest_sha256": sha256_file(
                repo_root / "experiments/run6/freeze_manifest.json"
            ),
        },
    )
    locked_thresholds = load_locked_threshold_table(
        detector_artifacts["thresholds.json"]
    )
    validate_formal_detector_artifacts(
        detector_artifacts,
        locked_thresholds,
        protocol_id=config["protocol_id"],
    )
    validate_detector_event_summaries(
        detector_artifacts,
        locked_thresholds,
        windows=config["event_windows"],
    )
    validate_frozen_shot_table(
        detector_artifacts["threshold_shots.csv"],
        threshold_scores,
        locked_thresholds,
        phase="threshold",
        pair_index_start=5_000,
        reference_start=5_000,
        monitor_start=15_000,
        windows=None,
    )

    shard_rows: list[list[Mapping[str, Any]]] = []
    shard_evidence: list[dict[str, Any]] = []
    checkpoint_hashes: set[str] = set()
    for shard_path in shard_paths:
        shard_manifest, shard = verify_randomization_shard_manifest(
            shard_path,
            repo_root=repo_root,
            config_path=config_path,
            config=config,
            ratification_path=ratification_path,
            repair_ratification_path=repair_ratification_path,
            detector_manifest_path=detector_manifest_path,
            detector_manifest=detector_manifest,
        )
        checkpoint_hashes.add(shard["warm_checkpoint_sha256"])
        shard_rows.append(shard["replicates"])
        shard_evidence.append(
            {
                "replicate_start": shard["replicate_start"],
                "replicate_stop_exclusive": shard["replicate_stop_exclusive"],
                "manifest_sha256": sha256_file(shard_path),
                "resources": shard_manifest["resources"],
            }
        )
    if checkpoint_hashes != {detector_manifest["warm_checkpoint_sha256"]}:
        raise ValueError("Randomization shards do not share the frozen checkpoint.")
    replicates = merge_randomization_rows(
        shard_rows,
        expected_replicates=256,
        horizon_shots=5_000,
        role_score_updates=255_000,
    )

    threshold_result = threshold_bootstrap(
        threshold_scores,
        locked_thresholds,
        replicates=2_000,
        seed_start=613_000,
        block_length=128,
    )
    threshold_result_path = output / "threshold_bootstrap.json"
    _write_canonical_json(threshold_result_path, threshold_result)
    result = summarize_replicates(
        replicates,
        checkpoint_sha256=detector_manifest["warm_checkpoint_sha256"],
    )
    result_path = output / "randomization_result.json"
    _write_canonical_json(result_path, result)

    indices = np.arange(256, dtype="<i8")
    seeds = np.arange(610700, 610956, dtype="<i8")
    merge_evidence = {
        "input_shard_count": len(shard_evidence),
        "input_shards": sorted(
            shard_evidence,
            key=lambda row: (
                row["replicate_start"],
                row["replicate_stop_exclusive"],
                row["manifest_sha256"],
            ),
        ),
        "replicate_index_range": [0, 256],
        "seed_range": [610700, 610956],
        "replicate_indices_sha256": hashlib.sha256(indices.tobytes()).hexdigest(),
        "seeds_sha256": hashlib.sha256(seeds.tobytes()).hexdigest(),
        "every_replicate_index_exactly_once": True,
        "every_seed_exactly_once": True,
        "shared_warm_checkpoint_sha256": detector_manifest["warm_checkpoint_sha256"],
        "canonical_result_sha256": sha256_file(result_path),
        "canonical_result_independent_of_shard_layout": True,
    }
    manifest = {
        "schema_version": "run6-google-randomization-manifest-v1",
        "protocol_id": config["protocol_id"],
        "config_sha256": sha256_file(config_path),
        "method_spec_sha256": sha256_file(
            repo_root / config["normative_method_spec"]["path"]
        ),
        "freeze_ratification_sha256": sha256_file(ratification_path),
        "repair_ratification_path": REPAIR_RATIFICATION_RELATIVE,
        "repair_ratification_sha256": sha256_file(repair_ratification_path),
        "detector_manifest_sha256": sha256_file(detector_manifest_path),
        "detector_manifest_git_commit": detector_manifest["git_commit"],
        "script_sha256": sha256_file(__file__),
        "git_commit": _git_commit(repo_root),
        "outcome_accessed": False,
        "source_archive_sha256": config["source"]["sha256"],
        "verified_zip_member_sha256": detector_manifest["verified_zip_member_sha256"],
        "warm_checkpoint_sha256": detector_manifest["warm_checkpoint_sha256"],
        "rng": {
            "algorithm": "numpy.random.Generator(PCG64)",
            "randomization_seed_start": 610700,
            "randomization_seed_stop_exclusive": 610956,
            "threshold_bootstrap_seed_start": 613000,
            "threshold_bootstrap_seed_stop_exclusive": 615000,
        },
        "execution_mode": "deterministic_gap_free_shard_merge",
        "merge_evidence": merge_evidence,
        "environment": environment_fingerprint(),
        "command": sys.argv,
        "artifacts": [
            _artifact_record(result_path),
            _artifact_record(threshold_result_path),
        ],
        "resources": _randomization_resources(
            replicates,
            started_unix=started_unix,
            output=output,
        ),
    }
    manifest_path = output / "randomization_manifest.json"
    _write_self_accounting_manifest(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


def main() -> None:
    args = parse_args()
    if args.dry_run:
        print(json.dumps(synthetic_dry_run(), indent=2, sort_keys=True))
        return
    if args.merge_shard_manifest:
        run_merge_real(args)
    else:
        run_shard_real(args)


if __name__ == "__main__":
    main()
