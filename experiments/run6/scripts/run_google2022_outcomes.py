#!/usr/bin/env python3
"""Locked post-freeze decoder-outcome audit for Google 2022 Run 6.

The three ``.01`` files are opened only after the freeze ratification,
detector-only manifest, and every frozen detector artifact have been
hash-checked.  Detector scores are never refit or selected using outcomes.
The primary label is actual XOR correlated-matching prediction; PyMatching is
the predeclared secondary replication label.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
from aoc.run6_protocol import (
    RUN6_REQUIRED_FREEZE_PATHS,
    canonical_json_bytes,
    environment_fingerprint,
    load_google_lock,
    load_strict_json,
    require_exact_keys,
    require_thread_environment,
    sha256_file,
    verify_committed_freeze_chain,
)
from aoc.space_qec import (
    apply_strict_shot_threshold,
    exact_component_priors,
    select_strict_shot_threshold,
)
from scipy.stats import beta

METHOD_IDS = ("m0", "m0c", "m1", "m2", "m3", "m4", "m5", "space")
EXACT_METHOD_IDS = ("m0", "m1", "m3", "m4", "m5", "space")
NONFORMAL_METHOD_IDS = ("m0c", "m2")
RISK_BUDGETS = (2, 20, 200)
LABEL_IDS = ("correlated_matching_mismatch", "pymatching_mismatch")
METRIC_IDS = (
    "captured_mismatches",
    "mismatch_recall",
    "alert_precision",
    "retained_mismatch_rate",
)
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
RANDOMIZATION_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "protocol_id",
        "config_sha256",
        "method_spec_sha256",
        "freeze_ratification_sha256",
        "detector_manifest_sha256",
        "detector_manifest_git_commit",
        "script_sha256",
        "git_commit",
        "outcome_accessed",
        "source_archive_sha256",
        "verified_zip_member_sha256",
        "warm_checkpoint_sha256",
        "rng",
        "execution_mode",
        "merge_evidence",
        "environment",
        "command",
        "artifacts",
        "resources",
    }
)
RANDOMIZATION_RESULT_KEYS = frozenset(
    {
        "schema_version",
        "primary_method",
        "primary_statistic",
        "replicate_count",
        "seed_start",
        "seed_stop_exclusive",
        "rng",
        "one_orientation_draw_per_replicate",
        "complete_shot_swap_shared_across_roles",
        "horizon_paired_shots",
        "role_score_updates_per_replicate",
        "formal_expert_index",
        "formal_role_prior",
        "warm_checkpoint_sha256",
        "crossing_counts_at_100",
        "space_crossing_fraction",
        "space_crossing_clopper_pearson_95",
        "familywide_any_crossing_count_at_600",
        "interpretation",
        "replicates",
    }
)
PNNL_RESULT_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "protocol_id",
        "claim_label",
        "formal_alarm_unit",
        "within_shot_roles",
        "git_commit",
        "config_sha256",
        "pittsburgh_manifest_sha256",
        "freeze_ratification_sha256",
        "package_lock_sha256",
        "first_unblinding_record",
        "metadata_validation",
        "held_payload_sha256",
        "state_rows",
        "aggregate_results",
        "randomization_audit",
        "randomization_alarm_counts",
        "randomization_maximum_log_e",
        "trace_artifacts",
        "bootstrap_artifacts",
        "resource_ledger",
        "retention_pass",
        "environment",
        "command",
        "started_unix",
        "held_value_processing_started_unix",
        "finished_unix",
    }
)
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
OUTCOME_TABLE_FIELDS = (
    "monitor_archive_shot",
    "actual_observable_flip",
    "correlated_matching_prediction",
    "correlated_matching_mismatch",
    "pymatching_prediction",
    "pymatching_mismatch",
    "detector_manifest_sha256",
)
PNNL_METHOD_IDS = (
    "dfr",
    "online_logistic",
    "space_sparse",
    "space_spectral",
    "space_composite",
)
PNNL_STATE_FIELDS = (
    "cohort_id",
    "logical_state",
    "method",
    "threshold_seed",
    "threshold_log_e",
    "first_alarm_update",
    "pre_false_alarm",
    "miss",
    "post_alarm_shot",
    "post_alarm_role",
    "restricted_post_delay_fraction",
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
    parser.add_argument("--detector-manifest", type=Path)
    parser.add_argument("--randomization-manifest", type=Path)
    parser.add_argument("--pnnl-results-manifest", type=Path)
    parser.add_argument("--outcome-root", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def require_paired_final_inputs(
    randomization_manifest: Path | None,
    pnnl_results_manifest: Path | None,
) -> bool:
    """Require both independent-arm manifests together or neither."""

    present = (
        randomization_manifest is not None,
        pnnl_results_manifest is not None,
    )
    if present[0] != present[1]:
        raise ValueError(
            "Final aggregation requires both --randomization-manifest and "
            "--pnnl-results-manifest, or neither."
        )
    return all(present)


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


def verify_outcome_zip_members(
    archive: Path,
    outcome_paths: Mapping[str, Path],
    *,
    archive_member_root: str,
) -> dict[str, str]:
    """Bind each extracted outcome to one exact verified archive member."""

    result: dict[str, str] = {}
    for name, path in outcome_paths.items():
        expected = _sha256_zip_member(
            archive,
            f"{archive_member_root}/{name}",
        )
        if sha256_file(path) != expected:
            raise ValueError(
                f"Extracted outcome {name} differs from the verified ZIP member."
            )
        result[name] = expected
    return result


def verify_freeze_ratification(
    path: Path,
    *,
    repo_root: Path,
    config_path: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify the canonical, committed, pushed freeze chain before outcomes."""

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
    return verify_committed_freeze_chain(
        path,
        repo_root=repo_root,
        required_paths=RUN6_REQUIRED_FREEZE_PATHS,
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
    """Validate the detector freeze and return its hash-checked artifacts."""

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
    artifacts: dict[str, Path] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise TypeError("Each detector artifact record must be an object.")
        artifact = _resolve_artifact(record, path)
        if artifact.name in artifacts:
            raise ValueError(f"Duplicate detector artifact basename: {artifact.name}")
        artifacts[artifact.name] = artifact
    expected_artifacts = expected_detector_artifact_names()
    if set(artifacts) != expected_artifacts:
        raise ValueError(
            "Detector artifact contract mismatch; "
            f"missing={sorted(expected_artifacts - artifacts.keys())}, "
            f"unknown={sorted(artifacts.keys() - expected_artifacts)}."
        )
    if manifest["threshold_table_sha256"] != sha256_file(artifacts["thresholds.json"]):
        raise ValueError("Detector threshold-table binding changed.")
    _validate_threshold_stage(artifacts, manifest, config)
    return manifest, artifacts


def _verify_artifact_records_recursive(
    value: Any,
    *,
    manifest_path: Path,
    observed: dict[str, Path],
) -> None:
    if isinstance(value, Mapping):
        if frozenset(value) == ARTIFACT_KEYS:
            artifact = _resolve_artifact(value, manifest_path)
            relative = artifact.relative_to(manifest_path.parent.resolve()).as_posix()
            if relative in observed:
                raise ValueError(f"Duplicate embedded artifact record: {relative}")
            observed[relative] = artifact
            return
        for child in value.values():
            _verify_artifact_records_recursive(
                child,
                manifest_path=manifest_path,
                observed=observed,
            )
    elif isinstance(value, list):
        for child in value:
            _verify_artifact_records_recursive(
                child,
                manifest_path=manifest_path,
                observed=observed,
            )


def clopper_pearson_interval(
    successes: int,
    trials: int,
    *,
    confidence: float = 0.95,
) -> tuple[float, float]:
    if trials < 1 or not 0 <= successes <= trials:
        raise ValueError("Require trials >= 1 and 0 <= successes <= trials.")
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


def verify_randomization_result(
    path: Path,
    *,
    repo_root: Path,
    config_path: Path,
    config: Mapping[str, Any],
    ratification_path: Path,
    detector_manifest_path: Path,
    detector_manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate the separate Google randomization command and all artifacts."""

    manifest = load_strict_json(path)
    require_exact_keys(
        manifest,
        RANDOMIZATION_MANIFEST_KEYS,
        context="Google randomization manifest",
    )
    expected = {
        "schema_version": "run6-google-randomization-manifest-v1",
        "protocol_id": config["protocol_id"],
        "config_sha256": sha256_file(config_path),
        "method_spec_sha256": sha256_file(
            repo_root / config["normative_method_spec"]["path"]
        ),
        "freeze_ratification_sha256": sha256_file(ratification_path),
        "detector_manifest_sha256": sha256_file(detector_manifest_path),
        "detector_manifest_git_commit": detector_manifest["git_commit"],
        "script_sha256": sha256_file(
            repo_root / "experiments/run6/scripts/run_google2022_randomization.py"
        ),
        "outcome_accessed": False,
        "source_archive_sha256": config["source"]["sha256"],
        "verified_zip_member_sha256": detector_manifest["verified_zip_member_sha256"],
        "warm_checkpoint_sha256": detector_manifest["warm_checkpoint_sha256"],
    }
    for key, expected_value in expected.items():
        if manifest[key] != expected_value:
            raise ValueError(f"Randomization manifest binding changed: {key}")
    if manifest["environment"] != environment_fingerprint():
        raise ValueError("Randomization environment differs from the current one.")
    if not isinstance(manifest["git_commit"], str):
        raise TypeError("Randomization git_commit must be a string.")
    _require_ancestor(repo_root, manifest["git_commit"], _git_commit(repo_root))
    if manifest["rng"] != {
        "algorithm": "numpy.random.Generator(PCG64)",
        "randomization_seed_start": 610700,
        "randomization_seed_stop_exclusive": 610956,
        "threshold_bootstrap_seed_start": 613000,
        "threshold_bootstrap_seed_stop_exclusive": 615000,
    }:
        raise ValueError("Randomization manifest seed registry changed.")
    if manifest["execution_mode"] != "deterministic_gap_free_shard_merge":
        raise ValueError("Randomization execution mode changed.")

    records = manifest["artifacts"]
    if not isinstance(records, list):
        raise TypeError("Randomization artifacts must be a list.")
    artifacts: dict[str, Path] = {}
    _verify_artifact_records_recursive(
        records,
        manifest_path=path,
        observed=artifacts,
    )
    by_name = {artifact.name: artifact for artifact in artifacts.values()}
    if set(by_name) != {"randomization_result.json", "threshold_bootstrap.json"}:
        raise ValueError("Randomization artifact set differs from the lock.")
    result = load_strict_json(by_name["randomization_result.json"])
    merge = manifest["merge_evidence"]
    if not isinstance(merge, Mapping):
        raise TypeError("Randomization merge evidence must be an object.")
    require_exact_keys(
        merge,
        {
            "input_shard_count",
            "input_shards",
            "replicate_index_range",
            "seed_range",
            "replicate_indices_sha256",
            "seeds_sha256",
            "every_replicate_index_exactly_once",
            "every_seed_exactly_once",
            "shared_warm_checkpoint_sha256",
            "canonical_result_sha256",
            "canonical_result_independent_of_shard_layout",
        },
        context="randomization merge evidence",
    )
    expected_indices = np.arange(256, dtype="<i8")
    expected_seeds = np.arange(610700, 610956, dtype="<i8")
    shard_rows = merge["input_shards"]
    if (
        isinstance(merge["input_shard_count"], bool)
        or not isinstance(merge["input_shard_count"], int)
        or not isinstance(shard_rows, list)
        or merge["input_shard_count"] != len(shard_rows)
        or len(shard_rows) < 1
        or merge["replicate_index_range"] != [0, 256]
        or merge["seed_range"] != [610700, 610956]
        or merge["replicate_indices_sha256"]
        != hashlib.sha256(expected_indices.tobytes()).hexdigest()
        or merge["seeds_sha256"] != hashlib.sha256(expected_seeds.tobytes()).hexdigest()
        or merge["every_replicate_index_exactly_once"] is not True
        or merge["every_seed_exactly_once"] is not True
        or merge["shared_warm_checkpoint_sha256"]
        != detector_manifest["warm_checkpoint_sha256"]
        or merge["canonical_result_sha256"]
        != sha256_file(by_name["randomization_result.json"])
        or merge["canonical_result_independent_of_shard_layout"] is not True
    ):
        raise ValueError("Randomization merge coverage evidence changed.")
    cursor = 0
    for shard_index, shard_row in enumerate(shard_rows):
        if not isinstance(shard_row, Mapping):
            raise TypeError("Randomization merge shard row must be an object.")
        require_exact_keys(
            shard_row,
            {
                "replicate_start",
                "replicate_stop_exclusive",
                "manifest_sha256",
                "resources",
            },
            context=f"randomization merge shard {shard_index}",
        )
        start = shard_row["replicate_start"]
        stop = shard_row["replicate_stop_exclusive"]
        if (
            start != cursor
            or isinstance(stop, bool)
            or not isinstance(stop, int)
            or not start < stop <= 256
            or not isinstance(shard_row["manifest_sha256"], str)
            or len(shard_row["manifest_sha256"]) != 64
        ):
            raise ValueError("Randomization merge shard ranges are not gap-free.")
        shard_resources = shard_row["resources"]
        if isinstance(shard_resources, Mapping):
            require_exact_keys(
                shard_resources,
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
                context=f"randomization merge shard {shard_index}.resources",
            )
        if (
            not isinstance(shard_resources, Mapping)
            or shard_resources.get("replicate_count") != stop - start
            or shard_resources.get("formal_eprocess_shot_updates")
            != (stop - start) * 5_000
            or shard_resources.get("role_score_updates") != (stop - start) * 255_000
            or shard_resources.get("worker_process_count") != 1
            or shard_resources.get("external_concurrency_not_inferred") is not True
            or any(
                isinstance(shard_resources.get(key), bool)
                or not isinstance(shard_resources.get(key), (int, float))
                or not np.isfinite(float(shard_resources[key]))
                or float(shard_resources[key]) < 0.0
                for key in (
                    "wall_seconds",
                    "peak_rss_kib_linux_ru_maxrss",
                    "output_bytes_excluding_manifest",
                    "output_bytes_including_manifest",
                )
            )
            or shard_resources.get("output_bytes_including_manifest", 0)
            < shard_resources.get("output_bytes_excluding_manifest", 0)
        ):
            raise ValueError("Randomization shard resource evidence changed.")
        cursor = stop
    if cursor != 256:
        raise ValueError("Randomization merge shard ranges ended early.")

    resources = manifest["resources"]
    if not isinstance(resources, Mapping):
        raise TypeError("Randomization resources must be an object.")
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
        context="randomization resources",
    )
    expected_output_bytes = sum(path.stat().st_size for path in by_name.values())
    if (
        resources["replicate_count"] != 256
        or resources["formal_eprocess_shot_updates"] != 256 * 5_000
        or resources["role_score_updates"] != 256 * 255_000
        or resources["worker_process_count"] != 1
        or resources["external_concurrency_not_inferred"] is not True
        or resources["output_bytes_excluding_manifest"] != expected_output_bytes
        or resources["output_bytes_including_manifest"]
        != expected_output_bytes + path.stat().st_size
    ):
        raise ValueError("Randomization resource accounting changed.")
    for key in ("wall_seconds", "peak_rss_kib_linux_ru_maxrss"):
        value = resources[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise ValueError(f"Randomization resource changed: {key}")
    require_exact_keys(
        result,
        RANDOMIZATION_RESULT_KEYS,
        context="Google randomization result",
    )
    if (
        result["schema_version"] != "run6-google-randomization-result-v1"
        or result["primary_method"] != "space"
        or result["replicate_count"] != 256
        or result["seed_start"] != 610700
        or result["seed_stop_exclusive"] != 610956
        or result["horizon_paired_shots"] != 5_000
        or result["role_score_updates_per_replicate"] != 255_000
        or result["warm_checkpoint_sha256"]
        != detector_manifest["warm_checkpoint_sha256"]
    ):
        raise ValueError("Randomization result dimensions or primary changed.")
    rows = result["replicates"]
    if not isinstance(rows, list) or len(rows) != 256:
        raise ValueError("Randomization result must contain all 256 replicates.")
    crossing_count = 0
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise TypeError("Randomization replicate must be an object.")
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
            context=f"randomization replicate {index}",
        )
        if (
            row["replicate_index"] != index
            or row["seed"] != 610700 + index
            or row["checkpoint_restored"] is not True
            or row["formal_eprocess_updates"] != 5_000
            or row["role_score_updates"] != 255_000
            or row["formal_experts"]
            != {
                "m0": 408,
                "m1": 408,
                "m3": 612,
                "m4": 3264,
                "m5": 1224,
                "space": 4488,
            }
        ):
            raise ValueError(f"Randomization replicate accounting changed: {index}")
        for key in (
            "crossed_100",
            "first_crossing_shot_number_one_based",
            "maximum_log_e",
            "final_log_e",
        ):
            if not isinstance(row[key], Mapping):
                raise TypeError(f"Randomization replicate {key} must be an object.")
            require_exact_keys(
                row[key],
                ("m0", "m1", "m3", "m4", "m5", "space"),
                context=f"randomization replicate {index}.{key}",
            )
        crossing_count += int(bool(row["crossed_100"]["space"]))
    if (
        result["crossing_counts_at_100"]["space"] != crossing_count
        or result["space_crossing_fraction"] != crossing_count / 256
    ):
        raise ValueError("Randomization primary crossing summary is inconsistent.")
    interval = clopper_pearson_interval(crossing_count, 256)
    observed_interval = result["space_crossing_clopper_pearson_95"]
    if observed_interval != {"lower": interval[0], "upper": interval[1]}:
        raise ValueError("Randomization Clopper-Pearson interval changed.")

    threshold = load_strict_json(by_name["threshold_bootstrap.json"])
    require_exact_keys(
        threshold,
        {
            "schema_version",
            "status",
            "unit",
            "block_length_shots",
            "replicates",
            "seed_start",
            "seed_stop_exclusive",
            "rng",
            "blocks_per_replicate",
            "primary_maximum_alerts",
            "secondary_maximum_alerts",
            "summaries",
            "replicate_results",
        },
        context="threshold bootstrap",
    )
    if (
        threshold.get("schema_version") != "run6-google-threshold-bootstrap-v1"
        or threshold.get("replicates") != 2_000
        or threshold.get("seed_start") != 613_000
        or threshold.get("seed_stop_exclusive") != 615_000
        or threshold.get("block_length_shots") != 128
        or threshold.get("blocks_per_replicate") != 40
        or not isinstance(threshold.get("replicate_results"), list)
        or len(threshold["replicate_results"]) != 2_000
    ):
        raise ValueError("Threshold bootstrap artifact differs from the lock.")
    return manifest, result


def recompute_pnnl_retention_from_state_rows(
    path: Path,
    *,
    cohort_ids: Sequence[str],
) -> dict[str, Any]:
    """Recompute the locked PNNL Boolean directly from canonical state rows."""

    cohorts = tuple(cohort_ids)
    if len(cohorts) != 11 or len(set(cohorts)) != len(cohorts):
        raise ValueError("PNNL retention requires the 11 unique frozen cohorts.")
    expected_keys = {
        (cohort, state, method)
        for cohort in cohorts
        for state in (0, 1)
        for method in PNNL_METHOD_IDS
    }
    indexed: dict[tuple[str, int, str], tuple[int, float]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != PNNL_STATE_FIELDS:
            raise ValueError("PNNL state-row columns changed.")
        for row in reader:
            try:
                state = int(row["logical_state"])
            except ValueError as error:
                raise ValueError("PNNL logical_state must be integer 0/1.") from error
            method = row["method"]
            key = (row["cohort_id"], state, method)
            if key not in expected_keys or key in indexed:
                raise ValueError(
                    f"PNNL state-row key is unexpected or duplicate: {key}"
                )
            if row["pre_false_alarm"] not in {"0", "1"}:
                raise ValueError("PNNL pre_false_alarm must be integer 0/1.")
            try:
                delay = float(row["restricted_post_delay_fraction"])
            except ValueError as error:
                raise ValueError("PNNL restricted delay must be numeric.") from error
            if not np.isfinite(delay) or not 0.0 <= delay <= 1.0:
                raise ValueError("PNNL restricted delay must lie in [0,1].")
            indexed[key] = (int(row["pre_false_alarm"]), delay)
    if set(indexed) != expected_keys:
        raise ValueError("PNNL state-row coverage differs from the lock.")

    pre_counts: dict[str, int] = {}
    cohort_delays: dict[str, list[float]] = {}
    macro_delays: dict[str, float] = {}
    for method in PNNL_METHOD_IDS:
        pre_counts[method] = sum(
            indexed[(cohort, state, method)][0]
            for cohort in cohorts
            for state in (0, 1)
        )
        cohort_delays[method] = [
            float(np.mean([indexed[(cohort, state, method)][1] for state in (0, 1)]))
            for cohort in cohorts
        ]
        macro_delays[method] = float(np.mean(cohort_delays[method]))

    target = "space_composite"
    comparisons: dict[str, dict[str, Any]] = {}
    for comparator in ("dfr", "online_logistic"):
        differences = np.asarray(cohort_delays[target], dtype=np.float64) - np.asarray(
            cohort_delays[comparator], dtype=np.float64
        )
        no_worse_pre = pre_counts[target] <= pre_counts[comparator]
        strict_delay = float(np.mean(differences)) < 0.0
        comparisons[comparator] = {
            "cohort_delay_differences": differences.tolist(),
            "macro_delay_difference": float(np.mean(differences)),
            "no_worse_pre_false_alarm": bool(no_worse_pre),
            "strictly_lower_macro_delay": bool(strict_delay),
            "retention_condition_pass": bool(no_worse_pre and strict_delay),
        }
    return {
        "pre_false_alarm_state_count": pre_counts,
        "macro_restricted_post_delay_fraction": macro_delays,
        "comparisons": comparisons,
        "retention_pass": all(
            row["retention_condition_pass"] for row in comparisons.values()
        ),
    }


def validate_pnnl_artifact_contract(
    manifest: Mapping[str, Any],
    artifacts: Mapping[str, Path],
    *,
    pittsburgh: Mapping[str, Any],
) -> None:
    """Validate the exact portable PNNL output set and array dimensions."""

    scalar_names = {
        "first_unblinding_record.json",
        "path_state_method_results.csv",
        "aggregate_results.json",
        "randomization_audit.json",
        "randomization_alarm_counts.npy",
        "randomization_maximum_log_e.npy",
    }
    schema = tuple(pittsburgh["cohort_row_schema"])
    if "m" not in schema:
        raise ValueError("PNNL cohort schema lacks m.")
    m_index = schema.index("m")
    rows = pittsburgh["cohort_pairs"]
    order = pittsburgh["cohort_order"]
    if not isinstance(rows, list) or len(rows) != 11 or len(order) != 11:
        raise ValueError("PNNL cohort coverage changed.")
    trace_names: set[str] = set()
    bootstrap_names: set[str] = set()
    trace_horizons: dict[str, int] = {}
    for cohort_index, raw_row in enumerate(rows):
        if (
            not isinstance(raw_row, list)
            or len(raw_row) != len(schema)
            or raw_row[0] != order[cohort_index]
        ):
            raise ValueError("PNNL cohort row/order changed.")
        m = raw_row[m_index]
        if isinstance(m, bool) or not isinstance(m, int) or m < 2:
            raise ValueError("PNNL cohort m must be an integer >= 2.")
        horizon = (m - m // 2) + m
        for state in (0, 1):
            for method in PNNL_METHOD_IDS:
                stem = f"{cohort_index:02d}_s{state}_{method}"
                trace_name = f"{stem}_log_e.npy"
                maxima_name = f"{stem}_bootstrap_maxima.npy"
                trace_names.add(trace_name)
                bootstrap_names.add(maxima_name)
                trace_horizons[trace_name] = horizon
    expected_names = scalar_names | trace_names | bootstrap_names
    by_name: dict[str, Path] = {}
    for artifact in artifacts.values():
        if artifact.name in by_name:
            raise ValueError(f"Duplicate PNNL artifact basename: {artifact.name}")
        by_name[artifact.name] = artifact
    if set(by_name) != expected_names:
        raise ValueError(
            "PNNL artifact contract mismatch; "
            f"missing={sorted(expected_names - by_name.keys())}, "
            f"unknown={sorted(by_name.keys() - expected_names)}."
        )
    expected_scalar_bindings = {
        "first_unblinding_record": "first_unblinding_record.json",
        "state_rows": "path_state_method_results.csv",
        "aggregate_results": "aggregate_results.json",
        "randomization_audit": "randomization_audit.json",
        "randomization_alarm_counts": "randomization_alarm_counts.npy",
        "randomization_maximum_log_e": "randomization_maximum_log_e.npy",
    }
    for key, name in expected_scalar_bindings.items():
        record = manifest[key]
        if not isinstance(record, Mapping) or record.get("path") != name:
            raise ValueError(f"PNNL portable artifact path changed: {key}")

    for name in trace_names:
        values = np.load(by_name[name], allow_pickle=False)
        if (
            values.dtype.str != "<f8"
            or values.shape != (trace_horizons[name],)
            or not np.all(np.isfinite(values))
        ):
            raise ValueError(f"PNNL trace array changed: {name}")
    for name in bootstrap_names:
        values = np.load(by_name[name], allow_pickle=False)
        if (
            values.dtype.str != "<f8"
            or values.shape != (4_096,)
            or np.any(np.isnan(values))
        ):
            raise ValueError(f"PNNL bootstrap-maxima array changed: {name}")
    counts = np.load(by_name["randomization_alarm_counts.npy"], allow_pickle=False)
    maxima = np.load(by_name["randomization_maximum_log_e.npy"], allow_pickle=False)
    if (
        counts.dtype.str != "<i8"
        or counts.shape != (256, len(PNNL_METHOD_IDS))
        or np.any(counts < 0)
        or np.any(counts > 22)
        or maxima.dtype.str != "<f8"
        or maxima.shape != counts.shape
        or np.any(np.isnan(maxima))
    ):
        raise ValueError("PNNL randomization arrays changed.")


def verify_pnnl_result(
    path: Path,
    *,
    repo_root: Path,
    ratification_path: Path,
) -> dict[str, Any]:
    """Validate the independent PNNL results and every embedded artifact."""

    manifest = load_strict_json(path)
    require_exact_keys(
        manifest,
        PNNL_RESULT_MANIFEST_KEYS,
        context="PNNL results manifest",
    )
    pnnl_config_path = repo_root / "experiments/run6/configs/pnnl_snapshot_locked.json"
    pnnl_config = load_strict_json(pnnl_config_path)
    if pnnl_config["status"] != "frozen_before_held_value_access":
        raise ValueError("PNNL configuration has not reached frozen status.")
    pittsburgh_path = repo_root / pnnl_config["normative_pittsburgh_manifest"]["path"]
    pittsburgh = load_strict_json(pittsburgh_path)
    expected = {
        "schema_version": "run6-pnnl-snapshot-results-v1",
        "protocol_id": pnnl_config["protocol_id"],
        "config_sha256": sha256_file(pnnl_config_path),
        "pittsburgh_manifest_sha256": sha256_file(pittsburgh_path),
        "freeze_ratification_sha256": sha256_file(ratification_path),
        "package_lock_sha256": sha256_file(
            repo_root / "experiments/run6/configs/python_environment_lock.txt"
        ),
    }
    for key, expected_value in expected.items():
        if manifest[key] != expected_value:
            raise ValueError(f"PNNL results binding changed: {key}")
    if manifest["environment"] != environment_fingerprint():
        raise ValueError("PNNL environment differs from the current one.")
    if not isinstance(manifest["git_commit"], str):
        raise TypeError("PNNL git_commit must be a string.")
    _require_ancestor(repo_root, manifest["git_commit"], _git_commit(repo_root))
    if not isinstance(manifest["retention_pass"], bool):
        raise TypeError("PNNL retention_pass must be Boolean.")
    timing_values = (
        manifest["started_unix"],
        manifest["held_value_processing_started_unix"],
        manifest["finished_unix"],
    )
    if (
        any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(float(value))
            for value in timing_values
        )
        or not timing_values[0] <= timing_values[1] <= timing_values[2]
    ):
        raise ValueError("PNNL processing timestamps are invalid.")

    artifacts: dict[str, Path] = {}
    for key in (
        "first_unblinding_record",
        "state_rows",
        "aggregate_results",
        "randomization_audit",
        "randomization_alarm_counts",
        "randomization_maximum_log_e",
        "trace_artifacts",
        "bootstrap_artifacts",
    ):
        _verify_artifact_records_recursive(
            manifest[key],
            manifest_path=path,
            observed=artifacts,
        )
    validate_pnnl_artifact_contract(
        manifest,
        artifacts,
        pittsburgh=pittsburgh,
    )
    unblinding_path = _resolve_artifact(manifest["first_unblinding_record"], path)
    unblinding = load_strict_json(unblinding_path)
    require_exact_keys(
        unblinding,
        {
            "schema_version",
            "utc",
            "git_commit",
            "config_sha256",
            "manifest_sha256",
            "freeze_ratification_sha256",
            "package_lock",
            "package_environment",
            "held_payloads",
            "scores_computed_before_record",
        },
        context="PNNL first-unblinding record",
    )
    if (
        unblinding["schema_version"] != "run6-pnnl-first-unblinding-v1"
        or unblinding["git_commit"] != manifest["git_commit"]
        or unblinding["config_sha256"] != sha256_file(pnnl_config_path)
        or unblinding["manifest_sha256"] != sha256_file(pittsburgh_path)
        or unblinding["freeze_ratification_sha256"] != sha256_file(ratification_path)
        or unblinding["package_environment"] != environment_fingerprint()
        or unblinding["scores_computed_before_record"] is not False
    ):
        raise ValueError("PNNL first-unblinding binding changed.")
    package_lock = unblinding["package_lock"]
    if not isinstance(package_lock, Mapping):
        raise TypeError("PNNL first-unblinding package lock must be an object.")
    require_exact_keys(
        package_lock,
        ARTIFACT_KEYS,
        context="PNNL first-unblinding package lock",
    )
    package_lock_path = (
        repo_root / "experiments/run6/configs/python_environment_lock.txt"
    ).resolve()
    if (
        package_lock["path"] != "experiments/run6/configs/python_environment_lock.txt"
        or isinstance(package_lock["bytes"], bool)
        or not isinstance(package_lock["bytes"], int)
        or package_lock["bytes"] != package_lock_path.stat().st_size
        or package_lock["sha256"] != sha256_file(package_lock_path)
        or package_lock["sha256"] != manifest["package_lock_sha256"]
    ):
        raise ValueError("PNNL first-unblinding package lock changed.")
    payloads = unblinding["held_payloads"]
    payload_hashes = manifest["held_payload_sha256"]
    if not isinstance(payloads, list) or not isinstance(payload_hashes, Mapping):
        raise TypeError("PNNL held-payload records must be a list and hash object.")
    observed_payloads: dict[str, str] = {}
    for payload in payloads:
        if not isinstance(payload, Mapping):
            raise TypeError("PNNL held-payload row must be an object.")
        require_exact_keys(
            payload,
            {"snapshot_id", "path", "bytes", "sha256"},
            context="PNNL held-payload row",
        )
        snapshot_id = payload["snapshot_id"]
        relative = payload["path"]
        if not isinstance(snapshot_id, str) or not isinstance(relative, str):
            raise TypeError("PNNL held-payload IDs/paths must be strings.")
        source = (repo_root / relative).resolve()
        if not source.is_relative_to(repo_root.resolve()) or not source.is_file():
            raise ValueError("PNNL held-payload path escapes or is absent.")
        if (
            isinstance(payload["bytes"], bool)
            or not isinstance(payload["bytes"], int)
            or source.stat().st_size != payload["bytes"]
            or sha256_file(source) != payload["sha256"]
        ):
            raise ValueError(f"PNNL held payload changed: {snapshot_id}")
        if snapshot_id in observed_payloads:
            raise ValueError(f"Duplicate PNNL held payload: {snapshot_id}")
        observed_payloads[snapshot_id] = payload["sha256"]
    if dict(payload_hashes) != observed_payloads:
        raise ValueError("PNNL held-payload hash summary is inconsistent.")
    aggregate_record = manifest["aggregate_results"]
    aggregate_path = _resolve_artifact(aggregate_record, path)
    aggregate = load_strict_json(aggregate_path)
    require_exact_keys(
        aggregate,
        {"cohort_rows", "macro_by_method", "comparisons", "retention_pass"},
        context="PNNL aggregate results",
    )
    state_path = _resolve_artifact(manifest["state_rows"], path)
    recomputed = recompute_pnnl_retention_from_state_rows(
        state_path,
        cohort_ids=pittsburgh["cohort_order"],
    )
    if (
        aggregate["retention_pass"] != recomputed["retention_pass"]
        or manifest["retention_pass"] != recomputed["retention_pass"]
    ):
        raise ValueError("PNNL retention differs from canonical state rows.")
    macro = aggregate["macro_by_method"]
    comparisons = aggregate["comparisons"]
    if (
        not isinstance(macro, Mapping)
        or set(macro) != set(PNNL_METHOD_IDS)
        or not isinstance(comparisons, Mapping)
        or set(comparisons) != {"dfr", "online_logistic"}
    ):
        raise ValueError("PNNL aggregate method coverage changed.")
    for method in PNNL_METHOD_IDS:
        row = macro[method]
        if (
            not isinstance(row, Mapping)
            or row.get("pre_false_alarm_state_count")
            != recomputed["pre_false_alarm_state_count"][method]
            or row.get("macro_restricted_post_delay_fraction")
            != recomputed["macro_restricted_post_delay_fraction"][method]
        ):
            raise ValueError(f"PNNL aggregate macro row changed: {method}")
    for comparator in ("dfr", "online_logistic"):
        observed = comparisons[comparator]
        expected_row = recomputed["comparisons"][comparator]
        if not isinstance(observed, Mapping) or any(
            observed.get(key) != expected_value
            for key, expected_value in expected_row.items()
        ):
            raise ValueError(f"PNNL aggregate comparison changed: {comparator}")
    return manifest


def integrate_full_run6_decision(
    partial: Mapping[str, Any],
    *,
    randomization_manifest_path: Path,
    randomization_result: Mapping[str, Any],
    pnnl_manifest_path: Path,
    pnnl_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Create the full locked Boolean after all independent arms are verified."""

    result = dict(partial)
    pnnl_pass = bool(pnnl_manifest["retention_pass"])
    result["summary_scope"] = "full_run6_locked_decision"
    result["randomization_audit"] = {
        "status": "completed_and_hash_verified",
        "manifest_sha256": sha256_file(randomization_manifest_path),
        "space_crossing_count_at_100": randomization_result["crossing_counts_at_100"][
            "space"
        ],
        "replicates": randomization_result["replicate_count"],
        "clopper_pearson_95": randomization_result["space_crossing_clopper_pearson_95"],
        "changes_primary_boolean": False,
    }
    result["pnnl_retention_pass"] = pnnl_pass
    result["pnnl_results_manifest_sha256"] = sha256_file(pnnl_manifest_path)
    result["overall_run6_advantage"] = bool(result["google_primary_pass"] and pnnl_pass)
    reasons = [
        reason
        for reason in result["negative_result_reasons"]
        if reason != "pnnl_retention_not_run"
    ]
    if not pnnl_pass:
        reasons.append("pnnl_retention_failed")
    result["negative_result_reasons"] = reasons
    return result


def parse_dot01_outcome_slice(
    path: Path,
    *,
    expected_count: int,
    start: int,
    stop: int,
) -> np.ndarray:
    """Stream one selected ``.01`` interval without decoding embargoed records."""

    bounds = (expected_count, start, stop)
    if (
        any(isinstance(value, bool) or not isinstance(value, int) for value in bounds)
        or expected_count < 1
        or not 0 <= start < stop <= expected_count
    ):
        raise ValueError("Invalid outcome count or half-open slice.")
    selected = np.empty(stop - start, dtype=np.uint8)
    record_count = 0
    with path.open("rb") as handle:
        for record_count, raw_line in enumerate(handle, start=1):
            index = record_count - 1
            if start <= index < stop:
                payload = raw_line[:-1] if raw_line.endswith(b"\n") else raw_line
                if payload.endswith(b"\r"):
                    payload = payload[:-1]
                if payload == b"0":
                    selected[index - start] = 0
                elif payload == b"1":
                    selected[index - start] = 1
                else:
                    raise ValueError(f"Selected outcome record {index} is not binary.")
    if record_count != expected_count:
        raise ValueError(
            f"Expected {expected_count} outcome lines, found {record_count}."
        )
    return selected


def frozen_ranking(scores: np.ndarray, archive_shots: np.ndarray) -> np.ndarray:
    """Rank by descending exact score and then lower archive shot."""

    values = np.asarray(scores, dtype=np.float64)
    shots = np.asarray(archive_shots, dtype=np.int64)
    if values.ndim != 1 or shots.shape != values.shape or len(values) < 1:
        raise ValueError("scores/archive_shots must be equal nonempty vectors.")
    if not np.all(np.isfinite(values)) or len(np.unique(shots)) != len(shots):
        raise ValueError("Scores must be finite and archive shots unique.")
    return np.lexsort((shots, -values)).astype(np.int64, copy=False)


def _load_cycle_score(path: Path, *, expected_shape: tuple[int, int]) -> np.ndarray:
    values = np.load(path, allow_pickle=False)
    if values.dtype.str != "<f8":
        raise TypeError(f"{path.name} must be persisted little-endian float64.")
    if values.shape != expected_shape or not np.all(np.isfinite(values)):
        raise ValueError(f"{path.name} has invalid shape or non-finite scores.")
    return np.asarray(values, dtype=np.float64)


def load_frozen_shot_scores(
    artifacts: Mapping[str, Path],
    *,
    shot_count: int = 20_000,
    role_count: int = 51,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Load cycle arrays and derive the frozen per-shot maxima."""

    cycles: dict[str, np.ndarray] = {}
    shots: dict[str, np.ndarray] = {}
    for method in METHOD_IDS:
        name = f"held__{method}__empirical_cycle_score.npy"
        if name not in artifacts:
            raise KeyError(f"Missing frozen cycle-score artifact: {name}")
        values = _load_cycle_score(
            artifacts[name],
            expected_shape=(shot_count, role_count),
        )
        cycles[method] = values
        shots[method] = np.max(values, axis=1)
    return cycles, shots


def _shared_outcome_label_bundle_sha256(
    labels: Mapping[str, np.ndarray],
    archive_shots: np.ndarray,
) -> str:
    """Hash the one outcome-label bundle shared by every frozen method."""

    shots = np.asarray(archive_shots, dtype="<i8")
    digest = hashlib.sha256()
    digest.update(b"run6-google-shared-outcome-label-bundle-v1\0")
    digest.update(len(shots).to_bytes(8, byteorder="little", signed=False))
    digest.update(shots.tobytes(order="C"))
    for label_id in LABEL_IDS:
        encoded_id = label_id.encode("utf-8")
        values = np.asarray(labels[label_id], dtype=np.uint8)
        digest.update(len(encoded_id).to_bytes(4, byteorder="little", signed=False))
        digest.update(encoded_id)
        digest.update(len(values).to_bytes(8, byteorder="little", signed=False))
        digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def derive_method_input_parity_evidence(
    cycles: Mapping[str, np.ndarray],
    labels: Mapping[str, np.ndarray],
    archive_shots: np.ndarray,
    *,
    expected_shot_count: int,
    expected_role_count: int,
) -> dict[str, Any]:
    """Derive the locked equal-record/equal-label predicate from inputs."""

    expected_shape = (int(expected_shot_count), int(expected_role_count))
    observed_method_ids = sorted(str(method) for method in cycles)
    exact_method_set = set(cycles) == set(METHOD_IDS)
    method_rows: dict[str, Any] = {}
    detector_inputs_match = exact_method_set
    for method in METHOD_IDS:
        if method not in cycles:
            method_rows[method] = {
                "shape": None,
                "record_count": None,
                "matches_locked_shape_and_count": False,
            }
            detector_inputs_match = False
            continue
        values = np.asarray(cycles[method])
        matches = values.shape == expected_shape
        method_rows[method] = {
            "shape": list(values.shape),
            "record_count": int(values.size),
            "matches_locked_shape_and_count": bool(matches),
        }
        detector_inputs_match = detector_inputs_match and matches

    shots = np.asarray(archive_shots)
    archive_valid = (
        shots.shape == (expected_shape[0],)
        and np.issubdtype(shots.dtype, np.integer)
        and len(np.unique(shots)) == expected_shape[0]
    )
    exact_label_set = set(labels) == set(LABEL_IDS)
    label_record_counts: dict[str, int | None] = {}
    label_inputs_valid = exact_label_set
    for label_id in LABEL_IDS:
        if label_id not in labels:
            label_record_counts[label_id] = None
            label_inputs_valid = False
            continue
        values = np.asarray(labels[label_id])
        label_record_counts[label_id] = int(values.size)
        label_inputs_valid = (
            label_inputs_valid
            and values.shape == (expected_shape[0],)
            and bool(np.all((values == 0) | (values == 1)))
        )

    shared_bundle_valid = bool(
        exact_method_set and archive_valid and label_inputs_valid
    )
    shared_bundle_sha256 = (
        _shared_outcome_label_bundle_sha256(labels, shots)
        if shared_bundle_valid
        else None
    )
    verified = bool(detector_inputs_match and shared_bundle_valid)
    return {
        "schema_version": "run6-google-method-input-parity-evidence-v1",
        "expected_method_ids": list(METHOD_IDS),
        "observed_method_ids": observed_method_ids,
        "expected_held_score_shape": list(expected_shape),
        "held_detector_score_inputs": method_rows,
        "all_methods_have_locked_detector_record_shape_and_count": bool(
            detector_inputs_match
        ),
        "shared_outcome_label_bundle": {
            "serialization": (
                "domain_tag_then_little_endian_archive_shots_then_locked_label_ids"
                "_and_uint8_vectors"
            ),
            "sha256": shared_bundle_sha256,
            "label_ids": list(LABEL_IDS),
            "label_record_counts": label_record_counts,
            "archive_shot_count": int(shots.size),
            "consumer_method_ids": observed_method_ids,
            "single_shared_bundle_for_all_methods": shared_bundle_valid,
        },
        "no_method_received_extra_detector_records_or_outcome_labels": verified,
    }


def load_and_validate_thresholds(
    artifacts: Mapping[str, Path],
    *,
    shot_count: int = 5_000,
    role_count: int = 51,
) -> dict[str, dict[str, Any]]:
    """Recompute every primary threshold from the frozen validation scores."""

    table = load_strict_json(artifacts["thresholds.json"])
    require_exact_keys(table, METHOD_IDS, context="locked threshold table")
    result: dict[str, dict[str, Any]] = {}
    for method in METHOD_IDS:
        row = table[method]
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
        threshold_is_valid = (
            isinstance(threshold, (int, float))
            and not isinstance(threshold, bool)
            and np.isfinite(float(threshold))
        ) or threshold == "+inf"
        if (
            row["max_validation_alerts"] != 2
            or not threshold_is_valid
            or isinstance(row["secondary_zero_alert_threshold"], bool)
            or not isinstance(
                row["secondary_zero_alert_threshold"],
                (int, float),
            )
            or not np.isfinite(float(row["secondary_zero_alert_threshold"]))
            or row["secondary_validation_alert_count"] != 0
        ):
            raise ValueError(f"Threshold alert budget changed for {method}.")
        name = f"threshold__{method}__empirical_cycle_score.npy"
        values = _load_cycle_score(
            artifacts[name],
            expected_shape=(shot_count, role_count),
        )
        selected = select_strict_shot_threshold(values, 2)
        serialized_selected: float | str = selected.threshold
        if np.isposinf(selected.threshold):
            serialized_selected = "+inf"
        if (
            row["threshold"] != serialized_selected
            or row["validation_alert_count"] != selected.alert_count
            or row["secondary_zero_alert_threshold"]
            != float(np.max(selected.shot_scores))
        ):
            raise ValueError(f"Frozen threshold table is inconsistent for {method}.")
        result[method] = dict(row)
    return result


def validate_formal_detector_artifacts(
    artifacts: Mapping[str, Path],
    thresholds: Mapping[str, Mapping[str, Any]],
    *,
    protocol_id: str,
    shot_count: int = 20_000,
    role_count: int = 51,
) -> None:
    """Validate formal traces/scopes and the role/base-component prior ledger."""

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
                valid = (
                    np.all(np.isfinite(values))
                    if method in EXACT_METHOD_IDS
                    else np.all(np.isnan(values))
                )
            else:
                valid = method in EXACT_METHOD_IDS or not np.any(values)
            if not valid:
                raise ValueError(f"Formal detector array semantics changed: {name}")
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
                sidecar.get("schema_version") != "run6-cycle-array-v1"
                or sidecar.get("protocol_id") != protocol_id
                or sidecar.get("run_id") != "google2022-canonical-detector"
                or sidecar.get("phase") != "held"
                or sidecar.get("method_id") != method
                or sidecar.get("array_id") != array_id
                or sidecar.get("data_file") != name
                or sidecar.get("data_sha256") != sha256_file(artifacts[name])
                or sidecar.get("shape") != [shot_count]
                or sidecar.get("dtype") != expected_dtype
                or sidecar.get("flatten_order") != ["paired_shot"]
                or sidecar.get("pair_index_range") != [0, shot_count]
                or sidecar.get("reference_archive_start") != 20_000
                or sidecar.get("monitor_archive_start") != 40_000
                or sidecar.get("threshold") != thresholds[method]["threshold"]
                or sidecar.get("formal_claim_scope") != expected_scope
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
        expected_weights = np.tile(base_weights / role_count, role_count)
        proper = summary["proper_prior"][method]
        sr = summary["shiryaev_roberts"][method]
        metadata = summary["expert_metadata"][method]
        if (
            not isinstance(proper, Mapping)
            or not isinstance(sr, Mapping)
            or not isinstance(metadata, Mapping)
        ):
            raise TypeError("Formal component summary rows must be objects.")
        accumulator_keys = {
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
        require_exact_keys(
            proper,
            accumulator_keys,
            context=f"formal proper-prior summary {method}",
        )
        require_exact_keys(
            sr,
            accumulator_keys,
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
            or sr.get("role_count") != role_count
            or proper["base_component_count"] != base_count
            or sr.get("base_component_count") != base_count
            or proper["expert_flatten_order"] != ["role", "base_component"]
            or sr["expert_flatten_order"] != ["role", "base_component"]
            or metadata["expert_flatten_order"] != ["role", "base_component"]
            or metadata["role_prior"] != 1.0 / role_count
            or metadata["within_shot_factor_compounding"] is not False
            or metadata["expert_count"] != expert_count
            or metadata["base_component_ids"]
            != [list(identifier) for identifier in priors[method].component_ids]
            or not np.array_equal(
                np.asarray(metadata["base_component_weights"]),
                base_weights,
            )
            or not np.array_equal(
                np.asarray(proper["component_weights"]),
                expected_weights,
            )
            or not np.array_equal(
                np.asarray(sr["component_weights"]),
                expected_weights,
            )
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
            or not np.isclose(
                np.sum(np.asarray(proper["component_weights"], dtype=np.float64)),
                1.0,
                rtol=0.0,
                atol=1e-15,
            )
            or not np.isclose(
                np.sum(np.asarray(sr["component_weights"], dtype=np.float64)),
                1.0,
                rtol=0.0,
                atol=1e-15,
            )
            or proper["expert_id_rule"]
            != "(role, *base_component_id), role-major then base-component-major"
            or sr["expert_id_rule"]
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
        for family, crossing_id in (
            (proper, "first_e_crossing"),
            (sr, "first_sr_crossing"),
        ):
            expected_crossing = np.zeros(shot_count, dtype=np.bool_)
            crossing = family.get("first_crossing_update")
            if crossing is not None:
                if (
                    isinstance(crossing, bool)
                    or not isinstance(crossing, int)
                    or not 1 <= crossing <= shot_count
                ):
                    raise ValueError("Formal first crossing is out of range.")
                expected_crossing[crossing - 1] = True
            if not np.array_equal(traces[method][crossing_id], expected_crossing):
                raise ValueError(f"Formal crossing mask changed: {method}")


def validate_frozen_shot_table(
    path: Path,
    cycles: Mapping[str, np.ndarray],
    thresholds: Mapping[str, Mapping[str, Any]],
    *,
    phase: str = "held",
    pair_index_start: int = 0,
    reference_start: int = 20_000,
    monitor_start: int = 40_000,
    windows: Mapping[str, Sequence[int]] | None = None,
) -> None:
    """Cross-check the canonical CSV against every frozen score array."""

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
                expected_score = float(applied.shot_scores[local_index])
                expected_role = int(applied.shot_score_roles[local_index])
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
                    or float(row["shot_score"]) != expected_score
                    or int(row["argmax_role"]) != expected_role
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


def recompute_event_summary(
    cycles: Mapping[str, np.ndarray],
    thresholds: Mapping[str, Mapping[str, Any]],
    *,
    monitor_start: int,
    windows: Mapping[str, Sequence[int]],
    threshold_key: str = "threshold",
) -> dict[str, Any]:
    """Recompute the detector-only event table from hash-frozen cycle scores."""

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


def _ratio(numerator: float, denominator: float) -> float | None:
    return None if denominator == 0.0 else float(numerator / denominator)


def one_label_risk_metrics(
    scores: Mapping[str, np.ndarray],
    labels: np.ndarray,
    archive_shots: np.ndarray,
    *,
    budgets: Sequence[int] = RISK_BUDGETS,
) -> dict[str, Any]:
    """Compute the predeclared exact top-budget risk summaries."""

    outcome = np.asarray(labels, dtype=np.uint8)
    shots = np.asarray(archive_shots, dtype=np.int64)
    if outcome.shape != shots.shape or np.any(outcome > 1):
        raise ValueError("Risk labels must be one binary value per archive shot.")
    total = int(np.sum(outcome, dtype=np.int64))
    result: dict[str, Any] = {}
    for method in METHOD_IDS:
        if method not in scores:
            raise KeyError(f"Missing frozen method score: {method}")
        ranking = frozen_ranking(scores[method], shots)
        rows: dict[str, Any] = {}
        recalls: list[float] = []
        fractions: list[float] = []
        for raw_budget in budgets:
            budget = int(raw_budget)
            if not 0 < budget < len(shots):
                raise ValueError("Every risk budget must lie in (0, shot_count).")
            selected = ranking[:budget]
            captured = int(np.sum(outcome[selected], dtype=np.int64))
            recall = _ratio(captured, total)
            rows[str(budget)] = {
                "alert_budget_shots": budget,
                "alert_fraction": budget / len(shots),
                "captured_mismatches": captured,
                "total_mismatches": total,
                "mismatch_recall": recall,
                "alert_precision": captured / budget,
                "retained_mismatch_rate": (total - captured) / (len(shots) - budget),
                "coverage": 1.0 - budget / len(shots),
                "selected_archive_shots": shots[selected].tolist(),
            }
            if recall is not None:
                recalls.append(recall)
                fractions.append(budget / len(shots))
        partial_area = (
            float(np.trapezoid(recalls, fractions))
            if len(recalls) == len(tuple(budgets))
            else None
        )
        result[method] = {
            "budgets": rows,
            "partial_trapezoidal_recall_area": partial_area,
        }
    return result


def circular_block_bootstrap_indices(
    seed: int,
    *,
    shot_count: int,
    block_length: int,
) -> np.ndarray:
    """Draw the locked circular complete-shot moving-block resample."""

    if shot_count < 1 or block_length < 1:
        raise ValueError("shot_count and block_length must be positive.")
    block_count = math.ceil(shot_count / block_length)
    generator = np.random.Generator(np.random.PCG64(int(seed)))
    starts = generator.integers(0, shot_count, size=block_count)
    offsets = np.arange(block_length, dtype=np.int64)
    return ((starts[:, None] + offsets[None, :]) % shot_count).reshape(-1)[:shot_count]


def _top_budget_captures_from_counts(
    ranking: np.ndarray,
    counts: np.ndarray,
    labels: np.ndarray,
    budgets: Sequence[int],
) -> dict[int, int]:
    """Top-budget capture in a row bootstrap without re-sorting duplicates."""

    ordered_counts = counts[ranking]
    cumulative = np.cumsum(ordered_counts, dtype=np.int64)
    weighted = ordered_counts * labels[ranking]
    weighted_cumulative = np.cumsum(weighted, dtype=np.int64)
    result: dict[int, int] = {}
    for raw_budget in budgets:
        budget = int(raw_budget)
        position = int(np.searchsorted(cumulative, budget, side="left"))
        count_before = 0 if position == 0 else int(cumulative[position - 1])
        capture_before = 0 if position == 0 else int(weighted_cumulative[position - 1])
        take_at_boundary = budget - count_before
        result[budget] = capture_before + take_at_boundary * int(
            labels[ranking[position]]
        )
    return result


def _percentile_interval(values: np.ndarray) -> dict[str, Any]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return {"lower": None, "upper": None, "valid_replicates": 0}
    lower, upper = np.percentile(finite, [2.5, 97.5], method="linear")
    return {
        "lower": float(lower),
        "upper": float(upper),
        "valid_replicates": len(finite),
    }


def bootstrap_risk_uncertainty(
    scores: Mapping[str, np.ndarray],
    labels: Mapping[str, np.ndarray],
    archive_shots: np.ndarray,
    *,
    budgets: Sequence[int] = RISK_BUDGETS,
    replicates: int = 2_000,
    seed_start: int = 611_000,
    block_length: int = 128,
) -> dict[str, Any]:
    """Run the locked paired circular moving-block risk bootstrap."""

    shots = np.asarray(archive_shots, dtype=np.int64)
    shot_count = len(shots)
    if replicates < 1:
        raise ValueError("replicates must be positive.")
    method_rankings = {
        method: frozen_ranking(scores[method], shots) for method in METHOD_IDS
    }
    label_arrays: dict[str, np.ndarray] = {}
    for label_id in LABEL_IDS:
        if label_id not in labels:
            raise KeyError(f"Missing bootstrap label: {label_id}")
        values = np.asarray(labels[label_id], dtype=np.uint8)
        if values.shape != (shot_count,) or np.any(values > 1):
            raise ValueError(f"Bootstrap label {label_id} must be binary.")
        label_arrays[label_id] = values

    budget_tuple = tuple(int(value) for value in budgets)
    estimates = {
        label_id: {
            method: {
                budget: {
                    metric: np.empty(replicates, dtype=np.float64)
                    for metric in METRIC_IDS
                }
                for budget in budget_tuple
            }
            for method in METHOD_IDS
        }
        for label_id in LABEL_IDS
    }
    for replicate in range(replicates):
        indices = circular_block_bootstrap_indices(
            seed_start + replicate,
            shot_count=shot_count,
            block_length=block_length,
        )
        counts = np.bincount(indices, minlength=shot_count).astype(
            np.int64,
            copy=False,
        )
        for label_id, outcome in label_arrays.items():
            total = int(counts @ outcome.astype(np.int64))
            for method, ranking in method_rankings.items():
                captures = _top_budget_captures_from_counts(
                    ranking,
                    counts,
                    outcome,
                    budget_tuple,
                )
                for budget, captured in captures.items():
                    target = estimates[label_id][method][budget]
                    target["captured_mismatches"][replicate] = captured
                    target["mismatch_recall"][replicate] = (
                        np.nan if total == 0 else captured / total
                    )
                    target["alert_precision"][replicate] = captured / budget
                    target["retained_mismatch_rate"][replicate] = (total - captured) / (
                        shot_count - budget
                    )

    intervals: dict[str, Any] = {}
    differences: dict[str, Any] = {}
    for label_id in LABEL_IDS:
        intervals[label_id] = {}
        for method in METHOD_IDS:
            intervals[label_id][method] = {}
            for budget in budget_tuple:
                intervals[label_id][method][str(budget)] = {
                    metric: _percentile_interval(
                        estimates[label_id][method][budget][metric]
                    )
                    for metric in METRIC_IDS
                }
        differences[label_id] = {}
        for comparator in ("m0", "m3"):
            key = f"space_minus_{comparator}"
            differences[label_id][key] = {}
            for budget in budget_tuple:
                differences[label_id][key][str(budget)] = {
                    metric: _percentile_interval(
                        estimates[label_id]["space"][budget][metric]
                        - estimates[label_id][comparator][budget][metric]
                    )
                    for metric in METRIC_IDS
                }
    return {
        "kind": "paired_circular_moving_complete_shot_blocks",
        "block_length_shots": block_length,
        "replicates": replicates,
        "seed_start": seed_start,
        "seed_stop_exclusive": seed_start + replicates,
        "rng": "numpy.random.Generator(PCG64)",
        "blocks_per_replicate": math.ceil(shot_count / block_length),
        "percentile_interval": [2.5, 97.5],
        "percentile_method": "linear",
        "method_intervals": intervals,
        "space_comparator_difference_intervals": differences,
    }


def _write_outcome_table(
    path: Path,
    *,
    archive_shots: np.ndarray,
    actual: np.ndarray,
    correlated: np.ndarray,
    pymatching: np.ndarray,
    detector_manifest_sha256: str,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTCOME_TABLE_FIELDS)
        writer.writeheader()
        for index, archive_shot in enumerate(archive_shots):
            correlated_mismatch = int(actual[index] ^ correlated[index])
            pymatching_mismatch = int(actual[index] ^ pymatching[index])
            writer.writerow(
                {
                    "monitor_archive_shot": int(archive_shot),
                    "actual_observable_flip": int(actual[index]),
                    "correlated_matching_prediction": int(correlated[index]),
                    "correlated_matching_mismatch": correlated_mismatch,
                    "pymatching_prediction": int(pymatching[index]),
                    "pymatching_mismatch": pymatching_mismatch,
                    "detector_manifest_sha256": detector_manifest_sha256,
                }
            )


def _event_summary_complete(event_summary: Mapping[str, Any]) -> bool:
    try:
        return all(
            all(
                window in event_summary[method]["windows"]
                for window in ("primary", "narrow", "wide")
            )
            for method in METHOD_IDS
        )
    except (KeyError, TypeError):
        return False


def _method_input_parity_evidence_verified(evidence: Mapping[str, Any]) -> bool:
    """Verify the serialized evidence before using it as an atomic predicate."""

    try:
        expected_shape = evidence["expected_held_score_shape"]
        shot_count, role_count = (int(value) for value in expected_shape)
        method_rows = evidence["held_detector_score_inputs"]
        shared = evidence["shared_outcome_label_bundle"]
        digest = shared["sha256"]
        digest_is_sha256 = (
            isinstance(digest, str)
            and len(digest) == 64
            and all(character in "0123456789abcdef" for character in digest)
        )
        detector_rows_match = set(method_rows) == set(METHOD_IDS) and all(
            method_rows[method]["shape"] == [shot_count, role_count]
            and method_rows[method]["record_count"] == shot_count * role_count
            and method_rows[method]["matches_locked_shape_and_count"] is True
            for method in METHOD_IDS
        )
        labels_match = (
            shared["label_ids"] == list(LABEL_IDS)
            and set(shared["label_record_counts"]) == set(LABEL_IDS)
            and all(
                shared["label_record_counts"][label_id] == shot_count
                for label_id in LABEL_IDS
            )
            and shared["archive_shot_count"] == shot_count
            and shared["consumer_method_ids"] == list(METHOD_IDS)
            and shared["single_shared_bundle_for_all_methods"] is True
            and digest_is_sha256
        )
        return bool(
            evidence["schema_version"] == "run6-google-method-input-parity-evidence-v1"
            and evidence["expected_method_ids"] == list(METHOD_IDS)
            and evidence["observed_method_ids"] == list(METHOD_IDS)
            and shot_count > 0
            and role_count > 0
            and evidence["all_methods_have_locked_detector_record_shape_and_count"]
            is True
            and detector_rows_match
            and labels_match
            and evidence["no_method_received_extra_detector_records_or_outcome_labels"]
            is True
        )
    except (KeyError, TypeError, ValueError):
        return False


def build_decision_summary(
    *,
    config: Mapping[str, Any],
    event_summary: Mapping[str, Any],
    primary_risk: Mapping[str, Any],
    uncertainty: Mapping[str, Any],
    detector_freeze_verified: bool,
    method_input_parity_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate every locked Google atomic predicate without rescue analyses."""

    capture = {
        method: int(primary_risk[method]["budgets"]["20"]["captured_mismatches"])
        for method in METHOD_IDS
    }
    predicate_values = {
        "fixed_space_composite_alerts_inside_primary_event_window_at_locked_primary_threshold": bool(
            event_summary["space"]["windows"]["primary"]["detected"]
        ),
        "fixed_space_composite_has_at_most_9_pre_event_alerts": int(
            event_summary["space"]["pre_event_alert_count"]
        )
        <= 9,
        "detector_scores_thresholds_and_resource_ledger_frozen_before_outcome_access": bool(
            detector_freeze_verified
        ),
        "space_top20_primary_mismatch_capture_at_least_dfr_plus_1": capture["space"]
        >= capture["m0"] + 1,
        "space_top20_primary_mismatch_capture_at_least_online_logistic_plus_1": capture[
            "space"
        ]
        >= capture["m3"] + 1,
        "all_window_sensitivity_and_uncertainty_results_reported": _event_summary_complete(
            event_summary
        )
        and uncertainty.get("replicates") == 2_000,
        "no_method_received_extra_detector_records_or_outcome_labels": (
            _method_input_parity_evidence_verified(method_input_parity_evidence)
        ),
    }
    expected_predicates = config["decision"]["google_primary_pass_all"]
    if list(predicate_values) != expected_predicates:
        raise ValueError("Decision predicate implementation differs from the lock.")
    contextual_controls_reported = all(
        method in primary_risk and method in event_summary
        for method in ("m0c", "m1", "m2")
    )
    google_pass = all(predicate_values.values()) and contextual_controls_reported
    reasons = [
        identifier for identifier, passed in predicate_values.items() if not passed
    ]
    if not contextual_controls_reported:
        reasons.append("mandatory_contextual_controls_not_reported")
    reasons.append("pnnl_retention_not_run")
    return {
        "schema_version": "run6-google-decision-v1",
        "summary_scope": "google_detector_and_outcome_only_not_full_run6_summary",
        "primary_label": "actual_xor_correlated_matching_prediction",
        "primary_budget_shots": 20,
        "top20_capture": capture,
        "atomic_predicates": predicate_values,
        "method_input_parity_evidence": dict(method_input_parity_evidence),
        "mandatory_contextual_controls_reported": contextual_controls_reported,
        "google_primary_pass": google_pass,
        "randomization_audit": "not_run_in_this_command",
        "pnnl_retention_pass": "not_run",
        "overall_run6_advantage": False,
        "negative_result_reasons": reasons,
        "bootstrap_changes_primary_boolean": False,
    }


def synthetic_dry_run() -> dict[str, Any]:
    """Exercise ranking, XOR labels, bootstrap, and decision logic synthetically."""

    generator = np.random.Generator(np.random.PCG64(611999))
    shot_count = 32
    shots = np.arange(40_000, 40_000 + shot_count, dtype=np.int64)
    base = np.linspace(0.0, 1.0, shot_count, dtype=np.float64)
    scores = {method: np.roll(base, index) for index, method in enumerate(METHOD_IDS)}
    actual = generator.integers(0, 2, size=shot_count, dtype=np.uint8)
    correlated = generator.integers(0, 2, size=shot_count, dtype=np.uint8)
    pymatching = generator.integers(0, 2, size=shot_count, dtype=np.uint8)
    labels = {
        "correlated_matching_mismatch": actual ^ correlated,
        "pymatching_mismatch": actual ^ pymatching,
    }
    budgets = (2, 4, 8)
    risk = {
        label_id: one_label_risk_metrics(
            scores,
            outcome,
            shots,
            budgets=budgets,
        )
        for label_id, outcome in labels.items()
    }
    bootstrap = bootstrap_risk_uncertainty(
        scores,
        labels,
        shots,
        budgets=budgets,
        replicates=12,
        seed_start=611000,
        block_length=8,
    )
    return {
        "status": "synthetic_dry_run_passed",
        "raw_run6_values_opened": False,
        "primary_label_equals_actual_xor_correlated": bool(
            np.array_equal(labels["correlated_matching_mismatch"], actual ^ correlated)
        ),
        "secondary_label_equals_actual_xor_pymatching": bool(
            np.array_equal(labels["pymatching_mismatch"], actual ^ pymatching)
        ),
        "risk": risk,
        "bootstrap": bootstrap,
    }


def run_real(args: argparse.Namespace) -> None:
    required = {
        "--config": args.config,
        "--freeze-ratification": args.freeze_ratification,
        "--detector-manifest": args.detector_manifest,
        "--outcome-root": args.outcome_root,
        "--output": args.output,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError(f"Missing outcome-audit arguments: {missing}")

    repo_root = Path(__file__).resolve().parents[3]
    config_path = args.config.resolve()
    ratification_path = args.freeze_ratification.resolve()
    detector_manifest_path = args.detector_manifest.resolve()
    outcome_root = args.outcome_root.resolve()
    output = args.output.resolve()
    config = load_google_lock(config_path)
    verify_freeze_ratification(
        ratification_path,
        repo_root=repo_root,
        config_path=config_path,
        config=config,
    )
    detector_manifest, artifacts = verify_detector_manifest(
        detector_manifest_path,
        config_path=config_path,
        config=config,
        ratification_path=ratification_path,
        repo_root=repo_root,
    )
    full_aggregation_requested = require_paired_final_inputs(
        args.randomization_manifest,
        args.pnnl_results_manifest,
    )
    randomization_manifest_path: Path | None = None
    pnnl_manifest_path: Path | None = None
    randomization_result: dict[str, Any] | None = None
    pnnl_manifest: dict[str, Any] | None = None
    if full_aggregation_requested:
        randomization_manifest_path = args.randomization_manifest.resolve()
        pnnl_manifest_path = args.pnnl_results_manifest.resolve()
        _, randomization_result = verify_randomization_result(
            randomization_manifest_path,
            repo_root=repo_root,
            config_path=config_path,
            config=config,
            ratification_path=ratification_path,
            detector_manifest_path=detector_manifest_path,
            detector_manifest=detector_manifest,
        )
        pnnl_manifest = verify_pnnl_result(
            pnnl_manifest_path,
            repo_root=repo_root,
            ratification_path=ratification_path,
        )
    require_thread_environment(config["numeric_policy"]["thread_environment"])

    detector_root = detector_manifest_path.parent.resolve()
    if output == detector_root or output.is_relative_to(detector_root):
        raise ValueError("Outcome artifacts must be physically separate.")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Outcome output directory is not empty.")
    output.mkdir(parents=True, exist_ok=True)

    thresholds = load_and_validate_thresholds(artifacts)
    validate_formal_detector_artifacts(
        artifacts,
        thresholds,
        protocol_id=config["protocol_id"],
    )
    threshold_cycles = {
        method: _load_cycle_score(
            artifacts[f"threshold__{method}__empirical_cycle_score.npy"],
            expected_shape=(5_000, 51),
        )
        for method in METHOD_IDS
    }
    validate_frozen_shot_table(
        artifacts["threshold_shots.csv"],
        threshold_cycles,
        thresholds,
        phase="threshold",
        pair_index_start=5_000,
        reference_start=5_000,
        monitor_start=15_000,
        windows=None,
    )
    cycles, shot_scores = load_frozen_shot_scores(artifacts)
    validate_frozen_shot_table(
        artifacts["held_shots.csv"],
        cycles,
        thresholds,
        windows=config["event_windows"],
    )
    event_summary = load_strict_json(artifacts["event_summary_detector_only.json"])
    recomputed_event_summary = recompute_event_summary(
        cycles,
        thresholds,
        monitor_start=40_000,
        windows=config["event_windows"],
    )
    if canonical_json_bytes(event_summary) != canonical_json_bytes(
        recomputed_event_summary
    ):
        raise ValueError("Detector event summary differs from frozen cycle scores.")
    secondary_event_summary = load_strict_json(
        artifacts["secondary_event_summary_detector_only.json"]
    )
    recomputed_secondary_event_summary = recompute_event_summary(
        cycles,
        thresholds,
        monitor_start=40_000,
        windows=config["event_windows"],
        threshold_key="secondary_zero_alert_threshold",
    )
    if canonical_json_bytes(secondary_event_summary) != canonical_json_bytes(
        recomputed_secondary_event_summary
    ):
        raise ValueError(
            "Secondary detector event summary differs from frozen cycle scores."
        )

    outcome_paths = {
        name: outcome_root / name for name in config["risk_audit"]["labels"]
    }
    if set(outcome_paths) != {
        "obs_flips_actual.01",
        "obs_flips_predicted_by_correlated_matching.01",
        "obs_flips_predicted_by_pymatching.01",
    }:
        raise ValueError("Outcome filename set differs from the lock.")

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
    outcome_member_hashes = verify_outcome_zip_members(
        archive,
        outcome_paths,
        archive_member_root=archive_member_root,
    )

    # Outcome bytes are parsed only after every freeze/artifact/archive gate.
    actual = parse_dot01_outcome_slice(
        outcome_paths["obs_flips_actual.01"],
        expected_count=500_000,
        start=40_000,
        stop=60_000,
    )
    correlated = parse_dot01_outcome_slice(
        outcome_paths["obs_flips_predicted_by_correlated_matching.01"],
        expected_count=500_000,
        start=40_000,
        stop=60_000,
    )
    pymatching = parse_dot01_outcome_slice(
        outcome_paths["obs_flips_predicted_by_pymatching.01"],
        expected_count=500_000,
        start=40_000,
        stop=60_000,
    )
    labels = {
        "correlated_matching_mismatch": actual ^ correlated,
        "pymatching_mismatch": actual ^ pymatching,
    }
    archive_shots = np.arange(40_000, 60_000, dtype=np.int64)
    method_input_parity_evidence = derive_method_input_parity_evidence(
        cycles,
        labels,
        archive_shots,
        expected_shot_count=20_000,
        expected_role_count=51,
    )
    method_input_parity_evidence_sha256 = hashlib.sha256(
        canonical_json_bytes(method_input_parity_evidence)
    ).hexdigest()
    detector_manifest_hash = sha256_file(detector_manifest_path)

    outcome_table = output / "outcomes.csv"
    _write_outcome_table(
        outcome_table,
        archive_shots=archive_shots,
        actual=actual,
        correlated=correlated,
        pymatching=pymatching,
        detector_manifest_sha256=detector_manifest_hash,
    )
    point_estimates = {
        label_id: one_label_risk_metrics(
            shot_scores,
            label,
            archive_shots,
            budgets=RISK_BUDGETS,
        )
        for label_id, label in labels.items()
    }
    uncertainty = bootstrap_risk_uncertainty(
        shot_scores,
        labels,
        archive_shots,
        budgets=RISK_BUDGETS,
        replicates=2_000,
        seed_start=611_000,
        block_length=128,
    )
    risk_summary = {
        "schema_version": "run6-google-risk-summary-v1",
        "primary_label": "actual_xor_correlated_matching_prediction",
        "secondary_label": "actual_xor_pymatching_prediction",
        "ranking": "descending_frozen_shot_score_then_ascending_archive_shot",
        "budgets_shots": list(RISK_BUDGETS),
        "outcome_table_sha256": sha256_file(outcome_table),
        "detector_manifest_sha256": detector_manifest_hash,
        "point_estimates": point_estimates,
        "uncertainty": uncertainty,
        "interpretation": "retrospective_veto_or_triage_not_a_decoder",
    }
    risk_path = output / "risk_summary.json"
    _write_canonical_json(risk_path, risk_summary)
    decision = build_decision_summary(
        config=config,
        event_summary=event_summary,
        primary_risk=point_estimates["correlated_matching_mismatch"],
        uncertainty=uncertainty,
        detector_freeze_verified=True,
        method_input_parity_evidence=method_input_parity_evidence,
    )
    if (
        randomization_manifest_path is not None
        and randomization_result is not None
        and pnnl_manifest_path is not None
        and pnnl_manifest is not None
    ):
        decision = integrate_full_run6_decision(
            decision,
            randomization_manifest_path=randomization_manifest_path,
            randomization_result=randomization_result,
            pnnl_manifest_path=pnnl_manifest_path,
            pnnl_manifest=pnnl_manifest,
        )
    decision_path = output / "decision_summary.json"
    _write_canonical_json(decision_path, decision)

    manifest = {
        "schema_version": "run6-google-outcome-manifest-v1",
        "protocol_id": config["protocol_id"],
        "config_sha256": sha256_file(config_path),
        "method_spec_sha256": sha256_file(
            repo_root / config["normative_method_spec"]["path"]
        ),
        "freeze_ratification_sha256": sha256_file(ratification_path),
        "detector_manifest_sha256": detector_manifest_hash,
        "detector_manifest_git_commit": detector_manifest["git_commit"],
        "script_sha256": sha256_file(__file__),
        "git_commit": _git_commit(repo_root),
        "outcome_accessed_after_detector_freeze": True,
        "outcome_source_hashes": {
            name: sha256_file(path) for name, path in outcome_paths.items()
        },
        "verified_outcome_zip_member_sha256": outcome_member_hashes,
        "primary_label": "actual_xor_correlated_matching_prediction",
        "secondary_label": "actual_xor_pymatching_prediction",
        "method_input_parity_evidence_sha256": method_input_parity_evidence_sha256,
        "shared_outcome_label_bundle_sha256": method_input_parity_evidence[
            "shared_outcome_label_bundle"
        ]["sha256"],
        "final_aggregation_inputs": (
            {
                "status": "completed_and_hash_verified",
                "randomization_manifest_sha256": sha256_file(
                    randomization_manifest_path
                ),
                "pnnl_results_manifest_sha256": sha256_file(pnnl_manifest_path),
            }
            if randomization_manifest_path is not None
            and pnnl_manifest_path is not None
            else {
                "status": "not_run_in_this_command",
                "randomization_manifest_sha256": None,
                "pnnl_results_manifest_sha256": None,
            }
        ),
        "rng": {
            "algorithm": "numpy.random.Generator(PCG64)",
            "risk_bootstrap_seed_start": 611_000,
            "risk_bootstrap_seed_stop_exclusive": 613_000,
        },
        "environment": environment_fingerprint(),
        "command": sys.argv,
        "artifacts": [
            _artifact_record(outcome_table),
            _artifact_record(risk_path),
            _artifact_record(decision_path),
        ],
    }
    manifest_path = output / "outcome_manifest.json"
    _write_canonical_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


def main() -> None:
    args = parse_args()
    if args.dry_run:
        print(json.dumps(synthetic_dry_run(), indent=2, sort_keys=True))
        return
    run_real(args)


if __name__ == "__main__":
    main()
