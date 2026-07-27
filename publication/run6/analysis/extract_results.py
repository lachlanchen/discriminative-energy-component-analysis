#!/usr/bin/env python3
"""Validate Run 6 result artifacts and emit publication-ready summaries.

This program consumes only derived Run 6 result artifacts and provenance
metadata.  It never opens the Google ``.b8``/``.01`` sources or a PNNL
``bitstrings.json`` payload.  It is deliberately fail closed: the original
pre-access freeze, the disclosed post-detector repair chain, all completed
result manifests, their hash-bound artifacts, and the full locked decision
are required.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

GOOGLE_PROTOCOL = "run6-google2022-v2"
PNNL_PROTOCOL = "run6-pnnl-snapshot-v2"
METHOD_IDS = ("m0", "m0c", "m1", "m2", "m3", "m4", "m5", "space")
PNNL_METHOD_IDS = (
    "dfr",
    "online_logistic",
    "space_sparse",
    "space_spectral",
    "space_composite",
)
PNNL_RANDOMIZATION_REPLICATES = 256
PNNL_PATH_STATE_EPISODES = 22
GOOGLE_THRESHOLD_BOOTSTRAP_REPLICATES = 2_000
FORMAL_METHOD_IDS = ("m0", "m1", "m3", "m4", "m5", "space")
LABEL_IDS = ("correlated_matching_mismatch", "pymatching_mismatch")
RISK_BUDGETS = (2, 20, 200)
WINDOWS = {
    "primary": (57_750, 57_800),
    "narrow": (57_770, 57_780),
    "wide": (57_725, 57_825),
}
ORIGINAL_RATIFICATION_PATH = "experiments/run6/freeze_ratification.json"
REPAIR_MANIFEST_PATH = "experiments/run6/repair_manifest.json"
REPAIR_RATIFICATION_PATH = "experiments/run6/repair_ratification.json"
INCIDENT_COMMIT = "c36a3b0980588db4663f2a51692294fcdfddc9a5"
ORIGINAL_RATIFICATION_COMMIT = "7e378d7f1d99818fc5e366bb14a7200767722d6c"
ORIGINAL_IMPLEMENTATION_COMMIT = "edb14d5b671ea9346b9953e211e8447618b585cb"
ORIGINAL_FREEZE_COMMIT = "eabc640d446c239f0c84d4f7d8f6a9b256fca72f"
REPAIR_IMPLEMENTATION_COMMIT = "1bd2d3af1e5a74b791f46c4034cda02cf8118b61"
REPAIR_MANIFEST_COMMIT = "fc92f7bda9f9711e8f2439d5d95c3330e4361bf1"
REPAIR_RATIFICATION_COMMIT = "eb2df6a4d8962e2c561a7d4dbb242c464b033d8c"

PRODUCTION_EVIDENCE_PATHS = {
    "detector_manifest": (
        "experiments/run6/results/google_detector/detector_freeze_manifest.json"
    ),
    "freeze_ratification": ORIGINAL_RATIFICATION_PATH,
    "repair_manifest": REPAIR_MANIFEST_PATH,
    "repair_ratification": REPAIR_RATIFICATION_PATH,
    "randomization_manifest": (
        "experiments/run6/results/google_randomization_repair1/merged/"
        "randomization_manifest.json"
    ),
    "pnnl_manifest": ("experiments/run6/results/pnnl_snapshot/results_manifest.json"),
    "pittsburgh_manifest": ("experiments/run6/configs/pnnl_pittsburgh_locked.json"),
    "outcome_manifest": (
        "experiments/run6/results/google_outcomes/outcome_manifest.json"
    ),
}


@dataclass(frozen=True)
class ValidationProfile:
    """Immutable anchors used by provenance validation.

    Production constructs this profile only from hardcoded Git commits.
    Tests may inject a private hermetic profile through Python APIs; no CLI
    argument exposes that route.
    """

    original_ratification_bytes: bytes
    repair_manifest_bytes: bytes
    repair_ratification_bytes: bytes
    implementation_hashes: tuple[tuple[str, str], ...]
    evidence_paths: tuple[tuple[str, str], ...]
    repository_root: Path | None

    def implementation_hash_map(self) -> dict[str, str]:
        return dict(self.implementation_hashes)

    def evidence_path_map(self) -> dict[str, str]:
        return dict(self.evidence_paths)


REPAIR_DIFF_STATUS = {
    "experiments/aoc/run6_repair.py": "A",
    "experiments/run6/scripts/create_repair_chain.py": "A",
    "experiments/run6/scripts/launch_google2022_randomization.py": "M",
    "experiments/run6/scripts/run_google2022_outcomes.py": "M",
    "experiments/run6/scripts/run_google2022_randomization.py": "M",
    "experiments/run6/scripts/run_pnnl_snapshot.py": "M",
    "experiments/run6/tests/test_google2022_outcomes.py": "M",
    "experiments/run6/tests/test_google2022_randomization.py": "M",
    "experiments/run6/tests/test_google2022_randomization_launcher.py": "M",
    "experiments/run6/tests/test_google_detector_runner.py": "M",
    "experiments/run6/tests/test_pnnl_snapshot_runner.py": "M",
    "experiments/run6/tests/test_run6_repair.py": "A",
    "references/run6_post_detector_schema_incident.md": "M",
}
REPAIR_ACCESS_RECORD = {
    "detector_values_accessed_before_repair": True,
    "detector_numeric_values_exposed_during_incident_diagnosis": True,
    "detector_numeric_values_used_to_select_repair": False,
    "decoder_outcomes_accessed_before_repair": False,
    "pnnl_held_payload_accessed_before_repair": False,
    "completed_randomization_replicates_before_repair": 0,
    "randomization_shard_manifests_before_repair": 0,
    "failed_attempts_preserved": True,
    "detector_artifacts_reused_without_modification": True,
    "detector_rerun_performed_for_repair": False,
}
REPAIR_THREAD_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}
REPAIR_PYTHON_LOCK_SHA256 = (
    "9c99a789e3202f93a8c6d5c517fb4fdbc4f9ee0ebcb30619648a37441c841d38"
)
FAILED_ATTEMPT_STDERR_SHA256 = (
    "d3bc72d114336901c1b502f2722cc3e4a7c44030f03fa38e7f861fc8c5e6dd3e"
)
EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
REPAIR_RUNTIME_MODULE_ORIGINS = {
    "aoc": "experiments/aoc/__init__.py",
    "aoc.qec_real": "experiments/aoc/qec_real.py",
    "aoc.run6_protocol": "experiments/aoc/run6_protocol.py",
    "aoc.run6_repair": "experiments/aoc/run6_repair.py",
    "aoc.space": "experiments/aoc/space.py",
    "aoc.space_qec": "experiments/aoc/space_qec.py",
}

METHOD_METADATA = {
    "m0": ("DFR", "Predesignated comparator"),
    "m0c": ("Within-shot Page CUSUM", "Mandatory contextual control"),
    "m1": ("Diagonal likelihood", "Mandatory contextual control"),
    "m2": ("Hotelling", "Mandatory contextual control"),
    "m3": ("Online logistic", "Predesignated comparator"),
    "m4": ("S-PACE sparse", "Preregistered component"),
    "m5": ("S-PACE spectral", "Preregistered component"),
    "space": ("S-PACE composite", "Primary proposed method"),
}
PNNL_METHOD_METADATA = {
    "dfr": ("DFR", "Predesignated comparator"),
    "online_logistic": ("Online logistic", "Predesignated comparator"),
    "space_sparse": ("S-PACE sparse", "Preregistered component"),
    "space_spectral": ("S-PACE spectral", "Preregistered component"),
    "space_composite": ("S-PACE composite", "Primary auxiliary target"),
}

ATOMIC_PREDICATES = (
    "fixed_space_composite_alerts_inside_primary_event_window_at_locked_primary_threshold",
    "fixed_space_composite_has_at_most_9_pre_event_alerts",
    "detector_scores_thresholds_and_resource_ledger_frozen_before_outcome_access",
    "space_top20_primary_mismatch_capture_at_least_dfr_plus_1",
    "space_top20_primary_mismatch_capture_at_least_online_logistic_plus_1",
    "all_window_sensitivity_and_uncertainty_results_reported",
    "no_method_received_extra_detector_records_or_outcome_labels",
)
PREDICATE_LABELS = {
    ATOMIC_PREDICATES[0]: "S-PACE alerts in the primary event window",
    ATOMIC_PREDICATES[1]: "S-PACE has at most nine pre-event alerts",
    ATOMIC_PREDICATES[2]: "Detector and resource artifacts frozen before labels",
    ATOMIC_PREDICATES[3]: "Top-20 capture is at least DFR plus one",
    ATOMIC_PREDICATES[4]: "Top-20 capture is at least online logistic plus one",
    ATOMIC_PREDICATES[5]: "All windows and locked uncertainty are reported",
    ATOMIC_PREDICATES[6]: "All methods receive the same records and labels",
}

ARTIFACT_KEYS = frozenset({"path", "bytes", "sha256"})
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
OUTCOME_MANIFEST_KEYS = frozenset(
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
        "outcome_accessed_after_detector_freeze",
        "outcome_source_hashes",
        "verified_outcome_zip_member_sha256",
        "primary_label",
        "secondary_label",
        "method_input_parity_evidence_sha256",
        "shared_outcome_label_bundle_sha256",
        "final_aggregation_inputs",
        "rng",
        "environment",
        "command",
        "artifacts",
    }
)
RANDOMIZATION_MANIFEST_KEYS = frozenset(
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
        "execution_mode",
        "merge_evidence",
        "environment",
        "command",
        "artifacts",
        "resources",
    }
)
PNNL_MANIFEST_KEYS = frozenset(
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
        "repair_ratification_path",
        "repair_ratification_sha256",
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
GOOGLE_THRESHOLD_BOOTSTRAP_KEYS = frozenset(
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
    }
)
GOOGLE_THRESHOLD_BOOTSTRAP_SUMMARY_KEYS = frozenset(
    {
        "frozen_threshold",
        "selected_primary_threshold_percentiles",
        "selected_primary_alert_count_frequency",
        "alert_count_at_frozen_threshold_percentiles",
        "alert_count_at_frozen_threshold_frequency",
        "selected_zero_alert_threshold_percentiles",
    }
)
GOOGLE_THRESHOLD_BOOTSTRAP_REPLICATE_KEYS = frozenset(
    {"replicate_index", "seed", "methods"}
)
GOOGLE_THRESHOLD_BOOTSTRAP_METHOD_KEYS = frozenset(
    {
        "selected_primary_threshold",
        "selected_primary_alert_count",
        "alert_count_at_frozen_threshold",
        "selected_zero_alert_threshold",
        "selected_zero_alert_count",
    }
)
PERCENTILE_SUMMARY_KEYS = frozenset({"lower_2_5", "median", "upper_97_5"})
PNNL_RANDOMIZATION_KEYS = frozenset(
    {
        "schema_version",
        "seeds",
        "method_order",
        "path_state_method_rows",
        "overall_episode_alarm_fraction",
        "alarmed_episode_count_histogram",
        "maximum_log_e_summary",
        "claim_scope",
    }
)
PNNL_RANDOMIZATION_ROW_KEYS = frozenset(
    {
        "cohort_index",
        "cohort_id",
        "logical_state",
        "method",
        "alarm_fraction",
        "maximum_log_e_over_replicates",
    }
)
PNNL_MAXIMUM_SUMMARY_KEYS = frozenset({"minimum", "median", "maximum"})

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
RISK_SUMMARY_KEYS = frozenset(
    {
        "schema_version",
        "primary_label",
        "secondary_label",
        "ranking",
        "budgets_shots",
        "outcome_table_sha256",
        "detector_manifest_sha256",
        "point_estimates",
        "uncertainty",
        "interpretation",
    }
)
DECISION_KEYS = frozenset(
    {
        "schema_version",
        "repair_ratification_path",
        "repair_ratification_sha256",
        "summary_scope",
        "primary_label",
        "primary_budget_shots",
        "top20_capture",
        "atomic_predicates",
        "method_input_parity_evidence",
        "mandatory_contextual_controls_reported",
        "google_primary_pass",
        "randomization_audit",
        "pnnl_retention_pass",
        "pnnl_results_manifest_sha256",
        "overall_run6_advantage",
        "negative_result_reasons",
        "bootstrap_changes_primary_boolean",
    }
)
ORIGINAL_RATIFICATION_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "freeze_commit",
        "hashes",
        "environment",
        "thread_environment",
        "held_value_access_before_ratification",
    }
)
REPAIR_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "incident_commit",
        "original_ratification_commit",
        "implementation_commit",
        "repair_diff",
        "hashes",
        "original_freeze",
        "detector_evidence",
        "failed_attempt_evidence",
        "access_record",
        "environment",
        "thread_environment",
        "python_environment_lock_sha256",
        "runtime_module_origins",
    }
)
REPAIR_RATIFICATION_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "repair_manifest_commit",
        "hashes",
        "original_ratification_sha256",
        "detector_manifest_sha256",
        "access_record",
        "environment",
        "thread_environment",
        "python_environment_lock_sha256",
    }
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
PNNL_COHORT_FIELDS = (
    "cohort_id",
    "distance",
    "rounds",
    "basis",
    "register_suffix",
    "data_qubits",
    "syndrome_qubits",
    "oriented_path",
    "early_snapshot_id",
    "late_snapshot_id",
    "m",
    "raw_qasm_pair_identical",
    "normalized_qasm_pair_identical_audit_only",
    "claim_label",
    "calibration_pair_id",
)
PNNL_ARTIFACT_TUPLE_SCHEMAS = {
    "metadata": (
        "distance",
        "rounds",
        "basis",
        "shots_per_state",
        "n_chains",
        "backend_property_date_source",
        "backend_property_date_utc",
    ),
    "info": ("bytes", "raw_sha256", "python_canonical_sha256"),
    "calibration": ("bytes", "raw_sha256", "python_canonical_sha256"),
    "qasm_state": ("bytes", "raw_sha256", "normalized_sha256"),
    "qasm_pair": ("raw_pair_sha256", "normalized_pair_sha256"),
    "held_bitstrings": (
        "bytes_from_stat_only",
        "content_read_for_manifest",
        "sha256_before_unblinding",
    ),
}


class PublicationDataError(ValueError):
    """Raised when a publication bundle cannot be supported by locked results."""


def _reject_json_constant(value: str) -> None:
    raise PublicationDataError(f"Non-finite JSON constant is forbidden: {value}")


def _reject_duplicate_pairs(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PublicationDataError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(path: Path) -> Any:
    """Load strict JSON with duplicate and non-finite values rejected."""

    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(
                handle,
                parse_constant=_reject_json_constant,
                object_pairs_hook=_reject_duplicate_pairs,
            )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PublicationDataError(
            f"Cannot load strict JSON {path}: {error}"
        ) from error


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_json_bytes(value: bytes, *, context: str) -> Any:
    try:
        return json.loads(
            value.decode("utf-8"),
            parse_constant=_reject_json_constant,
            object_pairs_hook=_reject_duplicate_pairs,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise PublicationDataError(f"Cannot load strict {context}: {error}") from error


def require_canonical_repository_path(value: Any, *, context: str) -> str:
    if not isinstance(value, str):
        raise PublicationDataError(f"{context} must be text.")
    candidate = PurePosixPath(value)
    if (
        not value
        or candidate.is_absolute()
        or candidate.as_posix() != value
        or any(part in {"", ".", ".."} for part in candidate.parts)
        or "\\" in value
    ):
        raise PublicationDataError(
            f"{context} must be a canonical repository-relative POSIX path."
        )
    return value


def require_canonical_repository_file(
    path: Path,
    *,
    repository_root: Path,
    relative: str,
    context: str,
) -> Path:
    """Require one exact in-repository file with no symlinked path component."""

    require_canonical_repository_path(relative, context=f"{context} path")
    try:
        root = repository_root.resolve(strict=True)
    except OSError as error:
        raise PublicationDataError(
            f"{context} repository root cannot be resolved: {error}"
        ) from error
    lexical_root = Path(os.path.abspath(repository_root))
    if lexical_root != root or repository_root.is_symlink():
        raise PublicationDataError(
            f"{context} repository root must itself be canonical and non-symlinked."
        )

    declared = root / relative
    if Path(os.path.abspath(path)) != declared:
        raise PublicationDataError(
            f"{context} must use canonical repository path {relative}."
        )

    component = root
    for part in PurePosixPath(relative).parts:
        component /= part
        if component.is_symlink():
            raise PublicationDataError(
                f"{context} canonical path contains a symlink component: {relative}."
            )
    try:
        resolved = declared.resolve(strict=True)
    except OSError as error:
        raise PublicationDataError(
            f"{context} canonical repository file cannot be resolved: {error}"
        ) from error
    if not resolved.is_relative_to(root):
        raise PublicationDataError(
            f"{context} canonical repository file escapes the repository root."
        )
    if resolved != declared or not resolved.is_file():
        raise PublicationDataError(
            f"{context} must be a regular non-symlink repository file."
        )
    return resolved


def git_blob(repository_root: Path, commit: str, path: str) -> bytes:
    """Read one path from an immutable commit, failing closed."""

    require_commit(commit, context="Git anchor commit")
    require_canonical_repository_path(path, context="Git anchor path")
    completed = subprocess.run(
        ["git", "-C", str(repository_root), "show", f"{commit}:{path}"],
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise PublicationDataError(
            f"Cannot resolve immutable Git anchor {commit}:{path}: {detail}"
        )
    return completed.stdout


def _make_internal_validation_profile(
    *,
    original_ratification_bytes: bytes,
    repair_manifest_bytes: bytes,
    repair_ratification_bytes: bytes,
    implementation_hashes: Mapping[str, str],
    evidence_paths: Mapping[str, str],
) -> ValidationProfile:
    """Construct a hermetic test profile; production CLI never calls this."""

    return ValidationProfile(
        original_ratification_bytes=bytes(original_ratification_bytes),
        repair_manifest_bytes=bytes(repair_manifest_bytes),
        repair_ratification_bytes=bytes(repair_ratification_bytes),
        implementation_hashes=tuple(sorted(implementation_hashes.items())),
        evidence_paths=tuple(sorted(evidence_paths.items())),
        repository_root=None,
    )


def load_production_validation_profile() -> ValidationProfile:
    """Load hardcoded production anchors from immutable Git objects."""

    repository_root = Path(__file__).resolve().parents[3]
    original_bytes = git_blob(
        repository_root,
        ORIGINAL_RATIFICATION_COMMIT,
        ORIGINAL_RATIFICATION_PATH,
    )
    repair_manifest_bytes = git_blob(
        repository_root,
        REPAIR_MANIFEST_COMMIT,
        REPAIR_MANIFEST_PATH,
    )
    repair_ratification_bytes = git_blob(
        repository_root,
        REPAIR_RATIFICATION_COMMIT,
        REPAIR_RATIFICATION_PATH,
    )
    implementation_hashes = {
        path: sha256_bytes(
            git_blob(repository_root, REPAIR_IMPLEMENTATION_COMMIT, path)
        )
        for path in REPAIR_DIFF_STATUS
    }
    canonical_manifest = require_mapping(
        load_json_bytes(repair_manifest_bytes, context="Git repair manifest"),
        context="Git repair manifest",
    )
    if canonical_manifest.get("hashes") != implementation_hashes:
        raise PublicationDataError(
            "Git repair manifest hashes do not match implementation-commit blobs."
        )
    return ValidationProfile(
        original_ratification_bytes=original_bytes,
        repair_manifest_bytes=repair_manifest_bytes,
        repair_ratification_bytes=repair_ratification_bytes,
        implementation_hashes=tuple(sorted(implementation_hashes.items())),
        evidence_paths=tuple(sorted(PRODUCTION_EVIDENCE_PATHS.items())),
        repository_root=repository_root,
    )


def evidence_input_records(
    paths: Mapping[str, Path],
    *,
    profile: ValidationProfile,
) -> dict[str, dict[str, str]]:
    """Validate source locations and return exact path/hash bundle records."""

    expected_paths = profile.evidence_path_map()
    if set(paths) != set(PRODUCTION_EVIDENCE_PATHS) or set(expected_paths) != set(
        PRODUCTION_EVIDENCE_PATHS
    ):
        raise PublicationDataError("Evidence roles differ from the locked eight roles.")
    records: dict[str, dict[str, str]] = {}
    for role in sorted(paths):
        relative = require_canonical_repository_path(
            expected_paths[role],
            context=f"evidence path for {role}",
        )
        if profile.repository_root is not None:
            observed = require_canonical_repository_file(
                paths[role],
                repository_root=profile.repository_root,
                relative=relative,
                context=role,
            )
        else:
            declared = paths[role]
            if declared.is_symlink() or not declared.is_file():
                raise PublicationDataError(
                    f"{role} must be a regular non-symlink file."
                )
            observed = declared.resolve(strict=True)
        records[role] = {"path": relative, "sha256": sha256_file(observed)}
    return records


def require_mapping(value: Any, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PublicationDataError(f"{context} must be an object.")
    return value


def require_exact_keys(
    value: Mapping[str, Any],
    expected: Iterable[str],
    *,
    context: str,
) -> None:
    expected_set = set(expected)
    observed = set(value)
    if observed != expected_set:
        missing = sorted(expected_set - observed)
        extra = sorted(observed - expected_set)
        raise PublicationDataError(
            f"{context} schema mismatch; missing={missing}, extra={extra}."
        )


def require_bool(value: Any, *, context: str) -> bool:
    if not isinstance(value, bool):
        raise PublicationDataError(f"{context} must be Boolean.")
    return value


def require_int(value: Any, *, context: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PublicationDataError(f"{context} must be an integer.")
    if minimum is not None and value < minimum:
        raise PublicationDataError(f"{context} must be at least {minimum}.")
    return value


def require_number(value: Any, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PublicationDataError(f"{context} must be numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise PublicationDataError(f"{context} must be finite.")
    return result


def require_sha256(value: Any, *, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise PublicationDataError(f"{context} must be a lowercase SHA-256.")
    return value


def require_commit(value: Any, *, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise PublicationDataError(f"{context} must be a full lowercase Git commit.")
    return value


def validate_hash_registry(value: Any, *, context: str) -> dict[str, str]:
    rows = require_mapping(value, context=context)
    if not rows:
        raise PublicationDataError(f"{context} must not be empty.")
    result: dict[str, str] = {}
    for path, digest in rows.items():
        if (
            not isinstance(path, str)
            or not path
            or PurePosixPath(path).is_absolute()
            or PurePosixPath(path).as_posix() != path
            or any(part in {"", ".", ".."} for part in PurePosixPath(path).parts)
            or "\\" in path
        ):
            raise PublicationDataError(f"{context} contains a noncanonical path.")
        result[path] = require_sha256(digest, context=f"{context}.{path}")
    return result


def validate_original_ratification(
    path: Path,
    *,
    detector_manifest: Mapping[str, Any],
    profile: ValidationProfile,
) -> Mapping[str, Any]:
    ratification = require_mapping(
        load_json(path), context="original freeze ratification"
    )
    expected = require_mapping(
        load_json_bytes(
            profile.original_ratification_bytes,
            context="anchored original freeze ratification",
        ),
        context="anchored original freeze ratification",
    )
    require_exact_keys(
        ratification,
        ORIGINAL_RATIFICATION_KEYS,
        context="original freeze ratification",
    )
    if (
        ratification["schema_version"] != "run6-freeze-ratification-v1"
        or ratification["status"] != "frozen_before_held_value_access"
        or ratification["held_value_access_before_ratification"] is not False
        or ratification["freeze_commit"] != ORIGINAL_FREEZE_COMMIT
        or detector_manifest["freeze_ratification_sha256"] != sha256_file(path)
        or sha256_file(path) != sha256_bytes(profile.original_ratification_bytes)
        or ratification != expected
    ):
        raise PublicationDataError(
            "Detector does not bind the valid original pre-access ratification."
        )
    validate_hash_registry(
        ratification["hashes"], context="original ratification hashes"
    )
    require_mapping(
        ratification["environment"], context="original ratification environment"
    )
    threads = require_mapping(
        ratification["thread_environment"],
        context="original ratification thread environment",
    )
    if dict(threads) != REPAIR_THREAD_ENVIRONMENT:
        raise PublicationDataError("Original ratification thread lock changed.")
    return ratification


def _detector_registry_for_repair(
    detector_manifest: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    prefix = "experiments/run6/results/google_detector"
    registry: dict[str, dict[str, Any]] = {}
    rows = detector_manifest["artifacts"]
    if not isinstance(rows, list):
        raise PublicationDataError("Detector artifacts must be a list.")
    for index, raw in enumerate(rows):
        row = require_mapping(raw, context=f"detector artifact[{index}]")
        require_exact_keys(row, ARTIFACT_KEYS, context=f"detector artifact[{index}]")
        relative = row["path"]
        if not isinstance(relative, str) or PurePosixPath(relative).name != relative:
            raise PublicationDataError("Detector repair registry requires flat paths.")
        registry[f"{prefix}/{relative}"] = {
            "bytes": require_int(
                row["bytes"],
                context=f"detector artifact[{index}].bytes",
                minimum=0,
            ),
            "sha256": require_sha256(
                row["sha256"], context=f"detector artifact[{index}].sha256"
            ),
        }
    return registry


def validate_repair_manifest(
    path: Path,
    *,
    original_ratification_path: Path,
    original_ratification: Mapping[str, Any],
    detector_path: Path,
    detector_manifest: Mapping[str, Any],
    profile: ValidationProfile,
) -> Mapping[str, Any]:
    manifest = require_mapping(load_json(path), context="repair manifest")
    expected_manifest = require_mapping(
        load_json_bytes(
            profile.repair_manifest_bytes,
            context="anchored repair manifest",
        ),
        context="anchored repair manifest",
    )
    require_exact_keys(manifest, REPAIR_MANIFEST_KEYS, context="repair manifest")
    if (
        manifest["schema_version"] != "run6-post-detector-repair-manifest-v1"
        or manifest["status"]
        != "post_detector_pre_outcome_repair_implementation_frozen"
        or manifest["incident_commit"] != INCIDENT_COMMIT
        or manifest["original_ratification_commit"] != ORIGINAL_RATIFICATION_COMMIT
        or manifest["implementation_commit"] != REPAIR_IMPLEMENTATION_COMMIT
        or manifest["repair_diff"] != REPAIR_DIFF_STATUS
        or manifest["access_record"] != REPAIR_ACCESS_RECORD
    ):
        raise PublicationDataError(
            "Repair manifest is not the exact disclosed post-detector amendment."
        )
    hashes = validate_hash_registry(
        manifest["hashes"], context="repair implementation hashes"
    )
    anchored_implementation_hashes = profile.implementation_hash_map()
    if (
        set(hashes) != set(REPAIR_DIFF_STATUS)
        or hashes != anchored_implementation_hashes
        or hashes != expected_manifest["hashes"]
    ):
        raise PublicationDataError(
            "Repair implementation hashes differ from immutable implementation "
            "commit blobs."
        )

    original = require_mapping(
        manifest["original_freeze"], context="repair original freeze"
    )
    require_exact_keys(
        original,
        {
            "implementation_commit",
            "freeze_commit",
            "freeze_manifest_sha256",
            "ratification_commit",
            "ratification_path",
            "ratification_sha256",
        },
        context="repair original freeze",
    )
    if (
        original["implementation_commit"] != ORIGINAL_IMPLEMENTATION_COMMIT
        or original["freeze_commit"] != ORIGINAL_FREEZE_COMMIT
        or original["ratification_commit"] != ORIGINAL_RATIFICATION_COMMIT
        or original["ratification_path"] != ORIGINAL_RATIFICATION_PATH
        or original["ratification_sha256"] != sha256_file(original_ratification_path)
        or original != expected_manifest["original_freeze"]
    ):
        raise PublicationDataError("Repair manifest changed the original freeze.")
    require_sha256(
        original["freeze_manifest_sha256"],
        context="original freeze-manifest hash",
    )

    evidence = require_mapping(
        manifest["detector_evidence"], context="repair detector evidence"
    )
    require_exact_keys(
        evidence,
        {
            "manifest_path",
            "manifest_bytes",
            "manifest_sha256",
            "detector_only",
            "outcome_accessed",
            "outcome_join_authorized",
            "held_joint_replay_digest_count",
            "held_joint_replay_all_identical",
            "artifact_count",
            "artifacts",
        },
        context="repair detector evidence",
    )
    artifact_rows = require_mapping(
        evidence["artifacts"], context="repair detector artifact registry"
    )
    expected_registry = _detector_registry_for_repair(detector_manifest)
    if (
        evidence["manifest_path"]
        != "experiments/run6/results/google_detector/detector_freeze_manifest.json"
        or evidence["manifest_bytes"] != detector_path.stat().st_size
        or evidence["manifest_sha256"] != sha256_file(detector_path)
        or evidence["detector_only"] is not True
        or evidence["outcome_accessed"] is not False
        or evidence["outcome_join_authorized"] is not False
        or evidence["held_joint_replay_digest_count"] != 3
        or evidence["held_joint_replay_all_identical"] is not True
        or evidence["artifact_count"] != 231
        or dict(artifact_rows) != expected_registry
        or evidence != expected_manifest["detector_evidence"]
    ):
        raise PublicationDataError(
            "Repair manifest does not bind the immutable detector bundle."
        )

    failures = require_mapping(
        manifest["failed_attempt_evidence"], context="repair failure evidence"
    )
    require_exact_keys(
        failures,
        {
            "root",
            "attempt_count",
            "attempt_shard_ranges",
            "file_count",
            "files",
            "common_stderr_sha256",
            "all_stderr_logs_byte_identical",
            "all_stdout_logs_empty",
            "shard_manifest_count",
            "completed_randomization_replicates",
            "empty_result_directory_count",
            "empty_result_directories",
        },
        context="repair failure evidence",
    )
    expected_failures = require_mapping(
        expected_manifest["failed_attempt_evidence"],
        context="anchored repair failure evidence",
    )
    failure_files = require_mapping(failures["files"], context="repair failure files")
    if (
        failures["root"] != "experiments/run6/results/google_randomization"
        or failures["attempt_count"] != 32
        or failures["attempt_shard_ranges"]
        != [[start, start + 8] for start in range(0, 256, 8)]
        or failures["file_count"] != 64
        or len(failure_files) != 64
        or failures["all_stderr_logs_byte_identical"] is not True
        or failures["all_stdout_logs_empty"] is not True
        or failures["shard_manifest_count"] != 0
        or failures["completed_randomization_replicates"] != 0
        or failures["empty_result_directory_count"] != 32
        or failures != expected_failures
    ):
        raise PublicationDataError("Repair failure chronology changed.")
    common_stderr_sha256 = require_sha256(
        failures["common_stderr_sha256"],
        context="common failed-attempt stderr hash",
    )
    if common_stderr_sha256 != FAILED_ATTEMPT_STDERR_SHA256:
        raise PublicationDataError("Common failed-attempt stderr digest changed.")
    expected_failure_files = require_mapping(
        expected_failures["files"], context="anchored repair failure files"
    )
    if set(failure_files) != set(expected_failure_files):
        raise PublicationDataError("Repair failed-attempt filenames changed.")
    for name, raw in failure_files.items():
        row = require_mapping(raw, context=f"repair failure file {name}")
        require_exact_keys(row, {"bytes", "sha256"}, context=f"failure file {name}")
        observed_bytes = require_int(
            row["bytes"], context=f"failure file {name}.bytes", minimum=0
        )
        observed_sha256 = require_sha256(
            row["sha256"], context=f"failure file {name}.sha256"
        )
        expected_row = expected_failure_files[name]
        if (
            row != expected_row
            or (
                name.endswith("/stderr.log")
                and (observed_bytes != 891 or observed_sha256 != common_stderr_sha256)
            )
            or (
                name.endswith("/stdout.log")
                and (observed_bytes != 0 or observed_sha256 != EMPTY_SHA256)
            )
        ):
            raise PublicationDataError("Repair failed-attempt file evidence changed.")

    environment = require_mapping(manifest["environment"], context="repair environment")
    threads = require_mapping(
        manifest["thread_environment"], context="repair thread environment"
    )
    if (
        dict(environment) != dict(original_ratification["environment"])
        or environment != expected_manifest["environment"]
        or dict(threads) != REPAIR_THREAD_ENVIRONMENT
        or threads != expected_manifest["thread_environment"]
    ):
        raise PublicationDataError("Repair environment or thread lock changed.")
    python_lock = require_sha256(
        manifest["python_environment_lock_sha256"],
        context="repair Python environment lock",
    )
    original_hashes = validate_hash_registry(
        original_ratification["hashes"],
        context="original ratification hashes for repair",
    )
    if (
        python_lock
        != original_hashes.get("experiments/run6/configs/python_environment_lock.txt")
        or python_lock != REPAIR_PYTHON_LOCK_SHA256
        or python_lock != expected_manifest["python_environment_lock_sha256"]
    ):
        raise PublicationDataError("Repair Python environment-lock binding changed.")
    origins = require_mapping(
        manifest["runtime_module_origins"], context="repair runtime module origins"
    )
    if (
        origins != REPAIR_RUNTIME_MODULE_ORIGINS
        or origins != expected_manifest["runtime_module_origins"]
    ):
        raise PublicationDataError("Repair runtime-module origins changed.")
    if (
        sha256_file(path) != sha256_bytes(profile.repair_manifest_bytes)
        or manifest != expected_manifest
    ):
        raise PublicationDataError(
            "Repair manifest differs from its immutable Git-committed anchor."
        )
    return manifest


def validate_repair_ratification(
    path: Path,
    *,
    repair_manifest_path: Path,
    repair_manifest: Mapping[str, Any],
    original_ratification_path: Path,
    detector_path: Path,
    profile: ValidationProfile,
) -> Mapping[str, Any]:
    ratification = require_mapping(load_json(path), context="repair ratification")
    expected_ratification = require_mapping(
        load_json_bytes(
            profile.repair_ratification_bytes,
            context="anchored repair ratification",
        ),
        context="anchored repair ratification",
    )
    require_exact_keys(
        ratification,
        REPAIR_RATIFICATION_KEYS,
        context="repair ratification",
    )
    hashes = validate_hash_registry(
        ratification["hashes"], context="repair ratification hashes"
    )
    expected_hash_paths = set(REPAIR_DIFF_STATUS) | {REPAIR_MANIFEST_PATH}
    if (
        ratification["schema_version"] != "run6-post-detector-repair-ratification-v1"
        or ratification["status"] != "post_detector_pre_outcome_repair_ratified"
        or ratification["repair_manifest_commit"] != REPAIR_MANIFEST_COMMIT
        or sha256_file(path) != sha256_bytes(profile.repair_ratification_bytes)
        or ratification != expected_ratification
        or set(hashes) != expected_hash_paths
        or hashes[REPAIR_MANIFEST_PATH] != sha256_file(repair_manifest_path)
        or any(
            hashes[artifact] != repair_manifest["hashes"][artifact]
            for artifact in REPAIR_DIFF_STATUS
        )
        or ratification["original_ratification_sha256"]
        != sha256_file(original_ratification_path)
        or ratification["detector_manifest_sha256"] != sha256_file(detector_path)
        or ratification["access_record"] != REPAIR_ACCESS_RECORD
        or ratification["access_record"] != repair_manifest["access_record"]
        or ratification["environment"] != repair_manifest["environment"]
        or ratification["thread_environment"] != repair_manifest["thread_environment"]
        or ratification["python_environment_lock_sha256"]
        != repair_manifest["python_environment_lock_sha256"]
    ):
        raise PublicationDataError(
            "Repair ratification does not recursively bind the repair evidence."
        )
    return ratification


def close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)


def resolve_artifact(record: Any, manifest_path: Path, *, context: str) -> Path:
    """Resolve and hash-check one canonical artifact beside its manifest."""

    row = require_mapping(record, context=context)
    require_exact_keys(row, ARTIFACT_KEYS, context=context)
    raw_path = row["path"]
    if not isinstance(raw_path, str):
        raise PublicationDataError(f"{context}.path must be a string.")
    relative = PurePosixPath(raw_path)
    if (
        not raw_path
        or relative.is_absolute()
        or relative.as_posix() != raw_path
        or any(part in {"", ".", ".."} for part in relative.parts)
        or "\\" in raw_path
    ):
        raise PublicationDataError(
            f"{context}.path must be canonical relative POSIX text."
        )
    if relative.name == "bitstrings.json" or relative.suffix in {".b8", ".01"}:
        raise PublicationDataError(
            f"{context} points to a forbidden raw-payload filename."
        )
    candidate = (manifest_path.parent / relative).resolve()
    root = manifest_path.parent.resolve()
    if not candidate.is_relative_to(root):
        raise PublicationDataError(f"{context} escapes its result directory.")
    if not candidate.is_file():
        raise PublicationDataError(f"{context} is missing: {candidate}")
    expected_bytes = require_int(row["bytes"], context=f"{context}.bytes", minimum=0)
    expected_hash = require_sha256(row["sha256"], context=f"{context}.sha256")
    if candidate.stat().st_size != expected_bytes:
        raise PublicationDataError(f"{context} byte count changed.")
    if sha256_file(candidate) != expected_hash:
        raise PublicationDataError(f"{context} SHA-256 changed.")
    return candidate


def artifact_map(
    records: Any,
    manifest_path: Path,
    *,
    context: str,
) -> dict[str, Path]:
    if not isinstance(records, list):
        raise PublicationDataError(f"{context} must be a list.")
    result: dict[str, Path] = {}
    for index, record in enumerate(records):
        path = resolve_artifact(
            record,
            manifest_path,
            context=f"{context}[{index}]",
        )
        name = path.relative_to(manifest_path.parent.resolve()).as_posix()
        if name in result:
            raise PublicationDataError(f"Duplicate artifact path in {context}: {name}")
        result[name] = path
    return result


def expected_detector_artifact_names() -> set[str]:
    threshold_ids = (
        "empirical_cycle_score",
        "above_threshold",
        "notification_emitted",
        "cooldown_active",
    )
    held_ids = (
        *threshold_ids,
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
        for array_id in threshold_ids:
            stem = f"threshold__{method}__{array_id}"
            names.update({f"{stem}.npy", f"{stem}.json"})
        for array_id in held_ids:
            stem = f"held__{method}__{array_id}"
            names.update({f"{stem}.npy", f"{stem}.json"})
        for array_id in ("candidate_threshold", "shot_alert_count"):
            stem = f"threshold__{method}__frontier_{array_id}"
            names.update({f"{stem}.npy", f"{stem}.json"})
    return names


def validate_event_summary(value: Any, *, context: str) -> Mapping[str, Any]:
    summary = require_mapping(value, context=context)
    require_exact_keys(summary, METHOD_IDS, context=context)
    for method in METHOD_IDS:
        row = require_mapping(summary[method], context=f"{context}.{method}")
        require_exact_keys(
            row,
            {"pre_event_alert_count", "pre_event_alert_shots", "windows"},
            context=f"{context}.{method}",
        )
        count = require_int(
            row["pre_event_alert_count"],
            context=f"{context}.{method}.pre_event_alert_count",
            minimum=0,
        )
        shots = row["pre_event_alert_shots"]
        if not isinstance(shots, list) or len(shots) != count:
            raise PublicationDataError(
                f"{context}.{method} pre-event list/count mismatch."
            )
        parsed_shots = [
            require_int(
                shot,
                context=f"{context}.{method}.pre_event_alert_shots",
                minimum=40_000,
            )
            for shot in shots
        ]
        if parsed_shots != sorted(set(parsed_shots)) or any(
            shot >= WINDOWS["primary"][0] for shot in parsed_shots
        ):
            raise PublicationDataError(
                f"{context}.{method} has invalid pre-event alert shots."
            )
        windows = require_mapping(row["windows"], context=f"{context}.{method}.windows")
        require_exact_keys(windows, WINDOWS, context=f"{context}.{method}.windows")
        for window, (start, stop) in WINDOWS.items():
            cell = require_mapping(
                windows[window],
                context=f"{context}.{method}.windows.{window}",
            )
            require_exact_keys(
                cell,
                {"detected", "first_alert_shot", "first_alert_role"},
                context=f"{context}.{method}.windows.{window}",
            )
            detected = require_bool(
                cell["detected"],
                context=f"{context}.{method}.windows.{window}.detected",
            )
            shot = cell["first_alert_shot"]
            role = cell["first_alert_role"]
            if detected:
                shot_value = require_int(
                    shot,
                    context=f"{context}.{method}.windows.{window}.first_alert_shot",
                )
                role_value = require_int(
                    role,
                    context=f"{context}.{method}.windows.{window}.first_alert_role",
                )
                if not start <= shot_value < stop or not 0 <= role_value < 51:
                    raise PublicationDataError(
                        f"{context}.{method}.{window} first alert is out of bounds."
                    )
            elif shot is not None or role is not None:
                raise PublicationDataError(
                    f"{context}.{method}.{window} non-detection has an alert location."
                )
    return summary


def validate_thresholds(value: Any) -> Mapping[str, Any]:
    thresholds = require_mapping(value, context="Google threshold table")
    require_exact_keys(thresholds, METHOD_IDS, context="Google threshold table")
    for method in METHOD_IDS:
        row = require_mapping(
            thresholds[method], context=f"Google threshold table.{method}"
        )
        require_exact_keys(
            row,
            {
                "threshold",
                "validation_alert_count",
                "max_validation_alerts",
                "secondary_zero_alert_threshold",
                "secondary_validation_alert_count",
            },
            context=f"Google threshold table.{method}",
        )
        require_number(row["threshold"], context=f"Google primary threshold {method}")
        primary_count = require_int(
            row["validation_alert_count"],
            context=f"Google primary validation count {method}",
            minimum=0,
        )
        if row["max_validation_alerts"] != 2 or primary_count > 2:
            raise PublicationDataError(
                f"Google primary threshold budget changed for {method}."
            )
        require_number(
            row["secondary_zero_alert_threshold"],
            context=f"Google secondary threshold {method}",
        )
        if row["secondary_validation_alert_count"] != 0:
            raise PublicationDataError(
                f"Google secondary threshold is not zero-alert for {method}."
            )
    return thresholds


def validate_detector_manifest(
    path: Path,
) -> tuple[
    Mapping[str, Any],
    dict[str, Path],
    Mapping[str, Any],
    Mapping[str, Any],
]:
    manifest = require_mapping(load_json(path), context="detector manifest")
    require_exact_keys(manifest, DETECTOR_MANIFEST_KEYS, context="detector manifest")
    if (
        manifest["schema_version"] != "run6-google-detector-freeze-v1"
        or manifest["protocol_id"] != GOOGLE_PROTOCOL
        or manifest["detector_only"] is not True
        or manifest["outcome_accessed"] is not False
        or manifest["outcome_join_authorized"] is not False
    ):
        raise PublicationDataError(
            "Detector manifest is not the frozen detector-only run."
        )
    resources = require_mapping(
        manifest["resources"], context="detector manifest.resources"
    )
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
        context="detector manifest.resources",
    )
    exposure = require_mapping(
        resources["record_exposure"],
        context="detector manifest.resources.record_exposure",
    )
    expected_exposure = {
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
    require_exact_keys(
        exposure,
        expected_exposure,
        context="detector manifest.resources.record_exposure",
    )
    for stage, expected_row in expected_exposure.items():
        observed = require_mapping(
            exposure[stage],
            context=f"detector resource exposure {stage}",
        )
        require_exact_keys(
            observed,
            expected_row,
            context=f"detector resource exposure {stage}",
        )
        if dict(observed) != expected_row:
            raise PublicationDataError(f"Detector exposure row changed: {stage}.")
    formal = require_mapping(
        resources["formal_accumulator"],
        context="detector manifest.resources.formal_accumulator",
    )
    if dict(formal) != {
        "time_unit": "complete_paired_shot",
        "held_updates": 20_000,
        "role_prior": "uniform_1_over_51",
        "within_shot_factor_compounding": False,
    }:
        raise PublicationDataError("Detector formal-time resource ledger changed.")
    performance = require_mapping(
        manifest["performance"], context="detector manifest.performance"
    )
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
        context="detector manifest.performance",
    )
    if (
        performance["canonical_joint_pipeline_only"] is not True
        or performance["not_a_per_method_speed_comparison"] is not True
        or performance["relative_method_speed_claim_authorized"] is not False
    ):
        raise PublicationDataError("Detector timing scope is not joint-pipeline-only.")
    for key in timing_keys:
        value = require_number(performance[key], context=f"detector performance {key}")
        if value < 0:
            raise PublicationDataError("Detector timing values cannot be negative.")
    timings = performance["held_joint_replay_all_three_seconds"]
    digests = performance["held_joint_replay_digests"]
    if (
        not isinstance(timings, list)
        or len(timings) != 3
        or any(
            require_number(value, context="detector held replay timing") < 0
            for value in timings
        )
        or performance["held_replay_seconds"] != timings[0]
        or not close(
            require_number(
                performance["held_joint_replay_median_seconds"],
                context="detector held replay median",
            ),
            sorted(float(value) for value in timings)[1],
        )
        or not isinstance(digests, list)
        or len(digests) != 3
        or len(set(digests)) != 1
    ):
        raise PublicationDataError("Detector repeated joint timing ledger is invalid.")
    require_int(
        performance["peak_rss_kib_linux_ru_maxrss"],
        context="detector process-wide peak RSS",
        minimum=0,
    )
    artifacts = artifact_map(
        manifest["artifacts"], path, context="detector manifest.artifacts"
    )
    expected = expected_detector_artifact_names()
    if set(artifacts) != expected:
        raise PublicationDataError(
            "Detector artifact contract is incomplete or has unknown files; "
            f"missing={sorted(expected - set(artifacts))}, "
            f"extra={sorted(set(artifacts) - expected)}."
        )
    if manifest["threshold_table_sha256"] != sha256_file(artifacts["thresholds.json"]):
        raise PublicationDataError("Detector manifest/threshold table hash mismatch.")
    event = validate_event_summary(
        load_json(artifacts["event_summary_detector_only.json"]),
        context="primary event summary",
    )
    thresholds = validate_thresholds(load_json(artifacts["thresholds.json"]))
    validate_event_summary(
        load_json(artifacts["secondary_event_summary_detector_only.json"]),
        context="secondary event summary",
    )
    return manifest, artifacts, event, thresholds


def percentile_summary(values: Sequence[float | int]) -> dict[str, float]:
    """Match the producer's NumPy linear-percentile serialization exactly."""

    numeric = np.asarray(values, dtype=np.float64)
    if numeric.ndim != 1 or numeric.size < 1 or not np.all(np.isfinite(numeric)):
        raise PublicationDataError(
            "Percentile source values must be a nonempty finite vector."
        )
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


def integer_histogram(values: Sequence[int] | np.ndarray) -> dict[str, int]:
    """Match the producer's sorted ``np.unique`` histogram exactly."""

    numeric = np.asarray(values)
    if numeric.ndim != 1 or numeric.size < 1 or numeric.dtype.kind not in {"i", "u"}:
        raise PublicationDataError(
            "Histogram source values must be a nonempty integer vector."
        )
    unique, frequency = np.unique(numeric, return_counts=True)
    return {
        str(int(value)): int(count)
        for value, count in zip(unique, frequency, strict=True)
    }


def validate_percentile_summary(
    value: Any,
    *,
    expected: Mapping[str, float],
    context: str,
) -> Mapping[str, Any]:
    row = require_mapping(value, context=context)
    require_exact_keys(row, PERCENTILE_SUMMARY_KEYS, context=context)
    for key in PERCENTILE_SUMMARY_KEYS:
        require_number(row[key], context=f"{context}.{key}")
    if dict(row) != dict(expected):
        raise PublicationDataError(f"{context} disagrees with replicate values.")
    return row


def validate_integer_histogram(
    value: Any,
    *,
    expected: Mapping[str, int],
    maximum_bin: int,
    total: int,
    context: str,
) -> Mapping[str, Any]:
    row = require_mapping(value, context=context)
    observed: dict[str, int] = {}
    for key, frequency in row.items():
        if (
            not isinstance(key, str)
            or not key
            or not key.isascii()
            or not key.isdecimal()
            or key != str(int(key))
        ):
            raise PublicationDataError(f"{context} has a noncanonical bin key.")
        bin_value = int(key)
        if not 0 <= bin_value <= maximum_bin:
            raise PublicationDataError(f"{context} has a bin outside its support.")
        observed[key] = require_int(
            frequency,
            context=f"{context}[{key}]",
            minimum=1,
        )
    if sum(observed.values()) != total:
        raise PublicationDataError(f"{context} frequencies do not sum to {total}.")
    if observed != dict(expected):
        raise PublicationDataError(f"{context} disagrees with replicate values.")
    return row


def validate_google_threshold_bootstrap(
    value: Any,
    *,
    detector_thresholds: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Recompute every descriptive Google threshold-bootstrap summary."""

    bootstrap = require_mapping(value, context="Google threshold bootstrap")
    require_exact_keys(
        bootstrap,
        GOOGLE_THRESHOLD_BOOTSTRAP_KEYS,
        context="Google threshold bootstrap",
    )
    if (
        bootstrap["schema_version"] != "run6-google-threshold-bootstrap-v1"
        or bootstrap["status"] != "descriptive_only_does_not_replace_frozen_threshold"
        or bootstrap["unit"] != "complete_paired_shot"
        or bootstrap["block_length_shots"] != 128
        or bootstrap["replicates"] != GOOGLE_THRESHOLD_BOOTSTRAP_REPLICATES
        or bootstrap["seed_start"] != 613_000
        or bootstrap["seed_stop_exclusive"] != 615_000
        or bootstrap["rng"] != "numpy.random.Generator(PCG64)"
        or bootstrap["blocks_per_replicate"] != 40
        or bootstrap["primary_maximum_alerts"] != 2
        or bootstrap["secondary_maximum_alerts"] != 0
    ):
        raise PublicationDataError(
            "Google threshold bootstrap differs from the locked descriptive design."
        )

    rows = bootstrap["replicate_results"]
    if not isinstance(rows, list) or len(rows) != GOOGLE_THRESHOLD_BOOTSTRAP_REPLICATES:
        raise PublicationDataError(
            "Google threshold bootstrap must contain exactly 2,000 replicates."
        )
    selected_thresholds = {
        method: np.empty(GOOGLE_THRESHOLD_BOOTSTRAP_REPLICATES, dtype=np.float64)
        for method in METHOD_IDS
    }
    selected_counts = {
        method: np.empty(GOOGLE_THRESHOLD_BOOTSTRAP_REPLICATES, dtype=np.int64)
        for method in METHOD_IDS
    }
    frozen_counts = {
        method: np.empty(GOOGLE_THRESHOLD_BOOTSTRAP_REPLICATES, dtype=np.int64)
        for method in METHOD_IDS
    }
    zero_thresholds = {
        method: np.empty(GOOGLE_THRESHOLD_BOOTSTRAP_REPLICATES, dtype=np.float64)
        for method in METHOD_IDS
    }
    for index, raw in enumerate(rows):
        row = require_mapping(
            raw,
            context=f"Google threshold bootstrap replicate[{index}]",
        )
        require_exact_keys(
            row,
            GOOGLE_THRESHOLD_BOOTSTRAP_REPLICATE_KEYS,
            context=f"Google threshold bootstrap replicate[{index}]",
        )
        if row["replicate_index"] != index or row["seed"] != 613_000 + index:
            raise PublicationDataError(
                "Google threshold-bootstrap replicate identity or seed changed."
            )
        methods = require_mapping(
            row["methods"],
            context=f"Google threshold bootstrap replicate[{index}].methods",
        )
        require_exact_keys(
            methods,
            METHOD_IDS,
            context=f"Google threshold bootstrap replicate[{index}].methods",
        )
        for method in METHOD_IDS:
            cell = require_mapping(
                methods[method],
                context=(f"Google threshold bootstrap replicate[{index}].{method}"),
            )
            require_exact_keys(
                cell,
                GOOGLE_THRESHOLD_BOOTSTRAP_METHOD_KEYS,
                context=(f"Google threshold bootstrap replicate[{index}].{method}"),
            )
            selected = require_number(
                cell["selected_primary_threshold"],
                context=(
                    f"Google threshold bootstrap replicate[{index}].{method} "
                    "selected threshold"
                ),
            )
            selected_count = require_int(
                cell["selected_primary_alert_count"],
                context=(
                    f"Google threshold bootstrap replicate[{index}].{method} "
                    "selected alert count"
                ),
                minimum=0,
            )
            frozen_count = require_int(
                cell["alert_count_at_frozen_threshold"],
                context=(
                    f"Google threshold bootstrap replicate[{index}].{method} "
                    "frozen-threshold alert count"
                ),
                minimum=0,
            )
            selected_zero = require_number(
                cell["selected_zero_alert_threshold"],
                context=(
                    f"Google threshold bootstrap replicate[{index}].{method} "
                    "zero-alert threshold"
                ),
            )
            zero_count = require_int(
                cell["selected_zero_alert_count"],
                context=(
                    f"Google threshold bootstrap replicate[{index}].{method} "
                    "zero-alert count"
                ),
                minimum=0,
            )
            if (
                selected_count > 2
                or frozen_count > 5_000
                or zero_count != 0
                or selected_zero < selected
            ):
                raise PublicationDataError(
                    "Google threshold-bootstrap replicate violates alert-budget "
                    "or threshold-order semantics."
                )
            selected_thresholds[method][index] = selected
            selected_counts[method][index] = selected_count
            frozen_counts[method][index] = frozen_count
            zero_thresholds[method][index] = selected_zero

    summaries = require_mapping(
        bootstrap["summaries"],
        context="Google threshold bootstrap summaries",
    )
    require_exact_keys(
        summaries,
        METHOD_IDS,
        context="Google threshold bootstrap summaries",
    )
    require_exact_keys(
        detector_thresholds,
        METHOD_IDS,
        context="detector threshold rows for bootstrap",
    )
    for method in METHOD_IDS:
        summary = require_mapping(
            summaries[method],
            context=f"Google threshold bootstrap summary {method}",
        )
        require_exact_keys(
            summary,
            GOOGLE_THRESHOLD_BOOTSTRAP_SUMMARY_KEYS,
            context=f"Google threshold bootstrap summary {method}",
        )
        frozen_threshold = require_number(
            summary["frozen_threshold"],
            context=f"Google threshold bootstrap {method} frozen threshold",
        )
        expected_threshold = require_number(
            detector_thresholds[method]["threshold"],
            context=f"detector frozen threshold {method}",
        )
        if frozen_threshold != expected_threshold:
            raise PublicationDataError(
                "Google threshold bootstrap does not use the detector-frozen "
                f"threshold for {method}."
            )
        validate_percentile_summary(
            summary["selected_primary_threshold_percentiles"],
            expected=percentile_summary(selected_thresholds[method]),
            context=f"Google threshold bootstrap {method} selected threshold",
        )
        validate_integer_histogram(
            summary["selected_primary_alert_count_frequency"],
            expected=integer_histogram(selected_counts[method]),
            maximum_bin=2,
            total=GOOGLE_THRESHOLD_BOOTSTRAP_REPLICATES,
            context=f"Google threshold bootstrap {method} selected-count histogram",
        )
        validate_percentile_summary(
            summary["alert_count_at_frozen_threshold_percentiles"],
            expected=percentile_summary(frozen_counts[method]),
            context=f"Google threshold bootstrap {method} frozen-count percentiles",
        )
        validate_integer_histogram(
            summary["alert_count_at_frozen_threshold_frequency"],
            expected=integer_histogram(frozen_counts[method]),
            maximum_bin=5_000,
            total=GOOGLE_THRESHOLD_BOOTSTRAP_REPLICATES,
            context=f"Google threshold bootstrap {method} frozen-count histogram",
        )
        validate_percentile_summary(
            summary["selected_zero_alert_threshold_percentiles"],
            expected=percentile_summary(zero_thresholds[method]),
            context=f"Google threshold bootstrap {method} zero-alert threshold",
        )
    return bootstrap


def validate_randomization_result(value: Any) -> Mapping[str, Any]:
    result = require_mapping(value, context="Google randomization result")
    require_exact_keys(
        result, RANDOMIZATION_RESULT_KEYS, context="Google randomization result"
    )
    if (
        result["schema_version"] != "run6-google-randomization-result-v1"
        or result["primary_method"] != "space"
        or result["primary_statistic"] != "ever_proper_prior_eprocess_ge_100"
        or result["replicate_count"] != 256
        or result["seed_start"] != 610_700
        or result["seed_stop_exclusive"] != 610_956
        or result["one_orientation_draw_per_replicate"] is not True
        or result["complete_shot_swap_shared_across_roles"] is not True
        or result["interpretation"] != "exact_design_based_implementation_diagnostic"
    ):
        raise PublicationDataError("Google randomization result differs from the lock.")
    counts = require_mapping(
        result["crossing_counts_at_100"],
        context="Google randomization crossing counts",
    )
    require_exact_keys(
        counts, FORMAL_METHOD_IDS, context="Google randomization crossing counts"
    )
    for method in FORMAL_METHOD_IDS:
        count = require_int(
            counts[method],
            context=f"Google randomization crossing count {method}",
            minimum=0,
        )
        if count > 256:
            raise PublicationDataError("Randomization crossing count exceeds 256.")
    space_count = int(counts["space"])
    fraction = require_number(
        result["space_crossing_fraction"],
        context="Google randomization S-PACE fraction",
    )
    if not close(fraction, space_count / 256):
        raise PublicationDataError("Randomization S-PACE count/fraction mismatch.")
    interval = require_mapping(
        result["space_crossing_clopper_pearson_95"],
        context="Google randomization interval",
    )
    require_exact_keys(
        interval, {"lower", "upper"}, context="Google randomization interval"
    )
    lower = require_number(interval["lower"], context="randomization interval lower")
    upper = require_number(interval["upper"], context="randomization interval upper")
    if not 0 <= lower <= fraction <= upper <= 1:
        raise PublicationDataError("Randomization interval is invalid.")
    replicates = result["replicates"]
    if not isinstance(replicates, list) or len(replicates) != 256:
        raise PublicationDataError("Randomization replicate rows are incomplete.")
    recomputed_counts = {method: 0 for method in FORMAL_METHOD_IDS}
    familywide_count = 0
    for index, raw in enumerate(replicates):
        row = require_mapping(raw, context=f"randomization replicate[{index}]")
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
            context=f"randomization replicate[{index}]",
        )
        if (
            row["replicate_index"] != index
            or row["seed"] != 610_700 + index
            or row["checkpoint_restored"] is not True
            or row["formal_eprocess_updates"] != 5_000
            or row["role_score_updates"] != 255_000
        ):
            raise PublicationDataError("Randomization replicate identity changed.")
        require_sha256(
            row["swap_sha256"],
            context=f"randomization replicate[{index}] swap hash",
        )
        swaps = require_int(
            row["swapped_shot_count"],
            context=f"randomization replicate[{index}] swapped shots",
            minimum=0,
        )
        if swaps > 5_000:
            raise PublicationDataError(
                "Randomization swapped-shot count exceeds horizon."
            )
        crossed = require_mapping(
            row["crossed_100"],
            context=f"randomization replicate[{index}] crossings",
        )
        first = require_mapping(
            row["first_crossing_shot_number_one_based"],
            context=f"randomization replicate[{index}] first crossings",
        )
        maxima = require_mapping(
            row["maximum_log_e"],
            context=f"randomization replicate[{index}] maximum log-e",
        )
        final = require_mapping(
            row["final_log_e"],
            context=f"randomization replicate[{index}] final log-e",
        )
        experts = require_mapping(
            row["formal_experts"],
            context=f"randomization replicate[{index}] experts",
        )
        for mapping_name, mapping in (
            ("crossings", crossed),
            ("first crossings", first),
            ("maximum log-e", maxima),
            ("final log-e", final),
            ("experts", experts),
        ):
            require_exact_keys(
                mapping,
                FORMAL_METHOD_IDS,
                context=f"randomization replicate[{index}] {mapping_name}",
            )
        for method in FORMAL_METHOD_IDS:
            did_cross = require_bool(
                crossed[method],
                context=f"randomization replicate[{index}] {method} crossing",
            )
            crossing_shot = first[method]
            if did_cross:
                crossing = require_int(
                    crossing_shot,
                    context=f"randomization replicate[{index}] {method} first crossing",
                    minimum=1,
                )
                if crossing > 5_000:
                    raise PublicationDataError(
                        "Randomization first crossing exceeds horizon."
                    )
                recomputed_counts[method] += 1
            elif crossing_shot is not None:
                raise PublicationDataError(
                    "Randomization non-crossing has a first-crossing shot."
                )
            if (
                require_number(
                    maxima[method],
                    context=f"randomization replicate[{index}] {method} maximum",
                )
                < 0
            ):
                raise PublicationDataError(
                    "Randomization maximum log-e cannot be negative."
                )
            require_number(
                final[method],
                context=f"randomization replicate[{index}] {method} final",
            )
            require_int(
                experts[method],
                context=f"randomization replicate[{index}] {method} expert count",
                minimum=1,
            )
        if require_bool(
            row["familywide_any_crossed_600"],
            context=f"randomization replicate[{index}] familywide crossing",
        ):
            familywide_count += 1
    if (
        recomputed_counts != dict(counts)
        or familywide_count != result["familywide_any_crossing_count_at_600"]
    ):
        raise PublicationDataError("Randomization summary/replicate rows disagree.")
    return result


def validate_randomization_manifest(
    path: Path,
    *,
    detector_path: Path,
    detector_manifest: Mapping[str, Any],
    detector_thresholds: Mapping[str, Any],
    repair_ratification_path: Path,
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    manifest = require_mapping(load_json(path), context="randomization manifest")
    require_exact_keys(
        manifest,
        RANDOMIZATION_MANIFEST_KEYS,
        context="randomization manifest",
    )
    if (
        manifest["schema_version"] != "run6-google-randomization-manifest-v1"
        or manifest["protocol_id"] != GOOGLE_PROTOCOL
        or manifest["outcome_accessed"] is not False
        or manifest["execution_mode"] != "deterministic_gap_free_shard_merge"
        or manifest["detector_manifest_sha256"] != sha256_file(detector_path)
        or manifest["detector_manifest_git_commit"] != detector_manifest["git_commit"]
        or manifest["config_sha256"] != detector_manifest["config_sha256"]
        or manifest["method_spec_sha256"] != detector_manifest["method_spec_sha256"]
        or manifest["freeze_ratification_sha256"]
        != detector_manifest["freeze_ratification_sha256"]
        or manifest["repair_ratification_path"] != REPAIR_RATIFICATION_PATH
        or manifest["repair_ratification_sha256"]
        != sha256_file(repair_ratification_path)
        or manifest["warm_checkpoint_sha256"]
        != detector_manifest["warm_checkpoint_sha256"]
        or manifest["source_archive_sha256"]
        != detector_manifest["source_archive_sha256"]
        or manifest["verified_zip_member_sha256"]
        != detector_manifest["verified_zip_member_sha256"]
    ):
        raise PublicationDataError("Randomization manifest bindings are inconsistent.")
    rng = require_mapping(manifest["rng"], context="randomization manifest RNG")
    require_exact_keys(
        rng,
        {
            "algorithm",
            "randomization_seed_start",
            "randomization_seed_stop_exclusive",
            "threshold_bootstrap_seed_start",
            "threshold_bootstrap_seed_stop_exclusive",
        },
        context="randomization manifest RNG",
    )
    if dict(rng) != {
        "algorithm": "numpy.random.Generator(PCG64)",
        "randomization_seed_start": 610_700,
        "randomization_seed_stop_exclusive": 610_956,
        "threshold_bootstrap_seed_start": 613_000,
        "threshold_bootstrap_seed_stop_exclusive": 615_000,
    }:
        raise PublicationDataError(
            "Randomization manifest RNG or threshold-bootstrap seed range changed."
        )
    artifacts = artifact_map(
        manifest["artifacts"], path, context="randomization manifest.artifacts"
    )
    if set(artifacts) != {"randomization_result.json", "threshold_bootstrap.json"}:
        raise PublicationDataError("Randomization artifact contract is incomplete.")
    result = validate_randomization_result(
        load_json(artifacts["randomization_result.json"])
    )
    threshold_bootstrap = validate_google_threshold_bootstrap(
        load_json(artifacts["threshold_bootstrap.json"]),
        detector_thresholds=detector_thresholds,
    )
    if result["warm_checkpoint_sha256"] != manifest["warm_checkpoint_sha256"]:
        raise PublicationDataError(
            "Randomization result/manifest warm checkpoint mismatch."
        )
    return manifest, result, threshold_bootstrap


def validate_interval(value: Any, *, context: str) -> None:
    row = require_mapping(value, context=context)
    require_exact_keys(row, {"lower", "upper", "valid_replicates"}, context=context)
    valid = require_int(row["valid_replicates"], context=f"{context}.valid", minimum=0)
    if valid == 0:
        if row["lower"] is not None or row["upper"] is not None:
            raise PublicationDataError(f"{context} invalid empty interval.")
        return
    lower = require_number(row["lower"], context=f"{context}.lower")
    upper = require_number(row["upper"], context=f"{context}.upper")
    if lower > upper:
        raise PublicationDataError(f"{context} has reversed bounds.")


def validate_risk_summary(
    value: Any,
    *,
    detector_hash: str,
) -> Mapping[str, Any]:
    risk = require_mapping(value, context="Google risk summary")
    require_exact_keys(risk, RISK_SUMMARY_KEYS, context="Google risk summary")
    if (
        risk["schema_version"] != "run6-google-risk-summary-v1"
        or risk["primary_label"] != "actual_xor_correlated_matching_prediction"
        or risk["secondary_label"] != "actual_xor_pymatching_prediction"
        or risk["ranking"] != "descending_frozen_shot_score_then_ascending_archive_shot"
        or risk["budgets_shots"] != list(RISK_BUDGETS)
        or risk["detector_manifest_sha256"] != detector_hash
        or risk["interpretation"] != "retrospective_veto_or_triage_not_a_decoder"
    ):
        raise PublicationDataError("Google risk-summary lock fields changed.")
    estimates = require_mapping(
        risk["point_estimates"], context="Google point estimates"
    )
    require_exact_keys(estimates, LABEL_IDS, context="Google point estimates")
    for label in LABEL_IDS:
        methods = require_mapping(estimates[label], context=f"point estimates.{label}")
        require_exact_keys(methods, METHOD_IDS, context=f"point estimates.{label}")
        common_total: int | None = None
        for method in METHOD_IDS:
            method_row = require_mapping(
                methods[method], context=f"point estimates.{label}.{method}"
            )
            require_exact_keys(
                method_row,
                {"budgets", "partial_trapezoidal_recall_area"},
                context=f"point estimates.{label}.{method}",
            )
            budgets = require_mapping(
                method_row["budgets"],
                context=f"point estimates.{label}.{method}.budgets",
            )
            require_exact_keys(
                budgets,
                {str(value) for value in RISK_BUDGETS},
                context=f"point estimates.{label}.{method}.budgets",
            )
            recall_points: list[float] = []
            alert_fractions: list[float] = []
            for budget in RISK_BUDGETS:
                row = require_mapping(
                    budgets[str(budget)],
                    context=f"point estimates.{label}.{method}.{budget}",
                )
                require_exact_keys(
                    row,
                    {
                        "alert_budget_shots",
                        "alert_fraction",
                        "captured_mismatches",
                        "total_mismatches",
                        "mismatch_recall",
                        "alert_precision",
                        "retained_mismatch_rate",
                        "coverage",
                        "selected_archive_shots",
                    },
                    context=f"point estimates.{label}.{method}.{budget}",
                )
                if row["alert_budget_shots"] != budget:
                    raise PublicationDataError("Risk budget label/value mismatch.")
                captured = require_int(
                    row["captured_mismatches"],
                    context=f"risk captured {label}/{method}/{budget}",
                    minimum=0,
                )
                total = require_int(
                    row["total_mismatches"],
                    context=f"risk total {label}/{method}/{budget}",
                    minimum=0,
                )
                if captured > min(total, budget):
                    raise PublicationDataError("Impossible mismatch-capture count.")
                if common_total is None:
                    common_total = total
                elif common_total != total:
                    raise PublicationDataError(
                        f"Mismatch total varies across {label} rows."
                    )
                expected_values = {
                    "alert_fraction": budget / 20_000,
                    "alert_precision": captured / budget,
                    "retained_mismatch_rate": (total - captured) / (20_000 - budget),
                    "coverage": 1 - budget / 20_000,
                }
                for key, expected in expected_values.items():
                    observed = require_number(
                        row[key], context=f"risk {key} {label}/{method}/{budget}"
                    )
                    if not close(observed, expected):
                        raise PublicationDataError(f"Risk {key} is inconsistent.")
                recall = row["mismatch_recall"]
                if total == 0:
                    if recall is not None:
                        raise PublicationDataError("Zero-total recall must be null.")
                else:
                    numeric_recall = require_number(
                        recall,
                        context=f"risk recall {label}/{method}/{budget}",
                    )
                    if not close(numeric_recall, captured / total):
                        raise PublicationDataError("Risk recall is inconsistent.")
                    recall_points.append(numeric_recall)
                    alert_fractions.append(budget / 20_000)
                selected = row["selected_archive_shots"]
                if (
                    not isinstance(selected, list)
                    or len(selected) != budget
                    or len(set(selected)) != budget
                    or any(
                        isinstance(shot, bool)
                        or not isinstance(shot, int)
                        or not 40_000 <= shot < 60_000
                        for shot in selected
                    )
                ):
                    raise PublicationDataError("Risk selected-shot list is invalid.")
            area = method_row["partial_trapezoidal_recall_area"]
            if common_total == 0:
                if area is not None:
                    raise PublicationDataError(
                        "Zero-total partial recall area must be null."
                    )
            else:
                expected_area = float(
                    np.trapezoid(
                        np.asarray(recall_points, dtype=np.float64),
                        np.asarray(alert_fractions, dtype=np.float64),
                    )
                )
                if not close(
                    require_number(
                        area,
                        context=f"partial recall area {label}/{method}",
                    ),
                    expected_area,
                ):
                    raise PublicationDataError(
                        "Partial trapezoidal recall area is inconsistent."
                    )
    uncertainty = require_mapping(
        risk["uncertainty"], context="Google risk uncertainty"
    )
    require_exact_keys(
        uncertainty,
        {
            "kind",
            "block_length_shots",
            "replicates",
            "seed_start",
            "seed_stop_exclusive",
            "rng",
            "blocks_per_replicate",
            "percentile_interval",
            "percentile_method",
            "method_intervals",
            "space_comparator_difference_intervals",
        },
        context="Google risk uncertainty",
    )
    if (
        uncertainty["kind"] != "paired_circular_moving_complete_shot_blocks"
        or uncertainty["block_length_shots"] != 128
        or uncertainty["replicates"] != 2_000
        or uncertainty["seed_start"] != 611_000
        or uncertainty["seed_stop_exclusive"] != 613_000
        or uncertainty["percentile_interval"] != [2.5, 97.5]
        or uncertainty["percentile_method"] != "linear"
    ):
        raise PublicationDataError("Google risk uncertainty differs from the lock.")
    method_intervals = require_mapping(
        uncertainty["method_intervals"],
        context="Google method uncertainty",
    )
    require_exact_keys(method_intervals, LABEL_IDS, context="Google method uncertainty")
    for label in LABEL_IDS:
        methods = require_mapping(
            method_intervals[label], context=f"uncertainty.{label}"
        )
        require_exact_keys(methods, METHOD_IDS, context=f"uncertainty.{label}")
        for method in METHOD_IDS:
            budgets = require_mapping(
                methods[method], context=f"uncertainty.{label}.{method}"
            )
            require_exact_keys(
                budgets,
                {str(value) for value in RISK_BUDGETS},
                context=f"uncertainty.{label}.{method}",
            )
            for budget in RISK_BUDGETS:
                metrics = require_mapping(
                    budgets[str(budget)],
                    context=f"uncertainty.{label}.{method}.{budget}",
                )
                require_exact_keys(
                    metrics,
                    {
                        "captured_mismatches",
                        "mismatch_recall",
                        "alert_precision",
                        "retained_mismatch_rate",
                    },
                    context=f"uncertainty.{label}.{method}.{budget}",
                )
                for metric, interval in metrics.items():
                    validate_interval(
                        interval,
                        context=f"uncertainty.{label}.{method}.{budget}.{metric}",
                    )
    difference_intervals = require_mapping(
        uncertainty["space_comparator_difference_intervals"],
        context="Google S-PACE comparator-difference uncertainty",
    )
    require_exact_keys(
        difference_intervals,
        LABEL_IDS,
        context="Google S-PACE comparator-difference uncertainty",
    )
    for label in LABEL_IDS:
        comparators = require_mapping(
            difference_intervals[label],
            context=f"difference uncertainty.{label}",
        )
        require_exact_keys(
            comparators,
            {"space_minus_m0", "space_minus_m3"},
            context=f"difference uncertainty.{label}",
        )
        for comparator, budgets_value in comparators.items():
            budgets = require_mapping(
                budgets_value,
                context=f"difference uncertainty.{label}.{comparator}",
            )
            require_exact_keys(
                budgets,
                {str(value) for value in RISK_BUDGETS},
                context=f"difference uncertainty.{label}.{comparator}",
            )
            for budget in RISK_BUDGETS:
                metrics = require_mapping(
                    budgets[str(budget)],
                    context=f"difference uncertainty.{label}.{comparator}.{budget}",
                )
                require_exact_keys(
                    metrics,
                    {
                        "captured_mismatches",
                        "mismatch_recall",
                        "alert_precision",
                        "retained_mismatch_rate",
                    },
                    context=f"difference uncertainty.{label}.{comparator}.{budget}",
                )
                for metric, interval in metrics.items():
                    validate_interval(
                        interval,
                        context=(
                            f"difference uncertainty.{label}.{comparator}."
                            f"{budget}.{metric}"
                        ),
                    )
    return risk


def parity_evidence_valid(value: Any) -> bool:
    try:
        evidence = require_mapping(value, context="method-input parity evidence")
        require_exact_keys(
            evidence,
            {
                "schema_version",
                "expected_method_ids",
                "observed_method_ids",
                "expected_held_score_shape",
                "held_detector_score_inputs",
                "all_methods_have_locked_detector_record_shape_and_count",
                "shared_outcome_label_bundle",
                "no_method_received_extra_detector_records_or_outcome_labels",
            },
            context="method-input parity evidence",
        )
        if (
            evidence["schema_version"] != "run6-google-method-input-parity-evidence-v1"
            or evidence["expected_method_ids"] != list(METHOD_IDS)
            or evidence["observed_method_ids"] != list(METHOD_IDS)
            or evidence["expected_held_score_shape"] != [20_000, 51]
            or evidence["all_methods_have_locked_detector_record_shape_and_count"]
            is not True
            or evidence["no_method_received_extra_detector_records_or_outcome_labels"]
            is not True
        ):
            return False
        methods = require_mapping(
            evidence["held_detector_score_inputs"],
            context="method-input detector rows",
        )
        require_exact_keys(methods, METHOD_IDS, context="method-input detector rows")
        for method in METHOD_IDS:
            row = require_mapping(methods[method], context=f"method input {method}")
            require_exact_keys(
                row,
                {"shape", "record_count", "matches_locked_shape_and_count"},
                context=f"method input {method}",
            )
            if (
                row["shape"] != [20_000, 51]
                or row["record_count"] != 1_020_000
                or row["matches_locked_shape_and_count"] is not True
            ):
                return False
        shared = require_mapping(
            evidence["shared_outcome_label_bundle"],
            context="shared outcome-label bundle",
        )
        require_exact_keys(
            shared,
            {
                "serialization",
                "sha256",
                "label_ids",
                "label_record_counts",
                "archive_shot_count",
                "consumer_method_ids",
                "single_shared_bundle_for_all_methods",
            },
            context="shared outcome-label bundle",
        )
        counts = require_mapping(
            shared["label_record_counts"], context="shared label counts"
        )
        return bool(
            shared["label_ids"] == list(LABEL_IDS)
            and set(counts) == set(LABEL_IDS)
            and all(counts[label] == 20_000 for label in LABEL_IDS)
            and shared["archive_shot_count"] == 20_000
            and shared["consumer_method_ids"] == list(METHOD_IDS)
            and shared["single_shared_bundle_for_all_methods"] is True
            and require_sha256(
                shared["sha256"], context="shared outcome-label bundle hash"
            )
        )
    except PublicationDataError:
        return False


def require_nonempty_text(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise PublicationDataError(f"{context} must be nonempty text.")
    return value


def require_fixed_list(value: Any, length: int, *, context: str) -> list[Any]:
    if not isinstance(value, list) or len(value) != length:
        raise PublicationDataError(
            f"{context} must be a list with exactly {length} entries."
        )
    return value


def validate_pittsburgh_manifest(path: Path) -> Mapping[str, Any]:
    """Validate metadata-only Pittsburgh evidence without opening payload paths."""

    manifest = require_mapping(
        load_json(path), context="locked Pittsburgh metadata manifest"
    )
    if (
        manifest.get("manifest_id") != "run6-pnnl-pittsburgh-metadata-lock-v2"
        or manifest.get("status") != "frozen_before_held_value_access"
    ):
        raise PublicationDataError(
            "Pittsburgh metadata manifest is not the frozen v2 lock."
        )
    source = require_mapping(
        manifest.get("source"), context="Pittsburgh metadata source"
    )
    if source.get("backend") != "ibm_pittsburgh" or source.get("logical_states") != [
        0,
        1,
    ]:
        raise PublicationDataError("Pittsburgh source identity changed.")
    schemas = require_mapping(
        manifest.get("artifact_tuple_schemas"),
        context="Pittsburgh artifact tuple schemas",
    )
    require_exact_keys(
        schemas,
        PNNL_ARTIFACT_TUPLE_SCHEMAS,
        context="Pittsburgh artifact tuple schemas",
    )
    if any(
        tuple(schemas[key]) != expected
        for key, expected in PNNL_ARTIFACT_TUPLE_SCHEMAS.items()
    ):
        raise PublicationDataError("Pittsburgh artifact tuple schema changed.")

    snapshots = require_mapping(
        manifest.get("snapshots"), context="Pittsburgh snapshots"
    )
    if len(snapshots) != 20:
        raise PublicationDataError("Pittsburgh lock must contain 20 snapshots.")
    parsed_snapshots: dict[str, dict[str, Any]] = {}
    snapshot_keys = {
        "relative_job_dir",
        "metadata",
        "info",
        "calibration",
        "qasm_state0",
        "qasm_state1",
        "qasm_pair",
        "held_bitstrings",
    }
    for snapshot_id, raw in snapshots.items():
        snapshot_id = require_nonempty_text(
            snapshot_id, context="Pittsburgh snapshot ID"
        )
        row = require_mapping(raw, context=f"Pittsburgh snapshot {snapshot_id}")
        require_exact_keys(
            row,
            snapshot_keys,
            context=f"Pittsburgh snapshot {snapshot_id}",
        )
        require_nonempty_text(
            row["relative_job_dir"],
            context=f"Pittsburgh snapshot {snapshot_id} job directory",
        )
        metadata = require_fixed_list(
            row["metadata"],
            len(PNNL_ARTIFACT_TUPLE_SCHEMAS["metadata"]),
            context=f"Pittsburgh snapshot {snapshot_id} metadata",
        )
        distance = require_int(
            metadata[0],
            context=f"Pittsburgh snapshot {snapshot_id} distance",
            minimum=1,
        )
        rounds = require_int(
            metadata[1],
            context=f"Pittsburgh snapshot {snapshot_id} rounds",
            minimum=1,
        )
        if metadata[2] not in {"X", "Z"}:
            raise PublicationDataError(
                f"Pittsburgh snapshot {snapshot_id} basis must be X or Z."
            )
        shots = require_int(
            metadata[3],
            context=f"Pittsburgh snapshot {snapshot_id} shots",
            minimum=2_048,
        )
        require_int(
            metadata[4],
            context=f"Pittsburgh snapshot {snapshot_id} chain count",
            minimum=1,
        )
        for date_index, label in ((5, "property-date source"), (6, "property date")):
            require_nonempty_text(
                metadata[date_index],
                context=f"Pittsburgh snapshot {snapshot_id} {label}",
            )
        for artifact_name in ("info", "calibration", "qasm_state0", "qasm_state1"):
            schema_name = (
                "qasm_state"
                if artifact_name.startswith("qasm_state")
                else artifact_name
            )
            artifact = require_fixed_list(
                row[artifact_name],
                len(PNNL_ARTIFACT_TUPLE_SCHEMAS[schema_name]),
                context=f"Pittsburgh snapshot {snapshot_id} {artifact_name}",
            )
            require_int(
                artifact[0],
                context=f"Pittsburgh snapshot {snapshot_id} {artifact_name} bytes",
                minimum=1,
            )
            require_sha256(
                artifact[1],
                context=f"Pittsburgh snapshot {snapshot_id} {artifact_name} raw hash",
            )
            require_sha256(
                artifact[2],
                context=(
                    f"Pittsburgh snapshot {snapshot_id} {artifact_name} "
                    "canonical/normalized hash"
                ),
            )
        qasm_pair = require_fixed_list(
            row["qasm_pair"],
            len(PNNL_ARTIFACT_TUPLE_SCHEMAS["qasm_pair"]),
            context=f"Pittsburgh snapshot {snapshot_id} QASM pair",
        )
        for index, label in ((0, "raw"), (1, "normalized")):
            require_sha256(
                qasm_pair[index],
                context=f"Pittsburgh snapshot {snapshot_id} {label} QASM-pair hash",
            )
        held = require_fixed_list(
            row["held_bitstrings"],
            len(PNNL_ARTIFACT_TUPLE_SCHEMAS["held_bitstrings"]),
            context=f"Pittsburgh snapshot {snapshot_id} held-payload metadata",
        )
        require_int(
            held[0],
            context=f"Pittsburgh snapshot {snapshot_id} held-payload bytes",
            minimum=1,
        )
        if held[1:] != [False, None]:
            raise PublicationDataError(
                "Pittsburgh lock indicates held payload access before ratification."
            )
        parsed_snapshots[snapshot_id] = {
            "distance": distance,
            "rounds": rounds,
            "basis": metadata[2],
            "shots": shots,
            "calibration_hash": row["calibration"][2],
            "raw_qasm_pair_hash": qasm_pair[0],
            "normalized_qasm_pair_hash": qasm_pair[1],
        }

    if tuple(manifest.get("cohort_row_schema", ())) != PNNL_COHORT_FIELDS:
        raise PublicationDataError("Pittsburgh cohort row schema changed.")
    raw_order = manifest.get("cohort_order")
    if (
        not isinstance(raw_order, list)
        or len(raw_order) != 11
        or len(set(raw_order)) != 11
        or any(not isinstance(value, str) or not value for value in raw_order)
    ):
        raise PublicationDataError("Pittsburgh cohort order is invalid.")
    raw_pairs = manifest.get("cohort_pairs")
    if not isinstance(raw_pairs, list) or len(raw_pairs) != len(raw_order):
        raise PublicationDataError("Pittsburgh cohort row/order length mismatch.")

    cohorts: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_pairs):
        values = require_fixed_list(
            raw,
            len(PNNL_COHORT_FIELDS),
            context=f"Pittsburgh cohort row {index}",
        )
        row = dict(zip(PNNL_COHORT_FIELDS, values, strict=True))
        cohort_id = require_nonempty_text(
            row["cohort_id"], context=f"Pittsburgh cohort {index} ID"
        )
        if cohort_id != raw_order[index]:
            raise PublicationDataError(
                "Pittsburgh cohort rows are not in locked order."
            )
        distance = require_int(
            row["distance"],
            context=f"Pittsburgh cohort {cohort_id} distance",
            minimum=1,
        )
        rounds = require_int(
            row["rounds"],
            context=f"Pittsburgh cohort {cohort_id} rounds",
            minimum=1,
        )
        if row["basis"] not in {"X", "Z"}:
            raise PublicationDataError("Pittsburgh cohort basis must be X or Z.")
        require_nonempty_text(
            row["register_suffix"],
            context=f"Pittsburgh cohort {cohort_id} register suffix",
        )
        data = require_fixed_list(
            row["data_qubits"],
            distance,
            context=f"Pittsburgh cohort {cohort_id} data qubits",
        )
        syndrome = require_fixed_list(
            row["syndrome_qubits"],
            distance - 1,
            context=f"Pittsburgh cohort {cohort_id} syndrome qubits",
        )
        oriented = require_fixed_list(
            row["oriented_path"],
            2 * distance - 1,
            context=f"Pittsburgh cohort {cohort_id} oriented path",
        )
        for label, qubits in (
            ("data", data),
            ("syndrome", syndrome),
            ("oriented", oriented),
        ):
            for qubit in qubits:
                require_int(
                    qubit,
                    context=f"Pittsburgh cohort {cohort_id} {label} qubit",
                    minimum=0,
                )
        expected_oriented: list[int] = []
        for check_index, check in enumerate(syndrome):
            expected_oriented.extend((data[check_index], check))
        expected_oriented.append(data[-1])
        if oriented != expected_oriented:
            raise PublicationDataError(
                f"Pittsburgh cohort {cohort_id} oriented path is inconsistent."
            )
        early_id = require_nonempty_text(
            row["early_snapshot_id"],
            context=f"Pittsburgh cohort {cohort_id} early snapshot",
        )
        late_id = require_nonempty_text(
            row["late_snapshot_id"],
            context=f"Pittsburgh cohort {cohort_id} late snapshot",
        )
        if early_id == late_id or early_id not in parsed_snapshots:
            raise PublicationDataError(
                f"Pittsburgh cohort {cohort_id} early snapshot is invalid."
            )
        if late_id not in parsed_snapshots:
            raise PublicationDataError(
                f"Pittsburgh cohort {cohort_id} late snapshot is invalid."
            )
        early = parsed_snapshots[early_id]
        late = parsed_snapshots[late_id]
        expected_metadata = (distance, rounds, row["basis"])
        if any(
            (snapshot["distance"], snapshot["rounds"], snapshot["basis"])
            != expected_metadata
            for snapshot in (early, late)
        ):
            raise PublicationDataError(
                f"Pittsburgh cohort {cohort_id} snapshot metadata disagree."
            )
        m = require_int(row["m"], context=f"Pittsburgh cohort {cohort_id} m", minimum=1)
        if m != min(early["shots"] // 3, late["shots"]):
            raise PublicationDataError(
                f"Pittsburgh cohort {cohort_id} partition size is inconsistent."
            )
        raw_same = early["raw_qasm_pair_hash"] == late["raw_qasm_pair_hash"]
        normalized_same = (
            early["normalized_qasm_pair_hash"] == late["normalized_qasm_pair_hash"]
        )
        if (
            require_bool(
                row["raw_qasm_pair_identical"],
                context=f"Pittsburgh cohort {cohort_id} raw-QASM flag",
            )
            is not raw_same
            or require_bool(
                row["normalized_qasm_pair_identical_audit_only"],
                context=f"Pittsburgh cohort {cohort_id} normalized-QASM flag",
            )
            is not normalized_same
        ):
            raise PublicationDataError(
                f"Pittsburgh cohort {cohort_id} QASM-equality flag is inconsistent."
            )
        expected_claim = (
            "circuit_controlled_cross_property_snapshot"
            if raw_same
            else "circuit_and_hardware_domain_shift"
        )
        if row["claim_label"] != expected_claim:
            raise PublicationDataError(
                f"Pittsburgh cohort {cohort_id} claim class is inconsistent."
            )
        if early["calibration_hash"] == late["calibration_hash"]:
            raise PublicationDataError(
                f"Pittsburgh cohort {cohort_id} calibration hashes must differ."
            )
        calibration_pair_id = f"{early['calibration_hash']}--{late['calibration_hash']}"
        if row["calibration_pair_id"] != calibration_pair_id:
            raise PublicationDataError(
                f"Pittsburgh cohort {cohort_id} calibration-pair ID is inconsistent."
            )
        if raw_same:
            qasm_status = (
                "Raw QASM identical; circuit-controlled cross-property snapshot"
            )
        elif normalized_same:
            qasm_status = (
                "Raw QASM differs (normalized identical, audit only); "
                "circuit + hardware shift; no equal-QASM control"
            )
        else:
            qasm_status = (
                "Raw and normalized QASM differ; circuit + hardware shift; "
                "no equal-QASM control"
            )
        cohorts.append(
            {
                "cohort_id": cohort_id,
                "basis": row["basis"],
                "distance": distance,
                "rounds": rounds,
                "calibration_pair_id": calibration_pair_id,
                "raw_qasm_pair_identical": raw_same,
                "normalized_qasm_pair_identical": normalized_same,
                "claim_label": expected_claim,
                "qasm_status": qasm_status,
            }
        )
    return {
        "manifest": manifest,
        "snapshot_ids": tuple(parsed_snapshots),
        "cohorts": tuple(cohorts),
    }


def pnnl_control_summary(cohorts: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    """Summarize a per-cohort control classification without assuming its result."""

    total = len(cohorts)
    raw_controls = sum(bool(row["raw_qasm_pair_identical"]) for row in cohorts)
    normalized_matches = sum(
        bool(row["normalized_qasm_pair_identical"]) for row in cohorts
    )
    if raw_controls == 0:
        caption = (
            f"All {total} selected cohorts are circuit-and-hardware shifts; "
            "no equal-QASM controlled cohort is available."
        )
        figure = f"{total} circuit + hardware shifts; no equal-QASM control"
    elif raw_controls == total:
        caption = (
            f"All {total} selected cohorts have identical raw QASM and are "
            "circuit-controlled cross-property snapshot comparisons."
        )
        figure = f"{total} equal-QASM cross-property controls"
    else:
        caption = (
            f"{raw_controls} of {total} selected cohorts have identical raw QASM; "
            f"{total - raw_controls} combine circuit and hardware shifts."
        )
        figure = (
            f"{raw_controls}/{total} equal-QASM controls; "
            f"{total - raw_controls}/{total} circuit + hardware shifts"
        )
    return {
        "total": total,
        "raw_controls": raw_controls,
        "normalized_matches": normalized_matches,
        "caption": caption,
        "figure": figure,
    }


def validate_pnnl_state_and_aggregate(
    state_path: Path,
    aggregate: Any,
    *,
    pittsburgh_cohorts: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], Mapping[str, Any]]:
    aggregate_row = require_mapping(aggregate, context="PNNL aggregate")
    require_exact_keys(
        aggregate_row,
        {"cohort_rows", "macro_by_method", "comparisons", "retention_pass"},
        context="PNNL aggregate",
    )
    rows: list[dict[str, Any]] = []
    with state_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != PNNL_STATE_FIELDS:
            raise PublicationDataError("PNNL state CSV columns changed.")
        for raw in reader:
            cohort = raw["cohort_id"]
            method = raw["method"]
            if not cohort or method not in PNNL_METHOD_IDS:
                raise PublicationDataError("PNNL state row has an unknown ID.")
            state = int(raw["logical_state"])
            pre = int(raw["pre_false_alarm"])
            miss = int(raw["miss"])
            delay = float(raw["restricted_post_delay_fraction"])
            if state not in (0, 1) or pre not in (0, 1) or miss not in (0, 1):
                raise PublicationDataError("PNNL state flags must be binary.")
            if not math.isfinite(delay) or not 0 <= delay <= 1:
                raise PublicationDataError("PNNL state delay must lie in [0,1].")
            rows.append(
                {
                    "cohort_id": cohort,
                    "logical_state": state,
                    "method": method,
                    "pre_false_alarm": pre,
                    "miss": miss,
                    "restricted_post_delay_fraction": delay,
                }
            )
    indexed = {
        (row["cohort_id"], row["logical_state"], row["method"]): row for row in rows
    }
    if len(indexed) != len(rows):
        raise PublicationDataError("PNNL state CSV contains duplicate keys.")
    cohorts = sorted({row["cohort_id"] for row in rows})
    locked_order = [row["cohort_id"] for row in pittsburgh_cohorts]
    locked_by_id = {row["cohort_id"]: row for row in pittsburgh_cohorts}
    if (
        len(locked_order) != 11
        or len(locked_by_id) != 11
        or set(cohorts) != set(locked_order)
    ):
        raise PublicationDataError(
            "PNNL state cohort IDs disagree with the Pittsburgh lock."
        )
    expected = {
        (cohort, state, method)
        for cohort in locked_order
        for state in (0, 1)
        for method in PNNL_METHOD_IDS
    }
    if len(cohorts) != 11 or set(indexed) != expected or len(rows) != 110:
        raise PublicationDataError("PNNL state CSV does not contain 11 x 2 x 5 rows.")
    cohort_rows = aggregate_row["cohort_rows"]
    if not isinstance(cohort_rows, list) or len(cohort_rows) != 55:
        raise PublicationDataError("PNNL aggregate cohort rows are incomplete.")
    aggregate_index: dict[tuple[str, str], Mapping[str, Any]] = {}
    cohort_order: list[str] = []
    for raw in cohort_rows:
        row = require_mapping(raw, context="PNNL cohort row")
        require_exact_keys(
            row,
            {
                "cohort_id",
                "basis",
                "distance",
                "rounds",
                "calibration_pair_id",
                "method",
                "pre_false_alarm_mean",
                "restricted_post_delay_fraction",
                "miss_mean",
            },
            context="PNNL cohort row",
        )
        key = (row["cohort_id"], row["method"])
        if key in aggregate_index:
            raise PublicationDataError("Duplicate PNNL aggregate cohort row.")
        locked = locked_by_id.get(row["cohort_id"])
        if (
            locked is None
            or row["basis"] != locked["basis"]
            or row["distance"] != locked["distance"]
            or row["rounds"] != locked["rounds"]
            or row["calibration_pair_id"] != locked["calibration_pair_id"]
        ):
            raise PublicationDataError(
                "PNNL aggregate cohort metadata disagree with the Pittsburgh lock."
            )
        aggregate_index[key] = row
        if row["method"] == "dfr":
            cohort_order.append(row["cohort_id"])
    if cohort_order != locked_order:
        raise PublicationDataError(
            "PNNL aggregate cohort ordering disagrees with the Pittsburgh lock."
        )
    for cohort in cohort_order:
        for method in PNNL_METHOD_IDS:
            row = aggregate_index.get((cohort, method))
            if row is None:
                raise PublicationDataError("PNNL aggregate cohort-method row missing.")
            state_pair = [indexed[(cohort, state, method)] for state in (0, 1)]
            expected_values = {
                "pre_false_alarm_mean": sum(
                    state["pre_false_alarm"] for state in state_pair
                )
                / 2,
                "restricted_post_delay_fraction": sum(
                    state["restricted_post_delay_fraction"] for state in state_pair
                )
                / 2,
                "miss_mean": sum(state["miss"] for state in state_pair) / 2,
            }
            for key, expected_value in expected_values.items():
                if not close(
                    require_number(
                        row[key], context=f"PNNL aggregate {cohort}/{method}/{key}"
                    ),
                    expected_value,
                ):
                    raise PublicationDataError("PNNL cohort aggregate is inconsistent.")
    macro = require_mapping(aggregate_row["macro_by_method"], context="PNNL macro rows")
    require_exact_keys(macro, PNNL_METHOD_IDS, context="PNNL macro rows")
    for method in PNNL_METHOD_IDS:
        row = require_mapping(macro[method], context=f"PNNL macro {method}")
        require_exact_keys(
            row,
            {
                "pre_false_alarm_state_count",
                "miss_state_count",
                "macro_restricted_post_delay_fraction",
            },
            context=f"PNNL macro {method}",
        )
        expected_pre = sum(
            indexed[(cohort, state, method)]["pre_false_alarm"]
            for cohort in cohort_order
            for state in (0, 1)
        )
        expected_miss = sum(
            indexed[(cohort, state, method)]["miss"]
            for cohort in cohort_order
            for state in (0, 1)
        )
        expected_delay = (
            sum(
                aggregate_index[(cohort, method)]["restricted_post_delay_fraction"]
                for cohort in cohort_order
            )
            / 11
        )
        if (
            row["pre_false_alarm_state_count"] != expected_pre
            or row["miss_state_count"] != expected_miss
            or not close(
                require_number(
                    row["macro_restricted_post_delay_fraction"],
                    context=f"PNNL macro delay {method}",
                ),
                expected_delay,
            )
        ):
            raise PublicationDataError("PNNL macro aggregate is inconsistent.")
    comparisons = require_mapping(
        aggregate_row["comparisons"], context="PNNL comparisons"
    )
    require_exact_keys(
        comparisons, {"dfr", "online_logistic"}, context="PNNL comparisons"
    )
    retention_conditions: list[bool] = []
    for comparator in ("dfr", "online_logistic"):
        row = require_mapping(
            comparisons[comparator], context=f"PNNL comparison {comparator}"
        )
        require_exact_keys(
            row,
            {
                "cohort_delay_differences",
                "macro_delay_difference",
                "primary_95_percentile_interval",
                "calibration_pair_95_percentile_sensitivity",
                "exact_sign_flip_two_sided_p",
                "no_worse_pre_false_alarm",
                "strictly_lower_macro_delay",
                "retention_condition_pass",
            },
            context=f"PNNL comparison {comparator}",
        )
        expected_effects = [
            aggregate_index[(cohort, "space_composite")][
                "restricted_post_delay_fraction"
            ]
            - aggregate_index[(cohort, comparator)]["restricted_post_delay_fraction"]
            for cohort in cohort_order
        ]
        effects = row["cohort_delay_differences"]
        if (
            not isinstance(effects, list)
            or len(effects) != 11
            or any(
                not close(
                    require_number(value, context="PNNL cohort delay effect"),
                    expected,
                )
                for value, expected in zip(effects, expected_effects, strict=True)
            )
        ):
            raise PublicationDataError("PNNL cohort delay effects are inconsistent.")
        macro_effect = sum(expected_effects) / 11
        no_worse = (
            macro["space_composite"]["pre_false_alarm_state_count"]
            <= macro[comparator]["pre_false_alarm_state_count"]
        )
        lower_delay = macro_effect < 0
        passed = no_worse and lower_delay
        if (
            not close(
                require_number(
                    row["macro_delay_difference"],
                    context=f"PNNL macro effect {comparator}",
                ),
                macro_effect,
            )
            or row["no_worse_pre_false_alarm"] is not no_worse
            or row["strictly_lower_macro_delay"] is not lower_delay
            or row["retention_condition_pass"] is not passed
        ):
            raise PublicationDataError("PNNL comparison Boolean is inconsistent.")
        for interval_name in (
            "primary_95_percentile_interval",
            "calibration_pair_95_percentile_sensitivity",
        ):
            interval = row[interval_name]
            if (
                not isinstance(interval, list)
                or len(interval) != 2
                or require_number(interval[0], context=f"PNNL {interval_name} lower")
                > require_number(interval[1], context=f"PNNL {interval_name} upper")
            ):
                raise PublicationDataError("PNNL comparison interval is invalid.")
        p_value = require_number(
            row["exact_sign_flip_two_sided_p"],
            context=f"PNNL sign-flip p {comparator}",
        )
        if not 0 <= p_value <= 1:
            raise PublicationDataError("PNNL sign-flip p-value is invalid.")
        retention_conditions.append(passed)
    expected_retention = all(retention_conditions)
    if (
        require_bool(
            aggregate_row["retention_pass"], context="PNNL aggregate retention"
        )
        is not expected_retention
    ):
        raise PublicationDataError("PNNL aggregate retention Boolean is inconsistent.")
    return rows, aggregate_row


def load_exact_npy(
    path: Path,
    *,
    dtype: str,
    shape: tuple[int, ...],
    context: str,
) -> np.ndarray:
    """Load one canonical C-order NPY file with no trailing bytes."""

    try:
        with path.open("rb") as handle:
            array = np.lib.format.read_array(handle, allow_pickle=False)
            trailing = handle.read(1)
    except (OSError, EOFError, ValueError) as error:
        raise PublicationDataError(
            f"Cannot load {context} as strict NPY: {error}"
        ) from error
    if not isinstance(array, np.ndarray):
        raise PublicationDataError(f"{context} must be one NPY array.")
    if trailing:
        raise PublicationDataError(f"{context} NPY file has trailing bytes.")
    if (
        array.dtype.str != dtype
        or array.shape != shape
        or not array.flags.c_contiguous
        or array.flags.f_contiguous
    ):
        raise PublicationDataError(
            f"{context} must have dtype {dtype}, shape {shape}, and C order."
        )
    return array


def validate_pnnl_randomization(
    value: Any,
    *,
    alarm_counts_path: Path,
    maximum_log_e_path: Path,
    pittsburgh_cohorts: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Cross-check the complete PNNL paired-swap audit and both NPY arrays."""

    audit = require_mapping(value, context="PNNL randomization audit")
    require_exact_keys(
        audit,
        PNNL_RANDOMIZATION_KEYS,
        context="PNNL randomization audit",
    )
    if (
        audit["schema_version"] != "run6-pnnl-randomization-audit-v1"
        or audit["seeds"]
        != list(range(610_700, 610_700 + PNNL_RANDOMIZATION_REPLICATES))
        or audit["method_order"] != list(PNNL_METHOD_IDS)
        or audit["claim_scope"]
        != (
            "implementation and exact randomized paired design only; "
            "not a natural hardware null"
        )
    ):
        raise PublicationDataError(
            "PNNL randomization audit identity, seeds, method order, or scope changed."
        )
    if len(pittsburgh_cohorts) != 11:
        raise PublicationDataError(
            "PNNL randomization validation requires 11 Pittsburgh cohorts."
        )

    alarm_counts = load_exact_npy(
        alarm_counts_path,
        dtype="<i8",
        shape=(PNNL_RANDOMIZATION_REPLICATES, len(PNNL_METHOD_IDS)),
        context="PNNL randomization alarm counts",
    )
    maximum_log_e = load_exact_npy(
        maximum_log_e_path,
        dtype="<f8",
        shape=(PNNL_RANDOMIZATION_REPLICATES, len(PNNL_METHOD_IDS)),
        context="PNNL randomization maximum log-e",
    )
    if (
        np.any(alarm_counts < 0)
        or np.any(alarm_counts > PNNL_PATH_STATE_EPISODES)
        or not np.all(np.isfinite(maximum_log_e))
    ):
        raise PublicationDataError(
            "PNNL randomization arrays contain an impossible count or non-finite log-e."
        )
    crossed = maximum_log_e >= math.log(100.0)
    if not np.array_equal(alarm_counts > 0, crossed):
        raise PublicationDataError(
            "PNNL randomization count/max arrays disagree on whether any episode "
            "crossed 100."
        )

    rows = audit["path_state_method_rows"]
    expected_rows = [
        (cohort_index, cohort["cohort_id"], logical_state, method)
        for cohort_index, cohort in enumerate(pittsburgh_cohorts)
        for logical_state in (0, 1)
        for method in PNNL_METHOD_IDS
    ]
    if not isinstance(rows, list) or len(rows) != len(expected_rows):
        raise PublicationDataError(
            "PNNL randomization audit must contain 110 ordered path-state-method rows."
        )
    row_alarm_totals = {method: 0 for method in PNNL_METHOD_IDS}
    row_maxima = {method: -math.inf for method in PNNL_METHOD_IDS}
    for index, (raw, expected) in enumerate(zip(rows, expected_rows, strict=True)):
        row = require_mapping(
            raw,
            context=f"PNNL randomization path-state-method row[{index}]",
        )
        require_exact_keys(
            row,
            PNNL_RANDOMIZATION_ROW_KEYS,
            context=f"PNNL randomization path-state-method row[{index}]",
        )
        cohort_index, cohort_id, logical_state, method = expected
        if (
            row["cohort_index"] != cohort_index
            or row["cohort_id"] != cohort_id
            or row["logical_state"] != logical_state
            or row["method"] != method
        ):
            raise PublicationDataError(
                "PNNL randomization path-state-method row ordering changed."
            )
        fraction = require_number(
            row["alarm_fraction"],
            context=f"PNNL randomization row[{index}] alarm fraction",
        )
        if not 0 <= fraction <= 1:
            raise PublicationDataError(
                "PNNL randomization path-state alarm fraction is outside [0,1]."
            )
        alarm_count = round(fraction * PNNL_RANDOMIZATION_REPLICATES)
        if fraction != alarm_count / PNNL_RANDOMIZATION_REPLICATES:
            raise PublicationDataError(
                "PNNL path-state alarm fraction is not an exact 1/256 multiple."
            )
        maximum = require_number(
            row["maximum_log_e_over_replicates"],
            context=f"PNNL randomization row[{index}] maximum log-e",
        )
        if (alarm_count > 0) != (maximum >= math.log(100.0)):
            raise PublicationDataError(
                "PNNL path-state alarm fraction and maximum log-e disagree."
            )
        row_alarm_totals[method] += alarm_count
        row_maxima[method] = max(row_maxima[method], maximum)

    overall = require_mapping(
        audit["overall_episode_alarm_fraction"],
        context="PNNL randomization overall alarm fractions",
    )
    histograms = require_mapping(
        audit["alarmed_episode_count_histogram"],
        context="PNNL randomization alarmed-episode histograms",
    )
    maximum_summaries = require_mapping(
        audit["maximum_log_e_summary"],
        context="PNNL randomization maximum log-e summaries",
    )
    for collection, label in (
        (overall, "overall alarm fractions"),
        (histograms, "alarmed-episode histograms"),
        (maximum_summaries, "maximum log-e summaries"),
    ):
        require_exact_keys(
            collection,
            PNNL_METHOD_IDS,
            context=f"PNNL randomization {label}",
        )

    denominator = PNNL_RANDOMIZATION_REPLICATES * PNNL_PATH_STATE_EPISODES
    for method_index, method in enumerate(PNNL_METHOD_IDS):
        counts = alarm_counts[:, method_index]
        maxima = maximum_log_e[:, method_index]
        total = int(np.sum(counts))
        if row_alarm_totals[method] != total:
            raise PublicationDataError(
                f"PNNL {method} path-state alarm fractions disagree with counts NPY."
            )
        expected_overall = float(np.sum(counts) / denominator)
        observed_overall = require_number(
            overall[method],
            context=f"PNNL randomization {method} overall alarm fraction",
        )
        if observed_overall != expected_overall:
            raise PublicationDataError(
                f"PNNL {method} overall alarm fraction disagrees with counts NPY."
            )
        validate_integer_histogram(
            histograms[method],
            expected=integer_histogram(counts),
            maximum_bin=PNNL_PATH_STATE_EPISODES,
            total=PNNL_RANDOMIZATION_REPLICATES,
            context=f"PNNL randomization {method} alarmed-episode histogram",
        )
        summary = require_mapping(
            maximum_summaries[method],
            context=f"PNNL randomization {method} maximum log-e summary",
        )
        require_exact_keys(
            summary,
            PNNL_MAXIMUM_SUMMARY_KEYS,
            context=f"PNNL randomization {method} maximum log-e summary",
        )
        expected_summary = {
            "minimum": float(np.min(maxima)),
            "median": float(np.median(maxima)),
            "maximum": float(np.max(maxima)),
        }
        for key in PNNL_MAXIMUM_SUMMARY_KEYS:
            require_number(
                summary[key],
                context=f"PNNL randomization {method} maximum log-e {key}",
            )
        if dict(summary) != expected_summary:
            raise PublicationDataError(
                f"PNNL {method} maximum log-e summary disagrees with maxima NPY."
            )
        if row_maxima[method] != expected_summary["maximum"]:
            raise PublicationDataError(
                f"PNNL {method} per-row maxima disagree with maxima NPY."
            )
    return audit


def validate_pnnl_resources(value: Any) -> Mapping[str, Any]:
    resources = require_mapping(value, context="PNNL resource ledger")
    integer_keys = {
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
        "peak_rss_kib",
        "output_bytes_excluding_results_manifest",
        "output_bytes_including_results_manifest",
    }
    require_exact_keys(
        resources,
        {
            *integer_keys,
            "adaptive_state_ledger",
            "wall_seconds",
            "held_value_processing_wall_seconds",
        },
        context="PNNL resource ledger",
    )
    for key in integer_keys:
        require_int(resources[key], context=f"PNNL resource {key}", minimum=0)
    for key in ("wall_seconds", "held_value_processing_wall_seconds"):
        if require_number(resources[key], context=f"PNNL resource {key}") < 0:
            raise PublicationDataError(f"PNNL resource {key} cannot be negative.")
    require_mapping(
        resources["adaptive_state_ledger"],
        context="PNNL adaptive-state ledger",
    )
    if (
        resources["path_groups"] != 11
        or resources["path_state_streams"] != 22
        or resources["bootstrap_replicates_per_path_state_method"] != 4_096
        or resources["randomization_replicates"] != 256
        or resources["fit_physical_circuit_shots"]
        != 2 * resources["fit_paired_shot_pairs"]
        or resources["surveillance_physical_circuit_shots"]
        != 2 * resources["surveillance_paired_shot_pairs"]
        or resources["surveillance_formal_eprocess_shot_updates"]
        != resources["surveillance_paired_shot_pairs"]
        or resources["output_bytes_including_results_manifest"]
        < resources["output_bytes_excluding_results_manifest"]
    ):
        raise PublicationDataError("PNNL resource-ledger identities changed.")
    return resources


def validate_pnnl_manifest(
    path: Path,
    *,
    pittsburgh_path: Path,
    pittsburgh_evidence: Mapping[str, Any],
    original_ratification_path: Path,
    repair_ratification_path: Path,
) -> tuple[
    Mapping[str, Any],
    list[dict[str, Any]],
    Mapping[str, Any],
    Mapping[str, Any],
]:
    manifest = require_mapping(load_json(path), context="PNNL results manifest")
    require_exact_keys(manifest, PNNL_MANIFEST_KEYS, context="PNNL results manifest")
    if (
        manifest["schema_version"] != "run6-pnnl-snapshot-results-v1"
        or manifest["protocol_id"] != PNNL_PROTOCOL
        or manifest["claim_label"]
        != "constructed circuit-and-hardware domain shift; not temporal drift"
        or manifest["formal_alarm_unit"] != "one update per complete paired shot"
        or manifest["pittsburgh_manifest_sha256"] != sha256_file(pittsburgh_path)
        or manifest["freeze_ratification_sha256"]
        != sha256_file(original_ratification_path)
        or manifest["repair_ratification_path"] != REPAIR_RATIFICATION_PATH
        or manifest["repair_ratification_sha256"]
        != sha256_file(repair_ratification_path)
    ):
        raise PublicationDataError("PNNL results manifest differs from the lock.")
    cohorts = pittsburgh_evidence["cohorts"]
    snapshot_ids = pittsburgh_evidence["snapshot_ids"]
    metadata = require_mapping(
        manifest["metadata_validation"], context="PNNL metadata validation"
    )
    require_exact_keys(
        metadata,
        {"snapshots", "cohorts", "held_payloads_statted"},
        context="PNNL metadata validation",
    )
    if dict(metadata) != {
        "snapshots": len(snapshot_ids),
        "cohorts": len(cohorts),
        "held_payloads_statted": len(snapshot_ids),
    }:
        raise PublicationDataError("PNNL metadata-validation counts changed.")
    validate_pnnl_resources(manifest["resource_ledger"])
    core_records = {
        "first_unblinding_record": manifest["first_unblinding_record"],
        "state_rows": manifest["state_rows"],
        "aggregate_results": manifest["aggregate_results"],
        "randomization_audit": manifest["randomization_audit"],
        "randomization_alarm_counts": manifest["randomization_alarm_counts"],
        "randomization_maximum_log_e": manifest["randomization_maximum_log_e"],
    }
    expected_core_names = {
        "first_unblinding_record": "first_unblinding_record.json",
        "state_rows": "path_state_method_results.csv",
        "aggregate_results": "aggregate_results.json",
        "randomization_audit": "randomization_audit.json",
        "randomization_alarm_counts": "randomization_alarm_counts.npy",
        "randomization_maximum_log_e": "randomization_maximum_log_e.npy",
    }
    paths: dict[str, Path] = {}
    for key, record in core_records.items():
        candidate = resolve_artifact(record, path, context=f"PNNL {key}")
        relative = candidate.relative_to(path.parent.resolve()).as_posix()
        if relative != expected_core_names[key]:
            raise PublicationDataError(f"PNNL {key} artifact name changed.")
        if relative in paths:
            raise PublicationDataError("Duplicate PNNL artifact path.")
        paths[relative] = candidate
    for collection_name, expected_count in (
        ("trace_artifacts", 110),
        ("bootstrap_artifacts", 110),
    ):
        records = manifest[collection_name]
        if not isinstance(records, list) or len(records) != expected_count:
            raise PublicationDataError(
                f"PNNL {collection_name} must contain {expected_count} records."
            )
        for index, record in enumerate(records):
            candidate = resolve_artifact(
                record,
                path,
                context=f"PNNL {collection_name}[{index}]",
            )
            relative = candidate.relative_to(path.parent.resolve()).as_posix()
            expected_suffix = (
                "_log_e.npy"
                if collection_name == "trace_artifacts"
                else "_bootstrap_maxima.npy"
            )
            if not relative.endswith(expected_suffix):
                raise PublicationDataError(
                    f"PNNL {collection_name}[{index}] artifact name changed."
                )
            if relative in paths:
                raise PublicationDataError("Duplicate PNNL artifact path.")
            paths[relative] = candidate
    if len(paths) != 226:
        raise PublicationDataError("PNNL portable artifact contract is not 226 files.")
    unblinding_path = resolve_artifact(
        manifest["first_unblinding_record"],
        path,
        context="PNNL first-unblinding record",
    )
    unblinding = require_mapping(
        load_json(unblinding_path), context="PNNL first-unblinding record"
    )
    require_exact_keys(
        unblinding,
        {
            "schema_version",
            "utc",
            "git_commit",
            "config_sha256",
            "manifest_sha256",
            "freeze_ratification_sha256",
            "repair_ratification_path",
            "repair_ratification_sha256",
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
        or unblinding["config_sha256"] != manifest["config_sha256"]
        or unblinding["manifest_sha256"] != manifest["pittsburgh_manifest_sha256"]
        or unblinding["freeze_ratification_sha256"]
        != manifest["freeze_ratification_sha256"]
        or unblinding["repair_ratification_path"] != REPAIR_RATIFICATION_PATH
        or unblinding["repair_ratification_sha256"]
        != sha256_file(repair_ratification_path)
        or unblinding["repair_ratification_sha256"]
        != manifest["repair_ratification_sha256"]
        or unblinding["scores_computed_before_record"] is not False
    ):
        raise PublicationDataError("PNNL first-unblinding bindings are inconsistent.")
    package_lock = require_mapping(
        unblinding["package_lock"], context="PNNL unblinding package lock"
    )
    require_exact_keys(
        package_lock,
        {"path", "bytes", "sha256"},
        context="PNNL unblinding package lock",
    )
    if package_lock["sha256"] != manifest["package_lock_sha256"]:
        raise PublicationDataError("PNNL unblinding package-lock hash changed.")
    payload_hashes = require_mapping(
        manifest["held_payload_sha256"], context="PNNL held-payload hashes"
    )
    if set(payload_hashes) != set(snapshot_ids):
        raise PublicationDataError(
            "PNNL held-payload IDs disagree with the Pittsburgh snapshot lock."
        )
    for snapshot_id, digest in payload_hashes.items():
        if not isinstance(snapshot_id, str) or not snapshot_id:
            raise PublicationDataError("PNNL held-payload ID must be nonempty text.")
        require_sha256(digest, context=f"PNNL held-payload hash {snapshot_id}")
    payload_rows = unblinding["held_payloads"]
    if not isinstance(payload_rows, list) or len(payload_rows) != len(snapshot_ids):
        raise PublicationDataError(
            "PNNL first-unblinding payload registry is incomplete."
        )
    observed_payloads: dict[str, str] = {}
    for index, raw in enumerate(payload_rows):
        row = require_mapping(raw, context=f"PNNL unblinding payload[{index}]")
        require_exact_keys(
            row,
            {"snapshot_id", "path", "bytes", "sha256"},
            context=f"PNNL unblinding payload[{index}]",
        )
        snapshot_id = row["snapshot_id"]
        relative_path = row["path"]
        if (
            not isinstance(snapshot_id, str)
            or snapshot_id in observed_payloads
            or not isinstance(relative_path, str)
            or not relative_path
        ):
            raise PublicationDataError("PNNL unblinding payload metadata is invalid.")
        require_int(
            row["bytes"],
            context=f"PNNL unblinding payload bytes {snapshot_id}",
            minimum=1,
        )
        observed_payloads[snapshot_id] = require_sha256(
            row["sha256"],
            context=f"PNNL unblinding payload hash {snapshot_id}",
        )
    if observed_payloads != dict(payload_hashes):
        raise PublicationDataError(
            "PNNL first-unblinding and result payload hashes disagree."
        )
    state_path = resolve_artifact(
        manifest["state_rows"], path, context="PNNL state rows"
    )
    aggregate_path = resolve_artifact(
        manifest["aggregate_results"], path, context="PNNL aggregate results"
    )
    rows, aggregate = validate_pnnl_state_and_aggregate(
        state_path,
        load_json(aggregate_path),
        pittsburgh_cohorts=cohorts,
    )
    randomization = validate_pnnl_randomization(
        load_json(paths["randomization_audit.json"]),
        alarm_counts_path=paths["randomization_alarm_counts.npy"],
        maximum_log_e_path=paths["randomization_maximum_log_e.npy"],
        pittsburgh_cohorts=cohorts,
    )
    retention = require_bool(manifest["retention_pass"], context="PNNL retention")
    if retention is not aggregate["retention_pass"]:
        raise PublicationDataError("PNNL manifest/aggregate retention mismatch.")
    started = require_number(manifest["started_unix"], context="PNNL start time")
    unblinded = require_number(
        manifest["held_value_processing_started_unix"],
        context="PNNL held-value start time",
    )
    finished = require_number(manifest["finished_unix"], context="PNNL finish time")
    if not started <= unblinded <= finished:
        raise PublicationDataError("PNNL unblinding time order is invalid.")
    return manifest, rows, aggregate, randomization


def validate_decision(
    value: Any,
    *,
    event: Mapping[str, Any],
    risk: Mapping[str, Any],
    randomization: Mapping[str, Any],
    threshold_bootstrap: Mapping[str, Any],
    pnnl_manifest: Mapping[str, Any],
    pnnl_path: Path,
    repair_ratification_path: Path,
) -> Mapping[str, Any]:
    decision = require_mapping(value, context="Run 6 decision summary")
    require_exact_keys(decision, DECISION_KEYS, context="Run 6 decision summary")
    if (
        decision["schema_version"] != "run6-google-decision-v1"
        or decision["repair_ratification_path"] != REPAIR_RATIFICATION_PATH
        or decision["repair_ratification_sha256"]
        != sha256_file(repair_ratification_path)
        or decision["summary_scope"] != "full_run6_locked_decision"
        or decision["primary_label"] != "actual_xor_correlated_matching_prediction"
        or decision["primary_budget_shots"] != 20
        or decision["bootstrap_changes_primary_boolean"] is not False
        or decision["pnnl_results_manifest_sha256"] != sha256_file(pnnl_path)
    ):
        raise PublicationDataError("Decision summary is not the completed locked run.")
    estimates = risk["point_estimates"]["correlated_matching_mismatch"]
    capture = {
        method: int(estimates[method]["budgets"]["20"]["captured_mismatches"])
        for method in METHOD_IDS
    }
    if decision["top20_capture"] != capture:
        raise PublicationDataError("Decision top-20 capture differs from risk summary.")
    evidence = decision["method_input_parity_evidence"]
    threshold_bootstrap_complete = (
        threshold_bootstrap.get("schema_version")
        == "run6-google-threshold-bootstrap-v1"
        and threshold_bootstrap.get("status")
        == "descriptive_only_does_not_replace_frozen_threshold"
        and threshold_bootstrap.get("replicates")
        == GOOGLE_THRESHOLD_BOOTSTRAP_REPLICATES
        and set(
            require_mapping(
                threshold_bootstrap.get("summaries"),
                context="validated Google threshold-bootstrap summaries",
            )
        )
        == set(METHOD_IDS)
        and isinstance(threshold_bootstrap.get("replicate_results"), list)
        and len(threshold_bootstrap["replicate_results"])
        == GOOGLE_THRESHOLD_BOOTSTRAP_REPLICATES
    )
    expected_predicates = {
        ATOMIC_PREDICATES[0]: bool(event["space"]["windows"]["primary"]["detected"]),
        ATOMIC_PREDICATES[1]: int(event["space"]["pre_event_alert_count"]) <= 9,
        ATOMIC_PREDICATES[2]: True,
        ATOMIC_PREDICATES[3]: capture["space"] >= capture["m0"] + 1,
        ATOMIC_PREDICATES[4]: capture["space"] >= capture["m3"] + 1,
        ATOMIC_PREDICATES[5]: all(
            set(event[method]["windows"]) == set(WINDOWS) for method in METHOD_IDS
        )
        and risk["uncertainty"]["replicates"] == 2_000
        and threshold_bootstrap_complete,
        ATOMIC_PREDICATES[6]: parity_evidence_valid(evidence),
    }
    observed_predicates = require_mapping(
        decision["atomic_predicates"], context="decision atomic predicates"
    )
    require_exact_keys(
        observed_predicates, ATOMIC_PREDICATES, context="decision atomic predicates"
    )
    if dict(observed_predicates) != expected_predicates:
        raise PublicationDataError("Decision atomic predicates were not recomputed.")
    if decision["mandatory_contextual_controls_reported"] is not True:
        raise PublicationDataError("Mandatory contextual controls are missing.")
    google_pass = all(expected_predicates.values())
    if (
        require_bool(decision["google_primary_pass"], context="Google primary pass")
        is not google_pass
    ):
        raise PublicationDataError("Google primary Boolean is inconsistent.")
    random_audit = require_mapping(
        decision["randomization_audit"], context="decision randomization audit"
    )
    require_exact_keys(
        random_audit,
        {
            "status",
            "manifest_sha256",
            "space_crossing_count_at_100",
            "replicates",
            "clopper_pearson_95",
            "changes_primary_boolean",
        },
        context="decision randomization audit",
    )
    if (
        random_audit["status"] != "completed_and_hash_verified"
        or random_audit["space_crossing_count_at_100"]
        != randomization["crossing_counts_at_100"]["space"]
        or random_audit["replicates"] != 256
        or random_audit["clopper_pearson_95"]
        != randomization["space_crossing_clopper_pearson_95"]
        or random_audit["changes_primary_boolean"] is not False
    ):
        raise PublicationDataError("Decision randomization audit is inconsistent.")
    pnnl_pass = require_bool(pnnl_manifest["retention_pass"], context="PNNL pass")
    if (
        require_bool(decision["pnnl_retention_pass"], context="decision PNNL pass")
        is not pnnl_pass
    ):
        raise PublicationDataError("Decision PNNL Boolean is inconsistent.")
    overall = google_pass and pnnl_pass
    if (
        require_bool(
            decision["overall_run6_advantage"],
            context="locked conjunctive empirical gate",
        )
        is not overall
    ):
        raise PublicationDataError("Overall Run 6 Boolean is inconsistent.")
    expected_reasons = [
        key for key in ATOMIC_PREDICATES if not expected_predicates[key]
    ]
    if not pnnl_pass:
        expected_reasons.append("pnnl_retention_failed")
    if decision["negative_result_reasons"] != expected_reasons:
        raise PublicationDataError("Decision failure reasons are inconsistent.")
    return decision


def validate_outcome_manifest(
    path: Path,
    *,
    detector_path: Path,
    detector_manifest: Mapping[str, Any],
    event: Mapping[str, Any],
    randomization_path: Path,
    randomization: Mapping[str, Any],
    threshold_bootstrap: Mapping[str, Any],
    pnnl_path: Path,
    pnnl_manifest: Mapping[str, Any],
    repair_ratification_path: Path,
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    manifest = require_mapping(load_json(path), context="outcome manifest")
    require_exact_keys(manifest, OUTCOME_MANIFEST_KEYS, context="outcome manifest")
    if (
        manifest["schema_version"] != "run6-google-outcome-manifest-v1"
        or manifest["protocol_id"] != GOOGLE_PROTOCOL
        or manifest["outcome_accessed_after_detector_freeze"] is not True
        or manifest["detector_manifest_sha256"] != sha256_file(detector_path)
        or manifest["detector_manifest_git_commit"] != detector_manifest["git_commit"]
        or manifest["config_sha256"] != detector_manifest["config_sha256"]
        or manifest["method_spec_sha256"] != detector_manifest["method_spec_sha256"]
        or manifest["freeze_ratification_sha256"]
        != detector_manifest["freeze_ratification_sha256"]
        or manifest["repair_ratification_path"] != REPAIR_RATIFICATION_PATH
        or manifest["repair_ratification_sha256"]
        != sha256_file(repair_ratification_path)
        or manifest["primary_label"] != "actual_xor_correlated_matching_prediction"
        or manifest["secondary_label"] != "actual_xor_pymatching_prediction"
    ):
        raise PublicationDataError("Outcome manifest bindings are inconsistent.")
    final_inputs = require_mapping(
        manifest["final_aggregation_inputs"],
        context="outcome final aggregation inputs",
    )
    require_exact_keys(
        final_inputs,
        {
            "status",
            "randomization_manifest_sha256",
            "pnnl_results_manifest_sha256",
        },
        context="outcome final aggregation inputs",
    )
    if (
        final_inputs["status"] != "completed_and_hash_verified"
        or final_inputs["randomization_manifest_sha256"]
        != sha256_file(randomization_path)
        or final_inputs["pnnl_results_manifest_sha256"] != sha256_file(pnnl_path)
    ):
        raise PublicationDataError("Outcome manifest lacks the two completed arms.")
    artifacts = artifact_map(
        manifest["artifacts"], path, context="outcome manifest.artifacts"
    )
    if set(artifacts) != {
        "outcomes.csv",
        "risk_summary.json",
        "decision_summary.json",
    }:
        raise PublicationDataError("Outcome artifact contract is incomplete.")
    risk = validate_risk_summary(
        load_json(artifacts["risk_summary.json"]),
        detector_hash=sha256_file(detector_path),
    )
    if risk["outcome_table_sha256"] != sha256_file(artifacts["outcomes.csv"]):
        raise PublicationDataError("Risk summary/outcome table hash mismatch.")
    decision = validate_decision(
        load_json(artifacts["decision_summary.json"]),
        event=event,
        risk=risk,
        randomization=randomization,
        threshold_bootstrap=threshold_bootstrap,
        pnnl_manifest=pnnl_manifest,
        pnnl_path=pnnl_path,
        repair_ratification_path=repair_ratification_path,
    )
    evidence = decision["method_input_parity_evidence"]
    if (
        manifest["method_input_parity_evidence_sha256"]
        != hashlib.sha256(canonical_json_bytes(evidence)).hexdigest()
        or manifest["shared_outcome_label_bundle_sha256"]
        != evidence["shared_outcome_label_bundle"]["sha256"]
    ):
        raise PublicationDataError(
            "Outcome manifest/method-input parity evidence hash mismatch."
        )
    if decision["randomization_audit"]["manifest_sha256"] != sha256_file(
        randomization_path
    ):
        raise PublicationDataError("Decision/randomization manifest hash mismatch.")
    return manifest, risk, decision


def tex_escape(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in text)


def yes_no(value: bool) -> str:
    return "Pass" if value else "Fail"


def format_number(value: Any, digits: int = 4) -> str:
    if value is None:
        return "---"
    number = float(value)
    if not math.isfinite(number):
        raise PublicationDataError("Cannot render a non-finite publication number.")
    return f"{number:.{digits}f}"


def format_estimate_interval(
    estimate: Any,
    interval: Mapping[str, Any],
    *,
    digits: int = 4,
) -> str:
    if interval["valid_replicates"] == 0:
        return f"{format_number(estimate, digits)} [---, ---]"
    return (
        f"{format_number(estimate, digits)} "
        f"[{format_number(interval['lower'], digits)}, "
        f"{format_number(interval['upper'], digits)}]"
    )


def format_estimate_interval_with_count(
    estimate: Any,
    interval: Mapping[str, Any],
    *,
    digits: int = 4,
) -> str:
    return (
        f"{format_estimate_interval(estimate, interval, digits=digits)}; "
        f"n={interval['valid_replicates']}"
    )


def format_percentile_triplet(
    summary: Mapping[str, Any],
    *,
    digits: int = 4,
) -> str:
    return (
        f"{format_number(summary['lower_2_5'], digits)} / "
        f"{format_number(summary['median'], digits)} / "
        f"{format_number(summary['upper_97_5'], digits)}"
    )


def format_min_median_max(
    summary: Mapping[str, Any],
    *,
    digits: int = 4,
) -> str:
    return (
        f"{format_number(summary['minimum'], digits)} / "
        f"{format_number(summary['median'], digits)} / "
        f"{format_number(summary['maximum'], digits)}"
    )


def format_histogram(histogram: Mapping[str, Any]) -> str:
    return ", ".join(f"{key}:{histogram[key]}" for key in sorted(histogram, key=int))


def table(
    *,
    columns: str,
    header: Sequence[str],
    rows: Sequence[Sequence[Any]],
) -> str:
    lines = [
        r"\noindent\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{" + columns + "}",
        r"\toprule",
        " & ".join(tex_escape(cell) for cell in header) + r" \\",
        r"\midrule",
    ]
    lines.extend(" & ".join(tex_escape(cell) for cell in row) + r" \\" for row in rows)
    lines.extend((r"\bottomrule", r"\end{tabular}", r"}", r"\par", ""))
    return "\n".join(lines)


def write_tables(
    output: Path,
    *,
    event: Mapping[str, Any],
    thresholds: Mapping[str, Any],
    risk: Mapping[str, Any],
    decision: Mapping[str, Any],
    randomization: Mapping[str, Any],
    threshold_bootstrap: Mapping[str, Any],
    aggregate: Mapping[str, Any],
    pnnl_state_rows: Sequence[Mapping[str, Any]],
    pnnl_randomization: Mapping[str, Any],
    detector_manifest: Mapping[str, Any],
    pnnl_manifest: Mapping[str, Any],
    pittsburgh_cohorts: Sequence[Mapping[str, Any]],
) -> None:
    gate_rows = [
        (
            PREDICATE_LABELS[key],
            "Google primary gate",
            yes_no(decision["atomic_predicates"][key]),
        )
        for key in ATOMIC_PREDICATES
    ]
    gate_rows.extend(
        (
            (
                "Google aggregate gate",
                "Primary empirical arm",
                yes_no(decision["google_primary_pass"]),
            ),
            (
                "PNNL retention gate",
                "Required auxiliary arm",
                yes_no(decision["pnnl_retention_pass"]),
            ),
            (
                "Overall locked dual-provenance gate",
                "Google AND PNNL",
                yes_no(decision["overall_run6_advantage"]),
            ),
        )
    )
    (output / "gate_decision_table.tex").write_text(
        table(
            columns="p{0.55\\linewidth}p{0.27\\linewidth}c",
            header=("Locked condition", "Status in protocol", "Result"),
            rows=gate_rows,
        ),
        encoding="utf-8",
    )
    event_rows = []
    for method in METHOD_IDS:
        name, role = METHOD_METADATA[method]
        primary = event[method]["windows"]["primary"]
        event_rows.append(
            (
                name,
                role,
                event[method]["pre_event_alert_count"],
                (
                    ", ".join(
                        str(shot) for shot in event[method]["pre_event_alert_shots"]
                    )
                    or "none"
                ),
                yes_no(primary["detected"]),
                primary["first_alert_shot"]
                if primary["first_alert_shot"] is not None
                else "---",
                primary["first_alert_role"]
                if primary["first_alert_role"] is not None
                else "---",
                yes_no(event[method]["windows"]["narrow"]["detected"]),
                yes_no(event[method]["windows"]["wide"]["detected"]),
            )
        )
    (output / "google_event_table.tex").write_text(
        table(
            columns="llrp{0.22\\linewidth}rrrrr",
            header=(
                "Method",
                "Preregistered role",
                "Pre-event",
                "Pre-event shots",
                "Primary",
                "First shot",
                "Role",
                "Narrow",
                "Wide",
            ),
            rows=event_rows,
        ),
        encoding="utf-8",
    )

    threshold_rows = []
    for method in METHOD_IDS:
        name, role = METHOD_METADATA[method]
        row = thresholds[method]
        threshold_rows.append(
            (
                name,
                role,
                format_number(row["threshold"], 6),
                f"{row['validation_alert_count']}/{row['max_validation_alerts']}",
                format_number(row["secondary_zero_alert_threshold"], 6),
                row["secondary_validation_alert_count"],
                "Hash-verified",
            )
        )
    (output / "google_threshold_frontier_table.tex").write_text(
        table(
            columns="llrrrrl",
            header=(
                "Method",
                "Preregistered role",
                "Primary threshold",
                "Validation alerts/budget",
                "Zero-alert threshold",
                "Zero-alert count",
                "Complete frontier",
            ),
            rows=threshold_rows,
        ),
        encoding="utf-8",
    )

    threshold_bootstrap_rows = []
    for method in METHOD_IDS:
        name, _ = METHOD_METADATA[method]
        summary = threshold_bootstrap["summaries"][method]
        threshold_bootstrap_rows.append(
            (
                name,
                format_number(summary["frozen_threshold"], 6),
                format_percentile_triplet(
                    summary["selected_primary_threshold_percentiles"],
                    digits=6,
                ),
                format_histogram(summary["selected_primary_alert_count_frequency"]),
                format_percentile_triplet(
                    summary["alert_count_at_frozen_threshold_percentiles"],
                    digits=2,
                ),
                format_histogram(summary["alert_count_at_frozen_threshold_frequency"]),
                format_percentile_triplet(
                    summary["selected_zero_alert_threshold_percentiles"],
                    digits=6,
                ),
            )
        )
    (output / "google_threshold_bootstrap_table.tex").write_text(
        table(
            columns="lrrrrrr",
            header=(
                "Method",
                "Frozen threshold",
                "Selected threshold (2.5/50/97.5%)",
                "Selected-alert histogram",
                "Alerts at frozen threshold (2.5/50/97.5%)",
                "Frozen-alert histogram",
                "Zero-alert threshold (2.5/50/97.5%)",
            ),
            rows=threshold_bootstrap_rows,
        ),
        encoding="utf-8",
    )

    risk_rows = []
    label_metadata = {
        "correlated_matching_mismatch": "Primary: correlated matching",
        "pymatching_mismatch": "Secondary: PyMatching",
    }
    for label in LABEL_IDS:
        for budget in RISK_BUDGETS:
            for method in METHOD_IDS:
                point = risk["point_estimates"][label][method]["budgets"][str(budget)]
                intervals = risk["uncertainty"]["method_intervals"][label][method][
                    str(budget)
                ]
                captured_interval = format_estimate_interval_with_count(
                    point["captured_mismatches"],
                    intervals["captured_mismatches"],
                    digits=2,
                )
                risk_rows.append(
                    (
                        label_metadata[label],
                        budget,
                        METHOD_METADATA[method][0],
                        METHOD_METADATA[method][1],
                        (
                            f"{point['captured_mismatches']}/"
                            f"{point['total_mismatches']} {captured_interval}"
                        ),
                        format_estimate_interval_with_count(
                            point["mismatch_recall"],
                            intervals["mismatch_recall"],
                            digits=5,
                        ),
                        format_estimate_interval_with_count(
                            point["alert_precision"],
                            intervals["alert_precision"],
                            digits=5,
                        ),
                        format_estimate_interval_with_count(
                            point["retained_mismatch_rate"],
                            intervals["retained_mismatch_rate"],
                            digits=5,
                        ),
                        format_number(point["coverage"], 5),
                    )
                )
    (output / "google_risk_budget_table.tex").write_text(
        table(
            columns="lrlp{0.20\\linewidth}rrrrr",
            header=(
                "Label",
                "Budget",
                "Method",
                "Preregistered role",
                "Captured/total [95%; n]",
                "Recall [95%; n]",
                "Precision [95%; n]",
                "Retained rate [95%; n]",
                "Coverage",
            ),
            rows=risk_rows,
        ),
        encoding="utf-8",
    )

    uncertainty_rows = []
    differences = risk["uncertainty"]["space_comparator_difference_intervals"]
    for label in LABEL_IDS:
        for comparator in ("m0", "m3"):
            comparison_id = f"space_minus_{comparator}"
            for budget in RISK_BUDGETS:
                row = differences[label][comparison_id][str(budget)]
                point_space = risk["point_estimates"][label]["space"]["budgets"][
                    str(budget)
                ]
                point_comparator = risk["point_estimates"][label][comparator][
                    "budgets"
                ][str(budget)]
                point_differences = {
                    metric: (
                        None
                        if point_space[metric] is None
                        or point_comparator[metric] is None
                        else float(point_space[metric])
                        - float(point_comparator[metric])
                    )
                    for metric in (
                        "captured_mismatches",
                        "mismatch_recall",
                        "alert_precision",
                        "retained_mismatch_rate",
                    )
                }

                uncertainty_rows.append(
                    (
                        label_metadata[label],
                        budget,
                        f"S-PACE minus {METHOD_METADATA[comparator][0]}",
                        format_estimate_interval_with_count(
                            point_differences["captured_mismatches"],
                            row["captured_mismatches"],
                            digits=2,
                        ),
                        format_estimate_interval_with_count(
                            point_differences["mismatch_recall"],
                            row["mismatch_recall"],
                            digits=5,
                        ),
                        format_estimate_interval_with_count(
                            point_differences["alert_precision"],
                            row["alert_precision"],
                            digits=5,
                        ),
                        format_estimate_interval_with_count(
                            point_differences["retained_mismatch_rate"],
                            row["retained_mismatch_rate"],
                            digits=5,
                        ),
                        (
                            "Gate capture contrast"
                            if label == LABEL_IDS[0] and budget == 20
                            else "Descriptive"
                        ),
                    )
                )
    (output / "google_uncertainty_table.tex").write_text(
        table(
            columns="lrlrrrrl",
            header=(
                "Label",
                "Budget",
                "Locked contrast",
                "Captured difference [95%; n]",
                "Recall difference [95%; n]",
                "Precision difference [95%; n]",
                "Retained-rate difference [95%; n]",
                "Gate relevance",
            ),
            rows=uncertainty_rows,
        ),
        encoding="utf-8",
    )

    macro_rows = []
    for method in PNNL_METHOD_IDS:
        name, role = PNNL_METHOD_METADATA[method]
        row = aggregate["macro_by_method"][method]
        macro_rows.append(
            (
                name,
                role,
                row["pre_false_alarm_state_count"],
                row["miss_state_count"],
                format_number(row["macro_restricted_post_delay_fraction"]),
            )
        )
    (output / "pnnl_macro_table.tex").write_text(
        table(
            columns="llrrr",
            header=(
                "Method",
                "Preregistered role",
                "Pre-FA states",
                "Miss states",
                "Macro delay",
            ),
            rows=macro_rows,
        ),
        encoding="utf-8",
    )

    state_index = {
        (row["cohort_id"], row["logical_state"], row["method"]): row
        for row in pnnl_state_rows
    }
    pnnl_state_table_rows = []
    for cohort_index, cohort in enumerate(pittsburgh_cohorts, start=1):
        for logical_state in (0, 1):
            method_cells = []
            for method in PNNL_METHOD_IDS:
                row = state_index[(cohort["cohort_id"], logical_state, method)]
                method_cells.append(
                    f"{row['pre_false_alarm']}/{row['miss']}/"
                    f"{format_number(row['restricted_post_delay_fraction'], 4)}"
                )
            pnnl_state_table_rows.append(
                (
                    f"C{cohort_index:02d}",
                    cohort["cohort_id"],
                    logical_state,
                    *method_cells,
                )
            )
    if len(pnnl_state_table_rows) != 22:
        raise PublicationDataError("PNNL state reporting contract is not 22 rows.")
    (output / "pnnl_state_results_table.tex").write_text(
        table(
            columns="llrrrrrr",
            header=(
                "Plot ID",
                "Cohort ID",
                "State",
                *(PNNL_METHOD_METADATA[method][0] for method in PNNL_METHOD_IDS),
            ),
            rows=pnnl_state_table_rows,
        ),
        encoding="utf-8",
    )

    cohort_result_index = {
        (row["cohort_id"], row["method"]): row for row in aggregate["cohort_rows"]
    }
    pnnl_cohort_result_rows = []
    for cohort_index, cohort in enumerate(pittsburgh_cohorts, start=1):
        method_cells = []
        for method in PNNL_METHOD_IDS:
            row = cohort_result_index[(cohort["cohort_id"], method)]
            method_cells.append(
                f"{format_number(row['pre_false_alarm_mean'], 3)}/"
                f"{format_number(row['miss_mean'], 3)}/"
                f"{format_number(row['restricted_post_delay_fraction'], 4)}"
            )
        pnnl_cohort_result_rows.append(
            (
                f"C{cohort_index:02d}",
                cohort["cohort_id"],
                *method_cells,
            )
        )
    if len(pnnl_cohort_result_rows) != 11:
        raise PublicationDataError(
            "PNNL state-averaged cohort reporting contract is not 11 rows."
        )
    (output / "pnnl_cohort_results_table.tex").write_text(
        table(
            columns="llrrrrr",
            header=(
                "Plot ID",
                "Cohort ID",
                *(PNNL_METHOD_METADATA[method][0] for method in PNNL_METHOD_IDS),
            ),
            rows=pnnl_cohort_result_rows,
        ),
        encoding="utf-8",
    )

    comparison_rows = []
    for comparator in ("dfr", "online_logistic"):
        row = aggregate["comparisons"][comparator]
        primary_interval = row["primary_95_percentile_interval"]
        sensitivity_interval = row["calibration_pair_95_percentile_sensitivity"]
        comparison_rows.append(
            (
                PNNL_METHOD_METADATA[comparator][0],
                format_number(row["macro_delay_difference"]),
                (
                    f"[{format_number(primary_interval[0])}, "
                    f"{format_number(primary_interval[1])}]"
                ),
                (
                    f"[{format_number(sensitivity_interval[0])}, "
                    f"{format_number(sensitivity_interval[1])}]"
                ),
                format_number(row["exact_sign_flip_two_sided_p"], 6),
                yes_no(row["no_worse_pre_false_alarm"]),
                yes_no(row["strictly_lower_macro_delay"]),
                yes_no(row["retention_condition_pass"]),
            )
        )
    (output / "pnnl_comparison_table.tex").write_text(
        table(
            columns="lrrrrrrr",
            header=(
                "Comparator",
                "S-PACE minus comparator",
                "Path-bootstrap 95% interval",
                "Calibration-pair sensitivity 95% interval",
                "Exact sign-flip p",
                "No-worse pre-FA",
                "Lower delay",
                "Retention",
            ),
            rows=comparison_rows,
        ),
        encoding="utf-8",
    )

    cohort_rows = [
        (
            f"C{cohort_index:02d}",
            row["cohort_id"],
            row["basis"],
            row["distance"],
            row["rounds"],
            row["calibration_pair_id"],
            row["qasm_status"],
        )
        for cohort_index, row in enumerate(pittsburgh_cohorts, start=1)
    ]
    if len(cohort_rows) != 11:
        raise PublicationDataError("PNNL cohort/control reporting contract changed.")
    (output / "pnnl_cohort_control_table.tex").write_text(
        table(
            columns="llrrrrp{0.28\\linewidth}",
            header=(
                "Plot ID",
                "Cohort ID",
                "Basis",
                "d",
                "r",
                "Calibration pair",
                "QASM/control status",
            ),
            rows=cohort_rows,
        ),
        encoding="utf-8",
    )

    interval = randomization["space_crossing_clopper_pearson_95"]
    randomization_rows = []
    for method in FORMAL_METHOD_IDS:
        count = randomization["crossing_counts_at_100"][method]
        randomization_rows.append(
            (
                METHOD_METADATA[method][0],
                "Proper-prior e-process >= 100",
                count,
                randomization["replicate_count"],
                format_number(count / randomization["replicate_count"]),
                (
                    f"[{format_number(interval['lower'])}, "
                    f"{format_number(interval['upper'])}]"
                    if method == "space"
                    else "Not preregistered"
                ),
            )
        )
    randomization_rows.append(
        (
            "Any-family crossing at 600",
            "Proper-prior family diagnostic",
            randomization["familywide_any_crossing_count_at_600"],
            randomization["replicate_count"],
            "---",
            "---",
        )
    )
    (output / "randomization_table.tex").write_text(
        table(
            columns="llrrrr",
            header=(
                "Method/statistic",
                "Proper-prior quantity",
                "Count",
                "Replicates",
                "Fraction",
                "95% interval",
            ),
            rows=randomization_rows,
        ),
        encoding="utf-8",
    )

    randomization_index = {
        (row["cohort_id"], row["logical_state"], row["method"]): row
        for row in pnnl_randomization["path_state_method_rows"]
    }
    pnnl_randomization_episode_rows = []
    for cohort_index, cohort in enumerate(pittsburgh_cohorts, start=1):
        for logical_state in (0, 1):
            pnnl_randomization_episode_rows.append(
                (
                    f"C{cohort_index:02d}",
                    cohort["cohort_id"],
                    logical_state,
                    *(
                        format_number(
                            randomization_index[
                                (cohort["cohort_id"], logical_state, method)
                            ]["alarm_fraction"],
                            6,
                        )
                        for method in PNNL_METHOD_IDS
                    ),
                )
            )
    if len(pnnl_randomization_episode_rows) != 22:
        raise PublicationDataError(
            "PNNL randomization episode reporting contract is not 22 rows."
        )

    pnnl_randomization_summary_rows = []
    for method in PNNL_METHOD_IDS:
        name, _ = PNNL_METHOD_METADATA[method]
        pnnl_randomization_summary_rows.append(
            (
                name,
                format_number(
                    pnnl_randomization["overall_episode_alarm_fraction"][method],
                    6,
                ),
                format_histogram(
                    pnnl_randomization["alarmed_episode_count_histogram"][method]
                ),
                format_min_median_max(
                    pnnl_randomization["maximum_log_e_summary"][method],
                    digits=4,
                ),
            )
        )
    pnnl_randomization_text = (
        r"\noindent\textbf{Panel A: path--state alarm fractions.}\par"
        "\n"
        + table(
            columns="llrrrrrr",
            header=(
                "Plot ID",
                "Cohort ID",
                "State",
                *(PNNL_METHOD_METADATA[method][0] for method in PNNL_METHOD_IDS),
            ),
            rows=pnnl_randomization_episode_rows,
        )
        + "\n"
        + r"\noindent\textbf{Panel B: method summaries over 256 replicates.}\par"
        + "\n"
        + table(
            columns="lrrr",
            header=(
                "Method",
                "Episode alarm fraction",
                "Alarmed episodes/replicate histogram",
                "Maximum log-e (min/median/max)",
            ),
            rows=pnnl_randomization_summary_rows,
        )
    )
    (output / "pnnl_randomization_audit_table.tex").write_text(
        pnnl_randomization_text,
        encoding="utf-8",
    )

    google_exposure = detector_manifest["resources"]["record_exposure"]
    google_performance = detector_manifest["performance"]
    pnnl_resources = pnnl_manifest["resource_ledger"]
    resource_rows = []
    for stage, label in (
        ("fit_warmup", "Google fit/warmup"),
        ("threshold", "Google threshold clone"),
        ("held", "Google held detector replay"),
    ):
        row = google_exposure[stage]
        scope = "Input accounting"
        if stage == "held":
            replay_runs = "/".join(
                format_number(value, 3)
                for value in google_performance["held_joint_replay_all_three_seconds"]
            )
            scope = (
                f"Joint-pipeline runs {replay_runs} s; median "
                f"{format_number(google_performance['held_joint_replay_median_seconds'], 3)} s; "
                "process-wide peak "
                f"{format_number(google_performance['peak_rss_kib_linux_ru_maxrss'] / 1024, 1)} MiB"
            )
        resource_rows.append(
            (
                label,
                f"{row['paired_shots']:,}",
                f"{row['physical_shots']:,}",
                f"{row['paired_role_updates']:,}",
                f"{row['detector_bits_exposed']:,}",
                scope,
            )
        )
    resource_rows.extend(
        (
            (
                "PNNL fit across 22 state streams",
                f"{pnnl_resources['fit_paired_shot_pairs']:,}",
                f"{pnnl_resources['fit_physical_circuit_shots']:,}",
                f"{pnnl_resources['fit_role_score_updates']:,}",
                f"{pnnl_resources['fit_detector_event_bits_consumed']:,}",
                (
                    "Whole-arm accounting; eigendecompositions "
                    f"{pnnl_resources['fit_eigendecompositions']:,}"
                ),
            ),
            (
                "PNNL surveillance",
                f"{pnnl_resources['surveillance_paired_shot_pairs']:,}",
                f"{pnnl_resources['surveillance_physical_circuit_shots']:,}",
                f"{pnnl_resources['surveillance_role_score_updates']:,}",
                f"{pnnl_resources['surveillance_detector_event_bits_consumed']:,}",
                (
                    "Whole-run "
                    f"{format_number(pnnl_resources['wall_seconds'], 3)} s; "
                    "held processing "
                    f"{format_number(pnnl_resources['held_value_processing_wall_seconds'], 3)} s; "
                    "process-wide peak "
                    f"{format_number(pnnl_resources['peak_rss_kib'] / 1024, 1)} MiB; "
                    "eigendecompositions "
                    f"{pnnl_resources['actual_surveillance_eigendecompositions']:,}"
                ),
            ),
            (
                "PNNL threshold-bootstrap surrogates",
                f"{pnnl_resources['bootstrap_surrogate_shot_updates']:,}",
                "Resampled; no new exposure",
                f"{pnnl_resources['bootstrap_surrogate_role_score_updates']:,}",
                "No new exposure",
                (
                    "4,096 replicates per path/state/method; "
                    "eigendecompositions "
                    f"{pnnl_resources['bootstrap_eigendecompositions']:,}"
                ),
            ),
            (
                "PNNL paired-swap surrogates",
                f"{pnnl_resources['randomization_surrogate_shot_updates']:,}",
                "Resampled; no new exposure",
                f"{pnnl_resources['randomization_surrogate_role_score_updates']:,}",
                "No new exposure",
                (
                    f"{pnnl_resources['randomization_replicates']:,} replicates; "
                    "eigendecompositions "
                    f"{pnnl_resources['randomization_eigendecompositions']:,}"
                ),
            ),
        )
    )
    resource_text = table(
        columns="lrrrrp{0.3\\linewidth}",
        header=(
            "Stage",
            "Paired-shot updates",
            "Physical shots",
            "Role updates",
            "Detector bits",
            "Timing/memory scope",
        ),
        rows=resource_rows,
    )
    resource_text += (
        "\n"
        r"\par\smallskip\noindent "
        r"All three Google replay timings are for the canonical joint detector "
        r"pipeline. PNNL reports both whole-run and held-value processing wall "
        r"time. Peak RSS is process-wide; surrogate rows are resampling work, "
        r"not new physical-shot or detector-bit exposure. "
        r"No per-method timing, memory, or relative-speed claim is reported."
        "\n"
    )
    (output / "resource_ledger_table.tex").write_text(
        resource_text,
        encoding="utf-8",
    )


def _matplotlib() -> tuple[Any, Any, Any]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
    except ImportError as error:
        raise PublicationDataError(
            "matplotlib is required to generate Run 6 publication figures."
        ) from error
    return plt, Line2D, Patch


def save_figure(fig: Any, path: Path) -> None:
    fig.savefig(
        path,
        bbox_inches="tight",
        metadata={
            "Title": path.stem,
            "Author": "Run 6 reproducible analysis",
            "Creator": "publication/run6/analysis/extract_results.py",
            "CreationDate": None,
            "ModDate": None,
        },
    )


def write_figures(
    output: Path,
    *,
    event: Mapping[str, Any],
    risk: Mapping[str, Any],
    randomization: Mapping[str, Any],
    aggregate: Mapping[str, Any],
    pnnl_manifest: Mapping[str, Any],
    pittsburgh_cohorts: Sequence[Mapping[str, Any]],
) -> None:
    plt, Line2D, Patch = _matplotlib()
    method_colors = {
        method: plt.get_cmap("tab10")(index) for index, method in enumerate(METHOD_IDS)
    }
    method_markers = {
        "m0": "s",
        "m0c": "o",
        "m1": "o",
        "m2": "o",
        "m3": "s",
        "m4": "^",
        "m5": "v",
        "space": "D",
    }

    fig, (pre_axis, event_axis) = plt.subplots(
        1,
        2,
        figsize=(11.0, 4.8),
        sharey=True,
        gridspec_kw={"width_ratios": [1.8, 1.2], "wspace": 0.05},
        layout="constrained",
    )
    event_axis.axvspan(
        *WINDOWS["wide"], color="#d9d9d9", alpha=0.5, label="Wide window"
    )
    event_axis.axvspan(
        *WINDOWS["primary"], color="#9ecae1", alpha=0.55, label="Primary window"
    )
    event_axis.axvspan(
        *WINDOWS["narrow"], color="#3182bd", alpha=0.22, label="Narrow window"
    )
    for index, method in enumerate(METHOD_IDS):
        pre_shots = event[method]["pre_event_alert_shots"]
        if pre_shots:
            pre_axis.scatter(
                pre_shots,
                [index] * len(pre_shots),
                color=method_colors[method],
                marker="x",
                s=30,
                linewidths=1.2,
                zorder=3,
            )
        cell = event[method]["windows"]["primary"]
        if cell["detected"]:
            event_axis.scatter(
                cell["first_alert_shot"],
                index,
                color=method_colors[method],
                marker=method_markers[method],
                s=54,
                edgecolor="black",
                linewidth=0.4,
                zorder=3,
            )
    pre_axis.set_yticks(range(len(METHOD_IDS)))
    pre_axis.set_yticklabels([METHOD_METADATA[method][0] for method in METHOD_IDS])
    pre_axis.set_xlim(40_000, WINDOWS["primary"][0])
    pre_axis.set_xlabel("Archive shot before primary window")
    pre_axis.set_title("Every frozen pre-event alert")
    pre_axis.grid(axis="x", alpha=0.2)
    event_axis.set_xlim(WINDOWS["wide"][0] - 8, WINDOWS["wide"][1] + 8)
    event_axis.set_xlabel("Archive shot")
    event_axis.set_title("First alert inside primary window")
    event_axis.grid(axis="x", alpha=0.2)
    handles = [
        Line2D(
            [0],
            [0],
            marker="x",
            color="none",
            markeredgecolor="black",
            label="Pre-event alert",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            color="none",
            markerfacecolor="#7b3294",
            markeredgecolor="black",
            label="First primary-window alert",
        ),
        Patch(facecolor="#d9d9d9", alpha=0.5, label="Wide window"),
        Patch(facecolor="#9ecae1", alpha=0.55, label="Primary window"),
        Patch(facecolor="#3182bd", alpha=0.22, label="Narrow window"),
    ]
    event_axis.legend(handles=handles, fontsize=8, loc="upper right")
    fig.suptitle(
        "Google locked event replay: common shot axis split to preserve window resolution"
    )
    save_figure(fig, output / "google_event_alerts.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0), sharex=True)
    label_titles = {
        "correlated_matching_mismatch": "Primary: correlated-matching mismatch",
        "pymatching_mismatch": "Secondary: PyMatching mismatch",
    }
    metric_titles = {
        "captured_mismatches": "Captured mismatches",
        "retained_mismatch_rate": "Retained mismatch rate",
    }
    offsets = {
        method: (index - (len(METHOD_IDS) - 1) / 2) * 0.012
        for index, method in enumerate(METHOD_IDS)
    }
    for row_index, label in enumerate(LABEL_IDS):
        for column_index, metric in enumerate(
            ("captured_mismatches", "retained_mismatch_rate")
        ):
            axis = axes[row_index, column_index]
            for method in METHOD_IDS:
                x_values = [budget * (10 ** offsets[method]) for budget in RISK_BUDGETS]
                points = [
                    risk["point_estimates"][label][method]["budgets"][str(budget)][
                        metric
                    ]
                    for budget in RISK_BUDGETS
                ]
                intervals = [
                    risk["uncertainty"]["method_intervals"][label][method][str(budget)][
                        metric
                    ]
                    for budget in RISK_BUDGETS
                ]
                axis.plot(
                    x_values,
                    points,
                    color=method_colors[method],
                    marker=method_markers[method],
                    linewidth=1.0,
                    markersize=4.5,
                    label=METHOD_METADATA[method][0],
                )
                for x_value, interval in zip(x_values, intervals, strict=True):
                    if interval["valid_replicates"] == 0:
                        continue
                    axis.vlines(
                        x_value,
                        interval["lower"],
                        interval["upper"],
                        color=method_colors[method],
                        linewidth=0.8,
                        alpha=0.75,
                    )
                    axis.hlines(
                        [interval["lower"], interval["upper"]],
                        x_value / 1.035,
                        x_value * 1.035,
                        color=method_colors[method],
                        linewidth=0.8,
                        alpha=0.75,
                    )
            axis.set_xscale("log")
            axis.set_xticks(RISK_BUDGETS)
            axis.set_xticklabels(
                [
                    f"{budget}\n{100 * (1 - budget / 20_000):.2f}% cov."
                    for budget in RISK_BUDGETS
                ]
            )
            axis.set_title(f"{label_titles[label]}\n{metric_titles[metric]}")
            axis.grid(alpha=0.18)
            axis.set_ylabel(metric_titles[metric])
    axes[1, 0].set_xlabel("Alert budget (log scale) and retained coverage")
    axes[1, 1].set_xlabel("Alert budget (log scale) and retained coverage")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        fontsize=8,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle(
        "Frozen risk-coverage summaries at budgets 2, 20, and 200; "
        "vertical bars are locked 95% shot-block intervals"
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    save_figure(fig, output / "google_risk_coverage.pdf")
    plt.close(fig)

    counts = randomization["crossing_counts_at_100"]
    fractions = [
        counts[method] / randomization["replicate_count"]
        for method in FORMAL_METHOD_IDS
    ]
    fig, axis = plt.subplots(figsize=(8.4, 4.4))
    bar_colors = [
        "#7b3294" if method == "space" else method_colors[method]
        for method in FORMAL_METHOD_IDS
    ]
    bars = axis.bar(
        range(len(FORMAL_METHOD_IDS)),
        fractions,
        color=bar_colors,
        edgecolor="white",
    )
    axis.bar_label(
        bars,
        labels=[f"{counts[method]}/256" for method in FORMAL_METHOD_IDS],
        padding=2,
        fontsize=8,
    )
    space_index = FORMAL_METHOD_IDS.index("space")
    interval = randomization["space_crossing_clopper_pearson_95"]
    space_fraction = fractions[space_index]
    axis.errorbar(
        [space_index],
        [space_fraction],
        yerr=[
            [space_fraction - interval["lower"]],
            [interval["upper"] - space_fraction],
        ],
        fmt="none",
        ecolor="black",
        capsize=4,
        linewidth=1,
        label="S-PACE exact 95% Clopper--Pearson interval",
    )
    axis.axhline(
        0.01,
        color="black",
        linestyle="--",
        linewidth=0.8,
        label="Nominal alpha = 0.01",
    )
    axis.set_xticks(range(len(FORMAL_METHOD_IDS)))
    axis.set_xticklabels(
        [METHOD_METADATA[method][0] for method in FORMAL_METHOD_IDS],
        rotation=25,
        ha="right",
    )
    axis.set_ylabel("Fraction crossing proper-prior e-process threshold 100")
    axis.set_title(
        "Exact-design complete-shot randomization audit (256 replicates; no SR statistic)"
    )
    axis.text(
        0.99,
        0.96,
        (
            "Familywide threshold-600 crossings: "
            f"{randomization['familywide_any_crossing_count_at_600']}/256"
        ),
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=8,
    )
    axis.grid(axis="y", alpha=0.2)
    axis.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    save_figure(fig, output / "google_randomization_proper_prior.pdf")
    plt.close(fig)

    cohort_order = [
        row["cohort_id"] for row in aggregate["cohort_rows"] if row["method"] == "dfr"
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 5.2), sharey=True)
    for axis, comparator in zip(axes, ("dfr", "online_logistic"), strict=True):
        comparison = aggregate["comparisons"][comparator]
        effects = comparison["cohort_delay_differences"]
        y_values = list(range(len(cohort_order)))
        axis.axvline(0, color="black", linewidth=0.8, linestyle="--")
        axis.scatter(effects, y_values, color="#7b3294", s=30, zorder=3)
        interval = comparison["primary_95_percentile_interval"]
        macro = comparison["macro_delay_difference"]
        axis.hlines(-1, interval[0], interval[1], color="#d95f02", linewidth=2)
        axis.plot(
            [interval[0], interval[1]],
            [-1, -1],
            linestyle="none",
            marker="|",
            markersize=8,
            color="#d95f02",
        )
        axis.scatter([macro], [-1], marker="D", color="#d95f02", zorder=3)
        axis.set_title("S-PACE minus " + PNNL_METHOD_METADATA[comparator][0])
        axis.set_xlabel("Restricted delay-fraction difference")
        axis.grid(axis="x", alpha=0.2)
        axis.set_yticks([-1, *y_values])
        axis.set_yticklabels(
            ["Macro", *[f"C{index + 1:02d}" for index in range(len(cohort_order))]]
        )
    axes[0].set_ylabel("Constructed Pittsburgh path cohort")
    control_summary = pnnl_control_summary(pittsburgh_cohorts)
    fig.suptitle(
        f"PNNL constructed boundary: {control_summary['figure']} "
        "(negative favors S-PACE)"
    )
    if (
        pnnl_manifest["claim_label"]
        != "constructed circuit-and-hardware domain shift; not temporal drift"
    ):
        raise PublicationDataError("PNNL figure claim label changed.")
    fig.tight_layout()
    save_figure(fig, output / "pnnl_delay_forest.pdf")
    plt.close(fig)


def conclusion_sentence(overall: bool) -> str:
    if overall:
        return (
            "The locked Google and PNNL empirical gates were both satisfied for "
            "the specified datasets, implementations, endpoints, and budgets "
            "under the disclosed post-detector, pre-outcome repair; this does "
            "not establish superiority over same-feature or same-parity "
            "logistic or threshold classes, a correct likelihood or oracle "
            "rule, or a general algorithmic advantage."
        )
    return "No demonstrated S-PACE algorithmic advantage."


def write_manuscript_contract(
    output: Path,
    *,
    pittsburgh_cohorts: Sequence[Mapping[str, Any]],
) -> tuple[Path, Path]:
    """Write the fail-closed manuscript include and caption contract."""

    control_summary = pnnl_control_summary(pittsburgh_cohorts)
    contract = {
        "schema_version": "run6-manuscript-artifact-contract-v2",
        "evidence_classes": {
            "provenance": (
                "original pre-access detector freeze plus a separately ratified "
                "post-detector, pre-outcome consumer-validation repair"
            ),
            "google_event": "empirical replay of one author-identified approximate event",
            "google_risk": "retrospective frozen-ranking triage; not a decoder",
            "google_randomization": "exact-design complete-shot implementation diagnostic",
            "google_threshold_bootstrap": (
                "descriptive complete-shot threshold uncertainty; it neither "
                "replaces a frozen threshold nor enters a gate"
            ),
            "pnnl": (
                "constructed circuit-and-hardware snapshot-cohort boundary; "
                "not temporal drift"
            ),
            "pnnl_randomization": (
                "exact paired-swap implementation audit; not a natural-hardware "
                "null; it enters neither empirical gate nor the overall conjunction"
            ),
        },
        "figures": {
            "google_event": {
                "path": "google_event_alerts.pdf",
                "latex_macro": "RunSixVerifiedGoogleEventFigure",
                "caption_contract": (
                    "Left panel contains every frozen pre-event alert for every "
                    "method. Right panel contains the first alert inside the "
                    "primary window and all three predeclared window bands. "
                    "The event location is approximate, not an exact onset."
                ),
            },
            "google_risk": {
                "path": "google_risk_coverage.pdf",
                "latex_macro": "RunSixVerifiedGoogleRiskFigure",
                "caption_contract": (
                    "Both decoder-mismatch labels and budgets 2, 20, and 200 "
                    "are shown with locked 95% complete-shot block-bootstrap "
                    "intervals. Only correlated-matching capture at budget 20 "
                    "enters the primary gate."
                ),
            },
            "google_randomization": {
                "path": "google_randomization_proper_prior.pdf",
                "latex_macro": "RunSixVerifiedGoogleRandomizationFigure",
                "caption_contract": (
                    "Only proper-prior e-process crossings from 256 complete-shot "
                    "randomization replicates are shown. No SR statistic exists "
                    "in this result artifact and none may be reported."
                ),
            },
            "pnnl": {
                "path": "pnnl_delay_forest.pdf",
                "latex_macro": "RunSixVerifiedPNNLFigure",
                "caption_contract": (
                    "Eleven path-level effects average logical states before "
                    "macro aggregation. C01--C11 map through the cohort-control "
                    f"table. {control_summary['caption']}"
                ),
            },
        },
        "tables": {
            "gate": "gate_decision_table.tex",
            "google_event": "google_event_table.tex",
            "google_threshold_frontier": "google_threshold_frontier_table.tex",
            "google_risk_budget": "google_risk_budget_table.tex",
            "google_uncertainty": "google_uncertainty_table.tex",
            "google_threshold_bootstrap": {
                "path": "google_threshold_bootstrap_table.tex",
                "latex_macro": "RunSixVerifiedGoogleThresholdBootstrapTable",
                "caption_contract": (
                    "All 2,000 complete-shot circular block-bootstrap rows use "
                    "block length 128 and seeds 613000--614999. The table is "
                    "descriptive only, does not replace frozen thresholds, and "
                    "does not alter a locked Boolean."
                ),
            },
            "google_randomization": "randomization_table.tex",
            "pnnl_macro": "pnnl_macro_table.tex",
            "pnnl_state_results": {
                "path": "pnnl_state_results_table.tex",
                "latex_macro": "RunSixVerifiedPNNLStateResultsTable",
                "caption_contract": (
                    "All 22 ordered path-state rows report pre-boundary "
                    "false-alarm, miss, and restricted-delay values for all "
                    "five methods; no path or state is selected."
                ),
            },
            "pnnl_cohort_results": {
                "path": "pnnl_cohort_results_table.tex",
                "latex_macro": "RunSixVerifiedPNNLCohortResultsTable",
                "caption_contract": (
                    "All 11 ordered state-averaged cohorts report "
                    "pre-false-alarm mean, miss mean, and restricted delay for "
                    "all five methods; the forest separately shows the two "
                    "gate-relevant comparator differences."
                ),
            },
            "pnnl_comparison": "pnnl_comparison_table.tex",
            "pnnl_cohort_control": "pnnl_cohort_control_table.tex",
            "pnnl_randomization": {
                "path": "pnnl_randomization_audit_table.tex",
                "latex_macro": "RunSixVerifiedPNNLRandomizationTable",
                "caption_contract": (
                    "The exact 256-replicate paired-swap implementation audit "
                    "reports all five methods over all 22 path-state episodes. "
                    "It is not a natural-hardware null and enters neither "
                    "empirical gate nor the overall conjunction."
                ),
            },
            "resource_ledger": "resource_ledger_table.tex",
        },
        "resource_scope": (
            "Google runtime is the canonical joint detector pipeline and PNNL "
            "runtime is the whole auxiliary arm; peak RSS is process-wide. "
            "Per-method timing, memory, and relative-speed claims are unsupported."
        ),
        "validated_descriptive_not_gate_fields": {
            "partial_trapezoidal_recall_area": (
                "Recomputed from the three locked risk-budget recall points "
                "with NumPy linear trapezoidal integration; descriptive only "
                "and absent from every locked Boolean."
            )
        },
        "repair_scope": {
            "detector_values_accessed_before_repair": True,
            "detector_numeric_diagnostics_exposed": True,
            "detector_numeric_values_used_to_select_repair": False,
            "outcomes_or_pnnl_accessed_before_repair": False,
            "completed_randomization_before_repair": False,
            "detector_rerun": False,
            "claim": (
                "This is a disclosed validator amendment, not detector-blind "
                "repair or fully preregistered end-to-end execution."
            ),
        },
        "unsupported_manuscript_fields": {
            "RANDOMIZATION_SR_SUMMARY": (
                "The Google randomization result has no SR field; remove it."
            ),
            "S_PACE_WALL_TIME_SECONDS": (
                "Only joint-pipeline timing is measured; remove per-method timing."
            ),
            "S_PACE_PEAK_MEMORY_MIB": (
                "Peak RSS is process-wide; remove per-method memory."
            ),
            "PNNL_PATHS_RETAINED": (
                "Retention is one Boolean aggregate rule, not a path count."
            ),
            "GOOGLE_S_PACE_EVENT_SCORE_PERCENTILE": (
                "No event shot/role/denominator was frozen; omit this field."
            ),
            "GOOGLE_CONTEXTUAL_BEST_METHOD": (
                "No endpoint or tie rule was frozen; post-hoc best-method "
                "selection is forbidden."
            ),
            "POSITIVE_ALGORITHMIC_SUPERIORITY": (
                "A true conjunction records only satisfaction of both fixed "
                "empirical gates. It does not establish superiority over "
                "same-feature or same-parity logistic/threshold classes, a "
                "correct likelihood or oracle rule, or general algorithmic "
                "advantage."
            ),
        },
    }
    json_path = output / "manuscript_artifact_contract.json"
    json_path.write_bytes(canonical_json_bytes(contract) + b"\n")
    tex_lines = [
        "% Generated by publication/run6/analysis/extract_results.py.",
        "% Override this directory before input if the bundle is not at generated/.",
        r"\providecommand{\RunSixGeneratedDir}{generated}",
        (
            r"\newcommand{\RunSixVerifiedClaim}{"
            r"\input{\RunSixGeneratedDir/claim_sentence.tex}}"
        ),
        (
            r"\newcommand{\RunSixVerifiedGateTable}{"
            r"\input{\RunSixGeneratedDir/gate_decision_table.tex}}"
        ),
        (
            r"\newcommand{\RunSixVerifiedGoogleEventTable}{"
            r"\input{\RunSixGeneratedDir/google_event_table.tex}}"
        ),
        (
            r"\newcommand{\RunSixVerifiedGoogleThresholdFrontierTable}{"
            r"\input{\RunSixGeneratedDir/google_threshold_frontier_table.tex}}"
        ),
        (
            r"\newcommand{\RunSixVerifiedGoogleRiskBudgetTable}{"
            r"\input{\RunSixGeneratedDir/google_risk_budget_table.tex}}"
        ),
        (
            r"\newcommand{\RunSixVerifiedGoogleUncertaintyTable}{"
            r"\input{\RunSixGeneratedDir/google_uncertainty_table.tex}}"
        ),
        (
            r"\newcommand{\RunSixVerifiedGoogleThresholdBootstrapTable}{"
            r"\input{\RunSixGeneratedDir/google_threshold_bootstrap_table.tex}}"
        ),
        (
            r"\newcommand{\RunSixVerifiedGoogleRandomizationTable}{"
            r"\input{\RunSixGeneratedDir/randomization_table.tex}}"
        ),
        (
            r"\newcommand{\RunSixVerifiedPNNLMacroTable}{"
            r"\input{\RunSixGeneratedDir/pnnl_macro_table.tex}}"
        ),
        (
            r"\newcommand{\RunSixVerifiedPNNLStateResultsTable}{"
            r"\input{\RunSixGeneratedDir/pnnl_state_results_table.tex}}"
        ),
        (
            r"\newcommand{\RunSixVerifiedPNNLCohortResultsTable}{"
            r"\input{\RunSixGeneratedDir/pnnl_cohort_results_table.tex}}"
        ),
        (
            r"\newcommand{\RunSixVerifiedPNNLComparisonTable}{"
            r"\input{\RunSixGeneratedDir/pnnl_comparison_table.tex}}"
        ),
        (
            r"\newcommand{\RunSixVerifiedPNNLCohortControlTable}{"
            r"\input{\RunSixGeneratedDir/pnnl_cohort_control_table.tex}}"
        ),
        (
            r"\newcommand{\RunSixVerifiedPNNLRandomizationTable}{"
            r"\input{\RunSixGeneratedDir/pnnl_randomization_audit_table.tex}}"
        ),
        (
            r"\newcommand{\RunSixVerifiedResourceLedgerTable}{"
            r"\input{\RunSixGeneratedDir/resource_ledger_table.tex}}"
        ),
        (
            r"\newcommand{\RunSixVerifiedGoogleEventFigure}{"
            r"\noindent\includegraphics[width=\linewidth]{"
            r"\RunSixGeneratedDir/google_event_alerts.pdf}}"
        ),
        (
            r"\newcommand{\RunSixVerifiedGoogleRiskFigure}{"
            r"\noindent\includegraphics[width=\linewidth]{"
            r"\RunSixGeneratedDir/google_risk_coverage.pdf}}"
        ),
        (
            r"\newcommand{\RunSixVerifiedGoogleRandomizationFigure}{"
            r"\noindent\includegraphics[width=\linewidth]{"
            r"\RunSixGeneratedDir/google_randomization_proper_prior.pdf}}"
        ),
        (
            r"\newcommand{\RunSixVerifiedPNNLFigure}{"
            r"\noindent\includegraphics[width=\linewidth]{"
            r"\RunSixGeneratedDir/pnnl_delay_forest.pdf}}"
        ),
        "",
    ]
    tex_path = output / "manuscript_artifact_contract.tex"
    tex_path.write_text("\n".join(tex_lines), encoding="utf-8")
    return json_path, tex_path


def write_bundle(
    output: Path,
    *,
    evidence_records: Mapping[str, Mapping[str, str]],
    event: Mapping[str, Any],
    thresholds: Mapping[str, Any],
    risk: Mapping[str, Any],
    decision: Mapping[str, Any],
    randomization: Mapping[str, Any],
    threshold_bootstrap: Mapping[str, Any],
    aggregate: Mapping[str, Any],
    pnnl_state_rows: Sequence[Mapping[str, Any]],
    pnnl_randomization: Mapping[str, Any],
    detector_manifest: Mapping[str, Any],
    pnnl_manifest: Mapping[str, Any],
    pittsburgh_cohorts: Sequence[Mapping[str, Any]],
    original_ratification: Mapping[str, Any],
    repair_manifest: Mapping[str, Any],
    repair_ratification: Mapping[str, Any],
) -> None:
    if output.exists() and any(output.iterdir()):
        raise PublicationDataError("Publication output directory must be empty.")
    output_parent = output.parent.resolve()
    output_parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output_parent))
    try:
        write_tables(
            temporary,
            event=event,
            thresholds=thresholds,
            risk=risk,
            decision=decision,
            randomization=randomization,
            threshold_bootstrap=threshold_bootstrap,
            aggregate=aggregate,
            pnnl_state_rows=pnnl_state_rows,
            pnnl_randomization=pnnl_randomization,
            detector_manifest=detector_manifest,
            pnnl_manifest=pnnl_manifest,
            pittsburgh_cohorts=pittsburgh_cohorts,
        )
        write_figures(
            temporary,
            event=event,
            risk=risk,
            randomization=randomization,
            aggregate=aggregate,
            pnnl_manifest=pnnl_manifest,
            pittsburgh_cohorts=pittsburgh_cohorts,
        )
        conclusion = conclusion_sentence(decision["overall_run6_advantage"])
        (temporary / "claim_sentence.tex").write_text(
            tex_escape(conclusion) + "\n",
            encoding="utf-8",
        )
        contract_path, _ = write_manuscript_contract(
            temporary,
            pittsburgh_cohorts=pittsburgh_cohorts,
        )
        artifacts = []
        for path in sorted(temporary.iterdir()):
            if path.name == "publication_bundle_manifest.json":
                continue
            artifacts.append(
                {
                    "path": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
        bundle_manifest = {
            "schema_version": "run6-publication-bundle-v5",
            "evidence_inputs": {
                role: dict(record) for role, record in sorted(evidence_records.items())
            },
            "dual_provenance": {
                "original_detector_chain": {
                    "ratification_status": original_ratification["status"],
                    "ratification_path": evidence_records["freeze_ratification"][
                        "path"
                    ],
                    "ratification_sha256": repair_manifest["original_freeze"][
                        "ratification_sha256"
                    ],
                    "detector_manifest_path": evidence_records["detector_manifest"][
                        "path"
                    ],
                    "detector_manifest_sha256": repair_ratification[
                        "detector_manifest_sha256"
                    ],
                },
                "post_detector_repair_chain": {
                    "status": repair_ratification["status"],
                    "repair_manifest_commit": repair_ratification[
                        "repair_manifest_commit"
                    ],
                    "repair_manifest_path": evidence_records["repair_manifest"]["path"],
                    "repair_manifest_sha256": repair_ratification["hashes"][
                        REPAIR_MANIFEST_PATH
                    ],
                    "repair_ratification_path": evidence_records["repair_ratification"][
                        "path"
                    ],
                    "repair_ratification_sha256": evidence_records[
                        "repair_ratification"
                    ]["sha256"],
                    "access_record": dict(repair_ratification["access_record"]),
                },
                "interpretation": (
                    "The detector was produced under the original pre-access "
                    "freeze. Consumer validation and all downstream arms use a "
                    "separate post-detector, pre-outcome repair ratification."
                ),
            },
            "decision": {
                "google_primary_pass": decision["google_primary_pass"],
                "pnnl_retention_pass": decision["pnnl_retention_pass"],
                "overall_run6_advantage": decision["overall_run6_advantage"],
                "rendered_label": "locked conjunctive empirical gate",
                "negative_result_reasons": decision["negative_result_reasons"],
                "claim_sentence": conclusion,
            },
            "claim_scope": {
                "google": "real-hardware replay of one author-identified approximate event",
                "pnnl": "constructed circuit-and-hardware snapshot-cohort boundary",
                "randomization": "exact-design implementation diagnostic",
                "threshold_bootstrap": (
                    "descriptive complete-shot threshold uncertainty only; "
                    "does not replace thresholds or change a gate"
                ),
                "pnnl_randomization": (
                    "exact paired-swap implementation audit; not a natural-hardware "
                    "null; it enters neither empirical gate nor the overall conjunction"
                ),
                "forbidden": [
                    "quantum advantage",
                    "universal sample-efficiency advantage",
                    "scalable computational advantage",
                    (
                        "superiority to same-feature or same-parity logistic or "
                        "threshold classes"
                    ),
                    "superiority to a correct likelihood or oracle rule",
                    "general algorithmic advantage",
                    "detector-blind repair or fully preregistered end-to-end execution",
                ],
            },
            "manuscript_artifact_contract": {
                "path": contract_path.name,
                "sha256": sha256_file(contract_path),
            },
            "artifacts": artifacts,
        }
        (temporary / "publication_bundle_manifest.json").write_bytes(
            canonical_json_bytes(bundle_manifest) + b"\n"
        )
        if output.exists():
            output.rmdir()
        temporary.replace(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate completed Run 6 manifests and generate publication tables/figures."
        )
    )
    parser.add_argument("--detector-manifest", type=Path, required=True)
    parser.add_argument("--freeze-ratification", type=Path, required=True)
    parser.add_argument("--repair-manifest", type=Path, required=True)
    parser.add_argument("--repair-ratification", type=Path, required=True)
    parser.add_argument("--outcome-manifest", type=Path, required=True)
    parser.add_argument("--randomization-manifest", type=Path, required=True)
    parser.add_argument("--pnnl-manifest", type=Path, required=True)
    parser.add_argument("--pittsburgh-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def _run_with_validation_profile(
    argv: Sequence[str] | None = None,
    *,
    validation_profile: ValidationProfile,
) -> int:
    args = parse_args(argv)
    profile = validation_profile
    source_paths = {
        "detector_manifest": args.detector_manifest,
        "freeze_ratification": args.freeze_ratification,
        "repair_manifest": args.repair_manifest,
        "repair_ratification": args.repair_ratification,
        "outcome_manifest": args.outcome_manifest,
        "randomization_manifest": args.randomization_manifest,
        "pnnl_manifest": args.pnnl_manifest,
        "pittsburgh_manifest": args.pittsburgh_manifest,
    }
    evidence_records = evidence_input_records(source_paths, profile=profile)
    paths = {role: path.resolve(strict=True) for role, path in source_paths.items()}
    if len(set(paths.values())) != 8:
        raise PublicationDataError("The eight evidence inputs must be distinct files.")
    detector, _, event, thresholds = validate_detector_manifest(
        paths["detector_manifest"]
    )
    original_ratification = validate_original_ratification(
        paths["freeze_ratification"],
        detector_manifest=detector,
        profile=profile,
    )
    repair_manifest = validate_repair_manifest(
        paths["repair_manifest"],
        original_ratification_path=paths["freeze_ratification"],
        original_ratification=original_ratification,
        detector_path=paths["detector_manifest"],
        detector_manifest=detector,
        profile=profile,
    )
    repair_ratification = validate_repair_ratification(
        paths["repair_ratification"],
        repair_manifest_path=paths["repair_manifest"],
        repair_manifest=repair_manifest,
        original_ratification_path=paths["freeze_ratification"],
        detector_path=paths["detector_manifest"],
        profile=profile,
    )
    _, randomization, threshold_bootstrap = validate_randomization_manifest(
        paths["randomization_manifest"],
        detector_path=paths["detector_manifest"],
        detector_manifest=detector,
        detector_thresholds=thresholds,
        repair_ratification_path=paths["repair_ratification"],
    )
    pittsburgh = validate_pittsburgh_manifest(paths["pittsburgh_manifest"])
    pnnl, pnnl_state_rows, aggregate, pnnl_randomization = validate_pnnl_manifest(
        paths["pnnl_manifest"],
        pittsburgh_path=paths["pittsburgh_manifest"],
        pittsburgh_evidence=pittsburgh,
        original_ratification_path=paths["freeze_ratification"],
        repair_ratification_path=paths["repair_ratification"],
    )
    _, risk, decision = validate_outcome_manifest(
        paths["outcome_manifest"],
        detector_path=paths["detector_manifest"],
        detector_manifest=detector,
        event=event,
        randomization_path=paths["randomization_manifest"],
        randomization=randomization,
        threshold_bootstrap=threshold_bootstrap,
        pnnl_path=paths["pnnl_manifest"],
        pnnl_manifest=pnnl,
        repair_ratification_path=paths["repair_ratification"],
    )
    write_bundle(
        args.output_dir.resolve(),
        evidence_records=evidence_records,
        event=event,
        thresholds=thresholds,
        risk=risk,
        decision=decision,
        randomization=randomization,
        threshold_bootstrap=threshold_bootstrap,
        aggregate=aggregate,
        pnnl_state_rows=pnnl_state_rows,
        pnnl_randomization=pnnl_randomization,
        detector_manifest=detector,
        pnnl_manifest=pnnl,
        pittsburgh_cohorts=pittsburgh["cohorts"],
        original_ratification=original_ratification,
        repair_manifest=repair_manifest,
        repair_ratification=repair_ratification,
    )
    print(conclusion_sentence(decision["overall_run6_advantage"]))
    print(f"Publication bundle: {args.output_dir.resolve()}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Production CLI with immutable Git anchors and no profile override."""

    return _run_with_validation_profile(
        argv,
        validation_profile=load_production_validation_profile(),
    )


if __name__ == "__main__":
    raise SystemExit(main())
