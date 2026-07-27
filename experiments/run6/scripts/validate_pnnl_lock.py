#!/usr/bin/env python3
"""Strict metadata-only validator for the ratified Run 6 Pittsburgh lock.

The validator is deliberately incapable of inspecting held measurement
payloads.  It reads the lock, ``info.json``, ``calibration.json``, and the two
QASM files.  For every ``bitstrings.json`` it performs only ``lstat``/``stat``
and compares ``st_size`` with the lock.  The guarded read helper rejects a
held-payload path even if a future caller accidentally passes one to it.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import stat
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Any

HELD_PAYLOAD_NAME = "bitstrings.json"
INFO_NAME = "info.json"
CALIBRATION_NAME = "calibration.json"
QASM_NAMES = ("circuit_state0.qasm", "circuit_state1.qasm")

_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
_RELATIVE_JOB_RE_TEXT = (
    r"^d(?P<distance>[0-9]+)_r(?P<rounds>[0-9]+)/job_(?P<job>[0-9]+)$"
)
_DECLARATION_RE_TEXT = (
    r"^\s*bit\[([0-9]+)\]\s+"
    r"(c_(data|syndrome)_([A-Za-z0-9_]+))\s*;\s*(?://.*)?$"
)
_MEASUREMENT_RE_TEXT = (
    r"^\s*(c_(data|syndrome)_([A-Za-z0-9_]+))\[([0-9]+)\]"
    r"\s*=\s*measure\s+\$([0-9]+)\s*;\s*(?://.*)?$"
)
_RELATIVE_JOB_RE = re.compile(_RELATIVE_JOB_RE_TEXT)
_DECLARATION_RE = re.compile(_DECLARATION_RE_TEXT)
_MEASUREMENT_RE = re.compile(_MEASUREMENT_RE_TEXT)

_TOP_LEVEL_KEYS = {
    "manifest_id",
    "status",
    "created_from",
    "parent_artifacts",
    "source",
    "held_payload_policy",
    "hash_profiles",
    "directory_and_qasm_parser",
    "bitstring_parser_after_unblinding",
    "cohort_selection",
    "stream_construction",
    "features_and_methods",
    "alarm_calibration",
    "metrics_and_retention",
    "uncertainty",
    "randomization_audit",
    "artifact_tuple_schemas",
    "snapshots",
    "cohort_row_schema",
    "cohort_order",
    "cohort_pairs",
    "aggregate_resource_counts_per_pre_or_post_phase",
    "claim_boundary",
}

_STATIC_KEYSETS: dict[str, set[str]] = {
    "created_from": {
        "allowed_inputs",
        "held_bitstring_content_read",
        "held_bitstring_hash_computed",
        "selection_used_syndrome_or_data_values",
    },
    "parent_artifacts": {
        "experiments/run6/configs/google2022_locked.json",
        "references/run6_real_qec_preregistered_plan.md",
        "references/run6_pnnl_snapshot_audit.md",
        "references/run6_real_qec_data_and_comparator_audit.md",
        "references/run6_preregistration_adversarial_audit.md",
    },
    "source": {
        "record_url",
        "doi",
        "release_version",
        "license",
        "backend",
        "artifact_root",
        "logical_states",
    },
    "held_payload_policy": {
        "before_ratification",
        "metadata_only_exception",
        "after_ratification",
        "first_unblinding_record",
        "data_register_use",
    },
    "hash_profiles": {
        "raw_sha256_v1",
        "python_json_canonical_v1",
        "qasm_text_normalized_v1",
        "state_pair_sha256_v1",
        "snapshot_identity",
        "selection_requires",
        "circuit_control_label_requires",
    },
    "directory_and_qasm_parser": {
        "relative_job_regex",
        "cross_checks",
        "declaration_regex",
        "measurement_regex",
        "required_data_length",
        "required_syndrome_length",
        "index_coverage",
        "syndrome_flat_index",
        "stable_ancilla_rule",
        "oriented_path",
        "state_agreement",
        "duplicate_path_rule",
        "reversal_rule",
        "register_suffix_rule",
        "manifest_check",
    },
    "bitstring_parser_after_unblinding": {
        "top_level",
        "state_selection",
        "required_states",
        "metadata_checks",
        "register_container",
        "data_key",
        "syndrome_key",
        "required_rows",
        "data_shape",
        "syndrome_shape",
        "value_domain",
        "syndrome_tensor",
        "detection_events",
    },
    "cohort_selection": {
        "group_key",
        "eligibility",
        "date_order",
        "selected_dates",
        "duplicate_date_tie_break",
        "no_value_dependent_filtering",
        "audit_counts",
    },
    "stream_construction": {
        "states_are_separate_replicates",
        "row_pairing",
        "cross_state_pairing",
        "path_pooling_before_scoring",
        "m",
        "early_reference_pre",
        "early_monitor_pre",
        "early_reference_post",
        "late_post_monitor",
        "pre_pair",
        "post_pair",
        "boundary",
        "update_order",
        "paired_cycle_contrast",
        "fit_prefix_shots",
        "pre_surveillance_shots",
        "post_surveillance_shots",
        "state_at_boundary",
        "resource_accounting",
    },
    "features_and_methods": {
        "inherited_from",
        "method_order",
        "q",
        "global_detector_rate",
        "sparse_feature",
        "sparse_dimension",
        "sparse_top_k_adaptation",
        "spectral_state",
        "spectral_rank_positive",
        "all_hyperparameter_components",
        "update_after_score",
        "online_state_fit",
        "space_composite",
        "target_method",
        "sibling_ablations_not_eligible_for_target_selection",
    },
    "alarm_calibration": {
        "primary",
        "alarm_trace",
        "surveillance_horizon_shots",
        "surveillance_horizon_updates",
        "bootstrap_input",
        "bootstrap_kind",
        "block_length_shots",
        "replicates",
        "alpha_per_path_state_episode",
        "rng",
        "seed_formula",
        "cohort_index",
        "method_index",
        "draw",
        "replicate_initialization",
        "replicate_statistic",
        "threshold_order_statistic",
        "actual_replay",
        "alarm_rule",
        "nonfinite_policy",
        "secondary_fixed_e_threshold",
        "forbidden_empirical_claim",
    },
    "metrics_and_retention": {
        "alarm_time_unit",
        "pre_false_alarm",
        "restricted_post_delay_fraction",
        "within_cohort_pooling",
        "across_cohort_pooling",
        "required_strata",
        "target_method",
        "retention_comparators",
        "pairwise_delay_difference",
        "retention_for_each_comparator",
        "retention_pass",
        "all_method_outputs_required",
        "post_selection",
        "interpretation",
    },
    "uncertainty": {
        "independent_summary_unit",
        "primary_interval",
        "primary_replicates",
        "primary_rng",
        "primary_resampling",
        "calibration_pair_sensitivity",
        "percentile_rule",
        "exact_sign_flip_sensitivity",
        "required_reporting",
    },
    "randomization_audit": {
        "dataset",
        "replicates",
        "replicate_seeds",
        "rng",
        "iteration_order",
        "swap_draw",
        "swap_scope",
        "method_reset",
        "shared_masks",
        "score_order",
        "audit_alarm",
        "reported_statistics",
        "claim_scope",
    },
    "artifact_tuple_schemas": {
        "metadata",
        "info",
        "calibration",
        "qasm_state",
        "qasm_pair",
        "held_bitstrings",
    },
    "aggregate_resource_counts_per_pre_or_post_phase": {
        "path_groups",
        "path_state_streams",
        "paired_shots",
        "paired_cycle_updates",
    },
    "claim_boundary": {"all_selected_pairs", "forbidden"},
}

_NESTED_KEYSETS: dict[tuple[str, ...], set[str]] = {
    ("hash_profiles", "raw_sha256_v1"): {"input", "algorithm"},
    ("hash_profiles", "python_json_canonical_v1"): {
        "decode",
        "parse",
        "serialize",
        "trailing_newline",
        "algorithm",
        "scope",
        "not_claimed",
    },
    ("hash_profiles", "qasm_text_normalized_v1"): {
        "decode",
        "newline_normalization",
        "line_normalization",
        "file_end",
        "algorithm",
        "scope",
    },
    ("hash_profiles", "state_pair_sha256_v1"): {
        "input",
        "algorithm",
        "state_order",
    },
    ("bitstring_parser_after_unblinding", "detection_events"): {
        "first_round",
        "later_rounds",
        "shape",
        "terminal_detector",
        "flattening_for_storage_only",
    },
    ("cohort_selection", "audit_counts"): {
        "state0_state1_path_agree_chain_instances",
        "unique_configuration_path_groups",
        "eligible_selected_groups",
        "basis_counts",
        "raw_qasm_identical_selected_pairs",
        "normalized_qasm_identical_selected_pairs",
    },
    ("cohort_selection", "audit_counts", "basis_counts"): {"X", "Z"},
    ("alarm_calibration", "secondary_fixed_e_threshold"): {
        "alpha",
        "e_threshold",
        "claim_scope",
    },
    ("uncertainty", "calibration_pair_sensitivity"): {
        "cluster",
        "replicates",
        "rng",
        "resampling",
        "interpretation",
    },
}

_SNAPSHOT_KEYS = {
    "relative_job_dir",
    "metadata",
    "info",
    "calibration",
    "qasm_state0",
    "qasm_state1",
    "qasm_pair",
    "held_bitstrings",
}

_EXPECTED_ARTIFACT_SCHEMAS = {
    "metadata": [
        "distance",
        "rounds",
        "basis",
        "shots_per_state",
        "n_chains",
        "backend_property_date_source",
        "backend_property_date_utc",
    ],
    "info": ["bytes", "raw_sha256", "python_canonical_sha256"],
    "calibration": ["bytes", "raw_sha256", "python_canonical_sha256"],
    "qasm_state": ["bytes", "raw_sha256", "normalized_sha256"],
    "qasm_pair": ["raw_pair_sha256", "normalized_pair_sha256"],
    "held_bitstrings": [
        "bytes_from_stat_only",
        "content_read_for_manifest",
        "sha256_before_unblinding",
    ],
}

_EXPECTED_COHORT_SCHEMA = [
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
]

_EXPECTED_METHOD_ORDER = [
    "dfr",
    "online_logistic",
    "space_sparse",
    "space_spectral",
    "space_composite",
]

_PENDING_STATIC_CONTRACT_SHA256 = (
    "20f5d2931fbda933e725f32af2d56a61ad28cac88fceda1154f8631b7878ab8a"
)
_FROZEN_STATIC_CONTRACT_SHA256 = (
    "f35f116e7465d90d0a51562ccc068a4e67f39960817c9f1bbc4161ea153fbc74"
)
_LOCK_STATUS_BY_MODE = {
    "pending": "ratified_metadata_lock_pending_final_freeze_commit",
    "frozen": "frozen_before_held_value_access",
}
_ARTIFACT_DEPENDENT_TOP_LEVEL_FIELDS = {
    "snapshots",
    "cohort_order",
    "cohort_pairs",
}

_EXPECTED_INFO_KEYS = {
    "backend",
    "d",
    "rounds",
    "basis",
    "logical_states",
    "shots",
    "n_chains",
}

_EXPECTED_CALIBRATION_KEYS = {
    "backend_name",
    "backend_version",
    "last_update_date",
    "qubits",
    "gates",
    "general",
    "general_qlists",
    "coupling_map",
}


class LockValidationError(ValueError):
    """Raised when a metadata lock or one of its safe artifacts is invalid."""


@dataclass(frozen=True)
class ValidationReport:
    """Summary containing no held values and no held hashes."""

    lock_path: Path
    artifact_root: Path
    snapshots: int
    cohorts: int
    held_payloads_statted: int
    paired_shots_per_phase: int
    paired_cycle_updates_per_phase: int
    threshold_seed_count: int


@dataclass(frozen=True)
class QasmMap:
    data_qubits: tuple[int, ...]
    syndrome_qubits: tuple[int, ...]
    oriented_path: tuple[int, ...]


def _fail(path: str, message: str) -> None:
    raise LockValidationError(f"{path}: {message}")


def _as_mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(path, "expected an object")
    if not all(isinstance(key, str) for key in value):
        _fail(path, "all object keys must be strings")
    return value


def _as_list(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        _fail(path, "expected an array")
    return value


def _as_exact_int(value: Any, path: str, *, minimum: int | None = None) -> int:
    if type(value) is not int:
        _fail(path, "expected an integer (booleans are not accepted)")
    if minimum is not None and value < minimum:
        _fail(path, f"expected an integer >= {minimum}")
    return value


def _as_bool(value: Any, path: str) -> bool:
    if type(value) is not bool:
        _fail(path, "expected a boolean")
    return value


def _as_string(value: Any, path: str) -> str:
    if not isinstance(value, str):
        _fail(path, "expected a string")
    return value


def _exact_keys(value: Any, expected: set[str], path: str) -> Mapping[str, Any]:
    mapping = _as_mapping(value, path)
    actual = set(mapping)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing or unknown:
        _fail(path, f"missing fields={missing}, unknown fields={unknown}")
    return mapping


def _expect_equal(actual: Any, expected: Any, path: str) -> None:
    if actual != expected:
        _fail(path, f"expected {expected!r}, observed {actual!r}")


def _require_sha256(value: Any, path: str) -> str:
    text = _as_string(value, path)
    if _HEX64_RE.fullmatch(text) is None:
        _fail(path, "expected a lowercase 64-hex SHA-256 digest")
    return text


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise LockValidationError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _decode_json(raw: bytes, path: str) -> Any:
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise LockValidationError(f"{path}: invalid UTF-8") from exc

    def reject_constant(token: str) -> Any:
        raise LockValidationError(f"{path}: non-finite JSON constant {token!r}")

    try:
        return json.loads(
            text,
            object_pairs_hook=_strict_object,
            parse_constant=reject_constant,
        )
    except LockValidationError:
        raise
    except (json.JSONDecodeError, ValueError) as exc:
        raise LockValidationError(f"{path}: invalid JSON: {exc}") from exc


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise LockValidationError(f"cannot canonicalize JSON: {exc}") from exc


def _digest(raw: bytes) -> str:
    return sha256(raw).hexdigest()


def _normalize_qasm(raw: bytes, path: str) -> bytes:
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise LockValidationError(f"{path}: QASM is not valid UTF-8") from exc
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+$", "", line) for line in text.split("\n")]
    while lines and lines[-1] == "":
        lines.pop()
    return ("\n".join(lines) + "\n").encode("utf-8")


def _pair_digest(state0_digest: str, state1_digest: str) -> str:
    return _digest(
        _canonical_json_bytes(
            {
                "state0_sha256": state0_digest,
                "state1_sha256": state1_digest,
            }
        )
    )


def _regular_stat(path: Path, label: str) -> Any:
    try:
        link_status = path.lstat()
    except FileNotFoundError as exc:
        raise LockValidationError(f"{label}: missing file {path}") from exc
    if stat.S_ISLNK(link_status.st_mode):
        _fail(label, f"symbolic links are forbidden: {path}")
    try:
        status = path.stat()
    except OSError as exc:
        raise LockValidationError(f"{label}: cannot stat {path}: {exc}") from exc
    if not stat.S_ISREG(status.st_mode):
        _fail(label, f"expected a regular file: {path}")
    return status


def _read_allowed_bytes(path: Path, label: str) -> bytes:
    """Read a safe artifact and mechanically refuse the held payload name."""

    if path.name == HELD_PAYLOAD_NAME:
        _fail(
            label,
            "refusing to open or hash bitstrings.json; metadata validation "
            "permits stat size only",
        )
    _regular_stat(path, label)
    try:
        return path.read_bytes()
    except OSError as exc:
        raise LockValidationError(f"{label}: cannot read {path}: {exc}") from exc


def _held_payload_size(path: Path, label: str) -> int:
    """Return only ``st_size``; never open, parse, or hash the held payload."""

    if path.name != HELD_PAYLOAD_NAME:
        _fail(label, f"held-payload path must end in {HELD_PAYLOAD_NAME!r}")
    return int(_regular_stat(path, label).st_size)


def _safe_under(root: Path, relative: str, path: str) -> Path:
    raw = _as_string(relative, path)
    if "\\" in raw:
        _fail(path, "backslashes are forbidden in POSIX relative paths")
    pure = PurePosixPath(raw)
    if (
        pure.is_absolute()
        or not pure.parts
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        _fail(path, "expected a traversal-free POSIX relative path")
    cursor = root
    for part in pure.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            _fail(path, f"symbolic path components are forbidden: {cursor}")
    root_resolved = root.resolve()
    candidate = (root / Path(*pure.parts)).resolve(strict=False)
    if not candidate.is_relative_to(root_resolved):
        _fail(path, "path escapes the artifact root")
    return candidate


def _find_repo_root(lock_path: Path) -> Path:
    for ancestor in (lock_path.resolve().parent, *lock_path.resolve().parents):
        if (ancestor / "experiments" / "run6" / "configs").is_dir() and (
            ancestor / "experiments" / "pyproject.toml"
        ).is_file():
            return ancestor
    raise LockValidationError(
        "cannot infer repository root; pass --repo-root explicitly"
    )


def _parse_utc(source: str, path: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(source)
    except ValueError as exc:
        raise LockValidationError(f"{path}: invalid ISO datetime {source!r}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        _fail(path, "datetime must include an explicit UTC offset")
    return parsed.astimezone(timezone.utc)


def _format_utc(value: datetime) -> str:
    return value.isoformat(timespec="seconds").replace("+00:00", "Z")


def _parse_qasm_maps(
    raw: bytes,
    *,
    distance: int,
    rounds: int,
    label: str,
) -> dict[str, QasmMap]:
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise LockValidationError(f"{label}: QASM is not valid UTF-8") from exc

    declarations: dict[tuple[str, str], tuple[int, str]] = {}
    measurements: dict[tuple[str, str], dict[int, int]] = {}
    for line_number, line in enumerate(text.splitlines(), start=1):
        declaration = _DECLARATION_RE.fullmatch(line)
        if declaration is not None:
            size_text, full_name, kind, suffix = declaration.groups()
            key = (kind, suffix)
            if key in declarations:
                _fail(label, f"line {line_number}: duplicate declaration {full_name}")
            declarations[key] = (int(size_text), full_name)
            continue
        if "bit[" in line and ("c_data_" in line or "c_syndrome_" in line):
            _fail(label, f"line {line_number}: malformed selected-register declaration")

        measurement = _MEASUREMENT_RE.fullmatch(line)
        if measurement is not None:
            full_name, kind, suffix, index_text, qubit_text = measurement.groups()
            key = (kind, suffix)
            index = int(index_text)
            target = measurements.setdefault(key, {})
            if index in target:
                _fail(
                    label,
                    f"line {line_number}: duplicate assignment {full_name}[{index}]",
                )
            target[index] = int(qubit_text)
            continue
        if "= measure" in line:
            _fail(label, f"line {line_number}: malformed measurement assignment")

    data_suffixes = {suffix for kind, suffix in declarations if kind == "data"}
    syndrome_suffixes = {suffix for kind, suffix in declarations if kind == "syndrome"}
    if not data_suffixes:
        _fail(label, "no declared c_data_* registers")
    if data_suffixes != syndrome_suffixes:
        _fail(
            label,
            "data/syndrome declaration suffix sets differ: "
            f"data_only={sorted(data_suffixes - syndrome_suffixes)}, "
            f"syndrome_only={sorted(syndrome_suffixes - data_suffixes)}",
        )
    if set(measurements) != set(declarations):
        _fail(
            label,
            "measurement/declaration register sets differ: "
            f"undeclared_measurements={sorted(set(measurements) - set(declarations))}, "
            f"unmeasured_declarations={sorted(set(declarations) - set(measurements))}",
        )

    result: dict[str, QasmMap] = {}
    seen_paths: dict[tuple[int, ...], str] = {}
    for suffix in sorted(data_suffixes):
        expected_sizes = {
            "data": distance,
            "syndrome": rounds * (distance - 1),
        }
        for kind, expected_size in expected_sizes.items():
            key = (kind, suffix)
            declared_size, _ = declarations[key]
            if declared_size != expected_size:
                _fail(
                    label,
                    f"{kind} register {suffix!r} declares {declared_size}, "
                    f"expected {expected_size}",
                )
            assigned = measurements.get(key, {})
            expected_indices = set(range(expected_size))
            if set(assigned) != expected_indices:
                _fail(
                    label,
                    f"{kind} register {suffix!r} has incomplete indices; "
                    f"missing={sorted(expected_indices - set(assigned))}, "
                    f"extra={sorted(set(assigned) - expected_indices)}",
                )

        data = tuple(measurements[("data", suffix)][index] for index in range(distance))
        syndrome_flat = tuple(
            measurements[("syndrome", suffix)][index]
            for index in range(rounds * (distance - 1))
        )
        by_round = tuple(
            syndrome_flat[
                round_index * (distance - 1) : (round_index + 1) * (distance - 1)
            ]
            for round_index in range(rounds)
        )
        if any(current != by_round[0] for current in by_round[1:]):
            _fail(label, f"register {suffix!r} changes ancilla assignment by round")
        checks = by_round[0]
        oriented: list[int] = []
        for index, check in enumerate(checks):
            oriented.extend((data[index], check))
        oriented.append(data[-1])
        oriented_tuple = tuple(oriented)
        if len(set(oriented_tuple)) != len(oriented_tuple):
            _fail(label, f"register {suffix!r} repeats a physical qubit in its path")
        if oriented_tuple in seen_paths:
            _fail(
                label,
                f"registers {seen_paths[oriented_tuple]!r} and {suffix!r} "
                "resolve to the same oriented path",
            )
        seen_paths[oriented_tuple] = suffix
        result[suffix] = QasmMap(data, checks, oriented_tuple)
    return result


def _validate_lock_schema(lock: Any) -> Mapping[str, Any]:
    root = _exact_keys(lock, _TOP_LEVEL_KEYS, "lock")
    for section, keys in _STATIC_KEYSETS.items():
        _exact_keys(root[section], keys, f"lock.{section}")
    for path_parts, keys in _NESTED_KEYSETS.items():
        value: Any = root
        for part in path_parts:
            value = _as_mapping(value, "lock." + ".".join(path_parts))[part]
        _exact_keys(value, keys, "lock." + ".".join(path_parts))

    snapshots = _as_mapping(root["snapshots"], "lock.snapshots")
    for snapshot_id, snapshot in snapshots.items():
        _exact_keys(snapshot, _SNAPSHOT_KEYS, f"lock.snapshots.{snapshot_id}")
    return root


def _validate_static_contract(
    lock: Mapping[str, Any],
    *,
    expected_status: str,
) -> None:
    _expect_equal(
        lock["manifest_id"],
        "run6-pnnl-pittsburgh-metadata-lock-v2",
        "lock.manifest_id",
    )
    _expect_equal(
        lock["status"],
        expected_status,
        "lock.status",
    )

    created = lock["created_from"]
    _expect_equal(
        created["allowed_inputs"],
        [
            INFO_NAME,
            CALIBRATION_NAME,
            QASM_NAMES[0],
            QASM_NAMES[1],
            "filesystem metadata for bitstrings.json",
        ],
        "lock.created_from.allowed_inputs",
    )
    for field in (
        "held_bitstring_content_read",
        "held_bitstring_hash_computed",
        "selection_used_syndrome_or_data_values",
    ):
        if _as_bool(created[field], f"lock.created_from.{field}"):
            _fail(f"lock.created_from.{field}", "must remain false")

    source = lock["source"]
    expected_source = {
        "record_url": "https://zenodo.org/records/20768087",
        "doi": "10.5281/zenodo.20768087",
        "release_version": "0.1",
        "license": "CC BY 4.0",
        "backend": "ibm_pittsburgh",
        "artifact_root": "experiments/data/run6/pnnl_ibm/ibm_pittsburgh",
        "logical_states": [0, 1],
    }
    for field, expected in expected_source.items():
        _expect_equal(source[field], expected, f"lock.source.{field}")

    for path, digest in lock["parent_artifacts"].items():
        _require_sha256(digest, f"lock.parent_artifacts.{path}")

    profiles = lock["hash_profiles"]
    _expect_equal(
        profiles["raw_sha256_v1"],
        {"input": "exact file bytes", "algorithm": "SHA-256"},
        "lock.hash_profiles.raw_sha256_v1",
    )
    _expect_equal(
        profiles["python_json_canonical_v1"]["serialize"],
        "CPython json.dumps(obj, sort_keys=True, ensure_ascii=False, "
        "separators=(',', ':'), allow_nan=False).encode('utf-8')",
        "lock.hash_profiles.python_json_canonical_v1.serialize",
    )
    _expect_equal(
        profiles["python_json_canonical_v1"]["trailing_newline"],
        False,
        "lock.hash_profiles.python_json_canonical_v1.trailing_newline",
    )
    _expect_equal(
        profiles["qasm_text_normalized_v1"]["algorithm"],
        "SHA-256",
        "lock.hash_profiles.qasm_text_normalized_v1.algorithm",
    )
    _expect_equal(
        profiles["state_pair_sha256_v1"]["state_order"],
        [0, 1],
        "lock.hash_profiles.state_pair_sha256_v1.state_order",
    )
    _expect_equal(
        profiles["snapshot_identity"],
        ["backend", "calibration.python_canonical_sha256"],
        "lock.hash_profiles.snapshot_identity",
    )

    parser = lock["directory_and_qasm_parser"]
    _expect_equal(
        parser["relative_job_regex"],
        _RELATIVE_JOB_RE_TEXT,
        "lock.directory_and_qasm_parser.relative_job_regex",
    )
    _expect_equal(
        parser["declaration_regex"],
        _DECLARATION_RE_TEXT,
        "lock.directory_and_qasm_parser.declaration_regex",
    )
    _expect_equal(
        parser["measurement_regex"],
        _MEASUREMENT_RE_TEXT,
        "lock.directory_and_qasm_parser.measurement_regex",
    )
    _expect_equal(
        lock["bitstring_parser_after_unblinding"]["detection_events"][
            "terminal_detector"
        ],
        False,
        "lock.bitstring_parser_after_unblinding.detection_events.terminal_detector",
    )
    _expect_equal(
        lock["artifact_tuple_schemas"],
        _EXPECTED_ARTIFACT_SCHEMAS,
        "lock.artifact_tuple_schemas",
    )
    _expect_equal(
        lock["cohort_row_schema"],
        _EXPECTED_COHORT_SCHEMA,
        "lock.cohort_row_schema",
    )

    selection = lock["cohort_selection"]
    _expect_equal(
        selection["no_value_dependent_filtering"],
        True,
        "lock.cohort_selection.no_value_dependent_filtering",
    )
    _expect_equal(
        selection["audit_counts"],
        {
            "state0_state1_path_agree_chain_instances": 852,
            "unique_configuration_path_groups": 837,
            "eligible_selected_groups": 11,
            "basis_counts": {"X": 7, "Z": 4},
            "raw_qasm_identical_selected_pairs": 0,
            "normalized_qasm_identical_selected_pairs": 0,
        },
        "lock.cohort_selection.audit_counts",
    )

    stream = lock["stream_construction"]
    for field in (
        "states_are_separate_replicates",
        "path_pooling_before_scoring",
        "cross_state_pairing",
    ):
        _as_bool(stream[field], f"lock.stream_construction.{field}")
    _expect_equal(
        stream["states_are_separate_replicates"],
        True,
        "lock.stream_construction.states_are_separate_replicates",
    )
    _expect_equal(
        stream["path_pooling_before_scoring"],
        False,
        "lock.stream_construction.path_pooling_before_scoring",
    )
    _expect_equal(
        stream["cross_state_pairing"],
        False,
        "lock.stream_construction.cross_state_pairing",
    )

    methods = lock["features_and_methods"]
    _expect_equal(
        methods["method_order"],
        _EXPECTED_METHOD_ORDER,
        "lock.features_and_methods.method_order",
    )
    _expect_equal(
        methods["target_method"],
        "space_composite",
        "lock.features_and_methods.target_method",
    )
    _expect_equal(
        methods["space_composite"],
        "fixed prior mass 1/2 on sparse and 1/2 on spectral, with each "
        "branch internally renormalized after dimension-ineligible sparse k "
        "values are removed",
        "lock.features_and_methods.space_composite",
    )
    _expect_equal(
        methods["sibling_ablations_not_eligible_for_target_selection"],
        ["space_sparse", "space_spectral"],
        "lock.features_and_methods.sibling_ablations_not_eligible_for_target_selection",
    )
    _expect_equal(
        methods["update_after_score"],
        True,
        "lock.features_and_methods.update_after_score",
    )

    metrics = lock["metrics_and_retention"]
    _expect_equal(
        metrics["target_method"],
        "space_composite",
        "lock.metrics_and_retention.target_method",
    )
    _expect_equal(
        metrics["retention_comparators"],
        ["dfr", "online_logistic"],
        "lock.metrics_and_retention.retention_comparators",
    )
    _expect_equal(
        metrics["post_selection"],
        "forbidden",
        "lock.metrics_and_retention.post_selection",
    )
    _expect_equal(
        metrics["all_method_outputs_required"],
        True,
        "lock.metrics_and_retention.all_method_outputs_required",
    )

    alarm = lock["alarm_calibration"]
    exact_alarm_values = {
        "block_length_shots": 32,
        "replicates": 4096,
        "alpha_per_path_state_episode": 0.01,
        "rng": "NumPy Generator(PCG64(seed))",
        "seed_formula": ("611000 + 100*cohort_index + 10*logical_state + method_index"),
        "cohort_index": "zero-based position in cohort_order",
        "method_index": ("zero-based position in features_and_methods.method_order"),
    }
    for field, expected in exact_alarm_values.items():
        _expect_equal(alarm[field], expected, f"lock.alarm_calibration.{field}")
    _expect_equal(
        alarm["secondary_fixed_e_threshold"]["alpha"],
        0.01,
        "lock.alarm_calibration.secondary_fixed_e_threshold.alpha",
    )
    _expect_equal(
        alarm["secondary_fixed_e_threshold"]["e_threshold"],
        100.0,
        "lock.alarm_calibration.secondary_fixed_e_threshold.e_threshold",
    )

    uncertainty = lock["uncertainty"]
    _expect_equal(
        uncertainty["primary_replicates"],
        10000,
        "lock.uncertainty.primary_replicates",
    )
    _expect_equal(
        uncertainty["primary_rng"],
        "NumPy Generator(PCG64(612500))",
        "lock.uncertainty.primary_rng",
    )
    _expect_equal(
        uncertainty["calibration_pair_sensitivity"]["replicates"],
        10000,
        "lock.uncertainty.calibration_pair_sensitivity.replicates",
    )
    _expect_equal(
        uncertainty["calibration_pair_sensitivity"]["rng"],
        "NumPy Generator(PCG64(612501))",
        "lock.uncertainty.calibration_pair_sensitivity.rng",
    )

    randomization = lock["randomization_audit"]
    _expect_equal(
        randomization["replicates"],
        256,
        "lock.randomization_audit.replicates",
    )
    _expect_equal(
        randomization["replicate_seeds"],
        "integers 610700 through 610955 inclusive",
        "lock.randomization_audit.replicate_seeds",
    )
    _expect_equal(
        randomization["rng"],
        "one fresh NumPy Generator(PCG64(seed)) per replicate",
        "lock.randomization_audit.rng",
    )


def _validate_parent_artifacts(
    lock: Mapping[str, Any],
    repo_root: Path,
) -> None:
    """Bind every declared parent digest to the current safe repository file."""

    for relative, expected_digest in lock["parent_artifacts"].items():
        label = f"lock.parent_artifacts.{relative}"
        path = _safe_under(repo_root, relative, label)
        observed_digest = _digest(_read_allowed_bytes(path, label))
        if observed_digest != expected_digest:
            _fail(
                label,
                "raw SHA-256 mismatch: "
                f"expected {expected_digest}, observed {observed_digest}",
            )


def _validate_artifact_tuple(
    value: Any,
    path: str,
    *,
    length: int,
) -> list[Any]:
    items = _as_list(value, path)
    if len(items) != length:
        _fail(path, f"expected {length} fields, observed {len(items)}")
    return items


def _validate_hashed_json_file(
    path: Path,
    record: Any,
    label: str,
    *,
    expected_top_keys: set[str],
) -> Mapping[str, Any]:
    fields = _validate_artifact_tuple(record, label, length=3)
    expected_size = _as_exact_int(fields[0], f"{label}[0]", minimum=1)
    expected_raw = _require_sha256(fields[1], f"{label}[1]")
    expected_canonical = _require_sha256(fields[2], f"{label}[2]")
    status = _regular_stat(path, label)
    if status.st_size != expected_size:
        _fail(
            label, f"size mismatch: expected {expected_size}, observed {status.st_size}"
        )
    raw = _read_allowed_bytes(path, label)
    if _digest(raw) != expected_raw:
        _fail(label, "raw SHA-256 mismatch")
    value = _decode_json(raw, str(path))
    mapping = _exact_keys(value, expected_top_keys, str(path))
    if _digest(_canonical_json_bytes(mapping)) != expected_canonical:
        _fail(label, "python-canonical SHA-256 mismatch")
    return mapping


def _validate_qasm_file(
    path: Path,
    record: Any,
    label: str,
    *,
    distance: int,
    rounds: int,
) -> tuple[str, str, dict[str, QasmMap]]:
    fields = _validate_artifact_tuple(record, label, length=3)
    expected_size = _as_exact_int(fields[0], f"{label}[0]", minimum=1)
    expected_raw = _require_sha256(fields[1], f"{label}[1]")
    expected_normalized = _require_sha256(fields[2], f"{label}[2]")
    status = _regular_stat(path, label)
    if status.st_size != expected_size:
        _fail(
            label, f"size mismatch: expected {expected_size}, observed {status.st_size}"
        )
    raw = _read_allowed_bytes(path, label)
    raw_digest = _digest(raw)
    normalized_digest = _digest(_normalize_qasm(raw, str(path)))
    if raw_digest != expected_raw:
        _fail(label, "raw SHA-256 mismatch")
    if normalized_digest != expected_normalized:
        _fail(label, "normalized SHA-256 mismatch")
    maps = _parse_qasm_maps(
        raw,
        distance=distance,
        rounds=rounds,
        label=label,
    )
    return raw_digest, normalized_digest, maps


def _validate_snapshots(
    lock: Mapping[str, Any],
    artifact_root: Path,
) -> tuple[dict[str, dict[str, Any]], int]:
    snapshots = _as_mapping(lock["snapshots"], "lock.snapshots")
    if len(snapshots) != 20:
        _fail(
            "lock.snapshots",
            f"expected 20 selected snapshots, observed {len(snapshots)}",
        )

    computed: dict[str, dict[str, Any]] = {}
    held_count = 0
    for snapshot_id, raw_snapshot in snapshots.items():
        label = f"lock.snapshots.{snapshot_id}"
        snapshot = _as_mapping(raw_snapshot, label)
        relative = _as_string(snapshot["relative_job_dir"], f"{label}.relative_job_dir")
        match = _RELATIVE_JOB_RE.fullmatch(relative)
        if match is None:
            _fail(f"{label}.relative_job_dir", "does not match the locked job regex")
        distance_from_path = int(match.group("distance"))
        rounds_from_path = int(match.group("rounds"))
        job_from_path = int(match.group("job"))
        expected_snapshot_id = (
            f"pgh_d{distance_from_path}_r{rounds_from_path}_job_{job_from_path}"
        )
        _expect_equal(snapshot_id, expected_snapshot_id, label)
        job_dir = _safe_under(
            artifact_root,
            relative,
            f"{label}.relative_job_dir",
        )

        metadata = _validate_artifact_tuple(
            snapshot["metadata"],
            f"{label}.metadata",
            length=7,
        )
        distance = _as_exact_int(metadata[0], f"{label}.metadata[0]", minimum=2)
        rounds = _as_exact_int(metadata[1], f"{label}.metadata[1]", minimum=1)
        basis = _as_string(metadata[2], f"{label}.metadata[2]")
        if basis not in {"X", "Z"}:
            _fail(f"{label}.metadata[2]", "basis must be X or Z")
        shots = _as_exact_int(metadata[3], f"{label}.metadata[3]", minimum=2048)
        n_chains = _as_exact_int(metadata[4], f"{label}.metadata[4]", minimum=1)
        date_source = _as_string(metadata[5], f"{label}.metadata[5]")
        date_utc_text = _as_string(metadata[6], f"{label}.metadata[6]")
        date_utc = _parse_utc(date_source, f"{label}.metadata[5]")
        _expect_equal(
            date_utc_text,
            _format_utc(date_utc),
            f"{label}.metadata[6]",
        )
        _expect_equal(distance, distance_from_path, f"{label}.metadata[0]")
        _expect_equal(rounds, rounds_from_path, f"{label}.metadata[1]")

        info = _validate_hashed_json_file(
            job_dir / INFO_NAME,
            snapshot["info"],
            f"{label}.info",
            expected_top_keys=_EXPECTED_INFO_KEYS,
        )
        expected_info = {
            "backend": lock["source"]["backend"],
            "d": distance,
            "rounds": rounds,
            "basis": basis,
            "logical_states": [0, 1],
            "shots": shots,
            "n_chains": n_chains,
        }
        _expect_equal(dict(info), expected_info, f"{label}.info metadata")

        calibration = _validate_hashed_json_file(
            job_dir / CALIBRATION_NAME,
            snapshot["calibration"],
            f"{label}.calibration",
            expected_top_keys=_EXPECTED_CALIBRATION_KEYS,
        )
        _expect_equal(
            calibration["backend_name"],
            lock["source"]["backend"],
            f"{label}.calibration backend_name",
        )
        _expect_equal(
            calibration["last_update_date"],
            date_source,
            f"{label}.calibration last_update_date",
        )

        raw0, normalized0, maps0 = _validate_qasm_file(
            job_dir / QASM_NAMES[0],
            snapshot["qasm_state0"],
            f"{label}.qasm_state0",
            distance=distance,
            rounds=rounds,
        )
        raw1, normalized1, maps1 = _validate_qasm_file(
            job_dir / QASM_NAMES[1],
            snapshot["qasm_state1"],
            f"{label}.qasm_state1",
            distance=distance,
            rounds=rounds,
        )
        pair_record = _validate_artifact_tuple(
            snapshot["qasm_pair"],
            f"{label}.qasm_pair",
            length=2,
        )
        expected_raw_pair = _require_sha256(
            pair_record[0],
            f"{label}.qasm_pair[0]",
        )
        expected_normalized_pair = _require_sha256(
            pair_record[1],
            f"{label}.qasm_pair[1]",
        )
        _expect_equal(
            expected_raw_pair,
            _pair_digest(raw0, raw1),
            f"{label}.qasm_pair[0]",
        )
        _expect_equal(
            expected_normalized_pair,
            _pair_digest(normalized0, normalized1),
            f"{label}.qasm_pair[1]",
        )

        held = _validate_artifact_tuple(
            snapshot["held_bitstrings"],
            f"{label}.held_bitstrings",
            length=3,
        )
        expected_held_size = _as_exact_int(
            held[0],
            f"{label}.held_bitstrings[0]",
            minimum=1,
        )
        _expect_equal(
            held[1],
            False,
            f"{label}.held_bitstrings[1]",
        )
        _expect_equal(
            held[2],
            None,
            f"{label}.held_bitstrings[2]",
        )
        observed_held_size = _held_payload_size(
            job_dir / HELD_PAYLOAD_NAME,
            f"{label}.held_bitstrings",
        )
        if observed_held_size != expected_held_size:
            _fail(
                f"{label}.held_bitstrings",
                f"stat size mismatch: expected {expected_held_size}, "
                f"observed {observed_held_size}",
            )
        held_count += 1

        computed[snapshot_id] = {
            "distance": distance,
            "rounds": rounds,
            "basis": basis,
            "shots": shots,
            "date_utc": date_utc,
            "calibration_canonical": snapshot["calibration"][2],
            "qasm_raw_pair": expected_raw_pair,
            "qasm_normalized_pair": expected_normalized_pair,
            "maps0": maps0,
            "maps1": maps1,
        }
    return computed, held_count


def _int_tuple(value: Any, path: str, *, expected_length: int) -> tuple[int, ...]:
    items = _as_list(value, path)
    if len(items) != expected_length:
        _fail(path, f"expected length {expected_length}, observed {len(items)}")
    result = tuple(
        _as_exact_int(item, f"{path}[{index}]", minimum=0)
        for index, item in enumerate(items)
    )
    if len(set(result)) != len(result):
        _fail(path, "physical qubits must be unique")
    return result


def _validate_cohorts(
    lock: Mapping[str, Any],
    snapshots: Mapping[str, Mapping[str, Any]],
) -> dict[str, int]:
    order = _as_list(lock["cohort_order"], "lock.cohort_order")
    rows = _as_list(lock["cohort_pairs"], "lock.cohort_pairs")
    if len(order) != 11 or len(rows) != 11:
        _fail(
            "lock.cohort_order/cohort_pairs",
            f"expected 11 cohorts, observed order={len(order)}, rows={len(rows)}",
        )
    if len(set(order)) != len(order) or not all(
        isinstance(item, str) for item in order
    ):
        _fail("lock.cohort_order", "cohort IDs must be unique strings")

    parsed_rows: list[dict[str, Any]] = []
    referenced_snapshots: set[str] = set()
    seen_paths: set[tuple[int, int, str, tuple[int, ...]]] = set()
    basis_counts = {"X": 0, "Z": 0}
    raw_same_count = 0
    normalized_same_count = 0
    paired_shots = 0
    paired_cycles = 0
    calibration_pairs: set[str] = set()

    for cohort_index, raw_row in enumerate(rows):
        row_path = f"lock.cohort_pairs[{cohort_index}]"
        row = _as_list(raw_row, row_path)
        if len(row) != len(_EXPECTED_COHORT_SCHEMA):
            _fail(
                row_path,
                f"expected {len(_EXPECTED_COHORT_SCHEMA)} fields, observed {len(row)}",
            )
        values = dict(zip(_EXPECTED_COHORT_SCHEMA, row))
        cohort_id = _as_string(values["cohort_id"], f"{row_path}.cohort_id")
        _expect_equal(cohort_id, order[cohort_index], f"{row_path}.cohort_id")
        distance = _as_exact_int(values["distance"], f"{row_path}.distance", minimum=2)
        rounds = _as_exact_int(values["rounds"], f"{row_path}.rounds", minimum=1)
        basis = _as_string(values["basis"], f"{row_path}.basis")
        if basis not in basis_counts:
            _fail(f"{row_path}.basis", "basis must be X or Z")
        basis_counts[basis] += 1
        suffix = _as_string(values["register_suffix"], f"{row_path}.register_suffix")
        expected_id = f"pgh_d{distance}_r{rounds}_{basis.lower()}_{suffix}"
        _expect_equal(cohort_id, expected_id, f"{row_path}.cohort_id")
        data = _int_tuple(
            values["data_qubits"],
            f"{row_path}.data_qubits",
            expected_length=distance,
        )
        checks = _int_tuple(
            values["syndrome_qubits"],
            f"{row_path}.syndrome_qubits",
            expected_length=distance - 1,
        )
        oriented = _int_tuple(
            values["oriented_path"],
            f"{row_path}.oriented_path",
            expected_length=2 * distance - 1,
        )
        interleaved: list[int] = []
        for index, check in enumerate(checks):
            interleaved.extend((data[index], check))
        interleaved.append(data[-1])
        _expect_equal(oriented, tuple(interleaved), f"{row_path}.oriented_path")

        early_id = _as_string(
            values["early_snapshot_id"],
            f"{row_path}.early_snapshot_id",
        )
        late_id = _as_string(
            values["late_snapshot_id"],
            f"{row_path}.late_snapshot_id",
        )
        if early_id == late_id:
            _fail(row_path, "early and late snapshots must differ")
        if early_id not in snapshots or late_id not in snapshots:
            _fail(row_path, "references an unknown snapshot ID")
        referenced_snapshots.update((early_id, late_id))
        early = snapshots[early_id]
        late = snapshots[late_id]
        for side, snapshot in (("early", early), ("late", late)):
            observed_config = (
                snapshot["distance"],
                snapshot["rounds"],
                snapshot["basis"],
            )
            _expect_equal(
                observed_config,
                (distance, rounds, basis),
                f"{row_path}.{side}_snapshot configuration",
            )
            for state in (0, 1):
                maps = snapshot[f"maps{state}"]
                if suffix not in maps:
                    _fail(
                        row_path,
                        f"{side} state{state} QASM lacks register suffix {suffix!r}",
                    )
                parsed = maps[suffix]
                expected_map = QasmMap(data, checks, oriented)
                _expect_equal(
                    parsed,
                    expected_map,
                    f"{row_path}.{side}_snapshot state{state} QASM map",
                )

        if not early["date_utc"] < late["date_utc"]:
            _fail(row_path, "early backend-property UTC instant must precede late")
        if early["calibration_canonical"] == late["calibration_canonical"]:
            _fail(row_path, "early and late canonical calibration hashes must differ")

        m = _as_exact_int(values["m"], f"{row_path}.m", minimum=1)
        expected_m = min(early["shots"] // 3, late["shots"])
        _expect_equal(m, expected_m, f"{row_path}.m")
        if 3 * m > early["shots"] or m > late["shots"]:
            _fail(row_path, "partition size exceeds available shots")

        raw_same = early["qasm_raw_pair"] == late["qasm_raw_pair"]
        normalized_same = early["qasm_normalized_pair"] == late["qasm_normalized_pair"]
        _expect_equal(
            _as_bool(
                values["raw_qasm_pair_identical"],
                f"{row_path}.raw_qasm_pair_identical",
            ),
            raw_same,
            f"{row_path}.raw_qasm_pair_identical",
        )
        _expect_equal(
            _as_bool(
                values["normalized_qasm_pair_identical_audit_only"],
                f"{row_path}.normalized_qasm_pair_identical_audit_only",
            ),
            normalized_same,
            f"{row_path}.normalized_qasm_pair_identical_audit_only",
        )
        raw_same_count += int(raw_same)
        normalized_same_count += int(normalized_same)
        expected_label = (
            "circuit_controlled_cross_property_snapshot"
            if raw_same
            else "circuit_and_hardware_domain_shift"
        )
        _expect_equal(
            values["claim_label"],
            expected_label,
            f"{row_path}.claim_label",
        )
        calibration_pair = (
            f"{early['calibration_canonical']}--{late['calibration_canonical']}"
        )
        _expect_equal(
            values["calibration_pair_id"],
            calibration_pair,
            f"{row_path}.calibration_pair_id",
        )
        calibration_pairs.add(calibration_pair)

        group_key = (distance, rounds, basis, oriented)
        if group_key in seen_paths:
            _fail(row_path, "duplicate configuration/oriented-path cohort")
        seen_paths.add(group_key)
        paired_shots += 2 * m
        paired_cycles += 2 * m * rounds
        parsed_rows.append(values)

    if referenced_snapshots != set(snapshots):
        _fail(
            "lock.snapshots",
            "snapshot table must contain exactly the cohort-referenced snapshots; "
            f"unreferenced={sorted(set(snapshots) - referenced_snapshots)}, "
            f"missing={sorted(referenced_snapshots - set(snapshots))}",
        )
    if len(calibration_pairs) != 5:
        _fail(
            "lock.cohort_pairs",
            f"expected five ordered calibration-hash pairs, observed {len(calibration_pairs)}",
        )

    audit = lock["cohort_selection"]["audit_counts"]
    _expect_equal(
        basis_counts,
        audit["basis_counts"],
        "lock.cohort_selection.audit_counts.basis_counts",
    )
    _expect_equal(
        len(parsed_rows),
        audit["eligible_selected_groups"],
        "lock.cohort_selection.audit_counts.eligible_selected_groups",
    )
    _expect_equal(
        raw_same_count,
        audit["raw_qasm_identical_selected_pairs"],
        "lock.cohort_selection.audit_counts.raw_qasm_identical_selected_pairs",
    )
    _expect_equal(
        normalized_same_count,
        audit["normalized_qasm_identical_selected_pairs"],
        "lock.cohort_selection.audit_counts.normalized_qasm_identical_selected_pairs",
    )

    resources = lock["aggregate_resource_counts_per_pre_or_post_phase"]
    expected_resources = {
        "path_groups": len(parsed_rows),
        "path_state_streams": len(parsed_rows) * 2,
        "paired_shots": paired_shots,
        "paired_cycle_updates": paired_cycles,
    }
    _expect_equal(
        resources,
        expected_resources,
        "lock.aggregate_resource_counts_per_pre_or_post_phase",
    )
    return {
        "cohorts": len(parsed_rows),
        "paired_shots": paired_shots,
        "paired_cycles": paired_cycles,
    }


def _validate_seed_arithmetic(lock: Mapping[str, Any]) -> int:
    cohort_count = len(lock["cohort_order"])
    state_values = tuple(lock["source"]["logical_states"])
    method_count = len(lock["features_and_methods"]["method_order"])
    threshold_seeds = {
        611000 + 100 * cohort_index + 10 * logical_state + method_index
        for cohort_index in range(cohort_count)
        for logical_state in state_values
        for method_index in range(method_count)
    }
    expected_count = cohort_count * len(state_values) * method_count
    if len(threshold_seeds) != expected_count:
        _fail("lock.alarm_calibration.seed_formula", "threshold seeds collide")

    randomization_seeds = set(range(610700, 610956))
    uncertainty_seeds = {612500, 612501}
    if threshold_seeds & randomization_seeds:
        _fail("lock.alarm_calibration.seed_formula", "overlaps randomization seeds")
    if threshold_seeds & uncertainty_seeds:
        _fail("lock.alarm_calibration.seed_formula", "overlaps uncertainty seeds")
    if randomization_seeds & uncertainty_seeds:
        _fail("lock.uncertainty", "uncertainty seeds overlap randomization seeds")

    alpha = lock["alarm_calibration"]["alpha_per_path_state_episode"]
    replicates = lock["alarm_calibration"]["replicates"]
    order_index = math.ceil((replicates + 1) * (1.0 - alpha)) - 1
    if order_index != 4056 or not 0 <= order_index < replicates:
        _fail(
            "lock.alarm_calibration.threshold_order_statistic",
            f"unexpected locked order-statistic index {order_index}",
        )
    return expected_count


def _validate_static_contract_digest(
    lock: Mapping[str, Any],
    *,
    expected_digest: str,
) -> None:
    """Bind every non-artifact-dependent field to the ratified v2 contract."""

    static_contract = {
        key: value
        for key, value in lock.items()
        if key not in _ARTIFACT_DEPENDENT_TOP_LEVEL_FIELDS
    }
    observed = _digest(_canonical_json_bytes(static_contract))
    if observed != expected_digest:
        _fail(
            "lock",
            "non-artifact ratified contract differs from v2 "
            f"(expected SHA-256 {expected_digest}, observed {observed})",
        )


def validate_lock(
    lock_path: str | Path,
    *,
    repo_root: str | Path | None = None,
    mode: str = "frozen",
) -> ValidationReport:
    """Validate the lock and safe artifacts without reading held bitstrings."""

    if mode not in _LOCK_STATUS_BY_MODE:
        raise ValueError("mode must be exactly 'pending' or 'frozen'.")
    expected_status = _LOCK_STATUS_BY_MODE[mode]
    expected_digest = (
        _PENDING_STATIC_CONTRACT_SHA256
        if mode == "pending"
        else _FROZEN_STATIC_CONTRACT_SHA256
    )
    if re.fullmatch(r"[0-9a-f]{64}", expected_digest) is None:
        _fail("validator", f"{mode} static-contract digest is not finalized")
    lock_file = Path(lock_path)
    root = (
        Path(repo_root).resolve()
        if repo_root is not None
        else _find_repo_root(lock_file)
    )
    raw_lock = _read_allowed_bytes(lock_file, "lock file")
    lock = _validate_lock_schema(_decode_json(raw_lock, str(lock_file)))
    _validate_static_contract(lock, expected_status=expected_status)
    _validate_parent_artifacts(lock, root)

    artifact_root = _safe_under(
        root,
        lock["source"]["artifact_root"],
        "lock.source.artifact_root",
    )
    if not artifact_root.is_dir():
        _fail("lock.source.artifact_root", f"missing directory {artifact_root}")
    if artifact_root.is_symlink():
        _fail("lock.source.artifact_root", "symbolic links are forbidden")

    snapshots, held_count = _validate_snapshots(lock, artifact_root)
    cohort_summary = _validate_cohorts(lock, snapshots)
    threshold_seed_count = _validate_seed_arithmetic(lock)
    _validate_static_contract_digest(lock, expected_digest=expected_digest)
    return ValidationReport(
        lock_path=lock_file.resolve(),
        artifact_root=artifact_root.resolve(),
        snapshots=len(snapshots),
        cohorts=cohort_summary["cohorts"],
        held_payloads_statted=held_count,
        paired_shots_per_phase=cohort_summary["paired_shots"],
        paired_cycle_updates_per_phase=cohort_summary["paired_cycles"],
        threshold_seed_count=threshold_seed_count,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the ratified Pittsburgh metadata lock without opening "
            "or hashing any bitstrings.json."
        )
    )
    parser.add_argument(
        "lock",
        nargs="?",
        default="experiments/run6/configs/pnnl_pittsburgh_locked.json",
        help="path to the ratified JSON lock",
    )
    parser.add_argument(
        "--repo-root",
        help="repository root used to resolve the locked artifact_root",
    )
    parser.add_argument(
        "--mode",
        choices=tuple(_LOCK_STATUS_BY_MODE),
        default="frozen",
        help="require the explicit pending or final frozen contract",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        report = validate_lock(
            args.lock,
            repo_root=args.repo_root,
            mode=args.mode,
        )
    except LockValidationError as exc:
        print(f"PNNL metadata lock INVALID: {exc}", file=sys.stderr)
        return 2
    print(
        "PNNL metadata lock VALID "
        f"(snapshots={report.snapshots}, cohorts={report.cohorts}, "
        f"held_files_statted_only={report.held_payloads_statted}, "
        f"paired_shots_per_phase={report.paired_shots_per_phase}, "
        f"paired_cycles_per_phase={report.paired_cycle_updates_per_phase}, "
        f"threshold_seeds={report.threshold_seed_count})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
