"""Fail-closed provenance gate for the Run 6 post-detector repair.

The original Run 6 freeze remains the authority for the completed detector
run.  This module adds a separate implementation -> manifest -> ratification
chain for the narrowly scoped, post-detector and pre-outcome compatibility
repair documented in ``references/run6_post_detector_schema_incident.md``.

No helper in this module accepts a raw-data, decoder-outcome, or PNNL payload
path.  Detector artifacts are treated as opaque byte strings: their declared
sizes and SHA-256 digests are checked without interpreting numeric contents.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import re
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from aoc.run6_protocol import (
    FREEZE_MANIFEST_KEYS,
    FREEZE_MANIFEST_RELATIVE,
    FREEZE_RATIFICATION_KEYS,
    FREEZE_RATIFICATION_RELATIVE,
    canonical_json_bytes,
    environment_fingerprint,
    load_strict_json,
    require_exact_keys,
    require_thread_environment,
    sha256_file,
    verify_python_environment_lock,
    verify_runtime_module_origins,
)

INCIDENT_COMMIT = "c36a3b0980588db4663f2a51692294fcdfddc9a5"
ORIGINAL_RATIFICATION_COMMIT = "7e378d7f1d99818fc5e366bb14a7200767722d6c"

REPAIR_MANIFEST_RELATIVE = "experiments/run6/repair_manifest.json"
REPAIR_RATIFICATION_RELATIVE = "experiments/run6/repair_ratification.json"
DETECTOR_MANIFEST_RELATIVE = (
    "experiments/run6/results/google_detector/detector_freeze_manifest.json"
)
FAILED_ATTEMPT_ROOT_RELATIVE = "experiments/run6/results/google_randomization"
PYTHON_LOCK_RELATIVE = "experiments/run6/configs/python_environment_lock.txt"

EXPECTED_DETECTOR_MANIFEST_SHA256 = (
    "ed9d9dcdcb2b3e78d144f2a2ce3cec6b6269ffce3f7e18f443784e5d6174c0c3"
)
EXPECTED_DETECTOR_ARTIFACT_COUNT = 231
EXPECTED_FAILED_ATTEMPT_COUNT = 32
EXPECTED_FAILED_STDERR_SHA256 = (
    "d3bc72d114336901c1b502f2722cc3e4a7c44030f03fa38e7f861fc8c5e6dd3e"
)
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()

REPAIR_DIFF_STATUS: dict[str, str] = {
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
RUN6_REQUIRED_REPAIR_PATHS = tuple(REPAIR_DIFF_STATUS)

REPAIR_ACCESS_RECORD: dict[str, object] = {
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
DETECTOR_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "protocol_id",
        "git_commit",
        "command",
        "started_unix",
        "finished_unix",
        "config_sha256",
        "method_spec_sha256",
        "freeze_ratification_sha256",
        "detector_script_sha256",
        "deviation_ledger",
        "source_archive_sha256",
        "source_archive_bytes",
        "verified_zip_member_sha256",
        "detection_file_bytes",
        "circuit_sha256",
        "detector_layout_index_sha256",
        "warm_checkpoint_sha256",
        "threshold_checkpoint_sha256",
        "held_final_checkpoint_sha256",
        "threshold_table_sha256",
        "detector_only",
        "outcome_accessed",
        "outcome_join_authorized",
        "environment",
        "resources",
        "performance",
        "artifacts",
    }
)

_HEX_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40,64}$")
_ATTEMPT_RE = re.compile(r"^\.attempt_(\d{3})_(\d{3})_(\d+)_(\d+)$")


def _strict_json_bytes(payload: bytes, *, context: str) -> dict[str, Any]:
    def reject_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key in {context}: {key!r}.")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"Non-finite JSON number in {context}: {value}.")

    value = json.loads(
        payload.decode("utf-8-sig"),
        object_pairs_hook=reject_pairs,
        parse_constant=reject_constant,
    )
    if not isinstance(value, dict):
        raise TypeError(f"{context} must contain a JSON object.")
    return value


def _git(
    repo_root: Path,
    arguments: Sequence[str],
    *,
    context: str,
) -> bytes:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repo_root,
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(f"{context}: {detail or 'Git query failed'}")
    return result.stdout


def _commit(repo_root: Path, revision: str, *, context: str) -> str:
    if not isinstance(revision, str) or not revision:
        raise TypeError(f"{context} must be a nonempty string.")
    resolved = (
        _git(
            repo_root,
            ["rev-parse", "--verify", f"{revision}^{{commit}}"],
            context=f"{context} is not a Git commit",
        )
        .decode("ascii")
        .strip()
    )
    if _COMMIT_RE.fullmatch(resolved) is None:
        raise ValueError(f"{context} resolved to an invalid commit identifier.")
    return resolved


def _require_ancestor(
    repo_root: Path,
    ancestor: str,
    descendant: str,
    *,
    context: str,
) -> None:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=repo_root,
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        raise ValueError(f"{context}: {ancestor} is not an ancestor of {descendant}.")


def _blob_bytes(repo_root: Path, commit: str, relative: str) -> bytes:
    return _git(
        repo_root,
        ["show", f"{commit}:{relative}"],
        context=f"Cannot read {relative!r} at {commit}",
    )


def _blob_sha256(repo_root: Path, commit: str, relative: str) -> str:
    return hashlib.sha256(_blob_bytes(repo_root, commit, relative)).hexdigest()


def _safe_relative(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{context} must be a nonempty string.")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
        or "\\" in value
        or ":" in value
    ):
        raise ValueError(f"{context} is not a canonical repository-relative path.")
    return value


def _canonical_repo_path(
    repo_root: Path,
    supplied: str | Path,
    expected_relative: str,
    *,
    context: str,
) -> Path:
    expected = Path(os.path.abspath(repo_root / expected_relative))
    observed = Path(os.path.abspath(supplied))
    if observed != expected:
        raise ValueError(f"{context} must be canonical {expected_relative}.")
    try:
        relative = expected.relative_to(repo_root)
    except ValueError as exc:
        raise ValueError(f"{context} escapes the canonical repository root.") from exc
    current = repo_root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"{context} contains a symbolic-link path component.")
    return expected


def _require_digest(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or _HEX_RE.fullmatch(value) is None:
        raise ValueError(f"{context} is not a lowercase SHA-256 digest.")
    return value


def _validate_hashes(value: Any, *, context: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or not value:
        raise TypeError(f"{context} must be a nonempty object.")
    result: dict[str, str] = {}
    for raw_path, raw_digest in value.items():
        relative = _safe_relative(raw_path, context=f"{context} path")
        result[relative] = _require_digest(
            raw_digest,
            context=f"{context}.{relative}",
        )
    return result


def _diff_status(
    repo_root: Path,
    base: str,
    target: str,
) -> dict[str, str]:
    payload = _git(
        repo_root,
        ["diff", "--name-status", "--no-renames", "-z", base, target, "--"],
        context=f"Cannot inspect repair diff {base}..{target}",
    )
    fields = payload.split(b"\0")
    if fields and fields[-1] == b"":
        fields.pop()
    if len(fields) % 2:
        raise ValueError("Unexpected NUL-delimited Git name-status output.")
    observed: dict[str, str] = {}
    for index in range(0, len(fields), 2):
        status = fields[index].decode("ascii")
        relative = fields[index + 1].decode("utf-8")
        if status not in {"A", "M", "D"}:
            raise ValueError(f"Unsupported repair diff status {status!r}.")
        relative = _safe_relative(relative, context="repair diff path")
        if relative in observed:
            raise ValueError(f"Duplicate repair diff path: {relative}.")
        observed[relative] = status
    return observed


def _require_exact_diff(
    repo_root: Path,
    base: str,
    target: str,
    expected: Mapping[str, str],
    *,
    context: str,
) -> dict[str, str]:
    observed = _diff_status(repo_root, base, target)
    expected_dict = dict(expected)
    if observed != expected_dict:
        missing = sorted(set(expected_dict) - set(observed))
        unexpected = sorted(set(observed) - set(expected_dict))
        wrong_status = {
            path: {"expected": expected_dict[path], "observed": observed[path]}
            for path in set(expected_dict) & set(observed)
            if expected_dict[path] != observed[path]
        }
        raise ValueError(
            f"{context} differs from the audited whitelist; "
            f"missing={missing}, unexpected={unexpected}, "
            f"wrong_status={wrong_status}."
        )
    return observed


def _require_pushed(
    repo_root: Path,
    commit: str,
    remote_ref: str,
    *,
    context: str,
) -> None:
    remote = _commit(repo_root, remote_ref, context=remote_ref)
    _require_ancestor(repo_root, commit, remote, context=context)


def _path_introduction_commit(
    repo_root: Path,
    revision: str,
    relative: str,
) -> str:
    lines = (
        _git(
            repo_root,
            [
                "log",
                "--diff-filter=A",
                "--format=%H",
                revision,
                "--",
                relative,
            ],
            context=f"Cannot locate introduction commit for {relative}",
        )
        .decode("ascii")
        .splitlines()
    )
    if len(lines) != 1:
        raise ValueError(
            f"{relative} must have exactly one introduction commit; observed={lines}."
        )
    return _commit(repo_root, lines[0], context=f"{relative} introduction commit")


def _repair_diff_contract(
    required_repair_paths: Iterable[str] | None,
) -> dict[str, str]:
    if required_repair_paths is None:
        return dict(REPAIR_DIFF_STATUS)
    normalized = {
        _safe_relative(path, context="required repair path")
        for path in required_repair_paths
    }
    expected = set(REPAIR_DIFF_STATUS)
    if normalized != expected:
        raise ValueError(
            "Required repair paths must equal the audited whitelist; "
            f"missing={sorted(expected - normalized)}, "
            f"unexpected={sorted(normalized - expected)}."
        )
    return dict(REPAIR_DIFF_STATUS)


def _runtime_module_origins(repo_root: Path) -> dict[str, str]:
    observed = verify_runtime_module_origins(repo_root)
    module = importlib.import_module("aoc.run6_repair")
    raw_path = getattr(module, "__file__", None)
    expected_relative = "experiments/aoc/run6_repair.py"
    if not isinstance(raw_path, str):
        raise TypeError("Runtime module aoc.run6_repair has no source path.")
    if Path(raw_path).resolve() != (repo_root / expected_relative).resolve():
        raise RuntimeError(
            "Runtime module aoc.run6_repair did not load from repository source."
        )
    return {**observed, "aoc.run6_repair": expected_relative}


def _require_clean_runtime_worktree(repo_root: Path) -> None:
    payload = _git(
        repo_root,
        [
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
            "--",
            "experiments/aoc",
            "experiments/run6/scripts",
        ],
        context="Cannot inspect repair runtime worktree",
    )
    if payload:
        entries = [
            item.decode("utf-8", errors="replace")
            for item in payload.split(b"\0")
            if item
        ]
        raise ValueError(
            f"Repair runtime worktree is not committed and clean: {entries}."
        )


def verify_historical_original_freeze_chain(
    original_ratification_path: str | Path,
    *,
    repo_root: str | Path,
    incident_commit: str | None = None,
    original_ratification_commit: str | None = None,
    remote_ref: str = "origin/main",
) -> dict[str, Any]:
    """Verify the original freeze from historical Git blobs, not new worktree code."""

    root = Path(repo_root).resolve()
    ratification_path = _canonical_repo_path(
        root,
        original_ratification_path,
        FREEZE_RATIFICATION_RELATIVE,
        context="original ratification",
    )
    incident = _commit(
        root,
        incident_commit or INCIDENT_COMMIT,
        context="incident_commit",
    )
    if incident_commit is None and incident != INCIDENT_COMMIT:
        raise ValueError("Incident commit changed from the audited commit.")
    ratification_commit = _commit(
        root,
        original_ratification_commit or ORIGINAL_RATIFICATION_COMMIT,
        context="original_ratification_commit",
    )
    if (
        original_ratification_commit is None
        and ratification_commit != ORIGINAL_RATIFICATION_COMMIT
    ):
        raise ValueError("Original ratification commit changed.")
    _require_ancestor(
        root,
        ratification_commit,
        incident,
        context="Original ratification is not before the incident",
    )
    _require_pushed(
        root,
        incident,
        remote_ref,
        context="Incident commit has not been pushed",
    )
    _require_pushed(
        root,
        ratification_commit,
        remote_ref,
        context="Original ratification commit has not been pushed",
    )

    ratification_bytes = _blob_bytes(
        root,
        ratification_commit,
        FREEZE_RATIFICATION_RELATIVE,
    )
    if (
        not ratification_path.is_file()
        or ratification_path.is_symlink()
        or ratification_path.read_bytes() != ratification_bytes
    ):
        raise ValueError(
            "Original ratification worktree bytes differ from the historical blob."
        )
    head = _commit(root, "HEAD", context="HEAD")
    if (
        _blob_sha256(root, head, FREEZE_RATIFICATION_RELATIVE)
        != hashlib.sha256(ratification_bytes).hexdigest()
    ):
        raise ValueError("Original ratification differs from the current HEAD blob.")
    ratification = _strict_json_bytes(
        ratification_bytes,
        context="historical original ratification",
    )
    require_exact_keys(
        ratification,
        FREEZE_RATIFICATION_KEYS,
        context="historical original ratification",
    )
    if (
        ratification["schema_version"] != "run6-freeze-ratification-v1"
        or ratification["status"] != "frozen_before_held_value_access"
        or ratification["held_value_access_before_ratification"] is not False
    ):
        raise ValueError("Historical original ratification is invalid.")
    freeze_commit = _commit(
        root,
        ratification["freeze_commit"],
        context="historical freeze_commit",
    )
    _require_ancestor(
        root,
        freeze_commit,
        ratification_commit,
        context="Freeze commit is not before original ratification",
    )
    _require_pushed(
        root,
        freeze_commit,
        remote_ref,
        context="Historical freeze commit has not been pushed",
    )

    ratified_hashes = _validate_hashes(
        ratification["hashes"],
        context="historical ratification hashes",
    )
    if FREEZE_MANIFEST_RELATIVE not in ratified_hashes:
        raise ValueError("Historical ratification does not bind its manifest.")
    manifest_bytes = _blob_bytes(root, freeze_commit, FREEZE_MANIFEST_RELATIVE)
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    if ratified_hashes[FREEZE_MANIFEST_RELATIVE] != manifest_sha:
        raise ValueError("Historical freeze manifest digest is not ratified.")
    current_manifest_path = root / FREEZE_MANIFEST_RELATIVE
    if (
        not current_manifest_path.is_file()
        or current_manifest_path.is_symlink()
        or sha256_file(current_manifest_path) != manifest_sha
        or _blob_sha256(root, head, FREEZE_MANIFEST_RELATIVE) != manifest_sha
    ):
        raise ValueError(
            "Original freeze manifest differs from historical or current Git bytes."
        )
    manifest = _strict_json_bytes(
        manifest_bytes,
        context="historical freeze manifest",
    )
    require_exact_keys(
        manifest,
        FREEZE_MANIFEST_KEYS,
        context="historical freeze manifest",
    )
    if (
        manifest["schema_version"] != "run6-freeze-manifest-v1"
        or manifest["status"] != "implementation_frozen_before_held_value_access"
        or manifest["held_value_access_before_freeze"] is not False
        or manifest["source_payload_values_accessed_before_freeze"] is not False
    ):
        raise ValueError("Historical freeze manifest is invalid.")
    implementation_commit = _commit(
        root,
        manifest["implementation_commit"],
        context="historical implementation_commit",
    )
    _require_ancestor(
        root,
        implementation_commit,
        freeze_commit,
        context="Historical implementation is not before freeze manifest",
    )
    _require_pushed(
        root,
        implementation_commit,
        remote_ref,
        context="Historical implementation commit has not been pushed",
    )
    manifest_hashes = _validate_hashes(
        manifest["hashes"],
        context="historical freeze manifest hashes",
    )
    if ratified_hashes != {
        **manifest_hashes,
        FREEZE_MANIFEST_RELATIVE: manifest_sha,
    }:
        raise ValueError("Historical freeze registries are not acyclic and exact.")
    for relative, digest in manifest_hashes.items():
        if (
            _blob_sha256(root, implementation_commit, relative) != digest
            or _blob_sha256(root, freeze_commit, relative) != digest
        ):
            raise ValueError(f"Historical frozen blob changed: {relative}.")

    tracked_runtime = {
        line
        for line in _git(
            root,
            [
                "ls-tree",
                "-r",
                "--name-only",
                freeze_commit,
                "--",
                "experiments/aoc",
                "experiments/run6/scripts",
            ],
            context="Cannot enumerate historical runtime sources",
        )
        .decode("utf-8")
        .splitlines()
        if line.endswith(".py")
    }
    omitted = sorted(tracked_runtime - set(manifest_hashes))
    if omitted:
        raise ValueError(
            f"Historical freeze omitted tracked runtime sources: {omitted}."
        )
    return {
        "ratification": ratification,
        "ratification_sha256": hashlib.sha256(ratification_bytes).hexdigest(),
        "ratification_commit": ratification_commit,
        "freeze_manifest_sha256": manifest_sha,
        "freeze_commit": freeze_commit,
        "implementation_commit": implementation_commit,
        "environment": ratification["environment"],
        "thread_environment": ratification["thread_environment"],
    }


def collect_detector_evidence(
    repo_root: str | Path,
    detector_manifest_path: str | Path,
    *,
    original_ratification_sha256: str,
) -> dict[str, Any]:
    """Collect the opaque detector-byte registry without loading numeric values."""

    root = Path(repo_root).resolve()
    manifest_path = _canonical_repo_path(
        root,
        detector_manifest_path,
        DETECTOR_MANIFEST_RELATIVE,
        context="detector manifest",
    )
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise FileNotFoundError("Canonical detector manifest is missing or a symlink.")
    manifest_sha = sha256_file(manifest_path)
    if manifest_sha != EXPECTED_DETECTOR_MANIFEST_SHA256:
        raise ValueError("Detector manifest differs from the incident-record digest.")
    manifest = load_strict_json(manifest_path)
    require_exact_keys(
        manifest,
        DETECTOR_MANIFEST_KEYS,
        context="detector manifest",
    )
    if (
        manifest["schema_version"] != "run6-google-detector-freeze-v1"
        or manifest["detector_only"] is not True
        or manifest["outcome_accessed"] is not False
        or manifest["outcome_join_authorized"] is not False
        or manifest["freeze_ratification_sha256"] != original_ratification_sha256
        or manifest["git_commit"] != ORIGINAL_RATIFICATION_COMMIT
    ):
        raise ValueError(
            "Detector manifest does not preserve detector-only provenance."
        )
    replay_digests = manifest["performance"].get("held_joint_replay_digests")
    if (
        not isinstance(replay_digests, list)
        or len(replay_digests) != 3
        or any(_HEX_RE.fullmatch(item or "") is None for item in replay_digests)
        or len(set(replay_digests)) != 1
    ):
        raise ValueError("The three held joint-replay digests are not identical.")

    records = manifest["artifacts"]
    if (
        not isinstance(records, list)
        or len(records) != EXPECTED_DETECTOR_ARTIFACT_COUNT
    ):
        raise ValueError(
            "Detector manifest must declare exactly "
            f"{EXPECTED_DETECTOR_ARTIFACT_COUNT} artifacts."
        )
    artifact_registry: dict[str, dict[str, object]] = {}
    manifest_dir = manifest_path.parent.resolve()
    declared_local_paths: set[Path] = set()
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise TypeError(f"Detector artifact record {index} must be an object.")
        require_exact_keys(
            record,
            {"path", "bytes", "sha256"},
            context=f"detector artifact record {index}",
        )
        relative_from_manifest = _safe_relative(
            record["path"],
            context=f"detector artifact record {index} path",
        )
        byte_count = record["bytes"]
        if (
            isinstance(byte_count, bool)
            or not isinstance(byte_count, int)
            or byte_count < 0
        ):
            raise TypeError(f"Detector artifact record {index} bytes is invalid.")
        digest = _require_digest(
            record["sha256"],
            context=f"detector artifact record {index} sha256",
        )
        artifact = (manifest_dir / relative_from_manifest).resolve()
        try:
            artifact.relative_to(manifest_dir)
        except ValueError as exc:
            raise ValueError(
                "Detector artifact escapes its manifest directory."
            ) from exc
        if (
            not artifact.is_file()
            or artifact.is_symlink()
            or artifact.stat().st_size != byte_count
            or sha256_file(artifact) != digest
        ):
            raise ValueError(f"Detector artifact changed: {relative_from_manifest}.")
        repo_relative = artifact.relative_to(root).as_posix()
        if repo_relative in artifact_registry:
            raise ValueError(f"Duplicate detector artifact: {repo_relative}.")
        declared_local_paths.add(artifact)
        artifact_registry[repo_relative] = {
            "bytes": byte_count,
            "sha256": digest,
        }
    actual_entries = set(manifest_dir.iterdir())
    expected_entries = declared_local_paths | {manifest_path}
    if actual_entries != expected_entries or any(
        path.is_dir() for path in actual_entries
    ):
        raise ValueError("Detector result directory has unknown or missing entries.")
    return {
        "manifest_path": DETECTOR_MANIFEST_RELATIVE,
        "manifest_bytes": manifest_path.stat().st_size,
        "manifest_sha256": manifest_sha,
        "artifact_count": len(artifact_registry),
        "artifacts": artifact_registry,
        "held_joint_replay_digest_count": 3,
        "held_joint_replay_all_identical": True,
        "detector_only": True,
        "outcome_accessed": False,
        "outcome_join_authorized": False,
    }


def collect_failed_attempt_evidence(
    repo_root: str | Path,
    failed_attempt_root: str | Path,
) -> dict[str, Any]:
    """Collect the exact pre-repair failed-attempt log inventory."""

    root = Path(repo_root).resolve()
    attempt_root = _canonical_repo_path(
        root,
        failed_attempt_root,
        FAILED_ATTEMPT_ROOT_RELATIVE,
        context="failed-attempt root",
    )
    if not attempt_root.is_dir() or attempt_root.is_symlink():
        raise FileNotFoundError("Canonical failed-attempt root is missing.")
    attempts = sorted(
        path
        for path in attempt_root.iterdir()
        if path.is_dir() and _ATTEMPT_RE.fullmatch(path.name)
    )
    if len(attempts) != EXPECTED_FAILED_ATTEMPT_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_FAILED_ATTEMPT_COUNT} preserved failed attempts."
        )
    if set(attempt_root.iterdir()) != set(attempts):
        raise ValueError("Failed-attempt root contains unknown pre-repair entries.")

    ranges: list[tuple[int, int]] = []
    registry: dict[str, dict[str, object]] = {}
    empty_result_directories: list[str] = []
    stderr_digests: set[str] = set()
    for attempt in attempts:
        match = _ATTEMPT_RE.fullmatch(attempt.name)
        assert match is not None
        start, stop = int(match.group(1)), int(match.group(2))
        ranges.append((start, stop))
        entries = set(attempt.iterdir())
        result_directory = attempt / "result"
        expected_entries = {
            attempt / "stderr.log",
            attempt / "stdout.log",
            result_directory,
        }
        if entries != expected_entries or any(path.is_symlink() for path in entries):
            raise ValueError(f"Failed attempt inventory changed: {attempt.name}.")
        if not result_directory.is_dir() or any(result_directory.iterdir()):
            raise ValueError(
                f"Failed attempt result directory is not empty: {attempt.name}."
            )
        empty_result_directories.append(result_directory.relative_to(root).as_posix())
        for path in sorted(entries - {result_directory}):
            if not path.is_file():
                raise ValueError(f"Failed-attempt evidence is not a file: {path}.")
            digest = sha256_file(path)
            relative = path.relative_to(root).as_posix()
            registry[relative] = {
                "bytes": path.stat().st_size,
                "sha256": digest,
            }
            if path.name == "stderr.log":
                stderr_digests.add(digest)
            elif path.stat().st_size != 0 or digest != EMPTY_SHA256:
                raise ValueError(f"Failed attempt stdout is not empty: {attempt.name}.")
    if ranges != [(start, start + 8) for start in range(0, 256, 8)]:
        raise ValueError(
            "Failed-attempt shard ranges are not the locked 32x8 partition."
        )
    if stderr_digests != {EXPECTED_FAILED_STDERR_SHA256}:
        raise ValueError("The 32 failed stderr logs are not byte-identical evidence.")
    shard_manifests = [
        path for path in attempt_root.rglob("*.json") if "manifest" in path.name
    ]
    if shard_manifests:
        raise ValueError("A shard manifest exists despite zero completed replicates.")
    return {
        "root": FAILED_ATTEMPT_ROOT_RELATIVE,
        "attempt_count": len(attempts),
        "attempt_shard_ranges": [list(bounds) for bounds in ranges],
        "file_count": len(registry),
        "files": registry,
        "empty_result_directory_count": len(empty_result_directories),
        "empty_result_directories": empty_result_directories,
        "common_stderr_sha256": EXPECTED_FAILED_STDERR_SHA256,
        "all_stderr_logs_byte_identical": True,
        "all_stdout_logs_empty": True,
        "completed_randomization_replicates": 0,
        "shard_manifest_count": 0,
    }


def _verify_registered_failed_attempts(
    repo_root: Path,
    evidence: Mapping[str, Any],
) -> None:
    """Recheck registered failures while allowing later, separately named runs."""

    require_exact_keys(
        evidence,
        {
            "root",
            "attempt_count",
            "attempt_shard_ranges",
            "file_count",
            "files",
            "empty_result_directory_count",
            "empty_result_directories",
            "common_stderr_sha256",
            "all_stderr_logs_byte_identical",
            "all_stdout_logs_empty",
            "completed_randomization_replicates",
            "shard_manifest_count",
        },
        context="failed-attempt evidence",
    )
    if (
        evidence["root"] != FAILED_ATTEMPT_ROOT_RELATIVE
        or evidence["attempt_count"] != EXPECTED_FAILED_ATTEMPT_COUNT
        or evidence["attempt_shard_ranges"]
        != [[start, start + 8] for start in range(0, 256, 8)]
        or evidence["file_count"] != 64
        or evidence["empty_result_directory_count"] != 32
        or evidence["common_stderr_sha256"] != EXPECTED_FAILED_STDERR_SHA256
        or evidence["all_stderr_logs_byte_identical"] is not True
        or evidence["all_stdout_logs_empty"] is not True
        or evidence["completed_randomization_replicates"] != 0
        or evidence["shard_manifest_count"] != 0
    ):
        raise ValueError("Failed-attempt evidence metadata changed.")
    files = evidence["files"]
    if not isinstance(files, Mapping) or len(files) != 64:
        raise TypeError("Failed-attempt evidence must bind exactly 64 log files.")
    result_directories = evidence["empty_result_directories"]
    if (
        not isinstance(result_directories, list)
        or len(result_directories) != 32
        or len(set(result_directories)) != 32
    ):
        raise TypeError(
            "Failed-attempt evidence must bind 32 empty result directories."
        )
    expected_result_directories: set[Path] = set()
    for raw_relative in result_directories:
        relative = _safe_relative(
            raw_relative,
            context="failed-attempt empty result directory",
        )
        directory = (repo_root / relative).resolve()
        try:
            directory.relative_to(repo_root / FAILED_ATTEMPT_ROOT_RELATIVE)
        except ValueError as exc:
            raise ValueError(
                "Registered empty result directory escapes its evidence root."
            ) from exc
        if not directory.is_dir() or directory.is_symlink() or any(directory.iterdir()):
            raise ValueError(f"Registered failed result directory changed: {relative}.")
        expected_result_directories.add(directory)
    parent_dirs: set[Path] = set()
    for raw_relative, raw_record in files.items():
        relative = _safe_relative(raw_relative, context="failed log path")
        if not isinstance(raw_record, Mapping):
            raise TypeError(f"Failed log record must be an object: {relative}.")
        require_exact_keys(
            raw_record,
            {"bytes", "sha256"},
            context=f"failed log record {relative}",
        )
        path = (repo_root / relative).resolve()
        try:
            path.relative_to(repo_root / FAILED_ATTEMPT_ROOT_RELATIVE)
        except ValueError as exc:
            raise ValueError(
                "Registered failed log escapes its evidence root."
            ) from exc
        byte_count = raw_record["bytes"]
        digest = _require_digest(
            raw_record["sha256"],
            context=f"failed log digest {relative}",
        )
        if (
            isinstance(byte_count, bool)
            or not isinstance(byte_count, int)
            or not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != byte_count
            or sha256_file(path) != digest
        ):
            raise ValueError(f"Registered failed log changed: {relative}.")
        parent_dirs.add(path.parent)
        if path.name == "stderr.log" and digest != EXPECTED_FAILED_STDERR_SHA256:
            raise ValueError(f"Registered failed stderr changed: {relative}.")
        if path.name == "stdout.log" and (byte_count != 0 or digest != EMPTY_SHA256):
            raise ValueError(f"Registered failed stdout changed: {relative}.")
    if len(parent_dirs) != EXPECTED_FAILED_ATTEMPT_COUNT:
        raise ValueError("Registered failed logs do not cover exactly 32 attempts.")
    attempt_root = repo_root / FAILED_ATTEMPT_ROOT_RELATIVE
    if (
        not attempt_root.is_dir()
        or attempt_root.is_symlink()
        or set(attempt_root.iterdir()) != parent_dirs
    ):
        raise ValueError("Registered failed-attempt root inventory changed.")
    if {directory / "result" for directory in parent_dirs} != (
        expected_result_directories
    ):
        raise ValueError("Registered failed logs and result directories differ.")
    for directory in parent_dirs:
        if set(directory.iterdir()) != {
            directory / "stderr.log",
            directory / "stdout.log",
            directory / "result",
        }:
            raise ValueError(
                f"Registered failed-attempt directory changed: {directory}."
            )


def _repair_hashes(
    repo_root: Path,
    implementation_commit: str,
    diff_contract: Mapping[str, str],
) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative, status in diff_contract.items():
        if status == "D":
            raise ValueError("The repair whitelist may not delete files.")
        path = repo_root / relative
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"Required repair artifact is missing: {relative}.")
        worktree_digest = sha256_file(path)
        blob_digest = _blob_sha256(repo_root, implementation_commit, relative)
        if worktree_digest != blob_digest:
            raise ValueError(
                f"Repair worktree artifact differs from implementation: {relative}."
            )
        hashes[relative] = blob_digest
    return hashes


def _validate_access_record(value: Any) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError("Repair access record must be an object.")
    require_exact_keys(
        value,
        REPAIR_ACCESS_RECORD,
        context="repair access record",
    )
    observed = dict(value)
    if observed != REPAIR_ACCESS_RECORD:
        raise ValueError("Repair access record does not truthfully match the incident.")
    return observed


def build_repair_manifest(
    original_ratification_path: str | Path,
    *,
    repo_root: str | Path,
    implementation_commit: str,
    detector_manifest_path: str | Path | None = None,
    failed_attempt_root: str | Path | None = None,
    required_repair_paths: Iterable[str] | None = None,
    incident_commit: str | None = None,
    original_ratification_commit: str | None = None,
    remote_ref: str = "origin/main",
) -> dict[str, Any]:
    """Build, but do not write, the post-detector repair manifest."""

    root = Path(repo_root).resolve()
    incident = _commit(root, incident_commit or INCIDENT_COMMIT, context="incident")
    implementation = _commit(
        root,
        implementation_commit,
        context="repair implementation",
    )
    _require_ancestor(
        root,
        incident,
        implementation,
        context="Repair implementation is not after the incident",
    )
    _require_pushed(
        root,
        implementation,
        remote_ref,
        context="Repair implementation commit has not been pushed",
    )
    diff_contract = _repair_diff_contract(required_repair_paths)
    repair_diff = _require_exact_diff(
        root,
        incident,
        implementation,
        diff_contract,
        context="Incident-to-implementation repair diff",
    )
    original = verify_historical_original_freeze_chain(
        original_ratification_path,
        repo_root=root,
        incident_commit=incident,
        original_ratification_commit=(
            original_ratification_commit or ORIGINAL_RATIFICATION_COMMIT
        ),
        remote_ref=remote_ref,
    )
    current_environment = environment_fingerprint()
    if current_environment != original["environment"]:
        raise ValueError("Repair environment differs from the original freeze.")
    threads = original["thread_environment"]
    require_thread_environment(threads)
    verify_python_environment_lock(root / PYTHON_LOCK_RELATIVE)
    _require_clean_runtime_worktree(root)
    origins = _runtime_module_origins(root)
    detector = collect_detector_evidence(
        root,
        detector_manifest_path or root / DETECTOR_MANIFEST_RELATIVE,
        original_ratification_sha256=original["ratification_sha256"],
    )
    failures = collect_failed_attempt_evidence(
        root,
        failed_attempt_root or root / FAILED_ATTEMPT_ROOT_RELATIVE,
    )
    return {
        "schema_version": "run6-post-detector-repair-manifest-v1",
        "status": "post_detector_pre_outcome_repair_implementation_frozen",
        "incident_commit": incident,
        "original_ratification_commit": original["ratification_commit"],
        "implementation_commit": implementation,
        "repair_diff": repair_diff,
        "hashes": _repair_hashes(root, implementation, diff_contract),
        "original_freeze": {
            "ratification_path": FREEZE_RATIFICATION_RELATIVE,
            "ratification_sha256": original["ratification_sha256"],
            "ratification_commit": original["ratification_commit"],
            "freeze_manifest_sha256": original["freeze_manifest_sha256"],
            "freeze_commit": original["freeze_commit"],
            "implementation_commit": original["implementation_commit"],
        },
        "detector_evidence": detector,
        "failed_attempt_evidence": failures,
        "access_record": dict(REPAIR_ACCESS_RECORD),
        "environment": current_environment,
        "thread_environment": threads,
        "python_environment_lock_sha256": sha256_file(root / PYTHON_LOCK_RELATIVE),
        "runtime_module_origins": origins,
    }


def verify_repair_manifest_for_ratification(
    original_ratification_path: str | Path,
    repair_manifest_path: str | Path,
    *,
    repo_root: str | Path,
    manifest_commit: str = "HEAD",
    required_repair_paths: Iterable[str] | None = None,
    remote_ref: str = "origin/main",
) -> dict[str, Any]:
    """Verify a pushed repair manifest commit before ratification is written."""

    root = Path(repo_root).resolve()
    manifest_path = _canonical_repo_path(
        root,
        repair_manifest_path,
        REPAIR_MANIFEST_RELATIVE,
        context="repair manifest",
    )
    commit = _commit(root, manifest_commit, context="repair manifest commit")
    _require_pushed(
        root,
        commit,
        remote_ref,
        context="Repair manifest commit has not been pushed",
    )
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise FileNotFoundError("Canonical repair manifest is missing.")
    manifest_digest = sha256_file(manifest_path)
    if _blob_sha256(root, commit, REPAIR_MANIFEST_RELATIVE) != manifest_digest:
        raise ValueError("Repair manifest is not the manifest-commit blob.")
    manifest = load_strict_json(manifest_path)
    require_exact_keys(manifest, REPAIR_MANIFEST_KEYS, context="repair manifest")
    if (
        manifest["schema_version"] != "run6-post-detector-repair-manifest-v1"
        or manifest["status"]
        != "post_detector_pre_outcome_repair_implementation_frozen"
        or manifest["incident_commit"] != INCIDENT_COMMIT
        or manifest["original_ratification_commit"] != ORIGINAL_RATIFICATION_COMMIT
    ):
        raise ValueError("Repair manifest identity or status changed.")
    implementation = _commit(
        root,
        manifest["implementation_commit"],
        context="repair implementation commit",
    )
    _require_ancestor(
        root,
        implementation,
        commit,
        context="Repair implementation is not before its manifest",
    )
    _require_exact_diff(
        root,
        implementation,
        commit,
        {REPAIR_MANIFEST_RELATIVE: "A"},
        context="Implementation-to-manifest diff",
    )
    diff_contract = _repair_diff_contract(required_repair_paths)
    if manifest["repair_diff"] != diff_contract:
        raise ValueError("Repair manifest diff registry changed.")
    _require_exact_diff(
        root,
        INCIDENT_COMMIT,
        implementation,
        diff_contract,
        context="Incident-to-implementation repair diff",
    )
    hashes = _validate_hashes(manifest["hashes"], context="repair manifest hashes")
    if set(hashes) != set(diff_contract):
        raise ValueError("Repair manifest hash registry is not the exact whitelist.")
    for relative, digest in hashes.items():
        if (
            sha256_file(root / relative) != digest
            or _blob_sha256(root, implementation, relative) != digest
            or _blob_sha256(root, commit, relative) != digest
        ):
            raise ValueError(f"Repair artifact changed: {relative}.")

    original = verify_historical_original_freeze_chain(
        original_ratification_path,
        repo_root=root,
        incident_commit=INCIDENT_COMMIT,
        original_ratification_commit=ORIGINAL_RATIFICATION_COMMIT,
        remote_ref=remote_ref,
    )
    expected_original_freeze = {
        "ratification_path": FREEZE_RATIFICATION_RELATIVE,
        "ratification_sha256": original["ratification_sha256"],
        "ratification_commit": original["ratification_commit"],
        "freeze_manifest_sha256": original["freeze_manifest_sha256"],
        "freeze_commit": original["freeze_commit"],
        "implementation_commit": original["implementation_commit"],
    }
    if manifest["original_freeze"] != expected_original_freeze:
        raise ValueError("Repair manifest does not exactly bind the original freeze.")
    detector = collect_detector_evidence(
        root,
        root / DETECTOR_MANIFEST_RELATIVE,
        original_ratification_sha256=original["ratification_sha256"],
    )
    if manifest["detector_evidence"] != detector:
        raise ValueError("Repair manifest detector evidence changed.")
    failures = collect_failed_attempt_evidence(
        root,
        root / FAILED_ATTEMPT_ROOT_RELATIVE,
    )
    if manifest["failed_attempt_evidence"] != failures:
        raise ValueError("Repair manifest failed-attempt evidence changed.")
    _validate_access_record(manifest["access_record"])
    if (
        manifest["environment"] != original["environment"]
        or manifest["thread_environment"] != original["thread_environment"]
        or manifest["environment"] != environment_fingerprint()
    ):
        raise ValueError("Repair manifest environment changed.")
    require_thread_environment(manifest["thread_environment"])
    lock_sha = sha256_file(root / PYTHON_LOCK_RELATIVE)
    if manifest["python_environment_lock_sha256"] != lock_sha:
        raise ValueError("Repair manifest Python lock digest changed.")
    verify_python_environment_lock(root / PYTHON_LOCK_RELATIVE)
    _require_clean_runtime_worktree(root)
    if manifest["runtime_module_origins"] != _runtime_module_origins(root):
        raise ValueError("Repair manifest runtime module origins changed.")
    return {
        "manifest": manifest,
        "manifest_sha256": manifest_digest,
        "manifest_commit": commit,
        "original": original,
    }


def build_repair_ratification(
    original_ratification_path: str | Path,
    repair_manifest_path: str | Path,
    *,
    repo_root: str | Path,
    manifest_commit: str = "HEAD",
    required_repair_paths: Iterable[str] | None = None,
    remote_ref: str = "origin/main",
) -> dict[str, Any]:
    """Build, but do not write, a repair ratification."""

    verified = verify_repair_manifest_for_ratification(
        original_ratification_path,
        repair_manifest_path,
        repo_root=repo_root,
        manifest_commit=manifest_commit,
        required_repair_paths=required_repair_paths,
        remote_ref=remote_ref,
    )
    manifest = verified["manifest"]
    hashes = {
        **manifest["hashes"],
        REPAIR_MANIFEST_RELATIVE: verified["manifest_sha256"],
    }
    return {
        "schema_version": "run6-post-detector-repair-ratification-v1",
        "status": "post_detector_pre_outcome_repair_ratified",
        "repair_manifest_commit": verified["manifest_commit"],
        "hashes": hashes,
        "original_ratification_sha256": verified["original"]["ratification_sha256"],
        "detector_manifest_sha256": manifest["detector_evidence"]["manifest_sha256"],
        "access_record": dict(REPAIR_ACCESS_RECORD),
        "environment": manifest["environment"],
        "thread_environment": manifest["thread_environment"],
        "python_environment_lock_sha256": manifest["python_environment_lock_sha256"],
    }


def verify_post_detector_repair_chain(
    original_ratification_path: str | Path,
    repair_ratification_path: str | Path,
    repo_root: str | Path,
    *,
    detector_manifest_path: str | Path | None = None,
    failed_attempt_root: str | Path | None = None,
    required_repair_paths: Iterable[str] | None = None,
    expected_environment: Mapping[str, Any] | None = None,
    expected_thread_environment: Mapping[str, str] | None = None,
    remote_ref: str = "origin/main",
) -> dict[str, Any]:
    """Verify the pushed post-detector repair chain and immutable evidence.

    The return value contains only provenance, counts, hashes, access flags,
    and parsed repair records.  It never contains detector numeric values.
    """

    root = Path(repo_root).resolve()
    original_path = _canonical_repo_path(
        root,
        original_ratification_path,
        FREEZE_RATIFICATION_RELATIVE,
        context="original ratification",
    )
    ratification_path = _canonical_repo_path(
        root,
        repair_ratification_path,
        REPAIR_RATIFICATION_RELATIVE,
        context="repair ratification",
    )
    if failed_attempt_root is not None:
        _canonical_repo_path(
            root,
            failed_attempt_root,
            FAILED_ATTEMPT_ROOT_RELATIVE,
            context="failed-attempt root",
        )
    head = _commit(root, "HEAD", context="HEAD")
    _require_pushed(
        root,
        head,
        remote_ref,
        context="Current HEAD has not been pushed",
    )
    if not ratification_path.is_file() or ratification_path.is_symlink():
        raise FileNotFoundError("Canonical repair ratification is missing.")
    ratification_digest = sha256_file(ratification_path)
    if _blob_sha256(root, head, REPAIR_RATIFICATION_RELATIVE) != ratification_digest:
        raise ValueError("Repair ratification differs from the current HEAD blob.")
    ratification_commit = _path_introduction_commit(
        root,
        head,
        REPAIR_RATIFICATION_RELATIVE,
    )
    _require_pushed(
        root,
        ratification_commit,
        remote_ref,
        context="Repair ratification commit has not been pushed",
    )
    if (
        _blob_sha256(root, ratification_commit, REPAIR_RATIFICATION_RELATIVE)
        != ratification_digest
    ):
        raise ValueError("Repair ratification changed after its introduction commit.")
    ratification = load_strict_json(ratification_path)
    require_exact_keys(
        ratification,
        REPAIR_RATIFICATION_KEYS,
        context="repair ratification",
    )
    if (
        ratification["schema_version"] != "run6-post-detector-repair-ratification-v1"
        or ratification["status"] != "post_detector_pre_outcome_repair_ratified"
    ):
        raise ValueError("Repair ratification identity or status changed.")
    _validate_access_record(ratification["access_record"])
    manifest_commit = _commit(
        root,
        ratification["repair_manifest_commit"],
        context="repair_manifest_commit",
    )
    _require_ancestor(
        root,
        manifest_commit,
        ratification_commit,
        context="Repair manifest is not before repair ratification",
    )
    _require_pushed(
        root,
        manifest_commit,
        remote_ref,
        context="Repair manifest commit has not been pushed",
    )
    _require_exact_diff(
        root,
        manifest_commit,
        ratification_commit,
        {REPAIR_RATIFICATION_RELATIVE: "A"},
        context="Manifest-to-ratification diff",
    )

    manifest_path = root / REPAIR_MANIFEST_RELATIVE
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise FileNotFoundError("Canonical repair manifest is missing.")
    manifest_digest = sha256_file(manifest_path)
    if _blob_sha256(root, manifest_commit, REPAIR_MANIFEST_RELATIVE) != manifest_digest:
        raise ValueError("Repair manifest differs from its committed blob.")
    manifest = load_strict_json(manifest_path)
    require_exact_keys(manifest, REPAIR_MANIFEST_KEYS, context="repair manifest")
    if (
        manifest["schema_version"] != "run6-post-detector-repair-manifest-v1"
        or manifest["status"]
        != "post_detector_pre_outcome_repair_implementation_frozen"
        or manifest["incident_commit"] != INCIDENT_COMMIT
        or manifest["original_ratification_commit"] != ORIGINAL_RATIFICATION_COMMIT
    ):
        raise ValueError("Repair manifest identity or status changed.")
    implementation = _commit(
        root,
        manifest["implementation_commit"],
        context="repair implementation",
    )
    _require_ancestor(
        root,
        implementation,
        manifest_commit,
        context="Repair implementation is not before repair manifest",
    )
    _require_exact_diff(
        root,
        implementation,
        manifest_commit,
        {REPAIR_MANIFEST_RELATIVE: "A"},
        context="Implementation-to-manifest diff",
    )
    diff_contract = _repair_diff_contract(required_repair_paths)
    if manifest["repair_diff"] != diff_contract:
        raise ValueError("Repair diff registry changed.")
    _require_exact_diff(
        root,
        INCIDENT_COMMIT,
        implementation,
        diff_contract,
        context="Incident-to-implementation repair diff",
    )

    manifest_hashes = _validate_hashes(
        manifest["hashes"],
        context="repair manifest hashes",
    )
    ratified_hashes = _validate_hashes(
        ratification["hashes"],
        context="repair ratification hashes",
    )
    if set(manifest_hashes) != set(diff_contract) or ratified_hashes != {
        **manifest_hashes,
        REPAIR_MANIFEST_RELATIVE: manifest_digest,
    }:
        raise ValueError("Repair manifest and ratification hash registries differ.")
    for relative, digest in manifest_hashes.items():
        path = root / relative
        if (
            not path.is_file()
            or path.is_symlink()
            or sha256_file(path) != digest
            or _blob_sha256(root, implementation, relative) != digest
            or _blob_sha256(root, manifest_commit, relative) != digest
            or _blob_sha256(root, ratification_commit, relative) != digest
            or _blob_sha256(root, head, relative) != digest
        ):
            raise ValueError(f"Ratified repair artifact changed: {relative}.")
    if _blob_sha256(root, head, REPAIR_MANIFEST_RELATIVE) != manifest_digest:
        raise ValueError("Repair manifest differs from current HEAD blob.")

    original = verify_historical_original_freeze_chain(
        original_path,
        repo_root=root,
        incident_commit=INCIDENT_COMMIT,
        original_ratification_commit=ORIGINAL_RATIFICATION_COMMIT,
        remote_ref=remote_ref,
    )
    expected_original = {
        "ratification_path": FREEZE_RATIFICATION_RELATIVE,
        "ratification_sha256": original["ratification_sha256"],
        "ratification_commit": original["ratification_commit"],
        "freeze_manifest_sha256": original["freeze_manifest_sha256"],
        "freeze_commit": original["freeze_commit"],
        "implementation_commit": original["implementation_commit"],
    }
    if manifest["original_freeze"] != expected_original:
        raise ValueError("Repair manifest original-freeze binding changed.")
    if ratification["original_ratification_sha256"] != original["ratification_sha256"]:
        raise ValueError("Repair ratification original-freeze digest changed.")

    detector = collect_detector_evidence(
        root,
        detector_manifest_path or root / DETECTOR_MANIFEST_RELATIVE,
        original_ratification_sha256=original["ratification_sha256"],
    )
    if manifest["detector_evidence"] != detector:
        raise ValueError("Opaque detector evidence changed after repair freeze.")
    if ratification["detector_manifest_sha256"] != detector["manifest_sha256"]:
        raise ValueError("Repair ratification detector digest changed.")
    _verify_registered_failed_attempts(
        root,
        manifest["failed_attempt_evidence"],
    )
    _validate_access_record(manifest["access_record"])

    expected_env = (
        dict(expected_environment)
        if expected_environment is not None
        else original["environment"]
    )
    expected_threads = (
        dict(expected_thread_environment)
        if expected_thread_environment is not None
        else original["thread_environment"]
    )
    if (
        manifest["environment"] != expected_env
        or ratification["environment"] != expected_env
        or environment_fingerprint() != expected_env
        or manifest["thread_environment"] != expected_threads
        or ratification["thread_environment"] != expected_threads
    ):
        raise ValueError("Repair chain environment differs from runtime.")
    require_thread_environment(expected_threads)
    lock_sha = sha256_file(root / PYTHON_LOCK_RELATIVE)
    if (
        manifest["python_environment_lock_sha256"] != lock_sha
        or ratification["python_environment_lock_sha256"] != lock_sha
    ):
        raise ValueError("Repair chain Python environment lock changed.")
    verify_python_environment_lock(root / PYTHON_LOCK_RELATIVE)
    _require_clean_runtime_worktree(root)
    origins = _runtime_module_origins(root)
    if manifest["runtime_module_origins"] != origins:
        raise ValueError("Repair runtime module origins changed.")

    later_runtime_diff = {
        path: status
        for path, status in _diff_status(root, implementation, head).items()
        if path.startswith(("experiments/aoc/", "experiments/run6/scripts/"))
    }
    if later_runtime_diff:
        raise ValueError(
            f"Runtime Python changed after repair implementation: {later_runtime_diff}."
        )
    return {
        "repair_manifest": manifest,
        "repair_ratification": ratification,
        "repair_manifest_sha256": manifest_digest,
        "repair_ratification_sha256": ratification_digest,
        "repair_implementation_commit": implementation,
        "repair_manifest_commit": manifest_commit,
        "repair_ratification_commit": ratification_commit,
        "original_ratification_sha256": original["ratification_sha256"],
        "detector_manifest_sha256": detector["manifest_sha256"],
        "detector_artifact_count": detector["artifact_count"],
        "failed_attempt_count": manifest["failed_attempt_evidence"]["attempt_count"],
        "access_record": dict(REPAIR_ACCESS_RECORD),
        "runtime_module_origins": origins,
    }


def canonical_repair_json_bytes(value: Any) -> bytes:
    """Expose the same canonical JSON profile used by the freeze-chain creator."""

    return canonical_json_bytes(value) + b"\n"
