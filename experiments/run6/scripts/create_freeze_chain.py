#!/usr/bin/env python3
"""Create the acyclic, committed Run 6 freeze manifest and ratification.

This utility reads only repository-controlled source/configuration files and
Git blobs.  It has no data-root argument and cannot inspect Run 6 payloads.
The manifest must be generated from an already pushed implementation commit;
the ratification must then be generated from the later pushed commit that
contains that manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path
from typing import Any

from aoc.run6_protocol import (
    FREEZE_MANIFEST_KEYS,
    FREEZE_MANIFEST_RELATIVE,
    FREEZE_RATIFICATION_RELATIVE,
    RUN6_REQUIRED_FREEZE_PATHS,
    canonical_json_bytes,
    environment_fingerprint,
    load_google_lock,
    load_strict_json,
    require_exact_keys,
    require_thread_environment,
    sha256_file,
)


def _git(repo_root: Path, *arguments: str) -> bytes:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repo_root,
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(detail or f"Git query failed: {arguments!r}")
    return result.stdout


def _commit(repo_root: Path, revision: str) -> str:
    return (
        _git(repo_root, "rev-parse", "--verify", f"{revision}^{{commit}}")
        .decode("ascii")
        .strip()
    )


def _require_pushed(repo_root: Path, commit: str) -> None:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "origin/main"],
        cwd=repo_root,
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        raise ValueError(f"Commit {commit} is not contained in origin/main.")


def _blob_sha256(repo_root: Path, commit: str, relative: str) -> str:
    return hashlib.sha256(_git(repo_root, "show", f"{commit}:{relative}")).hexdigest()


def _write_exclusive(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(canonical_json_bytes(value) + b"\n")


def _implementation_hashes(repo_root: Path, commit: str) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative in RUN6_REQUIRED_FREEZE_PATHS:
        path = repo_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Required freeze artifact is missing: {relative}")
        current = sha256_file(path)
        committed = _blob_sha256(repo_root, commit, relative)
        if current != committed:
            raise ValueError(
                f"Current artifact differs from implementation commit: {relative}"
            )
        hashes[relative] = committed
    return hashes


def create_manifest(repo_root: Path, revision: str) -> Path:
    output = repo_root / FREEZE_MANIFEST_RELATIVE
    if output.exists():
        raise FileExistsError(f"Refusing to replace existing {output}.")
    implementation_commit = _commit(repo_root, revision)
    if implementation_commit != _commit(repo_root, "HEAD"):
        raise ValueError("Manifest must be generated with implementation HEAD.")
    _require_pushed(repo_root, implementation_commit)
    config = load_google_lock(
        repo_root / "experiments/run6/configs/google2022_locked.json"
    )
    if config["status"] != "frozen_before_held_value_access":
        raise ValueError("Google lock must have final frozen status.")
    threads = config["numeric_policy"]["thread_environment"]
    require_thread_environment(threads)
    manifest = {
        "schema_version": "run6-freeze-manifest-v1",
        "status": "implementation_frozen_before_held_value_access",
        "implementation_commit": implementation_commit,
        "hashes": _implementation_hashes(repo_root, implementation_commit),
        "environment": environment_fingerprint(),
        "thread_environment": threads,
        "held_value_access_before_freeze": False,
        "source_payload_values_accessed_before_freeze": False,
    }
    _write_exclusive(output, manifest)
    return output


def create_ratification(repo_root: Path, revision: str) -> Path:
    output = repo_root / FREEZE_RATIFICATION_RELATIVE
    if output.exists():
        raise FileExistsError(f"Refusing to replace existing {output}.")
    freeze_commit = _commit(repo_root, revision)
    if freeze_commit != _commit(repo_root, "HEAD"):
        raise ValueError("Ratification must be generated with freeze-manifest HEAD.")
    _require_pushed(repo_root, freeze_commit)
    manifest_path = repo_root / FREEZE_MANIFEST_RELATIVE
    manifest = load_strict_json(manifest_path)
    require_exact_keys(manifest, FREEZE_MANIFEST_KEYS, context="freeze manifest")
    if (
        manifest["schema_version"] != "run6-freeze-manifest-v1"
        or manifest["status"] != "implementation_frozen_before_held_value_access"
        or manifest["held_value_access_before_freeze"] is not False
        or manifest["source_payload_values_accessed_before_freeze"] is not False
    ):
        raise ValueError("Freeze manifest is not a valid pre-access record.")
    require_thread_environment(manifest["thread_environment"])
    if manifest["environment"] != environment_fingerprint():
        raise ValueError("Current environment differs from the freeze manifest.")
    hashes = dict(manifest["hashes"])
    for relative, expected in hashes.items():
        if (
            sha256_file(repo_root / relative) != expected
            or _blob_sha256(repo_root, freeze_commit, relative) != expected
        ):
            raise ValueError(f"Freeze-commit artifact changed: {relative}")
    manifest_digest = sha256_file(manifest_path)
    if (
        _blob_sha256(
            repo_root,
            freeze_commit,
            FREEZE_MANIFEST_RELATIVE,
        )
        != manifest_digest
    ):
        raise ValueError("Freeze manifest is not the freeze-commit blob.")
    hashes[FREEZE_MANIFEST_RELATIVE] = manifest_digest
    ratification = {
        "schema_version": "run6-freeze-ratification-v1",
        "status": "frozen_before_held_value_access",
        "freeze_commit": freeze_commit,
        "hashes": hashes,
        "environment": manifest["environment"],
        "thread_environment": manifest["thread_environment"],
        "held_value_access_before_ratification": False,
    }
    _write_exclusive(output, ratification)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("manifest", "ratification"))
    parser.add_argument("--revision", default="HEAD")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[3]
    if args.stage == "manifest":
        path = create_manifest(repo_root, args.revision)
    else:
        path = create_ratification(repo_root, args.revision)
    print(path.relative_to(repo_root).as_posix())


if __name__ == "__main__":
    main()
