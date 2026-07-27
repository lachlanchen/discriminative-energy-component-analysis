#!/usr/bin/env python3
"""Create the acyclic Run 6 post-detector repair provenance chain.

The command has no raw-data, decoder-outcome, or PNNL payload arguments.
``manifest`` must run from the pushed repair implementation commit.
``ratification`` must run from the later pushed manifest commit.
Both writes are exclusive and refuse to replace prior records.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from aoc.run6_protocol import FREEZE_RATIFICATION_RELATIVE
from aoc.run6_repair import (
    REPAIR_MANIFEST_RELATIVE,
    REPAIR_RATIFICATION_RELATIVE,
    build_repair_manifest,
    build_repair_ratification,
    canonical_repair_json_bytes,
)


def _git_commit(repo_root: Path, revision: str) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--verify", f"{revision}^{{commit}}"],
        cwd=repo_root,
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(detail or f"{revision!r} is not a Git commit.")
    return result.stdout.decode("ascii").strip()


def _write_exclusive(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(canonical_repair_json_bytes(payload))


def create_manifest(repo_root: Path, revision: str = "HEAD") -> Path:
    output = repo_root / REPAIR_MANIFEST_RELATIVE
    if output.exists():
        raise FileExistsError(f"Refusing to replace existing {output}.")
    implementation = _git_commit(repo_root, revision)
    if implementation != _git_commit(repo_root, "HEAD"):
        raise ValueError("Repair manifest must be generated with implementation HEAD.")
    payload = build_repair_manifest(
        repo_root / FREEZE_RATIFICATION_RELATIVE,
        repo_root=repo_root,
        implementation_commit=implementation,
    )
    _write_exclusive(output, payload)
    return output


def create_ratification(repo_root: Path, revision: str = "HEAD") -> Path:
    output = repo_root / REPAIR_RATIFICATION_RELATIVE
    if output.exists():
        raise FileExistsError(f"Refusing to replace existing {output}.")
    manifest_commit = _git_commit(repo_root, revision)
    if manifest_commit != _git_commit(repo_root, "HEAD"):
        raise ValueError(
            "Repair ratification must be generated with repair-manifest HEAD."
        )
    payload = build_repair_ratification(
        repo_root / FREEZE_RATIFICATION_RELATIVE,
        repo_root / REPAIR_MANIFEST_RELATIVE,
        repo_root=repo_root,
        manifest_commit=manifest_commit,
    )
    _write_exclusive(output, payload)
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
