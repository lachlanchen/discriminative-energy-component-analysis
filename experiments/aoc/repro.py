"""Small helpers for deterministic run manifests."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_value(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_manifest(
    path: Path,
    *,
    experiment: str,
    started_at: float,
    config: dict[str, Any],
    outputs: Iterable[Path],
) -> None:
    output_paths = [Path(item) for item in outputs]
    root_value = _git_value(["rev-parse", "--show-toplevel"])
    repository_root = Path(root_value).resolve() if root_value else None

    def portable_path(item: Path) -> str:
        resolved = item.resolve()
        if repository_root is not None:
            try:
                return str(resolved.relative_to(repository_root))
            except ValueError:
                pass
        return str(resolved)

    executable = Path(sys.executable)
    command_executable = portable_path(executable)
    packages = {}
    for package in (
        "numpy",
        "scipy",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "qiskit",
        "qiskit-aer",
        "pymatching",
        "stim",
    ):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    payload = {
        "experiment": experiment,
        "command": [command_executable, *sys.argv],
        "config": config,
        "started_unix": started_at,
        "finished_unix": time.time(),
        "wall_seconds": time.time() - started_at,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "git_commit": _git_value(["rev-parse", "HEAD"]),
        "git_dirty": bool(_git_value(["status", "--porcelain"])),
        "outputs": {
            portable_path(item): sha256_file(item)
            for item in output_paths
            if item.is_file()
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
