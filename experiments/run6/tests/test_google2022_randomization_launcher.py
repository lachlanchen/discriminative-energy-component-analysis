"""Value-free tests for the deterministic Run 6 randomization launcher."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from aoc.run6_protocol import sha256_file

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "experiments/run6/scripts/launch_google2022_randomization.py"


def _load_launcher():
    spec = importlib.util.spec_from_file_location("run6_randomization_launcher", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load randomization launcher.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_fixed_shards_are_gap_free_and_exactly_once() -> None:
    launcher = _load_launcher()
    assert len(launcher.SHARD_RANGES) == 32
    assert all(stop - start == 8 for start, stop in launcher.SHARD_RANGES)
    flattened = [
        replicate
        for start, stop in launcher.SHARD_RANGES
        for replicate in range(start, stop)
    ]
    assert flattened == list(range(256))


def test_dry_run_reports_value_blind_plan() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--dry-run"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)
    assert result["raw_run6_values_opened"] is False
    assert result["default_max_workers"] == 16
    assert len(result["replicate_ranges"]) == 32


def test_complete_shard_is_reused_without_starting_a_process(
    tmp_path: Path,
) -> None:
    launcher = _load_launcher()
    root = tmp_path / "audit"
    final = root / "shard_000_008"
    final.mkdir(parents=True)
    manifest = final / "randomization_shard_manifest.json"
    manifest.write_text('{"synthetic":true}\n', encoding="utf-8")
    row = launcher._run_shard(
        command=["this-command-must-not-run"],
        attempt=root / ".attempt",
        final=final,
        output_root=root,
        start=0,
        stop=8,
        env={},
    )
    assert row["status"] == "reused_complete_shard"
    assert row["manifest"] == "shard_000_008/randomization_shard_manifest.json"
    assert row["manifest_sha256"] == sha256_file(manifest)


def test_fresh_shard_receives_an_empty_child_output_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launcher = _load_launcher()
    root = tmp_path / "audit"
    attempt = root / ".attempt"
    final = root / "shard_000_008"
    command = [
        "synthetic-runner",
        "--output",
        str(attempt / "result"),
    ]

    def fake_run(arguments: list[str], **_: object) -> SimpleNamespace:
        child_output = Path(arguments[arguments.index("--output") + 1])
        assert not child_output.exists()
        child_output.mkdir(parents=True)
        (child_output / "randomization_shard_manifest.json").write_text(
            '{"synthetic":true}\n',
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(launcher.subprocess, "run", fake_run)
    concurrency = launcher.ConcurrencyTracker()
    row = launcher._run_shard(
        command=command,
        attempt=attempt,
        final=final,
        output_root=root,
        start=0,
        stop=8,
        env={},
        concurrency=concurrency,
    )
    assert row["status"] == "executed_this_launch"
    assert not attempt.exists()
    assert (final / "stdout.log").is_file()
    assert (final / "stderr.log").is_file()
    assert (final / "randomization_shard_manifest.json").is_file()
    assert concurrency.active == 0
    assert concurrency.peak == 1
