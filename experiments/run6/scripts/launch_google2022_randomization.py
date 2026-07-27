#!/usr/bin/env python3
"""Launch and record the fixed 32-shard Google randomization audit."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from aoc.run6_protocol import (
    RUN6_REQUIRED_FREEZE_PATHS,
    assert_no_outcome_paths,
    canonical_json_bytes,
    environment_fingerprint,
    load_google_lock,
    require_thread_environment,
    sha256_file,
    verify_committed_freeze_chain,
)

SHARD_WIDTH = 8
SHARD_RANGES = tuple((start, start + SHARD_WIDTH) for start in range(0, 256, 8))


class ConcurrencyTracker:
    """Thread-safe count of actually running shard subprocesses."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.active = 0
        self.peak = 0

    def enter(self) -> None:
        with self._lock:
            self.active += 1
            self.peak = max(self.peak, self.active)

    def exit(self) -> None:
        with self._lock:
            self.active -= 1
            if self.active < 0:
                raise RuntimeError("Shard concurrency accounting became negative.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/run6/configs/google2022_locked.json"),
    )
    parser.add_argument(
        "--freeze-ratification",
        type=Path,
        default=Path("experiments/run6/freeze_ratification.json"),
    )
    parser.add_argument("--detector-manifest", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-workers", type=int, default=16)
    return parser.parse_args()


def _write_json(path: Path, value: Any) -> None:
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def _git_commit(repo_root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _shard_command(
    *,
    runner: Path,
    config: Path,
    ratification: Path,
    detector_manifest: Path,
    data_root: Path,
    output: Path,
    start: int,
    stop: int,
) -> list[str]:
    return [
        sys.executable,
        str(runner),
        "--config",
        str(config),
        "--freeze-ratification",
        str(ratification),
        "--detector-manifest",
        str(detector_manifest),
        "--data-root",
        str(data_root),
        "--output",
        str(output),
        "--replicate-start",
        str(start),
        "--replicate-stop",
        str(stop),
    ]


def _run_shard(
    *,
    command: list[str],
    attempt: Path,
    final: Path,
    output_root: Path,
    start: int,
    stop: int,
    env: dict[str, str],
    concurrency: ConcurrencyTracker | None = None,
) -> dict[str, Any]:
    final_manifest = final / "randomization_shard_manifest.json"
    if final_manifest.is_file():
        return {
            "replicate_range": [start, stop],
            "status": "reused_complete_shard",
            "manifest": final_manifest.relative_to(output_root).as_posix(),
            "manifest_sha256": sha256_file(final_manifest),
            "wall_seconds_this_launch": 0.0,
            "stdout_sha256": None,
            "stderr_sha256": None,
        }
    if final.exists():
        raise FileExistsError(f"Incomplete canonical shard directory exists: {final}")
    attempt.mkdir(parents=True, exist_ok=False)
    stdout_path = attempt / "stdout.log"
    stderr_path = attempt / "stderr.log"
    started = time.time()
    tracker = concurrency or ConcurrencyTracker()
    tracker.enter()
    try:
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            completed = subprocess.run(
                command,
                check=False,
                stdout=stdout,
                stderr=stderr,
                env=env,
            )
    finally:
        tracker.exit()
    finished = time.time()
    if completed.returncode != 0:
        raise RuntimeError(
            f"Shard [{start},{stop}) failed with code {completed.returncode}; "
            f"preserved at {attempt}."
        )
    child_output = attempt / "result"
    attempt_manifest = child_output / "randomization_shard_manifest.json"
    if not attempt_manifest.is_file():
        raise FileNotFoundError(f"Successful shard omitted its manifest: {attempt}")
    stdout_path.replace(child_output / "stdout.log")
    stderr_path.replace(child_output / "stderr.log")
    child_output.replace(final)
    attempt.rmdir()
    final_manifest = final / "randomization_shard_manifest.json"
    return {
        "replicate_range": [start, stop],
        "status": "executed_this_launch",
        "manifest": final_manifest.relative_to(output_root).as_posix(),
        "manifest_sha256": sha256_file(final_manifest),
        "wall_seconds_this_launch": finished - started,
        "stdout_sha256": sha256_file(final / "stdout.log"),
        "stderr_sha256": sha256_file(final / "stderr.log"),
    }


def run_real(args: argparse.Namespace) -> None:
    if args.detector_manifest is None or args.data_root is None or args.output is None:
        raise ValueError(
            "Real launch requires detector manifest, data root, and output."
        )
    if isinstance(args.max_workers, bool) or not 1 <= args.max_workers <= 16:
        raise ValueError("--max-workers must be an integer in [1,16].")
    repo_root = Path(__file__).resolve().parents[3]
    config_path = args.config.resolve()
    ratification_path = args.freeze_ratification.resolve()
    detector_manifest = args.detector_manifest.resolve()
    data_root = args.data_root.resolve()
    output = args.output.resolve()
    expected_config = (
        repo_root / "experiments/run6/configs/google2022_locked.json"
    ).resolve()
    expected_ratification = (
        repo_root / "experiments/run6/freeze_ratification.json"
    ).resolve()
    if config_path != expected_config or ratification_path != expected_ratification:
        raise ValueError("Launcher requires canonical config and ratification paths.")
    assert_no_outcome_paths(
        [config_path, ratification_path, detector_manifest, data_root, output]
    )
    config = load_google_lock(config_path)
    threads = config["numeric_policy"]["thread_environment"]
    require_thread_environment(threads)
    verify_committed_freeze_chain(
        ratification_path,
        repo_root=repo_root,
        required_paths=RUN6_REQUIRED_FREEZE_PATHS,
        expected_environment=environment_fingerprint(),
        expected_thread_environment=threads,
    )
    output.mkdir(parents=True, exist_ok=True)
    orchestration_path = output / "orchestration_manifest.json"
    merge_root = output / "merged"
    if orchestration_path.exists() or merge_root.exists():
        raise FileExistsError("Final orchestration/merge output already exists.")

    runner = repo_root / "experiments/run6/scripts/run_google2022_randomization.py"
    env = os.environ.copy()
    for key, value in threads.items():
        env[key] = value
    launch_started = time.time()
    concurrency = ConcurrencyTracker()
    futures: list[concurrent.futures.Future[dict[str, Any]]] = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.max_workers,
    ) as executor:
        for start, stop in SHARD_RANGES:
            final = output / f"shard_{start:03d}_{stop:03d}"
            attempt = output / (
                f".attempt_{start:03d}_{stop:03d}_{os.getpid()}_{time.time_ns()}"
            )
            command = _shard_command(
                runner=runner,
                config=config_path,
                ratification=ratification_path,
                detector_manifest=detector_manifest,
                data_root=data_root,
                output=attempt / "result",
                start=start,
                stop=stop,
            )
            futures.append(
                executor.submit(
                    _run_shard,
                    command=command,
                    attempt=attempt,
                    final=final,
                    output_root=output,
                    start=start,
                    stop=stop,
                    env=env,
                    concurrency=concurrency,
                )
            )
        shard_rows = [future.result() for future in futures]

    shard_rows.sort(key=lambda row: row["replicate_range"])
    shard_manifests = [output / row["manifest"] for row in shard_rows]
    merge_command = [
        sys.executable,
        str(runner),
        "--config",
        str(config_path),
        "--freeze-ratification",
        str(ratification_path),
        "--detector-manifest",
        str(detector_manifest),
        "--output",
        str(merge_root),
    ]
    for manifest in shard_manifests:
        merge_command.extend(["--merge-shard-manifest", str(manifest)])
    merge_stdout = output / "merge_stdout.log"
    merge_stderr = output / "merge_stderr.log"
    with merge_stdout.open("wb") as stdout, merge_stderr.open("wb") as stderr:
        completed = subprocess.run(
            merge_command,
            check=False,
            stdout=stdout,
            stderr=stderr,
            env=env,
        )
    if completed.returncode != 0:
        raise RuntimeError(f"Randomization merge failed; see {merge_stderr}.")
    merge_manifest = merge_root / "randomization_manifest.json"
    if not merge_manifest.is_file():
        raise FileNotFoundError("Randomization merge omitted its final manifest.")
    launch_finished = time.time()
    orchestration = {
        "schema_version": "run6-google-randomization-orchestration-v1",
        "protocol_id": config["protocol_id"],
        "git_commit": _git_commit(repo_root),
        "config_sha256": sha256_file(config_path),
        "freeze_ratification_sha256": sha256_file(ratification_path),
        "detector_manifest_sha256": sha256_file(detector_manifest),
        "runner_sha256": sha256_file(runner),
        "launcher_sha256": sha256_file(__file__),
        "replicate_ranges": [list(bounds) for bounds in SHARD_RANGES],
        "shard_width": SHARD_WIDTH,
        "configured_max_concurrent_worker_processes": args.max_workers,
        "observed_peak_concurrent_worker_processes": concurrency.peak,
        "executed_shard_count": sum(
            row["status"] == "executed_this_launch" for row in shard_rows
        ),
        "reused_shard_count": sum(
            row["status"] == "reused_complete_shard" for row in shard_rows
        ),
        "worker_process_count_per_shard": 1,
        "numeric_threads_per_worker": 1,
        "external_concurrency_measured_not_inferred": True,
        "started_unix": launch_started,
        "finished_unix": launch_finished,
        "wall_seconds": launch_finished - launch_started,
        "shards": shard_rows,
        "merge_manifest": merge_manifest.relative_to(output).as_posix(),
        "merge_manifest_sha256": sha256_file(merge_manifest),
        "merge_stdout_sha256": sha256_file(merge_stdout),
        "merge_stderr_sha256": sha256_file(merge_stderr),
        "outcome_accessed": False,
        "environment": environment_fingerprint(),
    }
    _write_json(orchestration_path, orchestration)
    print(json.dumps(orchestration, indent=2, sort_keys=True))


def main() -> None:
    args = parse_args()
    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "synthetic_launch_plan_only",
                    "replicate_ranges": [list(bounds) for bounds in SHARD_RANGES],
                    "default_max_workers": 16,
                    "raw_run6_values_opened": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    run_real(args)


if __name__ == "__main__":
    main()
