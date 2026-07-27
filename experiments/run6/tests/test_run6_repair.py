from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pytest
from aoc import run6_repair

ROOT = Path(__file__).resolve().parents[3]
CREATOR = ROOT / "experiments/run6/scripts/create_repair_chain.py"


def _load_creator():
    spec = importlib.util.spec_from_file_location("run6_repair_creator", CREATOR)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load repair-chain creator.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_repair_contract_is_exact_and_truthful() -> None:
    assert run6_repair.REPAIR_DIFF_STATUS == {
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
    assert run6_repair.REPAIR_ACCESS_RECORD == {
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


def test_creator_cli_has_no_payload_path_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    creator = _load_creator()
    monkeypatch.setattr(
        "sys.argv",
        ["create_repair_chain.py", "manifest", "--revision", "HEAD"],
    )
    args = creator.parse_args()
    assert vars(args) == {"stage": "manifest", "revision": "HEAD"}


def test_creator_write_is_exclusive(tmp_path: Path) -> None:
    creator = _load_creator()
    output = tmp_path / "repair.json"
    creator._write_exclusive(output, {"b": 2, "a": 1})
    assert output.read_bytes() == b'{"a":1,"b":2}\n'
    with pytest.raises(FileExistsError):
        creator._write_exclusive(output, {"a": 1})


@pytest.mark.parametrize("link_parent", [False, True])
def test_canonical_repo_path_rejects_symlink_components(
    tmp_path: Path,
    link_parent: bool,
) -> None:
    root = tmp_path.resolve()
    target_parent = root / "actual"
    target_parent.mkdir()
    target = target_parent / "record.json"
    target.write_text("{}\n", encoding="utf-8")
    if link_parent:
        (root / "canonical").symlink_to(target_parent, target_is_directory=True)
        supplied = root / "canonical/record.json"
        expected_relative = "canonical/record.json"
    else:
        supplied = root / "record.json"
        supplied.symlink_to(target)
        expected_relative = "record.json"
    with pytest.raises(ValueError, match="symbolic-link path component"):
        run6_repair._canonical_repo_path(
            root,
            supplied,
            expected_relative,
            context="synthetic record",
        )


def test_failed_attempt_evidence_is_exact_and_opaque() -> None:
    evidence = run6_repair.collect_failed_attempt_evidence(
        ROOT,
        ROOT / run6_repair.FAILED_ATTEMPT_ROOT_RELATIVE,
    )
    assert evidence["attempt_count"] == 32
    assert evidence["file_count"] == 64
    assert evidence["empty_result_directory_count"] == 32
    assert evidence["completed_randomization_replicates"] == 0
    assert evidence["shard_manifest_count"] == 0
    assert evidence["common_stderr_sha256"] == (
        run6_repair.EXPECTED_FAILED_STDERR_SHA256
    )
    assert {
        record["sha256"]
        for path, record in evidence["files"].items()
        if path.endswith("/stdout.log")
    } == {hashlib.sha256(b"").hexdigest()}


def test_detector_evidence_binds_manifest_and_all_artifacts() -> None:
    original = run6_repair.verify_historical_original_freeze_chain(
        ROOT / "experiments/run6/freeze_ratification.json",
        repo_root=ROOT,
    )
    evidence = run6_repair.collect_detector_evidence(
        ROOT,
        ROOT / run6_repair.DETECTOR_MANIFEST_RELATIVE,
        original_ratification_sha256=original["ratification_sha256"],
    )
    assert evidence["manifest_sha256"] == (
        run6_repair.EXPECTED_DETECTOR_MANIFEST_SHA256
    )
    assert evidence["artifact_count"] == 231
    assert len(evidence["artifacts"]) == 231
    assert evidence["held_joint_replay_all_identical"] is True
    assert evidence["detector_only"] is True
    assert evidence["outcome_accessed"] is False


def test_failed_attempt_registered_file_tamper_is_rejected(
    tmp_path: Path,
) -> None:
    attempt_root = tmp_path / "experiments/run6/results/google_randomization"
    files: dict[str, dict[str, object]] = {}
    empty_result_directories: list[str] = []
    for index, start in enumerate(range(0, 256, 8)):
        directory = attempt_root / f".attempt_{start:03d}_{start + 8:03d}_1_{index + 1}"
        directory.mkdir(parents=True)
        stderr = directory / "stderr.log"
        stdout = directory / "stdout.log"
        result = directory / "result"
        result.mkdir()
        empty_result_directories.append(result.relative_to(tmp_path).as_posix())
        stderr.write_bytes(b"failure")
        stdout.write_bytes(b"")
        for path in (stderr, stdout):
            files[path.relative_to(tmp_path).as_posix()] = {
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
    evidence = {
        "root": run6_repair.FAILED_ATTEMPT_ROOT_RELATIVE,
        "attempt_count": 32,
        "attempt_shard_ranges": [[start, start + 8] for start in range(0, 256, 8)],
        "file_count": 64,
        "files": files,
        "empty_result_directory_count": 32,
        "empty_result_directories": empty_result_directories,
        "common_stderr_sha256": run6_repair.EXPECTED_FAILED_STDERR_SHA256,
        "all_stderr_logs_byte_identical": True,
        "all_stdout_logs_empty": True,
        "completed_randomization_replicates": 0,
        "shard_manifest_count": 0,
    }
    with pytest.raises(ValueError, match="failed stderr changed"):
        run6_repair._verify_registered_failed_attempts(tmp_path, evidence)


def test_failed_attempt_root_cannot_gain_unregistered_entries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempt_root = tmp_path / run6_repair.FAILED_ATTEMPT_ROOT_RELATIVE
    files: dict[str, dict[str, object]] = {}
    empty_result_directories: list[str] = []
    for index, start in enumerate(range(0, 256, 8)):
        directory = attempt_root / f".attempt_{start:03d}_{start + 8:03d}_1_{index + 1}"
        directory.mkdir(parents=True)
        stderr = directory / "stderr.log"
        stdout = directory / "stdout.log"
        result = directory / "result"
        result.mkdir()
        stderr.write_bytes(b"failure")
        stdout.write_bytes(b"")
        empty_result_directories.append(result.relative_to(tmp_path).as_posix())
        for path in (stderr, stdout):
            files[path.relative_to(tmp_path).as_posix()] = {
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
    evidence = {
        "root": run6_repair.FAILED_ATTEMPT_ROOT_RELATIVE,
        "attempt_count": 32,
        "attempt_shard_ranges": [[start, start + 8] for start in range(0, 256, 8)],
        "file_count": 64,
        "files": files,
        "empty_result_directory_count": 32,
        "empty_result_directories": empty_result_directories,
        "common_stderr_sha256": hashlib.sha256(b"failure").hexdigest(),
        "all_stderr_logs_byte_identical": True,
        "all_stdout_logs_empty": True,
        "completed_randomization_replicates": 0,
        "shard_manifest_count": 0,
    }
    monkeypatch.setattr(
        run6_repair,
        "EXPECTED_FAILED_STDERR_SHA256",
        hashlib.sha256(b"failure").hexdigest(),
    )
    (attempt_root / "unregistered").mkdir()
    with pytest.raises(ValueError, match="root inventory changed"):
        run6_repair._verify_registered_failed_attempts(tmp_path, evidence)


def test_repair_paths_cannot_be_relaxed() -> None:
    with pytest.raises(ValueError, match="must equal the audited whitelist"):
        run6_repair._repair_diff_contract(
            set(run6_repair.RUN6_REQUIRED_REPAIR_PATHS)
            - {"experiments/run6/scripts/run_google2022_outcomes.py"}
        )
