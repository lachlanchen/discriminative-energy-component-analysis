"""Strict-schema and outcome-embargo tests for the Run 6 lock."""

from __future__ import annotations

import copy
import hashlib
import subprocess
from pathlib import Path

import pytest
from aoc import run6_protocol
from aoc.run6_protocol import (
    assert_no_outcome_paths,
    canonical_json_bytes,
    canonical_json_sha256,
    load_google_lock,
    load_strict_json,
    require_thread_environment,
    sha256_file,
    validate_google_lock,
    verify_committed_freeze_chain,
    verify_python_environment_lock,
    verify_runtime_module_origins,
)

ROOT = Path(__file__).resolve().parents[3]
GOOGLE_LOCK = ROOT / "experiments/run6/configs/google2022_locked.json"


def test_repository_google_lock_passes_strict_validation() -> None:
    config = load_google_lock(GOOGLE_LOCK)
    assert config["protocol_id"] == "run6-google2022-v2"


def test_unknown_or_missing_top_level_fields_are_rejected() -> None:
    config = load_google_lock(GOOGLE_LOCK)
    unknown = copy.deepcopy(config)
    unknown["silent_default"] = 1
    with pytest.raises(ValueError, match="unknown"):
        validate_google_lock(unknown)

    missing = copy.deepcopy(config)
    del missing["decision"]
    with pytest.raises(ValueError, match="missing"):
        validate_google_lock(missing)


def test_unknown_or_missing_nested_fields_are_rejected() -> None:
    config = load_google_lock(GOOGLE_LOCK)
    unknown = copy.deepcopy(config)
    unknown["accumulators"]["primary"]["silent_default"] = 1
    with pytest.raises(ValueError, match="unknown"):
        validate_google_lock(unknown)

    missing = copy.deepcopy(config)
    del missing["methods"]["space_sparse"]["tie_rule"]
    with pytest.raises(ValueError, match="missing"):
        validate_google_lock(missing)


def test_unknown_method_and_changed_critical_choice_are_rejected() -> None:
    config = load_google_lock(GOOGLE_LOCK)
    extra_method = copy.deepcopy(config)
    extra_method["methods"]["post_hoc_oracle"] = {}
    with pytest.raises(ValueError, match="methods schema"):
        validate_google_lock(extra_method)

    changed = copy.deepcopy(config)
    changed["methods"]["space_composite"]["fixed_branch_prior"] = {
        "space_sparse": 1.0,
        "space_spectral": 0.0,
    }
    with pytest.raises(ValueError, match="composite"):
        validate_google_lock(changed)


def test_shared_shot_cannot_be_relabelled_as_51_formal_time_steps() -> None:
    config = copy.deepcopy(load_google_lock(GOOGLE_LOCK))
    config["accumulators"]["primary"]["horizon_shots"] = 1_020_000
    with pytest.raises(ValueError, match="complete-shot accumulator"):
        validate_google_lock(config)


def test_duplicate_and_nonfinite_json_are_rejected(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"x":1,"x":2}', encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        load_strict_json(duplicate)

    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"x":NaN}', encoding="utf-8")
    with pytest.raises(ValueError, match="Non-finite"):
        load_strict_json(nonfinite)


def test_canonical_json_and_raw_file_hash_are_deterministic(tmp_path: Path) -> None:
    left = {"b": [2, 3], "a": "量子"}
    right = {"a": "量子", "b": [2, 3]}
    assert canonical_json_bytes(left) == canonical_json_bytes(right)
    assert canonical_json_sha256(left) == canonical_json_sha256(right)

    path = tmp_path / "artifact.bin"
    path.write_bytes(b"run6\x00artifact")
    assert sha256_file(path) == sha256_file(path)


def test_detector_command_rejects_every_outcome_path() -> None:
    assert_no_outcome_paths(["circuit_ideal.stim", "detection_events.b8"])
    for filename in (
        "obs_flips_actual.01",
        "obs_flips_predicted_by_correlated_matching.01",
        "obs_flips_predicted_by_pymatching.01",
        "some_other_label.01",
    ):
        with pytest.raises(PermissionError, match="cannot receive"):
            assert_no_outcome_paths([filename])


def test_full_python_environment_lock_is_exact(tmp_path: Path) -> None:
    lock = tmp_path / "environment.txt"
    lock.write_text(
        "# exact pins\nFoo_Bar==1.0\nbaz==2.0\n",
        encoding="utf-8",
    )
    installed = {
        "foo-bar": "1.0",
        "Baz": "2.0",
        "deca_experiments": "6.0.0",
    }
    assert verify_python_environment_lock(
        lock,
        installed_versions=installed,
    ) == {"foo-bar": "1.0", "baz": "2.0"}

    changed = dict(installed)
    changed["Baz"] = "2.1"
    with pytest.raises(ValueError, match="mismatched"):
        verify_python_environment_lock(lock, installed_versions=changed)

    missing = dict(installed)
    del missing["Baz"]
    with pytest.raises(ValueError, match="missing"):
        verify_python_environment_lock(lock, installed_versions=missing)

    unexpected = {**installed, "surprise": "1"}
    with pytest.raises(ValueError, match="unexpected"):
        verify_python_environment_lock(lock, installed_versions=unexpected)

    lock.write_text("foo_bar==1\nfoo-bar==1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate normalized"):
        verify_python_environment_lock(lock, installed_versions=installed)


def test_runtime_modules_resolve_to_repository_sources() -> None:
    observed = verify_runtime_module_origins(ROOT)
    assert observed["aoc.run6_protocol"] == "experiments/aoc/run6_protocol.py"
    assert observed["aoc.space_qec"] == "experiments/aoc/space_qec.py"


def test_thread_environment_is_checked_not_mutated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RUN6_THREADS", "1")
    require_thread_environment({"RUN6_THREADS": "1"})
    monkeypatch.setenv("RUN6_THREADS", "8")
    with pytest.raises(RuntimeError, match="mismatch"):
        require_thread_environment({"RUN6_THREADS": "1"})


def _git(repo: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def test_freeze_chain_binds_pushed_committed_blobs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        run6_protocol,
        "verify_python_environment_lock",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(
        run6_protocol,
        "verify_runtime_module_origins",
        lambda *args, **kwargs: {},
    )
    repo = tmp_path / "repo"
    remote = tmp_path / "remote.git"
    repo.mkdir()
    subprocess.run(["git", "init", "--bare", remote], check=True, capture_output=True)
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.email", "run6@example.invalid")
    _git(repo, "config", "user.name", "Run 6 test")
    _git(repo, "remote", "add", "origin", str(remote))

    code_path = repo / "frozen" / "code.txt"
    code_path.parent.mkdir()
    code_path.write_bytes(b"exact implementation\n")
    _git(repo, "add", "frozen/code.txt")
    _git(repo, "commit", "-m", "implementation")
    implementation_commit = _git(repo, "rev-parse", "HEAD")

    environment = {"python": "synthetic"}
    threads = {"RUN6_TEST_THREADS": "1"}
    code_digest = hashlib.sha256(code_path.read_bytes()).hexdigest()
    manifest_path = repo / "experiments/run6/freeze_manifest.json"
    _write_json(
        manifest_path,
        {
            "schema_version": "run6-freeze-manifest-v1",
            "status": "implementation_frozen_before_held_value_access",
            "implementation_commit": implementation_commit,
            "hashes": {"frozen/code.txt": code_digest},
            "environment": environment,
            "thread_environment": threads,
            "held_value_access_before_freeze": False,
            "source_payload_values_accessed_before_freeze": False,
        },
    )
    _git(repo, "add", "experiments/run6/freeze_manifest.json")
    _git(repo, "commit", "-m", "freeze manifest")
    freeze_commit = _git(repo, "rev-parse", "HEAD")
    manifest_digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

    ratification_path = repo / "experiments/run6/freeze_ratification.json"
    _write_json(
        ratification_path,
        {
            "schema_version": "run6-freeze-ratification-v1",
            "status": "frozen_before_held_value_access",
            "freeze_commit": freeze_commit,
            "hashes": {
                "frozen/code.txt": code_digest,
                "experiments/run6/freeze_manifest.json": manifest_digest,
            },
            "environment": environment,
            "thread_environment": threads,
            "held_value_access_before_ratification": False,
        },
    )
    _git(repo, "add", "experiments/run6/freeze_ratification.json")
    _git(repo, "commit", "-m", "ratify freeze")
    _git(repo, "push", "-u", "origin", "main")
    monkeypatch.setenv("RUN6_TEST_THREADS", "1")

    observed = verify_committed_freeze_chain(
        ratification_path,
        repo_root=repo,
        required_paths={"frozen/code.txt"},
        expected_environment=environment,
        expected_thread_environment=threads,
    )
    assert observed["freeze_commit"] == freeze_commit

    code_path.write_bytes(b"uncommitted substitution\n")
    with pytest.raises(ValueError, match="worktree artifact changed"):
        verify_committed_freeze_chain(
            ratification_path,
            repo_root=repo,
            required_paths={"frozen/code.txt"},
            expected_environment=environment,
            expected_thread_environment=threads,
        )
    code_path.write_bytes(b"exact implementation\n")

    injected = repo / "experiments/aoc/injected.py"
    injected.parent.mkdir(parents=True)
    injected.write_text("raise RuntimeError('not ratified')\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Untracked Python"):
        verify_committed_freeze_chain(
            ratification_path,
            repo_root=repo,
            required_paths={"frozen/code.txt"},
            expected_environment=environment,
            expected_thread_environment=threads,
        )
    injected.unlink()

    unrelated = repo / "unpublished.txt"
    unrelated.write_bytes(b"not pushed\n")
    _git(repo, "add", "unpublished.txt")
    _git(repo, "commit", "-m", "unpublished")
    with pytest.raises(ValueError, match="has not been pushed"):
        verify_committed_freeze_chain(
            ratification_path,
            repo_root=repo,
            required_paths={"frozen/code.txt"},
            expected_environment=environment,
            expected_thread_environment=threads,
        )
