"""Metadata-only tests for the ratified Pittsburgh lock validator.

Synthetic ``bitstrings.json`` files contain sentinels, not measurements.
Every validation call is guarded so that attempting to open one raises an
``AssertionError``.  The real-metadata test applies the same guard.
"""

from __future__ import annotations

import json
import re
import sys
from hashlib import sha256
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN6_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = RUN6_ROOT / "scripts"
TEMPLATE_LOCK = RUN6_ROOT / "configs" / "pnnl_pittsburgh_locked.json"
sys.path.insert(0, str(SCRIPT_ROOT))

from validate_pnnl_lock import (
    LockValidationError,
    validate_lock,
)


def _digest(raw: bytes) -> str:
    return sha256(raw).hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _normalized_qasm(raw: bytes) -> bytes:
    text = raw.decode("utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+$", "", line) for line in text.split("\n")]
    while lines and lines[-1] == "":
        lines.pop()
    return ("\n".join(lines) + "\n").encode("utf-8")


def _pair_digest(state0: str, state1: str) -> str:
    return _digest(
        _canonical(
            {
                "state0_sha256": state0,
                "state1_sha256": state1,
            }
        )
    )


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def _write_json_record(path: Path, value: Any) -> list[Any]:
    raw = _json_bytes(value)
    path.write_bytes(raw)
    return [len(raw), _digest(raw), _digest(_canonical(value))]


def _write_qasm_record(path: Path, raw: bytes) -> list[Any]:
    path.write_bytes(raw)
    return [len(raw), _digest(raw), _digest(_normalized_qasm(raw))]


def _qasm_for_snapshot(
    snapshot_id: str,
    *,
    state: int,
    distance: int,
    rounds: int,
    registers: dict[str, tuple[tuple[int, ...], tuple[int, ...]]],
) -> bytes:
    lines = ["OPENQASM 3.0;", f"// synthetic {snapshot_id} state{state}"]
    for suffix in sorted(registers):
        lines.append(f"bit[{distance}] c_data_{suffix};")
        lines.append(f"bit[{rounds * (distance - 1)}] c_syndrome_{suffix};")
    for suffix in sorted(registers):
        data, checks = registers[suffix]
        for round_index in range(rounds):
            for check_index, qubit in enumerate(checks):
                flat_index = round_index * (distance - 1) + check_index
                lines.append(f"c_syndrome_{suffix}[{flat_index}] = measure ${qubit};")
        for index, qubit in enumerate(data):
            lines.append(f"c_data_{suffix}[{index}] = measure ${qubit};")
    return ("\n".join(lines) + "\n").encode("utf-8")


def _guard_held_payload_open(monkeypatch: pytest.MonkeyPatch) -> None:
    original_open = Path.open

    def guarded_open(path: Path, *args: Any, **kwargs: Any) -> Any:
        if path.name == "bitstrings.json":
            raise AssertionError("validator attempted to open held bitstrings")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)


@pytest.fixture
def synthetic_lock(tmp_path: Path) -> tuple[Path, Path, dict[str, Any]]:
    manifest = json.loads(TEMPLATE_LOCK.read_text(encoding="utf-8"))
    for relative in manifest["parent_artifacts"]:
        source = REPO_ROOT / relative
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    artifact_root = (
        tmp_path / "experiments" / "data" / "run6" / "pnnl_ibm" / "ibm_pittsburgh"
    )
    lock_path = (
        tmp_path / "experiments" / "run6" / "configs" / "pnnl_pittsburgh_locked.json"
    )
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    schema = manifest["cohort_row_schema"]
    register_maps: dict[
        str,
        dict[str, tuple[tuple[int, ...], tuple[int, ...]]],
    ] = {snapshot_id: {} for snapshot_id in manifest["snapshots"]}
    for raw_row in manifest["cohort_pairs"]:
        row = dict(zip(schema, raw_row))
        mapping = (
            tuple(row["data_qubits"]),
            tuple(row["syndrome_qubits"]),
        )
        for snapshot_id in (
            row["early_snapshot_id"],
            row["late_snapshot_id"],
        ):
            register_maps[snapshot_id][row["register_suffix"]] = mapping

    calibration_by_original_hash: dict[str, dict[str, Any]] = {}
    for snapshot_id, snapshot in manifest["snapshots"].items():
        job_dir = artifact_root / snapshot["relative_job_dir"]
        job_dir.mkdir(parents=True)
        distance, rounds, basis, shots, n_chains, date_source, _ = snapshot["metadata"]
        info = {
            "backend": "ibm_pittsburgh",
            "d": distance,
            "rounds": rounds,
            "basis": basis,
            "logical_states": [0, 1],
            "shots": shots,
            "n_chains": n_chains,
        }
        snapshot["info"] = _write_json_record(job_dir / "info.json", info)

        original_calibration_hash = snapshot["calibration"][2]
        calibration = calibration_by_original_hash.setdefault(
            original_calibration_hash,
            {
                "backend_name": "ibm_pittsburgh",
                "backend_version": f"synthetic-{original_calibration_hash[:12]}",
                "last_update_date": date_source,
                "qubits": [],
                "gates": [],
                "general": [],
                "general_qlists": [],
                "coupling_map": "[]",
            },
        )
        assert calibration["last_update_date"] == date_source
        snapshot["calibration"] = _write_json_record(
            job_dir / "calibration.json",
            calibration,
        )

        qasm_records: list[list[Any]] = []
        for state in (0, 1):
            qasm = _qasm_for_snapshot(
                snapshot_id,
                state=state,
                distance=distance,
                rounds=rounds,
                registers=register_maps[snapshot_id],
            )
            qasm_records.append(
                _write_qasm_record(job_dir / f"circuit_state{state}.qasm", qasm)
            )
        snapshot["qasm_state0"], snapshot["qasm_state1"] = qasm_records
        snapshot["qasm_pair"] = [
            _pair_digest(qasm_records[0][1], qasm_records[1][1]),
            _pair_digest(qasm_records[0][2], qasm_records[1][2]),
        ]

        held_sentinel = f"DO NOT OPEN {snapshot_id}\n".encode("ascii")
        (job_dir / "bitstrings.json").write_bytes(held_sentinel)
        snapshot["held_bitstrings"] = [len(held_sentinel), False, None]

    for raw_row in manifest["cohort_pairs"]:
        row = dict(zip(schema, raw_row))
        early = manifest["snapshots"][row["early_snapshot_id"]]
        late = manifest["snapshots"][row["late_snapshot_id"]]
        raw_row[schema.index("calibration_pair_id")] = (
            f"{early['calibration'][2]}--{late['calibration'][2]}"
        )
        raw_row[schema.index("raw_qasm_pair_identical")] = (
            early["qasm_pair"][0] == late["qasm_pair"][0]
        )
        raw_row[schema.index("normalized_qasm_pair_identical_audit_only")] = (
            early["qasm_pair"][1] == late["qasm_pair"][1]
        )

    lock_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return tmp_path, lock_path, manifest


def _rewrite_lock(lock_path: Path, manifest: dict[str, Any]) -> None:
    lock_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def test_synthetic_lock_validates_and_never_opens_payload(
    synthetic_lock: tuple[Path, Path, dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root, lock_path, _ = synthetic_lock
    _guard_held_payload_open(monkeypatch)
    report = validate_lock(lock_path, repo_root=repo_root)
    assert report.snapshots == 20
    assert report.cohorts == 11
    assert report.held_payloads_statted == 20
    assert report.paired_shots_per_phase == 16_370
    assert report.paired_cycle_updates_per_phase == 49_110
    assert report.threshold_seed_count == 110


def test_pending_and_frozen_status_modes_are_explicit(
    synthetic_lock: tuple[Path, Path, dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root, lock_path, manifest = synthetic_lock
    _guard_held_payload_open(monkeypatch)
    validate_lock(lock_path, repo_root=repo_root, mode="frozen")

    manifest["status"] = "ratified_metadata_lock_pending_final_freeze_commit"
    _rewrite_lock(lock_path, manifest)
    validate_lock(lock_path, repo_root=repo_root, mode="pending")
    with pytest.raises(LockValidationError, match="lock.status"):
        validate_lock(lock_path, repo_root=repo_root, mode="frozen")

    manifest["status"] = "unexpected_third_status"
    _rewrite_lock(lock_path, manifest)
    with pytest.raises(LockValidationError, match="lock.status"):
        validate_lock(lock_path, repo_root=repo_root, mode="frozen")
    with pytest.raises(ValueError, match="pending.*frozen"):
        validate_lock(lock_path, repo_root=repo_root, mode="other")


def test_real_locked_manifest_validates_metadata_without_opening_bitstrings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_root = (
        REPO_ROOT / "experiments" / "data" / "run6" / "pnnl_ibm" / "ibm_pittsburgh"
    )
    if not artifact_root.is_dir():
        pytest.skip("local PNNL Pittsburgh metadata artifacts are unavailable")
    _guard_held_payload_open(monkeypatch)
    report = validate_lock(TEMPLATE_LOCK, repo_root=REPO_ROOT)
    assert report.snapshots == 20
    assert report.cohorts == 11
    assert report.held_payloads_statted == 20


def test_parent_artifact_byte_drift_is_rejected(
    synthetic_lock: tuple[Path, Path, dict[str, Any]],
) -> None:
    repo_root, lock_path, manifest = synthetic_lock
    relative = next(iter(manifest["parent_artifacts"]))
    parent = repo_root / relative
    parent.write_bytes(parent.read_bytes() + b"\npost-lock drift\n")
    with pytest.raises(LockValidationError, match="parent_artifacts.*SHA-256 mismatch"):
        validate_lock(lock_path, repo_root=repo_root)


@pytest.mark.parametrize("location", ["top", "snapshot"])
def test_unknown_lock_fields_are_rejected(
    synthetic_lock: tuple[Path, Path, dict[str, Any]],
    location: str,
) -> None:
    repo_root, lock_path, manifest = synthetic_lock
    if location == "top":
        manifest["unexpected"] = True
    else:
        first = next(iter(manifest["snapshots"].values()))
        first["unexpected"] = True
    _rewrite_lock(lock_path, manifest)
    with pytest.raises(LockValidationError, match="unknown fields"):
        validate_lock(lock_path, repo_root=repo_root)


def test_unknown_info_metadata_field_is_rejected_after_valid_rehash(
    synthetic_lock: tuple[Path, Path, dict[str, Any]],
) -> None:
    repo_root, lock_path, manifest = synthetic_lock
    first_snapshot = next(iter(manifest["snapshots"].values()))
    info_path = (
        repo_root
        / manifest["source"]["artifact_root"]
        / first_snapshot["relative_job_dir"]
        / "info.json"
    )
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["unexpected"] = "metadata"
    first_snapshot["info"] = _write_json_record(info_path, info)
    _rewrite_lock(lock_path, manifest)
    with pytest.raises(LockValidationError, match="unknown fields"):
        validate_lock(lock_path, repo_root=repo_root)


def test_raw_metadata_hash_mismatch_is_rejected(
    synthetic_lock: tuple[Path, Path, dict[str, Any]],
) -> None:
    repo_root, lock_path, manifest = synthetic_lock
    first_snapshot = next(iter(manifest["snapshots"].values()))
    first_snapshot["info"][1] = "0" * 64
    _rewrite_lock(lock_path, manifest)
    with pytest.raises(LockValidationError, match="raw SHA-256 mismatch"):
        validate_lock(lock_path, repo_root=repo_root)


def test_qasm_derived_map_not_register_suffix_controls_cohort(
    synthetic_lock: tuple[Path, Path, dict[str, Any]],
) -> None:
    repo_root, lock_path, manifest = synthetic_lock
    schema = manifest["cohort_row_schema"]
    row = manifest["cohort_pairs"][0]
    data_index = schema.index("data_qubits")
    path_index = schema.index("oriented_path")
    row[data_index][0] += 1000
    row[path_index][0] += 1000
    _rewrite_lock(lock_path, manifest)
    with pytest.raises(LockValidationError, match="QASM map"):
        validate_lock(lock_path, repo_root=repo_root)


def test_reversed_early_late_property_date_order_is_rejected(
    synthetic_lock: tuple[Path, Path, dict[str, Any]],
) -> None:
    repo_root, lock_path, manifest = synthetic_lock
    schema = manifest["cohort_row_schema"]
    row = manifest["cohort_pairs"][0]
    early_index = schema.index("early_snapshot_id")
    late_index = schema.index("late_snapshot_id")
    row[early_index], row[late_index] = row[late_index], row[early_index]
    _rewrite_lock(lock_path, manifest)
    with pytest.raises(LockValidationError, match="UTC instant must precede"):
        validate_lock(lock_path, repo_root=repo_root)


def test_fixed_space_composite_target_is_enforced(
    synthetic_lock: tuple[Path, Path, dict[str, Any]],
) -> None:
    repo_root, lock_path, manifest = synthetic_lock
    manifest["features_and_methods"]["target_method"] = "space_sparse"
    _rewrite_lock(lock_path, manifest)
    with pytest.raises(
        LockValidationError,
        match="features_and_methods.target_method",
    ):
        validate_lock(lock_path, repo_root=repo_root)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("seed", "seed_formula"),
        ("resource", "aggregate_resource_counts"),
    ],
)
def test_seed_and_resource_arithmetic_are_locked(
    synthetic_lock: tuple[Path, Path, dict[str, Any]],
    mutation: str,
    message: str,
) -> None:
    repo_root, lock_path, manifest = synthetic_lock
    if mutation == "seed":
        manifest["alarm_calibration"]["seed_formula"] = "611000 + cohort_index"
    else:
        manifest["aggregate_resource_counts_per_pre_or_post_phase"][
            "paired_cycle_updates"
        ] += 1
    _rewrite_lock(lock_path, manifest)
    with pytest.raises(LockValidationError, match=message):
        validate_lock(lock_path, repo_root=repo_root)


def test_held_payload_stat_size_is_checked_without_opening(
    synthetic_lock: tuple[Path, Path, dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root, lock_path, manifest = synthetic_lock
    first_snapshot = next(iter(manifest["snapshots"].values()))
    first_snapshot["held_bitstrings"][0] += 1
    _rewrite_lock(lock_path, manifest)
    _guard_held_payload_open(monkeypatch)
    with pytest.raises(LockValidationError, match="stat size mismatch"):
        validate_lock(lock_path, repo_root=repo_root)
