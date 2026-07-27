"""Synthetic and metadata-only tests for the Google detector runner."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
import zipfile
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from aoc.run6_protocol import canonical_json_bytes, load_google_lock, sha256_file

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "experiments/run6/scripts/run_google2022_detector.py"
CONFIG = ROOT / "experiments/run6/configs/google2022_locked.json"


def _load_runner():
    spec = importlib.util.spec_from_file_location("run6_google_detector", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load the detector runner.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_synthetic_dry_run_is_deterministic_and_value_blind() -> None:
    runner = _load_runner()
    first = runner.synthetic_dry_run()
    second = runner.synthetic_dry_run()
    assert first == second
    assert first["status"] == "synthetic_dry_run_passed"
    assert first["raw_run6_values_opened"] is False
    assert first["role_update_counts"] == [6, 6]


def test_zip_member_hash_binds_extracted_bytes(tmp_path: Path) -> None:
    runner = _load_runner()
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as target:
        target.writestr("experiment/events.b8", b"\x01\x02\x03")
    extracted = tmp_path / "events.b8"
    extracted.write_bytes(b"\x01\x02\x03")
    assert runner._sha256_zip_member(
        archive,
        "experiment/events.b8",
    ) == sha256_file(extracted)
    with pytest.raises(ValueError, match="exactly one ZIP member"):
        runner._sha256_zip_member(archive, "missing.b8")


def test_threshold_payload_is_strict_and_canonical() -> None:
    runner = _load_runner()
    scores = {
        method: np.zeros((2, runner.ROLE_COUNT), dtype=np.float64)
        for method in runner.METHOD_IDS
    }
    payload = runner._threshold_payload(scores, max_alerts=0)
    encoded = canonical_json_bytes(payload)
    assert b"Infinity" not in encoded
    assert all(item["threshold"] == 0.0 for item in payload.values())
    assert all(item["validation_alert_count"] == 0 for item in payload.values())


def test_cycle_arrays_have_locked_dtypes_sidecars_and_one_notification(
    tmp_path: Path,
) -> None:
    runner = _load_runner()
    shape = (2, runner.ROLE_COUNT)
    scores = {method: np.zeros(shape, dtype=np.float64) for method in runner.METHOD_IDS}
    scores["space"][0, 3] = 0.8
    scores["space"][0, 5] = 0.9
    logs = {
        method: np.full(shape[0], np.nan, dtype=np.float64)
        for method in runner.METHOD_IDS
    }
    for method in runner.EXACT_METHOD_IDS:
        logs[method][:] = 0.0
    logs["space"][:] = 0.0
    logs["space"][1] = np.log(100.0)
    thresholds = {
        method: {
            "threshold": 0.5,
            "validation_alert_count": 0,
            "max_validation_alerts": 0,
        }
        for method in runner.METHOD_IDS
    }
    accumulator_summary = runner.ReplayAccumulatorSummary(
        proper_prior={
            method: {"first_crossing_update": 2 if method == "space" else None}
            for method in runner.EXACT_METHOD_IDS
        },
        shiryaev_roberts={
            method: {"first_crossing_update": None}
            for method in runner.EXACT_METHOD_IDS
        },
        expert_metadata={},
    )

    artifacts = runner._save_cycle_arrays(
        tmp_path,
        phase="held",
        scores=scores,
        log_e=logs,
        log_sr=logs,
        thresholds=thresholds,
        protocol_id="synthetic",
        run_id="synthetic",
        pair_index_start=0,
        reference_archive_start=20,
        monitor_archive_start=40,
        common_hashes={"test": "0" * 64},
        include_formal_accumulators=True,
        accumulator_summary=accumulator_summary,
    )

    score_path = tmp_path / "held__space__empirical_cycle_score.npy"
    score = np.load(score_path, allow_pickle=False)
    notification = np.load(
        tmp_path / "held__space__notification_emitted.npy",
        allow_pickle=False,
    )
    cooldown = np.load(
        tmp_path / "held__space__cooldown_active.npy",
        allow_pickle=False,
    )
    first_e = np.load(
        tmp_path / "held__space__first_e_crossing.npy",
        allow_pickle=False,
    )
    sidecar = json.loads(score_path.with_suffix(".json").read_text())
    exact_formal_sidecar = json.loads(
        (tmp_path / "held__space__log_eprocess.json").read_text()
    )
    inapplicable_formal_sidecar = json.loads(
        (tmp_path / "held__m2__log_eprocess.json").read_text()
    )
    assert score.dtype.str == "<f8"
    assert notification.dtype == np.bool_
    assert np.count_nonzero(notification[0]) == 1
    assert notification[0, 3]
    assert not np.any(cooldown[0, :4])
    assert np.all(cooldown[0, 4:])
    assert not np.any(cooldown[1])
    assert np.count_nonzero(first_e) == 1
    assert first_e[1]
    assert sidecar["data_sha256"] == sha256_file(score_path)
    assert exact_formal_sidecar["formal_claim_scope"].startswith("diagnostic_only")
    assert (
        inapplicable_formal_sidecar["formal_claim_scope"]
        == "not_applicable_no_formal_accumulator"
    )
    assert len(artifacts) == len(runner.METHOD_IDS) * 8 * 2


def test_complete_threshold_frontier_is_monotone_and_includes_infinities(
    tmp_path: Path,
) -> None:
    runner = _load_runner()
    template = np.asarray([[0.1, 0.4], [0.2, 0.2], [0.5, 0.1]])
    scores = {method: template for method in runner.METHOD_IDS}
    artifacts = runner._save_threshold_frontiers(
        tmp_path,
        scores=scores,
        protocol_id="synthetic",
        common_hashes={"test": "0" * 64},
        pair_index_start=5_000,
    )
    candidates = np.load(
        tmp_path / "threshold__space__frontier_candidate_threshold.npy",
        allow_pickle=False,
    )
    counts = np.load(
        tmp_path / "threshold__space__frontier_shot_alert_count.npy",
        allow_pickle=False,
    )
    sidecar = json.loads(
        (tmp_path / "threshold__space__frontier_candidate_threshold.json").read_text()
    )
    assert np.isneginf(candidates[0])
    assert np.isposinf(candidates[-1])
    assert counts[0] == 3
    assert counts[-1] == 0
    assert np.all(counts[1:] <= counts[:-1])
    assert sidecar["pair_index_range"] == [5_000, 5_003]
    assert len(artifacts) == len(runner.METHOD_IDS) * 2 * 2


def test_shot_table_contains_rank_cumulative_count_and_window_metadata(
    tmp_path: Path,
) -> None:
    runner = _load_runner()
    template = np.zeros((3, runner.ROLE_COUNT), dtype=np.float64)
    template[0, 2] = 0.8
    template[1, 1] = 0.8
    template[2, 4] = 0.2
    scores = {method: template for method in runner.METHOD_IDS}
    thresholds = {method: {"threshold": 0.5} for method in runner.METHOD_IDS}
    path = tmp_path / "held_shots.csv"
    runner._write_shot_table(
        path,
        phase="held",
        pair_index_start=0,
        reference_start=20,
        monitor_start=40,
        scores=scores,
        thresholds=thresholds,
        windows={
            "primary": [40, 41],
            "narrow": [41, 42],
            "wide": [40, 43],
        },
    )
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    first = rows[:3]
    assert [int(row["rank"]) for row in first] == [1, 2, 3]
    assert [int(row["cumulative_alert_count"]) for row in first] == [1, 2, 2]
    assert [int(row["reference_archive_shot"]) for row in first] == [20, 21, 22]
    assert [int(row["rank_tie_archive_shot"]) for row in first] == [40, 41, 42]
    assert [int(row["in_primary_window"]) for row in first] == [1, 0, 0]
    assert [int(row["in_narrow_window"]) for row in first] == [0, 1, 0]
    assert [int(row["in_wide_window"]) for row in first] == [1, 1, 1]


def test_formal_accumulator_updates_once_per_complete_shot() -> None:
    runner = _load_runner()

    class UnitFactorBank:
        def __init__(self) -> None:
            self.calls = 0

        def update(self, role, reference, monitor):
            self.calls += 1
            empirical = SimpleNamespace(
                m0=0.0,
                m1=0.0,
                m2=0.0,
                m3=0.0,
                m4=0.0,
                m5=0.0,
                space=0.0,
            )
            return SimpleNamespace(
                empirical=empirical,
                m0_factors=np.ones(8),
                m1_factors=np.ones(8),
                m3_factors=np.ones(12),
                m4_factors=np.ones(64),
                m5_factors=np.ones(24),
                space_factors=np.ones(88),
            )

    reference = np.zeros((2, runner.ROLE_COUNT, 24), dtype=np.uint8)
    monitor = np.zeros_like(reference)
    bank = UnitFactorBank()
    _, log_e, log_sr, summary = runner.replay_scores(
        bank,
        reference,
        monitor,
        with_accumulators=True,
        horizon=2,
    )
    assert bank.calls == 2 * runner.ROLE_COUNT
    assert log_e["space"].shape == (2,)
    assert log_sr["space"].shape == (2,)
    assert np.allclose(log_e["space"], 0.0)
    assert summary.proper_prior["space"]["role_count"] == runner.ROLE_COUNT
    assert summary.proper_prior["space"]["base_component_count"] == 88
    assert len(summary.proper_prior["space"]["final_log_components"]) == (
        runner.ROLE_COUNT * 88
    )
    assert summary.expert_metadata["space"]["factor_bounds_satisfied"] is True
    assert summary.expert_metadata["space"]["expert_count"] == (runner.ROLE_COUNT * 88)
    assert len(summary.expert_metadata["space"]["base_component_ids"]) == 88
    assert np.isclose(summary.expert_metadata["space"]["base_prior_sum"], 1.0)
    assert np.isclose(
        summary.expert_metadata["space"]["full_role_component_prior_sum"],
        1.0,
    )


def test_freeze_ratification_delegates_to_committed_chain_and_binds_spec(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    config = deepcopy(load_google_lock(CONFIG))
    config["status"] = "frozen_before_held_value_access"
    relative_config = "experiments/run6/configs/google2022_locked.json"
    relative_spec = config["normative_method_spec"]["path"]
    spec_path = tmp_path / relative_spec
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text("synthetic specification\n", encoding="utf-8")
    config["normative_method_spec"]["sha256"] = sha256_file(tmp_path / relative_spec)
    config_path = tmp_path / relative_config
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_bytes(canonical_json_bytes(config) + b"\n")
    ratification_path = tmp_path / "ratification.json"
    ratification_path.write_text("synthetic\n", encoding="utf-8")
    expected = {"freeze_commit": "a" * 40}
    calls: list[dict[str, object]] = []

    def fake_verify(path: Path, **kwargs: object) -> dict[str, str]:
        calls.append({"path": path, **kwargs})
        return expected

    monkeypatch.setattr(runner, "verify_committed_freeze_chain", fake_verify)

    observed = runner.verify_freeze_ratification(
        ratification_path,
        repo_root=tmp_path,
        config_path=config_path,
        config=config,
    )
    assert observed == expected
    assert calls[0]["required_paths"] == runner.RUN6_REQUIRED_FREEZE_PATHS

    spec_path.write_text("changed specification\n", encoding="utf-8")
    with pytest.raises(ValueError, match="embedded in config"):
        runner.verify_freeze_ratification(
            ratification_path,
            repo_root=tmp_path,
            config_path=config_path,
            config=config,
        )
