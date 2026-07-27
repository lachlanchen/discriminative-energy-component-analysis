"""Synthetic tests for the locked Google complete-pair randomization audit."""

from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from aoc.run6_protocol import (
    canonical_json_bytes,
    environment_fingerprint,
    load_google_lock,
    sha256_file,
)
from aoc.space_qec import CHECK_COUNT, DiagonalLikelihoodModel, RoleIsolatedQECBank

ROOT = Path(__file__).resolve().parents[3]
GOOGLE_LOCK = ROOT / "experiments/run6/configs/google2022_locked.json"
SCRIPT = ROOT / "experiments/run6/scripts/run_google2022_randomization.py"
SPEC = importlib.util.spec_from_file_location("run6_google_randomization", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = audit
SPEC.loader.exec_module(audit)


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _write_json(path: Path, value: object) -> None:
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def _artifact(path: Path) -> dict[str, object]:
    return {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _resources() -> dict[str, object]:
    exposure = {
        "fit_warmup": {
            "paired_shots": 5_000,
            "physical_shots": 10_000,
            "paired_role_updates": 255_000,
            "detector_bits_exposed": 12_240_000,
        },
        "threshold": {
            "paired_shots": 5_000,
            "physical_shots": 10_000,
            "paired_role_updates": 255_000,
            "detector_bits_exposed": 12_240_000,
        },
        "held": {
            "paired_shots": 20_000,
            "physical_shots": 40_000,
            "paired_role_updates": 1_020_000,
            "detector_bits_exposed": 48_960_000,
        },
    }
    operation_keys = {
        "paired_shots",
        "paired_role_updates",
        "m0_scores",
        "m1_diagonal_scores",
        "m2_quadratic_scores",
        "m3_pairwise_logistic_scores",
        "m3_sgd_updates",
        "m4_shared_ewma_updates",
        "m4_sort_operations",
        "m4_sparse_dot_products",
        "m5_shared_ewma_updates",
        "m5_trace_scores",
        "m5_shared_eigendecompositions",
        "within_shot_page_cusum_channel_steps",
    }
    empty_operation = dict.fromkeys(operation_keys, 0)
    return {
        "record_exposure": exposure,
        "high_level_operations": {
            "fit_warmup": empty_operation,
            "threshold": empty_operation,
            "held": empty_operation,
            "m2_covariance_fits": 1,
            "m2_precision_matrix_constructions": 1,
            "m2_fit_observations_used": 20_000,
            "held_joint_replay_repetitions": 3,
            "extra_timing_replay_role_updates": 2_040_000,
        },
        "warm_checkpoint_storage": {
            "total_numeric_state_bytes": 0,
            "bytes_by_prefix": {},
            "array_inventory": {},
        },
        "held_final_checkpoint_storage": {
            "total_numeric_state_bytes": 0,
            "bytes_by_prefix": {},
            "array_inventory": {},
        },
        "formal_accumulator": {
            "time_unit": "complete_paired_shot",
            "held_updates": 20_000,
            "role_prior": "uniform_1_over_51",
            "within_shot_factor_compounding": False,
        },
        "output_bytes_before_manifest": 0,
    }


def _performance() -> dict[str, object]:
    return {
        "canonical_joint_pipeline_only": True,
        "not_a_per_method_speed_comparison": True,
        "integrity_and_layout_seconds": 0.0,
        "validation_read_seconds": 0.0,
        "warm_fit_replay_seconds": 0.0,
        "threshold_replay_seconds": 0.0,
        "threshold_serialization_seconds": 0.0,
        "held_read_seconds": 0.0,
        "held_replay_seconds": 0.0,
        "held_joint_replay_all_three_seconds": [0.0, 0.0, 0.0],
        "held_joint_replay_median_seconds": 0.0,
        "held_joint_replay_digests": ["0" * 64, "0" * 64, "0" * 64],
        "held_serialization_seconds": 0.0,
        "elapsed_before_manifest_seconds": 0.0,
        "peak_rss_kib_linux_ru_maxrss": 0,
        "relative_method_speed_claim_authorized": False,
    }


def _detector_manifest_fixture(tmp_path: Path) -> tuple[Path, Path]:
    config = load_google_lock(GOOGLE_LOCK)
    ratification = tmp_path / "freeze_ratification.json"
    _write_json(ratification, {"synthetic": True})
    required = audit.expected_detector_artifact_names()
    artifact_paths: list[Path] = []
    for index, name in enumerate(sorted(required)):
        path = tmp_path / name
        path.write_bytes(f"synthetic-{index}".encode())
        artifact_paths.append(path)
    threshold = tmp_path / "thresholds.json"
    threshold_stage = tmp_path / "threshold_stage_manifest.json"
    _write_json(
        threshold_stage,
        {
            "schema_version": "run6-google-threshold-stage-v1",
            "protocol_id": config["protocol_id"],
            "held_values_decoded_or_scored": False,
            "config_sha256": sha256_file(GOOGLE_LOCK),
            "method_spec_sha256": sha256_file(
                ROOT / config["normative_method_spec"]["path"]
            ),
            "detector_script_sha256": sha256_file(
                ROOT / "experiments/run6/scripts/run_google2022_detector.py"
            ),
            "warm_checkpoint_sha256": "2" * 64,
            "threshold_final_checkpoint_sha256": "3" * 64,
            "threshold_table_sha256": sha256_file(threshold),
            "threshold_artifacts": [
                _artifact(path)
                for path in artifact_paths
                if path.name in {"thresholds.json", "threshold_shots.csv"}
                or path.name.startswith("threshold__")
            ],
        },
    )
    manifest = {
        "schema_version": "run6-google-detector-freeze-v1",
        "protocol_id": config["protocol_id"],
        "detector_only": True,
        "outcome_accessed": False,
        "outcome_join_authorized": False,
        "git_commit": _git_commit(),
        "config_sha256": sha256_file(GOOGLE_LOCK),
        "method_spec_sha256": sha256_file(
            ROOT / config["normative_method_spec"]["path"]
        ),
        "detector_script_sha256": sha256_file(
            ROOT / "experiments/run6/scripts/run_google2022_detector.py"
        ),
        "freeze_ratification_sha256": sha256_file(ratification),
        "deviation_ledger": {
            "path": config["deviation_ledger"],
            "sha256": sha256_file(ROOT / config["deviation_ledger"]),
        },
        "circuit_sha256": "0" * 64,
        "detector_layout_index_sha256": "1" * 64,
        "warm_checkpoint_sha256": "2" * 64,
        "threshold_checkpoint_sha256": "3" * 64,
        "held_final_checkpoint_sha256": "4" * 64,
        "source_archive_sha256": config["source"]["sha256"],
        "source_archive_bytes": config["source"]["archive_bytes"],
        "verified_zip_member_sha256": {
            "circuit_ideal.stim": "5" * 64,
            config["source"]["detection_event_file"]: "6" * 64,
        },
        "detection_file_bytes": config["source"]["detection_event_file_bytes"],
        "threshold_table_sha256": sha256_file(threshold),
        "artifacts": [_artifact(path) for path in artifact_paths],
        "resources": _resources(),
        "performance": _performance(),
        "environment": environment_fingerprint(),
        "command": ["synthetic"],
        "started_unix": 0.0,
        "finished_unix": 1.0,
    }
    path = tmp_path / "detector_freeze_manifest.json"
    _write_json(path, manifest)
    return path, ratification


def test_locked_swap_draw_and_complete_shot_orientation() -> None:
    observed = audit.draw_complete_shot_swaps(610700, shot_count=7)
    expected = np.random.Generator(np.random.PCG64(610700)).integers(
        0,
        2,
        size=7,
        dtype=np.uint8,
    )
    np.testing.assert_array_equal(observed, expected)

    reference = np.zeros((7, 3, CHECK_COUNT), dtype=np.uint8)
    monitor = np.ones_like(reference)
    left, right = audit.apply_complete_shot_swaps(reference, monitor, observed)
    for shot, swap in enumerate(observed):
        expected_left = int(swap)
        assert np.all(left[shot] == expected_left)
        assert np.all(right[shot] == 1 - expected_left)


def test_clopper_pearson_interval_has_exact_boundary_cases() -> None:
    lower, upper = audit.clopper_pearson_interval(0, 256)
    assert lower == 0.0
    assert np.isclose(upper, 1.0 - 0.025 ** (1.0 / 256))

    lower, upper = audit.clopper_pearson_interval(256, 256)
    assert np.isclose(lower, 0.025 ** (1.0 / 256))
    assert upper == 1.0

    with pytest.raises(ValueError):
        audit.clopper_pearson_interval(2, 1)


def test_threshold_bootstrap_uses_locked_circular_pcg64_blocks() -> None:
    observed = audit.draw_threshold_bootstrap_indices(
        613000,
        shot_count=10,
        block_length=4,
    )
    starts = np.random.Generator(np.random.PCG64(613000)).integers(
        0,
        10,
        size=3,
    )
    expected = np.concatenate([np.arange(start, start + 4) % 10 for start in starts])[
        :10
    ]
    np.testing.assert_array_equal(observed, expected)


def test_threshold_bootstrap_is_descriptive_and_byte_deterministic() -> None:
    generator = np.random.Generator(np.random.PCG64(612999))
    scores = {
        method: generator.random((16, 3), dtype=np.float64)
        for method in audit.METHOD_IDS
    }
    locked = {
        method: {
            "threshold": float(np.sort(np.max(values, axis=1))[::-1][2]),
            "validation_alert_count": 2,
            "max_validation_alerts": 2,
            "secondary_zero_alert_threshold": float(np.max(values)),
            "secondary_validation_alert_count": 0,
        }
        for method, values in scores.items()
    }
    first = audit.threshold_bootstrap(
        scores,
        locked,
        replicates=12,
        seed_start=613000,
        block_length=4,
    )
    second = audit.threshold_bootstrap(
        scores,
        locked,
        replicates=12,
        seed_start=613000,
        block_length=4,
    )
    assert canonical_json_bytes(first) == canonical_json_bytes(second)
    assert first["status"] == "descriptive_only_does_not_replace_frozen_threshold"
    assert first["replicates"] == 12
    for row in first["replicate_results"]:
        for method in audit.METHOD_IDS:
            assert row["methods"][method]["selected_primary_alert_count"] <= 2
            assert row["methods"][method]["selected_zero_alert_count"] == 0


def test_replicate_restores_checkpoint_and_is_byte_deterministic() -> None:
    generator = np.random.Generator(np.random.PCG64(610698))
    warm_left = generator.integers(0, 2, size=(3, 2, CHECK_COUNT), dtype=np.uint8)
    warm_right = generator.integers(0, 2, size=(3, 2, CHECK_COUNT), dtype=np.uint8)
    bank = RoleIsolatedQECBank(
        role_count=2,
        diagonal_model=DiagonalLikelihoodModel.fit(warm_left, warm_right),
    )
    for shot in range(3):
        for role in range(2):
            bank.update(role, warm_left[shot, role], warm_right[shot, role])
    checkpoint = bank.state_digest()
    reference = generator.integers(
        0,
        2,
        size=(5, 2, CHECK_COUNT),
        dtype=np.uint8,
    )
    monitor = generator.integers(
        0,
        2,
        size=(5, 2, CHECK_COUNT),
        dtype=np.uint8,
    )
    swaps = audit.draw_complete_shot_swaps(610700, shot_count=5)
    first = audit.run_randomization_replicate(
        bank,
        reference,
        monitor,
        swaps,
        seed=610700,
        expected_checkpoint_sha256=checkpoint,
    )
    second = audit.run_randomization_replicate(
        bank,
        reference,
        monitor,
        swaps,
        seed=610700,
        expected_checkpoint_sha256=checkpoint,
    )
    assert bank.state_digest() == checkpoint
    assert canonical_json_bytes(first) == canonical_json_bytes(second)
    assert first["formal_eprocess_updates"] == 5
    assert first["role_score_updates"] == 10
    assert first["formal_experts"]["space"] == 2 * 88
    assert first["checkpoint_restored"] is True
    assert set(first["crossed_100"]) == set(audit.EXACT_METHOD_IDS)


def test_randomization_summary_uses_space_as_the_only_primary_process() -> None:
    rows = []
    for index in range(4):
        rows.append(
            {
                "formal_eprocess_updates": 6,
                "role_score_updates": 12,
                "crossed_100": {
                    method: bool(method == "space" and index < 2)
                    for method in audit.EXACT_METHOD_IDS
                },
                "familywide_any_crossed_600": index == 0,
            }
        )
    summary = audit.summarize_replicates(rows, checkpoint_sha256="a" * 64)
    assert summary["primary_method"] == "space"
    assert summary["crossing_counts_at_100"]["space"] == 2
    assert summary["space_crossing_fraction"] == 0.5
    assert summary["familywide_any_crossing_count_at_600"] == 1
    assert summary["horizon_paired_shots"] == 6


def _synthetic_merge_row(index: int) -> dict[str, object]:
    return {
        "replicate_index": index,
        "seed": 610700 + index,
        "swap_sha256": f"{index:064x}",
        "swapped_shot_count": 2,
        "checkpoint_restored": True,
        "crossed_100": {
            method: bool(method == "space" and index % 2 == 0)
            for method in audit.EXACT_METHOD_IDS
        },
        "first_crossing_shot_number_one_based": {
            method: (3 if method == "space" and index % 2 == 0 else None)
            for method in audit.EXACT_METHOD_IDS
        },
        "maximum_log_e": {method: float(index) for method in audit.EXACT_METHOD_IDS},
        "final_log_e": {
            method: float(index) / 2.0 for method in audit.EXACT_METHOD_IDS
        },
        "familywide_any_crossed_600": False,
        "formal_eprocess_updates": 4,
        "role_score_updates": 8,
        "formal_experts": {
            "m0": 408,
            "m1": 408,
            "m3": 612,
            "m4": 3264,
            "m5": 1224,
            "space": 4488,
        },
    }


def test_shard_resume_and_layout_independent_canonical_merge() -> None:
    rows = [_synthetic_merge_row(index) for index in range(4)]
    one_shard = audit.merge_randomization_rows(
        [rows],
        expected_replicates=4,
        horizon_shots=4,
        role_score_updates=8,
    )
    multiple_shards = audit.merge_randomization_rows(
        [rows[2:], rows[:1], rows[1:2]],
        expected_replicates=4,
        horizon_shots=4,
        role_score_updates=8,
    )
    one_result = audit.summarize_replicates(
        one_shard,
        checkpoint_sha256="a" * 64,
    )
    multiple_result = audit.summarize_replicates(
        multiple_shards,
        checkpoint_sha256="a" * 64,
    )
    assert canonical_json_bytes(one_result) == canonical_json_bytes(multiple_result)

    with pytest.raises(ValueError, match="coverage is incomplete"):
        audit.merge_randomization_rows(
            [rows[:1], rows[2:]],
            expected_replicates=4,
            horizon_shots=4,
            role_score_updates=8,
        )
    resumed = audit.merge_randomization_rows(
        [rows[:1], rows[2:], rows[1:2]],
        expected_replicates=4,
        horizon_shots=4,
        role_score_updates=8,
    )
    assert [row["seed"] for row in resumed] == list(range(610700, 610704))
    with pytest.raises(ValueError, match="Duplicate"):
        audit.merge_randomization_rows(
            [rows, rows[:1]],
            expected_replicates=4,
            horizon_shots=4,
            role_score_updates=8,
        )


def test_threshold_frontier_sidecars_match_exact_producer_schema(
    tmp_path: Path,
) -> None:
    protocol_id = "run6-google2022-v2"
    common_hashes = {
        "config_sha256": "0" * 64,
        "method_spec_sha256": "1" * 64,
        "detector_script_sha256": "2" * 64,
        "warm_checkpoint_sha256": "3" * 64,
        "threshold_final_checkpoint_sha256": "4" * 64,
        "freeze_ratification_sha256": "5" * 64,
        "deviation_ledger_sha256": "6" * 64,
        "python_environment_lock_sha256": "7" * 64,
        "freeze_manifest_sha256": "8" * 64,
    }
    scores: dict[str, np.ndarray] = {}
    artifacts: dict[str, Path] = {}
    for method_index, method in enumerate(audit.METHOD_IDS):
        values = np.asarray(
            [[0.1, 0.2], [0.3, 0.4 + 0.01 * method_index]],
            dtype=np.float64,
        )
        scores[method] = values
        candidates = np.concatenate(
            (
                np.asarray([-np.inf], dtype="<f8"),
                np.unique(values).astype("<f8"),
                np.asarray([np.inf], dtype="<f8"),
            )
        )
        maxima = np.sort(np.max(values, axis=1))
        counts = (
            len(values) - np.searchsorted(maxima, candidates, side="right")
        ).astype("<i8")
        for array_id, array in (
            ("candidate_threshold", candidates),
            ("shot_alert_count", counts),
        ):
            data_path = tmp_path / f"threshold__{method}__frontier_{array_id}.npy"
            np.save(data_path, array, allow_pickle=False)
            sidecar_path = data_path.with_suffix(".json")
            _write_json(
                sidecar_path,
                {
                    "schema_version": "run6-threshold-frontier-array-v1",
                    "protocol_id": protocol_id,
                    "method_id": method,
                    "array_id": array_id,
                    "data_file": data_path.name,
                    "data_sha256": sha256_file(data_path),
                    "shape": list(array.shape),
                    "dtype": array.dtype.str,
                    "candidate_rule": ("[-inf] + sorted_unique_cycle_scores + [+inf]"),
                    "count_rule": (
                        "strict_greater_than_with_at_most_one_notification_per_shot"
                    ),
                    "pair_index_range": [5_000, 5_002],
                    "checkpoint_and_code_hashes": common_hashes,
                },
            )
            artifacts[data_path.name] = data_path
            artifacts[sidecar_path.name] = sidecar_path
    audit.validate_locked_threshold_frontiers(
        artifacts,
        scores,
        protocol_id=protocol_id,
        expected_common_hashes=common_hashes,
    )

    first_sidecar = artifacts["threshold__m0__frontier_candidate_threshold.json"]
    payload = json.loads(first_sidecar.read_text(encoding="utf-8"))
    payload["unexpected"] = True
    _write_json(first_sidecar, payload)
    with pytest.raises(ValueError, match="schema mismatch"):
        audit.validate_locked_threshold_frontiers(
            artifacts,
            scores,
            protocol_id=protocol_id,
            expected_common_hashes=common_hashes,
        )


def test_detector_manifest_gate_rejects_unknown_fields_and_tampering(
    tmp_path: Path,
) -> None:
    manifest_path, ratification_path = _detector_manifest_fixture(tmp_path)
    config = load_google_lock(GOOGLE_LOCK)
    audit.verify_detector_manifest(
        manifest_path,
        config_path=GOOGLE_LOCK,
        config=config,
        ratification_path=ratification_path,
        repo_root=ROOT,
    )

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["post_hoc"] = True
    _write_json(manifest_path, payload)
    with pytest.raises(ValueError, match="unknown"):
        audit.verify_detector_manifest(
            manifest_path,
            config_path=GOOGLE_LOCK,
            config=config,
            ratification_path=ratification_path,
            repo_root=ROOT,
        )

    del payload["post_hoc"]
    _write_json(manifest_path, payload)
    (tmp_path / "held__space__empirical_cycle_score.npy").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash/size"):
        audit.verify_detector_manifest(
            manifest_path,
            config_path=GOOGLE_LOCK,
            config=config,
            ratification_path=ratification_path,
            repo_root=ROOT,
        )


def test_freeze_gate_delegates_to_canonical_committed_chain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = copy.deepcopy(load_google_lock(GOOGLE_LOCK))
    config["status"] = "frozen_before_held_value_access"
    spec_path = ROOT / config["normative_method_spec"]["path"]
    config["normative_method_spec"]["sha256"] = sha256_file(spec_path)
    path = tmp_path / "ratification.json"
    path.write_text("delegated\n", encoding="utf-8")
    calls: list[dict[str, object]] = []

    def fake_verify(
        supplied: Path,
        **kwargs: object,
    ) -> dict[str, object]:
        calls.append({"supplied": supplied, **kwargs})
        return {"verified": True}

    monkeypatch.setattr(audit, "verify_committed_freeze_chain", fake_verify)
    observed = audit.verify_freeze_ratification(
        path,
        repo_root=ROOT,
        config_path=GOOGLE_LOCK,
        config=config,
    )
    assert observed == {"verified": True}
    assert calls[0]["supplied"] == path
    assert calls[0]["required_paths"] == audit.RUN6_REQUIRED_FREEZE_PATHS
    assert calls[0]["expected_environment"] == environment_fingerprint()
    assert (
        calls[0]["expected_thread_environment"]
        == config["numeric_policy"]["thread_environment"]
    )

    with pytest.raises(ValueError, match="canonical Google lock"):
        audit.verify_freeze_ratification(
            path,
            repo_root=ROOT,
            config_path=tmp_path / "google2022_locked.json",
            config=config,
        )


def test_synthetic_dry_run_opens_no_run6_values() -> None:
    result = audit.synthetic_dry_run()
    assert result["status"] == "synthetic_dry_run_passed"
    assert result["raw_run6_values_opened"] is False
