#!/usr/bin/env python3
"""Frozen detector-only replay for the Google 2022 Run 6 arm.

The detector command cannot accept or open any ``.01`` outcome file.  A
committed freeze-ratification JSON whose hashes match the configuration,
method specification, code, and environment is mandatory before the real
``detection_events.b8`` payload is opened.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import resource
import statistics
import subprocess
import sys
import time
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from aoc.qec_real import parse_stim_detector_layout, read_b8_detector_shots
from aoc.run6_protocol import (
    RUN6_REQUIRED_FREEZE_PATHS,
    assert_no_outcome_paths,
    canonical_json_bytes,
    environment_fingerprint,
    load_google_lock,
    sha256_file,
    verify_committed_freeze_chain,
)
from aoc.space import MixtureSRBank, ProperUniformStartEProcessBank
from aoc.space_qec import (
    FEATURE_DIM,
    ROLE_COUNT,
    DiagonalLikelihoodModel,
    ResourceCounts,
    RoleHotellingModel,
    RoleIsolatedQECBank,
    apply_strict_shot_threshold,
    exact_component_priors,
    paired_page_cusum_shot,
    paired_qec_contrasts,
    paired_resource_counts,
    select_role_fit_indices,
    select_strict_shot_threshold,
)
from sklearn.covariance import LedoitWolf

METHOD_IDS = ("m0", "m0c", "m1", "m2", "m3", "m4", "m5", "space")
EXACT_METHOD_IDS = ("m0", "m1", "m3", "m4", "m5", "space")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--freeze-ratification", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _git_commit(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _write_canonical_json(path: Path, value: Any) -> None:
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def _artifact_record(path: Path, *, base: Path | None = None) -> dict[str, Any]:
    recorded_path = path if base is None else path.relative_to(base)
    return {
        "path": recorded_path.as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _sha256_zip_member(archive: Path, member: str) -> str:
    """Hash one exact member of an already hash-verified ZIP archive."""

    digest = hashlib.sha256()
    with zipfile.ZipFile(archive) as source:
        matches = [name for name in source.namelist() if name == member]
        if matches != [member]:
            raise ValueError(f"Expected exactly one ZIP member {member!r}.")
        with source.open(member, "r") as handle:
            while block := handle.read(1 << 20):
                digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class ReplayAccumulatorSummary:
    """Final state of every formal accumulator in one replay."""

    proper_prior: dict[str, dict[str, Any]]
    shiryaev_roberts: dict[str, dict[str, Any]]
    expert_metadata: dict[str, dict[str, Any]]


def _state_resource_summary(bank: RoleIsolatedQECBank) -> dict[str, Any]:
    """Return exact numeric checkpoint storage without serializing values."""

    arrays = bank.export_numeric_state()
    inventory = {
        name: {
            "shape": list(values.shape),
            "dtype": values.dtype.str,
            "elements": int(values.size),
            "bytes": int(values.nbytes),
        }
        for name, values in sorted(arrays.items())
    }
    groups: dict[str, int] = {}
    for name, values in arrays.items():
        prefix = ".".join(name.split(".")[:2])
        groups[prefix] = groups.get(prefix, 0) + int(values.nbytes)
    return {
        "total_numeric_state_bytes": sum(item["bytes"] for item in inventory.values()),
        "bytes_by_prefix": dict(sorted(groups.items())),
        "array_inventory": inventory,
    }


def _operation_ledger(
    *,
    paired_shots: int,
    role_updates_before: int,
) -> dict[str, int]:
    """Count the locked high-level operations in one replay phase."""

    updates = paired_shots * ROLE_COUNT
    eig_per_role_half_life = (
        role_updates_before + paired_shots
    ) // 8 - role_updates_before // 8
    return {
        "paired_shots": paired_shots,
        "paired_role_updates": updates,
        "m0_scores": updates,
        "m1_diagonal_scores": updates,
        "m2_quadratic_scores": updates,
        "m3_pairwise_logistic_scores": 3 * updates,
        "m3_sgd_updates": 3 * updates,
        "m4_shared_ewma_updates": 4 * updates,
        "m4_sort_operations": 4 * updates,
        "m4_sparse_dot_products": 16 * updates,
        "m5_shared_ewma_updates": 3 * updates,
        "m5_trace_scores": 6 * updates,
        "m5_shared_eigendecompositions": (ROLE_COUNT * 3 * eig_per_role_half_life),
        "within_shot_page_cusum_channel_steps": paired_shots * ROLE_COUNT * 3 * 25,
    }


def _replay_digest(
    scores: Mapping[str, np.ndarray],
    log_e: Mapping[str, np.ndarray],
    log_sr: Mapping[str, np.ndarray],
    summary: ReplayAccumulatorSummary,
) -> str:
    """Hash canonical replay numerics for timing-repeat equality checks."""

    digest = hashlib.sha256()
    for family, arrays in (
        ("scores", scores),
        ("log_e", log_e),
        ("log_sr", log_sr),
    ):
        for method, values in sorted(arrays.items()):
            array = np.asarray(values)
            canonical = np.ascontiguousarray(
                array.astype(array.dtype.newbyteorder("<"), copy=False)
            )
            metadata = canonical_json_bytes(
                {
                    "family": family,
                    "method": method,
                    "dtype": canonical.dtype.str,
                    "shape": list(canonical.shape),
                }
            )
            digest.update(len(metadata).to_bytes(8, "little"))
            digest.update(metadata)
            payload = canonical.tobytes(order="C")
            digest.update(len(payload).to_bytes(8, "little"))
            digest.update(payload)
    summary_bytes = canonical_json_bytes(
        {
            "proper_prior": summary.proper_prior,
            "shiryaev_roberts": summary.shiryaev_roberts,
            "expert_metadata": summary.expert_metadata,
        }
    )
    digest.update(len(summary_bytes).to_bytes(8, "little"))
    digest.update(summary_bytes)
    return digest.hexdigest()


def verify_freeze_ratification(
    path: Path,
    *,
    repo_root: Path,
    config_path: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify the external, post-freeze commit ratification gate."""

    expected_config_path = (
        repo_root / "experiments/run6/configs/google2022_locked.json"
    ).resolve()
    if config_path.resolve() != expected_config_path:
        raise ValueError("Real replay requires the canonical Google lock path.")
    if config["status"] != "frozen_before_held_value_access":
        raise ValueError("Google configuration has not reached frozen status.")
    spec_relative = config["normative_method_spec"]["path"]
    spec_path = repo_root / spec_relative
    if config["normative_method_spec"]["sha256"] != sha256_file(spec_path):
        raise ValueError("Normative method-spec hash embedded in config changed.")

    expected_threads = config["numeric_policy"]["thread_environment"]
    return verify_committed_freeze_chain(
        path,
        repo_root=repo_root,
        required_paths=RUN6_REQUIRED_FREEZE_PATHS,
        expected_environment=environment_fingerprint(),
        expected_thread_environment=expected_threads,
    )


def fit_hotelling_from_bits(
    reference: np.ndarray,
    monitor: np.ndarray,
) -> RoleHotellingModel:
    """Fit locked M2 without materializing a 5000×51×300 tensor."""

    if reference.shape != monitor.shape or reference.shape != (5_000, 51, 24):
        raise ValueError("M2 fit arrays must have shape (5000,51,24).")
    selection = select_role_fit_indices()
    selected = np.empty((len(selection), FEATURE_DIM), dtype=np.float64)
    role_means = np.empty((ROLE_COUNT, FEATURE_DIM), dtype=np.float64)
    cursor = 0
    for role in range(ROLE_COUNT):
        pair_indices = selection[selection[:, 1] == role, 0]
        role_values = np.empty((len(pair_indices), FEATURE_DIM), dtype=np.float64)
        for local_index, pair_index in enumerate(pair_indices):
            role_values[local_index] = paired_qec_contrasts(
                reference[pair_index, role],
                monitor[pair_index, role],
            ).feature_difference
        role_means[role] = np.mean(role_values, axis=0)
        next_cursor = cursor + len(pair_indices)
        selected[cursor:next_cursor] = role_values - role_means[role]
        cursor = next_cursor
    if cursor != 20_000:
        raise RuntimeError("M2 selected-observation count changed.")
    estimator = LedoitWolf(store_precision=True, assume_centered=True)
    estimator.fit(selected)
    precision = np.asarray(estimator.precision_, dtype=np.float64)
    precision = 0.5 * (precision + precision.T)
    return RoleHotellingModel(
        role_means=role_means,
        precision=precision,
        selected_indices=selection,
    )


def _score_arrays(num_pairs: int) -> dict[str, np.ndarray]:
    return {
        method: np.empty((num_pairs, ROLE_COUNT), dtype=np.float64)
        for method in METHOD_IDS
    }


def _empirical_scores(update: Any) -> dict[str, float]:
    return {
        "m0": update.empirical.m0,
        "m1": update.empirical.m1,
        "m2": update.empirical.m2,
        "m3": update.empirical.m3,
        "m4": update.empirical.m4,
        "m5": update.empirical.m5,
        "space": update.empirical.space,
    }


def _factor_arrays(update: Any) -> dict[str, np.ndarray]:
    if update.m1_factors is None:
        raise RuntimeError("M1 factors are required in the frozen replay.")
    return {
        "m0": update.m0_factors,
        "m1": update.m1_factors,
        "m3": update.m3_factors,
        "m4": update.m4_factors,
        "m5": update.m5_factors,
        "space": update.space_factors,
    }


def replay_scores(
    bank: RoleIsolatedQECBank,
    reference: np.ndarray,
    monitor: np.ndarray,
    *,
    with_accumulators: bool,
    horizon: int | None = None,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    ReplayAccumulatorSummary,
]:
    """Causally replay complete paired shots in immutable logical order."""

    if reference.shape != monitor.shape:
        raise ValueError("Reference and monitor blocks must have equal shape.")
    if reference.ndim != 3 or reference.shape[1:] != (ROLE_COUNT, 24):
        raise ValueError("Replay blocks must have shape (pairs,51,24).")
    pair_count = reference.shape[0]
    scores = _score_arrays(pair_count)
    log_e = {
        method: np.full(pair_count, np.nan, dtype=np.float64) for method in METHOD_IDS
    }
    log_sr = {
        method: np.full(pair_count, np.nan, dtype=np.float64) for method in METHOD_IDS
    }

    priors = exact_component_priors()
    factor_minimum = {method: np.inf for method in EXACT_METHOD_IDS}
    factor_maximum = {method: -np.inf for method in EXACT_METHOD_IDS}
    factor_all_finite_nonnegative = {method: True for method in EXACT_METHOD_IDS}
    e_banks: dict[str, ProperUniformStartEProcessBank] = {}
    sr_banks: dict[str, MixtureSRBank] = {}
    if with_accumulators:
        if horizon != pair_count:
            raise ValueError(
                "Accumulator horizon must equal the complete-shot replay length."
            )
        for method in EXACT_METHOD_IDS:
            prior = priors[method]
            role_component_weights = np.tile(
                prior.weights / ROLE_COUNT,
                ROLE_COUNT,
            )
            e_banks[method] = ProperUniformStartEProcessBank(
                len(role_component_weights),
                horizon=horizon,
                alpha=0.01,
                component_weights=role_component_weights,
            )
            sr_banks[method] = MixtureSRBank(
                len(role_component_weights),
                gamma=1_000_000,
                component_weights=role_component_weights,
            )

    for pair_index in range(pair_count):
        shot_factors: dict[str, list[np.ndarray]] = {
            method: [] for method in EXACT_METHOD_IDS
        }
        cusum = paired_page_cusum_shot(
            reference[pair_index],
            monitor[pair_index],
        )
        scores["m0c"][pair_index] = cusum.cycle_scores
        for role in range(ROLE_COUNT):
            update = bank.update(
                role,
                reference[pair_index, role],
                monitor[pair_index, role],
            )
            for method, value in _empirical_scores(update).items():
                if value is None:
                    raise RuntimeError(f"{method} empirical score is missing.")
                scores[method][pair_index, role] = value
            if with_accumulators:
                for method, factors in _factor_arrays(update).items():
                    factor_minimum[method] = min(
                        factor_minimum[method],
                        float(np.min(factors)),
                    )
                    factor_maximum[method] = max(
                        factor_maximum[method],
                        float(np.max(factors)),
                    )
                    factor_all_finite_nonnegative[method] = (
                        factor_all_finite_nonnegative[method]
                        and bool(np.all(np.isfinite(factors)))
                        and bool(np.all(factors >= 0.0))
                    )
                    shot_factors[method].append(factors)
        if with_accumulators:
            for method in EXACT_METHOD_IDS:
                flattened = np.concatenate(shot_factors[method])
                e_update = e_banks[method].update(flattened)
                sr_update = sr_banks[method].update(flattened)
                log_e[method][pair_index] = e_update.log_statistic
                log_sr[method][pair_index] = sr_update.log_statistic
    accumulator_summary = ReplayAccumulatorSummary(
        proper_prior={
            method: {
                "component_weights": e_banks[method].component_weights.tolist(),
                "role_count": ROLE_COUNT,
                "base_component_count": int(
                    len(e_banks[method].component_weights) / ROLE_COUNT
                ),
                "expert_flatten_order": ["role", "base_component"],
                "expert_id_rule": (
                    "(role, *base_component_id), role-major then base-component-major"
                ),
                "final_log_components": e_banks[method].log_components.tolist(),
                "final_log_statistic": e_banks[method].log_statistic,
                "first_crossing_update": e_banks[method].alarm_time,
                "threshold": e_banks[method].threshold,
            }
            for method in e_banks
        },
        shiryaev_roberts={
            method: {
                "component_weights": sr_banks[method].component_weights.tolist(),
                "role_count": ROLE_COUNT,
                "base_component_count": int(
                    len(sr_banks[method].component_weights) / ROLE_COUNT
                ),
                "expert_flatten_order": ["role", "base_component"],
                "final_log_components": sr_banks[method].log_components.tolist(),
                "final_log_statistic": sr_banks[method].log_statistic,
                "first_crossing_update": sr_banks[method].alarm_time,
                "threshold": sr_banks[method].gamma,
            }
            for method in sr_banks
        },
        expert_metadata={
            method: {
                "expert_flatten_order": ["role", "base_component"],
                "role_prior": 1.0 / ROLE_COUNT,
                "within_shot_factor_compounding": False,
                "base_component_ids": [
                    list(identifier) for identifier in priors[method].component_ids
                ],
                "base_component_weights": priors[method].weights.tolist(),
                "expert_count": ROLE_COUNT * len(priors[method].component_ids),
                "observed_factor_minimum": factor_minimum[method],
                "observed_factor_maximum": factor_maximum[method],
                "all_factors_finite_and_nonnegative": (
                    factor_all_finite_nonnegative[method]
                ),
                "declared_factor_bounds": [0.1, 1.9],
                "factor_bounds_satisfied": (
                    factor_all_finite_nonnegative[method]
                    and factor_minimum[method] >= 0.1 - 1e-12
                    and factor_maximum[method] <= 1.9 + 1e-12
                ),
                "base_prior_sum": float(np.sum(priors[method].weights)),
                "full_role_component_prior_sum": float(
                    np.sum(e_banks[method].component_weights)
                ),
            }
            for method in e_banks
        },
    )
    return scores, log_e, log_sr, accumulator_summary


def warm_bank(
    reference: np.ndarray,
    monitor: np.ndarray,
) -> tuple[RoleIsolatedQECBank, str]:
    """Fit fixed controls, warm adaptive states, and return checkpoint digest."""

    diagonal = DiagonalLikelihoodModel.fit(reference, monitor)
    hotelling = fit_hotelling_from_bits(reference, monitor)
    bank = RoleIsolatedQECBank(
        role_count=ROLE_COUNT,
        diagonal_model=diagonal,
        hotelling_model=hotelling,
    )
    replay_scores(
        bank,
        reference,
        monitor,
        with_accumulators=False,
    )
    return bank, bank.state_digest()


def _threshold_payload(
    scores: Mapping[str, np.ndarray],
    *,
    max_alerts: int,
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for method in METHOD_IDS:
        selected = select_strict_shot_threshold(scores[method], max_alerts)
        threshold: float | str = selected.threshold
        if np.isposinf(selected.threshold):
            threshold = "+inf"
        payload[method] = {
            "threshold": threshold,
            "validation_alert_count": selected.alert_count,
            "max_validation_alerts": max_alerts,
            "secondary_zero_alert_threshold": float(np.max(selected.shot_scores)),
            "secondary_validation_alert_count": 0,
        }
    return payload


def _save_threshold_frontiers(
    output: Path,
    *,
    scores: Mapping[str, np.ndarray],
    protocol_id: str,
    common_hashes: Mapping[str, str],
    pair_index_start: int,
) -> list[dict[str, Any]]:
    """Persist every locked cycle-score candidate and its shot alert count."""

    artifacts: list[dict[str, Any]] = []
    for method in METHOD_IDS:
        values = np.asarray(scores[method], dtype=np.float64)
        candidates = np.concatenate(
            (
                np.asarray([-np.inf], dtype=np.float64),
                np.unique(values),
                np.asarray([np.inf], dtype=np.float64),
            )
        ).astype("<f8", copy=False)
        sorted_shot_maxima = np.sort(np.max(values, axis=1))
        counts = (
            values.shape[0]
            - np.searchsorted(sorted_shot_maxima, candidates, side="right")
        ).astype("<i8", copy=False)
        if np.any(counts[1:] > counts[:-1]):
            raise RuntimeError("Threshold frontier alert counts are not monotone.")
        for array_id, array in (
            ("candidate_threshold", candidates),
            ("shot_alert_count", counts),
        ):
            path = output / f"threshold__{method}__frontier_{array_id}.npy"
            np.save(path, array, allow_pickle=False)
            artifacts.append(_artifact_record(path, base=output))
            sidecar = {
                "schema_version": "run6-threshold-frontier-array-v1",
                "protocol_id": protocol_id,
                "method_id": method,
                "array_id": array_id,
                "data_file": path.name,
                "data_sha256": sha256_file(path),
                "shape": list(array.shape),
                "dtype": array.dtype.str,
                "candidate_rule": ("[-inf] + sorted_unique_cycle_scores + [+inf]"),
                "count_rule": (
                    "strict_greater_than_with_at_most_one_notification_per_shot"
                ),
                "pair_index_range": [
                    pair_index_start,
                    pair_index_start + values.shape[0],
                ],
                "checkpoint_and_code_hashes": dict(common_hashes),
            }
            sidecar_path = path.with_suffix(".json")
            _write_canonical_json(sidecar_path, sidecar)
            artifacts.append(_artifact_record(sidecar_path, base=output))
    return artifacts


def _save_cycle_arrays(
    output: Path,
    *,
    phase: str,
    scores: Mapping[str, np.ndarray],
    log_e: Mapping[str, np.ndarray],
    log_sr: Mapping[str, np.ndarray],
    thresholds: Mapping[str, Mapping[str, Any]],
    protocol_id: str,
    run_id: str,
    pair_index_start: int,
    reference_archive_start: int,
    monitor_archive_start: int,
    common_hashes: Mapping[str, str],
    include_formal_accumulators: bool,
    accumulator_summary: ReplayAccumulatorSummary | None = None,
) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    for method in METHOD_IDS:
        applied = apply_strict_shot_threshold(
            scores[method],
            float(thresholds[method]["threshold"]),
        )
        above_threshold = scores[method] > float(thresholds[method]["threshold"])
        notification_emitted = np.zeros_like(above_threshold, dtype=np.bool_)
        rows = np.flatnonzero(applied.shot_alerts)
        if len(rows):
            notification_emitted[
                rows,
                applied.first_crossing_roles[rows],
            ] = True
        cooldown_active = np.zeros_like(above_threshold, dtype=np.bool_)
        for row in rows:
            first_role = int(applied.first_crossing_roles[row])
            cooldown_active[row, first_role + 1 :] = True
        arrays: list[tuple[str, np.ndarray, list[str]]] = [
            (
                "empirical_cycle_score",
                np.asarray(scores[method], dtype="<f8"),
                ["paired_shot", "role"],
            ),
            (
                "above_threshold",
                np.asarray(above_threshold, dtype=np.bool_),
                ["paired_shot", "role"],
            ),
            (
                "notification_emitted",
                np.asarray(notification_emitted, dtype=np.bool_),
                ["paired_shot", "role"],
            ),
            (
                "cooldown_active",
                cooldown_active,
                ["paired_shot", "role"],
            ),
        ]
        if include_formal_accumulators:
            if accumulator_summary is None:
                raise ValueError("Held formal arrays require accumulator summary.")
            e_values = np.asarray(log_e[method], dtype="<f8")
            sr_values = np.asarray(log_sr[method], dtype="<f8")
            if e_values.shape != (scores[method].shape[0],):
                raise ValueError("Formal e-process trace must be shot indexed.")
            if sr_values.shape != e_values.shape:
                raise ValueError("Formal SR trace must match the e-process trace.")
            first_e = np.zeros_like(e_values, dtype=np.bool_)
            first_sr = np.zeros_like(sr_values, dtype=np.bool_)
            if method in EXACT_METHOD_IDS:
                if not np.all(np.isfinite(e_values)) or not np.all(
                    np.isfinite(sr_values)
                ):
                    raise RuntimeError(
                        "Exact formal accumulator traces must be finite."
                    )
                e_time = accumulator_summary.proper_prior[method][
                    "first_crossing_update"
                ]
                sr_time = accumulator_summary.shiryaev_roberts[method][
                    "first_crossing_update"
                ]
                if e_time is not None:
                    if not 1 <= e_time <= len(first_e):
                        raise RuntimeError("E-process crossing time is out of range.")
                    first_e[e_time - 1] = True
                if sr_time is not None:
                    if not 1 <= sr_time <= len(first_sr):
                        raise RuntimeError("SR crossing time is out of range.")
                    first_sr[sr_time - 1] = True
            elif not np.all(np.isnan(e_values)) or not np.all(np.isnan(sr_values)):
                raise RuntimeError(
                    "Methods without a formal accumulator must emit all-NaN traces."
                )
            arrays.extend(
                [
                    ("log_eprocess", e_values, ["paired_shot"]),
                    ("log_sr", sr_values, ["paired_shot"]),
                    ("first_e_crossing", first_e, ["paired_shot"]),
                    ("first_sr_crossing", first_sr, ["paired_shot"]),
                ]
            )
        for name, values, flatten_order in arrays:
            path = output / f"{phase}__{method}__{name}.npy"
            np.save(path, values, allow_pickle=False)
            artifacts.append(_artifact_record(path, base=output))
            sidecar = {
                "schema_version": "run6-cycle-array-v1",
                "protocol_id": protocol_id,
                "run_id": run_id,
                "phase": phase,
                "method_id": method,
                "array_id": name,
                "data_file": path.name,
                "data_sha256": sha256_file(path),
                "shape": list(values.shape),
                "dtype": values.dtype.str,
                "flatten_order": flatten_order,
                "pair_index_range": [
                    pair_index_start,
                    pair_index_start + scores[method].shape[0],
                ],
                "reference_archive_start": reference_archive_start,
                "monitor_archive_start": monitor_archive_start,
                "threshold": thresholds[method]["threshold"],
                "formal_claim_scope": (
                    "not_applicable"
                    if name
                    not in {
                        "log_eprocess",
                        "log_sr",
                        "first_e_crossing",
                        "first_sr_crossing",
                    }
                    else "not_applicable_no_formal_accumulator"
                    if method not in EXACT_METHOD_IDS
                    else (
                        "diagnostic_only_on_natural_hardware; "
                        "no exchangeable hardware null asserted"
                    )
                ),
                "cooldown_semantics": (
                    "true only after the first notification through the "
                    "remainder of the same shot; all model updates continue; "
                    "the next shot starts false"
                ),
                "checkpoint_and_code_hashes": dict(common_hashes),
            }
            sidecar_path = path.with_suffix(".json")
            _write_canonical_json(sidecar_path, sidecar)
            artifacts.append(_artifact_record(sidecar_path, base=output))
    return artifacts


def _write_shot_table(
    path: Path,
    *,
    phase: str,
    pair_index_start: int,
    reference_start: int,
    monitor_start: int,
    scores: Mapping[str, np.ndarray],
    thresholds: Mapping[str, Mapping[str, Any]],
    windows: Mapping[str, list[int]] | None,
) -> None:
    fieldnames = [
        "phase",
        "method",
        "pair_index",
        "reference_archive_shot",
        "monitor_archive_shot",
        "shot_score",
        "argmax_role",
        "first_crossing_role",
        "shot_alert",
        "cumulative_alert_count",
        "rank",
        "rank_tie_archive_shot",
        "in_primary_window",
        "in_narrow_window",
        "in_wide_window",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for method in METHOD_IDS:
            applied = apply_strict_shot_threshold(
                scores[method],
                float(thresholds[method]["threshold"]),
            )
            archive_shots = monitor_start + np.arange(scores[method].shape[0])
            order = np.lexsort((archive_shots, -applied.shot_scores))
            ranks = np.empty(len(order), dtype=np.int64)
            ranks[order] = np.arange(1, len(order) + 1, dtype=np.int64)
            cumulative_alerts = np.cumsum(
                applied.shot_alerts,
                dtype=np.int64,
            )
            for pair_index in range(scores[method].shape[0]):
                monitor_shot = monitor_start + pair_index
                memberships = {
                    name: (
                        0
                        if windows is None
                        else int(windows[name][0] <= monitor_shot < windows[name][1])
                    )
                    for name in ("primary", "narrow", "wide")
                }

                writer.writerow(
                    {
                        "phase": phase,
                        "method": method,
                        "pair_index": pair_index_start + pair_index,
                        "reference_archive_shot": reference_start + pair_index,
                        "monitor_archive_shot": monitor_shot,
                        "shot_score": format(
                            float(applied.shot_scores[pair_index]),
                            ".17g",
                        ),
                        "argmax_role": int(applied.shot_score_roles[pair_index]),
                        "first_crossing_role": int(
                            applied.first_crossing_roles[pair_index]
                        ),
                        "shot_alert": int(applied.shot_alerts[pair_index]),
                        "cumulative_alert_count": int(cumulative_alerts[pair_index]),
                        "rank": int(ranks[pair_index]),
                        "rank_tie_archive_shot": monitor_shot,
                        "in_primary_window": memberships["primary"],
                        "in_narrow_window": memberships["narrow"],
                        "in_wide_window": memberships["wide"],
                    }
                )


def _event_summary(
    scores: Mapping[str, np.ndarray],
    thresholds: Mapping[str, Mapping[str, Any]],
    *,
    monitor_start: int,
    windows: Mapping[str, list[int]],
    threshold_key: str = "threshold",
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    primary_start = int(windows["primary"][0])
    for method in METHOD_IDS:
        applied = apply_strict_shot_threshold(
            scores[method],
            float(thresholds[method][threshold_key]),
        )
        alert_shots = monitor_start + np.flatnonzero(applied.shot_alerts)
        method_result: dict[str, Any] = {
            "pre_event_alert_count": int(np.sum(alert_shots < primary_start)),
            "pre_event_alert_shots": alert_shots[alert_shots < primary_start].tolist(),
            "windows": {},
        }
        for window_name in ("primary", "narrow", "wide"):
            start, stop = windows[window_name]
            inside = alert_shots[(alert_shots >= start) & (alert_shots < stop)]
            if len(inside):
                first_shot = int(inside[0])
                pair_index = first_shot - monitor_start
                first_role = int(applied.first_crossing_roles[pair_index])
            else:
                first_shot = None
                first_role = None
            method_result["windows"][window_name] = {
                "detected": bool(len(inside)),
                "first_alert_shot": first_shot,
                "first_alert_role": first_role,
            }
        result[method] = method_result
    return result


def synthetic_dry_run() -> dict[str, Any]:
    """Exercise every detector primitive without accessing a source file."""

    rng = np.random.Generator(np.random.PCG64(610699))
    reference = rng.integers(0, 2, size=(12, 2, 24), dtype=np.uint8)
    monitor = rng.integers(0, 2, size=(12, 2, 24), dtype=np.uint8)
    diagonal = DiagonalLikelihoodModel.fit(reference[:6], monitor[:6])

    differences = np.empty((6, 2, FEATURE_DIM), dtype=np.float64)
    for pair_index in range(6):
        for role in range(2):
            differences[pair_index, role] = paired_qec_contrasts(
                reference[pair_index, role],
                monitor[pair_index, role],
            ).feature_difference
    hotelling = RoleHotellingModel.fit(
        differences,
        sample_size=12,
        seed=610601,
    )
    bank = RoleIsolatedQECBank(
        role_count=2,
        diagonal_model=diagonal,
        hotelling_model=hotelling,
    )
    first_digest = bank.state_digest()
    for pair_index in range(6):
        paired_page_cusum_shot(reference[pair_index], monitor[pair_index])
        for role in range(2):
            bank.update(role, reference[pair_index, role], monitor[pair_index, role])
    clone = bank.clone()
    if clone.state_digest() != bank.state_digest():
        raise RuntimeError("Checkpoint clone digest mismatch.")
    clone.update(0, reference[6, 0], monitor[6, 0])
    if clone.state_digest() == bank.state_digest():
        raise RuntimeError("Checkpoint clone did not diverge after input.")
    return {
        "status": "synthetic_dry_run_passed",
        "seed": 610699,
        "initial_checkpoint_sha256": first_digest,
        "warmed_checkpoint_sha256": bank.state_digest(),
        "role_update_counts": bank.role_update_counts.tolist(),
        "raw_run6_values_opened": False,
    }


def run_real(args: argparse.Namespace) -> None:
    required = {
        "--config": args.config,
        "--freeze-ratification": args.freeze_ratification,
        "--data-root": args.data_root,
        "--output": args.output,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError(f"Missing real-replay arguments: {missing}.")

    repo_root = Path(__file__).resolve().parents[3]
    config_path = args.config.resolve()
    data_root = args.data_root.resolve()
    output = args.output.resolve()
    config = load_google_lock(config_path)
    verify_freeze_ratification(
        args.freeze_ratification.resolve(),
        repo_root=repo_root,
        config_path=config_path,
        config=config,
    )
    assert_no_outcome_paths(
        [
            data_root / config["source"]["detection_event_file"],
            data_root / "circuit_ideal.stim",
        ]
    )
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Output directory exists and is not empty.")
    output.mkdir(parents=True, exist_ok=True)

    performance_start = time.perf_counter()
    archive = (
        repo_root
        / "experiments/data/run6/google_2022/google_qec3v5_experiment_data.zip"
    )
    if archive.stat().st_size != config["source"]["archive_bytes"]:
        raise ValueError("Source archive byte count changed.")
    if sha256_file(archive) != config["source"]["sha256"]:
        raise ValueError("Source archive SHA-256 changed.")
    circuit = data_root / "circuit_ideal.stim"
    detection_events = data_root / config["source"]["detection_event_file"]
    archive_member_root = config["source"]["experiment"]
    member_hashes = {
        "circuit_ideal.stim": _sha256_zip_member(
            archive,
            f"{archive_member_root}/circuit_ideal.stim",
        ),
        config["source"]["detection_event_file"]: _sha256_zip_member(
            archive,
            f"{archive_member_root}/{config['source']['detection_event_file']}",
        ),
    }
    for filename, expected_hash in member_hashes.items():
        extracted = data_root / filename
        if sha256_file(extracted) != expected_hash:
            raise ValueError(
                f"Extracted {filename} is not byte-identical to the verified ZIP."
            )
    layout = parse_stim_detector_layout(
        circuit,
        expected_roles=ROLE_COUNT,
        expected_checks_per_role=24,
    )
    integrity_and_layout_seconds = time.perf_counter() - performance_start

    start_time = time.time()
    # This is the first source-value access, after every freeze check above.
    validation_read_start = time.perf_counter()
    validation_a = read_b8_detector_shots(
        detection_events,
        layout,
        start=0,
        stop=10_000,
        total_shots=500_000,
    )
    validation_b = read_b8_detector_shots(
        detection_events,
        layout,
        start=10_000,
        stop=20_000,
        total_shots=500_000,
    )
    validation_read_seconds = time.perf_counter() - validation_read_start
    warm_start = time.perf_counter()
    warm, checkpoint_hash = warm_bank(
        validation_a[:5_000],
        validation_b[:5_000],
    )
    threshold_bank = warm.clone()
    held_bank = warm.clone()
    if (
        threshold_bank.state_digest() != checkpoint_hash
        or held_bank.state_digest() != checkpoint_hash
    ):
        raise RuntimeError("Threshold/held checkpoint clone mismatch.")
    warm_fit_replay_seconds = time.perf_counter() - warm_start

    threshold_replay_start = time.perf_counter()
    (
        threshold_scores,
        threshold_log_e,
        threshold_log_sr,
        _threshold_accumulators,
    ) = replay_scores(
        threshold_bank,
        validation_a[5_000:],
        validation_b[5_000:],
        with_accumulators=False,
    )
    threshold_replay_seconds = time.perf_counter() - threshold_replay_start
    thresholds = _threshold_payload(threshold_scores, max_alerts=2)
    _write_canonical_json(output / "thresholds.json", thresholds)

    config_hash = sha256_file(config_path)
    method_spec_hash = sha256_file(repo_root / config["normative_method_spec"]["path"])
    detector_script_hash = sha256_file(__file__)
    common_hashes = {
        "config_sha256": config_hash,
        "method_spec_sha256": method_spec_hash,
        "detector_script_sha256": detector_script_hash,
        "warm_checkpoint_sha256": checkpoint_hash,
        "threshold_final_checkpoint_sha256": threshold_bank.state_digest(),
        "freeze_ratification_sha256": sha256_file(args.freeze_ratification),
        "deviation_ledger_sha256": sha256_file(repo_root / config["deviation_ledger"]),
        "python_environment_lock_sha256": sha256_file(
            repo_root / "experiments/run6/configs/python_environment_lock.txt"
        ),
        "freeze_manifest_sha256": sha256_file(
            repo_root / "experiments/run6/freeze_manifest.json"
        ),
    }
    artifacts = _save_threshold_frontiers(
        output,
        scores=threshold_scores,
        protocol_id=config["protocol_id"],
        common_hashes=common_hashes,
        pair_index_start=5_000,
    )
    artifacts.extend(
        _save_cycle_arrays(
            output,
            phase="threshold",
            scores=threshold_scores,
            log_e=threshold_log_e,
            log_sr=threshold_log_sr,
            thresholds=thresholds,
            protocol_id=config["protocol_id"],
            run_id="google2022-canonical-detector",
            pair_index_start=5_000,
            reference_archive_start=5_000,
            monitor_archive_start=15_000,
            common_hashes=common_hashes,
            include_formal_accumulators=False,
            accumulator_summary=None,
        )
    )
    artifacts.append(_artifact_record(output / "thresholds.json", base=output))
    threshold_shot_table = output / "threshold_shots.csv"
    _write_shot_table(
        threshold_shot_table,
        phase="threshold",
        pair_index_start=5_000,
        reference_start=5_000,
        monitor_start=15_000,
        scores=threshold_scores,
        thresholds=thresholds,
        windows=None,
    )
    artifacts.append(_artifact_record(threshold_shot_table, base=output))
    threshold_freeze_path = output / "threshold_stage_manifest.json"
    _write_canonical_json(
        threshold_freeze_path,
        {
            "schema_version": "run6-google-threshold-stage-v1",
            "protocol_id": config["protocol_id"],
            "held_values_decoded_or_scored": False,
            "config_sha256": config_hash,
            "method_spec_sha256": method_spec_hash,
            "detector_script_sha256": detector_script_hash,
            "warm_checkpoint_sha256": checkpoint_hash,
            "threshold_final_checkpoint_sha256": threshold_bank.state_digest(),
            "threshold_table_sha256": sha256_file(output / "thresholds.json"),
            "threshold_artifacts": list(artifacts),
        },
    )
    artifacts.append(_artifact_record(threshold_freeze_path, base=output))
    threshold_serialization_seconds = (
        time.perf_counter() - threshold_replay_start - threshold_replay_seconds
    )

    # Held values are opened only after the complete threshold frontier and
    # threshold-stage manifest have been persisted.
    held_read_start = time.perf_counter()
    held_reference = read_b8_detector_shots(
        detection_events,
        layout,
        start=20_000,
        stop=40_000,
        total_shots=500_000,
    )
    held_monitor = read_b8_detector_shots(
        detection_events,
        layout,
        start=40_000,
        stop=60_000,
        total_shots=500_000,
    )
    held_read_seconds = time.perf_counter() - held_read_start
    held_replay_start = time.perf_counter()
    held_scores, held_log_e, held_log_sr, held_accumulators = replay_scores(
        held_bank,
        held_reference,
        held_monitor,
        with_accumulators=True,
        horizon=20_000,
    )
    held_replay_seconds = time.perf_counter() - held_replay_start
    held_common_hashes = {
        **common_hashes,
        "held_final_checkpoint_sha256": held_bank.state_digest(),
    }
    canonical_replay_digest = _replay_digest(
        held_scores,
        held_log_e,
        held_log_sr,
        held_accumulators,
    )
    held_replay_timings = [held_replay_seconds]
    repeat_digests = [canonical_replay_digest]
    for _ in range(2):
        repeat_bank = warm.clone()
        repeat_start = time.perf_counter()
        (
            repeat_scores,
            repeat_log_e,
            repeat_log_sr,
            repeat_accumulators,
        ) = replay_scores(
            repeat_bank,
            held_reference,
            held_monitor,
            with_accumulators=True,
            horizon=20_000,
        )
        held_replay_timings.append(time.perf_counter() - repeat_start)
        repeat_digest = _replay_digest(
            repeat_scores,
            repeat_log_e,
            repeat_log_sr,
            repeat_accumulators,
        )
        repeat_digests.append(repeat_digest)
        if (
            repeat_digest != canonical_replay_digest
            or repeat_bank.state_digest() != held_bank.state_digest()
        ):
            raise RuntimeError("Held timing replay is not byte-deterministic.")
        del (
            repeat_scores,
            repeat_log_e,
            repeat_log_sr,
            repeat_accumulators,
            repeat_bank,
        )
    held_serialization_start = time.perf_counter()
    artifacts.extend(
        _save_cycle_arrays(
            output,
            phase="held",
            scores=held_scores,
            log_e=held_log_e,
            log_sr=held_log_sr,
            thresholds=thresholds,
            protocol_id=config["protocol_id"],
            run_id="google2022-canonical-detector",
            pair_index_start=0,
            reference_archive_start=20_000,
            monitor_archive_start=40_000,
            common_hashes=held_common_hashes,
            include_formal_accumulators=True,
            accumulator_summary=held_accumulators,
        )
    )
    shot_table = output / "held_shots.csv"
    _write_shot_table(
        shot_table,
        phase="held",
        pair_index_start=0,
        reference_start=20_000,
        monitor_start=40_000,
        scores=held_scores,
        thresholds=thresholds,
        windows=config["event_windows"],
    )
    artifacts.append(_artifact_record(shot_table, base=output))
    event_summary = _event_summary(
        held_scores,
        thresholds,
        monitor_start=40_000,
        windows=config["event_windows"],
    )
    _write_canonical_json(output / "event_summary_detector_only.json", event_summary)
    artifacts.append(
        _artifact_record(output / "event_summary_detector_only.json", base=output)
    )
    secondary_event_summary = _event_summary(
        held_scores,
        thresholds,
        monitor_start=40_000,
        windows=config["event_windows"],
        threshold_key="secondary_zero_alert_threshold",
    )
    secondary_event_path = output / "secondary_event_summary_detector_only.json"
    _write_canonical_json(secondary_event_path, secondary_event_summary)
    artifacts.append(_artifact_record(secondary_event_path, base=output))
    component_summary_path = output / "formal_component_summary.json"
    _write_canonical_json(
        component_summary_path,
        {
            "schema_version": "run6-formal-component-summary-v1",
            "held_trace_interpretation": (
                "diagnostic_only_on_natural_hardware; exact calibration is "
                "assessed only by the separately randomized complete-pair audit"
            ),
            "proper_prior": held_accumulators.proper_prior,
            "shiryaev_roberts": held_accumulators.shiryaev_roberts,
            "expert_metadata": held_accumulators.expert_metadata,
        },
    )
    artifacts.append(_artifact_record(component_summary_path, base=output))
    held_serialization_seconds = time.perf_counter() - held_serialization_start

    exposure_resources: dict[str, ResourceCounts] = {
        "fit_warmup": paired_resource_counts(5_000),
        "threshold": paired_resource_counts(5_000),
        "held": paired_resource_counts(20_000),
    }
    resources = {
        "record_exposure": {
            name: vars(counts) for name, counts in exposure_resources.items()
        },
        "high_level_operations": {
            "fit_warmup": _operation_ledger(
                paired_shots=5_000,
                role_updates_before=0,
            ),
            "threshold": _operation_ledger(
                paired_shots=5_000,
                role_updates_before=5_000,
            ),
            "held": _operation_ledger(
                paired_shots=20_000,
                role_updates_before=5_000,
            ),
            "m2_covariance_fits": 1,
            "m2_precision_matrix_constructions": 1,
            "m2_fit_observations_used": 20_000,
            "held_joint_replay_repetitions": 3,
            "extra_timing_replay_role_updates": 2 * 20_000 * ROLE_COUNT,
        },
        "warm_checkpoint_storage": _state_resource_summary(warm),
        "held_final_checkpoint_storage": _state_resource_summary(held_bank),
        "formal_accumulator": {
            "time_unit": "complete_paired_shot",
            "held_updates": 20_000,
            "role_prior": "uniform_1_over_51",
            "within_shot_factor_compounding": False,
        },
        "output_bytes_before_manifest": sum(
            int(artifact["bytes"]) for artifact in artifacts
        ),
    }
    performance = {
        "canonical_joint_pipeline_only": True,
        "not_a_per_method_speed_comparison": True,
        "integrity_and_layout_seconds": integrity_and_layout_seconds,
        "validation_read_seconds": validation_read_seconds,
        "warm_fit_replay_seconds": warm_fit_replay_seconds,
        "threshold_replay_seconds": threshold_replay_seconds,
        "threshold_serialization_seconds": threshold_serialization_seconds,
        "held_read_seconds": held_read_seconds,
        "held_replay_seconds": held_replay_seconds,
        "held_joint_replay_all_three_seconds": held_replay_timings,
        "held_joint_replay_median_seconds": statistics.median(held_replay_timings),
        "held_joint_replay_digests": repeat_digests,
        "held_serialization_seconds": held_serialization_seconds,
        "elapsed_before_manifest_seconds": time.perf_counter() - performance_start,
        "peak_rss_kib_linux_ru_maxrss": int(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        ),
        "relative_method_speed_claim_authorized": False,
    }
    manifest = {
        "schema_version": "run6-google-detector-freeze-v1",
        "protocol_id": config["protocol_id"],
        "detector_only": True,
        "outcome_accessed": False,
        "outcome_join_authorized": False,
        "git_commit": _git_commit(repo_root),
        "config_sha256": config_hash,
        "method_spec_sha256": method_spec_hash,
        "detector_script_sha256": detector_script_hash,
        "freeze_ratification_sha256": sha256_file(args.freeze_ratification),
        "deviation_ledger": {
            "path": config["deviation_ledger"],
            "sha256": sha256_file(repo_root / config["deviation_ledger"]),
        },
        "circuit_sha256": sha256_file(circuit),
        "detector_layout_index_sha256": hashlib.sha256(
            layout.ordered_declaration_indices.tobytes()
        ).hexdigest(),
        "warm_checkpoint_sha256": checkpoint_hash,
        "threshold_checkpoint_sha256": threshold_bank.state_digest(),
        "held_final_checkpoint_sha256": held_bank.state_digest(),
        "source_archive_sha256": config["source"]["sha256"],
        "source_archive_bytes": config["source"]["archive_bytes"],
        "verified_zip_member_sha256": member_hashes,
        "detection_file_bytes": detection_events.stat().st_size,
        "threshold_table_sha256": sha256_file(output / "thresholds.json"),
        "artifacts": artifacts,
        "resources": resources,
        "performance": performance,
        "environment": environment_fingerprint(),
        "command": sys.argv,
        "started_unix": start_time,
        "finished_unix": time.time(),
    }
    _write_canonical_json(output / "detector_freeze_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


def main() -> None:
    args = parse_args()
    if args.dry_run:
        print(json.dumps(synthetic_dry_run(), indent=2, sort_keys=True))
        return
    run_real(args)


if __name__ == "__main__":
    main()
