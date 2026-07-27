#!/usr/bin/env python3
"""Frozen Pittsburgh snapshot replay for Run 6.

The real-data path is deliberately fail closed.  It first validates the
metadata-only Pittsburgh lock, verifies a committed freeze ratification and
every hash listed by that ratification, and records raw hashes of the exact
held ``bitstrings.json`` files.  Only then may it parse measurement values.

Pittsburgh repetition-code paths have ``q = distance - 1`` checks, unlike the
fixed 24-check Google arm.  This runner therefore implements the locked
dimension-adapted feature maps locally while reusing the Run 6 score,
e-process, QASM, and detection-event primitives.  Logical states and round
roles remain separate; all adaptive scores are predictable (score first,
learn second).
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import re
import resource
import subprocess
import sys
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any

import aoc.qec_real as qec_real_module
import aoc.run6_protocol as run6_protocol_module
import aoc.space as space_module
import aoc.space_qec as space_qec_module
import numpy as np
import validate_pnnl_lock as pnnl_validator_module
from aoc.qec_real import parse_qasm_register_maps, repetition_detection_events
from aoc.run6_protocol import (
    RUN6_REQUIRED_FREEZE_PATHS,
    canonical_json_bytes,
    environment_fingerprint,
    load_strict_json,
    require_exact_keys,
    sha256_file,
    verify_committed_freeze_chain,
)
from aoc.space import (
    EWMASpectralWitness,
    EWMATopKWitness,
    PairwiseOnlineLogistic,
    ProperUniformStartEProcessBank,
    linear_bet_factors,
    validate_bounded_score,
)
from aoc.space_qec import (
    LOGISTIC_LEARNING_RATES,
    SIGNED_BET_MAGNITUDES,
    SPARSE_HALF_LIVES,
    SPARSE_K_VALUES,
    SPECTRAL_HALF_LIVES,
    SPECTRAL_RANKS,
)
from validate_pnnl_lock import validate_lock

METHOD_ORDER = (
    "dfr",
    "online_logistic",
    "space_sparse",
    "space_spectral",
    "space_composite",
)
PACKAGE_LOCK_RELATIVE = "experiments/run6/configs/python_environment_lock.txt"
LOG_E_100 = float(np.log(100.0))
SCORE_TOLERANCE = 1e-12
EIGENVALUE_TOLERANCE = 1e-10
SPECTRAL_VALIDATION_TOLERANCE = 1e-9
SPECTRAL_UPDATE_STRIDE = 8
PNNL_CONFIG_KEYS = frozenset(
    {
        "protocol_id",
        "status",
        "data_status",
        "source",
        "roles",
        "normative_pittsburgh_manifest",
        "normative_auxiliary_spec",
        "cohort_filter",
        "stream_construction",
        "features_and_methods",
        "claim_labels",
        "uncertainty_unit",
        "advantage_retention_rule",
    }
)
PNNL_SOURCE_KEYS = frozenset(
    {
        "record_url",
        "doi",
        "license",
        "release_version",
        "backends",
    }
)
PNNL_ROLE_KEYS = frozenset(
    {
        "pilot_backend",
        "held_backend",
        "no_exact_repeat_backend",
    }
)
NORMATIVE_ARTIFACT_KEYS = frozenset({"path", "sha256"})
COHORT_FIELDS = (
    "cohort_id",
    "distance",
    "rounds",
    "basis",
    "register_suffix",
    "data_qubits",
    "syndrome_qubits",
    "oriented_path",
    "early_snapshot_id",
    "late_snapshot_id",
    "m",
    "raw_qasm_pair_identical",
    "normalized_qasm_pair_identical_audit_only",
    "claim_label",
    "calibration_pair_id",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the frozen replay command line."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/run6/configs/pnnl_snapshot_locked.json"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("experiments/run6/configs/pnnl_pittsburgh_locked.json"),
    )
    parser.add_argument("--freeze-ratification", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


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


def _artifact_record(path: Path, *, base: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(base).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _strict_positive_int(value: Any, *, context: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, np.integer),
    ):
        raise TypeError(f"{context} must be an integer.")
    result = int(value)
    if result < 1:
        raise ValueError(f"{context} must be positive.")
    return result


def _strict_nonnegative_int(value: Any, *, context: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, np.integer),
    ):
        raise TypeError(f"{context} must be an integer.")
    result = int(value)
    if result < 0:
        raise ValueError(f"{context} must be nonnegative.")
    return result


def validate_pnnl_config(config: Mapping[str, Any]) -> None:
    """Reject silent defaults and changes to the locked auxiliary design."""

    require_exact_keys(config, PNNL_CONFIG_KEYS, context="PNNL config")
    require_exact_keys(config["source"], PNNL_SOURCE_KEYS, context="PNNL source")
    require_exact_keys(config["roles"], PNNL_ROLE_KEYS, context="PNNL roles")
    for field in ("normative_pittsburgh_manifest", "normative_auxiliary_spec"):
        require_exact_keys(
            config[field],
            NORMATIVE_ARTIFACT_KEYS,
            context=f"PNNL {field}",
        )
    if config["protocol_id"] != "run6-pnnl-snapshot-v2":
        raise ValueError("Unexpected PNNL protocol_id.")
    if config["status"] not in {
        "executable_lock_pending_final_freeze_commit",
        "frozen_before_held_value_access",
    }:
        raise ValueError("Unexpected PNNL lock status.")
    if (
        config["data_status"]
        != "constructed_boundary_between_real_hardware_snapshot_cohorts"
    ):
        raise ValueError("PNNL data-status claim changed.")
    if config["roles"] != {
        "pilot_backend": "ibm_kingston",
        "held_backend": "ibm_pittsburgh",
        "no_exact_repeat_backend": "ibm_fez",
    }:
        raise ValueError("PNNL backend roles changed.")
    if config["source"] != {
        "record_url": "https://zenodo.org/records/20768087",
        "doi": "10.5281/zenodo.20768087",
        "license": "CC BY 4.0",
        "release_version": "0.1",
        "backends": ["ibm_kingston", "ibm_pittsburgh", "ibm_fez"],
    }:
        raise ValueError("PNNL source identity changed.")
    normative_paths = {
        "normative_pittsburgh_manifest": (
            "experiments/run6/configs/pnnl_pittsburgh_locked.json"
        ),
        "normative_auxiliary_spec": (
            "references/run6_pnnl_locked_manifest_recommendations.md"
        ),
    }
    for field, expected_path in normative_paths.items():
        if config[field]["path"] != expected_path:
            raise ValueError(f"PNNL {field} path changed.")
        digest = config[field]["sha256"]
        if config["status"] == "frozen_before_held_value_access":
            if (
                not isinstance(digest, str)
                or re.fullmatch(
                    r"[0-9a-f]{64}",
                    digest,
                )
                is None
            ):
                raise ValueError(f"PNNL {field} needs a frozen SHA-256.")
        elif digest != "TO_BE_FILLED_AFTER_FINAL_NO_HELD_REAUDIT":
            raise ValueError(f"PNNL {field} pending hash marker changed.")
    if config["cohort_filter"] != {
        "same_backend": True,
        "same_distance_rounds_basis": True,
        "same_oriented_qasm_derived_path": True,
        "state0_state1_path_must_agree": True,
        "distinct_full_calibration_hash": True,
        "minimum_shots_per_state_per_cohort": 2048,
        "choose_dates": "earliest_and_latest_backend_property_date",
        "choose_duplicate_date_job": ("largest_shot_count_then_lexicographic_path"),
        "do_not_infer_path_from_register_suffix": True,
    }:
        raise ValueError("PNNL cohort filter changed.")
    if config["stream_construction"] != {
        "logical_states_are_separate_replicates": True,
        "baseline_partition": [
            "reference_pre",
            "monitor_pre",
            "reference_post",
        ],
        "partition_sizes": "floor(earlier_cohort_shots/3) each",
        "post_monitor": "first partition_size shots of later cohort",
        "pre_pairs": "reference_pre[i] with monitor_pre[i]",
        "post_pairs": "reference_post[i] with post_monitor[i]",
        "boundary": "after all pre pairs",
        "primary_detection_events": "r by (d-1), no terminal detector",
        "terminal_detector_ablation": False,
    }:
        raise ValueError("PNNL stream construction changed.")
    if (
        config["features_and_methods"]
        != (
            "dimension-adapted definitions and fixed space_composite from "
            "the hash-pinned Google v2 config and Pittsburgh manifest"
        )
        or config["claim_labels"]
        != {
            "byte_identical_complete_qasm": (
                "circuit_controlled_cross_property_snapshot"
            ),
            "different_complete_qasm": "circuit_and_hardware_domain_shift",
            "temporal_drift": "forbidden",
        }
        or config["uncertainty_unit"] != "qasm_derived_physical_path_by_snapshot_pair"
    ):
        raise ValueError("PNNL method/claim contract changed.")
    retention = config["advantage_retention_rule"]
    if retention != {
        "comparison_methods": list(METHOD_ORDER),
        "target_method": "space_composite",
        "required_direction": (
            "fixed space_composite no-worse pre-false-alarm count and "
            "strictly lower macro restricted delay versus both dfr and "
            "online_logistic"
        ),
        "post_selection": "forbidden",
    }:
        raise ValueError("PNNL retention rule changed.")


def load_pnnl_config(path: str | Path) -> dict[str, Any]:
    config = load_strict_json(path)
    validate_pnnl_config(config)
    return config


def verify_freeze_ratification(
    path: Path,
    *,
    repo_root: Path,
    config_path: Path,
    config: Mapping[str, Any],
    manifest_path: Path,
) -> dict[str, Any]:
    """Verify the external post-freeze gate before held payload access."""

    expected_module_paths = {
        qec_real_module: "experiments/aoc/qec_real.py",
        run6_protocol_module: "experiments/aoc/run6_protocol.py",
        space_module: "experiments/aoc/space.py",
        space_qec_module: "experiments/aoc/space_qec.py",
        pnnl_validator_module: ("experiments/run6/scripts/validate_pnnl_lock.py"),
    }
    for module, relative in expected_module_paths.items():
        observed = Path(module.__file__).resolve()
        expected = (repo_root / relative).resolve()
        if observed != expected:
            raise RuntimeError(
                f"Runtime imported {module.__name__} from {observed}, "
                f"not ratified source {expected}."
            )

    expected_config_path = (
        repo_root / "experiments/run6/configs/pnnl_snapshot_locked.json"
    ).resolve()
    expected_manifest_path = (
        repo_root / "experiments/run6/configs/pnnl_pittsburgh_locked.json"
    ).resolve()
    if (
        config_path.resolve() != expected_config_path
        or manifest_path.resolve() != expected_manifest_path
    ):
        raise ValueError("Real replay requires the canonical PNNL lock paths.")
    if config["status"] != "frozen_before_held_value_access":
        raise ValueError("PNNL executable configuration has not been frozen.")

    manifest_relative = config["normative_pittsburgh_manifest"]["path"]
    if manifest_relative != manifest_path.relative_to(repo_root).as_posix():
        raise ValueError("PNNL config points to a different manifest.")
    if config["normative_pittsburgh_manifest"]["sha256"] != sha256_file(manifest_path):
        raise ValueError("Normative Pittsburgh manifest hash changed.")
    spec_path = repo_root / config["normative_auxiliary_spec"]["path"]
    if config["normative_auxiliary_spec"]["sha256"] != sha256_file(spec_path):
        raise ValueError("Normative Pittsburgh specification hash changed.")

    manifest = load_strict_json(manifest_path)
    for relative, expected_hash in manifest["parent_artifacts"].items():
        parent = repo_root / relative
        if not parent.is_file() or sha256_file(parent) != expected_hash:
            raise ValueError(f"PNNL manifest parent artifact changed: {relative}")
    google_relative = manifest["features_and_methods"]["inherited_from"]
    google_config = load_strict_json(repo_root / google_relative)
    expected_threads = google_config["numeric_policy"]["thread_environment"]
    return verify_committed_freeze_chain(
        path,
        repo_root=repo_root,
        required_paths=RUN6_REQUIRED_FREEZE_PATHS,
        expected_environment=environment_fingerprint(),
        expected_thread_environment=expected_threads,
    )


@dataclass(frozen=True)
class Cohort:
    """One immutable Pittsburgh path/snapshot comparison."""

    cohort_id: str
    distance: int
    rounds: int
    basis: str
    register_suffix: str
    data_qubits: tuple[int, ...]
    syndrome_qubits: tuple[int, ...]
    oriented_path: tuple[int, ...]
    early_snapshot_id: str
    late_snapshot_id: str
    m: int
    claim_label: str
    calibration_pair_id: str

    @property
    def q(self) -> int:
        return self.distance - 1

    @property
    def fit_shots(self) -> int:
        return self.m // 2

    @property
    def pre_surveillance_shots(self) -> int:
        return self.m - self.fit_shots

    @property
    def surveillance_shots(self) -> int:
        return self.pre_surveillance_shots + self.m

    @property
    def surveillance_updates(self) -> int:
        """Return formal e-process updates (one per paired shot)."""

        return self.surveillance_shots


def parse_cohorts(manifest: Mapping[str, Any]) -> tuple[Cohort, ...]:
    """Parse and cross-check the machine-readable cohort rows."""

    if tuple(manifest["cohort_row_schema"]) != COHORT_FIELDS:
        raise ValueError("Unexpected cohort row schema.")
    order = tuple(manifest["cohort_order"])
    rows = manifest["cohort_pairs"]
    if not isinstance(rows, list) or len(rows) != len(order):
        raise ValueError("Cohort rows/order length mismatch.")
    cohorts: list[Cohort] = []
    for index, raw in enumerate(rows):
        if not isinstance(raw, list) or len(raw) != len(COHORT_FIELDS):
            raise ValueError(f"Malformed cohort row {index}.")
        row = dict(zip(COHORT_FIELDS, raw))
        if row["cohort_id"] != order[index]:
            raise ValueError("Cohort rows are not in locked order.")
        distance = _strict_positive_int(
            row["distance"],
            context=f"cohort {index} distance",
        )
        rounds = _strict_positive_int(
            row["rounds"],
            context=f"cohort {index} rounds",
        )
        m = _strict_positive_int(row["m"], context=f"cohort {index} m")
        if row["basis"] not in {"X", "Z"}:
            raise ValueError("Cohort basis must be X or Z.")
        data = tuple(
            _strict_nonnegative_int(value, context="data qubit")
            for value in row["data_qubits"]
        )
        syndrome = tuple(
            _strict_nonnegative_int(value, context="syndrome qubit")
            for value in row["syndrome_qubits"]
        )
        oriented = tuple(
            _strict_nonnegative_int(value, context="oriented path qubit")
            for value in row["oriented_path"]
        )
        if len(data) != distance or len(syndrome) != distance - 1:
            raise ValueError("Cohort path dimensions disagree.")
        expected_oriented: list[int] = []
        for check_index, check in enumerate(syndrome):
            expected_oriented.extend((data[check_index], check))
        expected_oriented.append(data[-1])
        if oriented != tuple(expected_oriented):
            raise ValueError("Cohort oriented path is inconsistent.")
        if (
            row["raw_qasm_pair_identical"] is not False
            or row["normalized_qasm_pair_identical_audit_only"] is not False
            or row["claim_label"] != "circuit_and_hardware_domain_shift"
        ):
            raise ValueError("The locked Pittsburgh claim class changed.")
        cohorts.append(
            Cohort(
                cohort_id=row["cohort_id"],
                distance=distance,
                rounds=rounds,
                basis=row["basis"],
                register_suffix=row["register_suffix"],
                data_qubits=data,
                syndrome_qubits=syndrome,
                oriented_path=oriented,
                early_snapshot_id=row["early_snapshot_id"],
                late_snapshot_id=row["late_snapshot_id"],
                m=m,
                claim_label=row["claim_label"],
                calibration_pair_id=row["calibration_pair_id"],
            )
        )
    return tuple(cohorts)


def dimension_adapted_features(bits: np.ndarray) -> np.ndarray:
    """Return ``[e_i; 1{e_i=e_j}]`` for an arbitrary check dimension."""

    values = np.asarray(bits)
    if values.ndim < 1 or values.shape[-1] < 1:
        raise ValueError("bits must have a nonempty final check dimension.")
    if np.any((values != 0) & (values != 1)):
        raise ValueError("bits must be binary.")
    numeric = values.astype(np.float64, copy=False)
    q = numeric.shape[-1]
    left, right = np.triu_indices(q, k=1)
    equality = numeric[..., left] == numeric[..., right]
    return np.concatenate((numeric, equality.astype(np.float64)), axis=-1)


def dimension_adapted_density(bits: np.ndarray) -> np.ndarray:
    """Return ``z z.T/q`` for arbitrary binary check vectors."""

    values = np.asarray(bits)
    if values.ndim < 1 or values.shape[-1] < 1:
        raise ValueError("bits must have a nonempty final check dimension.")
    if np.any((values != 0) & (values != 1)):
        raise ValueError("bits must be binary.")
    spins = 1.0 - 2.0 * values.astype(np.float64, copy=False)
    q = values.shape[-1]
    return np.einsum("...i,...j->...ij", spins, spins, optimize=True) / q


def eligible_sparse_k(q: int) -> tuple[int, ...]:
    """Return the preregistered sparse supports valid at dimension ``q``."""

    dimension = q * (q + 1) // 2
    return tuple(k for k in SPARSE_K_VALUES if k <= dimension)


def method_component_weights(q: int) -> dict[str, np.ndarray]:
    """Return exact fixed within-method priors for one path dimension."""

    sparse_count = (
        len(SPARSE_HALF_LIVES) * len(eligible_sparse_k(q)) * len(SIGNED_BET_MAGNITUDES)
    )
    spectral_count = (
        len(SPECTRAL_HALF_LIVES) * len(SPECTRAL_RANKS) * len(SIGNED_BET_MAGNITUDES)
    )
    counts = {
        "dfr": 2 * len(SIGNED_BET_MAGNITUDES),
        "online_logistic": (len(LOGISTIC_LEARNING_RATES) * len(SIGNED_BET_MAGNITUDES)),
        "space_sparse": sparse_count,
        "space_spectral": spectral_count,
    }
    weights = {
        method: np.full(count, 1.0 / count, dtype=np.float64)
        for method, count in counts.items()
    }
    weights["space_composite"] = np.concatenate(
        (
            0.5 * weights["space_sparse"],
            0.5 * weights["space_spectral"],
        )
    )
    return weights


def shot_component_weights(q: int, roles: int) -> dict[str, np.ndarray]:
    """Expand base experts with an explicit uniform round-role prior."""

    role_count = _strict_positive_int(roles, context="roles")
    return {
        method: np.tile(weights / role_count, role_count)
        for method, weights in method_component_weights(q).items()
    }


def _stable_rank_one_from_decomposition(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
) -> np.ndarray:
    """Apply the locked deterministic tie rule to one eigendecomposition."""

    maximum = float(eigenvalues[-1])
    q = len(eigenvalues)
    if maximum <= EIGENVALUE_TOLERANCE:
        return np.zeros((q, q), dtype=np.complex128)
    tied = eigenvalues >= maximum - EIGENVALUE_TOLERANCE
    top_space = eigenvectors[:, tied]
    projector = top_space @ top_space.conj().T
    diagonal = np.clip(np.real(np.diag(projector)), 0.0, None)
    largest = float(np.max(diagonal))
    anchors = np.flatnonzero(diagonal >= largest - EIGENVALUE_TOLERANCE)
    anchor = int(anchors[0])
    vector = projector[:, anchor] / np.sqrt(diagonal[anchor])
    effect = np.outer(vector, vector.conj())
    return 0.5 * (effect + effect.conj().T)


def _stable_rank_one_effect(matrix: np.ndarray) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    return _stable_rank_one_from_decomposition(eigenvalues, eigenvectors)


class _SharedDimensionSparse:
    """One EWMA shared by every eligible top-k expert at one half-life."""

    def __init__(
        self,
        dimension: int,
        *,
        half_life: float,
        k_values: tuple[int, ...],
    ) -> None:
        self.k_values = k_values
        self.state = EWMATopKWitness(
            dimension,
            half_life=half_life,
            k=1,
            tolerance=SCORE_TOLERANCE,
        )

    def update(self, difference: np.ndarray) -> np.ndarray:
        direction = self.state.ewma
        if np.max(np.abs(direction)) <= SCORE_TOLERANCE:
            scores = np.zeros(len(self.k_values), dtype=np.float64)
        else:
            order = np.argsort(-np.abs(direction), kind="stable")
            signs = np.where(direction[order] < 0.0, -1.0, 1.0)
            cumulative = np.cumsum(signs * difference[order])
            sparse_k = np.asarray(self.k_values, dtype=np.int64)
            scores = cumulative[sparse_k - 1] / sparse_k
        self.state.update(difference)
        return np.asarray(
            [
                validate_bounded_score(
                    score,
                    tolerance=SCORE_TOLERANCE,
                )
                for score in scores
            ],
            dtype=np.float64,
        )


class _SharedDimensionSpectral:
    """One EWMA/eigendecomposition shared by rank-one and Jordan effects."""

    _DISABLED_PRIMITIVE_STRIDE = 2**62

    def __init__(self, q: int, *, half_life: float) -> None:
        self.state = EWMASpectralWitness(
            q,
            half_life=half_life,
            rank="positive",
            update_stride=self._DISABLED_PRIMITIVE_STRIDE,
            eigenvalue_tolerance=EIGENVALUE_TOLERANCE,
            validation_tolerance=SPECTRAL_VALIDATION_TOLERANCE,
        )
        self.rank_one_effect = np.zeros((q, q), dtype=np.complex128)
        self.positive_effect = np.zeros((q, q), dtype=np.complex128)

    def effects(self) -> np.ndarray:
        return np.stack((self.rank_one_effect, self.positive_effect))

    def _fit_effects(self) -> None:
        eigenvalues, eigenvectors = np.linalg.eigh(self.state.ewma)
        positive = eigenvalues > EIGENVALUE_TOLERANCE
        if np.any(positive):
            basis = eigenvectors[:, positive]
            effect = basis @ basis.conj().T
            self.positive_effect = 0.5 * (effect + effect.conj().T)
        else:
            self.positive_effect.fill(0.0)
        self.rank_one_effect = _stable_rank_one_from_decomposition(
            eigenvalues,
            eigenvectors,
        )

    def update(self, difference: np.ndarray) -> np.ndarray:
        scores = np.asarray(
            [
                validate_bounded_score(
                    float(np.real(np.trace(effect @ difference))),
                    tolerance=SPECTRAL_VALIDATION_TOLERANCE,
                )
                for effect in self.effects()
            ],
            dtype=np.float64,
        )
        self.state.update(difference)
        if self.state.time % SPECTRAL_UPDATE_STRIDE == 0:
            self._fit_effects()
        return scores


class DimensionAdaptedBank:
    """Role-isolated predictable DFR/logistic/sparse/spectral factor bank."""

    def __init__(self, *, q: int, role_count: int) -> None:
        self.q = _strict_positive_int(q, context="q")
        self.role_count = _strict_positive_int(role_count, context="role_count")
        self.feature_dimension = self.q * (self.q + 1) // 2
        self.sparse_k = eligible_sparse_k(self.q)
        self._logistic = [
            [
                PairwiseOnlineLogistic(
                    self.feature_dimension,
                    learning_rate=rate,
                    l2=1e-4,
                    tolerance=SCORE_TOLERANCE,
                )
                for rate in LOGISTIC_LEARNING_RATES
            ]
            for _ in range(self.role_count)
        ]
        self._sparse = [
            [
                _SharedDimensionSparse(
                    self.feature_dimension,
                    half_life=half_life,
                    k_values=self.sparse_k,
                )
                for half_life in SPARSE_HALF_LIVES
            ]
            for _ in range(self.role_count)
        ]
        self._spectral = [
            [
                _SharedDimensionSpectral(
                    self.q,
                    half_life=half_life,
                )
                for half_life in SPECTRAL_HALF_LIVES
            ]
            for _ in range(self.role_count)
        ]
        self.role_updates = np.zeros(self.role_count, dtype=np.int64)

    def clone(self) -> DimensionAdaptedBank:
        return copy.deepcopy(self)

    def state_digest(self) -> str:
        """Hash every mutable numeric state without pickle."""

        arrays: dict[str, np.ndarray] = {
            "role_updates": self.role_updates,
            "logistic": np.asarray(
                [
                    [state.weights for state in role_states]
                    for role_states in self._logistic
                ],
                dtype=np.float64,
            ),
            "sparse_ewma": np.asarray(
                [
                    [state.state.ewma for state in role_states]
                    for role_states in self._sparse
                ],
                dtype=np.float64,
            ),
            "sparse_witness": np.asarray(
                [
                    [state.state.witness for state in role_states]
                    for role_states in self._sparse
                ],
                dtype=np.float64,
            ),
            "spectral_ewma": np.asarray(
                [
                    [state.state.ewma for state in role_states]
                    for role_states in self._spectral
                ],
                dtype=np.complex128,
            ),
            "spectral_effect": np.asarray(
                [
                    [state.effects() for state in role_states]
                    for role_states in self._spectral
                ],
                dtype=np.complex128,
            ),
        }
        digest = hashlib.sha256()
        for name, raw in sorted(arrays.items()):
            array = np.asarray(raw)
            canonical = np.ascontiguousarray(
                array.astype(array.dtype.newbyteorder("<"), copy=False)
            )
            header = repr((name, canonical.dtype.str, canonical.shape)).encode()
            digest.update(len(header).to_bytes(8, "little"))
            digest.update(header)
            payload = canonical.tobytes(order="C")
            digest.update(len(payload).to_bytes(8, "little"))
            digest.update(payload)
        return digest.hexdigest()

    def state_nbytes(self) -> int:
        """Return exact mutable numeric storage used by this Python bank."""

        arrays: list[np.ndarray] = [self.role_updates]
        for role_states in self._logistic:
            arrays.extend(state.weights for state in role_states)
        for role_states in self._sparse:
            for state in role_states:
                arrays.extend((state.state.ewma, state.state.witness))
        for role_states in self._spectral:
            for state in role_states:
                arrays.extend(
                    (
                        state.state.ewma,
                        state.state.effect,
                        state.rank_one_effect,
                        state.positive_effect,
                    )
                )
        return int(sum(np.asarray(array).nbytes for array in arrays))

    def _validate_pair(
        self,
        role: int,
        reference: np.ndarray,
        monitor: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        role_index = _strict_nonnegative_int(role, context="role")
        if role_index >= self.role_count:
            raise IndexError("role is out of range.")
        left = np.asarray(reference)
        right = np.asarray(monitor)
        if left.shape != (self.q,) or right.shape != (self.q,):
            raise ValueError(f"Each cycle must contain {self.q} checks.")
        if np.any((left != 0) & (left != 1)) or np.any((right != 0) & (right != 1)):
            raise ValueError("Cycle checks must be binary.")
        return left.astype(np.uint8, copy=False), right.astype(np.uint8, copy=False)

    def update_all(
        self,
        role: int,
        reference: np.ndarray,
        monitor: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Return every locked factor vector, then learn from the pair."""

        left, right = self._validate_pair(role, reference, monitor)
        role_index = int(role)
        left_features = dimension_adapted_features(left)
        right_features = dimension_adapted_features(right)
        difference = right_features - left_features
        density_difference = dimension_adapted_density(
            right
        ) - dimension_adapted_density(left)

        dfr_score = validate_bounded_score(
            float(np.mean(right) - np.mean(left)),
            tolerance=SCORE_TOLERANCE,
        )
        dfr_factors = linear_bet_factors(
            dfr_score,
            SIGNED_BET_MAGNITUDES,
            two_sided=True,
        )

        logistic_scores = np.asarray(
            [
                learner.update(left_features, right_features).score
                for learner in self._logistic[role_index]
            ],
            dtype=np.float64,
        )
        logistic_factors = np.concatenate(
            [
                linear_bet_factors(score, SIGNED_BET_MAGNITUDES)
                for score in logistic_scores
            ]
        )

        sparse_scores: list[float] = []
        for state in self._sparse[role_index]:
            sparse_scores.extend(state.update(difference))
        sparse_factors = np.concatenate(
            [
                linear_bet_factors(score, SIGNED_BET_MAGNITUDES)
                for score in sparse_scores
            ]
        )

        spectral_scores: list[float] = []
        for state in self._spectral[role_index]:
            spectral_scores.extend(state.update(density_difference))
        spectral_factors = np.concatenate(
            [
                linear_bet_factors(score, SIGNED_BET_MAGNITUDES)
                for score in spectral_scores
            ]
        )
        self.role_updates[role_index] += 1
        return {
            "dfr": dfr_factors,
            "online_logistic": logistic_factors,
            "space_sparse": sparse_factors,
            "space_spectral": spectral_factors,
            "space_composite": np.concatenate((sparse_factors, spectral_factors)),
        }


def replay_actual(
    bank: DimensionAdaptedBank,
    pre_reference: np.ndarray,
    pre_monitor: np.ndarray,
    post_reference: np.ndarray,
    post_monitor: np.ndarray,
) -> dict[str, dict[str, Any]]:
    """Replay surveillance without resetting at the constructed boundary."""

    blocks = (
        (pre_reference, pre_monitor, True),
        (post_reference, post_monitor, False),
    )
    if any(reference.shape != monitor.shape for reference, monitor, _ in blocks):
        raise ValueError("Paired replay blocks must have equal shapes.")
    pre_shots, roles, q = pre_reference.shape
    post_shots = post_reference.shape[0]
    if (
        post_reference.shape[1:] != (roles, q)
        or roles != bank.role_count
        or q != bank.q
    ):
        raise ValueError("Replay block dimensions disagree with the bank.")
    horizon = pre_shots + post_shots
    priors = shot_component_weights(q, roles)
    e_banks = {
        method: ProperUniformStartEProcessBank(
            len(weights),
            horizon=horizon,
            alpha=0.01,
            component_weights=weights,
        )
        for method, weights in priors.items()
    }
    traces = {method: np.empty(horizon, dtype="<f8") for method in METHOD_ORDER}
    first_alarm_update: dict[str, int | None] = {
        method: None for method in METHOD_ORDER
    }
    update_index = 0
    for reference, monitor, _ in blocks:
        for shot in range(reference.shape[0]):
            shot_factors: dict[str, list[np.ndarray]] = {
                method: [] for method in METHOD_ORDER
            }
            for role in range(roles):
                factors = bank.update_all(
                    role,
                    reference[shot, role],
                    monitor[shot, role],
                )
                for method in METHOD_ORDER:
                    shot_factors[method].append(factors[method])
            for method in METHOD_ORDER:
                result = e_banks[method].update(np.concatenate(shot_factors[method]))
                traces[method][update_index] = result.log_statistic
            update_index += 1
    if update_index != horizon:
        raise RuntimeError("Actual replay did not exhaust its horizon.")
    return {
        method: {
            "log_e": traces[method],
            "first_alarm_update": first_alarm_update[method],
        }
        for method in METHOD_ORDER
    }


def circular_block_indices(
    *,
    rng: np.random.Generator,
    replicates: int,
    source_shots: int,
    horizon_shots: int,
    block_length: int,
) -> np.ndarray:
    """Draw the exact locked circular moving-block bootstrap indices."""

    replicate_count = _strict_positive_int(replicates, context="replicates")
    source_count = _strict_positive_int(source_shots, context="source_shots")
    horizon_count = _strict_positive_int(horizon_shots, context="horizon_shots")
    block = _strict_positive_int(block_length, context="block_length")
    blocks = math.ceil(horizon_count / block)
    starts = rng.integers(
        0,
        source_count,
        size=(replicate_count, blocks),
        endpoint=False,
        dtype=np.int64,
    )
    offsets = np.arange(block, dtype=np.int64)
    expanded = (starts[..., None] + offsets) % source_count
    return expanded.reshape(replicate_count, blocks * block)[:, :horizon_count]


def bootstrap_threshold_scalar(
    *,
    fit_bank: DimensionAdaptedBank,
    method: str,
    fit_reference: np.ndarray,
    fit_monitor: np.ndarray,
    seed: int,
    replicates: int,
    horizon_shots: int,
    block_length: int = 32,
    alpha: float = 0.01,
) -> tuple[float, np.ndarray]:
    """Reference implementation of the locked path-state threshold.

    This scalar implementation is intentionally simple and auditable.  The
    production CLI uses it exactly; tests use small replicate counts.  Every
    replicate clones the learned fit state and starts a fresh proper-prior
    accumulator with the real surveillance horizon.
    """

    if method not in METHOD_ORDER:
        raise ValueError(f"Unknown method {method!r}.")
    if fit_reference.shape != fit_monitor.shape:
        raise ValueError("Bootstrap fit blocks must have equal shape.")
    if fit_reference.ndim != 3:
        raise ValueError("Bootstrap input must have shape (shot,role,check).")
    fit_shots, roles, q = fit_reference.shape
    if fit_shots < 1 or roles != fit_bank.role_count or q != fit_bank.q:
        raise ValueError("Bootstrap input dimensions disagree with fit bank.")
    rng = np.random.Generator(np.random.PCG64(seed))
    indices = circular_block_indices(
        rng=rng,
        replicates=replicates,
        source_shots=fit_shots,
        horizon_shots=horizon_shots,
        block_length=block_length,
    )
    weights = shot_component_weights(q, roles)[method]
    maxima = np.full(replicates, -np.inf, dtype=np.float64)
    horizon_updates = horizon_shots
    for replicate in range(replicates):
        bank = fit_bank.clone()
        e_bank = ProperUniformStartEProcessBank(
            len(weights),
            horizon=horizon_updates,
            alpha=alpha,
            component_weights=weights,
        )
        maximum = -np.inf
        for source_shot in indices[replicate]:
            source = int(source_shot)
            shot_factors: list[np.ndarray] = []
            for role in range(roles):
                shot_factors.append(
                    bank.update_all(
                        role,
                        fit_reference[source, role],
                        fit_monitor[source, role],
                    )[method]
                )
            update = e_bank.update(np.concatenate(shot_factors))
            maximum = max(maximum, update.log_statistic)
        maxima[replicate] = maximum
    if np.any(np.isnan(maxima)):
        raise FloatingPointError("Bootstrap maxima contain NaN.")
    order_index = min(
        math.ceil((replicates + 1) * (1.0 - alpha)) - 1,
        replicates - 1,
    )
    threshold = float(np.sort(maxima)[order_index])
    return threshold, maxima


def _batched_spectral_effects(
    matrices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return stable rank-one and positive effects for batched Hermitians."""

    values = np.asarray(matrices, dtype=np.float64)
    if values.ndim != 3 or values.shape[1] != values.shape[2]:
        raise ValueError("matrices must have shape (batch,q,q).")
    eigenvalues, eigenvectors = np.linalg.eigh(values)
    positive = eigenvalues > EIGENVALUE_TOLERANCE
    positive_effect = np.einsum(
        "bik,bjk,bk->bij",
        eigenvectors,
        eigenvectors,
        positive.astype(np.float64),
        optimize=True,
    )

    maximum = eigenvalues[:, -1]
    tied = eigenvalues >= maximum[:, None] - EIGENVALUE_TOLERANCE
    top_projector = np.einsum(
        "bik,bjk,bk->bij",
        eigenvectors,
        eigenvectors,
        tied.astype(np.float64),
        optimize=True,
    )
    diagonal = np.clip(
        np.diagonal(top_projector, axis1=1, axis2=2),
        0.0,
        None,
    )
    largest = np.max(diagonal, axis=1)
    anchor_mask = diagonal >= largest[:, None] - EIGENVALUE_TOLERANCE
    anchors = np.argmax(anchor_mask, axis=1)
    batch = np.arange(values.shape[0])
    denominators = np.sqrt(diagonal[batch, anchors])
    active = maximum > EIGENVALUE_TOLERANCE
    safe = np.where(active, denominators, 1.0)
    vectors = top_projector[batch, :, anchors] / safe[:, None]
    vectors[~active] = 0.0
    rank_one = np.einsum("bi,bj->bij", vectors, vectors, optimize=True)
    return (
        0.5 * (rank_one + np.swapaxes(rank_one, 1, 2)),
        0.5 * (positive_effect + np.swapaxes(positive_effect, 1, 2)),
    )


def _update_vectorized_eprocess(
    log_components: np.ndarray,
    factors: np.ndarray,
    *,
    log_start_mass: float,
    log_component_weights: np.ndarray,
    remaining: int,
) -> np.ndarray:
    """Update a batch of proper-prior e-processes and return log statistics."""

    values = np.asarray(factors, dtype=np.float64)
    if values.shape != log_components.shape:
        raise ValueError("Factor and e-process component shapes disagree.")
    if np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise FloatingPointError("Bootstrap factors must be finite/nonnegative.")
    with np.errstate(divide="ignore"):
        log_factors = np.log(values)
    log_components[:] = log_factors + np.logaddexp(
        log_components,
        log_start_mass,
    )
    started = np.logaddexp.reduce(
        log_components + log_component_weights[None, :],
        axis=1,
    )
    if remaining > 0:
        return np.logaddexp(started, math.log(remaining) + log_start_mass)
    return started


def bootstrap_threshold_vectorized(
    *,
    fit_bank: DimensionAdaptedBank,
    method: str,
    fit_reference: np.ndarray,
    fit_monitor: np.ndarray,
    seed: int,
    replicates: int,
    horizon_shots: int,
    block_length: int = 32,
    alpha: float = 0.01,
    _episode_indices: np.ndarray | None = None,
) -> tuple[float, np.ndarray, int]:
    """Vectorized production implementation of the locked bootstrap.

    Replicates are rows, so one NumPy operation advances every independently
    resampled episode by one paired shot.  Round-role factors are fixed
    experts under a uniform role prior, so the formal process receives one
    update for the one shared shot orientation.  This preserves the scalar RNG draw order,
    score-before-update semantics, role isolation, and threshold order
    statistic while avoiding billions of Python-level updates.
    """

    if method not in METHOD_ORDER:
        raise ValueError(f"Unknown method {method!r}.")
    if fit_reference.shape != fit_monitor.shape or fit_reference.ndim != 3:
        raise ValueError("Bootstrap fit blocks must be equal 3-D arrays.")
    fit_shots, roles, q = fit_reference.shape
    if (
        fit_shots < 1
        or roles != fit_bank.role_count
        or q != fit_bank.q
        or horizon_shots < 1
    ):
        raise ValueError("Bootstrap dimensions disagree with the fit bank.")
    if _episode_indices is None:
        rng = np.random.Generator(np.random.PCG64(seed))
        indices = circular_block_indices(
            rng=rng,
            replicates=replicates,
            source_shots=fit_shots,
            horizon_shots=horizon_shots,
            block_length=block_length,
        )
    else:
        indices = np.asarray(_episode_indices)
        if (
            indices.shape != (replicates, horizon_shots)
            or not np.issubdtype(indices.dtype, np.integer)
            or np.any(indices < 0)
            or np.any(indices >= fit_shots)
        ):
            raise ValueError("Explicit episode indices are invalid.")
        indices = indices.astype(np.int64, copy=False)
    base_component_count = len(method_component_weights(q)[method])
    component_weights = shot_component_weights(q, roles)[method]
    with np.errstate(divide="ignore"):
        log_weights = np.log(component_weights)
    horizon_updates = horizon_shots
    log_start = -math.log(horizon_updates)
    log_components = np.full(
        (replicates, len(component_weights)),
        -np.inf,
        dtype=np.float64,
    )
    maxima = np.full(replicates, -np.inf, dtype=np.float64)
    bet_values = np.asarray(SIGNED_BET_MAGNITUDES, dtype=np.float64)
    signed_bets = np.column_stack((bet_values, -bet_values)).reshape(-1)

    logistic_weights: np.ndarray | None = None
    if method == "online_logistic":
        warm = np.asarray(
            [
                [state.weights for state in role_states]
                for role_states in fit_bank._logistic
            ],
            dtype=np.float64,
        )
        logistic_weights = np.broadcast_to(
            warm,
            (replicates, *warm.shape),
        ).copy()
        logistic_rates = np.asarray(
            LOGISTIC_LEARNING_RATES,
            dtype=np.float64,
        )

    sparse_ewma: np.ndarray | None = None
    sparse_times: np.ndarray | None = None
    if method in {"space_sparse", "space_composite"}:
        sparse_warm = np.asarray(
            [
                [state.state.ewma for state in role_states]
                for role_states in fit_bank._sparse
            ],
            dtype=np.float64,
        )
        sparse_ewma = np.broadcast_to(
            sparse_warm,
            (replicates, *sparse_warm.shape),
        ).copy()
        sparse_times = np.asarray(
            [
                [state.state.time for state in role_states]
                for role_states in fit_bank._sparse
            ],
            dtype=np.int64,
        )
        sparse_decay = np.exp2(-1.0 / np.asarray(SPARSE_HALF_LIVES, dtype=np.float64))
        sparse_alpha = 1.0 - sparse_decay
        sparse_k = np.asarray(fit_bank.sparse_k, dtype=np.int64)

    spectral_ewma: np.ndarray | None = None
    spectral_effects: np.ndarray | None = None
    spectral_times: np.ndarray | None = None
    eigendecompositions = 0
    if method in {"space_spectral", "space_composite"}:
        spectral_warm = np.asarray(
            [
                [state.state.ewma.real for state in role_states]
                for role_states in fit_bank._spectral
            ],
            dtype=np.float64,
        )
        effect_warm = np.asarray(
            [
                [state.effects().real for state in role_states]
                for role_states in fit_bank._spectral
            ],
            dtype=np.float64,
        )
        spectral_ewma = np.broadcast_to(
            spectral_warm,
            (replicates, *spectral_warm.shape),
        ).copy()
        spectral_effects = np.broadcast_to(
            effect_warm,
            (replicates, *effect_warm.shape),
        ).copy()
        spectral_times = np.asarray(
            [
                [state.state.time for state in role_states]
                for role_states in fit_bank._spectral
            ],
            dtype=np.int64,
        )
        spectral_decay = np.exp2(
            -1.0 / np.asarray(SPECTRAL_HALF_LIVES, dtype=np.float64)
        )
        spectral_alpha = 1.0 - spectral_decay

    update_index = 0
    for bootstrap_shot in range(horizon_shots):
        source_indices = indices[:, bootstrap_shot]
        shot_factors = np.empty(
            (replicates, roles, base_component_count),
            dtype=np.float64,
        )
        for role in range(roles):
            left = fit_reference[source_indices, role]
            right = fit_monitor[source_indices, role]
            factors_by_method: dict[str, np.ndarray] = {}

            if method == "dfr":
                score = np.mean(right, axis=1) - np.mean(left, axis=1)
                factors_by_method["dfr"] = 1.0 + score[:, None] * signed_bets

            left_features: np.ndarray | None = None
            right_features: np.ndarray | None = None
            difference: np.ndarray | None = None
            if method in {
                "online_logistic",
                "space_sparse",
                "space_composite",
            }:
                left_features = dimension_adapted_features(left)
                right_features = dimension_adapted_features(right)
                difference = right_features - left_features

            if method == "online_logistic":
                assert logistic_weights is not None
                assert left_features is not None and right_features is not None
                assert difference is not None
                weights = logistic_weights[:, role]
                margin = np.einsum(
                    "blp,bp->bl",
                    weights,
                    difference,
                    optimize=True,
                )
                denominator = np.maximum(1.0, np.sum(np.abs(weights), axis=2))
                score = np.tanh(margin / denominator)
                factors_by_method["online_logistic"] = (
                    1.0 + score[:, :, None] * bet_values
                ).reshape(replicates, -1)
                miss = np.empty_like(margin)
                nonnegative = margin >= 0.0
                exponential = np.exp(-margin[nonnegative])
                miss[nonnegative] = exponential / (1.0 + exponential)
                exponential = np.exp(margin[~nonnegative])
                miss[~nonnegative] = 1.0 / (1.0 + exponential)
                gradient = -miss[:, :, None] * difference[:, None, :] + 1e-4 * weights
                weights -= logistic_rates[None, :, None] * gradient

            sparse_factors: np.ndarray | None = None
            if method in {"space_sparse", "space_composite"}:
                assert sparse_ewma is not None and sparse_times is not None
                assert difference is not None
                direction = sparse_ewma[:, role]
                sparse_scores = np.empty(
                    (
                        replicates,
                        len(SPARSE_HALF_LIVES),
                        len(fit_bank.sparse_k),
                    ),
                    dtype=np.float64,
                )
                for half_life_index in range(len(SPARSE_HALF_LIVES)):
                    current_direction = direction[:, half_life_index]
                    # Stable sort retains increasing coordinate order on ties.
                    order = np.argsort(
                        -np.abs(current_direction),
                        axis=1,
                        kind="stable",
                    )
                    ordered_sign = np.where(
                        np.take_along_axis(
                            current_direction,
                            order,
                            axis=1,
                        )
                        < 0.0,
                        -1.0,
                        1.0,
                    )
                    ordered_difference = np.take_along_axis(
                        difference,
                        order,
                        axis=1,
                    )
                    cumulative = np.cumsum(
                        ordered_sign * ordered_difference,
                        axis=1,
                    )
                    sparse_scores[:, half_life_index] = (
                        cumulative[:, sparse_k - 1] / sparse_k[None, :]
                    )
                    zero = np.max(np.abs(current_direction), axis=1) <= SCORE_TOLERANCE
                    sparse_scores[zero, half_life_index] = 0.0
                    direction[:, half_life_index] = (
                        sparse_decay[half_life_index] * current_direction
                        + sparse_alpha[half_life_index] * difference
                    )
                    sparse_times[role, half_life_index] += 1
                sparse_factors = (
                    1.0 + sparse_scores[..., None] * bet_values[None, None, None, :]
                ).reshape(replicates, -1)
                factors_by_method["space_sparse"] = sparse_factors

            spectral_factors: np.ndarray | None = None
            if method in {"space_spectral", "space_composite"}:
                assert spectral_ewma is not None
                assert spectral_effects is not None
                assert spectral_times is not None
                density_difference = dimension_adapted_density(
                    right
                ) - dimension_adapted_density(left)
                effects = spectral_effects[:, role]
                spectral_scores = np.einsum(
                    "bhrij,bij->bhr",
                    effects,
                    density_difference,
                    optimize=True,
                )
                spectral_scores = np.clip(spectral_scores, -1.0, 1.0)
                spectral_factors = (
                    1.0 + spectral_scores[..., None] * bet_values[None, None, None, :]
                ).reshape(replicates, -1)
                factors_by_method["space_spectral"] = spectral_factors
                for half_life_index in range(len(SPECTRAL_HALF_LIVES)):
                    spectral_ewma[:, role, half_life_index] = (
                        spectral_decay[half_life_index]
                        * spectral_ewma[:, role, half_life_index]
                        + spectral_alpha[half_life_index] * density_difference
                    )
                    spectral_times[role, half_life_index] += 1
                    if (
                        spectral_times[role, half_life_index] % SPECTRAL_UPDATE_STRIDE
                        == 0
                    ):
                        rank_one, positive = _batched_spectral_effects(
                            spectral_ewma[:, role, half_life_index]
                        )
                        spectral_effects[:, role, half_life_index, 0] = rank_one
                        spectral_effects[:, role, half_life_index, 1] = positive
                        eigendecompositions += replicates

            if method == "space_composite":
                assert sparse_factors is not None and spectral_factors is not None
                factors_by_method["space_composite"] = np.concatenate(
                    (sparse_factors, spectral_factors),
                    axis=1,
                )
            shot_factors[:, role] = factors_by_method[method]

        update_index += 1
        trace = _update_vectorized_eprocess(
            log_components,
            shot_factors.reshape(replicates, -1),
            log_start_mass=log_start,
            log_component_weights=log_weights,
            remaining=horizon_updates - update_index,
        )
        maxima = np.maximum(maxima, trace)
    if update_index != horizon_updates or np.any(np.isnan(maxima)):
        raise FloatingPointError("Vectorized bootstrap replay failed.")
    order_index = min(
        math.ceil((replicates + 1) * (1.0 - alpha)) - 1,
        replicates - 1,
    )
    threshold = float(np.sort(maxima)[order_index])
    return threshold, maxima, eigendecompositions


def _first_threshold_crossing(trace: np.ndarray, threshold: float) -> int | None:
    if np.any(np.isnan(trace)) or math.isnan(threshold):
        raise FloatingPointError("Alarm trace and threshold must not contain NaN.")
    indices = np.flatnonzero(trace >= threshold)
    return None if not len(indices) else int(indices[0])


def stream_metrics(
    *,
    trace: np.ndarray,
    threshold: float,
    pre_surveillance_shots: int,
    post_shots: int,
    roles: int,
) -> dict[str, Any]:
    """Compute the locked pre-alarm and restricted post-delay endpoint."""

    _strict_positive_int(roles, context="roles")
    expected = pre_surveillance_shots + post_shots
    values = np.asarray(trace, dtype=np.float64)
    if values.shape != (expected,):
        raise ValueError("Alarm trace has the wrong surveillance horizon.")
    crossing = _first_threshold_crossing(values, threshold)
    boundary_update = pre_surveillance_shots
    pre_alarm = crossing is not None and crossing < boundary_update
    miss = crossing is None
    if pre_alarm or miss:
        delay = 1.0
        post_alarm_shot = None
        post_alarm_role = None
    else:
        post_alarm_shot = crossing - boundary_update
        post_alarm_role = None
        delay = (post_alarm_shot + 1) / post_shots
    return {
        "first_alarm_update": crossing,
        "pre_false_alarm": int(pre_alarm),
        "miss": int(miss),
        "post_alarm_shot": post_alarm_shot,
        "post_alarm_role": post_alarm_role,
        "restricted_post_delay_fraction": float(delay),
    }


def aggregate_results(
    state_rows: Sequence[Mapping[str, Any]],
    cohorts: Sequence[Cohort],
    *,
    bootstrap_replicates: int = 10_000,
) -> dict[str, Any]:
    """Macro-average states/paths and apply the locked retention rule."""

    expected_keys = {
        (cohort.cohort_id, state, method)
        for cohort in cohorts
        for state in (0, 1)
        for method in METHOD_ORDER
    }
    indexed: dict[tuple[str, int, str], Mapping[str, Any]] = {}
    for row in state_rows:
        key = (row["cohort_id"], int(row["logical_state"]), row["method"])
        if key in indexed:
            raise ValueError(f"Duplicate state result row: {key}")
        indexed[key] = row
    if set(indexed) != expected_keys:
        missing = sorted(expected_keys - set(indexed))
        extra = sorted(set(indexed) - expected_keys)
        raise ValueError(
            f"State result coverage mismatch; missing={missing}, extra={extra}"
        )

    cohort_rows: list[dict[str, Any]] = []
    for cohort in cohorts:
        for method in METHOD_ORDER:
            pair = [indexed[(cohort.cohort_id, state, method)] for state in (0, 1)]
            cohort_rows.append(
                {
                    "cohort_id": cohort.cohort_id,
                    "basis": cohort.basis,
                    "distance": cohort.distance,
                    "rounds": cohort.rounds,
                    "calibration_pair_id": cohort.calibration_pair_id,
                    "method": method,
                    "pre_false_alarm_mean": float(
                        np.mean([row["pre_false_alarm"] for row in pair])
                    ),
                    "restricted_post_delay_fraction": float(
                        np.mean([row["restricted_post_delay_fraction"] for row in pair])
                    ),
                    "miss_mean": float(np.mean([row["miss"] for row in pair])),
                }
            )

    by_method = {
        method: [row for row in cohort_rows if row["method"] == method]
        for method in METHOD_ORDER
    }
    macro = {
        method: {
            "pre_false_alarm_state_count": int(
                sum(
                    indexed[(cohort.cohort_id, state, method)]["pre_false_alarm"]
                    for cohort in cohorts
                    for state in (0, 1)
                )
            ),
            "miss_state_count": int(
                sum(
                    indexed[(cohort.cohort_id, state, method)]["miss"]
                    for cohort in cohorts
                    for state in (0, 1)
                )
            ),
            "macro_restricted_post_delay_fraction": float(
                np.mean(
                    [row["restricted_post_delay_fraction"] for row in by_method[method]]
                )
            ),
        }
        for method in METHOD_ORDER
    }

    target = "space_composite"
    comparisons: dict[str, Any] = {}
    for comparator in ("dfr", "online_logistic"):
        effects = np.asarray(
            [
                by_method[target][index]["restricted_post_delay_fraction"]
                - by_method[comparator][index]["restricted_post_delay_fraction"]
                for index in range(len(cohorts))
            ],
            dtype=np.float64,
        )
        rng = np.random.Generator(np.random.PCG64(612500))
        draws = rng.integers(
            0,
            len(cohorts),
            size=(bootstrap_replicates, len(cohorts)),
            endpoint=False,
            dtype=np.int64,
        )
        sampled = np.mean(effects[draws], axis=1)

        grouped: dict[str, list[int]] = defaultdict(list)
        for index, cohort in enumerate(cohorts):
            grouped[cohort.calibration_pair_id].append(index)
        cluster_ids = sorted(grouped)
        if len(cohorts) == 11 and len(cluster_ids) != 5:
            raise ValueError("Locked Pittsburgh calibration-pair count changed.")
        cluster_rng = np.random.Generator(np.random.PCG64(612501))
        cluster_draws = cluster_rng.integers(
            0,
            len(cluster_ids),
            size=(bootstrap_replicates, len(cluster_ids)),
            endpoint=False,
            dtype=np.int64,
        )
        cluster_samples = np.empty(bootstrap_replicates, dtype=np.float64)
        for draw_index, draw in enumerate(cluster_draws):
            selected: list[int] = []
            for cluster_index in draw:
                selected.extend(grouped[cluster_ids[int(cluster_index)]])
            cluster_samples[draw_index] = float(np.mean(effects[selected]))

        observed_abs = abs(float(np.mean(effects)))
        sign_tail = 0
        for signs in product((-1.0, 1.0), repeat=len(cohorts)):
            statistic = abs(float(np.mean(effects * np.asarray(signs))))
            sign_tail += statistic >= observed_abs - 1e-15
        sign_p = sign_tail / (2 ** len(cohorts))
        no_worse_pre = (
            macro[target]["pre_false_alarm_state_count"]
            <= macro[comparator]["pre_false_alarm_state_count"]
        )
        strict_delay = float(np.mean(effects)) < 0.0
        comparisons[comparator] = {
            "cohort_delay_differences": effects.tolist(),
            "macro_delay_difference": float(np.mean(effects)),
            "primary_95_percentile_interval": np.quantile(
                sampled,
                [0.025, 0.975],
                method="linear",
            ).tolist(),
            "calibration_pair_95_percentile_sensitivity": np.quantile(
                cluster_samples,
                [0.025, 0.975],
                method="linear",
            ).tolist(),
            "exact_sign_flip_two_sided_p": float(sign_p),
            "no_worse_pre_false_alarm": bool(no_worse_pre),
            "strictly_lower_macro_delay": bool(strict_delay),
            "retention_condition_pass": bool(no_worse_pre and strict_delay),
        }
    return {
        "cohort_rows": cohort_rows,
        "macro_by_method": macro,
        "comparisons": comparisons,
        "retention_pass": all(
            value["retention_condition_pass"] for value in comparisons.values()
        ),
    }


def randomization_audit(
    *,
    cohorts: Sequence[Cohort],
    event_cache: Mapping[str, Mapping[tuple[int, str], np.ndarray]],
    seeds: Sequence[int] = tuple(range(610700, 610956)),
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, int]:
    """Run the locked complete-shot paired-swap implementation audit."""

    seed_values = tuple(int(seed) for seed in seeds)
    if len(seed_values) < 1 or len(set(seed_values)) != len(seed_values):
        raise ValueError("Randomization seeds must be a nonempty unique sequence.")
    generators = [np.random.Generator(np.random.PCG64(seed)) for seed in seed_values]
    replicates = len(generators)
    alarm_counts = np.zeros(
        (replicates, len(METHOD_ORDER)),
        dtype=np.int64,
    )
    maximum_log_e = np.full(
        (replicates, len(METHOD_ORDER)),
        -np.inf,
        dtype=np.float64,
    )
    stream_rows: list[dict[str, Any]] = []
    eigendecompositions = 0
    for cohort_index, cohort in enumerate(cohorts):
        early = event_cache[cohort.early_snapshot_id]
        for logical_state in (0, 1):
            events = early[(logical_state, cohort.register_suffix)]
            reference = events[: cohort.m]
            monitor = events[cohort.m : 2 * cohort.m]
            if (
                reference.shape
                != (
                    cohort.m,
                    cohort.rounds,
                    cohort.q,
                )
                or monitor.shape != reference.shape
            ):
                raise ValueError("Randomization stream shape changed.")
            masks = np.stack(
                [
                    generator.integers(
                        0,
                        2,
                        size=cohort.m,
                        dtype=np.int8,
                    )
                    for generator in generators
                ]
            )
            source_indices = np.arange(cohort.m, dtype=np.int64)[
                None, :
            ] + cohort.m * masks.astype(np.int64)
            augmented_reference = np.concatenate((reference, monitor), axis=0)
            augmented_monitor = np.concatenate((monitor, reference), axis=0)
            zero_bank = DimensionAdaptedBank(
                q=cohort.q,
                role_count=cohort.rounds,
            )
            for method_index, method in enumerate(METHOD_ORDER):
                _, maxima, eig_count = bootstrap_threshold_vectorized(
                    fit_bank=zero_bank,
                    method=method,
                    fit_reference=augmented_reference,
                    fit_monitor=augmented_monitor,
                    seed=0,
                    replicates=replicates,
                    horizon_shots=cohort.m,
                    block_length=32,
                    alpha=0.01,
                    _episode_indices=source_indices,
                )
                alarms = maxima >= LOG_E_100
                alarm_counts[:, method_index] += alarms
                maximum_log_e[:, method_index] = np.maximum(
                    maximum_log_e[:, method_index],
                    maxima,
                )
                eigendecompositions += eig_count
                stream_rows.append(
                    {
                        "cohort_index": cohort_index,
                        "cohort_id": cohort.cohort_id,
                        "logical_state": logical_state,
                        "method": method,
                        "alarm_fraction": float(np.mean(alarms)),
                        "maximum_log_e_over_replicates": float(np.max(maxima)),
                    }
                )
    return (
        {
            "schema_version": "run6-pnnl-randomization-audit-v1",
            "seeds": list(seed_values),
            "method_order": list(METHOD_ORDER),
            "path_state_method_rows": stream_rows,
            "overall_episode_alarm_fraction": {
                method: float(
                    np.sum(alarm_counts[:, method_index])
                    / (replicates * 2 * len(cohorts))
                )
                for method_index, method in enumerate(METHOD_ORDER)
            },
            "alarmed_episode_count_histogram": {
                method: {
                    str(int(count)): int(frequency)
                    for count, frequency in zip(
                        *np.unique(
                            alarm_counts[:, method_index],
                            return_counts=True,
                        )
                    )
                }
                for method_index, method in enumerate(METHOD_ORDER)
            },
            "maximum_log_e_summary": {
                method: {
                    "minimum": float(np.min(maximum_log_e[:, method_index])),
                    "median": float(np.median(maximum_log_e[:, method_index])),
                    "maximum": float(np.max(maximum_log_e[:, method_index])),
                }
                for method_index, method in enumerate(METHOD_ORDER)
            },
            "claim_scope": (
                "implementation and exact randomized paired design only; "
                "not a natural hardware null"
            ),
        },
        alarm_counts,
        maximum_log_e,
        eigendecompositions,
    )


def _strict_binary_matrix(
    value: Any,
    *,
    rows: int,
    columns: int,
    context: str,
) -> np.ndarray:
    if not isinstance(value, list) or len(value) != rows:
        raise ValueError(f"{context} must contain exactly {rows} rows.")
    result = np.empty((rows, columns), dtype=np.uint8)
    for row_index, row in enumerate(value):
        if not isinstance(row, list) or len(row) != columns:
            raise ValueError(f"{context}[{row_index}] has the wrong shape.")
        for column_index, item in enumerate(row):
            if (
                isinstance(item, bool)
                or not isinstance(item, int)
                or item not in (0, 1)
            ):
                raise ValueError(
                    f"{context}[{row_index}][{column_index}] is not integer 0/1."
                )
            result[row_index, column_index] = item
    return result


def _strict_json_value(path: Path) -> Any:
    """Load any JSON value while rejecting duplicate keys/non-finite numbers."""

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key!r}.")
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> None:
        raise ValueError(f"Non-finite JSON number is forbidden: {value}.")

    return json.loads(
        path.read_text(encoding="utf-8-sig"),
        object_pairs_hook=reject_duplicates,
        parse_constant=reject_nonfinite,
    )


def load_snapshot_events(
    path: Path,
    *,
    snapshot: Mapping[str, Any],
    register_labels: Sequence[str],
) -> dict[tuple[int, str], np.ndarray]:
    """Strictly parse selected state/register syndrome events after unblinding."""

    payload = _strict_json_value(path)
    if not isinstance(payload, list):
        raise TypeError("PNNL bitstrings root must be an array.")
    distance, rounds, basis, shots = snapshot["metadata"][:4]
    states: dict[int, Mapping[str, Any]] = {}
    for entry_index, entry in enumerate(payload):
        if not isinstance(entry, dict):
            raise TypeError(f"bitstrings[{entry_index}] must be an object.")
        metadata = entry.get("metadata")
        registers = entry.get("per_shot_cregs")
        if not isinstance(metadata, dict) or not isinstance(registers, dict):
            raise TypeError("Each bitstring entry needs metadata and per_shot_cregs.")
        state = metadata.get("logical_state")
        if isinstance(state, bool) or not isinstance(state, int) or state not in (0, 1):
            raise ValueError("metadata.logical_state must be integer 0 or 1.")
        if state in states:
            raise ValueError(f"Duplicate logical state {state}.")
        metadata_rounds = metadata.get("n_syndrome_rounds")
        if (
            isinstance(metadata_rounds, bool)
            or not isinstance(metadata_rounds, int)
            or metadata_rounds != rounds
        ):
            raise ValueError("metadata.n_syndrome_rounds changed.")
        metadata_basis = metadata.get("basis")
        if not isinstance(metadata_basis, str) or metadata_basis != basis:
            raise ValueError("metadata.basis changed.")
        states[state] = entry
    if set(states) != {0, 1}:
        raise ValueError("Exactly one entry for each logical state is required.")

    output: dict[tuple[int, str], np.ndarray] = {}
    for state in (0, 1):
        registers = states[state]["per_shot_cregs"]
        for label in register_labels:
            data = _strict_binary_matrix(
                registers[f"c_data_{label}"],
                rows=shots,
                columns=distance,
                context=f"state{state}.c_data_{label}",
            )
            syndrome = _strict_binary_matrix(
                registers[f"c_syndrome_{label}"],
                rows=shots,
                columns=rounds * (distance - 1),
                context=f"state{state}.c_syndrome_{label}",
            )
            # Shape/domain validation of data is intentional; values never
            # enter any detector score.
            if data.shape != (shots, distance):
                raise RuntimeError("Validated final-data shape changed.")
            output[(state, label)] = repetition_detection_events(
                syndrome,
                distance=distance,
                rounds=rounds,
            )
    return output


def validate_qasm_maps_for_cohorts(
    *,
    artifact_root: Path,
    manifest: Mapping[str, Any],
    cohorts: Sequence[Cohort],
) -> None:
    """Independently check selected state QASM maps against every cohort."""

    by_snapshot: dict[str, list[Cohort]] = defaultdict(list)
    for cohort in cohorts:
        by_snapshot[cohort.early_snapshot_id].append(cohort)
        by_snapshot[cohort.late_snapshot_id].append(cohort)
    for snapshot_id, selected in by_snapshot.items():
        snapshot = manifest["snapshots"][snapshot_id]
        distance, rounds = snapshot["metadata"][:2]
        job_dir = artifact_root / snapshot["relative_job_dir"]
        state_maps = [
            parse_qasm_register_maps(
                job_dir / f"circuit_state{state}.qasm",
                distance=distance,
                rounds=rounds,
            )
            for state in (0, 1)
        ]
        for cohort in selected:
            label = cohort.register_suffix
            left = state_maps[0][label]
            right = state_maps[1][label]
            if (
                left.data_qubits != cohort.data_qubits
                or right.data_qubits != cohort.data_qubits
                or left.syndrome_qubits_by_round[0] != cohort.syndrome_qubits
                or right.syndrome_qubits_by_round[0] != cohort.syndrome_qubits
                or left.oriented_path != cohort.oriented_path
                or right.oriented_path != cohort.oriented_path
            ):
                raise ValueError(f"QASM map changed for {cohort.cohort_id}.")


def _snapshot_register_index(
    cohorts: Sequence[Cohort],
) -> dict[str, tuple[str, ...]]:
    selected: dict[str, set[str]] = defaultdict(set)
    for cohort in cohorts:
        selected[cohort.early_snapshot_id].add(cohort.register_suffix)
        selected[cohort.late_snapshot_id].add(cohort.register_suffix)
    return {
        snapshot_id: tuple(sorted(labels)) for snapshot_id, labels in selected.items()
    }


def _first_unblinding_record(
    *,
    repo_root: Path,
    output: Path,
    config_path: Path,
    manifest_path: Path,
    ratification_path: Path,
    manifest: Mapping[str, Any],
    snapshot_ids: Sequence[str],
    artifact_root: Path,
) -> tuple[dict[str, str], Path]:
    """Hash exact held files and persist the record before score computation."""

    payload_hashes: dict[str, str] = {}
    payload_records: list[dict[str, Any]] = []
    for snapshot_id in snapshot_ids:
        snapshot = manifest["snapshots"][snapshot_id]
        path = artifact_root / snapshot["relative_job_dir"] / "bitstrings.json"
        expected_bytes = int(snapshot["held_bitstrings"][0])
        if path.stat().st_size != expected_bytes:
            raise ValueError(f"Held payload size changed for {snapshot_id}.")
        digest = sha256_file(path)
        payload_hashes[snapshot_id] = digest
        payload_records.append(
            {
                "snapshot_id": snapshot_id,
                "path": path.relative_to(repo_root).as_posix(),
                "bytes": expected_bytes,
                "sha256": digest,
            }
        )
    record = {
        "schema_version": "run6-pnnl-first-unblinding-v1",
        "utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(repo_root),
        "config_sha256": sha256_file(config_path),
        "manifest_sha256": sha256_file(manifest_path),
        "freeze_ratification_sha256": sha256_file(ratification_path),
        "package_lock": {
            "path": PACKAGE_LOCK_RELATIVE,
            "bytes": (repo_root / PACKAGE_LOCK_RELATIVE).stat().st_size,
            "sha256": sha256_file(repo_root / PACKAGE_LOCK_RELATIVE),
        },
        "package_environment": environment_fingerprint(),
        "held_payloads": payload_records,
        "scores_computed_before_record": False,
    }
    record_path = output / "first_unblinding_record.json"
    _write_canonical_json(record_path, record)
    return payload_hashes, record_path


def _save_state_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = (
        "cohort_id",
        "logical_state",
        "method",
        "threshold_seed",
        "threshold_log_e",
        "first_alarm_update",
        "pre_false_alarm",
        "miss",
        "post_alarm_shot",
        "post_alarm_role",
        "restricted_post_delay_fraction",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def synthetic_dry_run() -> dict[str, Any]:
    """Exercise all dimension-adapted methods without any external payload."""

    rng = np.random.Generator(np.random.PCG64(613900))
    m = 12
    roles = 2
    q = 4
    reference = rng.integers(0, 2, size=(m, roles, q), dtype=np.uint8)
    monitor = reference.copy()
    monitor[6:, :, 0] ^= 1
    fit_shots = m // 2
    bank = DimensionAdaptedBank(q=q, role_count=roles)
    for shot in range(fit_shots):
        for role in range(roles):
            bank.update_all(role, reference[shot, role], monitor[shot, role])
    checkpoint = bank.state_digest()
    actual = replay_actual(
        bank.clone(),
        reference[fit_shots:],
        monitor[fit_shots:],
        reference,
        monitor,
    )
    state_rows: list[dict[str, Any]] = []
    for method_index, method in enumerate(METHOD_ORDER):
        threshold, maxima = bootstrap_threshold_scalar(
            fit_bank=bank,
            method=method,
            fit_reference=reference[:fit_shots],
            fit_monitor=monitor[:fit_shots],
            seed=613910 + method_index,
            replicates=8,
            horizon_shots=(m - fit_shots) + m,
            block_length=4,
            alpha=0.25,
        )
        metrics = stream_metrics(
            trace=actual[method]["log_e"],
            threshold=threshold,
            pre_surveillance_shots=m - fit_shots,
            post_shots=m,
            roles=roles,
        )
        state_rows.append(
            {
                "method": method,
                "threshold": threshold,
                "bootstrap_maximum_count": len(maxima),
                **metrics,
            }
        )
    return {
        "status": "synthetic_dry_run_passed",
        "seed": 613900,
        "q": q,
        "feature_dimension": q * (q + 1) // 2,
        "eligible_sparse_k": list(eligible_sparse_k(q)),
        "fit_checkpoint_sha256": checkpoint,
        "methods": state_rows,
        "raw_run6_values_opened": False,
    }


def run_real(args: argparse.Namespace) -> None:
    """Execute the locked Pittsburgh replay after all freeze checks."""

    run_started = time.time()
    if args.freeze_ratification is None or args.output is None:
        raise ValueError("Real replay requires --freeze-ratification and --output.")
    repo_root = Path(__file__).resolve().parents[3]
    config_path = args.config.resolve()
    manifest_path = args.manifest.resolve()
    ratification_path = args.freeze_ratification.resolve()
    output = args.output.resolve()
    config = load_pnnl_config(config_path)

    # The metadata validator performs only stat on held bitstrings.
    validation = validate_lock(
        manifest_path,
        repo_root=repo_root,
        mode="frozen",
    )
    manifest = load_strict_json(manifest_path)
    cohorts = parse_cohorts(manifest)
    verify_freeze_ratification(
        ratification_path,
        repo_root=repo_root,
        config_path=config_path,
        config=config,
        manifest_path=manifest_path,
    )
    if (
        config["status"] != "frozen_before_held_value_access"
        or manifest["status"] != "frozen_before_held_value_access"
    ):
        raise ValueError("PNNL config and manifest must have final frozen status.")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Output directory exists and is not empty.")
    output.mkdir(parents=True, exist_ok=True)

    artifact_root = repo_root / manifest["source"]["artifact_root"]
    validate_qasm_maps_for_cohorts(
        artifact_root=artifact_root,
        manifest=manifest,
        cohorts=cohorts,
    )
    register_index = _snapshot_register_index(cohorts)
    payload_hashes, unblinding_path = _first_unblinding_record(
        repo_root=repo_root,
        output=output,
        config_path=config_path,
        manifest_path=manifest_path,
        ratification_path=ratification_path,
        manifest=manifest,
        snapshot_ids=tuple(register_index),
        artifact_root=artifact_root,
    )

    started = time.time()
    event_cache: dict[str, dict[tuple[int, str], np.ndarray]] = {}
    for snapshot_id, labels in register_index.items():
        snapshot = manifest["snapshots"][snapshot_id]
        payload = artifact_root / snapshot["relative_job_dir"] / "bitstrings.json"
        if sha256_file(payload) != payload_hashes[snapshot_id]:
            raise ValueError("Held payload changed after the unblinding record.")
        parsed_events = load_snapshot_events(
            payload,
            snapshot=snapshot,
            register_labels=labels,
        )
        if sha256_file(payload) != payload_hashes[snapshot_id]:
            raise ValueError("Held payload changed during strict parsing.")
        event_cache[snapshot_id] = parsed_events

    state_rows: list[dict[str, Any]] = []
    trace_artifacts: list[dict[str, Any]] = []
    bootstrap_artifacts: list[dict[str, Any]] = []
    bootstrap_eigendecompositions = 0
    fit_eigendecompositions = 0
    actual_eigendecompositions = 0
    adaptive_state_ledger: list[dict[str, Any]] = []
    threshold_replicates = int(manifest["alarm_calibration"]["replicates"])
    block_length = int(manifest["alarm_calibration"]["block_length_shots"])
    alpha = float(manifest["alarm_calibration"]["alpha_per_path_state_episode"])
    for cohort_index, cohort in enumerate(cohorts):
        early = event_cache[cohort.early_snapshot_id]
        late = event_cache[cohort.late_snapshot_id]
        for logical_state in (0, 1):
            early_events = early[(logical_state, cohort.register_suffix)]
            late_events = late[(logical_state, cohort.register_suffix)]
            m = cohort.m
            pre_reference = early_events[:m]
            pre_monitor = early_events[m : 2 * m]
            post_reference = early_events[2 * m : 3 * m]
            post_monitor = late_events[:m]
            if any(
                block.shape != (m, cohort.rounds, cohort.q)
                for block in (
                    pre_reference,
                    pre_monitor,
                    post_reference,
                    post_monitor,
                )
            ):
                raise ValueError(f"Locked stream shape changed for {cohort.cohort_id}.")
            fit = cohort.fit_shots
            fit_bank = DimensionAdaptedBank(
                q=cohort.q,
                role_count=cohort.rounds,
            )
            for shot in range(fit):
                for role in range(cohort.rounds):
                    fit_bank.update_all(
                        role,
                        pre_reference[shot, role],
                        pre_monitor[shot, role],
                    )
            fit_eigendecompositions += (
                cohort.rounds
                * len(SPECTRAL_HALF_LIVES)
                * (fit // SPECTRAL_UPDATE_STRIDE)
            )
            actual_eigendecompositions += (
                cohort.rounds
                * len(SPECTRAL_HALF_LIVES)
                * (
                    (fit + cohort.surveillance_shots) // SPECTRAL_UPDATE_STRIDE
                    - fit // SPECTRAL_UPDATE_STRIDE
                )
            )
            shot_priors = shot_component_weights(cohort.q, cohort.rounds)
            adaptive_state_ledger.append(
                {
                    "cohort_id": cohort.cohort_id,
                    "logical_state": logical_state,
                    "q": cohort.q,
                    "roles": cohort.rounds,
                    "adaptive_bank_numeric_bytes": fit_bank.state_nbytes(),
                    "formal_accumulator_components": {
                        method: len(weights) for method, weights in shot_priors.items()
                    },
                    "formal_accumulator_numeric_bytes": int(
                        sum(3 * len(weights) * 8 for weights in shot_priors.values())
                    ),
                }
            )
            actual = replay_actual(
                fit_bank.clone(),
                pre_reference[fit:],
                pre_monitor[fit:],
                post_reference,
                post_monitor,
            )
            for method_index, method in enumerate(METHOD_ORDER):
                seed = 611000 + 100 * cohort_index + 10 * logical_state + method_index
                threshold, maxima, eigendecompositions = bootstrap_threshold_vectorized(
                    fit_bank=fit_bank,
                    method=method,
                    fit_reference=pre_reference[:fit],
                    fit_monitor=pre_monitor[:fit],
                    seed=seed,
                    replicates=threshold_replicates,
                    horizon_shots=cohort.surveillance_shots,
                    block_length=block_length,
                    alpha=alpha,
                )
                bootstrap_eigendecompositions += eigendecompositions
                trace = actual[method]["log_e"]
                metrics = stream_metrics(
                    trace=trace,
                    threshold=threshold,
                    pre_surveillance_shots=cohort.pre_surveillance_shots,
                    post_shots=m,
                    roles=cohort.rounds,
                )
                stem = f"{cohort_index:02d}_s{logical_state}_{method}"
                trace_path = output / f"{stem}_log_e.npy"
                maxima_path = output / f"{stem}_bootstrap_maxima.npy"
                np.save(trace_path, np.asarray(trace, dtype="<f8"), allow_pickle=False)
                np.save(
                    maxima_path,
                    np.asarray(maxima, dtype="<f8"),
                    allow_pickle=False,
                )
                trace_artifacts.append(_artifact_record(trace_path, base=output))
                bootstrap_artifacts.append(_artifact_record(maxima_path, base=output))
                state_rows.append(
                    {
                        "cohort_id": cohort.cohort_id,
                        "logical_state": logical_state,
                        "method": method,
                        "threshold_seed": seed,
                        "threshold_log_e": threshold,
                        **metrics,
                    }
                )

    aggregate = aggregate_results(state_rows, cohorts)
    (
        randomization_summary,
        randomization_alarm_counts,
        randomization_maximum_log_e,
        randomization_eigendecompositions,
    ) = randomization_audit(
        cohorts=cohorts,
        event_cache=event_cache,
    )
    state_csv = output / "path_state_method_results.csv"
    _save_state_rows(state_csv, state_rows)
    aggregate_path = output / "aggregate_results.json"
    _write_canonical_json(aggregate_path, aggregate)
    randomization_path = output / "randomization_audit.json"
    _write_canonical_json(randomization_path, randomization_summary)
    randomization_counts_path = output / "randomization_alarm_counts.npy"
    np.save(
        randomization_counts_path,
        np.asarray(randomization_alarm_counts, dtype="<i8"),
        allow_pickle=False,
    )
    randomization_maxima_path = output / "randomization_maximum_log_e.npy"
    np.save(
        randomization_maxima_path,
        np.asarray(randomization_maximum_log_e, dtype="<f8"),
        allow_pickle=False,
    )
    finished = time.time()
    fit_paired_shot_pairs = sum(2 * cohort.fit_shots for cohort in cohorts)
    surveillance_paired_shot_pairs = sum(
        2 * cohort.surveillance_shots for cohort in cohorts
    )
    fit_role_score_updates = sum(
        2 * cohort.fit_shots * cohort.rounds for cohort in cohorts
    )
    surveillance_role_score_updates = sum(
        2 * cohort.surveillance_shots * cohort.rounds for cohort in cohorts
    )
    fit_detector_bits = sum(
        2 * 2 * cohort.fit_shots * cohort.rounds * cohort.q for cohort in cohorts
    )
    surveillance_detector_bits = sum(
        2 * 2 * cohort.surveillance_shots * cohort.rounds * cohort.q
        for cohort in cohorts
    )
    bootstrap_surrogate_shot_updates = sum(
        2 * len(METHOD_ORDER) * threshold_replicates * cohort.surveillance_shots
        for cohort in cohorts
    )
    bootstrap_surrogate_role_score_updates = sum(
        2
        * len(METHOD_ORDER)
        * threshold_replicates
        * cohort.surveillance_shots
        * cohort.rounds
        for cohort in cohorts
    )
    randomization_replicates = len(randomization_summary["seeds"])
    randomization_surrogate_shot_updates = sum(
        2 * len(METHOD_ORDER) * randomization_replicates * cohort.m
        for cohort in cohorts
    )
    randomization_surrogate_role_score_updates = sum(
        2 * len(METHOD_ORDER) * randomization_replicates * cohort.m * cohort.rounds
        for cohort in cohorts
    )
    output_bytes_before_results_manifest = sum(
        path.stat().st_size for path in output.iterdir() if path.is_file()
    )
    manifest_output = {
        "schema_version": "run6-pnnl-snapshot-results-v1",
        "protocol_id": config["protocol_id"],
        "claim_label": (
            "constructed circuit-and-hardware domain shift; not temporal drift"
        ),
        "formal_alarm_unit": "one update per complete paired shot",
        "within_shot_roles": (
            "fixed experts mixed with an explicit uniform role prior; "
            "not sequential formal updates"
        ),
        "git_commit": _git_commit(repo_root),
        "config_sha256": sha256_file(config_path),
        "pittsburgh_manifest_sha256": sha256_file(manifest_path),
        "freeze_ratification_sha256": sha256_file(ratification_path),
        "package_lock_sha256": sha256_file(repo_root / PACKAGE_LOCK_RELATIVE),
        "first_unblinding_record": _artifact_record(unblinding_path, base=output),
        "metadata_validation": {
            "snapshots": validation.snapshots,
            "cohorts": validation.cohorts,
            "held_payloads_statted": validation.held_payloads_statted,
        },
        "held_payload_sha256": payload_hashes,
        "state_rows": _artifact_record(state_csv, base=output),
        "aggregate_results": _artifact_record(aggregate_path, base=output),
        "randomization_audit": _artifact_record(randomization_path, base=output),
        "randomization_alarm_counts": _artifact_record(
            randomization_counts_path,
            base=output,
        ),
        "randomization_maximum_log_e": _artifact_record(
            randomization_maxima_path,
            base=output,
        ),
        "trace_artifacts": trace_artifacts,
        "bootstrap_artifacts": bootstrap_artifacts,
        "resource_ledger": {
            "path_groups": len(cohorts),
            "path_state_streams": 2 * len(cohorts),
            "paired_shots_per_pre_or_post_phase": sum(
                2 * cohort.m for cohort in cohorts
            ),
            "paired_cycle_updates_per_pre_or_post_phase": sum(
                2 * cohort.m * cohort.rounds for cohort in cohorts
            ),
            "fit_paired_shot_pairs": fit_paired_shot_pairs,
            "fit_physical_circuit_shots": 2 * fit_paired_shot_pairs,
            "fit_role_score_updates": fit_role_score_updates,
            "fit_detector_event_bits_consumed": fit_detector_bits,
            "surveillance_paired_shot_pairs": (surveillance_paired_shot_pairs),
            "surveillance_physical_circuit_shots": (2 * surveillance_paired_shot_pairs),
            "surveillance_formal_eprocess_shot_updates": (
                surveillance_paired_shot_pairs
            ),
            "surveillance_role_score_updates": (surveillance_role_score_updates),
            "surveillance_detector_event_bits_consumed": (surveillance_detector_bits),
            "bootstrap_replicates_per_path_state_method": threshold_replicates,
            "bootstrap_surrogate_shot_updates": (bootstrap_surrogate_shot_updates),
            "bootstrap_surrogate_role_score_updates": (
                bootstrap_surrogate_role_score_updates
            ),
            "randomization_replicates": randomization_replicates,
            "randomization_surrogate_shot_updates": (
                randomization_surrogate_shot_updates
            ),
            "randomization_surrogate_role_score_updates": (
                randomization_surrogate_role_score_updates
            ),
            "fit_eigendecompositions": fit_eigendecompositions,
            "actual_surveillance_eigendecompositions": (actual_eigendecompositions),
            "bootstrap_eigendecompositions": bootstrap_eigendecompositions,
            "randomization_eigendecompositions": (randomization_eigendecompositions),
            "adaptive_state_ledger": adaptive_state_ledger,
            "wall_seconds": finished - run_started,
            "held_value_processing_wall_seconds": finished - started,
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "output_bytes_excluding_results_manifest": (
                output_bytes_before_results_manifest
            ),
            "output_bytes_including_results_manifest": 0,
        },
        "retention_pass": aggregate["retention_pass"],
        "environment": environment_fingerprint(),
        "command": sys.argv,
        "started_unix": run_started,
        "held_value_processing_started_unix": started,
        "finished_unix": finished,
    }
    result_manifest = output / "results_manifest.json"
    for _ in range(16):
        encoded_manifest = canonical_json_bytes(manifest_output) + b"\n"
        total_output_bytes = output_bytes_before_results_manifest + len(
            encoded_manifest
        )
        if (
            manifest_output["resource_ledger"][
                "output_bytes_including_results_manifest"
            ]
            == total_output_bytes
        ):
            break
        manifest_output["resource_ledger"][
            "output_bytes_including_results_manifest"
        ] = total_output_bytes
    else:  # pragma: no cover - integer digit length converges immediately
        raise RuntimeError("Self-sized results manifest did not converge.")
    result_manifest.write_bytes(encoded_manifest)
    print(json.dumps(manifest_output, indent=2, sort_keys=True))


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.dry_run:
        print(json.dumps(synthetic_dry_run(), indent=2, sort_keys=True))
        return 0
    run_real(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
