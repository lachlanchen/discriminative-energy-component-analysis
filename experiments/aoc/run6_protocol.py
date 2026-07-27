"""Strict configuration, hashing, and outcome-embargo helpers for Run 6."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import platform
import re
import subprocess
import sys
from collections.abc import Iterable, Mapping, Sequence
from importlib.metadata import PackageNotFoundError, distributions, version
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

GOOGLE_TOP_LEVEL_KEYS = frozenset(
    {
        "protocol_id",
        "status",
        "data_status",
        "normative_method_spec",
        "source",
        "shot_splits",
        "validation_pair_phases",
        "event_windows",
        "detector_parser",
        "pairing",
        "features",
        "methods",
        "accumulators",
        "empirical_operating_points",
        "randomization_audit",
        "risk_audit",
        "decision",
        "uncertainty",
        "numeric_policy",
        "resource_ledger",
        "output_contract",
        "deviation_ledger",
    }
)

GOOGLE_METHOD_KEYS = frozenset(
    {
        "dfr",
        "within_shot_page_cusum",
        "diagonal_likelihood",
        "hotelling",
        "online_logistic",
        "space_sparse",
        "space_spectral",
        "space_composite",
    }
)

GOOGLE_NESTED_KEYSETS: dict[tuple[str, ...], frozenset[str]] = {
    ("normative_method_spec",): frozenset({"path", "sha256"}),
    ("source",): frozenset(
        {
            "record_url",
            "doi",
            "license",
            "archive",
            "archive_bytes",
            "md5",
            "sha256",
            "experiment",
            "shots",
            "distance",
            "measurement_rounds",
            "detector_round_roles",
            "checks_per_round_role",
            "detectors_per_shot",
            "detection_event_file",
            "detection_event_file_bytes",
        }
    ),
    ("shot_splits",): frozenset(
        {
            "validation_a",
            "validation_b",
            "held_reference",
            "held_monitor",
            "untouched_future",
        }
    ),
    ("validation_pair_phases",): frozenset(
        {
            "fit_warmup_pair_indices",
            "threshold_pair_indices",
            "threshold_and_held_clone_same_post_warmup_checkpoint",
            "threshold_state_never_carried_into_held",
        }
    ),
    ("event_windows",): frozenset(
        {
            "readme_grep_line_one_based",
            "grep_line_stored_index_zero_based",
            "readme_prose_anchor_approximate",
            "primary",
            "narrow",
            "wide",
        }
    ),
    ("detector_parser",): frozenset(
        {
            "coordinate_source",
            "canonical_key",
            "canonical_order",
            "roles",
            "boundary_declaration_order_differs",
            "packed_bit_source_order",
            "bit_order_within_byte",
            "bytes_per_shot",
            "shot_byte_aligned",
            "padding_bits_per_shot",
            "expected_file_bytes",
            "logical_flatten_order",
            "empirical_cycle_index",
            "one_based_formal_time",
            "naive_reshape_forbidden",
            "require_24_unique_checks_per_role",
            "require_identical_canonical_check_set_across_roles",
        }
    ),
    ("pairing",): frozenset(
        {
            "validation",
            "held",
            "match_round_role",
            "preserve_shot_order",
            "role_processing_order",
            "adaptive_state_sharing",
            "updates_per_held_episode",
            "formal_accumulator_unit",
            "formal_accumulator_updates_per_held_episode",
            "formal_role_experts",
            "formal_role_prior",
            "within_shot_factor_compounding",
        }
    ),
    ("features",): frozenset(
        {
            "global_detector_rate",
            "raw_check_bits",
            "all_pair_equalities",
            "sparse_feature_dimension",
            "feature_order",
            "spectral_state",
            "geometry_selected_after_data",
        }
    ),
    ("methods",): GOOGLE_METHOD_KEYS,
    ("methods", "dfr"): frozenset(
        {
            "score",
            "two_sided_bet_magnitudes",
            "signed_component_order",
            "component_prior",
            "empirical_cycle_score",
        }
    ),
    ("methods", "within_shot_page_cusum"): frozenset(
        {
            "status",
            "channels",
            "two_sided",
            "drift_allowances",
            "reset",
            "empirical_cycle_score",
            "exact_e_factor_claim",
            "advantage_gate_comparator",
        }
    ),
    ("methods", "diagonal_likelihood"): frozenset(
        {
            "fit_validation_pair_indices",
            "fit_separately_for_each_role_and_check",
            "pool_reference_and_monitor_sides",
            "jeffreys_pseudocount",
            "probability_clip",
            "nll_reduction",
            "normalizer",
            "update_after_fit",
            "two_sided_bet_magnitudes",
            "component_prior",
            "empirical_cycle_score",
        }
    ),
    ("methods", "hotelling"): frozenset(
        {
            "feature_bank",
            "centering",
            "fit_validation_pair_indices",
            "fit_cycle_pairs",
            "selection_rng",
            "selection_per_role",
            "estimator",
            "assume_centered",
            "precision_symmetrization",
            "negative_roundoff_tolerance",
            "online_update",
            "exact_e_factor_claim",
        }
    ),
    ("methods", "online_logistic"): frozenset(
        {
            "learning_rates",
            "l2",
            "intercept",
            "initial_weights",
            "orientation",
            "loss",
            "optimizer",
            "score",
            "update_after_score",
            "bet_fractions",
            "expert_component_prior",
            "empirical_cycle_score",
        }
    ),
    ("methods", "space_sparse"): frozenset(
        {
            "half_lives_role_updates",
            "decay",
            "ewma_bias_correction",
            "top_k",
            "initial_ewma",
            "zero_state_witness",
            "tie_rule",
            "selected_zero_sign",
            "update_after_score",
            "bet_fractions",
            "component_prior",
            "empirical_cycle_score",
        }
    ),
    ("methods", "space_spectral"): frozenset(
        {
            "half_lives_role_updates",
            "decay",
            "ewma_bias_correction",
            "ranks",
            "initial_ewma_and_effect",
            "eigenvalue_tolerance",
            "rank_one_degeneracy",
            "no_positive_eigenvalue_effect",
            "effect_update_stride_role_updates",
            "stride_phase",
            "update_after_score",
            "bet_fractions",
            "component_prior",
            "empirical_cycle_score",
        }
    ),
    ("methods", "space_composite"): frozenset(
        {
            "fixed_branch_prior",
            "empirical_cycle_score",
            "branch_selection_from_held",
            "primary_proposed_method",
        }
    ),
    ("methods", "space_composite", "fixed_branch_prior"): frozenset(
        {"space_sparse", "space_spectral"}
    ),
    ("accumulators",): frozenset({"primary", "secondary"}),
    ("accumulators", "primary"): frozenset(
        {
            "kind",
            "start_prior",
            "horizon_unit",
            "horizon_shots",
            "randomization_horizon_shots",
            "role_component_experts",
            "role_prior",
            "alpha",
            "threshold",
            "reset_in_episode",
            "numeric_domain",
        }
    ),
    ("accumulators", "secondary"): frozenset(
        {
            "kind",
            "gamma_shots",
            "formal_summary",
            "reset_in_episode",
            "numeric_domain",
        }
    ),
    ("empirical_operating_points",): frozenset(
        {
            "threshold_validation_pair_indices",
            "primary",
            "secondary",
            "shot_aggregation",
            "notification_rule",
            "threshold_rule",
            "tie_rule",
            "notifications_per_shot",
            "after_notification",
            "next_shot_cooldown",
            "witness_reset_on_alert",
            "complete_frontier_required",
        }
    ),
    ("empirical_operating_points", "primary"): frozenset(
        {
            "unit",
            "alerts_per_opportunity",
            "validation_opportunities",
            "maximum_validation_alerts",
        }
    ),
    ("empirical_operating_points", "secondary"): frozenset(
        {
            "unit",
            "alerts_per_opportunity",
            "validation_opportunities",
            "maximum_validation_alerts",
        }
    ),
    ("randomization_audit",): frozenset(
        {
            "replicates",
            "replicate_seeds",
            "rng",
            "pair_indices",
            "swap_draw",
            "swap_unit",
            "same_orientation_for_all_51_roles",
            "restore_same_post_warmup_checkpoint_each_replicate",
            "primary_process",
            "primary_statistic",
            "horizon_unit",
            "horizon_shots",
            "alpha",
            "interval",
            "familywide_optional_threshold",
        }
    ),
    ("risk_audit",): frozenset(
        {
            "open_labels_only_after_detector_freeze",
            "labels",
            "primary_label",
            "secondary_label",
            "alert_rates_per_shot",
            "alert_budgets_shots",
            "primary_alert_budget_shots",
            "ranking",
            "bootstrap",
        }
    ),
    ("risk_audit", "bootstrap"): frozenset(
        {
            "kind",
            "block_length_shots",
            "replicates",
            "seeds",
            "rng",
            "interval",
        }
    ),
    ("decision",): frozenset(
        {
            "predesignated_comparators",
            "mandatory_contextual_controls",
            "google_primary_pass_all",
            "pnnl_auxiliary_required",
            "untouched_google_alternative_for_retention",
            "overall_run6_advantage",
            "default_if_any_predicate_fails",
            "score_comparison_tolerance",
        }
    ),
    ("uncertainty",): frozenset(
        {"natural_event_miss_probability_forbidden", "threshold_bootstrap"}
    ),
    ("uncertainty", "threshold_bootstrap"): frozenset(
        {"status", "block_length_shots", "replicates", "seeds", "rng"}
    ),
    ("numeric_policy",): frozenset(
        {
            "floating_dtype",
            "score_bound_tolerance",
            "eigenvalue_tolerance",
            "empirical_threshold_comparison",
            "eprocess_threshold_comparison",
            "implicit_global_rng_forbidden",
            "component_pruning_forbidden",
            "thread_environment",
        }
    ),
    ("numeric_policy", "thread_environment"): frozenset(
        {
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        }
    ),
    ("resource_ledger",): frozenset(
        {
            "fit_warmup",
            "threshold",
            "held",
            "timing_repetitions",
            "timing_summary",
            "unreported_warmup_replay",
            "timing_scope",
            "relative_method_speed_claim_authorized",
            "all_replay_digests_must_match",
            "threads",
            "outcome_join_excluded_from_detector_timing",
        }
    ),
    ("resource_ledger", "fit_warmup"): frozenset(
        {
            "paired_shots",
            "physical_archived_shots",
            "paired_role_updates",
            "detector_bits_exposed",
        }
    ),
    ("resource_ledger", "threshold"): frozenset(
        {
            "paired_shots",
            "physical_archived_shots",
            "paired_role_updates",
            "detector_bits_exposed",
        }
    ),
    ("resource_ledger", "held"): frozenset(
        {
            "paired_shots",
            "physical_archived_shots",
            "paired_role_updates",
            "detector_bits_exposed",
        }
    ),
    ("output_contract",): frozenset(
        {
            "cycle_arrays",
            "shot_table",
            "component_summary",
            "detector_freeze_manifest",
            "outcome_table",
            "decision_summary",
            "pickle_forbidden",
            "unknown_or_missing_schema_fields",
        }
    ),
}

OUTCOME_FILENAMES = frozenset(
    {
        "obs_flips_actual.01",
        "obs_flips_predicted_by_correlated_matching.01",
        "obs_flips_predicted_by_pymatching.01",
    }
)

PINNED_PACKAGES = (
    "cvxpy",
    "matplotlib",
    "numpy",
    "pandas",
    "pymatching",
    "pytest",
    "scikit-learn",
    "scipy",
    "seaborn",
    "stim",
)

FREEZE_MANIFEST_RELATIVE = "experiments/run6/freeze_manifest.json"
FREEZE_RATIFICATION_RELATIVE = "experiments/run6/freeze_ratification.json"
FREEZE_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "implementation_commit",
        "hashes",
        "environment",
        "thread_environment",
        "held_value_access_before_freeze",
        "source_payload_values_accessed_before_freeze",
    }
)
FREEZE_RATIFICATION_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "freeze_commit",
        "hashes",
        "environment",
        "thread_environment",
        "held_value_access_before_ratification",
    }
)
_HEX_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_LOCK_PIN_RE = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([^=\s]+)$")

RUN6_REQUIRED_FREEZE_PATHS = (
    "experiments/aoc/__init__.py",
    "experiments/aoc/change_detection.py",
    "experiments/aoc/chemistry.py",
    "experiments/aoc/contrast.py",
    "experiments/aoc/gauge.py",
    "experiments/aoc/manybody.py",
    "experiments/aoc/multiclass.py",
    "experiments/aoc/physics.py",
    "experiments/aoc/qec_real.py",
    "experiments/aoc/quantum.py",
    "experiments/aoc/repro.py",
    "experiments/aoc/run6_protocol.py",
    "experiments/aoc/space.py",
    "experiments/aoc/space_qec.py",
    "experiments/aoc/states.py",
    "experiments/aoc/streaming.py",
    "experiments/aoc/surface_code.py",
    "experiments/aoc/surface_code_stim.py",
    "experiments/aoc/symmetry.py",
    "experiments/pyproject.toml",
    "experiments/run6/README.md",
    "experiments/run6/configs/google2022_locked.json",
    "experiments/run6/configs/pnnl_pittsburgh_locked.json",
    "experiments/run6/configs/pnnl_snapshot_locked.json",
    "experiments/run6/configs/python_environment_lock.txt",
    "experiments/run6/deviations.json",
    "experiments/run6/scripts/create_freeze_chain.py",
    "experiments/run6/scripts/launch_google2022_randomization.py",
    "experiments/run6/scripts/run_google2022_detector.py",
    "experiments/run6/scripts/run_google2022_outcomes.py",
    "experiments/run6/scripts/run_google2022_randomization.py",
    "experiments/run6/scripts/run_pnnl_snapshot.py",
    "experiments/run6/scripts/validate_pnnl_lock.py",
    "experiments/run6/tests/test_google2022_outcomes.py",
    "experiments/run6/tests/test_google2022_randomization.py",
    "experiments/run6/tests/test_google2022_randomization_launcher.py",
    "experiments/run6/tests/test_google_detector_runner.py",
    "experiments/run6/tests/test_pnnl_lock_metadata.py",
    "experiments/run6/tests/test_pnnl_snapshot_runner.py",
    "experiments/run6/tests/test_qec_real.py",
    "experiments/run6/tests/test_run6_protocol.py",
    "experiments/run6/tests/test_space.py",
    "experiments/run6/tests/test_space_qec.py",
    "references/run6_cross_domain_application_audit.md",
    "references/run6_final_predata_audit.md",
    "references/run6_google2022_data_map.md",
    "references/run6_method_lock_recommendations.md",
    "references/run6_pnnl_locked_manifest_recommendations.md",
    "references/run6_pnnl_snapshot_audit.md",
    "references/run6_predata_blocker_resolution.md",
    "references/run6_preregistration_adversarial_audit.md",
    "references/run6_publication_literature_audit.md",
    "references/run6_real_qec_data_and_comparator_audit.md",
    "references/run6_real_qec_preregistered_plan.md",
    "references/run6_real_structural_monitoring_audit.md",
    "references/run6_runner_integration_audit.md",
    "references/run6_space_final_theory.md",
    "references/run6_space_math_audit.md",
    "references/run6_space_math_reaudit.md",
    "references/run6_symmetry_scan_eprocess_theory.md",
    "references/run6_theory_adversarial_audit.md",
    "references/run6_theory_integration_plan.md",
)


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key!r}.")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON number is forbidden: {value}.")


def load_strict_json(path: str | Path) -> dict[str, Any]:
    """Load one UTF-8 JSON object, rejecting duplicates and non-finite values."""

    source = Path(path)
    value = json.loads(
        source.read_text(encoding="utf-8-sig"),
        object_pairs_hook=_reject_duplicate_pairs,
        parse_constant=_reject_nonfinite_constant,
    )
    if not isinstance(value, dict):
        raise TypeError(f"{source} must contain a JSON object.")
    return value


def canonical_json_bytes(value: Any) -> bytes:
    """Return the repository's declared compact, sorted UTF-8 JSON profile."""

    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    """Hash :func:`canonical_json_bytes` with SHA-256."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: str | Path, *, chunk_bytes: int = 1 << 20) -> str:
    """Hash exact file bytes without loading the complete file."""

    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def _safe_repo_relative(relative: Any, *, context: str) -> str:
    """Validate one canonical repository-relative POSIX path."""

    if not isinstance(relative, str) or not relative:
        raise TypeError(f"{context} must be a nonempty string.")
    candidate = PurePosixPath(relative)
    if (
        candidate.is_absolute()
        or candidate.as_posix() != relative
        or any(part in {"", ".", ".."} for part in candidate.parts)
        or "\\" in relative
        or ":" in relative
    ):
        raise ValueError(f"{context} is not a canonical repository-relative path.")
    return relative


def _git_output(
    repo_root: Path,
    arguments: Sequence[str],
    *,
    error: str,
) -> bytes:
    """Run one non-mutating Git query and return its exact stdout."""

    result = subprocess.run(
        ["git", *arguments],
        cwd=repo_root,
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(f"{error}: {detail or 'git query failed'}")
    return result.stdout


def _git_commit(repo_root: Path, revision: str, *, context: str) -> str:
    if not isinstance(revision, str) or not revision:
        raise TypeError(f"{context} must be a nonempty string.")
    resolved = (
        _git_output(
            repo_root,
            ["rev-parse", "--verify", f"{revision}^{{commit}}"],
            error=f"{context} is not a Git commit",
        )
        .decode("ascii")
        .strip()
    )
    if re.fullmatch(r"[0-9a-f]{40,64}", resolved) is None:
        raise ValueError(f"{context} resolved to an invalid object identifier.")
    return resolved


def _require_git_ancestor(
    repo_root: Path,
    ancestor: str,
    descendant: str,
    *,
    context: str,
) -> None:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=repo_root,
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        raise ValueError(f"{context}: {ancestor} is not an ancestor of {descendant}.")


def _git_blob_sha256(repo_root: Path, commit: str, relative: str) -> str:
    payload = _git_output(
        repo_root,
        ["show", f"{commit}:{relative}"],
        error=f"Cannot read committed artifact {relative!r} at {commit}",
    )
    return hashlib.sha256(payload).hexdigest()


def _validate_hash_registry(
    value: Any,
    *,
    context: str,
) -> dict[str, str]:
    if not isinstance(value, Mapping) or not value:
        raise TypeError(f"{context} must be a nonempty object.")
    result: dict[str, str] = {}
    for raw_relative, raw_digest in value.items():
        relative = _safe_repo_relative(
            raw_relative,
            context=f"{context} path",
        )
        if (
            not isinstance(raw_digest, str)
            or _HEX_DIGEST_RE.fullmatch(raw_digest) is None
        ):
            raise ValueError(f"{context}.{relative} is not a SHA-256 digest.")
        result[relative] = raw_digest
    return result


def _normalized_distribution_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def verify_python_environment_lock(
    lock_path: str | Path,
    *,
    allowed_unlocked: Iterable[str] = ("deca-experiments",),
    installed_versions: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Require an exact installed-distribution match to the Run 6 lock."""

    pins: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        Path(lock_path).read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _LOCK_PIN_RE.fullmatch(line)
        if match is None:
            raise ValueError(
                f"Invalid environment-lock pin on line {line_number}: {raw_line!r}."
            )
        normalized = _normalized_distribution_name(match.group(1))
        if normalized in pins:
            raise ValueError(
                f"Duplicate normalized distribution in environment lock: {normalized}."
            )
        pins[normalized] = match.group(2)
    if not pins:
        raise ValueError("Python environment lock contains no distribution pins.")

    allowed = {_normalized_distribution_name(name) for name in allowed_unlocked}
    observed_allowed: set[str] = set()
    if installed_versions is None:
        installed: dict[str, str] = {}
        for distribution in distributions():
            raw_name = distribution.metadata.get("Name")
            if not raw_name:
                raise ValueError("An installed distribution has no metadata Name.")
            normalized = _normalized_distribution_name(raw_name)
            if normalized in allowed:
                observed_allowed.add(normalized)
                continue
            observed_version = distribution.version
            if normalized in installed and installed[normalized] != observed_version:
                raise ValueError(
                    "Conflicting installed versions for normalized distribution "
                    f"{normalized}."
                )
            installed[normalized] = observed_version
    else:
        installed = {}
        for raw_name, observed_version in installed_versions.items():
            normalized = _normalized_distribution_name(raw_name)
            if normalized in allowed:
                observed_allowed.add(normalized)
                continue
            if normalized in installed:
                raise ValueError(
                    f"Duplicate normalized installed distribution: {normalized}."
                )
            if not isinstance(observed_version, str) or not observed_version:
                raise TypeError(
                    f"Installed version for {normalized} must be a nonempty string."
                )
            installed[normalized] = observed_version

    missing = sorted(set(pins) - set(installed))
    mismatched = sorted(
        name for name in set(pins) & set(installed) if pins[name] != installed[name]
    )
    unexpected = sorted(set(installed) - set(pins) - allowed)
    if missing or mismatched or unexpected:
        mismatch_detail = {
            name: {"locked": pins[name], "installed": installed[name]}
            for name in mismatched
        }
        raise ValueError(
            "Python environment differs from the full lock; "
            f"missing={missing}, mismatched={mismatch_detail}, "
            f"unexpected={unexpected}."
        )
    absent_allowed = sorted(allowed - observed_allowed)
    if absent_allowed:
        raise ValueError(
            "Explicitly excluded editable distribution is not installed: "
            f"{absent_allowed}."
        )
    return pins


def verify_runtime_module_origins(repo_root: str | Path) -> dict[str, str]:
    """Require every Run 6 shared module to load from the frozen repository."""

    root = Path(repo_root).resolve()
    expected = {
        "aoc": "experiments/aoc/__init__.py",
        "aoc.qec_real": "experiments/aoc/qec_real.py",
        "aoc.run6_protocol": "experiments/aoc/run6_protocol.py",
        "aoc.space": "experiments/aoc/space.py",
        "aoc.space_qec": "experiments/aoc/space_qec.py",
    }
    observed: dict[str, str] = {}
    for module_name, relative in expected.items():
        module = importlib.import_module(module_name)
        raw_path = getattr(module, "__file__", None)
        if not isinstance(raw_path, str):
            raise TypeError(f"Runtime module {module_name} has no source path.")
        path = Path(raw_path).resolve()
        if path != (root / relative).resolve():
            raise RuntimeError(
                f"Runtime module {module_name} loaded from {path}, "
                f"not frozen source {root / relative}."
            )
        observed[module_name] = relative
    return observed


def verify_committed_freeze_chain(
    ratification_path: str | Path,
    *,
    repo_root: str | Path,
    required_paths: Iterable[str],
    expected_environment: Mapping[str, Any],
    expected_thread_environment: Mapping[str, str],
    remote_ref: str = "origin/main",
) -> dict[str, Any]:
    """Verify the complete pushed Run 6 freeze chain.

    This check is intentionally stronger than comparing worktree hashes.  It
    proves that every frozen byte is present in both the implementation
    commit and the later freeze commit, that both commits are reachable from
    the locally recorded pushed branch, and that the ratification itself is
    the tracked byte sequence at the current pushed ``HEAD``.
    """

    root = Path(repo_root).resolve()
    supplied_ratification = Path(ratification_path).resolve()
    expected_ratification = (root / FREEZE_RATIFICATION_RELATIVE).resolve()
    if supplied_ratification != expected_ratification:
        raise ValueError(
            "The real-data gate requires the repository's canonical "
            f"{FREEZE_RATIFICATION_RELATIVE}."
        )

    head = _git_commit(root, "HEAD", context="HEAD")
    remote = _git_commit(root, remote_ref, context=remote_ref)
    _require_git_ancestor(
        root,
        head,
        remote,
        context="Current HEAD has not been pushed to the recorded remote branch",
    )
    if not supplied_ratification.is_file():
        raise FileNotFoundError("The freeze ratification is missing.")
    ratification_bytes = supplied_ratification.read_bytes()
    if hashlib.sha256(ratification_bytes).hexdigest() != _git_blob_sha256(
        root,
        head,
        FREEZE_RATIFICATION_RELATIVE,
    ):
        raise ValueError("Freeze ratification differs from the tracked HEAD blob.")

    ratification = load_strict_json(supplied_ratification)
    require_exact_keys(
        ratification,
        FREEZE_RATIFICATION_KEYS,
        context="freeze ratification",
    )
    if (
        ratification["schema_version"] != "run6-freeze-ratification-v1"
        or ratification["status"] != "frozen_before_held_value_access"
        or ratification["held_value_access_before_ratification"] is not False
    ):
        raise ValueError("Freeze ratification does not authorize held-value access.")
    freeze_commit = _git_commit(
        root,
        ratification["freeze_commit"],
        context="freeze_commit",
    )
    _require_git_ancestor(
        root,
        freeze_commit,
        head,
        context="Freeze commit is not in current history",
    )
    _require_git_ancestor(
        root,
        freeze_commit,
        remote,
        context="Freeze commit has not been pushed",
    )

    ratified_hashes = _validate_hash_registry(
        ratification["hashes"],
        context="ratification.hashes",
    )
    if FREEZE_MANIFEST_RELATIVE not in ratified_hashes:
        raise ValueError("Ratification does not bind the freeze manifest.")
    manifest_path = root / FREEZE_MANIFEST_RELATIVE
    if not manifest_path.is_file():
        raise FileNotFoundError("Freeze manifest is missing.")
    manifest_digest = sha256_file(manifest_path)
    if ratified_hashes[FREEZE_MANIFEST_RELATIVE] != manifest_digest:
        raise ValueError("Freeze manifest differs from the ratified digest.")
    if (
        _git_blob_sha256(root, freeze_commit, FREEZE_MANIFEST_RELATIVE)
        != manifest_digest
    ):
        raise ValueError("Freeze manifest is not the blob in freeze_commit.")

    manifest = load_strict_json(manifest_path)
    require_exact_keys(manifest, FREEZE_MANIFEST_KEYS, context="freeze manifest")
    if (
        manifest["schema_version"] != "run6-freeze-manifest-v1"
        or manifest["status"] != "implementation_frozen_before_held_value_access"
        or manifest["held_value_access_before_freeze"] is not False
        or manifest["source_payload_values_accessed_before_freeze"] is not False
    ):
        raise ValueError("Freeze manifest records an invalid pre-access state.")
    implementation_commit = _git_commit(
        root,
        manifest["implementation_commit"],
        context="implementation_commit",
    )
    _require_git_ancestor(
        root,
        implementation_commit,
        freeze_commit,
        context="Implementation commit is not an ancestor of freeze_commit",
    )
    _require_git_ancestor(
        root,
        implementation_commit,
        remote,
        context="Implementation commit has not been pushed",
    )

    manifest_hashes = _validate_hash_registry(
        manifest["hashes"],
        context="freeze_manifest.hashes",
    )
    expected_ratified_hashes = {
        **manifest_hashes,
        FREEZE_MANIFEST_RELATIVE: manifest_digest,
    }
    if ratified_hashes != expected_ratified_hashes:
        raise ValueError(
            "Ratification hashes must equal the implementation registry plus "
            "the freeze manifest."
        )

    normalized_required = {
        _safe_repo_relative(path, context="required freeze path")
        for path in required_paths
    }
    missing = sorted(normalized_required - set(manifest_hashes))
    if missing:
        raise ValueError(f"Freeze manifest omits required artifacts: {missing}")

    runtime_roots = ("experiments/aoc", "experiments/run6/scripts")
    tracked_runtime_sources = {
        line
        for line in _git_output(
            root,
            ["ls-files", "--", *runtime_roots],
            error="Cannot enumerate tracked runtime sources",
        )
        .decode("utf-8")
        .splitlines()
        if line.endswith(".py")
    }
    omitted_runtime_sources = sorted(tracked_runtime_sources - set(manifest_hashes))
    if omitted_runtime_sources:
        raise ValueError(
            f"Freeze manifest omits tracked runtime sources: {omitted_runtime_sources}"
        )
    untracked_runtime_sources = [
        line
        for line in _git_output(
            root,
            [
                "ls-files",
                "--others",
                "--exclude-standard",
                "--",
                *runtime_roots,
            ],
            error="Cannot enumerate untracked runtime sources",
        )
        .decode("utf-8")
        .splitlines()
        if line.endswith(".py")
    ]
    if untracked_runtime_sources:
        raise ValueError(
            "Untracked Python under Run 6 runtime roots is forbidden: "
            f"{sorted(untracked_runtime_sources)}"
        )

    for relative, expected_digest in manifest_hashes.items():
        artifact = (root / relative).resolve()
        try:
            artifact.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"Frozen path escapes repository: {relative}") from exc
        if not artifact.is_file() or sha256_file(artifact) != expected_digest:
            raise ValueError(f"Frozen worktree artifact changed: {relative}")
        if _git_blob_sha256(root, implementation_commit, relative) != expected_digest:
            raise ValueError(
                f"Frozen artifact is not the implementation-commit blob: {relative}"
            )
        if _git_blob_sha256(root, freeze_commit, relative) != expected_digest:
            raise ValueError(
                f"Frozen artifact is not the freeze-commit blob: {relative}"
            )

    expected_environment_dict = dict(expected_environment)
    expected_threads_dict = dict(expected_thread_environment)
    if (
        manifest["environment"] != expected_environment_dict
        or ratification["environment"] != expected_environment_dict
    ):
        raise ValueError("Frozen numeric environment differs from runtime.")
    if (
        manifest["thread_environment"] != expected_threads_dict
        or ratification["thread_environment"] != expected_threads_dict
    ):
        raise ValueError("Frozen thread environment differs from the lock.")
    require_thread_environment(expected_threads_dict)
    verify_python_environment_lock(
        root / "experiments/run6/configs/python_environment_lock.txt"
    )
    verify_runtime_module_origins(root)
    return ratification


def require_exact_keys(
    value: Mapping[str, Any],
    expected: Iterable[str],
    *,
    context: str,
) -> None:
    """Reject both missing and unknown fields."""

    expected_set = frozenset(expected)
    observed = frozenset(value)
    missing = sorted(expected_set - observed)
    unknown = sorted(observed - expected_set)
    if missing or unknown:
        raise ValueError(
            f"{context} schema mismatch; missing={missing}, unknown={unknown}."
        )


def _require_interval(
    value: Any,
    *,
    context: str,
    expected: tuple[int, int] | None = None,
) -> tuple[int, int]:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        raise TypeError(f"{context} must be a two-integer half-open interval.")
    start, stop = value
    if start < 0 or stop <= start:
        raise ValueError(f"{context} is not a valid half-open interval.")
    result = (start, stop)
    if expected is not None and result != expected:
        raise ValueError(f"{context} changed from locked value {expected}.")
    return result


def validate_google_lock(config: Mapping[str, Any]) -> None:
    """Validate critical Run 6 Google choices without supplying defaults."""

    require_exact_keys(config, GOOGLE_TOP_LEVEL_KEYS, context="Google lock")
    for path, expected_keys in GOOGLE_NESTED_KEYSETS.items():
        nested: Any = config
        for part in path:
            if not isinstance(nested, Mapping):
                raise TypeError(f"{'.'.join(path)} must be an object.")
            nested = nested[part]
        if not isinstance(nested, Mapping):
            raise TypeError(f"{'.'.join(path)} must be an object.")
        require_exact_keys(
            nested,
            expected_keys,
            context="Google lock." + ".".join(path),
        )
    if config["protocol_id"] != "run6-google2022-v2":
        raise ValueError("Unexpected Google protocol_id.")
    if config["status"] not in {
        "executable_lock_pending_final_freeze_commit",
        "frozen_before_held_value_access",
    }:
        raise ValueError("Google lock has an invalid status.")

    source = config["source"]
    if not isinstance(source, Mapping):
        raise TypeError("source must be an object.")
    if (
        source.get("shots") != 500_000
        or source.get("detectors_per_shot") != 1_224
        or source.get("detection_event_file_bytes") != 76_500_000
    ):
        raise ValueError("Google source dimensions or byte size changed.")

    splits = config["shot_splits"]
    if not isinstance(splits, Mapping):
        raise TypeError("shot_splits must be an object.")
    require_exact_keys(
        splits,
        {
            "validation_a",
            "validation_b",
            "held_reference",
            "held_monitor",
            "untouched_future",
        },
        context="shot_splits",
    )
    locked_splits = {
        "validation_a": (0, 10_000),
        "validation_b": (10_000, 20_000),
        "held_reference": (20_000, 40_000),
        "held_monitor": (40_000, 60_000),
        "untouched_future": (60_000, 500_000),
    }
    for name, expected in locked_splits.items():
        _require_interval(
            splits[name], context=f"shot_splits.{name}", expected=expected
        )

    phases = config["validation_pair_phases"]
    if not isinstance(phases, Mapping):
        raise TypeError("validation_pair_phases must be an object.")
    if _require_interval(
        phases.get("fit_warmup_pair_indices"),
        context="fit_warmup_pair_indices",
    ) != (0, 5_000):
        raise ValueError("Fit/warm-up phase changed.")
    if _require_interval(
        phases.get("threshold_pair_indices"),
        context="threshold_pair_indices",
    ) != (5_000, 10_000):
        raise ValueError("Threshold phase changed.")
    if not phases.get("threshold_and_held_clone_same_post_warmup_checkpoint"):
        raise ValueError("Threshold and held must clone one checkpoint.")
    if not phases.get("threshold_state_never_carried_into_held"):
        raise ValueError("Threshold state must not enter held replay.")

    parser = config["detector_parser"]
    if not isinstance(parser, Mapping):
        raise TypeError("detector_parser must be an object.")
    if (
        parser.get("bit_order_within_byte") != "little"
        or parser.get("bytes_per_shot") != 153
        or parser.get("padding_bits_per_shot") != 0
        or parser.get("expected_file_bytes") != 76_500_000
        or parser.get("logical_flatten_order") != ["shot", "role", "canonical_check"]
        or parser.get("naive_reshape_forbidden") is not True
    ):
        raise ValueError("Locked detector binary/parser semantics changed.")

    pairing = config["pairing"]
    if not isinstance(pairing, Mapping):
        raise TypeError("pairing must be an object.")
    if (
        pairing.get("adaptive_state_sharing") != "separate_state_for_each_of_51_roles"
        or pairing.get("updates_per_held_episode") != 1_020_000
        or pairing.get("formal_accumulator_unit") != "complete_paired_shot"
        or pairing.get("formal_accumulator_updates_per_held_episode") != 20_000
        or pairing.get("formal_role_experts") != 51
        or pairing.get("formal_role_prior") != "uniform_1_over_51"
        or pairing.get("within_shot_factor_compounding") is not False
    ):
        raise ValueError("Locked role-state or formal experimental unit changed.")

    methods = config["methods"]
    if not isinstance(methods, Mapping):
        raise TypeError("methods must be an object.")
    require_exact_keys(methods, GOOGLE_METHOD_KEYS, context="methods")
    composite = methods["space_composite"]
    if (
        not isinstance(composite, Mapping)
        or composite.get("fixed_branch_prior")
        != {"space_sparse": 0.5, "space_spectral": 0.5}
        or composite.get("branch_selection_from_held") is not False
    ):
        raise ValueError("Fixed S-PACE composite changed.")

    accumulators = config["accumulators"]
    if not isinstance(accumulators, Mapping):
        raise TypeError("accumulators must be an object.")
    primary_accumulator = accumulators.get("primary")
    secondary_accumulator = accumulators.get("secondary")
    if (
        not isinstance(primary_accumulator, Mapping)
        or primary_accumulator.get("horizon_unit") != "complete_paired_shot"
        or primary_accumulator.get("horizon_shots") != 20_000
        or primary_accumulator.get("randomization_horizon_shots") != 5_000
        or primary_accumulator.get("role_component_experts") is not True
        or primary_accumulator.get("role_prior") != "uniform_1_over_51"
        or not isinstance(secondary_accumulator, Mapping)
        or secondary_accumulator.get("gamma_shots") != 1_000_000
    ):
        raise ValueError("Formal complete-shot accumulator semantics changed.")

    operating = config["empirical_operating_points"]
    if not isinstance(operating, Mapping):
        raise TypeError("empirical_operating_points must be an object.")
    primary = operating.get("primary")
    if (
        not isinstance(primary, Mapping)
        or primary.get("alerts_per_opportunity") != 1e-5
        or primary.get("validation_opportunities") != 255_000
        or primary.get("maximum_validation_alerts") != 2
        or operating.get("notification_rule") != "strict_score_greater_than_threshold"
    ):
        raise ValueError("Primary empirical alarm rule changed.")

    decision = config["decision"]
    if not isinstance(decision, Mapping):
        raise TypeError("decision must be an object.")
    if (
        decision.get("predesignated_comparators") != ["dfr", "online_logistic"]
        or decision.get("pnnl_auxiliary_required") is not True
        or decision.get("untouched_google_alternative_for_retention") is not False
    ):
        raise ValueError("Advantage decision changed.")

    randomization = config["randomization_audit"]
    if (
        not isinstance(randomization, Mapping)
        or randomization.get("replicates") != 256
        or randomization.get("pair_indices") != [5_000, 10_000]
        or randomization.get("rng") != "numpy.random.Generator(PCG64)"
        or randomization.get("horizon_unit") != "complete_paired_shot"
        or randomization.get("horizon_shots") != 5_000
    ):
        raise ValueError("Randomization audit changed.")

    numeric = config["numeric_policy"]
    if not isinstance(numeric, Mapping):
        raise TypeError("numeric_policy must be an object.")
    required_threads = {
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }
    if (
        numeric.get("floating_dtype") != "float64"
        or numeric.get("score_bound_tolerance") != 1e-12
        or numeric.get("thread_environment") != required_threads
    ):
        raise ValueError("Numeric determinism policy changed.")


def load_google_lock(path: str | Path) -> dict[str, Any]:
    """Load and validate the exact Google configuration schema."""

    config = load_strict_json(path)
    validate_google_lock(config)
    return config


def assert_no_outcome_paths(paths: Sequence[str | Path]) -> None:
    """Enforce the detector-only command's outcome embargo."""

    forbidden: list[str] = []
    for raw in paths:
        path = Path(raw)
        if path.name in OUTCOME_FILENAMES or path.suffix == ".01":
            forbidden.append(str(path))
    if forbidden:
        raise PermissionError(
            "Detector-only replay cannot receive outcome paths: "
            + ", ".join(sorted(forbidden))
        )


def require_thread_environment(expected: Mapping[str, str]) -> None:
    """Fail rather than silently overwrite a frozen numeric environment."""

    mismatches = {
        key: {"expected": expected_value, "observed": os.environ.get(key)}
        for key, expected_value in expected.items()
        if os.environ.get(key) != expected_value
    }
    if mismatches:
        raise RuntimeError(f"Thread environment mismatch: {mismatches}.")


def environment_fingerprint() -> dict[str, Any]:
    """Return exact interpreter, platform, dependency, and thread metadata."""

    packages: dict[str, str] = {}
    for package in PINNED_PACKAGES:
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = "NOT_INSTALLED"
    thread_keys = (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    )
    return {
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": str(Path(sys.executable).resolve()),
            "compiler": platform.python_compiler(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "packages": packages,
        "numpy_build": np.__config__.show(mode="dicts"),
        "thread_environment": {key: os.environ.get(key) for key in thread_keys},
    }
