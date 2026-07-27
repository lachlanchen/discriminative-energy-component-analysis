from __future__ import annotations

import importlib.util
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from aoc.surface_code import PeriodicSurfaceSyndromeModel

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_syndrome_drift.py"
SPEC = importlib.util.spec_from_file_location("run5_syndrome_drift", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
aggregate_metrics = MODULE.aggregate_metrics
metric_rows = MODULE.metric_rows


def test_locked_config_is_cycle_fair_and_seed_disjoint() -> None:
    config = MODULE.load_config(
        Path(__file__).resolve().parents[1] / "configs" / "paper.json"
    )
    assert config["locked"] is True
    assert config["calibration_cycles"] == 4096
    assert config["change_time_cycles"] == 256
    assert config["post_change_horizon_cycles"] == 1024
    assert config["null_horizon_cycles"] == 5000
    assert config["validation_streams"] == 8
    assert config["ridge_validation_horizon_cycles"] == 512
    assert (
        config["validation_streams"] * config["ridge_validation_horizon_cycles"]
        == config["calibration_cycles"]
        == config["validation_trained_baseline"]["total_physical_cycles_per_class"]
    )
    assert (
        config["validation_trained_baseline"]["training_streams_per_class"]
        == config["validation_streams"]
    )
    assert (
        config["validation_trained_baseline"]["cycles_per_stream"]
        == config["ridge_validation_horizon_cycles"]
    )
    assert MODULE.cycles_to_updates(4096, "spatial") == 4096
    assert MODULE.cycles_to_updates(4096, "temporal") == 2048
    assert config["detector_threshold_updates"] == {
        "spatial": 1000,
        "temporal": 500,
    }
    intervals = MODULE.sampling_seed_intervals(config)
    for left_index, (_, left_start, left_end) in enumerate(intervals):
        for _, right_start, right_end in intervals[left_index + 1 :]:
            assert left_end < right_start or right_end < left_start


def test_locked_config_rejects_odd_temporal_cycle_budget(tmp_path) -> None:
    source = Path(__file__).resolve().parents[1] / "configs" / "paper.json"
    config = json.loads(source.read_text(encoding="utf-8"))
    config["null_horizon_cycles"] = 4999
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="must be even"):
        MODULE.load_config(path)


def test_publication_config_must_be_locked(tmp_path) -> None:
    source = Path(__file__).resolve().parents[1] / "configs" / "paper.json"
    config = json.loads(source.read_text(encoding="utf-8"))
    config["locked"] = False
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="must be locked"):
        MODULE.load_config(path)


def test_auxiliary_stream_structure_and_cycle_budgets_must_match(
    tmp_path,
) -> None:
    source = Path(__file__).resolve().parents[1] / "configs" / "paper.json"
    config = json.loads(source.read_text(encoding="utf-8"))
    config["validation_trained_baseline"]["cycles_per_stream"] = 256
    config["validation_trained_baseline"]["training_streams_per_class"] = 16
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="same number of independent streams"):
        MODULE.load_config(path)


def test_validation_logistic_direction_is_a_bounded_simplex_effect() -> None:
    null = np.asarray(
        [
            [0.8, 0.1, 0.1],
            [0.7, 0.2, 0.1],
            [0.75, 0.15, 0.1],
        ]
    )
    alternative = np.asarray(
        [
            [0.1, 0.2, 0.7],
            [0.1, 0.1, 0.8],
            [0.15, 0.15, 0.7],
        ]
    )
    effect, coefficient, accuracy = MODULE.fit_linear_logistic_effect(
        null,
        alternative,
        c_value=1.0,
        max_iter=1000,
    )
    assert coefficient.shape == (3,)
    assert np.all((effect >= 0.0) & (effect <= 1.0))
    assert np.isclose(effect.min(), 0.0)
    assert np.isclose(effect.max(), 1.0)
    assert effect[2] > effect[0]
    assert accuracy == 1.0


def test_prechange_alarm_is_excluded_from_conditional_delay() -> None:
    rows = metric_rows(
        {"method": 7},
        family="spatial",
        effect=0.55,
        replicate=0,
        seed=1,
        stream_type="changed",
        evaluation="surveillance",
        horizon=20,
        change_time=8,
        time_unit_cycles=1,
        threshold=10.0,
    )
    row = rows[0]
    assert row["false_alarm"]
    assert not row["conditional_eligible"]
    assert not row["post_change_detected"]
    assert not row["censored"]
    assert np.isnan(row["restricted_delay_cycles"])


def test_restart_metric_removes_blind_sr_age_artifact() -> None:
    data = []
    data.extend(
        metric_rows(
            {"blind": 10},
            family="spatial",
            effect=0.55,
            replicate=0,
            seed=1,
            stream_type="no_change",
            evaluation="null_arl",
            horizon=30,
            change_time=None,
            time_unit_cycles=1,
            threshold=10.0,
        )
    )
    data.extend(
        metric_rows(
            {"blind": 10},
            family="spatial",
            effect=0.55,
            replicate=0,
            seed=2,
            stream_type="changed",
            evaluation="surveillance",
            horizon=20,
            change_time=4,
            time_unit_cycles=1,
            threshold=10.0,
        )
    )
    data.extend(
        metric_rows(
            {"blind": 10},
            family="spatial",
            effect=0.55,
            replicate=0,
            seed=2,
            stream_type="changed",
            evaluation="restart_at_change",
            horizon=20,
            change_time=0,
            time_unit_cycles=1,
            threshold=10.0,
        )
    )
    aggregate = aggregate_metrics(pd.DataFrame(data)).iloc[0]
    assert aggregate["surveillance_conditional_mean_delay_cycles"] == 6.0
    assert aggregate["mean_restart_delay_cycles"] == 10.0
    assert aggregate["no_change_rmst_at_restart_horizon_cycles"] == 10.0
    assert aggregate["restart_delay_reduction_vs_no_change_cycles"] == 0.0
    assert aggregate["restart_delay_ratio_to_no_change"] == 1.0


def test_no_change_label_survives_default_pandas_csv_parsing(tmp_path) -> None:
    rows = metric_rows(
        {"method": None},
        family="spatial",
        effect=0.55,
        replicate=0,
        seed=1,
        stream_type="no_change",
        evaluation="null_arl",
        horizon=30,
        change_time=None,
        time_unit_cycles=1,
        threshold=10.0,
    )
    path = tmp_path / "metrics.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    restored = pd.read_csv(path)
    assert restored.loc[0, "stream_type"] == "no_change"
    assert pd.notna(restored.loc[0, "stream_type"])


def test_full_hmm_block_increments_equal_exact_path_likelihood() -> None:
    model = PeriodicSurfaceSyndromeModel(
        size=3,
        event_probability=0.4,
        readout_error=0.08,
        allow_small_for_test=True,
    )
    q = 0.37
    kappa = 0.62
    observed = model.sample_temporal(
        4,
        q=q,
        kappa=kappa,
        seed=184,
    )[0]
    increments = model.temporal_hmm_log_likelihood_ratio_increments(
        observed,
        q=q,
        kappa=kappa,
    )
    assert increments.shape == (2,)

    emissions = np.stack(
        [
            model.conditional_length_emission_likelihoods(observed, 1),
            model.conditional_length_emission_likelihoods(observed, 2),
        ],
        axis=1,
    )
    stationary = np.asarray([1.0 - q, q])
    transition = model.length_transition_matrix(q, kappa)
    alternative = 0.0
    for states in itertools.product(range(2), repeat=len(observed)):
        probability = stationary[states[0]] * emissions[0, states[0]]
        for time_index in range(1, len(observed)):
            probability *= (
                transition[states[time_index - 1], states[time_index]]
                * emissions[time_index, states[time_index]]
            )
        alternative += probability
    null_log = model.emission_log_likelihoods(observed, q).sum()
    assert np.isclose(increments.sum(), np.log(alternative) - null_log)

    null_increments = model.temporal_hmm_log_likelihood_ratio_increments(
        observed,
        q=q,
        kappa=0.0,
    )
    assert np.allclose(null_increments, 0.0, atol=1e-12)


def test_two_cycle_hmm_increment_matches_pair_restricted_likelihood() -> None:
    model = PeriodicSurfaceSyndromeModel(
        size=3,
        event_probability=0.4,
        readout_error=0.08,
        allow_small_for_test=True,
    )
    q = 0.37
    kappa = 0.62
    observed = model.sample_temporal(
        2,
        q=q,
        kappa=kappa,
        seed=185,
    )[0]
    increment = model.temporal_hmm_log_likelihood_ratio_increments(
        observed,
        q=q,
        kappa=kappa,
    )[0]
    pair_log = model.nonoverlapping_pair_log_likelihood(
        observed[0],
        observed[1],
        q=q,
        kappa=kappa,
    )
    null_log = model.nonoverlapping_pair_log_likelihood(
        observed[0],
        observed[1],
        q=q,
        kappa=0.0,
    )
    assert np.isclose(increment, pair_log - null_log)


def test_bounded_delay_hoeffding_formula() -> None:
    differences = np.asarray([-80.0] * 64)
    p_value, upper = MODULE.bounded_delay_hoeffding_inference(
        differences,
        horizon_cycles=100.0,
        one_sided_alpha=0.025,
    )
    expected_p = np.exp(-64 * 80.0**2 / (2.0 * 100.0**2))
    expected_upper = -80.0 + 100.0 * np.sqrt(2.0 * np.log(1.0 / 0.025) / 64)
    assert np.isclose(p_value, expected_p)
    assert np.isclose(upper, expected_upper)

    null_direction_p, _ = MODULE.bounded_delay_hoeffding_inference(
        np.asarray([1.0, -1.0, 2.0, -2.0]),
        horizon_cycles=2.0,
        one_sided_alpha=0.025,
    )
    assert null_direction_p == 1.0


def test_primary_comparison_requires_hoeffding_bound_and_holm_test() -> None:
    rows: list[dict[str, object]] = []
    methods = {
        MODULE.PROPOSED_METHOD: 10,
        MODULE.VALIDATION_BASELINE_METHOD: 90,
    }
    for family, effect, unit in (
        ("spatial", 0.55, 1),
        ("temporal", 0.75, 2),
    ):
        for replicate in range(64):
            rows.extend(
                metric_rows(
                    methods,
                    family=family,
                    effect=effect,
                    replicate=replicate,
                    seed=700000 + 1000 * unit + replicate,
                    stream_type="changed",
                    evaluation="restart_at_change",
                    horizon=100,
                    change_time=0,
                    time_unit_cycles=unit,
                    threshold=1000.0 / unit,
                )
            )
    audit = MODULE.paired_primary_comparator_audit(
        pd.DataFrame(rows),
        hypotheses={
            "proposed_method": MODULE.PROPOSED_METHOD,
            "baseline_method": MODULE.VALIDATION_BASELINE_METHOD,
            "spatial_effect": 0.55,
            "temporal_effect": 0.75,
            "familywise_alpha": 0.05,
        },
        bootstrap_resamples=1000,
        bootstrap_seed=123,
        arl_requirement_met=True,
    )
    assert len(audit) == 2
    assert (audit["descriptive_bootstrap_ci95_high_cycles"] < 0.0).all()
    assert (audit["simultaneous_upper_cycles"] < 0.0).all()
    assert (audit["holm_adjusted_one_sided_p"] < 0.05).all()
    assert audit["comparison_supported_for_hypothesis"].all()
    assert audit["overall_two_hypothesis_comparison_supported"].all()

    failed_arl = MODULE.paired_primary_comparator_audit(
        pd.DataFrame(rows),
        hypotheses={
            "proposed_method": MODULE.PROPOSED_METHOD,
            "baseline_method": MODULE.VALIDATION_BASELINE_METHOD,
            "spatial_effect": 0.55,
            "temporal_effect": 0.75,
            "familywise_alpha": 0.05,
        },
        bootstrap_resamples=200,
        bootstrap_seed=123,
        arl_requirement_met=False,
    )
    assert not failed_arl["comparison_supported_for_hypothesis"].any()
    assert not failed_arl["overall_two_hypothesis_comparison_supported"].any()
