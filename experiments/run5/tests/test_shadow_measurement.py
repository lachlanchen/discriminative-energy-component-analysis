from __future__ import annotations

import importlib.util
import sys
from itertools import product
from pathlib import Path

import numpy as np
from aoc.surface_code import PeriodicSurfaceSyndromeModel

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_shadow_measurement_audit.py"
)
SPEC = importlib.util.spec_from_file_location("run5_shadow_measurement_audit", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = AUDIT
SPEC.loader.exec_module(AUDIT)


def test_local_pauli_zz_inverse_channel_factor_is_unbiased() -> None:
    bases = np.asarray(list(product((0, 1, 2), repeat=2)), dtype=np.int8)
    # The Z eigenvalues are +1 and -1. Outcomes in X/Y can be arbitrary here:
    # they are multiplied by zero unless both selected bases are Z.
    outcomes = np.ones_like(bases)
    outcomes[bases[:, 0] == 2, 0] = 1
    outcomes[bases[:, 1] == 2, 1] = -1
    estimate = AUDIT.shadow_zz_estimators(
        bases,
        outcomes,
        np.asarray([[0, 1]]),
    )
    assert AUDIT.LOCAL_INVERSE_FACTOR == 3
    assert AUDIT.ZZ_INVERSE_FACTOR == 9
    assert np.isclose(estimate.mean(), -1.0)
    assert np.count_nonzero(estimate) == 1


def test_exact_zz_expectation_matches_exhaustive_diagonal_distribution() -> None:
    model = PeriodicSurfaceSyndromeModel(
        size=3,
        event_probability=0.57,
        readout_error=0.09,
        allow_small_for_test=True,
    )
    q = 0.38
    pairs = np.asarray([[0, 1], [0, 4], [2, 7]], dtype=np.int64)
    observations = np.asarray(
        list(product((0, 1), repeat=model.num_detectors)),
        dtype=np.uint8,
    )
    probabilities = model.emission_likelihoods(observations, q)
    z = 1.0 - 2.0 * observations
    exhaustive = probabilities @ (z[:, pairs[:, 0]] * z[:, pairs[:, 1]])
    np.testing.assert_allclose(
        AUDIT.exact_zz_expectations(model, q, pairs),
        exhaustive,
        atol=3e-15,
    )


def test_measurement_snapshot_sampling_is_seeded_and_uses_all_local_bases() -> None:
    model = PeriodicSurfaceSyndromeModel(
        size=5,
        event_probability=0.65,
        readout_error=0.03,
    )
    first = AUDIT.simulate_measurement_copies(
        model,
        0.35,
        256,
        rng=np.random.default_rng(71003),
    )
    second = AUDIT.simulate_measurement_copies(
        model,
        0.35,
        256,
        rng=np.random.default_rng(71003),
    )
    np.testing.assert_array_equal(first.syndromes, second.syndromes)
    np.testing.assert_array_equal(first.shadow_bases, second.shadow_bases)
    np.testing.assert_array_equal(first.shadow_outcomes, second.shadow_outcomes)
    assert set(np.unique(first.shadow_bases)) == {0, 1, 2}
    z_locations = first.shadow_bases == 2
    np.testing.assert_array_equal(
        first.shadow_outcomes[z_locations],
        first.native_z[z_locations],
    )


def test_declared_pair_bank_is_complete_and_translation_regular() -> None:
    bank = AUDIT.declared_pair_bank(
        5,
        [[0, 1], [1, 0], [0, 2], [2, 0]],
    )
    assert bank.indices.shape == (100, 2)
    assert len(bank.metadata) == 100
    assert np.all(bank.indices[:, 0] != bank.indices[:, 1])
    counts = bank.metadata.groupby("displacement_index").size()
    np.testing.assert_array_equal(counts.to_numpy(), np.full(4, 25))
