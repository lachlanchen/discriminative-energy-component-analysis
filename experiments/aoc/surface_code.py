"""Exact controlled syndrome models for periodic error-detection experiments.

The primary :class:`PeriodicSurfaceSyndromeModel` is an ``L x L`` torus whose
nonempty faults are uniformly translated horizontal or vertical open chains
of length one or two.  Both chain lengths have two syndrome endpoints.
Consequently their mixture is invisible to the full detection-count
distribution and every single-detector marginal, even after independent
binary-symmetric readout noise, while remaining visible to correlations and
translation-invariant Fourier power.

The secondary one-dimensional :class:`PeriodicSyndromeModel` is deliberately
small enough to admit exact likelihoods while retaining two structures that
matter in quantum error correction:

* an open error chain has a syndrome only at its two endpoints; and
* adding the noncontractible closed loop changes the logical class without
  changing any syndrome bit.

For a ring of ``size`` detectors, an emission is empty with probability
``1-event_probability``.  Otherwise, a uniformly translated chain of
``chain_length`` consecutive edges is emitted.  Every nonempty emission
therefore has exactly two detection events and every detector has marginal
probability ``2 * event_probability / size``, independently of chain length.
Changing the chain length is consequently invisible to detector marginals and
to the *entire* detection-fraction/count distribution, but visible to spatial
correlations and translation-invariant Fourier power.

The temporal sampler uses a refresh-or-repeat latent Markov chain.  Its
stationary one-cycle distribution is identical for every persistence value,
so a persistence change is invisible to every single-cycle statistic while
remaining detectable through lagged correlations.

This is a controlled periodic phenomenological model, not a replacement for a
circuit-level surface-code simulation.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb

import numpy as np
from numpy.typing import ArrayLike, NDArray

BinaryArray = NDArray[np.uint8]
FloatArray = NDArray[np.float64]


def _log_probability(probability: float) -> float:
    return float("-inf") if probability == 0.0 else float(np.log(probability))


def _as_binary_last_axis(values: ArrayLike, size: int, *, name: str) -> BinaryArray:
    array = np.asarray(values)
    if array.ndim == 0 or array.shape[-1] != size:
        raise ValueError(f"{name} must have final dimension {size}.")
    if not np.all((array == 0) | (array == 1)):
        raise ValueError(f"{name} must contain only binary values.")
    return array.astype(np.uint8, copy=False)


def _resolve_rng(
    *,
    seed: int | None,
    rng: np.random.Generator | None,
) -> np.random.Generator:
    if seed is not None and rng is not None:
        raise ValueError("Supply either seed or rng, not both.")
    return np.random.default_rng(seed) if rng is None else rng


@dataclass(frozen=True)
class PeriodicSurfaceSyndromeModel:
    """Exact translated-chain syndrome mixture on a square periodic lattice.

    With probability ``1 - event_probability``, the latent physical syndrome
    is empty.  Otherwise, a start vertex and one of the positive coordinate
    directions are chosen uniformly.  The two endpoints of a length-one or
    length-two open chain are marked; ``q`` is the probability of length two.
    Every detector bit is then independently flipped with probability
    ``readout_error``.

    The temporal model makes the length state a stationary refresh-or-repeat
    Markov chain.  Given ``q`` and persistence ``kappa``, the next length
    repeats with probability ``kappa`` and is independently refreshed from
    ``[(1-q), q]`` otherwise.  Event occurrence, translation, direction, and
    readout noise remain conditionally independent between nonoverlapping
    cycles.
    """

    size: int
    event_probability: float
    readout_error: float
    allow_small_for_test: bool = False

    def __post_init__(self) -> None:
        minimum = 3 if self.allow_small_for_test else 5
        if self.size < minimum or self.size % 2 == 0:
            qualifier = "odd and at least 3" if self.allow_small_for_test else (
                "odd and at least 5"
            )
            raise ValueError(f"size must be {qualifier}.")
        if not 0.0 <= self.event_probability <= 1.0:
            raise ValueError("event_probability must lie in [0, 1].")
        if not 0.0 <= self.readout_error <= 1.0:
            raise ValueError("readout_error must lie in [0, 1].")

    @property
    def num_detectors(self) -> int:
        return self.size**2

    @staticmethod
    def _validate_mixture(q: float) -> float:
        value = float(q)
        if not 0.0 <= value <= 1.0:
            raise ValueError("q must lie in [0, 1].")
        return value

    @staticmethod
    def _validate_persistence(kappa: float) -> float:
        value = float(kappa)
        if not 0.0 <= value <= 1.0:
            raise ValueError("kappa must lie in [0, 1].")
        return value

    def _coerce_observations(
        self,
        observed: ArrayLike,
        *,
        name: str,
    ) -> tuple[BinaryArray, tuple[int, ...]]:
        array = np.asarray(observed)
        if array.ndim >= 2 and array.shape[-2:] == (self.size, self.size):
            leading_shape = array.shape[:-2]
            flat = array.reshape(-1, self.num_detectors)
        elif array.ndim >= 1 and array.shape[-1] == self.num_detectors:
            leading_shape = array.shape[:-1]
            flat = array.reshape(-1, self.num_detectors)
        else:
            raise ValueError(
                f"{name} must end in ({self.size}, {self.size}) or "
                f"({self.num_detectors},)."
            )
        if not np.all((flat == 0) | (flat == 1)):
            raise ValueError(f"{name} must contain only binary values.")
        return flat.astype(np.uint8, copy=False), leading_shape

    def template_table(self, length: int) -> BinaryArray:
        """Return all uniformly translated ``+x`` and ``+y`` endpoint templates."""

        endpoints = self.template_endpoints(length)
        templates = np.zeros(
            (len(endpoints), self.num_detectors),
            dtype=np.uint8,
        )
        template_indices = np.arange(len(endpoints))[:, None]
        templates[template_indices, endpoints] = 1
        return templates

    def template_endpoints(self, length: int) -> NDArray[np.int64]:
        """Return endpoint indices for every translated and oriented chain."""

        if length not in (1, 2):
            raise ValueError("length must be 1 or 2.")
        endpoints = np.empty((2 * self.num_detectors, 2), dtype=np.int64)
        for row in range(self.size):
            for column in range(self.size):
                start = row * self.size + column
                horizontal = row * self.size + (column + length) % self.size
                vertical = ((row + length) % self.size) * self.size + column
                translation = row * self.size + column
                endpoints[translation] = [start, horizontal]
                endpoints[self.num_detectors + translation] = [start, vertical]
        return endpoints

    def length_transition_matrix(self, q: float, kappa: float) -> FloatArray:
        """Return the stationary refresh-or-repeat transition on lengths 1 and 2."""

        mixture = self._validate_mixture(q)
        persistence = self._validate_persistence(kappa)
        stationary = np.asarray([1.0 - mixture, mixture])
        refresh = np.broadcast_to(stationary, (2, 2)).copy()
        return persistence * np.eye(2) + (1.0 - persistence) * refresh

    def _bsc_log_probability_from_distance(
        self,
        distance: NDArray[np.integer],
    ) -> FloatArray:
        mismatch = np.asarray(distance)
        if self.readout_error == 0.0:
            return np.where(mismatch == 0, 0.0, float("-inf"))
        if self.readout_error == 1.0:
            return np.where(
                mismatch == self.num_detectors,
                0.0,
                float("-inf"),
            )
        return (
            mismatch * np.log(self.readout_error)
            + (self.num_detectors - mismatch) * np.log1p(-self.readout_error)
        )

    def _zero_syndrome_log_likelihoods(self, observed: BinaryArray) -> FloatArray:
        return self._bsc_log_probability_from_distance(observed.sum(axis=1))

    def _template_log_likelihoods(
        self,
        observed: BinaryArray,
        length: int,
    ) -> FloatArray:
        endpoints = self.template_endpoints(length)
        observed_count = observed.sum(axis=1, dtype=np.int64)
        endpoint_count = observed[:, endpoints].sum(axis=2, dtype=np.int64)
        distance = observed_count[:, None] + 2 - 2 * endpoint_count
        component_logs = self._bsc_log_probability_from_distance(distance)
        return np.logaddexp.reduce(component_logs, axis=1) - np.log(len(endpoints))

    def conditional_length_emission_log_likelihoods(
        self,
        observed: ArrayLike,
        length: int,
    ) -> FloatArray:
        """Exact BSC emission log likelihood conditional on the length state."""

        flat, leading_shape = self._coerce_observations(observed, name="observed")
        no_event = (
            _log_probability(1.0 - self.event_probability)
            + self._zero_syndrome_log_likelihoods(flat)
        )
        event = (
            _log_probability(self.event_probability)
            + self._template_log_likelihoods(flat, length)
        )
        values = np.logaddexp(no_event, event)
        return values.reshape(leading_shape)

    def conditional_length_emission_likelihoods(
        self,
        observed: ArrayLike,
        length: int,
    ) -> FloatArray:
        """Exact BSC likelihood conditional on a length state."""

        return np.exp(
            self.conditional_length_emission_log_likelihoods(observed, length)
        )

    def conditional_length_emission_log_likelihood(
        self,
        observed: ArrayLike,
        length: int,
    ) -> float:
        """Exact conditional-length log likelihood of one observation."""

        flat, leading_shape = self._coerce_observations(observed, name="observed")
        if leading_shape:
            raise ValueError(
                "conditional_length_emission_log_likelihood expects one observation."
            )
        return float(
            self.conditional_length_emission_log_likelihoods(flat[0], length)
        )

    def emission_log_likelihoods(self, observed: ArrayLike, q: float) -> FloatArray:
        """Exact batch log likelihood under the stationary length mixture ``q``."""

        mixture = self._validate_mixture(q)
        length_one = (
            _log_probability(1.0 - mixture)
            + self.conditional_length_emission_log_likelihoods(observed, 1)
        )
        length_two = (
            _log_probability(mixture)
            + self.conditional_length_emission_log_likelihoods(observed, 2)
        )
        return np.logaddexp(length_one, length_two)

    def emission_likelihoods(self, observed: ArrayLike, q: float) -> FloatArray:
        """Exact batch BSC likelihood under mixture ``q``."""

        return np.exp(self.emission_log_likelihoods(observed, q))

    def emission_log_likelihood(self, observed: ArrayLike, q: float) -> float:
        """Exact log likelihood of one flattened or square-grid observation."""

        flat, leading_shape = self._coerce_observations(observed, name="observed")
        if leading_shape:
            raise ValueError("emission_log_likelihood expects one observation.")
        return float(self.emission_log_likelihoods(flat[0], q))

    def emission_likelihood(self, observed: ArrayLike, q: float) -> float:
        """Exact likelihood of one flattened or square-grid observation."""

        return float(np.exp(self.emission_log_likelihood(observed, q)))

    def nonoverlapping_pair_log_likelihoods(
        self,
        first: ArrayLike,
        second: ArrayLike,
        *,
        q: float,
        kappa: float,
    ) -> FloatArray:
        """Exact likelihood for conditionally independent adjacent cycle emissions."""

        mixture = self._validate_mixture(q)
        transition = self.length_transition_matrix(mixture, kappa)
        stationary = np.asarray([1.0 - mixture, mixture])
        first_logs = np.stack(
            [
                self.conditional_length_emission_log_likelihoods(first, 1),
                self.conditional_length_emission_log_likelihoods(first, 2),
            ],
            axis=-1,
        )
        second_logs = np.stack(
            [
                self.conditional_length_emission_log_likelihoods(second, 1),
                self.conditional_length_emission_log_likelihoods(second, 2),
            ],
            axis=-1,
        )
        if first_logs.shape != second_logs.shape:
            raise ValueError("first and second must contain equally shaped batches.")
        component_logs = []
        for previous in range(2):
            for current in range(2):
                component_logs.append(
                    _log_probability(float(stationary[previous]))
                    + _log_probability(float(transition[previous, current]))
                    + first_logs[..., previous]
                    + second_logs[..., current]
                )
        return np.logaddexp.reduce(np.stack(component_logs), axis=0)

    def nonoverlapping_pair_likelihoods(
        self,
        first: ArrayLike,
        second: ArrayLike,
        *,
        q: float,
        kappa: float,
    ) -> FloatArray:
        """Exact adjacent-pair likelihood for nonoverlapping readout cycles."""

        return np.exp(
            self.nonoverlapping_pair_log_likelihoods(
                first,
                second,
                q=q,
                kappa=kappa,
            )
        )

    def nonoverlapping_pair_log_likelihood(
        self,
        first: ArrayLike,
        second: ArrayLike,
        *,
        q: float,
        kappa: float,
    ) -> float:
        """Exact log likelihood of one adjacent nonoverlapping observation pair."""

        first_flat, first_shape = self._coerce_observations(first, name="first")
        second_flat, second_shape = self._coerce_observations(second, name="second")
        if first_shape or second_shape:
            raise ValueError(
                "nonoverlapping_pair_log_likelihood expects two observations."
            )
        return float(
            self.nonoverlapping_pair_log_likelihoods(
                first_flat[0],
                second_flat[0],
                q=q,
                kappa=kappa,
            )
        )

    def nonoverlapping_pair_likelihood(
        self,
        first: ArrayLike,
        second: ArrayLike,
        *,
        q: float,
        kappa: float,
    ) -> float:
        """Exact likelihood of one adjacent nonoverlapping observation pair."""

        return float(
            np.exp(
                self.nonoverlapping_pair_log_likelihood(
                    first,
                    second,
                    q=q,
                    kappa=kappa,
                )
            )
        )

    def temporal_hmm_log_likelihood_ratio_increments(
        self,
        observed: ArrayLike,
        *,
        q: float,
        kappa: float,
    ) -> FloatArray:
        """Return exact two-cycle increments of the full temporal HMM LR.

        The alternative is the stationary two-state length HMM with
        persistence ``kappa``.  The null has the same one-cycle mixture
        ``q`` and zero persistence, so its observations are iid.  Entry
        ``j`` is the conditional log likelihood ratio of cycles
        ``2*j`` and ``2*j + 1`` given every preceding cycle.  Consequently,
        the cumulative sum through entry ``j`` is the exact full-path HMM
        log likelihood ratio through cycle ``2*j + 1``.
        """

        mixture = self._validate_mixture(q)
        transition = self.length_transition_matrix(mixture, kappa)
        flat, leading_shape = self._coerce_observations(
            observed,
            name="observed",
        )
        if len(leading_shape) != 1:
            raise ValueError("observed must be one temporal sequence.")
        if len(flat) % 2:
            raise ValueError("Full-HMM block increments require an even cycle count.")

        emission_logs = np.stack(
            [
                self.conditional_length_emission_log_likelihoods(flat, 1),
                self.conditional_length_emission_log_likelihoods(flat, 2),
            ],
            axis=1,
        )
        null_logs = self.emission_log_likelihoods(flat, mixture)
        stationary = np.asarray([1.0 - mixture, mixture], dtype=np.float64)
        log_filter = np.asarray(
            [_log_probability(float(value)) for value in stationary],
            dtype=np.float64,
        )
        log_transition = np.asarray(
            [
                [_log_probability(float(value)) for value in row]
                for row in transition
            ],
            dtype=np.float64,
        )

        block_increments = np.empty(len(flat) // 2, dtype=np.float64)
        alternative_block_log = 0.0
        null_block_log = 0.0
        for time_index, emission in enumerate(emission_logs):
            if time_index:
                predicted = np.asarray(
                    [
                        np.logaddexp.reduce(
                            log_filter + log_transition[:, current]
                        )
                        for current in range(2)
                    ]
                )
            else:
                predicted = log_filter
            joint = predicted + emission
            cycle_log_likelihood = float(np.logaddexp.reduce(joint))
            log_filter = joint - cycle_log_likelihood
            alternative_block_log += cycle_log_likelihood
            null_block_log += float(null_logs[time_index])
            if time_index % 2:
                block_increments[time_index // 2] = (
                    alternative_block_log - null_block_log
                )
                alternative_block_log = 0.0
                null_block_log = 0.0
        return block_increments

    @property
    def count_pmf(self) -> FloatArray:
        """Analytic observed syndrome-count PMF after binary-symmetric readout."""

        detector_count = self.num_detectors

        def conditional_count(true_weight: int) -> FloatArray:
            values = np.zeros(detector_count + 1, dtype=np.float64)
            for retained in range(true_weight + 1):
                retained_probability = (
                    comb(true_weight, retained)
                    * (1.0 - self.readout_error) ** retained
                    * self.readout_error ** (true_weight - retained)
                )
                for false_positive in range(detector_count - true_weight + 1):
                    false_positive_probability = (
                        comb(detector_count - true_weight, false_positive)
                        * self.readout_error**false_positive
                        * (1.0 - self.readout_error)
                        ** (detector_count - true_weight - false_positive)
                    )
                    values[retained + false_positive] += (
                        retained_probability * false_positive_probability
                    )
            return values

        return (
            (1.0 - self.event_probability) * conditional_count(0)
            + self.event_probability * conditional_count(2)
        )

    @property
    def detector_marginal(self) -> FloatArray:
        """Exact observed event probability at all ``size**2`` detectors."""

        true_marginal = 2.0 * self.event_probability / self.num_detectors
        observed_marginal = self.readout_error + (
            1.0 - 2.0 * self.readout_error
        ) * true_marginal
        return np.full(self.num_detectors, observed_marginal)

    def translation_pair_features(self, observed: ArrayLike) -> FloatArray:
        """Return the sufficient translation statistics ``(z_bar, g1, g2)``.

        For ``z = 1 - 2 y``,

        ``g_l = (2m)^-1 sum_{v,o in {x,y}} z_v z_{v+l e_o}``.

        The exact likelihood family indexed by ``q`` is a function of these
        three values. This sufficiency is specific to the translated
        two-endpoint/BSC model.
        """

        flat, leading_shape = self._coerce_observations(observed, name="observed")
        signed = 1.0 - 2.0 * flat.astype(np.float64)
        grids = signed.reshape(-1, self.size, self.size)
        features = [grids.mean(axis=(1, 2))]
        for length in (1, 2):
            horizontal = grids * np.roll(grids, -length, axis=2)
            vertical = grids * np.roll(grids, -length, axis=1)
            features.append(0.5 * (horizontal.mean((1, 2)) + vertical.mean((1, 2))))
        return np.stack(features, axis=-1).reshape(*leading_shape, 3)

    def expected_translation_pair_features(self, q: float) -> FloatArray:
        """Return the exact mean of ``(z_bar, g1, g2)`` under mixture ``q``."""

        mixture = self._validate_mixture(q)
        detector_mean = self.detector_marginal[0]
        signed_mean = 1.0 - 2.0 * detector_mean
        contrast = 1.0 - 2.0 * self.readout_error
        first = contrast**2 * (
            1.0
            - self.event_probability * (6.0 + 2.0 * mixture) / self.num_detectors
        )
        second = contrast**2 * (
            1.0
            - self.event_probability * (8.0 - 2.0 * mixture) / self.num_detectors
        )
        return np.asarray([signed_mean, first, second], dtype=np.float64)

    def posterior_standardized_length_score(
        self,
        observed: ArrayLike,
        q: float,
    ) -> FloatArray:
        """Return ``E[(H-q)/sqrt(q(1-q)) | observed]``.

        This posterior score gives the exact pair likelihood ratio
        ``1 + kappa * a(y1) * a(y2)`` against the independent-length null.
        """

        mixture = self._validate_mixture(q)
        if not 0.0 < mixture < 1.0:
            raise ValueError("posterior length score requires q strictly in (0, 1).")
        first_log = (
            _log_probability(1.0 - mixture)
            + self.conditional_length_emission_log_likelihoods(observed, 1)
        )
        second_log = (
            _log_probability(mixture)
            + self.conditional_length_emission_log_likelihoods(observed, 2)
        )
        normalizer = np.logaddexp(first_log, second_log)
        posterior_second = np.exp(second_log - normalizer)
        return (posterior_second - mixture) / np.sqrt(
            mixture * (1.0 - mixture)
        )

    def fourier_power_features(self, observed: ArrayLike) -> FloatArray:
        """Normalized 2-D Fourier power of ``z = 1 - 2 y``.

        Output modes are flattened in row-major order.  Since every entry of
        ``z`` has unit magnitude, Parseval's identity makes every output row
        nonnegative and sum exactly to one up to floating-point roundoff.
        """

        flat, leading_shape = self._coerce_observations(observed, name="observed")
        signed = 1.0 - 2.0 * flat.astype(np.float64)
        grids = signed.reshape(-1, self.size, self.size)
        spectrum = np.fft.fft2(grids, axes=(-2, -1), norm="ortho")
        power = np.abs(spectrum) ** 2 / self.num_detectors
        return np.asarray(
            power.reshape(*leading_shape, self.num_detectors),
            dtype=np.float64,
        )

    def expected_fourier_spectrum(self, q: float) -> FloatArray:
        """Exact expected normalized Fourier power after BSC readout."""

        mixture = self._validate_mixture(q)
        empty = np.zeros((1, self.num_detectors), dtype=np.uint8)
        clean = (
            (1.0 - self.event_probability) * self.fourier_power_features(empty)[0]
            + self.event_probability
            * (
                (1.0 - mixture)
                * self.fourier_power_features(self.template_table(1)).mean(axis=0)
                + mixture
                * self.fourier_power_features(self.template_table(2)).mean(axis=0)
            )
        )
        readout_contrast = 1.0 - 2.0 * self.readout_error
        noise_floor = (
            1.0 - readout_contrast**2
        ) / self.num_detectors
        return noise_floor + readout_contrast**2 * clean

    def null_fourier_spectrum(self, q: float) -> FloatArray:
        """Alias for the exact stationary expected Fourier spectrum."""

        return self.expected_fourier_spectrum(q)

    def fourier_power_state_features(self, observed: ArrayLike) -> FloatArray:
        """Alias emphasizing that every Fourier-power row is a probability state."""

        return self.fourier_power_features(observed)

    def _sample_observed_given_lengths(
        self,
        lengths: NDArray[np.int64],
        rng: np.random.Generator,
    ) -> BinaryArray:
        original_shape = lengths.shape
        flat_lengths = lengths.reshape(-1)
        count = len(flat_lengths)
        truth = np.zeros((count, self.num_detectors), dtype=np.uint8)
        active = rng.random(count) < self.event_probability
        active_rows = np.flatnonzero(active)
        if len(active_rows):
            starts = rng.integers(0, self.num_detectors, size=len(active_rows))
            directions = rng.integers(0, 2, size=len(active_rows))
            rows = starts // self.size
            columns = starts % self.size
            selected_lengths = flat_lengths[active_rows]
            horizontal_end = (
                rows * self.size + (columns + selected_lengths) % self.size
            )
            vertical_end = (
                (rows + selected_lengths) % self.size
            ) * self.size + columns
            ends = np.where(directions == 0, horizontal_end, vertical_end)
            truth[active_rows, starts] = 1
            truth[active_rows, ends] = 1
        readout_flips = rng.random(truth.shape) < self.readout_error
        observed = np.bitwise_xor(truth, readout_flips.astype(np.uint8))
        return observed.reshape(*original_shape, self.num_detectors)

    def sample_spatial(
        self,
        shots: int,
        q: float,
        *,
        seed: int | None = None,
        rng: np.random.Generator | None = None,
        return_lengths: bool = False,
    ) -> BinaryArray | tuple[BinaryArray, NDArray[np.int64]]:
        """Sample independent observations from the exact spatial mixture."""

        if shots < 0:
            raise ValueError("shots must be nonnegative.")
        mixture = self._validate_mixture(q)
        generator = _resolve_rng(seed=seed, rng=rng)
        lengths = 1 + (generator.random(shots) < mixture).astype(np.int64)
        observed = self._sample_observed_given_lengths(lengths, generator)
        if return_lengths:
            return observed, lengths
        return observed

    def sample_temporal(
        self,
        cycles: int,
        *,
        q: float,
        kappa: float,
        streams: int = 1,
        seed: int | None = None,
        rng: np.random.Generator | None = None,
        return_lengths: bool = False,
    ) -> BinaryArray | tuple[BinaryArray, NDArray[np.int64]]:
        """Sample stationary Markov length streams and their noisy syndromes."""

        if cycles < 0:
            raise ValueError("cycles must be nonnegative.")
        if streams <= 0:
            raise ValueError("streams must be positive.")
        mixture = self._validate_mixture(q)
        persistence = self._validate_persistence(kappa)
        generator = _resolve_rng(seed=seed, rng=rng)
        lengths = np.empty((streams, cycles), dtype=np.int64)
        if cycles:
            lengths[:, 0] = 1 + (
                generator.random(streams) < mixture
            ).astype(np.int64)
            for cycle in range(1, cycles):
                refresh = generator.random(streams) >= persistence
                lengths[:, cycle] = lengths[:, cycle - 1]
                if np.any(refresh):
                    lengths[refresh, cycle] = 1 + (
                        generator.random(int(refresh.sum())) < mixture
                    ).astype(np.int64)
        observed = self._sample_observed_given_lengths(lengths, generator)
        if return_lengths:
            return observed, lengths
        return observed


def periodic_boundary_syndrome(edge_paths: ArrayLike) -> BinaryArray:
    """Return the mod-2 boundary of edge paths on a periodic one-dimensional ring.

    Edge ``i`` connects detector vertices ``i`` and ``i + 1`` modulo the ring
    size.  The syndrome at vertex ``i`` is therefore
    ``edge[i - 1] XOR edge[i]``.
    """

    paths = np.asarray(edge_paths)
    if paths.ndim == 0 or paths.shape[-1] < 2:
        raise ValueError("edge_paths must have a final ring dimension of at least 2.")
    if not np.all((paths == 0) | (paths == 1)):
        raise ValueError("edge_paths must contain only binary values.")
    binary = paths.astype(np.uint8, copy=False)
    return np.bitwise_xor(binary, np.roll(binary, 1, axis=-1))


def toggle_closed_logical_loop(edge_paths: ArrayLike) -> BinaryArray:
    """Add the unique noncontractible all-edge loop to periodic edge paths."""

    paths = np.asarray(edge_paths)
    if paths.ndim == 0 or paths.shape[-1] < 2:
        raise ValueError("edge_paths must have a final ring dimension of at least 2.")
    if not np.all((paths == 0) | (paths == 1)):
        raise ValueError("edge_paths must contain only binary values.")
    return np.bitwise_xor(paths.astype(np.uint8, copy=False), np.uint8(1))


@dataclass(frozen=True)
class LogicalLoopNoGoCertificate:
    """Pathwise certificate for the blindness of every syndrome-only observer."""

    paths_checked: int
    ring_size: int
    pathwise_syndrome_equal: bool
    maximum_syndrome_difference: int
    syndrome_total_variation: float
    optimal_equal_prior_syndrome_success: float


def logical_loop_access_no_go(edge_paths: ArrayLike) -> LogicalLoopNoGoCertificate:
    """Pair paths with their closed-loop translates and certify syndrome blindness.

    Under equal priors on the original and loop-shifted members of every pair,
    their syndrome distributions agree path by path.  The total-variation
    distance available to a syndrome-only observer is exactly zero and its
    optimal equal-prior classification success is exactly one half.
    """

    paths = np.asarray(edge_paths)
    if paths.ndim == 1:
        paths = paths[None, :]
    if paths.ndim < 2:
        raise ValueError("edge_paths must contain at least one path.")
    original = periodic_boundary_syndrome(paths)
    shifted = periodic_boundary_syndrome(toggle_closed_logical_loop(paths))
    difference = np.abs(original.astype(np.int8) - shifted.astype(np.int8))
    maximum = int(difference.max(initial=0))
    return LogicalLoopNoGoCertificate(
        paths_checked=int(np.prod(paths.shape[:-1])),
        ring_size=int(paths.shape[-1]),
        pathwise_syndrome_equal=bool(maximum == 0),
        maximum_syndrome_difference=maximum,
        syndrome_total_variation=0.0 if maximum == 0 else float("nan"),
        optimal_equal_prior_syndrome_success=0.5 if maximum == 0 else float("nan"),
    )


@dataclass(frozen=True)
class PeriodicSyndromeModel:
    """Uniformly translated open-chain emissions on a periodic detector ring."""

    size: int
    event_probability: float
    chain_length: int
    persistence: float = 0.0

    def __post_init__(self) -> None:
        if self.size < 3:
            raise ValueError("size must be at least 3.")
        if not 0.0 <= self.event_probability <= 1.0:
            raise ValueError("event_probability must lie in [0, 1].")
        if not 1 <= self.chain_length < self.size:
            raise ValueError("chain_length must lie in [1, size - 1].")
        if not 0.0 <= self.persistence <= 1.0:
            raise ValueError("persistence must lie in [0, 1].")

    @property
    def stationary_latent_distribution(self) -> FloatArray:
        """Stationary probabilities for empty state 0 and starts 1 through size."""

        probabilities = np.full(
            self.size + 1,
            self.event_probability / self.size,
            dtype=np.float64,
        )
        probabilities[0] = 1.0 - self.event_probability
        return probabilities

    def transition_matrix(self, persistence: float | None = None) -> FloatArray:
        """Return the refresh-or-repeat latent Markov transition matrix."""

        value = self.persistence if persistence is None else float(persistence)
        if not 0.0 <= value <= 1.0:
            raise ValueError("persistence must lie in [0, 1].")
        dimension = self.size + 1
        refresh = np.broadcast_to(
            self.stationary_latent_distribution,
            (dimension, dimension),
        ).copy()
        return value * np.eye(dimension) + (1.0 - value) * refresh

    def edge_path(self, start: int) -> BinaryArray:
        """Return a consecutive open chain beginning at edge ``start``."""

        if not 0 <= start < self.size:
            raise ValueError("start must lie in [0, size).")
        path = np.zeros(self.size, dtype=np.uint8)
        path[(start + np.arange(self.chain_length)) % self.size] = 1
        return path

    def syndrome_from_start(self, start: int) -> BinaryArray:
        """Return the two-endpoint syndrome for a translated open chain."""

        return periodic_boundary_syndrome(self.edge_path(start))

    def syndrome_from_latent(self, latent: int) -> BinaryArray:
        """Map latent 0 to no event and latent ``start + 1`` to an open chain."""

        if not 0 <= latent <= self.size:
            raise ValueError("latent must lie in [0, size].")
        if latent == 0:
            return np.zeros(self.size, dtype=np.uint8)
        return self.syndrome_from_start(latent - 1)

    @property
    def emission_table(self) -> BinaryArray:
        """Deterministic syndrome emitted by each latent state."""

        return np.stack(
            [self.syndrome_from_latent(latent) for latent in range(self.size + 1)]
        )

    def emission_probability(self, syndrome: ArrayLike) -> float:
        """Return the exact single-cycle likelihood of a binary syndrome."""

        value = _as_binary_last_axis(syndrome, self.size, name="syndrome")
        if value.ndim != 1:
            raise ValueError("emission_probability expects one syndrome vector.")
        matches = np.all(self.emission_table == value[None, :], axis=1)
        return float(self.stationary_latent_distribution[matches].sum())

    def emission_likelihoods(self, syndromes: ArrayLike) -> FloatArray:
        """Vectorized exact single-cycle likelihoods."""

        values = _as_binary_last_axis(syndromes, self.size, name="syndromes")
        flat = values.reshape(-1, self.size)
        table = self.emission_table
        matches = np.all(flat[:, None, :] == table[None, :, :], axis=2)
        likelihoods = matches @ self.stationary_latent_distribution
        return likelihoods.reshape(values.shape[:-1])

    @property
    def count_pmf(self) -> FloatArray:
        """Exact PMF of the number of detection events, indexed by count."""

        pmf = np.zeros(self.size + 1, dtype=np.float64)
        table_counts = self.emission_table.sum(axis=1)
        for count, probability in zip(
            table_counts,
            self.stationary_latent_distribution,
            strict=True,
        ):
            pmf[int(count)] += probability
        return pmf

    @property
    def detector_marginal(self) -> FloatArray:
        """Exact one-cycle event probability at every detector."""

        return self.stationary_latent_distribution @ self.emission_table

    def fourier_power_features(self, syndromes: ArrayLike) -> FloatArray:
        """Return translation-invariant, Parseval-normalized Fourier power.

        The transform is divided by ``sqrt(size)``.  Consequently, the sum of
        all power features equals the syndrome Hamming weight.
        """

        values = _as_binary_last_axis(syndromes, self.size, name="syndromes")
        spectrum = np.fft.fft(values, axis=-1) / np.sqrt(self.size)
        return np.asarray(np.abs(spectrum) ** 2, dtype=np.float64)

    @property
    def null_fourier_spectrum(self) -> FloatArray:
        """Exact expected Fourier-power vector under the stationary emission law."""

        return (
            self.stationary_latent_distribution[:, None]
            * self.fourier_power_features(self.emission_table)
        ).sum(axis=0)

    def centered_fourier_power_features(self, syndromes: ArrayLike) -> FloatArray:
        """Fourier-power features centered by their exact null expectation."""

        return self.fourier_power_features(syndromes) - self.null_fourier_spectrum

    def _sample_stationary_latents(
        self,
        count: int,
        rng: np.random.Generator,
    ) -> NDArray[np.int64]:
        return rng.choice(
            self.size + 1,
            size=count,
            p=self.stationary_latent_distribution,
        )

    def sample_spatial(
        self,
        shots: int,
        *,
        seed: int | None = None,
        rng: np.random.Generator | None = None,
        return_latents: bool = False,
    ) -> BinaryArray | tuple[BinaryArray, NDArray[np.int64]]:
        """Sample independent stationary syndrome emissions."""

        if shots < 0:
            raise ValueError("shots must be nonnegative.")
        generator = _resolve_rng(seed=seed, rng=rng)
        latents = self._sample_stationary_latents(shots, generator)
        syndromes = self.emission_table[latents]
        if return_latents:
            return syndromes, latents
        return syndromes

    def sample_temporal(
        self,
        cycles: int,
        *,
        streams: int = 1,
        persistence: float | None = None,
        seed: int | None = None,
        rng: np.random.Generator | None = None,
        return_latents: bool = False,
    ) -> BinaryArray | tuple[BinaryArray, NDArray[np.int64]]:
        """Sample stationary refresh-or-repeat syndrome streams.

        The returned syndrome array always has shape
        ``(streams, cycles, size)``.  Latent state zero is the empty emission;
        states one through ``size`` encode the translated chain start.
        """

        if cycles < 0:
            raise ValueError("cycles must be nonnegative.")
        if streams <= 0:
            raise ValueError("streams must be positive.")
        value = self.persistence if persistence is None else float(persistence)
        if not 0.0 <= value <= 1.0:
            raise ValueError("persistence must lie in [0, 1].")
        generator = _resolve_rng(seed=seed, rng=rng)
        latents = np.empty((streams, cycles), dtype=np.int64)
        if cycles:
            latents[:, 0] = self._sample_stationary_latents(streams, generator)
            for cycle in range(1, cycles):
                refresh = generator.random(streams) >= value
                latents[:, cycle] = latents[:, cycle - 1]
                if np.any(refresh):
                    latents[refresh, cycle] = self._sample_stationary_latents(
                        int(refresh.sum()),
                        generator,
                    )
        syndromes = self.emission_table[latents]
        if return_latents:
            return syndromes, latents
        return syndromes

    def temporal_likelihood(
        self,
        syndrome_sequence: ArrayLike,
        *,
        persistence: float | None = None,
    ) -> float:
        """Return the exact stationary HMM likelihood of one syndrome sequence."""

        sequence = _as_binary_last_axis(
            syndrome_sequence,
            self.size,
            name="syndrome_sequence",
        )
        if sequence.ndim != 2:
            raise ValueError("temporal_likelihood expects shape (cycles, size).")
        if len(sequence) == 0:
            return 1.0
        transition = self.transition_matrix(persistence)
        table = self.emission_table
        forward = self.stationary_latent_distribution.copy()
        for cycle, syndrome in enumerate(sequence):
            emission_mask = np.all(table == syndrome[None, :], axis=1)
            forward *= emission_mask
            if cycle + 1 < len(sequence):
                forward = forward @ transition
        return float(forward.sum())
