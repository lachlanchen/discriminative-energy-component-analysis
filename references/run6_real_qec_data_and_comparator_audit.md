# Run 6 audit: real-hardware QEC streams and drift comparators

**Audit date:** 2026-07-27
**Scope:** publicly downloadable *experimental* surface-code or repetition-code syndrome/measurement data, plus reproducible drift/noise-estimation baselines. Simulator-only data and records containing only aggregate statistics are not treated as real syndrome streams.

## Decision

**GO**, but only for a tightly scoped real-hardware **stability / transient-burst / witness-recovery** study.

**NO-GO** for a definitive benchmark of naturally occurring online changepoints with exact detection delay: none of the audited raw QEC deposits supplies both wall-clock timestamps and machine-readable natural change labels. The most useful Google data preserve several kinds of order, but experiment order, shot order, QEC-cycle order, wall-clock time, and intervention labels must not be conflated.

**NO-GO** for calling a fixed-syndrome-data baseline “eSCD.” The published eSCD protocol requires its own randomized Clifford measurement settings and outcomes; these cannot be reconstructed from archived stabilizer syndrome bits. Only its classical e-detector layer can be applied faithfully to already observed QEC quantities.

The strongest feasible three-stage study is:

1. **Pilot:** Google 2022, 315.5 MB, including the author-identified high-energy event near shot 57,775.
2. **Transient test:** Google 2024 distance-29 repetition code, 65.0 GB, with documented sequential sample acquisition and the six large bursts discussed in the paper.
3. **Long-horizon stability/intervention test:** Google 2024 surface-code `set1`, whose final 16 experiments were acquired sequentially over 15 hours; the processor was recalibrated between groups of four runs. Those recalibrations are intervention markers, not ground-truth drift labels.

## What “time ordered” means here

Four distinct indices occur in the archives:

1. **QEC round within one shot.**
2. **Shot-row order within one sample/experiment.**
3. **Sample/experiment order.**
4. **Wall-clock acquisition time.**

The Google repetition-code archive explicitly documents sequential acquisition at level 3, and the circuits preserve level 1. The deposits do **not** provide a wall-clock timestamp per shot or per QEC round. A recalibration boundary or a visually identified burst is not automatically the unknown physical changepoint that a detector is supposed to discover.

## Raw real-hardware datasets

All Zenodo records below expose a metadata endpoint at `https://zenodo.org/api/records/<record_id>` and a stable file endpoint of the form `https://zenodo.org/api/records/<record_id>/files/<filename>/content`. Exact byte counts and licenses below were read from those APIs.

### 1. Google Quantum AI: below-threshold surface and repetition codes — primary Run 6 source

- **Official record:** [Zenodo 13273331](https://zenodo.org/records/13273331), DOI [10.5281/zenodo.13273331](https://doi.org/10.5281/zenodo.13273331)
- **Metadata API:** <https://zenodo.org/api/records/13273331>
- **Associated paper:** [Quantum error correction below the surface code threshold, Nature 2025](https://www.nature.com/articles/s41586-024-08449-y)
- **License:** CC BY 4.0
- **Total deposited size:** 112,527,420,393 bytes
- **Format:** ZIP archives containing hardware `measurements.b8`, derived `detection_events.b8`, actual logical flips, sweep bits where applicable, Stim circuits/noisy detector-error models, metadata JSON, and decoder predictions/results. The `.b8` rows are byte-aligned little-endian bit-packed records; circuit/metadata determine the number and meaning of bits per shot.

| Archive and exact content endpoint | Bytes | Temporal value | Labels and limitations |
|---|---:|---|---|
| [`google_72Q_surface_code_d3_d5_set1.zip`](https://zenodo.org/api/records/13273331/files/google_72Q_surface_code_d3_d5_set1.zip/content) | 30,152,649,005 | README: the last 16 experiments were performed sequentially over 15 hours. Nature paper: processor recalibrated between every four experimental runs. | No per-shot wall-clock timestamp, calibration event file, or natural changepoint label. Treat every-four-run boundaries as known interventions only. |
| [`google_72Q_repetition_code_d29.zip`](https://zenodo.org/api/records/13273331/files/google_72Q_repetition_code_d29.zip/content) | 65,033,044,069 | `X/` and `Z/` each contain `sample_00`–`sample_99`, documented as sequentially acquired; each sample has 100,000 shots and each shot has 1,000 QEC cycles. The paper reports 20 million shots, 20 billion cycles, and 5.5 hours of device execution. | The paper discusses six large error bursts, but the archive has no canonical burst-label file or per-shot wall-clock timestamp. Strong natural transient case; weak exact-delay ground truth. |
| [`google_105Q_surface_code_d3_d5_d7.zip`](https://zenodo.org/api/records/13273331/files/google_105Q_surface_code_d3_d5_d7.zip/content) | 5,716,907,033 | Raw experimental shots at multiple distances. | No documented long chronological series/change labels; useful control and scaling data, not the primary drift stream. |
| [`google_72Q_surface_code_d3_d5_set2.zip`](https://zenodo.org/api/records/13273331/files/google_72Q_surface_code_d3_d5_set2.zip/content) | 11,624,820,286 | Raw surface-code experiments. | No natural change labels or sufficient timestamp metadata for a clean online drift benchmark. |

**Recommended use.** The distance-29 archive is the best public source found for real transient syndrome dynamics. `set1` is the best for long-horizon cohort stability. Keep shot boundaries explicit: a new shot reinitializes the quantum state, so the last QEC round of one shot must not be treated as physically adjacent to the first round of the next shot without a boundary indicator.

### 2. Google Quantum AI 2022 surface/repetition-code data — manageable pilot

- **Official record:** [Zenodo 6804040](https://zenodo.org/records/6804040), DOI [10.5281/zenodo.6804040](https://doi.org/10.5281/zenodo.6804040)
- **Metadata API:** <https://zenodo.org/api/records/6804040>
- **File:** [`google_qec3v5_experiment_data.zip`](https://zenodo.org/api/records/6804040/files/google_qec3v5_experiment_data.zip/content), **315,490,804 bytes**
- **Associated paper:** [Suppressing quantum errors by scaling a surface code logical qubit, Nature 2023](https://www.nature.com/articles/s41586-022-05434-1)
- **License:** CC BY 4.0
- **Format:** raw bit-packed measurements and detector events, actual/predicted logical flips, circuits, detector-error models, and device properties.
- **Ordering/labels:** shot-row order is retained. The archive README points to a known high-energy event in `repetition_code_bZ_d25_r50_center_5_5` near shot 57,775, which is an unusually useful pre-existing witness. It is still an approximate author-identified event, not a complete changepoint annotation, and no per-shot wall-clock timestamps are supplied.

This is the best low-cost first run because it tests parsing, causal windows, detector calibration, witness localization, and false alarms before downloading 65 GB.

### 3. Google Sycamore decoder-prior experiments — real, but not a drift stream

- **Official record:** [Zenodo 11403595](https://zenodo.org/records/11403595), concept DOI [10.5281/zenodo.11403594](https://doi.org/10.5281/zenodo.11403594)
- **Metadata API:** <https://zenodo.org/api/records/11403595>
- **License:** CC BY 4.0
- **Format:** raw hardware measurements/detector events, Stim circuits and decoding artifacts.
- **Files:** [`google_sycamore_surface_code_d3_d5.zip`](https://zenodo.org/api/records/11403595/files/google_sycamore_surface_code_d3_d5.zip/content), **6,216,793,617 bytes**; [`google_sycamore_repetition_code_d21.zip`](https://zenodo.org/api/records/11403595/files/google_sycamore_repetition_code_d21.zip/content), **4,921,810,549 bytes**.
- **Ordering/labels:** samples represent experiments/layouts; the README does not document them as one chronological acquisition stream. No wall-clock timestamps or change labels.

Use it as a cross-layout/domain-shift control, not as evidence of natural drift detection.

### 4. Google dynamic surface-code experiments — real, but treatment arms rather than time

- **Official record:** [Zenodo 14238907](https://zenodo.org/records/14238907), DOI [10.5281/zenodo.14238907](https://doi.org/10.5281/zenodo.14238907)
- **Metadata API:** <https://zenodo.org/api/records/14238907>
- **File:** [`google_dynamic_circuits_d3_d5.zip`](https://zenodo.org/api/records/14238907/files/google_dynamic_circuits_d3_d5.zip/content), **2,081,742,123 bytes**
- **Associated paper:** [Demonstrating dynamic surface codes, Nature Physics 2026](https://www.nature.com/articles/s41567-025-03070-w)
- **License:** CC BY 4.0
- **Format:** hardware measurements/detection events plus Stim circuits and decoder outputs for `iswap`, `hexagonal`, and `walking` implementations.
- **Ordering/labels:** implementation identity is known, but the samples are not documented as one chronological drift stream. This supports static domain/treatment discrimination, not changepoint claims.

### 5. IBM/PNNL calibration-conditioned repetition-code snapshots — useful auxiliary validation

- **Official author deposit:** [Zenodo 20768087](https://zenodo.org/records/20768087), DOI [10.5281/zenodo.20768087](https://doi.org/10.5281/zenodo.20768087)
- **Metadata API:** <https://zenodo.org/api/records/20768087>
- **Associated preprint:** [Calibration-Conditioned FiLM Decoders…, arXiv:2601.16123](https://arxiv.org/abs/2601.16123)
- **License:** CC BY 4.0
- **Format:** compressed TAR archives containing per-shot `bitstrings.json`, `calibration.json`, transpiled QASM and job metadata for repetition-code distances 3–11, rounds 1–11, and X/Z bases. The record reports 352 hardware snapshots, 3,779,584 shots, and 20,654,080 extracted chain samples.
- **Files:** [`ibm_fez.tar.gz`](https://zenodo.org/api/records/20768087/files/ibm_fez.tar.gz/content), **9,021,652 bytes**; [`ibm_kingston.tar.gz`](https://zenodo.org/api/records/20768087/files/ibm_kingston.tar.gz/content), **71,399,630 bytes**; [`ibm_pittsburgh.tar.gz`](https://zenodo.org/api/records/20768087/files/ibm_pittsburgh.tar.gz/content), **65,285,202 bytes**; plus README/DESCRIPTION, for **145,713,052 bytes total**.
- **Ordering/labels:** calibration records contain dates/`last_update_date`, but anonymized job folders and `info.json` do not provide a reliable shot-acquisition timestamp or a canonical total order. Calibration time is not the same as job execution time. No natural change labels.
- **Reproducibility warning:** the README/description mention an `index.csv`, but it is absent from the Zenodo file list and from the three archives checked on the audit date. The linked GitHub repository, `Samuelstein1224/calibration-conditioned-decoding`, returned HTTP 404 on 2026-07-27.

Use this as a held-out, cross-device/calibration-conditioned robustness test. Do not present it as an ordered online-drift benchmark unless the authors restore the index and acquisition chronology.

### Aggregate-only record excluded from the raw-stream benchmark

[DAQEC-Benchmark, Zenodo 18045662](https://zenodo.org/records/18045662) ([metadata API](https://zenodo.org/api/records/18045662)) is a primary author deposit claiming IBM hardware validation and is CC BY 4.0, but its entire deposit is only **408,943 bytes**. It contains `master.parquet` (109,439 bytes), aggregate CSV/JSON summaries and experiment scripts, not a raw per-shot syndrome stream. The associated manuscript was described by the depositor as submitted, not peer-reviewed. It can be an external aggregate sanity check, but it cannot support round-level Run 6 detection or witness recovery.

## Comparator audit

### Comparator matrix

| Comparator | What it tests | Evidence status | Reproducibility and Run 6 use |
|---|---|---|---|
| **Static calibrated MWPM** | Decoder with fixed detector-error-model edge weights | Mature standard baseline | Use [PyMatching](https://github.com/oscarhiggott/PyMatching) and [Stim](https://github.com/quantumlib/Stim), both Apache-2.0. The Google archives also contain circuits/DEMs and decoder artifacts. This is the essential no-adaptation decoder control. |
| **Detector-fire-rate rolling window / ReloQate** | Directly observed mean detector activity and DFR-to-logical-error proxy | [ReloQate, arXiv:2603.00837](https://arxiv.org/abs/2603.00837), 2026 preprint; it derives volatile traces from Google hardware data but evaluates a broader simulated system architecture | Implement the disclosed rolling DFR baseline. No official code link was found in the arXiv source as of the audit date. This is a particularly important simple baseline: ECA must beat it, not merely beat PCA. |
| **DGR graph reweighting** | Adapts MWPM edge and edge-pair weights from decoded matching statistics | [arXiv:2311.16214](https://arxiv.org/abs/2311.16214); reported gains are from simulated surface/honeycomb noise mismatch | No official implementation was found. It must be reimplemented from the paper. DGR is an adaptive decoder, not a calibrated changepoint detector; compare downstream logical error and adaptation budget, not just alarm delay. |
| **QEC sliding/overlapping-window noise tracking** | Tracks time-varying Pauli noise from syndrome statistics and updates decoding | [Adaptive Estimation of Drifting Noise in QEC](https://arxiv.org/abs/2511.09491), accepted by [PRX Quantum](https://journals.aps.org/prxquantum/accepted/10.1103/z1hc-nqw5) on 2026-06-23 | Validated on phenomenological/circuit-level simulation, not the audited real streams; no official code found. Strongest current QEC-native estimator conceptually, but a paper reimplementation is required. |
| **Bayesian MCMC/SMC tracking** | Posterior estimation of general stationary/time-varying surface-code noise parameters using a tensor-network likelihood | [Kobori and Todo, arXiv:2406.08981](https://arxiv.org/abs/2406.08981), [Physical Review A DOI](https://doi.org/10.1103/wg5h-spy6) | Numerical validation only; no official code found. Strong model-based comparator when the channel is identifiable, but expensive and model-dependent. Report failed identifiability as a result, not silently tuned success. |
| **Windowed detector-rate likelihood / GLR / CUSUM** | Sequential mean/rate change in an observed scalar or low-dimensional vector | Standard statistical control | Implement Bernoulli/Poisson-binomial likelihood CUSUM for detector rate or decoder negative log likelihood. Calibrate every method to the same average run length (ARL) or false-alarm probability using a held-out stationary prefix. A post-change-known CUSUM is an oracle upper bound; mark it as such. |
| **e-CUSUM / e-SR on observed QEC features** | Nonparametric sequential e-process with explicit ARL control | [E-detectors paper](https://arxiv.org/abs/2203.03532); [paper scripts](https://github.com/shinjaehyeok/e_detector_paper), MIT; recommended [`stcpR6`](https://github.com/shinjaehyeok/stcpR6) package, GPL-3.0 | Reproducible and statistically preferable to an uncalibrated threshold. Apply to a predeclared bounded DFR, decoder score, Wilson/parity observable or ECA score. Call it observable-specific e-CUSUM/e-SR, not eSCD. |
| **Covariance-change scan** | Change in multivariate second-order structure | Reproducible with [`ruptures`](https://github.com/deepcharles/ruptures), BSD-2-Clause; `CostNormal` uses a segment log-determinant covariance cost | Useful offline localization control. Binary detector dimension often exceeds the window length, so raw covariance is singular; predeclare block aggregation plus shrinkage or a fixed training-only projection. An offline scan cannot be compared as if it were an online alarm. |
| **Page–Hinkley** | Lightweight online scalar-mean drift | Reproducible with [`river`](https://github.com/online-ml/river), BSD-3-Clause | Good engineering baseline on DFR/decoder NLL/ECA score, but its threshold does not by itself provide the same exact ARL guarantee as an e-detector. Calibrate empirically under the same stream and budget. |
| **pyGSTi spectral stability analysis** | Detects/characterizes nonstationary quantum click probabilities by power spectra | [Nature Communications 2020](https://www.nature.com/articles/s41467-020-18953-0); [official data/notebooks](https://zenodo.org/records/4033077), 331,358,713 bytes, CC BY 4.0; [pyGSTi](https://github.com/sandialabs/pyGSTi), Apache-2.0 | This is the strongest directly reproducible general quantum-drift control found. Adapt a detector bit/rate clickstream, preserving approximately equal spacing. It is spectral stability analysis, not automatically a causal quickest-change detector. |
| **eSCD + classical shadows** | Universal sequential detection when the relevant observables are unknown to the measurement device | [arXiv:2602.11846](https://arxiv.org/abs/2602.11846); theory and numerical experiments | No official code found. Not faithfully runnable on the archived QEC files: eSCD requires the randomized local/joint Clifford setting \(U_t\), computational-basis outcome, and inverse shadow channel at each time. Fixed stabilizer measurements supply neither the randomized settings nor an informationally complete shadow. |

### The key eSCD boundary

The archived QEC measurements are extremely valuable, but they are **observable-specific**. They measure a fixed family of stabilizer/parity quantities designed by the code. eSCD's “measure once, test many observables later” advantage arises from a different acquisition protocol: a randomized classical-shadow measurement is made before the detector chooses its observable family.

Therefore:

- Running an e-process on syndrome-derived observables is legitimate.
- Calling that procedure e-CUSUM/e-SR on observed QEC features is legitimate.
- Calling it eSCD, claiming classical-shadow universality, or importing eSCD's shadow sample-complexity/delay guarantee is not legitimate.
- A faithful eSCD comparison requires a new hardware experiment or simulator that records every randomized Clifford setting and measurement outcome under the same shot budget.

## Fair Run 6 experimental contract

### Predeclare the units and budget

- Use an ordered **calibration prefix of exactly \(B\) shots** (or \(B\) completed QEC rounds) for every learned detector and adaptive decoder.
- Give every method the same stream, detector subset, causal history, and recalibration opportunities.
- Never train a projection, choose a threshold, or select a detector after seeing the held-out event.
- Preserve shot IDs, sample IDs, basis, code distance, round index and known intervention boundaries.
- Report compute time/memory separately; a detector that sees future samples or an offline covariance scan is not an online competitor.

### Primary metrics

1. **False alarm:** empirical ARL, false alarms per \(10^6\) cycles, and run-wise confidence intervals.
2. **Detection:** conditional delay in QEC rounds/shots after a predeclared event interval; miss probability by a fixed horizon.
3. **Localization/witness:** precision/recall and stability of selected detectors/edges/qubits, assessed only where an event window or injected mechanism supplies truth.
4. **Decoder consequence:** logical error rate or excess decoder loss under the same calibration/shots budget.
5. **Resources:** retained features, update cost, wall-clock latency and peak memory.

### Three label regimes must remain separate

1. **Natural, approximately located events:** the 2022 event near shot 57,775 and the six 2024 repetition-code bursts. Report sensitivity to the declared event-window width.
2. **Known interventions:** the `set1` recalibration boundaries. These test whether distributions change around intervention, not whether spontaneous drift was discovered.
3. **Constructed changepoints on real hardware data:** concatenate held-out stationary blocks from different experiments/calibrations/bases/layouts, or inject a fully specified bit-flip mechanism. Label every result “constructed from real-hardware samples”; it is not a naturally observed drift claim.

### What would constitute an ECA advantage

An advantage is supported only if a locked ECA procedure improves a meaningful Pareto frontier:

- lower detection delay at the **same empirical/theoretical ARL**, or
- lower logical error at the **same shot/calibration/update budget**, or
- comparable detection/decoding with materially lower causal compute/memory,

and the gain persists across devices/events/distances, not just one post-selected burst. Sparse or visually separable features are useful diagnostics, but are not by themselves an algorithmic advantage. ECA must be compared against raw DFR, an oracle/GLR CUSUM, e-CUSUM/e-SR, covariance scan, pyGSTi spectral analysis, static MWPM, and at least one adaptive decoder/noise tracker.

## Claim boundaries and final recommendation

Run 6 can claim:

- a reproducible audit and parser for real Google/IBM QEC measurements;
- calibrated, causal detection on fixed syndrome observables;
- natural-burst case studies with explicitly approximate labels;
- constructed-changepoint benchmarks on real hardware samples;
- witness recovery and downstream decoder effects under a shared budget.

Run 6 cannot claim from these data alone:

- exact natural changepoint delay or causal origin of the Google bursts;
- faithful eSCD/classical-shadow performance;
- superiority to DGR, Bayesian/SMC tracking or adaptive QEC estimation without implementing and budget-matching them;
- quantum speedup, quantum measurement advantage, universal sample-efficiency, or a general scalable-computation advantage.

**Go recommendation:** begin with the 315.5 MB 2022 pilot and lock the data model, calibration rule, ARL target and event windows. Only then download the 65.0 GB repetition archive. Treat the 30.2 GB `set1` study as ordered cohort/intervention analysis. A publication is plausible if ECA survives the simple DFR and calibrated e-/likelihood-CUSUM baselines and improves either matched-ARL delay or downstream logical error on held-out natural events; otherwise the honest output is a valuable negative result and reproducible real-hardware benchmark, not an ECA-advantage paper.
