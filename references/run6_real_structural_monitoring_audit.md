# Run 6 audit: real structural-health and rotating-machinery monitoring

**Audit date:** 2026-07-27
**Question:** Is a maximum-difference covariance witness a credible real-data contribution for structural or machine condition monitoring, and which public experiment can test it without manufacturing an onset label?
**Scope:** Primary-source audit of real experimental vibration streams with chronological degradation or intervention information. Large archives were not downloaded. Sizes below come from repository metadata or a one-byte HTTP range query; only small documentation files were inspected.

## Executive verdict

There is a real and important problem here, but there is **not yet evidence that the present algorithm has an advantage**.

The best publication-grade target is the [Paderborn University 17-bearing run-to-failure release](https://zenodo.org/records/10868257). It contains natural, unseeded bearing failures under time-varying speed and load, two raw vibration channels, temperature, and measured operating conditions. It is openly licensed under CC BY 4.0 and contains enough independent runs for leave-one-bearing-out evaluation. Its central practical question is:

> Can a compact, condition-adjusted covariance or cross-spectral witness give earlier and better-calibrated warning of impending failure than mature vibration diagnostics, when every method receives the same sensors, history, labels, and false-alarm budget?

The strongest full-scale civil-structure target is the [Z24 bridge benchmark](https://bwk.kuleuven.be/bwm/z24). Its progressive physical damage schedule is exceptionally well documented, and lagged covariance and cross-spectral structure are genuinely tied to modal dynamics. Access is restricted to approved non-commercial research, however, and the short-term tests are separate before/after blocks rather than one continuously observed damage onset.

The fastest labeled mechanism check is the [NIST linear-axis rail-degradation experiment](https://data.nist.gov/od/id/6EF435207EF17114E0532457068155831934). It has 15 measured wear stages, repeated triaxial accelerometer and gyroscope runs, three speeds, two travel directions, encoder position, and laser metrology. It is an excellent test of sensitivity, nuisance adjustment, and localization, but the degradation was deliberately applied between acquisition blocks. It must not be described as natural online onset detection.

The decisive scientific boundary is that a whitened maximum covariance-difference direction is algebraically a generalized covariance eigenvector, closely overlapping common-spatial-pattern and generalized Rayleigh-quotient methods. On bearings, a second-order covariance statistic may also lose to envelope analysis, cyclostationary analysis, or spectral kurtosis because incipient spalls produce sparse impacts and non-Gaussian transients. A publishable contribution therefore requires more than another eigendecomposition:

1. an honest online formulation with no post-change oracle;
2. condition adjustment for speed, load, and temperature;
3. valid threshold calibration after adaptive direction search;
4. strong signal-processing and same-feature statistical controls; and
5. a held-out physical-run advantage at a fixed event-level false-alarm rate.

If those tests are negative, the useful result is a reproducible **no-go boundary** showing when a rank-one covariance witness is insufficient. That would still be more credible than a broad superiority claim.

## 1. Inclusion standard

A candidate was ranked only if the primary provider establishes that the data came from a physical experiment. The audit distinguishes four kinds of time information:

1. **Natural chronology with a certified onset:** best case, but none of the shortlisted natural bearing releases supplies an independently verified incipient-fault onset time.
2. **Natural chronology with a terminal event:** the run is ordered and the failure or stopping criterion is known, but onset is latent.
3. **Known physical intervention boundary:** damage is applied between measurement blocks, so the first post-intervention block has a defensible boundary but is not a continuously observed onset.
4. **Configuration index only:** independently acquired states can be ordered or concatenated, but the resulting change point is constructed.

An “online detection-delay” claim is valid only in cases 1 or 3, with case 3 explicitly called an intervention boundary. Case 2 supports lead time to terminal failure, time-to-event prediction, and prospective alarm analysis, but not delay from true incipient onset. Case 4 is a classification or sensitivity benchmark, not an online field trial.

## 2. Ranked candidates

| Rank | Primary dataset | Physical evidence and channels | Time label that is actually available | Access, license, and exact size status | Decision |
|---:|---|---|---|---|---|
| 1 | [Paderborn time-varying run-to-failure bearings](https://zenodo.org/records/10868257) | 17 unseeded run-to-failure experiments; two raw housing accelerometers; bearing and ambient temperatures; measured speed, static load, and dynamic load | Natural chronological snapshots and terminal stopping criterion/teardown fault; **no incipient-onset annotation** | Open, [CC BY 4.0 in the Zenodo API record](https://zenodo.org/api/records/10868257); **152,008,932,259 bytes** total; B01 alone is **965,866,495 bytes** | **Primary rotating-machine study** |
| 2 | [KU Leuven Z24 bridge](https://bwk.kuleuven.be/bwm/z24) | Full-scale bridge; 16-channel long-term acceleration plus 48 environmental channels; progressive damage tests with 291 measured DOF | Dated, known physical interventions with before/after forced and ambient tests; not continuous onset during intervention | Approved non-commercial research only; attribution/citation required; no third-party transfer; official archive size **not published** | **Best civil-structure validation**, pending access |
| 3 | [NIST linear-axis rail degradation](https://data.nist.gov/od/id/6EF435207EF17114E0532457068155831934) | 15 intentional wear stages; triaxial acceleration, triaxial angular velocity, encoder position, temperature, and six-DOF laser reference; repeated speed/direction runs | Known stage boundary and measured wear profile; deliberately staged, not natural or continuous | Public under the [NIST data terms](https://www.nist.gov/open/license); 14 ZIPs plus README, **4,423,616,111 bytes** total | **Run first as a mechanism and calibration check** |
| 4 | [NASA/IMS bearing archive](https://data.nasa.gov/dataset/ims-bearings) | Three real run-to-failure bearing tests with raw vibration snapshots and terminal fault inspection | Natural timestamp order and terminal fault; **no independently certified onset** | Public NASA portal; metadata license field points to [U.S. Government Works](https://www.usa.gov/government-works); archive is **1,061,902,801 bytes** | Useful legacy external check; annotation and license provenance need care |
| 5 | [LANL three-story structure in SHMTools](https://sourceforge.net/projects/shmtools/) | Physical aluminum structure; force input and four accelerometers; 17 mass/stiffness/impact-gap configurations | Configuration blocks only; any online boundary is created by concatenation | SHMTools software has BSD-3-Clause-like terms, but an independent raw-data license is not stated; array payload is about 55.7 MB or 278.5 MB depending version | Sensitivity/confound benchmark only |
| Watch | [XJTU-SY run-to-failure bearings](https://github.com/WangBiaoXJTU/xjtu-sy-bearing-datasets) | 15 physical run-to-failure bearings, two vibration directions, three operating conditions | Natural order and terminal teardown labels; onset remains inferred | Provider says anyone may use it for prognostics validation and requests citation, but the [repository API reports no standard license](https://api.github.com/repos/WangBiaoXJTU/xjtu-sy-bearing-datasets); official page gives no exact archive size | Technically attractive; confirm rights and provenance before use or redistribution |

The ranking reflects publication value, not download order. A sensible execution order is NIST for a rapid controlled check, a bounded Paderborn subset for the natural-failure study, and Z24 after official access is granted.

## 3. Evidence and limitations by dataset

### 3.1 Paderborn: strongest natural online target

The [official Zenodo record](https://zenodo.org/records/10868257) states that the release contains 17 physical run-to-failure experiments on 61806-2RS ball bearings, that no defect was initiated before testing, and that speed and load vary during operation. The accompanying primary description specifies:

- two provided one-directional housing accelerometers, positions A and C;
- bearing temperatures at two locations and ambient temperature;
- measured and commanded rotating speed, static load, and dynamic load;
- 1.6-second acquisitions approximately every 12 seconds;
- 128 kHz vibration sampling for B01--B09 and 64 kHz for B10--B17;
- sinusoidal dynamic loading for B01--B06 and Gaussian-noise loading for B07--B17;
- stopping thresholds based on housing vibration, \(T_{\rm vibration}\in[6,10]g\), or bearing temperature, \(T_{\rm temperature}=110^\circ{\rm C}\), for most runs; and
- teardown fault labels for inner race, outer race, and rolling elements.

The release is large but divisible. The [machine-readable Zenodo record](https://zenodo.org/api/records/10868257) reports:

\[
152{,}008{,}932{,}259\ {\rm bytes}
\quad (152.01\ {\rm GB},\ 141.57\ {\rm GiB})
\]

across 49 files. B01 is 965,866,495 bytes. B01--B05 together are 9,243,200,290 bytes, which is a practical pilot subset, but B01 had no preset terminal criterion and B05 was deliberately interrupted. They must be treated as atypical or censored rather than silently pooled with threshold-terminated runs.

**Why it fits.** Vibration channels are approximately centered oscillatory signals, and changes in resonance, cross-channel coupling, modulation, and band energy can appear in covariance or cross-spectral structure. The recorded operating variables make it possible to test condition adjustment instead of pretending that speed and load are stationary.

**Why it may defeat the proposed method.** A bearing spall generates intermittent impacts, resonance excitation, and cyclostationary modulation. A zero-lag \(2\times2\) covariance has very little structure and may mostly measure amplitude. Spectral kurtosis and envelope/cyclic methods use precisely the higher-order and periodic information that a plain covariance discards. The proposed method should therefore use a fixed, preregistered lag or filter-bank representation, with every comparator receiving the same representation.

**Ground-truth boundary.** The terminal threshold and teardown fault are physical evidence. There is no labeled first microscopic defect or first detectable incipient fault. Alarm time may be reported relative to shutdown, but it is invalid to call a data-derived change point “ground-truth onset.”

### 3.2 Z24: best full-scale structural application

The [official KU Leuven benchmark page](https://bwk.kuleuven.be/bwm/z24) documents both:

- one year of long-term monitoring with 16 acceleration channels and 48 environmental channels, storing eight averages of 8,192 acceleration samples per hour; and
- a month of progressive damage tests with forced and ambient vibration measurements before and after each applied scenario.

The short-term campaign measured 291 DOF in nine setups with five reference channels. Forced/drop tests contain 65,536 samples at 100 Hz with a 30 Hz anti-alias cutoff. The dated physical schedule runs from an undamaged test on 4 August 1998 through pier settlement, concrete spalling, a landslide, hinge and anchor-head failures, and rupture of 2, 4, and 6 of 16 tendons by 9 September.

This is unusually strong physical ground truth. It is still blockwise: the bridge is measured before and after a completed intervention. The correct claim is “detection across known progressive intervention boundaries,” not observation of damage nucleation in real time.

The environmental problem is scientifically central. Temperature and other operational variables can move modal quantities substantially. The primary Z24 page explicitly lists environmental normalization, operational modal analysis, and damage identification as established uses. A recent primary study on [confounder-adjusted output covariances](https://doi.org/10.1016/j.ymssp.2024.111983) further shows on real bridges that sensor covariance itself changes with temperature. An unconditioned maximum-difference eigenvector could therefore be an excellent temperature detector and a poor damage detector.

Access terms are exact and restrictive:

- non-commercial research only;
- acknowledge the KU Leuven Structural Mechanics Section;
- cite relevant KU Leuven publications;
- do not transfer the data to third parties; and
- submit the [registration form](https://bwk.kuleuven.be/bwm/z24/registration) for review and a temporary account.

The official page does not publish the archive byte size. A third-party mirror with a permissive license would conflict with the no-transfer term and should not be used as the legal source.

### 3.3 NIST linear axis: strongest controlled witness-recovery test

The [NIST metadata record](https://data.nist.gov/od/id/6EF435207EF17114E0532457068155831934) describes a rail raceway intentionally worn over a 10 cm region. The damage zone grows from no degradation at Stage 1 to approximately 75 mm at Stage 15, normally in increments near 5.4 mm. Micrometer profiles provide an external severity measurement.

For each stage, the experiment collected 50 bidirectional IMU runs at 0.02, 0.1, and 0.5 m/s over 322 mm, plus encoder position. It also collected ten bidirectional laser-reference runs at 1 mm spatial increments. The IMU provides:

- triaxial acceleration;
- triaxial angular velocity;
- gyroscope temperature; and
- synchronized axis position.

The laser system supplies three translational and three angular error motions. This makes it possible to ask whether a learned witness points to physically relevant motion components and whether its score tracks independently measured wear, rather than judging only classification accuracy.

The repository exposes 14 daily ZIP archives and a README. Exact sizes in the official metadata sum to:

\[
4{,}423{,}616{,}111\ {\rm bytes}
\quad (4.424\ {\rm GB},\ 4.120\ {\rm GiB}).
\]

The [NIST license statement](https://www.nist.gov/open/license) permits use, modification, derivative works, and distribution with source acknowledgement and change notices, subject to its disclaimers.

**Boundary.** This is an intentional degradation ladder. Stage 2 is a known first intervention, not a naturally arising incipient failure. A valid online emulation presents held-out runs in stage order, but the paper must call it a sequential staged benchmark.

### 3.4 NASA/IMS: useful, compact, but weakly annotated

The [NASA data catalog](https://data.nasa.gov/dataset/ims-bearings) identifies the archive as physical bearing experiments supplied by the University of Cincinnati Center for Intelligent Maintenance Systems. The archive contains three timestamped run-to-failure tests. The accompanying dataset documentation used by the field describes one-second vibration snapshots, conventionally treated as 20,480 samples, acquired at approximately ten-minute intervals, with terminal inner-race, outer-race, or rolling-element faults depending on the test.

The official archive URL is [IMS.zip](https://data.nasa.gov/docs/legacy/IMS.zip). A one-byte HTTP range query returned:

\[
1{,}061{,}902{,}801\ {\rm bytes}
\quad (1.062\ {\rm GB},\ 0.989\ {\rm GiB}).
\]

Two cautions are mandatory:

1. the catalog is sparse and does not provide a certified incipient-onset annotation; and
2. its license field links to U.S. Government Works even though the catalog says the data were supplied by a university center.

The official portal is sufficient for a research download, but the university-origin/license mismatch should be documented before redistributing any raw archive. IMS is appropriate as a legacy external test, not as the only evidence for an online-onset claim.

### 3.5 LANL/SHMTools: a configuration benchmark, not an onset stream

The primary [SHMTools three-story example](https://svn.code.sf.net/p/shmtools/code/Examples/ExampleUsageScripts/threeStoryDataSet.m) describes a real aluminum three-story structure driven by a band-limited random shaker. Channel 1 is input force and Channels 2--5 are floor accelerometers. Signals were originally digitized at 2,560 Hz for 65,536 samples and downsampled to 8,192 samples at 320 Hz, giving 25.6-second records.

There are 17 configurations with 50 tests each in the full release:

- State 1 is the baseline;
- States 2--9 are labeled operational/environmental variability, implemented through mass and stiffness changes;
- States 10--14 introduce impact nonlinearity with bumper gaps from 0.20 to 0.05 mm; and
- States 15--17 combine nonlinearity with mass changes.

The small `data3SS.mat` uses ten records per state. If the array uses MATLAB's default 8-byte double representation, its raw payload is:

\[
8192\times5\times170\times8
=55{,}705{,}600\ {\rm bytes}.
\]

Under the same representation, the full 50-record version has:

\[
8192\times5\times850\times8
=278{,}528{,}000\ {\rm bytes}.
\]

These are uncompressed array payloads, not exact MAT archive sizes.

The [SHMTools copyright file](https://svn.code.sf.net/p/shmtools/code/copyright.txt) contains permissive BSD-3-Clause-like software terms. It does not clearly grant a separate license for redistributing the experimental data. Code reuse is defensible under those terms; raw-data redistribution should wait for confirmation.

Most importantly, each state is a stationary configuration. Concatenating State 1 and State 10 produces a convenient change point but not an observed damage event. LANL remains useful for checking nonlinearity, nuisance variation, and cross-record generalization, not for field-onset claims.

### 3.6 XJTU-SY: technically strong, legally under-specified

The provider's [official repository page](https://github.com/WangBiaoXJTU/xjtu-sy-bearing-datasets) describes complete run-to-failure data from 15 bearings under three operating conditions and says that anyone may use the data to validate prognostic algorithms, with citation of the associated IEEE Transactions on Reliability paper.

That is a clear research-use invitation, but it is not a standard redistribution license. The [GitHub repository metadata](https://api.github.com/repos/WangBiaoXJTU/xjtu-sy-bearing-datasets) reports `license: null`; the repository itself contains only a README and configuration file, while raw data are hosted through external file-sharing services. The official page does not publish an exact archive size.

Recommendation: do not copy the raw archive into this repository or label it MIT/CC. Ask the authors to confirm reuse and redistribution terms, then use it as a multi-bearing validation set. Until then, Paderborn has cleaner legal and technical provenance.

### 3.7 Deliberate exclusions

The conventional CWRU and Paderborn Bearing Data Center fault-classification collections were not ranked as online targets. They are valuable diagnostic benchmarks, but their common tasks compare separately prepared healthy and seeded-fault specimens or stationary operating states; they do not expose a natural fault-onset stream. Randomly splitting their many windows would answer classification, not change detection.

PRONOSTIA/FEMTO and several gearbox challenge collections may contain useful physical degradation sequences, but this bounded audit did not find a primary provider page that simultaneously made the current raw download, exact reuse/redistribution terms, and onset/terminal annotation clear enough to outrank the sources above. They should be added only after a separate rights-and-label audit, not through an unofficial mirror.

## 4. Mathematical fit and novelty boundary

### 4.1 The basic maximum-difference direction

For centered feature vectors \(z\), let \(C_0\) and \(C_1\) be baseline and changed-state covariances. The Euclidean rank-one contrast is

\[
\max_{\lVert w\rVert_2=1}
\left|w^\top(C_1-C_0)w\right|.
\]

Its solution is an eigenvector associated with the eigenvalue of \(C_1-C_0\) having largest magnitude. This is a compact discriminative witness: the scalar energy \((w^\top z)^2\) changes maximally among Euclidean unit directions.

For heterogeneous sensor scales, a more meaningful objective is

\[
\max_{w^\top(C_0+\lambda I)w=1}
\left|w^\top(C_1-C_0)w\right|.
\]

Ignoring regularization for clarity,

\[
\frac{w^\top(C_1-C_0)w}{w^\top C_0w}
=
\frac{w^\top C_1w}{w^\top C_0w}-1.
\]

Therefore the maximizing directions are generalized eigenvectors of \((C_1,C_0)\). This is closely related to common spatial patterns and covariance-ratio discrimination. Whitening improves scale invariance, but it also makes the prior-art boundary sharper: the generalized direction itself is not a new theorem.

### 4.2 Zero-lag covariance is usually insufficient

Structural and machine vibration are time series. For lag \(\ell\),

\[
C_c(\ell)=
\mathbb E\!\left[z_tz_{t-\ell}^{\top}\mid c\right],
\qquad
S_c(f)=
\sum_{\ell=-L}^{L} C_c(\ell)e^{-i2\pi f\ell},
\]

where \(S_c(f)\) is a cross-spectral density matrix. A loss of stiffness, resonance shift, coupling change, or bearing impact can be much clearer in \(C_c(\ell)\) or \(S_c(f)\) than in \(C_c(0)\).

Two defensible extensions are:

1. form a fixed lag embedding

   \[
   y_t=[z_t^\top,z_{t-1}^\top,\ldots,z_{t-L}^\top]^\top
   \]

   and learn a regularized covariance contrast on \(y_t\); or

2. estimate cross-spectral matrices in preregistered bands and learn a complex generalized eigen-witness per band.

For bearings, speed should be used for order tracking or conditioning before the contrast is learned. Otherwise, the largest “damage” direction may simply identify the current RPM.

### 4.3 Operational conditioning is part of the null

Let \(u_t\) contain speed, load, direction, temperature, or other operating variables. The relevant comparison is not generally

\[
C_1-C_0,
\]

but

\[
\Delta C(u)
=
C_{\rm candidate}(u)-C_{\rm baseline}(u).
\]

Practical implementations may stratify by operating regime, regress mean and covariance on \(u\), or use matched baseline neighbors. Every baseline must receive the same operating variables. Supplying speed and temperature only to the proposed method would be an information-budget advantage, not an algorithmic advantage.

The null is therefore:

\[
H_0:\quad
C_t(\ell\mid u_t)=C_0(\ell\mid u_t),
\]

not “the unconditional covariance never changes.” This distinction is essential on Z24 and Paderborn.

### 4.4 Rank one is not the Gaussian oracle

If two centered Gaussian states have full covariances \(C_0\) and \(C_1\), the log-likelihood ratio contains

\[
\frac12 z^\top(C_0^{-1}-C_1^{-1})z
+\frac12\log\frac{\det C_0}{\det C_1}.
\]

It uses all changed eigenmodes and a log-determinant term. A single energy \((w^\top z)^2\) is generally a lossy rank-one approximation. It can be near-optimal only under additional structure, such as a dominant rank-one covariance change or a strict sensor/compute budget.

Consequently, the method must be compared with a full regularized covariance GLR or KL statistic. It may claim compactness or robustness if supported, but it may not claim statistical optimality in general.

### 4.5 Adaptive direction search inflates the score

Learning \(w_t\) on a candidate window and scoring the same samples selects the largest noise eigenvalue even under the null. Repeating this at every time creates an additional optional-selection bias. A valid experiment must use one of:

- cross-fitting: learn \(w_t\) on one block and score a disjoint future block;
- a frozen witness learned only from other physical runs;
- a null block bootstrap that repeats the complete adaptive search and sequential stopping rule; or
- a proved anytime-valid construction whose conditional supermartingale assumptions include the adaptive witness.

A pointwise permutation \(p\)-value is not automatically an event-level online false-alarm guarantee.

## 5. Strong baselines that cannot be omitted

### 5.1 Structural vibration

| Family | Required comparator | Why it is strong |
|---|---|---|
| Simple surveillance | RMS/variance, bandpower, peak and kurtosis CUSUM | Exposes whether the witness only rediscovers amplitude |
| Modal analysis | Frequency-domain decomposition and covariance-driven/reference-based stochastic subspace identification | Natural frequencies, damping, and mode shapes are established Z24/OMA features; lag covariance is already central |
| Time-series residual | AR/ARX coefficients or prediction residuals plus Mahalanobis/CUSUM | Standard LANL and sequential-SHM route; captures dynamics beyond zero lag |
| Environmental normalization | Temperature/load regression, cointegration or conditional covariance before damage scoring | Prevents nuisance changes from becoming false damage |
| Full covariance | Regularized Gaussian GLR/KL, Frobenius/operator norm, and log-Euclidean or affine-invariant SPD distance | Tests whether the rank-one witness adds anything over the whole covariance |
| Nonparametric change | Energy distance or MMD scan, with the same window and calibration | Detects changes beyond second moments |
| Same-feature classifier | Logistic regression, linear SVM, and a scalar threshold on the learned witness | Separates feature value from classifier branding |

The primary Z24 source already identifies operational modal analysis, environmental normalization, and damage-identification methods as established benchmark uses. A recent [sequential SHM change-point method](https://arxiv.org/abs/1812.02824) also demonstrates that unknown post-damage distributions and delay/false-alarm trade-offs are existing parts of the field.

### 5.2 Bearings and rotating machinery

| Family | Required comparator | Why it is strong |
|---|---|---|
| Marginal health indicators | RMS, standard deviation, peak-to-peak, crest factor, skewness and kurtosis | Very difficult to beat near terminal failure; catches amplitude leakage |
| Spectrum | Welch PSD/bandpower and order-tracked spectrum | Directly tracks resonance and speed-related components |
| Envelope/cyclostationary | Band-pass Hilbert envelope spectrum, cyclic spectral coherence, bearing characteristic frequencies where geometry permits | Purpose-built for repeated bearing impacts |
| Transient selection | Spectral kurtosis and fast kurtogram | Uses higher-order non-Gaussian transients that covariance misses |
| Predictive residual | AR residual energy and change detector | Low-cost online dynamic baseline |
| Covariance geometry | Full covariance GLR and an SPD/log-Euclidean classifier | Closest prior family to the proposed representation |
| Classical learning | One-class SVM/isolation forest and supervised logistic/gradient boosting on the identical feature bank | Strong tabular controls |
| Deep sequence | A modest 1-D CNN/autoencoder and, only if enough independent bearings exist, an LSTM/transformer | Prevents comparison only with weak classical methods; report its larger training/compute budget separately |

The prior-art floor is high. [Antoni and Randall's spectral-kurtosis work](https://doi.org/10.1016/j.ymssp.2004.09.002) introduced the kurtogram for detecting and filtering transient rotating-machine faults; the [bearing-diagnostics tutorial by Randall and Antoni](https://doi.org/10.1016/j.ymssp.2010.07.017) covers envelope and cyclostationary diagnostics. Covariance-manifold bearing classifiers also already exist, for example the [statistical-enhanced covariance/log-Euclidean method](https://doi.org/10.1016/j.isatra.2020.11.018). These methods must be treated as foundations and competitors, not renamed as new discoveries.

## 6. Executable same-budget experiments

### 6.1 Common resource contract

Before running any model, lock:

- identical physical channels and raw acquisition blocks;
- identical downsampling, filtering, order tracking, and missing-data policy;
- identical chronological calibration duration;
- identical labeled training bearings or intervention blocks;
- identical candidate-window length and alarm opportunities;
- hyperparameters chosen only from training runs or baseline resampling;
- thresholds calibrated without the evaluated event;
- compute, memory, parameter count, and latency reported rather than conflated with accuracy; and
- confidence intervals resampled by independent bearing/run/intervention, never by overlapping windows.

All overlapping windows from one physical run stay in the same split. Random window-level train/test splitting is prohibited.

### 6.2 Track A: Paderborn natural-failure study

Use two tasks because the dataset does not contain an onset label.

#### Task A1: held-out-bearing failure-horizon warning

For a horizon \(H\), define the target from the recorded terminal time:

\[
y_t(H)=
\begin{cases}
1,&0<T_{\rm end}-t\le H,\\
0,&T_{\rm end}-t>H+G,
\end{cases}
\]

where \(G\) is a preregistered guard interval. Train a witness and classifier on complete training bearings and deploy it prospectively on an unseen bearing. Use group leave-one-bearing-out splits and treat deliberately interrupted or no-threshold runs as censored.

Report event-level AUROC/AUPRC, warning lead time, missed-event rate, and false warnings per operating hour. Evaluate several fixed horizons selected before seeing test results. This is a real impending-failure task, not an onset-detection claim.

#### Task A2: label-free sequential change alarm

Use a short, fixed initial calibration interval from each new bearing, justified by the provider's statement that defects were not seeded. Then compare rolling condition-adjusted features with the baseline. The exact healthy duration remains unknown, so report:

- alarm time relative to terminal shutdown;
- proportion of bearings alarmed before shutdown at a common threshold;
- alarms during the initial calibration/holdout portion;
- stability under load/speed strata; and
- sensitivity to calibration length.

Do not compute “detection delay from onset.” An internally estimated change point cannot also serve as its own ground truth.

#### Representation ablation

At minimum compare:

1. raw two-channel zero-lag covariance;
2. per-window demeaned covariance;
3. correlation after marginal variance normalization;
4. fixed lag covariance;
5. cross-spectral/filter-bank covariance;
6. higher-order transient features; and
7. marginal plus covariance features.

This reveals whether any gain comes from coupling structure, temporal structure, or simply amplitude.

### 6.3 Track B: NIST staged-damage study

Use Stage 1 only for baseline calibration. Within each speed and direction, cross-fit the 50 repeated runs: one fold learns or updates the witness and a disjoint fold is scored. Present held-out runs in stage order.

Primary endpoints:

- detection at the first intervention, Stage 2, at a threshold calibrated on Stage-1 repetitions;
- score monotonicity and uncertainty across Stages 1--15;
- correlation with micrometer wear profile and laser-derived geometric error;
- generalization to held-out speed and direction; and
- witness stability across cross-fit folds.

Key comparators are RMS/bandpower, full covariance GLR, AR residual, logistic/threshold on the same projected energy, and models using all six IMU channels. Position-gated features can test whether the witness localizes the known damaged rail region, but localization must be evaluated against encoder/laser position, not inferred from the witness alone.

Passing NIST means the method detects controlled wear robustly. It does not establish natural-failure performance.

### 6.4 Track C: Z24 civil-structure validation

First obtain data through the official registration route. Keep the long-term and progressive-test sensor geometries separate unless a documented channel mapping supports fusion.

For the progressive campaign:

- use the five reference channels or setup-blocked statistics to avoid treating roving sensor position as damage;
- learn thresholds only from undamaged/new-reference repetitions;
- score each known intervention chronologically;
- compare forced and ambient excitation separately; and
- report delay in measurement blocks from a completed intervention, not wall-clock damage onset.

For long-term monitoring:

- fit temperature/environment conditioning on an early chronological period;
- use later undamaged periods to estimate false alarms across seasons;
- compare conditional covariance against modal-frequency/SSI baselines; and
- do not use the known demolition campaign to tune the alarm threshold.

This track is the best test of whether the method sees physical structural dynamics rather than bearing-specific impulses.

## 7. Advantage gate and claim boundaries

### 7.1 What would demonstrate a narrow advantage?

The result supports an advantage only if, on held-out physical runs:

1. thresholds are matched to the same event-level false-alarm exposure;
2. the proposed detector improves warning lead time, delay, or missed-event rate over full covariance GLR, marginal CUSUM, AR/modal methods, and domain signal-processing baselines;
3. a threshold or logistic model on the identical learned scalar does not reproduce the claimed classifier gain;
4. the result survives speed/load/temperature conditioning;
5. the gain is not caused by more sensors, longer history, damage labels, or hyperparameter trials;
6. uncertainty is computed over bearings or interventions; and
7. runtime or parameter-efficiency claims are backed by measured resources.

A particularly meaningful positive result would be:

> Under a fixed calibration, sensor, and false-alarm budget, a cross-fitted rank-\(k\) condition-adjusted cross-spectral witness improves held-out-bearing failure warning over scalar, full-covariance, and specialist spectral baselines while using a compact interpretable score.

Even this claim is benchmark- and protocol-specific.

### 7.2 Claims that remain prohibited

This audit does not support claims of:

- superiority to an oracle likelihood-ratio detector;
- general superiority to spectral kurtosis, envelope analysis, OMA/SSI, or full covariance GLR;
- a new generalized-eigenvalue, CSP, Wilson-loop, topological, quantum, or string-theory theorem;
- quantum acceleration or quantum advantage;
- universal sample efficiency or scalability;
- exact incipient-fault onset on Paderborn, NASA/IMS, or XJTU-SY;
- natural online degradation for NIST, Z24, or LANL configuration concatenations;
- physical localization unless independently checked against position, metrology, or known damage location;
- privacy, decentralization, or edge efficiency without a concrete protocol and measured communication/compute cost; or
- real-time operation without measured end-to-end latency on the target hardware.

### 7.3 Go/no-go rule

**Go to a full paper** only if Paderborn shows a held-out-bearing benefit at matched false-alarm rate and NIST or Z24 supplies an independent mechanism check. A full paper should include either a genuine theoretical contribution in adaptive calibration/structured low-rank change detection or a clearly framed application contribution; generalized eigenvectors alone are insufficient novelty.

**Stop or reframe** if:

- gains disappear after order/speed/load conditioning;
- envelope or spectral-kurtosis baselines dominate;
- the learned direction is unstable across bearings;
- only random-window splits look positive;
- only terminal high-amplitude data are detected; or
- a scalar threshold/logistic model on the same feature matches the proposed classifier.

In that case, publishable value may remain in an adversarial benchmark and no-go certificate, but not in a broad algorithmic-superiority paper.

## 8. Minimal publication sequence

1. **Controlled smoke test:** NIST Stages 1--15, cross-fitted, all six IMU channels, speed/direction holdouts, and metrology correlation.
2. **Bounded natural pilot:** download B01 only to validate parsing, then B02--B04 for threshold-terminated experiments. Do not use B01 as a successful endpoint example merely because it is smallest.
3. **Locked bearing study:** preregister feature banks, horizons, calibration windows, split units, thresholds, and baselines; then expand to all usable Paderborn runs.
4. **Independent structural validation:** obtain Z24 officially and test the same mathematical witness with environment-conditioned modal/cross-spectral inputs.
5. **Legacy replication:** NASA/IMS, with all onset claims removed.

The strongest venue fit would be an algorithm-and-signal-processing venue such as *Mechanical Systems and Signal Processing*, *Structural Health Monitoring*, IEEE Transactions on Instrumentation and Measurement, or a PHM conference. The manuscript should be led by the monitored engineering problem and a precise statistical contribution, not by a loose quantum analogy.

## 9. Primary-source ledger

### Data and licenses

- KU Leuven Structural Mechanics, [Z24 Bridge benchmark](https://bwk.kuleuven.be/bwm/z24) and [registration terms](https://bwk.kuleuven.be/bwm/z24/registration).
- Aimiyekagbon, [Run-to-failure data set of ball bearings subjected to time-varying load and speed conditions](https://zenodo.org/records/10868257), DOI 10.5281/zenodo.10868257; exact sizes and CC BY 4.0 identifier from the [Zenodo API record](https://zenodo.org/api/records/10868257).
- NIST, [Linear Axis Testbed -- Rail Degradation Experiment 01 metadata](https://data.nist.gov/od/id/6EF435207EF17114E0532457068155831934), DOI 10.18434/T4/1502585, and [NIST data-use terms](https://www.nist.gov/open/license).
- NASA Open Data Portal, [IMS Bearings](https://data.nasa.gov/dataset/ims-bearings) and official [IMS archive](https://data.nasa.gov/docs/legacy/IMS.zip).
- Los Alamos National Laboratory, [Structural Health Monitoring Algorithm Comparisons Using Standard Data Sets, LA-14393](https://www.osti.gov/servlets/purl/961604); [SHMTools project](https://sourceforge.net/projects/shmtools/), [three-story primary example](https://svn.code.sf.net/p/shmtools/code/Examples/ExampleUsageScripts/threeStoryDataSet.m), and [software terms](https://svn.code.sf.net/p/shmtools/code/copyright.txt).
- Wang et al., [XJTU-SY provider repository](https://github.com/WangBiaoXJTU/xjtu-sy-bearing-datasets), associated paper DOI [10.1109/TR.2018.2882682](https://doi.org/10.1109/TR.2018.2882682), and [repository license metadata](https://api.github.com/repos/WangBiaoXJTU/xjtu-sy-bearing-datasets).

### Methodological boundaries

- Peeters and De Roeck, [One-year monitoring of the Z24 bridge: environmental effects versus damage events](https://doi.org/10.1002/1096-9845%28200102%2930%3A2%3C149%3A%3AAID-EQE1%3E3.0.CO%3B2-Z).
- Reynders and De Roeck, [Reference-based combined deterministic-stochastic subspace identification](https://doi.org/10.1016/j.ymssp.2007.09.004).
- Neumann et al., [Confounder-adjusted covariances of system outputs and applications to structural health monitoring](https://doi.org/10.1016/j.ymssp.2024.111983).
- Liao et al., [Structural damage detection and localization with unknown post-damage feature distribution using sequential change-point detection](https://arxiv.org/abs/1812.02824).
- Antoni, [The spectral kurtosis: a useful tool for characterising non-stationary signals](https://doi.org/10.1016/j.ymssp.2004.09.001).
- Antoni and Randall, [The spectral kurtosis: application to vibratory surveillance and diagnostics of rotating machines](https://doi.org/10.1016/j.ymssp.2004.09.002).
- Randall and Antoni, [Rolling element bearing diagnostics -- a tutorial](https://doi.org/10.1016/j.ymssp.2010.07.017).
- Li et al., [A bearing fault diagnosis scheme with statistical-enhanced covariance matrix and a Log-Euclidean Riemannian classifier](https://doi.org/10.1016/j.isatra.2020.11.018).
