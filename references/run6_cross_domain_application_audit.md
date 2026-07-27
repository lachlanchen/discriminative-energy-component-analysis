# Run 6 cross-domain application audit

**Date:** 2026-07-27
**Question:** Where is an unknown translated or localized change in covariance/correlation genuinely the sequential signal, and where could the Run 4/5 witness-bank idea plausibly add value over strong baselines?

## Executive decision

Run 6 should begin with the **UCI dynamic gas-mixture sensor array**, not with another simulated lattice and not with a medical claim. It provides two uninterrupted, real, approximately 12-hour streams from 16 physical sensors sampled at 100 Hz. Gas-concentration set points change at random times separated by 80--120 seconds, and the set points are recorded sample by sample. This is unusually close to the desired problem: an online algorithm sees only the sensor array, does not know when or which component changes, and must alarm quickly under a fixed false-alarm budget.

The first experiment must nevertheless be adversarial to our own hypothesis. Gas transitions strongly affect channel means. The study is only evidence for a correlation-based advantage if the proposed score adds value beyond scalar and multivariate mean detectors, using the same samples and calibration budget. The decisive ablation is therefore:

1. raw sensor channels;
2. per-window demeaned channels;
3. per-window demeaned and variance-normalized channels;
4. marginal features only;
5. correlation features only;
6. marginal plus correlation features.

If the proposed method wins only because it indirectly recovers a mean or amplitude change, that is not a distinctive ECA/correlation result.

The recommended application order is:

1. **Dynamic chemical-sensor monitoring** -- best combination of continuous real data, controlled ground truth, safety relevance, and a plausible label-light advantage.
2. **EEG seizure-onset detection** -- strongest natural unknown-onset dataset and highest human impact, but medically high-stakes and already served by strong specialist models.
3. **Robot tactile contact/slip** -- the cleanest approximate two-dimensional translation symmetry and good onset labels.
4. **Earthquake P-arrival detection** -- excellent physical labels and impact, but the standard public dataset is event-windowed and amplitude/deep-learning baselines are very strong.
5. **Optical speckle/wavefront monitoring** -- strong correlation physics, but available labels mostly come from controlled ramps or have no change-point annotation.
6. **NMR reaction monitoring** -- meaningful chemical-shift nuisance and real sequences, but very few independent reaction trajectories and predominantly first-order spectral changes.
7. **Surface-code drift on public Google hardware data** -- best mathematical match to lattice-localized witnesses, but no naturally labeled drift onset; it remains an important continuity benchmark rather than the best first real application.

This ranking is a research-priority decision, not an experimental result and not evidence that the current algorithm already has an advantage.

## 1. What problem is being transferred?

Let \(X_t\) denote a multichannel observation or a short window ending at time \(t\). A useful common model is

\[
H_0:\quad \Sigma_t=\Sigma_0,
\qquad
H_1:\quad
\Sigma_t=\Sigma_0+P_g\Delta P_g^\top,\quad t\ge \nu,
\]

where:

- \(\nu\) is an unknown change time;
- \(\Delta\) is a structured local change;
- \(g\) is an unknown location, channel relabeling, graph neighborhood, time shift, or other nuisance transformation;
- \(P_g\) is the corresponding action on features;
- \(\Sigma_t\) is a covariance, correlation, coherence, or other second-order observable.

A generic localized witness bank has a score such as

\[
Z_t
=
\max_{g\in\mathcal G}
\left\langle
W,\,
P_g^\top(\widehat{\Sigma}_t-\widehat{\Sigma}_0)P_g
\right\rangle ,
\]

followed by an online detector, for example

\[
C_t=\max\{0,C_{t-1}+Z_t-\kappa\},
\qquad
\tau=\inf\{t:C_t\ge h\}.
\]

The max over \(g\) is useful only when the nuisance family is physically justified. It also creates a multiple-search penalty, which must be included when calibrating \(h\). Cyclic translation is appropriate on a periodic lattice, approximate two-dimensional translation may be appropriate in a tactile image, while EEG electrodes require a graph-local scan rather than a fictitious cyclic shift.

### What would count as an algorithmic advantage?

An advantage is established only if all of the following hold:

- **Equal surveillance budget:** the same samples, history length, calibration period, and alarm opportunities are available to every detector.
- **Equal false-alarm operating point:** compare delay or sensitivity only after thresholds have been independently calibrated to the same average run length (ARL), false alarms per hour, or false alarms per million cycles.
- **Strong same-information controls:** at minimum include a scalar/rate CUSUM, a multivariate mean GLR or Hotelling detector, a covariance/correlation detector, and logistic/threshold models on exactly the same feature bank.
- **Domain controls:** compare with a credible specialist method, such as decoder likelihood for QEC, Sparsh features for tactile sensing, or a patient-specific seizure detector.
- **Time-respecting evaluation:** no random split of overlapping windows and no fitting a threshold on the test stream.
- **Incremental mechanism test:** report whether correlation information improves on marginals. A gain from more features, more labels, or knowledge of the true changed location is not an ECA advantage.
- **Uncertainty:** confidence intervals must resample independent subjects, physical runs, transitions, or episodes rather than treating adjacent frames as independent.

The defensible claim would be narrow: *under a specified calibration and compute budget, the structured max-witness detector improves the delay/false-alarm trade-off for a stated family of translated or localized changes*. It would not establish universal sample efficiency, computational superiority, quantum advantage, or superiority to an oracle that knows the changed feature and its location.

## 2. Ranking rubric

Each score is from 0 (poor) to 5 (excellent):

- **D -- data availability:** public access, size, documentation, and usable license;
- **G -- ground truth:** trustworthy timing and identity of the physical change;
- **I -- safety/operational impact:** consequence of earlier reliable detection;
- **P -- plausible advantage:** chance of improving a strong baseline for the specific structured-correlation hypothesis.

Because the purpose is to find an honest algorithmic win, the priority index is

\[
\text{priority}=D+G+I+2P.
\]

The scores are prospective judgments and must not appear in a paper as measured performance.

| Rank | Domain and primary real dataset | Real-data status | Symmetry/locality fit | D | G | I | P | Priority | Decision |
|---:|---|---|---|---:|---:|---:|---:|---:|---|
| 1 | UCI dynamic gas mixtures | Two uninterrupted controlled physical experiments; random real set-point transitions | Medium: repeated sensor types and unknown affected subset, not an exact spatial translation | 5 | 4 | 5 | 4 | 22 | **Run first** |
| 2 | CHB-MIT scalp EEG | Continuous clinical recordings with naturally occurring annotated seizures | Medium-high: unknown graph-local electrode/coherence change | 5 | 5 | 5 | 3 | 21 | Strong second application; research-only medical framing |
| 3 | Touch and Go / RH20T tactile data | Real contact sequences; Touch and Go has labeled touch onsets | High: approximate 2-D translation of a local contact field | 4 | 4 | 4 | 3 | Best robotics test |
| 4 | STEAD earthquake waveforms | Real earthquakes with P/S picks, but distributed as selected 60-s event/noise windows | Medium: unknown time shift; only three components spatially | 5 | 5 | 5 | 1 | Stress test, not first advantage target |
| 5 | Fiber-speckle temperature / AO telemetry | Real optical acquisitions; controlled temperature ramps or unlabeled telescope telemetry | Medium: strong correlation physics, but often global decorrelation | 4 | 3 | 3 | 2 | Good physics validation after gas/tactile |
| 6 | Time-resolved NMR reactions | Real spectra indexed by time; reaction start is controlled | High for chemical-shift translation, low for covariance-specific signal | 4 | 2 | 4 | 2 | Exploratory chemistry study |
| 7 | Google surface-code hardware data | Real processor measurements, but no labeled operational drift event | Very high: local lattice translations and parity correlations | 4 | 2 | 3 | 2 | Preserve as theory/QEC continuity benchmark |

The tension in this table is informative. QEC has the cleanest mathematical group action, but the gas and EEG datasets provide much more credible sequential ground truth. Conversely, STEAD has excellent labels and impact, but it is unlikely that a correlation-only method will beat mature amplitude-aware phase pickers.

## 3. Exact public real-data resources and licenses

| Domain | Dataset and official source | What is actually available | License and access note |
|---|---|---|---|
| Dynamic gas sensing | [Gas sensor array under dynamic gas mixtures, UCI; DOI 10.24432/C5WP4C](https://archive.ics.uci.edu/dataset/322/gas) | 4,178,504 rows; two approximately 12-h uninterrupted streams; time, two concentration set points, and 16 sensor readings; 100 Hz; 351.9 MB compressed | Current UCI metadata says **CC BY 4.0**. The older free-text description also says research-only and excludes commercial use. Use is clear for this research benchmark; obtain clarification before commercial reuse. |
| Long-term gas drift | [Gas Sensor Array Drift at Different Concentrations, UCI; DOI 10.24432/C5MK6M](https://archive.ics.uci.edu/dataset/270/gas%2Bsensor%2Barray%2Bdrift%2Bdataset%2Bat%2Bdifferent%2Bconcentrations) | 13,910 physical exposures, 16 sensors, 128 aggregate features, six gases over 36 months, organized into ten batches | Current UCI metadata says **CC BY 4.0**, while legacy text again says research-only/non-commercial. Batches merge nonconsecutive months, so this is a drift/domain-adaptation set, not a fine-grained online stream. |
| EEG | [CHB-MIT Scalp EEG Database, PhysioNet; DOI 10.13026/C2K01R](https://physionet.org/content/chbmit/1.0.0/) | 664 EDF files, 129 with seizures, 198 total annotated seizures; 22 pediatric subjects in 23 cases; mostly 23 channels at 256 Hz; 42.6 GB | **Open Data Commons Attribution 1.0**; open access subject to attribution terms. |
| Tactile contact | [Touch and Go official project](https://touch-and-go.github.io/) and [NeurIPS paper/datasheet](https://proceedings.neurips.cc/paper_files/paper/2022/file/354892587fe39b17c2b727af02abff4a-Paper-Datasets_and_Benchmarks.pdf) | Paired egocentric RGB and GelSight videos from real human probing; frame timestamps, material labels, and identified touch-onset frames | Dataset datasheet states **CC BY**; the official project is presented under **CC BY 4.0**. |
| Robot manipulation | [RH20T official dataset](https://rh20t.github.io/) | More than 110,000 real contact-rich sequences; RGB-D, 6-DoF force/torque at 100 Hz, actions, audio, and fingertip tactile at 200 Hz for configuration 7; reduced release is multi-terabyte | `scene_0001`--`scene_0005`: **CC BY-SA 4.0** (RH20T-C). `scene_0006`--`scene_0010`: **CC BY-NC 4.0** (RH20T-NC). Faces/voices may occur and require careful handling. |
| Earthquakes | [STEAD official repository](https://github.com/smousavi05/STEAD) | Approximately 85 GB merged; chunks of roughly 200,000 three-component waveforms; local-earthquake/noise labels and P-arrival, S-arrival, and coda-end metadata; 100 Hz | **CC BY 4.0**. The official repository warns that some back azimuths are misplaced and should be recomputed. |
| QEC hardware | [Google Quantum AI surface/repetition-code data; DOI 10.5281/zenodo.13273331](https://zenodo.org/records/13273331) | Four archives totaling 112.5 GB, including 105-qubit surface-code \(d=3,5,7\) data and 72-qubit surface/repetition-code data; detection events, measurements, observable flips, circuits, metadata, and decoder artifacts | **CC BY 4.0** in Zenodo metadata. These are real processor experiments, not a naturally labeled drift stream. |
| Fiber speckle | [Specklegram Temperature Dataset; DOI 10.17605/OSF.IO/8NXVK](https://doi.org/10.17605/OSF.IO/8NXVK) and [official data paper](https://www.mdpi.com/2306-5729/10/4/44) | 24,528 real TIFF specklegrams in 14 acquisition sets; 25--200 °C, approximately 0.175 °C increments; heating/cooling identity in filenames; 633-nm source and 62.5-\(\mu\)m-core multimode fiber | **CC BY-SA 4.0** according to the data paper. |
| Adaptive optics | [AOT proof-of-concept telemetry; DOI 10.5281/zenodo.8192742](https://zenodo.org/records/8192742) | 1.6 GB of real FITS telemetry from CIAO, ERIS, GALACSI, NAOMI, and PAPYRUS, including Shack-Hartmann and pyramid wavefront sensors | **CC BY 4.0**. There are no authoritative change-onset labels. |
| NMR reaction monitoring | [NMR reaction monitoring robust to spectral distortions; DOI 10.5281/zenodo.14814657](https://zenodo.org/records/14814657) | 2.3 GB; time-indexed raw and preprocessed spectra for sucrose hydrolysis and pentene/hexene hydrosilylation | **CC BY 4.0**. |
| NMR click reactions | [ShimNetV2-RM reaction-monitoring data; DOI 10.5281/zenodo.18474776](https://zenodo.org/records/18474776) | 749.4 MB; two real click reactions on an Agilent 600-MHz system; one scan every 20 s for 5 h, with well-shimmed before/after spectra | **CC BY 4.0**. |

## 4. Domain-specific audit and executable tests

### 4.1 Dynamic gas mixtures -- recommended Run 6A

#### Why this is the best first test

The [UCI dynamic-mixture page](https://archive.ics.uci.edu/dataset/322/gas) states that concentration transitions occur at random times in the interval 80--120 s and include increases, decreases, and setting one gas to zero while the other is fixed. It also states that response magnitudes were deliberately made similar enough that mixtures cannot be identified simply from magnitude. Four copies of each of four Figaro sensor types provide a meaningful repeated-channel structure.

This does not give an exact translation group. A defensible symmetry is permutation or pooling within nominally identical sensor types, plus a max over unknown sensor subsets/templates. The group must be learned or declared from hardware type labels; arbitrary cyclic shifts over the 16 listed channels have no physical justification.

#### Locked task

- Hide the two gas-concentration columns from every detector.
- Treat a change in either set-point column as an event at \(\nu_j\).
- Build one-second windows, updated every 0.1 or 1 s, from the 16 resistance/conductivity channels.
- Exclude a predeclared post-transition settling interval only from false-alarm exposure, not from delay measurement.
- Calibrate all thresholds using pre-transition material in the calibration split.
- Evaluate within each stream by forward chaining, then evaluate cross-mixture transfer: ethylene/methane to ethylene/CO and the reverse.
- Never randomly mix overlapping windows between train and test.

The recorded concentration is a commanded set point, not a direct chamber concentration measurement. Detection delay therefore includes gas-delivery and sensor response lag. The paper must call it delay relative to commanded transition, not an exact molecular arrival time.

#### Required baselines

1. max single-channel absolute-difference threshold;
2. max single-channel CUSUM;
3. multivariate mean Hotelling \(T^2\) or Gaussian mean GLR;
4. covariance GLR or Frobenius covariance-distance CUSUM;
5. PCA residual/SPE detector;
6. logistic or shallow supervised threshold on the identical marginal/correlation feature bank;
7. a strong temporal model, ideally the reservoir-computing approach associated with the dataset or a compact TCN, with the same causal history.

#### Primary metrics

- event sensitivity within 10, 30, and 60 s of a set-point transition;
- median and 90th-percentile detection delay in seconds;
- false alarms per hour and mean time between false alarms;
- time in alarm;
- area under the event-level false-alarm/delay curve;
- runtime, peak memory, and causal history length;
- performance stratified by increase/decrease/return-to-zero, gas identity, and concentration step size.

Use a transition-cluster bootstrap, not a frame bootstrap. The primary comparison should be the delay difference at a predeclared false-alarm rate, with confidence intervals over transition events and a sensitivity analysis that blocks adjacent transitions.

#### Publishable advantage criterion

Proceed to an algorithm paper only if the max-witness method:

- improves delay at the same false alarms/hour against the best same-information baseline;
- retains a gain after demeaning and variance normalization;
- transfers across the two gas-mixture experiments or clearly identifies why it cannot;
- does not obtain the gain solely from searching more templates without paying the corresponding threshold penalty.

If it loses to a simple mean CUSUM, that is a useful negative result: this dataset does not validate a second-order mechanism.

### 4.2 CHB-MIT EEG -- recommended high-impact Run 6B

#### Real-data status and fit

CHB-MIT contains long clinical recordings with seizures that occur without the algorithm knowing their time. Onset and end annotations are given to the second. This is the strongest natural unknown-onset source in the audit.

The physically sensible structure is an electrode graph derived from the International 10--20 montage. A seizure may produce a local change in band-limited covariance, coherence, or phase-locking that spreads over this graph. Cyclic translation over channel number would be wrong. A graph-local max over candidate neighborhoods is defensible, with montage differences handled explicitly.

Seizures also produce changes in power and waveform amplitude. Therefore, a covariance-based result requires the same marginal-versus-correlation ladder used for gas data.

#### Evaluation design

- Patient-specific: leave one seizure out, fit/calibrate only on earlier recordings from the same patient, and evaluate on all intervening interictal time.
- Patient-independent: leave one subject out, with no windows from that subject used for feature learning or threshold selection.
- Respect file gaps documented by PhysioNet; do not silently concatenate gaps as if they were observed normal EEG.
- Fix a causal window and update rate before opening the test subjects.
- Report results by subject and aggregate with subject-level uncertainty.

#### Strong baselines

- line length, energy, variance, and band-power CUSUMs;
- multichannel mean/covariance GLR;
- Riemannian covariance-distance detector;
- the patient-specific SVM lineage cited by the [PhysioNet dataset page](https://physionet.org/content/chbmit/1.0.0/);
- a causal CNN/TCN or transformer baseline with explicit parameter count and latency.

#### Metrics

- event sensitivity;
- false alarms per hour over all available interictal recording time;
- onset latency in seconds relative to the clinical annotation;
- fraction of seizures detected before 5, 10, and 30 s after onset;
- time in warning;
- per-subject miss and false-alarm distributions;
- compute, memory, and number of patient-specific labeled seizures required.

This can support a research claim, not a clinical-device claim. Clinical utility, treatment timing, and safety require prospective validation that this retrospective dataset cannot provide.

### 4.3 Robot tactile contact and slip -- recommended Run 6C

#### Real-data status and fit

[Touch and Go](https://touch-and-go.github.io/) provides real paired RGB/GelSight videos and explicitly identifies touch-onset frames. A local contact patch moves approximately by translation over the tactile image as the user probes different positions. Boundary effects, illumination, marker layout, shear, and elastomer deformation make this only an approximate group.

[RH20T](https://rh20t.github.io/) provides a much broader real-robot validation set with synchronized force/torque and tactile data, but contact onset is not directly annotated. A force/torque threshold can create a reference label, which must be described as a **derived proxy**, not ground truth. RH20T is also very large and has privacy-sensitive human video/audio, so Touch and Go is the practical first dataset.

The [Sparsh official repository](https://github.com/facebookresearch/sparsh) supplies a strong representation baseline and real force/slip datasets measured with an ATI Nano17 sensor. Its collection page should be checked for the exact per-dataset license before redistributing derived data; the model/repository license must not be assumed to license every upstream dataset.

#### Tasks

1. **Touch onset:** alarm from GelSight frames only, then compare against the annotated onset.
2. **Slip onset:** use a real force/slip-labeled trajectory dataset if its license is verified.
3. **Cross-session robustness:** train/calibrate on collectors or days and test on held-out collectors/days.
4. **Approximate translation test:** compare a fixed-location witness, a max-translated witness, and a small translation-equivariant CNN under the same causal window.

#### Strong baselines

- frame difference, SSIM, normalized cross-correlation, and optical flow;
- mean deformation, marker displacement, and variance thresholds;
- logistic/Hotelling detectors on the identical witness bank;
- Sparsh embeddings with a causal linear or temporal head;
- force/torque threshold where force is available, labeled as a privileged sensor baseline.

#### Metrics

- contact/slip event sensitivity;
- false alarms per minute;
- median and 90th-percentile delay in frames and milliseconds;
- time in alarm;
- performance by material, contact direction, session, and sensor;
- inference latency on a declared robot CPU/GPU and memory footprint.

A good result would be a better false-alarm/delay trade-off under sensor/session shift, not merely higher offline frame accuracy.

### 4.4 STEAD earthquake P-arrivals -- valuable stress test, weak first target

[STEAD](https://github.com/smousavi05/STEAD) contains real three-component seismograms and physical P/S arrival picks. Unknown time translation is genuine: an online detector does not know when an arrival occurs. However, STEAD distributes preselected 60-s earthquake and noise windows rather than continuous station chronology. It can test pick accuracy and within-window false positives, but it cannot by itself establish an operational false-alarm rate per station-day without reconstructing continuous data from another archive.

The correlation structure across E/N/Z components can change at an arrival, but the marginal amplitude and energy change is often much stronger. The candidate method is therefore unlikely to beat mature methods on raw accuracy. The honest niche would be low-amplitude arrivals, station transfer, or a very small causal compute budget.

Required controls include STA/LTA, three-component energy/CUSUM, covariance GLR, PhaseNet-like picking, and [EQTransformer](https://github.com/smousavi05/EQTransformer). Report P-pick absolute error, event F1 at predeclared tolerances, latency after P arrival, station-held-out performance, magnitude/distance strata, and compute. Do not report false alarms/hour from balanced or event-centered windows as if they represented continuous surveillance.

### 4.5 Optical speckle and adaptive-optics telemetry

#### Fiber-speckle temperature

The [Specklegram Temperature Dataset](https://doi.org/10.17605/OSF.IO/8NXVK) is a genuine physical acquisition with exact temperature labels and 14 experimental sets. Heating and cooling provide ordered sequences and session variation. It is useful for minimum-detectable-change and calibration-transfer experiments.

Its limitations are important:

- temperature is a programmed monotone ramp, not an unknown natural fault;
- speckle decorrelation can be global rather than a translated local perturbation;
- defining an alarm at a chosen temperature threshold creates the detection event after data collection;
- normalized cross-correlation is already a physically strong baseline.

Use held-out acquisition sets, never random held-out images from the same ramp. Compare normalized cross-correlation, SSIM, speckle contrast, PCA/PLS, covariance/Hotelling scores, and the data-paper CNN/regression baseline. Metrics are temperature MAE/RMSE, minimum detectable \(\Delta T\), alarm delay in frames or degrees Celsius at a fixed false-alarm probability, hysteresis between heating/cooling, cross-session calibration error, and compute.

#### Adaptive-optics telemetry

The [AOT telemetry release](https://zenodo.org/records/8192742) is real telescope/test-bench telemetry across five AO systems and is highly relevant to changing wavefront covariance. It lacks authoritative turbulence-change or fault-onset labels. It is therefore suitable for unsupervised stability analysis and external validity, not a primary supervised advantage claim.

A later collaboration with an observatory could supply synchronized seeing estimates, loop-state changes, fault logs, and injected calibration events. Strong controls would include wavefront-sensor residual variance/PSD, modal covariance, AO loop residual likelihood, and known turbulence-profile estimators. Until such labels exist, any detected “change” is a candidate event, not verified ground truth.

### 4.6 NMR reaction monitoring

The two Zenodo releases contain real time-indexed NMR spectra. Chemical-shift and line-shape distortions make a max over small translations on the frequency axis physically meaningful. The 2026 click-reaction set is especially clear operationally: a spectrum every 20 s for 5 h and well-shimmed before/after references.

The main obstacle is replication. The releases contain only a few independent reactions, so hundreds of consecutive spectra do not imply hundreds of independent experiments. Reaction progress also changes expected peak intensities directly; covariance is not obviously the principal signal.

An exploratory task can hide the known reaction initiation, detect the first sustained spectral change, and estimate reaction progress. Compare:

- peak integration and matched spectral templates;
- PCA/PLS and multivariate curve resolution (MCR-ALS);
- mean-spectrum GLR and covariance GLR;
- dynamic time warping;
- Wasserstein/“Magnetstein” alignment from the associated robust-monitoring work;
- ShimNetV2-RM correction followed by the same detector.

Metrics are delay in seconds/spectra from controlled initiation, false alarms per pre-reaction hour, concentration or reaction-progress RMSE where references exist, robustness to artificial *evaluation-only* frequency shifts, held-reaction-out performance, and compute. Frequency-shift augmentation is simulation layered onto real spectra and must be labeled as such.

### 4.7 Surface-code drift -- real hardware, constructed change point

The [Google Quantum AI Zenodo release](https://zenodo.org/records/13273331) is unusually rich real hardware data. The archives expose detection events and observable outcomes, making it possible to connect a surveillance alarm to logical performance. It is the strongest continuation of Run 4/5.

It does **not** provide a naturally occurring, independently annotated “drift began here” stream. A benchmark made by concatenating real blocks from different distances, rounds, calibration conditions, or experiment sets uses genuine measurements but has a **constructed change point**. That is stronger than a fully simulated benchmark but weaker than observed operational drift.

The fair task is:

- use the same number of shots/cycles and the same initial calibration for every method;
- detect a hidden boundary between predeclared real experimental blocks;
- separately run controlled Stim noise changes as simulation;
- evaluate whether an alarm triggers a decoder-prior refresh that improves subsequent logical error.

Strong controls are detector-event-rate CUSUM, decoder/detector negative log likelihood, covariance CUSUM, a logistic/threshold model on the same parity features, Wilson-loop diagnostics when applicable, and the 2026 classical-shadow/eSCD method when its observation model is matched.

Report false alarms per million QEC cycles, median/90th-percentile delay in cycles and shots, miss rate, calibration shots, decoder-update latency, post-alarm logical error, bytes processed, and runtime. A lower-cost interpretable alarm could be useful even if it does not beat a fully informed decoder likelihood. It cannot be described as quantum speedup or as superiority to Helstrom, a Wilson oracle, or an oracle logistic/threshold using the true parity feature.

## 5. Real data, constructed changes, and simulation must remain separate

### Category R1: natural continuous event with independent annotation

- **CHB-MIT seizures.** Real continuous clinical EEG; naturally occurring onset and end annotations.

This is the strongest evidence class in the audit. It is still retrospective and cannot alone establish clinical safety.

### Category R2: continuous controlled physical experiment

- **UCI dynamic gas mixtures.** Real sensors and randomly timed commanded concentration changes.
- **Touch and Go.** Real tactile videos and annotated human contact onsets.
- **Fiber-speckle temperature.** Real optics under a controlled temperature ramp.
- **NMR reaction monitoring.** Real reactions and spectra, with controlled initiation.

These are real data. The physical event is controlled, and the algorithm can be blinded to it. They should not be called naturally occurring operational faults.

### Category R3: real measurements but selected windows, derived labels, or constructed boundary

- **STEAD.** Real natural earthquakes, but event/noise windows are preselected; operational exposure is missing.
- **RH20T contact.** Real robot episodes; contact onset inferred from force is a derived proxy.
- **Google QEC.** Real processor shots; block concatenation creates the benchmark change point.
- **AOT telemetry.** Real telescope/test-bench telemetry; change labels are absent.
- **Long-term UCI gas drift.** Real 36-month measurements, but ten batches merge months and are not a continuous fine-resolution stream.

These datasets can validate robustness or mechanism, but their constructed/derived aspects must appear in the abstract, methods, tables, and figure captions.

### Category S: simulation or synthetic perturbation

Simulation is useful for parameter recovery, controlled effect size, and debugging; it is not real-world evidence.

- QEC circuits generated with [Stim](https://github.com/quantumlib/Stim) and injected Pauli/leakage/drift models are simulated, even if calibrated from hardware.
- Fourier-optics or synthetic speckle propagation is simulated.
- Tactile renderers and virtual object/contact datasets are simulated.
- Synthetic NMR spectra, added chemical shifts, and generated kinetic trajectories are simulated.
- Artificial covariance changes added to EEG, gas, or seismic recordings are semi-synthetic.

Every result table should include a `data_status` column with `R1`, `R2`, `R3`, or `S`. Do not pool these categories into one average “accuracy.”

## 6. Recommended Run 6 experimental package

### Run 6A: dynamic gas, preregistered primary experiment

1. Download and checksum both UCI mixture files.
2. Preserve the original chronological order.
3. Define causal windows and transition labels from set-point columns before inspecting detector scores.
4. Use a forward calibration/tuning/test split inside each stream and a cross-mixture transfer test.
5. Calibrate thresholds to a fixed false-alarm budget using only calibration data.
6. Run the marginal/correlation information ladder and all strong controls.
7. Report transition-cluster confidence intervals, latency, false alarms/hour, compute, and failure strata.
8. Freeze the code and configuration before the held-out test.

### Run 6B: Touch and Go

Use the same detector implementation on tactile-image correlation patches. Test whether a max over 2-D translations helps under held-session/material shift. The key comparison is against normalized frame correlation, optical flow, same-feature logistic/Hotelling models, and Sparsh features.

### Run 6C: CHB-MIT

Proceed only after the detector and reporting protocol are frozen on nonmedical domains. Use patient-level splits, all interictal exposure, and research-only language. A medical collaborator should review any later clinical interpretation.

### Run 6Q: QEC continuity study

Keep this as a separate evidence track. Use real hardware blocks with an explicitly constructed hidden boundary and a separate fully simulated drift suite. Evaluate logical downstream value after alarm, not just detection AUC.

## 7. Go/no-go rules for a publication claim

### Go: narrow algorithmic advantage

A positive paper is justified if at least two physical domains show a statistically supported improvement at matched false-alarm and surveillance budgets, including:

- one continuous R1 or R2 dataset;
- one held-domain/session/subject transfer test;
- a gain over the best same-feature linear/threshold detector;
- a mechanism ablation showing that the claimed correlation/localization structure matters;
- honest runtime and calibration cost.

The title and abstract should name the actual scope, for example *structured localized correlation scans for label-light sequential sensing*. They should not lead with quantum inspiration unless quantum formalism contributes a testable capability beyond established scan statistics, covariance change detection, or matched-subspace detection.

### Conditional go: domain tool without a new general algorithm

If the method is useful only in one domain, publish it as a domain-specific detector with a modest claim, an open benchmark, and strong specialist controls.

### No-go for advantage; possible negative benchmark

Do not claim algorithmic superiority if:

- a scalar/mean CUSUM wins at matched false alarms;
- the gain disappears when search multiplicity is correctly calibrated;
- performance depends on random frame/window splits;
- only simulation or constructed QEC boundaries succeed;
- a same-feature logistic/threshold model matches the result;
- an oracle location, parity, seizure channel, or transition identity is supplied only to our method;
- confidence intervals treat adjacent samples as independent.

A careful negative result can still be valuable: it would identify where localized covariance witnesses do and do not add information beyond marginals.

## 8. Bottom line

The most important practical problem exposed by this audit is not “apply ECA everywhere.” It is:

> **Can a symmetry-aware bank of localized correlation witnesses detect an unknown physical transition earlier, at the same false-alarm and calibration budget, than strong marginal and covariance baselines?**

The UCI dynamic gas array is the best place to answer that question first. CHB-MIT is the strongest natural-event validation, Touch and Go is the cleanest approximate spatial-translation test, and Google QEC remains the cleanest lattice-theory test but not the cleanest real-drift dataset. Until those comparisons are run, the correct conclusion is that an advantage is **plausible and sharply testable**, not demonstrated.
