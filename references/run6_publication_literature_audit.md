# Run 6 publication and literature-positioning audit

**Audit date:** 2026-07-28
**Scope:** S-PACE, sequential e-inference, QEC syndrome monitoring, the
Google 2022 event archive, the PNNL/IBM snapshot release, and 2026
classical-shadow eSCD
**Evidence policy:** primary papers, official journal pages, arXiv records,
and official data deposits only
**Data embargo:** no Google `.b8` or `.01` value and no PNNL
`bitstrings.json` content was opened, decoded, streamed, grepped, or hashed
for this audit

## 1. Executive decision

Run 6 has a defensible paper if it is positioned as:

> a preregistered, resource-matched evaluation of predictable sparse and
> spectral contrast monitoring on fixed QEC syndrome records, with exact
> design-based guarantees confined to an independently randomized
> complete-pair audit and empirical conclusions confined to the two real
> hardware arms.

The most defensible contribution is the **integration and audit discipline**,
not a claim that its mathematical ingredients were unknown:

1. past-only maximum-difference witnesses are converted into bounded
   one-step betting factors;
2. sparse and spectral witnesses are combined by fixed priors rather than
   post-selected maxima;
3. all round roles sharing one pair-orientation coin are treated as one
   formal time step, preventing pseudoreplication;
4. common-mode and correlation/spectral branches are both retained, with an
   explicit common-mode blind-spot certificate for pure orbit centering;
5. the method is evaluated against detector firing rate, within-shot CUSUM,
   fixed diagonal surprise, shrinkage covariance/Hotelling, and an online
   logistic witness using the same feature bank; and
6. detector outputs are frozen before decoder outcomes are opened.

The literature search found close foundations for every individual element:
test martingales, predictable betting, e-process mixtures, e-CUSUM/e-SR,
randomization e-processes, sparse projection monitoring, QEC detector-rate
and detector-likelihood diagnostics, adaptive syndrome-noise estimation,
and classical-shadow eSCD. It did **not** identify a primary source using the
exact locked S-PACE composite and Run 6 protocol on these two releases.
That negative search result is not proof of priority. The manuscript should
say “we introduce the following preregistered composite” rather than “we
discover the first general theory.”

No empirical advantage exists until the locked Google and PNNL gates have
both been evaluated. If either gate fails, the required conclusion is:

> Under the preregistered operating points and evidence arms, we find no
> demonstrated S-PACE algorithmic advantage.

That outcome can still support a useful negative-results/reproducibility
paper: one author-identified event, exact randomization calibration, two
real-hardware snapshot cohorts, complete same-information controls, and a
public freeze chain are more scientifically useful than a post-selected
positive result.

## 2. Keep the observation levels separate

Most dangerous overclaims arise from mixing three different experiments.

| Level | Object available to an algorithm | Relevant theory | Run 6 access |
|---|---|---|---|
| Quantum acquisition | an unmeasured state or channel and a choice of POVM/measurement setting | Helstrom discrimination, quantum quickest-change limits, classical-shadow acquisition | **No.** The measurements have already been made. |
| Classical syndrome stream | fixed stabilizer/repetition-code detector bits in archive order | classical likelihoods, CUSUM, covariance monitoring, predictable betting and e-processes | **Yes.** This is the detector input. |
| Retrospective utility | actual observable flips and decoder predictions | risk coverage, veto/triage, decoder comparison | **Only after detector scores are frozen.** |

This separation gives four immediate manuscript rules.

1. S-PACE in Run 6 is a **classical sequential procedure applied to quantum
   hardware records**. That does not make it a quantum algorithm.
2. A classical procedure on fixed syndrome bits cannot be compared as if it
   optimized the original quantum measurement.
3. Decoder outcomes can evaluate downstream utility, but cannot train,
   select, or rank the unsupervised detector.
4. A Wilson loop, logical operator, or other observable absent from the
   archived observation sigma-field cannot be reconstructed by naming a
   syndrome statistic after it.

## 3. Established sequential-statistics foundations

### 3.1 Test martingales, e-values, and predictable betting

Shafer, Shen, Vereshchagin and Vovk formalized nonnegative test martingales
as sequential evidence, and Vovk and Wang developed e-values and their
mixture properties. Howard et al. place the same nonnegative-supermartingale
machinery in the modern time-uniform inference literature. Waudby-Smith and
Ramdas develop predictable betting constructions for bounded means. These
are established foundations, not Run 6 discoveries:

- [Shafer et al., 2011](https://doi.org/10.1214/10-STS347);
- [Vovk and Wang, 2021](https://doi.org/10.1214/20-AOS2020);
- [Howard et al., 2021](https://doi.org/10.1214/20-AOS1991); and
- [Waudby-Smith and Ramdas, 2024](https://doi.org/10.1093/jrsssb/qkad009).

If a bounded score \(s_t\in[-1,1]\) satisfies

\[
\mathbb E_0[s_t\mid\mathcal F_{t-1}]=0
\]

and \(\beta_t\in[-1,1]\) is predictable, then

\[
L_t=1+\beta_t s_t
\]

is a nonnegative conditional e-factor. This is an elementary specialization
of established predictable-betting theory. Run 6's contribution is to
construct particular QEC contrast scores and enforce predictability in an
executable protocol.

Fixed convex mixtures of valid e-processes remain valid. Maxima generally do
not. Products require a sequential conditional-mean argument; dependence
between coordinates cannot be ignored. Consequently, the 51 Google round
roles sharing one complete-shot orientation must be scored from their
pre-shot states and mixed as experts in **one** update. Treating them as 51
successive coin revelations would expose the shared orientation after the
first role and invalidate the remaining conditional argument. This is a
correct experimental-unit application of martingale theory, not a claim for
a new independence theorem.

### 3.2 e-process lifetime control is not e-detector ARL control

Shin, Ramdas and Rinaldo introduced e-detectors and their CUSUM- and
Shiryaev--Roberts-style constructions:

- [Shin, Ramdas and Rinaldo, 2024](https://doi.org/10.51387/23-NEJSDS51)
  (also [arXiv:2203.03532](https://arxiv.org/abs/2203.03532)).

The paper must preserve the following distinction.

| Construction | Typical threshold | Meaning |
|---|---:|---|
| Nonnegative e-process with initial value 1 | \(1/\alpha\) | probability of ever crossing is at most \(\alpha\), under its declared null |
| Proper changepoint-prior mixture e-process | \(1/\alpha\) | the same lifetime statement, including the cost of mixing possible starts |
| Shiryaev--Roberts e-detector | \(\gamma\) | average run length at least \(\gamma\), under the e-detector conditions |
| Empirical validation quantile | fitted score threshold | only the measured/bootstrapped operating point that was actually calibrated |

An SR statistic with neutral factors \(L_t\equiv1\) advances like a clock and
eventually crosses a finite threshold. Its ARL statement must never be
rewritten as a lifetime false-alarm probability. Conversely, a proper-prior
e-process pays prior mass for possible starts and is not numerically
interchangeable with SR.

### 3.3 Randomized orientation is an exact design, not a hardware-null model

Run 6's independent A/B swap coin acts on an entire paired shot. Conditional
on the unordered observed pair, an antisymmetric score is centered under
that artificial randomization. This supplies an exact implementation and
design-based type-I audit.

Sequential randomization tests built from e-values already exist in current
primary literature; see Zampieri's trial-monitoring construction
([arXiv:2512.04366](https://arxiv.org/abs/2512.04366), v9 as of May 2026).
The application and experimental unit differ, but Run 6 must not claim to
have invented randomization-based e-processes.

The fixed natural Google orientation—earlier reference reservoir versus
later monitor stream—is not randomized by the experiment and is not known
to be conditionally exchangeable. The real-event replay therefore remains
empirical even if the same code produces exact factors in the randomized
audit. Likewise, a PNNL early/late snapshot boundary is not made into a
natural null by symmetric notation.

## 4. Established change-detection and multivariate controls

### 4.1 Page CUSUM and Shiryaev--Roberts

Page's cumulative-sum scheme is classical
([Page, 1954](https://doi.org/10.1093/biomet/41.1-2.100)). Roberts' control
chart comparison is an early source for the statistic now associated with
the Shiryaev--Roberts family
([Roberts, 1966](https://doi.org/10.1080/00401706.1966.10490374)).

Run 6 M0C resets at each shot and scans 51 roles within that shot. It is
therefore accurately described as a **within-shot two-sided Page--CUSUM
control**. It is not:

- a continuously running cross-shot CUSUM;
- a likelihood-optimal CUSUM for a known post-change distribution;
- an e-factor; or
- a quickest-change oracle.

### 4.2 Hotelling and shrinkage covariance

Hotelling's multivariate quadratic statistic and the Ledoit--Wolf
well-conditioned covariance estimator are established:

- [Hotelling, 1931](https://doi.org/10.1214/aoms/1177732979); and
- [Ledoit and Wolf, 2004](https://doi.org/10.1016/S0047-259X(03)00096-4).

M2 uses a fixed, training-only role centering and pooled Ledoit--Wolf
precision. It is a strong covariance-aware empirical control. It is not an
e-process and should not inherit an exact false-alarm statement from the
S-PACE branches.

### 4.3 Sparse discriminative directions predate S-PACE

The broad idea “monitor directions sensitive to change rather than the
directions of largest variance” is already present in high-dimensional
change detection. Tveten and Glad explicitly tailor projections for sparse
mean/covariance changes and emphasize that low-variance projections can be
important ([arXiv:1908.02029](https://arxiv.org/abs/1908.02029)).

Accordingly:

- the contrast-over-variance motivation is legitimate;
- the capped-simplex top-\(k\) support formula is a useful exact
  implementation result for the declared objective;
- neither fact establishes universal statistical optimality; and
- the manuscript should compare with covariance and same-feature learners,
  not only PCA.

### 4.4 The classical likelihood ceiling

The Neyman--Pearson likelihood-ratio result is classical
([Neyman and Pearson, 1933](https://doi.org/10.1098/rsta.1933.0009)).
For a simple same-information alternative \(P_1\ll P_0\), any one-step
e-factor \(L\) satisfying \(\mathbb E_0L\le1\) obeys

\[
\mathbb E_1[\log L]
\le
\mathrm{KL}(P_1\Vert P_0).
\]

One proof writes
\[
\mathbb E_1\log L
=
\mathbb E_1\log\!\left(
L\frac{dP_0}{dP_1}
\right)
+
\mathrm{KL}(P_1\Vert P_0)
\]
and applies Jensen to the first term. Equality is attained by the correct
likelihood ratio. Conditional versions give the same chain-rule ceiling for
adaptive factors.

Therefore S-PACE cannot be claimed to beat a correctly specified
same-observation likelihood ratio in expected log growth. Its potential
benefits are robustness to misspecification, sparse interpretation,
past-only adaptation, transparent calibration, or computation—not a
violation of the likelihood ceiling.

## 5. QEC syndrome monitoring: the prior-art floor

### 5.1 Detector rates and correlations are established observables

The Google Quantum AI surface-code paper defines detection events as changes
in stabilizer outcomes and analyzes their spatial/temporal probabilities and
pairwise correlations. In its distance-25 repetition-code experiment, a
single high-energy event set the observed logical-error floor; the paper
states that such events can be identified by spikes in detection-event
counts:

- [Google Quantum AI, *Nature* 614, 676--681
  (2023)](https://doi.org/10.1038/s41586-022-05434-1).

Radiation-induced, spatially correlated superconducting-qubit bursts were
already directly resolved by McEwen et al.:

- [McEwen et al., *Nature Physics* 18, 107--111
  (2022)](https://doi.org/10.1038/s41567-021-01432-8).

The archived Run 6 event is correctly called the **author-identified
high-energy event**. It should not be upgraded to a machine-verified cosmic
ray without an event-specific physical label. Detector firing rate is an
especially strong baseline because the source paper itself identifies count
spikes as an event signature.

### 5.2 Detector likelihood is prior art, but M1 is not a reproduction

Hesner, Hetényi and Wootton define average detector likelihood as a QEC
benchmark and show that it predicts code performance for two surface-code
variants:

- [Hesner, Hetényi and Wootton, *Physical Review A* 111, 052452
  (2025)](https://doi.org/10.1103/PhysRevA.111.052452).

Run 6 M1 is a fixed diagonal, per-role Bernoulli negative-log-likelihood
difference. It belongs to the same model-aware diagnostic family but is not
a faithful implementation of every definition or calibration in Hesner et
al. The manuscript should call it “fixed diagonal detector-surprise
control,” cite detector-likelihood benchmarking as motivation, and give its
own formula.

### 5.3 QEC-specific adaptation and drift work

The contemporary comparison set is broader than generic CUSUM:

- Wang et al. use decoded matching statistics to reweight a decoding graph
  under drifted and correlated simulated noise
  ([DGR, arXiv:2311.16214](https://arxiv.org/abs/2311.16214)).
- Bhardwaj et al. derive sliding and overlapping window estimators for
  time-dependent Pauli noise using syndrome statistics and validate adaptive
  decoding in simulations
  ([arXiv:2511.09491](https://arxiv.org/abs/2511.09491);
  [accepted in PRX Quantum, 23 June
  2026](https://doi.org/10.1103/z1hc-nqw5)).
- Poster, Chadwick and Baker map detector firing rate to logical error and
  study remapping/recalibration in a system architecture
  ([ReloQate, arXiv:2603.00837](https://arxiv.org/abs/2603.00837)).
- Stein et al. condition a neural repetition-code decoder on calibration
  records and evaluate cross-chain/cross-calibration generalization on IBM
  hardware
  ([arXiv:2601.16123](https://arxiv.org/abs/2601.16123)).
- Tan et al. analyze the resilience of surface codes to burst errors
  ([*Physical Review A* 113, 022450
  (2026)](https://doi.org/10.1103/vwtl-hwfx)).

These works solve different problems—decoder reweighting, noise estimation,
logical-error prediction, calibration-conditioned decoding, and burst
resilience—but they prevent any claim that Run 6 is the first work on QEC
drift, syndrome adaptation, detector-rate monitoring, or error bursts.

Run 6's narrower opening is:

> a frozen, same-information comparison of predictable sparse/spectral
> contrast witnesses with simple and model-aware monitors on one real
> author-identified event and independently selected real-hardware cohort
> shifts.

## 6. Dataset-specific positioning

### 6.1 Google 2022 deposit

The official source is:

- Google Quantum AI Team, *Data for “Suppressing quantum errors by scaling a
  surface code logical qubit”*, Zenodo record 6804040,
  [DOI 10.5281/zenodo.6804040](https://doi.org/10.5281/zenodo.6804040),
  deposited 14 July 2022, CC BY 4.0.

The associated paper reports 500,000 repetitions, 50 QEC cycles, and one
high-energy event in the distance-25 repetition-code experiment. The archive
README localizes a decoder-mismatch cluster near shot 57,775. These facts
support a predeclared approximate event window.

They do **not** supply:

- an exact physical onset or recovery time;
- a wall-clock timestamp or inter-shot cadence;
- independent labels for multiple natural events; or
- a population of event clusters from which to estimate a general miss
  probability.

Allowed description:

> We causally replay archive order around an approximately localized,
> author-identified high-energy event.

Forbidden descriptions:

- “real-time latency in seconds”;
- “exact quickest-change delay”;
- “prospective deployment”;
- “many independent event detections”; or
- “confirmed cosmic-ray onset at shot 57,775.”

One contiguous event cluster is one event, not 51 cycles times many affected
checks worth of independent evidence. Uncertainty must be clustered at a
shot/block or event level.

### 6.2 PNNL deposit of IBM-hardware experiments

The official sources are:

- Stein et al., *Calibration-Conditioned FiLM Decoders for Low-Latency
  Decoding of Quantum Error Correction Evaluated on IBM Repetition-Code
  Experiments*,
  [arXiv:2601.16123](https://arxiv.org/abs/2601.16123); and
- Stein/PNNL, accompanying dataset v0.1, Zenodo record 20768087,
  [DOI 10.5281/zenodo.20768087](https://doi.org/10.5281/zenodo.20768087),
  CC BY 4.0; the
  [official record API](https://zenodo.org/api/records/20768087) supplies the
  current metadata and file inventory.

The deposit metadata identify 352 hardware snapshots across IBM Fez,
Kingston and Pittsburgh and describe raw hardware records. It is an
author-released PNNL deposit containing experiments on named IBM processors;
it is not an official IBM dataset or IBM endorsement.

The release requires two explicit caveats.

1. The paper reports over 2.7 million shots and 400 contiguous-chain
   calibration snapshots, whereas v0.1 metadata and the extracted
   metadata-only audit describe 3,779,584 circuit shots and 352 hardware
   snapshots. The exact paper split must not be presumed.
2. The Zenodo description says `index.csv` is included, but the current
   official five-file record lists only the three backend archives,
   `DESCRIPTION.txt`, and `README.md`. The absent index is a release-level
   reproducibility discrepancy, not something to silently reconstruct and
   call author-provided.

The calibration object's `last_update_date` is not a per-shot execution
timestamp. Snapshot folders also do not establish a backend-wide continuous
acquisition order. Because the locked Pittsburgh pairs can differ in full
transpiled QASM, the conservative evidence label is:

> constructed boundary between real-hardware snapshot cohorts, potentially
> combining circuit and hardware/calibration domain shift.

Even a positive result is not “natural temporal drift detection.” PNNL
paths/snapshot pairs—not millions of component bits—are the independent
units for the auxiliary summary.

## 7. Classical-shadow eSCD: related but not runnable here

Huang, Kueng and Preskill introduced classical shadows as randomized
measurements that support prediction of many later-chosen properties:

- [Huang, Kueng and Preskill, *Nature Physics* 16, 1050--1057
  (2020)](https://doi.org/10.1038/s41567-020-0932-7).

Zecchin, Simeone and Ramdas combine that acquisition strategy with
e-detectors in shadow-based sequential changepoint e-detection (eSCD):

- [*Universal Sequential Changepoint Detection of Quantum Observables via
  Classical Shadows*, arXiv:2602.11846
  (2026)](https://arxiv.org/abs/2602.11846).

As of this audit, eSCD is a February 2026 v1 preprint with theory and
numerical experiments. Its protocol receives, at each time:

1. a recorded randomized local- or joint-Clifford setting \(U_t\);
2. the computational-basis measurement outcome \(b_t\); and
3. the inverse shadow channel needed to form an unbiased observable
   estimator.

The Google and PNNL archives instead contain outcomes of fixed
code-designed measurements. They do not contain an informationally complete
randomized shadow acquisition for each archived shot. Therefore:

- a syndrome-derived e-process is legitimate;
- calling it “eSCD” is not;
- importing eSCD's measurement universality is not;
- importing its shadow sample/delay bounds is not; and
- fabricating random Clifford labels after measurement does not repair the
  missing acquisition.

The closest correct relationship is:

> Both S-PACE and eSCD use predictable betting/e-detector machinery after a
> bounded unbiased or centered observable score is available. eSCD obtains
> observable universality from a randomized quantum measurement channel;
> Run 6 uses a fixed, observable-specific classical syndrome feature map and
> obtains exact centering only in its separate randomized-pair design.

Quantum quickest-change theory also sets a distinct ceiling when the
measurement itself can be optimized; see Fanizza, Hirche and Calsamiglia,
[*Physical Review Letters* 131, 020602
(2023)](https://doi.org/10.1103/PhysRevLett.131.020602). Run 6 cannot claim
to reach or beat that quantum-acquisition limit.

## 8. S-PACE novelty ledger

| S-PACE element | Established foundation | Defensible Run 6 contribution | Wording to avoid |
|---|---|---|---|
| “difference, not variance” | discriminative projections and sparse change monitoring predate Run 6 | a fixed QEC raw-check plus pair-correlation feature bank and locked top-\(k\) witness rule | “first discriminative feature method” |
| \(1+\beta s_t\) betting factor | predictable bounded betting/test martingales | exact interface from a declared QEC contrast to a runnable factor | “new martingale theorem” |
| fixed mixtures of experts | e-value mixture theory | complete prior ledger over role, half-life, cap, rank and bet experts | “posterior probability of the changed check” |
| top-\(k\) capped support | elementary linear optimization on a capped simplex | closed-form, deterministic support/tie rule for this algorithm | “statistically optimal sparse features” |
| positive-eigenspace effect | standard Jordan/Helstrom variational result | a past-only spectral QEC correlation witness with frozen refresh rules | “new Helstrom measurement” |
| e-CUSUM/e-SR accumulation | established e-detectors | both lifetime and ARL modes implemented without conflating guarantees | “SR threshold controls probability of ever alarming” |
| complete-pair randomization | established randomization inference and recent randomization e-processes | one swap coin per physical shot, with all role experts mixed at one formal time | “natural hardware exchangeability” |
| global plus relative branches | invariant/common-mode decomposition is standard linear algebra | an explicit certificate that pure orbit centering deletes a chip-wide additive mode, followed by a two-branch repair | “new topological-order theorem” |
| Google event replay | data and count-spike signature are established by Google | preregistered, same-budget benchmark with outcomes held behind detector freeze | “discovery of the event” |
| PNNL auxiliary | dataset and calibration-conditioned decoding are Stein et al.'s work | value-blind path reconstruction and constructed cross-snapshot monitoring gate | “natural IBM drift stream” |

### Recommended central contribution sentence

> We introduce S-PACE, a preregistered composite of past-only sparse and
> spectral contrast witnesses. Under an independently randomized
> complete-pair design, its bounded scores yield conditional e-factors; on
> the natural Google and constructed PNNL/IBM replays, we evaluate the same
> frozen algorithm empirically against rate, likelihood, covariance, CUSUM,
> and same-feature logistic controls at matched data and alert budgets.

### What may count as a technical contribution

The paper may prove and use the following bounded propositions, provided it
labels their ingredients honestly:

- a conditionally centered bounded contrast composed with a predictable
  bounded witness gives a conditional e-factor;
- all roles sharing one random orientation form one time step and may be
  mixed, but not naively multiplied as fresh revelations;
- pure relative/orbit centering is pathwise blind to a purely additive
  common mode;
- the declared capped-simplex linear-gap objective has the signed top-\(k\)
  closed form; and
- the positive eigenspace maximizes the declared one-step operator gap.

These propositions make the algorithm auditable. The first, fourth and fifth
are direct specializations of established martingale, linear-programming and
Jordan/Helstrom facts. The third is a projection identity. Novelty should be
claimed for their **joint protocol and consequences**, not for re-proving
the foundations.

## 9. Hard no-go and claim boundaries

### 9.1 Accessible-data no-go

Let \(Y\) be the archived syndrome record and let \(T(Y)\) be any Run 6
score, adaptive state, alarm, or randomized post-processing whose extra
randomness has the same law under both hypotheses. If two physical regimes
induce the same law for \(Y\), they induce the same law for \(T(Y)\).
Therefore no fixed-syndrome algorithm can detect a change that is invisible
in the archived syndrome sigma-field.

This is a pushforward/data-processing statement. It is not a new toric-code,
local-topological-order, or string-theory theorem.

### 9.2 Helstrom is a ceiling in another experiment

Helstrom's binary quantum decision theory optimizes a POVM for specified
quantum states, priors and costs:

- [Helstrom, 1969](https://doi.org/10.1007/BF01007479).

Run 6 receives classical outcomes after a fixed measurement. Its spectral
positive-part calculation is algebraically related to the standard
Jordan/Helstrom variational formula, but it is not a physical Helstrom
measurement on the processor state. The paper must not claim:

- superiority to Helstrom;
- attainment of Helstrom error;
- a new quantum measurement;
- a quantum sensing advantage; or
- a quantum speedup.

### 9.3 Wilson loops and topological order are not Run 6 results

Wilson loops originate in lattice gauge theory
([Wilson, 1974](https://doi.org/10.1103/PhysRevD.10.2445)); toric-code
logical/string structure is established in Kitaev's construction
([Kitaev, 2003](https://doi.org/10.1016/S0003-4916(02)00018-0)).
Run 6's primary dataset is a one-dimensional repetition-code experiment,
not a lattice-gauge Wilson-loop measurement campaign.

“Wilson oracle” is not a standard name for an optimal detector and should
not appear. A future surface-code experiment could predeclare logical-loop
or gauge-invariant observables as additional diagnostics, but that is a
different measurement/feature budget.

### 9.4 Symmetry language has nearby but distinct literatures

Hiai, Mosonyi and Hayashi study asymptotic quantum hypothesis testing when
measurements are constrained by a group symmetry
([J. Math. Phys. 50, 103304
(2009)](https://doi.org/10.1063/1.3234186)). Goldstein and Sela resolve
many-body entanglement into conserved-charge sectors
([Phys. Rev. Lett. 120, 200602
(2018)](https://doi.org/10.1103/PhysRevLett.120.200602)).

S-PACE's role/feature sectors and orbit centering are classical
representation bookkeeping on observed features. They neither extend those
quantum asymptotic error exponents nor compute symmetry-resolved
entanglement. Those papers can appear in a broad outlook, but not as
evidence that Run 6 solved either problem.

### 9.5 Same-feature learners and oracle features

The online logistic comparator receives the same 300-dimensional feature
bank as sparse S-PACE. If logistic ties or wins, sparsity or visual
interpretability cannot override that outcome. Likewise, a threshold on the
true changed parity or a correctly specified post-change likelihood is a
privileged ceiling; S-PACE cannot claim superiority merely because that
feature was omitted from an easier baseline.

### 9.6 Prohibited global claims

The final paper must not claim:

- universal sample-efficiency advantage;
- scalable computational advantage without measured scaling experiments;
- quantum acceleration;
- superiority to Helstrom, a correct same-information likelihood ratio,
  Wilson diagnostics, or an oracle using the true changed feature;
- discovery of high-energy QEC events, detector-rate signatures, Wilson
  loops, toric-code structure, or symmetry-resolved entanglement;
- natural PNNL drift or exact Google wall-clock delay;
- a string-theory result; or
- a holographic-duality result.

String theory and holography have no inferential role in the locked Run 6
data, algorithm, or endpoint. They should be absent from the title,
abstract, main claims, and conclusion.

## 10. Publication architecture

### 10.1 Recommended title

Safest:

> **Predictable Sparse and Spectral Contrast Monitoring of Real QEC
> Syndromes: A Preregistered Benchmark**

Acceptable if the acronym is retained:

> **S-PACE: Pair-Calibrated Contrast e-Detection for Real QEC Syndrome
> Monitoring**

The second title requires an early qualifier that exact pair calibration
belongs to the randomized-design arm, whereas natural hardware replay is
empirically calibrated.

Avoid “quantum advantage,” “quantum-inspired,” “universal,” “optimal,”
“topological,” “Wilson,” “string,” and “holographic” in the title.

### 10.2 Results-contingent abstract logic

The abstract should have five moves:

1. **Problem:** real QEC noise is nonstationary and burst-prone, while simple
   detector-rate signals are already strong.
2. **Method:** define a fixed composite of past-only sparse and spectral
   contrast witnesses with complete-shot expert mixtures.
3. **Validity:** state exact false-alarm control only for the independent
   randomized-pair design; label real hardware arms empirical.
4. **Evidence:** name one Google author-identified event and 11
   Pittsburgh-path constructed snapshot comparisons, with same-budget
   baselines.
5. **Outcome:** insert the locked pass/fail conclusion without changing the
   success criterion.

Positive-result wording:

> Under the preregistered Google event and PNNL retention gates, the fixed
> composite improved the declared endpoints over paired detector firing
> rate and the same-feature online logistic comparator. This is an
> empirical, dataset-specific algorithmic advantage under the reported
> budgets.

Negative-result wording:

> The fixed composite did not satisfy the preregistered two-arm advantage
> rule. The resulting benchmark identifies which simpler comparator matched
> or exceeded it and preserves an exact randomized-design calibration
> audit.

### 10.3 Required main-paper evidence

The main paper should expose, not hide:

- the one-event limitation;
- the Google event-window ambiguity;
- natural versus randomized orientation;
- one formal update per paired shot;
- lifetime e-process versus SR ARL thresholds;
- detector-only freeze before `.01` outcomes;
- all comparator thresholds and full alert frontiers;
- the same-feature logistic comparison;
- PNNL QASM and snapshot confounding;
- path-level rather than bit-level uncertainty;
- runtime, peak memory, eigendecomposition count, and input-bit budgets;
- all failed gates and negative controls; and
- the exact source hashes, environment lock, and freeze/ratification chain.

### 10.4 Recommended figures and tables

1. **Observation-level diagram:** quantum state → fixed code measurement →
   syndrome stream → frozen detector → retrospective decoder outcome.
2. **Protocol timeline:** fit, disjoint threshold, held reference/monitor,
   detector freeze, outcome join.
3. **Method table:** information, fitted state, score, accumulator,
   threshold, resource cost, and whether the guarantee is exact or
   empirical.
4. **Google event plot:** all methods on the same archive-shot axis with
   predeclared windows and pre-event false alarms.
5. **Risk-coverage plot:** frozen shot ranking against both decoder labels.
6. **PNNL forest plot:** 11 path-level paired delay differences, logical
   states averaged before macro aggregation.
7. **Calibration plot:** random-swap alarm distributions at the declared
   complete-shot threshold.
8. **Claim-boundary table:** exact-design, natural-event, constructed-shift,
   and unavailable classical-shadow evidence.

## 11. Reviewer-risk audit

| Likely objection | Required answer |
|---|---|
| “DFR already detects the event.” | Show the locked head-to-head result; do not redefine advantage after seeing it. |
| “The event window was selected from the same data.” | State that the window came from the archive README before detector values and report all three frozen windows. |
| “The pair reference is not simultaneous or exchangeable.” | Agree; reserve exact validity for independent random orientation and call natural replay empirical. |
| “You counted 51 correlated roles as 51 trials.” | Show one-shot formal update and the explicit \(1/51\) role prior. |
| “M1 is not the published detector-likelihood method.” | Call it diagonal detector surprise, give the formula, and cite Hesner et al. only as related motivation. |
| “PNNL is not a time series.” | Use “constructed boundary between real-hardware snapshot cohorts”; report QASM differences and property-date semantics. |
| “Why no eSCD baseline?” | Explain the missing randomized Clifford settings and inverse shadow channel; do not invent a proxy. |
| “Where is the quantum advantage?” | State there is none: all Run 6 processing is classical and consumes fixed measurements. |
| “Is the spectral branch just Helstrom?” | The positive-part variational step is standard; novelty is the predictable, budgeted sequential integration, if supported. |
| “One event cannot establish general power.” | Agree; report it as a case study plus design calibration and independent constructed-shift robustness arm. |

## 12. Citation metadata, ready for BibTeX

The entries below use stable journal DOIs, arXiv identifiers, and versioned
Zenodo record DOIs. Preprints are labeled as such.

```bibtex
@article{shafer2011testmartingales,
  author  = {Shafer, Glenn and Shen, Alexander and Vereshchagin, Nikolai and Vovk, Vladimir},
  title   = {Test Martingales, Bayes Factors and p-Values},
  journal = {Statistical Science},
  year    = {2011},
  volume  = {26},
  number  = {1},
  pages   = {84--101},
  doi     = {10.1214/10-STS347}
}

@article{vovk2021evalues,
  author  = {Vovk, Vladimir and Wang, Ruodu},
  title   = {E-values: Calibration, Combination and Applications},
  journal = {The Annals of Statistics},
  year    = {2021},
  volume  = {49},
  number  = {3},
  pages   = {1736--1754},
  doi     = {10.1214/20-AOS2020}
}

@article{howard2021confidence,
  author  = {Howard, Steven R. and Ramdas, Aaditya and McAuliffe, Jon and Sekhon, Jasjeet},
  title   = {Time-uniform, Nonparametric, Nonasymptotic Confidence Sequences},
  journal = {The Annals of Statistics},
  year    = {2021},
  volume  = {49},
  number  = {2},
  pages   = {1055--1080},
  doi     = {10.1214/20-AOS1991}
}

@article{waudbysmith2024betting,
  author  = {Waudby-Smith, Ian and Ramdas, Aaditya},
  title   = {Estimating Means of Bounded Random Variables by Betting},
  journal = {Journal of the Royal Statistical Society Series B: Statistical Methodology},
  year    = {2024},
  volume  = {86},
  number  = {1},
  pages   = {1--27},
  doi     = {10.1093/jrsssb/qkad009}
}

@article{shin2024edetectors,
  author  = {Shin, Jaehyeok and Ramdas, Aaditya and Rinaldo, Alessandro},
  title   = {E-detectors: A Nonparametric Framework for Sequential Change Detection},
  journal = {The New England Journal of Statistics in Data Science},
  year    = {2024},
  volume  = {2},
  pages   = {229--260},
  doi     = {10.51387/23-NEJSDS51},
  eprint  = {2203.03532},
  archivePrefix = {arXiv},
  primaryClass  = {stat.ME}
}

@article{zampieri2025randomization,
  author  = {Zampieri, Fernando G.},
  title   = {Sequential Randomization Tests Using e-values: Applications for Trial Monitoring},
  year    = {2025},
  eprint  = {2512.04366},
  archivePrefix = {arXiv},
  primaryClass  = {stat.ME},
  note    = {Preprint, v9 dated 10 May 2026}
}

@article{page1954cusum,
  author  = {Page, E. S.},
  title   = {Continuous Inspection Schemes},
  journal = {Biometrika},
  year    = {1954},
  volume  = {41},
  number  = {1--2},
  pages   = {100--115},
  doi     = {10.1093/biomet/41.1-2.100}
}

@article{roberts1966control,
  author  = {Roberts, S. W.},
  title   = {A Comparison of Some Control Chart Procedures},
  journal = {Technometrics},
  year    = {1966},
  volume  = {8},
  number  = {3},
  pages   = {411--430},
  doi     = {10.1080/00401706.1966.10490374}
}

@article{hotelling1931ratio,
  author  = {Hotelling, Harold},
  title   = {The Generalization of Student's Ratio},
  journal = {The Annals of Mathematical Statistics},
  year    = {1931},
  volume  = {2},
  number  = {3},
  pages   = {360--378},
  doi     = {10.1214/aoms/1177732979}
}

@article{ledoitwolf2004covariance,
  author  = {Ledoit, Olivier and Wolf, Michael},
  title   = {A Well-Conditioned Estimator for Large-Dimensional Covariance Matrices},
  journal = {Journal of Multivariate Analysis},
  year    = {2004},
  volume  = {88},
  number  = {2},
  pages   = {365--411},
  doi     = {10.1016/S0047-259X(03)00096-4}
}

@article{neymanpearson1933tests,
  author  = {Neyman, Jerzy and Pearson, Egon S.},
  title   = {On the Problem of the Most Efficient Tests of Statistical Hypotheses},
  journal = {Philosophical Transactions of the Royal Society of London. Series A},
  year    = {1933},
  volume  = {231},
  number  = {694--706},
  pages   = {289--337},
  doi     = {10.1098/rsta.1933.0009}
}

@article{tveten2019sparse,
  author  = {Tveten, Martin and Glad, Ingrid K.},
  title   = {Online Detection of Sparse Changes in High-Dimensional Data Streams Using Tailored Projections},
  year    = {2019},
  eprint  = {1908.02029},
  archivePrefix = {arXiv},
  primaryClass  = {stat.ME},
  note    = {Preprint}
}

@article{google2023surface,
  author  = {{Google Quantum AI}},
  title   = {Suppressing Quantum Errors by Scaling a Surface Code Logical Qubit},
  journal = {Nature},
  year    = {2023},
  volume  = {614},
  number  = {7949},
  pages   = {676--681},
  doi     = {10.1038/s41586-022-05434-1}
}

@misc{google2022data,
  author    = {{Google Quantum AI Team}},
  title     = {Data for ``Suppressing Quantum Errors by Scaling a Surface Code Logical Qubit''},
  publisher = {Zenodo},
  year      = {2022},
  doi       = {10.5281/zenodo.6804040},
  url       = {https://zenodo.org/records/6804040},
  note      = {CC BY 4.0}
}

@article{mcewen2022bursts,
  author  = {McEwen, Matt and Faoro, Lara and Arya, Kunal and others},
  title   = {Resolving Catastrophic Error Bursts from Cosmic Rays in Large Arrays of Superconducting Qubits},
  journal = {Nature Physics},
  year    = {2022},
  volume  = {18},
  pages   = {107--111},
  doi     = {10.1038/s41567-021-01432-8}
}

@article{hesner2025likelihood,
  author  = {Hesner, Ian and Het{\'e}nyi, Bence and Wootton, James R.},
  title   = {Using Detector Likelihood for Benchmarking Quantum Error Correction},
  journal = {Physical Review A},
  year    = {2025},
  volume  = {111},
  number  = {5},
  pages   = {052452},
  doi     = {10.1103/PhysRevA.111.052452}
}

@article{wang2023dgr,
  author  = {Wang, Hanrui and Liu, Pengyu and Liu, Yilian and Gu, Jiaqi and Baker, Jonathan and Chong, Frederic T. and Han, Song},
  title   = {{DGR}: Tackling Drifted and Correlated Noise in Quantum Error Correction via Decoding Graph Re-weighting},
  year    = {2023},
  eprint  = {2311.16214},
  archivePrefix = {arXiv},
  primaryClass  = {quant-ph},
  note    = {Preprint}
}

@article{bhardwaj2026drifting,
  author  = {Bhardwaj, Devansh and Takou, Evangelia and Lin, Yingjia and Brown, Kenneth R.},
  title   = {Adaptive Estimation of Drifting Noise in Quantum Error Correction},
  journal = {PRX Quantum},
  year    = {2026},
  doi     = {10.1103/z1hc-nqw5},
  eprint  = {2511.09491},
  archivePrefix = {arXiv},
  primaryClass  = {quant-ph},
  note    = {Accepted 23 June 2026}
}

@article{poster2026reloqate,
  author  = {Poster, Maxwell and Chadwick, Jason and Baker, Jonathan Mark},
  title   = {{ReloQate}: Transient Drift Detection and In-Situ Recalibration in Surface Code Quantum Error Correction},
  year    = {2026},
  eprint  = {2603.00837},
  archivePrefix = {arXiv},
  primaryClass  = {quant-ph},
  note    = {Preprint, v2}
}

@article{tan2026bursts,
  author  = {Tan, Shi Jie Samuel and Pattison, Christopher A. and McEwen, Matt and Preskill, John},
  title   = {Resilience of the Surface Code to Error Bursts},
  journal = {Physical Review A},
  year    = {2026},
  volume  = {113},
  number  = {2},
  pages   = {022450},
  doi     = {10.1103/vwtl-hwfx}
}

@article{stein2026film,
  author  = {Stein, Samuel and Kan, Shuwen and Liu, Chenxu and Harkness, Adrian and Garner, Sean and Du, Zefan and Ding, Yufei and Mao, Ying and Li, Ang},
  title   = {Calibration-Conditioned {FiLM} Decoders for Low-Latency Decoding of Quantum Error Correction Evaluated on {IBM} Repetition-Code Experiments},
  year    = {2026},
  eprint  = {2601.16123},
  archivePrefix = {arXiv},
  primaryClass  = {quant-ph},
  note    = {Preprint}
}

@misc{stein2026dataset,
  author    = {Stein, Samuel},
  title     = {Calibration-Conditioned {FiLM} Decoders for Low-Latency Decoding of Quantum Error Correction Evaluated on {IBM} Repetition-Code Experiments---Datasets},
  publisher = {Zenodo},
  year      = {2026},
  version   = {0.1},
  doi       = {10.5281/zenodo.20768087},
  url       = {https://zenodo.org/records/20768087},
  note      = {Pacific Northwest National Laboratory; CC BY 4.0}
}

@article{huang2020shadows,
  author  = {Huang, Hsin-Yuan and Kueng, Richard and Preskill, John},
  title   = {Predicting Many Properties of a Quantum System from Very Few Measurements},
  journal = {Nature Physics},
  year    = {2020},
  volume  = {16},
  number  = {10},
  pages   = {1050--1057},
  doi     = {10.1038/s41567-020-0932-7}
}

@article{zecchin2026escd,
  author  = {Zecchin, Matteo and Simeone, Osvaldo and Ramdas, Aaditya},
  title   = {Universal Sequential Changepoint Detection of Quantum Observables via Classical Shadows},
  year    = {2026},
  eprint  = {2602.11846},
  archivePrefix = {arXiv},
  primaryClass  = {quant-ph},
  note    = {Preprint, v1}
}

@article{fanizza2023quantumchange,
  author  = {Fanizza, Marco and Hirche, Christoph and Calsamiglia, John},
  title   = {Ultimate Limits for Quickest Quantum Change-Point Detection},
  journal = {Physical Review Letters},
  year    = {2023},
  volume  = {131},
  number  = {2},
  pages   = {020602},
  doi     = {10.1103/PhysRevLett.131.020602}
}

@article{hiai2009symmetry,
  author  = {Hiai, Fumio and Mosonyi, Mil{\'a}n and Hayashi, Masahito},
  title   = {Quantum Hypothesis Testing with Group Symmetry},
  journal = {Journal of Mathematical Physics},
  year    = {2009},
  volume  = {50},
  number  = {10},
  pages   = {103304},
  doi     = {10.1063/1.3234186},
  eprint  = {0904.0704},
  archivePrefix = {arXiv},
  primaryClass  = {quant-ph}
}

@article{goldstein2018symmetry,
  author  = {Goldstein, Moshe and Sela, Eran},
  title   = {Symmetry-Resolved Entanglement in Many-Body Systems},
  journal = {Physical Review Letters},
  year    = {2018},
  volume  = {120},
  number  = {20},
  pages   = {200602},
  doi     = {10.1103/PhysRevLett.120.200602},
  eprint  = {1711.09418},
  archivePrefix = {arXiv},
  primaryClass  = {cond-mat.stat-mech}
}

@article{helstrom1969detection,
  author  = {Helstrom, Carl W.},
  title   = {Quantum Detection and Estimation Theory},
  journal = {Journal of Statistical Physics},
  year    = {1969},
  volume  = {1},
  number  = {2},
  pages   = {231--252},
  doi     = {10.1007/BF01007479}
}

@article{wilson1974confinement,
  author  = {Wilson, Kenneth G.},
  title   = {Confinement of Quarks},
  journal = {Physical Review D},
  year    = {1974},
  volume  = {10},
  number  = {8},
  pages   = {2445--2459},
  doi     = {10.1103/PhysRevD.10.2445}
}

@article{kitaev2003anyons,
  author  = {Kitaev, A. Yu.},
  title   = {Fault-Tolerant Quantum Computation by Anyons},
  journal = {Annals of Physics},
  year    = {2003},
  volume  = {303},
  number  = {1},
  pages   = {2--30},
  doi     = {10.1016/S0003-4916(02)00018-0},
  eprint  = {quant-ph/9707021},
  archivePrefix = {arXiv}
}
```

## 13. Final claim checklist

Before submission, every item must be answerable “yes.”

- [ ] Does the abstract call the Google onset approximate rather than exact?
- [ ] Does it call PNNL a constructed snapshot-cohort boundary?
- [ ] Does it reserve exact e-validity for the declared randomized design?
- [ ] Does it distinguish e-process lifetime control from e-detector ARL?
- [ ] Does one complete paired shot equal one formal update?
- [ ] Are DFR and same-feature logistic both included in the advantage gate?
- [ ] Are detector-likelihood, covariance and CUSUM controls fully reported?
- [ ] Were detector artifacts frozen before decoder outcomes were opened?
- [ ] Are event and PNNL uncertainty units clustered correctly?
- [ ] Are negative or tied results reported without a sparsity override?
- [ ] Is eSCD described as unavailable from fixed syndrome records?
- [ ] Are Helstrom, Wilson, likelihood-oracle and true-feature ceilings stated?
- [ ] Are quantum speedup, universal sample efficiency, string theory and
      holography absent from the claims?

Passing this checklist does not prove an advantage. It ensures that whatever
the locked experiments show can be stated without crossing the statistical,
physical, or data-provenance boundary.
