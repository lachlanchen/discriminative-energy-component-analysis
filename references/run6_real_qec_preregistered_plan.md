# Run 6 preregistration: real-QEC contrast monitoring

**Status:** executable lock frozen before held-value access; no Google 2022
detection-event values or decoder outcomes were inspected before freeze
**Date:** 2026-07-27
**Primary evidence class:** real hardware with an author-identified,
approximately located event
**Secondary evidence class:** constructed boundaries between real hardware
snapshot cohorts

## 1. Scientific question

At the same calibration, observation and false-alert budget, does a
predictable maximum-difference correlation witness improve real QEC anomaly
detection or logical-risk triage over:

- detector firing rate;
- detector-likelihood surprise;
- covariance/Hotelling monitoring; and
- an online logistic witness using the identical feature bank?

The primary candidate is the paired, global-plus-correlation form of S-PACE
defined in `references/run6_space_final_theory.md`. The protocol is designed
to return a negative result if a simple rate or same-feature comparator is
as good.

## 2. Data roles fixed before scoring

### 2.1 Primary held event

Dataset:

- Google Quantum AI Team, Zenodo 6804040;
- archive `google_qec3v5_experiment_data.zip`;
- MD5 `a7fd8b481c3087090093106382dc217d`;
- experiment
  `repetition_code_bZ_d25_r50_center_5_5`;
- 500,000 shots, distance 25, 50 syndrome rounds, 1,224 detectors;
- CC BY 4.0.

The archive README independently identifies a “huge cluster of mismatches
near shot 57775” as the high-energy event discussed in the paper. This
statement was read before the detector values. It is an approximate
author-identified event, not a machine-readable physical-onset label.

The immutable index split is:

| Role | Zero-based shot indices | Use |
|---|---:|---|
| null pairing/threshold validation A | `[0, 10000)` | first half of a null pair |
| null pairing/threshold validation B | `[10000, 20000)` | second half of a null pair |
| held reference reservoir | `[20000, 40000)` | one stored reference shot per test shot |
| held monitored stream | `[40000, 60000)` | contains the author-identified event |
| untouched future stability data | `[60000, 500000)` | not opened until the primary table is frozen |

The primary event window is

\[
[57750,57800)
\]

in archive shot coordinates. Sensitivity windows are
`[57770, 57780)` and `[57725, 57825)`. The event window will not be moved
after scores are viewed. The README's localization command uses
`grep -n`, whose line 57,775 is stored zero-based shot 57,774; its prose says
“near shot 57775.” Both zero-based candidates 57,774 and 57,775 lie inside
every declared window, so the ambiguity cannot change the decision rule.

### 2.2 PNNL/IBM pilot and auxiliary arm

The already viewed Kingston cohorts are pilot-only. They may set generic
timescale and cap grids but may not supply fitted state, select a grid member
or appear as held-out advantage evidence.

The PNNL/IBM v0.1 release is an auxiliary cross-snapshot/domain-shift arm.
Physical paths are reconstructed from each state-specific QASM, never from
the register suffix alone. Results are labeled:

> constructed boundary between real-hardware snapshot cohorts.

They are not called natural temporal drift. Long-span cohorts with different
complete QASM are explicitly labeled circuit-and-hardware domain shifts.
Only byte-identical-QASM comparisons can be described as circuit controlled.

### 2.3 Exact randomization arm

An exact-design validation uses real observed pair values but independently
randomizes the A/B orientation of each complete paired shot before scoring.
Conditional on the unordered pairs, antisymmetric scores then have mean zero.
This validates the implementation and type-I statement under the declared
randomization; it is not a natural hardware-null claim.

## 3. Observation unit and pairing

The primary causal update is one QEC cycle within a shot. The 1,224 detector
bits are reordered using detector coordinates from `circuit_ideal.stim` into
51 round roles by 24 physical check coordinates. Shot boundaries and round
roles are retained.

The parser must read every `DETECTOR(x, y, t)` declaration in global Stim
declaration order and construct the permutation from the coordinate tuple,
not reshape the 1,224 packed bits and assume a fixed column order. At the
boundary roles \(t=0\) and \(t=50\), the declarations for spatial rows
3, 5 and 7 run in the reverse order from roles \(t=1,\ldots,49\). The
canonical check order is increasing spatial coordinate within each declared
check stratum; parser tests must verify that every role contains the same
24 unique canonical checks exactly once.

Held test shot \(i\in[40000,60000)\) is paired one-to-one with reference shot

\[
20000+(i-40000).
\]

Within a pair, cycle role \(r\) is matched only to the same role \(r\).
This gives 20,000 paired shots and

\[
20,000\times51=1,020,000
\]

cycle updates. A complete pair consumes two archived physical shots; both
are included in the resource ledger.

Validation uses the fixed one-to-one pairs

\[
i\in[0,10000)
\quad\leftrightarrow\quad
10000+i.
\]

These validation pairs have two disjoint roles:

- pair indices `[0, 5000)` are the common fit/warm-up block;
- pair indices `[5000, 10000)` are the empirical threshold block.

All methods may see the same 5,000 fit pairs (10,000 physical archived
shots). M1/M2 fit their fixed null controls there; M3--M5 perform ordinary
score-before-update warm-up there. The post-warm-up state is cloned twice.
One clone processes the threshold block; the other starts the held replay.
Threshold-block state is never carried into held monitoring.

Every adaptive M3--M5 state is separate for each of the 51 detector round
roles and persists across shots. A role state is updated once per shot and
never by another role. Thus every declared half-life is measured in updates
of that same role, not in the flattened 51-role stream.

No random pairing seed may be selected by event performance. A separate
random-swap seed list is predeclared for exact-design validation.

Archived shot order is preserved inside each block. The analysis is a causal
replay of archived order, not proof of a known wall-clock cadence.

## 4. Locked bounded features

For one cycle, let

\[
e\in\{0,1\}^{24},
\qquad
z=1-2e\in\{-1,+1\}^{24}.
\]

### Global branch

\[
g(e)=\frac1{24}\sum_{j=1}^{24}e_j\in[0,1].
\]

### Sparse local/correlation branch

Use the 300-dimensional bounded vector

\[
\phi(e)
=
\left(
e_1,\ldots,e_{24},
\left\{
\frac{1+z_iz_j}{2}
\right\}_{1\le i<j\le24}
\right)
\in[0,1]^{300}.
\]

All 276 check pairs are included. This avoids selecting a physical adjacency
graph after viewing the event. A local-neighbor-only bank may be reported
only as a predeclared metadata-derived ablation.

### Spectral AOC branch

\[
R(e)=\frac{zz^{\mathsf T}}{24}
\]

is positive semidefinite with unit trace. For a paired reference/monitor
cycle,

\[
D_t^\phi=\phi(e_t^{B})-\phi(e_t^{A}),
\qquad
D_t^R=R(e_t^{B})-R(e_t^{A}).
\]

The global, sparse and spectral branches receive exactly the same detector
bits. Observable flips and decoder predictions are unavailable to every
detector and are opened only for the frozen downstream utility audit.

## 5. Frozen methods

### M0: detector firing rate

Use the paired difference

\[
s_t^{\mathrm{DFR}}=g(e_t^B)-g(e_t^A)\in[-1,1]
\]

with a two-sided fixed bet mixture. Also report the conventional unpaired
cycle and shot DFR thresholds calibrated on the validation block.

### M0C: within-shot Page--CUSUM

As a mandatory empirical control, run the two-sided Page--CUSUM specified in
`references/run6_method_lock_recommendations.md` on the global DFR and all
24 raw paired check differences, with drift allowances
\(\{0.01,0.05,0.1\}\). Its state resets at every shot boundary, so it is a
within-shot scan and not an e-factor or an oracle quickest-change procedure.

### M1: diagonal detector-likelihood surprise

Estimate per-round-role, per-check Bernoulli probabilities by pooling both
sides of only the 5,000 fit pairs, with fixed Jeffreys smoothing and clipping.
Normalize the monitor-minus-reference mean NLL difference by
\(\log(0.9999/0.0001)\), yielding a bounded score. This is a model-aware
diagonal control and receives the same round/check data.

### M2: shrinkage covariance/Hotelling monitor

Fit role means and one pooled Ledoit--Wolf precision to the exact
PCG64(610601) role-stratified 20,000-observation subsample of the fit block.
Use an empirical role-centered quadratic score and the disjoint threshold
block; do not call it an e-factor.

### M3: same-feature online logistic witness

At update \(t\), use weights learned only from earlier reference/monitor
pairs. The bounded antisymmetric score is

\[
s_t^{\mathrm{logit}}
=
\tanh\!\left[
\frac{
w_{t-1}^{\mathsf T}
(\phi(e_t^B)-\phi(e_t^A))
}{2c}
\right].
\]

Use zero initialization, no intercept, L2 \(10^{-4}\), and three fixed
learning-rate experts \(\{0.001,0.01,0.1\}\) with uniform prior. The Kingston
pilot motivates this grid only and selects no expert. One overflow-safe
pairwise-logistic SGD step occurs only after scoring the current pair. This is
the named same-information comparator.

### M4: S-PACE sparse analytical witness

Maintain past-only exponentially weighted contrasts for half-lives

\[
\{4,16,64,256\}\ \text{updates of the same round-role state}.
\]

For each half-life and

\[
k\in\{1,4,16,64\},
\]

select the signed top-\(k\) coordinates analytically and assign equal weight.
Use bet fractions

\[
\beta\in\{0.1,0.3,0.6,0.9\}.
\]

Use decay \(2^{-1/h}\), zero initialization, no EWMA bias correction, and
the fixed absolute-magnitude/feature-index tie rule. All half-life, \(k\) and
\(\beta\) components have fixed uniform prior mass. No component is chosen
from the held event.

### M5: S-PACE spectral AOC

Maintain past-only exponentially weighted operator contrast for half-lives
\(\{4,16,64\}\) updates of the same role. Recompute after updates 8, 16,
24, ... and first use the new effect on the following observation. Report
both the full positive eigenspace and the deterministic degenerate-rank-one
variant. The current pair is scored before every update.

The proposed method \(S\) is fixed before held replay: one-half prior mass on
M4 and one-half on M5, uniform inside each branch. The empirical cycle score
is \(\max\{Z^{(4)},Z^{(5)}\}\). The better observed branch may never replace
this composite.

### Oracle labels

No post-event-trained logistic or likelihood model participates in the
primary advantage decision. If reported, a classifier fitted with event or
decoder-error labels is named **post hoc oracle** and appears only as a
privileged ceiling.

Wilson-loop diagnostics are not a matched comparator for this one-dimensional
repetition-code experiment. Classical-shadow eSCD is not runnable from fixed
syndrome bits and will not be represented by an invented proxy.

The implementation-complete formulas, component ordering, tie rules,
checkpoint semantics, numeric policy and schemas in
`references/run6_method_lock_recommendations.md` are normative and are pinned
by hash in the final freeze manifest.

## 6. Sequential accumulation and alarms

### Primary finite-horizon guarantee

Use the proper-prior e-process with a uniform start prior over the declared
20,000-complete-paired-shot held episode and threshold

\[
\alpha=0.01,\qquad E_t\ge100.
\]

Every internal method mixture includes its fixed component weights and an
explicit uniform \(1/51\) prior over role-specific experts. The 51 roles that
share one complete-shot orientation are scored from their pre-shot states and
mixed in one e-process update; they are never compounded as 51 independent
time steps. The mathematical guarantee is claimed only for the exact
paired/randomized null, not automatically for stationary hardware.

### Secondary ARL mode

Use the mixture Shiryaev--Roberts e-detector with

\[
\gamma=10^6\ \text{complete paired shots}.
\]

Report the \(L_t\equiv1\) clock behavior and do not translate this threshold
into a lifetime false-alarm probability.

### Empirical operating points

Thresholds use only the 5,000-pair threshold clone. The primary point is:

- at most one notification per 100,000 paired cycle-role opportunities;
- no more than two validation notifications among 255,000 opportunities.

Use strict `score > threshold`, suppress additional notifications only for
the remainder of the same shot, continue every model update, and use no
next-shot cooldown or witness reset. Select the smallest observed-or-infinite
threshold whose replay alert count is within budget. The secondary
one-per-10,000-shot point permits zero alerts in 5,000 threshold shots and
therefore uses the maximum validation shot score. Report the complete
frontier.

## 7. Primary endpoints

### Detection

- whether each method alarms inside the primary event window;
- first alarm shot and within-shot QEC cycle;
- false alarms before the event;
- event score rank among all held monitored shots;
- sensitivity to the two predeclared wider/narrower event windows.

Because the onset label is approximate, an exact natural detection-delay
claim is forbidden. One event cluster also cannot identify a population miss
probability; the random-swap audit estimates design-based false-alarm
behavior, not natural-event power.

### Logical-risk triage

After all detector outputs are frozen, open:

- `obs_flips_actual.01`;
- correlated-matching predictions; and
- PyMatching predictions.

For each method, aggregate its causal cycle score to a shot alert and report:

- decoder mismatches captured at fixed validation-calibrated alert rates;
- logical-error/mismatch rate among retained shots;
- coverage or valid-computation fraction;
- risk-coverage and alert-budget curves.

This is a retrospective veto/triage analysis, not a new decoder.

The primary downstream label is actual observable flip XOR the
correlated-matching prediction. Freeze detector scores first, rank the 20,000
held monitor shots by descending shot score and then ascending archive index,
and count mismatches among the first 20 shots. PyMatching is a secondary
replication. Detector labels cannot affect a rank or tie.

### Resources

Report:

- reference and monitored shots;
- QEC cycles and detector bits processed;
- features retained;
- update frequency;
- wall time, peak memory and output size;
- whether an eigendecomposition, covariance inverse or labeled update is
  required.

## 8. Statistical uncertainty

- The high-energy event is one event cluster; adjacent cycles are not
  independent replications.
- Use shot/block resampling for null thresholds and risk-coverage uncertainty.
- Use exact random-swap replication for the design-based type-I audit.
- Treat PNNL paths/snapshots, not individual syndrome bits, as the
  independent units in cross-cohort summaries.
- Do not report a narrow cycle-level confidence interval around one event as
  population-level evidence.

## 9. Locked advantage rule

Define `google_primary_pass = true` only if:

1. the fixed M4/M5 composite alarms inside the primary event window at the
   primary validation-locked threshold;
2. it emits at most nine pre-event alerts;
3. detector artifacts and the resource ledger were hashed before outcomes
   were opened;
4. its top-20 correlated-decoder mismatch capture is at least paired DFR
   capture plus one;
5. its capture is at least same-feature online-logistic capture plus one;
6. all fixed event-window sensitivity and uncertainty summaries are
   reported; and
7. no method receives extra detector records or outcome labels.

M0C, M1 and M2 are mandatory contextual controls but cannot rescue a failure.
If one exceeds S-PACE, “best overall method” is forbidden.

Define

\[
\texttt{overall\_run6\_advantage}
=
\texttt{google\_primary\_pass}
\land
\texttt{pnnl\_retention\_pass}.
\]

PNNL is the only locked auxiliary; the untouched Google future cannot be
chosen after results as a replacement. Calibration records and all compute
are reported per method even when model complexity differs.

Failure of any item forces:

> no demonstrated S-PACE algorithmic advantage.

A tie, an interpretable sparse witness or a successful exact-design
randomization audit does not override this rule.

## 10. Claim boundary

Permitted:

- real Google hardware causal replay with an approximately located
  author-identified event;
- exact design-based validation under randomized pair orientation;
- constructed real-hardware cohort shifts for PNNL;
- empirical hardware false alerts and downstream risk-coverage;
- conditional mathematical guarantees under the stated null.

Forbidden:

- exact natural changepoint onset or wall-clock delay;
- exact hardware e-validity from untested stationarity;
- natural PNNL temporal drift;
- faithful eSCD/classical-shadow performance;
- quantum acceleration or universal sample efficiency;
- superiority to Helstrom, Wilson, a correct likelihood ratio or an oracle
  classifier using the true changed feature;
- a new topological, string-theory or holographic result.

## 11. Allowed implementation and paper files

Implementation:

- `experiments/aoc/space.py`
- `experiments/aoc/qec_real.py`
- `experiments/run6/**`
- `experiments/pyproject.toml` only for Run 6 test registration/versioning

Publication, only after results are locked:

- `publication/run6/**`
- root and directory README/version/citation metadata needed to expose Run 6

Runs 1--5 remain immutable.

## 12. Freeze procedure

Before reading `detection_events.b8` values:

1. finish the metadata/data-map audit;
2. theorem-check the S-PACE final note;
3. serialize this plan into locked JSON configurations;
4. record file hashes, index splits, feature definitions, seeds and package
   versions;
5. commit and push the preregistration unit.

Only then may the primary held event stream and decoder outcomes be opened.
Any later deviation appears in a deviation ledger and cannot silently change
the primary advantage rule.
