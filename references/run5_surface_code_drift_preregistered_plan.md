# Run 5 preregistered plan: accessible-observable surface-code drift

## 1. Status and scope

- Created: 2026-07-27, before Run 5 implementation or locked-test execution.
- Theory refinement: 2026-07-27, before pilot or locked-test execution.
- Working title: **What Syndrome Data Can and Cannot Detect:
  Accessible-Observable Sequential Change Detection for Topological Quantum
  Error Correction**.
- Code target: `experiments/aoc/` and a new immutable
  `experiments/run5/`.
- Paper target: a new `publication/run5/`; Runs 1--4 remain unchanged.
- Primary practical question:

  > At the same syndrome-round, calibration, and false-alarm budgets, can a
  > structured observable recover correlation-only surface-code noise drift
  > that rate/count observables cannot see, and can any detection gain be
  > converted into lower decoder logical error?

This plan deliberately separates an exact controlled detector-error model
from a secondary circuit-level Stim/PyMatching validation. The former supports
exact no-go statements; the latter tests whether the method remains useful
under realistic stabilizer-extraction noise. Neither is called hardware data.

## 2. Prior-art and novelty boundary

The following are foundations, not claimed discoveries:

- online syndrome-based noise estimation and decoder adaptation;
- detector-correlation and decoding-graph reweighting;
- covariance change detection, Fisher/Hotelling whitening, CUSUM, and
  Shiryaev--Roberts procedures;
- e-detectors and anytime-valid average-run-length control;
- classical-shadow sequential quantum change detection;
- toric/surface codes, detector error models, Wilson loops, and local
  indistinguishability.

Run 5 will not claim the first QEC drift detector, the first correlation-aware
decoder, a new covariance test, quantum acceleration, universal sample
efficiency, superiority over an oracle likelihood ratio, or any new
string-theory/holography result.

The proposed contribution is the reproducible synthesis of:

1. an explicit hierarchy of observable information;
2. exact pushforward no-go certificates at each level;
3. a symmetry/locality-constrained, interpretable change witness;
4. a predictable bounded score with an average-run-length guarantee when the
   null state is known exactly;
5. physically matched auxiliary access for one named, predeclared logistic
   comparator plus separately labeled model-aware controls;
6. a decoder-utility and circuit-level audit.

## 3. Observable hierarchy and exact no-go theorem

Let \(Y_t\) denote one syndrome/detector round, \(C_t=h(Y_t)\) its total
detector count, and \(A_t\) an optional logical/Wilson audit. The experiments
separate

\[
\sigma(C_t)
\subset
\sigma(Y_t)
\subset
\sigma(Y_{1:t})
\subset
\sigma(Y_{1:t},A_t).
\]

**Accessible-process pushforward no-go.** Let \(P_0^{1:\infty}\) and
\(P_1^{1:\infty}\) be the two laws on complete observation paths and let
\(H(Y_{1:\infty})=(h(Y_1),h(Y_2),\ldots)\). If
\[
H_\#P_0^{1:\infty}=H_\#P_1^{1:\infty},
\]
then every randomized stopping rule adapted only to
\(h(Y_1),h(Y_2),\ldots\) has the same law under \(P_0\) and \(P_1\). Its
change-detection probability therefore cannot exceed its false-alarm
probability at any fixed horizon. Equality of only the one-round pushforwards
\(h_\#P_0^{(1)}=h_\#P_1^{(1)}\) is sufficient when the accessible rounds are
iid under both hypotheses, but it is not sufficient when temporal dependence
can differ.

The code and paper must test equality of the *entire* count distribution, not
only its mean.

Three controlled separations will be implemented:

1. **Spatial correlation drift.** Detector marginals and the full one-round
   count distribution are identical, while pair topology changes.
2. **Temporal correlation drift.** The full one-round syndrome distribution
   is identical, while lag correlations change. Nonoverlapping two-round
   blocks avoid an invalid independence claim.
3. **Logical-loop drift.** A homologically noncontractible closed error has
   zero syndrome boundary, so the complete syndrome history is pathwise
   identical. Only the separately costed logical/Wilson audit can detect it.

## 4. Exact controlled detector model

The primary exact model is a periodic \(L\times L\) check lattice with
\(m=L^2\) binary detectors. In one round:

1. \(J\sim\mathrm{Bernoulli}(\eta)\);
2. if \(J=1\), select a uniformly translated/oriented straight chain whose
   length is one with probability \(1-q\) and two with probability \(q\);
3. record the two chain endpoints as the clean syndrome;
4. pass every detector through independent binary readout noise
   \(\epsilon\).

Primary parameters are \(L=5\), \(\eta=0.65\), \(\epsilon=0.03\), and
\(q_0=0.35\). Spatial alternatives are
\(q_1\in\{0.45,0.55,0.65\}\), with \(L\in\{5,7,9\}\) used for scaling.

The construction must verify analytically and numerically that:

\[
\Pr(Y_i=1)=\epsilon+(1-2\epsilon)\frac{2\eta}{m}
\]

is independent of \(q\), and that the full count distribution is independent
of \(q\). The exact one-round likelihood is a finite mixture over the clean
zero/two-endpoint templates and is evaluated without enumerating
\(2^m\) bit strings.

For \(z_i=1-2y_i\), define

\[
g_\ell(y)=\frac1{2m}\sum_{v,o}z_vz_{v+\ell e_o}.
\]

The exact BSC likelihood can be reduced to

\[
P_q(y)=B_0(y)\left\{(1-\eta)+
\eta[(1-q)K_1(y)+qK_2(y)]\right\},
\]

where each \(K_\ell\) is an affine combination of
\((1,\bar z,g_\ell)\). Thus \((\bar z,g_1,g_2)\) is sufficient for the
spatial \(q\)-family: translation reduction loses no likelihood information
in this controlled model. This statement is model-specific and will be
verified against direct template summation.

The exact population pair-feature gap is

\[
\left(\mathbb E_{q_1}g_1-\mathbb E_{q_0}g_1,\,
\mathbb E_{q_1}g_2-\mathbb E_{q_0}g_2\right)
=
\frac{2\eta(1-2\epsilon)^2}{m}(q_1-q_0)(-1,+1).
\]

For temporal drift, the chain-length state retains stationary probability
\(q_0\) but changes from independent draws to a stationary two-state Markov
chain with persistence
\(\kappa\in\{0.50,0.75,0.90\}\). Exact alternative likelihoods use independent
nonoverlapping two-round blocks; they are strong but intentionally omit
cross-block alternative correlations.

If \(a(y)\) is the posterior conditional expectation of the standardized
latent length state under the one-cycle null, the exact adjacent-pair
likelihood ratio against the independent null is

\[
L_\kappa(y,y')=1+\kappa a(y)a(y').
\]

This gives a closed-form pair-restricted oracle and exact null e-factor. It
must not be called the full alternative HMM likelihood; a forward-filter HMM
is retained as the oracle ceiling.

## 5. Observable-contrast methods

Encode \(z_t=1-2Y_t\) and

\[
R_t=\frac{z_tz_t^\top}{m},\qquad \operatorname{Tr}R_t=1.
\]

The raw AOC effect is the positive support of the projected difference
between a predictable live state estimate and the null state. Translation
twirling turns the spatial correlation state into a block-circulant object;
the implementation should use displacement features/FFT structure rather
than dense eigendecomposition in the scaling study.

### Variance/Kelly-aware accessible contrast

Raw trace contrast maximizes a bounded expectation difference; it is not a
quickest-change objective. For an accessible feature bank \(g(R)\), define

\[
\widehat\delta_{t-1}
  =\widehat{\mathbb E}_{t-1}g-\mu_0,\qquad
w_{t-1}=(V_0+\lambda I)^{-1}\widehat\delta_{t-1}.
\]

After deterministic scaling to make the centered score
\(s_t\in[-1,1]\), use

\[
L_t(\beta)=1+\beta s_t,\qquad
R_t(\beta)=(R_{t-1}(\beta)+1)L_t(\beta),
\]

for a fixed grid
\(\beta\in\{0.02,0.05,0.1,0.2,0.4,0.8\}\), and mix the \(R_t(\beta)\)
uniformly. With an exact null mean and a predictable witness,
\(\mathbb E_0[L_t(\beta)\mid\mathcal F_{t-1}]=1\), so threshold
\(\gamma\) gives average run length at least \(\gamma\).

The unconstrained solution is ordinary Fisher/Hotelling whitening and is
credited as such. Any Run 5 value must come from the declared
symmetry/operator restriction, interpretability, predictability, or downstream
QEC utility.

Locally, the variance-aware direction is the least-squares projection of the
likelihood score onto the accessible feature span. In the unrestricted
operator space the analogous equation is the established symmetric
logarithmic derivative/quantum-Fisher construction. Run 5 therefore claims a
restricted sequential synthesis, not new Fisher, Hotelling, SLD, or
quantum-Fisher mathematics.

The decision criteria are explicitly different:

- raw AOC/Helstrom maximizes a bounded one-shot mean gap/total variation;
- a likelihood e-factor maximizes Kelly expected log growth/KL information;
- variance-aware AOC is a regularized local projection toward that likelihood
  score.

No theorem says raw AOC must minimize detection delay.

Variants that replace the analytic null mean by a finite-sample null-mean
estimate do **not** automatically inherit the exact theorem; those variants
would require a separate false-alarm calibration or uncertainty argument. In
the locked design here, however, the null simplex mean is analytic.
Covariance/ridge selection and logistic-direction fitting use independent
auxiliary data only to choose an effect. Conditional on those frozen
auxiliary data, the effect is fixed or predictable and its score remains
centered at the exact analytic null, so the bounded-score SR ARL theorem is
preserved. Independent no-change streams audit the guarantee; they do not
choose its threshold.

## 6. Locked comparison set

Every syndrome method sees identical streams and the same calibration budget.

### Detection baselines

1. detector-count/DFR CUSUM or SR (provably blind control);
2. detector mean-vector scan;
3. matched signed displacement-correlation witness;
4. shrinkage covariance/Hotelling scan;
5. spectral projected second-moment detector;
6. RBF-Hamming MMD window scan;
7. direct observable-bank e-detector with fixed and adaptive weights;
8. raw AOC and variance-aware AOC on the same features;
9. a fixed **validation-trained linear logistic effect** on the identical
   Fourier simplex/pair-simplex, affinely mapped into \([0,1]\), centered at
   the analytic null, and run through the same bounded-score SR recursion;
10. exact \(q\)-grid or \(\kappa\)-grid likelihood-ratio SR;
11. known-post likelihood CUSUM as the oracle ceiling;
12. logistic regression and RBF SVM on identical fixed windows as offline
    diagnostics, not sequential guarantees.

Item 9 is the named non-oracle, model-agnostic sequential comparator for the
two primary paired tests. It was chosen after inspecting the independent
offline diagnostic audit but before the sequential locked test. It is not
claimed to be the strongest possible generic baseline. The matched signed
witness, exact likelihood grid, and known-post likelihood are model-aware
controls/ceilings and must not be substituted for this comparator.

### Measurement-policy audit

A secondary diagonal-state experiment will implement genuine local-Clifford
classical-shadow snapshots. A \(ZZ\) pair estimator uses the appropriate
inverse-channel factor. It will be compared per state copy with an all-\(Z\)
measurement matched to native syndrome readout. A loss from shadows is
reported as the cost of measurement universality, not as evidence that AOC
beats eSCD.

### Decoder baselines

The controlled DEM and circuit-level validation will compare, where supported:

1. a static mismatched MWPM decoder;
2. an oracle decoder built from the true post-change DEM;
3. a correlation/DEM-reweighted decoder learned from an independent window.

The paper will cite stronger published adaptive methods (DGR, sliding-window
adaptive DEM, differentiable likelihood methods) and will not falsely label a
lightweight reweighting reproduction as their full implementation.

## 7. Fixed budgets and success criterion

### Controlled exact arm

- null state/mean: analytic and shared by all eligible methods;
- every calibration and horizon field is measured in **physical syndrome
  cycles**, not detector updates;
- covariance calibration: 4096 independent physical cycles per family,
  yielding 4096 spatial one-cycle samples and 2048 temporal nonoverlapping
  pair samples;
- ridge validation: 8 independent fresh-start changed streams per family,
  with 512 physical cycles per stream, hence 4096 alternative-validation
  cycles per family;
- validation-trained comparator: 8 independent labeled streams per class,
  512 physical cycles per stream, hence 4096 physical cycles per class in both
  families (4096 spatial one-cycle examples and 2048 temporal pair examples);
- locked test: 512 null streams of horizon 5000 and 256 changed streams per
  scenario, change time 256 physical cycles, post-change horizon 1024 physical
  cycles;
- target ARL: 1000 physical cycles; the bounded-score threshold is 1000
  spatial one-cycle updates and 500 temporal pair updates;
- scaling: \(L\in\{5,7,9\}\), at least 128 paired repetitions;
- common random numbers across methods;
- fixed, disjoint seed partitions: 510001--510002 calibration,
  511001--511004 labeled-baseline training, 520000- and 521000-series ridge
  validation, 600000-series spatial locked test, and 20000000-series temporal
  locked test. The executable configuration validates all consumed seed
  intervals before running.

Thus vAOC receives 4096 null covariance-calibration cycles and 4096
alternative ridge-validation cycles; the logistic comparator receives 4096
null and 4096 alternative training cycles with the same \(8\times512\) stream
structure. The two families consume different numbers of detector updates
because a temporal update costs two physical cycles. No cross-family
sample-efficiency claim is permitted from those update counts. Locked-test
comparisons within a family use identical paired streams and physical-cycle
budgets.

If the full locked design is computationally prohibitive, any reduced run must
be explicitly marked a pilot and cannot support the primary named-comparator
claim.

### Metrics

- restricted mean detection delay through 1024 post-change rounds;
- median/IQR, miss fraction, and detection within 64/128 rounds;
- empirical null run length with censoring reported explicitly;
- offline ROC AUC;
- witness overlap/localization;
- runtime and peak memory;
- logical error rate before and after decoder adaptation.

For the middle spatial effect \(q_1=0.55\) and middle temporal effect
\(\kappa=0.75\), compare vAOC directly with the frozen validation-trained
linear logistic effect on paired restart delays. Let
\(D_i=T_i^{\rm vAOC}-T_i^{\rm logit}\). Since both restricted delays lie in
\([0,H]\), independent paired differences satisfy \(D_i\in[-H,H]\). Under
\(H_0:\mathbb E D_i\ge0\), report the distribution-free Hoeffding p-value

\[
p_{\rm H}=\min\!\left\{1,\exp\!\left[
-\frac{n\max(0,-\bar D)^2}{2H^2}\right]\right\},
\]

and apply Holm adjustment across the two hypotheses. The simultaneous
familywise-\(1-\alpha\) upper bounds use Bonferroni:

\[
U_{\rm H}=\bar D+
H\sqrt{\frac{2\log(m/\alpha)}{n}},\qquad m=2.
\]

These statements are conditional on the frozen auxiliary fits and require
independence across paired locked-test replicate streams and bounded
restricted delays, but no sign symmetry or parametric delay model.
A paired percentile-bootstrap interval is also reported descriptively; it is
not used for the comparison decision.

**Support for the named comparison is declared only if** the upper
simultaneous 95% confidence limit for

\[
\mathrm{RMDD}_{\mathrm{vAOC}}
-
\mathrm{RMDD}_{\mathrm{named\ logistic\ comparator}}
\]

is below zero, the corresponding Holm-adjusted one-sided Hoeffding
\(p\)-value is below 0.05, and the false-alarm/ARL requirement is met. A
repo-level overall comparison flag additionally requires **both**
preregistered middle-effect hypotheses to pass. Even a pass supports only
vAOC versus this named logistic comparator on these two controlled tasks; it
does not establish strongest-baseline or general algorithmic superiority. The
exact likelihood and known-post oracle are ceilings, not targets that must be
beaten.
Failure of this criterion is a publishable negative result and must not be
hidden.

## 8. Circuit-level validation

Install pinned optional dependencies for Stim and PyMatching. Generate
rotated surface-code memory circuits at distances \(3,5,7\), extract detector
error models, sample detector events and logical observables with Stim, and
decode with PyMatching. The exact injected drift and matched-budget protocol
will be frozen after verifying the available API and before running locked
seeds.

Generated example circuits are treated as circuit-level stabilizer simulation,
not hardware data or a production-complete device model. The publication must
record code distance, rounds, every physical noise parameter, Stim/PyMatching
versions, shot counts, seeds, and whether each result came from a circuit
sampler or a phenomenological DEM sampler.

## 9. Required tests

- exact detector marginal and full count-distribution invariance;
- normalized emission likelihood and null likelihood-ratio mean one;
- stationary Markov construction and exact one-cycle equality;
- closed-loop boundary equals zero and syndrome histories are identical;
- displacement twirl equals explicit translation average on small lattices;
- raw and variance-aware scores are bounded;
- learned witnesses are predictable (no current/future sample leakage);
- SR recursion satisfies null expectation identities in Monte Carlo;
- wrong rotational twirl erases an orientation drift (negative control);
- Stim-generated detector samples have expected shapes and PyMatching decodes
  their logical observables;
- result manifests contain seeds, dependency versions, commands, hashes, and
  Git state.

## 10. Publication and repository acceptance

Run 5 is complete only when:

1. reusable code and tests live under `experiments/aoc/`;
2. run-specific scripts, raw CSV/JSON, figures, and manifests live under
   `experiments/run5/`;
3. the locked result is summarized in an advantage audit that distinguishes
   theorem, controlled simulation, circuit-level simulation, and conjecture;
4. `publication/run5/main.tex` and `main.pdf` compile with no undefined
   citations/references or overfull boxes and every page is visually audited;
5. Runs 1--4 remain unchanged;
6. the complete unit passes tests/lint, is committed separately, and is pushed
   to `main`.

## 11. Pilot-triggered evaluation amendment (2026-07-27, before locked run)

This section is an explicit amendment made **after inspecting the reduced
pilot and before executing the publication-grade locked run**. It changes the
primary delay estimand described above; the original plan must not be
represented as if it already contained this correction.

The pilot exposed a deterministic property of the exactly blind SR control.
For a zero score, every betting factor is one and

\[
R_t=(R_{t-1}+1)\cdot 1=t.
\]

It therefore alarms at \(t=\gamma\) without receiving evidence. In a
surveillance stream whose changepoint is \(\nu<\gamma\), reporting
\(\gamma-\nu\) as an ordinary detection delay incorrectly credits the age of
the SR clock to the post-change distribution. This behavior is not a failure
of the ARL theorem: the blind control has null run length exactly
\(\gamma\). It is a failure to separate distinct evaluation estimands.

The locked evaluation is amended as follows.

1. The primary paired response is the **restart-at-change restricted mean
   detection delay**. Every detector and predictable witness is reinitialized
   at zero and evaluated on the same post-change segment, with the original
   post-change horizon and threshold unchanged. Ridge selection is performed
   on the corresponding fresh-start validation streams.
2. For each method, the restart delay is also compared with its no-change
   restricted mean run length at the same horizon. The exactly blind control
   should have zero delay reduction even if its deterministic SR clock reaches
   the threshold within the horizon.
3. The original surveillance experiment is retained as a secondary audit.
   Pre-change alarm probability is reported on all streams. Post-change delay
   and miss fraction are computed only among streams satisfying
   \(\tau>\nu\); a pre-change alarm is not recoded as a post-change miss or as
   a horizon-length delay.
4. The independent no-change experiment remains the ARL/censoring audit.
   Its run lengths are not pooled with either changed-stream estimand.
5. Persisted result rows use `stream_type=no_change` rather than the literal
   string `null`, because default CSV parsing in common dataframe software
   interprets `null` as a missing value. The statistical meaning is unchanged.

The temporal full-HMM ceiling is also restricted explicitly to changepoint
candidates at nonoverlapping two-cycle block boundaries. Let
\((y_{2k-1},y_{2k})\) be block \(k\), let \(e_h(y)\) be the emission density
for latent state \(h\), let \(p_0(y)\) be the iid one-cycle null density, and
let \(T\) and \(\pi\) be the post-change transition and stationary
distribution. The exact vector SR update is

\[
u_k(h')
=
\frac{e_{h'}(y_{2k-1})}{p_0(y_{2k-1})}
\left[
\pi_{h'}+\sum_h r_{k-1}(h)T_{hh'}
\right],
\]

\[
r_k(h')
=
\frac{e_{h'}(y_{2k})}{p_0(y_{2k})}
\sum_h u_k(h)T_{hh'},
\qquad
R_k=\sum_{h'}r_k(h').
\]

The injected \(\pi\) term gives every new block-boundary candidate its own
stationary latent filter. Feeding one full-path filter's predictive likelihood
increments into a generic scalar \((R+1)L\) recursion would not be an exact
unknown-changepoint HMM SR procedure. The adjacent-pair marginal likelihood
detector remains a separate, correctly labeled pair-restricted e-detector.

This amendment does not change the model parameters, seeds, calibration
budgets, threshold, alternatives, horizons, or claim boundaries. It changes
which delay statistic is primary and makes the full-HMM candidate resolution
explicit.

## 12. Independent pre-freeze audit amendments (2026-07-27)

This section records two corrections made after the reduced pilot and the
first offline diagnostic execution, but **before** the publication-grade
sequential run and before the corrected locked offline rerun. Earlier result
artifacts are not evidence for the corrected design.

### 12.1 Physical-cycle fairness and locked primary baseline

The original implementation treated `change_time`, `post_change_horizon`,
`null_horizon`, and `calibration_rounds` as one-cycle updates in the spatial
arm but as pair counts in the temporal arm. It therefore silently gave the
temporal arm twice the declared physical-cycle budget. The corrected
configuration replaces those ambiguous fields with explicit `*_cycles`
fields, requires every temporal budget to be even, samples exactly 4096
calibration cycles, changes at cycle 256, observes 1024 post-change cycles,
and uses 5000 no-change cycles. Its temporal detector receives half as many
updates, and its threshold is 500 pair updates for the same 1000-cycle target
ARL.

The audit also found a validation/test seed collision caused by deriving a
temporal validation start as `validation_start + 10000`. Every sampling role
now has a named, disjoint seed partition, and configuration loading rejects
overlapping consumed intervals.

Finally, the matched analytic witness and exact likelihood controls are not
model-agnostic competitors. After inspecting the independent offline
diagnostic audit, and before the sequential locked test, the corrected design
predeclared one named comparator: a fixed linear logistic coefficient learned
from independent labeled middle-effect streams on the same Fourier
simplex/pair-simplex. It is not claimed to be the strongest generic baseline.
The logistic fit uses 8 streams/class by 512 cycles/stream, matching vAOC's
8 by 512 alternative ridge-validation structure and its 4096-cycle null
calibration budget. It is affinely mapped to a valid diagonal effect, centered
at the analytic null, and uses exactly the same bounded-score SR and
family-specific physical-cycle ARL target as vAOC. It is frozen before locked
test streams are generated. Any positive result is limited to vAOC versus
this comparator on the two declared tasks.

### 12.2 Offline objective/label correction

The first offline script used a magnitude-weighted positive part of the
training mean difference for every feature family and labeled every such
direction “Raw AOC positive support.” That operator interpretation is valid
only for the declared commuting probability-simplex representation, and the
raw positive-support optimum there is an **indicator projector**, not a
magnitude-weighted vector.

The corrected locked rerun therefore uses
\(\mathbf 1\{\widehat\mu_1-\widehat\mu_0>0\}\) only on the \(D_4\) Fourier
simplex and its pair-lift simplex. Count sequences, detector first moments,
and translation statistics use an explicitly generic L1-normalized
positive-part mean direction and receive no Helstrom/ECA/AOC optimum label.
The prior v1 offline estimator labels/results are superseded. The complete
method table remains primary; any compact “best” row is selected by
validation AUC and only then reports held-out test performance.
