# Adversarial audit of Run 6 symmetry-scan theory for a PNNL/Google QEC task

**Audit date:** 2026-07-27
**Object audited:** [`run6_symmetry_scan_eprocess_theory.md`](run6_symmetry_scan_eprocess_theory.md)
**Decision standard:** theorem-level validity, operationally matched resources,
and advantage over strong same-information baselines
**Editing scope:** this audit does not modify the theory note

## Executive verdict

The algebraic core of the Run 6 note is correct **under its stated
conditional-mean assumptions**. Fixed convex mixtures of fixed-location
e-processes are valid under arbitrary same-round spatial dependence, and the
note correctly separates probability-of-ever-alarm control from
Shiryaev--Roberts average-run-length (ARL) control.

The main risk is not a flaw in those proofs. It is that the assumption needed
to invoke them is unlikely to hold exactly on current surface-code hardware:

\[
\mathbb E_0[
\Phi_t(x)-\overline\Phi_t
\mid\mathcal F_{t-1}
]=0
\quad\text{for every declared null, time, and location.}
\]

A surface-code layout is not one transitive set of interchangeable sensors.
Boundary stabilizers, \(X/Z\) roles, gate schedules, leakage-removal roles,
readout chains, fixed qubit heterogeneity, calibration actions, and latent
device states all break naive conditional translation symmetry. Unconditional
or visually approximate symmetry is insufficient.

The proposed paired/canary escape route is not currently supported by the
audited public data:

- the Google Quantum AI release contains real surface/repetition-code
  experiment data, but it does not supply a synchronized PNNL radiation-sensor
  stream or an identified exchangeable canary patch;
- PNNL sensor-assisted mitigation and radiation-spectroscopy work motivates a
  *prospective* side-sensor experiment, not a paired null for the historical
  Google data; and
- concatenating selected Google experiment blocks creates a valid
  retrospective stress test only if the blocks are homogeneous and chosen
  before scoring. It is a constructed changepoint, not observed online drift.

### Recommendation

| Proposed result | Audit decision |
|---|---|
| Conditional mathematical construction | **GO**, with the assumptions stated as the result, not as a hardware fact |
| Exact false-alarm guarantee on public Google data from orbit centering alone | **NO-GO** until a simultaneous conditional-mean certificate is supplied |
| “Paired” or “canary-controlled” Google hardware experiment | **NO-GO**: the required reference stream is not present in the audited release |
| Joint PNNL/Google real-hardware claim by combining separate datasets | **NO-GO** |
| Retrospective Google same-circuit block-concatenation benchmark | **CONDITIONAL GO**, labeled real data with a constructed boundary |
| Prospective PNNL sensor + QEC synchronization study | **CONDITIONAL GO**, if designed and budgeted as a new experiment |
| Localization “posterior” or credible region from e-wealth | **NO-GO** unless factors are declared likelihood ratios |
| Advantage over strong same-information QEC baselines | **NO-GO at present**; Run 5 is negative evidence, not support |
| Narrow publication after a locked, same-budget benchmark | **CONDITIONAL GO** |

The most realistic QEC design is a **two-channel monitor**:

1. a global detector for chip-wide/common-mode drift and radiation bursts; and
2. a stratified local residual scan for symmetry-breaking faults.

Orbit centering alone is a poor primary detector for the PNNL-motivated
failure mode because a spatially broad radiation event can be subtracted away.

## 1. Three null constructions that must not be conflated

### 1.1 Same-round orbit centering

For a field \(\Phi_t(x)\), Run 6 uses

\[
D_t(x)=\Phi_t(x)-\frac1M\sum_{u=1}^{M}\Phi_t(u).
\]

This consumes no extra quantum shots if every \(\Phi_t(x)\) is computed from
the native syndrome round. Its exact validity requires equal *conditional
feature means* across the declared orbit:

\[
\mathbb E_0[\Phi_t(x)\mid\mathcal F_{t-1}]
=m_t
\quad\text{for every }x.
\]

Full conditional group invariance is sufficient but stronger than necessary.
Equal unconditional means, exchangeable pooled histograms, or approximate
geometric translation symmetry are not sufficient.

### 1.2 A simultaneous paired reference

Suppose a target stream \(U_t\) and a reference stream \(V_t\) are both
observed at round \(t\). A paired antisymmetric score has an exact theorem.

#### Proposition 1: paired-reference e-factor

Let \((U_t,V_t)\), conditional on \(\mathcal F_{t-1}\), be exchangeable:

\[
(U_t,V_t)\mid\mathcal F_{t-1}
\overset d=
(V_t,U_t)\mid\mathcal F_{t-1}.
\]

Let \(g_{t-1}(u,v)\) be predictable, antisymmetric,
\(g_{t-1}(v,u)=-g_{t-1}(u,v)\), and bounded in \([-1,1]\). Then

\[
\mathbb E_0[
g_{t-1}(U_t,V_t)
\mid\mathcal F_{t-1}
]=0,
\]

and \(1+\beta g_{t-1}(U_t,V_t)\) is an e-factor for
\(|\beta|\le1\).

**Proof.** Conditional exchangeability and antisymmetry give

\[
\mathbb E[g(U,V)\mid\mathcal F_{t-1}]
=\mathbb E[g(V,U)\mid\mathcal F_{t-1}]
=-\mathbb E[g(U,V)\mid\mathcal F_{t-1}].
\]

The expectation is therefore zero. Boundedness gives nonnegativity of the
linear factor. \(\square\)

This is a different design from orbit centering. It requires a genuinely
exchangeable simultaneous pair or a design-based randomization that creates
one. Two fixed physical patches with different qubits, boundaries, workloads,
or readout chains are not exchangeable merely because they run the same code.

The “paired streams” in Run 5 were coupled simulation/evaluation replicates
used to reduce comparison variance. They were not an online target/reference
pair available to a hardware detector.

### 1.3 Canary rounds, canary qubits, and external sensors

These are three further designs, each with different assumptions.

- **Interleaved canary rounds:** a known diagnostic circuit is run between
  production rounds. It is asynchronous, changes the circuit context, and
  consumes time. A canary observation at \(t-1\) is not an exchangeable pair
  for a production observation at \(t\).
- **A canary patch:** a second patch runs simultaneously. It consumes qubits
  and controls, and its null law need not match the production patch. Exact
  paired validity requires exchangeability or randomized role assignment plus
  no interference.
- **A PNNL-style radiation/microresonator sensor:** this measures an external
  physical cause. It is valuable side information, not a draw from the QEC
  syndrome null. Sensor-only, syndrome-only, and fused monitors are different
  information sets.

If a side sensor \(Z_t\) is observed before the QEC outcome and is used to
choose a bet, the relevant filtration becomes

\[
\mathcal G_t=\sigma(\mathcal F_{t-1},Z_t).
\]

The required condition is then

\[
\mathbb E_0[L_t\mid\mathcal G_t]\le1.
\]

Using a contemporaneous sensor to choose the sign, location, or witness after
seeing both sensor and syndrome data is not automatically predictable. A
joint e-factor or an explicitly staged acquisition protocol is needed.

### Design comparison

| Design | Extra physical resource | Exact condition | Principal failure |
|---|---:|---|---|
| Orbit centering | None beyond native full syndrome field | Equal conditional means within each orbit/stratum | Hardware heterogeneity; removes common mode |
| Simultaneous paired patch | Additional qubits/control/readout | Conditional pair exchangeability | Patch mismatch and shared/interfering faults |
| Interleaved canary rounds | Lost production cycles | Stable/randomized potential-outcome model | Time lag and circuit mismatch |
| External PNNL sensor | Sensor area/readout/latency | Valid joint/conditional null model | Sensor is not a QEC reference draw |
| Historical block concatenation | No new hardware; retrospective compute | Homogeneous blocks and predeclared construction | Not natural drift; metadata confounding |

## 2. Counterexamples to naive orbit validity

### Counterexample 1: unconditional symmetry is not enough

Let a latent, fixed device orientation
\(\Theta\in\{-1,+1\}\) be uniformly random. Conditional on \(\Theta\), let

\[
X_{t,1}\sim\operatorname{Bernoulli}(p+\epsilon\Theta),
\qquad
X_{t,2}\sim\operatorname{Bernoulli}(p-\epsilon\Theta),
\]

independently over time. Unconditionally, swapping sites \(1\) and \(2\) has
the same law because it is equivalent to replacing \(\Theta\) by
\(-\Theta\). Thus the full experiment is unconditionally exchangeable across
the two sites.

After observing the past, however, \(\Theta\) becomes learnable. For the
orbit-centered score

\[
s_t=X_{t,1}-X_{t,2}\in[-1,1],
\]

\[
\mathbb E[s_t\mid\mathcal F_{t-1}]
=2\epsilon\,
\mathbb E[\Theta\mid\mathcal F_{t-1}],
\]

which is nonzero with positive probability once the past contains information
about the fixed orientation. More explicitly,

\[
\mathbb E[s_t\mid\Theta,\mathcal F_{t-1}]
=2\epsilon\Theta.
\]

A predictable witness that estimates \(\Theta\) from the past can choose the
corresponding sign and obtain positive expected growth under a perfectly
stationary no-change process. The alleged e-factor is then invalid.

**Hardware interpretation.** A fixed hot qubit, readout chain, TLS
environment, or boundary role is precisely such a latent orientation.

**Repair.** Stratify or model the fixed location effect; randomize physical
roles when scientifically possible; or use a simultaneous, uniformly valid
conditional-bias envelope. Do not test conditional validity with a pooled
unconditional permutation plot.

### Counterexample 2: surface-code geometry is not one orbit

Even an ideal planar surface code has inequivalent positions:

- interior and boundary checks have different support;
- \(X\)- and \(Z\)-type stabilizers have different circuit roles;
- cycle endpoints differ from bulk rounds;
- leakage-removal and measurement/reset paths can differ; and
- a real chip assigns distinct frequencies, couplers, amplifiers, and
  calibration histories to nominally translated locations.

Pooling these into one orbit creates nonzero null residual means. A defensible
partition must at least consider

\[
\text{check type}\times
\text{boundary degree}\times
\text{round role}\times
\text{circuit/basis}\times
\text{hardware patch}.
\]

After stratification, some orbits may be too small for a meaningful
unknown-location scan. That is a scientific result, not a reason to pool
inequivalent sites.

### Counterexample 3: common-mode drift is exactly invisible

Let the observation field change by a common offset:

\[
\Phi'_t(x)=\Phi_t(x)+c_t
\quad\text{for every }x.
\]

Then

\[
\Phi'_t(x)-\overline\Phi'_t
=\Phi_t(x)-\overline\Phi_t.
\]

No detector that sees only the orbit-centered field can distinguish these
two processes, regardless of sample size or algorithm.

The same no-go applies to a uniform change in a correlation channel. PNNL-
motivated radiation and quasiparticle events can produce broad spatially
correlated disturbances. Recent superconducting-qubit experiments explicitly
analyze their temporal, spatial, and frequency signatures. If the effect is
chip-wide, orbit centering can delete the most relevant signal.

**Repair.** Run a global channel in parallel:

\[
G_t=\frac1M\sum_x\Phi_t(x)
\]

or a calibrated decoder/detector likelihood, while the centered scan handles
relative local structure. Treat global and local alarms as a predeclared
multiple-monitor family.

### Counterexample 4: centered data can confound location and sign

With two sites, these two mean changes have identical centered
representations:

\[
(\mu+\delta,\mu)
\quad\text{and}\quad
(\mu,\mu-\delta).
\]

Both become

\[
(\delta/2,-\delta/2)
\]

after subtracting their orbit means. Thus “site 1 increased” and “site 2
decreased” are not identifiable from the centered field alone.

**Repair.** Report a relative anomaly unless an uncentered/global reference
resolves the sign. Do not describe the location evidence as identifying a
physical fault mechanism without an identifiability audit.

## 3. Paired/canary counterexamples and repairs

### Counterexample 5: a fixed reference mismatch creates false evidence

Under an operational no-change regime, suppose

\[
U_t\sim\operatorname{Bernoulli}(p+\delta),
\qquad
V_t\sim\operatorname{Bernoulli}(p-\delta)
\]

because the target and canary patches have stable but different readout
errors. Then

\[
\mathbb E[U_t-V_t]=2\delta.
\]

The linear paired factor has positive null drift. Simultaneous acquisition
does not repair nonexchangeability.

**Repair.** Randomize which physically comparable patch receives each role,
or calibrate a simultaneous bound on the conditional paired bias. Fixed
device labels plus one pre-experiment mean correction do not create an exact
paired theorem under later drift.

### Counterexample 6: an asynchronous canary aliases smooth drift

Let production and canary observations share a no-event mean
\(\mu_t=at\), but let the canary be observed one round earlier:

\[
U_t=\mu_t+\varepsilon_t,\qquad
V_{t-1}=\mu_{t-1}+\eta_{t-1}.
\]

Then

\[
\mathbb E[U_t-V_{t-1}\mid\mathcal F_{t-1}]=a.
\]

The canary difference signals a changepoint even though the system follows
the declared smooth null.

**Repair.** Synchronize acquisition, explicitly model admissible smooth drift,
or randomize interleaving and derive a design-based estimator for the exact
potential-outcome null. Count the loss of production throughput.

### When randomized canaries can be exact

If two comparable patches are both observed and a fair random bit \(A_t\) is
drawn independently before assigning their target/reference labels, then

\[
(2A_t-1)\{f(U_t)-f(V_t)\}
\]

has conditional mean zero under the assignment mechanism, even when the two
physical patches differ. This is a design-based result.

It is useful only if:

1. roles can actually be randomized;
2. assignment does not change the potential syndrome outcomes under the null;
3. there is no cross-patch interference;
4. the randomization is logged and included in the filtration; and
5. the resource cost of the second patch is counted.

Running a logical workload on one patch and a structurally different
diagnostic circuit on the other does not satisfy these conditions.

## 4. Predictable learned witnesses: the exact theorem and two traps

Predictability is necessary, but it is not the centering theorem.

### Proposition 2: sufficient conditions for a learned-witness e-factor

Let \(D_t\in\mathbb R^p\) satisfy

\[
\mathbb E_0[D_t\mid\mathcal F_{t-1}]=0.
\]

Let \(w_{t-1}\) and \(B_{t-1}>0\) be
\(\mathcal F_{t-1}\)-measurable, with

\[
|w_{t-1}^{\mathsf T}D_t|
\le B_{t-1}
\quad\text{almost surely}.
\]

Then, for predictable \(|\beta_{t-1}|\le1\),

\[
L_t
=1+\beta_{t-1}
\frac{w_{t-1}^{\mathsf T}D_t}{B_{t-1}}
\]

is a nonnegative conditional e-factor.

The witness may be an AOC/Jordan effect, covariance/Hotelling direction,
logistic direction, neural score, or an online expert selected from all past
data. The theorem depends on conditional centering, predictability, and the
range bound—not on the witness's name.

### Trap 1: a predictable plug-in mean is still biased

Let \(X_t\overset{\rm iid}{\sim}\operatorname{Bernoulli}(p)\), and let
\(\widehat p_{t-1}\) be any past-based estimator. Then

\[
\mathbb E[
X_t-\widehat p_{t-1}
\mid\mathcal F_{t-1}]
=p-\widehat p_{t-1},
\]

which is generally nonzero. The same remains true conditional on an
independent finite calibration sample: once its estimate is frozen, its
realized error is a fixed null bias.

**Repair.**

- use a known analytic null;
- use a design-based orbit/pair symmetry;
- construct a simultaneous confidence sequence or robust null envelope; or
- include calibration uncertainty inside a joint e-process.

If a bound

\[
\left|
\mathbb E_0[s_t\mid\mathcal F_{t-1}]
\right|
\le\varepsilon_t
\]

holds simultaneously, then

\[
\widetilde L_t
=\frac{1+\beta s_t}{1+|\beta|\varepsilon_t}
\]

is safe. If the envelope fails with probability at most \(\delta\), a simple
union argument gives at best an \(\alpha+\delta\) overall error statement
unless the budgets are combined more carefully. The envelope must be
simultaneous over time, location, shape, witness tuning, and all nulls claimed.

### Trap 2: choosing a witness on the current round

Let \(D_t\) be a null Rademacher variable. If the witness is selected after
seeing \(D_t\), choose \(w_t=\operatorname{sign}(D_t)\). The score becomes

\[
w_tD_t=|D_t|=1,
\]

so \(1+\beta|D_t|\) has expectation \(1+\beta>1\).

“Update after scoring” is therefore essential. Current-round selection of a
location, sign, eigenvector, threshold, variance, or clipping rule can create
the same bias.

### Additional QEC caveat: control changes the filtration

If an alarm triggers decoder reweighting, recalibration, qubit remapping, or
reinforcement-learning control, future syndrome laws are action-dependent.
The action history belongs in the filtration. A null guarantee derived for a
fixed circuit cannot simply continue across adaptive interventions. The
monitoring episode must be reset under a predeclared valid protocol or
analyzed as a controlled stochastic process.

## 5. Multiple locations: what remains valid

### 5.1 Fixed mixtures are safe; maxima and products are not

If \(E_t^j\) are component e-processes and fixed
\(\pi_j\ge0\), \(\sum_j\pi_j=1\), then

\[
E_t=\sum_j\pi_jE_t^j
\]

is valid even when all locations are dependent.

Two one-step counterexamples show why shortcuts fail.

#### Invalid maximum

Let an event \(A\) have probability \(1/2\), and set

\[
L_1=2\mathbf1_A,\qquad
L_2=2\mathbf1_{A^c}.
\]

Each has expectation one, but

\[
\max(L_1,L_2)=2
\]

always. A raw maximum of e-values is not an e-value.

#### Invalid spatial product

Set

\[
L_1=L_2=2\mathbf1_A.
\]

Again each has expectation one, but

\[
\mathbb E[L_1L_2]
=\mathbb E[4\mathbf1_A]
=2.
\]

Multiplying same-round location factors is invalid without a proved
conditional factorization or a direct joint expectation bound.

### 5.2 Valid scan threshold

The Run 6 weighted scan

\[
\max_j\pi_jE_t^j\ge1/\alpha
\]

is conservative because it implies
\(\sum_j\pi_jE_t^j\ge1/\alpha\). Equivalently, component \(j\) uses threshold
\(1/(\alpha\pi_j)\).

An unweighted maximum at threshold \(1/\alpha\) is not protected. Neither is a
shape/location prior selected after viewing the current round. Predictable
one-step expert weights can be valid, but they define an alternative whose
location may move; they are not the same as evidence for one persistent
unknown site.

### 5.3 Multiple affected sites

For a predeclared structured set \(S\), one may construct one bounded
aggregate score and one e-factor. One may also mix over a finite shape bank.
One may not multiply the site factors merely because \(S\) is sparse.

Orbit centering also creates \(M-|S|\) weak opposite-sign residuals outside a
positive patch. With a two-sided bet bank, those components accumulate
opposite-sign evidence. The true patch may dominate asymptotically, but early
evidence can be diffuse or favor a broad complement. Localization and
detection must be audited separately.

## 6. Localization is not automatically inference

Normalized component wealth

\[
q_t(j)
=\frac{\pi_jW_{t,j}}{\sum_u\pi_uW_{t,u}}
\]

is a useful descriptive evidence allocation. It is a Bayesian posterior only
when \(W_{t,j}\) is the declared likelihood ratio for component \(j\) and
\(\pi_j\) is a declared prior. For generic bounded bets:

- \(q_t(j)=0.95\) is not a calibrated \(95\%\) probability;
- a \(95\%\)-mass location set has no coverage guarantee;
- selection at the alarm time changes its sampling distribution; and
- the component with greatest log-growth need not be the physical source.

### Proposition 3: an exact localization no-go

Let \(\theta\in\{1,2\}\) denote two possible fault locations with equal prior.
If the accessible stream has the same law under both,

\[
P_1^Y=P_2^Y,
\]

then every location estimator \(\widehat\theta(Y)\) has average success
probability at most \(1/2\).

**Proof.** With the common law \(Q\),

\[
\frac12P_1(\widehat\theta=1)
+\frac12P_2(\widehat\theta=2)
=\frac12\{Q(\widehat\theta=1)+Q(\widehat\theta=2)\}
\le\frac12.
\quad\square
\]

Symmetry-related locations can become observationally identical after
coarse-graining or orbit centering. No heatmap repairs this.

### Repair

Choose one of three honest goals:

1. **descriptive localization:** report top-\(k\) and distance error on
   independent labeled experiments;
2. **Bayesian localization:** specify actual likelihood components and check
   posterior calibration; or
3. **frequentist localization:** formulate location hypotheses and use a
   simultaneous sequential multiple-testing/confidence-set method.

Post-alarm extra rounds can be reserved for confirmation/localization, but
their shot cost and stopping rule must be declared.

## 7. ARL and probability of ever alarming are operationally different

### Counterexample 7: a valid SR e-detector that always false-alarms

Take the uninformative factor

\[
L_t\equiv1.
\]

The Shiryaev--Roberts recursion is

\[
R_t=(R_{t-1}+1)L_t=t.
\]

At threshold \(\gamma\), it alarms deterministically at \(t=\gamma\):

\[
\mathbb E_0[\tau_\gamma]=\gamma,
\qquad
\mathbb P_0(\tau_\gamma<\infty)=1.
\]

This exactly satisfies the ARL theorem while providing no evidence of change.
Run 5 observed the analogous “clock behavior” for an exactly blind score.

Therefore:

- an SR threshold of \(10^3\) means at least \(10^3\) *updates in expected
  time to false alarm*, not a \(10^{-3}\) lifetime false-alarm probability;
- an operational monitor that runs for billions of QEC cycles needs a
  threshold and reset policy tied to that exposure; and
- delay should be measured relative to a same-horizon null run, not merely
  from a fresh zeroed statistic.

### Proper-prior e-process

The proper changepoint-prior construction in the theory note can control

\[
\mathbb P_0(\text{ever alarm})\le\alpha.
\]

It pays a late-start prior penalty and covers only the declared monitoring
episode. Resetting after each alarm does not retain one global \(\alpha\)
guarantee without alpha spending or another valid repeated-testing protocol.

### Required choice for QEC

- For one finite retrospective block or one safety-critical run, use a
  probability-of-ever-alarm guarantee.
- For indefinite plant-style surveillance, ARL/false-alarm rate may be the
  appropriate engineering metric, but report expected alarms per physical
  time, the reset/intervention policy, and alarm cost.
- Do not tune one method by ARL and another by finite-horizon familywise error,
  then compare delays.

## 8. Shot, cycle, qubit, and intervention budgets

“Same number of samples” is not a sufficient QEC resource statement.

### 8.1 Required ledger

For every method report:

\[
C_{\rm total}
=C_{\rm calibration}
+C_{\rm surveillance}
+C_{\rm canary}
+C_{\rm confirmation}
+C_{\rm post\text{-}alarm}.
\]

Also report qubit-cycle exposure

\[
Q_{\rm total}
=\sum_r
(\text{physical qubits used in role }r)
\times
(\text{cycles in role }r).
\]

The ledger must include:

- shots and physical QEC cycles used to estimate the null;
- shots used to learn/select the witness, shape bank, ridge, bet grid, and
  threshold;
- native syndrome rounds used for surveillance;
- reference-patch qubits and cycles;
- interleaved canary rounds and lost logical throughput;
- external-sensor acquisition and latency;
- post-alarm confirmation, decoder fitting, and calibration;
- rejected logical computations after a sensor veto; and
- classical runtime, memory, and control-loop latency.

### 8.2 Updates are not physical cycles

One Google QEC shot can contain many syndrome-extraction cycles. A temporal
feature using nonoverlapping pairs receives half as many updates as a
one-cycle detector. Overlapping windows may update every cycle, but reuse data
and must still satisfy the conditional e-factor assumption.

All delay and ARL comparisons should be converted to physical cycles and wall
time. If a fraction \(c\) of time is used for interleaved canaries, \(n\)
production updates require at least \(n/(1-c)\) wall-clock slots before other
latency; the lost production fraction must appear in utility.

### 8.3 Orbit centering is cheap but not free information

Same-round centering normally requires no additional quantum measurement
beyond native syndrome readout, but it uses the full spatial syndrome field.
A comparator given only a scalar detector-firing rate is not
same-information. Same-information baselines must receive the same
location-resolved correlation features.

A separate canary patch may not consume more *shots* in parallel, but it
consumes physical qubits, couplers, control channels, cryogenic readout
bandwidth, and decoder capacity. Calling it “free paired calibration” would be
misleading.

## 9. Reality check: the public Google and PNNL evidence

### 9.1 Google public QEC data

The [Google Quantum AI Zenodo
release](https://zenodo.org/records/13273331) contains real surface- and
repetition-code memory experiment data. The associated
[surface-code paper](https://arxiv.org/abs/2408.13687) reports, among other
experiments:

- different code distances, bases, subgrids, and processors;
- rare correlated events in long repetition-code operation;
- 16 measurements over 15 hours with recalibration after every four runs; and
- decoder adaptation and detector-likelihood diagnostics.

These are scientifically valuable, but they do not automatically form one
stationary pre-change stream followed by one naturally labeled drift stream.

#### Prohibited constructions

- Concatenating \(d=3\) and \(d=7\) blocks and calling the boundary “drift.”
  The observation dimension, circuit, geometry, and expected detector rate
  change.
- Treating nested/overlapping subcodes as independent paired canaries.
- Randomly shuffling shots and then interpreting the result sequentially.
- Using experiment-set metadata to select a boundary while withholding that
  information only from selected comparators.
- Calling recalibration times the true onset of drift. They are interventions,
  not necessarily change onsets.

#### Defensible retrospective construction

Use only blocks with the same code, basis, circuit, round count, patch, and
measurement representation. Preserve documented chronology when it exists.
Choose block pairs and the hidden boundary without inspecting detector scores.
Label the result:

> real processor measurements with a constructed changepoint.

If chronology or homogeneous context cannot be established from metadata, do
not make a sequential hardware claim.

### 9.2 PNNL sensor and radiation work

PNNL's [sensor-assisted fault-mitigation
paper](https://arxiv.org/abs/2012.12423) proposes co-located sensors for
environmental energy deposition and studies illustrative small-code
mitigation, including the cost of rejecting computations. PNNL's
[substrate-spectroscopy work](https://doi.org/10.1103/PRXQuantum.5.040323)
shows that superconducting microresonators can detect deposited energy in a
cryogenic silicon substrate and motivates integration near quantum circuits.

These sources establish physical motivation. They do not provide, in the
audited public materials, a time-synchronized PNNL sensor stream paired with
the Google surface-code archive. A file-level join between unrelated
experiments would have no paired-null or causal interpretation.

Recent work on
[distinguishing correlated qubit-error
types](https://arxiv.org/abs/2603.16494) already uses temporal, spatial, and
frequency-domain features together with accelerometer data. A Run 6
PNNL-style experiment must compare against that physically motivated feature
analysis, not only against detector firing rate.

### 9.3 Current operational baselines are stronger than the Run 5 baseline set

As of this audit, relevant primary baselines include:

- Google's 2026
  [reinforcement-learning control of
  QEC](https://www.nature.com/articles/s41586-026-10759-2), which uses native
  error-detection events to continuously steer physical control parameters;
- [DGR](https://arxiv.org/abs/2311.16214), which updates matching-edge and
  edge-pair statistics for drifted/correlated noise;
- [detector error-model estimation from syndrome
  data](https://arxiv.org/abs/2504.14643);
- [logical-error/DEM estimation on Google Willow
  data](https://arxiv.org/abs/2606.11496);
- [detector likelihood](https://arxiv.org/abs/2408.02082);
- [QECali/CaliQEC](https://doi.org/10.1145/3695053.3731042), a
  PNNL-coauthored in-situ surface-code calibration architecture; and
- direct external-sensor veto/fusion for radiation-associated events.

Some use different control authority or modeling assumptions. They must be
matched by information and resource class rather than placed in one
undifferentiated leaderboard.

## 10. Can the Run 6 factor honestly beat a strong baseline?

### Proposition 4: likelihood-ratio log-growth ceiling

Let \(P_1\ll P_0\), and let \(L\ge0\) be any one-step e-factor satisfying
\(\mathbb E_0L\le1\). Then

\[
\mathbb E_1[\log L]
\le
\mathrm{KL}(P_1\|P_0),
\]

with equality for

\[
L^\star=\frac{dP_1}{dP_0}
\]

under the usual integrability conditions.

**Proof.** Write \(r=dP_1/dP_0\). By Jensen's inequality,

\[
\begin{aligned}
\mathbb E_1\log L
&=\mathbb E_1\log r
 +\mathbb E_1\log(L/r)\\
&\le\mathrm{KL}(P_1\|P_0)
 +\log\mathbb E_1[L/r]\\
&=\mathrm{KL}(P_1\|P_0)
 +\log\mathbb E_0L\\
&\le\mathrm{KL}(P_1\|P_0).
\end{aligned}
\quad\square
\]

Thus a bounded AOC/orbit bet cannot have greater expected log growth than the
correct likelihood ratio on the same observation. Classical likelihood CUSUM,
SR, and scan procedures remain the model-aware ceilings for the corresponding
simple alternatives and criteria.

### 10.1 Existing repo evidence is adverse

Run 5 already found:

- the correlation representation contains information absent from detector
  marginals;
- exact/model-aware likelihood detectors are much faster;
- same-feature logistic/Hotelling controls match or beat the AOC direction
  offline; and
- the locked sequential test gives no support for vAOC over the named
  same-feature logistic effect.

At the locked spatial middle effect, the reported restart mean delays were
approximately \(88\) cycles for the model-aware grid, \(470\) for logistic,
and \(540\) for vAOC. The Run 6 location scan is not identical to Run 5, so
these numbers do not settle its performance. They do remove any prior reason
to expect an AOC witness to dominate a strong same-feature classifier.

### 10.2 Required same-information baselines

Every detector should receive the same location-resolved syndrome/correlation
stream and the same calibration budget. The minimum comparison set is:

1. per-check detector-firing-rate EWMA/CUSUM with multiplicity correction;
2. global and local detector-likelihood CUSUM/SR;
3. covariance/Hotelling and matched pair-correlation scans;
4. logistic and boosted-tree scores on exactly the same bounded feature bank;
5. mixture/GLR scan statistics for unknown affected sites;
6. decoder negative log likelihood/residuals;
7. DGR or a faithful edge/edge-pair update;
8. sensor-only, syndrome-only, and fused models in a PNNL prospective arm;
9. the exact likelihood oracle in simulation only; and
10. the Run 6 orbit bet with all calibration and component-prior costs.

### 10.3 Plausible narrow advantages

Run 6 could honestly win on:

- exact design-based validity under a genuinely randomized/exchangeable
  paired design;
- lower calibration cost from justified symmetry pooling;
- robustness to arbitrary same-round spatial dependence under that exact
  symmetry;
- interpretable relative-location evidence; or
- lower classical compute for a large dense translation bank.

None is currently established for the QEC task. On a distance-7 layout,
local kernels are sparse and the array is small; direct stencil evaluation may
beat FFT overhead. Fixed hardware heterogeneity weakens symmetry pooling. A
PNNL-relevant chip-wide burst weakens orbit centering. Classical scan/e-value
methods can also use the same paired symmetry, so any gain is not inherently
quantum-inspired or ECA-specific.

## 11. Minimum defensible experimental designs

### 11.1 Google retrospective arm

1. Inventory archive metadata before looking at scores.
2. Identify homogeneous same-code/same-patch chronological blocks.
3. Freeze a constructed-boundary protocol and held-out blocks.
4. Keep global and centered-local channels separate.
5. Calibrate every component envelope simultaneously or report the e-values
   as diagnostic only.
6. Give all same-information baselines the same field and calibration blocks.
7. Measure physical-cycle delay, false alarms per million cycles,
   localization error, compute, and downstream logical-error utility.
8. State in every result table that the boundary is constructed.

This arm can test robustness and implementation. It cannot establish a
PNNL sensor benefit or natural online drift detection.

### 11.2 Prospective PNNL-style sensor-fusion arm

Required synchronized streams:

- timestamped native syndrome events from one fixed code/circuit;
- logical outcomes or a declared decoder utility measure;
- radiation/microresonator and, where relevant, vibration/accelerometer data;
- all calibration and control actions;
- blinded controlled-source event times plus a natural-background holdout;
- optional simultaneous canary patch with randomized roles if hardware allows.

Predeclare:

- sensor-only, global-syndrome, centered-local, and fused monitors;
- the acquisition order defining predictability;
- physical and qubit-cycle budgets;
- false-alarm metric and reset rule;
- event matching window;
- veto/recalibration/remapping action and cost; and
- a held-run rather than random-cycle split.

The operational endpoint should be expected logical failures or valid
computations per wall-clock/qubit-cycle budget, not detector AUC alone.

## 12. Final go/no-go gates

Proceed to an advantage paper only if every gate below passes.

| Gate | Pass condition | Failure decision |
|---|---|---|
| Stream integrity | Chronology, circuit context, calibration actions, and block construction are auditable | No sequential hardware claim |
| Null validity | Exact design-based centering or a simultaneous conditional-bias envelope | No exact e-factor claim |
| Global visibility | A common-mode channel accompanies orbit centering | No claim on radiation/global drift |
| Canary validity | Simultaneous exchangeability/randomization or an explicit causal time-series model | Do not call it paired |
| Multiplicity | Fixed mixture or valid weighted thresholds; no raw max/product | No familywise statement |
| Localization | Identifiability plus calibrated likelihood or separate error evaluation | Evidence heatmap only |
| False-alarm metric | ARL or ever-alarm guarantee chosen before delay tuning | No fair delay comparison |
| Resource parity | Calibration, canary, qubit-cycle, sensor, confirmation, and intervention costs included | No efficiency claim |
| Baseline strength | Best same-information likelihood/scan/logistic/decoder controls run | No advantage claim |
| Locked result | Held-out improvement with uncertainty and downstream utility | Publish negative benchmark or methods note |

### Overall recommendation

**NO-GO now** for a claim that Run 6 has demonstrated an algorithmic advantage
on the PNNL/Google QEC problem.

**CONDITIONAL GO** for two scoped studies:

1. a retrospective Google real-measurement/constructed-boundary benchmark,
   explicitly noncausal and nonpaired; and
2. a prospective PNNL-style synchronized sensor-fusion experiment designed to
   make the filtration, canary role, and resource costs real.

The theory should be presented as:

> an exact conditional construction whose practical value depends on whether
> a hardware experiment can supply design-based symmetry or a defensible
> conditional-bias envelope.

That is mathematically meaningful. It is also falsifiable. It does not yet
show superiority to detector likelihood, classical scan/CUSUM/SR,
same-feature logistic models, decoder-aware adaptation, PNNL side sensors, or
Google's continuous QEC control.

## Primary evidence consulted

- Run 6 theory note:
  [`run6_symmetry_scan_eprocess_theory.md`](run6_symmetry_scan_eprocess_theory.md).
- Run 5 locked results:
  [`run5_surface_code_drift_results_and_advantage_audit.md`](run5_surface_code_drift_results_and_advantage_audit.md).
- Google Quantum AI public data:
  [Zenodo 13273331](https://zenodo.org/records/13273331).
- Google Quantum AI, “Quantum error correction below the surface code
  threshold”:
  [arXiv:2408.13687](https://arxiv.org/abs/2408.13687).
- Google/DeepMind, “Reinforcement learning control of quantum error
  correction”:
  [Nature](https://www.nature.com/articles/s41586-026-10759-2).
- Orrell and Loer, “Sensor-assisted fault mitigation in quantum computation”:
  [arXiv:2012.12423](https://arxiv.org/abs/2012.12423).
- Fowler et al., “Spectroscopic measurements and models of energy deposition
  in the substrate of quantum circuits by natural ionizing radiation”:
  [PRX Quantum](https://doi.org/10.1103/PRXQuantum.5.040323).
- Binney et al., “Distinguishing types of correlated errors in
  superconducting qubits”:
  [arXiv:2603.16494](https://arxiv.org/abs/2603.16494).
- Shin, Ramdas, and Rinaldo, “E-detectors”:
  [NEJSDS](https://nejsds.nestat.org/journal/NEJSDS/article/59/read).
- Dandapanthula and Ramdas, multi-stream sequential change detection:
  [arXiv:2501.04130](https://arxiv.org/abs/2501.04130).
- Wang et al., DGR:
  [arXiv:2311.16214](https://arxiv.org/abs/2311.16214).
- Blume-Kohout and Young, detector error-model estimation:
  [arXiv:2504.14643](https://arxiv.org/abs/2504.14643).
- Takou et al., logical-error/DEM estimation on surface-code experiments:
  [arXiv:2606.11496](https://arxiv.org/abs/2606.11496).
- Hesner, Hetényi, and Wootton, detector likelihood:
  [arXiv:2408.02082](https://arxiv.org/abs/2408.02082).
- Fang et al., QECali/CaliQEC:
  [ISCA 2025](https://doi.org/10.1145/3695053.3731042).
