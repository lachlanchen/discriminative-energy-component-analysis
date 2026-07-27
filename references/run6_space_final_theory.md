# Run 6 integrated theory: S-PACE

**Provisional expansion:** symmetry-/pair-calibrated additive contrast
e-detection
**Status:** final theory specification for implementation, not a novelty or
performance claim
**Date:** 2026-07-27

## 1. Result in one sentence

The original ECA principle—learn what differs rather than what merely
varies—can be made sequential and finite-sample valid when the observed
contrast is conditionally centered by design:

\[
\boxed{
\text{valid conditional contrast}
\;+\;
\text{past-measurable bounded witness}
\;\Longrightarrow\;
\text{e-factor}.
}
\]

The contrast can come from a known reference, a genuinely exchangeable
simultaneous pair, or a scientifically valid conditional symmetry orbit.
The witness can be the standard positive-eigenspace effect, an analytically
optimal top-\(k\) support for the declared capped linear-gap objective, or a
predictable online portfolio. A proper
changepoint-prior mixture gives probability-of-ever-alarm control; a
Shiryaev--Roberts sum gives average-run-length control.

The practical repair to the first Run 6 draft is equally important:

\[
\text{feature field}
=
\underbrace{\text{global/trivial sector}}_{\text{common mode}}
\oplus
\underbrace{\text{nontrivial sectors}}_{\text{relative/local structure}}.
\]

Both must be monitored. Pure orbit centering deletes the global sector and
can therefore miss a chip-wide radiation or calibration event.

Nothing in this construction permits a claim of superiority to a correctly
specified likelihood ratio, Helstrom or Wilson oracle, or an oracle
classifier using the same informative feature. Exact hardware validity is
also not implied by approximate geometric symmetry or a finite plug-in
baseline.

## 2. Filtration and observation model

Let \(\mathcal F_t\) contain all observations, randomizations, model updates,
alarms and control actions available after update \(t\). Anything used to
score observation \(t\)—a witness, location, sparsity level, bet size,
calibration envelope or pruning rule—must be
\(\mathcal F_{t-1}\)-measurable.

Let

\[
\phi:\mathcal Y\longrightarrow[0,1]^p
\]

be a declared bounded feature map. Its coordinates may be detector bits,
local parity products, co-firing indicators, bounded spectral bands,
correlations, graph stencils, optical intensities, tactile patches or
chemical-sensor channels.

The theory begins only after constructing a contrast

\[
D_t\in[-1,1]^p
\quad\text{such that}\quad
\mathbb E_0[D_t\mid\mathcal F_{t-1}]=0.
\tag{1}
\]

Predictability alone does not imply (1). The mechanism creating (1) is part
of the statistical model and must be declared.

## 3. Three exact contrast interfaces and one empirical interface

### 3.1 Known conditional reference

If the conditional feature mean

\[
\mu_t=\mathbb E_0[\phi(Y_t)\mid\mathcal F_{t-1}]
\]

is known, then

\[
D_t=\phi(Y_t)-\mu_t
\]

satisfies (1). An analytic simulator null can have this property. A mean
estimated from finite calibration data is not known after conditioning on
the realized estimate.

### 3.2 Simultaneous paired exchangeability

Suppose a target \(U_t\) and reference \(V_t\) are observed together and

\[
(U_t,V_t)\mid\mathcal F_{t-1}
\overset d=
(V_t,U_t)\mid\mathcal F_{t-1}.
\tag{2}
\]

Then

\[
D_t=\phi(U_t)-\phi(V_t)
\tag{3}
\]

satisfies (1). Independence within a pair is unnecessary.

More generally, let
\(g_{t-1}(u,v)=-g_{t-1}(v,u)\in[-1,1]\) be predictable. Conditional
exchangeability gives

\[
\mathbb E_0[g_{t-1}(U_t,V_t)\mid\mathcal F_{t-1}]=0.
\tag{4}
\]

Equation (4) is the exact paired-reference theorem. It does not apply merely
because two fixed patches have the same nominal code distance. Different
qubits, boundaries, schedules, readout paths or workloads can destroy (2).

If one pair contains several coordinates or round roles that all share the
same swap/orientation variable, the entire pair is one filtration time step.
For predictable coordinate scores \(g_{t,r,c}\), each `(role, component)` may
be a fixed e-process expert and the experts may be mixed with priors summing
to one. Their same-pair factors may not be multiplied or revealed
sequentially as though they were conditionally independent trials: the first
coordinate can reveal the shared orientation. This experimental-unit rule is
essential in the QEC replay below.

### 3.3 Conditional orbit centering

Let a finite group \(G\) act transitively on a scientifically valid location
set \(\mathcal X\), and let \(\Phi_t(x)\in[0,1]^p\). If, conditional on the
past, the joint field is invariant under the group action, then

\[
D_t(x)
=
\Phi_t(x)-\frac1{|\mathcal X|}
                  \sum_{u\in\mathcal X}\Phi_t(u)
\tag{5}
\]

has conditional mean zero for every \(x\), without any independence
assumption between locations.

Full conditional invariance is sufficient. Equality of conditional means is
the weaker property actually needed. Unconditional symmetry is insufficient:
a fixed latent hot location can be symmetric before observing history and
asymmetric after conditioning on it.

Real planar QEC hardware is not one undifferentiated orbit. At minimum, a
candidate partition must consider check type, boundary degree, circuit round
role, basis, patch and stable hardware role.

### 3.4 Empirical reference

For a finite baseline estimate \(\widehat\mu\),

\[
\mathbb E_0[
\phi(Y_t)-\widehat\mu
\mid\mathcal F_{t-1}]
=\mu_t-\widehat\mu,
\]

which is generally nonzero. A score centered by \(\widehat\mu\) is therefore
an empirically calibrated monitor, not automatically an exact e-factor.

Suppose \(s_t\in[-1,1]\), and a predictable simultaneous bound
\(\varepsilon_t\ge0\) is proved,

\[
\left|
\mathbb E_0[s_t\mid\mathcal F_{t-1}]
\right|\le\varepsilon_t,
\tag{6}
\]

then for \(|\beta|<1\)

\[
\widetilde L_t
=
\frac{1+\beta s_t}
     {1+|\beta|\varepsilon_t}
\tag{7}
\]

is nonnegative and satisfies

\[
\mathbb E_0[\widetilde L_t\mid\mathcal F_{t-1}]\le1.
\]

It is therefore a conditional e-factor, generally with expectation at most
one rather than exactly one. The range assumption (or another predictable
lower bound guaranteeing a nonnegative numerator) is essential: a bound on
the conditional mean alone does not prevent \(1+\beta s_t<0\). The envelope
in (6) must cover time, feature, location, tuning and the complete claimed
null class simultaneously. A pointwise standard error does not suffice.

## 4. Sector-resolved global and local monitoring

Let \(U_g\) be a fixed orthogonal or unitary representation of \(G\) on the
feature space \(V\). For a location field, \(U_g\) permutes locations while
leaving its \(p\) channels unchanged, so
\(\mathbf1_{\mathcal X}\otimes c\in V_{\mathrm{triv}}\). Write the isotypic
decomposition

\[
V=\bigoplus_{\lambda}V_\lambda,
\qquad
D_t=\sum_\lambda P_\lambda D_t,
\tag{8}
\]

where \(P_\lambda\) are mutually orthogonal representation projectors.
The Reynolds projector onto the invariant or trivial representation is

\[
P_{\mathrm{triv}}
=\frac1{|G|}\sum_{g\in G}U_g.
\tag{9}
\]

For a transitive scalar location field, \(P_{\mathrm{triv}}\) is the spatial
average. The complementary projector

\[
P_{\mathrm{rel}}=I-P_{\mathrm{triv}}
\]

contains relative or symmetry-breaking structure.

### Proposition 1: conditional centering survives sector projection

If (1) holds and \(P_\lambda\) is fixed, then

\[
\mathbb E_0[P_\lambda D_t\mid\mathcal F_{t-1}]=0.
\]

Thus each sector can supply a valid bounded score after an explicit range
normalization. More precisely, let \(\ell_{t-1}\) be a predictable real
linear functional. Let \(\mathcal D\subset V\) be a declared set containing
the support of \(D_t\); for a stacked \(p\)-channel field on \(\mathcal X\),
one conservative choice is
\(\mathcal D=[-1,1]^{p|\mathcal X|}\). Choose a predictable bound

\[
B_{t-1}\ge
\sup_{d\in\mathcal D}
\left|
\ell_{t-1}(P_\lambda d)
\right|.
\]

Define

\[
s_{t,\lambda}
=
\begin{cases}
\ell_{t-1}(P_\lambda D_t)/B_{t-1},
&B_{t-1}>0,\\
0,&B_{t-1}=0.
\end{cases}
\]

Then \(s_{t,\lambda}\in[-1,1]\) and is conditionally centered. For a complex
unitary representation, conjugate sectors must be combined into a real
invariant block, or the real part of a declared complex functional must be
used. Fixed mixtures across sectors remain exact only when every branch has
first been converted into a valid conditional e-factor; mixing an empirical
branch with an exact branch does not make the empirical branch exact.

### Proposition 2: common-mode no-go for pure orbit centering

Suppose the observed post-change field is exactly
\(\Phi'_t=\Phi_t+\mathbf 1_{\mathcal X}\otimes c_t\), with no clipping,
saturation, nonlinear feature remapping or simultaneous change in the
relative field. Then

\[
P_{\mathrm{rel}}(\Phi_t+\mathbf 1_{\mathcal X}\otimes c_t)
=P_{\mathrm{rel}}\Phi_t.
\]

Every detector measurable only with respect to the
nontrivial/orbit-centered field has identical law before and after that
common-mode change. A physical event called “chip-wide” can also alter
variances, correlations or relative responses; this proposition does not
rule out detecting those additional effects.

This is not a small-power effect; it is an information-theoretic blind spot.
For the orbit-centered contrast (5),

\[
\sum_{x\in\mathcal X}D_t(x)=0
\quad\text{and}\quad
P_{\mathrm{triv}}D_t=0
\]

pathwise. Its trivial projection therefore cannot be used as a global
detector. S-PACE instead keeps two separately referenced branches:

1. **global branch:** a contrast such as
   \(\overline\Phi_t-\mu_t^{\mathrm{global}}\) with a known conditional
   reference, a paired difference of simultaneous global averages, or an
   explicitly empirical detector-likelihood/common-mode monitor;
2. **relative branch:** nontrivial sectors or stratified local residuals.

Conditional orbit invariance alone does not center \(\overline\Phi_t\).
A fixed prior mixes the branches only after each has its own valid e-factor;
otherwise the combined hardware replay remains empirical. An empirical
hardware replay calibrates both branches separately at the same false-alarm
budget.

## 5. Predictable bounded witnesses

Let \(w_{t-1}\) be predictable with
\(\|w_{t-1}\|_1\le1\). From a vector contrast satisfying (1), define

\[
s_t=w_{t-1}^{\mathsf T}D_t.
\tag{10}
\]

Then \(|s_t|\le1\) and its conditional null mean is zero. Therefore

\[
L_t(\beta)=1+\beta s_t,
\qquad |\beta|\le1,
\tag{11}
\]

is nonnegative and has conditional expectation one.

The witness may be learned from every past observation. It may not be chosen
after seeing \(D_t\). Selecting
\(w_t=\operatorname{sign}(D_t)\) on the current round turns a null
Rademacher score into \(|D_t|=1\) and invalidates (11).

## 6. The original ECA/AOC spectral witness

Suppose each observation is represented by a positive unit-trace operator
\(R(Y)\). For a pair, define

\[
\Delta_t=R(U_t)-R(V_t).
\tag{12}
\]

For any predictable effect
\(0\preceq E_{t-1}\preceq I\),

\[
s_t=\operatorname{Tr}(E_{t-1}\Delta_t)\in[-1,1].
\tag{13}
\]

Conditional pair exchangeability makes (13) mean zero under the null.
The same holds for a known operator reference or a valid orbit-centered
operator field.

Given a past average contrast \(\overline\Delta_{t-1}\), the effect
maximizing the next one-step empirical expectation gap is

\[
E_{t-1}^{\mathrm{gap}}
=\mathbf 1_{\overline\Delta_{t-1}>0},
\tag{14}
\]

and

\[
\max_{0\preceq E\preceq I}
\operatorname{Tr}(E\overline\Delta_{t-1})
=\operatorname{Tr}(
  \overline\Delta_{t-1,+}).
\tag{15}
\]

Equations (14)--(15) are the standard Jordan/Helstrom result. Under an
accessible observable algebra \(\mathcal A\), replace the contrast by its
trace-preserving conditional expectation onto \(\mathcal A\) before taking
the positive part.

This solves a one-step linear-gap objective. It need not maximize sequential
log wealth or minimize detection delay.

## 7. Analytical discriminative features

The user's original motivation asks for the features with the largest class
difference instead of the directions with largest pooled variance. That
principle has a simple exact coordinate form.

For a mean contrast \(\delta\in\mathbb R^p\), define the signed expansion

\[
u=(\delta_1,\ldots,\delta_p,-\delta_1,\ldots,-\delta_p)
\in\mathbb R^{2p}
\]

and the capped simplex

\[
\mathcal C_k
=
\left\{
q\in\mathbb R_+^{2p}:
\sum_iq_i=1,\quad q_i\le\frac1k
\right\},
\qquad 1\le k\le p.
\tag{16}
\]

### Proposition 3: closed-form top-\(k\) contrast

\[
\max_{q\in\mathcal C_k}q^{\mathsf T}u
=
\frac1k\sum_{j=1}^{k}|\delta|_{(j)},
\tag{17}
\]

where \(|\delta|_{(1)}\ge\cdots\) are the ordered absolute contrasts. An
optimizer places weight \(1/k\) on the signed copies of the \(k\) largest
absolute coordinates.

**Proof.** A linear functional on the capped simplex is maximized by placing
the largest permitted mass on its largest coordinates. Because \(k\le p\),
the \(k\) largest coordinates of \((\delta,-\delta)\) are the appropriately
signed \(k\) largest absolute entries of \(\delta\). \(\square\)

The \(k=1\) case is the maximum-difference feature. The domain
\(\mathcal C_k\) is a capped or diversified convex portfolio and may contain
dense points; a linear objective has a \(k\)-support extreme-point
optimizer. In that extreme-point sense, larger \(k\) trades sparsity for
stability. Equation (17) is an analytical support-function solution, not a
claim that the selected feature is statistically optimal.

## 8. Predictable online portfolio and regret

At time \(t\), form the signed centered observation

\[
u_t=(D_t,-D_t)\in[-1,1]^{2p}.
\]

Use a predictable \(q_t\in\mathcal C_k\) and score

\[
s_t=q_t^{\mathsf T}u_t.
\tag{18}
\]

For a fixed \(0<\beta<1\), the log gain is

\[
f_t(q)=\log(1+\beta q^{\mathsf T}u_t).
\tag{19}
\]

This is concave in \(q\), and

\[
\|\nabla f_t(q)\|_\infty
\le G_\beta:=\frac{\beta}{1-\beta}.
\tag{20}
\]

Initialize \(q_1\) uniformly and, only after scoring \(u_t\), use the
KL/Bregman-projected entropic mirror-ascent update

\[
g_t=\nabla f_t(q_t),\qquad
q_{t+1}
=
\arg\max_{q\in\mathcal C_k}
\left\{
\eta\langle g_t,q\rangle
-\mathrm{KL}(q\|q_t)
\right\}.
\]

Equivalently, exponentiate the weights by \(\eta g_t\) and take their KL
projection onto \(\mathcal C_k\). Euclidean projection defines a different
algorithm and is not covered by the following constant.

### Proposition 4: pathwise log-wealth regret

For the standard entropic mirror-ascent update and every
\(q^\star\in\mathcal C_k\),

\[
\sum_{t=1}^{T}
\left[
f_t(q^\star)-f_t(q_t)
\right]
\le
\frac{\log(2p/k)}{\eta}
+\frac{\eta G_\beta^2T}{2}.
\tag{21}
\]

For a horizon \(T\) declared before the run, choose

\[
\eta=
\sqrt{
\frac{2\log(2p/k)}
     {G_\beta^2T}
},
\]

\[
\boxed{
\log W_T
\ge
\max_{q\in\mathcal C_k}
\sum_{t=1}^{T}\log(1+\beta q^{\mathsf T}u_t)
-
G_\beta\sqrt{2T\log(2p/k)}.
}
\tag{22}
\]

Here

\[
W_0=1,\qquad
W_T=\prod_{t=1}^{T}
\left(1+\beta q_t^{\mathsf T}u_t\right).
\]

The bound is the standard negative-entropy online-mirror-descent regret
inequality. The only Run 6 observation is how it interfaces with the valid
contrast: because \(q_t\) is predictable, each wealth factor remains an
e-factor under (1).

The displayed optimal constant step size is a fixed-horizon statement. An
unknown or unbounded horizon requires a declared doubling trick or
time-varying learning-rate schedule and its corresponding anytime regret
constant; e-validity itself only requires that the update remain
predictable.

Equation (22) gives a deterministic comparison with the best fixed capped
betting portfolio in hindsight. It is not a minimax
quickest-change-delay theorem. A separate learner for every candidate start,
or a strongly adaptive online-learning construction, is required for a
start-specific regret statement.

The sufficient state for the simple linear-gap update is additive:

\[
S_t=S_{t-1}+D_t,\qquad N_t=N_{t-1}+1.
\]

Independent nodes can merge \((S,N)\) by addition. Exponential forgetting
and finite windows change the alternative; a finite sliding window also
requires the outgoing samples or a ring buffer.

## 9. Unknown location, sector, shape, cap and bet size

Let a fixed component index be

\[
j=(\lambda,x,h,k,\beta),
\]

where \(\lambda\) is a representation sector, \(x\) a candidate location,
\(h\) a shape/channel, \(k\) a cap/diversification level whose linear
extreme optimizer has \(k\)-support, and \(\beta\) a bet size.
For every \(j\), maintain a compounded wealth from its own bounded
conditional e-factors.

Use fixed prior weights \(\pi_j\ge0\), \(\sum_j\pi_j=1\). The mixture

\[
\sum_j\pi_jW_{t,j}
\]

is valid under arbitrary dependence between components. In contrast:

- \(\max_jW_{t,j}\) is not generally an e-value;
- \(\prod_jL_{t,j}\) is not generally an e-factor under dependent locations;
- for raw scalar orbit contrasts—or common predictable linear witnesses
  using a common bet size and common normalization—the instantaneous
  uniform location mixture cancels identically to one; this need not hold
  for location-specific witnesses, bets or normalizations; and
- a prior selected after viewing the current round is invalid.

For a transitive orbit and equal component information, a uniform location
prior has worst-case location penalty \(\log|\mathcal X|\). This is the
standard cost of not knowing the location.

Normalized component wealth is an **evidence allocation**. It is not a
Bayesian posterior or a confidence set unless the components are genuine
likelihood ratios or a separate localization theorem is supplied.

## 10. Two sequential guarantees

### 10.1 Probability of ever alarming

Let \(\rho_\nu>0\), \(\sum_{\nu\ge1}\rho_\nu=1\), be a proper start-time
prior and let \(\rho_{>t}=\sum_{\nu>t}\rho_\nu\). Define

\[
A_{t,j}=L_{t,j}(A_{t-1,j}+\rho_t),
\qquad A_{0,j}=0,
\]

and

\[
E_t=\rho_{>t}+\sum_j\pi_jA_{t,j}.
\tag{23}
\]

Then \(E_t\) is a nonnegative supermartingale under the declared exact null,
with \(E_0=1\). Ville's inequality gives

\[
\mathbb P_0\!\left(
\sup_tE_t\ge\frac1\alpha
\right)\le\alpha.
\tag{24}
\]

A heavy-tailed prior such as
\(\rho_\nu=1/\{\nu(\nu+1)\}\) pays a logarithmic late-start penalty.
Repeated resets require alpha spending or another explicit protocol.

### 10.2 Average run length

Define per-component Shiryaev--Roberts recursions

\[
R_{t,j}=(R_{t-1,j}+1)L_{t,j},
\qquad R_{0,j}=0,
\]

and

\[
R_t=\sum_j\pi_jR_{t,j}.
\tag{25}
\]

Then \(R_t-t\) is a supermartingale and the threshold rule

\[
\tau_\gamma=\inf\{t:R_t\ge\gamma\}
\]

satisfies

\[
\mathbb E_0[\tau_\gamma]\ge\gamma
\tag{26}
\]

under the standard optional-sampling conditions.

This is an ARL statement, not lifetime false-alarm probability. If
\(L_t\equiv1\), then \(R_t=t\) and the detector alarms deterministically at
\(\lceil\gamma\rceil\). The statistic behaves as a clock while still
satisfying (26).

The experiment must choose (24) or (26) before comparing delay. Physical
shots, QEC cycles, paired/canary resources, resets and interventions must be
counted in the same unit for every method.

## 11. Power, delay and the likelihood ceiling

For a component whose witness and score law have stabilized and are
time-homogeneous after the change, define

\[
I_j(\beta)
=
\mathbb E_1[
\log(1+\beta s_{t,j})
].
\]

For small \(\beta\),

\[
I_j(\beta)
=\beta\delta_j-\frac{\beta^2v_j}{2}+O(\beta^3),
\]

where
\(\delta_j=\mathbb E_1s_{t,j}\) and
\(v_j=\mathbb E_1s_{t,j}^2\). This is a Kelly/log-growth objective. It is
different from the one-step Jordan/top-\(k\) mean-gap objective.

If the witness or portfolio continues adapting, use the time-indexed
quantities \(I_{t,j}\), \(\delta_{t,j}\) and \(v_{t,j}\); stationarity of the
raw observations alone does not make the adaptive score stationary.

For a correctly specified simple alternative \(P_1\ll P_0\), every
one-step e-factor \(L\) satisfies

\[
\mathbb E_1\log L
\le
\mathrm{KL}(P_1\|P_0),
\tag{27}
\]

with equality for the likelihood ratio \(dP_1/dP_0\). Therefore S-PACE
cannot beat the correct same-information likelihood ratio in expected
one-step log growth. Its possible value is robustness, transparent
calibration, sparse interpretation, reduced model fitting or efficient
symmetry reuse—not model-aware optimality.

For adaptive sequential factors the applicable statement is conditional:
almost surely for each realized past,

\[
\mathbb E_1[
\log L_t\mid\mathcal F_{t-1}]
\le
\mathrm{KL}\!\left(
P_{1,t}(\,\cdot\mid\mathcal F_{t-1})
\middle\|
P_{0,t}(\,\cdot\mid\mathcal F_{t-1})
\right),
\]

provided \(L_t\) is a conditional e-factor for the same current-observation
sigma-field. Summing gives the chain-rule ceiling. Methods receiving
different information are not covered by a same-information comparison.

First-order delay expressions that divide a threshold-plus-prior penalty by
\(I_j\) are renewal heuristics unless stronger iid or mixing assumptions are
added. They must not be presented as finite-sample delay bounds.

## 12. Hardware evidence classes

Run 6 uses four visibly separate evidence labels:

1. **Exact-model validation:** simulated or randomized data satisfying the
   stated conditional centering theorem.
2. **Natural approximate event:** real hardware order with an
   author-identified event interval but no exact physical onset timestamp.
3. **Known intervention:** a recalibration or commanded change; the
   intervention is not automatically the drift onset.
4. **Constructed boundary:** real hardware cohorts concatenated by the
   benchmark designer.

The public Google archive can support classes 2--4 depending on the subset.
The current PNNL/IBM release supports cross-calibration cohorts and class 4,
not a continuously timestamped drift stream. Fixed syndrome files cannot
support a faithful classical-shadow eSCD comparison because the randomized
measurement settings and outcomes required by that protocol were never
recorded.

On real archived data, finite baseline estimates and approximate stationarity
will normally force empirical threshold calibration. The paper must call the
result a causal replay score with empirical false alarms, not an exact
hardware e-process, unless an exact contrast interface is genuinely present.

## 13. Required physical resource ledger

For every method report:

\[
C_{\mathrm{total}}
=C_{\mathrm{calibration}}
+C_{\mathrm{surveillance}}
+C_{\mathrm{canary}}
+C_{\mathrm{confirmation}}
+C_{\mathrm{post-alarm}},
\]

plus qubit-cycle exposure, retained features, classical update cost, peak
memory and latency.

A simultaneous canary patch may cost no additional wall-clock shot, but it
uses qubits, controls and readout. A paired offline benchmark uses two
physical records per update. A temporal pair feature with nonoverlapping
cycles has half the update rate of a per-cycle feature. All delay and ARL
results must be converted to common physical units.

## 14. Falsifiable advantage gate

S-PACE supports an algorithmic-advantage statement only if a frozen
same-information comparison shows one of:

- lower delay at the same verified false-alarm criterion;
- better localization at the same delay and false-alarm criterion, with
  independent location truth;
- equal detection with fewer calibration labels or materially lower causal
  compute; or
- lower downstream logical loss at the same total physical budget.

Required controls include global and per-check detector-rate CUSUM,
detector-likelihood/e-SR, covariance or Hotelling monitoring, a sparse scan,
same-feature logistic/threshold models, decoder residual/likelihood, and a
model-aware oracle where the model is actually known. QEC-native adaptive
decoder/noise trackers must be compared by downstream utility when
reproducible.

Sparse heatmaps, a large one-shot mean gap, or success on one selected event
are not sufficient. A negative held-out result is the required conclusion
when the gate fails.

## 15. What is and is not being claimed

The defensible theory claim is:

> Known-reference, paired-exchangeable and conditionally symmetric designs
> provide a common source of bounded mean-zero contrasts. Predictable
> spectral or sparse witnesses can be learned online without breaking this
> validity; fixed mixtures and standard e-process/e-detector accumulators
> then give explicit sequential guarantees. Representation sectors expose
> the necessary split between global and relative changes.

This is a synthesis built from established state discrimination, invariant
testing, online convex optimization, betting inference and quickest-change
foundations. Literature-wide priority has not been established.

It is not a claim of:

- a new Helstrom/Jordan theorem;
- the first sparse, paired, group or quantum change detector;
- superiority to a correct likelihood ratio, Wilson diagnostic or
  same-feature oracle classifier;
- exact false-alarm control on public hardware from approximate symmetry;
- quantum speedup or universal sample efficiency;
- calibrated localization posterior from generic wealth; or
- a result in string theory or holographic duality.

## 16. Implementation decision

The locked implementation should expose four independent objects:

1. `ContrastInterface`: known, paired, orbit or empirical;
2. `SectorBank`: a separately referenced global branch plus valid
   nontrivial strata/sectors;
3. `Witness`: Jordan effect, top-\(k\) gap or online capped portfolio;
4. `Accumulator`: proper-prior e-process or SR e-detector.

This separation makes every assumption testable. It also prevents a good
feature representation from being mistaken for a valid null calibration, or
a valid e-factor from being mistaken for an optimal quickest-change rule.

The immediate real-data experiment is a held protocol on Google hardware
replay, with the 2022 author-identified event kept unseen until the feature
map, baseline partitions, false-alarm rule and comparators are frozen.
PNNL/IBM data are an auxiliary cross-snapshot/domain-shift arm only.
