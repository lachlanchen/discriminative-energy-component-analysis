# Run 6 theory: symmetry-orbit scan e-processes for unknown-location correlation drift

**Status:** theory/design note, not a novelty or performance claim
**Date:** 2026-07-27
**Scope:** unknown-location sparse correlation drift on a finite symmetry orbit,
with predictable bounded scores, exact false-alarm statements, streaming
updates, localization evidence, and group-convolution acceleration

## Executive result

There is a clean algorithmic synthesis worth testing:

1. represent each round by bounded local correlation features over a symmetry
   orbit;
2. subtract the *same-round orbit average*;
3. apply a direction or template chosen using the past only;
4. compound a separate bounded betting factor for each fixed
   location/shape/effect-size hypothesis;
5. mix the compounded wealths, not the instantaneous local factors; and
6. choose explicitly between:
   - a proper changepoint-prior **e-process**, which controls the probability of
     ever raising a false alarm; or
   - a mixture Shiryaev--Roberts **e-detector**, which controls average run
     length under indefinite monitoring.

The central finite-sample observation is simple but useful. Conditional
invariance of the null field over a transitive group orbit makes every
orbit-centered local score conditionally mean zero, even when all locations
within a round are strongly dependent. This yields valid local betting factors
without estimating a full spatial covariance or multiplying across dependent
sites.

This does **not** establish that the resulting detector is faster than a
correct likelihood-ratio scan, CUSUM, Shiryaev--Roberts procedure, matched
threshold/logistic detector, Wilson-loop diagnostic, or decoder likelihood.
The Jordan/positive-eigenspace direction used by accessible-observable
contrast (AOC) is the standard Helstrom/Jordan solution for a one-step
expectation-gap objective; it need not maximize sequential log growth.

Most ingredients below are established. The potentially useful contribution
is their symmetry-aware, dependence-robust and computationally explicit
combination, together with a benchmark that can falsify an advantage claim.
Literature-wide novelty has not been established.

## 1. Statistical problem and exact assumptions

### 1.1 Orbit of possible drift locations

Let a finite group \(G\) act transitively on a finite set of locations

\[
\mathcal X \simeq G/H,\qquad M=|\mathcal X|,
\]

where \(H\) is the stabilizer of a reference location. Translation on a
periodic lattice is the main example, but rotations of an array, symmetry-
equivalent bonds in a molecule, and equivalent plaquettes in a lattice gauge
model fit the same notation.

At round \(t\), an observation \(Y_t\) becomes a field of bounded correlation
features

\[
\Phi_{t,h}(x)\in[-1,1]^p,\qquad x\in\mathcal X,\quad h\in\mathcal H.
\]

Here \(h\) indexes a finite bank of shapes, scales, correlation channels, or
accessible observables. For binary syndrome/spin data \(z_t(x)\in\{-1,1\}\),
a basic example is

\[
\Phi_{t,h}(x)
=\left(
z_t(x+a_{h,r})z_t(x+b_{h,r})
\right)_{r=1}^{p}.
\]

The objective is to detect a persistent change affecting an unknown translated
location or sparse translated patch, and then localize it.

Let \(\mathcal F_t\) contain all information available after round \(t\).
Every learned template, bet size, calibration bound, and pruning decision used
to score round \(t\) must be \(\mathcal F_{t-1}\)-measurable.

### 1.2 Conditional orbit invariance

The exact null assumption used in the main theorem is:

> **Conditional orbit invariance.** For every \(a\in G\), conditional on
> \(\mathcal F_{t-1}\), the joint field
> \((\Phi_{t,h}(a x))_{x,h}\) has the same law as
> \((\Phi_{t,h}(x))_{x,h}\).

This is stronger than unconditional symmetry. A fixed unknown hot sensor, a
past-dependent preferred direction, or location-dependent readout calibration
can violate conditional invariance even when an unconditional histogram looks
symmetric. The assumption must therefore be tested scientifically and
replaced by a calibrated conditional baseline when it is implausible.

No conditional independence between locations is assumed. Locations may share
latent noise, conservation constraints, parity constraints, or decoder-induced
dependence.

## 2. Orbit centering yields exact predictable local bets

Define the same-round orbit mean

\[
\overline\Phi_{t,h}
=\frac1M\sum_{x\in\mathcal X}\Phi_{t,h}(x).
\]

Let \(w_{t-1,h}\in\mathbb R^p\) be any predictable direction. It may be fixed,
learned from a separate calibration set, or updated from rounds strictly before
\(t\). Define

\[
s_{t,x,h}
=
\begin{cases}
\displaystyle
\frac{
\left\langle w_{t-1,h},
\Phi_{t,h}(x)-\overline\Phi_{t,h}\right\rangle
}{
2\|w_{t-1,h}\|_1
},
&\|w_{t-1,h}\|_1>0,\\[1.2em]
0,&\|w_{t-1,h}\|_1=0.
\end{cases}
\]

Because every coordinate of
\(\Phi_{t,h}(x)-\overline\Phi_{t,h}\) lies in \([-2,2]\),

\[
|s_{t,x,h}|\le 1.
\]

### Theorem 1: exact mean-zero score without spatial independence

Under conditional orbit invariance and transitivity,

\[
\mathbb E_0[s_{t,x,h}\mid\mathcal F_{t-1}]=0
\quad\text{for every }t,x,h.
\]

**Proof.** Conditional invariance and transitivity imply that, for fixed \(h\),
\(\mathbb E_0[\Phi_{t,h}(x)\mid\mathcal F_{t-1}]\) is the same vector for every
\(x\). Its orbit average has that same conditional expectation. Their
difference therefore has conditional mean zero. Taking an inner product with
the predictable \(w_{t-1,h}\) and applying the deterministic normalization
preserves zero conditional mean. No factorization of the joint field is used.
\(\square\)

For a bet \(\beta\in[-\beta_{\max},\beta_{\max}]\) with
\(\beta_{\max}<1\), define

\[
L_{t,x,h,\beta}=1+\beta s_{t,x,h}.
\]

Then

\[
L_{t,x,h,\beta}\ge 1-\beta_{\max}>0,
\qquad
\mathbb E_0[L_{t,x,h,\beta}\mid\mathcal F_{t-1}]=1.
\]

Thus \(L_{t,x,h,\beta}\) is a one-round e-factor. A grid containing both
positive and negative \(\beta\) handles either direction of change. A prior
over several magnitudes adapts to unknown effect size at a logarithmic mixture
cost.

### 2.1 A useful warning: average only after compounding

Orbit centering implies

\[
\sum_{x\in\mathcal X}s_{t,x,h}=0.
\]

Consequently, an instantaneous uniform mixture over locations is identically
uninformative:

\[
\frac1M\sum_x L_{t,x,h,\beta}
=1+\frac{\beta}{M}\sum_xs_{t,x,h}
=1.
\]

Therefore the algorithm must keep fixed-location wealths

\[
W_{k:t,x,h,\beta}
=\prod_{i=k}^{t}L_{i,x,h,\beta}
\]

and average those *after* temporal compounding. The operation

\[
\frac1M\sum_x\prod_iL_{i,x}
\]

is not interchangeable with

\[
\prod_i\frac1M\sum_xL_{i,x}.
\]

The first accumulates evidence for a persistent unknown location; the second
is exactly one under this centering. This is an elementary algebraic fact, not
a new statistical theorem, but it is a decisive implementation detail.

### 2.2 Signal attenuation and a genuine blind spot

Suppose the post-change mean shift in one scalar feature is \(\delta\) on a
subset \(S\subset\mathcal X\) and zero elsewhere. Orbit centering changes the
mean by

\[
\begin{cases}
(1-|S|/M)\delta,&x\in S,\\
-(|S|/M)\delta,&x\notin S.
\end{cases}
\]

Sparse changes lose little signal when \(|S|\ll M\). A uniform orbit-wide
change is removed completely. The method is therefore designed for
*symmetry-breaking or localized* drift, not common-mode drift. A separate
global detector is needed for the latter.

### 2.3 Robustification when symmetry is only approximately certified

Finite calibration by itself does not prove the conditional mean-zero
property. Suppose instead that a predictable, simultaneous bound is available:

\[
\left|
\mathbb E_0[s_{t,x,h}\mid\mathcal F_{t-1}]
\right|
\le \varepsilon_{t,x,h}.
\]

Then

\[
\widetilde L_{t,x,h,\beta}
=
\frac{1+\beta s_{t,x,h}}
     {1+|\beta|\varepsilon_{t,x,h}}
\]

is nonnegative and satisfies

\[
\mathbb E_0[\widetilde L_{t,x,h,\beta}\mid\mathcal F_{t-1}]
\le1.
\]

The calibration price is explicit:

\[
\log \widetilde L
=\log(1+\beta s)
-\log(1+|\beta|\varepsilon).
\]

The bound \(\varepsilon\) must itself be justified uniformly over the claimed
null class and under the relevant conditioning. Plugging in a sample mean or
standard error without an anytime-valid or held-out envelope does not preserve
the theorem.

In the rest of the note, \(L_{t,j}\) denotes either the exact factor or this
conservatively normalized factor, and

\[
j=(x,h,\beta)\in\mathcal J.
\]

## 3. Two monitoring guarantees that must not be conflated

### 3.1 Mode A: probability of ever making a false alarm

Choose a proper prior \((\rho_k)_{k\ge1}\) over changepoint starts,
\(\rho_k>0\), \(\sum_{k\ge1}\rho_k=1\), and a prior
\((\pi_j)_{j\in\mathcal J}\) over location/shape/bet components. Let

\[
\rho_{>t}=\sum_{k>t}\rho_k.
\]

For every component, initialize \(A_{0,j}=0\) and update

\[
A_{t,j}=L_{t,j}\bigl(A_{t-1,j}+\rho_t\bigr).
\]

The global process is

\[
\boxed{
E_t=\rho_{>t}+\sum_{j\in\mathcal J}\pi_jA_{t,j}
}
\]

or, equivalently,

\[
E_t
=\rho_{>t}
+\sum_{k=1}^{t}\rho_k
  \sum_j\pi_j\prod_{i=k}^{t}L_{i,j}.
\]

The tail \(\rho_{>t}\) is the still-unspent prior mass for future changes.

#### Theorem 2: anytime-valid false-alarm control

If every \(L_{t,j}\) is a nonnegative conditional e-factor under the null, then
\((E_t)\) is a nonnegative supermartingale with \(E_0=1\). Hence

\[
\mathbb P_0\!\left(\sup_{t\ge0}E_t\ge\frac1\alpha\right)
\le\alpha.
\]

**Proof.** Conditional on \(\mathcal F_{t-1}\),

\[
\mathbb E_0[A_{t,j}\mid\mathcal F_{t-1}]
\le A_{t-1,j}+\rho_t.
\]

After mixing over \(j\) and using
\(\rho_{>t-1}=\rho_t+\rho_{>t}\),

\[
\mathbb E_0[E_t\mid\mathcal F_{t-1}]\le E_{t-1}.
\]

Ville's inequality gives the crossing-probability statement. \(\square\)

Two valid stopping rules are

\[
\tau_{\mathrm{mix}}
=\inf\{t:E_t\ge1/\alpha\}
\]

and the more conservative weighted scan

\[
\tau_{\mathrm{scan}}
=\inf\left\{
t:\max_j\pi_jA_{t,j}\ge1/\alpha
\right\}.
\]

The scan event implies the mixture event because
\(E_t\ge\max_j\pi_jA_{t,j}\), so it inherits the same probability bound.
Taking an unweighted maximum at the same \(1/\alpha\) threshold generally does
not account for multiplicity.

A convenient heavy-tailed start prior is

\[
\rho_k=\frac1{k(k+1)},\qquad \rho_{>t}=\frac1{t+1}.
\]

It gives late changepoint \(k\) a penalty
\(\log(1/\rho_k)=\log(k(k+1))\), rather than the linear-in-\(k\) log penalty
of a geometric prior. No proper prior can avoid some penalty for arbitrarily
late starts while retaining one global probability-of-ever-alarm guarantee.

If the detector is reset after an alarm, the theorem does not automatically
cover repeated restarts. Alpha spending or another explicitly valid repeated-
testing protocol is then required.

### 3.2 Mode B: indefinite monitoring with an ARL guarantee

For each component initialize \(R_{0,j}=0\) and use the
Shiryaev--Roberts recursion

\[
R_{t,j}=(R_{t-1,j}+1)L_{t,j}.
\]

Mix with the fixed prior:

\[
\boxed{
R_t=\sum_j\pi_jR_{t,j}.
}
\]

The expansion is

\[
R_{t,j}
=\sum_{k=1}^{t}\prod_{i=k}^{t}L_{i,j}.
\]

#### Theorem 3: e-detector and average-run-length control

Under the same conditional e-factor assumption, \(R_t\) is an e-detector:
for every integrable stopping time \(\tau\),

\[
\mathbb E_0[R_\tau]\le\mathbb E_0[\tau].
\]

Consequently, for

\[
\tau_\gamma=\inf\{t\ge1:R_t\ge\gamma\},
\]

\[
\boxed{\mathbb E_0[\tau_\gamma]\ge\gamma.}
\]

**Proof sketch.** The recursion gives

\[
\mathbb E_0[R_t\mid\mathcal F_{t-1}]
\le R_{t-1}+1,
\]

so \(R_t-t\) is a supermartingale. Apply optional sampling first to
\(\tau\wedge n\), then pass to the limit using nonnegativity and standard
integrability conditions. If \(\tau_\gamma\) has positive probability of
being infinite, its expectation is already infinite; otherwise
\(R_{\tau_\gamma}\ge\gamma\), yielding
\(\gamma\le\mathbb E R_{\tau_\gamma}\le\mathbb E\tau_\gamma\).
\(\square\)

A weighted component scan

\[
\tau_{\gamma,\mathrm{scan}}
=\inf\left\{
t:\max_j\pi_jR_{t,j}\ge\gamma
\right\}
\]

also has ARL at least \(\gamma\), since
\(R_t\ge\max_j\pi_jR_{t,j}\) implies
\(\tau_\gamma\le\tau_{\gamma,\mathrm{scan}}\).

This SR statistic is an **e-detector, not an e-process**. The ARL statement
does not imply

\[
\mathbb P_0(\tau_\gamma<\infty)\le1/\gamma.
\]

Indeed, a repeatedly sensitive long-running detector can eventually false
alarm with probability one while still having a large mean time to alarm.
Run 6 must select the intended operational guarantee before comparing delays.

## 4. A precise streaming algorithm

### Inputs

- orbit \(\mathcal X=G/H\);
- finite shape/channel bank \(\mathcal H\);
- bet grid \(\mathcal B\subset(-1,1)\), including both signs when needed;
- fixed prior
  \(\pi_{x,h,\beta}=\pi_x\pi_{h\mid x}\pi_{\beta\mid x,h}\);
- either \((\alpha,\rho)\) for Mode A or \(\gamma\) for Mode B;
- predictable template update rule;
- exact symmetry assumption or predictable calibration-error envelope.

### Per-round procedure

1. Before seeing \(Y_t\), freeze \(w_{t-1,h}\), bet grid/weights, active
   components, and all calibration envelopes.
2. Observe \(Y_t\) and compute bounded local features
   \(\Phi_{t,h}(x)\).
3. Compute orbit means and local scores \(s_{t,x,h}\).
4. Form exact or robustified e-factors \(L_{t,x,h,\beta}\).
5. Update every \(A_{t,j}\) in Mode A or \(R_{t,j}\) in Mode B.
6. Aggregate the fixed-component statistics and compare with the selected
   threshold.
7. If an alarm occurs, report the normalized evidence allocation over
   location, shape, effect size, and candidate start.
8. Only after scoring round \(t\), update calibration sufficient statistics
   and templates for use at \(t+1\).

Memory is \(O(|\mathcal J|)\), independent of elapsed time. Arithmetic is
\(O(|\mathcal J|)\) per round after the feature field has been computed.

### Pseudocode

```text
initialize predictable templates and calibration state
initialize A[j] = 0 or R[j] = 0

for t = 1, 2, ...:
    freeze all objects that must be F[t-1]-measurable
    observe Y[t]
    Phi = bounded_correlation_field(Y[t])
    Phi_bar[h] = orbit_average_x Phi[x,h]

    for each (x,h):
        s[x,h] = normalized_inner_product(
            w_previous[h], Phi[x,h] - Phi_bar[h]
        )

    for each j = (x,h,beta):
        L[j] = calibrated_e_factor(s[x,h], beta)
        if mode == ANYTIME_PROBABILITY:
            A[j] = L[j] * (A[j] + rho[t])
        else if mode == ARL:
            R[j] = L[j] * (R[j] + 1)

    aggregate and test the mode-specific threshold
    if alarm:
        report evidence map and stop (or invoke a separately valid reset rule)

    update templates/calibration using Y[t] for the next round
```

## 5. Sparse patches, shape banks, and spatial dependence

The component index need not represent a single pixel. Let \(h\) define a
structured patch, oriented edge, plaquette, correlation stencil, or
multiscale kernel. Translate that shape over the group orbit and reduce its
bounded feature vector to one normalized scalar score. The finite prior over
\((x,h,\beta)\) then pays an explicit log complexity penalty.

This is preferable to multiplying same-round local e-factors over a candidate
subset:

\[
\prod_{x\in S}L_{t,x}.
\]

Such a product is not generally an e-factor when locations are dependent.
Spatial products are valid only if a suitable conditional factorization or
another direct expectation bound has been proved. Safe alternatives are:

- one bounded score for each structured patch;
- a prior over a finite dictionary of patches;
- a bounded weighted sum followed by one e-factor; or
- a model-based joint likelihood ratio when the joint law is specified.

An arbitrary subset scan over \(2^M\) subsets is both statistically expensive
and computationally infeasible. A scientifically motivated shape dictionary
turns sparsity assumptions into inspectable model choices.

### Predictable expert mixtures

At a single round, a predictable convex combination

\[
\sum_j p_{t-1,j}L_{t,j}
\]

is also an e-factor. Multiplying these factors connects the method to online
learning with experts. It permits the favored location to switch over time.
That can help moving anomalies, but it changes the alternative and can weaken
localization of a persistent fixed site.

For the fixed-location problem, maintaining one compounded wealth per
component and then mixing them is the cleaner default. It pays a transparent
\(\log(1/\pi_j)\) price for not knowing the component.

## 6. Growth rate and delay approximation

Let \(j_\star\) be the component best aligned with the post-change regime.
Assume, for this section only, that after the change the log factors are
stationary ergodic and integrable. Define

\[
I_j(\beta)
=\mathbb E_1[\log(1+\beta s_{t,j})].
\]

Positive \(I_j\) means exponential wealth growth. Write

\[
\delta_j=\mathbb E_1[s_{t,j}],
\qquad
v_j=\mathbb E_1[s_{t,j}^2].
\]

For small \(\beta\),

\[
I_j(\beta)
=\beta\delta_j-\frac{\beta^2v_j}{2}+O(\beta^3),
\]

so the quadratic approximation suggests

\[
\beta^\star\approx\frac{\delta_j}{v_j},
\]

clipped to the admissible interval. This is a Kelly/log-growth calculation,
not an AOC expectation-gap calculation.

For every \(|\beta|<1\), a nonasymptotic analytic bound follows from the power
series of \(\log(1+x)\):

\[
\boxed{
\beta\delta_j
-\frac{\beta^2v_j}{2(1-|\beta|)}
\le I_j(\beta)
\le\beta\delta_j.
}
\]

For the robustified factor, subtract
\(\mathbb E_1\log(1+|\beta|\varepsilon_{t,j})\) from the growth rate.

### 6.1 First-order delay formulas

If the change begins at \(\nu\), the correct component contributes
\(\rho_\nu\pi_{j_\star}W_{\nu:t,j_\star}\) to Mode A. Ignoring overshoot and
lower-order renewal terms gives

\[
\mathbb E_1[\tau-\nu+1]
\approx
\frac{
\log(1/\alpha)
+\log(1/\rho_\nu)
+\log(1/\pi_{j_\star})
}{
I_{j_\star}
}.
\]

For the SR e-detector,

\[
\mathbb E_1[\tau-\nu+1]
\approx
\frac{
\log\gamma+\log(1/\pi_{j_\star})
}{
I_{j_\star}
}.
\]

If \(\beta\), scale, or shape has a separate prior, its
\(\log(1/\text{prior mass})\) term is included in
\(\log(1/\pi_{j_\star})\).

These are Wald/renewal heuristics unless stronger iid or mixing conditions are
imposed. They are not finite-sample upper bounds. With an exactly specified
pre/post model, the likelihood-ratio factor has

\[
I=\mathrm{KL}(P_1\|P_0)
\]

per observation and is the natural model-aware benchmark. A bounded linear
bet sacrifices model-specific efficiency in exchange for robustness and
simple validity; it cannot be assumed to beat the correct likelihood ratio.

### Proposition 4: minimax location prior on a transitive orbit

If all \(M\) locations are a priori symmetric and have equal post-change
information rate, the uniform prior minimizes the worst location penalty:

\[
\min_{\pi_x>0,\ \sum_x\pi_x=1}
\max_x\log\frac1{\pi_x}
=\log M,
\]

with equality at \(\pi_x=1/M\).

**Proof.** Some location has \(\pi_x\le1/M\), so the maximum penalty is at
least \(\log M\). Uniform weights attain it. \(\square\)

For \(\mathcal X=G/H\), the location price is therefore
\(\log|G/H|\), not necessarily \(\log|G|\). For a nontransitive action, each
orbit should be treated separately and receive an explicit between-orbit
prior. This proposition is elementary decision theory, not a claim of
literature novelty.

## 7. Localization as evidence allocation

For the SR statistic,

\[
R_{t,j}
=\sum_{k=1}^{t}W_{k:t,j},
\qquad
W_{k:t,j}=\prod_{i=k}^{t}L_{i,j}.
\]

This gives a natural joint evidence map

\[
q_t(k,j)
=
\frac{\pi_jW_{k:t,j}}
{\sum_{\ell=1}^{t}\sum_{u\in\mathcal J}
\pi_uW_{\ell:t,u}}.
\]

In Mode A, replace the numerator by
\(\rho_k\pi_jW_{k:t,j}\). Marginalizing over \(k,h,\beta\) gives a location
heatmap; marginalizing over \(x,h,\beta\) gives a candidate changepoint
distribution.

These normalized weights are **evidence allocations**, not automatically
Bayesian posterior probabilities. They become a Bayesian posterior (or the
posterior conditional on a change having occurred) only when the factors are
genuine likelihood ratios for the declared generative components and the
weights are genuine priors. Otherwise:

- a \(95\%\) mass set is not a \(95\%\) confidence/credible set;
- calibration of localization probabilities is not guaranteed;
- top-\(k\) localization and distance error must be evaluated empirically or
  supported by a separate theorem.

### Proposition 5: fixed-start evidence concentration

Fix a candidate start \(k\). Suppose under the post-change law that
\(\{\log L_{t,j}\}_{t\ge k}\) is stationary ergodic and integrable for each
finite \(j\), and that one component has a unique largest rate:

\[
I_{j_\star}>\max_{j\ne j_\star}I_j.
\]

Then normalized fixed-start wealth concentrates on \(j_\star\):

\[
\frac{\pi_jW_{k:t,j}}
{\sum_u\pi_uW_{k:t,u}}
\longrightarrow
\mathbf 1\{j=j_\star\}
\quad\text{almost surely}.
\]

Moreover,

\[
\frac1{t-k+1}
\log\frac{\pi_jW_{k:t,j}}
         {\pi_{j_\star}W_{k:t,j_\star}}
\longrightarrow I_j-I_{j_\star}.
\]

**Proof.** Apply the ergodic theorem to each log-wealth ratio and use the
finite component bank. \(\square\)

For the start-summed SR evidence, the same conclusion requires additional
conditions ensuring that the sums over candidate starts have the claimed
exponential rates. It should not be asserted from Proposition 5 alone.

## 8. Additive and distributed updates

The detector recursion itself has constant memory in time. Templates and
calibration models can also be based on mergeable sufficient statistics. For
a feature vector \(u_t\), maintain

\[
N_t=N_{t-1}+a_t,\qquad
S_t=S_{t-1}+a_tu_t,\qquad
Q_t=Q_{t-1}+a_tu_tu_t^\top.
\]

Independent sites can merge summaries exactly:

\[
(N,S,Q)=(N_A+N_B,\;S_A+S_B,\;Q_A+Q_B).
\]

This supports mean directions, covariance-aware directions, and AOC state
differences chosen for the next round. Exponential forgetting uses

\[
S_t=\lambda S_{t-1}+u_t,\qquad
Q_t=\lambda Q_{t-1}+u_tu_t^\top.
\]

An exact finite sliding window additionally needs a ring buffer, or the
summary of the item being removed.

The order of operations matters:

> score \(Y_t\) with the template frozen at \(t-1\), then update the template.

Learning a direction from \(Y_t\) and using it to bet on the same \(Y_t\)
breaks predictability. Updating after scoring preserves predictability, but it
does not by itself prove the null centering/calibration bound.

For numerical stability, maintain log component statistics and normalize
location evidence with log-sum-exp. Very small components can underflow even
when their total prior mass is scientifically important.

Data-dependent pruning based on the current observation can invalidate the
guarantee. Safe pruning must be decided predictably, or the discarded
component/prior mass must remain represented by a conservative aggregate.

## 9. FFT and group-convolution acceleration

### 9.1 Abelian translations

Let \(X_{t,r}(x)\) be an orbit-centered residual correlation channel, and let
\(w_{h,r}(u)\) be a translated template. The score numerator at every
translation is a cross-correlation:

\[
a_t(x,h)
=\sum_{r=1}^{p}\sum_u
w_{h,r}(u)X_{t,r}(x+u).
\]

On a periodic abelian lattice,

\[
\widehat a_t(\omega,h)
=\sum_{r=1}^{p}
\overline{\widehat w_{h,r}(\omega)}
\widehat X_{t,r}(\omega).
\]

Template spectra can be precomputed. Computing all translated scores costs
approximately

\[
O\!\left((p+|\mathcal H|)M\log M
+|\mathcal H|pM\right),
\]

using \(p\) forward transforms, one inverse transform per shape, and spectral
channel accumulation. A direct implementation with support size \(r_0\)
costs roughly

\[
O(|\mathcal H|p r_0M).
\]

Direct convolution is usually better for tiny local stencils; FFT becomes
attractive for dense kernels, many translations, or reusable channel
transforms. After score computation, evidence updates still cost
\(O(M|\mathcal H||\mathcal B|)\) and may dominate.

### 9.2 Nonabelian groups and homogeneous spaces

For a finite or compact nonabelian group, group convolution becomes block
multiplication over irreducible representations. On \(G/H\), one may lift to
\(G\) or use a spherical/homogeneous-space Fourier transform. This is the
established harmonic-analysis basis of group-equivariant convolution.

There is no universal \(O(M\log M)\) algorithm for every group and
representation. Complexity depends on available fast transforms, irrep
dimensions, multiplicities, sparsity, and the cost of moving between \(G/H\)
and \(G\). “FFT acceleration” must therefore be benchmarked for the actual
symmetry rather than claimed generically.

## 10. Relation to AOC, Jordan decomposition, and invariant testing

Suppose the local representation is a positive unit-trace state
\(R_t(x)\), and define its orbit mean

\[
\overline R_t=\frac1M\sum_xR_t(x).
\]

For a predictable effect \(0\preceq E_{t-1,h}\preceq I\),

\[
s_{t,x,h}
=\operatorname{Tr}\!\left[
E_{t-1,h}(R_t(x)-\overline R_t)
\right]\in[-1,1].
\]

Conditional orbit invariance again gives conditional mean zero. Thus an
AOC-learned effect can instantiate the local score, provided it is learned
from prior/calibration data and the stated invariance or calibration envelope
is valid.

Given a training difference

\[
\Delta=\rho_1-\rho_0,
\]

the effect maximizing a one-step expectation gap is

\[
E^\star=\mathbf 1_{\Delta>0},
\qquad
\max_{0\preceq E\preceq I}\operatorname{Tr}(E\Delta)
=\operatorname{Tr}(\Delta_+).
\]

This is the standard Jordan/Helstrom result. Under an accessible observable
algebra \(\mathcal A\), the corresponding solution uses the positive part of
the conditional expectation of \(\Delta\) onto \(\mathcal A\). Group averaging
and restricted/invariant tests also have established statistical and quantum
decision-theoretic foundations.

Three objectives must remain separate:

1. **AOC/Jordan:** maximize one-step mean separation;
2. **Kelly/e-process design:** maximize expected log evidence growth under
   validity constraints;
3. **quickest change detection:** minimize an explicitly chosen delay
   functional subject to an ARL or false-alarm constraint.

The optimizer of one is not generally the optimizer of the others. A
covariance-whitened direction, likelihood score, logistic classifier, or
directly optimized betting direction can beat the positive-eigenspace effect
on delay while the AOC effect has a larger raw mean gap.

## 11. Prior-art and novelty boundary

| Ingredient | Established foundation | What Run 6 may honestly add |
|---|---|---|
| Local/spatial scan | Spatial scan statistics, graph scans, matched-filter and multiple-testing literature | One declared symmetry-orbit feature/shape bank and reproducible benchmark |
| Unknown affected sensors/sites | Multisensor mixture/GLR/CUSUM/SR change detection | Bounded nonparametric factors specialized to a group orbit |
| Sequential likelihood accumulation | Page CUSUM, Shiryaev--Roberts, Pollak and mixture rules | A common implementation exposing both FWER-style and ARL modes |
| E-values/e-processes/e-detectors | Safe anytime-valid inference, bounded-mean betting, e-SR/e-CUSUM and e-detector mixtures | Orbit-centered local factor construction and explicit location evidence |
| Symmetry/invariance | Hunt--Stein/invariant testing, group averaging and symmetry-constrained quantum tests | Conditional orbit centering under arbitrary within-round dependence |
| FFT/group convolution | Classical convolution theorem and group-equivariant/homogeneous-space convolution | Reuse for all translated detector templates, with honest complexity accounting |
| Positive spectral witness | Jordan decomposition, Helstrom discrimination, restricted-measurement norms | A predictable bounded feature inside the sequential detector |
| Adaptive expert weights | Prediction with expert advice and predictable e-value mixtures | Comparison of moving-location and fixed-location alternatives |

In particular, Run 6 must not claim the first:

- scan statistic;
- sparse multisensor change detector;
- mixture CUSUM or Shiryaev--Roberts method;
- e-process/e-detector for change detection;
- symmetry test or group convolution;
- positive-eigenspace discriminant;
- online expert mixture; or
- quantum change detector.

The e-detector literature itself already contains an example described as a
change from symmetry to asymmetry. That example concerns distributional
central symmetry; it does not by itself establish this exact
group-orbit/location construction, but it prevents broad “first symmetry
e-detector” language.

### 11.1 Candidate contribution, stated at the right level

The defensible candidate contribution is:

> A symmetry-orbit implementation of fixed-location sequential evidence in
> which same-round orbit centering produces bounded conditionally mean-zero
> local scores without assuming spatial independence; compounded component
> wealths are combined through either a proper-prior e-process or an
> SR e-detector; the same recursion yields an interpretable location/start
> evidence map and admits translation/group-convolution acceleration.

Even this should initially be described as a **candidate synthesis**. A
literature-wide priority search and comparison with permutation martingales,
exchangeability e-processes, invariant sequential tests, and scan-e-value
methods is still required before making a novelty claim.

Potential theorem-level advances that would be genuinely more substantial are:

1. validity and sharp power under a useful composite *approximately invariant*
   null, with a data-driven but provably simultaneous calibration envelope;
2. minimax or near-minimax sparse detection boundaries on \(G/H\);
3. finite-sample localization-error bounds, not merely normalized evidence
   plots;
4. regret/delay guarantees for adaptive moving versus fixed locations;
5. computational-statistical tradeoffs for truncated group Fourier scans; and
6. a proof that a restricted AOC/template update improves a declared delay
   criterion over a specified baseline under explicit assumptions.

## 12. Falsifiable experimental program

An algorithmic advantage is not visible from these equations alone. It must be
tested at matched information and resource budgets.

### 12.1 Required baselines

For a specified generative model and accessible stream, compare against:

- exact likelihood-ratio scan and model-aware CUSUM/SR oracle;
- mixture/GLR multisensor change detector;
- covariance or Hotelling/CUSUM score;
- matched parity/correlation threshold and logistic detector;
- spatial or graph scan statistic;
- MMD/kernel sequential detector where appropriate;
- AOC/Jordan direction, log-growth-optimized direction, and simple raw
  correlation direction;
- Wilson/decoder likelihood diagnostics in QEC settings; and
- eSCD only when the measurement/observable setting is genuinely matched.

No method may receive a richer observation, more calibration samples, more
shots, a different false-alarm criterion, or an unreported hyperparameter
search.

### 12.2 Metrics

Report:

- empirical probability of ever alarming for Mode A over the declared horizon;
- null ARL and full run-length distribution for Mode B;
- conditional detection delay versus change time and effect size;
- top-\(1\), top-\(k\), and distance-based localization error;
- robustness to spatial dependence and calibrated symmetry breaking;
- sensitivity to shape-bank and bet-grid misspecification;
- calibration samples, physical shots/copies, memory, latency, and energy where
  relevant; and
- direct versus FFT/group-transform wall time at several orbit sizes.

### 12.3 Critical ablations

- orbit centering versus a known/learned location-specific baseline;
- compounding fixed locations before mixing versus instantaneous uniform
  mixing;
- proper-prior e-process versus SR e-detector at their correct guarantees;
- exact symmetry versus robustified \(\varepsilon_t\) factors;
- fixed versus predictable adaptive templates;
- one shape versus multiscale shape bank;
- direct convolution versus FFT; and
- AOC direction versus log-growth-optimized and likelihood directions.

### 12.4 What would count as an advantage

A credible practical advantage would be one of the following, under matched
access and budgets:

- lower delay at the same verified false-alarm guarantee;
- better localization at the same delay and false-alarm guarantee;
- comparable detection with materially less calibration;
- exact robustness to arbitrary same-round spatial dependence when a competing
  calibration assumes independence; or
- comparable statistics with lower measured computation through reusable
  group transforms.

A small accuracy or delay improvement on one simulated dataset is not evidence
of universal sample efficiency, scalability, quantum acceleration, or
optimality.

## 13. Applications and their limits

The same mathematical problem can occur in:

- **quantum error correction:** unknown-location syndrome-correlation drift on
  symmetry-equivalent checks, compared against decoder likelihood and Wilson
  diagnostics;
- **optics and vision:** localized phase, polarization, aberration, or defect
  changes across translated/rotated sensor or image patches;
- **robotics:** contact, slip, wear, or damage patches on tactile skins and
  repeated joints;
- **chemistry and materials:** changes on symmetry-equivalent atoms, bonds, or
  local difference-density/correlation features;
- **condensed matter:** nucleation or symmetry-breaking correlation drift on a
  lattice; and
- **lattice gauge theory/tensor networks:** changes in gauge-invariant
  plaquette, charge, or flux-sector observables over equivalent regions.

The last item is a concrete computational direction, not a solution to string
theory or holographic duality. Gauge constraints can create nonfactorizing
regional algebras and centers, so the accessible features and orbit action
must be defined physically. The method cannot recover information absent from
the accessible observable algebra.

## 14. Primary sources and research positioning

### Sequential testing and e-values

- Page, “Continuous Inspection Schemes,” 1954:
  [Biometrika](https://doi.org/10.1093/biomet/41.1-2.100).
- Roberts, “A Comparison of Some Control Chart Procedures,” 1966:
  [Technometrics](https://doi.org/10.1080/00401706.1966.10490374).
- Pollak, “Optimal Detection of a Change in Distribution,” 1985:
  [Annals of Statistics](https://doi.org/10.1214/aos/1176346587).
- Tartakovsky, “Asymptotic Optimality of Mixture Rules for Detecting Changes
  in General Stochastic Models”:
  [arXiv:1807.08980](https://arxiv.org/abs/1807.08980).
- Ramdas, Grünwald, Vovk, and Shafer, “Game-Theoretic Statistics and Safe
  Anytime-Valid Inference”:
  [Statistical Science](https://doi.org/10.1214/23-STS894).
- Waudby-Smith and Ramdas, “Estimating Means of Bounded Random Variables by
  Betting”:
  [arXiv:2010.09686](https://arxiv.org/abs/2010.09686).
- Shin, Ramdas, and Rinaldo, “E-detectors: A Nonparametric Framework for
  Sequential Change Detection”:
  [NEJSDS](https://nejsds.nestat.org/journal/NEJSDS/article/59/read).
- Dandapanthula and Ramdas, “Multiple Testing in Multi-Stream Sequential
  Change Detection”:
  [arXiv:2501.04130](https://arxiv.org/abs/2501.04130).
- Vovk and Wang, “E-values: Calibration, Combination and Applications” and
  sequential merging context:
  [arXiv:2007.06382](https://arxiv.org/abs/2007.06382).

### Scan and sparse multisensor detection

- Kulldorff, “A Spatial Scan Statistic”:
  [Communications in Statistics](https://doi.org/10.1080/03610929708831995).
- Xie and Siegmund, “Sequential Multi-Sensor Change-Point Detection”:
  [Annals of Statistics manuscript](https://www2.isye.gatech.edu/~yxie77/multisensor_published.pdf).
- Wang, Neill, and Chen, “Calibrated Nonparametric Scan Statistics for
  Anomalous Pattern Detection in Graphs”:
  [AAAI](https://doi.org/10.1609/aaai.v36i4.20339).

### Symmetry, convolution, and online experts

- Cohen and Welling, “Group Equivariant Convolutional Networks”:
  [ICML/PMLR](https://proceedings.mlr.press/v48/cohenc16.html).
- Kondor and Trivedi, “On the Generalization of Equivariance and Convolution
  in Neural Networks to the Action of Compact Groups”:
  [ICML/PMLR](https://proceedings.mlr.press/v80/kondor18a.html).
- Cohen, Geiger, and Weiler, “A General Theory of Equivariant CNNs on
  Homogeneous Spaces”:
  [NeurIPS](https://papers.nips.cc/paper_files/paper/2019/hash/b9cfe8b6042cf759dc4c0cccb27a6737-Abstract.html).
- Bousquet and Warmuth, “Tracking a Small Set of Experts by Mixing Past
  Posteriors”:
  [JMLR](https://www.jmlr.org/papers/v3/bousquet02b.html).
- Hunt--Stein theorem overview:
  [Encyclopedia of Mathematics](https://encyclopediaofmath.org/wiki/Hunt-Stein_theorem).

### Quantum decision theory and quantum change detection

- Helstrom, *Quantum Detection and Estimation Theory*:
  [Elsevier](https://shop.elsevier.com/books/quantum-detection-and-estimation-theory/helstrom/978-0-12-340050-5).
- Matthews, Wehner, and Winter, “Distinguishability of Quantum States Under
  Restricted Families of Measurements”:
  [Communications in Mathematical Physics](https://doi.org/10.1007/s00220-009-0890-5).
- Hiai, Mosonyi, and Hayashi, “Quantum Hypothesis Testing with Group
  Symmetry”:
  [arXiv:0904.0704](https://arxiv.org/abs/0904.0704).
- Zecchin, Simeone, and Ramdas, “Quantum Sequential Change Detection Through
  Classical Shadows”:
  [arXiv:2602.11846](https://arxiv.org/abs/2602.11846).

## Bottom line

Run 6 has a rigorous algorithmic core:

\[
\text{conditional orbit symmetry}
\Longrightarrow
\text{bounded mean-zero local score}
\Longrightarrow
\text{fixed-location e-factors}
\Longrightarrow
\begin{cases}
\text{proper-prior e-process with ever-alarm control},\\
\text{mixture SR e-detector with ARL control}.
\end{cases}
\]

It also has a practical implementation path through additive sufficient
statistics and group convolution. What it does **not** yet have is evidence
that these choices beat matched classical or model-aware alternatives. The
next scientific step is a preregistered, same-budget sparse-drift benchmark,
with the guarantee, observation algebra, calibration budget, and localization
criterion fixed before results are inspected.
