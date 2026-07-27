# Run 6 Google executable method lock

**Status:** normative specification frozen before held-value access; no held
detector or decoder/outcome values were inspected before freeze.
**Purpose:** remove every value-dependent implementation choice before the
held replay.
**Scope:** Google 2022 primary arm only. The separately hashed configuration,
code, tests and freeze manifest agree with this specification at freeze.

## 1. Locked decision

Use the following fixed design:

- one paired update compares the same round role in one reference and one monitored shot;
- adaptive states are **separate for all 51 round roles** and persist across shots;
- the first 5,000 validation pairs are a common fit/warm-up block;
- the last 5,000 validation pairs are a disjoint threshold block;
- threshold and held replays both clone the same post-warm-up checkpoint;
- the primary empirical operating point is at most one alert per 100,000 validation cycles;
- “S-PACE” is a fixed M4/M5 composite, never the better held branch;
- the Google primary improvement scalar is decoder-mismatch capture among the 20 highest-ranked monitored shots, after the detector score table is frozen;
- M0 and M3 are the two predesignated advantage comparators; M0C, M1 and M2
  remain mandatory contextual controls;
- a full Run 6 advantage remains false until one separately locked auxiliary arm passes. Remove the current post-result choice between PNNL and untouched Google data.

All natural-hardware guarantees remain empirical. Exact e-process language applies only to the independently randomized complete-pair orientation audit.

## 2. Immutable indexing and data phases

### 2.1 Pair indices

Let \(q\) denote a zero-based pair index and \(r\in\{0,\ldots,50\}\) the canonical detector round role.

| Phase | Pair indices | Reference archive shot \(A(q)\) | Monitor archive shot \(B(q)\) | Purpose |
|---|---:|---:|---:|---|
| fit/warm-up | \(q=0,\ldots,4999\) | \(q\) | \(10000+q\) | M1/M2 fit and M3–M5 warm-up |
| threshold | \(q=5000,\ldots,9999\) | \(q\) | \(10000+q\) | empirical thresholds only |
| held | \(q=0,\ldots,19999\) | \(20000+q\) | \(40000+q\) | primary replay |

For every phase, process pairs in increasing \(q\), and roles in increasing
\(r\). For storage and empirical cycle bookkeeping only, the zero-based
flattened row index is

\[
n_{\rm emp}(q,r)=51q+r.
\]

Its inverse is

\[
q=\lfloor n_{\rm emp}/51\rfloor,\qquad r=n_{\rm emp}\bmod 51.
\]

Formal e-process/SR time is instead one-based complete-paired-shot time:
\(u=q_{\rm local}+1\), where \(q_{\rm local}=0,\ldots,H-1\) within the
threshold/randomization or held episode. The 51 roles at fixed \(u\) are
experts mixed in one update. Never allow batching to change either order or
to turn the empirical row index into formal time.

### 2.2 Binary parser lock

The future machine-readable lock should include:

- `detection_events.b8` has 1,224 bits and exactly 153 bytes per shot;
- every shot is independently byte aligned;
- bit \(k\) is `(record[k // 8] >> (k % 8)) & 1`;
- expected file size is \(500000\times153=76{,}500{,}000\) bytes;
- stored detector bits follow global Stim `DETECTOR` declaration order;
- each declaration is remapped by its parsed \((x,y,t)\) coordinate;
- canonical check order is increasing \((x,y)\) and must contain the same 24 unique checks for every role;
- the role-0 and role-50 declaration-order reversal must be exercised by tests;
- a held runner must reject a raw `[shots,51,24]` reshape that lacks the coordinate permutation.

The logical stream is shot-major, then role-major. This is an analysis update order, not a claim that role 50 of one shot is a continuous physical trajectory with role 0 of the next.

## 3. Common features and notation

For phase/pair \(q\), role \(r\), and side \(X\in\{A,B\}\), let

\[
e_{q,r}^{X}\in\{0,1\}^{24},
\qquad
z_{q,r}^{X}=1-2e_{q,r}^{X}\in\{-1,+1\}^{24}.
\]

Use exactly:

\[
g(e)=\frac1{24}\sum_{j=0}^{23}e_j,
\]

\[
\phi(e)=
\left(
e_0,\ldots,e_{23},
\left\{\frac{1+z_i z_j}{2}\right\}_{0\le i<j<24}
\right)\in[0,1]^{300},
\]

and

\[
R(e)=\frac{zz^\mathsf T}{24}.
\]

Feature ordering is immutable:

1. raw checks \(0,\ldots,23\) in canonical coordinate order;
2. pair equalities in lexicographic order
   \((0,1),(0,2),\ldots,(0,23),(1,2),\ldots,(22,23)\).

Define

\[
D_{q,r}=\phi(e^B_{q,r})-\phi(e^A_{q,r})\in[-1,1]^{300},
\]

\[
\Delta_{q,r}=R(e^B_{q,r})-R(e^A_{q,r}),
\qquad
d^g_{q,r}=g(e^B_{q,r})-g(e^A_{q,r}).
\]

Every score uses the current \(D,\Delta,d^g\) before updating any state with that pair.

## 4. Shared calibration and initialization

### 4.1 Common information budget

All six methods may access the same first 5,000 validation pairs, corresponding to 10,000 physical archived shots and 255,000 paired cycle-role observations. No method may fit from shots 20,000–39,999 or from any held monitor shot.

- M0 is stateless.
- M1 fits its fixed Bernoulli model.
- M2 fits its fixed role means and shrinkage precision.
- M3–M5 run their ordinary causal updates through the fit/warm-up stream; their scores and alarms from this phase are discarded.

At the end of pair 4,999, serialize one immutable method-state checkpoint. Clone it twice:

1. threshold clone: process validation pairs 5,000–9,999;
2. held clone: process the 20,000 held pairs.

Do not carry threshold-block state into the held replay. This gives threshold and held streams identical initialization and prevents threshold data from becoming hidden model fitting.

### 4.2 Role sharing

Maintain independent adaptive state for every \(r=0,\ldots,50\):

- M3: one weight vector per `(role, learning_rate)`;
- M4: one EWMA per `(role, half_life)`;
- M5: one EWMA/effect state per `(role, half_life, rank_variant)`.

No adaptive state is reset at an ordinary shot boundary. A state at role \(r\) is updated once per shot and never by another role. Consequently the existing names `half_lives_cycles` should be replaced by `half_lives_role_updates`; the numerical grids remain unchanged. This avoids pooling the distinct boundary and bulk role laws.

## 5. M0–M5 exact method definitions

### M0: paired detector firing rate

The causal paired score is

\[
s^{(0)}_{q,r}=d^g_{q,r}\in[-1,1].
\]

Use eight exact components

\[
\beta\in
\{-0.9,-0.6,-0.3,-0.1,0.1,0.3,0.6,0.9\}
\]

with prior \(1/8\), each having

\[
L^{(0,\beta)}_{q,r}=1+\beta s^{(0)}_{q,r}.
\]

For the empirical two-sided comparison define

\[
Z^{(0)}_{q,r}=|s^{(0)}_{q,r}|.
\]

The conventional unpaired DFR is secondary only:

\[
Z^{(0u)}_{q,r}=g(e^B_{q,r}),
\qquad
Z^{(0u)}_q=\max_r Z^{(0u)}_{q,r}.
\]

It receives a separately calibrated threshold and does not replace paired M0 in the advantage gate.

### M0C: within-shot two-sided Page--CUSUM control

This empirical control addresses the requested CUSUM comparison without
pretending that a fixed archived stream supplies a known stationary
likelihood. It uses no correlation feature beyond the 24 raw paired check
differences.

At the start of every paired shot, initialize, for the global channel and all
24 canonical check channels,

\[
C^+_{\kappa,c}=C^-_{\kappa,c}=0,
\qquad
\kappa\in\{0.01,0.05,0.1\}.
\]

At role \(r\), set

\[
x_{r,0}=d^g_{q,r},
\qquad
x_{r,j+1}=e^B_{q,r,j}-e^A_{q,r,j},
\]

and update

\[
C^+_{\kappa,c}
=
\max\{0,C^+_{\kappa,c}+x_{r,c}-\kappa\},
\qquad
C^-_{\kappa,c}
=
\max\{0,C^-_{\kappa,c}-x_{r,c}-\kappa\}.
\]

Its empirical cycle score is

\[
Z^{(0C)}_{q,r}
=
\max_{\kappa,c}\{C^+_{\kappa,c},C^-_{\kappa,c}\}.
\]

State resets at every ordinary shot boundary whether or not an alert occurs,
so scores do not depend on the threshold. This is explicitly a
within-shot Page--CUSUM scan, not a cross-shot quickest-change optimum and
not an e-factor. It is mandatory contextual control but does not enter the
predesignated advantage Boolean.

### M1: fixed diagonal likelihood surprise

Use only fit/warm-up pairs. For each role \(r\) and check \(j\), pool both sides:

\[
c_{rj}
=
\sum_{q=0}^{4999}
\left(e^A_{q,r,j}+e^B_{q,r,j}\right),
\]

\[
\widehat p_{rj}
=
\operatorname{clip}
\left(
\frac{c_{rj}+1/2}{10000+1},
10^{-4},
1-10^{-4}
\right).
\]

Freeze these probabilities. Define the average diagonal negative log likelihood

\[
\ell_r(e)
=
-\frac1{24}\sum_{j=0}^{23}
\left[
e_j\log\widehat p_{rj}
+(1-e_j)\log(1-\widehat p_{rj})
\right].
\]

With

\[
C_\epsilon
=
\log\frac{1-10^{-4}}{10^{-4}},
\]

use the bounded antisymmetric score

\[
s^{(1)}_{q,r}
=
\frac{\ell_r(e^B_{q,r})-\ell_r(e^A_{q,r})}
{C_\epsilon}
\in[-1,1].
\]

Use the same eight signed bet fractions and uniform prior as M0:

\[
L^{(1,\beta)}_{q,r}=1+\beta s^{(1)}_{q,r}.
\]

The empirical statistic is

\[
Z^{(1)}_{q,r}=|s^{(1)}_{q,r}|.
\]

Do not update \(\widehat p\) in the threshold or held streams.

### M2: role-centered Ledoit–Wolf/Hotelling control

M2 is empirical and never enters an e-process or e-SR claim.

Use \(D_{q,r}\in\mathbb R^{300}\). Select exactly 20,000 fit observations using NumPy `Generator(PCG64(610601))`:

- iterate roles \(r=0,\ldots,50\);
- select 393 distinct \(q\)'s from `0..4999` for roles \(0,\ldots,7\);
- select 392 distinct \(q\)'s for roles \(8,\ldots,50\);
- use `choice(5000, size=n_r, replace=False)` and sort each selected index list.

This gives \(8(393)+43(392)=20{,}000\) observations.

For each role, compute \(\widehat\mu_r\) from its selected observations. Pool the centered residuals

\[
x_{q,r}=D_{q,r}-\widehat\mu_r.
\]

Fit scikit-learn `LedoitWolf(store_precision=True, assume_centered=True)` in float64 to the pooled 20,000 residuals. Freeze

\[
\widehat\Omega
=
\frac{\texttt{precision\_}+\texttt{precision\_}^{\mathsf T}}2.
\]

The causal empirical score is

\[
Z^{(2)}_{q,r}
=
\max\left\{
0,\,
(D_{q,r}-\widehat\mu_r)^\mathsf T
\widehat\Omega
(D_{q,r}-\widehat\mu_r)
\right\}.
\]

Only a negative value in \([-10^{-12},0)\) may be rounded to zero; a smaller value is an error. No online centering or covariance update is allowed.

### M3: same-feature online pairwise logistic witness

Maintain \(w_{r,\eta}\in\mathbb R^{300}\) with

\[
\eta\in\{0.001,0.01,0.1\},
\qquad
\lambda_{\rm L2}=10^{-4}.
\]

There is no intercept, momentum, minibatching, shuffling or adaptive optimizer. Initialize every vector to zero before fit/warm-up.

For current \(D=D_{q,r}\), use the pre-update quantities

\[
a=w_{r,\eta}^{\mathsf T}D,
\qquad
2c=\max\{1,\|w_{r,\eta}\|_1\},
\]

\[
s^{(3,\eta)}_{q,r}
=
\tanh\left(\frac{a}{2c}\right).
\]

This is antisymmetric and satisfies
\(|s^{(3,\eta)}| \le \tanh(1)<1\).

After scoring, minimize the pairwise loss

\[
\mathcal L(w;D)
=
\log(1+\exp(-w^\mathsf TD))
+\frac{\lambda_{\rm L2}}2\|w\|_2^2
\]

by one exact SGD step:

\[
w_{r,\eta}
\leftarrow
w_{r,\eta}
+\eta\,\sigma(-a)D
-\eta\lambda_{\rm L2}w_{r,\eta},
\]

where every quantity on the right-hand side is pre-update and
\(\sigma(-a)\) is evaluated by an overflow-safe sigmoid.

The component bank is

\[
(\eta,\beta)
\in
\{0.001,0.01,0.1\}
\times
\{0.1,0.3,0.6,0.9\},
\]

with 12 equal prior weights and

\[
L^{(3,\eta,\beta)}_{q,r}
=
1+\beta s^{(3,\eta)}_{q,r}.
\]

The empirical statistic, fixed before observing outcomes, is

\[
Z^{(3)}_{q,r}
=
\max_{\eta}s^{(3,\eta)}_{q,r}.
\]

The external Kingston pilot motivates only the frozen grid. It supplies no weight state and does not select one learning rate.

### M4: role-stratified analytical top-\(k\) witness

For

\[
h\in\{4,16,64,256\},
\qquad
\lambda_h=2^{-1/h},
\]

maintain a per-role state \(M_{r,h}\in\mathbb R^{300}\), initialized to zero.
No finite-time EWMA bias correction is used. Such a correction is a positive
scalar common to all coordinates and would not alter a nonzero top-\(k\)
support or signs; omitting it also removes a tolerance-dependent early-time
branch.

For each \(k\in\{1,4,16,64\}\):

1. if \(\max_i|M_{r,h,i}|\le10^{-12}\), set \(w_{r,h,k}=0\);
2. otherwise sort coordinates by the deterministic key
   \((-\lvert M_i\rvert,i)\);
3. take the first \(k\) coordinates;
4. assign

\[
w_i=
\begin{cases}
+1/k,&M_i\ge0,\\
-1/k,&M_i<0,
\end{cases}
\]

and set every unselected coordinate to zero.

Thus exact magnitude ties use lower feature index first, and an exact zero uses positive sign.

Score before updating:

\[
s^{(4,h,k)}_{q,r}
=
w_{r,h,k}^{\mathsf T}D_{q,r}\in[-1,1].
\]

Then update only the current role:

\[
M_{r,h}
\leftarrow
\lambda_h M_{r,h}
+(1-\lambda_h)D_{q,r}.
\]

Use 64 exact components

\[
(h,k,\beta)
\in
\{4,16,64,256\}
\times
\{1,4,16,64\}
\times
\{0.1,0.3,0.6,0.9\}
\]

with uniform prior and

\[
L^{(4,h,k,\beta)}=1+\beta s^{(4,h,k)}.
\]

The empirical method statistic is

\[
Z^{(4)}_{q,r}
=
\max_{h,k}s^{(4,h,k)}_{q,r}.
\]

This empirical maximum is not itself called an e-factor.

### M5: role-stratified spectral AOC

For \(h\in\{4,16,64\}\), maintain per-role
\(M^R_{r,h}\in\mathbb R^{24\times24}\), a prior-observation count
\(n_{r,h}\), and two effects: full-positive and rank-one. Initialize all
matrices, counts and effects to zero.

Use the same \(\lambda_h=2^{-1/h}\) with no finite-time bias correction.
Positive scalar correction would leave the spectral subspaces unchanged
away from the declared eigenvalue tolerance.

Recompute effects from past data exactly when

\[
n_{r,h}>0
\quad\text{and}\quad
n_{r,h}\bmod8=0,
\]

before scoring the next observation of that role. Symmetrize the stored
EWMA first:

\[
H=(M^R_{r,h}+(M^R_{r,h})^\mathsf T)/2.
\]

Use float64 `numpy.linalg.eigh(H)` and absolute eigenvalue tolerance
\(\tau_{\rm eig}=10^{-10}\).

**Full positive effect.** If \(V_+\) contains all eigenvectors with
\(\lambda_i>\tau_{\rm eig}\), set

\[
E_+=V_+V_+^\mathsf T.
\]

If no eigenvalue passes, set \(E_+=0\).

**Rank-one effect.** If
\(\lambda_{\max}\le\tau_{\rm eig}\), set \(E_1=0\). Otherwise identify all eigenvalues within \(\tau_{\rm eig}\) of \(\lambda_{\max}\), form their projector \(P_{\max}\), choose the smallest coordinate \(j\) with
diagonal equal, within \(\tau_{\rm eig}\), to the largest diagonal of
\(P_{\max}\), and define

\[
v=
\frac{P_{\max}e_j}{\|P_{\max}e_j\|_2},
\qquad
E_1=vv^\mathsf T.
\]

This “largest diagonal, then smallest tied index” anchor avoids amplifying a
nearly zero projected coordinate. Fix the sign of \(v\) by making its first
entry with magnitude above \(\tau_{\rm eig}\) positive, although the
projector is sign invariant. This removes rank-one degeneracy freedom.

Score the current \(\Delta=\Delta_{q,r}\) before update:

\[
s^{(5,h,\mathrm{rank})}_{q,r}
=
\operatorname{Tr}(E_{\mathrm{rank}}\Delta)
\in[-1,1].
\]

After scoring:

\[
M^R_{r,h}
\leftarrow
\lambda_hM^R_{r,h}
+(1-\lambda_h)\Delta,
\qquad
n_{r,h}\leftarrow n_{r,h}+1.
\]

Use 24 equal-prior components over

\[
(h,\mathrm{rank},\beta)
\in
\{4,16,64\}
\times
\{\mathrm{rank1},\mathrm{positive}\}
\times
\{0.1,0.3,0.6,0.9\},
\]

with \(L=1+\beta s\). The empirical statistic is

\[
Z^{(5)}_{q,r}
=
\max_{h,\mathrm{rank}}s^{(5,h,\mathrm{rank})}_{q,r}.
\]

Again, this current-cycle maximum is empirical, not an e-factor.

## 6. Fixed expert and S-PACE composition

### 6.1 Exact component mixtures

Never select a learning rate, half-life, \(k\), rank or bet fraction from the held replay.

| Method | Exact components | Prior |
|---|---:|---:|
| M0 | 8 signed bets | \(1/8\) each |
| M1 | 8 signed bets | \(1/8\) each |
| M3 | 3 learning rates × 4 positive bets | \(1/12\) each |
| M4 | 4 half-lives × 4 \(k\)'s × 4 bets | \(1/64\) each |
| M5 | 3 half-lives × 2 ranks × 4 bets | \(1/24\) each |

M2 has no exact components.

The fixed exact S-PACE composite is the union of M4 and M5:

- total prior mass \(1/2\) on M4, uniform inside M4;
- total prior mass \(1/2\) on M5, uniform inside M5.

The empirical S-PACE cycle statistic is fixed as

\[
Z^{(S)}_{q,r}
=
\max\{Z^{(4)}_{q,r},Z^{(5)}_{q,r}\}.
\]

M4-only and M5-only results are secondary. The better observed branch may never replace \(S\).

### 6.2 Proper-prior e-process

For a declared episode horizon \(H\), let

\[
\rho_t=1/H,\quad t=1,\ldots,H,
\qquad
\rho_{>t}=(H-t)/H.
\]

The complete paired shot, not an individual role, is the exact randomization
unit. For shot \(u\), role \(r\), and component \(c\), compute the predictable
factor \(L_{u,r,c}\) from that role's state before incorporating shot \(u\).
Each `(role, component)` is a separate expert with fixed prior
\(\pi_c/51\). Never multiply or serially compound the 51 factors that share
one shot-orientation bit.

For every role-component expert \((r,c)\):

\[
A_{0,r,c}=0,
\qquad
A_{u,r,c}=L_{u,r,c}(A_{u-1,r,c}+1/H).
\]

For method prior \(\pi_c\):

\[
E_u=\rho_{>u}+\frac1{51}\sum_{r=0}^{50}\sum_c\pi_cA_{u,r,c}.
\]

Use:

- held replay \(H=20{,}000\) complete paired shots;
- randomization threshold-block replay \(H=5{,}000\) complete paired shots.

The first exact alarm is

\[
\tau_E=\inf\{u:E_u\ge100\}.
\]

Do not reset this process in a primary episode.

### 6.3 SR mode

For each role-component expert:

\[
R_{0,r,c}=0,
\qquad
R_{u,r,c}=(R_{u-1,r,c}+1)L_{u,r,c},
\]

\[
R_u=\frac1{51}\sum_{r=0}^{50}\sum_c\pi_cR_{u,r,c}.
\]

The first secondary alarm is

\[
\tau_R=\inf\{u:R_u\ge10^6\}.
\]

Only this first alarm receives the ARL interpretation. A repeated-alert diagnostic may reset all \(R_c\) to zero at the next shot boundary while retaining witness state, but it must be labeled empirical and must not reuse the ARL theorem.

### 6.4 Numeric accumulation

Use float64 and log-domain recursions:

\[
\log A_{t,c}
=
\log L_{t,c}
+\operatorname{logaddexp}(\log A_{t-1,c},-\log H),
\]

with \(\log0=-\infty\). Compute mixtures by `logsumexp`. Use `log1p(beta * score)` for \(\log L\). Any component with \(L=0\) has log factor \(-\infty\).

## 7. Empirical score, shot aggregation and alarm state machine

### 7.1 Common empirical cycle scores

Use exactly the \(Z^{(m)}_{q,r}\) definitions above:

| Method | Empirical cycle score |
|---|---|
| M0 | \(|d^g|\) |
| M0C | within-shot two-sided Page--CUSUM maximum |
| M1 | absolute bounded NLL difference |
| M2 | role-centered Hotelling quadratic |
| M3 | maximum over three predeclared logistic experts |
| M4 | maximum over predeclared \((h,k)\) witnesses |
| M5 | maximum over predeclared \((h,\mathrm{rank})\) witnesses |
| S | \(\max(M4,M5)\) |

These maxima are permitted only in the empirical branch whose complete multiplicity is calibrated on validation data.

The shot score is

\[
Z^{(m)}_q=\max_{r=0,\ldots,50}Z^{(m)}_{q,r}.
\]

If several roles tie within exact float equality, use the smallest role. The causal alert time is the first role satisfying the threshold, which need not be the role attaining the final shot maximum.

When an empirical maximum over experts is attained by multiple components,
attribute it to the lexicographically smallest component parameter tuple:
increasing learning rate for M3, then increasing `(half_life, k)` for M4,
and increasing `(half_life, rank1-before-positive)` for M5. An M4/M5
composite tie is attributed to M4. These attribution rules do not alter the
numeric maximum.

### 7.2 Threshold calibration

Generate threshold scores from the threshold clone only: validation pairs 5,000–9,999. There are

\[
N_{\rm val,cycle}=5000\times51=255{,}000
\]

cycle opportunities and 5,000 shot opportunities.

The primary empirical budget is

\[
b_{\rm cycle}=10^{-5},
\qquad
K_{\rm cycle}=
\left\lfloor
255000\times10^{-5}
\right\rfloor=2.
\]

For a proposed threshold \(h\), process each shot causally and count at most one alert in that shot. An alert occurs only when

\[
Z^{(m)}_{q,r}>h.
\]

After an alert, suppress notifications for the remaining roles of that shot. Continue all witness updates during suppression. There is no cooldown in the next shot and no witness reset.

Choose the **smallest** threshold among
\(\{-\infty\}\cup\{Z^{(m)}_{q,r}\}\cup\{+\infty\}\) whose state-machine alert count is at most two. Strict `>` makes ties conservative. This is the most sensitive threshold that does not exceed the budget; “most conservative” must not be interpreted as always choosing \(+\infty\).

The one-alert-per-10,000-shot point is secondary. On only 5,000 disjoint threshold shots its permitted count is zero. Set its threshold to the maximum validation shot score with strict `>` and explicitly report that this discrete point has no validation alert.

The complete threshold/count frontier must be stored before held replay.
Persist the sorted cycle-score candidates and corresponding one-alert-per-shot
counts as separate non-object arrays for every method, hash them, and write a
threshold-stage manifest with `held_values_decoded_or_scored=false`. Raw
cryptographic integrity hashing of the extracted file against the exact
member of the already verified ZIP is allowed and recorded before parsing;
it is not a statistical access. Only after the threshold-stage manifest is
durably written may the detector runner decode or score the held reference
or monitor blocks.

### 7.3 Held alarm rules

Clone the common checkpoint and use the primary threshold without retuning.

The held monitor shot for local pair \(q\) is archive shot \(40000+q\). Before the primary event window there are

\[
(57750-40000)\times51=905{,}250
\]

cycle opportunities, so the nominal pre-event budget is

\[
\left\lfloor905250/100000\right\rfloor=9
\]

alerts under the same one-per-shot state machine.

Record:

- all pre-event alert shots and first crossing roles;
- whether any alert occurs in each predeclared event window;
- the first alert `(archive_shot, role)` inside each window;
- no exact onset or physical-time delay.

The proper-prior e-process and SR modes stop at their first threshold crossing for their formal summaries. Empirical scores continue to the end because they are needed for frozen shot ranking.

## 8. Primary utility and strict-improvement rule

### 8.1 Detector freeze before outcomes

Complete and hash all cycle scores, shot scores, alert tables, thresholds and resource ledgers before opening or joining any observable/decoder outcome. Outcome processing must be a separate command that requires the detector-result manifest hash.

### 8.2 Primary downstream label and budget

The primary decoder mismatch label is fixed as

\[
y_q
=
\texttt{obs\_flips\_actual}[B(q)]
\oplus
\texttt{obs\_flips\_predicted\_by\_correlated\_matching}[B(q)].
\]

PyMatching mismatch is a secondary replication outcome.

The primary triage budget is alert fraction \(0.001\), exactly

\[
B_{\rm risk}=20
\]

of the 20,000 monitored shots. For each method, rank frozen shot scores descending. At a score tie, lower archive shot index ranks first. Let

\[
\mathcal T_m(20)
\]

be the first 20 shots, and define mismatch capture

\[
C_m(20)=\sum_{q\in\mathcal T_m(20)}y_q.
\]

No decoder label participates in ranking or tie resolution.

### 8.3 Executable Google decision

Define `google_primary_pass = true` only if all conditions hold:

1. fixed composite \(S\) has an empirical alarm inside the primary event window at its validation-locked primary threshold;
2. \(S\) has no more than nine pre-event alerts;
3. the detector table and resource ledger were frozen before outcome access;
4. \(C_S(20)\ge C_{M0}(20)+1\);
5. \(C_S(20)\ge C_{M3}(20)+1\);
6. all event-window sensitivity results and uncertainty summaries are reported;
7. M0C, M1 and M2 contextual controls are reported whether or not they favor
   \(S\); and
8. no method received additional detector records or outcome labels.

The integer \(+1\) is the strict minimum effect. Equal capture is a tie and
fails. M0C/M1/M2 results cannot rescue a failure. If any of them exceeds
\(S\), the allowed wording is at most “improvement over the predesignated DFR
and same-feature logistic comparators,” not “best overall method.”

Timing is secondary. A timing improvement requires a lead of at least one complete shot:

\[
t_S+51\le t_b
\]

in flattened cycle units. A smaller lead is reported as a timing tie because the natural event boundary is approximate. Timing and risk-coverage cannot rescue the primary capture rule.

### 8.4 Risk-coverage summaries

Report secondary fixed budgets \(B\in\{2,20,200\}\), corresponding to alert fractions
\(\{10^{-4},10^{-3},10^{-2}\}\). At each budget report:

- captured mismatches \(\sum_{q\in\mathcal T_m(B)}y_q\);
- mismatch recall;
- alert precision;
- mismatch rate among retained/non-alerted shots;
- coverage \(1-B/20000\).

If a partial area is reported, use trapezoidal area of mismatch recall versus alert fraction over these three fixed points. It is descriptive and not an alternative advantage gate.

### 8.5 Overall Run 6 decision

Eliminate the current `PNNL or untouched Google` choice. Recommend:

\[
\texttt{overall\_run6\_advantage}
=
\texttt{google\_primary\_pass}
\land
\texttt{pnnl\_retention\_pass}.
\]

`pnnl_retention_pass` must come from a separate value-blind, hashed PNNL cohort manifest and decision specification. Until that exists and passes, the overall Boolean is false even if the Google primary arm passes.

## 9. Uncertainty and randomization

### 9.1 Natural event

There is one author-identified event cluster. Report a binary detection result and the fixed sensitivity windows. Do **not** report a population miss probability from its adjacent cycles.

### 9.2 Exact pair-orientation audit

Use only threshold-block pairs 5,000–9,999, initialized from the common fit/warm-up checkpoint.

For replicate \(b=0,\ldots,255\):

- RNG is NumPy `Generator(PCG64(610700+b))`;
- call `rng.integers(0, 2, size=5000, dtype=np.uint8)` exactly once;
  bit 1 swaps A/B and bit 0 retains the stored orientation;
- use the same swap orientation for all 51 roles in that shot;
- apply the swap to the complete shot, score every role using only its
  pre-shot role state, mix the 51 role-component experts, and only then
  retain the per-role updates;
- reset all witnesses and accumulators to an identical checkpoint copy;
- use the complete-shot horizon \(H=5{,}000\);
- process the complete stream in fixed order.

For fault-tolerant execution, the 256 replicate indices are partitioned into
the 32 fixed half-open shards
\([8j,8(j+1))\), \(j=0,\ldots,31\). At most 16 one-process, one-numeric-thread
shards run concurrently. Every shard restores and records the same warm
checkpoint. The merger must reject gaps, overlaps, duplicate indices, changed
seeds or checkpoints and must sort by replicate index before writing the
canonical result. Completed shard manifests may be reused after interruption;
partial attempts are preserved but never merged. This is a computational
partition only and does not change any RNG call or statistical experimental
unit.

The primary audit statistic is

\[
\mathbf1\{\sup_tE^{(S)}_t\ge100\}.
\]

Report its count over 256 replicates with a two-sided 95% Clopper–Pearson
interval. This is an implementation diagnostic, not a power estimate.
Individual M0/M1/M3/M4/M5 crossing rates are descriptive marginal checks.
M0C and M2 have no exact-process crossing rate. If a family-wide alarm is
claimed, use Bonferroni threshold \(6/0.01=600\) for the six processes M0,
M1, M3, M4, M5 and \(S\).

### 9.3 Risk uncertainty

After the detector freeze and outcome join, use a paired circular moving-block bootstrap of complete monitored shots:

- block length 128 shots;
- 2,000 replicates;
- replicate \(b\) uses `Generator(PCG64(611000+b))`;
- draw `ceil(20000/128)=157` start indices with
  `rng.integers(0, 20000, size=157)`;
- expand each start to 128 consecutive indices modulo 20,000,
  concatenate in draw order, and retain the first 20,000 indices;
- resample score/outcome rows together;
- recompute top-\(B\) sets and metric differences;
- report percentile 95% intervals.

These intervals describe stability under shot-block resampling. They are not independent-event confidence intervals and do not change the primary Boolean.

### 9.4 Threshold uncertainty

The locked threshold is deterministic and is never replaced by a bootstrap estimate. A descriptive threshold/count bootstrap may use:

- circular complete-shot blocks of length 128;
- 2,000 replicates;
- seeds `613000+b`, \(b=0,\ldots,1999\).
- for each replicate, draw `ceil(5000/128)=40` starts uniformly from
  `0..4999`, expand modulo 5,000, concatenate, and retain the first
  5,000 indices.

## 10. Determinism and numerical rules

- All numerical arrays and persisted scores use IEEE float64.
- Set BLAS/OpenMP thread counts to one for the frozen run.
- Record Python, NumPy, SciPy, scikit-learn and BLAS versions.
- No implicit/global RNG is permitted.
- Score-bound check tolerance is \(10^{-12}\):
  values in \([-1-10^{-12},1+10^{-12}]\) may be clipped to \([-1,1]\);
  larger violations abort.
- Eigenvalue/degeneracy tolerance is the locked \(10^{-10}\).
- Exact score ties use the deterministic rules above; do not use approximate ties for ranking.
- For reporting floating metric equality, use
  `abs(a-b) <= 1e-12 + 1e-12*max(abs(a),abs(b))`.
- E-process/SR thresholds use `>=`; empirical raw-score thresholds use strict `>`.
- No pruning of experts/components is permitted.
- Formal expert IDs contain both `role` and the method component tuple.
  The role prior is exactly \(1/51\), so adding role experts does not multiply
  total prior mass. Role-specific adaptive-state IDs use the same role.

## 11. Resource and timing ledger

Every method is offered the same records:

| Phase | Paired shots | Physical archived shots | Paired role updates | Detector bits exposed |
|---|---:|---:|---:|---:|
| fit/warm-up | 5,000 | 10,000 | 255,000 | 12,240,000 |
| threshold | 5,000 | 10,000 | 255,000 | 12,240,000 |
| held | 20,000 | 40,000 | 1,020,000 | 48,960,000 |

One paired role update exposes \(2\times24=48\) detector bits. M2 may use only its locked 20,000-observation subsample for fitting, but its ledger must still distinguish records made available from records actually used.

For the canonical joint pipeline report:

- physical calibration, threshold and held shots;
- paired role updates and detector bits read;
- fixed parameters and mutable-state floats/bytes, separated by method-state
  prefix;
- feature evaluations, dot products and model updates;
- M2 covariance-fit/inversion count;
- M5 eigendecomposition count, with one decomposition shared by its two rank variants;
- output bytes and peak resident memory.

Timing protocol:

- pin the exact CPU model, OS, Python environment and one BLAS/OpenMP thread;
- use the threshold-block replay as one unreported joint-pipeline warm-up;
- time three full held detector-only joint-pipeline replays from the identical checkpoint;
- report the median and all three elapsed times;
- collect peak RSS from the same standalone process;
- use the first deterministic replay as the canonical score artifact;
- require all three numeric replay digests and final checkpoint hashes to
  agree;
- never include outcome loading/joining in detector latency.

The implementations share parsing and feature construction, so this protocol
does not authorize a relative per-method speed claim. Isolated optimized
implementations would require a separate frozen benchmark. The external
Kingston work is provenance for a predeclared grid, not additional fitted
state. It should be listed qualitatively and cannot be described as equal
record-level calibration.

## 12. Reset and cooldown table

| Context | Witness/model state | Accumulator | Notification cooldown |
|---|---|---|---|
| ordinary next shot | retain role-specific state | retain | none |
| threshold clone start | clone warm-up checkpoint | reset | none |
| held clone start | clone same warm-up checkpoint | reset | none |
| empirical alert | keep updating | unchanged/not applicable | suppress remainder of current shot |
| proper-prior primary alarm | stop formal first-alarm clock; scores may continue separately | no reset | not applicable |
| SR primary alarm | stop formal first-alarm clock | no reset | not applicable |
| optional repeated SR diagnostic | retain witness | reset all \(R_c=0\) at next shot boundary | one shot at most |
| randomization replicate | restore identical warm-up checkpoint | reset all | none |

No event-window boundary, decoder outcome or observed alarm may trigger a model reset.

## 13. Required output contract

### 13.1 Freeze manifest JSON

Require a strict, versioned manifest with:

- protocol/spec/config/code commit hashes;
- source archive size/checksums and extracted metadata hashes;
- parser permutation hash;
- warm-up checkpoint hash;
- threshold-table hash;
- dependency/platform/BLAS/thread settings;
- all RNG algorithms and seed ranges;
- start/end time and command line;
- outcome-access flag, initially false;
- deviation-ledger path and hash.

Unknown or missing fields must fail validation.

### 13.2 Cycle arrays

Avoid introducing an unfrozen Parquet engine. Persist one little-endian
float64/int64/bool `.npy` array per phase and method, all in fixed
shot-major/role-major order. Both the threshold and held phases contain:

- `empirical_cycle_score`;
- `above_threshold`;
- `notification_emitted`;
- `cooldown_active`;

The held phase additionally contains shot-indexed `log_eprocess`, `log_sr`,
`first_e_crossing`, and `first_sr_crossing`. These arrays have one row per
complete paired shot, because the 51 roles are role experts sharing one
randomization unit, not 51 formal time steps. Formal accumulators are not run
on the empirical threshold-selection clone, so threshold-phase formal arrays
are omitted rather than filled with misleading missing values. In the held
phase, use `NaN` only for M0C and M2, to which the formal statistic is
inapplicable; their first-crossing masks are all false.

A canonical JSON sidecar supplies `protocol_id`, `run_id`, phase, method ID,
shape, dtype, pair/archive-shot mapping, threshold and every
checkpoint/config/code hash. Hash the raw `.npy` bytes and the canonical JSON
(`sort_keys=True`, UTF-8, compact separators). Do not use pickle or object
arrays.

Component-level factors need not be repeated in this table; write a separate component summary/checkpoint table containing IDs, priors, final log wealth and bound checks.

### 13.3 Shot table

One row per `(phase, pair, method)`:

- archive shot IDs and pair index;
- maximum empirical shot score;
- earliest argmax role;
- earliest threshold-crossing role;
- shot alert;
- cumulative alert count;
- rank and rank tie key;
- event-window membership only as metadata, never as score input.

Write both `threshold_shots.csv` before held decoding/scoring and
`held_shots.csv` after the held replay. In `cooldown_active`, roles strictly
after the first notification in that same shot are true, marking notification
suppression while all model updates continue. Every next shot starts false;
there is no cross-shot cooldown.

### 13.4 Outcome table

Create only after detector freeze. Keep it physically separate and join by monitored archive shot:

- actual observable flip;
- correlated-matching prediction and mismatch;
- PyMatching prediction and mismatch;
- detector-result manifest hash authorizing the join.

### 13.5 Summary and decision JSON

Include:

- validation counts/frontiers and locked thresholds;
- primary and zero-validation-alert secondary event-window summaries;
- event-window alarms and pre-event counts;
- exact randomization audit;
- top-\(B\) risk metrics and uncertainty;
- M0–M5 and fixed-\(S\) resource ledger;
- every atomic decision predicate;
- `google_primary_pass`;
- `pnnl_retention_pass` or explicit `not_run`;
- `overall_run6_advantage`;
- a machine-readable negative-result reason list.

## 14. Minimum synthetic/no-held-values tests

The implementation gate should require at least:

1. **B8 parser:** synthetic little-endian records verify byte offsets and per-shot alignment.
2. **Coordinate remap:** a synthetic Stim declaration fixture verifies boundary-role reversal and identical canonical check sets.
3. **Index round trip:** all edge indices satisfy
   \(n=51q+r\) and its inverse.
4. **Feature order:** dimension is 300, pair order is lexicographic, all values lie in \([0,1]\), and \(R\succeq0\) with unit trace.
5. **Antisymmetry:** swapping A/B negates every M0, M1 and fixed-witness M3–M5 score.
6. **Bounds:** randomized synthetic pairs keep all exact scores in \([-1,1]\) and every factor nonnegative.
7. **Causality:** changing current \(D_t\) cannot change the witness used to score that same \(t\); it may change only \(t+1\) for the same role.
8. **Role isolation:** updating role 1 leaves roles 0 and 2–50 byte-identical.
9. **EWMA half-life:** an impulse decays by one half after exactly \(h\) subsequent updates of the same role; bias-correction state matches the formula.
10. **Top-\(k\) ties:** equal magnitudes select lower indices and zero contrast gets positive sign.
11. **Spectral zero case:** no positive eigenvalue yields zero effects/scores.
12. **Spectral degeneracy:** a repeated maximum eigenvalue yields the prescribed deterministic rank-one projector.
13. **Logistic gradient:** one hand-computed SGD step, no intercept, and score-before-update match exactly.
14. **M2 selection:** PCG64 seed produces exactly 20,000 unique role-stratified fit indices; threshold indices are absent.
15. **Mixture priors:** every method and composite prior sums to one within \(10^{-15}\).
16. **Neutral process:** \(L_t\equiv1\) gives \(E_t\equiv1\) and \(R_t=t\).
17. **Threshold ties:** strict `>` and the smallest qualifying threshold produce no more than the permitted alert count.
18. **Cooldown:** at most one notification occurs per shot while all 51 updates still execute.
19. **Checkpoint cloning:** threshold and held clones initially hash identically and diverge only after their first input.
20. **Random swap:** one seeded orientation applies to all 51 roles, and every replicate fully resets state.
21. **Deterministic replay:** two synthetic runs have byte-identical score/summary artifacts.
22. **Outcome embargo:** the detector command rejects outcome paths; the outcome command rejects an absent/mismatched detector-freeze hash.
23. **Schema rejection:** unknown/missing config or output fields fail rather than receiving defaults.
24. **Resource ledger:** paired physical shots, cycles, detector bits, model updates and decompositions agree with synthetic expected counts.

## 15. Scope and freeze gate

Before any held-value run, serialize this specification into the strict
versioned config, implement only against synthetic fixtures, and freeze:

- plan/spec/config and inherited-config hashes;
- parser and warm-up-checkpoint code;
- runner and output schemas;
- environment/threads;
- seed registry;
- deviation ledger;
- git commit and remote commit identity.

The locked claim should be narrowed to:

> At a predeclared empirical false-alert operating point, does the fixed M4/M5 S-PACE composite detect the author-identified Google event and capture more correlated-decoder mismatches at a fixed 20-shot triage budget than paired DFR and the same-feature online logistic comparator?

Global/per-check DFR CUSUM and a separate sparse scan are not silently
represented by M0–M5. Either give them their own value-blind specification
before the run or remove them from any list of baselines that the primary
claim purports to beat.

No result from this arm alone permits exact natural-drift delay, exact hardware e-validity, superiority to every model-aware detector, or an overall Run 6 algorithmic-advantage claim.
