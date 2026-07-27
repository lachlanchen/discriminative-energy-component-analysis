# Independent mathematical audit of the Run 6 S-PACE theory

**Audit date:** 2026-07-27
**Audited file:** `references/run6_space_final_theory.md`
**Scope:** theorem validity, predictability, normalization and exact-versus-empirical claims. No performance or novelty assessment is made here.

## Verdict

**CONDITIONAL PASS.** The central construction is mathematically sound:

\[
\mathbb E_0[D_t\mid\mathcal F_{t-1}]=0,\quad
s_t\in[-1,1],\quad s_t\ \text{formed predictably}
\quad\Longrightarrow\quad
1+\beta s_t
\]

is a conditional e-factor for \(|\beta|\le 1\). The paired-exchangeability theorem, orbit-centering calculation, sector-projection identity, capped-simplex support function, mirror-ascent regret constant, proper-prior e-process, Shiryaev--Roberts ARL argument and KL ceiling all pass under their intended hypotheses.

The file should nevertheless receive **five validity/implementation repairs** before it is treated as a locked specification:

1. Equation (7) must explicitly assume a bounded score, or another lower bound ensuring nonnegativity.
2. A global/trivial-sector score cannot be recovered by projecting the orbit-centered contrast in (5); that projection is identically zero. It needs a separate valid reference interface.
3. Sector outputs must be converted to predictable **real-valued** scores and normalized by an explicit predictable range bound.
4. The claim that a uniform instantaneous location mixture is identically one is true only for common witnesses/bet sizes and zero-sum location scores, not for arbitrary location-specific components.
5. The mirror-ascent theorem must specify the KL/Bregman projection, define \(W_T\), and either declare the horizon \(T\) in advance or use an anytime learning-rate schedule with a correspondingly revised bound.

Several narrower wording repairs are also recommended: scope the common-mode no-go to a pure additive, unclipped observation model; describe \(\mathcal C_k\) as a capped/diversified convex domain rather than a generally sparse one; replace the SR clock's alarm time by \(\lceil\gamma\rceil\); and state the conditional KL version when applying the ceiling sequentially.

## Summary table

| Item | Verdict | Main condition or repair |
|---|---|---|
| Conditional paired exchangeability, (2)–(4) | **PASS** | Antisymmetric score must be a jointly measurable \(\mathcal F_{t-1}\)-measurable random function |
| Conditional orbit centering, (5) | **PASS** | Conditional equality of location means is sufficient; unconditional invariance is not |
| Empirical-envelope e-factor, (6)–(7) | **CONDITIONAL PASS** | Add \(s_t\in[-1,1]\), predictable \(\varepsilon_t\), and note expectation is \(\le1\) |
| Sector projection, Proposition 1 | **PASS WITH REPAIR** | Use a predictable real functional and explicit predictable range normalization |
| Common-mode no-go, Proposition 2 | **PASS WITH SCOPE REPAIR** | Exact for \(\Phi'_t=\Phi_t+\mathbf1\otimes c_t\) before clipping/nonlinear remapping |
| Spectral/Jordan effect, (12)–(15) | **PASS** | The past average and accessible-algebra map must be predictable/fixed |
| Capped-simplex top-\(k\), Proposition 3 | **PASS** | Domain permits dense portfolios; a linear optimum has a \(k\)-support representative |
| Entropic mirror-ascent regret, Proposition 4 | **PASS WITH SPECIFICATION REPAIR** | KL projection, uniform initialization, fixed \(T\), and \(W_T=\prod_t(1+\beta q_t^\top u_t)\) |
| Fixed component wealth mixture | **PASS** | Each component must already be an e-process; fixed weights cannot repair empirical invalidity |
| Proper start-prior e-process, (23)–(24) | **PASS** | Nonnegative conditional e-factors and fixed prior/component weights |
| SR recursion and ARL, (25)–(26) | **PASS** | Use stopped-process argument; \(L_t\equiv1\) alarms at \(\lceil\gamma\rceil\) |
| Likelihood KL ceiling, (27) | **PASS** | Apply on the same one-step sigma-field; use conditional KL for adaptive sequences |
| Exact-versus-empirical hardware language | **PASS** | The manuscript consistently denies exact hardware validity absent an exact contrast interface |

## 1. Filtration and paired exchangeability

### 1.1 Paired theorem: pass

Condition (2) implies the vector result directly. For any coordinate \(a\),

\[
\begin{aligned}
\mathbb E_0[
\phi_a(U_t)-\phi_a(V_t)
\mid\mathcal F_{t-1}]
&=
\mathbb E_0[
\phi_a(V_t)-\phi_a(U_t)
\mid\mathcal F_{t-1}]\\
&=
-\mathbb E_0[
\phi_a(U_t)-\phi_a(V_t)
\mid\mathcal F_{t-1}],
\end{aligned}
\]

so the conditional expectation is zero. No within-pair independence is needed.

Likewise, for each realized past, conditional swap invariance and antisymmetry give

\[
\int g_{t-1}(u,v)\,dP_t(u,v)
=
\int g_{t-1}(v,u)\,dP_t(u,v)
=
-\int g_{t-1}(u,v)\,dP_t(u,v).
\]

Boundedness supplies integrability. Equation (4) is therefore correct.

### 1.2 Predictability warning

The theorem requires more than declaring two records a pair. The following must be fixed from \(\mathcal F_{t-1}\), or randomized independently of the current values under the stated null:

- which physical records enter the pair;
- which member is called \(U_t\) and which is called \(V_t\);
- all preprocessing and missing-data rules;
- the antisymmetric function and its tuning parameters.

A rule such as “call the larger-current-error patch \(U_t\)” breaks the ordered-pair exchangeability used in the proof. The current filtration paragraph already points in the right direction; the implementation should test this at the pairing API boundary.

## 2. Orbit centering, sectors and the common-mode no-go

### 2.1 Conditional orbit centering: pass

Let

\[
m_t(x)=
\mathbb E_0[\Phi_t(x)\mid\mathcal F_{t-1}].
\]

Conditional invariance and transitivity imply \(m_t(x)=m_t(u)\) for all \(x,u\) in the orbit. Hence

\[
\mathbb E_0[D_t(x)\mid\mathcal F_{t-1}]
=
m_t(x)-|\mathcal X|^{-1}\sum_um_t(u)
=0.
\]

The manuscript correctly states that equality of conditional means is the minimal property used by this calculation and that unconditional symmetry is insufficient.

### 2.2 Mandatory repair: the global score is not inside the orbit-centered contrast

For (5),

\[
\sum_{x\in\mathcal X}D_t(x)=0
\quad\text{pathwise},
\qquad
P_{\mathrm{triv}}D_t=0.
\]

Therefore projecting the contrast from (5) cannot create a global branch. Conditional orbit symmetry centers only the **relative** field; it says nothing about the null mean of

\[
\overline\Phi_t
=
\frac1{|\mathcal X|}\sum_x\Phi_t(x).
\]

Insert the following clarification after Proposition 2:

> When the contrast interface is orbit centering (5), \(P_{\mathrm{triv}}D_t=0\) identically. The global branch must be constructed separately, for example as \(\overline\Phi_t-\mu_t^{\mathrm{global}}\) with a known conditional reference, as a paired difference of simultaneous global averages, or as an explicitly empirical monitor. Conditional orbit invariance alone does not make \(\overline\Phi_t\) mean zero.

This is the most important structural repair. Without it, an implementation could mistakenly label the zero trivial projection—or an uncentered global mean—as an exact e-factor.

### 2.3 Sector projection: algebra passes; score construction needs specification

For an \(\mathcal F_{t-1}\)-measurable bounded linear map \(P_{t-1}\),

\[
\mathbb E_0[P_{t-1}D_t\mid\mathcal F_{t-1}]
=
P_{t-1}
\mathbb E_0[D_t\mid\mathcal F_{t-1}]
=0.
\]

Thus Proposition 1 is correct. Orthogonal projection, however, does not by itself guarantee the coordinatewise \([-1,1]\) range needed later. A complex unitary representation can also produce complex sector coordinates, whereas an e-factor must be real and nonnegative.

Use an explicit construction. Let \(\ell_{t-1}\) be a predictable **real** linear functional (or the real part of a declared complex functional), and choose a predictable bound

\[
B_{t-1}
\ge
\sup_{d\in[-1,1]^p}
\left|
\ell_{t-1}(P_{\lambda,t-1}d)
\right|.
\]

Define

\[
s_{t,\lambda}
=
\begin{cases}
\ell_{t-1}(P_{\lambda,t-1}D_t)/B_{t-1},
&B_{t-1}>0,\\
0,&B_{t-1}=0.
\end{cases}
\]

Then \(s_{t,\lambda}\in[-1,1]\) and is conditionally centered. For complex irreducible sectors, either combine conjugate sectors into a real invariant block or declare the real functional explicitly.

Also replace “fixed mixtures across sectors remain valid” by:

> Fixed mixtures remain exact only when every mixed branch has already been converted into a valid conditional e-factor/e-process. Mixing an empirically calibrated branch with an exact branch does not make the empirical branch exact.

### 2.4 Common-mode no-go: pass under its exact additive model

Write a common-mode translation as

\[
\Phi'_t=\Phi_t+\mathbf1_{\mathcal X}\otimes c_t.
\]

Because \(\mathbf1_{\mathcal X}\otimes c_t\) lies in the invariant subspace,

\[
P_{\mathrm{rel}}\Phi'_t
=P_{\mathrm{rel}}\Phi_t
\underbrace{P_{\mathrm{rel}}
(\mathbf1_{\mathcal X}\otimes c_t)}_{0}
=P_{\mathrm{rel}}\Phi_t.
\]

The no-go is pathwise and therefore stronger than a low-power statement. The sentence claiming identical pre/post law should nevertheless be scoped to the model actually proved:

> If the post-change observed field is exactly \(\Phi'_t=\Phi_t+\mathbf1\otimes c_t\), with no clipping, saturation, nonlinear feature remapping or simultaneous change in the relative field, every statistic measurable only with respect to \(P_{\mathrm{rel}}\Phi_t\) has the same law before and after the change.

A physical event described colloquially as “chip-wide” may also alter variances, correlations or relative responses; Proposition 2 does not rule out detecting those additional effects.

## 3. Empirical-envelope correction

Equation (7) has the correct expectation normalization but omits the assumption that makes its numerator nonnegative.

The complete statement should be:

> Suppose \(s_t\in[-1,1]\), \(\varepsilon_t\ge0\) is \(\mathcal F_{t-1}\)-measurable, and
> \(\left|\mathbb E_0[s_t\mid\mathcal F_{t-1}]\right|\le\varepsilon_t\) almost surely uniformly over the claimed null. For \(|\beta|<1\),
> \[
> \widetilde L_t=
> \frac{1+\beta s_t}{1+|\beta|\varepsilon_t}
> \]
> is nonnegative and satisfies
> \[
> \mathbb E_0[\widetilde L_t\mid\mathcal F_{t-1}]
> \le1.
> \]

Indeed,

\[
1+\beta\mathbb E_0[s_t\mid\mathcal F_{t-1}]
\le1+|\beta|\varepsilon_t.
\]

The corrected factor is generally a conditional e-factor with expectation at most one, not exactly one. This is sufficient for all later supermartingale and ARL arguments.

Without \(s_t\in[-1,1]\), a mean envelope alone does not prevent \(1+\beta s_t<0\). A more general theorem may replace the boundedness assumption by a predictable lower bound chosen to guarantee numerator nonnegativity.

## 4. Spectral witness

This section passes.

For density operators \(R(U),R(V)\) and an effect \(0\preceq E\preceq I\),

\[
0\le\operatorname{Tr}(ER(U)),\operatorname{Tr}(ER(V))\le1,
\]

so their difference lies in \([-1,1]\). Conditional exchangeability centers it. Since \(\overline\Delta_{t-1}\) uses only past observations, its positive-support projector is predictable.

For any Hermitian \(A\),

\[
\max_{0\preceq E\preceq I}\operatorname{Tr}(EA)
=\operatorname{Tr}(A_+),
\]

attained by the positive spectral projector, with arbitrary choice on the zero eigenspace. The accessible-algebra statement is also correct when the conditional expectation onto \(\mathcal A\) is the trace-preserving, trace-inner-product projection: effects in \(\mathcal A\) pair with \(\Delta\) exactly as they pair with its conditional expectation.

The manuscript correctly limits this result to a one-step linear-gap objective.

## 5. Capped-simplex top-\(k\) optimizer

Proposition 3 passes exactly.

The extreme points of

\[
\left\{q\ge0:\sum_iq_i=1,\ q_i\le1/k\right\}
\]

place mass \(1/k\) on \(k\) coordinates. Therefore a linear objective selects the \(k\) largest entries of \(u\). For every original coordinate \(i\), the pair \((\delta_i,-\delta_i)\) contains \(+|\delta_i|\) and \(-|\delta_i|\). Since \(k\le p\), the \(k\) largest entries of \(u\) can be chosen as the signed copies of the \(k\) largest \(|\delta_i|\), proving (17). Ties and zero contrasts can make the optimizer nonunique but do not change the value.

The interpretation needs one correction:

- \(\mathcal C_k\) enforces **at least** \(k\) nonzero coordinates, not at most \(k\);
- a linear objective has a \(k\)-support extreme-point optimizer;
- the online entropic portfolio will generally be dense.

Use “capped/diversified portfolio whose linear extreme-point optimum selects \(k\) signed features,” rather than implying that every \(q\in\mathcal C_k\) is \(k\)-sparse. “Larger \(k\) trades sparsity for stability” remains reasonable when referring to the extreme optimizer.

## 6. Entropic mirror-ascent regret

### 6.1 Constants and domain: pass

For

\[
f_t(q)=\log(1+\beta q^\top u_t),
\]

\[
\nabla f_t(q)
=
\frac{\beta u_t}{1+\beta q^\top u_t}.
\]

Since \(q\) is a probability vector, \(u_t\in[-1,1]^{2p}\), and \(0<\beta<1\),

\[
q^\top u_t\ge-1,\qquad
\|\nabla f_t(q)\|_\infty
\le\frac{\beta}{1-\beta}=G_\beta.
\]

With \(q_1=(1/(2p),\ldots,1/(2p))\),

\[
\mathrm{KL}(q^\star\|q_1)
=\log(2p)-H(q^\star).
\]

The cap \(q_i^\star\le1/k\) implies \(H(q^\star)\ge\log k\), so

\[
\mathrm{KL}(q^\star\|q_1)\le\log(2p/k).
\]

Negative entropy is 1-strongly convex with respect to \(\ell_1\), whose dual norm is \(\ell_\infty\). Concavity of \(f_t\) and the standard mirror-ascent inequality yield

\[
\sum_{t=1}^T[f_t(q^\star)-f_t(q_t)]
\le
\frac{\log(2p/k)}{\eta}
+\frac{\eta G_\beta^2T}{2}.
\]

Optimizing the right side gives exactly

\[
G_\beta\sqrt{2T\log(2p/k)}.
\]

There is no missing factor of two in (21) or (22).

### 6.2 Mandatory algorithm specification

“Standard entropic mirror ascent with projection” should be made unambiguous:

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

Equivalently, exponentiate by \(\eta g_t\) and take the **KL/Bregman** projection onto \(\mathcal C_k\). An ordinary Euclidean projection is a different algorithm and does not inherit the displayed negative-entropy constant automatically.

Define the wealth used in (22):

\[
W_0=1,\qquad
W_T=
\prod_{t=1}^T
\left(1+\beta q_t^\top u_t\right),
\]

so \(\log W_T=\sum_tf_t(q_t)\).

Finally, the displayed optimizer for \(\eta\) depends on \(T\). State either:

- “For a horizon \(T\) declared before the run, choose the displayed constant \(\eta\),” or
- use a doubling trick/time-varying rate and state the resulting anytime regret constant separately.

This does not affect e-validity: any predictable update preserves validity. It affects only the claimed pathwise regret constant.

## 7. Component mixtures and the instantaneous-location statement

The fixed mixture

\[
W_t^{\mathrm{mix}}=\sum_j\pi_jW_{t,j}
\]

passes because conditional expectation is linear; no independence among components is needed. The weights must be fixed (or managed by a separately valid predictable switching construction), nonnegative and sum to one. Every \(W_{t,j}\) must already be valid under the same claimed null.

The statement “an instantaneous uniform location mixture after orbit centering is identically one” is too broad. If

\[
L_{t,x}=1+\beta w^\top D_t(x)
\]

uses the same \(\beta\) and same predictable linear witness \(w\) at all locations, then

\[
\frac1{|\mathcal X|}\sum_xL_{t,x}
=
1+\beta w^\top
\left\{\frac1{|\mathcal X|}\sum_xD_t(x)\right\}
=1.
\]

But cancellation need not hold for location-specific \(w_x\), \(\beta_x\), range normalizations or nonlinear scores.

Replace the bullet by:

> For raw scalar orbit contrasts, or common predictable linear witnesses with a common bet size and normalization, the instantaneous uniform location mixture cancels identically to one. This cancellation is not automatic for location-specific witnesses or normalizations.

## 8. Proper-prior e-process

Equations (23)–(24) pass.

If

\[
\mathbb E_0[L_{t,j}\mid\mathcal F_{t-1}]\le1,
\]

then

\[
\mathbb E_0[A_{t,j}\mid\mathcal F_{t-1}]
\le A_{t-1,j}+\rho_t.
\]

Using \(\sum_j\pi_j=1\) and
\(\rho_{>t-1}=\rho_t+\rho_{>t}\),

\[
\mathbb E_0[E_t\mid\mathcal F_{t-1}]
\le
\sum_j\pi_jA_{t-1,j}+\rho_t+\rho_{>t}
=E_{t-1}.
\]

Moreover \(E_0=\rho_{>0}=1\). Ville's inequality therefore gives the stated probability-of-ever-alarm guarantee. The example
\(\rho_\nu=1/\{\nu(\nu+1)\}\) is proper because it telescopes, and its \(-\log\rho_\nu\) late-start cost is \(2\log\nu+O(1)\), logarithmic in order.

No normalization error was found here.

## 9. Shiryaev--Roberts recursion and ARL

Equations (25)–(26) pass.

For each component,

\[
\mathbb E_0[R_{t,j}\mid\mathcal F_{t-1}]
\le R_{t-1,j}+1.
\]

After mixing,

\[
\mathbb E_0[R_t-t\mid\mathcal F_{t-1}]
\le R_{t-1}-(t-1),
\]

so \(R_t-t\) is a supermartingale. Apply optional sampling to
\(\tau_\gamma\wedge n\). If \(\tau_\gamma<\infty\) almost surely, localization followed by the usual integrability/limit argument gives

\[
\mathbb E_0[\tau_\gamma]
\ge
\mathbb E_0[R_{\tau_\gamma}]
\ge\gamma.
\]

If \(\mathbb P_0(\tau_\gamma=\infty)>0\), the extended expectation of \(\tau_\gamma\) is infinite and the bound is trivial. The manuscript's “standard optional-sampling conditions” caveat is adequate, though this stopped-process proof would make the statement more self-contained.

One exact wording repair is needed. In discrete time, if \(L_t\equiv1\), then \(R_t=t\) and

\[
\tau_\gamma=\lceil\gamma\rceil,
\]

not literally \(\gamma\) unless \(\gamma\) is an integer.

The manuscript correctly distinguishes this ARL guarantee from lifetime false-alarm probability.

## 10. Likelihood KL ceiling and log-growth expansion

### 10.1 KL ceiling: pass

Let \(r=dP_1/dP_0\), let \(L\ge0\), and assume \(\mathbb E_0L\le1\). If \(L=0\) on a set of positive \(P_1\)-mass, the left side is \(-\infty\). Otherwise Jensen's inequality gives

\[
\begin{aligned}
\mathbb E_1\log L-\mathrm{KL}(P_1\|P_0)
&=
\mathbb E_1\log\frac{L}{r}\\
&\le
\log\mathbb E_1\frac{L}{r}\\
&=
\log\mathbb E_0[L\mathbf1_{\{r>0\}}]\\
&\le0.
\end{aligned}
\]

Taking \(L=r\) attains equality. Equation (27) and the “cannot beat the correct same-information likelihood ratio in expected log growth” boundary are correct.

For an adaptive sequential application, state the conditional form. Almost surely for each past,

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

provided \(L_t\) is a conditional e-factor for \(P_{0,t}\) on the same current-observation sigma-field. Summing then gives a chain-rule ceiling. This prevents an accidental comparison between methods seeing different information.

### 10.2 Stationarity wording

The Taylor expansion is correct for \(|s_{t,j}|\le1\):

\[
\mathbb E_1\log(1+\beta s_{t,j})
=
\beta\mathbb E_1s_{t,j}
-\frac{\beta^2}{2}\mathbb E_1s_{t,j}^2
+O(\beta^3).
\]

If the witness or portfolio continues adapting, however, the law of \(s_{t,j}\) need not be stationary even when the raw post-change observations are stationary. Use \(I_{t,j}\), \(\delta_{t,j}\), and \(v_{t,j}\), or explicitly assume a fixed/stabilized component with a time-homogeneous score law before suppressing \(t\).

The manuscript correctly labels threshold-over-\(I_j\) delay formulas as renewal heuristics rather than finite-sample delay bounds.

## 11. Exact-versus-empirical language

This part passes and is unusually careful. In particular, the theory correctly states that:

- approximate geometry does not establish conditional orbit invariance;
- a finite plug-in baseline is not a known conditional mean;
- public hardware replay normally needs empirical threshold calibration;
- a constructed cohort boundary is not a naturally observed drift onset;
- a generic e-wealth allocation is not a posterior;
- fixed syndrome data are not a faithful classical-shadow eSCD experiment;
- no likelihood-ratio, Helstrom/Wilson-oracle, quantum-speedup or universal sample-efficiency advantage follows.

To keep this separation watertight in implementation and publication:

1. Label every result as **exact-model**, **natural approximate event**, **known intervention**, or **constructed boundary**.
2. Attach “exact” only when the conditional centering assumption is justified after conditioning on all training, selection and control history.
3. Treat a finite calibration replay without a proved simultaneous envelope as empirical even if its observed mean is numerically close to zero.
4. Do not state an exact mixture guarantee when any included global/sector branch is only empirically calibrated.

## Final disposition

No fatal theorem error or regret-constant error was found. After the five mandatory repairs, the theory supports the following narrow claim:

> Under a declared exact interface yielding a bounded conditionally mean-zero contrast, a predictable real bounded witness produces valid e-factors. Fixed component/start mixtures yield lifetime alarm control, and the SR recursion yields an ARL lower bound. Orbit centering detects only relative structure; global monitoring requires a separate referenced contrast. The capped-simplex and Jordan constructions optimize declared linear objectives, while entropic mirror ascent competes pathwise with the best fixed capped portfolio.

It does **not** turn empirical hardware centering into an exact e-process, make every capped portfolio sparse, provide a start-specific or quickest-delay theorem, or permit a same-information likelihood-ratio advantage claim.
