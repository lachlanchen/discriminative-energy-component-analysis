# Re-audit addendum: patched S-PACE theory

**Date:** 2026-07-27
**Re-audited file:** `references/run6_space_final_theory.md`
**Baseline audit:** `references/run6_space_math_audit.md`

## Verdict

**PASS WITH MINOR CONSISTENCY EDITS.** All five mandatory repairs from the first audit have been implemented in substance. The paired-exchangeability, orbit-centering, common-mode no-go, capped-simplex optimizer, entropic mirror-ascent regret, proper-prior e-process, Shiryaev--Roberts ARL and likelihood-KL claims remain correct. No predictability, factor-normalization or regret-constant blocker remains.

Three non-blocking notation/wording issues remain:

1. the sector space and normalization domain need one consistent dimension and projector notation;
2. the empirical-reference equation uses an undefined/static \(\mu\) where the preceding notation requires \(\mu_t\);
3. two residual uses of “sparse” should be qualified because \(\mathcal C_k\) itself permits dense portfolios.

## Repair-by-repair disposition

| Prior audit item | Status in patched theory |
|---|---|
| Equation (7) needs score boundedness/nonnegativity | **RESOLVED.** It now assumes \(s_t\in[-1,1]\), predictable \(\varepsilon_t\), proves nonnegativity and states conditional expectation \(\le1\). |
| Orbit-centered trivial sector cannot supply global score | **RESOLVED.** It now states \(P_{\mathrm{triv}}D_t=0\) pathwise and constructs a separately referenced global branch. |
| Sector score must be predictable, real and range normalized | **SUBSTANTIVELY RESOLVED.** A real predictable functional and predictable \(B_{t-1}\) are supplied; only the ambient-dimension notation remains inconsistent. |
| Uniform instantaneous location-mixture claim was too broad | **RESOLVED.** Cancellation is now limited to common witnesses, bet size and normalization. |
| Mirror ascent needs KL projection, \(W_T\), and horizon declaration | **RESOLVED.** The KL/Bregman update, product wealth and fixed-horizon qualification are explicit; unknown horizons are separated. |
| Common-mode no-go needed an additive/unclipped scope | **RESOLVED.** The exact additive model and excluded clipping/nonlinear/relative changes are stated. |
| \(\mathcal C_k\) should not be called generally sparse | **MOSTLY RESOLVED.** The top-\(k\) section correctly says the domain may be dense and only a linear extreme optimizer is \(k\)-supported. |
| SR clock should alarm at \(\lceil\gamma\rceil\) | **RESOLVED.** |
| Adaptive likelihood ceiling needs conditional KL/same information | **RESOLVED.** |
| Adaptive scores need time-indexed growth unless stabilized | **RESOLVED.** The manuscript now distinguishes \(I_j\) from \(I_{t,j}\). |
| Exact and empirical branches must not be conflated by mixing | **RESOLVED.** The text explicitly keeps a mixed empirical replay empirical. |

## Exact remaining edits

### 1. Unify the sector ambient space and projector notation

Section 3 defines a per-location feature in \([0,1]^p\), while the sector section acts on a complete location field and uses
\(\mathbf1_{\mathcal X}\otimes c_t\). Its ambient dimension is therefore generally \(p|\mathcal X|\), not \(p\). Yet the range certificate currently uses

\[
\sup_{d\in[-1,1]^p}
\left|\ell_{t-1}(P_{\lambda,t-1}d)\right|.
\]

It also switches from \(P_\lambda\) to an otherwise undefined \(P_{\lambda,t-1}\).

Use one of these exact forms:

> Let \(\mathcal D\subset V\) be a declared set containing the support of \(D_t\). If the projectors are predictable, denote them consistently by \(P_{\lambda,t-1}\). Choose
> \[
> B_{t-1}\ge
> \sup_{d\in\mathcal D}
> |\ell_{t-1}(P_{\lambda,t-1}d)|.
> \]
> For a stacked \(p\)-channel field on \(\mathcal X\), one conservative choice is
> \(\mathcal D=[-1,1]^{p|\mathcal X|}\).

Alternatively, if the representation/projectors are fixed, retain \(P_\lambda\) everywhere. Also state that the replicated common mode
\(\mathbf1_{\mathcal X}\otimes c_t\) belongs to \(V_{\mathrm{triv}}\), or define \(U_g\) as permuting locations while leaving channels unchanged. This removes the only remaining normalization/domain ambiguity.

### 2. Restore the conditional-mean index in the empirical interface

The known-reference subsection defines

\[
\mu_t=\mathbb E_0[\phi(Y_t)\mid\mathcal F_{t-1}],
\]

but the empirical subsection writes

\[
\mathbb E_0[\phi(Y_t)-\widehat\mu\mid\mathcal F_{t-1}]
=\mu-\widehat\mu.
\]

Unless a stationary constant mean \(\mu\) is separately defined, the right side should be

\[
\mu_t-\widehat\mu.
\]

This is a notation repair, not a change to the conclusion that plug-in centering is generally nonzero.

### 3. Qualify the remaining sparse terminology

The detailed theorem now correctly distinguishes a dense capped domain from its \(k\)-support linear extreme optimizer. Make the introduction and component index match it:

- replace “an analytically optimal sparse feature set” with
  “an analytically optimal top-\(k\) support for the declared capped linear-gap objective”;
- replace “\(k\) a sparsity level” with
  “\(k\) a cap/diversification level whose linear extreme optimizer has \(k\)-support.”

The later phrase “spectral or sparse witnesses” can remain if it refers specifically to the predictable top-\(k\) extreme witness rather than every online \(q_t\in\mathcal C_k\).

## Lightweight consistency scan

- Equation tags are unique and complete from (1) through (27).
- Display-math delimiters are balanced: 55 openings and 55 closings.
- `\left`/`\right` delimiters are balanced.
- The e-factor distinction is consistent: exact centering gives conditional expectation one; the envelope correction gives at most one.
- The proper-prior recursion retains \(E_0=1\), and the SR mixture retains the unit drift needed for the ARL proof.
- The OMD update occurs only after scoring \(u_t\), so \(q_t\) remains predictable.
- Exact-model, natural-event, intervention, constructed-boundary and empirical-hardware language remains separated.

## Final disposition

The patched theory is mathematically ready for implementation after the three small consistency edits above. None changes a theorem, bound or experimental claim. The strongest remaining implementation guard is still:

> An orbit-centered relative contrast and a referenced global contrast are separate statistical objects; neither a projection nor a fixed mixture can manufacture exact validity for an empirically centered branch.
