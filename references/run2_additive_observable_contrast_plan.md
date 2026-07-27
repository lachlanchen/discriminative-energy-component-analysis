# Research plan: run 2 additive observable contrast

Date: 2026-07-27
Status: executed and verified
Target manuscript: `publication/run2/main.tex`
Target code: `experiments/aoc/` and `experiments/run2/`

## Scientific goal

Replace the narrow view of ECA as a classifier with an operator-theoretic
problem:

> Given two streams of physical or statistical states, find the bounded
> observable whose expectation changes the most, update it from one arriving
> sample at a time, and report both the magnitude and the physical mode of the
> change.

Classification is a corollary. The primary tasks are state discrimination,
change detection, mode localization, and online experimental diagnosis.

## Mathematical core

For a positive trace-one encoding \(R(x)\), maintain

\[
S_c=\sum_i w_iR(x_i),\qquad m_c=\sum_iw_i,\qquad
\rho_c=S_c/m_c.
\]

For \(\Delta=\rho_1-\rho_0\), prove and implement

\[
\sup_{0\preceq E\preceq I}\operatorname{Tr}(E\Delta)
=\operatorname{Tr}(\Delta_+)
=\tfrac12\|\Delta\|_1,
\quad
E^\star=\mathbf 1_{\Delta>0},
\]

where the last equality uses \(\operatorname{Tr}\Delta=0\).
This is the exact formalization of “maximum class difference rather than
maximum variance.”

The run must also establish:

1. the general Jordan-decomposition result for unequal trace/priors;
2. the dual integral-probability-metric interpretation over bounded quadratic
   observables;
3. the projective-kernel mean-embedding relation
   \(k(x,z)=|\langle\phi(x),\phi(z)\rangle|^2\);
4. rank-constrained Ky Fan solutions;
5. exact batch/online/merge equivalence;
6. sliding-window and exponentially forgotten states;
7. perturbation and finite-sample witness stability under an eigengap;
8. a predictable-witness sequential score, with false-alarm claims limited to
   the assumptions actually verified.

Known Helstrom, trace-distance, kernel-mean, CSP, and quantum change-point
results must be credited. The manuscript must not claim invention of those
results.

## Algorithms

- `AdditiveState`: add, remove, decay, snapshot, and associative merge.
- `maximum_observable_contrast`: full and rank-constrained analytic witness.
- `ObservableContrastDetector`: fixed-reference streaming detector with
  interpretable witness history.
- Incremental binary DECA under empirical priors using the unnormalized signed
  sum \(G_t=\sum_i y_iR(x_i)\).
- Qiskit compilation of the current witness measurement for finite shots.

## Required experiments

1. Algebraic audit: online state equals batch state to floating-point error
   under random arrival and merge orders.
2. Null calibration: stationary streams, many seeds, empirical false-alarm
   behavior reported without extrapolation.
3. Detection-delay curves against change magnitude and window length.
4. Two-dimensional Ising phase/order-mode discovery with random global spin
   flips.
5. Ambiently excited mass-spring structural damage and mode localization.
6. Polarization/coherence drift with an actual Qiskit Aer measurement circuit.
7. Matched phase-blind benchmark where mean-only methods are provably blind.
8. Strong baselines: mean CUSUM, Frobenius/MMD contrast, covariance GLR or QDA
   when its assumptions apply, PCA, and fixed physical observables.

The desired “overwhelming” result is acceptable only on predeclared matched
regimes. Universal superiority is explicitly out of scope and contradicted by
no-free-lunch considerations.

## Manuscript scope

The run 2 manuscript will use a physics-facing journal layout and contain:

- operator-algebra formulation and proofs;
- online algorithms and complexity;
- quantum implementation and resource assumptions;
- physical experiments with effect sizes and uncertainty;
- a related-work table separating known theory from new synthesis;
- limitations, including information lost by second-order encodings.

## Verification

- Unit/property tests for every algebraic identity and state operation.
- Deterministic experiment manifests and hashes.
- Statistical assertions recomputed from raw run 2 CSV files.
- Clean LaTeX build, cross-reference audit, bibliography audit, and PDF visual
  inspection.
- No claim of dominance unless the stored results and protocol prove it.

## Execution notes

- Implemented additive states, rank-constrained contrast, projective MMD
  identity, predictable betting e-process, multiclass POVM SDP, physical
  generators, and Qiskit measurement under `experiments/aoc/`.
- Added 13 run 2 tests and four fixed-seed experiment entry points.
- Stored algebra, Ising, structural, and optical evidence under
  `experiments/run2/results/`.
- The initial exponential/Hoeffding-style detector was replaced by a bounded
  betting martingale; exact anytime validity is restricted to a known,
  independent reference state.
- The manuscript builds to `publication/run2/main.pdf`; visual and unresolved
  reference audits pass.
- Git commit and push are recorded in repository history.
