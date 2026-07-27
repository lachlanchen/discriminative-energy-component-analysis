# Run 5 locked results and advantage audit

- **Audit date:** 2026-07-27
- **Protocol:** [Run 5 preregistered plan](run5_surface_code_drift_preregistered_plan.md)
- **Prior-art and claim boundary:** [Run 5 prior-art audit](run5_prior_art_and_claim_boundary.md)
- **Protocol/source freeze:** `9d59cba22f944b6574b429bcae39060a39c43c73`
- **Offline-only provenance-label repair:** `3ebf211821599d6bdd73687cce5dc22273597ca3`
- **Sequential metrics-completeness repair:** `1b55355370880872bcb853490cff4febfa2c1ea4`

## 1. Executive answer

The locked Run 5 evidence does **not** show an algorithmic detection
advantage for raw ECA/AOC or for variance-aware AOC (vAOC).

The preregistered primary comparison was vAOC versus a fixed,
validation-trained linear logistic effect on the **same** Fourier
simplex/pair-simplex, under matched physical-cycle budgets and the same
bounded-score Shiryaev--Roberts recursion. It failed:

- on spatial drift at \(q_1=0.55\), vAOC was **70.176 physical cycles
  slower** on average;
- on temporal drift at \(\kappa=0.75\), vAOC was 9.180 cycles faster as a
  point estimate, but its descriptive interval included zero and the
  preregistered distribution-free inference did not support an advantage;
- neither hypothesis passed its Bonferroni--Hoeffding upper-bound and
  Holm-adjusted \(p\)-value criteria, so the repository-level two-task
  comparison flag is `false`.

The results do support three narrower statements, which must not be conflated:

| Type of result | Verdict | Exact scope |
|---|---:|---|
| Accessible-information / representation separation | **Yes** | Count-only or one-round access can be exactly blind while spatial, pair-history, or separately audited logical information is discriminative in the declared controlled model. |
| ECA/AOC or vAOC detector advantage | **No** | The preregistered vAOC-versus-logistic criterion failed; raw AOC was also not consistently competitive with logistic, Hotelling, or likelihood methods. |
| Correlation-aware decoder utility | **Yes, conditionally** | With the injected correlated channel known, correlation-aware PyMatching reduced the \(d=7\) logical-error rate relative to ordinary post-model matching in a controlled circuit simulation. This was not an ECA/AOC alarm-and-update experiment. |
| Quantum, universal sample-efficiency, or scalable computational advantage | **No** | No such theorem or matched experiment was performed. |

Thus the strongest honest answer to “does our algorithm have an advantage?” is:

> **Not as a detector under the locked Run 5 criterion.** The positive result
> is an access/representation result---retain the correlations that carry the
> change---plus a separate known-channel decoder-model utility result. Neither
> establishes that ECA/AOC is better than a same-feature logistic rule,
> likelihood method, Helstrom/Wilson oracle, or another general detector.

## 2. Provenance and integrity

### 2.1 Freeze and configuration hashes

The identifiability, shadow, and circuit arms record
`9d59cba22f944b6574b429bcae39060a39c43c73`, the repository's “Freeze
cycle-fair Run 5 protocol” commit. The corrected offline arm records
`3ebf211821599d6bdd73687cce5dc22273597ca3`. That commit changes only the
persisted no-change class label from the CSV-sensitive literal `null` to
`no_change` and adds its regression test; the offline configuration and
scientific performance values are unchanged. The offline publication arm was
rerun after that schema-only repair.

The final sequential arm records
`1b55355370880872bcb853490cff4febfa2c1ea4`. Relative to the protocol freeze,
that commit adds the preregistered delay quartiles, detection-within-64/128
summaries, and separate `tracemalloc` allocation measurements plus regression
coverage. The formal sequential arm was rerun after this reporting-completeness
repair. Its primary delays and Hoeffding--Holm decision are unchanged.

| Arm | Locked configuration | SHA-256 |
|---|---|---|
| Sequential paper run | [`paper.json`](../experiments/run5/configs/paper.json) | `8b08de8022b7fd6983c2456db2ca871b1b9d244a7dea1b546bfd3f83078ada41` |
| Corrected offline diagnostic | [`offline_diagnostic_locked.json`](../experiments/run5/configs/offline_diagnostic_locked.json) | `d5938c08099e858dcfd6ba266a8a48123040c0d1a7322e0802f4b49949bc053f` |
| Circuit-level decoder arm | [`circuit_level_locked.json`](../experiments/run5/configs/circuit_level_locked.json) | `7b208f844ab90cca53815dbcc82e67989716c42e5297d5b4ef6fbe35d7376e9a` |
| Shadow measurement arm | [`shadow_measurement_locked.json`](../experiments/run5/configs/shadow_measurement_locked.json) | `9d789c64204ec7c51c42c2ed324335779b5ed2d94d7b1aa229d1ec0cb9acab48` |
| Identifiability arm | Configuration embedded in its manifest | Canonical sorted compact-JSON digest: `b0a7279561a02de0f0d6b2c97486244ce60f5f7eccbc81aa4f0d41770699b45e` |

The pilot directory is excluded from all conclusions. Its manifest records
the earlier Run 4 commit and it predates the cycle-unit, seed-partition,
baseline, and inferential corrections.

### 2.2 Result manifests

| Arm | Manifest | Manifest SHA-256 | Recorded wall time |
|---|---|---|---:|
| Identifiability | [`manifest.json`](../experiments/run5/results/identifiability/manifest.json) | `69bd6affccebd6e0f4f0a2e03c68b4fa340188d06650605ba73e4140cfc3a2aa` | 9.62 s |
| Corrected offline diagnostic | [`manifest.json`](../experiments/run5/results/offline_diagnostic_locked/manifest.json) | `35d2771e64b125245ce75674ce780db54f2053d4cc11251d6c65385a74796ef0` | 21.03 s |
| Shadow measurement | [`manifest.json`](../experiments/run5/results/shadow_measurement/manifest.json) | `b5984f351859207dc66f18829e102d3d54d56093183781d5486ae69d04353d34` | 4.63 s |
| Circuit-level decoder | [`manifest.json`](../experiments/run5/results/circuit_level_locked/manifest.json) | `673835b28a93569ab1b529a2a14f627cfe49c015d909582322b6766280eeba62` | 14.77 s |
| Sequential paper run | [`manifest.json`](../experiments/run5/results/syndrome_drift_paper/manifest.json) | `3c6a6ff25631f6175f9e20ba309da3f8a94262b4c497e88289758eac4b96f6d9` | 1195.46 s |

The recorded wall times sum to 1245.52 s (20.76 min), but these are
single-machine engineering timings, not benchmark-quality complexity
evidence. The manifests record Linux, Python 3.10.13, NumPy 2.2.6,
SciPy 1.15.3, scikit-learn 1.7.2, Stim 1.16.0, and PyMatching 2.4.0.

At audit time:

- every file listed in the five publication-grade manifests was rehashed;
  every SHA-256 matched;
- each arm's relevant executable, configuration, and test paths had no diff
  from its recorded commit;
- all manifests nevertheless record `git_dirty: true`.

The dirty flag means these must not be described as runs from a formally clean
checkout. It is consistent with generated/untracked result artifacts, but the
manifests do not preserve the contemporaneous dirty-path list. The defensible
provenance claim is therefore that the three recorded commits, locked
configuration hashes, commands, dependency versions, and output hashes are
internally consistent---not that either worktree was pristine.

## 3. Locked model, budgets, and estimands

The exact controlled arm uses an \(L\times L\) periodic phenomenological
endpoint-syndrome model. Its primary setting is

\[
L=5,\qquad m=L^2=25,\qquad
\eta=0.65,\qquad \epsilon=0.03,\qquad q_0=0.35.
\]

Spatial alternatives are \(q_1\in\{0.45,0.55,0.65\}\). Temporal alternatives
change latent persistence to
\(\kappa\in\{0.50,0.75,0.90\}\) while preserving the complete one-cycle
syndrome law. This is a controlled QEC-motivated model, not hardware data or
a complete surface-code circuit model.

The locked sequential design uses:

- 4096 physical cycles of covariance calibration per family;
- 8 fresh ridge-validation streams of 512 cycles, or 4096 changed cycles per
  family;
- 8 logistic-training streams per class of 512 cycles, or 4096 physical
  cycles per class and family;
- 512 no-change streams of 5000 cycles per scenario;
- 256 paired changed streams per scenario;
- change time 256 cycles for the secondary surveillance audit;
- a 1024-cycle post-change horizon;
- target ARL 1000 physical cycles;
- threshold 1000 one-cycle updates spatially and 500 nonoverlapping
  two-cycle updates temporally.

Calibration, logistic fitting, ridge validation, locked spatial tests, locked
temporal tests, scaling, timing, and bootstrap use declared disjoint seed
partitions. The temporal arm has half as many updates because each update
costs two physical cycles; no cross-family sample-efficiency claim is valid.

The primary delay estimand is **restart-at-change restricted mean detection
delay**. Every method is restarted at the known changepoint and evaluated on
the same post-change segment. This prevents the deterministic age of an SR
statistic from being misreported as change evidence. It is a controlled
method-comparison estimand, not an operational unknown-changepoint workflow.
The ordinary surveillance experiment and independent no-change audit are
reported separately.

## 4. Exact no-go and identifiability certificates

Let \(Z=\mathcal H(Y_{1:\infty})\) be the complete process visible to a
detector. If

\[
\mathcal H_\#P_0^{1:\infty}=\mathcal H_\#P_1^{1:\infty},
\]

then any stopping rule adapted only to \(Z\), including independent
randomization, has the same stopping-time law under the two regimes. Equality
of a one-round marginal is enough only when it implies equality of the
accessible **process** law, as it does for the iid spatial count arm. It is
not enough under temporal dependence.

The locked certificate gives:

| Drift | Accessible data | Result |
|---|---|---:|
| Spatial \(q:0.35\to0.55\) | Complete detector-count process | Exact TV \(=0\); ROC AUC \(=0.5\) |
| Spatial \(q:0.35\to0.55\) | Translation sufficient statistic | LLR AUC \(=0.5737747\) |
| Spatial \(q:0.35\to0.55\) | Full one-round syndrome | Same LLR AUC \(=0.5737747\) |
| Temporal \(\kappa=0.75\) | One-cycle full syndrome | Exact no-go; ROC AUC \(=0.5\) |
| Temporal \(\kappa=0.75\) | Nonoverlapping two-cycle likelihood | Pair-LLR AUC \(=0.5769941\) |
| Added logical loop | Complete syndrome history | Pathwise exact no-go; success \(=0.5\) |
| Added logical loop | Separate logical/Wilson audit | Success \(=1.0\) |

Additional identity checks were at numerical precision:

- complete spatial count-distribution gap: \(0\);
- maximum single-detector marginal gap: \(0\);
- direct versus sufficient spatial log-likelihood-ratio error:
  \(5.55\times10^{-15}\);
- direct versus closed-form temporal pair log-likelihood-ratio error:
  \(8.91\times10^{-15}\);
- logical-loop syndrome-history difference: \(0\).

These establish an **information-access hierarchy**: a compressed observation
can be provably blind, and the next richer declared observation can recover a
witness. They do not establish estimator optimality. The underlying
topological-code/QEC indistinguishability, Wilson/logical observables, and
restricted-measurement distinguishability are established prior concepts;
Run 5's contribution is an executable, jointly audited instance, not a new
toric-code or Wilson-loop theorem.

Source tables:
[`accessibility_hierarchy.csv`](../experiments/run5/results/identifiability/accessibility_hierarchy.csv),
[`identity_checks.csv`](../experiments/run5/results/identifiability/identity_checks.csv), and
[`summary.json`](../experiments/run5/results/identifiability/summary.json).

## 5. Corrected locked offline diagnostics

The v2 offline run supersedes the earlier v1 output. The correction matters:
only the \(D_4\) Fourier/pair-lift probability simplex uses the indicator
positive-support projector and receives an ECA/AOC interpretation. Positive
parts of count, first-moment, or translation mean differences are generic
linear contrasts, not Helstrom or AOC optima.

For each effect and class, the corrected run uses 800 training, 400 validation,
and 1000 held-out test windows. Spatial windows contain 64 cycles; temporal
windows contain 64 nonoverlapping pairs, or 128 physical cycles. Fixed model
hyperparameters and validation-selected thresholds never use test labels.
There are 18 methods per task across six tasks.

The complete count-sequence feature family was near chance across all tasks
and estimators: test ROC AUC ranged from 0.488435 to 0.524807, consistent with
the exact count-process no-go and finite-sample variation.

At the two middle effects:

| Scenario | Method | Test ROC AUC | Test balanced accuracy |
|---|---|---:|---:|
| Spatial \(q_1=0.55\) | Exact spatial window LLR ceiling | 0.935718 | 0.8625 |
|  | Fourier linear logistic | **0.922158** | 0.8375 |
|  | Fourier regularized Hotelling | 0.921865 | 0.8380 |
|  | Fourier simplex positive-support projector | 0.674646 | 0.6115 |
| Temporal \(\kappa=0.75\) | Exact full-HMM window LLR ceiling | 0.999246 | 0.9840 |
|  | Fourier linear logistic | **0.946676** | 0.8685 |
|  | Fourier RBF SVM | 0.940566 | 0.8640 |
|  | Fourier regularized Hotelling | 0.768531 | 0.6905 |
|  | Fourier simplex positive-support projector | 0.518481 | 0.5155 |

Validation selection among non-ceilings chose:

| Scenario/effect | Validation-selected method | Validation AUC | Held-out test AUC |
|---|---|---:|---:|
| Spatial 0.45 | Translation linear logistic | 0.778700 | 0.779360 |
| Spatial 0.55 | Fourier regularized Hotelling | 0.925344 | 0.921865 |
| Spatial 0.65 | Fourier regularized Hotelling | 0.988119 | 0.984515 |
| Temporal 0.50 | Fourier linear logistic | 0.866506 | 0.849434 |
| Temporal 0.75 | Fourier linear logistic | 0.950150 | 0.946676 |
| Temporal 0.90 | Fourier RBF SVM | 0.977369 | 0.974985 |

The offline result is positive for the **correlation representation**, not for
the raw positive-support rule. Once the relevant Fourier/pair representation
is available, conventional logistic, Hotelling, and RBF methods use it at
least as well and often much better. The exact likelihood/HMM values are
known-model ceilings. This arm has no sequential ARL or delay guarantee.

Source:
[`metrics.csv`](../experiments/run5/results/offline_diagnostic_locked/metrics.csv),
[`sample_manifest.csv`](../experiments/run5/results/offline_diagnostic_locked/sample_manifest.csv), and
[`summary.json`](../experiments/run5/results/offline_diagnostic_locked/summary.json).

## 6. Measurement-policy audit

The standalone shadow arm estimates 100 declared \(ZZ\) observables on two
controlled diagonal states, using 256 repetitions at copy budgets
64--4096. Uniform local-Pauli shadows contribute to a given \(ZZ\) observable
only when both local bases are \(Z\), with probability \(1/9\), and use the
inverse-channel factor 9. Native all-\(Z\) readout contributes every copy to
every declared observable.

At 4096 copies:

| State | Local-Pauli shadow mean RMSE | Native all-\(Z\) mean RMSE | Ratio |
|---|---:|---:|---:|
| \(q=0.35\) | 0.045054 | 0.010737 | 4.196 |
| \(q=0.65\) | 0.045192 | 0.010674 | 4.234 |

This is the expected cost of a universal measurement policy on a known
diagonal observable bank. It supports task-matched measurement design; it
does **not** compare AOC with eSCD, implement sequential shadow change
detection, or establish a quantum advantage.

Source:
[`aggregate_metrics.csv`](../experiments/run5/results/shadow_measurement/aggregate_metrics.csv) and
[`summary.json`](../experiments/run5/results/shadow_measurement/summary.json).

## 7. Circuit-level decoder utility

The locked circuit arm uses Stim rotated-memory-\(Z\) circuits with
rounds \(=d\), PyMatching, distances \(d\in\{3,5,7\}\), and 200,000 shots per
distance and stationary regime: 1.2 million sampled shots in total. The
reference regime has independent data-\(X\) faults; the correlated regime has
known common-plus-residual paired faults. There is no within-circuit
changepoint.

The exact detector-marginal audit passed at every distance, with maximum gap
\(3.33\times10^{-16}\) against tolerance \(10^{-12}\). The largest empirical
reference-versus-correlated detector-marginal gap was 0.002595. The two
regimes therefore change correlations while maintaining the designed
marginals to numerical precision.

At the predeclared \(d=7\) endpoint:

| Regime | Static null DEM | Ordinary post DEM | Correlation-aware post DEM |
|---|---:|---:|---:|
| Reference | 0.019785 | 0.021605 | 0.022175 |
| Correlated | 0.016530 | 0.015260 | **0.011200** |

For correlated shots, the primary paired comparison was

\[
\widehat p_{\rm ordinary}-\widehat p_{\rm corr\text{-}aware}
=0.004060,
\]

with paired multinomial-bootstrap 95% interval
\([0.003690,0.004435]\), a 26.6% relative reduction from 0.015260 to
0.011200. This supports lower logical error for the correlation-aware decoder
**when the injected post-change channel is known**.

The same model is harmful when applied to reference shots: at \(d=7\), the
correlation-aware decoder has error 0.022175 versus 0.019785 for the static
decoder; the paired difference
\(p_{\rm static}-p_{\rm corr\text{-}aware}=-0.002390\) has interval
\([-0.002930,-0.001855]\). Operational model selection therefore matters.

This arm tests decoder utility, not detector advantage. It contains neither a
sequential drift detector nor a delayed alarm--calibration--decoder-update
pipeline. It is controlled simulation, not hardware evidence, and neither
decoder is an oracle-independent benchmark.

Source:
[`logical_error_rates.csv`](../experiments/run5/results/circuit_level_locked/logical_error_rates.csv),
[`paired_decoder_differences.csv`](../experiments/run5/results/circuit_level_locked/paired_decoder_differences.csv),
[`dem_marginal_audit.csv`](../experiments/run5/results/circuit_level_locked/dem_marginal_audit.csv), and
[`summary.json`](../experiments/run5/results/circuit_level_locked/summary.json).

## 8. Locked sequential results

### 8.1 False-alarm/ARL basis

For a predictable bounded score \(s_t\in[-1,1]\), analytically centered under
the null,

\[
L_t(\beta)=1+\beta s_t,\qquad
\mathbb E_0[L_t(\beta)\mid\mathcal F_{t-1}]=1.
\]

The bounded-score SR mixtures therefore inherit the established e-detector
ARL result at their declared threshold. Independent auxiliary data choose
covariance, ridge, or logistic direction, but the effect is then fixed and
centered at the exact analytic null. The guarantee is conditional on those
frozen fits and on correct score bounds and null specification. It would not
automatically survive a plug-in finite-sample null mean.

The table below reports empirical restricted mean run length (RMRL) through
the 5000-cycle no-change horizon and its right-censoring fraction. A truncated
finite-sample RMRL is not itself a proof or disproof of the full ARL theorem.
For example, temporal raw AOC's point estimate 992.52 is compatible with a
full-mean lower bound because truncation can only reduce a mean and no
uncertainty interval was used for that descriptive estimate.

The exactly blind DFR/count score is identically zero. Its SR recursion grows
as \(R_t=t\) and deterministically alarms at its threshold. Its 1000-cycle
run length and 1000-cycle restart delay are clock behavior, not evidence for
the change; the same-horizon delay reduction is exactly zero.

### 8.2 Spatial middle effect: \(q_1=0.55\)

All delays are physical cycles. `Miss` is the fraction not detected through
the 1024-cycle restart horizon and hence assigned the restricted horizon.

| Method | Restart mean | Median | Miss | Null RMRL @5000 | Null censor | Same-horizon no-change RMRL | Delay reduction |
|---|---:|---:|---:|---:|---:|---:|---:|
| DFR/count exact pushforward | 1000.00 | 1000.0 | 0% | 1000.00 | 0% | 1000.00 | 0.00 |
| Exact one-cycle likelihood grid SR | 88.26 | 80.0 | 0% | 1169.83 | 1.56% | 725.46 | 637.21 |
| Known-post one-cycle likelihood SR | **86.82** | 79.5 | 0% | 1159.54 | 1.76% | 722.61 | 635.79 |
| Matched correlation witness | 459.78 | 459.5 | 0% | 1001.90 | 0% | 955.66 | 495.88 |
| Raw symmetry-resolved AOC | 592.77 | 587.0 | 3.13% | 1035.22 | 0% | 869.52 | 276.75 |
| vAOC / same-feature Hotelling | 540.30 | 535.0 | 0% | 1009.43 | 0% | 943.99 | 403.68 |
| Validation-trained logistic effect | 470.13 | 468.0 | 0% | 1001.64 | 0% | 957.05 | 486.92 |

The corresponding restart-delay quartiles and early-detection fractions are:

| Method | IQR [Q1, Q3], cycles | Detected by 64 cycles | Detected by 128 cycles |
|---|---:|---:|---:|
| DFR/count exact pushforward | [1000.00, 1000.00] | 0% | 0% |
| Exact one-cycle likelihood grid SR | [55.00, 114.00] | 34.77% | 82.42% |
| Known-post one-cycle likelihood SR | [56.75, 107.00] | 35.16% | 83.59% |
| Matched correlation witness | [419.00, 490.25] | 0% | 0% |
| Raw symmetry-resolved AOC | [435.25, 712.50] | 0% | 0% |
| vAOC / same-feature Hotelling | [473.00, 591.25] | 0% | 0% |
| Validation-trained logistic effect | [432.75, 501.00] | 0% | 0% |

vAOC is faster than raw AOC by 52.46 cycles in this arm, but slower than the
matched witness by 80.52, the named logistic comparator by 70.18, and the
likelihood methods by about 452--453 cycles. A spatial raw-to-variance-aware
improvement alone is therefore not a general detector advantage.

### 8.3 Temporal middle effect: \(\kappa=0.75\)

| Method | Restart mean | Median | Miss | Null RMRL @5000 | Null censor | Same-horizon no-change RMRL | Delay reduction |
|---|---:|---:|---:|---:|---:|---:|---:|
| DFR/count exact pushforward | 1000.00 | 1000.0 | 0% | 1000.00 | 0% | 1000.00 | 0.00 |
| Exact pair-restricted likelihood grid SR | 110.91 | 100.0 | 0% | 1203.71 | 0.78% | 741.46 | 630.55 |
| Full-HMM block-boundary grid SR | 53.24 | 48.0 | 0% | 1364.43 | 2.93% | 747.88 | 694.64 |
| Known-post full-HMM block-boundary SR | **51.84** | 46.0 | 0% | 1412.97 | 3.13% | 748.80 | 696.97 |
| Known-post pair-restricted likelihood SR | 109.54 | 98.0 | 0% | 1226.83 | 0.98% | 738.07 | 628.54 |
| Matched correlation witness | 963.73 | 964.0 | 0% | 1001.11 | 0% | 1001.11 | 37.38 |
| Raw symmetry-resolved AOC | 862.91 | 969.0 | 45.31% | 992.52 | 0% | 865.98 | 3.07 |
| vAOC / same-feature Hotelling | 959.36 | 990.0 | 29.69% | 1000.89 | 0% | 977.86 | 18.50 |
| Validation-trained logistic effect | 968.54 | 1000.0 | 39.84% | 1002.86 | 0% | 971.27 | 2.73 |

The corresponding restart-delay quartiles and early-detection fractions are:

| Method | IQR [Q1, Q3], cycles | Detected by 64 cycles | Detected by 128 cycles |
|---|---:|---:|---:|
| DFR/count exact pushforward | [1000.00, 1000.00] | 0% | 0% |
| Exact pair-restricted likelihood grid SR | [72.00, 140.00] | 17.97% | 69.53% |
| Full-HMM block-boundary grid SR | [34.00, 68.50] | 71.09% | 98.44% |
| Known-post full-HMM block-boundary SR | [36.00, 66.00] | 74.61% | 98.83% |
| Known-post pair-restricted likelihood SR | [71.50, 136.50] | 19.14% | 71.09% |
| Matched correlation witness | [960.00, 968.00] | 0% | 0% |
| Raw symmetry-resolved AOC | [727.00, 1024.00] | 0% | 0% |
| vAOC / same-feature Hotelling | [914.00, 1024.00] | 0% | 0% |
| Validation-trained logistic effect | [934.00, 1024.00] | 0% | 0% |

The raw restricted mean is lower than vAOC's but has a much larger miss
fraction; this mixed behavior is one reason not to infer an ordering from a
single summary. All three learned/constrained effects are far behind the
correctly specified pair and HMM likelihood controls. This shows that
retaining temporal-pair information is necessary but not sufficient for a
good sequential betting score.

The secondary surveillance audit conditions post-change delay on no
pre-change alarm. At the middle effects, the model-aware likelihood methods
had pre-change alarm fractions of approximately 9.8--13.3% before cycle 256;
the matched witness, vAOC, logistic effect, and blind control had zero, while
spatial raw AOC had 1.95%. These are fixed-horizon surveillance summaries,
not substitutes for the independent ARL audit.

Source:
[`aggregate_metrics.csv`](../experiments/run5/results/syndrome_drift_paper/aggregate_metrics.csv) and
[`replicate_metrics.csv`](../experiments/run5/results/syndrome_drift_paper/replicate_metrics.csv).

### 8.4 Preregistered Hoeffding--Holm decision

For each of the two middle effects, the paired estimand is

\[
D_i=T_i^{\rm vAOC}-T_i^{\rm logistic},\qquad
D_i\in[-H,H],\quad H=1024,\quad n=256.
\]

Under \(H_0:\mathbb E D_i\ge 0\), the locked analysis uses

\[
p_{\rm H}
=\min\left\{1,\exp\left[
-\frac{n\max(0,-\bar D)^2}{2H^2}
\right]\right\},
\]

Holm adjustment over the two hypotheses, and the simultaneous
Bonferroni--Hoeffding upper bound

\[
U_{\rm H}
=\bar D+H\sqrt{\frac{2\log(2/0.05)}{n}}.
\]

| Hypothesis | \(\bar D\), cycles | Descriptive bootstrap 95% interval | Simultaneous upper bound | Raw \(p_{\rm H}\) | Holm \(p\) | Pass |
|---|---:|---:|---:|---:|---:|---:|
| Spatial \(q_1=0.55\) | +70.176 | [61.937, 78.672] | +244.013 | 1.0000 | 1.0000 | **No** |
| Temporal \(\kappa=0.75\) | -9.180 | [-21.758, 3.148] | +164.657 | 0.9898 | 1.0000 | **No** |

The bootstrap intervals are descriptive only. The preregistered decision
requires both simultaneous upper bounds below zero, both Holm-adjusted
\(p<0.05\), and the ARL condition. Neither individual hypothesis passes and
the overall comparison flag is `false`.

This negative result must not be weakened into “competitive” or replaced by a
post-hoc comparison. The Hoeffding test is conservative, especially at
\(H=1024\), but it was the frozen distribution-free rule.

Source:
[`primary_named_comparator_audit.csv`](../experiments/run5/results/syndrome_drift_paper/primary_named_comparator_audit.csv) and
[`summary.json`](../experiments/run5/results/syndrome_drift_paper/summary.json).

## 9. Scaling and runtime audit

The scaling arm fixes the \(L=5\) spatial ridge \(\lambda=0.001\), uses
\(q_1=0.55\), and runs 128 paired restart replicates at
\(L\in\{5,7,9\}\), corresponding to 25, 49, and 81 detectors.

| Method | \(L=5\) mean delay | \(L=7\) | \(L=9\) |
|---|---:|---:|---:|
| DFR/count exact pushforward | 1000.00 | 1000.00 | 1000.00 |
| Raw symmetry-resolved AOC | 608.77 | 719.54 | 827.98 |
| vAOC / same-feature Hotelling | 542.25 | 744.07 | 867.81 |
| Matched correlation witness | 462.38 | 636.72 | 740.40 |
| Known-post one-cycle likelihood SR | 85.02 | 85.51 | 99.89 |

The analytic translation-feature \(L_1\) gap decreases from 0.018379 to
0.009377 to 0.005672 as \(L\) grows. The constrained detectors' delay
correspondingly worsens, while the known-model likelihood remains near
85--100 cycles. This is evidence **against** claiming a statistical scaling
advantage from this arm.

The recorded single-process timings and separately traced allocation peaks
per 10,000 samples are:

| \(L\) | Translation feature | Exact LLR | Exact/feature time ratio | Translation peak traced bytes | Exact-LLR peak traced bytes |
|---|---:|---:|---:|---:|---:|
| 5 | 0.002921 s | 0.035271 s | 12.07 | 8,167,876 | 20,454,076 |
| 7 | 0.015791 s | 0.162449 s | 10.29 | 15,847,876 | 39,654,812 |
| 9 | 0.008974 s | 0.122711 s | 13.67 | 26,087,932 | 65,255,836 |

The translation feature is cheaper in this small implementation, but these
numbers time only the declared routines after a ten-sample warm-up. They do
not include an end-to-end system, establish asymptotic complexity, or
compensate for the worsening delay. The allocation columns come from a
separate `tracemalloc` call over the 10,000-sample routine. They are peaks of
traced Python/NumPy allocations, **not** total process RSS, accelerator
memory, an end-to-end detector footprint, or a space-complexity theorem. The
nonmonotone wall times also underscore that these are engineering
measurements rather than scaling laws.

Source:
[`spatial_scaling.csv`](../experiments/run5/results/syndrome_drift_paper/spatial_scaling.csv).

### 9.1 Unfulfilled witness-localization metric

The preregistered secondary metric list named “witness
overlap/localization,” but the frozen protocol did not define a target
witness, a localization score, or a time-aggregation rule, and the dynamic
vAOC witness history was not persisted by the formal run. That metric is
therefore unfulfilled/inapplicable to the locked artifacts. This audit does
not invent a post-hoc overlap statistic.

Consequently, Run 5 cannot support a witness-localization claim. This omission
does not change the primary negative inference: the named-comparator decision
uses the persisted paired restricted delays, its frozen Hoeffding--Holm rule,
and the separate ARL basis, none of which depends on an overlap metric.

## 10. What the evidence means for ECA/AOC

### 10.1 What worked

1. **Access-first diagnosis.** Exact pushforward tests identify statistics
   that cannot detect the change, before fitting an algorithm.
2. **Symmetry-resolved feature recovery.** Fourier/translation or temporal
   pair lifts expose discriminative information that count and one-round
   marginals erase.
3. **Predictable bounded effects.** Analytic-null centering turns a frozen or
   predictable bounded contrast into a valid e-detector under the stated
   assumptions.
4. **Downstream correlation modeling.** Once the correct correlated channel
   is supplied, a correlation-aware decoder can materially reduce logical
   error in the controlled circuit arm.

### 10.2 What did not work

1. The raw positive-support projector did not provide a general detection
   advantage. It was particularly weak on the temporal offline task.
2. Hotelling whitening improved raw AOC spatially but did not consistently
   dominate raw AOC temporally and did not beat the same-feature logistic
   comparator under the locked two-task rule.
3. The constrained witnesses were far slower than exact likelihood/HMM
   controls when the post-change model was known.
4. The detector and decoder experiments were not joined into an operational
   alarm--estimation--update pipeline.
5. Delay worsened with lattice size in the limited scaling arm. The
   `tracemalloc` peaks are bounded implementation measurements, not total RSS
   or evidence of an asymptotic resource advantage.

### 10.3 Exact claim language

Permitted:

> In a controlled marginal-preserving drift model, complete count or
> one-round pushforwards can be exactly invariant while spatial or temporal
> correlation features remain identifiable. The locked benchmark recovered
> those witnesses and instantiated analytically centered bounded
> e-detectors.

Permitted for the circuit arm:

> In a known-channel Stim/PyMatching simulation, correlation-aware decoding
> reduced the predeclared \(d=7\) correlated-regime logical-error rate from
> 0.01526 to 0.01120 relative to ordinary post-model matching.

Required negative statement:

> The preregistered vAOC-versus-same-feature-logistic detection comparison did
> not pass on either individual hypothesis or jointly; Run 5 therefore
> provides no locked ECA/AOC detector-advantage claim.

Not permitted:

- superiority to Helstrom, a Wilson/logical oracle, exact likelihood/HMM, or
  a correctly specified threshold/logistic rule on the same sufficient
  feature;
- superiority to DGR, Bayesian syndrome inference, syndrome-to-DEM
  estimation, or classical-shadow eSCD without faithful matched
  implementations;
- quantum acceleration, universal sample efficiency, or scalable
  computational advantage;
- hardware robustness or generality across code families, coherent/leakage
  noise, schedules, and decoders;
- discovery of local indistinguishability, Wilson loops, topological order,
  a toric-code theorem, string theory, or holographic duality.

Gauge-theory, charge/flux-sector, tensor-network, string-theory, or
holographic applications remain future research questions. Run 5 supplies no
evidence for those claims.

## 11. Practical next experiment

The present evidence motivates, but does not yet perform, an end-to-end test:
generate a circuit-level or hardware syndrome stream with a hidden
changepoint; let each detector alarm without knowing that changepoint; reserve
fresh post-alarm data for channel/DEM estimation; account for calibration and
update latency; and compare decoded logical outcomes against static,
correlation-aware, covariance/CUSUM, decoder-likelihood, DGR-style, and
faithful eSCD-style baselines under matched physical shots.

That experiment should make logical error after the full alarm-and-update
pipeline the primary endpoint. It is the shortest route from the valid
information-access result here to a genuinely practical advantage question.
