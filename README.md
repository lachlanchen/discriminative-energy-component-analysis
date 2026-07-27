# Observable Contrast Research

[![Research](https://img.shields.io/badge/status-reproducible%20working%20papers-2b6cb0)](publication/)
[![Tests](https://img.shields.io/badge/tests-350%20passing-2ea44f)](experiments/)
[![Runs](https://img.shields.io/badge/research%20runs-6-6f42c1)](experiments/)
[![ISCAS 2025](https://img.shields.io/badge/ECA%20origin-ISCAS%202025-00629B)](https://doi.org/10.1109/ISCAS56072.2025.11044249)
[![中文](https://img.shields.io/badge/README-简体中文-red)](README.zh-Hans.md)

This repository follows one research idea from ECA to a broader mathematical
and physical framework:

> Encode observations as positive states, specify the measurements that are
> actually accessible, and learn the accessible observable whose expectation
> differs most between regimes.

The point is not to replace every classifier. It is to solve a precise problem
that appears in state discrimination, streaming change detection, physical
mode localization, optics, chemistry, invariant signals, and many-body
physics.

## The central result

For empirical states \(\rho_0,\rho_1\), let
\(\Delta=\rho_1-\rho_0\). Among all bounded effects,

\[
\max_{0\preceq E\preceq I}\operatorname{Tr}(E\Delta)
=\operatorname{Tr}(\Delta_+)
=\frac12\|\Delta\|_1.
\]

The optimizer is the positive spectral projector
\(E^\star=\mathbf 1_{\Delta>0}\). This is the known Helstrom/Jordan result,
not a newly claimed quantum theorem. It gives an exact operational meaning to
the original ECA motivation: maximize class difference rather than pooled
variance.

The deeper run 3 statement handles physical constraints. If
\(\mathcal A\) is the accessible observable algebra and
\(\mathcal E_{\mathcal A}\) is the trace-preserving conditional expectation
onto it, then

\[
\max_{\substack{E\in\mathcal A\\0\preceq E\preceq I}}
\operatorname{Tr}(E\Delta)
=\operatorname{Tr}\!\left[
  \mathcal E_{\mathcal A}(\Delta)_+
\right].
\]

Group symmetry is the special case where
\(\mathcal E_{\mathcal A}\) is a twirl. The resulting representation blocks
say which parity, frequency, charge, or other physical sector carries the
change.

## Six immutable research runs

| Run | Question | Paper |
|---|---|---|
| [run 1](experiments/run1/) | When is ECA an optimal commuting measurement, and what is lost relative to a general POVM? | [DECA PDF](publication/run1/main.pdf) |
| [run 2](experiments/run2/) | Can maximum-difference witnesses be accumulated, merged, capacity-limited, and used online? | [AOC PDF](publication/run2/main.pdf) |
| [run 3](experiments/run3/) | What is the exact optimum under symmetry or a physical readout algebra, and which sector changed? | [SAOC PDF](publication/run3/main.pdf) |
| [run 4](experiments/run4/) | Can a local observable algebra be certified blind before the first noncontractible gauge-sector witness is recovered? | [integrated Run 3+4 PDF](publication/run4/main.pdf); [advantage audit](references/run4_gauge_sector_results_and_advantage_audit.md) |
| [run 5](experiments/run5/) | When syndrome marginals are unchanged, can correlation-aware sequential evidence detect drift and improve decoding under equal physical-cycle budgets? | [Run 5 PDF](publication/run5/main.pdf); [result and advantage audit](references/run5_surface_code_drift_results_and_advantage_audit.md) |
| [run 6](experiments/run6/) | What does a predeclared, calibration-aware comparison show on public real-device QEC syndrome data under fixed shot budgets, separated detector/randomization/outcome access, and exact provenance? | [Run 6 PDF](publication/run6/main.pdf); [result and claim audit](references/run6_real_qec_results_and_advantage_audit.md); [v6.0.0 derived artifacts](https://github.com/lachlanchen/discriminative-energy-component-analysis/releases/tag/v6.0.0) |

Run 1 is the frozen DECA baseline. Later runs do not overwrite its code,
results, or manuscript. Shared maintained code lives in
[`experiments/aoc/`](experiments/aoc/).

Run 6 is a fixed empirical benchmark, not a theorem or a general advantage
claim. Its provenance records separate detector construction, blinded
randomization, external-snapshot processing, delayed outcome access, and a
disclosed post-detector execution repair. Any final gate must be read only for
the named implementations, datasets, budgets, and endpoints. It cannot
establish superiority to Helstrom or Wilson oracles, the same parity feature
with logistic/threshold rules, quantum speedup, universal sample efficiency,
or scalable computational advantage.

The locked Run 6 result is:

> **No demonstrated S-PACE algorithmic advantage.**

The composite missed the Google primary event and captured `0/31` primary
decoder mismatches at top 20, versus `9/31` for DFR.  Its PNNL delay was
lower than the named online-logistic implementation but higher than DFR, so
the required two-comparator retention conjunction also failed.  The spectral
component's event detection is exploratory and cannot replace the frozen
composite after outcome access.

## What is new—and what is prior art

Established foundations are credited directly: Helstrom state
discrimination, POVMs, trace distance, group-invariant hypothesis testing,
conditional expectations, CSP/generalized eigendecomposition, kernel MMD, and
symmetry-resolved entanglement.

The repository's contribution is the tested synthesis:

- additive and exactly mergeable positive-state summaries;
- full and rank-constrained maximum-difference witnesses;
- a multiclass minimum-error POVM solver with feasibility diagnostics;
- predictable learned witnesses coupled to a betting e-process under a known
  independent reference;
- symmetry-/subalgebra-restricted optimal effects;
- additive sector diagnostics and an exact \(O(d\log d)\) cyclic-translation
  implementation;
- an exact toric-code local no-information certificate and recovery of the
  first Wilson-loop-equivalent flux witness;
- accessible-process no-go certificates, cycle-fair e-detectors, and a
  controlled Stim/PyMatching decoder-utility audit;
- a predeclared real-QEC benchmark with separated access phases, a disclosed
  post-detector execution repair, and artifact-level provenance;
- cross-domain experiments that report ties, failures, and claim boundaries.

The detailed derivation and application map are in
[`additive_symmetry_observable_contrast_theory.md`](references/additive_symmetry_observable_contrast_theory.md).

## Selected evidence

| Matched problem | Result | Honest comparison |
|---|---:|---|
| Algebraic identities | max analytic/SDP error \(6.67\times10^{-9}\) | numerical precision audit |
| Sign-paired Ising phases | AOC \(0.9998\) accuracy; linear \(0.5000\) | RBF SVM ties; physical energy oracle is perfect |
| 35% spring damage, 128-sample windows | rank-1 AOC AUC \(0.9768\); mean logistic \(0.4985\) | covariance centroid \(0.9733\); oracle \(0.9774\) |
| Diagonal/antidiagonal polarization | learned analyzer \(0.9500\) success; fixed H/V \(0.5000\) | finite-shot Aer matches the analytic result |
| Translation nuisance, 2 samples/class | invariant AOC \(0.999861\); raw methods \(0.5000\) | correctly specified Fourier-power logistic is \(1.0\) |
| 10-qubit TFIM reduced state | response peak \(g/J=0.9625\) | known thermodynamic critical field is 1 |
| Hückel difference density | attachment/detachment balance error \(5.55\times10^{-17}\) | controlled tight-binding model only |
| Simulated 6-axis robot contact | learned screw overlap \(0.99925\) | structured simulation, not deployed hardware |
| \(3\times3\) toric-code flux sectors | all 1,431 sub-distance Pauli gaps are 0; Wilson-loop trace distance is 1 | AOC ties the Wilson and Helstrom oracles when given the correct symmetry/access model |
| \(5\times5\) marginal-preserving syndrome drift | count TV and maximum detector-marginal gap are 0; correlation likelihood AUCs are \(0.5738/0.5770\) | an information-access separation, not an AOC advantage |
| Equal-budget sequential detection | spatial vAOC is 70.18 cycles slower than the named logistic effect; temporal is 9.18 cycles faster but inconclusive | the predeclared two-task advantage decision is false |
| Stim/PyMatching, injected correlated channel, \(d=7\) | correlation-aware decoding lowers logical error \(1.526\%\to1.120\%\) | 26.6% relative reduction for a known simulated post-channel; not a detector or hardware result |

The strong differences occur in deliberately matched mean-blind,
low-rank, or symmetry-nuisance regimes. The project makes no claim of
universal accuracy, quantum speedup, privacy, ASIC/FPGA advantage, or
real-world robot/chemistry deployment.

## Applications

Direct uses are the ones where the state and observable already have physical
meaning:

- optical polarization and coherence analyzers;
- quantum-state/device drift and bounded observable change;
- data-driven order parameters and symmetry-sector response;
- zero-mean structural vibration and covariance-mode localization;
- translation-/phase-nuisance vibration, radar, sonar, and vision signals.

Promising uses requiring real domain validation include:

- robot force/torque and tactile contact modes;
- electronic difference densities and spectroscopy;
- EEG/CSP-style spatial power changes;
- rotational molecular features built on established SOAP/ACE
  representations.

Run 4 completes the finite fixed-point version of the proposed
lattice-gauge calculation: it compares toric-code flux sectors, proves local
blindness below code distance, and recovers a Wilson-loop-equivalent effect.
This is a \(\mathbb Z_2\) lattice-gauge/QEC testbed, not a string-theory or
holography result. Run 5 executes the equal-budget surface-code
syndrome-drift step. It finds no vAOC advantage over its named learned
comparator, but it does verify that correlation access restores information
lost by marginals and that a correctly selected post-change decoder can help
in controlled simulation. The next practical step requires end-to-end
hardware syndrome streams, online model selection, and decoder switching.
Run 6 evaluates a frozen comparison protocol on public real-device QEC
syndrome data. Its paper and claim audit, rather than this overview, are the
canonical record of the locked empirical outcome. Neither outcome would be a
new Wilson-loop, toric-code, string-theory, or holographic-duality result.

## Quick start

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e './experiments[quantum,qec,test]'

.venv/bin/python -m pytest -q
```

Run 2 experiments:

```bash
.venv/bin/python experiments/run2/scripts/run_algebraic_validation.py
.venv/bin/python experiments/run2/scripts/run_ising_order.py
.venv/bin/python experiments/run2/scripts/run_structural_monitoring.py
.venv/bin/python experiments/run2/scripts/run_optical_quantum.py
```

Run 3 experiments:

```bash
.venv/bin/python experiments/run3/scripts/run_translation_vision.py
.venv/bin/python experiments/run3/scripts/run_quantum_phase.py
.venv/bin/python experiments/run3/scripts/run_huckel_difference.py
.venv/bin/python experiments/run3/scripts/run_robot_contact.py
```

Run 4 exact gauge-sector experiment:

```bash
.venv/bin/python experiments/run4/scripts/run_topological_flux.py
```

Run 5 locked syndrome-drift suite:

```bash
.venv/bin/python experiments/run5/scripts/run_identifiability_certificate.py
.venv/bin/python experiments/run5/scripts/run_syndrome_drift.py \
  --config experiments/run5/configs/paper.json
.venv/bin/python experiments/run5/scripts/run_offline_diagnostic_audit.py \
  --config experiments/run5/configs/offline_diagnostic_locked.json
.venv/bin/python experiments/run5/scripts/run_shadow_measurement_audit.py
.venv/bin/python experiments/run5/scripts/run_circuit_level_validation.py
```

Run 6 uses staged detector, randomization, external-snapshot, and outcome
runners. The exact locked sequence and artifact hashes are documented in the
[Run 6 paper](publication/run6/main.pdf) and
[result and claim audit](references/run6_real_qec_results_and_advantage_audit.md).
Large derived outputs are intentionally excluded from Git and distributed
with the
[v6.0.0 release](https://github.com/lachlanchen/discriminative-energy-component-analysis/releases/tag/v6.0.0);
verify its release checksums and embedded manifests before reanalysis.

Build all manuscripts:

```bash
make -C publication
```

No IBM Quantum account is required; the quantum circuits run locally on Aer.
Every new result directory includes raw CSV/JSON evidence and a manifest with
the command, dependencies, Git state, runtime, and output hashes.

## Repository map

| Path | Purpose |
|---|---|
| [`experiments/aoc/`](experiments/aoc/) | Maintained additive, multiclass, streaming, symmetry, physics, chemistry, and quantum primitives |
| [`experiments/run1/`](experiments/run1/) | Frozen DECA code, tests, scripts, and evidence |
| [`experiments/run2/`](experiments/run2/) | Additive/streaming observable-contrast validation |
| [`experiments/run3/`](experiments/run3/) | Symmetry-resolved and cross-domain validation |
| [`experiments/run4/`](experiments/run4/) | Exact local-blindness and topological-flux validation |
| [`experiments/run5/`](experiments/run5/) | Cycle-fair syndrome-drift, measurement, and decoder validation |
| [`experiments/run6/`](experiments/run6/) | Predeclared real-QEC evaluation, access separation, randomization, and provenance |
| [`publication/`](publication/) | Six independent paper sources and compiled PDFs for runs 1–6 |
| [`references/`](references/) | Original ECA/Ising materials, reviews, research plans, and deep theory analysis |
| [v6.0.0 release](https://github.com/lachlanchen/discriminative-energy-component-analysis/releases/tag/v6.0.0) | Hashed Run 6 derived artifacts kept outside Git history |
| [`CITATION.cff`](CITATION.cff) | GitHub citation metadata |
| [`LICENSE.md`](LICENSE.md) | Public/private and mixed-rights boundary |

## Historical lineage

The original ECA paper was published at IEEE ISCAS 2025
([DOI](https://doi.org/10.1109/ISCAS56072.2025.11044249)). The 2020 preprint,
ISCAS source, exploratory Ising clustering manuscript, and the author's
earlier discussion are preserved under [`references/`](references/) as
provenance. They are not rewritten to make later results appear historical.

The working papers use `Rongzhou (Lachlan) Chen` as the repository-author
placeholder. Final authorship, affiliations, acknowledgments, and target
venues must be confirmed by the human contributors before submission.

## Citation

GitHub renders “Cite this repository” from [`CITATION.cff`](CITATION.cff).

```bibtex
@software{chen2026observablecontrast,
  author  = {Chen, Rongzhou},
  title   = {Observable Contrast Research:
             From Eigen-Components to Additive and
             Symmetry-Resolved Physical Witnesses},
  year    = {2026},
  version = {6.0.0},
  url     = {https://github.com/lachlanchen/discriminative-energy-component-analysis}
}
```

Original code under [`experiments/`](experiments/) is MIT-licensed.
Historical manuscripts, figures, templates, and publications can have
different rights; see [`LICENSE.md`](LICENSE.md).
