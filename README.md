# Discriminative Energy Component Analysis

[![Research](https://img.shields.io/badge/status-validated%20research-2b6cb0)](publication/main.pdf)
[![Tests](https://img.shields.io/badge/tests-17%20passing-2ea44f)](experiments/tests)
[![Qiskit](https://img.shields.io/badge/Qiskit-2.5.1-6929c4)](experiments/scripts/run_quantum_simulation.py)
[![ISCAS 2025](https://img.shields.io/badge/ECA%20origin-ISCAS%202025-00629B)](https://doi.org/10.1109/ISCAS56072.2025.11044249)
[![中文](https://img.shields.io/badge/README-简体中文-red)](README.zh-Hans.md)

This repository turns the original Eigen-Component Analysis (ECA) idea into a
tested mathematical and computational framework:

> Learn class-discriminative energy components and compile them into the
> simplest measurement that the problem structure permits.

The central result is **Discriminative Energy Component Analysis (DECA)**,
formulated as minimum-error classification under a commuting/shared-basis
measurement constraint. The repository includes:

- a journal-style [paper source](publication/main.tex) and
  [compiled PDF](publication/main.pdf);
- analytical binary and commuting-multiclass results;
- a monotone multiclass Jacobi algorithm;
- SDP and Pretty Good Measurement (PGM) oracles;
- real Qiskit Aer PVM and Naimark-dilation simulations;
- 17 passing tests and 1,100 repeated-CV benchmark fits;
- the original 2020 preprint, ISCAS 2025 source archive, and exploratory Ising
  clustering work for provenance.

## The idea in three equations

Encode an input as a unit state and estimate one class operator per class:

\[
\rho_x=\phi(x)\phi(x)^\dagger,\qquad
A_c=\pi_c\,\mathbb E[\rho_x\mid y=c].
\]

For a shared basis \(P=[p_1,\ldots,p_d]\), the best commuting measurement has
an analytical hard decoder:

\[
S_{\mathrm{DECA}}(P)=
\sum_{j=1}^d\max_c p_j^\dagger A_cp_j.
\]

For two classes, let \(\Delta=A_1-A_2\). Its eigenbasis is globally optimal:

\[
S_{\mathrm{DECA}}^\star
=\frac12(1+\|\Delta\|_1)
=S_{\mathrm{Helstrom}}^\star.
\]

This binary equality is the known Helstrom result, not a new quantum
discrimination theorem. The DECA contribution is the constrained-measurement
reformulation, hard-decoder elimination, multiclass exactness and gap bound,
Jacobi solver, and the distinction between:

- **PVM-DECA:** keeps only eigenvalue signs for optimal single-shot binary
  measurement;
- **Spectral-DECA:** retains eigenvalue magnitudes for deterministic quadratic
  classification or repeated-shot observable estimation.

## What the experiments show

| Evidence | Result |
|---|---|
| 30 random binary trials | closed form vs. SDP gap \(\le 2.0\times10^{-8}\) |
| 16 commuting multiclass trials | DECA vs. SDP gap \(\le 1.45\times10^{-8}\) |
| 72 noncommuting trials | zero violations of the proved residual bound |
| Qiskit Aer circuits | max probability total-variation error \(0.0073\) |
| Trine-state example | general POVM improves success by \(0.04466\) |
| Classical benchmark | 10 datasets, 11 methods, 1,100 outer fits |

The controlled covariance experiment validates the original intuition:
Spectral-DECA reaches \(0.786\pm0.014\), versus \(0.496\pm0.019\) for logistic
regression. The public-data results are deliberately less flattering. PGM
usually improves over a projective DECA measurement, but RBF-SVM and random
forest remain stronger on most tasks. On 26-class Letter Recognition with
only 17 encoded dimensions, the experiment exposes the exact \(K>d\) PVM
capacity limit.

The project therefore claims a rigorous **mechanism and resource trade-off**,
not universal accuracy, quantum speedup, privacy, or hardware advantage.

## Quick start

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e './experiments[quantum,test]'

.venv/bin/python -m pytest experiments/tests -q
.venv/bin/python experiments/scripts/run_theory_validation.py
.venv/bin/python experiments/scripts/run_quantum_simulation.py
```

Run the complete classical study:

```bash
.venv/bin/python experiments/scripts/run_classical_benchmarks.py \
  --folds 5 --repeats 2
.venv/bin/python experiments/scripts/analyze_classical_results.py
```

The UCI downloader uses fixed URLs and SHA-256 verification. Downloaded data
are cached under ignored `experiments/data/`; all CSV, JSON, PDF, and PNG
evidence is under [`experiments/results/`](experiments/results/).

Build the paper:

```bash
.venv/bin/python experiments/scripts/export_paper_tables.py
make -C publication
```

No IBM Quantum account is required for the included local Aer simulations.

## Repository map

| Path | Purpose |
|---|---|
| [`publication/main.pdf`](publication/main.pdf) | Current journal-style paper |
| [`publication/main.tex`](publication/main.tex) | Canonical LaTeX source |
| [`experiments/deca/`](experiments/deca/) | Encodings, operators, solvers, classifier, and circuits |
| [`experiments/tests/`](experiments/tests/) | Analytical, API, and Qiskit tests |
| [`experiments/scripts/`](experiments/scripts/) | Reproducible experiment and table entry points |
| [`experiments/results/`](experiments/results/) | Frozen raw records, summaries, and figures |
| [`references/deca_theory_and_novelty_spec.md`](references/deca_theory_and_novelty_spec.md) | Detailed Chinese theory/novelty specification |
| [`references/eca_deep_research_analysis.md`](references/eca_deep_research_analysis.md) | Audit of the original ECA and Ising directions |
| [`references/README.md`](references/README.md) | Historical provenance and public/private boundary |

## Historical lineage

The original ECA paper was published at IEEE ISCAS 2025
([DOI](https://doi.org/10.1109/ISCAS56072.2025.11044249)). This repository
preserves the 2020 preprint and author source materials but does not alter them
to make the new theory appear historical.

The new manuscript uses `Rongzhou (Lachlan) Chen` as a placeholder author.
Final authorship, affiliations, acknowledgments, and target venue must be
confirmed by the human contributors before submission.

## Citation

```bibtex
@software{chen2026deca,
  author  = {Chen, Rongzhou},
  title   = {Discriminative Energy Component Analysis:
             Optimal Commuting Measurements and Spectral
             Quadratic Classification},
  year    = {2026},
  version = {1.0.0},
  url     = {https://github.com/lachlanchen/discriminative-energy-component-analysis}
}
```

The original ISCAS paper has its own citation in [`CITATION.cff`](CITATION.cff)
and the manuscript bibliography.

## Rights and reuse

Original code under [`experiments/`](experiments/) is MIT-licensed. Historical
papers, figures, manuscripts, templates, and other archive material can have
different rights. See [`LICENSE.md`](LICENSE.md) before reuse.
