# Discriminative Energy Component Analysis

[![Research status](https://img.shields.io/badge/status-research%20roadmap-8a2be2)](#project-status)
[![ISCAS 2025](https://img.shields.io/badge/ISCAS%202025-10.1109%2FISCAS56072.2025.11044249-00629B)](https://doi.org/10.1109/ISCAS56072.2025.11044249)
[![Implementation](https://img.shields.io/badge/code-lachlanchen%2Feca-181717?logo=github)](https://github.com/lachlanchen/eca)
[![中文](https://img.shields.io/badge/README-简体中文-red)](README.zh-Hans.md)

This repository is the research archive and forward-looking mathematical
roadmap for **Eigen-Component Analysis (ECA)**. It connects the original
2020 idea, the published ISCAS 2025 model, a critical implementation audit,
and a more rigorous next-generation formulation:

> Learn a low-rank orthogonal basis whose class-conditional directional
> energies differ as strongly and reliably as possible.

The main output is the
[deep research analysis](references/eca_deep_research_analysis.md). It
separates established results, proposed theory, implementation findings, and
claims that still require experiments.

## The model in one equation

For an orthogonal basis \(P=[p_1,\ldots,p_r]\), ECA uses squared projection
features

\[
z_P(x)=(P^\top x)\odot(P^\top x).
\]

A class score

\[
s_c(x)=\sum_j L_{jc}(p_j^\top x)^2
\]

is equivalently

\[
s_c(x)=x^\top Q_cx,\qquad
Q_c=P\operatorname{diag}(L_{:c})P^\top.
\]

ECA is therefore linear in the learned **energy features**, but a structured
quadratic classifier in the original input. Its class operators share an
eigenbasis and commute.

The proposed discriminative objective replaces total variance with
class-conditional energy contrast:

\[
\max_{P^\top P=I}
\sum_c\pi_c
\left\|
\operatorname{diag}\!\left(P^\top(R_c-\bar R)P\right)
\right\|_2^2.
\]

This is equivalent to approximately jointly diagonalizing centered
class-conditional second-moment matrices. The full derivation, limits, and
experimental plan are in the analysis.

## Why this repository exists

The published work introduced an interpretable squared-component model, but
several questions remained open:

- What precisely is maximized when the goal is class difference rather than
  total variance?
- Is ECA a linear model, a quadratic model, a subspace classifier, or a
  quantum measurement model?
- Which quantum-mechanical statements are exact, and which are analogies?
- When does a shared eigenbasis help, and when does noncommutativity make it
  too restrictive?
- How should parameter counts, orthogonality, probabilities, scaling, and
  privacy claims be audited?
- How can the exploratory Ising/Potts clustering branch be repaired?

This repository turns those questions into definitions, falsifiable
hypotheses, ablations, and a staged research plan.

## Repository map

| Path | Contents | Status |
|---|---|---|
| [`references/eca_deep_research_analysis.md`](references/eca_deep_research_analysis.md) | Mathematical, physical, ML, implementation, reviewer, and experiment analysis | Primary research document |
| [`references/2003.10199v3.pdf`](references/2003.10199v3.pdf) | Original ECA preprint artifact | Historical source |
| [`references/_Rongzhou___ISCAS2025__Eigen_Component_Analysis__A_Quantum_Theory_Inspired_Linear_Model/`](references/_Rongzhou___ISCAS2025__Eigen_Component_Analysis__A_Quantum_Theory_Inspired_Linear_Model/) | ISCAS-era LaTeX and figures, plus later candidate manuscript material | Research source archive |
| [`references/_Rongzhou___ICIP2025__Ising_Clustering__A_Democratic_Voting_Approach/`](references/_Rongzhou___ICIP2025__Ising_Clustering__A_Democratic_Voting_Approach/) | Exploratory Ising/Potts clustering draft | Unpublished exploratory work |
| [`references/1. 更 general 的数学：ECA 代表什么通用概念？.md`](references/1.%20更%20general%20的数学：ECA%20代表什么通用概念？.md) | Earlier ChatGPT discussion retained for provenance | Exploratory notes, not evidence |
| [`references/README.md`](references/README.md) | Provenance and public/private boundary | Archive guide |
| [`CITATION.cff`](CITATION.cff) | Repository citation metadata | Citable research record |

The maintained Python implementation remains in
[`lachlanchen/eca`](https://github.com/lachlanchen/eca). This repository does
not duplicate it.

## Key research conclusions

- An invertible orthogonal transform alone cannot make linearly inseparable
  data linearly separable; the extra capacity comes from squaring the
  projected coordinates.
- Strictly normalized ECA with row-stochastic class effects is a commuting
  POVM; hard component allocation is a projective measurement.
- Elementwise sigmoid class weights do not by themselves form a categorical
  probability model.
- Adding a learned diagonal term inside the skew-matrix exponential generally
  breaks orthogonality and energy conservation.
- The strongest defensible novelty is not generic “maximum separability.”
  It is discriminative class-energy spectra represented by shared-eigenbasis
  quadratic operators and supervised joint diagonalization.
- The current Ising-clustering pairwise-difference objective needs an explicit
  sign choice, anti-collapse constraints, and a Potts/max-\(K\)-cut
  formulation for more than two clusters.

## Project status

This repository intentionally distinguishes three levels of maturity:

1. **Published:** the ECA paper appeared at IEEE ISCAS 2025
   ([DOI](https://doi.org/10.1109/ISCAS56072.2025.11044249),
   [author-hosted PDF](https://www.eee.hku.hk/optima/pub/conference/2505_ISCASa.pdf)).
2. **Audited:** the analysis derives the actual quadratic model and documents
   mathematical and implementation inconsistencies that should be corrected.
3. **Proposed, not yet validated:** DECA's class-contrast joint-diagonalization
   objective, worst-pair energy margin, low-rank Stiefel optimization, and the
   repaired Ising/Potts extension require implementation and benchmark study.

No quantum speedup, privacy guarantee, hardware advantage, or universal
accuracy advantage is claimed here.

## Suggested validation path

The proposed work should be tested on:

- controlled mean-only, covariance-only, commuting, noncommuting, and
  higher-moment synthetic problems;
- OpenML and TabZilla tabular benchmarks;
- strong baselines including LDA/MMC, CSP, QDA, polynomial SVM,
  nearest-subspace methods, NCA/LMNN, boosted trees, and MLPs;
- repeated nested cross-validation with confidence intervals;
- strict parameter, storage, FLOP, training-time, latency, and calibration
  reporting;
- ablations for orthogonality, class-effect normalization, rank, sparsity,
  loss design, and initialization.

See the [research roadmap](references/eca_deep_research_analysis.md#13-分阶段研究路线)
for the full checklist.

## Citation

GitHub reads [`CITATION.cff`](CITATION.cff) and exposes a
**Cite this repository** action. To cite the published ECA paper:

```bibtex
@inproceedings{chen2025eca,
  title     = {Eigen-Component Analysis: A Quantum Theory-Inspired Linear Model},
  author    = {Chen, Rongzhou and Zhao, Yaping and Liu, Hanghang and
               Xu, Haohan and Ma, Shaohua and Lam, Edmund Y.},
  booktitle = {2025 IEEE International Symposium on Circuits and Systems (ISCAS)},
  pages     = {1--5},
  year      = {2025},
  doi       = {10.1109/ISCAS56072.2025.11044249}
}
```

To cite this research archive:

```bibtex
@software{chen2026deca,
  author = {Chen, Rongzhou},
  title  = {Discriminative Energy Component Analysis: Research Archive and Roadmap},
  year   = {2026},
  url    = {https://github.com/lachlanchen/discriminative-energy-component-analysis}
}
```

## Contributing and reuse

Corrections, counterexamples, reproducible experiments, and carefully scoped
implementations are welcome; see [`CONTRIBUTING.md`](CONTRIBUTING.md).

This archive contains mixed-origin research material, manuscript sources, and
publisher/template assets. No single open-source license applies to every
file. Read [`LICENSE.md`](LICENSE.md) before reuse.
