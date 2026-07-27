# Reference archive

This directory preserves the research lineage behind ECA and the proposed
Discriminative Energy Component Analysis (DECA).

## Public contents

| Artifact | Provenance | How it should be read |
|---|---|---|
| [`2003.10199v3.pdf`](2003.10199v3.pdf) | Local copy of the original ECA preprint | Historical source; see [arXiv:2003.10199](https://arxiv.org/abs/2003.10199) |
| [`_Rongzhou___ISCAS2025__Eigen_Component_Analysis__A_Quantum_Theory_Inspired_Linear_Model/`](_Rongzhou___ISCAS2025__Eigen_Component_Analysis__A_Quantum_Theory_Inspired_Linear_Model/) | ISCAS-era manuscript source, figures, and later candidate manuscript material | Source archive; the authoritative published paper is the [ISCAS 2025 version](https://doi.org/10.1109/ISCAS56072.2025.11044249) |
| [`_Rongzhou___ICIP2025__Ising_Clustering__A_Democratic_Voting_Approach/`](_Rongzhou___ICIP2025__Ising_Clustering__A_Democratic_Voting_Approach/) | Author research draft | Exploratory and unpublished; its objective is critically audited in the deep analysis |
| [`1. 更 general 的数学：ECA 代表什么通用概念？.md`](1.%20更%20general%20的数学：ECA%20代表什么通用概念？.md) | Earlier ChatGPT conversation supplied by the author | Research-process provenance, not peer-reviewed evidence |
| [`eca_deep_research_analysis.md`](eca_deep_research_analysis.md) | Evidence-backed synthesis produced from the local artifacts, public code, and primary literature | Main analysis and future research roadmap |
| [`deca_theory_and_novelty_spec.md`](deca_theory_and_novelty_spec.md) | Formal DECA/POVM/Spectral-DECA derivation plus frozen experiment findings | Current mathematical specification |
| [`additive_symmetry_observable_contrast_theory.md`](additive_symmetry_observable_contrast_theory.md) | Unified AOC/SAOC derivation, multiclass POVM formulation, applications, and string-theory boundary | Main run 2/run 3 theory and research program |
| [`global_run_architecture_plan.md`](global_run_architecture_plan.md) | Immutable run layout and verification record | Repository provenance |
| [`run2_additive_observable_contrast_plan.md`](run2_additive_observable_contrast_plan.md) | Pre-change run 2 claim/evidence plan | Traceability record |
| [`run3_symmetry_resolved_observable_contrast_plan.md`](run3_symmetry_resolved_observable_contrast_plan.md) | Pre-change run 3 claim/evidence plan | Traceability record |
| [`publication_plan.md`](publication_plan.md) | Claim, evidence, file, and verification plan for the new manuscript | Traceability record |

The executable implementation and paper generated from this analysis now live
in [`../experiments/`](../experiments/) and [`../publication/`](../publication/).

## Deliberately excluded from the public repository

- The full TCAS-II decision email and recipient addresses.
- Private editorial correspondence and earlier internal scratch plans.
- A duplicate TCAS candidate-source directory whose main files are
  byte-identical to material already preserved in the ISCAS-era archive and
  which cannot be verified as the exact submitted artifact.
- Tangential notes that do not support the repository's main research claim.

The substantive, anonymized reviewer concerns are analyzed in
[`eca_deep_research_analysis.md`](eca_deep_research_analysis.md). Excluding the
private correspondence protects coauthor contact information and avoids
publishing confidential editorial records.

## Historical-source build status

These manuscript directories are preserved as research artifacts, not as a
clean release package:

- `eca_short.tex` and the ICIP `main.tex` both completed a first PDF pass during
  repository validation when supplied with an external `IEEEtran.cls`.
- The public archive does not vendor `IEEEtran.cls` or `IEEEtran.bst`; install
  the IEEEtran LaTeX distribution before attempting a full build.
- `ISCAS2025/conference_101719.tex` refers to legacy figure filenames that are
  not present in the supplied directory.
- Generated PDFs and LaTeX intermediates are intentionally excluded from Git.

These limitations are documented instead of silently modifying historical
manuscript source.

## Evidence labels

The documents use the following distinctions:

- **Published:** supported by a stable publication record.
- **Source observation:** directly verified from a local artifact or public
  implementation.
- **Derivation:** follows mathematically from an explicitly stated model.
- **Proposal:** a new algorithm, interpretation, or experiment not yet
  validated.
- **Validated experiment:** a result backed by a committed entry point, raw
  records, pinned versions, and tests.
- **Exploratory note:** useful research context, but not independent evidence.

## Rights

The directory contains author manuscripts, published-paper artifacts, IEEE
template material, and generated analysis. These items may have different
copyright and reuse terms. See the repository-level [`LICENSE.md`](../LICENSE.md).
