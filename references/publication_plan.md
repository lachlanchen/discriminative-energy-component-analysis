# Publication plan: DECA journal-style manuscript

Date: 2026-07-27
Status: approved by the user's request to produce a high-quality LaTeX paper
Frozen manuscript target: `publication/run1/main.tex`

## Scope and baseline

This is a new, journal-length manuscript derived from the historical ECA
preprint and ISCAS paper, not an in-place edit of either archived source.
Historical files under `references/` remain unchanged and provide provenance.
The new paper must not claim that prior Helstrom, PGM, density-classifier, SDP,
or joint-diagonalization results are original.

The manuscript will use a neutral journal/preprint layout. It may be adapted
to TPAMI or another algorithm-oriented venue after authorship and venue choice
are confirmed; the repository must not imply acceptance or submission.

## Central argument

The paper will present DECA as a measurement-compression problem:

1. A common unitary cannot improve an unrestricted POVM oracle.
2. Under a commuting/shared-basis constraint, the optimal decoder is hard.
3. Binary DECA reduces analytically to the known Helstrom measurement.
4. Commuting multiclass ensembles are also exact.
5. A joint-diagonalization residual bounds the general-POVM success gap.
6. Jacobi-DECA provides monotone analytical pair updates for the noncommuting
   multiclass approximation.
7. PVM-DECA and Spectral-DECA are distinct operational rules: the first
   optimizes single-shot discrimination; the second retains eigenvalue
   magnitudes for deterministic quadratic classification.
8. Qiskit circuits expose the PVM-versus-Naimark resource trade-off.

## Claim discipline

The paper will explicitly report:

- no quantum speedup, privacy, ASIC/FPGA, power, or hardware-latency claim;
- no universal tabular accuracy claim;
- binary Helstrom and general POVM theory as prior art;
- classical state loading as an unresolved cost;
- the exact \(K>d\) PVM output-capacity limitation;
- strong negative results on Dry Bean and Letter;
- only dataset-level blocks in statistical tests, not overlapping CV folds as
  independent observations;
- fixed, non-nested baseline configurations, so the empirical study is a
  mechanism/resource evaluation rather than a state-of-the-art benchmark.

## Planned manuscript structure

1. Abstract
2. Introduction and contribution boundary
3. Related work
4. Problem formulation and unrestricted POVM oracle
5. Commuting-measurement theorems
6. Spectral-DECA and shot complexity
7. Multiclass Jacobi-DECA
8. Quantum circuit compilation
9. Experimental protocol
10. Theory and circuit validation
11. Classical results and resource analysis
12. Applications, limitations, and ethics/reproducibility
13. Conclusion
14. Appendices with proofs, pseudocode, and detailed tables

## Evidence and figure mapping

| Manuscript evidence | Frozen source |
|---|---|
| Binary/commuting exactness and gap bound | `experiments/run1/results/theory/*.csv` |
| Noncommuting residual/gap plot | `experiments/run1/results/theory/noncommutativity_gap.pdf` |
| Trine and binary circuit sampling | `experiments/run1/results/quantum/*.csv` |
| Quantum success plot | `experiments/run1/results/quantum/quantum_simulation_success.pdf` |
| 1,100-fit benchmark | `experiments/run1/results/classical/benchmark_folds.csv` |
| Accuracy heatmap | `experiments/run1/results/classical/benchmark_accuracy.pdf` |
| Mean-rank plot | `experiments/run1/results/classical/benchmark_mean_rank.pdf` |
| Dataset-level tests | `experiments/run1/results/classical/statistical_summary.json` and `paired_dataset_comparisons.csv` |
| Runtime/storage/sweep diagnostics | `experiments/run1/results/classical/resource_summary.csv` |

All numerical values in prose and tables must be checked against these files.

## Files to create

- `publication/run1/main.tex`
- `publication/run1/references.bib`
- `publication/run1/Makefile`
- `publication/run1/README.md`
- `publication/run1/main.pdf` (compiled artifact)

## Verification

```bash
.venv/bin/python -m pytest experiments/run1/tests -q
.venv/bin/python experiments/run1/scripts/analyze_classical_results.py
make -C publication
git diff --check
```

Before the final commit:

- verify every theorem reference and equation label;
- inspect the compiled PDF page images for overflow or unreadable tables;
- ensure bibliography has no unresolved citations;
- compare key paper numbers to JSON/CSV sources programmatically;
- keep `publication/run1/main.tex` as the run 1 source and `main.pdf` as its
  compiled counterpart.

## Authorship boundary

The draft will use `Rongzhou (Lachlan) Chen` as a placeholder corresponding to
the repository owner. Historical coauthors will be cited through the ISCAS
paper, not silently added to the new manuscript. Final authorship, affiliation,
acknowledgments, and corresponding-author details must be confirmed by the
humans before submission.
