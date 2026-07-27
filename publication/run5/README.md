# Run 5 manuscript

This directory contains the Run 5 journal-style working paper:

> *What Syndrome Data Can and Cannot Detect: Accessible-Observable Sequential
> Change Detection for Topological Quantum Error Correction*

The draft distinguishes exact statements, controlled phenomenological
evidence, circuit-level Stim/PyMatching evidence, and absent hardware
evidence. It reports the objective-corrected locked offline audit and the
cycle-fair publication-grade sequential test. The primary result is negative:
neither predeclared vAOC-versus-named-logistic comparison is supported, and
the overall named-comparator advantage flag is therefore false.

Build with:

```bash
make
```

Before release, rebuild, check for undefined references/citations and overfull
boxes, and inspect every PDF page. Numerical edits must remain traceable to
the frozen raw records and manifests under `experiments/run5/results/`.
