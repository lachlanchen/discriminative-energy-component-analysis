# Run 5 manuscript

This directory contains the staged Run 5 journal-style working paper:

> *What Syndrome Data Can and Cannot Detect: Accessible-Observable Sequential
> Change Detection for Topological Quantum Error Correction*

The draft distinguishes exact statements, controlled phenomenological
evidence, circuit-level Stim/PyMatching evidence, and absent hardware
evidence. Its sequential and offline locked-result fields are deliberately
represented by searchable `\TBD...` macros until those outputs pass the
declared audit. They must not be replaced by pilot values.

Build with:

```bash
make
```

Before release, search the TeX source for `TBD`, replace only from audited raw
records/manifests, rebuild, check for undefined references/citations and
overfull boxes, and inspect every PDF page.
