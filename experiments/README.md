# Reproducible experiments

The repository uses immutable research runs:

| Run | Question | Status |
|---|---|---|
| [`run1/`](run1/) | DECA as an optimal commuting measurement/classifier | frozen baseline |
| [`run2/`](run2/) | additive maximum-observable contrast for online physical states | validated working paper |
| [`run3/`](run3/) | symmetry-resolved observable contrast and sector diagnostics | validated working paper |
| [`run4/`](run4/) | exact local blindness and topological-flux witness recovery | validated exact experiment |
| [`run5/`](run5/) | accessible-process certificates and cycle-fair syndrome-drift/decoder audits | validated working paper |
| [`run6/`](run6/) | predeclared real-QEC comparison with separated access phases and exact provenance | locked benchmark; derived artifacts released separately |

Reusable maintained code lives in `aoc/`. A run owns its scripts, tests,
result tables, figures, and manifest; later runs never overwrite earlier
evidence.

## Environment

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e './experiments[quantum,test]'
```

## Validation

```bash
.venv/bin/python -m pytest
```

Runs 2–6 have deterministic entry points under their respective `scripts/`
directories. The smaller generated CSV/JSON evidence, figures, and manifests
for the earlier runs are committed under each run's `results/` directory.

Run 6 result artifacts are substantially larger and are intentionally ignored
at `experiments/run6/results/`. The versioned, hashed archive is distributed
with the
[v6.0.0 release](https://github.com/lachlanchen/discriminative-energy-component-analysis/releases/tag/v6.0.0).
Verify the release checksums and embedded manifests before using those
artifacts. The
[Run 6 paper](../publication/run6/main.pdf) and
[result and claim audit](../references/run6_real_qec_results_and_advantage_audit.md)
define the locked interpretation; the archive does not establish a general
algorithmic or quantum advantage.

The exact locked conclusion is:

> **No demonstrated S-PACE algorithmic advantage.**

Downloaded public datasets are cached under the ignored `experiments/data/`
directory. Original code in this tree is licensed under
[`experiments/LICENSE`](LICENSE).
