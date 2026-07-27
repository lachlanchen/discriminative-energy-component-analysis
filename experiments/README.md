# Reproducible experiments

The repository uses immutable research runs:

| Run | Question | Status |
|---|---|---|
| [`run1/`](run1/) | DECA as an optimal commuting measurement/classifier | frozen baseline |
| [`run2/`](run2/) | additive maximum-observable contrast for online physical states | validated working paper |
| [`run3/`](run3/) | symmetry-resolved observable contrast and sector diagnostics | validated working paper |

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

Run 2 and run 3 have deterministic entry points under their respective
`scripts/` directories. Generated CSV/JSON evidence, figures, and manifests
are committed under each run's `results/` directory.

Downloaded public datasets are cached under the ignored `experiments/data/`
directory. Original code in this tree is licensed under
[`experiments/LICENSE`](LICENSE).
