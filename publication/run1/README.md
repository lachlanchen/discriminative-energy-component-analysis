# Run 1: DECA manuscript

`main.tex` is the canonical journal-style manuscript. The author line and
target venue are placeholders until the human contributors confirm
authorship, affiliations, and submission plans.

Build from the repository root:

```bash
.venv/bin/python experiments/run1/scripts/export_paper_tables.py
make -C publication/run1
```

The generated tables are derived directly from the frozen CSV results under
`experiments/run1/results/`; the manuscript figures include the corresponding
PDF plots by relative path. `main.pdf` is the compiled artifact.
