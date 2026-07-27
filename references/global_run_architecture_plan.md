# Repository plan: immutable research runs

Date: 2026-07-27
Status: executed and verified

## Goal

Preserve every completed research pass as a self-contained run, while keeping
licenses, packaging, shared algorithms, research notes, and build entry points
at the repository root.

## Target layout

```text
experiments/
├── aoc/                 # shared, maintained implementation
├── run1/                # delivered DECA baseline
├── run2/                # additive observable contrast
├── run3/                # symmetry-resolved observable contrast
├── LICENSE
├── README.md
└── pyproject.toml

publication/
├── run1/                # delivered DECA manuscript and PDF
├── run2/                # additive-observable manuscript and PDF
├── run3/                # symmetry-resolved manuscript and PDF
├── Makefile
└── README.md
```

## Scope

### Files allowed to move

- `experiments/deca`, `experiments/scripts`, `experiments/tests`, and
  `experiments/results` into `experiments/run1/`.
- The current manuscript sources, generated tables, bibliography, PDF, and
  run-specific Makefile into `publication/run1/`.

### Root files allowed to change

- `experiments/README.md`, `experiments/pyproject.toml`
- `publication/README.md`, `publication/Makefile`
- root READMEs, citation metadata, contribution notes, and `.gitignore`

### Preservation rules

- Run 1 numerical CSV/JSON results and compiled PDF are immutable evidence.
- Git history remains the byte-level provenance of the original layout.
- Run 2 and run 3 never overwrite a run 1 artifact.
- Shared source is allowed only under `experiments/aoc/`; each run keeps its
  scripts, tests, manifests, results, and environment record.
- Private TCAS correspondence remains ignored.

## Verification

- Run 1 tests pass from the new path.
- Every manuscript builds through `make -C publication`.
- No LaTeX source points to the obsolete directory layout.
- Each result manifest records the command, random seed, dependency versions,
  wall time, and output hashes.
- `git diff --check` passes.

## Execution notes

- The former experiment and publication roots were preserved as `run1/`
  through Git-aware moves.
- Shared maintained primitives were added under `experiments/aoc/`.
- Run 2 and run 3 own independent scripts, tests, results, manifests, paper
  sources, bibliographies, and compiled PDFs.
- All 36 tests pass and all three manuscripts build through the root
  publication Makefile.
- Final Git commit and push are recorded in repository history rather than
  hard-coded here.
