# Run 1: DECA experiments

This directory contains the executable research code for the theory in
[`references/deca_theory_and_novelty_spec.md`](../references/deca_theory_and_novelty_spec.md).

## Methods

- `binary_helstrom`: analytical binary DECA/Helstrom solution.
- `optimal_povm`: globally optimal multi-class minimum-error measurement
  solved by semidefinite programming.
- `pretty_good_measurement`: analytical PGM baseline.
- `jacobi_deca`: ancilla-free commuting measurement optimized by hard
  component assignment and analytical two-dimensional Jacobi rotations.
- `DECAClassifier(decision_rule="measurement")`: physical POVM/PVM outcome
  probabilities and deterministic argmax over those probabilities.
- `DECAClassifier(decision_rule="spectral")`: common-basis quadratic
  affinities that retain discriminative eigenvalue magnitudes.
- `quantum`: Qiskit circuits for the ancilla-free PVM and Naimark-dilated
  general POVM.

## Environment

From the repository root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e './experiments[quantum,test]'
```

The quantum dependencies follow the current IBM documentation recommendation
to install Qiskit and Qiskit Aer separately.

## Validation

```bash
.venv/bin/python -m pytest experiments/run1/tests
```

## Reproducible runs

```bash
.venv/bin/python experiments/run1/scripts/run_theory_validation.py
.venv/bin/python experiments/run1/scripts/run_quantum_simulation.py
.venv/bin/python experiments/run1/scripts/run_classical_benchmarks.py
.venv/bin/python experiments/run1/scripts/analyze_classical_results.py
.venv/bin/python experiments/run1/scripts/run_storage_audit.py
.venv/bin/python experiments/run1/scripts/export_paper_tables.py
```

The classical benchmark downloads four fixed UCI archives, verifies their
SHA-256 digests, and caches them under the ignored `data/uci/` directory.
Use `--quick --repeats 1` for a smoke run. The paper's fixed protocol is
`--folds 5 --repeats 2`. Generated CSV, JSON, PDF, and PNG evidence is written
under `run1/results/`.

The code distinguishes:

1. class-ensemble single-shot success;
2. deterministic argmax classification;
3. finite-shot circuit simulation.
4. repeated-shot spectral observable estimation.

These metrics are not interchangeable.

## License

Original code in this directory is available under the
[MIT License](LICENSE). Historical manuscript material elsewhere in the
repository has separate rights.
