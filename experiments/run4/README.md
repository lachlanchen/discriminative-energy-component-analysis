# Run 4: local blindness and topological-flux contrast

Run 4 is an exact, finite \(\mathbb Z_2\) toric-code / quantum-double
fixed-point benchmark on a \(3\times3\) torus.  It asks whether observable
contrast can certify that a declared local measurement class contains no
topological-sector information and then recover the first accessible
noncontractible witness.

The benchmark compares two ground sectors with opposite eigenvalues of one
logical \(Z\) Wilson loop.  It predeclares both outcomes:

- every Pauli observable below code distance and every reduced state on one
  or two links must be blind;
- a three-link noncontractible Wilson loop must distinguish the sectors
  perfectly.

This is a physics-correctness and representation-access benchmark.  It does
not claim an advantage over Helstrom discrimination or a supplied Wilson-loop
oracle; those methods must tie the learned witness at the fixed point.

The ordinary link-qubit partial trace used here is the extended-link
(electric-center) prescription.  A physical gauge Hilbert space does not
possess a unique spatial tensor-product factorization, so the algebra and
center choice remain part of the claim.

## Reproduce

```bash
.venv/bin/python -m pytest experiments/run4/tests -q
.venv/bin/python experiments/run4/scripts/run_topological_flux.py
```

Generated CSV/JSON evidence, the robustness figure, and a hash manifest are
written under `results/topological_flux/`.

## Interpretation boundary

The exact experiment can establish:

1. a local no-information certificate below code distance;
2. recovery of a Wilson-loop-equivalent effect after the correct
   label-preserving nuisance twirl;
3. failure under a deliberately label-flipping twirl;
4. calibrated loss under logical-sector mixing and readout flips.

It cannot establish a new toric-code theorem, generic algorithmic advantage,
efficient global tomography, confinement, holography, or a string-theory
result.  The next practical test is held-out surface-code syndrome drift with
equal shot budgets and decoder, covariance, CUSUM, Wilson-loop, and observable
change-detection baselines.
