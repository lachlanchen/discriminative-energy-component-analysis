# Run 4 plan: gauge-sector observable contrast

Date: 2026-07-27
Status: executed and verified; see
[`run4_gauge_sector_results_and_advantage_audit.md`](run4_gauge_sector_results_and_advantage_audit.md)

## Scientific question

Can symmetry-/subalgebra-accessible observable contrast identify information
that is provably invisible to every sufficiently local observable, and can a
known label-preserving logical symmetry recover the correct witness from one
exact representative state description per class?

The controlled model is the \(L=3\) toric-code fixed point on a torus,
equivalently a finite \(\mathbb Z_2\) quantum-double/lattice-gauge model. Two
ground states differ only in a noncontractible Wilson-loop eigenvalue.

## Predeclared claims and boundaries

The experiment may claim:

1. exact gauge/stabilizer constraints and orthogonal topological sectors;
2. equality of all reduced states on fewer than the code-distance number of
   links;
3. zero expectation contrast for every Pauli string below code distance;
4. perfect discrimination by the noncontractible Wilson-loop sector
   projector;
5. one-representative-per-class recovery after twirling over the known
   label-preserving nuisance group;
6. explicit degradation under predeclared logical-sector mixing.

It must not claim:

- a new toric-code, gauge-theory, Wilson-loop, or quantum-error-correction
  theorem;
- a solution to string theory, holography, or a continuum gauge theory;
- discovery of the Wilson loop without a supplied topology/symmetry prior;
- universal classifier or computational advantage;
- a result independent of the chosen local observable algebra or
  extended-Hilbert-space convention for reduced states.

The gauge-invariant Wilson-loop statement is algebraic. Reduced density
matrices are calculated in the tensor product of link Hilbert spaces (the
extended-Hilbert-space embedding); gauge-theory boundary centers and edge
modes are therefore discussed explicitly.

## Exact model

Put one qubit on each of \(2L^2\) links. Define

\[
A_s=\prod_{e\ni s}X_e,\qquad
B_p=\prod_{e\in\partial p}Z_e,\qquad
H=-\sum_s A_s-\sum_p B_p .
\]

Ground states satisfy \(A_s=B_p=+1\). On a torus the four-dimensional ground
space is labeled by two independent noncontractible logical Wilson loops.
Run 4 compares the \(W_x^Z=+1\) and \(W_x^Z=-1\) states at fixed
\(W_y^Z=+1\).

For a loop region \(R\) containing the three links of \(W_x^Z\), the expected
reduced states are

\[
\rho_\pm^R=\frac{P_\pm}{2^{L-1}},\qquad
P_\pm=\frac{I\pm W_x^Z}{2}.
\]

Thus \(D(\rho_+^R,\rho_-^R)=1\), while every region with fewer than \(L\)
links has identical reduced states.

## Experiments

1. Construct all four ground states from the star group and verify
   normalization, orthogonality, \(A_s=B_p=1\), and logical eigenvalues.
2. Enumerate all Pauli strings of weights 1 and 2 and record their maximum
   expectation gap; scan weight 3 to locate the first nonzero/topological
   contrast.
3. Compare trace distance on one link, two links, and the noncontractible
   three-link loop.
4. Compare measurement success:
   - every sub-distance local bounded observable;
   - an untwirled AOC effect from one logical representative state per class;
   - correct label-preserving and deliberately label-flipping twirls;
   - the analytical Wilson-loop oracle.
5. Mix opposite logical sectors with probability \(p\in[0,0.5]\) and verify
   the predicted optimal success \(1-p\).
6. Report wall time, state dimension, state support, observable weight, and
   the supplied-prior boundary.

## Verification

- Unit tests for lattice indexing, stabilizers, logical sectors, partial
  trace, and parity twirling.
- Deterministic raw CSV/JSON, figure, and manifest under
  `experiments/run4/results/`.
- Full repository tests, Ruff, manifest-hash audit, and no private artifacts.
- Results and advantage interpretation reviewed against primary literature
  before commit.
