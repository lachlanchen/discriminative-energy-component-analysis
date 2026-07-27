# Contributing

Contributions that make the ECA/DECA claims more precise, reproducible, or
falsifiable are welcome.

## Useful contributions

- Corrections to a derivation, source citation, or implementation audit.
- Counterexamples that identify the limits of squared energy features or the
  shared-eigenbasis assumption.
- Reproducible implementations of the proposed low-rank Stiefel model,
  class-energy contrast objective, or worst-class-pair margin.
- Fair benchmark scripts with fold-local preprocessing, fixed seeds, nested
  validation, and strong baselines.
- Repairs to the exploratory Ising/Potts clustering formulation that include
  explicit sign, assignment, balance, and anti-collapse constraints.

## Evidence standard

Please label a contribution as one of:

- reproduction of a published result;
- observation from source code or a local artifact;
- mathematical derivation;
- new proposal;
- empirical result.

New empirical claims should include the data split, preprocessing, hardware,
software versions, hyperparameter budget, random seeds, and uncertainty
estimates.

## Pull requests

1. Open an issue or concise design note for substantial algorithm changes.
2. Keep one pull request focused on one claim or implementation unit.
3. Add tests or a reproducible command when code is introduced.
4. Run `git diff --check` and verify all Markdown links you touched.
5. Do not include confidential review correspondence, private datasets,
   credentials, personal contact lists, or publisher-restricted artifacts.

By contributing, you confirm that you have the right to submit the material.
No repository-wide license is implied; see [`LICENSE.md`](LICENSE.md).
