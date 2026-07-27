# Research plan: run 3 symmetry-resolved observable contrast

Date: 2026-07-27
Status: executed and verified
Target manuscript: `publication/run3/main.tex`
Target code: `experiments/aoc/` and `experiments/run3/`

## Scientific goal

Solve maximum-observable contrast when the instrument or the physical question
must respect a known compact symmetry group. The output is not only a scalar
distance: it identifies which irreducible symmetry sectors carry the change.

## Mathematical core

For a unitary representation \(U_g\), define the twirling conditional
expectation

\[
\mathcal T_G(X)=\int_G U_gXU_g^\dagger\,dg.
\]

The central result to prove and test is

\[
\sup_{\substack{0\preceq E\preceq I\\[E,U_g]=0}}
\operatorname{Tr}(E\Delta)
=\operatorname{Tr}\!\left[\mathcal T_G(\Delta)_+\right].
\]

With
\(\mathcal H=\bigoplus_\lambda V_\lambda\otimes M_\lambda\), derive the
commutant-block form

\[
\mathcal T_G(\Delta)
=\bigoplus_\lambda
\frac{I_{V_\lambda}}{\dim V_\lambda}\otimes\Delta_\lambda
\]

and an additive sector decomposition of the distinguishability. This theorem
is expected to follow from known group-invariant hypothesis testing and
conditional-expectation theory; novelty claims are restricted to the online
algorithm, diagnostic decomposition, and cross-domain applications unless the
literature audit proves more.

## Exact special cases

- \(U(1)\) global phase: rank-one projectors are already invariant.
- Global \(Z_2\): sign-flip-invariant order-parameter discovery.
- Cyclic translations: twirling is diagonal in the DFT basis, giving an
  \(O(d\log d)\) exact streaming spectral-contrast algorithm.
- Finite rotations/reflections: orbit averaging with exact finite groups.
- SO(3): spherical-harmonic irrep blocks, described and tested on controlled
  molecular/optical data without pretending to replace SOAP or ACE.

## Required experiments

1. Translation-randomized optical/vision textures: recover the discriminative
   frequency bands; compare with raw mean, PCA, FFT-power baselines, and RBF.
2. Ising \(Z_2\) order discovery: quantify overlap between the learned witness
   and the known magnetization/correlation modes across lattice sizes and
   temperatures.
3. A molecular-orbital or rotational-density case that uses physically valid
   density matrices; if no reliable chemistry data are available, label a
   controlled tight-binding example as a toy rather than a chemical result.
4. Small transverse-field Ising reduced states: symmetry-sector
   distinguishability across a quantum phase transition.
5. Optical Jones/coherency states: convert the optimal witness to analyzer
   settings and verify finite-shot performance.

## Domain map and claim boundaries

- Direct: optics, quantum devices, phase/order diagnostics, streaming
  covariance systems, translation-invariant signal/vision changes.
- Plausible with domain validation: robot contact modes, structural health,
  molecular environments and spectroscopy.
- Theoretical frontier only: algebraic QFT, modular Hamiltonians, holography,
  and string-theory tensor-network states. No claim that this work solves a
  string-theory problem without a concrete model, calculation, and expert
  validation.

## Manuscript and verification

Run 3 is a separate manuscript, not a silent overwrite of run 2. It must:

- cite symmetry-constrained quantum hypothesis testing, invariant statistics,
  harmonic analysis, SOAP/ACE, and symmetry-resolved entanglement;
- compile from `publication/run3/main.tex`;
- include a reproducible theorem/experiment audit;
- visually inspect every generated figure and the final PDF;
- report negative or non-identifiable cases, especially where second-order
  power spectra lose phase information.

## Execution notes

- Implemented finite-group and FFT cyclic twirls, invariant contrast, sector
  decomposition, small-system TFIM states, and Hückel densities.
- Added six run 3 tests and four fixed-seed experiments covering translation
  nuisance, quantum parity response, controlled difference-density chemistry,
  and simulated robot contact.
- A reliable real chemistry or robot dataset was not introduced; both results
  are explicitly labeled controlled models.
- String theory remains a concrete future reduced-state/gauge-sector
  calculation, not a result claim.
- The manuscript builds to `publication/run3/main.pdf`; visual and unresolved
  reference audits pass.
- Git commit and push are recorded in repository history.
