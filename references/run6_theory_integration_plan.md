# Run 6 theory integration plan

**Stage:** internal theory revision before any Run 6 manuscript is edited
**Date:** 2026-07-27
**Revision type:** scientific integration after adversarial audit

## Problem to solve

The first Run 6 theory draft gives a correct orbit-centered e-factor under
conditional orbit invariance, but that assumption is not a fact of current
QEC hardware. Pure orbit centering also deletes a chip-wide/common-mode
change, which is precisely a plausible signature of a high-energy or
radiation-associated event. The public Google and PNNL archives do not contain
an exchangeable synchronized canary stream.

The revised theory must therefore separate:

1. mathematical conditions that give exact sequential validity;
2. retrospective hardware scores whose thresholds are calibrated
   empirically;
3. global and symmetry-breaking sectors;
4. one-step maximum difference, sequential log growth, and quickest-change
   objectives; and
5. evidence allocation from calibrated localization inference.

## Target theory

Use the provisional name **S-PACE**:

> symmetry-/pair-calibrated additive contrast e-detection.

The name is provisional and is not a novelty claim. The final note will
derive a common bounded-contrast interface with three distinct inputs:

- a known conditional reference;
- a conditionally exchangeable simultaneous pair and an antisymmetric score;
- within-round centering over a scientifically valid conditional symmetry
  orbit or stratum.

An estimated finite reference is a fourth, empirical mode. It is not to be
called exact unless calibration uncertainty is included in a simultaneous
conditional-bias envelope or joint e-process.

For a unitary group representation, decompose every paired/reference
contrast into the trivial representation and nontrivial isotypic sectors.
The trivial sector is the global/common-mode channel. Nontrivial sectors
carry relative or localized symmetry breaking. A fixed mixture over sector,
location, shape, sparsity and bet-size components must be used; an
uncorrected maximum or same-round spatial product is forbidden.

## Analytical components to include

1. **Paired-exchangeability theorem.** A predictable bounded antisymmetric
   score has conditional mean zero and yields a linear e-factor.
2. **Sector decomposition.** The Reynolds/isotypic projections preserve
   conditional centering; removing the trivial sector creates an exact
   common-mode blind spot.
3. **Accessible-observable solution.** For a past contrast operator, the
   positive spectral projector is the standard Jordan/Helstrom solution of
   the one-step expectation-gap problem.
4. **Sparse analytical solution.** Over the capped signed simplex, the
   linear-gap optimizer selects the \(k\) largest absolute feature
   contrasts with their signs and equal weight.
5. **Predictable online portfolio.** Entropic mirror descent on
   \(-\log(1+\beta s_t)\) remains e-valid because the next witness is chosen
   from the past. State a deterministic regret bound relative to the best
   fixed capped sparse witness, crediting online convex optimization and
   universal-portfolio foundations.
6. **Change accumulation.** Keep the proper-prior e-process and
   Shiryaev--Roberts e-detector separate; state ever-alarm versus ARL
   guarantees and the \(L_t\equiv1\) clock counterexample.
7. **Ceilings and no-go results.** Include the likelihood-ratio KL
   log-growth ceiling, global-blindness result, conditional-symmetry
   counterexample, localization-identifiability boundary, and physical
   resource ledger.

## Prior art and claim boundary

The final note must explicitly credit:

- Helstrom/Jordan state discrimination and restricted observable norms;
- invariant/group hypothesis testing and Reynolds/group Fourier
  decompositions;
- Page CUSUM, Shiryaev--Roberts procedures, e-processes and e-detectors;
- pairwise betting and exchangeability martingales;
- online mirror descent, expert advice and universal portfolios;
- sparse multisensor/scan/higher-criticism change detection; and
- detector-likelihood, adaptive decoder and QEC noise-tracking work.

Permitted candidate contribution language is limited to a QEC-oriented,
resource-explicit synthesis and its reproducible benchmark. Do not claim the
first paired e-process, sparse scan, group detector, online portfolio,
positive-eigenspace discriminant, quantum change detector or optimal
quickest-change procedure.

## Allowed files

- `references/run6_space_final_theory.md`
- later, after the experimental protocol and results are locked:
  `publication/run6/main.tex` and `publication/run6/references.bib`

The existing first draft and adversarial audit remain immutable evidence:

- `references/run6_symmetry_scan_eprocess_theory.md`
- `references/run6_theory_adversarial_audit.md`

## Out of scope

- altering Runs 1--5;
- asserting exact hardware false-alarm control from approximate symmetry;
- calling archived fixed-syndrome data a faithful classical-shadow eSCD run;
- joining unrelated PNNL side-sensor and Google QEC datasets as a pair;
- claiming natural temporal drift from PNNL calibration-property dates;
- claiming a localization posterior without likelihood calibration;
- quantum speedup, universal sample efficiency, or superiority to a correct
  likelihood ratio, Helstrom/Wilson oracle, or same-feature oracle classifier.

## Verification

- Check every theorem algebraically and with adversarial counterexamples.
- Add deterministic unit tests for capped-simplex selection, predictability,
  sector reconstruction, mixture validity and the clock behavior.
- Validate exact false-alarm statements only on a declared model satisfying
  their assumptions.
- Label Google replay, PNNL cohort shift and simulation separately in every
  result table.
- Verify all manuscript wording from the compiled PDF, including the
  guarantee assumptions and negative/no-advantage conclusion if applicable.

## Acceptance criteria

The theory revision is ready for implementation only when:

1. the global channel cannot be silently removed;
2. every exact guarantee names its filtration and null condition;
3. estimated-reference hardware use is visibly separated from exact theory;
4. the sparse update has a reproducible analytical optimizer and a correctly
   stated regret result;
5. all scan and mixture weights are fixed or predictably updated in a valid
   way; and
6. no performance advantage is stated before a held-out, matched-budget
   experiment passes its locked gate.
