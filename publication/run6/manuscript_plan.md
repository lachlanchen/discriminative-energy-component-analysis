# Run 6 manuscript-skeleton plan

**Stage:** pre-results manuscript construction

**Scientific goal:** prepare a rigorous, results-contingent paper for a
high-quality quantum-information or quantum-engineering venue without
opening held payloads or implying an empirical outcome.

## Allowed scope

- Create or edit files only under `publication/run6/`.
- Use the already written Run 6 theory, preregistration, method lock, and
  publication/literature audit as read-only sources.
- State exact claims only under their declared randomized-design or
  conditional-centering assumptions.
- Label the natural Google replay empirical and the PNNL arm a constructed
  boundary between real-hardware snapshot cohorts.
- Represent every unknown empirical quantity by a visible,
  machine-detectable `TBD(KEY)` fallback.

## Forbidden scope

- Do not edit a frozen Run 6 configuration, implementation, reference,
  manifest, ratification, or result.
- Do not open, decode, stream, grep, hash, or otherwise inspect Google
  `.b8`/`.01` values or PNNL bitstrings.
- Do not state or imply superiority to Helstrom, a correct same-information
  likelihood ratio, Wilson diagnostics, or same-feature oracle models.
- Do not claim quantum speedup, universal sample efficiency, a new
  topological-order theorem, string theory, or holography.
- Do not choose the abstract conclusion until the locked two-arm gate is
  evaluated.

## Files

- `main.tex`: stable active manuscript.
- `references.bib`: primary-source bibliography from the literature audit.
- `check_placeholders.py`: audits visible PDF fallback markers.
- `verify_generated_bundle.py`: freshly regenerates and byte-compares the
  hash-bound analysis bundle.
- `analysis/`: fail-closed extractor and tests maintained in the same
  publication scope.
- `Makefile`: build, placeholder audit, PDF text audit, and clean targets.
- `README.md`: stage, scope, build instructions, and result-insertion rules.

No baseline or redline exists because this is a new manuscript, not a
revision. Redline generation is therefore intentionally unavailable.

## Planned paper architecture

1. Results-contingent abstract.
2. Observation-level and claim boundary.
3. Locked problem, data roles, and paired-shot experimental unit.
4. Predictable sparse and spectral contrast construction.
5. Theorems: bounded predictable factor, complete-shot mixture,
   common-mode blind spot, top-\(k\) solution, proper-prior lifetime
   control, SR average-run-length distinction, and likelihood ceiling.
6. Predeclared baselines, endpoints, uncertainty, conjunctive gate, and
   resource accounting.
7. Results section containing only placeholders and fixed interpretation
   logic.
8. Discussion, limitations, reproducibility, and conclusion.

## Acceptance criteria

- `make` produces `main.pdf`.
- LaTeX has no undefined citation/reference warnings and no overfull boxes.
- `make audit` extracts the PDF and confirms that pending markers remain
  visible at this pre-results stage.
- `make check-placeholders` intentionally exits nonzero until a verified
  generated bundle replaces every visible fallback.
- The PDF explicitly distinguishes exact randomized-design validity,
  empirical natural replay, and constructed snapshot boundaries.
- The PDF states that one complete paired shot is one formal update and
  that the 51 roles are simultaneous fixed-prior experts.
- The PDF includes the locked negative conclusion if either empirical gate
  fails and contains no fallback “sparsity advantage.”

## Verification record

- `make`: passed with pdfLaTeX/BibTeX on 28 July 2026.
- `make audit`: passed for the pre-results manuscript.
- Compiled pre-results PDF: 20 pages with 20 visible empirical artifact keys
  and 20 corresponding source fallbacks.
- Citation/reference audit: no undefined citations or references.
- Layout audit: no overfull boxes.
- Extractor and bundle-verifier tests: 44 passed, including negative tests
  for coordinated repair-manifest/ratification fabrication, false
  detector-blind access, a claimed detector rerun, missing repair bindings
  in each downstream stage, a changed evidence path, an overbroad positive
  claim, a tampered Pittsburgh lock, inconsistent cohort metadata, a
  manually edited claim, an unconsumed manuscript-contract macro,
  malformed or cross-artifact-inconsistent PNNL paired-swap JSON/NPY
  records, and malformed or summary-inconsistent Google threshold-bootstrap
  records.
- Static checks: Ruff lint and format checks, Python byte-compilation, and
  `git diff --check` passed.
- The publication contract declares exactly 22 generated artifacts:
  fourteen tables, four figures, one generated claim sentence, two contract
  files, and one bundle manifest under schema
  `run6-publication-bundle-v5`.
- Full positive and negative synthetic integrations generated and freshly
  reproduced the 22-artifact bundle.  The 25-page positive and 24-page
  negative final PDFs contained no pending marker, undefined
  reference/citation, or overfull box; the positive PDF carried the explicit
  class/oracle non-superiority sentence and the negative PDF carried the
  exact locked no-advantage sentence.
- `final-audit` regenerates the bundle into a temporary directory from all
  eight declared evidence inputs---the original freeze ratification, repair
  manifest and ratification, four result manifests, and metadata-only frozen
  Pittsburgh cohort/QASM manifest---and byte-compares every artifact.
  Missing, extra, symlinked, or manually edited generated files therefore
  fail closed.  It also requires every verified contract macro to occur
  exactly once inside a true result branch in `main.tex`.
- Production extraction and `final-audit` expose no validation-profile
  override.  They load three provenance blobs from hardcoded Git commits and
  recompute all 13 allowlisted repair-file hashes from the hardcoded
  implementation commit.  Only private Python test helpers accept the
  immutable synthetic fixture profile.
- Template deviation: `revtex4-2.cls` and `apsrev4-2.bst` are absent from the
  current TeX installation. `main.tex` therefore used its one-column article
  fallback. It will select the PRX Quantum REVTeX 4.2 preprint class
  automatically when the publisher package is installed.
- Bibliography: 28 cited primary papers/data deposits resolve; BibTeX emits
  only expected empty-journal warnings for six explicitly labeled
  preprints.
- Data-boundary audit: this repair-provenance revision opened no real
  detector number, randomization result, decoder outcome, or PNNL held/result
  payload; all integration tests used synthetic fixtures.
- Value-blind completeness audit: the PNNL consumer now validates exact
  seeds/method/cohort order, 110 JSON rows, `<i8` counts and `<f8` maxima
  with shape `(256,5)`, crossing identities, totals, histograms, and extrema.
  The Google consumer validates all 2,000 threshold-bootstrap rows, exact
  seeds and block design, frozen-threshold identity, and every NumPy-linear
  percentile/frequency summary.  The two generated tables and captions
  explicitly remain implementation/descriptive audits outside the locked
  empirical gate.

## Revision unit: hash-bound result integration

**Origin:** post-skeleton publication audit.

**Allowed files:** `publication/run6/` only, including coordination with the
already added `publication/run6/analysis/` extractor.

**Problems to close:**

1. replace the incorrect generic PNNL resampling description with all three
   frozen uncertainty designs: 4,096 within-stream complete-shot block
   bootstraps for thresholds; 10,000 path/snapshot-pair bootstraps after
   averaging logical states; 10,000 calibration-hash-cluster sensitivity
   bootstraps; plus exhaustive \(2^{11}\) paired sign flips;
2. describe the PNNL decision as one aggregate retention gate against both
   named comparators, not a count of ``retained paths'';
3. remove the nonexistent empirical Google SR result;
4. remove per-method timing/RSS placeholders and state only process-wide
   joint-pipeline resource measurements unless the hash-bound extractor
   supplies a supported aggregate table;
5. render each pre-results marker as visible `TBD(KEY)`;
6. replace manual `results_pending.tex` insertion with conditional includes
   from the extractor's exact generated fragment names; and
7. make `final-audit` require a complete hash-bound bundle, revalidate the
   four result manifests plus the frozen Pittsburgh metadata/QASM lock,
   recompute the locked Boolean and claim sentence, and verify every generated
   artifact digest.

**Out of scope:** raw result access, edits to frozen paths, changes to
experimental logic, or independent numerical interpretation.

**Acceptance criteria:**

- the pre-results manuscript still compiles with visible keyed markers;
- extractor tests and a new bundle-verifier test pass;
- `make audit` passes before a bundle exists;
- `make final-audit` fails closed without the bundle, its four bound result
  manifests, and the bound Pittsburgh metadata/QASM lock;
- the manuscript consumes only the extractor's declared tables, figures,
  and claim sentence in final mode;
- no source-edit of a placeholder file can satisfy `final-audit`; and
- the rebuilt PDF has no undefined citations/references or overfull boxes.

## Revision unit: post-detector repair provenance

**Origin:** Run 6 validator-repair ratification after detector execution and
before decoder-outcome or held-PNNL access.

**Allowed files:** `publication/run6/` only.

**Problems to close:**

1. replace the obsolete single-freeze publication contract with a
   fail-closed dual-provenance contract covering the original freeze
   ratification and the separate repair manifest/ratification;
2. require every downstream randomization, PNNL, outcome, and decision record
   to bind the repair ratification in addition to its original provenance;
3. disclose that detector numeric values were exposed during incident
   diagnosis, while recording that they did not select the source/schema-only
   repair;
4. disclose that no decoder outcome, held PNNL payload, completed
   randomization replicate, or randomization shard manifest existed before
   repair, and that detector artifacts were reused without modification or
   rerun;
5. remove any implication that the complete execution remained
   detector-blind or pristine end-to-end preregistration; and
6. preserve all scientific gates, thresholds, states, scores, estimands,
   budgets, and forbidden-claim boundaries.

**Out of scope:** opening real detector values, decoder outcomes, held PNNL
payloads, completed randomization results, or filling any empirical
`TBD(KEY)` marker.

**Acceptance criteria:**

- the extractor accepts exactly eight distinct evidence inputs and validates
  both provenance chains recursively;
- exact repair scope, access record, environment, failure chronology, and
  downstream bindings fail closed under mutation;
- synthetic positive and negative tests pass without reading real result
  payloads;
- the pre-results PDF states the chronology and limitations truthfully,
  compiles without undefined references/citations or overfull boxes, and
  retains visible pending markers; and
- `make check-placeholders` fails as expected while `make audit` and
  `make expect-placeholders` pass.

## Revision unit: adversarial provenance and claim audit

**Origin:** independent NO-GO audit of the first dual-provenance extractor.

**Allowed files:** `publication/run6/` only.

**Problems to close:**

1. prevent a mutually edited repair manifest and ratification from fabricating
   chronology, environment, package-lock, runtime-origin, or implementation
   evidence;
2. ground every repair implementation digest in immutable Git blobs at the
   fixed implementation commit, while retaining a hermetic internal fixture
   profile that is unavailable from the production CLI;
3. require the exact failed-attempt root, 32 shard ranges, 32 empty-result
   paths, 64 stdout/stderr records and digests, fixed common stderr digest,
   exact four one-thread settings, original environment equality, Python-lock
   binding, and runtime-module origins;
4. record and verify each evidence input as an exact
   role--canonical-path--SHA tuple rather than only role--SHA;
5. use the repaired randomization output directory as the Makefile default;
   and
6. ensure that a positive conjunction says only that both fixed empirical
   gates were satisfied for the named data, implementations, endpoints, and
   budgets, explicitly denying same-feature/same-parity threshold, logistic,
   class-wide, or oracle superiority.

**Out of scope:** real detector/randomization/outcome/PNNL result access,
changes to experimental records, or empirical interpretation.

**Acceptance criteria:**

- a concrete coordinated-manifest/ratification fabrication fails;
- production validation has no flag or CLI route to bypass immutable anchors;
- synthetic positive and negative bundles remain hermetic;
- the bundle verifier checks exact evidence roles, paths, and hashes;
- positive and negative final PDFs compile with the locked claim language;
  and
- all publication tests, formatting, lint, byte-compilation, pre-results
  placeholder checks, and PDF audits pass.

## Revision unit: value-blind randomization and bootstrap completeness

**Origin:** pre-outcome publication-completeness audit of the locked result
contracts.

**Allowed files:** `publication/run6/` only.  Frozen configurations, producer
code, and producer tests may be read to recover their declared schemas.
Synthetic fixtures may be created under the publication test tree.

**Problems to close:**

1. independently validate the PNNL paired-swap randomization JSON and its
   two hash-bound NumPy arrays, including exact seeds, method and cohort
   ordering, all 110 path--state--method rows, array dtype/shape/finiteness,
   and every recomputable fraction, histogram, and extrema summary;
2. return that validated object from the PNNL manifest validator and render
   a compact, complete five-method descriptive table while retaining
   validation of every unrendered row;
3. independently validate the Google 2,000-replicate, block-length-128
   threshold bootstrap artifact, including exact seeds and all internally
   recomputable summaries; and
4. render its fixed descriptive uncertainty summary without changing or
   entering any empirical gate;
5. render all 22 ordered PNNL path--state rows and five method fractions,
   not only the aggregate method summaries;
6. render the full Google risk contract---both labels, all three budgets,
   and all eight methods---with every available point estimate, interval,
   and valid-replicate count; and
7. render both PNNL comparison uncertainty designs and the exact sign-flip
   \(p\)-value alongside the three locked comparison Booleans; and
8. render all 11 state-averaged PNNL cohort rows for all five methods, so
   the frozen reporting requirement is literal rather than implicit in the
   two comparator-difference forest panels.

**Claim boundary:** both additions are implementation/descriptive audits.
The PNNL paired-swap audit is not a natural-hardware null and the Google
bootstrap is descriptive threshold uncertainty only.  Neither changes or
enters either empirical gate or the overall conjunction, and neither supports
class-wide, oracle, computational, sample-efficiency, quantum, topological,
string-theory, or holographic claims.

**Out of scope:** reading any path under `experiments/run6/results/`, opening
held payloads or decoder outcomes, changing producer artifacts, changing a
locked gate, or selecting claims from real values.

**Acceptance criteria:**

- malformed JSON, wrong order/seeds/schema, wrong NumPy dtype/shape,
  non-finite values, cross-artifact inconsistencies, and summary tampering
  fail closed;
- both new tables are declared in the artifact contract and consumed
  exactly once in the manuscript result branch, with visible keyed
  pre-results fallbacks;
- the generated PNNL randomization fragment contains 22 ordered path--state
  rows plus five complete method-summary rows, the Google risk table
  contains all \(2\times3\times8=48\) cells, and the PNNL comparison table
  exposes path-bootstrap, calibration-pair sensitivity, and exact sign-flip
  results; the separate PNNL cohort table contains all \(11\times5\)
  state-averaged method cells;
- synthetic positive and negative final PDFs contain both tables and their
  claim-boundary captions;
- publication tests, Ruff lint/format, Python byte-compilation,
  `git diff --check`, pre-results audits, and both synthetic final builds
  pass; and
- no real Run 6 numerical result file is inspected during this revision.

## Revision unit: post-outcome publication-consumer schema repair

**Origin:** the first strict production extraction after all scientific
producers and the outcome join had completed.

**Observed failure:** the extractor stopped before bundle emission and before
reaching the outcome manifest because it required the PNNL
`adaptive_state_ledger` to be an object, while the frozen producer emits an
ordered list of 22 per-cohort/per-state records.

**Allowed scope:** the publication extractor and its synthetic tests,
manuscript disclosure, and separate publication-repair provenance records.
No scientific producer, result manifest, experimental gate, threshold, seed,
endpoint, outcome, or recorded runtime may be changed or rerun.

**Repair contract:**

1. validate exactly 22 ledger rows in Pittsburgh cohort order and logical
   state order \(0,1\);
2. require an exact seven-field row schema;
3. derive \(q=d-1\) and the role count from locked cohort metadata;
4. reconstruct the DFR, online-logistic, sparse, spectral, and composite
   accumulator dimensions from the frozen bank design;
5. reconstruct `DimensionAdaptedBank.state_nbytes()` and the three-array
   formal-accumulator byte count independently; and
6. preserve the failed extraction and bind the allowlisted publication-only
   diff plus unchanged hashes of all eight experimental evidence inputs in a
   separate post-outcome provenance chain.

**Claim boundary:** this is a reporting-pipeline repair, not new experimental
evidence.  Outcome production preceded it, so neither the repair nor the
final execution is described as outcome-blind or preregistered.  The repair
cannot change either empirical gate or strengthen any algorithmic,
computational, quantum, topological, string-theory, or holographic claim.

**Acceptance criteria:**

- object/list confusion, wrong row count/order/schema, wrong \(q\), wrong
  role count, any component-count change, and either byte-identity change
  fail closed;
- all eight upstream experimental input hashes are unchanged;
- the canonical outcome output is not rerun;
- the publication-repair manifest and ratification bind the historical
  implementation commit, incident record, allowlisted diff, access truth,
  and upstream hashes;
- a fresh extraction and final PDF audit pass only through the repaired
  consumer; and
- the paper discloses the post-outcome chronology without outcome-blind
  language.
