# Run 6 manuscript

This directory contains the pre-results manuscript for:

> *Predictable Sparse and Spectral Contrast Monitoring of Real QEC
> Syndromes: A Predeclared Benchmark with a Disclosed Post-Detector Repair*

The manuscript targets PRX Quantum-style quantum-engineering readership.
`main.tex` selects REVTeX 4.2 automatically when that class is installed and
otherwise uses a compileable one-column preprint fallback. The fallback was
chosen because `revtex4-2.cls` is not present in the current TeX
installation; no scientific content depends on the class.

## Evidence status

The original detector was run under the pre-access freeze.  Its values were
therefore read, scored, and serialized before a producer--consumer validation
mismatch was found.  Detector-derived numeric diagnostics were exposed during
incident diagnosis, but they were not used to choose the source-derived
validator repair.  Decoder outcomes, PNNL held payloads, and completed
randomization replicates remained unavailable; the detector was not rerun or
modified.  The repair has its own implementation--manifest--ratification
chain.  This is not a detector-blind repair or a fully preregistered
end-to-end execution.

After all locked producers and the outcome join had completed, the first
strict publication extraction failed before bundle emission because the
publication consumer expected the PNNL adaptive-state ledger to be an object
although the producer emits an ordered 22-row array.  That automated attempt
validated derived detector/randomization records and loaded the PNNL results
manifest before the schema failure; it had not reached the outcome manifest,
and no performance number was manually inspected to select the repair.  The
repair is nevertheless post-outcome and is not described as outcome-blind or
preregistered.  It is confined to the publication extractor, tests, and
provenance documentation.  No scientific producer, upstream artifact,
threshold, seed, endpoint, gate, outcome, or recorded runtime was changed or
rerun.

The repaired consumer requires exactly 22 adaptive-state rows in locked
cohort/state order and reconstructs every row's path dimension, role count,
five accumulator sizes, adaptive-bank bytes, and formal-accumulator bytes.
Separate publication-provenance records bind the failed attempt, the
allowlisted repair, and unchanged hashes of all eight experimental inputs.
Before a verified bundle exists, every unresolved value appears visibly as
`TBD(KEY)`.

The locked interpretation is:

```text
overall_run6_advantage = google_primary_pass AND pnnl_retention_pass
```

`overall_run6_advantage` is retained only as the frozen schema field name.
Rendered prose and tables call it the locked conjunctive empirical gate; a
true value records satisfaction of those two implementation-specific
conditions and is not a class-wide or oracle superiority claim.

If either Boolean is false, the paper must state:

> no demonstrated S-PACE algorithmic advantage.

A successful exact randomization audit, sparsity, or interpretability cannot
override that rule.

## Build and audit

```bash
make
make audit
make test
```

During the pre-results stage, this command intentionally exits nonzero and
detects the keyed markers in the compiled PDF:

```bash
make check-placeholders
```

No empirical macro is edited manually. The only supported insertion route is
the fail-closed extractor:

```bash
python analysis/extract_results.py \
  --detector-manifest ../../experiments/run6/results/google_detector/detector_freeze_manifest.json \
  --freeze-ratification ../../experiments/run6/freeze_ratification.json \
  --repair-manifest ../../experiments/run6/repair_manifest.json \
  --repair-ratification ../../experiments/run6/repair_ratification.json \
  --randomization-manifest ../../experiments/run6/results/google_randomization_repair1/merged/randomization_manifest.json \
  --pnnl-manifest ../../experiments/run6/results/pnnl_snapshot/results_manifest.json \
  --pittsburgh-manifest ../../experiments/run6/configs/pnnl_pittsburgh_locked.json \
  --outcome-manifest ../../experiments/run6/results/google_outcomes/outcome_manifest.json \
  --output-dir generated
```

It validates the original freeze and separate repair ratification, then
cross-binds all eight inputs: the three provenance records, four result
manifests, and locked Pittsburgh metadata manifest.  The detector must
continue to name only the original ratification; randomization, PNNL,
outcome, and decision records must also name the repair ratification.  The
production validator anchors the three provenance records to hardcoded Git
commits and recomputes every repair implementation digest from its immutable
implementation-commit blob.  The eight input paths are canonical,
repository-relative, every path component must be non-symlinked, and every
resolved file must remain inside the repository root.  Their canonical paths
are recorded together with their hashes.  The
extractor recomputes the atomic predicates and conjunctive claim and emits
the exact tables, figures, evidence contract, claim include, and
`publication_bundle_manifest.json`.

The consumer also closes two value-blind reporting contracts.  For Google it
validates all 2,000 threshold-bootstrap rows (complete-shot circular blocks of
length 128; seeds 613000--614999), recomputes every linear percentile and
frequency summary, and matches every frozen threshold to the detector table.
This is descriptive uncertainty only and cannot alter a locked Boolean.  For
PNNL it validates the exact 256 seeds, five-method order, all 110 ordered
path--state--method rows, and the `<i8` count and `<f8` maximum-log-e arrays
with shape `(256,5)`.  The rows are hash-bound and aggregate-cross-checked:
the arrays recover totals, histograms, crossing identities, and global
extrema, but cannot reconstruct each path's replicate allocation.  This
paired-swap audit is not a natural-hardware null and enters neither empirical
gate nor the overall conjunction.

The resulting `run6-publication-bundle-v5` contract contains 22 files:
fourteen tables, four figures, the generated claim sentence, two manuscript
contracts, and the bundle manifest.

The final gate then performs a fresh extraction in a temporary directory and
requires every generated byte to match before rebuilding. It also requires
every `\RunSixVerified...` contract macro to appear exactly once in a
result-bundle branch of `main.tex`. Before resolving paths, it rejects a
symlink in any component of the supplied bundle directory or manuscript
source, as well as every symlinked bundle entry:

```bash
make final-audit
```

Editing a generated table, figure, claim sentence, bundle manifest, or
manuscript fallback cannot satisfy this target. `make check-placeholders`
must pass and `make expect-placeholders` must fail in final mode.

## Claim boundary

This paper does not claim a quantum algorithm or speedup, universal
sample-efficiency or scaling advantage, superiority over same-feature or
same-parity logistic/threshold classes, Helstrom/a correct same-information
likelihood ratio/oracle features, a new Wilson-loop or toric-code theorem, or
a result in string theory or holography.

No original/redlined baseline exists because this directory starts a new
manuscript. Once empirical values are inserted, preserve this version as the
pre-results baseline before producing a redline.
