# Run 6 publication result extractor

`extract_results.py` is the only supported route from completed Run 6 result
artifacts to manuscript tables and figures. It reads the original freeze
ratification, the separate repair manifest and ratification, four derived
result manifests, and the locked Pittsburgh metadata-only manifest. It never
reads a Google `.b8` or `.01` source or a PNNL `bitstrings.json` payload.

The extractor fails unless:

- all eight evidence inputs have the exact schemas, statuses, commits,
  paths, and recursive cross-hashes;
- the original ratification, repair manifest, and repair ratification match
  their hardcoded immutable Git blobs, and every repair implementation digest
  is recomputed from the hardcoded implementation commit;
- the original ratification binds the immutable detector chain, while the
  repair manifest/ratification bind the exact allowlisted source/schema
  repair, detector registry, failed-attempt chronology, access record,
  environment, and package/runtime provenance;
- randomization, PNNL, outcome, and decision records bind the repair
  ratification, while the detector continues to bind only the original
  ratification;
- every declared derived artifact has the recorded byte count and SHA-256;
- the Google outcome summary is the completed
  `full_run6_locked_decision`;
- the seven Google predicates can be reproduced from the frozen event,
  complete threshold-bootstrap artifact, and primary top-20 risk summaries;
- the Google threshold bootstrap contains all 2,000 ordered rows, exact
  seeds 613000--614999 and block design, detector-frozen thresholds, and
  exactly recomputed NumPy-linear percentiles and frequencies;
- the randomization audit is the complete 256-replicate exact-design audit;
- the PNNL paired-swap JSON has exact seeds/method/cohort order and 110 rows,
  and its `<i8` alarm-count and `<f8` maximum-log-e arrays have exact
  `(256,5)` shape, finite values, crossing identities, totals, histograms,
  and extrema;
- all 110 PNNL path–state–method rows reproduce the macro results and
  two-comparator retention gate, while every cohort ID, basis, distance,
  rounds, calibration pair, and QASM/control class matches the Pittsburgh
  lock; and
- `overall_run6_advantage` equals the conjunction of the Google and PNNL
  gates. This frozen field name is schema provenance, not the rendered claim.

Run it only after both provenance chains and the four real-data stages are
complete:

```bash
python publication/run6/analysis/extract_results.py \
  --detector-manifest experiments/run6/results/google_detector/detector_freeze_manifest.json \
  --freeze-ratification experiments/run6/freeze_ratification.json \
  --repair-manifest experiments/run6/repair_manifest.json \
  --repair-ratification experiments/run6/repair_ratification.json \
  --randomization-manifest experiments/run6/results/google_randomization_repair1/merged/randomization_manifest.json \
  --pnnl-manifest experiments/run6/results/pnnl_snapshot/results_manifest.json \
  --pittsburgh-manifest experiments/run6/configs/pnnl_pittsburgh_locked.json \
  --outcome-manifest experiments/run6/results/google_outcomes/outcome_manifest.json \
  --output-dir publication/run6/generated
```

The output directory must be empty. The command generates:

- fourteen LaTeX tables covering the gate, complete event alert-shot lists,
  complete threshold frontiers and descriptive bootstrap, all 48
  label--budget--method risk rows, all comparator-difference uncertainty,
  exact-design randomization, all 22 PNNL state rows, PNNL
  macro/comparator/control status, all 11 state-averaged cohort rows, all 22
  paired-swap path--state fractions, and joint/process-wide and surrogate
  resources;
- four PDF figures: every pre-event alert plus the event-window zoom,
  both-label risk–coverage with locked intervals, proper-prior-only
  randomization calibration, and the PNNL path-level forest plot;
- `manuscript_artifact_contract.json` and
  `manuscript_artifact_contract.tex`, which provide caption/evidence-class
  requirements and non-colliding `\RunSixVerified...` include macros; and
- a single-sentence claim include;
- `publication_bundle_manifest.json`, which binds each generated file to all
  eight exact role--canonical-path--hash evidence records, records both
  provenance chains and the truthful repair access record, and records the
  exact locked conclusion. Its schema is `run6-publication-bundle-v5`.

The bundle has 22 files in total: fourteen tables, four figures, one claim
include, two manuscript contracts, and the bundle manifest.  PNNL paired-swap
path rows are hash-bound and aggregate-cross-checked: the aggregate arrays
cannot reconstruct each path's replicate allocation.  That audit is not a
natural-hardware null and enters neither empirical gate nor the overall
conjunction.  The Google threshold bootstrap and partial trapezoidal recall
areas are validated descriptive quantities and do not alter a locked Boolean.

The manuscript contract explicitly rejects fields absent from the locked
artifacts: a Google randomization SR summary, per-method timing or memory,
a count of “retained” PNNL paths, an undefined event-score percentile, and a
post-hoc “best contextual method,” or a detector-blind repair claim. Runtime
is reported only for the Google joint detector pipeline or the whole PNNL
arm; peak RSS is process-wide.

The positive sentence says only that both fixed gates were satisfied for the
specified datasets, implementations, endpoints, and budgets. It explicitly
denies superiority over same-feature or same-parity logistic/threshold
classes, a correct likelihood or oracle rule, and general algorithmic
advantage. Any failed predicate emits exactly: **“No demonstrated S-PACE
algorithmic advantage.”**
