# Run 6 runner integration audit

Date: 2026-07-27
Scope: metadata- and synthetic-only integration audit of the Google detector,
Google randomization, Google outcome, Pittsburgh snapshot, and Pittsburgh
metadata-validation runners.

## Verdict

**BLOCK: do not decode, score, parse, or otherwise unblind any real Run 6
payload yet.**

The complete-shot statistical implementation is internally consistent, the
Google threshold-stage boundary is now exact, and the PNNL retention Boolean is
now independently recomputed by the final Google outcome runner. The remaining
hard blockers are freeze provenance, the impossible pending/frozen Pittsburgh
validator transition, and a held-data policy/code contradiction. These are
pre-data contract defects, not evidence against the algorithm.

This audit did **not** open or hash:

- Google `detection_events.b8`;
- any Google `.01` outcome file;
- any real PNNL `bitstrings.json`.

The PNNL metadata validator statted the 20 locked `bitstrings.json` files only,
as allowed by the preregistration.

## Evidence that passed

- The initial 118-test Run 6 metadata/synthetic suite passed. After the
  downstream exact-schema tests were expanded during this audit, the latest
  run was 123 passed and one non-semantic assertion-message mismatch:
  `test_threshold_frontier_sidecars_match_exact_producer_schema` expects
  `keys differ`, while the shared strict validator correctly reports
  `schema mismatch; missing=[], unknown=['unexpected']`. This test wording
  must be synchronized before freeze.
- The focused detector/randomization/outcome/PNNL runner tests passed.
- All four runner CLIs passed their synthetic `--dry-run` and reported
  `raw_run6_values_opened=false`.
- The real Pittsburgh metadata lock passed without payload reads or hashes:
  20 snapshots, 11 cohorts, 20 held files statted, 16,370 paired shots and
  49,110 cycle updates per pre/post phase, and 110 unique threshold seeds.
- `python_environment_lock.txt` matched the current `pip freeze --all` output
  after excluding its explanatory comments and the separately bound editable
  repository line.
- Python compilation passed for all five runners and the shared Run 6 modules.
  Ruff presently also reports two unused imports (`subprocess` and
  `PurePosixPath`) in `run6_protocol.py`, apparently reserved for the
  not-yet-implemented strong freeze helper.

## Integration findings

| Area | Status | Finding |
| --- | --- | --- |
| Google formal time | PASS | Held horizon is 20,000 complete paired shots; randomized horizon is 5,000. The 51 round roles are simultaneous experts, not sequential time points. |
| Google role prior | PASS | Each exact method tiles its fixed base-component prior with mass `1/51` per role; within-shot factor compounding is disabled. |
| Google threshold frontier | PASS | The persisted frontier is `[-inf] + sorted unique cycle scores + [+inf]`; alert counts are recomputed from strict shot maxima and are monotone. |
| Threshold/held state separation | PASS | Threshold and held replays clone the same post-warm-up checkpoint; threshold state is not carried into held replay. |
| Threshold-stage artifact boundary | PASS | Both downstream consumers now require the exact set of threshold arrays, sidecars, frontiers, `thresholds.json`, and `threshold_shots.csv`, and reject held artifacts in the stage manifest. |
| Google randomization | PASS | One PCG64 orientation bit is shared by all 51 roles of each shot, the warm checkpoint is restored per replicate, and the proper-prior process updates once per complete shot. |
| PNNL formal time and priors | PASS | Each path-state episode uses one update per complete paired shot and a uniform round-role prior. Vectorized bootstrap results match the scalar reference in tests. |
| PNNL retention aggregation | PASS | Producer aggregation averages states 1/2–1/2, then paths 1/11–1/11, and uses total pre-false-alarm state counts. The final outcome runner now independently reconstructs the Boolean from all 110 canonical state rows. |
| PNNL producer/consumer keys | PASS | Genuine result fields `package_lock_sha256`, `held_value_processing_started_unix`, and nested `package_lock` are now included in the exact consumer schemas and verified. |
| Metadata embargo | PASS | The Pittsburgh validator mechanically refuses to open/hash `bitstrings.json` and uses `lstat/stat` size only. |
| Freeze provenance | **BLOCK** | Current gates validate current files and ancestry only; they do not bind each hash to the corresponding Git blob in `freeze_commit`, prove the freeze/ratification is pushed, or parse the freeze manifest contract. |
| Pittsburgh pending-to-frozen transition | **BLOCK** | The validator requires the pending status while the real runner requires the frozen status, so a correctly frozen manifest cannot pass `validate_lock`. |
| PNNL final-data policy | **BLOCK** | The lock says data-register values remain unopened until a separate risk audit, but `load_snapshot_events` fully parses and validates all `c_data_*` binary values. |
| Package lock semantics | FIX BEFORE UNBLINDING | The lock file and hashes are recorded, but installed distributions are not mechanically compared against the full lock. |
| Formal diagnostic sidecars | FIX BEFORE UNBLINDING | M0C/M2 formal arrays are correctly NaN/inapplicable, but their sidecars currently label those arrays as diagnostic formal statistics. |
| Portable artifact records | FIX BEFORE PUBLICATION | PNNL result artifact records use absolute paths, unlike the relative Google records. |

## Hard blocker 1: strong freeze provenance is not implemented

The detector, randomization, outcome, and PNNL gates presently establish that:

1. `freeze_commit` is an ancestor of the current `HEAD`; and
2. current filesystem bytes equal the hash map in the supplied ratification.

They do **not** establish that the ratified hash for each path is the SHA-256 of
`git show <freeze_commit>:<path>`. Consequently, a ratification can name an old
ancestor while authorizing bytes that were never committed in that ancestor.
They also do not prove that the freeze and ratification commits are contained
in `origin/main`, or that ratified paths are free of staged/unstaged changes.

The Google gates require a current hash for
`experiments/run6/freeze_manifest.json` but do not parse or validate its exact
schema, implementation commit, status, hash registry, environment, thread
settings, or pre-freeze access flags. The per-runner required-path sets also
differ.

### Required patch

Implement one shared helper in `experiments/aoc/run6_protocol.py` and call it
from every real runner. It must:

1. load the freeze manifest and ratification with exact-key schemas;
2. require both access flags to be false and both statuses to be final;
3. require the implementation commit to be an ancestor of `freeze_commit`,
   and `freeze_commit` to be an ancestor of current `HEAD`;
4. for every ratified path, compute SHA-256 over the exact Git blob from
   `freeze_commit` and require equality with both the ratification hash and the
   current file hash;
5. reject staged or unstaged changes to every ratified path;
6. require `freeze_commit` and the commit containing the unchanged
   `freeze_ratification.json` to be ancestors of `origin/main`;
7. validate the freeze-manifest hash registry, rather than merely hash the
   manifest file;
8. require the same union of result-affecting paths in every runner.

The required union should contain, at minimum:

- `experiments/aoc/{__init__,qec_real,run6_protocol,space,space_qec}.py`;
- `experiments/pyproject.toml`;
- all five Run 6 scripts;
- both Run 6 top-level configs, the Pittsburgh manifest, environment lock,
  deviation ledger, and freeze manifest;
- the Google method specification, Google preregistration, PNNL auxiliary
  specification, and every path in Pittsburgh `parent_artifacts`.

Do not use a self-referential hash. Use this acyclic dependency graph:

```text
implementation/config/spec files
        ↓
freeze_manifest (hashes the files above, never itself)
        ↓ committed as freeze_commit
freeze_ratification (names freeze_commit and hashes freeze_manifest + files,
                     never itself)
        ↓ committed and pushed
result manifests (hash freeze_ratification and upstream result manifests)
```

### Required tests

- A current file matching the ratification but differing from
  `git show freeze_commit:path` must fail.
- A staged or unstaged change to any ratified path must fail.
- A freeze commit absent from `origin/main` must fail.
- A ratification file not committed/pushed unchanged must fail.
- Missing/unknown freeze-manifest fields and any embedded hash mismatch must
  fail.
- A self-hash entry for the freeze manifest or ratification must fail.
- Every runner must reject omission of any member of the shared path union.

## Hard blocker 2: Pittsburgh cannot transition to its final status

`run_pnnl_snapshot.run_real` first calls `validate_lock`, then requires:

```text
pnnl_snapshot_locked.status == frozen_before_held_value_access
pnnl_pittsburgh_locked.status == frozen_before_held_value_access
```

But `validate_pnnl_lock._validate_static_contract` currently requires the
Pittsburgh status to equal:

```text
ratified_metadata_lock_pending_final_freeze_commit
```

The static-contract digest also includes the status and `parent_artifacts`.
Therefore changing the manifest to the status required by the runner makes the
validator fail before the runner reaches its explicit frozen-status check.

Two Pittsburgh parent hashes are also currently stale:

- `experiments/run6/configs/google2022_locked.json`;
- `references/run6_real_qec_preregistered_plan.md`.

### Required patch

1. Make the metadata validator explicitly support a pre-freeze pending mode and
   a real-run frozen mode; `run_real` must request frozen mode.
2. Keep one explicit final static-contract digest (or two explicit
   pending/frozen digests). Do not silently exclude arbitrary contract fields.
3. Refresh every parent hash only after all parent files are final.
4. Recompute the final static digest and lock it in the validator.
5. Add a test that changes pending to frozen and proves the frozen validator
   succeeds, while any third status fails.

## Hard blocker 3: PNNL final-data policy contradicts execution

The Pittsburgh manifest says:

> validate shape only; final-data values remain unavailable to detectors and
> may be opened only in a separately frozen downstream logical-risk audit

However, after unblinding, `load_snapshot_events` calls
`_strict_binary_matrix` for every selected `c_data_*` register. This
materializes and validates every data bit's type/domain, even though the matrix
is discarded and never enters a detector.

Strict parsing is defensible, but the present wording is false.

### Required patch

Prefer an honest contract:

> data-register values are loaded as an unavoidable part of the joint JSON
> payload and are strictly validated for type, binary domain, and shape, then
> discarded; they are never supplied to feature construction, detector
> updates, cohort selection, thresholds, or the Run 6 retention decision, and
> no logical-risk result is computed.

Update the validator's static-contract digest after this wording change. Add a
test showing that perturbing `c_data` changes no detector events/scores while
invalid type/domain/shape still fails.

## Required integration fixes before unblinding

### Full package-lock enforcement

The package-lock bytes are now hash-bound correctly, and the checked-in lock
matches the current environment. Still, `environment_fingerprint()` records
only a subset of distributions. Add a strict package-lock parser that:

- accepts only unique normalized `name==version` pins after comments/blanks;
- compares every pin with `importlib.metadata`;
- rejects missing, mismatched, and unexpected distributions except the
  explicitly excluded editable repository package;
- runs before any real source payload is opened.

Test a changed version, missing package, duplicate normalized name, unexpected
package, and the one allowed editable-project exception.

### Formal-array semantics

Google correctly uses shot-indexed formal traces only for
`m0,m1,m3,m4,m5,space`; M0C and M2 are inapplicable and carry NaNs. Change
their sidecar `formal_claim_scope` to
`not_applicable_no_formal_accumulator`, not the natural-hardware diagnostic
label used for exact methods.

Downstream validation should additionally require:

- finite length-20,000 `log_eprocess`/`log_sr` arrays for exact methods;
- all-NaN formal traces and all-false crossing masks only for M0C/M2;
- exact expert counts, base priors summing to one, uniform role mass, and no
  within-shot compounding in `formal_component_summary.json`.

Also independently recompute the required secondary zero-alert event summary,
which is currently hash-frozen and required but not semantically rechecked.

### Portable recursive artifacts

Change PNNL `_artifact_record` to store paths relative to the results-manifest
directory. Then require exactly:

- one unblinding record;
- one 110-row state table;
- one aggregate file and three randomization artifacts;
- 110 trace arrays and 110 bootstrap-maxima arrays with unique locked stems.

Validate little-endian dtype and the cohort-specific complete-shot horizon of
every array. Absolute build-machine paths should not enter the publication
artifact.

## Release gate

Run 6 may be unblinded only after all of the following are true:

1. the three hard blockers above are fixed;
2. package/formal semantic checks are implemented;
3. all placeholder hashes and pending statuses are replaced;
4. Pittsburgh parent hashes and final static digest agree;
5. the shared freeze helper passes Git-blob, clean-tree, and pushed-commit
   checks in every runner;
6. `ruff`, Python compilation, and the complete synthetic/metadata suite pass;
7. freeze manifest, freeze commit, ratification, and ratification commit are
   created and pushed in that order.

At the audited state, no result exists that can establish an advantage for
S-PACE/ECA. The correct conclusion remains **not yet run**, rather than
advantage or no advantage.

---

## Closure re-audit — 2026-07-28

### Updated verdict

**IMPLEMENTATION GO: no remaining implementation blocker was found before the
controlled freeze-chain step.**

This is not yet an authorization to open a real Run 6 payload. Real
unblinding remains blocked until the already prepared implementation is
committed and pushed, the freeze manifest is generated from that pushed
implementation commit and itself committed and pushed, the ratification is
generated from that pushed freeze commit and itself committed and pushed, and
the common gate verifies the complete chain.

The absent `freeze_manifest.json` and `freeze_ratification.json` are therefore
the expected products of the next controlled step, not defects in the current
implementation. No empirical advantage result exists at this point.

This closure re-audit did not open, hash, decode, grep, sample, or summarize
any real Google `.b8` value, any real Google `.01` value, or any real PNNL
`bitstrings.json` value. The PNNL validator performed the preregistered
metadata/QASM checks and `stat`-only checks of the 20 held files.

### Final verification evidence

- The complete Run 6 suite passed: **137 passed**.
- Ruff, Python compilation, `pip check`, and `git diff --check` all passed.
- Detector, randomization, launcher, outcome, and PNNL synthetic dry-runs all
  passed and reported that no raw Run 6 values were opened.
- The frozen PNNL metadata lock passed with 20 snapshots, 11 cohorts, 20 held
  files statted only, 16,370 paired shots per phase, 49,110 paired cycle
  updates per phase, and 110 threshold seeds.
- The exact installed-environment comparison passed for all 52 locked
  distributions, with only the separately Git-bound editable repository
  handled by the explicit exception.
- All 62 paths in `RUN6_REQUIRED_FREEZE_PATHS` now exist.
- The Google and PNNL statuses are final, and the Google method-spec, PNNL
  Pittsburgh-manifest, and PNNL auxiliary-spec hashes agree with current
  bytes.
- A temporary value-free Git repository exercised the complete creator path:
  pushed implementation commit, manifest creation, pushed freeze commit,
  ratification creation, pushed ratification commit, and successful shared
  chain verification.
- A value-free transition probe validated the same Pittsburgh contract in
  both explicit `pending` and `frozen` modes and confirmed that the wrong mode
  is rejected.

### B1–B6 disposition

| Blocker | Closure result | Re-audit evidence |
| --- | --- | --- |
| B1 — weak freeze provenance | **GO** | `verify_committed_freeze_chain` validates canonical paths, exact schemas and false pre-access flags; implementation → freeze → current/pushed ancestry; current bytes against implementation and freeze Git blobs; the ratification against the pushed `HEAD` blob; the shared required-path union; tracked/untracked runtime Python; exact environment/thread settings; the full package lock; and runtime module origins. Every real runner and the launcher calls it. The acyclic creator also passed a synthetic end-to-end Git test. |
| B2 — impossible Pittsburgh status transition | **GO** | The metadata validator has explicit `pending` and `frozen` modes with separate static-contract digests. The real runner requests `frozen`; a value-free copy passed in pending mode after only the declared status transition, while a mode mismatch failed. The current frozen lock also passes. |
| B3 — false `c_data` policy | **GO** | The contract now states that `c_data` values are parsed only for strict structure, binary-domain, and shape validation and are then discarded. A regression flips every valid synthetic final-data bit while holding syndrome bits fixed and obtains identical detector inputs and downstream endpoints; malformed data still fails. |
| B4 — future Google outcomes decoded | **GO** | The streaming `.01` reader interprets only records `[40000,60000)`. It counts record boundaries outside that interval without converting their payloads to values. Tests place invalid and non-UTF-8 bytes outside the selected interval and confirm they remain uninterpreted. Full-file raw hashing remains only an integrity binding after the freeze gate. |
| B5 — non-resumable randomization | **GO** | The 256 fixed seeds are split into 32 gap-free shards of eight. Every row binds its replicate index, seed, one complete-shot swap vector, restored checkpoint, update counts, and expert counts. Merge rejects gaps, overlaps, duplicates, seed changes, and checkpoint changes and produces the same canonical result across shard layouts. A fresh shard receives a pristine child output directory; logs remain outside it until atomic promotion. Completed shards are reused and failed attempts are preserved. The launcher separately records the configured concurrency cap, thread-safe observed peak running subprocesses, and executed/reused shard counts. Per-shard, merge, and orchestration wall/RSS/update/byte evidence remain separated. |
| B6 — asserted fairness predicate | **GO** | The predicate is derived from exact per-method detector shapes/counts, the archive-shot vector, exact label coverage, and one shared serialized outcome-label bundle. Extra detector records or an invalid label bundle force the atomic predicate and Google primary gate to fail. |

### Additional contract closures

#### Formal arrays and summaries

Both downstream consumers now enforce the formal-data contract rather than
only trusting hashes:

- exact methods `m0`, `m1`, `m3`, `m4`, `m5`, and `space` require finite,
  shot-indexed e-process and SR traces;
- nonformal `m0c` and `m2` require all-NaN traces, all-false crossing masks,
  and `not_applicable_no_formal_accumulator` sidecar scope;
- sidecars and nested component-summary rows have exact schemas;
- base-component IDs and weights, expert counts, role-major flattening,
  uniform role mass, total prior mass, e/SR thresholds, first crossings, and
  the prohibition on within-shot factor compounding are independently
  checked; and
- declared and observed factor bounds, finite/nonnegative flags, and the
  bound-satisfaction Boolean are checked, including a negative tampering
  regression.

The primary and secondary detector event summaries and the expanded
threshold/held shot tables are also recomputed from the frozen arrays.

#### PNNL exact portable artifact contract

The final consumer requires exactly **226** result artifacts excluding the
results manifest:

- six scalar artifacts, including the unblinding record, 110-row state table,
  aggregate, and three randomization artifacts;
- 110 locked trace arrays; and
- 110 locked bootstrap-maxima arrays.

Artifact paths must be canonical relative POSIX paths; absolute paths,
traversal, noncanonical forms, duplicates, missing names, and unknown names
fail. Locked stems, cohort-specific horizons, shapes, little-endian dtypes,
finite/NaN policies, and randomization-array dimensions are checked. The
retention Boolean is independently reconstructed from the exact 11 × 2 × 5
state-row coverage and must agree with both the aggregate and result manifest.

#### Package and source identity

The environment lock is a full installed-distribution equality check, not a
short version fingerprint. Normalized duplicate pins, missing packages,
version mismatches, unexpected packages, and absence of the one declared
editable distribution all fail. Frozen worktree bytes must also equal their
implementation-commit and freeze-commit Git blobs, and runtime shared modules
must resolve to this repository.

### Final GO/BLOCK list

**GO**

1. Finalize the implementation commit containing this closure record and all
   62 required freeze paths, then push it.
2. Run the value-blind freeze creator for the manifest, commit and push that
   manifest, then create the ratification, commit and push it.
3. Invoke the shared gate from a real runner and require it to pass before the
   first payload access.
4. After that gate passes, execute the detector-only Google replay, fixed
   randomization shards/merge, independent PNNL arm, and finally the authorized
   Google outcome slice in the preregistered order.

**BLOCK**

- **No implementation BLOCK remains.**
- Real payload access is still procedurally blocked until the controlled
  commit/manifest/ratification chain above exists and verifies.
- Any claim that the algorithm has an advantage is blocked until both held
  arms have actually run and the locked conjunction passes. A failed gate
  requires the conclusion **“no demonstrated S-PACE algorithmic advantage.”**
- Even a positive narrow result cannot support superiority to Helstrom,
  Wilson or same-parity oracle methods; quantum acceleration; universal
  sample efficiency or scalable computational advantage; a new Wilson-loop
  or toric-code theorem; string theory; or holographic duality.
