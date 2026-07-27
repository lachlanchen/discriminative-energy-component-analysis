# Run 6 final pre-data audit

**Audit snapshot:** 2026-07-27 23:51 Asia/Hong_Kong
**Verdict:** **BLOCK — do not open any held Google detector/outcome values or
PNNL `bitstrings.json` values yet.**

## Embargo statement

This audit did not open, parse, grep, sample, hash, or statistically inspect:

- Google `detection_events.b8`;
- any Google `.01` outcome file; or
- any PNNL `bitstrings.json`.

The PNNL metadata validator was run only in its documented `stat`-only mode.
Repository code/configuration, synthetic fixtures, Git metadata, file sizes,
and hashes of non-payload specifications/configurations were inspected.
No detector result, unblinding record, randomization result, outcome table, or
aggregate result exists in the repository at this snapshot.

## What is ready

The scientific core is substantially stronger than the earlier draft:

1. The formal unit is now one complete paired shot. The 51 round roles are
   role-component experts with a fixed uniform role prior, not 51 serial
   e-process updates sharing one randomization coin.
2. The proper-prior e-process, mixture SR recursion, sparse top-\(k\) witness,
   spectral effect, deterministic tie rules, and score-before-update rule are
   mathematically coherent. Boundary tests now distinguish exact `>=`
   e/SR thresholds from strict `>` empirical thresholds.
3. Natural-hardware e/SR traces are explicitly diagnostic. Exact validity is
   claimed only for the complete-pair randomization design.
4. The Google parser binds the extracted circuit/detector files to exact ZIP
   members. Its role permutation, checkpoint cloning, phase separation,
   threshold frontiers, threshold-stage freeze, primary/secondary event
   summaries, full shot tables, cooldown masks, component IDs/priors/bounds,
   repeated replay digests, and exact detector artifact set are implemented.
5. The Google lock rejects unknown nested fields. The detector, randomization,
   outcome, and PNNL paths use strict JSON and hash-verified artifacts.
6. The PNNL core runner has a strict parser, value-blind metadata selection,
   one formal update per paired shot, fixed cohorts, full calibration and
   randomization, a detailed resource ledger, and a first-unblinding record.
   The exact Python environment lock matches the current `pip freeze --all`,
   and `pip check` reports no broken requirements.
7. Claim boundaries are appropriate: no Helstrom/Wilson/oracle superiority,
   quantum speedup, universal sample-efficiency/scalability, new toric-code
   theorem, string-theory result, or holographic-duality result is authorized.

Verification at this snapshot:

```text
Run 6 synthetic/unit tests: 121 passed in 4.86 s
Ruff:                       all checks passed
git diff --check:           passed
PNNL metadata validation:   VALID
                             20 snapshots, 11 cohorts,
                             20 held files statted only
```

Passing synthetic tests do not ratify a freeze and are not experimental
results.

## Blocking findings

### B1 — The committed freeze does not exist, and the stronger gate is not wired into runners

Severity: **fatal pre-access blocker**.

Current state:

- `experiments/run6/freeze_manifest.json` is absent.
- `experiments/run6/freeze_ratification.json` is absent.
- Google, PNNL snapshot, and Pittsburgh statuses remain pending.
- Normative SHA fields remain `TO_BE_FILLED_AFTER_FINAL_NO_HELD_REAUDIT`.
- Run 6 implementation/reference files are untracked or modified; `HEAD`
  remains the Run 5 publication commit `5947ad84fa867474481163b7e660d47d7256ac07`.

The new
`experiments/aoc/run6_protocol.py:663` `verify_committed_freeze_chain`
correctly adds Git-blob verification, canonical freeze files, pushed
`origin/main` containment, a non-self-referential manifest/ratification DAG,
and environment equality. At this snapshot no real runner calls it. The
legacy gates at:

- `experiments/run6/scripts/run_google2022_detector.py:226`;
- `experiments/run6/scripts/run_google2022_randomization.py:228`;
- `experiments/run6/scripts/run_google2022_outcomes.py:350`; and
- `experiments/run6/scripts/run_pnnl_snapshot.py:342`

still prove only “ancestor commit + current worktree hashes.” That does not by
itself prove that the executed bytes are blobs in the stated freeze commit or
that the freeze was pushed.

**Required repair and acceptance test**

1. Make all four real-data gates delegate to
   `verify_committed_freeze_chain` before any payload hashing/parsing.
2. Include every executable transitive import in the manifest, or additionally
   require a clean tracked worktree and no untracked Python under the runtime
   package/script roots. `aoc/__init__.py` imports modules beyond the four Run
   6 modules.
3. Add end-to-end tests rejecting:
   - an uncommitted but hash-matching worktree edit;
   - a ratified hash that differs from `git show <commit>:<path>`;
   - an unpushed commit;
   - an omitted transitive source;
   - a malformed or incomplete freeze manifest; and
   - a ratification not tracked at the canonical path.
4. Complete the final implementation commit, freeze-manifest commit,
   ratification commit, and push before invoking any real runner.

### B2 — PNNL cannot transition from pending metadata lock to executable frozen lock

Severity: **fatal PNNL execution blocker**.

`experiments/run6/scripts/run_pnnl_snapshot.py:2207-2211` requires both the
snapshot config and Pittsburgh manifest to have status
`frozen_before_held_value_access`. In contrast,
`experiments/run6/scripts/validate_pnnl_lock.py:801-805` requires the
Pittsburgh status to remain
`ratified_metadata_lock_pending_final_freeze_commit`. The static contract hash
at `validate_pnnl_lock.py:386-393,1580-1594` also binds status and
`parent_artifacts`, so merely editing the JSON status cannot pass validation.

Two Pittsburgh parent hashes are stale:

| Parent | Manifest value | Current pre-fix value |
|---|---|---|
| `experiments/run6/configs/google2022_locked.json` | `502be5b8aadf906ca65d02afedcc37daee86dc257c357b7844241954ff82b25d` | `aead2c847062ab0b4891b4c5d7bb23e584c6aa7f6252c118291fdb4ca4f1755c` |
| `references/run6_real_qec_preregistered_plan.md` | `9253376a7f79b71c81c94485de723a03faccc048598a67a56356d4d61febfdce` | `6b6fbb6d736d4ee90fd4b18809b721276b163d4d8ca790fc565186656fe6c212` |

The pending fields are visible at:

- `experiments/run6/configs/google2022_locked.json:3-7`;
- `experiments/run6/configs/pnnl_pittsburgh_locked.json:3,16-21`; and
- `experiments/run6/configs/pnnl_snapshot_locked.json:3,17-23`.

**Required repair and acceptance test**

Finalize in dependency order: method/spec/code → final Google config and hash
→ Pittsburgh parent hashes/status/static digest → Pittsburgh hash → PNNL
snapshot embedded hashes/status → implementation commit. The metadata
validator must pass in final frozen mode, and the real PNNL runner must accept
that same immutable manifest without a post-validation edit.

### B3 — The PNNL final-data policy contradicts the parser

Severity: **fatal audit-trail wording/implementation mismatch**.

`experiments/run6/configs/pnnl_pittsburgh_locked.json:40` says that `c_data`
registers are used for shape validation only and their values may be opened
only in a separately frozen downstream logical-risk audit. However,
`experiments/run6/scripts/run_pnnl_snapshot.py:1937-1939` materializes the
entire JSON payload, and lines `1971-1987` materialize every selected
`c_data_*` matrix and validate every value's binary domain.

The values do not enter detector scores, but “remain unavailable” and
“validate shape only” are factually false.

**Required repair and acceptance test**

Either:

- amend the frozen policy to say that, after general payload unblinding,
  `c_data` is parsed only for locked shape/domain validation, discarded, and
  never supplied to detector, retention, or logical-risk metrics; or
- implement a streaming parser that skips final-data values and proves they
  are never materialized.

Add a test demonstrating that changing only `c_data` cannot change any score,
threshold, alarm, uncertainty result, or retention Boolean.

### B4 — The outcome command decodes the nominally untouched Google future

Severity: **high; blocks preservation of a future replication set**.

`experiments/aoc/qec_real.py:445-459` reads and converts every line of a
`.01` file. The outcome runner invokes it for all 500,000 rows at
`experiments/run6/scripts/run_google2022_outcomes.py:1908-1924` and only then
slices `[40000,60000)`. Thus `[60000,500000)`, still named
`untouched_future`, is decoded unnecessarily.

This occurs after the detector table is frozen, so it does not alter the
current detector gate, but the future block is no longer honestly untouched.

**Required repair and acceptance test**

Use a binary streaming slice parser that returns and validates values only for
`[40000,60000)`. Outside the slice, count record boundaries without decoding
values; raw source integrity is already supplied by the ZIP-member hash.
Alternatively, explicitly retire the word “untouched” and document that the
future outcomes were decoded but not analyzed. Preserving the future set is
preferred.

### B5 — The 256-replicate Google randomization audit is a non-resumable single process

Severity: **high operational blocker for the requested experiment**.

The CLI has no shard/range/resume controls
(`run_google2022_randomization.py:175-178`), and
`run_google2022_randomization.py:1540-1558` executes all 256 replicates
sequentially, writing the result only after the final replicate.

The locked workload is 65,280,000 role updates. A value-free 5,000-update
synthetic kernel benchmark on this host measured 947.4 updates/s, projecting
about **19.14 hours** for role updates alone; full replay adds factor/e-process
work. A late interruption loses the complete audit. The manifest at
`run_google2022_randomization.py:1563-1592` also omits wall time, peak RSS,
operation counts, and checkpoint/resume provenance.

**Required repair and acceptance test**

Before freeze, implement either deterministic process parallelism or
replicate-range shards plus an exact merger. The merger must prove that seeds
`610700..610955` occur exactly once, rows are sorted by replicate index,
every replicate restored the same checkpoint, and merged bytes are
independent of shard layout. Record worker/process count, all shard hashes,
wall time, peak RSS, role updates, formal updates, and output bytes. Test
interrupted resume and byte-identical 1-worker versus multi-worker synthetic
outputs.

### B6 — One advantage predicate is asserted rather than audited

Severity: **medium, but part of the locked Boolean**.

At `experiments/run6/scripts/run_google2022_outcomes.py:1698`,
`no_method_received_extra_detector_records_or_outcome_labels` is hard-coded
to `True`. The surrounding artifact validators make unequal exposure
unlikely, but the published atomic predicate should be evidence-derived.

**Required repair and acceptance test**

Compute the predicate from verified per-method phase shapes/record counts and
one shared outcome-label hash, pass that evidence into
`build_decision_summary`, and serialize the evidence beside the Boolean.
Add a negative test in which one method receives one extra record or a
different label vector.

## Freeze/release order

After B1–B6 are resolved:

1. Run the complete value-free suite, Ruff, `pip check`, `git diff --check`,
   strict config injection tests, and final PNNL stat-only validation.
2. Fill all normative hashes and final statuses in the dependency order
   described in B2.
3. Commit the implementation and push it.
4. Generate the non-self-referential freeze manifest from committed blobs;
   commit and push it.
5. Generate the canonical ratification, commit and push it, then verify the
   full Git-blob chain from every real runner.
6. Run detector-only Google replay and freeze all detector artifacts.
7. Run the exact randomization audit and PNNL auxiliary arm.
8. Open only the authorized Google outcome slice in the separate outcome
   command and produce the full integrated decision.
9. Publish every contextual M0C/M1/M2 result and every negative reason,
   irrespective of whether they favor S-PACE.

## Does Run 6 currently show an advantage?

**No. There is no Run 6 result yet.** Held values remain unopened, and the
locked Boolean cannot be evaluated.

Even if the final Boolean becomes true, the permitted claim is narrow:
S-PACE improved the predesignated DFR and same-feature online-logistic
comparators at the locked Google event/top-20 endpoints and retained its
locked result on the constructed Pittsburgh circuit-and-hardware boundary.
It would not establish best-overall performance, superiority to M0C/M1/M2,
decoder likelihood, Helstrom/Wilson/oracle methods, quantum acceleration,
universal sample efficiency, scalable computational advantage, string
theory, or holographic duality.

If any atomic gate fails, the required conclusion is:

> no demonstrated S-PACE algorithmic advantage.

A negative result remains publishable as a reproducible no-go certificate,
paired-shot calibration study, and real-QEC benchmark.
