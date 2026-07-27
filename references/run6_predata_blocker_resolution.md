# Run 6 pre-data blocker resolution

**Scope:** implementation-only closure record before any held-value access
**Payload status:** no Google `.b8`/`.01` value and no PNNL
`bitstrings.json` value was opened, decoded, sampled, grepped, or analyzed
while making these repairs.

This record resolves the six blockers identified in
`run6_final_predata_audit.md` and the additional integration requirements in
`run6_runner_integration_audit.md`. The historical audit reports are retained
unchanged; their `BLOCK` verdict describes the earlier audited snapshot.

## B1 — committed and pushed freeze provenance

Resolved in the implementation:

- `aoc.run6_protocol.verify_committed_freeze_chain` requires the canonical
  manifest and ratification paths, exact schemas, false pre-access flags,
  implementation/freeze/current commit ancestry, containment in
  `origin/main`, and byte equality among the current file, implementation
  Git blob, freeze Git blob, and SHA-256 registry.
- The ratification file itself must equal the tracked blob at a pushed
  `HEAD`.
- Every tracked Python source under `experiments/aoc` and
  `experiments/run6/scripts` must be included; untracked runtime Python is
  forbidden.
- All four real runners and the parallel launcher call the same helper before
  payload access.
- `create_freeze_chain.py` generates the acyclic implementation → manifest →
  ratification chain and cannot accept a data path.

The final implementation, manifest, and ratification commits are created and
pushed only after all fields and tests below are final.

## B2 — PNNL pending/frozen contradiction

Resolved:

- the metadata validator has explicit `pending` and `frozen` modes;
- the real PNNL runner requires `mode="frozen"`;
- each mode has an explicit locked static-contract digest;
- a third status is rejected;
- final parent hashes and the frozen digest are filled in dependency order
  before the implementation commit.

## B3 — PNNL `c_data` policy

Resolved by an honest contract. After ratification, `c_data` integers are
parsed only for strict structure, shape, and binary-domain validation. They
are never supplied to detector features, calibration, alarms, uncertainty,
retention, or logical-risk claims.

A value-free regression changes every valid synthetic `c_data` bit while
holding syndromes fixed and obtains exactly identical detection events,
component factors, checkpoints, bootstrap thresholds/maxima, e-traces,
alarms, retention rows, and aggregate decision. An invalid value is rejected.

## B4 — untouched Google future outcomes

Resolved with a streaming slice parser. The outcome runner interprets only
records `[40000,60000)`. Records outside the authorized slice are counted as
line boundaries but their payload bytes are not converted to values. The
full extracted file remains bound to the verified ZIP member by raw hash.

## B5 — non-resumable 256-replicate audit

Resolved without changing the locked randomization distribution:

- seeds remain exactly `610700..610955`;
- one PCG64 swap bit orients all 51 roles of one complete paired shot;
- every replicate restores the same frozen warm checkpoint;
- shard manifests use mandatory half-open ranges and record one worker,
  one-thread numeric execution, exact updates, wall time, RSS, and bytes;
- the merger rejects gaps, overlaps, duplicates, seed changes, and checkpoint
  changes, and produces canonical seed-sorted output independent of shard
  layout;
- the launcher fixes 32 shards of eight replicates, runs at most 16
  single-threaded workers, atomically promotes completed shards, reuses only
  completed manifests, preserves failed attempts, and records actual external
  concurrency through a thread-safe active-subprocess counter, together with
  the configured cap and executed/reused shard counts.

The scalar algorithm is intentionally retained. No unverified vectorized
eigendecomposition or floating-point reduction replaces it.

## B6 — fairness predicate

Resolved. The no-extra-input predicate is derived from verified per-method
cycle shapes/counts, archive-shot identity, shared outcome-label bundle hash,
and shared detector-manifest binding. A negative synthetic test changes one
method's evidence and forces failure.

## Additional integration closures

- The complete environment lock is parsed as unique normalized
  `name==version` pins and compared with every installed distribution; missing,
  changed, duplicate, and unexpected distributions fail. The separately
  Git-bound editable repository is the sole explicit exception.
- The PNNL metadata validator recomputes every declared parent-artifact
  SHA-256 from the current safe repository file, so a stale cross-document
  hash cannot be hidden by an internally self-consistent static digest.
- Runtime `aoc` modules must resolve to the frozen repository paths.
- Formal traces are finite and shot-indexed only for exact methods. M0C/M2
  traces are all-NaN with all-false crossings and are labeled
  `not_applicable_no_formal_accumulator`.
- Consumers verify expert counts, base-prior sums, uniform role mass, no
  within-shot factor compounding, both primary and secondary event summaries,
  and expanded threshold/held shot tables.
- The PNNL consumer requires exactly 226 portable result artifacts: one
  unblinding record, one state table, one aggregate, three randomization
  artifacts, 110 trace arrays, and 110 bootstrap-maxima arrays. Names,
  horizons, shapes, and little-endian dtypes are checked.

## Claim boundary

These repairs establish reproducibility and auditability, not performance.
Until held runs finish, the result is **not yet evaluated**. After running,
failure of any locked gate requires:

> no demonstrated S-PACE algorithmic advantage.

Even a positive narrow gate cannot support superiority to Helstrom, a Wilson
oracle, same-parity logistic/threshold methods outside the locked endpoints,
quantum acceleration, universal sample efficiency, scalable computational
advantage, a new Wilson-loop or toric-code theorem, string theory, or
holographic duality.

## Final pre-access verification — 2026-07-28

The independent integration re-audit returned **IMPLEMENTATION GO** with no
remaining implementation blocker. The complete Run 6 suite passed
**138/138** tests, and the full Runs 1--6 regression suite passed
**231/231** tests; Ruff, byte compilation, `pip check`, and `git diff --check`
passed; all five synthetic dry-runs reported
`raw_run6_values_opened=false`; and the frozen PNNL metadata-only validator
confirmed 20 snapshots and statted, but did not open or hash, 20 held payload
files.

This GO applies only to creating the committed and pushed three-stage freeze
chain. Real payload access remains blocked until the common verifier accepts
that chain.
