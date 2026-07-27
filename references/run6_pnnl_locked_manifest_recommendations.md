# Run 6 Pittsburgh auxiliary executable lock

**Status:** frozen metadata-only selection before held-value access; no selected
`bitstrings.json` content was opened before freeze.

## Outcome

The metadata-only Pittsburgh scan now determines one reproducible auxiliary
experiment before any selected syndrome or final-data value is read. The
machine-readable manifest is
`experiments/run6/configs/pnnl_pittsburgh_locked.json`.

Freeze dependencies are acyclic: the finalized shared method/config and
references are parents of the Pittsburgh cohort manifest; that manifest and
this auxiliary specification are parents of the PNNL executable config; the
top-level freeze manifest hashes all of them. The Pittsburgh manifest
deliberately does not hash the PNNL executable config that hashes it, because
mutual content hashes would be unsatisfiable.

The scan used only `info.json`, `calibration.json`, the two state QASM files,
and `stat` file sizes for `bitstrings.json`. It did **not** open, stream,
grep, hash, summarize, or score any `bitstrings.json`.

The deterministic filter reproduced:

- 852 state-0/state-1 path-agreeing chain instances;
- 837 unique `(d,r,basis,oriented physical path)` groups;
- exactly 11 eligible earliest/latest Pittsburgh snapshot pairs;
- seven X-basis and four Z-basis pairs; and
- zero byte-identical or normalized-identical complete two-state QASM pairs.

The last point fixes the claim class: **all 11 Pittsburgh comparisons are
simultaneous circuit-and-hardware domain shifts**. None can be called
circuit controlled, natural temporal drift, or a change with a verified
physical onset.

## Exact selected cohorts

`m` is the number of paired shots per logical-state replicate in each pre or
post phase:

\[
m=\min(\lfloor N_{\mathrm{early}}/3\rfloor,N_{\mathrm{late}}).
\]

| # | \(d,r\) | basis | QASM-derived oriented path | early job | late job | \(m\) |
|---:|---:|:---:|---|---|---|---:|
| 1 | 3,1 | X | `18-11-12-13-14` | `d3_r1/job_1` | `d3_r1/job_3` | 682 |
| 2 | 3,1 | X | `111-98-91-92-93` | `d3_r1/job_1` | `d3_r1/job_2` | 682 |
| 3 | 3,3 | X | `14-13-12-11-18` | `d3_r3/job_2` | `d3_r3/job_3` | 682 |
| 4 | 3,3 | X | `115-114-113-119-133` | `d3_r3/job_1` | `d3_r3/job_2` | 682 |
| 5 | 3,3 | Z | `151-152-153-154-155` | `d3_r3/job_4` | `d3_r3/job_6` | 1,365 |
| 6 | 5,1 | X | `18-11-12-13-14-15-19-35-34` | `d5_r1/job_1` | `d5_r1/job_3` | 682 |
| 7 | 5,1 | Z | `18-11-12-13-14-15-19-35-34` | `d5_r1/job_5` | `d5_r1/job_6` | 682 |
| 8 | 5,5 | X | `34-35-19-15-14-13-12-11-18` | `d5_r5/job_1` | `d5_r5/job_3` | 682 |
| 9 | 7,3 | Z | `80-81-82-83-84-85-77-65-66-67-57-47-46` | `d7_r3/job_6` | `d7_r3/job_7` | 682 |
| 10 | 7,7 | Z | `138-151-150-149-148-147-146-145-144-143-142-141-140` | `d7_r7/job_7` | `d7_r7/job_9` | 682 |
| 11 | 9,5 | X | `153-154-155-139-135-134-133-119-113-114-115-99-95-94-93-92-91` | `d9_r5/job_1` | `d9_r5/job_2` | 682 |

This produces 22 independent path-state replay streams. Each complete pre or
post phase has 16,370 paired shots and 49,110 paired cycle updates.

## Resolved implementation contract

### Artifact and snapshot identity

The manifest records exact byte counts and raw/canonical hashes for all 20
selected job directories. A snapshot is identified by backend plus the
canonical full calibration hash. The source property date is parsed as an
offset-aware datetime and compared in UTC; it remains a
**backend-property date**, not an execution timestamp.

Raw SHA-256 controls artifact integrity. Calibration semantic identity uses a
fully specified Python-canonical JSON profile: reject duplicate keys and
non-finite numbers, sort object keys, encode UTF-8, and serialize compactly
without a terminal newline. This profile is deliberately named and is not
misrepresented as RFC 8785/JCS.

QASM has both a raw hash and a normalized-text audit hash. Only equality of
both raw state QASM files permits the label `circuit_controlled`; normalized
equality can never upgrade that label.

The manifest intentionally records only `stat` sizes for the held bitstring
files. Their raw hashes should be computed and logged as the first
post-ratification unblinding action.

### QASM path and detection-event parsing

For register suffix \(\rho\), parse the state-specific assignments

\[
c_{\mathrm{data},\rho}[i]\leftarrow q=D_i,\qquad
c_{\mathrm{syndrome},\rho}[t(d-1)+i]\leftarrow q=A_{t,i}.
\]

Require complete unique index coverage, a stable \(A_{t,i}=A_i\) over
rounds, and identical state-0/state-1 maps. The oriented path is

\[
(D_0,A_0,D_1,A_1,\ldots,D_{d-2},A_{d-2},D_{d-1}).
\]

Never infer it from the register suffix and never merge its reversal.

After unblinding, select logical-state entries by
`metadata.logical_state`, not array position. Reshape syndrome registers in
row-major order,

\[
S_{n,t,i}=S^{\mathrm{flat}}_{n,t(d-1)+i},
\]

then use exactly

\[
\chi_{n,0,i}=S_{n,0,i},\qquad
\chi_{n,t,i}=S_{n,t,i}\oplus S_{n,t-1,i}\quad(t\ge1).
\]

There is no terminal detector. Final data bits are not detector inputs.

### Pairing and state handling

The early snapshot is partitioned into three consecutive blocks of `m`
shots; the late snapshot contributes its first `m` shots:

- pre: early `[0,m)` as A versus early `[m,2m)` as B;
- post: early `[2m,3m)` as A versus late `[0,m)` as B.

Pair equal row indices only, keep the same round role, and update
shot-major then round-major. Logical states 0 and 1 are separate circuit
replicates. They must never be paired or concatenated.

For a formal exact-randomization trace, one complete paired shot is one time
step. All round-role factors from that shot are scored using their respective
pre-shot states and mixed as `(round-role, base-component)` experts under a
uniform role prior. They are not compounded into multiple formal time steps,
because all rounds share the same A/B swap bit. Empirical cycle diagnostics
may remain round indexed.

For each path-state stream, the first `floor(m/2)` pre pairs form the causal
fit prefix. Preserve the learned witness state, reset only its alarm
accumulator, and replay the remaining pre pairs followed by the post pairs
without resetting at the constructed boundary.

### Matched threshold and endpoint

The Pittsburgh data cannot support the Google operating point of one alert
per 100,000 cycles: even the entire pre phase contains only 49,110 paired
cycle updates. The auxiliary primary threshold is therefore explicitly
different:

- 1% per-path-state finite-episode threshold;
- 4,096 circular moving-block bootstrap replicates;
- complete paired-shot blocks of length 32;
- threshold from the maximum shot-indexed log proper-prior e-process trace;
- PCG64 seed
  `611000 + 100*cohort_index + 10*logical_state + method_index`; and
- a conservative finite-bootstrap order statistic fixed in the manifest.

This is a matched empirical episode threshold, not a claim of
one-in-100,000-cycle hardware false-alarm control. The fixed \(E\ge100\)
result remains a secondary exact-randomized-design statement only.

The target is the same fixed `space_composite` used in the Google arm:
one-half prior mass on `space_sparse` and one-half on `space_spectral`.
When a sparse \(k\) exceeds the dimension-adjusted feature count, that
component is removed before data access and the sparse branch is
renormalized internally; the branch still receives total mass one-half.
Both branches are reported as sibling ablations and neither can replace the
composite after Pittsburgh outcomes are seen.
For every path-state-method stream report:

1. whether it first alarms before the boundary; and
2. restricted post-delay fraction: 1 for a pre-alarm or miss, otherwise
   `(post_alarm_shot+1)/m`.

Alarm time is a complete paired-shot index only. The formal uniform mixture
over round-role experts has no crossing-role coordinate; any later
dominant-role localization is a separate descriptive diagnostic.

Average the two logical states within a physical path, then macro-average the
11 paths equally. Do not weight by shots, rounds, basis, distance, feature
dimension, or number of syndrome bits.

Retention of the fixed composite against each of DFR and same-feature online
logistic requires both:

- no more pre-boundary false alarms over the 22 state streams; and
- a strictly lower equal-weight mean restricted post-delay fraction.

Ties fail the strict-delay condition. All methods and every cohort row must
still be reported.

### Uncertainty and deterministic seeds

The independent summary unit is the physical path/snapshot pair after
averaging logical states. The primary uncertainty report is a 10,000-draw
cohort bootstrap with PCG64 seed `612500`. A second 10,000-draw sensitivity
bootstrap clusters by the ordered calibration-hash pair with seed `612501`;
it is descriptive because only five such clusters occur. Also enumerate all
\(2^{11}\) sign flips of the 11 paired cohort effects.

The exact random-swap audit uses seeds 610700 through 610955 inclusive,
one fresh PCG64 generator per replicate, one swap bit per complete paired
shot, the same mask for every method, score-before-update for every round
role, one expert-mixture e-update per shot, and a full method reset per
path-state replicate. This validates the declared randomized design, not a
natural hardware null.

## Freeze gate

Before opening any selected `bitstrings.json`:

1. retain the ratified cohort rows without value-dependent amendment;
2. freeze and commit the manifest;
3. record its SHA-256 and the code/environment lock;
4. implement a dry-run validator that checks all metadata, QASM, hashes,
   dates, selected paths, and seeds without opening bitstrings; and
5. only then begin the logged unblinding run.

The manifest has already passed a metadata-only validator covering valid
JSON, all 20 artifact records, all 11 selected paths, state-map agreement,
partition sizes, calibration-pair IDs, and raw/normalized QASM labels.
