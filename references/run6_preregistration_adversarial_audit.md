# Run 6 preregistration adversarial audit

**Audit date:** 2026-07-27
**Gate:** before opening Google detector values or any decoder outcomes
**Data-access constraint observed:** this audit read only the preregistration,
locked JSON, theory, repository metadata and implementation interfaces. It did
not read `detection_events.b8` or any decoder/observable outcome values.

## Verdict

> **FAIL — DO NOT OPEN HELD VALUES.**

The scientific split, event windows, coordinate remap and claim boundaries are
largely sound. The protocol is not yet an executable lock, however. Several
unset choices can materially change alarms, ranks, risk coverage and the final
advantage Boolean. The three purported lock files are also untracked, and the
plan still says `draft lock`.

Audited file identities:

| File | SHA-256 at audit |
|---|---|
| `references/run6_real_qec_preregistered_plan.md` | `9253376a7f79b71c81c94485de723a03faccc048598a67a56356d4d61febfdce` |
| `experiments/run6/configs/google2022_locked.json` | `502be5b8aadf906ca65d02afedcc37daee86dc257c357b7844241954ff82b25d` |
| `experiments/run6/configs/pnnl_snapshot_locked.json` | `953b95529ea8e95d75a318cf17bc764c4b92393e81cb0602ddf7a1d1843dff12` |

## Checks that pass

- Both JSON files are syntactically valid objects.
- Google intervals are disjoint, contiguous and cover `[0,500000)`.
- All event windows lie inside `[40000,60000)` and satisfy
  `narrow ⊂ primary ⊂ wide`.
- The one-based README line versus zero-based shot ambiguity is correctly
  represented and cannot change window membership.
- `51 × 24 = 1224`, `20000 × 51 = 1020000`, and
  `24 + binom(24,2) = 300`.
- Pairing is one-to-one, round-role matched and stated in archive order.
- The Google coordinate parser correctly forbids a naive fixed-column reshape
  and requires the same 24 canonical checks at every role.
- Raw detector bits are shared across branches; decoder outcomes are explicitly
  embargoed until detector outputs freeze.
- M3–M5 state that scoring precedes the current update.
- The primary e-process threshold agrees with `alpha=0.01`; the SR threshold
  agrees with `gamma=10^6`.
- PNNL is correctly labeled a constructed cross-snapshot/domain-shift arm, not
  natural temporal drift, and physical paths are to come from state-specific
  QASM.
- The prohibited claims and the negative-result fallback are appropriate.

These checks are necessary, but they do not resolve the blockers below.

## Blocking findings

### B1. The lock has not been frozen

**Severity: fatal.**

- The plan says `Status: draft lock`; the JSON says
  `locked_before_detection_event_values`.
- All three audited targets are currently untracked.
- The plan's own freeze procedure requires hashes, package versions, commit and
  push, but no freeze manifest records the plan/config hashes, code revision,
  exact environment or commit ID.
- `pnnl_snapshot_locked.json` inherits semantics from a mutable filename,
  `google2022_locked.json`, without pinning its hash.
- No deviation-ledger path or schema is locked.

**Required before opening data:** resolve the status contradiction; create one
machine-readable manifest containing all input/config/plan hashes, git commit,
Python and package versions, platform/BLAS/thread settings and inherited-config
hashes; define the deviation ledger; commit and push the complete preregistration
unit.

### B2. Google binary and replay semantics are not fully locked

**Severity: fatal for parser reproducibility.**

The coordinate permutation is well specified, but the locked JSON omits:

- `.b8` little-endian bit order;
- 153 bytes per detector shot, per-shot byte alignment and zero padding count;
- expected detector-file byte count;
- the exact flattening order, e.g. shot-major then role `0,...,50`;
- global update index ↔ `(archive_shot, role)` conversion;
- whether adaptive state is shared across all 51 roles or maintained separately
  per role.

The last item changes the algorithm substantially because boundary and bulk
roles have different laws.

**Required:** serialize these points and add synthetic parser tests that verify
bit order, shot offsets, the boundary permutation and update ordering without
reading held detector values.

### B3. The five methods are not implementation-complete

**Severity: fatal; different reasonable implementations produce different
primary results.**

1. **DFR:** “two-sided bets” lists only positive magnitudes. It does not state
   whether the components are `±beta`, their prior weights, or the exact
   unpaired cycle/shot aggregation.
2. **Diagonal likelihood:** the exact NLL-difference formula and its mapping to
   `[-1,1]` are absent. Bet fractions and accumulator-component weights are
   absent. Clipping alone does not determine the normalization.
3. **Hotelling:** the input feature bank, centering, quadratic score, covariance
   inverse convention, selection of the 20,000 fit pairs and threshold sample
   are unset. Fitting covariance and calibrating a threshold on the same rows
   would leak calibration information unless a split or cross-fit is locked.
4. **Online logistic:** labels/orientation, loss, gradient, intercept,
   initialization, optimizer, per-pair update and expert-wealth combination are
   unset. The plan says a singular Kingston-tuned learning rate; the JSON gives
   three learning-rate experts. This is a direct inconsistency.
5. **Sparse S-PACE:** the half-life-to-decay formula, EWMA initialization/bias
   correction, role sharing, feature order and deterministic signed top-\(k\)
   tie-break are absent.
6. **Spectral S-PACE:** the same EWMA/role questions remain; “every eight
   cycles” lacks its phase; zero-positive-spectrum behavior and rank-one
   degeneracy handling are not fixed.

The accumulator section also does not say which bet grid and component bank is
attached to M1, M3 or each empirical comparator.

**Required:** lock equations or pseudocode for every score, update, mixture and
initial state. Lock deterministic feature ordering and tie rules. Resolve the
logistic inconsistency.

### B4. Thresholds, alarms and the advantage Boolean are not executable

**Severity: fatal for outcome-independent analysis.**

The following decisions remain open:

- whether empirical thresholds apply to a raw cycle score, shot aggregate,
  accumulated statistic or running maximum;
- how cycle scores become one shot score (`max`, mean, end-of-shot wealth,
  first crossing, or another rule);
- reset, cooldown and repeated-alert rules needed to define “alerts per
  100,000 cycles”;
- which of the two empirical operating points is primary;
- whether M4, M5, a fixed M4/M5 mixture or the better observed branch is
  “S-PACE” in the advantage rule;
- the exact scalar used for “strictly improves” timing, mismatch capture or
  risk coverage, including numerical tie tolerance and minimum effect;
- the risk-coverage summary and direction (point value, area or dominance);
- how one deterministic natural event obtains a “miss probability”;
- whether auxiliary retention must use PNNL or the unspecified untouched
  Google alternative. The current `or` permits post-result branch choice.

Most importantly, cycle-to-shot aggregation is still selectable when decoder
outcomes are opened. That is direct outcome-analysis flexibility.

**Required:** lock the alarm state machine, shot aggregation, primary operating
point, fixed S-PACE composite, exact comparison functional/tolerance and one
auxiliary branch before reading any held value.

### B5. Calibration and resource parity do not match the scientific question

**Severity: fatal for a “same budget” advantage claim.**

- M1 fits on 40,000 shots.
- M2 fits on at most 20,000 cycle pairs, with selection not specified.
- thresholds use the validation pairs, potentially also used for fitting.
- M3 hyperparameters use an external viewed Kingston pilot.
- M4/M5 use online paired observations but no comparable offline fit.

The plan says “same calibration, observation and false-alert budget,” yet no
per-method ledger fixes which records count as calibration versus surveillance.
The advantage rule merely requires excess compute to be disclosed, not matched.
Wall-time comparison also lacks hardware, thread count, warm-up/repetition and
measurement procedure.

**Required:** either equalize the information/calibration budget or change the
question to a transparently unequal-budget comparison. Lock disjoint fitting
and threshold-calibration roles, external-pilot accounting, physical-shot
counts, update counts and the common resource-measurement protocol.

### B6. Randomization and determinism are incomplete

**Severity: fatal for the exact-design audit; major for reproducibility.**

`replicates=256` and `seed_start=610700` do not define:

- which pair block is randomized;
- the exact seed list and RNG algorithm/version;
- Bernoulli swap generation and when the swap is revealed relative to scoring;
- whether every replicate resets all witnesses and accumulators;
- the statistic and multiplicity rule used by the randomization audit.

Hotelling subsampling, block/bootstrap uncertainty and any threshold resampling
lack complete seeds and algorithms. Floating dtype, log-domain overflow policy,
eigensolver/thread settings and deterministic tie handling are also absent.

**Required:** lock the randomization dataset, RNG (`PCG64` or equivalent),
seed-to-replicate mapping, swap timing, reset behavior, audit statistic,
resampling/block lengths, confidence level, replicate counts and numeric
determinism policy.

### B7. The PNNL auxiliary arm is a filter, not a frozen experiment

**Severity: fatal for the independent-retention clause.**

The config correctly restricts claims but does not determine a unique result:

- source archive sizes/checksums and an extraction/version manifest are absent;
- `backend_property_date` and calibration/QASM hash serialization are not
  defined;
- no metadata-only manifest enumerates the held Pittsburgh paths/snapshot pairs;
- QASM measurement parsing, logical-state selection and the detector formula
  \(\chi_{0,i}=S_{0,i}\),
  \(\chi_{t,i}=S_{t,i}\oplus S_{t-1,i}\) are not in the lock;
- behavior when the later cohort has fewer rows is unset;
- pooling/weighting across paths, bases, distances and logical-state replicates
  is unset;
- threshold calibration, alarm metric, uncertainty method/seeds and “same
  direction” are undefined;
- circuit-controlled and circuit-and-hardware-shift cohorts are labeled, but
  it is not fixed which class supplies retention evidence.

**Required:** generate and hash a metadata-only cohort manifest before syndrome
values are read; lock parsers/hashes, exact stream construction, pooling,
threshold, uncertainty and the retention statistic.

### B8. No implementation gate exists

**Severity: fatal operationally.**

There is no Run 6 runner, config schema, result schema or test suite.
`experiments/aoc/space.py` and `experiments/aoc/qec_real.py`, named by the plan,
do not exist. Nothing in code consumes either lock JSON. Existing dependency
ranges are broad and Run 6 is absent from pytest registration.

**Required:** implement strict config validation that rejects unknown/missing
fields; add metadata/synthetic-fixture parser, causality, bounds, determinism,
mixture and resource-ledger tests; run a no-held-values dry run; freeze the
runner/environment hash before opening held values.

The final theory also lists global/per-check DFR CUSUM and a sparse scan among
required advantage controls, but the preregistration freezes neither. Add them
or narrow the claimed advantage gate before lock.

## Minimum re-audit checklist

A PASS requires all of the following:

1. All B1–B8 decisions are serialized in versioned configs or exact referenced
   specifications.
2. A strict schema and synthetic tests prove the parser, causal score-before-
   update order, score bounds and deterministic replay.
3. Every method has a fixed data/resource ledger and disjoint fitting/threshold
   rule.
4. Shot aggregation, alarm state machine and the advantage functional are
   executable without outcomes.
5. A hashed PNNL cohort manifest and aggregation rule exist.
6. A freeze manifest pins inputs, inherited configs, code, environment and
   seeds.
7. The status is changed from draft only after the complete unit is committed
   and pushed.

Until a new adversarial audit records **PASS**, opening the Google detector
stream or decoder outcomes would turn later choices into post-observation
analysis decisions.
