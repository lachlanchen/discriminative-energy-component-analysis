# Run 6 post-detector schema incident

**Incident time:** 2026-07-28T01:53:12+08:00
**Stage:** after the detector-only Google replay; before any successful
randomization replicate, Google decoder-outcome access, or PNNL held-payload
access
**Original ratified HEAD:** `7e378d7f1d99818fc5e366bb14a7200767722d6c`

## Immutable evidence

- The completed detector manifest is
  `experiments/run6/results/google_detector/detector_freeze_manifest.json`,
  SHA-256
  `ed9d9dcdcb2b3e78d144f2a2ce3cec6b6269ffce3f7e18f443784e5d6174c0c3`.
- It declares `detector_only=true`, `outcome_accessed=false`, and
  `outcome_join_authorized=false`.
- Its 231 declared artifacts all passed byte-length and SHA-256 verification;
  the three held joint-replay digests are identical.
- The launcher created 32 preserved `.attempt_*` directories. All 32 stderr
  logs are byte-identical, with SHA-256
  `d3bc72d114336901c1b502f2722cc3e4a7c44030f03fa38e7f861fc8c5e6dd3e`.
  Every shard failed during detector-artifact schema validation, before a
  randomization replicate completed.

The failed attempts are retained as evidence. They must not be presented as
randomization results or silently deleted.

## Failure

The detector producer writes `expert_id_rule` in each
`proper_prior[method]` row of `formal_component_summary.json`, but not in the
corresponding `shiryaev_roberts[method]` row. Both downstream consumers
require the field in both rows. The first failure is therefore:

```text
ValueError: formal SR summary m0 schema mismatch;
missing=['expert_id_rule'], unknown=[]
```

This is a producer--consumer schema mismatch in redundant metadata. It is not
a numerical failure: the SR row already binds `component_weights`,
`role_count`, `base_component_count`, `expert_flatten_order`, component
states, statistic, crossing, and threshold. The expert identifier rule is
also present in the proper-prior row, and both accumulators share the same
role-major flattening and weights.

## Access and contamination record

- Google detector values have been read, scored, and serialized by the
  ratified detector-only runner.
- Google decoder outcomes have not been opened, parsed, hashed, or joined.
- PNNL held `bitstrings.json` values have not been opened, parsed, or hashed.
- No randomization replicate completed.
- During diagnosis, a broad result display unintentionally exposed some
  detector-derived numeric fields to the primary agent, and a delegated
  audit used a broad `rg` that exposed a numeric detector array and summary
  tail. Neither exposure contained decoder outcomes or PNNL values, and no
  exposed detector number may be used to choose or modify the repair.

The original claim “frozen before held detector access” remains true only for
the original chain and detector run. Any repair chain must explicitly say
**post-detector and pre-outcome**, not recreate or imply detector-value
blindness.

## Permitted repair scope

The proposed compatibility repair is fixed from source/schema evidence:

1. make the SR exact-key set omit only the redundant `expert_id_rule`;
2. continue requiring the proper-prior rule exactly;
3. continue requiring identical role-major flatten order, role count, base
   count, component weights, prior mass, factor bounds, thresholds, trace
   endpoints, and crossing semantics for both families;
4. apply the same schema rule in the randomization and outcome consumers;
5. add a producer-to-consumer integration regression using the actual
   detector producer shape;
6. change no score, feature, split, seed, threshold, event window,
   randomization distribution, PNNL rule, endpoint, or claim gate.

The completed detector artifacts will be reused only if an independent audit
confirms that the compatibility repair neither changes nor reconstructs any
numeric detector artifact. A detector rerun is required if that condition
cannot be proved.

## Required repair gate

Before another real shard starts, a new pushed implementation → repair
manifest → repair ratification chain must:

- bind this incident record, the original freeze chain, the completed detector
  manifest and all of its artifacts, and the exact repair diff;
- declare detector-value access `true`, decoder-outcome access `false`, PNNL
  held-payload access `false`, and completed randomization replicates `0`;
- reject changes outside the audited schema/verification scope;
- recheck Git ancestry, current blobs, environment, numeric thread settings,
  package lock, and runtime module origins; and
- fail closed if any original detector artifact differs.

Until that gate passes, Run 6 randomization and all advantage claims remain
blocked.
