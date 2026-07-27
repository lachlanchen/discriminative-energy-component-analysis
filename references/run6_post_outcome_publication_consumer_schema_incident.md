# Run 6 post-outcome publication-consumer schema incident

Date recorded: 2026-07-28 (Asia/Hong_Kong)

## Scope and chronology

All locked scientific producers had completed before this incident,
including the one-shot Google outcome join.  The canonical outcome directory
is `experiments/run6/results/google_outcomes`; its manifest SHA-256 is
`fac31ab5cff646fd081ba0d4e6a8a73ba42747618997f35702d172e563b7f62b`.
The outcome producer was not rerun.

The first strict publication extraction then failed before emitting
`publication/run6/generated`.  The atomic output directory remained absent.
The failure occurred inside PNNL resource-ledger validation, before the
extractor reached `validate_outcome_manifest`.

Exact observed exception:

```text
PublicationDataError: PNNL adaptive-state ledger must be an object.
```

The command used the eight canonical production inputs and requested
`publication/run6/generated` as its output.  This record reconstructs the
exception from the Codex terminal result because a separate stderr file was
not created at the time; it does not mislabel that reconstruction as a
pre-existing log artifact.

## Root cause

The frozen PNNL producer declares
`adaptive_state_ledger: list[dict[str, Any]]`, appends one record for each of
11 cohorts and each logical state \(0,1\), and serializes the resulting
22-row array in `resource_ledger`.  The pre-outcome publication consumer
instead called `require_mapping` on that field.  Its synthetic fixture had
also encoded an empty object, so the contract error escaped the value-blind
pre-outcome tests.

## Access truth

- Outcome production had completed before the failure: **true**.
- The failed extractor reached or opened the outcome manifest: **false**.
- The failed extractor automatically validated derived detector and
  randomization records before reaching the PNNL ledger: **true**.
- The failed extractor loaded the PNNL results manifest: **true**.
- It reached PNNL hash-bound performance arrays after the resource check:
  **false**.
- A human inspected performance values to diagnose or select this repair:
  **false**.
- The repair was selected from producer source and field schema alone:
  **true**.
- Any scientific producer, result artifact, threshold, seed, endpoint,
  empirical gate, outcome, or recorded runtime changed: **false**.
- Any scientific producer or the outcome join was rerun: **false**.

Because the outcome producer already existed, this amendment is not called
outcome-blind or preregistered even though the failed consumer had not yet
opened the outcome manifest.

## Repair

The repaired publication consumer treats the field as exactly 22 ordered
records and validates more than container type:

1. exact cohort/state order and seven-field row schema;
2. `q = distance - 1` and `roles = rounds` from the locked Pittsburgh
   metadata;
3. exact component counts for DFR, online logistic, sparse, spectral, and
   composite banks;
4. exact numeric storage reconstructed independently from the frozen array
   shapes; and
5. nine adversarial mutations covering object/list confusion, length,
   schema, order, dimension, role count, component counts, and both byte
   identities.

This is a publication-consumer repair only.  It supplies no new scientific
evidence and cannot alter the locked conjunction or its claim boundary.
