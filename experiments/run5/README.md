# Run 5: what syndrome data can and cannot detect

Run 5 tests accessible-observable sequential change detection on topological
quantum-error-correction data. It has two deliberately separate evidence
layers:

1. an exact periodic phenomenological detector model with provable
   count/rate no-go cases; and
2. a Stim/PyMatching circuit-level surface-code validation.

The exact model supports the theorem. The circuit simulation tests practical
transfer but is not hardware evidence.

The preregistered protocol and claim boundaries are in
[`../../references/run5_surface_code_drift_preregistered_plan.md`](../../references/run5_surface_code_drift_preregistered_plan.md).

Install the QEC dependencies:

```bash
.venv/bin/python -m pip install -e './experiments[qec,test]'
```

Run the fast pilot before the locked paper configuration:

```bash
.venv/bin/python experiments/run5/scripts/run_identifiability_certificate.py
.venv/bin/python experiments/run5/scripts/run_syndrome_drift.py \
  --config experiments/run5/configs/pilot.json
```

The sequential output keeps three estimands separate:

- `null_arl`: right-censored no-change run lengths, including the exact
  deterministic \(R_t=t\) behavior of a zero-score SR detector;
- `surveillance`: post-change delay conditional on no alarm at or before the
  declared changepoint, with pre-change alarms reported separately; and
- `restart_at_change`: every detector is restarted from zero on the same
  post-change segment. This is the primary paired delay comparison and avoids
  crediting the blind SR clock with an artificial
  `threshold - change_time` detection delay.

The aggregate table also subtracts each restart delay from the corresponding
no-change restricted mean run length at the same horizon. The exactly blind
control should therefore have zero acceleration, even though its SR statistic
still reaches the ARL threshold.

The publication configuration is immutable and writes to a separate result
directory:

```bash
.venv/bin/python experiments/run5/scripts/run_syndrome_drift.py \
  --config experiments/run5/configs/paper.json
.venv/bin/python experiments/run5/scripts/run_offline_diagnostic_audit.py \
  --config experiments/run5/configs/offline_diagnostic_locked.json
.venv/bin/python experiments/run5/scripts/run_shadow_measurement_audit.py
.venv/bin/python experiments/run5/scripts/run_circuit_level_validation.py
```

`paper.json` is a cycle-fair locked design. `calibration_cycles`,
`change_time_cycles`, `post_change_horizon_cycles`, and
`null_horizon_cycles` always mean physical syndrome cycles. The spatial
detector updates once per cycle and uses threshold 1000; the temporal detector
updates once per nonoverlapping two-cycle pair and uses threshold 500. Both
therefore have the same target ARL of at least 1000 physical cycles. Temporal
calibration consumes 4096 cycles (2048 pair updates), the change occurs at
cycle 256, and the post/no-change horizons are 1024/5000 cycles. Cross-family
sample-efficiency claims are prohibited because their update counts differ.

The named non-oracle, model-agnostic sequential comparator is
`validation-trained linear logistic effect`. It was chosen after inspecting
the independent offline diagnostic audit but before the sequential locked
test; it is not claimed to be the strongest possible generic baseline. It is
trained at the preregistered middle effect on 8 independent labeled streams
per class and 512 cycles per stream, or 4096 physical cycles per class in each
family. vAOC likewise receives 4096 null covariance-calibration cycles and
8 independent alternative ridge-validation streams of 512 cycles. The
logistic coefficient is affinely mapped into a valid effect in `[0,1]`,
centered at the analytic null, and run through the same `BoundedScoreSR` with
the same physical-cycle ARL target. Matched analytic witnesses and exact
likelihood ratios remain model-aware controls or ceilings.

`primary_named_comparator_audit.csv` compares vAOC to this comparator at the
two middle effects using independent paired restart delays. Because both
restricted delays lie in `[0,H]`, their differences lie in `[-H,H]`; the
inferential comparison uses distribution-free one-sided Hoeffding p-values,
Holm adjustment, and conservative Bonferroni-Hoeffding simultaneous upper
bounds. Paired percentile-bootstrap intervals are descriptive only. A
positive overall comparison flag requires both hypotheses and the ARL
condition to pass. Even a pass supports only vAOC versus this named comparator
on these two controlled tasks—not strongest-baseline or general algorithmic
superiority.

All calibration, baseline-training, ridge-validation, locked-test, scaling,
and timing sampling seeds occupy explicit disjoint partitions.
`load_config` rejects a publication configuration that is unlocked, has odd
temporal cycle budgets, omits unit/claim provenance, changes the frozen
primary comparison, or overlaps sampling seed intervals.

The locked offline audit is deliberately separate from sequential
surveillance. It compares count/rate, detector-first-moment,
translation-correlation, and \(D_4\)-resolved Fourier features using fixed
linear, kernel, and regularized Hotelling estimators. Only the \(D_4\)
Fourier/pair-lift probability simplex uses the indicator positive-support
projector with an ECA/AOC operator interpretation. Count, first-moment, and
translation feature families use a generic positive-part mean direction and
are not labeled Helstrom or AOC optima. Exact known-model likelihood ratios
are reported only as ceilings. The test split is never used to fit a
direction, select a threshold, tune a hyperparameter, or choose the compact
summary row; that row is selected by validation AUC before its test
performance is reported.

The publication run also writes `spatial_scaling.csv` for the locked
`scaling_sizes` and `scaling_repetitions`. It uses the middle spatial shift,
fresh-start streams, the declared post-change horizon, per-size covariance
calibration, and the spatial ridge selected at \(L=5\). The table records the
analytic translation-feature gap, restricted delay/miss metrics, and a fixed
10,000-sample feature/likelihood wall-clock measurement. The timing is an
engineering measurement, not a complexity theorem.

Generated result tables, figures, summaries, and manifests are committed.
Large raw detector tensors are regenerated from fixed seeds and are not
committed.
