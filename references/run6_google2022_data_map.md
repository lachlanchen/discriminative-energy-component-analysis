# Run 6 Google 2022 pilot: acquisition and causal data map

**Audit date:** 2026-07-27
**Scope:** acquisition, integrity, archive mapping and causal parsing only. No detector or decoder performance experiment was run.

## Acquisition and integrity

- **Official record:** [Zenodo 6804040](https://zenodo.org/records/6804040), DOI [10.5281/zenodo.6804040](https://doi.org/10.5281/zenodo.6804040)
- **Metadata API:** <https://zenodo.org/api/records/6804040>
- **Exact content endpoint:** <https://zenodo.org/api/records/6804040/files/google_qec3v5_experiment_data.zip/content>
- **Associated paper:** [Suppressing quantum errors by scaling a surface code logical qubit, Nature 2023](https://www.nature.com/articles/s41586-022-05434-1)
- **License:** CC BY 4.0
- **Official byte count:** `315490804`
- **Official checksum:** `md5:a7fd8b481c3087090093106382dc217d`
- **Local ZIP:** `experiments/data/run6/google_2022/google_qec3v5_experiment_data.zip`
- **Extracted root:** `experiments/data/run6/google_2022/extracted/`
- **Ignore rule:** existing `.gitignore` line 23, `/experiments/data/`

Local verification matched both official values exactly:

```text
315490804 bytes
a7fd8b481c3087090093106382dc217d
```

`unzip -tq` reported no errors. The archive has 2,095 ZIP entries: 1,964 regular files and 131 experiment-directory entries, with 548,415,888 uncompressed bytes. The extracted root additionally contains `README.txt`.

## Archive map

The archive contains seven million hardware shots:

| Experiment family | Directories | Shots per directory | Total shots |
|---|---:|---:|---:|
| Repetition code, Z basis, \(d=25\), 50 rounds | 1 | 500,000 | 500,000 |
| Surface code, X/Z bases, \(d=3\) | 104 | 50,000 | 5,200,000 |
| Surface code, X/Z bases, \(d=5\) | 26 | 50,000 | 1,300,000 |

For each surface-code distance and basis, the round counts are
`01,03,05,...,25`. Distance 3 uses four centers,
`(3,5),(5,3),(5,7),(7,5)`; distance 5 uses center `(5,5)`.
Directory names follow the documented pattern
`{code}_b{basis}_d{distance}_r{rounds}_center_{row}_{col}`.

## Author-identified high-energy event

The exact experiment directory is:

```text
experiments/data/run6/google_2022/extracted/
  repetition_code_bZ_d25_r50_center_5_5/
```

The archive README, lines 12–29, says this full repetition-code experiment contains the high-energy event discussed in the paper. It directs the reader to compare:

```text
repetition_code_bZ_d25_r50_center_5_5/obs_flips_actual.01
repetition_code_bZ_d25_r50_center_5_5/obs_flips_predicted_by_correlated_matching.01
```

and notes a “huge cluster of mismatches near shot 57775.” Re-running only that documented localization command confirms that the `paste | grep -n` output contains:

```text
1-based line     actual,predicted
57772            01
57774            01
57775            10
57778            10
57779            01
57781            10
```

There are many nearby mismatches; examples extend from at least line 57,747 through line 57,843 in a narrow inspected window. These are not asserted as physical event boundaries.

### Index ambiguity that must remain explicit

GNU `grep -n` reports **one-based line numbers**, so its line 57,775 maps to stored zero-based shot index 57,774. Elsewhere, the README explicitly says the first shot has index 0. Nevertheless, its informal phrase “near shot 57775” could refer to the approximate region rather than either exact convention. In fact, zero-based shot 57,775 is line 57,776 and has pair `00`.

Therefore:

- preserve the archive's row order;
- describe the event as an author-identified **region near 57,775**;
- predeclare a window for evaluation;
- do not silently set an exact physical changepoint \(\tau=57{,}775\);
- do not use decoder mismatch itself as a causal detector input if it is also the evaluation marker.

The README identifies a decoder-mismatch cluster associated with increased logical error, not a machine-readable timestamp for the physical onset of the high-energy event.

## Exact target schema

The authoritative metadata are in:

```text
repetition_code_bZ_d25_r50_center_5_5/properties.yml
repetition_code_bZ_d25_r50_center_5_5/circuit_ideal.stim
```

`properties.yml` reports:

| Property | Value |
|---|---:|
| basis | Z |
| distance | 25 |
| rounds | 50 |
| data qubits | 25 |
| measurement qubits | 24 |
| circuit qubits | 49 |
| shots | 500,000 |
| measurements per shot | 1,225 |
| sweep bits per shot | 25 |
| detectors per shot | 1,224 |
| observables per shot | 1 |

The following dimensions were independently confirmed from file sizes, line counts and circuit instructions:

| File | Meaning | Bits/shot or record | Bytes/shot | Total bytes |
|---|---|---:|---:|---:|
| `measurements.b8` | Raw hardware measurements | 1,225 bits | 154 | 77,000,000 |
| `sweep.b8` | Per-shot data-qubit initialization pattern | 25 bits | 4 | 2,000,000 |
| `detection_events.b8` | Derived detector fires | 1,224 bits | 153 | 76,500,000 |
| `obs_flips_actual.01` | Actual flip relative to the noiseless logical value | one ASCII bit plus LF | 2 | 1,000,000 |
| `obs_flips_predicted_by_correlated_matching.01` | Correlated-matching prediction | one ASCII bit plus LF | 2 | 1,000,000 |
| `obs_flips_predicted_by_pymatching.01` | PyMatching prediction | one ASCII bit plus LF | 2 | 1,000,000 |

For every `.b8` file, each shot is independently byte-aligned as specified by the official [Stim result formats](https://github.com/quantumlib/Stim/blob/main/doc/result_formats.md#b8). Within each byte, bit \(k\) is little-endian:

\[
x_k = \bigl(\mathrm{record}[\lfloor k/8\rfloor] \gg (k\bmod 8)\bigr)\mathbin{\&}1.
\]

Thus `measurements.b8` and `sweep.b8` each have seven unused padding bits at the end of every shot; `detection_events.b8` has none. A parser must discard padding per record, not concatenate the files into one continuous bit string.

For zero-based shot index \(s\), the byte offsets are:

\[
\begin{aligned}
o_{\rm meas}(s)&=154s,\\
o_{\rm sweep}(s)&=4s,\\
o_{\rm det}(s)&=153s,\\
o_{\rm obs}(s)&=2s \quad\text{for the ASCII `.01` files.}
\end{aligned}
\]

For example, zero-based indices 57,774 and 57,775 begin at:

| Index | measurement offset | detector offset | sweep offset | `.01` offset |
|---:|---:|---:|---:|---:|
| 57,774 | 8,897,196 | 8,839,422 | 231,096 | 115,548 |
| 57,775 | 8,897,350 | 8,839,575 | 231,100 | 115,550 |

## Circuit-order map

The ideal Stim circuit is the source of truth for bit semantics. It contains 49 `QUBIT_COORDS`, 51 `M` instructions, 1,225 measurement targets, 1,224 `DETECTOR` declarations and one `OBSERVABLE_INCLUDE(0)`.

### Measurement bits

The first 50 `M` instructions each measure these 24 ancilla qubits, in this target order:

```text
0,3,5,6,8,10,13,15,17,19,21,23,
25,27,29,31,33,35,38,40,42,43,45,48
```

Consequently, measurement bit `24*r + j` is target `j` in QEC round
`r=0,...,49`. The final 25 measurement bits, indices 1,200–1,224, are the data-qubit readout in this order:

```text
1,2,4,7,9,11,12,14,16,18,20,22,24,
26,28,30,32,34,36,37,39,41,44,46,47
```

Sweep-bit position `j` controls the `j`-th physical-qubit target in that same final-readout list. The sweep file is therefore required when regenerating detector events and observable flips; `measurements.b8` alone is insufficient.

### Detector bits and the boundary-order trap

There are 24 detector declarations at every circuit time coordinate
`t=0,...,50`. The stored detector-bit order is the **global declaration order in `circuit_ideal.stim`**:

\[
k=24t+j.
\]

For intermediate times `t=1,...,49`, spatial order is:

```text
(1,4),(2,5),(2,7),
(3,2),(3,4),(3,6),
(4,3),(4,5),(4,7),(4,9),
(5,2),(5,4),(5,6),(5,8),
(6,1),(6,3),(6,5),(6,7),
(7,4),(7,6),(7,8),
(8,3),(8,5),(9,6)
```

At both boundaries, `t=0` and `t=50`, rows 3, 5 and 7 occur in the reverse spatial order:

```text
(3,6),(3,4),(3,2)
(5,8),(5,6),(5,4),(5,2)
(7,8),(7,6),(7,4)
```

The other coordinates remain in the displayed positions. A naive reshape to
`[51,24]` followed by the assumption that column `j` always denotes the same spatial check is therefore wrong at the first and final boundary. Parse the coordinates from every `DETECTOR(...)` declaration or explicitly remap those two boundaries.

Semantically:

- `t=0` detectors use the first ancilla measurement boundary;
- `t=1,...,49` compare consecutive ancilla rounds;
- `t=50` combines final data measurements with the last ancilla round;
- the logical observable is the parity declared by the final
  `OBSERVABLE_INCLUDE(0)`.

The final-time detector bits and logical observable are not causally available until the final data readout has occurred.

## Files safe for causal parsing—and leakage hazards

- Use `circuit_ideal.stim`, `properties.yml`, `measurements.b8` and `sweep.b8` to establish raw semantics.
- `detection_events.b8` and `obs_flips_actual.01` are deterministic derived products; the README gives the exact `stim m2d` regeneration command.
- `circuit_noisy.stim` is an approximate Pauli noise model, not another hardware stream.
- `circuit_detector_error_model.dem` is derived from that noisy circuit and was used to configure the archived PyMatching and correlated-matching predictions.
- `pij_from_even_for_odd.dem` and `pij_from_odd_for_even.dem` were estimated from all even or odd detector rows. They include statistics from future rows relative to almost any online time and must be treated as full-stream/oracle artifacts—not causal calibration—unless their information is explicitly charged to the budget.
- The `.01` decoder predictions and actual observable flips are suitable as post hoc outcomes. They must not enter an unsupervised causal drift score if mismatch against them is the event reference.

One documentation typo is explicit: the README heading says
`obs_flips_actual.b8`, but the archive, its commands and its stated result format all use `obs_flips_actual.01`. The `.01` file is authoritative.

## Temporal ambiguities and parser contract

The archive preserves a canonical row order and uses it to reveal a localized high-energy event. It does **not** provide:

- wall-clock timestamps;
- an inter-shot duration;
- acquisition-batch or pause markers;
- calibration timestamps;
- an exact physical onset/recovery label for the high-energy event.

For Run 6:

1. Keep the original zero-based shot index and never shuffle before a chronological split.
2. Keep `(shot, round)` as two indices. A new shot reinitializes the logical experiment; do not treat `t=50` of shot \(s\) and `t=0` of shot \(s+1\) as an uninterrupted code trajectory.
3. If simulating round-by-round arrival, expose only detector groups whose circuit time has completed; do not expose the full 153-byte row at round 0.
4. Fit calibration, projections and thresholds only on a predeclared prefix ending before the event window.
5. Report delay in stored shots or circuit rounds, not seconds.
6. Treat “near 57,775” as an approximate external event region and publish sensitivity to the chosen window.

This pilot is now locally ready for a separately authorized benchmark, but the map itself supports no ECA-advantage claim.
