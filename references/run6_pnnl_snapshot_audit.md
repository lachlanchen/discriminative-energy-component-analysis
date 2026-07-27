# Run 6 PNNL/IBM snapshot audit

**Audit date:** 2026-07-27
**Scope:** the already extracted files under `experiments/data/run6/pnnl_ibm/`; no decoder or detector performance comparison was run.

## Executive verdict

The release supports a reproducible **cross-calibration-snapshot** or **hardware-domain-shift** experiment. It does **not**, by itself, support a claim of naturally observed continuous temporal drift or a quickest-change benchmark with a known change time.

There are useful repeated controls:

- 42 exact, oriented `(backend, d, rounds, basis, physical path)` groups recur under more than one distinct backend-property `last_update_date` after reconstructing the path from QASM. Of these, 31 are on `ibm_kingston` and 11 are on `ibm_pittsburgh`; none are on `ibm_fez`.
- The strongest long-span same-path examples cover about 28–41 days in backend-property dates, but their complete transpiled QASM changes across the long interval. They therefore mix calibration/hardware shift with circuit-compilation or parallel-layout changes.
- Only two cross-date configuration groups have byte-identical complete state-0 and state-1 QASM. They are both on Kingston and span only about 8 h 40 min and 56 min, respectively.
- No distance-11 exact-path/configuration group recurs across distinct `last_update_date` values.

The correct conservative statement is:

> We evaluate distribution shift between real IBM hardware measurement cohorts associated with different backend-property snapshots. For selected cohorts, the same oriented physical repetition-code path, code distance, syndrome depth and basis recur. The released artifacts do not contain execution timestamps or a verified global acquisition order, so we do not interpret these cohorts as a continuously monitored temporal-drift stream.

This audit establishes data semantics and candidate controls only. It provides no evidence yet that ECA, or any other detector, has an algorithmic advantage.

## 1. Provenance and release identity

The source is the author deposit [Zenodo 20768087](https://zenodo.org/records/20768087), DOI [10.5281/zenodo.20768087](https://doi.org/10.5281/zenodo.20768087), associated with [Stein et al., *Calibration-Conditioned FiLM Decoders for Low-Latency Decoding of Quantum Error Correction Evaluated on IBM Repetition-Code Experiments*](https://arxiv.org/abs/2601.16123). The release metadata identify version 0.1, Samuel Stein/PNNL as depositor, and a CC BY 4.0 license.

The extracted release contains:

| Backend | Hardware snapshots | Circuit shots | Per-chain samples |
|---|---:|---:|---:|
| `ibm_kingston` | 155 | 1,902,592 | 10,280,960 |
| `ibm_pittsburgh` | 156 | 1,712,128 | 9,273,344 |
| `ibm_fez` | 41 | 164,864 | 1,099,776 |
| **Total** | **352** | **3,779,584** | **20,654,080** |

Here a circuit shot measures several parallel repetition-code chains; “per-chain samples” multiplies the circuit-shot count by `n_chains`. Each snapshot contains two circuits, one for each prepared logical state, and `info.json["shots"]` is the number of shots **per circuit**.

There is a material release/paper mismatch:

- the arXiv manuscript reports 2,760,704 shots over 400 contiguous-chain calibration snapshots;
- Zenodo v0.1 reports 3,779,584 circuit shots over 352 hardware snapshots;
- the README says that paper splits can be recovered using `index.csv`, but no `index.csv` appears in the Zenodo file list, the three backend archives, or this extraction.

Consequently, the released v0.1 corpus must not be assumed to be numerically identical to the manuscript corpus, and the manuscript's exact train/validation/“one week later” split cannot be reconstructed unambiguously from the deposited files alone.

## 2. What the timestamps and ordering actually mean

The README describes `calibration.json` as IBM backend properties “captured at submission time.” Its top-level `last_update_date` is the date attached to the backend-properties object. It is **not** a job-submission, job-start, job-completion, circuit-execution, or shot timestamp.

| Available field or order | What it documents | What it does not document |
|---|---|---|
| `calibration.json:last_update_date` | Date attached to a backend-properties snapshot | Exact circuit execution time or a physical drift onset |
| Dates nested in qubit/gate/general properties | Last-update metadata for individual calibration quantities | A synchronized measurement of all properties |
| Folder `job_n` | An anonymized local label within one `d<D>_r<R>` directory | A backend-wide or global chronological rank |
| Row `n` in a `per_shot_cregs` array | Sample index in the returned array | Wall-clock time or a known cadence |
| QASM instruction order | Program order/dependencies in that compiled circuit | Order among separate hardware jobs |
| Zenodo publication/version date | Deposit history | Hardware acquisition chronology |

The top-level backend-property timestamps present in the extracted files are:

- **Fez (2):** `2025-08-25 17:03:11-07:00`, `2025-10-03 10:00:39-05:00`.
- **Kingston (12):** `2025-08-28 06:49:21-07:00`, `2025-08-28 15:29:33-07:00`, `2025-08-28 16:39:32-07:00`, `2025-08-29 10:09:52-07:00`, `2025-08-29 11:06:02-07:00`, `2025-08-30 12:49:00-04:00`, `2025-09-01 14:27:45-04:00`, `2025-09-04 14:05:29-04:00`, `2025-09-05 13:14:49-04:00`, `2025-10-02 10:10:32-05:00`, `2025-10-03 08:32:51-05:00`, `2025-10-08 09:14:51-05:00`.
- **Pittsburgh (6):** `2025-09-02 23:32:15-04:00`, `2025-09-05 14:29:14-04:00`, `2025-10-02 10:16:59-05:00`, `2025-10-02 10:57:52-05:00`, `2025-10-03 11:43:51-05:00`, `2025-10-08 09:28:09-05:00`.

The 352 directories contain 20 distinct timestamp strings but 21 distinct full calibration-file hashes: Kingston's `2025-08-29 10:09:52-07:00` timestamp occurs with two different calibration contents. Therefore even `last_update_date` is not a unique snapshot identifier. A robust identifier should include at least backend plus a hash of the normalized/full calibration JSON.

The paper is independent author evidence that selected Kingston experiments were performed on different chains about one week later. However, the anonymized v0.1 artifacts contain neither IBM Runtime job IDs nor execution timestamps nor machine-readable split labels. Sorting by `last_update_date` creates a **calibration-property-date order**, not a verified acquisition sequence. Concatenating such cohorts is permissible only as a **constructed changepoint from real-hardware samples**.

## 3. Physical paths must be reconstructed from QASM

The release README says that a register suffix lists the physical qubits used. That is usually, but not universally, true.

Across 2,051 parallel chain-register instances:

| Backend | Suffix agrees with both state QASMs | Suffix/QASM anomaly |
|---|---:|---:|
| Fez | 276 | 0 |
| Kingston | 839 | 8 |
| Pittsburgh | 852 | 76 |
| **Total** | **1,967 (95.9%)** | **84 (4.1%)** |

For each of the 84 anomalous instances, the suffix does not describe the physical measurement mapping in the corresponding QASM, and the actual state-0 and state-1 paths also differ. For example:

- `ibm_kingston/d7_r7/job_5` labels one pair of registers with the suffix
  `80_81_82_83_96_103_104_105_117_125_126_127_137`;
- state 0 actually measures the interleaved path
  `2-3-4-5-6-7-8-9-10-11-12-13-14`;
- state 1 actually measures
  `1-2-3-4-5-6-7-8-9-10-11-12-13`.

The QASM measurement assignments are therefore authoritative; the register suffix is only the JSON lookup label.

For each logical-state QASM and register label \(\rho\):

1. Parse `c_data_ρ[i] = measure $q` to obtain the ordered data-qubit map \(D_i=q\), \(0\le i<d\).
2. Parse `c_syndrome_ρ[t(d-1)+i] = measure $q` to obtain ancilla map \(A_{t,i}=q\).
3. Verify that the ancilla assignment is stable across rounds. It is stable in all 2,051 instances audited here.
4. Reconstruct the oriented physical path

\[
P=(D_0,A_{0,0},D_1,A_{0,1},\ldots,A_{0,d-2},D_{d-1}).
\]

5. Retain the state-0 and state-1 maps separately. Combine logical states into one physical-chain cohort only when the maps agree, or explicitly model the state-dependent circuit difference.

Path reversal was **not** merged in the exact counts below. Orientation can change data/stabilizer indices, gates and scheduling and therefore should not be discarded automatically.

## 4. Strongest exact-path candidates

The table keeps only groups in which:

- backend, distance \(d\), rounds \(r\), basis and oriented QASM-derived path agree;
- that path agrees in the logical-state-0 and logical-state-1 QASMs;
- at least two distinct `last_update_date` values occur.

“Chain shots” is the sum over both prepared logical-state circuits for that one path. It is a sample-count descriptor, not a count of independent time points.

| Backend/configuration | Exact oriented physical path | Distinct property dates / span | Snapshot labels | Chain shots | Complete-QASM control |
|---|---|---:|---|---:|---|
| Kingston, \(d=3,r=3,Z\) | `140-141-142-143-144` | 4 / 41.02 d | jobs 4, 5, 6, 7 | 38,912 | jobs 4–5 identical; later QASM differs |
| Kingston, \(d=5,r=5,Z\) | `0-1-2-3-16-23-22-21-36` | 3 / 34.85 d | jobs 3, 4, 5, 6, 7 | 45,056 | jobs 3–6 identical; job 7 differs |
| Kingston, \(d=3,r=3,Z\) | `0-1-2-3-16` | 3 / 35.99 d | jobs 4, 5, 6 | 22,528 | jobs 4–5 identical; job 6 differs |
| Kingston, \(d=9,r=9,Z\) | `118-129-128-127-137-147-148-149-150-151-152-153-154-155-139-135-134` | 2 / 27.85 d | jobs 4, 5, 6 | 36,864 | QASM differs across dates |
| Pittsburgh, \(d=5,r=5,Z\) | `140-141-142-143-144-145-146-147-148` | 2 / 35.46 d | jobs 4, 6 | 24,576 | QASM differs |
| Pittsburgh, \(d=3,r=3,Z\) | `151-152-153-154-155` | 2 / 35.46 d | jobs 4, 6 | 24,576 | QASM differs |
| Pittsburgh, \(d=5,r=1,X\) | `18-11-12-13-14-15-19-35-34` | 2 / 5.97 d | jobs 1, 3 | 20,480 | QASM differs |
| Pittsburgh, \(d=5,r=5,X\) | `34-35-19-15-14-13-12-11-18` | 2 / 5.97 d | jobs 1, 3 | 20,480 | QASM differs |
| Kingston, \(d=5,r=3,X\) | `8-7-6-5-4-3-2-1-0` | 2 / 5.03 d | jobs 1, 2 | 20,480 | QASM differs |
| Pittsburgh, \(d=7,r=3,Z\) | `80-81-82-83-84-85-77-65-66-67-57-47-46` | 2 / 4.91 d | jobs 6, 7 | 20,480 | QASM differs |
| Pittsburgh, \(d=7,r=7,Z\) | `138-151-150-149-148-147-146-145-144-143-142-141-140` | 2 / 4.91 d | jobs 7, 9 | 20,480 | QASM differs |

These are the strongest candidates by date span and/or depth. The full audit found 42 repeated exact groups: 31 Kingston and 11 Pittsburgh.

### The only byte-identical complete-QASM cross-date controls

Hashing the complete `circuit_state0.qasm` and `circuit_state1.qasm` pair leaves only:

| Backend/configuration | Snapshot labels | Distinct property timestamps | Parallel exact paths | Interpretation |
|---|---|---|---:|---|
| Kingston, \(d=3,r=3,Z\) | jobs 4 and 5 | `2025-08-28 06:49:21-07:00` and `2025-08-28 15:29:33-07:00` | 10 | Same compiled two-circuit workload, property dates separated by 8 h 40 min 12 s |
| Kingston, \(d=5,r=5,Z\) | jobs 3–6 | `2025-08-29 10:09:52-07:00` and `2025-08-29 11:06:02-07:00` | 11 | Same compiled two-circuit workload, property dates separated by 56 min 10 s; jobs 3–5 share the earlier date |

These are the cleanest circuit-controlled calibration-cohort comparisons, but the recorded time is still a backend-property timestamp rather than execution time. Conversely, the month-scale exact-path groups are useful robustness tests but are not fully circuit controlled.

## 5. Reconstructing syndrome tensors and detection events

`bitstrings.json` is an array with one entry for each logical-state circuit. Select entries by `metadata.logical_state`, not by assuming array order. Each entry contains:

- `metadata.logical_state`, `metadata.n_syndrome_rounds`, and `metadata.basis`;
- paired `c_data_ρ` and `c_syndrome_ρ` arrays in `per_shot_cregs`;
- one row per circuit shot.

For a distance-\(d\), \(r\)-round chain:

- the data array has shape \([N,d]\);
- the syndrome array has shape \([N,r(d-1)]\).

The QASM classical indices establish the flattening convention. For shot \(n\),

\[
S_{n,t,i}
=
\texttt{syndrome}[n,\;t(d-1)+i],
\qquad
0\le t<r,\quad 0\le i<d-1.
\]

The paper's detection-event definition is

\[
\chi_{n,0,i}=S_{n,0,i},
\qquad
\chi_{n,t,i}=S_{n,t,i}\oplus S_{n,t-1,i}
\quad (t\ge 1),
\]

which is equivalent to \(s_{0,i}=0\) in its one-indexed notation. The decoder input therefore has shape

\[
\chi_n\in\{0,1\}^{r\times(d-1)}.
\]

Do not silently append a terminal detector. A conventional offline terminal boundary can be formed from final data

\[
F_{n,i}=M_{n,i}\oplus M_{n,i+1},\qquad
\chi^{\mathrm{terminal}}_{n,i}=F_{n,i}\oplus S_{n,r-1,i},
\]

but the manuscript's neural input is explicitly \(r\times(d-1)\) and does not include final data. Adding the terminal boundary defines a different observation model and must be declared.

Additional alignment rules:

- Use `c_data_ρ[i] = measure $q` in the matching logical-state QASM to map data column \(i\) to a physical qubit.
- Use the JSON register label \(\rho\) only to locate the corresponding arrays; do not infer the physical map from its suffix.
- Do not pair state-0 shot \(n\) with state-1 shot \(n\). They are separate circuit executions.
- Rows with the same \(n\) across parallel registers in one logical-state entry are parallel outputs from the same circuit shot, as intended by the release layout, but they still have no wall-clock timestamp.
- Do not concatenate row order across snapshots and describe it as a continuous stream.

## 6. Reconstructing per-qubit targets and logical errors

Let the prepared logical value from metadata be \(\ell\in\{0,1\}\), and let

\[
Q_{n,i}=\ell,\qquad
M_{n,i}=\texttt{data}[n,i].
\]

The X-basis circuit rotates the final X measurement into the recorded computational basis, so the same recorded-bit convention applies. The paper's offline per-qubit target is

\[
Y_{n,i}=Q_{n,i}\oplus M_{n,i}.
\]

For a decoder correction \(C_n\in\{0,1\}^d\), define residual errors

\[
R_n=Y_n\oplus C_n.
\]

Because all audited distances are odd, a corrected logical error occurs exactly when

\[
\mathbb{1}_{\mathrm{LE},n}
=
\mathbb{1}\!\left[
\sum_{i=0}^{d-1}R_{n,i}>\frac d2
\right].
\]

Equivalently,

\[
\mathbb{1}_{\mathrm{LE},n}
=
\mathbb{1}\!\left[
\operatorname{majority}(M_n\oplus C_n)\ne\ell
\right].
\]

Setting \(C_n=0\) gives the raw-majority logical-error indicator. The final data \(M_n\), prepared value \(\ell\), and target \(Y_n\) are evaluation labels. A faithful reproduction of the manuscript's decoder must not expose them as input: its online observation is \(\chi_n\), with calibration data supplied separately.

## 7. What temporal claim is and is not supported

### Supported by the release

- Real IBM hardware bitstrings are associated with multiple backend-property snapshots.
- Selected exact physical paths and code configurations recur under distinct `last_update_date` values.
- Cohorts can be compared for cross-calibration robustness, calibration-conditioned transfer, and hardware distribution shift.
- A synthetic boundary can be constructed by concatenating held-out cohorts, provided it is called a **constructed changepoint between real-hardware snapshot cohorts**.
- The two byte-identical-QASM Kingston groups offer limited, sub-day circuit-controlled snapshot comparisons.

### Not supported by the release alone

- A continuously sampled hardware time series.
- Exact job execution or shot times.
- The claim that `last_update_date` is the execution time.
- A global chronology derived from `job_n`.
- A naturally observed change time, drift onset, average run length, or physical detection delay.
- The claim that one exact chain was continuously monitored over days or weeks.
- Treating adjacent JSON rows as measurements separated by a known wall-clock interval.
- An exact reconstruction of the paper's train/validation/one-week-later partition.
- Attribution of any observed cohort difference solely to calibration drift when transpiled QASM, parallel-chain composition, shot count, or state-dependent physical mapping also changes.

Recommended publication wording:

> The benchmark uses anonymized real-hardware repetition-code cohorts from three IBM processors. We identify cohorts by a hash of the complete calibration record and reconstruct each physical chain from state-specific QASM measurement assignments. The reported date is the backend-properties `last_update_date`, not a circuit-execution timestamp. Long-span results therefore quantify cross-snapshot distribution shift rather than natural online drift. Where we concatenate cohorts, the boundary is constructed and known by design.

If acquisition timestamps or the missing index/split manifest later become available from the authors, a genuine temporal audit can be reopened. Until then, temporal words such as “later” should be attributed to the manuscript, not inferred from folder order.

## 8. License and redistribution caveat

Zenodo marks the deposited dataset [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Redistribution and adaptation therefore require appropriate attribution, a link to the license, and an indication of changes. Preserve at minimum the dataset title, Samuel Stein/PNNL attribution, DOI `10.5281/zenodo.20768087`, release version, and a statement identifying generated detection events, parsed paths, filters, hashes, or train/test manifests as derived artifacts.

The deposit is an author-released dataset containing measurements from named IBM processors; it should be described as “the Stein et al. PNNL deposit of IBM-hardware experiments,” not as an official IBM dataset or IBM endorsement. CC BY does not grant rights in IBM names or trademarks. Do not restore or attempt to infer anonymized Runtime job identities, and do not imply that the Zenodo license establishes a broader right to redistribute material obtained independently through IBM services.

For repository hygiene, prefer scripts, checksums, manifests and download instructions over committing duplicate archives. If raw files are redistributed, retain their original directory structure, DOI/provenance notice, CC BY 4.0 notice, and a clear separation between untouched source data and this project's derived outputs.
