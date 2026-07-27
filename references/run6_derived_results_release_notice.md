# Run 6 derived-results data attribution and modification notice

Release: **v6.0.0** of *Observable Contrast Research: From Eigen-Components
to Additive and Symmetry-Resolved Physical Witnesses*, by Rongzhou
(Lachlan) Chen.

Repository:
<https://github.com/lachlanchen/discriminative-energy-component-analysis>

Locked conclusion:

> **No demonstrated S-PACE algorithmic advantage.**

The Run 6 data artifacts distributed with this release are project-generated
derived or adapted outputs. No original Zenodo archive or complete upstream
measurement/bitstring payload is redistributed. The Google outcome bundle
does include an adapted 20,000-shot join over the predeclared
`[40000,60000)` interval: decoded actual/predicted observable-flip bits are
placed beside project-generated mismatch labels and frozen-manifest hashes.
Those row-level adapted data remain subject to the Google source attribution
and CC BY 4.0 terms below. Obtain the complete source data directly from the
records below.

## Source data

1. **Google Quantum AI Team (2022)**, *Data for “Suppressing quantum errors
   by scaling a surface code logical qubit”*. Zenodo record 6804040.
   DOI: <https://doi.org/10.5281/zenodo.6804040>
   Record: <https://zenodo.org/records/6804040>
   License: [Creative Commons Attribution 4.0 International
   (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)

2. **Samuel Stein / Pacific Northwest National Laboratory (PNNL) (2026)**,
   *Calibration-Conditioned FiLM Decoders for Low-Latency Decoding of
   Quantum Error Correction Evaluated on IBM Repetition-Code
   Experiments—Datasets*, version 0.1. Zenodo record 20768087.
   DOI: <https://doi.org/10.5281/zenodo.20768087>
   Record: <https://zenodo.org/records/20768087>
   License: [Creative Commons Attribution 4.0 International
   (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)
   Associated work: Stein et al.,
   <https://arxiv.org/abs/2601.16123>

## Changes made

For the Google source, this project parsed the bit-packed detector records,
mapped circuit-declaration order into consistent physical-check and
round-role coordinates, preserved archive shot order, applied predeclared
causal partitions and event windows, joined the selected outcome interval to
two decoder-prediction streams, and generated project-specific mismatch
labels, hashes, manifests, scores, and summaries.

For the Stein/PNNL source, this project parsed syndrome registers,
reconstructed oriented physical paths from the state-specific QASM
measurement assignments, formed syndrome-change detection events, selected
predeclared Pittsburgh snapshot cohorts, constructed paired pre/post cohort
boundaries, and generated project-specific hashes, manifests, scores, and
summaries.

These parsing, selection, transformation, aggregation, and analysis steps are
modifications. The resulting files are not unchanged copies of either source
dataset and are not official products of the upstream creators. Attribution
does not imply that any upstream creator endorses this project, its results,
or its modifications.

## IBM notice

The Stein/PNNL source is an author-released PNNL deposit of experiments
performed on named IBM processors; it is not an official IBM dataset. No IBM
endorsement is claimed or implied. Neither CC BY 4.0 nor this notice grants
rights in IBM names or trademarks, or rights to redistribute material
obtained independently through IBM services. IBM names are used only to
identify the experimental hardware described by the source deposit.

## Citation for these derived outputs

Chen, Rongzhou (Lachlan Chen). *Observable Contrast Research: From
Eigen-Components to Additive and Symmetry-Resolved Physical Witnesses*,
version 6.0.0, 2026.
<https://github.com/lachlanchen/discriminative-energy-component-analysis>

## Local provenance sources

- Google record identity, DOI, checksum, and license:
  [`run6_google2022_data_map.md`](run6_google2022_data_map.md).
- Google parsing, coordinate mapping, and archive semantics:
  [`run6_google2022_data_map.md`](run6_google2022_data_map.md) and
  [`run6_real_qec_preregistered_plan.md`](run6_real_qec_preregistered_plan.md).
- PNNL record identity, release version, DOI, associated work, data semantics,
  and license:
  [`run6_pnnl_snapshot_audit.md`](run6_pnnl_snapshot_audit.md).
- Frozen Pittsburgh cohort selection and constructed-boundary interpretation:
  [`run6_pnnl_locked_manifest_recommendations.md`](run6_pnnl_locked_manifest_recommendations.md).

The assertion that no original archive or complete raw source payload is
redistributed must also be verified against the final release-asset
inventory; this notice alone is not evidence of archive contents. The
adapted Google outcome join described above must not be mislabeled as a
purely aggregate artifact.
