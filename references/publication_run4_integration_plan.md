# Revision Plan: global — integrate Run 3 and Run 4 into the Run 4 manuscript

## 1. Task Metadata

- Plan file: `references/publication_run4_integration_plan.md`
- Created: 2026-07-27
- Task origin: user-requested global manuscript extension
- Revision stage: new Run 4 working paper built from the immutable Run 3 paper
- Baseline: `publication/run3/main.tex` and
  `publication/run3/main.pdf`
- Active target: `publication/run4/main.tex`
- Other target files:
  - `publication/run4/references.bib`
  - `publication/run4/Makefile`
  - `publication/run4/README.md`
  - `publication/Makefile`
  - `publication/README.md`
  - repository README/citation metadata only after the PDF is verified
- Build command: `make -C publication/run4`
- Status: executed and verified

## 2. Source Motivation

The user asked to update a Run 4 PDF that contains the Run 3 theory and
experiments, while preserving the following boundary:

> The result cannot claim superiority over Helstrom, a Wilson-loop oracle, or
> methods given the same parity feature; quantum speedup, universal sample
> efficiency, scalable advantage, discovery of Wilson loops, a new toric-code
> theorem, string theory, or holographic duality are out of scope.

The completed Run 4 calculation now replaces Run 3's “future lattice-gauge
calculation” with an exact finite \(L=3\) toric-code / \(D(\mathbb Z_2)\)
benchmark.

## 3. Problem Diagnosis

The active Run 3 paper is a four-page paper with the correct accessible
observable theorem and four controlled applications, but:

1. its abstract, contribution list, QFT section, and conclusion still describe
   the gauge calculation as future work;
2. it contains no Run 4 local-blindness, Wilson-loop, correct/wrong-twirl, or
   robustness results;
3. the Run 3 bibliography has an incorrect DOI for Hiai--Mosonyi--Hayashi
   (`10.1063/1.3247342` rather than `10.1063/1.3234186`);
4. the cyclic \(O(d\log d)\)/\(O(d)\) statement is an algorithmic special case,
   but the current shipped convenience path still materializes dense
   matrices; the manuscript must separate theoretical structure from current
   implementation cost;
5. the paper needs an explicit gauge-Hilbert-factorization/electric-center
   convention and a sharper distinction among global charge sectors,
   gauge-boundary flux sectors, and torus topological holonomies.

## 4. Intended Scope

### Files allowed to change

- `publication/run4/*` — new active Run 4 paper, bibliography, build file, and
  paper-specific README.
- `publication/Makefile` and `publication/README.md` — register Run 4 after its
  PDF builds.
- `README.md`, `README.zh-Hans.md`, and `CITATION.cff` — update paper links and
  version text only after PDF verification.
- This plan file — record actual build and PDF audit results.

### Locations/content allowed in the Run 4 paper

- Title, abstract, keywords, introduction, contribution list.
- The Run 3 unrestricted, conditional-expectation, group-twirl, sector, and
  Fourier theory.
- A new locality/gauge/topological-sector subsection.
- A concise preservation of the Run 3 translation, Ising, chemistry, and robot
  evidence.
- A new exact Run 4 experimental-design and results subsection.
- One Run 4 result table and the existing Run 4 robustness figure.
- A revised limitations/future-work section identifying Run 5 as an
  equal-budget surface-code syndrome-drift test.
- Corrected and expanded primary bibliography.

### Explicitly forbidden changes

- Do not edit or overwrite `publication/run3/*`; it is the immutable baseline.
- Do not change Run 1–3 experiment data or manifests.
- Do not import unexecuted Run 5 numbers or imply that Run 5 exists.
- Do not claim a new Helstrom, conditional-expectation, toric-code, Wilson-loop,
  QEC, group-testing, symmetry-resolved-entanglement, string, or holography
  theorem.
- Do not call one exact representative state description “one-shot quantum
  training” or imply tomography from one physical copy.
- Do not call the ordinary link-qubit partial trace a unique gauge-theory RDM.
- Do not claim practical sample or compute advantage from unequal calibration
  budgets in earlier runs.

## 5. Proposed Edits

| Step | Location | Edit | Rationale | Acceptance criterion |
|---|---|---|---|---|
| 1 | Run 4 title/abstract | Retain SAOC theorem and summarize the exact local no-go/Wilson recovery result | Make the paper genuinely contain Run 3 + Run 4 | Abstract reports only values in committed summaries |
| 2 | Introduction/contributions | Replace the future gauge calculation with a completed finite benchmark and explicit prior-art boundary | Prevent novelty inflation | Wilson/local indistinguishability credited as established |
| 3 | Theory | Add \(D_{\mathcal A}\), correctable-region condition \(POP=cP\), and logical nuisance twirl | Connect accessible algebra to the exact experiment | Equations reproduce `summary.json` predictions |
| 4 | Experimental design | Specify 18 link qubits, stabilizers, logical sectors, exhaustive Pauli/RDM scans, and extended-link prescription | Ensure reproducibility and gauge precision | All counts match raw CSVs |
| 5 | Results | Add table: 1,431 blind sub-distance Paulis; 3 of 22,032 weight-3 discriminative; success 0.50/0.75/1.00 controls | Put the central evidence in the PDF | Table values match committed outputs |
| 6 | Figure | Include `experiments/run4/results/topological_flux/robustness.pdf` | Show logical mixing and readout calibration | Caption states cost/prior caveats |
| 7 | Boundaries | State observable-access and symmetry-prior advantage, but tie with Wilson/Helstrom and same-feature threshold | Answer the user's advantage question accurately | Prohibited claims appear explicitly |
| 8 | Prior work | Add Kitaev, Bravyi--Hastings--Michalakis, Casini--Huerta--Rosabal, Donnelly, Bridgeman--Flammia--Poulin, Che et al., and 2026 eSCD; correct Hiai DOI | Establish novelty boundary | No missing citations; BibTeX builds cleanly |
| 9 | Build integration | Add Run 4 target to publication Makefile/README only after clean build | Keep repository navigation current | Root publication build finds Run 4 |
| 10 | PDF audit | Inspect extracted text and rendered pages; check equations, table, figures, references, overflow, and claim wording | Source-only checks are insufficient | PDF page count and locations recorded below |

## 6. Response-Letter Impact

Not applicable. This is a new working-paper run, not a resubmission or a
response to the rejected TCAS-II manuscript. No response letter will be
created.

## 7. Verification Plan

- Run `make -C publication/run4`.
- Run `make -C publication`.
- Require zero undefined references/citations and zero LaTeX build errors.
- Check the log for overfull boxes and record any remaining warnings.
- Use `pdfinfo` and `pdftotext -layout` on `publication/run4/main.pdf`.
- Render every PDF page to PNG and inspect the page images.
- Check every numerical claim against:
  - `experiments/run3/results/*/summary.json`
  - `experiments/run4/results/topological_flux/summary.json`
  - Run 4 CSV tables.
- Run repository tests and scoped Ruff after source changes.
- Baseline/redline:
  - `publication/run3/main.tex` is the preserved baseline.
  - `latexdiff` is not installed at plan time. If it remains unavailable,
    retain a textual `git diff --no-index` audit and record redline as
    unavailable rather than inventing one.

## 8. Execution Notes

- Files changed:
  - created `publication/run4/{main.tex,references.bib,Makefile,README.md,main.pdf}`;
  - registered Run 4 in `publication/{Makefile,README.md}`;
  - linked the paper from both repository READMEs and updated the compiled-paper
    count in `CITATION.cff`;
  - clarified the historical `stabilizer_only` implementation label in the
    Run 4 advantage audit;
  - did not modify any file under `publication/run3/`.
- Deviations from plan:
  - the archived `IEEEtran.cls` is available, but `IEEEtran.bst` is not
    installed. The paper therefore uses the standard `unsrt` bibliography
    style rather than silently vendoring an unverified style file;
  - the robustness plot is one column rather than `figure*`, which avoids a
    poor page break and remains legible in the rendered PDF.
- Build results:
  - `make -C publication/run4` and `make -C publication` complete
    successfully;
  - the final Run 4 log contains no overfull boxes, undefined references, or
    undefined citations;
  - repository validation reports 41 tests passed and scoped Ruff checks
    passed.
- PDF locations verified:
  - `publication/run4/main.pdf`, five pages;
  - all pages were rasterized and visually inspected, including the equation
    layout, two result tables, five figures, claim-boundary section, and
    bibliography;
  - `pdftotext -layout` confirms the numerical results and explicit nonclaims
    are present in the compiled artifact.
- Redline status:
  - `publication/run3/main.tex` and its PDF remain the immutable baseline;
  - `latexdiff` is unavailable in the local toolchain, so no redline artifact
    is fabricated. The source-level baseline remains auditable with
    `git diff --no-index publication/run3/main.tex publication/run4/main.tex`.

## 9. Final Status

- Manuscript compiled: yes.
- Root publication build compiled: yes.
- PDF visually inspected: yes.
- Response letter: not applicable.
- Redline generated: no; `latexdiff` unavailable, immutable baseline retained.
- Git commit: pending.
- Remaining issues:
  - final authorship, affiliations, acknowledgments, and target venue require
    human confirmation before submission;
  - the bibliography-style fallback should be replaced with the selected
    venue's official style at submission time.
