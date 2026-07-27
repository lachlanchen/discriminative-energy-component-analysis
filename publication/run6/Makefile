LATEXMK ?= latexmk
PYTHON ?= ../../.venv/bin/python
PYTEST ?= ../../.venv/bin/pytest
DETECTOR_MANIFEST ?= ../../experiments/run6/results/google_detector/detector_freeze_manifest.json
FREEZE_RATIFICATION ?= ../../experiments/run6/freeze_ratification.json
REPAIR_MANIFEST ?= ../../experiments/run6/repair_manifest.json
REPAIR_RATIFICATION ?= ../../experiments/run6/repair_ratification.json
RANDOMIZATION_MANIFEST ?= ../../experiments/run6/results/google_randomization_repair1/merged/randomization_manifest.json
PNNL_MANIFEST ?= ../../experiments/run6/results/pnnl_snapshot/results_manifest.json
PITTSBURGH_MANIFEST ?= ../../experiments/run6/configs/pnnl_pittsburgh_locked.json
OUTCOME_MANIFEST ?= ../../experiments/run6/results/google_outcomes/outcome_manifest.json
BUNDLE_DIR ?= generated
GENERATED_INPUTS := $(wildcard $(BUNDLE_DIR)/*.tex $(BUNDLE_DIR)/*.pdf $(BUNDLE_DIR)/*.json)

.PHONY: all audit final-audit verify-generated check-placeholders expect-placeholders test clean

all: main.pdf

main.pdf: main.tex references.bib $(GENERATED_INPUTS)
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error main.tex

# This target is intentionally red before empirical insertion.
check-placeholders:
	$(PYTHON) check_placeholders.py

expect-placeholders:
	$(PYTHON) check_placeholders.py --expect-pending

audit: main.pdf expect-placeholders
	@! grep -E "LaTeX Warning: (Citation|Reference).*undefined|There were undefined references|There were undefined citations" main.log
	@! grep -F "Overfull \\hbox" main.log
	@pdftotext main.pdf - | grep -q "TBD("
	@printf '%s\n' "PDF audit passed: references resolve and pre-results markers are visible."

verify-generated:
	$(PYTHON) verify_generated_bundle.py \
		--detector-manifest $(DETECTOR_MANIFEST) \
		--freeze-ratification $(FREEZE_RATIFICATION) \
		--repair-manifest $(REPAIR_MANIFEST) \
		--repair-ratification $(REPAIR_RATIFICATION) \
		--randomization-manifest $(RANDOMIZATION_MANIFEST) \
		--pnnl-manifest $(PNNL_MANIFEST) \
		--pittsburgh-manifest $(PITTSBURGH_MANIFEST) \
		--outcome-manifest $(OUTCOME_MANIFEST) \
		--bundle-dir $(BUNDLE_DIR) \
		--manuscript main.tex

final-audit: verify-generated
	$(LATEXMK) -g -pdf -interaction=nonstopmode -halt-on-error main.tex
	$(PYTHON) check_placeholders.py
	@! grep -E "LaTeX Warning: (Citation|Reference).*undefined|There were undefined references|There were undefined citations" main.log
	@! grep -F "Overfull \\hbox" main.log
	@! pdftotext main.pdf - | grep -q "TBD("
	@printf '%s\n' "Final PDF audit passed: references resolve and no result markers remain."

test:
	PYTHONPATH=../../experiments $(PYTEST) -q \
		analysis/test_extract_results.py \
		test_verify_generated_bundle.py

clean:
	$(LATEXMK) -C main.tex
