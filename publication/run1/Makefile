LATEXMK ?= latexmk

.PHONY: all clean distclean

all: main.pdf

main.pdf: main.tex references.bib generated/main_accuracy_table.tex \
	generated/full_accuracy_table.tex generated/resource_table.tex
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error main.tex

clean:
	$(LATEXMK) -c main.tex

distclean:
	$(LATEXMK) -C main.tex
