LATEXMK ?= latexmk
TEXINPUTS := ../../references/_Rongzhou___ISCAS2025__Eigen_Component_Analysis__A_Quantum_Theory_Inspired_Linear_Model:$(TEXINPUTS)
export TEXINPUTS

.PHONY: all clean

all: main.pdf

main.pdf: main.tex references.bib
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error main.tex

clean:
	$(LATEXMK) -C main.tex
