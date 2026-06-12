# ── Thesis Makefile ─────────────────────────────────────────
# Usage (from thesis_latex/):
#   make          → full build (xelatex + biber + xelatex × 2)
#   make quick    → single xelatex pass (fast, no bib/refs update)
#   make clean    → remove all build artifacts
#   make open     → open thesis_ch1-4.pdf in Preview

JOBNAME  = thesis_ch1-4
SRCDIR   = src
BUILDDIR = build
PDF      = $(BUILDDIR)/$(JOBNAME).pdf
XELATEX  = xelatex -interaction=nonstopmode -jobname=$(JOBNAME) -output-directory=../$(BUILDDIR)

.PHONY: all quick clean open mkdirs

mkdirs:
	mkdir -p \
		$(BUILDDIR)/chapters/ch00_frontmatter \
		$(BUILDDIR)/chapters/ch01_introduction \
		$(BUILDDIR)/chapters/ch02_background \
		$(BUILDDIR)/chapters/ch03_data_analysis \
		$(BUILDDIR)/chapters/ch04_implementation \
		$(BUILDDIR)/chapters/ch05_contribution \
		$(BUILDDIR)/chapters/ch06_experiments \
		$(BUILDDIR)/chapters/ch07_conclusion \
		$(BUILDDIR)/chapters/appendices

all: mkdirs
	cd $(SRCDIR) && $(XELATEX) main.tex
	cd $(SRCDIR) && biber ../$(BUILDDIR)/$(JOBNAME)
	cd $(SRCDIR) && $(XELATEX) main.tex
	cd $(SRCDIR) && $(XELATEX) main.tex

quick: mkdirs
	cd $(SRCDIR) && $(XELATEX) main.tex

clean:
	rm -rf $(BUILDDIR)

open:
	open $(PDF)
