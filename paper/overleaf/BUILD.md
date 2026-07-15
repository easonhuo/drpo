# Build

The Overleaf entry point is `main.tex`; the full manuscript is in
`main_replacement.tex`.

Local build:

```bash
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

Run one additional `pdflatex` pass only if the log requests a reference
rerun. The included `algorithm.sty` and `algorithmic.sty` are local build
dependencies for the ICML style.
