# DRPO KDD 2027 Stage-A manuscript

This directory is the self-contained current KDD 2027 Research Track
submission generated from the locked manuscript in `source_locked.tex`.

- Format: `\documentclass[sigconf,anonymous,review]{acmart}`
- Scope: current KDD submission source with the registered appendix result update
- Compiler: pdfLaTeX
- Canonical TeX environment: TeX Live 2025
- Figures: local `figures/` directory
- Bibliography: local `example_paper.bib` and `missing_references.bib`
- Release PDF: `paper/releases/DRPO_KDD2027_STAGE_A.pdf`
- Overleaf upload bundle: `paper/releases/DRPO_KDD2027_OVERLEAF.zip`

## Overleaf

Upload `DRPO_KDD2027_OVERLEAF.zip` as a new project. The ZIP is packaged with
`main.tex` at the project root, so no path edits are required. In project
settings select:

- Main document: `main.tex`
- Compiler: pdfLaTeX
- TeX Live version: 2025

## Repository build

From the repository root:

```bash
bash paper/kdd2027/build.sh
```

The build requires TeX Live 2025 and verifies the locked source hash,
character-equivalent manuscript content, six local figure assets, resolved
LaTeX references/citations, anonymous PDF text and metadata, US Letter page
size, embedded fonts, and a complete PNG render of every page. Page count is
recorded but not gated in Stage A.
