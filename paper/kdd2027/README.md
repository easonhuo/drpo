# DRPO KDD 2027 Stage-A manuscript

This directory is the format-only KDD 2027 Research Track migration of the
locked manuscript in `source_locked.tex`.

- Format: `\documentclass[sigconf,anonymous,review]{acmart}`
- Scope: Stage A only; no page-limit compression or scientific-text editing
- Figures: reused from `paper/overleaf/figures/`
- Bibliography: reused from `paper/overleaf/example_paper.bib` and `paper/overleaf/missing_references.bib`
- Release PDF: `paper/releases/DRPO_KDD2027_STAGE_A.pdf`

Build from the repository root:

```bash
bash paper/kdd2027/build.sh
```

The build verifies the locked source hash, character-equivalent manuscript
content, five referenced figure assets, LaTeX references/citations, anonymous
PDF text and metadata, US Letter page size, embedded fonts, and a complete PNG
render of every page. Page count is recorded but not gated in Stage A.
