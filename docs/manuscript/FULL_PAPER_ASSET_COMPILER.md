# Full-paper asset compiler

`compile_full_manuscript.py` is the full downstream target for the existing manuscript graph. It does not create a second paper-writing system. It reuses `scripts/paper_pipeline.py render` for stable-ID outline/blueprint/prose projection, then adds four missing output responsibilities:

1. **Data and figures.** Registered result artifacts under `outputs/` are transformed into deterministic figures, numeric tables, and captions. The blueprint/graph decides the paragraph role; `full_paper_assets.yaml` binds each paragraph node to the generated evidence asset.
2. **Theory derivations.** Main-text theorem and proposition statements remain concise. Detailed derivations are inserted after the corresponding appendix nodes and are checked by statement/proof labels.
3. **Citations.** Every cited key must exist in `paper/overleaf/references.bib`; compilation uses BibTeX and fails on unresolved citations or references.
4. **Template-aware typesetting.** The release target is explicitly configured as the ICML 2026 two-column template. The build fails if the selected template or column contract is not active.

## Ownership rule

- The **paragraph blueprint** knows which evidence, figure, table, theorem, and citation a paragraph needs and what conclusion they support.
- The **prose layer** explains and interprets those values.
- Numeric values, plots, table cells, and caption tokens are generated from registered artifacts rather than copied by the language model.
- The **template layer** controls page geometry and publication formatting; it is not inferred from prose.

## Build

```bash
python3 scripts/compile_full_manuscript.py --repo-root .
```

The verified review PDF is written to:

```text
paper/releases/DRPO_FULL_REVIEW_V1.pdf
```

The current build includes:

- the original two conceptual figures;
- D-U1 E5 long-run reward and support-boundary figures plus a numeric table;
- C-U1 E3 targeted-intervention reward/failure figure plus a numeric table;
- C-U1 E4 phase and matched-near-retention control figures;
- bibliography and in-text citations;
- full appendix derivations for Theorem 1, the far-field proposition, Gaussian mean--variance dynamics, and categorical log-odds dynamics.

Hopper and Countdown formal result cells remain `TBD` because their registered formal experiments are not yet terminal-audited. The compiler never replaces missing formal evidence with pilot results.
