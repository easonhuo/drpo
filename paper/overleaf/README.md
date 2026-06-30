# DRPO Overleaf project

Upload the contents of the generated release ZIP in `paper/releases/` directly to Overleaf. `main.tex` is the root document.

The active TeX is generated from `docs/manuscript/paper_graph.yaml`; edit a stable manuscript layer and run `scripts/paper_pipeline.py sync` rather than hand-editing generated section files. The original user-supplied ICML/Overleaf package is preserved under `legacy_source/` for provenance and is excluded from the upload release.

Local build:

```bash
./build.sh
```
