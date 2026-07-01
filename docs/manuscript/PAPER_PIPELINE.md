# Legacy v1 stable-ID scaffold pipeline

> **Status:** preserved for historical compatibility. It is not the evidence-first review-draft generator. New Core development is specified in `docs/manuscript/PAPER_PIPELINE_V2_CORE.md` and implemented by `scripts/paper_pipeline_core.py`.


## 1. Purpose

`scripts/paper_pipeline.py` keeps the manuscript graph, outline, paragraph blueprint, prose, appendix, LaTeX, figures, tables, PDF, and Overleaf ZIP synchronized. The reconciliation source is `docs/manuscript/paper_graph.yaml`.

The pipeline is bidirectional for registered semantic fields and conflict-intolerant. It never resolves two incompatible edits by silently choosing one.

## 2. Layers

```text
scientific authority (handoff / registry)
        ↓
stable Guidance + DRPO Strategy + Playbook
        ↓
paper_graph.yaml
   ↙          ↓            ↘
outline    blueprint      prose
                 ↓
        TeX + appendix + figures + tables
                 ↓
             PDF + Overleaf ZIP
```

Every manuscript block has a stable ID such as `INTRO-P03` or `APP-DRO-P01`.

## 3. Forward generation

```bash
python scripts/paper_pipeline.py all --repo-root . \
  --output paper/releases/DRPO_OVERLEAF_DRAFT_V092.zip
```

This performs deterministic rendering, validation, PDF compilation, and Overleaf packaging. Unavailable formal results remain explicit `TBD` fields.

A trusted semantic generator may be attached:

```bash
python scripts/paper_pipeline.py sync --repo-root . \
  --generator-cmd 'python /ABS/PATH/trusted_generator.py'
```

The command receives a JSON node on stdin and returns updated `blueprint` and `prose` fields. Third-party skill code is never executed by default.

## 4. Editing any layer

### Edit the outline

Change only the stable block in `docs/paper_rewrite_outline_v0_9_2.md`, then run:

```bash
python scripts/paper_pipeline.py sync --repo-root .
```

The graph imports the changed block and regenerates its blueprint, prose, TeX, and appendix projection.

### Edit the paragraph blueprint

Change the registered block in `docs/paper_rewrite_blueprint_v0_6.md` and run the same command. The graph imports topic sentence, moves, evidence use, and transition, then updates outline metadata and prose.

### Edit the prose

Each prose block contains structured fields before `**Body:**`: Claim, Reader question, Role, Logical moves, and Evidence use. Change the semantic field together with the body when a scientific point changes. `sync` imports those fields upstream and regenerates the corresponding outline and blueprint block. Pure wording edits below `**Body:**` remain prose-only.

This distinction is intentional: a sentence-level copyedit must not silently rewrite the scientific outline, while a changed claim must propagate.

### Apply a structured multi-layer change

```bash
python scripts/paper_pipeline.py apply-delta --repo-root . \
  --delta path/to/change.yaml
```

Example:

```yaml
changes:
- id: INTRO-P03
  claim: New approved claim
  reader_question: New reader question
  must_include:
  - first required move
  - second required move
  prose: Full replacement paragraph.
```

## 5. Conflict handling

If the same node is edited in more than one layer since the previous render, `sync` stops. Resolve explicitly:

```bash
python scripts/paper_pipeline.py sync --repo-root . --prefer outline
```

No automatic last-write-wins behavior is allowed.

## 6. Validation

```bash
python scripts/paper_pipeline.py validate --repo-root .
python scripts/manuscript_cascade.py validate-artifacts \
  --repo-root . --config docs/manuscript/hierarchy.yaml
pytest -q tests/test_paper_pipeline.py \
  tests/test_manuscript_guidance_review.py \
  tests/test_manuscript_live_hierarchy.py
```

Validation checks stable IDs, parent hashes, graph coverage, TeX node coverage, prohibited sequel framing, C-U1 terminology, evidence status, and missing generated artifacts.

## 7. Overleaf behavior

`paper/overleaf/` is the active project. `paper/overleaf/legacy_source/` preserves the uploaded source package for provenance but is excluded from the upload ZIP. The generated release contains `main.tex`, active sections, appendix, figures, tables, bibliography, styles, and compiled `main.pdf`. It has no local absolute paths.
