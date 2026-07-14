# Active manuscript system

## Authority and writing layers

1. `docs/handoff.md` and `experiments/registry.yaml` own scientific conclusions, experiment status, frozen protocols, and execution order.
2. `docs/manuscript/RL_PAPER_WRITING_GUIDANCE.md` is the slow-moving cross-paper quality standard.
3. `docs/manuscript/DRPO_MANUSCRIPT_STRATEGY.md` owns the current DRPO-specific story and hard identity constraint: this is one manuscript being rewritten, never an old paper plus a sequel.
4. `docs/manuscript/RL_PAPER_WRITING_PLAYBOOK.md` is the detailed operational handbook distilled from high-quality RL papers and audited external skills.
5. `docs/manuscript/RL_WRITING_CORPUS_NOTES.md` records source reading and retained or rejected techniques.
6. `docs/manuscript/paper_graph.yaml` is the stable-ID reconciliation source for outline, blueprint, prose, appendix, TeX, figures, tables, PDF, and Overleaf package.
7. `docs/manuscript/PAPER_WRITING_OPTIMIZATION_ITERATION_LOG.md` records manuscript-process incidents, approved optimization designs, cost budgets, implementation status, and measured outcomes. It is an engineering iteration record, not a scientific or writing-rule authority.

## Current review candidate

- merge audit: `docs/manuscript/V092_MERGE_LEDGER.md`;
- claim evidence: `docs/manuscript/claim_evidence_matrix.yaml`;
- outline: `docs/paper_rewrite_outline_v0_9_2.md`;
- paragraph blueprint: `docs/paper_rewrite_blueprint_v0_6.md`;
- prose draft: `docs/paper_rewrite_prose_v0_1.md`;
- active Overleaf source: `paper/overleaf/`;
- compiled PDF: `paper/overleaf/main.pdf`;
- direct-upload release: `paper/releases/DRPO_OVERLEAF_DRAFT_V092.zip`.

`v0.9.2` is a review candidate, not automatically canonical. It mechanically merges the user-approved v0.9-review, the useful corrections in repository v0.9, and the confirmed non-buggy refinements in v0.9.1. The merge ledger records accepted and rejected changes.

Historical artifacts remain in the repository and must not be destructively deleted.

## Generation and synchronization

See `docs/manuscript/PAPER_PIPELINE.md`.

```bash
python scripts/paper_pipeline.py all --repo-root . \
  --output paper/releases/DRPO_OVERLEAF_DRAFT_V092.zip
```

A semantic change is made through a stable block's structured metadata or an explicit delta. Wording-only prose changes remain local. Conflicting edits stop the pipeline.

## Validation

```bash
python scripts/paper_pipeline.py validate --repo-root .
python scripts/manuscript_cascade.py validate-artifacts \
  --repo-root . --config docs/manuscript/hierarchy.yaml
python scripts/manuscript_cascade.py validate-issue \
  --config docs/manuscript/hierarchy.yaml \
  --issue docs/manuscript/issues/PAPER-V092-MERGE-PIPELINE-01.yaml
pytest -q tests/test_paper_pipeline.py \
  tests/test_manuscript_guidance_review.py \
  tests/test_manuscript_live_hierarchy.py
```

## Evidence-first Core pipeline

The reliable vertical-slice implementation is documented in `PAPER_PIPELINE_V2_CORE.md` and configured by `paper_spec_core.yaml`. The historical `paper_pipeline.py` remains a scaffold/compatibility path and must not be presented as a review-draft generator.

The Core now includes a faithful outline compiler. `parse-outline` produces a
39-node AST and explicit resolution; `build-blueprint` creates a one-to-one
structured blueprint and rejects merge, split, rename, reorder, omission, and
claim-copy regressions before prose generation.
