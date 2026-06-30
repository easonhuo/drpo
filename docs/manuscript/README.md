# Active manuscript hierarchy

The live manuscript cascade is configured in `docs/manuscript/hierarchy.yaml`.

## Governance layers

The manuscript now separates stable principles from evolving paper strategy:

1. **Scientific authority:** `docs/handoff.md` and `experiments/registry.yaml` own conclusions, experiment status, frozen protocols, and execution order.
2. **Stable writing standard:** `docs/manuscript/RL_PAPER_WRITING_GUIDANCE.md` owns durable writing principles and G01–G14 review gates. It changes only when a general rule is wrong or materially incomplete.
3. **DRPO manuscript strategy:** `docs/manuscript/DRPO_MANUSCRIPT_STRATEGY.md` owns the current paper lineage, central tension, theorem role, method bridge, environment responsibilities, novelty boundary, and evidence architecture. It may evolve with the science.
4. **Versioned manuscript artifacts:** outline → paragraph blueprint → prose. These may iterate while preserving parent hashes and superseded versions.
5. **Source corpus:** `docs/manuscript/RL_WRITING_CORPUS_NOTES.md` records paper/skill close reading. New corpus entries do not automatically modify Guidance.

## Current active artifacts

- stable quality gate: `docs/manuscript/RL_PAPER_WRITING_GUIDANCE.md`;
- DRPO strategy: `docs/manuscript/DRPO_MANUSCRIPT_STRATEGY.md`;
- source notes: `docs/manuscript/RL_WRITING_CORPUS_NOTES.md`;
- canonical full outline: `docs/paper_rewrite_outline_v0_9_1.md`;
- Introduction paragraph blueprint: `docs/paper_rewrite_intro_blueprint_v0_5.md`;
- Introduction prose: not created yet.

The v0.9.1 outline is an authorized refinement of v0.9. It makes the original DRPO lineage explicit, removes live experiment status from structural content, removes Product manifold from the main environment table, consolidates six RQs into four, and elevates the aggregate negative term to a mandatory theory–method–experiment observable. The v0.5 blueprint is its six-paragraph Introduction derivation.

## Historical artifacts

Historical files remain available and must not be destructively deleted:

- v0.9/v0.4 are the previous approved parent pair;
- v0.7/v0.3 are the earlier approved pair;
- v0.8/v0.2 preserve the superseded invalid reverse-alignment attempt;
- still earlier outline and blueprint files remain provenance unless separately governed.

## Validation

Before editing or activating manuscript material, run:

```bash
python3 scripts/manuscript_cascade.py validate-artifacts \
  --repo-root . \
  --config docs/manuscript/hierarchy.yaml

python3 scripts/manuscript_cascade.py validate-issue \
  --config docs/manuscript/hierarchy.yaml \
  --issue docs/manuscript/issues/PAPER-V091-STRATEGY-SEPARATION-01.yaml

pytest -q \
  tests/test_manuscript_guidance_review.py \
  tests/test_manuscript_live_hierarchy.py
```

Every new or modified outline, blueprint, prose section, main figure plan, or result narrative requires a review record under `docs/manuscript/reviews/` with all G01–G14 gates passing.

When a manuscript problem is reported, inspect the strategy and outline before changing a child artifact. Change an approved outline only when it is independently wrong and the user authorizes the revision; then cascade the change through every configured downstream layer and preserve the superseded pair.
