# Active manuscript hierarchy

The live manuscript cascade is configured in `docs/manuscript/hierarchy.yaml`.

Current active artifacts:

- manuscript writing quality gate: `docs/manuscript/RL_PAPER_WRITING_GUIDANCE.md`;
- canonical full outline: `docs/paper_rewrite_outline_v0_9.md`;
- Introduction paragraph blueprint: `docs/paper_rewrite_intro_blueprint_v0_4.md`;
- Introduction prose: not created yet.

The v0.9 outline is the user-approved structural contract after the RL-paper corpus and
open-source writing-skill review. The v0.4 blueprint is its six-paragraph Introduction
derivation. Both artifacts must pass the manuscript cascade and the guidance hard gates
before prose can become active.

Historical files remain available and must not be destructively deleted:

- `docs/paper_rewrite_outline_v0_7.md` and
  `docs/paper_rewrite_intro_blueprint_v0_3.md` are the previously approved parent pair;
- `docs/paper_rewrite_outline_v0_8.md` and
  `docs/paper_rewrite_intro_blueprint_v0_2.md` preserve the superseded invalid
  reverse-alignment attempt and are not active manuscript contracts.

Before editing or delivering manuscript material, run:

```bash
python3 scripts/manuscript_cascade.py validate-artifacts \
  --repo-root . \
  --config docs/manuscript/hierarchy.yaml

python3 scripts/manuscript_cascade.py validate-issue \
  --config docs/manuscript/hierarchy.yaml \
  --issue docs/manuscript/issues/PAPER-V09-GUIDANCE-REWRITE-01.yaml

pytest -q tests/test_manuscript_guidance_review.py
```

Every new or modified outline, blueprint, prose section, main figure plan, or result
narrative also requires a review record under `docs/manuscript/reviews/` with all
G01–G14 gates passing. `docs/handoff.md` remains the sole authority for scientific
conclusions, experiment status, frozen protocols, and execution order.

When a user reports a manuscript problem, inspect the outline first. A downstream
mismatch is a child failure by default. Change an approved outline only when the outline
itself is independently wrong and the user explicitly authorizes the revision; then
cascade the change through every configured downstream layer.
