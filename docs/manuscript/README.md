# Active manuscript hierarchy

The live manuscript cascade is configured in `docs/manuscript/hierarchy.yaml`.

Current active artifacts:

- canonical full outline: `docs/paper_rewrite_outline_v0_7.md`;
- Introduction paragraph blueprint: `docs/paper_rewrite_intro_blueprint_v0_3.md`;
- Introduction prose: not created yet.

The v0.7 outline is the user-approved structural contract. The v0.3 blueprint is a
seven-paragraph derivation of that outline. The live validator checks identical paragraph
IDs, titles, order, and parent SHA-256 fingerprints.

`docs/paper_rewrite_outline_v0_8.md` and
`docs/paper_rewrite_intro_blueprint_v0_2.md` are preserved as superseded provenance.
They record an invalid reverse-alignment attempt in which the approved seven-paragraph
outline was changed to match an eight-paragraph child blueprint. They are not active
manuscript contracts.

Before editing or delivering manuscript material, run:

```bash
python3 scripts/manuscript_cascade.py validate-artifacts \
  --repo-root . \
  --config docs/manuscript/hierarchy.yaml
```

When a user reports a problem, create or update a schema-version-2 change record under
`docs/manuscript/issues/`. Inspect the outline first. A child-parent mismatch is a child
failure by default; do not change the outline merely to make the mismatch disappear. If a
blueprint review reveals a genuine outline improvement, obtain or record explicit outline
revision authorization, then cascade the approved parent change downstream.
