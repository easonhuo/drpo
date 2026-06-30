# Active manuscript hierarchy

The live manuscript cascade is configured in `docs/manuscript/hierarchy.yaml`.

Current active artifacts:

- canonical full outline: `docs/paper_rewrite_outline_v0_8.md`;
- Introduction paragraph blueprint: `docs/paper_rewrite_intro_blueprint_v0_2.md`;
- Introduction prose: not created yet.

`docs/paper_rewrite_outline_v0_7.md` and
`docs/paper_rewrite_intro_blueprint_v0_1.md` are preserved as historical
provenance. They are not active manuscript contracts.

Before editing or delivering manuscript material, run:

```bash
python scripts/manuscript_cascade.py validate-artifacts \
  --repo-root . \
  --config docs/manuscript/hierarchy.yaml
```

When a user reports a problem, create or update a change record under
`docs/manuscript/issues/`, inspect the outline first, and propagate the first
failing layer through all configured downstream layers.
