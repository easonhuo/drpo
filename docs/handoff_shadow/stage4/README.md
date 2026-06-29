# Stage 4A Shadow Inventory and Dynamic Semantic Graph

This directory is a **non-authoritative Stage 4A shadow workspace** for
`GOV-HANDOFF-INDEX-01`.

`docs/handoff.md` remains the only research master. Nothing under this directory
may replace the manual handoff, change the startup protocol, alter an experiment,
or authorize Stage 4B, Stage 4C, or Stage 5.

The original `schema/` and `inventory/` directories remain the validated Stage 4A
bootstrap snapshot. The `dynamic/` directory adds a reusable semantic layer:

- `kernel/`: cross-project node, relation, lifecycle, review, and validation rules;
- `profiles/`: versioned project-specific semantic profiles;
- `overrides/`: small, explicit human approvals for ambiguous semantics;
- `generated/`: deterministic nodes, edges, review queue, manifest, Mermaid views,
  and Graphviz DOT generated from the same canonical graph.

The dynamic builder never calls a network service or an LLM. Mechanical discovery
is automatic; semantic proposals not resolved by the profile or an explicit
override remain in the review queue and never become accepted graph edges.

Run the bootstrap validator:

```bash
python3 scripts/validate_stage4a_inventory.py --repo-root .
```

Regenerate and validate the dynamic graph:

```bash
python3 scripts/build_stage4_semantic_graph.py --repo-root . --write
python3 scripts/build_stage4_semantic_graph.py --repo-root . --check
python3 scripts/validate_stage4_semantic_graph.py --repo-root .
```

Generated files must not be edited manually. The validator fails closed on stale
sources, duplicate or dangling graph objects, unknown types, supersedes cycles,
unaccepted semantic leakage, stale graph hashes, and visualization drift.
