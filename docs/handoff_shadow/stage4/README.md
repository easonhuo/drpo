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


## Dynamic governance overrides

`dynamic/overrides/SEMANTIC_OVERRIDES.yaml` is versioned input, not a generated
file. Increment `override_version` whenever accepted edges, rejected candidates,
module assignments, or lifecycle decisions change.

A rejected candidate records the exact deterministic review signature and a
human rationale. The builder suppresses the same candidate from the pending
queue while preserving the rejection in generated audit output. It does not use
fuzzy rejection matching.

Approved module evolution is expressed with `module_lifecycle_changes` using
`rename`, `supersede`, `split`, or `merge`. Every change supplies source and
target IDs, exact before/after versions, and rationale. Existing IDs are never
deleted: old modules remain queryable and are connected to replacements through
`supersedes` lineage.

The previous generated manifest is an enforcement checkpoint. Semantic profile
changes require a higher `profile_version`; semantic override changes require a
higher `override_version`; module changes require a higher per-module version.
Destructive module removal fails closed. The semantic-fingerprint algorithm is
itself versioned so that a tooling migration cannot be confused with a business
semantic change.

Current hardening validates the DRPO project workflow. Direct Stage 3 Delta
consumption, an AI proposal adapter, and broader cross-project acceptance remain
later work and do not authorize Stage 4B or Stage 4C.
