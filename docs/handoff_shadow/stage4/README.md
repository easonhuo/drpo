# Stage 4A Shadow Inventory

This directory is a **non-authoritative Stage 4A shadow workspace** for
`GOV-HANDOFF-INDEX-01`.

`docs/handoff.md` remains the only research master. Nothing under this directory
may replace the manual handoff, change the startup protocol, alter an experiment,
or authorize Stage 4B, Stage 4C, or Stage 5.

Stage 4A contains only:

- the frozen node, relation, lineage, ambiguity, and dependency-closure schema;
- the semantic-module inventory;
- exact heading, claim-locator, and experiment inventories;
- a read-only deterministic validator.

Run the validator from the repository root:

```bash
python3 scripts/validate_stage4a_inventory.py --repo-root .
```

The validator fails closed on stale source hashes, missing or duplicate objects,
unresolved classifications, dangling references, broken lineage, unknown node or
relation types, and any Stage 4B/4C output appearing before its own acceptance.
