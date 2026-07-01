# Blueprint-to-manuscript graph transform

Task: `PAPER-PIPELINE-V2-BLUEPRINT-DOWNSTREAM-01`

Parent: `PAPER-PIPELINE-V2-CORE-01`

Scientific result status: unchanged.

## Purpose

The existing outline compiler already treats manuscript construction as a
stable-ID graph transform. This increment reuses that model instead of adding a
second pipeline:

```text
approved outline -> executable blueprint -> downstream product graph
                                            |-> prose packets and draft
                                            |-> figure specifications and figures
```

The generic engine is `scripts/manuscript_graph_transform.py`. Paragraph-specific
choices live in `docs/manuscript/manuscript_downstream_contract.yaml`; they are
adapters, not a new orchestration system.

## Reused invariants

The downstream compiler preserves every source blueprint node, stable ID,
order, section, enabled/disabled state, and parent outline block hash. Disabled
nodes remain explicit. Enabled sentence-plan roles become stable sentence IDs
such as `EXP-P04-S03` in the same order.

Each sentence unit records:

- its blueprint role and instruction;
- exact evidence IDs;
- exact research-snapshot metric paths and resolved values;
- figure and table bindings; and
- a deterministic first-pass sentence.

The same units also form an LLM refinement packet with allowed conclusions,
forbidden conclusions, the reviewer objection, the required response, and the
transition. This makes later language refinement subordinate to the blueprint
instead of asking a model to reconstruct the argument from a chapter title.

## Visual mapping

The contract describes figures as products of the same sentence units. Each
figure declares the sentence IDs it supports, its source metrics, panel
questions, renderer, caption boundary, and output files. The first registered
renderer is a generic multi-panel bar renderer. `EXP-P04` uses it to keep four
outcomes separate:

1. fixed-variance held-out-context reward;
2. fixed-variance task-performance collapse;
3. learnable-variance support-boundary events; and
4. NaN/Inf numerical failures.

## Outputs

The default command writes to `paper/core_review_v2_core/downstream_v1/`:

- `product_graph.json`
- `prose_packets.json`
- `prose_draft.md`
- `figure_specs.json`
- `exp_p04_causal_mapping.pdf`
- `exp_p04_causal_mapping.png`
- `validation_report.json`

Generated outputs are deterministic functions of the blueprint, snapshot,
adapter contract, and renderer. The report records SHA-256 hashes for all three
source inputs.

## Commands

```bash
python scripts/manuscript_graph_transform.py --repo-root .
```

Compile and validate without invoking Matplotlib:

```bash
python scripts/manuscript_graph_transform.py --repo-root . --skip-figures
```

Run unit tests:

```bash
pytest -q tests/test_manuscript_graph_transform.py
```

## Claim boundary

This increment demonstrates reuse of the graph compiler and improves the
faithfulness of the first prose/visual realization. It does not claim that a
deterministic template is camera-ready prose, that every future figure needs no
new renderer, or that manuscript infrastructure changes constitute a new
scientific result.
