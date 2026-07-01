# DRPO Paper Pipeline v2.3 Core

## Status

- Task: `PAPER-PIPELINE-V2-CORE-01`
- Scope: first reliable implementation of the evidence-first manuscript core
- Build profile: `core_vertical_slice`
- Outline compiler increment: `PAPER-PIPELINE-V2-CORE-BLUEPRINT-01`
- Scientific result status: unchanged; this pipeline is manuscript infrastructure, not a new experiment

## Goal

The Core pipeline replaces the failed outline-expansion path with one short, auditable chain:

```text
repository evidence
  -> frozen research snapshot
  -> real figure, table, and proof assets
  -> executable paragraph blueprint
  -> evidence-bounded prose
  -> deterministic checks
  -> review PDF
```

The first acceptance target is deliberately small: one long-run-validated C-U1 result, one empirical figure, one result table, two evidence-bearing result paragraphs, one analytic proposition with proof, and a two-page review PDF. Passing this slice is required before expanding the system to the full manuscript.

## Non-goals

This version does **not** implement:

- bidirectional outline/blueprint/prose synchronization;
- a general waiver system;
- concurrent release promotion or CAS;
- hermetic file-read tracing;
- a general schema-migration engine;
- semantic three-way merging of author edits;
- autonomous literature search;
- submission-candidate generation.

Those items remain backlog work and must not be used to block the Core slice.

## Authority and direction

The authority order is:

1. `docs/handoff.md` and `experiments/registry.yaml` for status, terminology, protocol, and claim boundaries;
2. compact result artifacts and their checksums for reported values;
3. the approved outline for paragraph identity and rhetorical responsibility;
4. `docs/manuscript/paper_spec_core.yaml` for the current slice's evidence bindings and output contract.

`research_snapshot.json` is a generated, immutable build snapshot. It is not a competing research master document.

The manuscript direction is strictly:

```text
approved outline -> blueprint -> prose -> TeX/PDF
```

A child-layer problem may create a proposal but may not rewrite a verified parent.


## Faithful outline-to-blueprint compiler

The approved Markdown outline is compiled deterministically before any semantic
blueprint enrichment:

```text
approved outline
  -> outline_ast.json
  -> outline_resolution.json
  -> one-to-one blueprint.json
  -> rendered blueprint.md
```

The compiler enforces the following hard rules:

- every stable outline ID appears exactly once and in the same order in the AST,
  resolution, and blueprint;
- the current build profile may only mark a node `enabled` or
  `disabled_with_reason`; merge, split, rename, reorder, and silent omission are
  rejected;
- every enabled node carries its parent block SHA-256;
- an enabled blueprint claim may not be a normalized copy of the outline claim;
- experiment nodes require exact metric paths and a figure or table binding;
- theory or method nodes require theorem/equation bindings;
- all enabled nodes require an executable sentence plan, evidence refs, a
  reviewer objection and response, calibrated conclusions, and a transition.

For the Core slice, the 39-node approved outline resolves to exactly two enabled
nodes: `METHOD-P03` and `EXP-P04`. The earlier `EXP-P04-A` / `EXP-P04-B` split is
invalid and is replaced by one evidence-bearing `EXP-P04` paragraph. The other
37 nodes remain present with explicit disabled reasons; they are not silently
dropped.

## Evidence gate

The Core slice is enabled only when all of the following are true:

- the registered experiment status satisfies the spec's minimum status;
- the compact artifact index exists and all listed compact files match their SHA-256 values;
- the terminal audit exists;
- the C-U1 terminology remains held-out-context / unseen-state generalization, never OOD;
- task-performance collapse, support/variance-boundary events, and NaN/Inf failures remain separate fields;
- every prose number is rendered from the generated snapshot, not typed independently into the prose template.

A missing or conflicting input fails closed. The pipeline does not emit `TBD` result paragraphs and does not upgrade evidence status.

## Commands

```bash
python scripts/paper_pipeline_core.py snapshot --repo-root .
python scripts/paper_pipeline_core.py parse-outline --repo-root .
python scripts/paper_pipeline_core.py build-blueprint --repo-root .
python scripts/paper_pipeline_core.py validate-blueprint --repo-root .
python scripts/paper_pipeline_core.py build-slice --repo-root .
python scripts/paper_pipeline_core.py validate-slice --repo-root .
python scripts/paper_pipeline_core.py all --repo-root .
```

The default output directory is `paper/core_review_v2_core/`. `compile` and `all` are strict build commands and require `latexmk`. `validate-slice` is intentionally portable: it validates the committed PDF, LaTeX log, and all manifest hashes without rewriting tracked artifacts; when `pdfinfo` is unavailable it uses the page count recorded by a prior verified compile in `build_manifest.json`. Package tests always validate the committed slice and exercise deterministic source generation in a temporary directory, so applying the update does not require a local TeX installation.

## Acceptance contract

The vertical slice passes only when:

- the snapshot is built from `C-U1-E3-ADAM-RERUN` with `long_run_validated` status;
- the four primary fixed-variance interventions and the two registered fixed-budget controls are present;
- reported means and confidence intervals exactly match the verified compact aggregate;
- the figure and table are regenerated from the snapshot;
- the 39-node outline AST, explicit resolution, and one-to-one blueprint validate;
- the blueprint names exact metric paths and contains a nontrivial sentence plan;
- no outline paragraph is merged, split, renamed, reordered, or silently omitted;
- the prose contains no `TBD`, no OOD claim for C-U1, and no universal method ranking;
- task collapse, boundary events, and NaN/Inf are reported separately;
- Proposition 2 states a finite-order score-growth assumption and proves exponential domination without asserting exponential utility decay;
- LaTeX compiles and the review PDF has exactly two pages;
- the legacy v0.9.2 deterministic scaffold is not used as the source of the slice.

## Expansion gate

Full-manuscript work may start only after this vertical slice passes its unit tests, package tests, PDF render inspection, and author review. A failure must be repaired in the Core chain; it must not be hidden by adding more infrastructure.
