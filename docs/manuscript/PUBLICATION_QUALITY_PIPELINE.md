# Publication-quality outline, blueprint, and prose gate

This layer upgrades the first four manuscript sections without replacing the
stable-ID manuscript graph.

## Why it exists

The legacy graph renderer can project registered prose into Markdown and TeX,
but a compilable projection is not by itself a publication-quality generation
contract. The publication-quality layer makes three obligations executable:

1. the outline must state the reader question, claim, evidence, and required
   logical content for every node;
2. the paragraph blueprint must decompose that obligation into ordered sentence
   units with coverage anchors, citations, theorem/equation bindings, appendix
   proofs, claim boundaries, and a word budget;
3. the prose must realize those units, use only registered citations and formal
   labels, remain inside its budget, and preserve the allowed/forbidden claim
   boundary.

The first registered profile covers Introduction, Related Work, Problem Setup,
and Theory. Later sections can be added to the same contract without creating a
second pipeline.

## Files

- `docs/manuscript/paper_graph.yaml`: source outline, rich blueprints, and
  approved prose candidates.
- `docs/manuscript/publication_quality_contract.yaml`: scoped quality profile and
  required sentence roles.
- `scripts/manuscript_publication_pipeline.py`: packet builder and fail-closed
  validator.
- `paper/publication_quality_v1/`: generated blueprint, prose candidate,
  generation packets, and quality report.

## Commands

```bash
python scripts/paper_pipeline.py render --repo-root .
python scripts/compile_full_manuscript.py --repo-root . --skip-compile
python scripts/manuscript_publication_pipeline.py all --repo-root .
```

To use a trusted local prose generator, pass `--generator-cmd`. The command
receives one JSON packet on stdin and must return `{"prose": "..."}`. Generated
prose remains a candidate until its sentence-unit, citation, formula, proof,
terminology, and word-budget gates pass.

## Governance boundaries

This layer changes manuscript infrastructure only. It does not change experiment
status, seeds, thresholds, method rankings, or terminal audits. C-U1 remains
held-out-context or unseen-state generalization. Task-performance collapse,
support/variance boundary events, and NaN/Inf numerical failure remain separate.
