# Domain-agnostic publication-quality manuscript pipeline

This layer converts a stable-ID research outline into paragraph blueprints,
prose, figures/tables, and a release package without embedding the scientific
content of any one paper in the core engine.

## Architecture boundary

The pipeline is intentionally split into two layers.

### Generic core

The generic layer defines only reusable writing and release mechanics:

1. outline obligations such as claim, reader question, evidence, and scope;
2. paragraph-blueprint obligations such as ordered sentence units, rhetorical
   function, citation/formula/proof bindings, claim boundaries, and word budget;
3. prose realization checks, including coverage, citation validity, theorem and
   appendix links, word budget, and reference-independence auditing;
4. graph rendering, asset injection, template validation, bibliography checks,
   and portable TeX compilation.

The generic implementation is located in:

- `docs/manuscript/generic_publication_quality_profile.yaml`;
- `scripts/manuscript_publication_pipeline.py`;
- `scripts/manuscript_release_pipeline.py`;
- `scripts/paper_pipeline.py`.

These files must not contain a paper's method names, environments, datasets,
claims, theorem content, terminology rules, or experiment-specific captions.

### Project layer

A project supplies its own scientific content and governance through data and
plugins:

- the manuscript graph and outline;
- a project profile with terminology and claim-boundary rules;
- a project publication contract mapping opaque sentence roles to generic
  rhetorical functions;
- evidence, bibliography, theorem/proof labels, figures, tables, and template
  configuration;
- optional project asset builders.

For this repository, those inputs live in:

- `docs/manuscript/paper_graph.yaml`;
- `docs/manuscript/projects/drpo_profile.yaml`;
- `docs/manuscript/publication_quality_contract.yaml`;
- `docs/manuscript/full_paper_assets.yaml`;
- `scripts/projects/drpo/build_manuscript_assets.py`.

The current project can therefore request domain-specific content without
turning that content into a universal writing rule.

## Where quality comes from

Publication quality is enforced through domain-independent obligations rather
than similarity to a reference manuscript:

- every paragraph has a reader question and a defensible claim;
- definitions, results, reasoning, interpretation, limitations, and transitions
  are represented as distinct moves when the section archetype requires them;
- every empirical or formal statement is bound to registered evidence;
- citations, equations, theorem statements, proofs, and figures are checked as
  dependencies rather than invented during prose generation;
- every sentence-plan unit must be realized in the prose;
- allowed and forbidden conclusions are checked explicitly;
- reference papers are treated as quality rubrics only, with optional lexical
  overlap gates to detect copying.

A strong reference may inform the quality dimensions being audited, but it is
never a source of project claims, entities, theorem order, experimental story,
or wording unless those items are independently registered in the project
inputs.

## Commands

```bash
python scripts/paper_pipeline.py render --repo-root .
python scripts/manuscript_publication_pipeline.py all --repo-root .
python scripts/compile_full_manuscript.py --repo-root . --skip-compile
```

To use a trusted local prose generator, pass `--generator-cmd` to
`manuscript_publication_pipeline.py`. The command receives one JSON packet on
stdin and must return `{"prose": "..."}`. The packet separates generic quality
requirements from project content. Generated prose remains a candidate until
all coverage, evidence, citation, formula, proof, terminology, and
reference-independence gates pass.

## Genericity acceptance test

`tests/fixtures/generic_manuscript/` describes an unrelated distributed-sensor
paper. The same publication-quality and release engines must validate that
fixture end to end without emitting any terminology or narrative from this
repository's primary research paper. This is a required regression test for the
core/project boundary.

## Governance boundaries

This layer changes manuscript infrastructure only. It does not change an
experiment's status, seeds, thresholds, convergence criteria, method ranking,
or scientific conclusion. All project-specific terminology and reporting rules
remain project-profile data and are not promoted into generic pipeline logic.
