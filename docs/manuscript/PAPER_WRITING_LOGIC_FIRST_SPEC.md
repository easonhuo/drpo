# Paper Writing Logic-First Gate — Phase A Specification

**Initiative:** `PAPER-WRITING-LOGIC-FIRST-01`  
**Status:** opt-in Phase A implementation  
**Scientific impact:** none  
**Default pipeline impact:** none

## 1. Purpose

The logic-first gate is a constrained front-end for the existing manuscript hierarchy:

```text
Guidance rules + Playbook modules
  -> approved section/paragraph logic artifacts
  -> explicit source mapping and edit authorization
  -> prose candidate
  -> existing paper pipeline and release checks
```

It does not replace `paper_pipeline_core.py`, change scientific authority, generate
results, or promote manuscript evidence. Phase A is intentionally opt-in so that its
false-positive rate and latency can be measured before any default-policy change.

## 2. Runtime use of existing writing assets

`docs/manuscript/paper_logic_gate_policy.yaml` maps each edit level to:

- stable Guidance rule IDs that must exist in
  `RL_PAPER_WRITING_GUIDANCE.md`;
- exact Playbook headings that must exist in
  `RL_PAPER_WRITING_PLAYBOOK.md`;
- artifacts required before a prose candidate can pass.

The gate verifies those references at runtime. This converts Guidance and Playbook
from passive documents into selected inputs to a deterministic validation contract.
The iteration log remains historical design memory and is not loaded for each edit.

## 3. Edit levels

### `wording`

For grammar, concision, or local clarity that does not change paragraph responsibility,
claim strength, evidence, or order.

Required artifacts:

- approved paragraph logic map;
- complete source mapping;
- prose candidate.

No section-logic reapproval is required.

### `paragraph`

For a missing causal bridge, sentence-role change, paragraph transition, or a new
approved boundary sentence.

Required artifacts:

- approved section logic map;
- approved paragraph logic map;
- complete source mapping;
- prose candidate.

Only the targeted paragraphs and adjacent transitions are invalidated.

### `section`

For a changed central question, paragraph responsibility, theory--method bridge,
evidence architecture, or section-level order.

Required artifacts are the same as `paragraph`, but the invalidation scope covers the
section's paragraph logic, source mappings, and prose descendants.

## 4. Artifact contract

Every artifact is bound to the exact source SHA-256. A source change makes old logic,
mapping, and candidate artifacts stale.

### Section logic map

Required fields:

```yaml
schema_version: 1
artifact_type: section_logic_map
status: approved
source_sha256: <sha256>
section_id: INTRO
central_question: ...
entry_point: ...
exit_point: ...
chain: [...]
paragraph_ids: [INTRO-P01, INTRO-P02]
approval:
  approved_by: user
  approved_at: 2026-07-14
```

### Paragraph logic map

Required fields:

```yaml
schema_version: 1
artifact_type: paragraph_logic_map
status: approved
source_sha256: <sha256>
section_id: INTRO
paragraphs:
  - id: INTRO-P01
    responsibility: ...
    topic_claim: ...
    sentence_nodes:
      - id: INTRO-P01.S01
        role: ...
        instruction: ...
approval:
  approved_by: user
  approved_at: 2026-07-14
```

Section and paragraph maps must agree exactly on paragraph IDs and order.

### Source mapping

Each approved sentence node receives exactly one operation:

```text
KEEP | TRIM | REVISE | MOVE | ADD
```

`KEEP` and `MOVE` preserve the source text byte-for-byte. `TRIM`, `REVISE`, and `ADD`
require a reason. `ADD` may only target a sentence node already present in the approved
paragraph logic map. Claim strengthening fails unless the manifest explicitly records
that the user approved it.

### Prose candidate

The candidate is structured by paragraph ID and sentence-node ID. Candidate nodes and
source-mapping nodes must match exactly. Every approved node in each targeted paragraph
must be represented, preventing silent omission. Sentence order must match the
paragraph logic map unless a `MOVE` operation explicitly authorizes reordering.

### Authoring manifest

The manifest binds the edit level, target scope, source, artifact hashes, and approval:

```yaml
schema_version: 1
initiative: PAPER-WRITING-LOGIC-FIRST-01
edit_level: paragraph
section_id: INTRO
target_paragraph_ids: [INTRO-P02]
source:
  path: paper/source.md
  sha256: <sha256>
section_logic:
  path: work/section_logic.yaml
  sha256: <sha256>
paragraph_logic:
  path: work/paragraph_logic.yaml
  sha256: <sha256>
source_mapping:
  path: work/source_mapping.yaml
  sha256: <sha256>
candidate:
  path: work/candidate.yaml
  sha256: <sha256>
authorization:
  approved_by: user
  approved_at: 2026-07-14
  allow_claim_strengthening: false
```

## 5. Fail-closed behavior

The gate rejects:

- a missing required artifact;
- a path outside the repository;
- a checksum mismatch;
- draft or stale logic maps;
- source SHA mismatch;
- section/paragraph identity or order mismatch;
- candidate nodes without source-mapping authorization;
- mapping nodes absent from the candidate;
- omission of an approved node in a targeted paragraph;
- modification of `KEEP` or `MOVE` text;
- sentence reordering without `MOVE`;
- target-scope expansion;
- undeclared claim strengthening;
- a policy reference to a missing Guidance rule or Playbook module.

Phase A validates authorization and declared claim impact. It does not claim to solve
full semantic equivalence or automatically detect a dishonest claim-impact annotation.
Those remain review responsibilities and possible later enhancements.

## 6. Commands

Inspect the required contract without changing files:

```bash
python scripts/paper_logic_gate.py --repo-root . plan \
  --edit-level paragraph \
  --section-id INTRO \
  --paragraph-id INTRO-P02
```

Validate a prepared authoring packet:

```bash
python scripts/paper_logic_gate.py --repo-root . validate \
  --manifest path/to/authoring_manifest.yaml \
  --report path/to/gate_report.json
```

A successful report records selected Guidance rules, selected Playbook modules,
required artifacts, validated node count, source hash, authorization owner, and
incremental invalidation scope.

## 7. Integration boundary

Phase A does not modify the default behavior of `paper_pipeline_core.py`. The intended
shadow sequence is:

```text
paper_logic_gate.py validate
  -> PASS report
  -> existing Core blueprint/prose/release path
```

Default integration requires a separate user-approved change after shadow replay on a
real manuscript revision. Until then, a PASS proves only that the supplied authoring
packet satisfies the Phase A authorization contract.

## 8. Acceptance tests

The test suite covers:

- wording edits passing without a section map;
- paragraph edits failing when section approval is absent;
- frozen sentence modification rejection;
- unauthorized candidate-node rejection;
- stale parent rejection;
- claim-strengthening authorization;
- complete target-node coverage;
- approved `ADD` behavior;
- deterministic invalidation scopes;
- live policy references to repository Guidance and Playbook headings.

Run:

```bash
pytest -q tests/test_paper_logic_gate.py
python -m py_compile scripts/paper_logic_gate.py
ruff check scripts/paper_logic_gate.py tests/test_paper_logic_gate.py
```

## 9. Rollback

Because Phase A is opt-in and isolated, rollback is removal or disabling of:

- `scripts/paper_logic_gate.py`;
- `docs/manuscript/paper_logic_gate_policy.yaml`;
- the Phase A specification and tests.

No scientific artifact, manuscript prose, existing pipeline output, or default command
must be rewritten to roll back this phase.
