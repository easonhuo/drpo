# Manuscript cascade protocol

This protocol prevents the outline, paragraph blueprint, and manuscript prose from
becoming independent drafts.

## Canonical hierarchy

The hierarchy is strictly one-way:

1. **Outline** — the structural contract: paragraph order, paragraph responsibility,
   and allowed claims.
2. **Paragraph blueprint** — a derivation of the outline: topic sentence, supporting
   reasoning, evidence, equations, citations, word budget, and prohibited overclaim.
3. **Prose** — a realization of the blueprint.

The blueprint may expand an outline paragraph but may not add, remove, reorder, split,
merge, or rename paragraphs independently.  Prose may realize a blueprint but may not
change its responsibility independently.

## Mandatory top-down issue triage

For every reported manuscript problem:

1. Check the relevant **outline paragraph first**.
2. Only when the outline is correct, check the matching **blueprint paragraph**.
3. Only when both are correct, inspect the matching **prose paragraph**.
4. The first failing layer is the root-cause layer.
5. Modify that layer and every configured downstream layer:
   - outline failure -> outline, blueprint, prose;
   - blueprint failure -> blueprint, prose;
   - prose failure -> prose only.
6. Upstream layers already marked correct must not be rewritten.

This is the first-invalid-layer rule.  It forbids bottom-up patching and prevents a
child document from silently redefining the manuscript structure.

## Stable paragraph identity

Every paragraph must use the same stable ID and title in all configured layers:

```markdown
<!-- MANUSCRIPT:BEGIN INTRO-P01 -->
## [INTRO-P01] Background and problem setting
...
<!-- MANUSCRIPT:END INTRO-P01 -->
```

The order and titles must match exactly across the outline, blueprint, and prose.

## Parent fingerprints

A blueprint paragraph must place the normalized SHA-256 of its outline paragraph
immediately after the heading:

```markdown
Parent-Outline-SHA256: `0123...cdef`
```

A prose paragraph must analogously contain:

```markdown
Parent-Blueprint-SHA256: `0123...cdef`
```

Changing an outline paragraph therefore makes the matching blueprint stale.  Updating
that blueprint changes its own hash and makes the prose stale.  The validator rejects
an incomplete cascade even when paragraph IDs and titles still happen to match.

## Files

- `scripts/manuscript_cascade.py`: validator and Git cascade gate.
- `docs/manuscript_cascade/hierarchy.template.yaml`: hierarchy configuration template.
- `docs/manuscript_cascade/change_request.template.yaml`: top-down issue record template.

## Commands

Validate structural identity and parent fingerprints:

```bash
python3 scripts/manuscript_cascade.py validate-artifacts \
  --repo-root . \
  --config docs/manuscript/hierarchy.yaml
```

Validate a top-down issue record before editing:

```bash
python3 scripts/manuscript_cascade.py validate-issue \
  --config docs/manuscript/hierarchy.yaml \
  --issue docs/manuscript/changes/ISSUE-ID.yaml
```

After the edit, validate the issue, artifacts, and changed-file cascade together:

```bash
python3 scripts/manuscript_cascade.py validate-change \
  --repo-root . \
  --config docs/manuscript/hierarchy.yaml \
  --issue docs/manuscript/changes/ISSUE-ID.yaml \
  --base <BASE_COMMIT> \
  --head <CANDIDATE_COMMIT>
```

## Change-record states

During triage, use `resolution.state: planned` and keep `changed_layers: []`.
After all required layers are updated, set `resolution.state: completed` and list
exactly the computed cascade in `changed_layers`.

A downstream layer is `blocked` during triage whenever an upstream layer is wrong.
This prevents reviewing or rewriting a child against a parent contract already known
to be invalid.
