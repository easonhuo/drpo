# Paper Blind Replay Harness

`PAPER-BLIND-REPLAY-01` adds an audit layer for label-driven manuscript replay.
The purpose is to let the paper pipeline learn or optimize an outline from the
label paper, then prove that downstream blueprint/prose/TeX generation does not
look at the label paper or existing drafts.

## Protocol boundary

The replay is split into three phases.

1. **Label-to-outline phase.** The label paper may be read to extract a rhetorical
   skeleton or to improve the outline.
2. **Blind generation phase.** Once the outline is frozen, generation may read
   only files listed in `BLIND_INPUT_MANIFEST.json` inside the replay workspace.
   Label PDFs, release PDFs, old prose, old blueprints, old Overleaf sections,
   and the existing manuscript graph are forbidden.
3. **Post-generation evaluation phase.** The label paper may be read again only
   to score the generated result against the label. It must not be used as an
   input to the generation step.

This is a pipeline-quality experiment, not a DRPO scientific experiment. It does
not change the status of E8, C-U1, D-U1, Hopper, or Countdown.

## Why this exists

A high replay score is meaningless if the generator can see the answer. The
historical manuscript graph contains outline, blueprint, and prose fields, so a
plain rerun of the existing paper pipeline can accidentally reuse old prose even
without directly reading `paper/releases/DRPO_REWRITTEN_DRAFT_20260705.pdf`.
The blind replay harness prevents that by copying only allowlisted inputs into a
fresh workspace and auditing generated outputs for leakage.

## Current command

The deterministic scaffold is intentionally conservative. It is not a
publication-quality generator. It proves the input boundary and produces files
that a later generator can replace while keeping the same manifest/audit gate.

```bash
python3 scripts/paper_blind_replay.py all \
  --repo-root . \
  --workspace outputs/paper_blind_replay/REPLAY-001 \
  --optimized-outline docs/paper_rewrite_outline_v0_9_2.md \
  --label-source paper/releases/DRPO_REWRITTEN_DRAFT_20260705.pdf \
  --include-default-drpo-context \
  --sentinel DRPO_LABEL_SENTINEL_DO_NOT_COPY
```

Outputs:

- `BLIND_INPUT_MANIFEST.json`: exact generation-phase input list and hashes;
- `generated/replay_blueprint.md`: outline-only scaffold blueprint;
- `generated/replay_prose.md`: outline-only scaffold prose;
- `generated/replay_main.tex`: outline-only scaffold TeX;
- `BLIND_REPLAY_AUDIT.json`: leakage audit result.

## Forbidden generation inputs

The generation phase must not read these sources:

- `paper/releases/`;
- `paper/core_review_v2_core/`;
- `paper/publication_quality_v1/`;
- `paper/overleaf/sections/`;
- `paper/overleaf/generated/`;
- `paper/overleaf/main.tex`, `main_replacement.tex`, or `main.pdf`;
- `docs/manuscript/paper_graph.yaml`;
- `docs/paper_rewrite_blueprint*`;
- `docs/paper_rewrite_intro_blueprint*`;
- `docs/paper_rewrite_prose*`.

`paper/releases/DRPO_REWRITTEN_DRAFT_20260705.pdf` may be named as
`--label-source`, but the harness records it as pre-outline/evaluation-only and
does not copy it into the blind workspace.

## Audit behavior

`audit` verifies that every manifest input is still present and has the recorded
hash. It then scans generated text files for forbidden path tokens and configured
sentinel strings. If a generator copies a sentinel from the label or references a
forbidden source path, the audit fails.

```bash
python3 scripts/paper_blind_replay.py audit \
  --workspace outputs/paper_blind_replay/REPLAY-001
```

## Interpretation

A passing blind audit only proves that the generation workspace was isolated and
that no configured sentinel or forbidden path token leaked into generated text.
It does not prove that the regenerated manuscript is good or that it matches the
label. Semantic replay evaluation belongs to the post-generation evaluation
phase and should be reported separately.
