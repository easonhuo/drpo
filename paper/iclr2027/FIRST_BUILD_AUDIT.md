# First ICLR 2027 mechanical-build audit

Task: `MANUSCRIPT-ICLR2027-PORT-01`

Base `main`: `8b0616cf0f887f86ec04e398a7604c3d3940aa5d`

First successful build workflow: `33299142248` at branch head `58d4e63430a752477516a8be88b4f84c5b9a36c1`.

## Content integrity

The first build is a format-only migration. CI verified:

- active source blob: `e2e592caa780f0ba4c5edc31986f6ea4106ef000`;
- canonical source body SHA-256: `81eacb6234a7475c328238d431bbb69f11ef7762c66fef0491cd9cd7dc6513e5`;
- canonical generated body SHA-256: the same value;
- `body_lock=PASS`;
- abstract through appendix text change: none;
- figure and bibliography asset lock: pass;
- the only normalized body-level format change is `\\bibliographystyle{icml2026}` -> `\\bibliographystyle{iclr2027_conference}`.

## Template provenance

The successful CI build downloaded the official ICLR 2027 archive from:

`https://media.iclr.cc/Conferences/ICLR2027/iclr-2027-style-files.zip`

Recorded hashes:

- archive: `0d940dfa9398ae99a18f24a85a8a683f367204b6af6d17d2899e60a67102529e`;
- `iclr2027_conference.sty`: `797deef41724e93761426ac0cbcca46279a91cc650dd1f0ce76a4f08d2098ea6`;
- `iclr2027_conference.bst`: `2d67552db7ed38ccfccb5957b52f95656e25c249724761d3cf5f7922ad1844c5`.

## First-PDF findings

The PDF compiles successfully in anonymous ICLR review format on US Letter. It has 48 total pages. Main-text content continues onto page 14, where the Conclusion is followed by References; references continue through page 16, and Appendix A starts on page 17. Therefore the unchanged manuscript is materially above the ICLR 2027 submission limit of nine main-text pages. This is a diagnostic finding only; no prose, equation, table, caption, figure, or appendix content is edited in this stage.

Visual inspection covered pages 1--17 plus representative appendix/figure pages 29, 33, 47, and 48. No clipped text, broken glyphs, figure clipping, or element overlap was observed. The final LaTeX log contains no overfull-box diagnostic; four underfull-vbox diagnostics remain, consistent with float/page packing rather than content loss.

Two visible format issues are intentionally left for the next approved stage:

1. Hyperref draws colored rectangular borders around citations/cross-references/URLs. This is a pure formatting issue and can be removed without changing manuscript content.
2. Several single-column pages have non-optimal vertical packing because the source was authored under a different venue layout. This is also a format/layout issue, but it is not expected to recover the roughly four-plus pages needed to satisfy the nine-page main-text limit by itself.

## ICLR-specific compliance item deliberately not inserted

ICLR 2027 requires an AI-use statement and states that this section does not count toward the main-text page limit. The active manuscript does not currently contain that ICLR-specific section. Because this migration is under an explicit zero-content-change lock, the first mechanical version does **not** add it. It must be added only in a later content-approved submission-compliance pass.

No experimental protocol, result, `docs/handoff.md`, or `experiments/registry.yaml` content was modified by this task.
