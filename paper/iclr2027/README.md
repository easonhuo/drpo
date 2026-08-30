# DRPO ICLR 2027 mechanical port

This directory defines the **format-only ICLR 2027 port** of the active full DRPO manuscript.

- Task / claim: `MANUSCRIPT-ICLR2027-PORT-01`
- Base commit: `8b0616cf0f887f86ec04e398a7604c3d3940aa5d`
- Scientific/content source: `paper/overleaf/main_replacement.tex`
- Source blob at the base commit: `e2e592caa780f0ba4c5edc31986f6ea4106ef000`
- Official ICLR 2027 style archive: `https://media.iclr.cc/Conferences/ICLR2027/iclr-2027-style-files.zip`

## Hard content lock

This first port is intentionally mechanical. The abstract, main text, section titles, equations, theorem/proposition text, captions, table text and values, citations, appendices, bibliography entries, and figure assets must remain unchanged. The only permitted transformations are template plumbing: replacing the ICML wrapper/title machinery with the ICLR 2027 wrapper, changing the bibliography *style selector* from `icml2026` to `iclr2027_conference`, and anonymous-review formatting supplied by the official ICLR style.

`generate_iclr.sh` verifies a byte-for-byte canonical body lock from `\\begin{abstract}` through the end of the appendix after normalizing only the bibliography-style selector. It also verifies that the source Git blob is the expected blob before generating anything.

## Build

From the repository root:

```bash
bash paper/iclr2027/build.sh
```

The build downloads the official ICLR 2027 style archive, generates an ICLR wrapper without editing the manuscript source, copies bibliography and figure assets byte-for-byte, compiles the paper, and produces:

- `paper/iclr2027/release/main.pdf`
- `paper/iclr2027/release/main.tex`
- `paper/iclr2027/release/iclr2027_conference.sty`
- `paper/iclr2027/release/iclr2027_conference.bst`
- unchanged bibliography and figure assets
- `CONTENT_LOCK_REPORT.txt`
- build diagnostics and an uploadable `DRPO_ICLR2027_MECHANICAL_PORT.zip`

## Scope of this first PDF

The first PDF is a diagnostic migration, **not a content-optimized submission draft**. Page-limit pressure, float placement, line/table overflow, appendix pagination, and ICLR-specific submission statements are to be audited after the mechanical build. No manuscript wording is changed to solve those issues in this stage.
