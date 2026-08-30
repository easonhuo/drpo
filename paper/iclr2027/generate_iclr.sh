#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SOURCE="$ROOT/paper/overleaf/main_replacement.tex"
OUT_DIR="${1:-$ROOT/paper/iclr2027/build}"
EXPECTED_BLOB="e2e592caa780f0ba4c5edc31986f6ea4106ef000"

mkdir -p "$OUT_DIR"

ACTUAL_BLOB="$(git -C "$ROOT" rev-parse "HEAD:paper/overleaf/main_replacement.tex")"
if [[ "$ACTUAL_BLOB" != "$EXPECTED_BLOB" ]]; then
  echo "CONTENT LOCK FAILURE: source blob drifted" >&2
  echo "expected=$EXPECTED_BLOB" >&2
  echo "actual=$ACTUAL_BLOB" >&2
  exit 2
fi

python3 - "$SOURCE" "$OUT_DIR/main.tex" "$OUT_DIR/CONTENT_LOCK_REPORT.txt" "$EXPECTED_BLOB" <<'PY'
from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

source_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])
report_path = Path(sys.argv[3])
expected_blob = sys.argv[4]

source = source_path.read_text(encoding="utf-8")

abstract_marker = r"\begin{abstract}"
if source.count(abstract_marker) != 1:
    raise SystemExit(f"expected one {abstract_marker}, found {source.count(abstract_marker)}")
body_start = source.index(abstract_marker)
body = source[body_start:]

old_bibstyle = r"\bibliographystyle{icml2026}"
new_bibstyle = r"\bibliographystyle{iclr2027_conference}"
if body.count(old_bibstyle) != 1:
    raise SystemExit(f"expected one {old_bibstyle}, found {body.count(old_bibstyle)}")

# No ICML-specific document machinery is allowed to leak into the preserved body.
remaining_icml = [line for line in body.splitlines() if "\\icml" in line]
if remaining_icml:
    raise SystemExit("ICML-specific macro found inside locked manuscript body: " + repr(remaining_icml[:5]))

m = re.search(r"\\icmltitle\{([^{}]+)\}", source)
if not m:
    raise SystemExit("could not recover the source title")
title = m.group(1)
expected_title = "Breaking the Curse of Repulsion: Remoteness-Aware Control of Negative Off-Policy Updates"
if title != expected_title:
    raise SystemExit(f"title drift: {title!r}")

# Format-only wrapper. Manuscript content begins at \begin{abstract} below.
preamble = rf'''\documentclass{{article}}

% ICLR 2027 review format. The official style file is supplied by build.sh.
\usepackage{{iclr2027_conference,times}}

% Packages retained from the active manuscript wrapper.
\usepackage{{microtype}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\usepackage{{makecell}}
\usepackage{{hyperref}}
\usepackage{{amsmath}}
\usepackage{{amssymb}}
\usepackage{{mathtools}}
\usepackage{{amsthm}}
\usepackage[capitalize,noabbrev]{{cleveref}}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% THEOREMS -- unchanged numbering contract from the active manuscript wrapper
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\theoremstyle{{plain}}
\newtheorem{{theorem}}{{Theorem}}[section]
\newtheorem{{proposition}}[theorem]{{Proposition}}
\newtheorem{{lemma}}[theorem]{{Lemma}}
\newtheorem{{corollary}}[theorem]{{Corollary}}
\theoremstyle{{definition}}
\newtheorem{{definition}}[theorem]{{Definition}}
\newtheorem{{assumption}}[theorem]{{Assumption}}
\theoremstyle{{remark}}
\newtheorem{{remark}}[theorem]{{Remark}}

\setlength{{\abovedisplayskip}}{{3pt}}
\setlength{{\belowdisplayskip}}{{3pt}}

% The title string is copied character-for-character from the active source.
\title{{{title}}}

% Author metadata is retained in the TeX source. The ICLR review style renders
% the anonymous double-blind author block when \iclrfinalcopy is not enabled.
\author{{Yusen Huo \And Changping Wang \And Yangru Huang \And Jun Zhang \And Jie Jiang}}

% Source metadata retained verbatim for provenance; it is not typeset in review mode.
% Tencent Inc, China
% Correspondence to: Jun Zhang <neoxzhang@tencent.com>
% Off-Policy Reinforcement Learning, Negative Feedback, Policy Optimization, Stability

\begin{{document}}
\maketitle

'''

generated_body = body.replace(old_bibstyle, new_bibstyle)
generated = preamble + generated_body
out_path.write_text(generated, encoding="utf-8")

# Canonical content lock: normalize only the bibliography style selector.
def canonicalize(s: str, style: str) -> str:
    return s.replace(style, r"\bibliographystyle{<FORMAT_STYLE>}")

source_canonical = canonicalize(body, old_bibstyle)
generated_canonical = canonicalize(generated_body, new_bibstyle)
if source_canonical != generated_canonical:
    raise SystemExit("canonical manuscript body differs after format-only migration")

sha = lambda text: hashlib.sha256(text.encode("utf-8")).hexdigest()
report = [
    "MANUSCRIPT-ICLR2027-PORT-01 CONTENT LOCK",
    f"source={source_path}",
    f"expected_source_git_blob={expected_blob}",
    f"source_body_sha256={sha(source_canonical)}",
    f"generated_body_sha256={sha(generated_canonical)}",
    f"body_lock={'PASS' if sha(source_canonical) == sha(generated_canonical) else 'FAIL'}",
    f"title={title}",
    "allowed_format_change=bibliographystyle icml2026 -> iclr2027_conference",
    "abstract_to_appendix_text_change=NONE",
]
report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
print("\n".join(report))
PY
