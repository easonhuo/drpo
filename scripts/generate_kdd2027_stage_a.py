#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / 'paper' / 'kdd2027'
SOURCE = OUT / 'source_locked.tex'
EXPECTED_SOURCE_SHA256 = '6e5ee53daf390f50ee2f2098826c02882a53257cbbe97ccee4a4ace1cde45dae'
EXPECTED_CONTENT_SHA256 = '842bc044055bf9695a23ce26411df6b93859d2582c6eb877ab7751bf8e5d6708'

PREAMBLE = r'''\documentclass[sigconf,anonymous,review]{acmart}

% KDD 2027 Research Track review format: ACM sigconf, anonymous, review.
% The scientific manuscript content is locked to source_locked.tex.
\usepackage{mathtools}
\usepackage[capitalize,noabbrev]{cleveref}
\graphicspath{{../overleaf/}}

% Keep the two manuscript-specific theorem environments while using acmart's
% native theorem/proposition/lemma/corollary/definition environments.
\AtEndPreamble{%
  \theoremstyle{acmdefinition}
  \newtheorem{assumption}[theorem]{Assumption}
  \newtheorem{remark}[theorem]{Remark}
}

% Review copies must not print placeholder copyright/DOI/ISBN metadata.
\setcopyright{none}
\settopmatter{printacmref=false,printccs=false}
\renewcommand\footnotetextcopyrightpermission[1]{}

\begin{document}
'''

DESCRIPTIONS = {
    'fig:external_far_field': 'Two-panel diagnostic figure. The Hopper panel compares relative actor-gradient magnitude and matched absolute advantage across standardized-distance bins. The Countdown panel compares relative actor-gradient magnitude and fixed negative coefficient across mean-token-surprisal deciles.',
    'fig:controlled_6_3': 'Two-part controlled-mechanism figure. The top heatmap factorizes coefficient magnitude and policy-score response across remoteness bins. The bottom intervention plot compares reward retention and task-collapse counts across near-field and far-field interventions.',
    'fig:phase_transition_6_4_1': 'Phase-transition figure showing held-out-context reward and policy displacement as effective negative strength increases, with separate indicators for task collapse, boundary events, and numerical failure.',
    'fig:taper_control_and_transfer': 'Composite figure. The controlled panel compares near-field and far-field retained negative-update magnitude under matched taper operating points. The transfer panel is a reserved table for Hopper return and Countdown success.',
}

VERIFY = r'''from __future__ import annotations

import hashlib
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "source_locked.tex"
KDD = ROOT / "main.tex"
EXPECTED_SOURCE_SHA256 = "6e5ee53daf390f50ee2f2098826c02882a53257cbbe97ccee4a4ace1cde45dae"
EXPECTED_CONTENT_SHA256 = "842bc044055bf9695a23ce26411df6b93859d2582c6eb877ab7751bf8e5d6708"


def require_match(pattern: str, text: str, name: str) -> str:
    match = re.search(pattern, text)
    if match is None:
        raise SystemExit(f"missing {name}")
    return match.group(1)


def strip_descriptions(text: str) -> str:
    return re.sub(r"\n?\s*\\Description\{[^{}]*\}", "", text)


def canonical_original(text: str) -> str:
    title = require_match(r"\\icmltitle\{([^{}]+)\}", text, "ICML title")
    keywords = require_match(r"\\icmlkeywords\{([^{}]+)\}", text, "ICML keywords")
    abstract_start = text.index("\\begin{abstract}")
    abstract_end = text.index("\\end{abstract}", abstract_start) + len("\\end{abstract}")
    introduction_start = text.index("\\section{Introduction}", abstract_end)
    bibliography_start = text.index("\\bibliography{", introduction_start)
    appendix_start = text.index("\\appendix", bibliography_start)
    document_end = text.rindex("\\end{document}")
    appendix = text[appendix_start:document_end].replace("\\appendix\n\\onecolumn", "\\appendix", 1)
    return "\n".join([title, keywords, text[abstract_start:abstract_end], text[introduction_start:bibliography_start], appendix])


def canonical_kdd(text: str) -> str:
    text = strip_descriptions(text)
    title = require_match(r"\\title\[[^]]*\]\{([^{}]+)\}", text, "ACM title")
    keywords = require_match(r"\\keywords\{([^{}]+)\}", text, "ACM keywords")
    abstract_start = text.index("\\begin{abstract}")
    abstract_end = text.index("\\end{abstract}", abstract_start) + len("\\end{abstract}")
    introduction_start = text.index("\\section{Introduction}", abstract_end)
    bibliography_start = text.index("\\bibliographystyle{", introduction_start)
    appendix_start = text.index("\\appendix", bibliography_start)
    document_end = text.rindex("\\end{document}")
    return "\n".join([title, keywords, text[abstract_start:abstract_end], text[introduction_start:bibliography_start], text[appendix_start:document_end]])


source = SOURCE.read_text(encoding="utf-8")
kdd = KDD.read_text(encoding="utf-8")
source_sha = hashlib.sha256(source.encode("utf-8")).hexdigest()
if source_sha != EXPECTED_SOURCE_SHA256:
    raise SystemExit(f"source snapshot mismatch: expected {EXPECTED_SOURCE_SHA256}, got {source_sha}")
original_content = canonical_original(source)
kdd_content = canonical_kdd(kdd)
if original_content != kdd_content:
    import difflib
    diff = "".join(difflib.unified_diff(original_content.splitlines(True), kdd_content.splitlines(True), fromfile="source-canonical", tofile="kdd-canonical", n=3))
    (ROOT / "content_lock.diff").write_text(diff, encoding="utf-8")
    raise SystemExit("canonical manuscript content differs; see content_lock.diff")
content_sha = hashlib.sha256(original_content.encode("utf-8")).hexdigest()
if content_sha != EXPECTED_CONTENT_SHA256:
    raise SystemExit(f"canonical content hash mismatch: expected {EXPECTED_CONTENT_SHA256}, got {content_sha}")
for token in [r"\documentclass[sigconf,anonymous,review]{acmart}", r"\bibliographystyle{ACM-Reference-Format}", r"\maketitle"]:
    if token not in kdd:
        raise SystemExit(f"missing required KDD format token: {token}")
for forbidden in [r"\usepackage[preprint]{icml2026}", r"\icmlauthor", r"\icmlaffiliation", r"\icmlcorrespondingauthor", r"\onecolumn", "Tencent Inc", "neoxzhang@tencent.com", "Yusen Huo", "Changping Wang", "Yangru Huang", "Jun Zhang", "Jie Jiang"]:
    if forbidden in kdd:
        raise SystemExit(f"anonymous KDD source contains forbidden token: {forbidden}")
figures = re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^{}]+)\}", kdd, flags=re.S)
expected_figures = ["figures/figure1_external_gradient_bottom_labels.pdf", "figures/fig_6_3_1_source_heatmap.pdf", "figures/fig_6_3_2_rescue_plot.pdf", "figures/fig_6_4_1_phase_transition.pdf", "figures/fig_6_4_2_leftfig_bigtext_legend_protocol.pdf"]
if figures != expected_figures:
    raise SystemExit(f"figure inventory mismatch: {figures!r}")
if kdd.count("\\Description{") != 4:
    raise SystemExit("expected four ACM figure descriptions")
print(f"source_sha256={source_sha}")
print(f"canonical_content_sha256={content_sha}")
print("content_lock=PASS")
print(f"figure_inventory=PASS ({len(figures)} files)")
'''

BUILD = r'''#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
python3 verify_content_lock.py
required_figures=(
  ../overleaf/figures/figure1_external_gradient_bottom_labels.pdf
  ../overleaf/figures/fig_6_3_1_source_heatmap.pdf
  ../overleaf/figures/fig_6_3_2_rescue_plot.pdf
  ../overleaf/figures/fig_6_4_1_phase_transition.pdf
  ../overleaf/figures/fig_6_4_2_leftfig_bigtext_legend_protocol.pdf
)
for file in "${required_figures[@]}"; do test -s "$file" || { echo "missing figure: $file" >&2; exit 1; }; done
for file in ../overleaf/example_paper.bib ../overleaf/missing_references.bib; do test -s "$file" || { echo "missing bibliography: $file" >&2; exit 1; }; done
latexmk -C main.tex >/dev/null 2>&1 || true
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
test -s main.pdf
if grep -Eqi 'Undefined control sequence|LaTeX Error|Citation .* undefined|Reference .* undefined|There were undefined references|There were undefined citations|multiply defined|Fatal error' main.log; then
  echo 'Unresolved LaTeX diagnostic detected' >&2
  grep -Ein 'Undefined control sequence|LaTeX Error|Citation .* undefined|Reference .* undefined|There were undefined references|There were undefined citations|multiply defined|Fatal error' main.log >&2 || true
  exit 1
fi
pdfinfo main.pdf > PDFINFO.txt
pdffonts main.pdf > PDFFONTS.txt
pdftotext main.pdf main.txt
for token in 'Yusen Huo' 'Changping Wang' 'Yangru Huang' 'Jun Zhang' 'Jie Jiang' 'Tencent Inc' 'neoxzhang@tencent.com'; do
  if grep -Fqi "$token" main.txt PDFINFO.txt; then echo "anonymous PDF contains forbidden identity token: $token" >&2; exit 1; fi
done
pages="$(awk '/^Pages:/ {print $2}' PDFINFO.txt)"
page_size="$(awk -F': +' '/^Page size:/ {print $2}' PDFINFO.txt)"
[[ -n "$pages" && "$pages" -ge 1 ]] || { echo 'invalid PDF page count' >&2; exit 1; }
[[ "$page_size" == *'612 x 792 pts'* ]] || { echo "unexpected page size: $page_size" >&2; exit 1; }
if awk 'NR>2 && ($5 != "yes" || $6 != "yes") {bad=1} END {exit bad}' PDFFONTS.txt; then :; else echo 'non-embedded or non-subset font detected' >&2; cat PDFFONTS.txt >&2; exit 1; fi
mkdir -p renders
rm -f renders/page-*.png
pdftoppm -png -r 150 main.pdf renders/page >/dev/null 2>&1
render_count="$(find renders -maxdepth 1 -name 'page-*.png' -type f | wc -l | tr -d ' ')"
[[ "$render_count" == "$pages" ]] || { echo "render count mismatch: pages=$pages renders=$render_count" >&2; exit 1; }
{
  echo "source_sha256=6e5ee53daf390f50ee2f2098826c02882a53257cbbe97ccee4a4ace1cde45dae"
  echo "canonical_content_sha256=842bc044055bf9695a23ce26411df6b93859d2582c6eb877ab7751bf8e5d6708"
  echo "pages=$pages"
  echo "page_size=$page_size"
  echo "required_figures=${#required_figures[@]}"
  echo "rendered_pages=$render_count"
  echo "pdf_sha256=$(sha256sum main.pdf | awk '{print $1}')"
  echo "content_lock=PASS"
  echo "anonymous_pdf=PASS"
  echo "latex_diagnostics=PASS"
  echo "font_embedding=PASS"
  echo "render_gate=PASS"
} > BUILD_AUDIT.txt
cat BUILD_AUDIT.txt
'''

README = r'''# DRPO KDD 2027 Stage-A manuscript

This directory is the format-only KDD 2027 Research Track migration of the
locked manuscript in `source_locked.tex`.

- Format: `\documentclass[sigconf,anonymous,review]{acmart}`
- Scope: Stage A only; no page-limit compression or scientific-text editing
- Figures: reused from `paper/overleaf/figures/`
- Bibliography: reused from `paper/overleaf/example_paper.bib` and `paper/overleaf/missing_references.bib`
- Release PDF: `paper/releases/DRPO_KDD2027_STAGE_A.pdf`

Build from the repository root:

```bash
bash paper/kdd2027/build.sh
```

The build verifies the locked source hash, character-equivalent manuscript
content, five referenced figure assets, LaTeX references/citations, anonymous
PDF text and metadata, US Letter page size, embedded fonts, and a complete PNG
render of every page. Page count is recorded but not gated in Stage A.
'''


def match(pattern: str, text: str, name: str) -> str:
    m = re.search(pattern, text)
    if not m:
        raise SystemExit(f'missing {name}')
    return m.group(1)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    source = SOURCE.read_text(encoding='utf-8')
    digest = hashlib.sha256(source.encode()).hexdigest()
    if digest != EXPECTED_SOURCE_SHA256:
        raise SystemExit(f'locked source mismatch: {digest}')
    title = match(r'\\icmltitle\{([^{}]+)\}', source, 'title')
    keywords = match(r'\\icmlkeywords\{([^{}]+)\}', source, 'keywords')
    a0 = source.index('\\begin{abstract}')
    a1 = source.index('\\end{abstract}', a0) + len('\\end{abstract}')
    i0 = source.index('\\section{Introduction}', a1)
    b0 = source.index('\\bibliography{', i0)
    app0 = source.index('\\appendix', b0)
    e0 = source.rindex('\\end{document}')
    abstract = source[a0:a1]
    body = source[i0:b0]
    appendix = source[app0:e0].replace('\\appendix\n\\onecolumn', '\\appendix', 1)
    for label, desc in DESCRIPTIONS.items():
        marker = f'\\label{{{label}}}'
        if marker not in body:
            raise SystemExit(f'missing figure label: {label}')
        m = re.search(r'(?m)^([ \t]*)' + re.escape(marker), body)
        if not m:
            raise SystemExit(f'missing figure label line: {label}')
        indent = m.group(1)
        body = body[:m.start()] + indent + f'\\Description{{{desc}}}\n' + indent + marker + body[m.end():]
    out = PREAMBLE
    out += f'\n\\title[Remoteness-Aware Control of Negative Off-Policy Updates]{{{title}}}\n\n'
    out += '% Author names, affiliations, emails, acknowledgments, and funding are\n% intentionally omitted from the anonymous KDD review manuscript.\n\n'
    out += abstract + '\n\n'
    out += f'\\keywords{{{keywords}}}\n\n\\maketitle\n\n'
    out += body
    out += '\\bibliographystyle{ACM-Reference-Format}\n\\bibliography{../overleaf/example_paper,../overleaf/missing_references}\n\n'
    out += appendix + '\\end{document}\n'
    (OUT / 'main.tex').write_text(out, encoding='utf-8')
    (OUT / 'verify_content_lock.py').write_text(VERIFY, encoding='utf-8')
    (OUT / 'build.sh').write_text(BUILD, encoding='utf-8')
    (OUT / 'build.sh').chmod(0o755)
    (OUT / 'README.md').write_text(README, encoding='utf-8')
    (OUT / 'CONTENT_LOCK.txt').write_text(f'source_sha256={EXPECTED_SOURCE_SHA256}\ncanonical_content_sha256={EXPECTED_CONTENT_SHA256}\nclaim=PAPER-KDD-2027-TEMPLATE-MIGRATION-01\n', encoding='utf-8')
    print(f'generated {OUT / "main.tex"}')


if __name__ == '__main__':
    main()
