#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "paper" / "kdd2027"
MAIN = ROOT / "main.tex"
VERIFY = ROOT / "verify_content_lock.py"
BUILD = ROOT / "build.sh"
README = ROOT / "README.md"

VERIFIER_ORIGINAL = r'''\begin{equation}
    R(x,y)
    =
    \mathbf 1
    \left[
        y
        \text{ is syntactically valid, uses }
        \mathcal N
        \text{ exactly once, and evaluates to }T
    \right].
    \label{eq:countdown_reward}
\end{equation}'''

VERIFIER_FORMATTED = r'''\begin{equation}
    R(x,y)
    =
    \mathbf 1
    \left[
    \begin{aligned}
        &y
        \text{ is syntactically valid, uses }
        \mathcal N
        \text{ exactly once,} \\
        &\text{and evaluates to }T
    \end{aligned}
    \right].
    \label{eq:countdown_reward}
\end{equation}'''

TOPR_ORIGINAL = r'''\[
\begin{aligned}
g_{\mathrm{TOPR}}(\theta)
={}&
\mathbb{E}_{(x,y)\sim\mu}
\Big[
\mathbf{1}\{(x,y)\in\mathcal{T}^{+}\}
R(x,y)
\nabla_\theta\log\pi_\theta(y\mid x)
\Big]
\\
&+
\mathbb{E}_{(x,y)\sim\mu}
\Big[
\mathbf{1}\{(x,y)\in\mathcal{T}^{-}\}
\min\{\rho_\theta(x,y),1\}
R(x,y)
\nabla_\theta\log\pi_\theta(y\mid x)
\Big].
\end{aligned}
\]'''

TOPR_FORMATTED = r'''\[
\begin{aligned}
g_{\mathrm{TOPR}}(\theta)
={}&
\mathbb{E}_{(x,y)\sim\mu}
\Big[
\mathbf{1}\{(x,y)\in\mathcal{T}^{+}\}
R(x,y) \\
&\hspace{4.8em}{}
\nabla_\theta\log\pi_\theta(y\mid x)
\Big]
\\
&+
\mathbb{E}_{(x,y)\sim\mu}
\Big[
\mathbf{1}\{(x,y)\in\mathcal{T}^{-}\}
\min\{\rho_\theta(x,y),1\}
R(x,y) \\
&\hspace{4.8em}{}
\nabla_\theta\log\pi_\theta(y\mid x)
\Big].
\end{aligned}
\]'''

VERIFY_CONSTANTS = f'''\nVERIFIER_ORIGINAL = r\'\'\'{VERIFIER_ORIGINAL}\'\'\'\n\nVERIFIER_FORMATTED = r\'\'\'{VERIFIER_FORMATTED}\'\'\'\n\nTOPR_ORIGINAL = r\'\'\'{TOPR_ORIGINAL}\'\'\'\n\nTOPR_FORMATTED = r\'\'\'{TOPR_FORMATTED}\'\'\'\n'''

FORMAT_HELPER = r'''


def strip_formatting_only(text: str) -> str:
    text = text.replace(VERIFIER_FORMATTED, VERIFIER_ORIGINAL)
    text = text.replace(TOPR_FORMATTED, TOPR_ORIGINAL)
    return text
'''

README_TEXT = r'''# DRPO KDD 2027 Stage-A manuscript

This directory is a self-contained, format-only KDD 2027 Research Track
migration of the locked manuscript in `source_locked.tex`.

- Format: `\documentclass[sigconf,anonymous,review]{acmart}`
- Scope: Stage A only; no page-limit compression or scientific-text editing
- Compiler: pdfLaTeX
- Canonical TeX environment: TeX Live 2025
- Figures: local `figures/` directory
- Bibliography: local `example_paper.bib` and `missing_references.bib`
- Release PDF: `paper/releases/DRPO_KDD2027_STAGE_A.pdf`
- Overleaf upload bundle: `paper/releases/DRPO_KDD2027_OVERLEAF.zip`

## Overleaf

Upload `DRPO_KDD2027_OVERLEAF.zip` as a new project. The ZIP is packaged with
`main.tex` at the project root, so no path edits are required. In project
settings select:

- Main document: `main.tex`
- Compiler: pdfLaTeX
- TeX Live version: 2025

## Repository build

From the repository root:

```bash
bash paper/kdd2027/build.sh
```

The build requires TeX Live 2025 and verifies the locked source hash,
character-equivalent manuscript content, five local figure assets, resolved
LaTeX references/citations, anonymous PDF text and metadata, US Letter page
size, embedded fonts, and a complete PNG render of every page. Page count is
recorded but not gated in Stage A.
'''


def replace_once(text: str, old: str, new: str, name: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"expected one {name}, found {count}")
    return text.replace(old, new, 1)


def main() -> None:
    text = MAIN.read_text(encoding="utf-8")
    text = replace_once(
        text,
        r"\graphicspath{{../overleaf/}}",
        "\n".join(
            [
                r"\graphicspath{{figures/}}",
                r"\setlength{\emergencystretch}{3em}",
                r"\acmConference[KDD '27]{The 33rd ACM SIGKDD Conference on Knowledge Discovery and Data Mining}{August 1--5, 2027}{San Jose, CA, USA}",
                r"\acmYear{2027}",
            ]
        ),
        "graphics path",
    )
    text = replace_once(
        text,
        r"\bibliography{../overleaf/example_paper,../overleaf/missing_references}",
        r"\bibliography{example_paper,missing_references}",
        "bibliography path",
    )
    text = replace_once(
        text,
        r"\keywords{Off-Policy Reinforcement Learning, Negative Feedback, Policy Optimization, Stability}",
        "\n".join(
            [
                r"\ccsdesc[500]{Computing methodologies~Reinforcement learning}",
                "",
                r"\keywords{Off-Policy Reinforcement Learning, Negative Feedback, Policy Optimization, Stability}",
            ]
        ),
        "keywords",
    )
    text = replace_once(text, VERIFIER_ORIGINAL, VERIFIER_FORMATTED, "Countdown verifier equation")
    text = replace_once(text, TOPR_ORIGINAL, TOPR_FORMATTED, "TOPR equation")
    MAIN.write_text(text, encoding="utf-8")

    verify = VERIFY.read_text(encoding="utf-8")
    hash_line = 'EXPECTED_CONTENT_SHA256 = "842bc044055bf9695a23ce26411df6b93859d2582c6eb877ab7751bf8e5d6708"'
    verify = replace_once(verify, hash_line, hash_line + VERIFY_CONSTANTS, "content hash line")
    strip_fn = '''def strip_descriptions(text: str) -> str:\n    return re.sub(r"\\n?\\s*\\\\Description\\{[^{}]*\\}", "", text)\n'''
    verify = replace_once(verify, strip_fn, strip_fn + FORMAT_HELPER, "description stripper")
    verify = replace_once(
        verify,
        "def canonical_original(text: str) -> str:\n    title =",
        "def canonical_original(text: str) -> str:\n    text = strip_formatting_only(text)\n    title =",
        "original canonicalizer",
    )
    verify = replace_once(
        verify,
        "def canonical_kdd(text: str) -> str:\n    text = strip_descriptions(text)",
        "def canonical_kdd(text: str) -> str:\n    text = strip_formatting_only(strip_descriptions(text))",
        "KDD canonicalizer",
    )
    VERIFY.write_text(verify, encoding="utf-8")

    build = BUILD.read_text(encoding="utf-8")
    build = build.replace("../overleaf/figures/", "figures/")
    build = replace_once(
        build,
        "for file in ../overleaf/example_paper.bib ../overleaf/missing_references.bib; do",
        "for file in example_paper.bib missing_references.bib; do",
        "bibliography inventory",
    )
    compile_block = """latexmk -C main.tex >/dev/null 2>&1 || true
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
test -s main.pdf
"""
    compile_replacement = """if [[ "${DRPO_SKIP_LATEX_COMPILE:-0}" != "1" ]]; then
  pdflatex --version > TEXLIVE_ENVIRONMENT.txt
  latexmk -C main.tex >/dev/null 2>&1 || true
  latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
fi
test -s main.pdf
test -s TEXLIVE_ENVIRONMENT.txt
if ! grep -Fq 'TeX Live 2025' TEXLIVE_ENVIRONMENT.txt; then
  echo 'KDD canonical build requires TeX Live 2025' >&2
  cat TEXLIVE_ENVIRONMENT.txt >&2
  exit 1
fi
"""
    build = replace_once(build, compile_block, compile_replacement, "TeX Live compile block")
    diagnostic_gate = """if grep -Fq 'Overfull \\hbox' main.log; then
  echo 'Overfull horizontal box detected' >&2
  grep -Fn 'Overfull \\hbox' main.log >&2 || true
  exit 1
fi
"""
    anchor = "pdfinfo main.pdf > PDFINFO.txt\n"
    build = replace_once(build, anchor, diagnostic_gate + anchor, "PDF audit anchor")
    build = replace_once(
        build,
        '  echo "page_size=$page_size"\n',
        '  echo "page_size=$page_size"\n  echo "texlive_version=2025"\n',
        "TeX Live audit field",
    )
    BUILD.write_text(build, encoding="utf-8")
    README.write_text(README_TEXT, encoding="utf-8")


if __name__ == "__main__":
    main()
