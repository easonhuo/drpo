#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "paper" / "kdd2027"
MAIN = ROOT / "main.tex"
VERIFY = ROOT / "verify_content_lock.py"
BUILD = ROOT / "build.sh"

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
                r"\graphicspath{{../overleaf/}}",
                r"\setlength{\emergencystretch}{3em}",
                r"\acmConference[KDD '27]{The 33rd ACM SIGKDD Conference on Knowledge Discovery and Data Mining}{August 1--5, 2027}{San Jose, CA, USA}",
                r"\acmYear{2027}",
            ]
        ),
        "graphics path",
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
    diagnostic_gate = """if grep -Fq 'Overfull \\hbox' main.log; then
  echo 'Overfull horizontal box detected' >&2
  grep -Fn 'Overfull \\hbox' main.log >&2 || true
  exit 1
fi
"""
    anchor = "pdfinfo main.pdf > PDFINFO.txt\n"
    build = replace_once(build, anchor, diagnostic_gate + anchor, "PDF audit anchor")
    BUILD.write_text(build, encoding="utf-8")


if __name__ == "__main__":
    main()
