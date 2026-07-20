from __future__ import annotations

import hashlib
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "source_locked.tex"
KDD = ROOT / "main.tex"
EXPECTED_SOURCE_SHA256 = "6e5ee53daf390f50ee2f2098826c02882a53257cbbe97ccee4a4ace1cde45dae"
EXPECTED_CONTENT_SHA256 = "842bc044055bf9695a23ce26411df6b93859d2582c6eb877ab7751bf8e5d6708"
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



def require_match(pattern: str, text: str, name: str) -> str:
    match = re.search(pattern, text)
    if match is None:
        raise SystemExit(f"missing {name}")
    return match.group(1)


def strip_descriptions(text: str) -> str:
    return re.sub(r"\n?\s*\\Description\{[^{}]*\}", "", text)


def strip_formatting_only(text: str) -> str:
    text = text.replace(VERIFIER_FORMATTED, VERIFIER_ORIGINAL)
    text = text.replace(TOPR_FORMATTED, TOPR_ORIGINAL)
    return text


def canonical_original(text: str) -> str:
    text = strip_formatting_only(text)
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
    text = strip_formatting_only(strip_descriptions(text))
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
