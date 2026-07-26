from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OLD_SOURCE = "9f7415af1a60898266c99d31cc8573c030dfbcf274ca58ed8828a8842a7575e7"
NEW_SOURCE = "e64b74012a050f61e87c1601a301f2c5c452a9421d7e570d28098c74eb21a966"
OLD_CONTENT = "c20958a1e307633d80ddca46cf017e407ad80cfc629c56aec63f3ce2a36a2bdd"
NEW_CONTENT = "56260fe2712a462e03cfe3b61bb8ef23f6440bd413e823b8426f6bb12a56682f"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)


source_path = Path("paper/kdd2027/source_locked.tex")
text = source_path.read_text(encoding="utf-8")

replacements = [
    (
        """provided numbers and reaches the target value. Greedy verifier success
is the primary metric, while Pass@$k$ measures sampling-level solution
coverage and valid-expression rate distinguishes reasoning failure from
formatting or execution failure.""",
        """provided numbers and reaches the target value. Pass@8 is the primary task metric because it measures
sampling-level solution coverage under stochastic decoding, while
greedy verifier success provides a deterministic-decoding check.
Valid-expression rate is reported separately to distinguish reasoning
failure from formatting or execution failure.""",
    ),
    (
        """variant~\\citep{leroux2025tapered} as off-policy baselines.
The shared objectives and taper functions are defined in""",
        """variant~\\citep{leroux2025tapered} as off-policy baselines.
The corresponding validation sweeps and selected coefficients are
reported in Appendix~\\ref{app:countdown_baseline_sensitivity}.
The shared objectives and taper functions are defined in""",
    ),
    (
        """Greedy success (\\%)
& --
& --
& --
\\\\

Pass@$k$ (\\%)
& --
& --
& --
\\\\

Valid expression (\\%)
& --
& --
& --
\\\\""",
        """Greedy success (\\%)
& 6.80
& 7.53
& \\textbf{7.62}
\\\\

Pass@8 (\\%)
& 14.38
& 14.41
& \\textbf{15.72}
\\\\

Valid expression (\\%)
& \\textbf{99.48}
& 99.14
& 99.40
\\\\""",
    ),
    (
        """All methods initialize a fresh policy LoRA on the same
Qwen2.5-0.5B-Instruct backbone and share the frozen V2 negative bank,
validation split, test puzzles, and training budget. The TOPR arm
implements the joint fitted-reference \\(\\beta\\)-TOPR variant; it is not a
canonical frozen-behavior reproduction of \\citet{leroux2025tapered}. A
dash denotes a result pending the formal run.""",
        """All methods initialize a fresh policy LoRA on the same
Qwen2.5-0.5B-Instruct backbone. All entries are arithmetic means over
evaluation steps 800, 900, 1000, 1100, and 1200 on the held-out
validation split. The TOPR arm implements the joint fitted-reference
\\(\\beta\\)-TOPR variant; it is not a canonical frozen-behavior
reproduction of \\citet{leroux2025tapered}. Bold indicates the highest
observed value in each row.""",
    ),
    (
        """Table~\\ref{tab:countdown_performance} reserves the greedy success,
Pass@$k$, and validity fields for the formal run. Because all entries are
pending, we make no Countdown ranking or improvement claim in this
version.""",
        """DRPO delivers the strongest observed task performance among the
compared methods. The near-saturated valid-expression rates across all
three methods indicate that this difference is not obtained through a
loss of output well-formedness.""",
    ),
    (
        """otherwise. Hyperparameters are selected using the registered validation
rule. The held-out test set is evaluated only after checkpoint selection.""",
        """otherwise. Hyperparameters are selected using the registered validation
rule.""",
    ),
]

for index, (old, new) in enumerate(replacements, start=1):
    text = replace_once(text, old, new, f"manuscript replacement {index}")

anchor = """\\subsection{Countdown Taper-Coefficient Sensitivity}
\\label{app:countdown_taper_sensitivity}"""
insertion = r'''\subsection{Countdown Baseline-Coefficient Sensitivity}
\label{app:countdown_baseline_sensitivity}

Figure~\ref{fig:app_countdown_baseline_parameter_response} reports the
complete parameter-response sweeps of the two tuned Countdown baselines.
For AsymRE, performance is maximized at \(\delta_v=-1\) and drops
substantially as \(\delta_v\) increases, placing the strongest tested
finite-horizon configuration at the zero-negative boundary. For joint
fitted-reference \(\beta\)-TOPR, \(\beta=0\) performs poorly, while
positive \(\beta\) values recover performance and form a broad plateau
rather than a sharply isolated optimum. We therefore use
\(\delta_v=-1\) for AsymRE and \(\beta=0.25\) for joint
fitted-reference \(\beta\)-TOPR in the main comparison. The complete
sweeps show that DRPO's advantage is not an artifact of comparing
against isolated or untuned baseline settings.

\begin{figure}[t]
    \centering

    \begin{minipage}[t]{0.49\columnwidth}
        \centering
        \vspace{0pt}
        \textbf{(a) AsymRE}\\[0.05em]
        \includegraphics[
            width=\linewidth
        ]{figures/fig_app_countdown_asymre_parameter_response.pdf}
    \end{minipage}
    \hfill
    \begin{minipage}[t]{0.49\columnwidth}
        \centering
        \vspace{0pt}
        \textbf{(b) Joint fitted-reference \(\beta\)-TOPR}\\[0.05em]
        \includegraphics[
            width=\linewidth
        ]{figures/fig_app_countdown_topr_parameter_response.pdf}
    \end{minipage}

    \vspace{-0.35em}
    \caption{
        \textbf{Countdown-0.5B baseline-coefficient response.}
        \textbf{(a)} AsymRE late-window Pass@8 versus \(\delta_v\).
        \textbf{(b)} Joint fitted-reference \(\beta\)-TOPR late-window
        Pass@8 versus \(\beta\). Points denote means over the available
        development trajectories; error bars span their minimum--maximum
        range. The dashed line denotes DRPO at \(c=1.897\), and stars mark
        the baseline settings used in Table~\ref{tab:countdown_performance}:
        \(\delta_v=-1\) and \(\beta=0.25\).
    }
    \label{fig:app_countdown_baseline_parameter_response}
\end{figure}

'''
text = replace_once(text, anchor, insertion + anchor, "Countdown appendix insertion")
source_path.write_text(text, encoding="utf-8")
Path("paper/overleaf/main_replacement.tex").write_text(text, encoding="utf-8")

for filename in [
    "paper/kdd2027/verify_content_lock.py",
    "paper/kdd2027/build.sh",
    "paper/kdd2027/CONTENT_LOCK.txt",
    "scripts/generate_kdd2027_stage_a.py",
    "scripts/postprocess_kdd2027_stage_a.py",
    ".github/workflows/build-kdd2027-stage-a.yml",
]:
    path = Path(filename)
    data = path.read_text(encoding="utf-8")
    data = data.replace(OLD_SOURCE, NEW_SOURCE).replace(OLD_CONTENT, NEW_CONTENT)
    path.write_text(data, encoding="utf-8")

generator = Path("scripts/generate_kdd2027_stage_a.py")
data = generator.read_text(encoding="utf-8")
old_inventory = "'figures/fig_app_d4rl9_gradient_panels.pdf', 'figures/fig_app_countdown_taper_coefficient_response.pdf'"
new_inventory = "'figures/fig_app_d4rl9_gradient_panels.pdf', 'figures/fig_app_countdown_asymre_parameter_response.pdf', 'figures/fig_app_countdown_topr_parameter_response.pdf', 'figures/fig_app_countdown_taper_coefficient_response.pdf'"
data = replace_once(data, old_inventory, new_inventory, "generator verifier inventory")
data = replace_once(
    data,
    'kdd.count("\\\\Description{") != 5',
    'kdd.count("\\\\Description{") != 6',
    "generator description count",
)
data = replace_once(
    data,
    "expected five ACM figure descriptions",
    "expected six ACM figure descriptions",
    "generator description error",
)
old_build_inventory = "  ../overleaf/figures/fig_app_d4rl9_gradient_panels.pdf\n  ../overleaf/figures/fig_app_countdown_taper_coefficient_response.pdf"
new_build_inventory = "  ../overleaf/figures/fig_app_d4rl9_gradient_panels.pdf\n  ../overleaf/figures/fig_app_countdown_asymre_parameter_response.pdf\n  ../overleaf/figures/fig_app_countdown_topr_parameter_response.pdf\n  ../overleaf/figures/fig_app_countdown_taper_coefficient_response.pdf"
data = replace_once(data, old_build_inventory, new_build_inventory, "generator build inventory")
data = data.replace("five referenced figure assets", "nine referenced figure assets")
generator.write_text(data, encoding="utf-8")

post = Path("scripts/postprocess_kdd2027_stage_a.py")
data = post.read_text(encoding="utf-8")
write_anchor = '    MAIN.write_text(text, encoding="utf-8")\n'
description_code = r'''    countdown_description_anchor = r''' + "'''" + r'''    }
    \label{fig:app_countdown_baseline_parameter_response}''' + "'''" + r'''
    countdown_description_block = r''' + "'''" + r'''    }
    \Description{Single-column two-panel figure showing late-window Pass@8
    parameter-response curves for AsymRE and joint fitted-reference beta-TOPR.
    Both panels include the DRPO reference and mark the selected baseline
    setting.}
    \label{fig:app_countdown_baseline_parameter_response}''' + "'''" + r'''
    if countdown_description_anchor in text:
        text = replace_once(
            text,
            countdown_description_anchor,
            countdown_description_block,
            "Countdown baseline sensitivity description",
        )
    MAIN.write_text(text, encoding="utf-8")
'''
data = replace_once(data, write_anchor, description_code, "postprocess main write")
data = data.replace("seven local figure assets", "nine local figure assets")
post.write_text(data, encoding="utf-8")

workflow = Path(".github/workflows/build-kdd2027-stage-a.yml")
data = workflow.read_text(encoding="utf-8")
figure_anchor = "            fig_app_d4rl9_gradient_panels.pdf\n            fig_app_countdown_taper_coefficient_response.pdf"
figure_replacement = "            fig_app_d4rl9_gradient_panels.pdf\n            fig_app_countdown_asymre_parameter_response.pdf\n            fig_app_countdown_topr_parameter_response.pdf\n            fig_app_countdown_taper_coefficient_response.pdf"
data = replace_once(data, figure_anchor, figure_replacement, "CI figure inventory")
data = data.replace(')" -eq 7', ')" -eq 9')
data = data.replace("required_figures=7", "required_figures=9")
old_pages = "          grep -Fxq 'pages=27' paper/kdd2027/BUILD_AUDIT.txt\n          grep -Fxq 'required_figures=9' paper/kdd2027/BUILD_AUDIT.txt\n          grep -Fxq 'rendered_pages=27' paper/kdd2027/BUILD_AUDIT.txt"
new_pages = "          pages=\"$(awk -F= '$1 == \\\"pages\\\" {print $2}' paper/kdd2027/BUILD_AUDIT.txt)\"\n          rendered_pages=\"$(awk -F= '$1 == \\\"rendered_pages\\\" {print $2}' paper/kdd2027/BUILD_AUDIT.txt)\"\n          [[ \"$pages\" =~ ^[1-9][0-9]*$ ]]\n          [[ \"$rendered_pages\" == \"$pages\" ]]\n          grep -Fxq 'required_figures=9' paper/kdd2027/BUILD_AUDIT.txt"
data = replace_once(data, old_pages, new_pages, "CI dynamic page audit")
workflow.write_text(data, encoding="utf-8")

# Generate the two approved parameter-response figures from registered summaries.
drpo = 15.72
asymre = {
    -1.00: [14.52, 14.24, 14.52, 14.24],
    -0.95: [10.08, 10.60],
    -0.90: [10.04, 10.72],
    -0.85: [9.16, 10.84],
    -0.80: [10.68, 10.40],
    -0.70: [9.68, 7.72],
    -0.60: [10.00, 9.80],
    -0.50: [10.48, 10.32, 11.04, 10.76],
    -0.30: [10.52, 10.28],
    -0.20: [8.28, 9.00],
    -0.10: [7.24, 7.40],
    -0.05: [4.36, 5.28],
    0.00: [3.04, 2.24],
    0.10: [1.92, 1.68],
}
topr = {
    0.000: [2.60, 1.64, 2.04, 2.96],
    0.010: [12.28, 12.24],
    0.020: [11.28, 12.12],
    0.040: [11.72, 12.76],
    0.080: [13.04, 13.44],
    0.125: [14.68, 13.48],
    0.250: [15.76, 14.48, 13.68, 13.72],
    0.500: [12.68, 14.48, 14.40, 14.48],
    0.750: [13.92, 13.88],
    1.000: [13.08, 14.28],
    1.500: [14.08, 13.80],
    2.000: [14.76, 13.96],
    4.000: [13.80, 14.08],
}


def summarize(values: dict[float, list[float]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.array(sorted(values), dtype=float)
    y = np.array([np.mean(values[v]) for v in x])
    lo = np.array([y[i] - min(values[v]) for i, v in enumerate(x)])
    hi = np.array([max(values[v]) - y[i] for i, v in enumerate(x)])
    return x, y, np.vstack([lo, hi])


def draw(values: dict[float, list[float]], selected: float, xlabel: str, output: Path, *, symlog: bool = False) -> None:
    x, y, yerr = summarize(values)
    fig, ax = plt.subplots(figsize=(3.15, 2.65))
    ax.errorbar(x, y, yerr=yerr, marker="o", linewidth=1.4, markersize=3.3, capsize=2)
    ax.axhline(drpo, linestyle="--", linewidth=1.3, label="DRPO")
    index = int(np.where(np.isclose(x, selected))[0][0])
    ax.plot([x[index]], [y[index]], marker="*", markersize=8.5, linestyle="None")
    if symlog:
        ax.set_xscale("symlog", linthresh=0.01, linscale=0.8, base=10)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel("Late-window Pass@8 (%)", fontsize=8)
    ax.set_ylim(0, 17.2)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.tick_params(labelsize=6.5)
    fig.tight_layout(pad=0.45)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


out = Path("paper/kdd2027/figures")
out.mkdir(parents=True, exist_ok=True)
draw(asymre, -1.0, r"AsymRE coefficient $\delta_v$", out / "fig_app_countdown_asymre_parameter_response.pdf")
draw(topr, 0.25, r"TOPR coefficient $\beta$", out / "fig_app_countdown_topr_parameter_response.pdf", symlog=True)
overleaf = Path("paper/overleaf/figures")
overleaf.mkdir(parents=True, exist_ok=True)
for name in [
    "fig_app_countdown_asymre_parameter_response.pdf",
    "fig_app_countdown_topr_parameter_response.pdf",
]:
    (overleaf / name).write_bytes((out / name).read_bytes())

print("approved manuscript transformation prepared")
