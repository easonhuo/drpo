from pathlib import Path
import shutil
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import Rectangle

# Source data from the 7-point selected phase-transition variant.
# If unavailable, fall back to the full template and subset it.
SRC7 = Path('/mnt/data/DRPO_6_4_1_PHASE_TRANSITION_V6_SUBSAMPLED/fig_6_4_1_phase_transition_7pt_data.csv')
FULL = Path('/mnt/data/DRPO_6_4_1_PHASE_TRANSITION_V6_SUBSAMPLED/fig_6_4_1_phase_transition_full_template.csv')
outdir = Path('/mnt/data/DRPO_FIG3_7PT_PAPER_READY')
if outdir.exists():
    shutil.rmtree(outdir)
(outdir / 'figures').mkdir(parents=True)
(outdir / 'data').mkdir(parents=True)
(outdir / 'scripts').mkdir(parents=True)
(outdir / 'tex').mkdir(parents=True)

if SRC7.exists():
    df = pd.read_csv(SRC7)
else:
    full = pd.read_csv(FULL).sort_values('strength_q_over_p').reset_index(drop=True)
    df = full.iloc[[0,2,4,5,7,8,9]].copy().reset_index(drop=True)

df.to_csv(outdir / 'data' / 'fig3_phase_transition_7pt_data.csv', index=False)

def fmt(v: float) -> str:
    if abs(v) < 1e-12:
        return '0'
    if float(v).is_integer():
        return str(int(v))
    return f'{v:g}'

def text_color(rgba):
    r, g, b, _ = rgba
    lum = 0.2126*r + 0.7152*g + 0.0722*b
    return 'white' if lum < 0.52 else 'black'

x = np.arange(len(df), dtype=float)
strength = df['strength_q_over_p'].to_numpy(float)
reward = df['heldout_reward'].to_numpy(float)
lo = reward - df['heldout_ci_low'].to_numpy(float)
hi = df['heldout_ci_high'].to_numpy(float) - reward
shift = df['policy_shift'].to_numpy(float)
ceiling = float(df['positive_only_ceiling'].iloc[0])

# Publication-ready vector figure: 7 points, larger fonts, thicker lines,
# different top/middle colors.
fig = plt.figure(figsize=(7.35, 4.35))
gs = fig.add_gridspec(3, 1, height_ratios=[3.0, 1.2, 1.1], hspace=0.17)
ax = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1, 0], sharex=ax)
ax3 = fig.add_subplot(gs[2, 0], sharex=ax)

blue = '#0B5EA8'
orange = '#FF6B00'
separator = '#0B5EA8'

def setup_spines(a):
    a.spines['top'].set_visible(False)
    a.spines['right'].set_visible(False)
    a.spines['left'].set_linewidth(1.15)
    a.spines['bottom'].set_linewidth(1.15)

# Phase separators for 7 selected q/p values.
sep_positions = [0.5, 3.5, 4.5]
for xl in sep_positions:
    ax.axvline(xl, linestyle=':', linewidth=1.3, color=separator)
    ax2.axvline(xl, linestyle=':', linewidth=1.3, color=separator)

ax.errorbar(
    x, reward, yerr=np.vstack([lo, hi]), marker='o', markersize=7.0,
    linewidth=2.85, elinewidth=1.55, capsize=3.8, color=blue,
    markerfacecolor=blue, markeredgecolor=blue, zorder=3,
)
ax.axhline(ceiling, linestyle='--', linewidth=1.45, color=blue)
ax.text(len(df)-0.02, ceiling+0.025, 'positive-only ceiling', ha='right', va='bottom', fontsize=12.2)
ax.text(1.4, 1.075, 'useful\nrepulsion', ha='center', va='top', fontsize=13.2)
ax.text(4.0, 1.075, 'over-\nextrapolation', ha='center', va='top', fontsize=13.2)
ax.text(5.2, 1.075, 'collapse', ha='center', va='top', fontsize=13.2)
ax.set_ylabel('Held-out\nreward', fontsize=15.6)
ax.set_ylim(0.05, 1.12)
ax.tick_params(axis='y', labelsize=12.8, width=1.15)
ax.tick_params(axis='x', labelbottom=False, length=0)
ax.grid(axis='y', linestyle='--', linewidth=0.55, alpha=0.35)
setup_spines(ax)

ax2.plot(
    x, shift, marker='s', markersize=6.4, linewidth=2.65,
    color=orange, markerfacecolor=orange, markeredgecolor=orange, zorder=3,
)
ax2.set_ylabel('Policy\nshift', fontsize=14.8)
ax2.tick_params(axis='y', labelsize=12.4, width=1.15)
ax2.tick_params(axis='x', labelbottom=False, length=0)
ax2.grid(axis='y', linestyle='--', linewidth=0.55, alpha=0.30)
ax2.text(0.01, 0.90, 'distance from positive-only target', transform=ax2.transAxes, ha='left', va='top', fontsize=11.4)
setup_spines(ax2)

ax3.set_ylim(0, 3)
ylocs = [2.35, 1.50, 0.65]
ax3.set_yticks(ylocs)
ax3.set_yticklabels(['Task coll.', 'Boundary', 'NaN/Inf'], fontsize=12.1)
ax3.set_xlabel(r'Effective negative strength $q/p$', fontsize=15.2, labelpad=4)
ax3.set_xticks(x)
ax3.set_xticklabels([fmt(v) for v in strength], fontsize=13.0)
ax3.tick_params(axis='y', length=0, pad=4)
ax3.tick_params(axis='x', length=3.2, pad=2, width=1.15)

cmap = plt.get_cmap('Reds')
norm = colors.Normalize(vmin=0.0, vmax=1.0)
totals = df['n_seeds'].astype(int).to_numpy()
event_rows = [
    df['task_collapse_count'].astype(int).to_numpy(),
    df['boundary_event_count'].astype(int).to_numpy(),
    df['nan_inf_count'].astype(int).to_numpy(),
]
for vals, yy in zip(event_rows, ylocs):
    for xi, count, total in zip(x, vals, totals):
        frac = count / max(total, 1)
        rgba = cmap(norm(frac))
        ax3.add_patch(Rectangle((xi-0.36, yy-0.23), 0.72, 0.46, facecolor=rgba, edgecolor='0.62', linewidth=0.45))
        ax3.text(xi, yy, f'{count}/{total}', ha='center', va='center', fontsize=9.8, color=text_color(rgba), fontweight='semibold')
for side in ['top', 'right', 'left']:
    ax3.spines[side].set_visible(False)
ax3.spines['bottom'].set_linewidth(1.15)

ax.set_xlim(-0.35, len(df)-0.65)
fig.subplots_adjust(left=0.13, right=0.995, top=0.985, bottom=0.14)

stem = outdir / 'figures' / 'fig3_phase_transition_7pt_paper_ready'
fig.savefig(stem.with_suffix('.pdf'), bbox_inches='tight', pad_inches=0.03)
fig.savefig(stem.with_suffix('.svg'), bbox_inches='tight', pad_inches=0.03)
fig.savefig(stem.with_suffix('.png'), dpi=600, bbox_inches='tight', pad_inches=0.03)
plt.close(fig)

script_text = Path(__file__).read_text(encoding='utf-8') if '__file__' in globals() else ''
(outdir / 'scripts' / 'plot_fig3_phase_transition_7pt_paper_ready.py').write_text(script_text, encoding='utf-8')
(outdir / 'tex' / 'include_fig3_phase_transition.tex').write_text(r'''\begin{figure}[t]
    \centering
    \includegraphics[width=\columnwidth]{figures/fig3_phase_transition_7pt_paper_ready.pdf}
    \caption{Stable extrapolation from controlled negative feedback.}
    \label{fig:controlled_phase_transition}
\end{figure}
''', encoding='utf-8')
(outdir / 'README.md').write_text('''# Figure 3 paper-ready package

Use `figures/fig3_phase_transition_7pt_paper_ready.pdf` in the LaTeX paper.
The PDF and SVG are vector outputs generated from Matplotlib; the PNG is only for preview.

Files:
- figures/fig3_phase_transition_7pt_paper_ready.pdf: paper-ready vector figure
- figures/fig3_phase_transition_7pt_paper_ready.svg: vector figure for editing
- figures/fig3_phase_transition_7pt_paper_ready.png: preview image
- data/fig3_phase_transition_7pt_data.csv: plotted 7-point data
- scripts/plot_fig3_phase_transition_7pt_paper_ready.py: source plotting script
- tex/include_fig3_phase_transition.tex: minimal LaTeX include snippet
''', encoding='utf-8')

zip_path = Path('/mnt/data/DRPO_FIG3_7PT_PAPER_READY.zip')
if zip_path.exists():
    zip_path.unlink()
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
    for p in outdir.rglob('*'):
        if p.is_file():
            z.write(p, p.relative_to(outdir.parent))
print(zip_path)
