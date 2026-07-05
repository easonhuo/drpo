#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

def _tc(cmap, norm, value: float) -> str:
    r, g, b, _ = cmap(norm(value))
    return "white" if 0.2126*r + 0.7152*g + 0.0722*b < 0.50 else "black"

def _save(fig, stem: Path) -> None:
    fig.savefig(str(stem) + ".pdf", bbox_inches="tight", pad_inches=0.025)
    fig.savefig(str(stem) + ".svg", bbox_inches="tight", pad_inches=0.025)
    fig.savefig(str(stem) + ".png", dpi=600, bbox_inches="tight", pad_inches=0.025)

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--summary", type=Path, default=Path("data/fig_6_3_2_causal_summary.csv"))
    p.add_argument("--out", type=Path, default=Path("figures/fig_6_3_2_rescue_plot"))
    a = p.parse_args()

    sm = pd.read_csv(a.summary)
    order = ["baseline", "near_zero", "far_to_near", "far_zero", "far_cap", "global_scale", "positive_only"]
    labels = ["Base", "Near-zero", "Far→near", "Far-zero", "Far-cap", "Global", "Pos-only"]
    rows = sm.set_index("method").loc[order].reset_index()

    x = np.arange(len(rows))
    y = rows["retention"].to_numpy(float)
    lo = y - rows["ret_ci_low"].to_numpy(float)
    hi = rows["ret_ci_high"].to_numpy(float) - y
    n = rows["n"].astype(int).to_numpy() if "n" in rows.columns else np.full(len(rows), 20)
    cnt = rows["collapse_count"].astype(int).to_numpy()
    rate = rows["collapse_rate"].to_numpy(float)

    fig = plt.figure(figsize=(6.85, 2.02))
    gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[3.05, 0.62], hspace=0.028)
    ax = fig.add_subplot(gs[0, 0])
    strip = fig.add_subplot(gs[1, 0], sharex=ax)

    ax.errorbar(
        x, y,
        yerr=np.vstack([lo, hi]),
        marker="o",
        markersize=4.9,
        linewidth=1.50,
        elinewidth=1.15,
        capsize=2.7,
    )
    ax.plot(x, y, linewidth=1.0, alpha=0.65)

    th = 0.45
    ax.axhline(th, linestyle="--", linewidth=0.90, alpha=0.75)
    ax.text(0.02, th + 0.025, "task-collapse threshold", fontsize=9.2, va="bottom")

    # Edge-safe: no long rotated y-label. Caption defines the metric.
    ax.set_ylabel("Ret.", fontsize=12.2, labelpad=6)
    ax.set_ylim(0.10, 1.10)
    ax.tick_params(axis="y", labelsize=10.8)
    ax.tick_params(axis="x", labelbottom=False, length=0)
    ax.grid(axis="y", linestyle="--", linewidth=0.50, alpha=0.32)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    cmap = plt.get_cmap("Reds")
    norm = colors.Normalize(vmin=0.0, vmax=1.0)
    strip.set_ylim(0, 1)
    strip.set_yticks([0.5])
    strip.set_yticklabels(["Task\ncoll."], fontsize=9.8)
    strip.set_xticks(x)
    strip.set_xticklabels(labels, fontsize=10.1, rotation=18, ha="right")
    strip.tick_params(axis="x", length=0, pad=1)
    strip.tick_params(axis="y", length=0, pad=4)

    for i, (r, c, total) in enumerate(zip(rate, cnt, n)):
        strip.add_patch(
            Rectangle(
                (i - 0.43, 0.20),
                0.86,
                0.60,
                facecolor=cmap(norm(r)),
                edgecolor="black",
                linewidth=0.50,
            )
        )
        strip.text(
            i, 0.50, f"{c}/{total}",
            ha="center", va="center",
            fontsize=9.1,
            color=_tc(cmap, norm, r),
            fontweight="semibold",
        )

    for side in ["top", "right", "left"]:
        strip.spines[side].set_visible(False)
    strip.spines["bottom"].set_linewidth(0.65)
    strip.set_xlim(-0.55, len(rows) - 0.45)

    # Larger left margin protects labels; bbox_inches also prevents output-edge clipping.
    fig.subplots_adjust(left=0.095, right=0.995, bottom=0.260, top=0.985)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    _save(fig, a.out)
    plt.close(fig)

if __name__ == "__main__":
    main()
