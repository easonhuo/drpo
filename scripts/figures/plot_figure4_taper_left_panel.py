#!/usr/bin/env python3
"""Plot Figure 4 left panel: controlled taper operating-point comparison.

The right external-transfer table is LaTeX/tabular in
`results/FIGURE4_TAPER_CONTROL_TRANSFER/figure4_composite_tex_snippet.tex`.
"""
from __future__ import annotations
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path("results/FIGURE4_TAPER_CONTROL_TRANSFER/fig_6_4_2_leftfig_template.csv"))
    ap.add_argument("--out", type=Path, default=Path("paper/figures/fig_6_4_2_leftfig_bigtext_legend_protocol"))
    args = ap.parse_args()
    df = pd.read_csv(args.input)

    fs_axis, fs_tick, fs_legend, fs_legend_title, fs_point_val = 15.0, 13.2, 11.5, 11.0, 10.8
    methods = df["method"].tolist()
    x = np.arange(len(methods), dtype=float)
    bar_width, offset = 0.25, 0.16
    x_n, x_b = x - offset, x + offset

    fig, ax = plt.subplots(figsize=(7.15, 3.70))
    ax2 = ax.twinx()

    bars_near = ax.bar(x_n, df["near_match_near"], width=bar_width)
    near_color = bars_near.patches[0].get_facecolor()
    bars_far = ax.bar(x_n, df["near_match_far"], width=bar_width, bottom=df["near_match_near"])
    far_color = bars_far.patches[0].get_facecolor()
    ax.bar(x_b, df["budget_match_near"], width=bar_width, color=near_color)
    ax.bar(x_b, df["budget_match_far"], width=bar_width, bottom=df["budget_match_near"], color=far_color)

    reward_n, reward_b = df["near_match_reward"].to_numpy(), df["budget_match_reward"].to_numpy()
    ax2.plot(x_n, reward_n, marker="o", linestyle="--", linewidth=1.35, markersize=5.9)
    ax2.plot(x_b, reward_b, marker="s", linestyle="--", linewidth=1.35, markersize=5.7)

    for i, (xn, yn) in enumerate(zip(x_n, reward_n)):
        dx, dy = (0.0, 0.018)
        if i == 1:
            dx, dy = -0.025, 0.021
        if i == 2:
            dx, dy = 0.0, 0.017
        ax2.text(xn + dx, yn + dy, f"{yn:.2f}", ha="center", va="bottom", fontsize=fs_point_val)
    for i, (xb, yb) in enumerate(zip(x_b, reward_b)):
        dx, dy = (0.0, 0.018)
        if i == 1:
            dx, dy = 0.025, 0.021
        if i == 2:
            dx, dy = 0.0, 0.017
        if i == 3:
            dx = 0.01
        ax2.text(xb + dx, yb + dy, f"{yb:.2f}", ha="center", va="bottom", fontsize=fs_point_val)

    ax.set_ylim(0.0, 1.50)
    ax2.set_ylim(0.48, 1.02)
    ax.set_xlim(-0.55, len(methods) - 0.45)
    ax.set_ylabel("Realized negative influence", fontsize=fs_axis)
    ax2.set_ylabel("Held-out-context reward", fontsize=fs_axis)
    ax.set_xlabel("Method", fontsize=fs_axis, labelpad=3)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=fs_tick)
    ax.tick_params(axis="y", labelsize=fs_tick)
    ax2.tick_params(axis="y", labelsize=fs_tick)
    ax.tick_params(axis="x", length=0, pad=3)
    ax.grid(axis="y", linestyle="--", linewidth=0.55, alpha=0.32)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)

    handles = [
        Patch(facecolor=near_color, label="Near retained"),
        Patch(facecolor=far_color, label="Far retained"),
        Line2D([0], [0], marker="o", linestyle="--", linewidth=1.35, markersize=6.3, label="Reward (N)"),
        Line2D([0], [0], marker="s", linestyle="--", linewidth=1.35, markersize=6.1, label="Reward (B)"),
    ]
    leg = fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=fs_legend,
                     bbox_to_anchor=(0.52, 0.995), frameon=True,
                     title="Within each method, left bar = N; right bar = B")
    leg.get_title().set_fontsize(fs_legend_title)
    fig.subplots_adjust(top=0.765, bottom=0.19, left=0.115, right=0.875)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ["pdf", "svg", "png"]:
        fig.savefig(str(args.out) + "." + ext, dpi=320 if ext == "png" else None, bbox_inches="tight", pad_inches=0.03)

if __name__ == "__main__":
    main()
