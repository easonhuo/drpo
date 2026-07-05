#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import Rectangle

def _save(fig, stem: Path) -> None:
    fig.savefig(str(stem) + ".pdf", bbox_inches="tight", pad_inches=0.025)
    fig.savefig(str(stem) + ".svg", bbox_inches="tight", pad_inches=0.025)
    fig.savefig(str(stem) + ".png", dpi=600, bbox_inches="tight", pad_inches=0.025)

def _fmt_strength(v: float) -> str:
    if abs(v) < 1e-12:
        return "0"
    return f"{v:g}"

def _text_color(rgba):
    r, g, b, _ = rgba
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "white" if luminance < 0.52 else "black"

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data/fig_6_4_1_phase_transition_template.csv"))
    p.add_argument("--out", type=Path, default=Path("figures/fig_6_4_1_phase_transition"))
    a = p.parse_args()

    df = pd.read_csv(a.data).sort_values("strength_q_over_p").reset_index(drop=True)
    xpos = np.arange(len(df), dtype=float)
    strength = df["strength_q_over_p"].to_numpy(float)
    reward = df["heldout_reward"].to_numpy(float)
    lo = reward - df["heldout_ci_low"].to_numpy(float)
    hi = df["heldout_ci_high"].to_numpy(float) - reward
    shift = df["policy_shift"].to_numpy(float)
    ceiling = float(df["positive_only_ceiling"].iloc[0])

    # Slightly taller than v3 and with larger internal fonts.
    fig = plt.figure(figsize=(7.10, 3.78))
    gs = fig.add_gridspec(nrows=3, ncols=1, height_ratios=[3.05, 1.14, 1.08], hspace=0.13)
    ax = fig.add_subplot(gs[0, 0])
    ax_shift = fig.add_subplot(gs[1, 0], sharex=ax)
    ax_evt = fig.add_subplot(gs[2, 0], sharex=ax)

    ax.errorbar(
        xpos, reward, yerr=np.vstack([lo, hi]),
        marker="o", markersize=5.0, linewidth=1.60,
        elinewidth=1.15, capsize=2.8,
    )
    ax.axhline(ceiling, linestyle="--", linewidth=1.05)
    ax.text(len(df) - 0.06, ceiling + 0.020, "positive-only ceiling",
            ha="right", va="bottom", fontsize=9.4)

    phase_lines = [0.5, 5.5, 7.5]
    for xline in phase_lines:
        ax.axvline(xline, linestyle=":", linewidth=0.9)
        ax_shift.axvline(xline, linestyle=":", linewidth=0.9)

    ax.text(1.08, 1.055, "useful\nrepulsion", ha="center", va="top", fontsize=9.6)
    ax.text(6.58, 1.055, "over-\nextrapolation", ha="center", va="top", fontsize=9.6)
    ax.text(8.55, 1.055, "collapse", ha="center", va="top", fontsize=9.6)

    if "is_placeholder" in df.columns and int(df["is_placeholder"].max()) == 1:
        ax.text(0.01, 0.975, "layout draft", transform=ax.transAxes,
                ha="left", va="top", fontsize=8.8)

    ax.set_ylabel("Held-out\nreward", fontsize=12.2)
    ax.set_ylim(0.05, 1.12)
    ax.tick_params(axis="y", labelsize=9.8)
    ax.tick_params(axis="x", labelbottom=False, length=0)
    ax.grid(axis="y", linestyle="--", linewidth=0.50, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax_shift.plot(xpos, shift, marker="s", markersize=4.5, linewidth=1.45)
    ax_shift.set_ylabel("Policy\nshift", fontsize=11.4)
    ax_shift.tick_params(axis="y", labelsize=9.5)
    ax_shift.tick_params(axis="x", labelbottom=False, length=0)
    ax_shift.grid(axis="y", linestyle="--", linewidth=0.50, alpha=0.30)
    ax_shift.spines["top"].set_visible(False)
    ax_shift.spines["right"].set_visible(False)
    ax_shift.text(
        0.01, 0.90,
        "distance from positive-only target",
        transform=ax_shift.transAxes, ha="left", va="top", fontsize=8.7
    )

    ax_evt.set_ylim(0, 3)
    ylocs = [2.35, 1.50, 0.65]
    ax_evt.set_yticks(ylocs)
    ax_evt.set_yticklabels(["Task coll.", "Boundary", "NaN/Inf"], fontsize=9.4)
    ax_evt.set_xlabel(r"Effective negative strength $q/p$", fontsize=11.6, labelpad=2)
    ax_evt.set_xticks(xpos)
    ax_evt.set_xticklabels([_fmt_strength(v) for v in strength], fontsize=9.2)
    ax_evt.tick_params(axis="y", length=0, pad=4)
    ax_evt.tick_params(axis="x", length=2.5, pad=2)

    totals = df["n_seeds"].astype(int).to_numpy()
    event_rows = [
        df["task_collapse_count"].astype(int).to_numpy(),
        df["boundary_event_count"].astype(int).to_numpy(),
        df["nan_inf_count"].astype(int).to_numpy(),
    ]

    cmap = plt.get_cmap("Reds")
    norm = colors.Normalize(vmin=0.0, vmax=1.0)

    for vals, yy in zip(event_rows, ylocs):
        for xi, count, total in zip(xpos, vals, totals):
            frac = count / max(total, 1)
            rgba = cmap(norm(frac))
            ax_evt.add_patch(
                Rectangle(
                    (xi - 0.35, yy - 0.23), 0.70, 0.46,
                    facecolor=rgba, edgecolor="0.62", linewidth=0.45
                )
            )
            ax_evt.text(
                xi, yy, f"{count}/{total}",
                ha="center", va="center", fontsize=8.05,
                color=_text_color(rgba), fontweight="semibold"
            )

    for side in ["top", "right", "left"]:
        ax_evt.spines[side].set_visible(False)

    ax.set_xlim(-0.35, len(df) - 0.65)
    fig.subplots_adjust(left=0.112, right=0.992, top=0.985, bottom=0.130)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    _save(fig, a.out)
    plt.close(fig)

if __name__ == "__main__":
    main()
