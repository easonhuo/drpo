#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import pandas as pd

def _tc(cmap, norm, value: float) -> str:
    r, g, b, _ = cmap(norm(value))
    return "white" if 0.2126*r + 0.7152*g + 0.0722*b < 0.48 else "black"

def _save(fig, stem: Path) -> None:
    fig.savefig(str(stem) + ".pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(str(stem) + ".svg", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(str(stem) + ".png", dpi=600, bbox_inches="tight", pad_inches=0.02)

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data/fig_6_3_1_source_heatmap_data.csv"))
    p.add_argument("--out", type=Path, default=Path("figures/fig_6_3_1_source_heatmap"))
    a = p.parse_args()

    df = pd.read_csv(a.data)
    rows = ["C-U1 coefficient", "C-U1 score / |A|", "D-U1 coefficient", "D-U1 logit score"]
    row_labels = [r"C-COE", r"C-SC", r"D-COE", r"D-SC"]
    cols = ["near", "Q2", "Q3", "Q4", "Q5", "far"]

    vals = np.array([
        [df[(df["row"] == r) & (df["remoteness_bin"] == c)]["relative_value"].iloc[0] for c in cols]
        for r in rows
    ], dtype=float)
    shown = np.log2(np.maximum(vals, 1e-12))

    cmap = plt.get_cmap("cividis")
    norm = colors.Normalize(vmin=float(shown.min()), vmax=float(shown.max()))

    fig, ax = plt.subplots(figsize=(6.85, 1.50))
    im = ax.imshow(shown, aspect="auto", cmap=cmap, norm=norm)

    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(cols, fontsize=10.5)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=10.5)
    ax.set_xlabel("Remoteness quantile", fontsize=10.9, labelpad=2)

    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            val = vals[i, j]
            lab = f"{val:.1f}x" if val >= 10 else f"{val:.2f}x"
            ax.text(
                j, i, lab,
                ha="center", va="center",
                fontsize=8.9,
                color=_tc(cmap, norm, shown[i, j]),
                fontweight="semibold",
            )

    ax.axhline(1.5, color="white", linewidth=1.0, alpha=0.95)
    ax.axhline(1.5, color="black", linewidth=0.30, alpha=0.55)

    for s in ax.spines.values():
        s.set_linewidth(0.75)

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.008)
    cbar.set_label(r"$\log_2$ scale", fontsize=9.4, labelpad=3)
    cbar.ax.tick_params(labelsize=8.6, length=2.0)

    # Extra safety against left-edge clipping; short labels keep cells large.
    fig.subplots_adjust(left=0.083, right=0.964, bottom=0.318, top=0.985)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    _save(fig, a.out)
    plt.close(fig)

if __name__ == "__main__":
    main()
