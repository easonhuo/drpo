#!/usr/bin/env python3
"""Generate deterministic conceptual figures for the DRPO manuscript draft."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def story_figure(output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.7))

    ax = axes[0]
    x = np.linspace(0, 1.15, 300)
    positive = 1.0 - np.exp(-5 * x)
    balanced = 1.12 - np.exp(-4 * x)
    runaway = np.where(x < 0.55, 0.95 * (1 - np.exp(-5 * x)), 0.88 - 1.2 * (x - 0.55) ** 2)
    ax.plot(x, positive, lw=2.2, label="Positive-only")
    ax.plot(x, balanced, lw=2.2, label="Controlled repulsion")
    ax.plot(x, runaway, lw=2.2, label="Excessive repulsion")
    ax.axvline(0.55, ls="--", lw=1)
    ax.text(0.58, 0.12, "far-field\nonset", fontsize=9)
    ax.set_xlabel("Training progress / policy movement")
    ax.set_ylabel("Task utility (schematic)")
    ax.set_title("Useful-to-destructive transition")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    d = np.linspace(0, 5, 300)
    raw = 0.5 + d + 0.35 * d**2
    weighted = raw * np.exp(-0.9 * d)
    ax.plot(d, raw, lw=2.4, label="Raw negative influence")
    ax.plot(d, weighted, lw=2.4, label="DRPO-weighted influence")
    ax.fill_between(d, weighted, raw, alpha=0.15)
    ax.set_xlabel("Learner-relative distance / rarity")
    ax.set_ylabel("Influence magnitude (schematic)")
    ax.set_title("Selective far-field control")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[2]
    ax.axis("off")
    boxes = [
        (0.03, 0.60, 0.25, 0.22, "External\noccurrence"),
        (0.38, 0.60, 0.25, 0.22, "Matched source\n+ causal control"),
        (0.72, 0.60, 0.25, 0.22, "External task\nclosure"),
        (0.38, 0.15, 0.25, 0.22, "Theory phase\n+ DRPO"),
    ]
    for x0, y0, w, h, text in boxes:
        patch = FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0.02", fill=False, lw=1.5)
        ax.add_patch(patch)
        ax.text(x0 + w / 2, y0 + h / 2, text, ha="center", va="center", fontsize=9)
    arrows = [
        ((0.28, 0.71), (0.38, 0.71)),
        ((0.63, 0.71), (0.72, 0.71)),
        ((0.505, 0.60), (0.505, 0.37)),
        ((0.63, 0.26), (0.82, 0.60)),
    ]
    for p0, p1 in arrows:
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="->", mutation_scale=12, lw=1.3))
    ax.text(0.5, 0.93, "Evidence architecture", ha="center", va="center", fontsize=11, fontweight="bold")

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def phase_figure(output: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    x = np.linspace(0, 1, 400)
    eq = 0.25 + 0.75 * x / np.maximum(1 - x, 0.08)
    eq = np.minimum(eq, 3.2)
    ax.plot(x, eq, lw=2.5)
    ax.axvspan(0, 0.35, alpha=0.10, label="Positive-only / weak repulsion")
    ax.axvspan(0.35, 0.68, alpha=0.10, label="Stable extrapolation")
    ax.axvspan(0.68, 0.86, alpha=0.10, label="Boundary approach")
    ax.axvspan(0.86, 1.0, alpha=0.10, label="No finite equilibrium")
    ax.set_xlabel("Aggregate negative contribution")
    ax.set_ylabel("Equilibrium displacement (schematic)")
    ax.set_ylim(0, 3.4)
    ax.set_title("Theorem 1 phase map")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper left")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("paper/overleaf/figures/generated"))
    args = parser.parse_args()
    story_figure(args.output_dir / "fig1_story.pdf")
    phase_figure(args.output_dir / "fig2_phase_map.pdf")


if __name__ == "__main__":
    main()
