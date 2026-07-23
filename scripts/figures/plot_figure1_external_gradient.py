#!/usr/bin/env python3
"""Regenerate Figure 1 D4RL-9 + Countdown and the appendix D4RL-9 panels.

Usage:
  python scripts/figures/plot_figure1_external_gradient.py \
    --d4rl-main results/FIGURE1_EXTERNAL_GRADIENT_D4RL9/main_aggregate.csv \
    --d4rl-panels results/FIGURE1_EXTERNAL_GRADIENT_D4RL9/dataset_plot_data.csv \
    --d4rl-manifest results/FIGURE1_EXTERNAL_GRADIENT_D4RL9/MANIFEST.json \
    --countdown results/FIGURE1_EXTERNAL_GRADIENT/countdown_gradient_deciles_seed100.csv \
    --out paper/kdd2027/figures/figure1_external_gradient_d4rl9 \
    --appendix-out paper/kdd2027/figures/fig_app_d4rl9_gradient_panels
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def bool_series(series):
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin(["true", "1", "yes", "y"])


def decile_stats_np(surprisal, gradient, coefficient, bins=10):
    order = np.argsort(surprisal)
    s2 = surprisal[order]
    g2 = gradient[order]
    c2 = coefficient[order]
    count = len(s2)
    bin_id = np.minimum((np.arange(count) * bins // count), bins - 1)
    rows = []
    for index in range(bins):
        mask = bin_id == index
        rows.append(
            (
                index + 1,
                s2[mask].mean(),
                g2[mask].mean(),
                c2[mask].mean(),
                int(mask.sum()),
            )
        )
    array = np.array(rows, dtype=float)
    relative_gradient = array[:, 2] / array[0, 2]
    relative_coefficient = (
        array[:, 3] / array[0, 3]
        if array[0, 3] != 0
        else np.full(bins, np.nan)
    )
    return array, relative_gradient, relative_coefficient


def filter_countdown(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    mask = np.ones(len(frame), dtype=bool)
    if "verifier_category" in frame.columns:
        mask &= frame["verifier_category"].astype(str).eq("arithmetic_wrong").to_numpy()
    if "valid_format" in frame.columns:
        mask &= bool_series(frame["valid_format"]).to_numpy()
    if "uses_numbers" in frame.columns:
        mask &= bool_series(frame["uses_numbers"]).to_numpy()
    if "correct" in frame.columns:
        mask &= ~bool_series(frame["correct"]).to_numpy()
    columns = [
        "mean_token_surprisal",
        "trainable_parameter_gradient_norm",
        "negative_coefficient_abs",
    ]
    for column in columns:
        mask &= np.isfinite(pd.to_numeric(frame[column], errors="coerce")).to_numpy()
    output = frame.loc[mask].copy()
    for column in columns:
        output[column] = pd.to_numeric(output[column], errors="coerce")
    return output


def countdown_deciles(
    frame: pd.DataFrame,
    bootstrap: int = 400,
    seed: int = 20260705,
) -> pd.DataFrame:
    surprisal = frame["mean_token_surprisal"].to_numpy(float)
    gradient = frame["trainable_parameter_gradient_norm"].to_numpy(float)
    coefficient = frame["negative_coefficient_abs"].to_numpy(float)
    array, relative_gradient, relative_coefficient = decile_stats_np(
        surprisal,
        gradient,
        coefficient,
    )
    rng = np.random.default_rng(seed)
    boot_gradient = np.empty((bootstrap, 10), dtype=float)
    boot_coefficient = np.empty((bootstrap, 10), dtype=float)
    if "puzzle_id" in frame.columns and frame.groupby("puzzle_id").size().nunique() == 1:
        groups = list(frame.groupby("puzzle_id", sort=False))
        sizes = [len(value) for _, value in groups]
    else:
        groups = []
        sizes = []
    if groups and len(set(sizes)) == 1:
        stacked_s = np.stack(
            [value["mean_token_surprisal"].to_numpy(float) for _, value in groups]
        )
        stacked_g = np.stack(
            [value["trainable_parameter_gradient_norm"].to_numpy(float) for _, value in groups]
        )
        stacked_c = np.stack(
            [value["negative_coefficient_abs"].to_numpy(float) for _, value in groups]
        )
        puzzles = stacked_s.shape[0]
        for draw in range(bootstrap):
            indices = rng.integers(0, puzzles, size=puzzles)
            _, bg, bc = decile_stats_np(
                stacked_s[indices].reshape(-1),
                stacked_g[indices].reshape(-1),
                stacked_c[indices].reshape(-1),
            )
            boot_gradient[draw] = bg
            boot_coefficient[draw] = bc
    else:
        for draw in range(bootstrap):
            indices = rng.integers(0, len(surprisal), size=len(surprisal))
            _, bg, bc = decile_stats_np(
                surprisal[indices],
                gradient[indices],
                coefficient[indices],
            )
            boot_gradient[draw] = bg
            boot_coefficient[draw] = bc
    return pd.DataFrame(
        {
            "surprisal_decile": array[:, 0].astype(int),
            "mean_surprisal": array[:, 1],
            "mean_gradient": array[:, 2],
            "mean_negative_coefficient_abs": array[:, 3],
            "n": array[:, 4].astype(int),
            "relative_gradient_mean": relative_gradient,
            "relative_gradient_ci_low": np.percentile(boot_gradient, 2.5, axis=0),
            "relative_gradient_ci_high": np.percentile(boot_gradient, 97.5, axis=0),
            "relative_coeff_mean": relative_coefficient,
        }
    )


def load_countdown_deciles(path: Path, bootstrap: int) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "surprisal_decile",
        "relative_gradient_mean",
        "relative_gradient_ci_low",
        "relative_gradient_ci_high",
        "relative_coeff_mean",
    }
    if required.issubset(frame.columns):
        return frame.copy()
    return countdown_deciles(filter_countdown(path), bootstrap=bootstrap)


def save_all(figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    for extension in ["pdf", "svg", "png"]:
        figure.savefig(
            str(stem) + "." + extension,
            dpi=320 if extension == "png" else None,
            bbox_inches="tight",
            pad_inches=0.03,
        )


def validate_d4rl(main: pd.DataFrame, panels: pd.DataFrame, manifest: dict) -> None:
    main_columns = {
        "relative_distance",
        "relative_gradient_mean",
        "relative_gradient_ci_low",
        "relative_gradient_ci_high",
        "n_datasets",
        "n_seeds_per_dataset",
    }
    panel_columns = {
        "dataset_id",
        "distance_bin",
        "relative_distance",
        "relative_gradient_mean",
        "relative_gradient_ci_low",
        "relative_gradient_ci_high",
        "relative_abs_advantage_mean",
        "relative_abs_advantage_ci_low",
        "relative_abs_advantage_ci_high",
        "n_seeds",
    }
    if not main_columns.issubset(main.columns):
        raise ValueError("D4RL aggregate schema mismatch")
    if not panel_columns.issubset(panels.columns):
        raise ValueError("D4RL panel schema mismatch")
    if len(main) != 8 or not main["relative_distance"].is_monotonic_increasing:
        raise ValueError("D4RL aggregate-grid mismatch")
    if set(main["n_datasets"].astype(int)) != {9}:
        raise ValueError("D4RL aggregate must cover nine dataset cells")
    if set(main["n_seeds_per_dataset"].astype(int)) != {10}:
        raise ValueError("D4RL aggregate must use ten seeds per dataset")
    if panels["dataset_id"].nunique() != 9:
        raise ValueError("D4RL appendix must cover nine dataset cells")
    if set(panels["distance_bin"].astype(int)) != set(range(-1, 7)):
        raise ValueError("D4RL panel-bin mismatch")
    if manifest.get("scientific_status") != "pilot":
        raise ValueError("unexpected D4RL result status")
    if manifest.get("formal_evidence_allowed") is not False:
        raise ValueError("plotting must not promote the pilot result")


def plot_main(main: pd.DataFrame, countdown: pd.DataFrame, manifest: dict, out: Path) -> None:
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(7.25, 2.85),
        gridspec_kw={"width_ratios": [1.22, 1.0]},
    )
    axis = axes[0]
    x = main["relative_distance"].to_numpy(float)
    axis.fill_between(
        x,
        main["relative_gradient_ci_low"],
        main["relative_gradient_ci_high"],
        alpha=0.18,
        linewidth=0,
    )
    axis.plot(
        x,
        main["relative_gradient_mean"],
        marker="o",
        linewidth=1.75,
        markersize=4.2,
        label="Gradient norm",
    )
    axis.axhline(1.0, linestyle=":", linewidth=1.0)
    advantage_ratio = float(manifest["pairwise_absolute_advantage"]["mean_ratio"])
    axis.text(
        0.035,
        0.955,
        "9 datasets × 10 seeds\n"
        + rf"pair-matched $|A|$ far/near = {advantage_ratio:.3f}",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=7.7,
    )
    axis.set_title("(a) D4RL locomotion aggregate", fontsize=11.1, pad=3)
    axis.set_xlabel("Relative standardized distance", fontsize=9.7)
    axis.set_ylabel("Relative magnitude", fontsize=9.7)
    axis.tick_params(labelsize=8.4)
    axis.grid(axis="y", linestyle="--", linewidth=0.45, alpha=0.33)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(fontsize=7.9, frameon=True, loc="lower right")

    axis = axes[1]
    x_countdown = countdown["surprisal_decile"].to_numpy()
    axis.plot(
        x_countdown,
        countdown["relative_gradient_mean"],
        marker="o",
        linewidth=1.75,
        markersize=4.2,
        label="Gradient norm",
    )
    axis.fill_between(
        x_countdown,
        countdown["relative_gradient_ci_low"],
        countdown["relative_gradient_ci_high"],
        alpha=0.18,
        linewidth=0,
    )
    axis.plot(
        x_countdown,
        countdown["relative_coeff_mean"],
        marker="s",
        linestyle="--",
        linewidth=1.35,
        markersize=3.8,
        label="Neg. coeff.",
    )
    axis.axhline(1.0, linestyle=":", linewidth=1.0)
    axis.set_title("(b) Countdown", fontsize=11.6, pad=3)
    axis.set_xlabel("Mean-token surprisal decile", fontsize=9.7)
    axis.set_ylabel("Relative magnitude", fontsize=9.7)
    axis.set_xticks([1, 5, 10])
    axis.set_xticklabels(["low", "mid", "high"], fontsize=8.4)
    axis.tick_params(axis="y", labelsize=8.4)
    axis.grid(axis="y", linestyle="--", linewidth=0.45, alpha=0.33)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(fontsize=7.9, frameon=True, loc="lower right")
    figure.tight_layout(pad=0.35, w_pad=0.85)
    save_all(figure, out)
    plt.close(figure)


def plot_appendix(panels: pd.DataFrame, out: Path) -> None:
    order = [
        "halfcheetah-medium-v2",
        "halfcheetah-medium-replay-v2",
        "halfcheetah-medium-expert-v2",
        "hopper-medium-v2",
        "hopper-medium-replay-v2",
        "hopper-medium-expert-v2",
        "walker2d-medium-v2",
        "walker2d-medium-replay-v2",
        "walker2d-medium-expert-v2",
    ]
    titles = {
        "halfcheetah-medium-v2": "HalfCheetah — medium",
        "halfcheetah-medium-replay-v2": "HalfCheetah — medium-replay",
        "halfcheetah-medium-expert-v2": "HalfCheetah — medium-expert",
        "hopper-medium-v2": "Hopper — medium",
        "hopper-medium-replay-v2": "Hopper — medium-replay",
        "hopper-medium-expert-v2": "Hopper — medium-expert",
        "walker2d-medium-v2": "Walker2d — medium",
        "walker2d-medium-replay-v2": "Walker2d — medium-replay",
        "walker2d-medium-expert-v2": "Walker2d — medium-expert",
    }
    if set(panels["dataset_id"]) != set(order):
        raise ValueError("unexpected D4RL dataset inventory")
    y_max = math.ceil(float(panels["relative_gradient_ci_high"].max()) + 0.25)
    figure, axes = plt.subplots(3, 3, figsize=(10.5, 8.0))
    for axis, dataset in zip(axes.flat, order):
        data = panels[panels["dataset_id"] == dataset].sort_values("relative_distance")
        x = data["relative_distance"].to_numpy(float)
        axis.fill_between(
            x,
            data["relative_gradient_ci_low"],
            data["relative_gradient_ci_high"],
            alpha=0.18,
            linewidth=0,
        )
        axis.plot(
            x,
            data["relative_gradient_mean"],
            marker="o",
            linewidth=1.75,
            markersize=3.4,
            label="Relative gradient",
        )
        axis.fill_between(
            x,
            data["relative_abs_advantage_ci_low"],
            data["relative_abs_advantage_ci_high"],
            alpha=0.10,
            linewidth=0,
        )
        axis.plot(
            x,
            data["relative_abs_advantage_mean"],
            marker="s",
            linestyle="--",
            linewidth=1.2,
            markersize=2.9,
            label=r"Relative $|A|$",
        )
        axis.axhline(1.0, linestyle=":", linewidth=0.9)
        axis.set_title(titles[dataset], fontsize=9.2, pad=3)
        axis.set_xlim(0.9, float(x.max()) + 0.15)
        axis.set_ylim(0.45, y_max)
        axis.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.3)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(labelsize=7.2)
    for axis in axes[-1, :]:
        axis.set_xlabel("Relative standardized distance", fontsize=8.0)
    for axis in axes[:, 0]:
        axis.set_ylabel("Relative magnitude", fontsize=8.0)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        fontsize=8.0,
        bbox_to_anchor=(0.5, 1.005),
    )
    figure.tight_layout(rect=(0, 0, 1, 0.975), h_pad=1.0, w_pad=0.8)
    save_all(figure, out)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--d4rl-main", type=Path, required=True)
    parser.add_argument("--d4rl-panels", type=Path, required=True)
    parser.add_argument("--d4rl-manifest", type=Path, required=True)
    parser.add_argument("--countdown", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--appendix-out", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=400)
    args = parser.parse_args()

    d4rl_main = pd.read_csv(args.d4rl_main)
    d4rl_panels = pd.read_csv(args.d4rl_panels)
    manifest = json.loads(args.d4rl_manifest.read_text(encoding="utf-8"))
    countdown = load_countdown_deciles(args.countdown, bootstrap=args.bootstrap)
    validate_d4rl(d4rl_main, d4rl_panels, manifest)
    countdown.to_csv(
        args.out.with_name(args.out.name + "_countdown_deciles.csv"),
        index=False,
    )
    plot_main(d4rl_main, countdown, manifest, args.out)
    plot_appendix(d4rl_panels, args.appendix_out)


if __name__ == "__main__":
    main()
