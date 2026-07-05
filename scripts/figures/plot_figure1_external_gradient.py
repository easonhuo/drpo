#!/usr/bin/env python3
"""Regenerate Figure 1 Hopper + Countdown exact decile panel.

Usage:
  # Use a plot-ready Countdown decile table already stored in the repository.
  python scripts/figures/plot_figure1_external_gradient.py \
    --hopper results/FIGURE1_EXTERNAL_GRADIENT/hopper_e7_q2_plot_data.csv \
    --countdown results/FIGURE1_EXTERNAL_GRADIENT/countdown_gradient_deciles_seed100.csv \
    --out paper/figures/figure1_external_gradient

  # Or pass the full raw Countdown probe CSV; the script will compute deciles.
  python scripts/figures/plot_figure1_external_gradient.py \
    --hopper results/FIGURE1_EXTERNAL_GRADIENT/hopper_e7_q2_plot_data.csv \
    --countdown /path/to/countdown_gradient_samples_seed100_full.csv \
    --out paper/figures/figure1_external_gradient
"""
from __future__ import annotations
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def bool_series(s):
    if s.dtype == bool:
        return s
    return s.astype(str).str.lower().isin(["true", "1", "yes", "y"])

def decile_stats_np(s, g, coeff, bins=10):
    order = np.argsort(s)
    s2 = s[order]
    g2 = g[order]
    c2 = coeff[order]
    n = len(s2)
    bin_id = np.minimum((np.arange(n) * bins // n), bins - 1)
    out = []
    for b in range(bins):
        m = bin_id == b
        out.append((b+1, s2[m].mean(), g2[m].mean(), c2[m].mean(), int(m.sum())))
    arr = np.array(out, dtype=float)
    rel_g = arr[:,2] / arr[0,2]
    rel_c = arr[:,3] / arr[0,3] if arr[0,3] != 0 else np.full(bins, np.nan)
    return arr, rel_g, rel_c

def filter_countdown(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    mask = np.ones(len(df), dtype=bool)
    if "verifier_category" in df.columns:
        mask &= df["verifier_category"].astype(str).eq("arithmetic_wrong").to_numpy()
    if "valid_format" in df.columns:
        mask &= bool_series(df["valid_format"]).to_numpy()
    if "uses_numbers" in df.columns:
        mask &= bool_series(df["uses_numbers"]).to_numpy()
    if "correct" in df.columns:
        mask &= ~bool_series(df["correct"]).to_numpy()
    for c in ["mean_token_surprisal", "trainable_parameter_gradient_norm", "negative_coefficient_abs"]:
        mask &= np.isfinite(pd.to_numeric(df[c], errors="coerce")).to_numpy()
    out = df.loc[mask].copy()
    for c in ["mean_token_surprisal", "trainable_parameter_gradient_norm", "negative_coefficient_abs"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def countdown_deciles(cd: pd.DataFrame, bootstrap: int = 400, seed: int = 20260705) -> pd.DataFrame:
    s = cd["mean_token_surprisal"].to_numpy(float)
    g = cd["trainable_parameter_gradient_norm"].to_numpy(float)
    coeff = cd["negative_coefficient_abs"].to_numpy(float)
    arr, rel_g, rel_c = decile_stats_np(s, g, coeff)
    rng = np.random.default_rng(seed)
    boot_rel_g = np.empty((bootstrap, 10), dtype=float)
    boot_rel_c = np.empty((bootstrap, 10), dtype=float)

    if "puzzle_id" in cd.columns and cd.groupby("puzzle_id").size().nunique() == 1:
        groups = list(cd.groupby("puzzle_id", sort=False))
        sizes = [len(v) for _, v in groups]
        if len(set(sizes)) == 1:
            sp = np.stack([v["mean_token_surprisal"].to_numpy(float) for _, v in groups], axis=0)
            gp = np.stack([v["trainable_parameter_gradient_norm"].to_numpy(float) for _, v in groups], axis=0)
            cp = np.stack([v["negative_coefficient_abs"].to_numpy(float) for _, v in groups], axis=0)
            P = sp.shape[0]
            for b in range(bootstrap):
                idx = rng.integers(0, P, size=P)
                _, bg, bc = decile_stats_np(sp[idx].reshape(-1), gp[idx].reshape(-1), cp[idx].reshape(-1))
                boot_rel_g[b] = bg
                boot_rel_c[b] = bc
        else:
            for b in range(bootstrap):
                idx = rng.integers(0, len(s), size=len(s))
                _, bg, bc = decile_stats_np(s[idx], g[idx], coeff[idx])
                boot_rel_g[b] = bg
                boot_rel_c[b] = bc
    else:
        for b in range(bootstrap):
            idx = rng.integers(0, len(s), size=len(s))
            _, bg, bc = decile_stats_np(s[idx], g[idx], coeff[idx])
            boot_rel_g[b] = bg
            boot_rel_c[b] = bc

    return pd.DataFrame({
        "surprisal_decile": arr[:,0].astype(int),
        "mean_surprisal": arr[:,1],
        "mean_gradient": arr[:,2],
        "mean_negative_coefficient_abs": arr[:,3],
        "n": arr[:,4].astype(int),
        "relative_gradient_mean": rel_g,
        "relative_gradient_ci_low": np.percentile(boot_rel_g, 2.5, axis=0),
        "relative_gradient_ci_high": np.percentile(boot_rel_g, 97.5, axis=0),
        "relative_coeff_mean": rel_c,
    })

def load_countdown_deciles(path: Path, bootstrap: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    required_decile_cols = {
        "surprisal_decile",
        "relative_gradient_mean",
        "relative_gradient_ci_low",
        "relative_gradient_ci_high",
        "relative_coeff_mean",
    }
    if required_decile_cols.issubset(df.columns):
        # Repository-stored plot-ready path. Do not pretend to recompute from raw.
        return df.copy()
    cd = filter_countdown(path)
    return countdown_deciles(cd, bootstrap=bootstrap)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hopper", type=Path, required=True)
    ap.add_argument("--countdown", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--bootstrap", type=int, default=400)
    args = ap.parse_args()

    hopper = pd.read_csv(args.hopper)
    dec = load_countdown_deciles(args.countdown, bootstrap=args.bootstrap)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    dec.to_csv(args.out.with_name(args.out.name + "_countdown_deciles.csv"), index=False)

    fig, axes = plt.subplots(1, 2, figsize=(7.25, 2.85), gridspec_kw={"width_ratios": [1.22, 1.0]})
    ax = axes[0]
    x = hopper["relative_distance"].to_numpy()
    ax.plot(x, hopper["relative_gradient_mean"], marker="o", linewidth=1.75, markersize=4.2, label="Gradient norm")
    ax.fill_between(x, hopper["relative_gradient_ci_low"], hopper["relative_gradient_ci_high"], alpha=0.18, linewidth=0)
    ax.plot(x, hopper["relative_abs_advantage_mean"], marker="s", linestyle="--", linewidth=1.35, markersize=3.8, label=r"$|A|$")
    ax.fill_between(x, hopper["relative_abs_advantage_ci_low"], hopper["relative_abs_advantage_ci_high"], alpha=0.10, linewidth=0)
    ax.axhline(1.0, linestyle=":", linewidth=1.0)
    ax.set_title("(a) Hopper", fontsize=11.6, pad=3)
    ax.set_xlabel("Relative standardized distance", fontsize=9.7)
    ax.set_ylabel("Relative magnitude", fontsize=9.7)
    ax.tick_params(labelsize=8.4)
    ax.grid(axis="y", linestyle="--", linewidth=0.45, alpha=0.33)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=7.9, frameon=True, loc="upper left")

    ax = axes[1]
    xd = dec["surprisal_decile"].to_numpy()
    ax.plot(xd, dec["relative_gradient_mean"], marker="o", linewidth=1.75, markersize=4.2, label="Gradient norm")
    ax.fill_between(xd, dec["relative_gradient_ci_low"], dec["relative_gradient_ci_high"], alpha=0.18, linewidth=0)
    ax.plot(xd, dec["relative_coeff_mean"], marker="s", linestyle="--", linewidth=1.35, markersize=3.8, label="Neg. coeff.")
    ax.axhline(1.0, linestyle=":", linewidth=1.0)
    ax.set_title("(b) Countdown", fontsize=11.6, pad=3)
    ax.set_xlabel("Mean-token surprisal decile", fontsize=9.7)
    ax.set_ylabel("Relative magnitude", fontsize=9.7)
    ax.set_xticks([1,5,10])
    ax.set_xticklabels(["low","mid","high"], fontsize=8.4)
    ax.tick_params(axis="y", labelsize=8.4)
    ax.grid(axis="y", linestyle="--", linewidth=0.45, alpha=0.33)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=7.9, frameon=True, loc="lower right")

    fig.tight_layout(pad=0.35, w_pad=0.85)
    for ext in ["pdf", "svg", "png"]:
        fig.savefig(str(args.out) + "." + ext, dpi=320 if ext == "png" else None, bbox_inches="tight", pad_inches=0.03)

if __name__ == "__main__":
    main()
