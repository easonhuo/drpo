"""Render the E8 structured-generation response figures.\n\nThe default mode reproduces the nine-task EXP/DRPO coefficient-response Figure 6.\nPass `--reciprocal` to render the matched-strength Reciprocal Linear vs\nReciprocal Quadratic 400-cell follow-up with the same paper-working layout.

The default input is a checked-in plot-ready summary derived from the canonical
historical E8 response-data locator plus the approved 22-point Countdown curve.
This script performs visualization only: it does not change experiment status,
select coefficients, or upgrade pilot response-shape evidence.

Smoothing is intentionally simple and uniform across all nine tasks:

1. interpolate the observed mean/std series onto a uniform log10(lambda) grid;
2. apply one Gaussian smoothing kernel with the same sigma to every task;
3. use the same smoothing rule for the uncertainty band.

The defaults reproduce the approved presentation style used for the 9-task
response plot. The PDF default is the exact manuscript asset referenced by the
current ICLR source; PNG/SVG defaults remain paper-working previews. Figure-level
title/subtitle text is intentionally omitted because the manuscript caption
already provides that context.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["svg.fonttype"] = "none"

TASK_ORDER = [
    "countdown",
    "word_sorting",
    "spiral_matrix",
    "mini_sudoku",
    "maze",
    "word_ladder",
    "knights_knaves",
    "graph_color",
    "wikisql",
]
DISPLAY = {
    "countdown": "Countdown",
    "word_sorting": "Word Sorting",
    "spiral_matrix": "Spiral Matrix",
    "mini_sudoku": "Mini Sudoku",
    "maze": "Maze",
    "word_ladder": "Word Ladder",
    "knights_knaves": "Knights & Knaves",
    "graph_color": "Graph Coloring",
    "wikisql": "WikiSQL",
}

DENSE_POINTS = 320
MEAN_SIGMA = 12.0
BAND_SIGMA = 12.0
BAND_SCALE = 1.45
MIN_BAND = 0.30
COUNTDOWN_EXPECTED_POINTS = 22

REQUIRED_COLUMNS = {
    "task",
    "lambda",
    "mean_pass8_pct",
    "std_pass8_pct",
    "positive_only_pct",
    "point_count",
    "source",
}


def gaussian_kernel(sigma: float) -> np.ndarray:
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError(f"sigma must be positive and finite, got {sigma!r}")
    radius = max(1, int(np.ceil(4.0 * sigma)))
    grid = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (grid / sigma) ** 2)
    return kernel / kernel.sum()


def gaussian_smooth(values: np.ndarray, sigma: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    kernel = gaussian_kernel(sigma)
    radius = len(kernel) // 2
    padded = np.pad(values, radius, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def smooth_log_series(
    x: np.ndarray,
    y: np.ndarray,
    *,
    sigma: float,
    dense_points: int = DENSE_POINTS,
) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(x)
    x = np.asarray(x, dtype=float)[order]
    y = np.asarray(y, dtype=float)[order]
    log_x = np.log10(x)
    log_dense = np.linspace(log_x.min(), log_x.max(), dense_points)
    dense = np.interp(log_dense, log_x, y)
    return 10.0**log_dense, gaussian_smooth(dense, sigma)


def countdown_local_band(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if len(y) < 2:
        return np.full_like(y, MIN_BAND)
    diffs = np.zeros_like(y)
    diffs[0] = abs(y[1] - y[0])
    diffs[-1] = abs(y[-1] - y[-2])
    if len(y) > 2:
        diffs[1:-1] = 0.5 * (
            np.abs(y[1:-1] - y[:-2]) + np.abs(y[2:] - y[1:-1])
        )
    return np.maximum(diffs * 0.60, MIN_BAND)


def load_plot_ready(path: Path) -> dict[str, pd.DataFrame]:
    if path.is_dir():
        csv_paths = sorted(path.glob("*.csv"))
        if not csv_paths:
            raise RuntimeError(f"No CSV files found in {path}")
        frame = pd.concat([pd.read_csv(csv_path) for csv_path in csv_paths], ignore_index=True)
    else:
        frame = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise RuntimeError(f"Missing columns in {path}: {sorted(missing)}")

    if set(frame["task"]) != set(TASK_ORDER):
        raise RuntimeError(
            "Plot-ready task set mismatch: "
            f"expected {TASK_ORDER}, got {sorted(set(frame['task']))}"
        )

    result: dict[str, pd.DataFrame] = {}
    for task in TASK_ORDER:
        task_frame = frame[frame["task"] == task].copy().sort_values("lambda")
        numeric = task_frame[["lambda", "mean_pass8_pct", "positive_only_pct"]].to_numpy(
            dtype=float
        )
        if not np.all(np.isfinite(numeric)):
            raise RuntimeError(f"Non-finite required numeric value for {task}")
        if np.any(task_frame["lambda"].to_numpy(dtype=float) <= 0):
            raise RuntimeError(f"Non-positive lambda for {task}")
        if task_frame["lambda"].duplicated().any():
            raise RuntimeError(f"Duplicate lambda rows for {task}")
        baselines = task_frame["positive_only_pct"].to_numpy(dtype=float)
        if np.max(np.abs(baselines - baselines[0])) > 1.0e-10:
            raise RuntimeError(f"Positive-only baseline is not constant for {task}")
        if task == "countdown" and len(task_frame) != COUNTDOWN_EXPECTED_POINTS:
            raise RuntimeError(
                f"Countdown must contain {COUNTDOWN_EXPECTED_POINTS} approved points; "
                f"got {len(task_frame)}"
            )
        result[task] = task_frame
    return result


def render(
    data: dict[str, pd.DataFrame],
    *,
    output_png: Path,
    output_pdf: Path,
    output_svg: Path | None,
) -> None:
    figure, axes = plt.subplots(3, 3, figsize=(15, 10.4))
    axes = axes.flatten()
    legend_handles = None
    legend_labels = None

    for axis, task in zip(axes, TASK_ORDER, strict=True):
        frame = data[task]
        x = frame["lambda"].to_numpy(dtype=float)
        y = frame["mean_pass8_pct"].to_numpy(dtype=float)
        baseline = float(frame["positive_only_pct"].iloc[0])

        if task == "countdown":
            raw_band = countdown_local_band(y)
        else:
            raw_band = frame["std_pass8_pct"].to_numpy(dtype=float)
            raw_band = np.nan_to_num(raw_band, nan=0.0, posinf=0.0, neginf=0.0)
            raw_band = np.maximum(raw_band, MIN_BAND)

        dense_x, smooth_mean = smooth_log_series(x, y, sigma=MEAN_SIGMA)
        _, smooth_band = smooth_log_series(x, raw_band, sigma=BAND_SIGMA)
        smooth_band *= BAND_SCALE

        line = axis.plot(
            dense_x,
            smooth_mean,
            linewidth=2.0,
            label="Smoothed EXP mean",
        )[0]
        band = axis.fill_between(
            dense_x,
            smooth_mean - smooth_band,
            smooth_mean + smooth_band,
            alpha=0.18,
            label="EXP local mean ± std",
        )
        baseline_line = axis.axhline(
            baseline,
            linestyle="--",
            linewidth=1.5,
            label="Positive-only baseline",
        )

        axis.set_xscale("log")
        axis.set_xlim(x.min() / 1.18, x.max() * 1.18)
        ymin = min(float(np.min(smooth_mean - smooth_band)), baseline)
        ymax = max(float(np.max(smooth_mean + smooth_band)), baseline)
        padding = max((ymax - ymin) * 0.08, 0.8)
        axis.set_ylim(ymin - padding, ymax + padding)
        axis.set_title(DISPLAY[task], fontsize=14)
        axis.set_xlabel(r"$\lambda$", fontsize=11)
        axis.set_ylabel("Late-window Pass@8 (%)", fontsize=10)
        axis.grid(True, alpha=0.35)
        axis.tick_params(axis="both", labelsize=9)

        if legend_handles is None:
            legend_handles = [line, band, baseline_line]
            legend_labels = [
                "Smoothed EXP mean",
                "EXP local mean ± std",
                "Positive-only baseline",
            ]

    figure.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        fontsize=10,
        bbox_to_anchor=(0.5, 0.985),
    )
    figure.subplots_adjust(
        top=0.91,
        left=0.06,
        right=0.99,
        bottom=0.065,
        wspace=0.23,
        hspace=0.34,
    )

    output_png.parent.mkdir(parents=True, exist_ok=True)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_png, dpi=220)
    figure.savefig(output_pdf)
    if output_svg is not None:
        output_svg.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_svg)
    plt.close(figure)



RECIPROCAL_TASK_ORDER = [
    "word_sorting",
    "spiral_matrix",
    "mini_sudoku",
    "maze",
    "word_ladder",
    "knights_knaves",
    "graph_color",
    "wikisql",
]
RECIPROCAL_DISPLAY = {
    "word_sorting": "Word Sorting",
    "spiral_matrix": "Spiral Matrix",
    "mini_sudoku": "Mini Sudoku",
    "maze": "Maze",
    "word_ladder": "Word Ladder",
    "knights_knaves": "Knights & Knaves",
    "graph_color": "Graph Coloring",
    "wikisql": "WikiSQL",
}
RECIPROCAL_METHOD_ORDER = ["reciprocal_linear", "reciprocal_quadratic"]
RECIPROCAL_METHOD_DISPLAY = {
    "reciprocal_linear": "Reciprocal Linear",
    "reciprocal_quadratic": "Reciprocal Quadratic",
}
RECIPROCAL_REQUIRED_COLUMNS = {
    "task",
    "method",
    "lambda",
    "seed",
    "late_window_pass8_mean",
    "late_window_greedy_mean",
}


def load_reciprocal_plot_ready(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = RECIPROCAL_REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise RuntimeError(f"Missing reciprocal columns in {path}: {sorted(missing)}")
    if len(frame) != 400:
        raise RuntimeError(f"Reciprocal compact table must contain 400 rows; got {len(frame)}")
    keys = frame[["task", "method", "lambda", "seed"]]
    if keys.drop_duplicates().shape[0] != 400:
        raise RuntimeError("Duplicate reciprocal (task, method, lambda, seed) rows")
    if set(frame["seed"].astype(int)) != {4000, 5000}:
        raise RuntimeError(
            f"Unexpected reciprocal seeds: {sorted(set(frame['seed'].astype(int)))}"
        )
    if set(frame["task"]) != set(RECIPROCAL_TASK_ORDER):
        raise RuntimeError(
            "Reciprocal task set mismatch: "
            f"expected {RECIPROCAL_TASK_ORDER}, got {sorted(set(frame['task']))}"
        )
    if set(frame["method"]) != set(RECIPROCAL_METHOD_ORDER):
        raise RuntimeError(
            "Reciprocal method set mismatch: "
            f"expected {RECIPROCAL_METHOD_ORDER}, got {sorted(set(frame['method']))}"
        )

    numeric = frame[
        ["lambda", "seed", "late_window_pass8_mean", "late_window_greedy_mean"]
    ].to_numpy(dtype=float)
    if not np.all(np.isfinite(numeric)):
        raise RuntimeError("Non-finite reciprocal curve value")
    if np.any(frame["lambda"].to_numpy(dtype=float) <= 0):
        raise RuntimeError("Reciprocal lambda must stay positive")

    result = frame.copy()
    result["matched_strength"] = np.where(
        result["method"].eq("reciprocal_linear"),
        result["lambda"].to_numpy(dtype=float),
        np.sqrt(result["lambda"].to_numpy(dtype=float)),
    )
    summary = (
        result.groupby(["task", "method", "matched_strength"], as_index=False)
        .agg(
            mean_pass8=("late_window_pass8_mean", "mean"),
            std_pass8=("late_window_pass8_mean", "std"),
            seed_count=("seed", "nunique"),
        )
        .sort_values(["task", "method", "matched_strength"])
    )
    if not np.all(summary["seed_count"].to_numpy(dtype=int) == 2):
        raise RuntimeError("Every reciprocal curve point must contain both registered seeds")
    if len(summary) != 200:
        raise RuntimeError(f"Expected 200 two-seed reciprocal means; got {len(summary)}")
    return summary


def render_reciprocal(
    summary: pd.DataFrame,
    *,
    output_png: Path,
    output_pdf: Path,
    output_svg: Path | None,
) -> None:
    figure, axes = plt.subplots(3, 3, figsize=(15, 10.4))
    axes = axes.flatten()
    legend_handles = None
    legend_labels = None

    for axis, task in zip(axes[:8], RECIPROCAL_TASK_ORDER, strict=True):
        task_frame = summary[summary["task"] == task]
        task_handles = []
        for method in RECIPROCAL_METHOD_ORDER:
            curve = task_frame[task_frame["method"] == method].sort_values(
                "matched_strength"
            )
            x = curve["matched_strength"].to_numpy(dtype=float)
            y = 100.0 * curve["mean_pass8"].to_numpy(dtype=float)
            band = 100.0 * curve["std_pass8"].fillna(0.0).to_numpy(dtype=float)
            line = axis.plot(
                x,
                y,
                marker="o",
                markersize=3.0,
                linewidth=1.6,
                label=RECIPROCAL_METHOD_DISPLAY[method],
            )[0]
            axis.fill_between(
                x,
                y - band,
                y + band,
                alpha=0.12,
                color=line.get_color(),
            )
            task_handles.append(line)

        if legend_handles is None:
            legend_handles = task_handles
            legend_labels = [
                RECIPROCAL_METHOD_DISPLAY[method]
                for method in RECIPROCAL_METHOD_ORDER
            ]

        axis.set_xscale("log")
        x_all = task_frame["matched_strength"].to_numpy(dtype=float)
        axis.set_xlim(float(np.min(x_all)) / 1.18, float(np.max(x_all)) * 1.18)
        means = 100.0 * task_frame["mean_pass8"].to_numpy(dtype=float)
        bands = 100.0 * task_frame["std_pass8"].fillna(0.0).to_numpy(dtype=float)
        ymin = float(np.min(means - bands))
        ymax = float(np.max(means + bands))
        padding = max((ymax - ymin) * 0.08, 0.8)
        axis.set_ylim(ymin - padding, ymax + padding)
        axis.set_title(RECIPROCAL_DISPLAY[task], fontsize=14)
        axis.set_xlabel("Matched taper strength", fontsize=11)
        axis.set_ylabel("Late-window Pass@8 (%)", fontsize=10)
        axis.grid(True, alpha=0.35)
        axis.tick_params(axis="both", labelsize=9)

    note_axis = axes[-1]
    note_axis.axis("off")
    note_axis.text(
        0.03,
        0.92,
        "Countdown was not part of the\nreciprocal 400-cell sweeps.",
        fontsize=13,
        va="top",
    )
    note_axis.text(
        0.03,
        0.63,
        "Matched x-coordinate",
        fontsize=11,
        fontweight="bold",
    )
    note_axis.text(
        0.03,
        0.53,
        "Linear: x = λ\nQuadratic: x = √λ",
        fontsize=11,
        va="top",
    )
    note_axis.text(
        0.03,
        0.28,
        "400 cells = 200 two-seed means\nSeeds: 4000, 5000",
        fontsize=11,
        va="top",
    )

    figure.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        fontsize=10,
        bbox_to_anchor=(0.5, 0.985),
    )
    figure.subplots_adjust(
        top=0.91,
        left=0.06,
        right=0.99,
        bottom=0.065,
        wspace=0.23,
        hspace=0.34,
    )

    output_png.parent.mkdir(parents=True, exist_ok=True)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_png, dpi=220)
    figure.savefig(output_pdf)
    if output_svg is not None:
        output_svg.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_svg)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=repo_root
        / "paper/iclr2027/figures/data/e8_multitask_exp_response",
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=repo_root
        / "paper/iclr2027/figures/fig_e8_multitask_exp_response_nine_panel.png",
    )
    parser.add_argument(
        "--output-pdf",
        type=Path,
        default=repo_root
        / "paper/overleaf/figures/fig_app_structured9_drpo_coefficient_response.pdf",
    )
    parser.add_argument(
        "--output-svg",
        type=Path,
        default=repo_root
        / "paper/iclr2027/figures/fig_e8_multitask_exp_response_nine_panel.svg",
    )
    parser.add_argument(
        "--reciprocal",
        action="store_true",
        help="Render the registered reciprocal 400-cell follow-up instead of Figure 6.",
    )
    parser.add_argument(
        "--reciprocal-data",
        type=Path,
        default=repo_root
        / "paper/iclr2027/figures/data/e8_reciprocal_response_20260921/RECIPROCAL_400_CURVE_POINTS.csv",
    )
    parser.add_argument(
        "--reciprocal-output-png",
        type=Path,
        default=repo_root
        / "paper/iclr2027/figures/fig_e8_reciprocal_400_response_eight_panel.png",
    )
    parser.add_argument(
        "--reciprocal-output-pdf",
        type=Path,
        default=repo_root
        / "paper/overleaf/figures/fig_app_structured8_reciprocal_400_response.pdf",
    )
    parser.add_argument(
        "--reciprocal-output-svg",
        type=Path,
        default=repo_root
        / "paper/iclr2027/figures/fig_e8_reciprocal_400_response_eight_panel.svg",
    )
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.reciprocal:
        reciprocal = load_reciprocal_plot_ready(args.reciprocal_data)
        if args.verify_only:
            return
        render_reciprocal(
            reciprocal,
            output_png=args.reciprocal_output_png,
            output_pdf=args.reciprocal_output_pdf,
            output_svg=args.reciprocal_output_svg,
        )
        return

    data = load_plot_ready(args.data)
    if args.verify_only:
        return
    render(
        data,
        output_png=args.output_png,
        output_pdf=args.output_pdf,
        output_svg=args.output_svg,
    )


if __name__ == "__main__":
    main()
