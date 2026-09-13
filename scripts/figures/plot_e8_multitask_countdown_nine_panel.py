#!/usr/bin/env python3
"""Render the E8 multitask + Countdown surprisal-gradient 3x3 diagnostic.

The checked-in E8 plot-ready CSV contains per-task 95% bootstrap intervals
recomputed from the frozen P0 raw diagnostics. The recomputation extends the
original aggregate bootstrap without changing its task bins, normalization,
prompt resampling unit, replicate count, or seed. If --raw-e8-root is supplied,
this script reruns that derivation and requires exact reproduction of both the
saved per-task means and the saved task-equal aggregate mean/CI before plotting.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np

TASKS = [
    "word_sorting",
    "spiral_matrix",
    "mini_sudoku",
    "maze",
    "word_ladder",
    "knights_knaves",
    "graph_color",
    "wikisql",
]
PANEL_ORDER = [
    "word_sorting",
    "spiral_matrix",
    "wikisql",
    "maze",
    "word_ladder",
    "knights_knaves",
    "mini_sudoku",
    "graph_color",
    "countdown",
]
DISPLAY = {
    "word_sorting": "word sorting",
    "spiral_matrix": "spiral matrix",
    "wikisql": "WikiSQL",
    "maze": "maze",
    "word_ladder": "word ladder",
    "knights_knaves": "knights & knaves",
    "mini_sudoku": "mini sudoku",
    "graph_color": "graph coloring",
    "countdown": "Countdown",
}
BINS = 10
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 161803


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def percentile_interval(values: Iterable[float]) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=float)
    low, high = np.percentile(array, [2.5, 97.5])
    return float(low), float(high)


def assign_task_bins(
    points: list[dict[str, Any]],
) -> tuple[dict[tuple[str, str], int], list[float]]:
    surprisals = np.asarray([float(point["mean_token_surprisal"]) for point in points])
    order = np.argsort(surprisals, kind="stable")
    split_indices = np.array_split(order, BINS)
    assignments: dict[tuple[str, str], int] = {}
    gradients: list[float] = []
    for bin_index, indices in enumerate(split_indices):
        selected = [points[int(index)] for index in indices]
        gradient = float(
            np.mean([float(point["implemented_actor_gradient_norm"]) for point in selected])
        )
        gradients.append(gradient)
        for point in selected:
            assignments[(str(point["prompt_id"]), str(point["negative_id"]))] = bin_index
    if gradients[0] <= 0:
        raise RuntimeError("Non-positive lowest-surprisal normalization anchor")
    relative = [value / gradients[0] for value in gradients]
    return assignments, relative


def recompute_e8_per_task_ci(raw_root: Path) -> dict[str, list[tuple[float, float, float]]]:
    diagnostics = raw_root / "diagnostics"
    aggregate = raw_root / "aggregate"
    points_by_task = {task: read_jsonl(diagnostics / f"{task}.jsonl") for task in TASKS}

    assignments: dict[str, dict[tuple[str, str], int]] = {}
    exact_relative: dict[str, list[float]] = {}
    for task in TASKS:
        assignments[task], exact_relative[task] = assign_task_bins(points_by_task[task])

    saved_per_task = read_csv(aggregate / "per_task_bins.csv")
    saved_means: dict[str, dict[int, float]] = defaultdict(dict)
    for row in saved_per_task:
        saved_means[row["task"]][int(row["bin_index"])] = float(
            row["relative_implemented_actor_gradient"]
        )
    max_mean_error = max(
        abs(exact_relative[task][bin_index] - saved_means[task][bin_index])
        for task in TASKS
        for bin_index in range(BINS)
    )
    if max_mean_error > 1.0e-12:
        raise RuntimeError(f"Per-task mean reproduction failed: {max_mean_error:.3e}")

    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for task, points in points_by_task.items():
        prompt_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for point in points:
            prompt_groups[str(point["prompt_id"])].append(point)
        grouped[task] = prompt_groups

    rng = random.Random(BOOTSTRAP_SEED)
    task_samples = {(task, bin_index): [] for task in TASKS for bin_index in range(BINS)}
    aggregate_samples = {bin_index: [] for bin_index in range(BINS)}
    valid_replicates = 0

    for _ in range(BOOTSTRAP_REPLICATES):
        task_curves: dict[str, list[float]] = {}
        valid = True
        for task in TASKS:
            prompt_groups = grouped[task]
            prompt_ids = sorted(prompt_groups)
            selected: list[dict[str, Any]] = []
            for _prompt_id in prompt_ids:
                chosen = rng.choice(prompt_ids)
                selected.extend(prompt_groups[chosen])

            bin_values: list[float] = []
            for bin_index in range(BINS):
                values = [
                    float(point["implemented_actor_gradient_norm"])
                    for point in selected
                    if assignments[task][
                        (str(point["prompt_id"]), str(point["negative_id"]))
                    ]
                    == bin_index
                ]
                if not values:
                    valid = False
                    break
                bin_values.append(float(np.mean(values)))
            if not valid or bin_values[0] <= 0:
                valid = False
                break
            task_curves[task] = [value / bin_values[0] for value in bin_values]

        if not valid:
            continue
        valid_replicates += 1
        for task in TASKS:
            for bin_index in range(BINS):
                task_samples[(task, bin_index)].append(task_curves[task][bin_index])
        for bin_index in range(BINS):
            aggregate_samples[bin_index].append(
                float(np.mean([task_curves[task][bin_index] for task in sorted(task_curves)]))
            )

    if valid_replicates != BOOTSTRAP_REPLICATES:
        raise RuntimeError(
            f"Expected {BOOTSTRAP_REPLICATES} valid bootstrap replicates, got {valid_replicates}"
        )

    saved_aggregate = {
        int(row["bin_index"]): row for row in read_csv(aggregate / "task_equal_aggregate.csv")
    }
    aggregate_errors: list[float] = []
    for bin_index in range(BINS):
        direct_mean = float(np.mean([exact_relative[task][bin_index] for task in TASKS]))
        saved_mean = float(
            saved_aggregate[bin_index]["relative_implemented_actor_gradient"]
        )
        aggregate_errors.append(abs(direct_mean - saved_mean))
        low, high = percentile_interval(aggregate_samples[bin_index])
        aggregate_errors.extend(
            [
                abs(
                    low
                    - float(
                        saved_aggregate[bin_index][
                            "relative_implemented_actor_gradient_ci95_low"
                        ]
                    )
                ),
                abs(
                    high
                    - float(
                        saved_aggregate[bin_index][
                            "relative_implemented_actor_gradient_ci95_high"
                        ]
                    )
                ),
            ]
        )
    max_aggregate_error = max(aggregate_errors)
    if max_aggregate_error > 1.0e-12:
        raise RuntimeError(f"Aggregate reproduction failed: {max_aggregate_error:.3e}")

    result: dict[str, list[tuple[float, float, float]]] = {}
    for task in TASKS:
        result[task] = []
        for bin_index in range(BINS):
            low, high = percentile_interval(task_samples[(task, bin_index)])
            result[task].append((exact_relative[task][bin_index], low, high))
    return result


def load_e8_plot_ready(path: Path) -> dict[str, list[tuple[float, float, float]]]:
    grouped: dict[str, list[tuple[int, float, float, float]]] = defaultdict(list)
    rows = read_csv(path)
    for row in rows:
        task = row["task"]
        grouped[task].append(
            (
                int(row["bin_index"]),
                float(row["relative_implemented_actor_gradient"]),
                float(row["ci95_low"]),
                float(row["ci95_high"]),
            )
        )
    result: dict[str, list[tuple[float, float, float]]] = {}
    for task in TASKS:
        values = sorted(grouped[task])
        if [entry[0] for entry in values] != list(range(BINS)):
            raise RuntimeError(f"Unexpected bin coverage in {path}: {task}")
        result[task] = [(mean, low, high) for _, mean, low, high in values]
    return result


def compare_e8_tables(
    observed: dict[str, list[tuple[float, float, float]]],
    expected: dict[str, list[tuple[float, float, float]]],
) -> None:
    error = max(
        abs(a - b)
        for task in TASKS
        for observed_row, expected_row in zip(observed[task], expected[task], strict=True)
        for a, b in zip(observed_row, expected_row, strict=True)
    )
    if error > 1.0e-12:
        raise RuntimeError(
            "Checked-in E8 plot data differs from frozen-bootstrap recomputation: "
            f"{error:.3e}"
        )


def load_countdown(path: Path) -> list[tuple[float, float, float]]:
    rows = read_csv(path)
    rows.sort(key=lambda row: int(row["surprisal_decile"]))
    if [int(row["surprisal_decile"]) for row in rows] != list(range(1, 11)):
        raise RuntimeError("Countdown CSV must contain surprisal deciles 1..10")
    return [
        (
            float(row["relative_gradient_mean"]),
            float(row["relative_gradient_ci_low"]),
            float(row["relative_gradient_ci_high"]),
        )
        for row in rows
    ]


def adaptive_ylim(rows: list[tuple[float, float, float]]) -> tuple[float, float]:
    lows = [row[1] for row in rows]
    highs = [row[2] for row in rows]
    low = min(lows + [1.0])
    high = max(highs + [1.0])
    span = max(high - low, 0.08)
    margin = 0.08 * span
    lower = max(0.0, low - margin)
    upper = high + margin
    return lower, upper


def render(
    e8: dict[str, list[tuple[float, float, float]]],
    countdown: list[tuple[float, float, float]],
    output_png: Path,
    output_pdf: Path,
) -> None:
    all_panels = dict(e8)
    all_panels["countdown"] = countdown
    x = np.arange(1, 11)

    figure, axes = plt.subplots(3, 3, figsize=(15.2, 12.2), constrained_layout=True)
    for axis, task in zip(axes.flat, PANEL_ORDER, strict=True):
        rows = all_panels[task]
        mean = np.asarray([row[0] for row in rows])
        low = np.asarray([row[1] for row in rows])
        high = np.asarray([row[2] for row in rows])
        axis.fill_between(x, low, high, alpha=0.18, linewidth=0)
        axis.plot(x, mean, marker="o", linewidth=2.0, markersize=4.5)
        axis.axhline(1.0, linestyle="--", linewidth=1.0, alpha=0.75)
        axis.set_xlim(1, 10)
        axis.set_xticks([1, 3, 5, 7, 10])
        axis.set_xticklabels(["1\n(near)", "3", "5", "7", "10\n(far)"])
        axis.set_ylim(*adaptive_ylim(rows))
        axis.grid(True, alpha=0.25, linewidth=0.6)
        axis.set_title(f"{DISPLAY[task]}\nnear->far = {mean[-1] / mean[0]:.2f}x", fontsize=12)
        axis.set_xlabel("Relative surprisal bin")
        axis.set_ylabel("Relative implemented actor-gradient")
        y0, y1 = axis.get_ylim()
        offset = 0.02 * (y1 - y0)
        axis.text(1.05, mean[0] + offset, f"{mean[0]:.2f}", fontsize=8)
        axis.text(9.65, mean[-1] + offset, f"{mean[-1]:.2f}", fontsize=8, ha="right")

    figure.suptitle("Nine-task surprisal-gradient diagnostic (3x3)", fontsize=16)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_png, dpi=240, bbox_inches="tight")
    figure.savefig(output_pdf, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--e8-ci-csv",
        type=Path,
        default=repo_root
        / "paper/iclr2027/figures/data/e8_multitask_per_task_gradient_ci95.csv",
    )
    parser.add_argument(
        "--countdown-csv",
        type=Path,
        default=repo_root
        / "results/FIGURE1_EXTERNAL_GRADIENT/countdown_gradient_deciles_seed100.csv",
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=repo_root
        / "paper/iclr2027/figures/fig_e8_multitask_countdown_gradient_nine_panel.png",
    )
    parser.add_argument(
        "--output-pdf",
        type=Path,
        default=repo_root
        / "paper/iclr2027/figures/fig_e8_multitask_countdown_gradient_nine_panel.pdf",
    )
    parser.add_argument(
        "--raw-e8-root",
        type=Path,
        help=(
            "Optional path to the extracted E8 P0 run root containing diagnostics/ and "
            "aggregate/. When supplied, recompute the frozen 2000-replicate prompt bootstrap "
            "and require exact agreement with the checked-in plot-ready CSV."
        ),
    )
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    e8 = load_e8_plot_ready(args.e8_ci_csv)
    if args.raw_e8_root is not None:
        recomputed = recompute_e8_per_task_ci(args.raw_e8_root)
        compare_e8_tables(recomputed, e8)
    countdown = load_countdown(args.countdown_csv)
    if args.verify_only:
        return
    render(e8, countdown, args.output_png, args.output_pdf)


if __name__ == "__main__":
    main()
