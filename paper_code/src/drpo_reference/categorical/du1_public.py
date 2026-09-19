"""Public runner for the D-U1 six-method experiment."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from drpo_reference.common import atomic_json

from .du1_protocol import METHODS, DU1Protocol, method_specs
from .du1_training import build_shared_start, run_method


def run_du1(
    *,
    output_root: Path,
    seeds: Sequence[int] | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    """Run every D-U1 method from the same per-seed initialization."""

    protocol = DU1Protocol()
    selected_seeds = protocol.seeds if seeds is None else tuple(int(seed) for seed in seeds)
    target = torch.device(
        "cuda"
        if device == "auto" and torch.cuda.is_available()
        else "cpu"
        if device == "auto"
        else device
    )
    specs = method_specs()
    trajectories: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []

    for seed in selected_seeds:
        base_state, calibration = build_shared_start(protocol, seed, target)
        for spec in specs:
            method_trajectory, summary = run_method(
                protocol=protocol,
                seed=seed,
                spec=spec,
                base_state=base_state,
                calibration=calibration,
                device=target,
            )
            trajectories.extend(method_trajectory)
            summaries.append(summary)

    order = {method: index for index, method in enumerate(METHODS)}
    trajectories.sort(
        key=lambda row: (
            int(row["seed"]),
            order[str(row["method"])],
            int(row["step"]),
        )
    )
    summaries.sort(
        key=lambda row: (
            int(row["seed"]),
            order[str(row["method"])],
        )
    )

    positive_only = {
        int(row["seed"]): float(row["final_expected_semantic_reward"])
        for row in summaries
        if row["method"] == "positive_only"
    }
    for row in summaries:
        reference = positive_only[int(row["seed"])]
        row["paired_positive_only_reward"] = reference
        row["task_performance_collapse"] = bool(
            float(row["final_expected_semantic_reward"])
            < protocol.task_collapse_ratio_to_paired_positive_only * reference
        )

    result = {
        "experiment": "D-U1",
        "terminology": "same-distribution held-out-context generalization",
        "seeds": list(selected_seeds),
        "methods": list(METHODS),
        "summaries": summaries,
        "trajectories": trajectories,
    }
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    atomic_json(root / "results.json", result)
    return result
