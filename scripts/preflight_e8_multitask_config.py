"""Fail-fast, non-scientific preflight for an E8 multitask sweep config."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from drpo import e8_experiment_config as experiment_config
from drpo import e8_multitask_exp_tuning as tuning
from drpo.e8_multitask_tasks import stable_hash


def build_summary(config_path: Path, repo_root: Path) -> dict[str, object]:
    resolved, relative, blob = experiment_config.require_tracked_config(config_path, repo_root)
    value = experiment_config.load_strict_yaml(resolved)
    experiment_config.validate_profile_experiment_id(value)
    experiment_config.validate_historical_config_identity(resolved, value, repo_root=repo_root)
    tuning.validate_config(value)
    cells = tuning.build_cells(value)
    waves = tuning.build_waves(value)
    counts = Counter(cell.task for cell in cells)
    active_tasks = [task for task in value["suite"]["tasks"] if counts[task]]
    effective_runtime = (
        {task: experiment_config.effective_coldstart_runtime(value, task) for task in active_tasks}
        if experiment_config.sweep_profile(value) == experiment_config.SWEEP_PROFILE_COLDSTART
        else {}
    )
    return {
        "schema_version": 1,
        "scientific_status": "not_run",
        "experiment_id": experiment_config.experiment_id(value),
        "profile": experiment_config.sweep_profile(value),
        "config_path": relative,
        "config_git_blob_sha": blob,
        "config_hash": stable_hash(value),
        "active_tasks": active_tasks,
        "cells_per_task": {task: counts[task] for task in value["suite"]["tasks"]},
        "effective_runtime": effective_runtime,
        "cell_count": len(cells),
        "wave_count": len(waves),
        "wave_sizes": [len(wave) for wave in waves],
        "model": value.get("model", {}),
        "initialization": value.get("initialization", {}),
        "split": value.get("split", {}),
        "training": value.get("training", {}),
        "evaluation": value.get("evaluation", {}),
        "task_runtime": value.get("task_runtime", {}),
        "note": "Resolved effective runtime only; no model, optimizer, GPU, or scientific metric executed.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    args = parser.parse_args()
    summary = build_summary(Path(args.config), Path(args.repo_root))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
