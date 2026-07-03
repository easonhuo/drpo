#!/usr/bin/env python3
"""Run the D-U1 E6 development separation grid with durable per-run outputs.

This runner is intentionally development-only. It excludes Quartic from the
active matrix, preserves historical Quartic code/results, never accesses formal
seeds, and reports task collapse, support boundary, and NaN/Inf separately.

The runner expects protocol revision 4 (dynamic oracle utility + task-visible
hidden rare support) to be present in ``src/drpo/du1_e6_cartesian_taper.py``.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import dataclasses
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

EXPERIMENT_ID = "D-U1-E6-CARTESIAN-TAPER-01"
MIN_PROTOCOL_REVISION = 4
ACTIVE_METHODS = (
    "positive_only",
    "all_negative",
    "global_matched",
    "reciprocal_linear_distance",
    "reciprocal_quadratic_distance",
    "exponential_quadratic_distance",
)
FORBIDDEN_ACTIVE_METHODS = ("reciprocal_quartic_distance",)
FORMAL_SEED_MIN = 200


@dataclasses.dataclass(frozen=True)
class GridTask:
    seed: int
    method: str
    negative_alpha: float
    anchor: float
    retention: float
    steps: int
    eval_every: int
    device: str
    source_module: str
    base_config: str
    output_dir: str

    @property
    def cell_id(self) -> str:
        return (
            f"alpha_{float_token(self.negative_alpha)}"
            f"__anchor_{float_token(self.anchor)}"
            f"__rho_{float_token(self.retention)}"
        )

    @property
    def run_id(self) -> str:
        return f"{self.cell_id}__seed_{self.seed}__{self.method}"


def float_token(value: float) -> str:
    text = f"{float(value):.8g}"
    return text.replace("-", "m").replace(".", "p")


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temp.write_text(text)
    os.replace(temp, path)


def json_dump(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def yaml_dump(path: Path, value: Any) -> None:
    atomic_write_text(path, yaml.safe_dump(value, sort_keys=False))


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    atomic_write_text(
        path,
        "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows),
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_text(repo_root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def _float_list(plan: Mapping[str, Any], key: str) -> list[float]:
    values = plan["grid"][key]
    if not isinstance(values, list) or not values:
        raise ValueError(f"grid.{key} must be a non-empty list")
    out = [float(value) for value in values]
    if any(not math.isfinite(value) for value in out):
        raise ValueError(f"grid.{key} contains non-finite values")
    return out


def validate_plan(plan: Mapping[str, Any]) -> None:
    if plan.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError(f"experiment_id must be {EXPERIMENT_ID}")
    if plan.get("stage") != "development_calibration":
        raise ValueError("stage must be development_calibration")
    if bool(plan.get("formal_result")):
        raise ValueError("development grid cannot be marked formal")
    if bool(plan.get("formal_seed_access_allowed")):
        raise ValueError("formal seed access must remain disabled")

    seeds = [int(seed) for seed in plan.get("seeds", [])]
    if seeds != [0, 1, 2, 3, 4]:
        raise ValueError("development seeds must remain exactly [0,1,2,3,4]")
    if any(seed >= FORMAL_SEED_MIN for seed in seeds):
        raise ValueError("formal seeds are forbidden")

    methods = tuple(str(method) for method in plan.get("methods", []))
    if methods != ACTIVE_METHODS:
        raise ValueError(f"methods must be exactly {list(ACTIVE_METHODS)}")
    forbidden = sorted(set(methods) & set(FORBIDDEN_ACTIVE_METHODS))
    if forbidden:
        raise ValueError(f"Quartic is excluded from the active matrix: {forbidden}")

    alphas = _float_list(plan, "negative_alpha")
    anchors = _float_list(plan, "rarity_logit_anchor_coefficient")
    retentions = _float_list(plan, "reference_rare_retention")
    if alphas != [0.25, 0.5]:
        raise ValueError("negative_alpha grid must remain [0.25, 0.5]")
    if anchors != [0.25, 0.1]:
        raise ValueError("anchor grid must remain [0.25, 0.1]")
    if retentions != [0.25]:
        raise ValueError("reference_rare_retention must remain fixed at 0.25")
    if any(not 0.0 < value < 1.0 for value in retentions):
        raise ValueError("retention must be in (0,1)")

    training = plan.get("training", {})
    if int(training.get("maximum_steps", 0)) != 8000:
        raise ValueError("maximum_steps must remain 8000")
    if int(training.get("evaluation_interval_steps", 0)) != 100:
        raise ValueError("evaluation_interval_steps must remain 100")
    if str(training.get("device")) != "cpu":
        raise ValueError("development grid is frozen to CPU")
    if int(training.get("cpu_threads_per_run", 0)) != 1:
        raise ValueError("cpu_threads_per_run must remain 1")

    events = plan.get("events", {})
    collapse_ratio = float(events.get("task_collapse_ratio_to_paired_positive_only", -1.0))
    if collapse_ratio != 0.2:
        raise ValueError("task collapse ratio must remain 0.2")

    audit = plan.get("terminal_audit", {})
    if list(audit.get("window_1_steps", [])) != [4000, 6000]:
        raise ValueError("terminal window 1 must remain [4000,6000]")
    if list(audit.get("window_2_steps", [])) != [6000, 8000]:
        raise ValueError("terminal window 2 must remain [6000,8000]")
    if not bool(audit.get("required")):
        raise ValueError("terminal audit is required")


def build_tasks(plan: Mapping[str, Any], repo_root: Path, output_root: Path) -> list[GridTask]:
    validate_plan(plan)
    training = plan["training"]
    tasks: list[GridTask] = []
    for alpha in _float_list(plan, "negative_alpha"):
        for anchor in _float_list(plan, "rarity_logit_anchor_coefficient"):
            for retention in _float_list(plan, "reference_rare_retention"):
                for seed in plan["seeds"]:
                    for method in plan["methods"]:
                        tasks.append(
                            GridTask(
                                seed=int(seed),
                                method=str(method),
                                negative_alpha=float(alpha),
                                anchor=float(anchor),
                                retention=float(retention),
                                steps=int(training["maximum_steps"]),
                                eval_every=int(training["evaluation_interval_steps"]),
                                device=str(training["device"]),
                                source_module=str((repo_root / plan["source_module"]).resolve()),
                                base_config=str((repo_root / plan["base_config"]).resolve()),
                                output_dir=str((output_root / "runs").resolve()),
                            )
                        )
    run_ids = [task.run_id for task in tasks]
    if len(run_ids) != len(set(run_ids)):
        raise RuntimeError("grid generated duplicate run IDs")
    expected = 2 * 2 * 1 * 5 * 6
    if len(tasks) != expected:
        raise RuntimeError(f"expected {expected} runs, generated {len(tasks)}")
    return tasks


def load_source_module(path: Path):
    spec = importlib.util.spec_from_file_location("drpo_du1_e6_grid_source", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import source module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    revision = int(getattr(module, "PROTOCOL_REVISION", 0))
    if revision < MIN_PROTOCOL_REVISION:
        raise RuntimeError(
            f"protocol revision {revision} is too old; revision >= {MIN_PROTOCOL_REVISION} required"
        )
    required = ("load_config", "method_specs", "run_seed_bundle")
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise RuntimeError(f"source module is missing required symbols: {missing}")
    return module


def _completed_run(run_root: Path) -> bool:
    marker = run_root / "RUN_COMPLETE.json"
    summary = run_root / "summary.json"
    trajectory = run_root / "trajectory.jsonl"
    if not (marker.is_file() and summary.is_file() and trajectory.is_file()):
        return False
    try:
        payload = json.loads(marker.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    return bool(payload.get("completed")) and payload.get("status") == "complete"


def run_task(task: GridTask, resume: bool) -> dict[str, Any]:
    run_root = Path(task.output_dir) / task.run_id
    if resume and _completed_run(run_root):
        return {"run_id": task.run_id, "status": "skipped_complete"}

    run_root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    json_dump(run_root / "TASK.json", dataclasses.asdict(task))
    try:
        module = load_source_module(Path(task.source_module))
        config = module.load_config(Path(task.base_config))
        config["formal_parameter_freeze"] = False
        config["scientific_status"] = "pilot"
        config["seeds"]["held_out_formal"] = [task.seed]
        config["optimization"]["negative_alpha"] = task.negative_alpha
        config["optimization"]["rarity_logit_anchor_coefficient"] = task.anchor
        config["optimization"]["maximum_steps"] = task.steps
        config["optimization"]["evaluation_interval_steps"] = task.eval_every
        config["optimization"]["cpu_threads_per_run"] = 1
        config["taper"]["reference_rare_retention"] = task.retention
        config.setdefault("development_grid", {})
        config["development_grid"].update(
            {
                "active_method": task.method,
                "cell_id": task.cell_id,
                "quartic_excluded": True,
                "formal_result": False,
            }
        )

        specs = {spec.method: spec for spec in module.method_specs()}
        if task.method not in specs:
            raise RuntimeError(f"source module does not define method {task.method}")
        if task.method in FORBIDDEN_ACTIVE_METHODS:
            raise RuntimeError("Quartic must not run in this grid")

        yaml_dump(run_root / "resolved_config.yaml", config)
        bundle = module.run_seed_bundle(config, task.seed, [specs[task.method]], task.device)
        summaries = list(bundle["summaries"])
        if len(summaries) != 1:
            raise RuntimeError(f"expected one summary, got {len(summaries)}")
        summary = dict(summaries[0])
        summary.update(
            {
                "grid_cell_id": task.cell_id,
                "negative_alpha": task.negative_alpha,
                "rarity_logit_anchor_coefficient": task.anchor,
                "reference_rare_retention": task.retention,
                "formal_result": False,
                "method_ranking_allowed": False,
                "environment_validity_failure": not bool(bundle["audit"].get("passed", False)),
            }
        )
        json_dump(run_root / "summary.json", summary)
        write_jsonl(run_root / "trajectory.jsonl", bundle["trajectories"])
        json_dump(run_root / "environment_audit.json", bundle["audit"])
        json_dump(run_root / "coordinate_calibration.json", bundle["calibration"])
        checksums = {
            name: sha256_file(run_root / name)
            for name in (
                "resolved_config.yaml",
                "summary.json",
                "trajectory.jsonl",
                "environment_audit.json",
                "coordinate_calibration.json",
            )
        }
        elapsed = time.time() - started
        json_dump(
            run_root / "RUN_COMPLETE.json",
            {
                "completed": True,
                "status": "complete",
                "run_id": task.run_id,
                "elapsed_seconds": elapsed,
                "checksums": checksums,
                "formal_result": False,
            },
        )
        return {"run_id": task.run_id, "status": "complete", "elapsed_seconds": elapsed}
    except Exception as exc:  # noqa: BLE001 - preserve complete failure artifact
        json_dump(
            run_root / "RUN_FAILED.json",
            {
                "completed": False,
                "status": "failed",
                "run_id": task.run_id,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "elapsed_seconds": time.time() - started,
            },
        )
        return {
            "run_id": task.run_id,
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def paired_delta(rows: Sequence[Mapping[str, Any]], lhs: str, rhs: str, metric: str) -> dict[str, Any]:
    index: dict[str, dict[int, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        index[str(row["method"])][int(row["seed"])] = row
    seeds = sorted(set(index.get(lhs, {})) & set(index.get(rhs, {})))
    values = [float(index[lhs][seed][metric]) - float(index[rhs][seed][metric]) for seed in seeds]
    return {
        "lhs": lhs,
        "rhs": rhs,
        "metric": metric,
        "seeds": seeds,
        "mean": None if not values else sum(values) / len(values),
        "wins": sum(value > 0 for value in values),
        "values": values,
    }


def assign_task_collapse(
    rows: Sequence[dict[str, Any]], ratio: float
) -> None:
    """Assign collapse against the paired Positive-only run in each grid cell/seed."""

    positive = {
        (str(row["grid_cell_id"]), int(row["seed"])): float(
            row["final_expected_semantic_reward"]
        )
        for row in rows
        if row["method"] == "positive_only"
    }
    for row in rows:
        key = (str(row["grid_cell_id"]), int(row["seed"]))
        if key not in positive:
            row["task_performance_collapse"] = False
            row["task_collapse_reference_missing"] = True
            continue
        reference = positive[key]
        row["task_performance_collapse"] = bool(
            float(row["final_expected_semantic_reward"]) < ratio * reference
        )
        row["task_collapse_reference_missing"] = False
        row["paired_positive_only_reward"] = reference


def aggregate_results(
    output_root: Path,
    tasks: Sequence[GridTask],
    task_collapse_ratio: float,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    failed: list[str] = []
    for task in tasks:
        run_root = output_root / "runs" / task.run_id
        if (run_root / "RUN_FAILED.json").exists():
            failed.append(task.run_id)
            continue
        summary_path = run_root / "summary.json"
        if not _completed_run(run_root) or not summary_path.is_file():
            missing.append(task.run_id)
            continue
        rows.append(json.loads(summary_path.read_text()))

    assign_task_collapse(rows, task_collapse_ratio)
    for row in rows:
        # Persist the paired collapse classification back into each durable summary.
        matching = [
            candidate
            for candidate in tasks
            if candidate.seed == int(row["seed"])
            and candidate.method == str(row["method"])
            and candidate.cell_id == str(row["grid_cell_id"])
        ]
        if len(matching) == 1:
            json_dump(output_root / "runs" / matching[0].run_id / "summary.json", row)

    by_cell: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_cell[str(row["grid_cell_id"])].append(row)

    metrics = (
        "final_expected_semantic_reward",
        "final_hidden_optimal_family_probability",
        "final_prototype_effective_support",
        "final_rare_total_probability",
    )
    cells: dict[str, Any] = {}
    csv_rows: list[dict[str, Any]] = []
    for cell_id, cell_rows in sorted(by_cell.items()):
        by_method: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in cell_rows:
            by_method[str(row["method"])].append(row)
        method_summary: dict[str, Any] = {}
        for method, method_rows in sorted(by_method.items()):
            entry = {
                "runs": len(method_rows),
                **{
                    f"{metric}_mean": sum(float(row[metric]) for row in method_rows)
                    / len(method_rows)
                    for metric in metrics
                },
                "task_performance_collapse_events": sum(
                    bool(row.get("task_performance_collapse")) for row in method_rows
                ),
                "support_boundary_events": sum(
                    bool(row.get("support_boundary_event")) for row in method_rows
                ),
                "nan_inf_numerical_failures": sum(
                    bool(row.get("nan_inf_numerical_failure")) for row in method_rows
                ),
                "environment_validity_failures": sum(
                    bool(row.get("environment_validity_failure")) for row in method_rows
                ),
                "terminal_plateaus": sum(
                    row.get("terminal_class") == "terminal_plateau" for row in method_rows
                ),
            }
            method_summary[method] = entry
            csv_rows.append({"grid_cell_id": cell_id, "method": method, **entry})
        contrasts: dict[str, Any] = {}
        for method in ACTIVE_METHODS:
            if method == "positive_only":
                continue
            for reference in ("positive_only", "global_matched"):
                if method == reference:
                    continue
                key = f"{method}_minus_{reference}"
                contrasts[key] = {
                    metric: paired_delta(cell_rows, method, reference, metric)
                    for metric in metrics
                }
        cells[cell_id] = {"methods": method_summary, "paired_contrasts": contrasts}

    fields = sorted({key for row in csv_rows for key in row}) if csv_rows else []
    if fields:
        path = output_root / "grid_summary.csv"
        temp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
        with temp.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(csv_rows)
        os.replace(temp, path)

    terminal = {
        "experiment_id": EXPERIMENT_ID,
        "stage": "development_calibration",
        "formal_result": False,
        "expected_runs": len(tasks),
        "actual_runs": len(rows),
        "missing_runs": missing,
        "failed_runs": failed,
        "all_registered_runs_present": not missing and not failed and len(rows) == len(tasks),
        "task_performance_collapse_events": sum(
            bool(row.get("task_performance_collapse")) for row in rows
        ),
        "support_boundary_events": sum(bool(row.get("support_boundary_event")) for row in rows),
        "nan_inf_numerical_failures": sum(
            bool(row.get("nan_inf_numerical_failure")) for row in rows
        ),
        "environment_validity_failures": sum(
            bool(row.get("environment_validity_failure")) for row in rows
        ),
        "method_ranking_allowed": False,
    }
    aggregate = {
        "experiment_id": EXPERIMENT_ID,
        "stage": "development_calibration",
        "formal_result": False,
        "quartic_excluded_from_active_matrix": True,
        "cells": cells,
        "terminal_audit": terminal,
        "selection_gate": (
            "diagnostic_only_no_method_ranking; review separation, environment validity, "
            "and three failure classes before any later freeze"
        ),
    }
    json_dump(output_root / "aggregate_summary.json", aggregate)
    json_dump(output_root / "terminal_audit.json", terminal)
    return aggregate


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--grid-config",
        type=Path,
        default=Path("configs/du1_e6_separation_grid.yaml"),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-workers", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.resolve()
    grid_path = args.grid_config
    if not grid_path.is_absolute():
        grid_path = repo_root / grid_path
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    plan = load_yaml(grid_path)
    tasks = build_tasks(plan, repo_root, output_root)
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "stage": "development_calibration",
        "formal_result": False,
        "git_commit": git_text(repo_root, "rev-parse", "HEAD"),
        "git_status_porcelain": git_text(repo_root, "status", "--porcelain"),
        "grid_config": str(grid_path),
        "grid_config_sha256": sha256_file(grid_path),
        "expected_runs": len(tasks),
        "quartic_excluded_from_active_matrix": True,
        "formal_seeds_accessed": False,
        "tasks": [dataclasses.asdict(task) | {"run_id": task.run_id} for task in tasks],
    }
    json_dump(output_root / "grid_plan.json", manifest)
    if args.plan_only:
        print(json.dumps({"status": "plan_only", "expected_runs": len(tasks)}, sort_keys=True))
        return 0

    max_workers = args.max_workers or int(plan["training"]["max_parallel_runs"])
    if max_workers < 1:
        raise ValueError("max-workers must be positive")
    max_workers = min(max_workers, len(tasks))
    heartbeat = output_root / "HEARTBEAT.json"
    completed = 0
    failed = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(run_task, task, args.resume): task for task in tasks}
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            completed += 1
            failed += result["status"] == "failed"
            json_dump(
                heartbeat,
                {
                    "completed_or_skipped": completed,
                    "expected_runs": len(tasks),
                    "failed_runs": failed,
                    "last_result": result,
                    "updated_unix_time": time.time(),
                },
            )
            print(
                json.dumps(
                    {
                        "progress": f"{completed}/{len(tasks)}",
                        "failed": failed,
                        **result,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    aggregate = aggregate_results(
        output_root,
        tasks,
        float(plan["events"]["task_collapse_ratio_to_paired_positive_only"]),
    )
    terminal = aggregate["terminal_audit"]
    json_dump(
        output_root / "RUN_COMPLETE.json",
        {
            "completed": bool(terminal["all_registered_runs_present"]),
            "status": "complete" if terminal["all_registered_runs_present"] else "incomplete",
            "experiment_id": EXPERIMENT_ID,
            "stage": "development_calibration",
            "formal_result": False,
            "method_ranking_allowed": False,
            "expected_runs": len(tasks),
            "actual_runs": terminal["actual_runs"],
        },
    )
    return 0 if terminal["all_registered_runs_present"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
