"""Command builders and evidence inspection for runtime-resource acceptance."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from drpo.runtime_resource_acceptance import AcceptanceError

NUMERICAL_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])(?:nan|inf|infinity|non[- ]finite)(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
THREAD_ENVIRONMENT_NAMES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def pool_command(
    repo: Path,
    *,
    cpu_pool: str,
    identity: Path,
    command: Sequence[str],
    gpu_ids: Sequence[str] = (),
    dry_run: bool = False,
) -> list[str]:
    values = [
        sys.executable,
        str(repo / "scripts/run_with_resource_pool.py"),
        "--cpu-pool",
        cpu_pool,
        "--pool-identity",
        str(identity),
    ]
    if gpu_ids:
        values.extend(["--gpu-pool", ",".join(gpu_ids)])
    if dry_run:
        values.append("--dry-run")
    return [*values, "--", *(str(item) for item in command)]


def gpu_selection_command(
    gpu_repo: Path,
    profile: Mapping[str, Any],
    *,
    work_dir: Path,
    gpu_ids: Sequence[str],
    max_devices: int | None = None,
    max_slots: int | None = None,
) -> list[str]:
    e8 = profile["e8"]
    return [
        sys.executable,
        str(gpu_repo / "scripts/run_countdown_e8_oracle_offline_v2_taper_auto.py"),
        "--selection-only",
        "--model_path",
        e8["model_path"],
        "--work_dir",
        str(work_dir),
        "--bank",
        e8["bank"],
        "--val",
        e8["val"],
        "--global_calibration",
        e8["global_calibration"],
        "--base_config",
        e8["base_config"],
        "--sweep_config",
        e8["sweep_config"],
        "--gpus",
        ",".join(gpu_ids),
        "--required-free-gpu-memory-gib",
        str(e8["required_free_gpu_memory_gib"]),
        "--required-host-memory-gib-per-worker",
        str(e8["required_host_memory_gib_per_worker"]),
        "--gpu-memory-headroom-fraction",
        str(e8["gpu_memory_headroom_fraction"]),
        "--host-memory-headroom-fraction",
        str(e8["host_memory_headroom_fraction"]),
        "--per-worker-host-memory-safety-factor",
        str(e8["per_worker_host_memory_safety_factor"]),
        "--per-worker-vram-safety-factor",
        str(e8["per_worker_vram_safety_factor"]),
        "--cpu-fraction",
        str(e8["cpu_fraction"]),
        "--per-worker-cpu-safety-factor",
        str(e8["per_worker_cpu_safety_factor"]),
        "--minimum-cpu-cores-per-worker",
        str(e8["minimum_cpu_cores_per_worker"]),
        "--maximum-gpu-utilization-percent",
        str(e8["maximum_gpu_utilization_percent"]),
        "--max-devices",
        str(max_devices or e8["max_devices"]),
        "--max-slots-per-gpu",
        str(max_slots or e8["max_slots_per_gpu"]),
        "--single-probe-seconds",
        str(e8["single_probe_seconds"]),
        "--validation-probe-seconds",
        str(e8["validation_probe_seconds"]),
        "--probe-budget-seconds",
        str(e8["probe_budget_seconds"]),
        "--probe-free-floor-gib",
        str(e8["probe_free_floor_gib"]),
    ]


def e7_plan_command(repo: Path, profile: Mapping[str, Any], work_dir: Path) -> list[str]:
    e7 = profile["e7"]
    values = [
        sys.executable,
        str(repo / "scripts/run_e7_ppo_w0_grid_pilot_auto.py"),
        "plan",
        "--repo-root",
        str(repo),
        "--contract",
        e7["contract"],
        "--run-spec",
        e7["run_spec"],
        "--grid",
        e7["grid"],
        "--work-dir",
        str(work_dir),
        "--fallback-workers",
        str(e7["fallback_workers"]),
        "--probe-steps",
        str(e7["probe_steps"]),
        "--probe-seed",
        str(e7["probe_seed"]),
        "--probe-seconds",
        str(e7["probe_seconds"]),
        "--throughput-retention-fraction",
        str(e7["throughput_retention_fraction"]),
        "--cpu-fraction",
        str(e7["cpu_fraction"]),
        "--memory-headroom-fraction",
        str(e7["memory_headroom_fraction"]),
        "--per-worker-safety-factor",
        str(e7["per_worker_safety_factor"]),
        "--per-worker-cpu-safety-factor",
        str(e7["per_worker_cpu_safety_factor"]),
        "--minimum-cpu-cores-per-worker",
        str(e7["minimum_cpu_cores_per_worker"]),
        "--max-growth-factor",
        str(e7["max_growth_factor"]),
        "--minimum-branches-for-probe",
        str(e7["minimum_branches_for_probe"]),
        "--revalidation-samples",
        str(e7["revalidation_samples"]),
        "--revalidation-sample-seconds",
        str(e7["revalidation_sample_seconds"]),
    ]
    if e7["max_workers"] is not None:
        values.extend(["--max-workers", str(e7["max_workers"])])
    return values


def internal_e7_command(
    repo: Path,
    profile_path: Path,
    action: str,
    work_dir: Path,
    output: Path,
) -> list[str]:
    return [
        sys.executable,
        str(repo / "scripts/run_runtime_resource_acceptance.py"),
        "--profile",
        str(profile_path),
        "--internal-e7-action",
        action,
        "--e7-work-dir",
        str(work_dir),
        "--internal-output",
        str(output),
    ]


def recursive_values(value: Any, key: str) -> list[Any]:
    values: list[Any] = []
    if isinstance(value, Mapping):
        for current_key, current_value in value.items():
            if current_key == key:
                values.append(current_value)
            values.extend(recursive_values(current_value, key))
    elif isinstance(value, list):
        for item in value:
            values.extend(recursive_values(item, key))
    return values


def candidate_above_one(document: Mapping[str, Any]) -> bool:
    """Return true only for a measured/executed worker or slot count above one."""

    for key in ("concurrency", "workers", "slots_per_gpu"):
        for value in recursive_values(document, key):
            try:
                if int(value) > 1:
                    return True
            except (TypeError, ValueError):
                continue
    return False


def gpu_failures(document: Mapping[str, Any]) -> list[str]:
    failures: list[str] = []
    for key, label in (
        ("oom_detected", "oom_detected"),
        ("controller_terminated_workers", "controller_terminated_workers"),
        ("process_group_alive_after_cleanup", "orphan_process_group"),
    ):
        if any(value is True for value in recursive_values(document, key)):
            failures.append(label)
    for value in recursive_values(document, "worker_returncodes"):
        if isinstance(value, list) and any(item != 0 for item in value):
            failures.append("nonzero_worker_returncode")
    return sorted(set(failures))


def first_numeric(document: Mapping[str, Any], key: str) -> float | None:
    for value in recursive_values(document, key):
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def numerical_matches(paths: Sequence[Path]) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for path in paths:
        if not path.is_file() or path.stat().st_size > 20 * 1024 * 1024:
            continue
        if path.suffix.lower() not in {".log", ".txt", ".json", ".jsonl", ".csv"}:
            continue
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1
        ):
            if NUMERICAL_PATTERN.search(line):
                matches.append({"path": str(path), "line": line_number, "text": line[:500]})
                if len(matches) == 100:
                    return matches
    return matches


def last_json_line(path: Path) -> dict[str, Any]:
    for line in reversed(path.read_text(encoding="utf-8", errors="replace").splitlines()):
        if not line.strip().startswith("{"):
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise AcceptanceError(f"no JSON object in log: {path}")
