"""Classify fail-closed runtime capacity exhaustion for server acceptance."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from drpo.runtime_resource_acceptance import StageResult, stage_result
from drpo.runtime_resource_autotune import load_json

E7_CAPACITY_FAILURES = frozenset({"cpu_capacity_changed", "memory_capacity_changed"})
E7_PLAN_CAPACITY_SIGNATURES = (
    "measured cpu capacity cannot support one worker",
    "measured cpu/ram capacity produced no worker slot",
)
GPU_CAPACITY_SIGNATURES = (
    "measured system cpu occupancy leaves no safe capacity for one gpu worker",
    "measured worker cannot fit after host/cpu headroom",
    "insufficient host memory for one worker after safety headroom",
)
CAPACITY_FATAL_SIGNATURES = (
    "cuda out of memory",
    "outofmemoryerror",
    "oom signature detected",
    "nan/inf numerical failure",
)


def _relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _read_log(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace").lower()
    except OSError:
        return None


def _latest_revalidation(stage_dir: Path) -> tuple[Path, dict[str, Any]] | None:
    candidates = sorted(
        stage_dir.glob("work/_runtime_resource_attempts/*/RUNTIME_REVALIDATION.json"),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    for path in candidates:
        try:
            payload = load_json(path)
        except Exception:  # noqa: BLE001 - malformed evidence must not be normalized
            continue
        if isinstance(payload, dict):
            return path, payload
    return None


def _e7_plan_capacity_block(root: Path) -> dict[str, Any] | None:
    log_path = root / "stage2_e7_cpu_v2" / "plan.log"
    lowered = _read_log(log_path)
    if lowered is None:
        return None
    if any(signature in lowered for signature in CAPACITY_FATAL_SIGNATURES):
        return None
    matched = next(
        (signature for signature in E7_PLAN_CAPACITY_SIGNATURES if signature in lowered),
        None,
    )
    if matched is None:
        return None
    return {
        "capacity_unavailable": True,
        "capacity_source": "e7_plan_cpu_or_memory_envelope",
        "capacity_failures": [matched.replace(" ", "_")],
        "capacity_evidence_paths": [_relative(log_path, root)],
        "selected_workers": None,
    }


def _e7_capacity_block(root: Path) -> dict[str, Any] | None:
    stage_dir = root / "stage2_e7_cpu_v2"
    found = _latest_revalidation(stage_dir)
    if found is None:
        return _e7_plan_capacity_block(root)
    path, payload = found
    raw_failures = payload.get("failures")
    if not isinstance(raw_failures, list) or not raw_failures:
        return None
    failures = {str(value) for value in raw_failures}
    if payload.get("decision") != "BLOCK" or not failures <= E7_CAPACITY_FAILURES:
        return None

    cpu = payload.get("cpu_revalidation")
    memory = payload.get("memory_revalidation")
    details: dict[str, Any] = {
        "capacity_unavailable": True,
        "capacity_source": "e7_runtime_revalidation",
        "capacity_failures": sorted(failures),
        "capacity_evidence_paths": [_relative(path, root)],
        "selected_workers": payload.get("selected_workers"),
    }
    if isinstance(cpu, Mapping):
        details["cpu_capacity"] = {
            "conservative_system_busy_cores": cpu.get(
                "conservative_system_busy_cores"
            ),
            "planned_worker_cpu_cores": cpu.get("planned_worker_cpu_cores"),
            "affinity_budget_cores": cpu.get("affinity_budget_cores"),
            "affinity_projected_total_busy_cores": cpu.get(
                "affinity_projected_total_busy_cores"
            ),
            "quota_domains": cpu.get("quota_domains"),
            "ok": cpu.get("ok"),
        }
    if isinstance(memory, Mapping):
        details["memory_capacity"] = dict(memory)
    return details


def _gpu_capacity_block(root: Path) -> dict[str, Any] | None:
    log_path = root / "stage3_gpu_placement" / "selection.log"
    lowered = _read_log(log_path)
    if lowered is None:
        return None
    if any(signature in lowered for signature in CAPACITY_FATAL_SIGNATURES):
        return None
    matched = next(
        (signature for signature in GPU_CAPACITY_SIGNATURES if signature in lowered),
        None,
    )
    if matched is None:
        return None
    return {
        "capacity_unavailable": True,
        "capacity_source": "gpu_placement_cpu_or_memory_envelope",
        "capacity_failures": [matched.replace(" ", "_")],
        "capacity_evidence_paths": [_relative(log_path, root)],
    }


def normalize_capacity_block(root: Path, result: StageResult) -> StageResult:
    """Convert pure safe-capacity exhaustion from FAIL to BLOCKED.

    Selector arithmetic and fail-closed behavior remain unchanged. This function only
    reconciles the stage status with the acceptance state machine, where unavailable
    safe capacity is BLOCKED rather than a contract/process/code failure.
    """

    if result.status != "FAIL":
        return result
    if result.name == "stage2_e7_cpu_v2":
        capacity = _e7_capacity_block(root)
    elif result.name == "stage3_gpu_placement":
        capacity = _gpu_capacity_block(root)
    else:
        capacity = None
    if capacity is None:
        return result
    details = {
        **dict(result.details),
        **capacity,
        "original_status": result.status,
        "classification": "safe_capacity_unavailable",
        "scientific_matrix_changed": False,
    }
    return stage_result(root, result.name, "BLOCKED", result.started_utc, details)
