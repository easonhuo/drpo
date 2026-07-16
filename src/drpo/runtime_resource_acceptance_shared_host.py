"""Shared-host semantics for runtime-resource acceptance.

The default target-server route contains DRPO-owned processes inside explicit CPU/GPU
ceilings and measures residual capacity inside those ceilings. It does not require
exclusive CPUs. Permanent external workloads are observation-only, while competing DRPO
runs remain blocking conflicts.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from drpo.runtime_resource_acceptance import StageResult, stage_result
from drpo.runtime_resource_autotune import atomic_write_json

TopologyStage = Callable[
    [Path, Path, Path, Mapping[str, Any]],
    StageResult,
]
E7Stage = Callable[
    [Path, Path, Path, Mapping[str, Any], Path],
    StageResult,
]
GPUStage = Callable[
    [Path, Path, Path, Mapping[str, Any], Path],
    StageResult,
]
ReportFunction = Callable[
    [Path, Mapping[str, Any], Mapping[str, Any], Sequence[StageResult], Mapping[str, Any]],
    dict[str, Any],
]

PERMANENT_EXTERNAL_PATTERNS = (
    "researchbench",
    "collector_v2.py",
    "aide",
    "joblib.externals.loky",
    "loky.process_executor",
)
OOM_KEYS = frozenset(
    {
        "oom",
        "oom_detected",
        "oom_failure",
        "oom_killed",
        "cuda_oom",
        "out_of_memory",
        "outofmemoryerror",
    }
)
OOM_TEXT = re.compile(
    r"(?:\bcuda\s+out\s+of\s+memory\b|\bout\s+of\s+memory\b|"
    r"\boutofmemoryerror\b|\bcuda\s+oom\b|\boom(?:[-_ ]killed)?\b)",
    re.IGNORECASE,
)


def shared_host_capacity_contract(profile: Mapping[str, Any]) -> dict[str, Any]:
    """Return the explicit hard ceilings and dynamic-selection semantics."""

    pools = profile["resource_pools"]
    e7 = profile["e7"]
    e8 = profile["e8"]
    gpu_count = min(len(pools["e8_gpu_ids"]), int(e8["max_devices"]))
    return {
        "mode": "shared_host_dynamic_measured_capacity",
        "exclusive_cpu_guarantee_required": False,
        "exclusive_partition_required": False,
        "permanent_external_process_presence_blocks": False,
        "competing_drpo_process_presence_blocks": True,
        "hard_limits": {
            "e7_cpu_pool": pools["e7_cpu_pool"],
            "e7_cpu_count": len(pools["e7_cpu_ids"]),
            "e7_max_workers": e7["max_workers"],
            "e8_cpu_pool": pools["e8_cpu_pool"],
            "e8_cpu_count": len(pools["e8_cpu_ids"]),
            "e8_gpu_ids": list(pools["e8_gpu_ids"]),
            "e8_max_devices": int(e8["max_devices"]),
            "e8_max_slots_per_gpu": int(e8["max_slots_per_gpu"]),
            "e8_max_total_slots": gpu_count * int(e8["max_slots_per_gpu"]),
        },
        "dynamic_selection": {
            "measurement_scope": "active_process_affinity",
            "e7_inputs": [
                "pool_local_cpu_occupancy",
                "measured_per_worker_cpu",
                "measured_per_worker_rss",
                "host_memory_headroom",
                "configured_worker_caps",
            ],
            "e8_inputs": [
                "pool_local_cpu_occupancy",
                "host_memory_headroom",
                "gpu_utilization",
                "free_vram",
                "measured_phase_envelopes",
                "configured_slot_caps",
            ],
            "launch_revalidation": True,
            "positive_prelaunch_downshift_allowed": True,
            "running_worker_resize_allowed": False,
            "unsafe_capacity_status": "BLOCKED",
            "automatic_external_process_control": False,
        },
        "owned_process_controls": {
            "exact_cpu_affinity_required": True,
            "e7_e8_pool_overlap_allowed": False,
            "terminate_unrelated_processes": False,
            "modify_unrelated_affinity": False,
            "owned_process_group_cleanup_only": True,
        },
    }


def _matches_patterns(row: Mapping[str, Any], patterns: Sequence[str]) -> bool:
    command = str(row.get("command", "")).lower()
    return any(str(pattern).strip().lower() in command for pattern in patterns)


def observe_external_workloads(
    inventory: Sequence[Mapping[str, Any]],
    *,
    patterns: Sequence[str],
    excluded_pids: set[int],
) -> dict[str, Any]:
    """Match permanent external workloads without turning presence into a gate."""

    normalized = [str(pattern).strip().lower() for pattern in patterns if str(pattern).strip()]
    matches = [
        dict(row)
        for row in inventory
        if int(row["pid"]) not in excluded_pids and _matches_patterns(row, normalized)
    ]
    return {
        "policy": "observe_only",
        "presence_blocks_stage0": False,
        "patterns": normalized,
        "match_count": len(matches),
        "matches": matches,
    }


def _load_mapping(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return dict(payload) if isinstance(payload, Mapping) else None


def _read_process_inventory(directory: Path) -> list[dict[str, Any]]:
    path = directory / "PROCESS_INVENTORY.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(payload, list):
        return []
    return [dict(row) for row in payload if isinstance(row, Mapping)]


def shared_host_topology_stage(
    original_stage: TopologyStage,
    root: Path,
    repo: Path,
    gpu_worktree: Path,
    profile: Mapping[str, Any],
) -> StageResult:
    """Keep permanent external processes observational and DRPO conflicts blocking."""

    original = original_stage(root, repo, gpu_worktree, profile)
    directory = root / "stage0_topology"
    details = dict(original.details)
    legacy_matches = [
        dict(row)
        for row in details.get("conflicts", [])
        if isinstance(row, Mapping)
    ]
    blocking_conflicts = [
        row
        for row in legacy_matches
        if not _matches_patterns(row, PERMANENT_EXTERNAL_PATTERNS)
    ]
    inventory = _read_process_inventory(directory)
    if not inventory:
        inventory = legacy_matches
    observation = observe_external_workloads(
        inventory,
        patterns=PERMANENT_EXTERNAL_PATTERNS,
        excluded_pids=set(),
    )
    observation["source"] = "stage0 full process inventory"
    contract = shared_host_capacity_contract(profile)
    atomic_write_json(directory / "OBSERVED_EXTERNAL_WORKLOADS.json", observation)
    atomic_write_json(directory / "SHARED_HOST_CAPACITY_CONTRACT.json", contract)

    details.update(
        {
            "legacy_topology_status": original.status,
            "conflicts": blocking_conflicts,
            "observed_external_workloads": observation,
            "shared_host_capacity_contract": contract,
        }
    )

    process_only_block = original.status == "BLOCKED" and "error" not in original.details
    if original.status == "PASS":
        status = "PASS"
    elif process_only_block and not blocking_conflicts:
        status = "PASS"
    else:
        status = original.status
    return stage_result(
        root,
        "stage0_topology",
        status,
        original.started_utc,
        details,
    )


def shared_host_e7_stage(
    original_stage: E7Stage,
    root: Path,
    repo: Path,
    profile_path: Path,
    profile: Mapping[str, Any],
    ledger: Path,
) -> StageResult:
    """Expose the actually admitted E7 worker count in the terminal stage result."""

    original = original_stage(root, repo, profile_path, profile, ledger)
    if original.status not in {"PASS", "INCONCLUSIVE"}:
        return original
    evidence = _load_mapping(root / "stage2_e7_cpu_v2" / "SELECTED_LIVENESS.json")
    if evidence is None:
        return stage_result(
            root,
            "stage2_e7_cpu_v2",
            "FAIL",
            original.started_utc,
            {
                **dict(original.details),
                "error": "selected liveness passed without admission evidence",
            },
        )
    proposed = int(evidence.get("proposed_workers", 0) or 0)
    admitted = int(evidence.get("admitted_workers", 0) or 0)
    if proposed < 1 or admitted < 1 or admitted > proposed:
        return stage_result(
            root,
            "stage2_e7_cpu_v2",
            "FAIL",
            original.started_utc,
            {
                **dict(original.details),
                "error": "selected liveness contains invalid admission counts",
                "proposed_workers": proposed,
                "admitted_workers": admitted,
            },
        )
    details = {
        **dict(original.details),
        "proposed_workers": proposed,
        "admitted_workers": admitted,
        "selected_workers": admitted,
        "downshifted": bool(evidence.get("downshifted")),
        "runtime_admission": evidence.get("runtime_admission"),
        "revalidation": evidence.get("revalidation"),
        "candidate_above_one_observed": admitted > 1,
    }
    status = "PASS" if admitted > 1 else "INCONCLUSIVE"
    return stage_result(
        root,
        "stage2_e7_cpu_v2",
        status,
        original.started_utc,
        details,
    )


def shared_host_gpu_stage(
    original_stage: GPUStage,
    root: Path,
    repo: Path,
    gpu_repo: Path,
    profile: Mapping[str, Any],
    ledger: Path,
) -> StageResult:
    """Separate selected E8 capacity from actually probed concurrency."""

    original = original_stage(root, repo, gpu_repo, profile, ledger)
    if original.status not in {"PASS", "INCONCLUSIVE"}:
        return original
    document = _load_mapping(
        root / "stage3_gpu_placement" / "work" / "RUNTIME_SELECTION.json"
    )
    if document is None:
        return stage_result(
            root,
            "stage3_gpu_placement",
            "FAIL",
            original.started_utc,
            {
                **dict(original.details),
                "error": "GPU placement passed without RUNTIME_SELECTION.json",
            },
        )
    selection = document.get("selection")
    probe = document.get("probe")
    if not isinstance(selection, Mapping) or not isinstance(probe, Mapping):
        return stage_result(
            root,
            "stage3_gpu_placement",
            "FAIL",
            original.started_utc,
            {
                **dict(original.details),
                "error": "GPU placement selection or probe payload is missing",
            },
        )
    selected_devices = [str(value) for value in selection.get("selected_device_ids", [])]
    slots_per_gpu = int(selection.get("slots_per_gpu", 0) or 0)
    selected_total_slots = int(selection.get("total_runtime_slots", 0) or 0)
    records = probe.get("records", [])
    if not isinstance(records, list):
        records = []
    probed = [
        int(row.get("concurrency", 0) or 0)
        for row in records
        if isinstance(row, Mapping)
    ]
    max_probed_concurrency = max(probed, default=0)
    details = {
        **dict(original.details),
        "selected_device_ids": selected_devices,
        "selected_device_count": len(selected_devices),
        "selected_slots_per_gpu": slots_per_gpu,
        "selected_total_runtime_slots": selected_total_slots,
        "selected_capacity_above_one": selected_total_slots > 1,
        "maximum_actually_probed_concurrency": max_probed_concurrency,
        "candidate_above_one_observed": max_probed_concurrency > 1,
        "multi_slot_capacity_actually_validated": max_probed_concurrency > 1,
    }
    return stage_result(
        root,
        "stage3_gpu_placement",
        original.status,
        original.started_utc,
        details,
    )


def _contains_explicit_oom(value: object) -> bool:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).strip().lower()
            if normalized in OOM_KEYS and bool(item):
                return True
            if normalized in {"error", "message", "reason", "classification"}:
                if isinstance(item, str) and OOM_TEXT.search(item):
                    return True
            if _contains_explicit_oom(item):
                return True
        return False
    if isinstance(value, (list, tuple)):
        return any(_contains_explicit_oom(item) for item in value)
    if isinstance(value, str):
        return bool(OOM_TEXT.search(value))
    return False


def explicit_oom_stage_names(results: Sequence[StageResult]) -> list[str]:
    return [result.name for result in results if _contains_explicit_oom(result.details)]


def shared_host_report(
    original_report: ReportFunction,
    root: Path,
    checkout: Mapping[str, Any],
    profile: Mapping[str, Any],
    results: Sequence[StageResult],
    final_audit: Mapping[str, Any],
) -> dict[str, Any]:
    """Correct shared-host failure classes without substring-based OOM false positives."""

    summary = original_report(root, checkout, profile, results, final_audit)
    classes = summary.get("separate_failure_classes")
    if not isinstance(classes, dict):
        classes = {}
        summary["separate_failure_classes"] = classes
    oom_stages = explicit_oom_stage_names(results)
    classes["resource_boundary_or_oom"] = oom_stages
    classes["oom_failure"] = oom_stages
    atomic_write_json(root / "ACCEPTANCE_SUMMARY.json", summary)
    return summary
