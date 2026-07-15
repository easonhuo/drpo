"""Shared-host preflight semantics for runtime-resource acceptance.

The default target-server route contains DRPO-owned processes inside explicit CPU/GPU
ceilings and measures residual capacity inside those ceilings. It does not require
exclusive CPUs and does not use external process names as a readiness gate.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from drpo.runtime_resource_acceptance import StageResult, stage_result
from drpo.runtime_resource_autotune import atomic_write_json

TopologyStage = Callable[
    [Path, Path, Path, Mapping[str, Any]],
    StageResult,
]


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
        "external_process_presence_blocks": False,
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


def observe_external_workloads(
    inventory: Sequence[Mapping[str, Any]],
    *,
    patterns: Sequence[str],
    excluded_pids: set[int],
) -> dict[str, Any]:
    """Match configured process patterns without turning presence into a gate."""

    normalized = [
        str(pattern).strip().lower()
        for pattern in patterns
        if str(pattern).strip()
    ]
    matches = [
        dict(row)
        for row in inventory
        if int(row["pid"]) not in excluded_pids
        and any(
            pattern in str(row.get("command", "")).lower()
            for pattern in normalized
        )
    ]
    return {
        "policy": "observe_only",
        "presence_blocks_stage0": False,
        "patterns": normalized,
        "match_count": len(matches),
        "matches": matches,
    }


def shared_host_topology_stage(
    original_stage: TopologyStage,
    root: Path,
    repo: Path,
    gpu_worktree: Path,
    profile: Mapping[str, Any],
) -> StageResult:
    """Reinterpret only process-name matches as observation-only evidence.

    The original stage remains authoritative for checkout identity, external inputs,
    inherited affinity, requested pool availability, and unexpected implementation
    errors. Only its legacy process-pattern block is superseded.
    """

    original = original_stage(root, repo, gpu_worktree, profile)
    directory = root / "stage0_topology"
    details = dict(original.details)
    legacy_matches = [
        dict(row)
        for row in details.get("conflicts", [])
        if isinstance(row, Mapping)
    ]
    observation = {
        "policy": "observe_only",
        "presence_blocks_stage0": False,
        "patterns": [
            str(pattern).strip().lower()
            for pattern in profile["conflict_process_patterns"]
            if str(pattern).strip()
        ],
        "match_count": len(legacy_matches),
        "matches": legacy_matches,
        "source": "legacy topology process-pattern inventory",
    }
    contract = shared_host_capacity_contract(profile)
    atomic_write_json(directory / "OBSERVED_EXTERNAL_WORKLOADS.json", observation)
    atomic_write_json(directory / "SHARED_HOST_CAPACITY_CONTRACT.json", contract)

    details.update(
        {
            "legacy_topology_status": original.status,
            "conflicts": [],
            "observed_external_workloads": observation,
            "shared_host_capacity_contract": contract,
        }
    )

    conflict_only_block = (
        original.status == "BLOCKED"
        and "error" not in original.details
        and bool(legacy_matches)
    )
    if original.status == "PASS" or conflict_only_block:
        return stage_result(
            root,
            "stage0_topology",
            "PASS",
            original.started_utc,
            details,
        )
    return stage_result(
        root,
        "stage0_topology",
        original.status,
        original.started_utc,
        details,
    )
