from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from drpo.runtime_resource_acceptance import StageResult, utc_now
from drpo.runtime_resource_acceptance_shared_host import (
    observe_external_workloads,
    shared_host_capacity_contract,
    shared_host_topology_stage,
)


def _profile() -> dict[str, Any]:
    return {
        "conflict_process_patterns": ["ResearchBench", "AIDE", "loky"],
        "resource_pools": {
            "e7_cpu_pool": "0-3",
            "e7_cpu_ids": [0, 1, 2, 3],
            "e8_cpu_pool": "4-7",
            "e8_cpu_ids": [4, 5, 6, 7],
            "e8_gpu_ids": ["0", "1"],
        },
        "e7": {"max_workers": 3},
        "e8": {"max_devices": 1, "max_slots_per_gpu": 2},
    }


def _result(status: str, details: Mapping[str, Any]) -> StageResult:
    now = utc_now()
    return StageResult("stage0_topology", status, now, now, dict(details))


def test_capacity_contract_exposes_hard_caps_without_exclusivity() -> None:
    contract = shared_host_capacity_contract(_profile())
    assert contract["mode"] == "shared_host_dynamic_measured_capacity"
    assert contract["exclusive_cpu_guarantee_required"] is False
    assert contract["exclusive_partition_required"] is False
    assert contract["external_process_presence_blocks"] is False
    assert contract["hard_limits"]["e7_cpu_count"] == 4
    assert contract["hard_limits"]["e7_max_workers"] == 3
    assert contract["hard_limits"]["e8_cpu_count"] == 4
    assert contract["hard_limits"]["e8_max_total_slots"] == 2
    assert contract["dynamic_selection"]["launch_revalidation"] is True


def test_external_workload_matching_is_observation_only() -> None:
    observation = observe_external_workloads(
        [
            {"pid": 10, "command": "python ResearchBench/collector_v2.py"},
            {"pid": 11, "command": "python -m joblib.externals.loky"},
            {"pid": 12, "command": "python unrelated.py"},
        ],
        patterns=["ResearchBench", "loky"],
        excluded_pids={11},
    )
    assert observation["presence_blocks_stage0"] is False
    assert observation["match_count"] == 1
    assert observation["matches"][0]["pid"] == 10


def test_conflict_only_legacy_block_becomes_shared_host_pass(tmp_path: Path) -> None:
    legacy_match = {
        "pid": 100,
        "command": "python ResearchBench/collector_v2.py",
        "affinity_cpu_ids": list(range(8)),
    }

    def original(
        root: Path,
        repo: Path,
        gpu_worktree: Path,
        profile: Mapping[str, Any],
    ) -> StageResult:
        del root, repo, gpu_worktree, profile
        return _result("BLOCKED", {"conflicts": [legacy_match]})

    result = shared_host_topology_stage(
        original,
        tmp_path,
        tmp_path / "repo",
        tmp_path / "gpu",
        _profile(),
    )
    assert result.status == "PASS"
    assert result.details["legacy_topology_status"] == "BLOCKED"
    assert result.details["conflicts"] == []
    assert result.details["observed_external_workloads"]["match_count"] == 1
    assert (
        result.details["shared_host_capacity_contract"]
        ["external_process_presence_blocks"]
        is False
    )
    audit = json.loads(
        (tmp_path / "stage0_topology" / "OBSERVED_EXTERNAL_WORKLOADS.json").read_text()
    )
    assert audit["matches"][0]["pid"] == 100


def test_real_topology_block_is_not_reclassified(tmp_path: Path) -> None:
    def original(
        root: Path,
        repo: Path,
        gpu_worktree: Path,
        profile: Mapping[str, Any],
    ) -> StageResult:
        del root, repo, gpu_worktree, profile
        return _result("BLOCKED", {"error": "CPU pool unavailable"})

    result = shared_host_topology_stage(
        original,
        tmp_path,
        tmp_path / "repo",
        tmp_path / "gpu",
        _profile(),
    )
    assert result.status == "BLOCKED"
    assert result.details["error"] == "CPU pool unavailable"


def test_implementation_failure_is_not_reclassified(tmp_path: Path) -> None:
    def original(
        root: Path,
        repo: Path,
        gpu_worktree: Path,
        profile: Mapping[str, Any],
    ) -> StageResult:
        del root, repo, gpu_worktree, profile
        return _result("FAIL", {"error_type": "RuntimeError", "error": "boom"})

    result = shared_host_topology_stage(
        original,
        tmp_path,
        tmp_path / "repo",
        tmp_path / "gpu",
        _profile(),
    )
    assert result.status == "FAIL"
    assert result.details["error_type"] == "RuntimeError"
