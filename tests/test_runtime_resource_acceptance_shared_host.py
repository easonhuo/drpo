from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from drpo.runtime_resource_acceptance import StageResult, utc_now
from drpo.runtime_resource_acceptance_shared_host import (
    PERMANENT_EXTERNAL_PATTERNS,
    explicit_oom_stage_names,
    observe_external_workloads,
    shared_host_capacity_contract,
    shared_host_e7_stage,
    shared_host_gpu_stage,
    shared_host_report,
    shared_host_topology_stage,
)


def _profile() -> dict[str, Any]:
    return {
        "conflict_process_patterns": ["run_e7_", "countdown_e8", "EXT-H-E7", "EXT-C-E8"],
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


def _result(
    status: str,
    details: Mapping[str, Any],
    *,
    name: str = "stage0_topology",
) -> StageResult:
    now = utc_now()
    return StageResult(name, status, now, now, dict(details))


def _write_inventory(root: Path, rows: list[dict[str, Any]]) -> None:
    directory = root / "stage0_topology"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "PROCESS_INVENTORY.json").write_text(
        json.dumps(rows),
        encoding="utf-8",
    )


def test_capacity_contract_exposes_hard_caps_without_exclusivity() -> None:
    contract = shared_host_capacity_contract(_profile())
    assert contract["mode"] == "shared_host_dynamic_measured_capacity"
    assert contract["exclusive_cpu_guarantee_required"] is False
    assert contract["exclusive_partition_required"] is False
    assert contract["permanent_external_process_presence_blocks"] is False
    assert contract["competing_drpo_process_presence_blocks"] is True
    assert contract["hard_limits"]["e7_cpu_count"] == 4
    assert contract["hard_limits"]["e7_max_workers"] == 3
    assert contract["hard_limits"]["e8_cpu_count"] == 4
    assert contract["hard_limits"]["e8_max_total_slots"] == 2
    assert contract["dynamic_selection"]["launch_revalidation"] is True
    assert (
        contract["dynamic_selection"]["positive_prelaunch_downshift_allowed"]
        is True
    )
    assert contract["dynamic_selection"]["running_worker_resize_allowed"] is False


def test_external_workload_matching_is_observation_only() -> None:
    observation = observe_external_workloads(
        [
            {"pid": 10, "command": "python ResearchBench/collector_v2.py"},
            {"pid": 11, "command": "python -m joblib.externals.loky"},
            {"pid": 12, "command": "python unrelated.py"},
        ],
        patterns=PERMANENT_EXTERNAL_PATTERNS,
        excluded_pids={11},
    )
    assert observation["presence_blocks_stage0"] is False
    assert observation["match_count"] == 1
    assert observation["matches"][0]["pid"] == 10


def test_permanent_external_only_legacy_block_becomes_pass(tmp_path: Path) -> None:
    external = {
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
        del repo, gpu_worktree, profile
        _write_inventory(root, [external])
        return _result("BLOCKED", {"conflicts": [external]})

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
    audit = json.loads(
        (tmp_path / "stage0_topology" / "OBSERVED_EXTERNAL_WORKLOADS.json").read_text()
    )
    assert audit["matches"][0]["pid"] == 100


def test_competing_drpo_process_remains_blocking(tmp_path: Path) -> None:
    external = {
        "pid": 100,
        "command": "python ResearchBench/collector_v2.py",
        "affinity_cpu_ids": list(range(8)),
    }
    competing = {
        "pid": 101,
        "command": "python scripts/run_e7_existing.py",
        "affinity_cpu_ids": [0, 1],
    }

    def original(
        root: Path,
        repo: Path,
        gpu_worktree: Path,
        profile: Mapping[str, Any],
    ) -> StageResult:
        del repo, gpu_worktree, profile
        _write_inventory(root, [external, competing])
        return _result("BLOCKED", {"conflicts": [external, competing]})

    result = shared_host_topology_stage(
        original,
        tmp_path,
        tmp_path / "repo",
        tmp_path / "gpu",
        _profile(),
    )
    assert result.status == "BLOCKED"
    assert [row["pid"] for row in result.details["conflicts"]] == [101]
    assert result.details["observed_external_workloads"]["match_count"] == 1
    assert result.details["observed_external_workloads"]["matches"][0]["pid"] == 100


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


def test_e7_stage_reports_actually_admitted_workers(tmp_path: Path) -> None:
    directory = tmp_path / "stage2_e7_cpu_v2"
    directory.mkdir()
    (directory / "SELECTED_LIVENESS.json").write_text(
        json.dumps(
            {
                "proposed_workers": 123,
                "admitted_workers": 113,
                "selected_workers": 113,
                "downshifted": True,
                "runtime_admission": {"decision": "ALLOW", "admitted_workers": 113},
                "revalidation": {"decision": "ALLOW_WITH_DOWNSHIFT"},
            }
        ),
        encoding="utf-8",
    )

    def original(
        root: Path,
        repo: Path,
        profile_path: Path,
        profile: Mapping[str, Any],
        ledger: Path,
    ) -> StageResult:
        del root, repo, profile_path, profile, ledger
        return _result(
            "PASS",
            {"selected_workers": 123, "selection_digest": "digest"},
            name="stage2_e7_cpu_v2",
        )

    result = shared_host_e7_stage(
        original,
        tmp_path,
        tmp_path / "repo",
        tmp_path / "profile.json",
        _profile(),
        tmp_path / "ledger.jsonl",
    )
    assert result.status == "PASS"
    assert result.details["proposed_workers"] == 123
    assert result.details["admitted_workers"] == 113
    assert result.details["selected_workers"] == 113
    assert result.details["downshifted"] is True
    assert result.details["candidate_above_one_observed"] is True


def test_gpu_stage_separates_selected_slots_from_probed_concurrency(
    tmp_path: Path,
) -> None:
    work = tmp_path / "stage3_gpu_placement" / "work"
    work.mkdir(parents=True)
    (work / "RUNTIME_SELECTION.json").write_text(
        json.dumps(
            {
                "selection": {
                    "selected_device_ids": [str(index) for index in range(8)],
                    "slots_per_gpu": 1,
                    "total_runtime_slots": 8,
                },
                "probe": {"records": [{"concurrency": 1, "success": True}]},
            }
        ),
        encoding="utf-8",
    )

    def original(
        root: Path,
        repo: Path,
        gpu_repo: Path,
        profile: Mapping[str, Any],
        ledger: Path,
    ) -> StageResult:
        del root, repo, gpu_repo, profile, ledger
        return _result(
            "INCONCLUSIVE",
            {"candidate_above_one_observed": False},
            name="stage3_gpu_placement",
        )

    result = shared_host_gpu_stage(
        original,
        tmp_path,
        tmp_path / "repo",
        tmp_path / "gpu",
        _profile(),
        tmp_path / "ledger.jsonl",
    )
    assert result.status == "INCONCLUSIVE"
    assert result.details["selected_device_count"] == 8
    assert result.details["selected_slots_per_gpu"] == 1
    assert result.details["selected_total_runtime_slots"] == 8
    assert result.details["selected_capacity_above_one"] is True
    assert result.details["maximum_actually_probed_concurrency"] == 1
    assert result.details["candidate_above_one_observed"] is False
    assert result.details["multi_slot_capacity_actually_validated"] is False


def test_oom_detection_does_not_match_headroom_substring() -> None:
    results = [
        _result(
            "PASS",
            {
                "memory_headroom_fraction": 0.15,
                "gpu_memory_headroom_fraction": 0.12,
                "oom_detected": False,
            },
            name="stage0_topology",
        ),
        _result(
            "FAIL",
            {"error": "CUDA out of memory during probe"},
            name="stage3_gpu_placement",
        ),
    ]
    assert explicit_oom_stage_names(results) == ["stage3_gpu_placement"]


def test_shared_host_report_replaces_substring_oom_classification(
    tmp_path: Path,
) -> None:
    results: Sequence[StageResult] = [
        _result(
            "PASS",
            {"memory_headroom_fraction": 0.15},
            name="stage0_topology",
        )
    ]

    def original_report(
        root: Path,
        checkout: Mapping[str, Any],
        profile: Mapping[str, Any],
        stages: Sequence[StageResult],
        final_audit: Mapping[str, Any],
    ) -> dict[str, Any]:
        del checkout, profile, stages, final_audit
        summary = {
            "separate_failure_classes": {
                "resource_boundary_or_oom": ["stage0_topology"]
            }
        }
        (root / "ACCEPTANCE_SUMMARY.json").write_text(
            json.dumps(summary), encoding="utf-8"
        )
        return summary

    summary = shared_host_report(
        original_report,
        tmp_path,
        {},
        {},
        results,
        {},
    )
    assert summary["separate_failure_classes"]["resource_boundary_or_oom"] == []
    assert summary["separate_failure_classes"]["oom_failure"] == []
