from __future__ import annotations

import json
from pathlib import Path

from drpo import runtime_resource_acceptance as acceptance
from drpo import runtime_resource_acceptance_capacity as capacity


def _failure(name: str) -> acceptance.StageResult:
    now = acceptance.utc_now()
    return acceptance.StageResult(
        name=name,
        status="FAIL",
        started_utc=now,
        finished_utc=now,
        details={"error_type": "AcceptanceError", "error": "delegated stage failed"},
    )


def test_e7_pure_capacity_change_is_blocked(tmp_path: Path) -> None:
    attempt = (
        tmp_path
        / "stage2_e7_cpu_v2"
        / "work"
        / "_runtime_resource_attempts"
        / "attempt-1"
        / "RUNTIME_REVALIDATION.json"
    )
    attempt.parent.mkdir(parents=True)
    attempt.write_text(
        json.dumps(
            {
                "decision": "BLOCK",
                "failures": ["cpu_capacity_changed"],
                "selected_workers": 60,
                "cpu_revalidation": {
                    "conservative_system_busy_cores": 190.0,
                    "planned_worker_cpu_cores": 74.0,
                    "affinity_budget_cores": 163.2,
                    "affinity_projected_total_busy_cores": 264.0,
                    "ok": False,
                },
                "memory_revalidation": {"ok": True},
            }
        ),
        encoding="utf-8",
    )

    result = capacity.normalize_capacity_block(
        tmp_path, _failure("stage2_e7_cpu_v2")
    )

    assert result.status == "BLOCKED"
    assert result.details["classification"] == "safe_capacity_unavailable"
    assert result.details["capacity_failures"] == ["cpu_capacity_changed"]
    written = json.loads(
        (tmp_path / "stage2_e7_cpu_v2" / "STAGE_RESULT.json").read_text()
    )
    assert written["status"] == "BLOCKED"


def test_gpu_safe_capacity_error_is_blocked(tmp_path: Path) -> None:
    log = tmp_path / "stage3_gpu_placement" / "selection.log"
    log.parent.mkdir(parents=True)
    log.write_text(
        "RuntimeResourceError: measured system CPU occupancy leaves no safe "
        "capacity for one GPU worker\n",
        encoding="utf-8",
    )

    result = capacity.normalize_capacity_block(
        tmp_path, _failure("stage3_gpu_placement")
    )

    assert result.status == "BLOCKED"
    assert result.details["capacity_source"] == "gpu_placement_cpu_or_memory_envelope"


def test_generic_failure_remains_fail(tmp_path: Path) -> None:
    log = tmp_path / "stage3_gpu_placement" / "selection.log"
    log.parent.mkdir(parents=True)
    log.write_text("RuntimeError: malformed selection document\n", encoding="utf-8")

    original = _failure("stage3_gpu_placement")
    result = capacity.normalize_capacity_block(tmp_path, original)

    assert result is original
    assert result.status == "FAIL"
