from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from drpo.runtime_resource_admission import revalidate_with_safe_downshift
from drpo.runtime_resource_autotune import RuntimeResourceError


def _write_selection(
    work: Path,
    *,
    workers: int,
    digest: str = "selection-digest",
    reserved_cpu: float = 1.25,
    reserved_memory: int = 100,
) -> dict[str, Any]:
    work.mkdir(parents=True, exist_ok=True)
    document = {
        "selection_digest": digest,
        "selection": {
            "selected_workers": workers,
            "per_worker_reserved_cpu_cores": reserved_cpu,
            "per_worker_reserved_bytes": reserved_memory,
        },
    }
    (work / "RUNTIME_SELECTION.json").write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return document


def _record(
    work: Path,
    *,
    failures: list[str],
    affinity_budget: float,
    system_busy: float,
    usable_memory: int,
) -> Path:
    attempt = work / "_runtime_resource_attempts" / "attempt-test"
    attempt.mkdir(parents=True, exist_ok=False)
    path = attempt / "RUNTIME_REVALIDATION.json"
    path.write_text(
        json.dumps(
            {
                "attempt_id": "attempt-test",
                "decision": "BLOCK" if failures else "ALLOW",
                "failures": failures,
                "cpu_revalidation": {
                    "affinity_budget_cores": affinity_budget,
                    "conservative_system_busy_cores": system_busy,
                    "quota_domains": [],
                },
                "memory_revalidation": {
                    "usable_memory_bytes": usable_memory,
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_capacity_block_downshifts_before_launch(tmp_path: Path) -> None:
    work = tmp_path / "work"
    immutable = _write_selection(
        work,
        workers=123,
        reserved_cpu=152.75519936932926 / 123.0,
        reserved_memory=997_387_469,
    )
    original_bytes = (work / "RUNTIME_SELECTION.json").read_bytes()

    def blocked(**_: Any) -> dict[str, Any]:
        _record(
            work,
            failures=["cpu_capacity_changed"],
            affinity_budget=163.2,
            system_busy=21.776100628930816,
            usable_memory=1_156_020_868_710,
        )
        raise RuntimeResourceError(
            "RUNTIME_CAPACITY_CHANGED_REPLAN_REQUIRED: cpu_capacity_changed"
        )

    result = revalidate_with_safe_downshift(
        revalidate_runtime=blocked,
        work_dir=work,
        proposed_workers=123,
        selection_digest="selection-digest",
        revalidate_kwargs={},
    )

    admission = result["runtime_admission"]
    assert admission["decision"] == "ALLOW"
    assert admission["proposed_workers"] == 123
    assert admission["admitted_workers"] == 113
    assert admission["downshifted"] is True
    assert admission["capacity"]["cpu_worker_limit"] == 113
    assert result["revalidation"]["decision"] == "ALLOW_WITH_DOWNSHIFT"
    assert (work / "RUNTIME_SELECTION.json").read_bytes() == original_bytes
    assert json.loads((work / "RUNTIME_SELECTION.json").read_text()) == immutable
    admission_path = Path(admission["path"])
    assert admission_path.is_file()
    persisted = json.loads(admission_path.read_text())
    assert persisted["admitted_workers"] == 113
    assert persisted["scientific_matrix_changed"] is False
    assert persisted["running_workers_resized"] is False


def test_safe_plan_is_admitted_without_downshift(tmp_path: Path) -> None:
    work = tmp_path / "work"
    document = _write_selection(work, workers=4, reserved_cpu=1.0, reserved_memory=100)

    def allowed(**_: Any) -> dict[str, Any]:
        path = _record(
            work,
            failures=[],
            affinity_budget=8.0,
            system_busy=1.0,
            usable_memory=10_000,
        )
        return {
            **document,
            "revalidation": {
                "decision": "ALLOW",
                "attempt_id": "attempt-test",
                "path": str(path),
            },
        }

    result = revalidate_with_safe_downshift(
        revalidate_runtime=allowed,
        work_dir=work,
        proposed_workers=4,
        selection_digest="selection-digest",
        revalidate_kwargs={},
    )

    assert result["runtime_admission"]["admitted_workers"] == 4
    assert result["runtime_admission"]["downshifted"] is False
    assert result["revalidation"]["decision"] == "ALLOW"


def test_zero_safe_capacity_remains_blocked(tmp_path: Path) -> None:
    work = tmp_path / "work"
    _write_selection(work, workers=4, reserved_cpu=2.0, reserved_memory=100)

    def blocked(**_: Any) -> dict[str, Any]:
        _record(
            work,
            failures=["cpu_capacity_changed"],
            affinity_budget=8.0,
            system_busy=8.0,
            usable_memory=10_000,
        )
        raise RuntimeResourceError(
            "RUNTIME_CAPACITY_CHANGED_REPLAN_REQUIRED: cpu_capacity_changed"
        )

    with pytest.raises(RuntimeResourceError, match="RUNTIME_CAPACITY_CHANGED"):
        revalidate_with_safe_downshift(
            revalidate_runtime=blocked,
            work_dir=work,
            proposed_workers=4,
            selection_digest="selection-digest",
            revalidate_kwargs={},
        )

    admission = json.loads(
        (
            work
            / "_runtime_resource_attempts"
            / "attempt-test"
            / "RUNTIME_ADMISSION.json"
        ).read_text()
    )
    assert admission["decision"] == "BLOCK"
    assert admission["admitted_workers"] == 0


def test_zero_capacity_can_return_structured_wait_evidence(tmp_path: Path) -> None:
    work = tmp_path / "work"
    _write_selection(work, workers=4, reserved_cpu=2.0, reserved_memory=100)

    def blocked(**_: Any) -> dict[str, Any]:
        _record(
            work,
            failures=["cpu_capacity_changed"],
            affinity_budget=8.0,
            system_busy=8.0,
            usable_memory=10_000,
        )
        raise RuntimeResourceError(
            "RUNTIME_CAPACITY_CHANGED_REPLAN_REQUIRED: cpu_capacity_changed"
        )

    result = revalidate_with_safe_downshift(
        revalidate_runtime=blocked,
        work_dir=work,
        proposed_workers=4,
        selection_digest="selection-digest",
        revalidate_kwargs={},
        allow_zero=True,
    )

    admission = result["runtime_admission"]
    assert admission["decision"] == "BLOCK"
    assert admission["admitted_workers"] == 0
    assert admission["reason"] == "no_safe_worker_capacity"
    assert result["revalidation"]["decision"] == "BLOCK"
    assert Path(admission["path"]).is_file()
    assert result["selection_digest"] == "selection-digest"


def test_non_capacity_failure_is_never_downshifted(tmp_path: Path) -> None:
    work = tmp_path / "work"
    _write_selection(work, workers=4)

    def blocked(**_: Any) -> dict[str, Any]:
        _record(
            work,
            failures=["source_commit_mismatch"],
            affinity_budget=8.0,
            system_busy=1.0,
            usable_memory=10_000,
        )
        raise RuntimeResourceError(
            "RUNTIME_CAPACITY_CHANGED_REPLAN_REQUIRED: source_commit_mismatch"
        )

    with pytest.raises(RuntimeResourceError, match="source_commit_mismatch"):
        revalidate_with_safe_downshift(
            revalidate_runtime=blocked,
            work_dir=work,
            proposed_workers=4,
            selection_digest="selection-digest",
            revalidate_kwargs={},
            allow_zero=True,
        )
    assert not (
        work
        / "_runtime_resource_attempts"
        / "attempt-test"
        / "RUNTIME_ADMISSION.json"
    ).exists()
