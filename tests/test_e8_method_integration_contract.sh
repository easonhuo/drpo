#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
python3 -m py_compile \
  "${ROOT_DIR}/src/drpo/e8_multitask_orchestration.py" \
  "${ROOT_DIR}/src/drpo/e8_multitask_results.py" \
  "${ROOT_DIR}/src/drpo/e8_multitask_runtime.py"
python3 - <<'PY'
from __future__ import annotations

import json
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from drpo.e8_multitask_orchestration import (
    SchedulerCallbacks,
    execution_geometry,
    nominal_batches,
    run_dynamic_queue,
)
from drpo.e8_multitask_results import (
    MethodResultProjection,
    common_result_row,
    grouped_curve,
)
from drpo.e8_multitask_runtime import (
    MethodAuditResult,
    audit_terminal_cells,
    file_inventory,
    recovery_identity,
)


@dataclass(frozen=True)
class DummyCell:
    task: str
    method: str
    seed: int
    stage: str
    parameter: float

    @property
    def key(self) -> str:
        return f"{self.task}__{self.method}_{self.parameter}__seed{self.seed}"


cells = tuple(
    DummyCell(task, "dummy_future_method", seed, "transfer", parameter)
    for seed in (4000, 5000)
    for task in ("a", "b")
    for parameter in (0.1, 0.2)
)
geometry = execution_geometry(
    cells,
    gpu_ids=(0, 1),
    slots_per_gpu=2,
    max_concurrent_cells=4,
    seed_barrier=True,
    seed_order=(4000, 5000),
)
assert geometry.cell_count == 8
assert geometry.seed_order == (4000, 5000)
assert geometry.expected_cells_by_seed == {4000: 4, 5000: 4}
assert geometry.nominal_batch_count == 2
assert [
    len(batch)
    for batch in nominal_batches(
        cells,
        slot_count=4,
        seed_barrier=True,
        seed_order=(4000, 5000),
    )
] == [4, 4]

# Mirror the frozen baseline-matrix scheduling geometry without importing any
# scientific method implementation: 88 cells per reviewed seed, 16 slots.
formal_shape_cells = tuple(
    DummyCell(
        task=f"task-{index % 8}",
        method="opaque_method",
        seed=seed,
        stage="task_transfer",
        parameter=float(index),
    )
    for seed in (4000, 5000)
    for index in range(88)
)
formal_batches = nominal_batches(
    formal_shape_cells,
    slot_count=16,
    seed_barrier=True,
    seed_order=(4000, 5000),
)
assert [len(batch) for batch in formal_batches] == [16, 16, 16, 16, 16, 8] * 2

started: list[tuple[str, int, int, int, float]] = []
lock = threading.Lock()
seed4000_success_hooks = 0
seed5000_started_before_release = False


def run_cell(cell: DummyCell, slot: int, gpu_id: int) -> dict[str, object]:
    global seed5000_started_before_release
    with lock:
        if cell.seed == 5000 and seed4000_success_hooks < 4:
            seed5000_started_before_release = True
        started.append((cell.key, cell.seed, slot, gpu_id, time.monotonic()))
    time.sleep(0.002)
    return {"returncode": 0, "payload": cell.parameter}


def after_success(cell: DummyCell, row: dict[str, object]) -> None:
    global seed4000_success_hooks
    del row
    if cell.seed == 4000:
        time.sleep(0.001)
        with lock:
            seed4000_success_hooks += 1


results = run_dynamic_queue(
    cells,
    gpu_ids=(0, 1),
    slots_per_gpu=2,
    max_concurrent_cells=4,
    seed_barrier=True,
    seed_order=(4000, 5000),
    callbacks=SchedulerCallbacks(run_cell=run_cell, after_success=after_success),
)
assert len(results) == 8
assert not seed5000_started_before_release
assert seed4000_success_hooks == 4
assert [row["cell_key"] for row in results] == [cell.key for cell in cells]

# The post-success hook is part of the success transaction. A publication or
# recovery failure must not release the next reviewed seed batch.
failed_starts: list[DummyCell] = []


def run_cell_fail_hook(cell: DummyCell, slot: int, gpu_id: int) -> dict[str, object]:
    del slot, gpu_id
    failed_starts.append(cell)
    return {"returncode": 0}


def fail_hook(cell: DummyCell, row: dict[str, object]) -> None:
    del row
    if cell.seed == 4000 and cell.task == "a" and cell.parameter == 0.1:
        raise RuntimeError("publish failed")


failed_results = run_dynamic_queue(
    cells,
    gpu_ids=(0, 1),
    slots_per_gpu=2,
    max_concurrent_cells=4,
    seed_barrier=True,
    seed_order=(4000, 5000),
    callbacks=SchedulerCallbacks(run_cell=run_cell_fail_hook, after_success=fail_hook),
)
assert all(cell.seed == 4000 for cell in failed_starts)
assert any(
    row["returncode"] == 1 and "after_success RuntimeError" in row.get("error", "")
    for row in failed_results
)


def project_method(cell: DummyCell) -> MethodResultProjection:
    return MethodResultProjection(
        parameters={"temperature": cell.parameter},
        compatibility_columns={"legacy_parameter": cell.parameter},
    )


rows = [
    common_result_row(
        cell,
        {"score": cell.parameter + cell.seed / 100000.0},
        source="dummy",
        project_method=project_method,
        project_metrics=lambda value: {"score": value["score"]},
    )
    for cell in cells
]
curve = grouped_curve(
    rows,
    metric_names=("score",),
    group_order=lambda task, method, params: (task, method, params["temperature"]),
)
assert len(curve) == 4
assert [row["method_parameters"]["temperature"] for row in curve[:2]] == [0.1, 0.2]
assert all(row["seed_count"] == 2 for row in curve)

identity = recovery_identity(
    cells[0],
    experiment_id="DEV-DUMMY",
    config_hash="cfg",
    method_identity_fields={"method_parameters": {"temperature": 0.1}},
    common_identity_fields={"bank_hash": "bank"},
)
assert identity["method_parameters"] == {"temperature": 0.1}
assert len(identity["identity_sha256"]) == 64

records = {
    cell.key: {
        "cell_key": cell.key,
        "method": cell.method,
        "seed": cell.seed,
        "complete": True,
        "evaluation_status": "complete",
        "terminal_step": 1200,
        "stop_reason": "max_steps",
        "nan_inf_failure": False,
    }
    for cell in cells
}


def method_audit(cell: DummyCell, record: dict[str, object]) -> MethodAuditResult:
    del record
    return MethodAuditResult(True, {"dummy_parameter": cell.parameter})


audit = audit_terminal_cells(
    cells,
    records,
    expected_terminal_step=1200,
    expected_stop_reason="max_steps",
    method_audit=method_audit,
)
assert audit["passed"]
assert audit["expected_cell_count"] == 8

with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    first = root / "a.txt"
    second = root / "nested" / "b.txt"
    second.parent.mkdir()
    first.write_text("a\n", encoding="utf-8")
    second.write_text("b\n", encoding="utf-8")
    inventory = file_inventory(root, (second, first))
    assert [row["path"] for row in inventory] == ["a.txt", "nested/b.txt"]
    assert all(len(row["sha256"]) == 64 for row in inventory)

# Architecture acceptance: generic production modules must not know current
# scientific method names. This is a maintenance-contract check, not a new
# experiment acceptance gate.
repo_root = Path(__import__("os").environ["PYTHONPATH"].split(":", 1)[0]).parent
for relative in (
    "src/drpo/e8_multitask_orchestration.py",
    "src/drpo/e8_multitask_results.py",
    "src/drpo/e8_multitask_runtime.py",
):
    text = (repo_root / relative).read_text(encoding="utf-8").lower()
    for forbidden in ("asymre", "topr", "canonical_dpo"):
        assert forbidden not in text, (relative, forbidden)

print(
    json.dumps(
        {
            "compile": "pass",
            "dummy_cells": len(cells),
            "formal_nominal_batches": [len(batch) for batch in formal_batches],
            "method_name_leakage": "pass",
            "package_inventory": "pass",
            "recovery_identity": "pass",
            "result_projection": "pass",
            "seed_barrier": "pass",
            "success_hook_barrier": "pass",
            "terminal_audit": "pass",
        },
        sort_keys=True,
    )
)
PY
