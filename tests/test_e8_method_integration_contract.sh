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

import inspect
import json
import tempfile
import threading
import time
from pathlib import Path

import yaml

from drpo import e8_multitask_exp_tuning as exp_tuning
from drpo.e8_multitask_orchestration import (
    SchedulerCallbacks,
    execution_geometry,
    nominal_batches,
    plan_rows,
    run_dynamic_queue,
)
from drpo.e8_multitask_runtime import MethodAuditResult, recovery_identity

repo_root = Path(__import__("os").environ["PYTHONPATH"].split(":", 1)[0]).parent
for relative in (
    "src/drpo/e8_multitask_orchestration.py",
    "src/drpo/e8_multitask_results.py",
    "src/drpo/e8_multitask_runtime.py",
):
    source = (repo_root / relative).read_text(encoding="utf-8").lower()
    for forbidden in ("asymre", "topr", "canonical_dpo"):
        assert forbidden not in source, (relative, forbidden)

dummy_name = "dummy_contract_method"

def build_dummy(*, task, method, seed, stage, value, lambda_only, dpo_initialization):
    del lambda_only, dpo_initialization
    return exp_tuning.Cell(
        task=task,
        method=method,
        rho=None,
        seed=seed,
        stage=stage,
        method_parameters={"temperature": float(value)},
    )

def parameters(cell):
    return dict(cell.method_parameters or {})

def compatibility(cell):
    return {
        "delta_v": None,
        "beta": None,
        "dpo_initialization": None,
        "rho": None,
        "lambda": None,
        "temperature": parameters(cell)["temperature"],
    }

def cell_key(cell):
    tag = f"{parameters(cell)['temperature']:.3f}".replace(".", "p")
    return f"{cell.task}__{cell.method}_{tag}__seed{cell.seed}"

spec = exp_tuning.MethodSpec(
    name=dummy_name,
    build_cell=build_dummy,
    cell_key=cell_key,
    parameters=parameters,
    compatibility_columns=compatibility,
    cell_initialization=lambda config: None,
    initialization_identity=lambda config: {},
    train_cold=lambda cell, **kwargs: {"complete": True},
    liveness_task=lambda config: "countdown",
    liveness_runner=lambda **kwargs: {"complete": True},
    canonical_liveness_grid=None,
    canonical_liveness_parameter="representative_temperature",
    paper_grid_paths=None,
    paper_cell_parameters=None,
    paper_formula="dummy_contract_only",
    audit_record=lambda cell, record: MethodAuditResult(
        True, {"temperature": parameters(cell)["temperature"]}
    ),
    audit_failure_bucket="terminal_contract_failures",
    scientific_kernel="dummy_contract_only",
    single_aggregate_metadata=lambda config: {},
    matrix_aggregate_metadata=lambda config: {},
)
exp_tuning._register_method_spec(spec)
try:
    cells = tuple(
        exp_tuning._coldstart_method_cell(
            task,
            dummy_name,
            seed,
            "task_transfer",
            value,
            lambda_only=False,
        )
        for seed in (4000, 5000)
        for task in ("a", "b")
        for value in (0.1, 0.2)
    )
    batches = nominal_batches(
        cells,
        slot_count=4,
        seed_barrier=True,
        seed_order=(4000, 5000),
    )
    rows = plan_rows(
        batches,
        gpu_ids=(0, 1),
        project_cell=lambda cell: exp_tuning._method_spec(
            cell.method
        ).compatibility_columns(cell),
    )
    assert len(rows) == 8
    assert {row["temperature"] for row in rows} == {0.1, 0.2}
    geometry = execution_geometry(
        cells,
        gpu_ids=(0, 1),
        slots_per_gpu=2,
        max_concurrent_cells=4,
        seed_barrier=True,
        seed_order=(4000, 5000),
    )
    assert geometry.expected_cells_by_seed == {4000: 4, 5000: 4}

    lock = threading.Lock()
    state = {"seed4000_hooks": 0, "released_early": False}

    def run_cell(cell, slot, gpu_id):
        del slot, gpu_id
        with lock:
            if cell.seed == 5000 and state["seed4000_hooks"] < 4:
                state["released_early"] = True
        time.sleep(0.001)
        return {"returncode": 0}

    def after_success(cell, row):
        del row
        if cell.seed == 4000:
            with lock:
                state["seed4000_hooks"] += 1

    run = run_dynamic_queue(
        cells,
        gpu_ids=(0, 1),
        slots_per_gpu=2,
        max_concurrent_cells=4,
        seed_barrier=True,
        seed_order=(4000, 5000),
        callbacks=SchedulerCallbacks(
            run_cell=run_cell,
            after_success=after_success,
        ),
    )
    assert len(run) == 8
    assert not state["released_early"]
    assert state["seed4000_hooks"] == 4

    aggregate_rows = [
        {
            "cell_key": cell.key,
            "task": cell.task,
            "method": cell.method,
            "seed": cell.seed,
            "late_window_pass8_mean": parameters(cell)["temperature"],
            "late_window_greedy_mean": 0.0,
            "terminal_pass8": parameters(cell)["temperature"],
            "terminal_greedy_valid_rate": 1.0,
            "nan_inf_failure": False,
        }
        for cell in cells
    ]
    curve = exp_tuning._coldstart_method_grouped_curve(
        task="a",
        method=dummy_name,
        method_rows=[row for row in aggregate_rows if row["task"] == "a"],
        cells_by_key={cell.key: cell for cell in cells},
    )
    assert [row["temperature"] for row in curve] == [0.1, 0.2]

    identity = recovery_identity(
        cells[0],
        experiment_id="DEV-DUMMY-METHOD-SPEC",
        config_hash="cfg",
        cell_identity_fields=compatibility(cells[0]),
        common_identity_fields={"bank_hash": "bank"},
    )
    assert len(identity["identity_hash"]) == 64

    with tempfile.TemporaryDirectory() as temporary:
        grid_path = Path(temporary) / "liveness.yaml"
        grid_path.write_text(
            yaml.safe_dump(
                {
                    "execution": {
                        "liveness": {
                            "representative_family": dummy_name,
                            "representative_temperature": 0.1,
                        }
                    },
                    "sweep": {"seed_offsets": [4000]},
                }
            ),
            encoding="utf-8",
        )
        live_cell = exp_tuning._canonical_cold_liveness_cell(grid_path)
        assert parameters(live_cell) == {"temperature": 0.1}

    assert spec.audit_record(cells[0], {}).passed
    audit_source = inspect.getsource(exp_tuning.cmd_audit)
    assert "_method_spec(cell.method).audit_record" in audit_source
    matrix_source = inspect.getsource(
        exp_tuning._aggregate_coldstart_matrix_unranked
    )
    for forbidden in ("METHOD_ASYMRE", "METHOD_TOPR", "METHOD_DPO"):
        assert forbidden not in matrix_source
    method_spec_fields = set(exp_tuning.MethodSpec.__dataclass_fields__)
    assert not {
        "group_projection",
        "group_order",
        "plot_columns",
        "plot_projection",
    }.intersection(method_spec_fields)
finally:
    removed = exp_tuning._METHOD_SPECS.pop(dummy_name)
    assert removed is spec

print(
    json.dumps(
        {
            "dummy_cells": len(cells),
            "generic_liveness_cell": "pass",
            "matrix_parameter_extensibility": "pass",
            "method_name_leakage": "pass",
            "methodspec_surface_reduced": "pass",
            "production_audit_hook": "pass",
            "production_grouping": "pass",
            "recovery_identity": "pass",
            "seed_barrier": "pass",
        },
        sort_keys=True,
    )
)
PY
