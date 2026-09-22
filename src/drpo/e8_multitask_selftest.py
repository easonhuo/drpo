"""Non-scientific engineering self-test harness for E8 multitask execution.

Production/scientific helpers are supplied explicitly by the composition root.
This module never imports e8_multitask_exp_tuning, preventing an import cycle.
"""

from __future__ import annotations

import copy
import json
import os
import shutil
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from drpo.e8_multitask_p0 import bank_path, with_smoke_overrides


@dataclass(frozen=True)
class SelfTestBindings:
    """Call-scoped dependencies supplied by the E8 composition root."""

    host: Any


_INTENTIONAL_FAILURE_RETURNCODE = 73


def _engineering_self_test_config(
    config: Mapping[str, Any],
    *,
    bindings: SelfTestBindings,
) -> dict[str, Any]:
    host = bindings.host
    updated = copy.deepcopy(dict(config))
    split_overrides = {
        "p0_train_rows": 2,
        "p0_validation_rows": 1,
        "p0_test_rows": 1,
    }
    if "countdown" in set(config["suite"]["tasks"]):
        split_overrides.update(
            {
                "countdown_train_rows": 2,
                "countdown_validation_rows": 1,
            }
        )
    updated["split"].update(split_overrides)
    updated["engineering_self_test"] = {
        "placeholder_backend": True,
        "scientific_evidence_allowed": False,
        "purpose": "non_gpu_end_to_end_delivery_acceptance",
    }
    host.validate_config(updated)
    return updated


def _write_engineering_input_fixtures(
    config: Mapping[str, Any],
    output_root: Path,
    *,
    bindings: SelfTestBindings,
) -> tuple[Path, Path, Path | None, Path | None]:
    host = bindings.host
    fixture_root = output_root / "engineering_fixtures"
    p0_work_dir = fixture_root / "p0"
    sources_root = p0_work_dir / "sources"
    sources_root.mkdir(parents=True, exist_ok=True)
    p0_config_path = host._repo_root() / "configs" / "e8_multitask_p0.yaml"
    p0_config = yaml.safe_load(p0_config_path.read_text(encoding="utf-8"))
    if not isinstance(p0_config, dict):
        raise TypeError("P0 configuration root must be a mapping")
    row_count = (
        int(config["split"]["p0_train_rows"])
        + int(config["split"]["p0_validation_rows"])
        + int(config["split"]["p0_test_rows"])
    )
    task_qualification: dict[str, Any] = {}
    for task_value in config["suite"]["p0_tasks"]:
        task = str(task_value)
        rows: list[dict[str, Any]] = []
        for row_index in range(row_count):
            rows.append(
                {
                    "schema_version": 1,
                    "task": task,
                    "prompt_id": f"{task}-placeholder-{row_index:03d}",
                    "prompt": f"[{task}] engineering prompt {row_index}",
                    "oracle_completion": f"{task}-oracle-{row_index}",
                    "metadata": {"engineering_placeholder": True},
                    "generation_seed": 2026080901,
                    "negatives": [
                        {
                            "negative_id": f"{task}-{row_index:03d}-neg-{negative:02d}",
                            "completion": f"{task}-wrong-{row_index}-{negative}",
                            "format_valid": True,
                            "binary_correct": False,
                            "error_class": "engineering_placeholder_wrong",
                        }
                        for negative in range(16)
                    ],
                }
            )
        host.atomic_jsonl(bank_path(p0_work_dir, task), rows)
        task_qualification[task] = {"passed": True, "engineering_placeholder": True}
    qualification = {
        "schema_version": 1,
        "experiment_id": host.P0_EXPERIMENT_ID,
        "config_hash": host.stable_config_hash(
            with_smoke_overrides(p0_config, rows=None, negatives=None)
        ),
        "tasks": task_qualification,
        "passed": True,
        "scientific_status": "not_run",
        "engineering_placeholder_backend": True,
    }
    host.atomic_json(p0_work_dir / "qualification_audit.json", qualification)

    countdown_bank: Path | None = None
    countdown_validation: Path | None = None
    if "countdown" in set(config["suite"]["tasks"]):
        countdown_bank = fixture_root / "countdown" / "offline_bank_v2.jsonl"
        countdown_rows = []
        for row_index in range(int(config["split"]["countdown_train_rows"])):
            countdown_rows.append(
                {
                    "row_id": f"countdown-placeholder-train-{row_index:03d}",
                    "source_prompt_id": f"countdown-placeholder-source-{row_index:03d}",
                    "prompt": f"Use 1 and 2 to make {3 + row_index}",
                    "oracle_positive": "1 + 2",
                    "numbers": [1, 2],
                    "target": 3 + row_index,
                    "negative_bank": [
                        {
                            "expression": f"1 - 2 + {negative}",
                            "valid_format": True,
                            "correct": False,
                            "source": "engineering_placeholder_wrong",
                        }
                        for negative in range(16)
                    ],
                }
            )
        host.atomic_jsonl(countdown_bank, countdown_rows)
        countdown_validation = fixture_root / "countdown" / "val.jsonl"
        host.atomic_jsonl(
            countdown_validation,
            [
                {
                    "id": "countdown-placeholder-validation-000",
                    "prompt": "Use 2 and 2 to make 4",
                    "oracle": "2 + 2",
                    "numbers": [2, 2],
                    "target": 4,
                }
            ],
        )
    return p0_work_dir, p0_config_path, countdown_bank, countdown_validation


def _write_engineering_gates(
    config: Mapping[str, Any],
    output_root: Path,
    *,
    base_model_path: str,
    bindings: SelfTestBindings,
) -> None:
    host = bindings.host
    splits, _ = host._load_ready_inputs(output_root, config, base_model_path=base_model_path)
    calibration_tasks: dict[str, Any] = {}
    for task_value in config["suite"]["tasks"]:
        task = str(task_value)
        identity = host._canonical_calibration_identity(
            task,
            split_manifest=splits,
            base_model_path=base_model_path,
            config=config,
        )
        result = {
            **identity,
            **host.experiment_config.coldstart_remoteness_metadata(config),
            "complete": True,
            "scientific_status": "not_run",
            "engineering_placeholder_backend": True,
        }
        host.atomic_json(output_root / "calibration" / f"{task}.json", result)
        calibration_tasks[task] = result
        log_path = output_root / "logs" / "calibration" / f"{task}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("engineering placeholder calibration complete\n", encoding="utf-8")
    host.atomic_json(
        output_root / "calibration" / "calibration_manifest.json",
        {
            "schema_version": 1,
            "experiment_id": host.experiment_id(config),
            "config_hash": host.stable_config_hash(config),
            "requested_tasks": list(config["suite"]["tasks"]),
            "tasks": calibration_tasks,
            "complete": True,
            "scientific_status": "not_run",
            "engineering_placeholder_backend": True,
        },
    )
    base_identity = host.model_identity(base_model_path, None)["model"]
    methods = host._coldstart_methods(config) if host._is_coldstart(config) else (host._coldstart_method(config),)
    for method in methods:
        task = (
            str(config["dpo"]["liveness_task"])
            if method == host.METHOD_DPO
            else "countdown"
        )
        liveness_key = f"{task}__{method}__engineering_placeholder_liveness"
        liveness = {
            "schema_version": 1,
            "experiment_id": host.experiment_id(config),
            "config_hash": host.stable_config_hash(config),
            "base_model_identity": base_identity,
            "engineering_liveness": True,
            "engineering_placeholder_backend": True,
            "optimizer_updates": 2,
            "complete": True,
            "reload_gate_passed": True,
            "adapter_weight_changed": True,
            "fresh_process_reload_passed": True,
            "liveness_parent_process_id": 1001,
            "reload_process_id": 1002,
            "nan_inf_failure": False,
            "canonical_dispatch_verified": True,
            "finite_old_core_updates": True,
            "optimizer_update_norm": 1.0,
            "initial_adapter_weight_sha256": "0" * 64,
            "terminal_adapter_weight_sha256": "1" * 64,
            "cell": {
                "task": task,
                "method": method,
            },
            "scientific_status": "not_run",
        }
        host.atomic_json(output_root / "liveness" / liveness_key / "cell_manifest.json", liveness)
    (output_root / "logs" / "liveness.log").write_text(
        "engineering placeholder liveness complete\n",
        encoding="utf-8",
    )


def _audit_engineering_queue(
    config: Mapping[str, Any],
    output_root: Path,
    scheduler: Mapping[str, Any],
    *,
    bindings: SelfTestBindings,
) -> dict[str, Any]:
    host = bindings.host
    cells = host.build_cells(config)
    events = [
        json.loads(line)
        for line in (output_root / "scheduler" / "queue_events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    events = [row for row in events if row["scheduler_run_id"] == scheduler["scheduler_run_id"]]
    active_by_gpu = {int(gpu): 0 for gpu in config["execution"]["gpu_ids"]}
    maximum_by_gpu = dict(active_by_gpu)
    starts: dict[str, float] = {}
    finishes: dict[str, float] = {}
    for event in events:
        gpu_id = int(event["gpu_id"])
        cell_key = str(event["cell_key"])
        if event["event"] == "start":
            active_by_gpu[gpu_id] += 1
            maximum_by_gpu[gpu_id] = max(maximum_by_gpu[gpu_id], active_by_gpu[gpu_id])
            starts[cell_key] = float(event["unix_time"])
        elif event["event"] == "finish":
            active_by_gpu[gpu_id] -= 1
            finishes[cell_key] = float(event["unix_time"])
    expected_keys = {cell.key for cell in cells}
    if set(starts) != expected_keys or set(finishes) != expected_keys:
        raise RuntimeError(f"Engineering queue did not observe all {len(cells)} starts/finishes")
    slots_per_gpu = int(config["execution"]["slots_per_gpu"])
    if (
        any(value != 0 for value in active_by_gpu.values())
        or max(maximum_by_gpu.values()) > slots_per_gpu
    ):
        raise RuntimeError("Engineering queue exceeded the declared per-GPU capacity")
    slot_count = int(config["execution"]["max_concurrent_cells"])
    initial_keys = {cell.key for cell in cells[:slot_count]}
    replacement_keys = {cell.key for cell in cells[slot_count:]}
    if not replacement_keys:
        raise RuntimeError("Engineering queue requires replacement cells to audit dynamic refill")
    if min(starts[key] for key in replacement_keys) >= max(finishes[key] for key in initial_keys):
        raise RuntimeError(
            f"Engineering queue did not refill before the initial {slot_count} cells finished"
        )
    return {
        "all_cells_observed": True,
        "maximum_active_by_gpu": maximum_by_gpu,
        "dynamic_refill_observed": True,
        "nominal_batch_barrier_absent": True,
        "nominal_batch_count": len(host.build_waves(config)),
        "slots_per_gpu": slots_per_gpu,
    }


def _intentional_failure_evidence(
    scheduler: Mapping[str, Any],
    *,
    expected_cell_key: str,
) -> dict[str, Any]:
    failed_cells = [str(value) for value in scheduler.get("failed_cells", ())]
    if failed_cells != [expected_cell_key]:
        raise RuntimeError(
            "Engineering failure injection hit unexpected cells: "
            f"expected={[expected_cell_key]} actual={failed_cells}"
        )
    results = scheduler.get("results")
    if not isinstance(results, list):
        raise TypeError("Engineering failure scheduler results are missing")
    matching = [
        row
        for row in results
        if isinstance(row, Mapping) and str(row.get("cell_key")) == expected_cell_key
    ]
    if len(matching) != 1:
        raise RuntimeError(
            "Engineering failure injection did not produce exactly one result for "
            f"{expected_cell_key}"
        )
    try:
        returncode = int(matching[0]["returncode"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("Engineering failure injection return code is invalid") from exc
    if returncode != _INTENTIONAL_FAILURE_RETURNCODE:
        raise RuntimeError(
            "Engineering failure injection returned unexpected code: "
            f"expected={_INTENTIONAL_FAILURE_RETURNCODE} actual={returncode}"
        )
    return {"cell_key": expected_cell_key, "returncode": returncode}


def cmd_engineering_self_test(
    config: Mapping[str, Any],
    output_root: Path,
    *,
    source_commit: str,
    bindings: SelfTestBindings,
) -> dict[str, Any]:
    """Exercise the delivery pipeline with an isolated, non-scientific backend."""

    host = bindings.host
    if not host._is_coldstart(config):
        raise RuntimeError("Engineering self-test is available only for the cold-start profile")
    if len(source_commit) != 40 or any(char not in "0123456789abcdef" for char in source_commit):
        raise ValueError("Engineering self-test requires one full lowercase source commit")
    output_root = host.validate_work_dir(output_root)
    fresh_run = not (output_root / "prepare_manifest.json").is_file()
    self_test_config = _engineering_self_test_config(config, bindings=bindings)
    config_path = output_root / "engineering_self_test_config.yaml"
    base_model = output_root / "engineering_fixtures" / "placeholder_model"
    if fresh_run:
        config_path.write_text(yaml.safe_dump(self_test_config, sort_keys=False), encoding="utf-8")
        p0_work_dir, p0_config_path, countdown_bank, countdown_validation = (
            _write_engineering_input_fixtures(self_test_config, output_root, bindings=bindings)
        )
        host.cmd_prepare(
            self_test_config,
            output_root,
            p0_work_dir=p0_work_dir,
            p0_config=p0_config_path,
            countdown_bank=countdown_bank,
            countdown_validation=countdown_validation,
            countdown_adapter=None,
        )
        base_model.mkdir(parents=True, exist_ok=True)
        host.atomic_json(base_model / "config.json", {"engineering_placeholder_backend": True})
        host.atomic_json(
            output_root / "source_provenance.json",
            {
                "schema_version": 1,
                "run_id": output_root.name,
                "source_commit": source_commit,
                "model_repo": "engineering-placeholder-no-model-loaded",
                "model_revision": "not_applicable",
                "model_path": str(base_model.resolve()),
                "test_partition_accessed": False,
                "engineering_placeholder_backend": True,
            },
        )
        _write_engineering_gates(
            self_test_config,
            output_root,
            base_model_path=str(base_model),
            bindings=bindings,
        )
    else:
        recovered_config = host.load_config(config_path)
        if recovered_config != self_test_config:
            raise RuntimeError("Engineering recovery config differs from the reviewed config")
        provenance = host._read_json_object(output_root / "source_provenance.json")
        if provenance.get("source_commit") != source_commit:
            raise RuntimeError("Engineering recovery source commit mismatch")
        host._load_prepared(output_root, self_test_config)
        _write_engineering_gates(
            self_test_config,
            output_root,
            base_model_path=str(base_model),
            bindings=bindings,
        )
        host._require_calibration_gate(
            self_test_config,
            output_root,
            base_model_path=str(base_model),
        )
        host._require_liveness_gate(
            self_test_config,
            output_root,
            base_model_path=str(base_model),
        )

    cells = host.build_cells(self_test_config)
    cell_index = {cell.key: index for index, cell in enumerate(cells)}
    failed_once = False
    failure_lock = threading.Lock()
    failure_evidence: dict[str, Any] | None = None

    def placeholder_cell_runner(
        *,
        config_path: Path,
        output_root: Path,
        base_model_path: str,
        cell: Any,
        gpu_id: int,
        force: bool,
    ) -> dict[str, Any]:
        del config_path, base_model_path, force
        nonlocal failed_once
        started_at = time.time()
        cell_root = output_root / "cells" / cell.key
        manifest_path = cell_root / "cell_manifest.json"
        log_path = output_root / "logs" / f"{cell.key}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if manifest_path.is_file():
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (
                existing.get("config_hash") != host.stable_config_hash(self_test_config)
                or existing.get("engineering_placeholder_backend") is not True
                or existing.get("complete") is not True
            ):
                raise RuntimeError(f"Placeholder resume identity mismatch: {cell.key}")
            if cell.key == cells[0].key:
                time.sleep(0.05)
            return {
                "cell_key": cell.key,
                "gpu_id": gpu_id,
                "returncode": 0,
                "log": str(log_path.resolve()),
                "started_unix": started_at,
                "finished_unix": time.time(),
                "reused_complete": True,
            }
        with failure_lock:
            should_fail = fresh_run and cell.key == cells[0].key and not failed_once
            if should_fail:
                failed_once = True
        if should_fail:
            cell_root.mkdir(parents=True, exist_ok=True)
            log_path.write_text("intentional engineering failure\n", encoding="utf-8")
            host.atomic_json(
                cell_root / "failure.json",
                {
                    "experiment_id": host.experiment_id(self_test_config),
                    "cell_key": cell.key,
                    "engineering_placeholder_backend": True,
                    "complete": False,
                },
            )
            return {
                "cell_key": cell.key,
                "gpu_id": gpu_id,
                "returncode": _INTENTIONAL_FAILURE_RETURNCODE,
                "log": str(log_path.resolve()),
                "started_unix": started_at,
                "finished_unix": time.time(),
            }
        if cell.key == cells[0].key:
            time.sleep(0.05)
        else:
            time.sleep(0.001)
        index = cell_index[cell.key]
        score = round(0.1 + (index % 20) * 0.01, 6)
        manifest = {
            "schema_version": 1,
            "experiment_id": host.experiment_id(self_test_config),
            "config_hash": host.stable_config_hash(self_test_config),
            "source_commit": source_commit,
            "cell_key": cell.key,
            "validation_best_pass8": score,
            "validation_terminal_pass8": max(0.0, score - 0.005),
            "validation_best_greedy": max(0.0, score - 0.02),
            "validation_terminal_greedy": max(0.0, score - 0.025),
            "validation_best_greedy_valid_rate": 1.0,
            "validation_terminal_greedy_valid_rate": 1.0,
            "best_step": 2,
            "terminal_step": 2,
            "stop_reason": "engineering_placeholder_complete",
            "nan_inf_failure": False,
            "evaluation_status": "complete",
            "complete": True,
            "scientific_status": "not_run",
            "engineering_placeholder_backend": True,
        }
        host.atomic_json(manifest_path, manifest)
        log_path.write_text("engineering placeholder cell complete\n", encoding="utf-8")
        return {
            "cell_key": cell.key,
            "gpu_id": gpu_id,
            "returncode": 0,
            "log": str(log_path.resolve()),
            "started_unix": started_at,
            "finished_unix": time.time(),
            "reused_complete": False,
        }

    original_runner = host._run_subprocess_cell
    host._run_subprocess_cell = placeholder_cell_runner
    try:
        if fresh_run:
            first_failure: dict[str, Any]
            try:
                host.cmd_run_dynamic(
                    self_test_config,
                    config_path,
                    output_root,
                    base_model_path=str(base_model),
                    force=False,
                    retry_incomplete=True,
                )
            except RuntimeError:
                first_failure = json.loads(
                    (output_root / "scheduler" / "dynamic_run.json").read_text(encoding="utf-8")
                )
            else:
                raise RuntimeError("Engineering failure injection did not fail closed")
            failure_evidence = _intentional_failure_evidence(
                first_failure,
                expected_cell_key=cells[0].key,
            )
            if not first_failure["unscheduled_cells"]:
                raise RuntimeError("Engineering failure did not preserve unscheduled work")
            resumed = host.cmd_run_dynamic(
                self_test_config,
                config_path,
                output_root,
                base_model_path=str(base_model),
                force=False,
                retry_incomplete=True,
            )
        else:
            resumed = host.cmd_run_dynamic(
                self_test_config,
                config_path,
                output_root,
                base_model_path=str(base_model),
                force=False,
                retry_incomplete=True,
            )
        queue_audit = _audit_engineering_queue(self_test_config, output_root, resumed, bindings=bindings)
        before = {
            cell.key: host.sha256_file(output_root / "cells" / cell.key / "cell_manifest.json")
            for cell in cells
        }
        repeated = host.cmd_run_dynamic(
            self_test_config,
            config_path,
            output_root,
            base_model_path=str(base_model),
            force=False,
            retry_incomplete=True,
        )
        after = {
            cell.key: host.sha256_file(output_root / "cells" / cell.key / "cell_manifest.json")
            for cell in cells
        }
        if before != after or not repeated["complete"]:
            raise RuntimeError("Engineering repeat run changed a completed cell")
    finally:
        host._run_subprocess_cell = original_runner

    failure_stage = os.environ.get("E8_COLDSTART_ENGINEERING_FAIL_STAGE", "").strip()
    if failure_stage == "after_queue":
        raise RuntimeError("Intentional engineering failure after all cells completed")
    aggregate = host.cmd_aggregate(self_test_config, output_root)
    if failure_stage == "after_aggregate":
        raise RuntimeError("Intentional engineering failure after aggregate")
    audit = host.cmd_audit(self_test_config, output_root)
    if failure_stage == "after_audit":
        raise RuntimeError("Intentional engineering failure after audit")
    finalized = host.cmd_finalize(self_test_config, output_root)
    if failure_stage == "after_finalize":
        raise RuntimeError("Intentional engineering failure after finalize")
    preliminary_package = host.cmd_package(self_test_config, output_root)
    package_manifest_path = output_root / "packages" / "package_manifest.json"
    tampered = output_root / "packages" / "tampered_self_test.zip"
    shutil.copyfile(preliminary_package["full_results_zip"], tampered)
    with tampered.open("ab") as handle:
        handle.write(b"tamper")
    tamper_rejected = False
    try:
        host.verify_result_package(package_manifest_path, zip_override=tampered)
    except RuntimeError:
        tamper_rejected = True
    finally:
        tampered.unlink(missing_ok=True)
    if not tamper_rejected:
        raise RuntimeError("Result-package verification accepted a tampered ZIP")
    report = {
        "schema_version": 1,
        "experiment_id": host.experiment_id(self_test_config),
        "source_commit": source_commit,
        "scientific_status": "not_run",
        "engineering_placeholder_backend": True,
        "prepare_complete": True,
        "canonical_source_lock_passed": True,
        "intentional_failure_cell_key": (
            failure_evidence["cell_key"] if failure_evidence is not None else None
        ),
        "intentional_failure_returncode": (
            failure_evidence["returncode"] if failure_evidence is not None else None
        ),
        "intentional_failure_observed_this_attempt": failure_evidence is not None,
        "failure_preserved_unscheduled_work": True if failure_evidence is not None else None,
        "recovered_from_previous_attempt": not fresh_run,
        "resume_completed_cells": resumed["completed_cells"],
        "repeat_run_preserved_cell_hashes": True,
        "analysis_ready_tasks": sorted(repeated.get("analysis_ready_tasks", ())),
        "task_result_count": len(repeated.get("task_results", {})),
        "queue_audit": queue_audit,
        "aggregate_cell_count": aggregate["cell_count"],
        "terminal_audit_complete": audit["all_training_and_evaluation_complete"],
        "canonical_archive_owner": finalized["canonical_archive_owner"],
        "package_reopen_verification_passed": True,
        "tampered_package_rejected": True,
        "complete": True,
        "note": "No model, GPU, optimizer, or scientific metric was executed.",
    }
    host.atomic_json(output_root / "ENGINEERING_SELF_TEST_REPORT.json", report)
    final_package = host.cmd_package(self_test_config, output_root)
    return {
        **report,
        "output_root": str(output_root.resolve()),
        "full_results_zip": final_package["full_results_zip"],
        "full_results_zip_sha256": final_package["full_results_zip_sha256"],
        "plot_curve_points_csv": final_package["plot_curve_points_csv"],
        "plot_curve_points_csv_sha256": final_package["plot_curve_points_csv_sha256"],
    }
