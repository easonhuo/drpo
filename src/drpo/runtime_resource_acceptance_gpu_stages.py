"""GPU placement, E8 thread scan, and concurrent pool acceptance stages."""
from __future__ import annotations

import csv
import json
import os
import shutil
from pathlib import Path
from typing import Any, Mapping

from drpo.runtime_resource_acceptance import AcceptanceError, StageResult, sha256_file
from drpo.runtime_resource_acceptance import stage_result, utc_now
from drpo.runtime_resource_acceptance_commands import (
    THREAD_ENVIRONMENT_NAMES,
    candidate_above_one,
    first_numeric,
    gpu_failures,
    gpu_selection_command,
    internal_e7_command,
    numerical_matches,
    pool_command,
)
from drpo.runtime_resource_acceptance_process import run_command, run_concurrent
from drpo.runtime_resource_autotune import load_json


def _reuse_calibration(source_work: Path, target_work: Path) -> Path:
    source = source_work / "calibration" / "taper_budget_calibration.json"
    if not source.is_file():
        raise AcceptanceError(f"validated calibration is missing: {source}")
    target = target_work / "calibration" / source.name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    if sha256_file(source) != sha256_file(target):
        raise AcceptanceError("reused calibration checksum mismatch")
    return target


def gpu_stage(
    root: Path,
    repo: Path,
    gpu_repo: Path,
    profile: Mapping[str, Any],
    ledger: Path,
) -> StageResult:
    started = utc_now()
    directory = root / "stage3_gpu_placement"
    if not profile["e8"]["enabled"]:
        return stage_result(
            root, "stage3_gpu_placement", "NOT_RUN", started, {"reason": "E8 disabled"}
        )
    pools = profile["resource_pools"]
    work = directory / "work"
    try:
        result = run_command(
            pool_command(
                repo,
                cpu_pool=pools["e8_cpu_pool"],
                identity=directory / "RESOURCE_POOL.json",
                command=gpu_selection_command(
                    gpu_repo, profile, work_dir=work, gpu_ids=pools["e8_gpu_ids"]
                ),
                gpu_ids=pools["e8_gpu_ids"],
            ),
            cwd=gpu_repo,
            environment={**os.environ, "PYTHONPATH": str(gpu_repo / "src")},
            timeout_seconds=float(profile["e8"]["selection_timeout_seconds"]),
            log_path=directory / "selection.log",
            samples_path=directory / "selection_samples.jsonl",
            command_ledger=ledger,
        )
        if not result.ok:
            raise AcceptanceError("GPU placement selection-only command failed")
        selection = load_json(work / "RUNTIME_SELECTION.json")
        failures = gpu_failures(selection)
        if failures:
            raise AcceptanceError(f"GPU placement structured failures: {failures}")
        fingerprint = selection.get("workload_fingerprint", {})
        if not isinstance(fingerprint, Mapping):
            raise AcceptanceError("GPU placement fingerprint is missing")
        if fingerprint.get("test_split_access") != "not_accessed_selection_only":
            raise AcceptanceError("GPU placement did not prove test-split isolation")
        if fingerprint.get("test_sha256") is not None:
            raise AcceptanceError("GPU placement unexpectedly recorded a test hash")
        numerical = numerical_matches(list(directory.rglob("*")))
        if numerical:
            raise AcceptanceError("GPU placement logs contain NaN/Inf indicators")
        above_one = candidate_above_one(selection)
        status = "PASS" if above_one else "INCONCLUSIVE"
        return stage_result(
            root,
            "stage3_gpu_placement",
            status,
            started,
            {
                "command": result.as_dict(),
                "selection_sha256": sha256_file(work / "RUNTIME_SELECTION.json"),
                "candidate_above_one_observed": above_one,
                "single_worker_envelope_passed": True,
                "test_split_access": fingerprint.get("test_split_access"),
                "structured_failures": failures,
                "nan_inf_matches": numerical,
                "full_scientific_sweep_started": False,
            },
        )
    except BaseException as exc:
        return stage_result(
            root,
            "stage3_gpu_placement",
            "FAIL",
            started,
            {"error_type": type(exc).__name__, "error": str(exc)},
        )


def _thread_environment(value: int | None, gpu_repo: Path) -> dict[str, str]:
    environment = {**os.environ, "PYTHONPATH": str(gpu_repo / "src")}
    if value is not None:
        for name in THREAD_ENVIRONMENT_NAMES:
            environment[name] = str(value)
    return environment


def thread_scan_stage(
    root: Path,
    repo: Path,
    gpu_repo: Path,
    profile: Mapping[str, Any],
    ledger: Path,
    gpu_result: StageResult,
) -> StageResult:
    started = utc_now()
    directory = root / "stage4_e8_thread_scan"
    if not profile["e8"]["enabled"]:
        return stage_result(
            root, "stage4_e8_thread_scan", "NOT_RUN", started, {"reason": "E8 disabled"}
        )
    if gpu_result.status not in {"PASS", "INCONCLUSIVE"}:
        return stage_result(
            root,
            "stage4_e8_thread_scan",
            "BLOCKED",
            started,
            {"reason": "independent GPU selection stage failed"},
        )
    pools = profile["resource_pools"]
    gpu_ids = [str(pools["e8_gpu_ids"][0])]
    source_work = root / "stage3_gpu_placement" / "work"
    rows: list[dict[str, Any]] = []
    try:
        for candidate in profile["e8"]["thread_candidates"]:
            label = "unbounded" if candidate is None else f"threads_{candidate}"
            candidate_dir = directory / label
            work = candidate_dir / "work"
            calibration = _reuse_calibration(source_work, work)
            result = run_command(
                pool_command(
                    repo,
                    cpu_pool=pools["e8_cpu_pool"],
                    identity=candidate_dir / "RESOURCE_POOL.json",
                    command=gpu_selection_command(
                        gpu_repo,
                        profile,
                        work_dir=work,
                        gpu_ids=gpu_ids,
                        max_devices=1,
                        max_slots=1,
                    ),
                    gpu_ids=gpu_ids,
                ),
                cwd=gpu_repo,
                environment=_thread_environment(candidate, gpu_repo),
                timeout_seconds=float(profile["e8"]["selection_timeout_seconds"]),
                log_path=candidate_dir / "selection.log",
                samples_path=candidate_dir / "samples.jsonl",
                command_ledger=ledger,
            )
            if not result.ok:
                raise AcceptanceError(f"E8 thread candidate failed: {label}")
            selection = load_json(work / "RUNTIME_SELECTION.json")
            failures = gpu_failures(selection)
            if failures:
                raise AcceptanceError(f"E8 thread candidate failure {label}: {failures}")
            rows.append(
                {
                    "candidate": "unbounded" if candidate is None else candidate,
                    "elapsed_seconds": result.elapsed_seconds,
                    "peak_rss_bytes": result.peak_rss_bytes,
                    "peak_process_count": result.peak_process_count,
                    "average_cpu_cores": first_numeric(selection, "average_cpu_cores"),
                    "peak_host_rss_bytes": first_numeric(selection, "peak_host_rss_bytes"),
                    "peak_incremental_vram_bytes": first_numeric(
                        selection, "peak_incremental_vram_bytes"
                    ),
                    "minimum_free_vram_bytes": first_numeric(
                        selection, "minimum_free_vram_bytes"
                    ),
                    "calibration_sha256": sha256_file(calibration),
                    "selection_sha256": sha256_file(work / "RUNTIME_SELECTION.json"),
                }
            )
        with (directory / "THREAD_SCAN.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        numerical = numerical_matches(list(directory.rglob("*")))
        if numerical:
            raise AcceptanceError("thread scan logs contain NaN/Inf indicators")
        return stage_result(
            root,
            "stage4_e8_thread_scan",
            "PASS",
            started,
            {
                "rows": rows,
                "calibration_reused_from_stage3": True,
                "nan_inf_matches": numerical,
                "permanent_thread_policy_selected": False,
                "full_scientific_sweep_started": False,
            },
        )
    except BaseException as exc:
        return stage_result(
            root,
            "stage4_e8_thread_scan",
            "FAIL",
            started,
            {"error_type": type(exc).__name__, "error": str(exc), "rows": rows},
        )


def _affinity_violations(samples: Path, pools: Mapping[str, set[int]]) -> list[dict[str, Any]]:
    if not samples.is_file():
        return [{"reason": "samples_missing"}]
    violations: list[dict[str, Any]] = []
    for line_number, line in enumerate(samples.read_text(encoding="utf-8").splitlines(), 1):
        payload = json.loads(line)
        for name, command in payload.get("commands", {}).items():
            for process in command.get("processes", []):
                observed = set(int(item) for item in process.get("affinity_cpu_ids", []))
                if not observed or not observed.issubset(pools[name]):
                    violations.append(
                        {
                            "line": line_number,
                            "command": name,
                            "pid": process.get("pid"),
                            "observed": sorted(observed),
                            "allowed": sorted(pools[name]),
                        }
                    )
    return violations


def concurrent_stage(
    root: Path,
    repo: Path,
    gpu_repo: Path,
    profile_path: Path,
    profile: Mapping[str, Any],
    ledger: Path,
    e7_result: StageResult,
    gpu_result: StageResult,
) -> StageResult:
    started = utc_now()
    directory = root / "stage5_concurrent_pool"
    if not profile["concurrent"]["enabled"]:
        return stage_result(
            root,
            "stage5_concurrent_pool",
            "NOT_RUN",
            started,
            {"reason": "concurrent stage disabled"},
        )
    acceptable = {"PASS", "INCONCLUSIVE"}
    if e7_result.status not in acceptable or gpu_result.status not in acceptable:
        return stage_result(
            root,
            "stage5_concurrent_pool",
            "BLOCKED",
            started,
            {
                "reason": "independent E7 and GPU liveness must be usable",
                "e7_status": e7_result.status,
                "gpu_status": gpu_result.status,
            },
        )
    pools = profile["resource_pools"]
    e7_work = root / "stage2_e7_cpu_v2" / "work"
    gpu_ids = [str(pools["e8_gpu_ids"][0])]
    e8_work = directory / "e8_work"
    try:
        calibration = _reuse_calibration(
            root / "stage3_gpu_placement" / "work", e8_work
        )
        e7_command = pool_command(
            repo,
            cpu_pool=pools["e7_cpu_pool"],
            identity=root / "stage2_e7_cpu_v2" / "RESOURCE_POOL.json",
            command=internal_e7_command(
                repo,
                profile_path,
                "liveness",
                e7_work,
                directory / "E7_SELECTED_LIVENESS.json",
            ),
        )
        e8_command = pool_command(
            repo,
            cpu_pool=pools["e8_cpu_pool"],
            identity=directory / "E8_RESOURCE_POOL.json",
            command=gpu_selection_command(
                gpu_repo,
                profile,
                work_dir=e8_work,
                gpu_ids=gpu_ids,
                max_devices=1,
                max_slots=1,
            ),
            gpu_ids=gpu_ids,
        )
        samples = directory / "process_samples.jsonl"
        results = run_concurrent(
            {"e7": e7_command, "e8": e8_command},
            cwd_by_name={"e7": repo, "e8": gpu_repo},
            environment_by_name={
                "e7": os.environ.copy(),
                "e8": {**os.environ, "PYTHONPATH": str(gpu_repo / "src")},
            },
            timeout_seconds=float(profile["concurrent"]["timeout_seconds"]),
            log_dir=directory / "logs",
            samples_path=samples,
            sample_interval_seconds=float(
                profile["concurrent"]["sample_interval_seconds"]
            ),
            command_ledger=ledger,
        )
        if not all(result.ok for result in results.values()):
            raise AcceptanceError("concurrent commands did not complete cleanly")
        violations = _affinity_violations(
            samples,
            {
                "e7": set(int(item) for item in pools["e7_cpu_ids"]),
                "e8": set(int(item) for item in pools["e8_cpu_ids"]),
            },
        )
        if violations:
            raise AcceptanceError("concurrent affinity escaped declared pool")
        numerical = numerical_matches(list(directory.rglob("*")))
        if numerical:
            raise AcceptanceError("concurrent logs contain NaN/Inf indicators")
        return stage_result(
            root,
            "stage5_concurrent_pool",
            "PASS",
            started,
            {
                "commands": {name: result.as_dict() for name, result in results.items()},
                "independent_e7_status": e7_result.status,
                "independent_gpu_status": gpu_result.status,
                "calibration_sha256": sha256_file(calibration),
                "affinity_violations": violations,
                "nan_inf_matches": numerical,
                "full_scientific_sweep_started": False,
            },
        )
    except BaseException as exc:
        return stage_result(
            root,
            "stage5_concurrent_pool",
            "FAIL",
            started,
            {"error_type": type(exc).__name__, "error": str(exc)},
        )
