"""PPO-specific CPU capacity adapter for EXT-H-E7-PPO-STABILITY-01.

This module applies the existing conservative runtime-resource V1 contract to
one representative PPO branch.  It selects a safe active subprocess count from
CPU availability and measured peak host memory.  It does not change the frozen
scientific matrix and does not claim to locate the globally throughput-optimal
worker count.
"""

from __future__ import annotations

import concurrent.futures
import dataclasses
import hashlib
import math
import os
import shutil
import signal
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

from drpo import e7_canonical_ppo_stability as pilot
from drpo import e7_canonical_ppo_stability_entry as entry
from drpo import e7_canonical_sweep as base
from drpo.runtime_resource_autotune import (
    CPUSelection,
    MachineSnapshot,
    RuntimeResourceError,
    atomic_write_json,
    canonical_json_sha256,
    load_json,
    measure_command_peak_memory,
    select_cpu_workers,
    selection_document,
    utc_now,
)

ADAPTER_ID = "e7_canonical_ppo_stability_cpu_v1"
REPRESENTATIVE_DATASET = "walker2d-medium-v2"
REPRESENTATIVE_SEED = 200
REPRESENTATIVE_COEFFICIENT = 1.5


def _file_sha256(path: str | Path) -> str:
    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _flag_value(argv: list[str], flag: str) -> str:
    positions = [index for index, token in enumerate(argv) if token == flag]
    if len(positions) != 1 or positions[0] + 1 >= len(argv):
        raise RuntimeResourceError(
            f"trainer_argv_template must contain exactly one {flag}"
        )
    return argv[positions[0] + 1]


def _probe_trainer_template(
    run_spec: Mapping[str, Any],
    *,
    probe_steps: int,
) -> list[str]:
    """Return a terminating probe template without changing formal training.

    The canonical trainer assumes at least one evaluation before it writes its
    summary.  Capacity probes are shorter than the frozen 50k evaluation
    interval, so the runtime-only probe performs one terminal evaluation episode
    at ``probe_steps``.  This keeps the probe process structurally valid while
    leaving the scientific 10-episode/50k evaluation protocol untouched.
    """

    if probe_steps < 1:
        raise RuntimeResourceError("probe_steps must be positive")
    argv = [str(value) for value in run_spec["trainer_argv_template"]]
    expected = {
        "--eval_interval": "50000",
        "--eval_episodes": "10",
    }
    replacements = {
        "--eval_interval": str(probe_steps),
        "--eval_episodes": "1",
    }
    for flag, expected_value in expected.items():
        positions = [index for index, token in enumerate(argv) if token == flag]
        if len(positions) != 1 or positions[0] + 1 >= len(argv):
            raise RuntimeResourceError(
                f"probe trainer template must contain exactly one {flag}"
            )
        current = argv[positions[0] + 1]
        if current != expected_value:
            raise RuntimeResourceError(
                f"canonical probe source {flag} changed: {current} != {expected_value}"
            )
        argv[positions[0] + 1] = replacements[flag]
    return argv


def _load_run_spec(path: str | Path) -> tuple[dict[str, Any], str]:
    previous = pilot._BASE_LOAD_RUN_SPEC  # noqa: SLF001
    pilot._BASE_LOAD_RUN_SPEC = entry._load_source_run_spec  # noqa: SLF001
    try:
        return pilot.load_ppo_run_spec(path)
    finally:
        pilot._BASE_LOAD_RUN_SPEC = previous  # noqa: SLF001


def resource_fingerprint(
    *,
    repo_root: str | Path,
    contract_path: str | Path,
    run_spec_path: str | Path,
    grid_path: str | Path,
    probe_steps: int,
    probe_seed: int,
    probe_seconds: float,
    throughput_retention_fraction: float,
    fallback_workers: int,
    cpu_fraction: float,
    memory_headroom_fraction: float,
    per_worker_safety_factor: float,
    max_workers: int | None,
    max_growth_factor: float,
) -> dict[str, Any]:
    """Build a workload fingerprint that excludes scientific sweep coordinates.

    Seeds, EXP coefficients, horizon, and method count do not alter this PPO
    branch's compute graph and therefore do not invalidate the capacity profile.
    Batch size, network/source fingerprints, old-policy refresh cadence,
    diagnostics cadence, evaluator load, and thread environment remain included.
    """

    repo = Path(repo_root).resolve()
    run_spec, _ = _load_run_spec(run_spec_path)
    grid, _ = pilot.load_ppo_grid(grid_path)
    argv = [str(value) for value in run_spec["trainer_argv_template"]]
    environment = {
        name: str(run_spec.get("environment", {}).get(name))
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
    }
    source_paths = (
        "src/drpo/e7_canonical_ppo_injection.py",
        "src/drpo/e7_canonical_ppo_bootstrap.py",
        "src/drpo/e7_canonical_ppo_stability.py",
        "src/drpo/e7_canonical_ppo_stability_entry.py",
    )
    return {
        "schema_version": 1,
        "adapter_id": ADAPTER_ID,
        "hard_fields": {
            "contract_sha256": _file_sha256(contract_path),
            "source_sha256": {
                path: _file_sha256(repo / path)
                for path in source_paths
            },
            "actor_update_mode": "ppo_clip",
            "batch_size": int(_flag_value(argv, "--batch")),
            "optimizer_learning_rate": float(_flag_value(argv, "--lr")),
            "updates_per_old_policy": int(
                grid["ppo"]["updates_per_old_policy"]
            ),
            "diagnostics_interval": int(grid["ppo"]["diagnostics_interval"]),
            "thread_environment": environment,
            "representative_dataset": REPRESENTATIVE_DATASET,
        },
        "soft_fields": {
            "evaluation_interval": int(_flag_value(argv, "--eval_interval")),
            "evaluation_episodes": int(_flag_value(argv, "--eval_episodes")),
            "probe_terminal_evaluation_episodes": 1,
            "probe_steps": int(probe_steps),
            "probe_seconds": float(probe_seconds),
            "probe_seed_namespace": int(probe_seed),
        },
        "ignored_scientific_fields": [
            "development_seed_values",
            "held_out_seed_values",
            "exp_coefficients",
            "negative_control_label",
            "training_horizon",
            "method_count",
            "branch_count_except_task_limit",
        ],
        "selection_policy": {
            "fallback_workers": int(fallback_workers),
            "cpu_fraction": float(cpu_fraction),
            "memory_headroom_fraction": float(memory_headroom_fraction),
            "per_worker_safety_factor": float(per_worker_safety_factor),
            "max_workers": None if max_workers is None else int(max_workers),
            "max_growth_factor": float(max_growth_factor),
            "throughput_retention_fraction": float(
                throughput_retention_fraction
            ),
        },
        "tuned_runtime_field": "active_subprocess_count",
        "scientific_matrix_changed": False,
    }


def _cached_selection(
    path: Path,
    *,
    fingerprint: Mapping[str, Any],
    machine: MachineSnapshot,
    memory_headroom_fraction: float,
    per_worker_safety_factor: float,
    cpu_fraction: float,
) -> int | None:
    if not path.is_file():
        return None
    try:
        document = load_json(path)
    except Exception:  # noqa: BLE001 - damaged cache is a miss
        return None
    if document.get("adapter_id") != ADAPTER_ID:
        return None
    if document.get("resource_fingerprint_sha256") != canonical_json_sha256(
        dict(fingerprint)
    ):
        return None
    if document.get("machine_static_sha256") != canonical_json_sha256(
        machine.static_identity()
    ):
        return None
    selection = document.get("selection")
    probe = document.get("probe")
    if not isinstance(selection, dict) or not isinstance(probe, dict):
        return None
    workers = selection.get("selected_workers")
    memory_probe = probe.get("single_branch_memory_probe")
    if not isinstance(memory_probe, dict):
        return None
    peak = memory_probe.get("peak_rss_bytes")
    if not isinstance(workers, int) or workers < 1:
        return None
    if not isinstance(peak, int) or peak < 1:
        return None
    reserved = max(1, int(peak * per_worker_safety_factor))
    usable = int(
        machine.effective_memory_available_bytes * (1.0 - memory_headroom_fraction)
    )
    if workers * reserved > usable:
        return None
    current_cpu_limit = max(
        1,
        int(machine.logical_cpu_count * cpu_fraction - machine.load_average_1m),
    )
    if workers > current_cpu_limit:
        return None
    return workers


def build_probe_command(
    *,
    contract_path: str | Path,
    run_spec_path: str | Path,
    grid_path: str | Path,
    probe_root: str | Path,
    probe_steps: int,
    probe_seed: int,
) -> tuple[list[str], Path, dict[str, str], int, Path]:
    if probe_steps < 1:
        raise RuntimeResourceError("probe_steps must be positive")
    contract_source = Path(contract_path).expanduser().resolve()
    contract = base.CanonicalContract.load(contract_source)
    contract.verify_runtime()
    run_spec, _ = _load_run_spec(run_spec_path)
    grid, _ = pilot.load_ppo_grid(grid_path)
    branches = pilot.build_ppo_branches(contract, run_spec, grid)
    matches = [
        branch
        for branch in branches
        if branch.dataset.id == REPRESENTATIVE_DATASET
        and branch.seed == REPRESENTATIVE_SEED
        and branch.template_values.get("actor_update_mode") == "ppo_clip"
        and branch.negative_control is not None
        and branch.negative_control.method == "exponential"
        and abs(
            branch.negative_control.exponential_coefficient
            - REPRESENTATIVE_COEFFICIENT
        )
        < 1e-12
    ]
    if len(matches) != 1:
        raise RuntimeResourceError(
            f"expected one representative PPO branch, found {len(matches)}"
        )
    representative = matches[0]
    representative.dataset.verify()
    probe_branch = dataclasses.replace(
        representative,
        branch_id=(
            f"resource_probe__seed{probe_seed}__"
            f"{representative.branch_id.split('__', 2)[-1]}"
        ),
        seed=int(probe_seed),
        template_values={
            **representative.template_values,
            "steps": str(probe_steps),
            "diagnostics_interval": str(min(1000, probe_steps)),
        },
    )
    root = Path(probe_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    branch_dir = Path(tempfile.mkdtemp(prefix="attempt-", dir=root))
    command, _ = pilot.ppo_branch_command(
        contract_path=contract_source,
        contract=contract,
        branch=probe_branch,
        branch_dir=branch_dir,
        trainer_argv_template=_probe_trainer_template(
            run_spec,
            probe_steps=probe_steps,
        ),
    )
    environment = os.environ.copy()
    environment.update(
        {str(key): str(value) for key, value in run_spec.get("environment", {}).items()}
    )
    environment["DRPO_E7_BRANCH_ID"] = probe_branch.branch_id
    environment["DRPO_RUNTIME_RESOURCE_PROBE"] = "1"
    return command, branch_dir, environment, len(branches), contract.source_root


def _cleanup_probe_payload(probe_dir: Path) -> None:
    for relative in (
        "trainer_output",
        "checkpoints",
        "checkpoint",
        "ppo_diagnostics.jsonl",
        "PPO_DIAGNOSTICS_LATEST.json",
    ):
        target = probe_dir / relative
        if target.is_dir() and not target.is_symlink():
            shutil.rmtree(target)
        elif target.is_file() and not target.is_symlink():
            target.unlink()



def _terminate_process_group(process: subprocess.Popen[str]) -> int:
    if process.poll() is not None:
        return int(process.returncode)
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        return int(process.wait(timeout=5.0))
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        return int(process.wait())


def _run_probe_process(
    *,
    command: list[str],
    cwd: Path,
    environment: Mapping[str, str],
    log_path: Path,
    timeout_seconds: float,
) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    timed_out = False
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            env=dict(environment),
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        try:
            returncode = int(process.wait(timeout=timeout_seconds))
        except subprocess.TimeoutExpired:
            timed_out = True
            returncode = _terminate_process_group(process)
    return {
        "returncode": returncode,
        "timed_out": timed_out,
        "elapsed_seconds": time.monotonic() - started,
        "log_path": str(log_path),
    }


def _candidate_workers(safe_cap: int, fallback_workers: int) -> list[int]:
    if safe_cap < 1:
        raise RuntimeResourceError("safe worker cap must be positive")
    values = {safe_cap}
    if safe_cap >= 4:
        values.update(
            {
                max(1, int(round(safe_cap * 0.50))),
                max(1, int(round(safe_cap * 0.75))),
            }
        )
    if 1 <= fallback_workers <= safe_cap:
        values.add(fallback_workers)
    return sorted(values)


def _build_throughput_commands(
    *,
    contract_path: str | Path,
    run_spec_path: str | Path,
    grid_path: str | Path,
    candidate_root: Path,
    concurrency: int,
    probe_steps: int,
    probe_seed: int,
) -> tuple[list[tuple[list[str], Path, dict[str, str], Path]], Path]:
    contract_source = Path(contract_path).expanduser().resolve()
    contract = base.CanonicalContract.load(contract_source)
    contract.verify_runtime()
    run_spec, _ = _load_run_spec(run_spec_path)
    grid, _ = pilot.load_ppo_grid(grid_path)
    branches = pilot.build_ppo_branches(contract, run_spec, grid)
    matches = [
        branch
        for branch in branches
        if branch.dataset.id == REPRESENTATIVE_DATASET
        and branch.seed == REPRESENTATIVE_SEED
        and branch.template_values.get("actor_update_mode") == "ppo_clip"
        and branch.negative_control is not None
        and branch.negative_control.method == "exponential"
        and math.isclose(
            branch.negative_control.exponential_coefficient,
            REPRESENTATIVE_COEFFICIENT,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ]
    if len(matches) != 1:
        raise RuntimeResourceError(
            f"expected one throughput representative branch, found {len(matches)}"
        )
    representative = matches[0]
    representative.dataset.verify()
    candidate_root.mkdir(parents=True, exist_ok=True)
    values: list[tuple[list[str], Path, dict[str, str], Path]] = []
    for slot in range(concurrency):
        branch = dataclasses.replace(
            representative,
            branch_id=(
                f"throughput_probe__workers{concurrency}__slot{slot:03d}__"
                f"{representative.branch_id}"
            ),
            seed=probe_seed + slot,
            template_values={
                **representative.template_values,
                "steps": str(probe_steps),
                "diagnostics_interval": str(max(1, probe_steps)),
            },
        )
        branch_dir = candidate_root / f"slot-{slot:03d}"
        branch_dir.mkdir(parents=True, exist_ok=True)
        command, _ = pilot.ppo_branch_command(
            contract_path=contract_source,
            contract=contract,
            branch=branch,
            branch_dir=branch_dir,
            trainer_argv_template=_probe_trainer_template(
                run_spec,
                probe_steps=probe_steps,
            ),
        )
        environment = os.environ.copy()
        environment.update(
            {
                str(key): str(value)
                for key, value in run_spec.get("environment", {}).items()
            }
        )
        environment["DRPO_E7_BRANCH_ID"] = branch.branch_id
        environment["DRPO_RUNTIME_RESOURCE_PROBE"] = "1"
        values.append(
            (
                command,
                contract.source_root,
                environment,
                branch_dir / "stdout_stderr.log",
            )
        )
    return values, contract.source_root


def _benchmark_concurrency(
    *,
    contract_path: str | Path,
    run_spec_path: str | Path,
    grid_path: str | Path,
    probe_root: Path,
    concurrency: int,
    probe_steps: int,
    probe_seed: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    candidate_root = probe_root / f"workers-{concurrency:03d}"
    if candidate_root.exists():
        shutil.rmtree(candidate_root)
    commands, _ = _build_throughput_commands(
        contract_path=contract_path,
        run_spec_path=run_spec_path,
        grid_path=grid_path,
        candidate_root=candidate_root,
        concurrency=concurrency,
        probe_steps=probe_steps,
        probe_seed=probe_seed,
    )
    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(
                _run_probe_process,
                command=command,
                cwd=cwd,
                environment=environment,
                log_path=log_path,
                timeout_seconds=timeout_seconds,
            )
            for command, cwd, environment, log_path in commands
        ]
        results = [future.result() for future in futures]
    elapsed = time.monotonic() - started
    completed = sum(
        row["returncode"] == 0 and not row["timed_out"] for row in results
    )
    failed = concurrency - completed
    aggregate_updates = completed * probe_steps
    aggregate_updates_per_second = (
        aggregate_updates / elapsed if elapsed > 0.0 else 0.0
    )
    per_branch_updates_per_second = (
        aggregate_updates_per_second / concurrency if concurrency else 0.0
    )
    summary = {
        "concurrency": concurrency,
        "probe_steps_per_branch": probe_steps,
        "elapsed_seconds": elapsed,
        "completed": completed,
        "failed": failed,
        "timed_out": sum(bool(row["timed_out"]) for row in results),
        "aggregate_updates_per_second": aggregate_updates_per_second,
        "per_branch_updates_per_second": per_branch_updates_per_second,
        "branch_elapsed_seconds_mean": (
            sum(float(row["elapsed_seconds"]) for row in results) / len(results)
        ),
        "valid": failed == 0 and aggregate_updates_per_second > 0.0,
    }
    atomic_write_json(candidate_root / "BENCHMARK_SUMMARY.json", summary)
    for slot_dir in candidate_root.glob("slot-*"):
        for relative in (
            "trainer_output",
            "checkpoints",
            "checkpoint",
            "ppo_diagnostics.jsonl",
            "PPO_DIAGNOSTICS_LATEST.json",
        ):
            target = slot_dir / relative
            if target.is_dir() and not target.is_symlink():
                shutil.rmtree(target)
            elif target.is_file() and not target.is_symlink():
                target.unlink()
    return summary


def _select_from_throughput(
    benchmarks: list[Mapping[str, Any]],
    *,
    retention_fraction: float,
) -> tuple[int, dict[str, Any]]:
    if not 0.5 <= retention_fraction <= 1.0:
        raise RuntimeResourceError(
            "throughput_retention_fraction must be in [0.5, 1.0]"
        )
    valid = [row for row in benchmarks if row.get("valid") is True]
    if not valid:
        raise RuntimeResourceError("no concurrency candidate completed successfully")
    peak = max(float(row["aggregate_updates_per_second"]) for row in valid)
    threshold = peak * retention_fraction
    eligible = [
        row
        for row in valid
        if float(row["aggregate_updates_per_second"]) >= threshold
    ]
    selected = min(int(row["concurrency"]) for row in eligible)
    return selected, {
        "peak_aggregate_updates_per_second": peak,
        "retention_fraction": retention_fraction,
        "eligible_threshold_updates_per_second": threshold,
        "rule": "smallest_successful_candidate_at_or_above_retained_peak",
    }

def select_runtime(
    *,
    machine: MachineSnapshot,
    repo_root: str | Path,
    contract_path: str | Path,
    run_spec_path: str | Path,
    grid_path: str | Path,
    work_dir: str | Path,
    fallback_workers: int,
    probe_steps: int,
    probe_seed: int,
    probe_seconds: float,
    throughput_retention_fraction: float,
    cpu_fraction: float,
    memory_headroom_fraction: float,
    per_worker_safety_factor: float,
    max_workers: int | None,
    max_growth_factor: float,
    minimum_branches_for_probe: int,
) -> dict[str, Any]:
    work = Path(work_dir).resolve()
    selection_path = work / "RUNTIME_SELECTION.json"
    fingerprint = resource_fingerprint(
        repo_root=repo_root,
        contract_path=contract_path,
        run_spec_path=run_spec_path,
        grid_path=grid_path,
        probe_steps=probe_steps,
        probe_seed=probe_seed,
        probe_seconds=probe_seconds,
        throughput_retention_fraction=throughput_retention_fraction,
        fallback_workers=fallback_workers,
        cpu_fraction=cpu_fraction,
        memory_headroom_fraction=memory_headroom_fraction,
        per_worker_safety_factor=per_worker_safety_factor,
        max_workers=max_workers,
        max_growth_factor=max_growth_factor,
    )
    cached = _cached_selection(
        selection_path,
        fingerprint=fingerprint,
        machine=machine,
        memory_headroom_fraction=memory_headroom_fraction,
        per_worker_safety_factor=per_worker_safety_factor,
        cpu_fraction=cpu_fraction,
    )
    if cached is not None:
        previous = load_json(selection_path)
        previous["mode"] = "cached"
        previous["cache_validated_machine_snapshot"] = machine.as_dict()
        previous["cache_validated_utc"] = utc_now()
        atomic_write_json(selection_path, previous)
        return previous

    command, probe_dir, environment, total_tasks, command_cwd = build_probe_command(
        contract_path=contract_path,
        run_spec_path=run_spec_path,
        grid_path=grid_path,
        probe_root=work / "_runtime_resource_probe" / "e7_ppo",
        probe_steps=probe_steps,
        probe_seed=probe_seed,
    )
    if total_tasks < minimum_branches_for_probe:
        selection = select_cpu_workers(
            machine,
            total_tasks=total_tasks,
            fallback_workers=fallback_workers,
            per_worker_peak_bytes=None,
            cpu_fraction=cpu_fraction,
            memory_headroom_fraction=memory_headroom_fraction,
            per_worker_safety_factor=per_worker_safety_factor,
            max_workers=max_workers,
            max_growth_factor=max_growth_factor,
        )
        document = selection_document(
            adapter_id=ADAPTER_ID,
            resource_fingerprint=fingerprint,
            machine=machine,
            mode="exempt",
            selection=selection.as_dict(),
            probe=None,
            fallback={"workers": fallback_workers, "reason": "small_task_exemption"},
            repo_root=repo_root,
            limitations=[
                "capacity_guard_not_throughput_knee_search",
                "no_real_probe_for_small_task_exemption",
            ],
        )
        atomic_write_json(selection_path, document)
        return document

    try:
        probe = measure_command_peak_memory(
            command,
            cwd=command_cwd,
            environment=environment,
            log_path=probe_dir / "stdout_stderr.log",
            sample_seconds=probe_seconds,
            accept_timeout=True,
        )
    finally:
        _cleanup_probe_payload(probe_dir)

    safe_selection: CPUSelection = select_cpu_workers(
        machine,
        total_tasks=total_tasks,
        fallback_workers=fallback_workers,
        per_worker_peak_bytes=probe.peak_rss_bytes,
        cpu_fraction=cpu_fraction,
        memory_headroom_fraction=memory_headroom_fraction,
        per_worker_safety_factor=per_worker_safety_factor,
        max_workers=max_workers,
        max_growth_factor=max_growth_factor,
    )
    candidates = _candidate_workers(
        safe_selection.selected_workers,
        fallback_workers,
    )
    throughput_root = work / "_runtime_resource_probe" / "e7_ppo_throughput"
    benchmarks: list[dict[str, Any]] = []
    for index, concurrency in enumerate(candidates):
        benchmark = _benchmark_concurrency(
            contract_path=contract_path,
            run_spec_path=run_spec_path,
            grid_path=grid_path,
            probe_root=throughput_root,
            concurrency=concurrency,
            probe_steps=probe_steps,
            probe_seed=probe_seed + index * 10_000,
            timeout_seconds=probe_seconds,
        )
        benchmarks.append(benchmark)
        if not benchmark["valid"]:
            break
    selected_workers, throughput_rule = _select_from_throughput(
        benchmarks,
        retention_fraction=throughput_retention_fraction,
    )
    selection_payload = safe_selection.as_dict()
    selection_payload.update(
        {
            "selected_workers": selected_workers,
            "safe_capacity_ceiling": safe_selection.selected_workers,
            "reason": "empirical_short_throughput_grid_with_safe_capacity_ceiling",
        }
    )
    probe_payload = {
        "single_branch_memory_probe": probe.as_dict(),
        "candidate_workers": candidates,
        "throughput_benchmarks": benchmarks,
        "throughput_selection_rule": throughput_rule,
    }
    document = selection_document(
        adapter_id=ADAPTER_ID,
        resource_fingerprint=fingerprint,
        machine=machine,
        mode="auto",
        selection=selection_payload,
        probe=probe_payload,
        fallback={"workers": fallback_workers, "reason": "legacy_verified_schedule"},
        repo_root=repo_root,
        limitations=[
            "candidate_grid_not_continuous_global_optimization",
            "probe_uses_one_terminal_eval_episode_not_formal_10_episode_burst",
            "single_representative_ppo_workload_family",
        ],
    )
    atomic_write_json(selection_path, document)
    return document
