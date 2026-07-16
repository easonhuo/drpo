"""Measured CPU/RAM and bounded-throughput autotuning for E7 PPO-family grids."""
from __future__ import annotations

import concurrent.futures
import copy
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
from typing import Any, Mapping, Sequence

from drpo import e7_canonical_sweep as base
from drpo import e7_ppo_w0_grid_pilot as pilot
from drpo import runtime_cpu_capacity as cpu
from drpo.runtime_resource_autotune import (
    MachineSnapshot,
    RuntimeResourceError,
    atomic_write_json,
    canonical_json_sha256,
    git_state,
    load_json,
    process_tree_rss,
    selection_document,
    utc_now,
)

ADAPTER_ID = "e7_ppo_w0_exp_grid_cpu_v2"
SELECTOR_POLICY_VERSION = 2
SELECTION_SCHEMA_VERSION = 2
REPRESENTATIVE_DATASET = "walker2d-medium-v2"
REPRESENTATIVE_W0 = 0.11
REPRESENTATIVE_COEFFICIENT = 1.0


def _file_sha256(path: str | Path) -> str:
    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _selector_implementation_identity(repo_root: str | Path) -> dict[str, str]:
    repo = Path(repo_root).resolve()
    return {
        "e7_ppo_w0_runtime_autotune.py": _file_sha256(Path(__file__).resolve()),
        "runtime_cpu_capacity.py": _file_sha256(
            repo / "src/drpo/runtime_cpu_capacity.py"
        ),
    }


def _probe_trainer_template(
    run_spec: Mapping[str, Any], *, probe_steps: int
) -> list[str]:
    if probe_steps < 1:
        raise RuntimeResourceError("probe_steps must be positive")
    argv = [str(value) for value in run_spec["trainer_argv_template"]]
    expected = {"--eval_interval": "50000", "--eval_episodes": "10"}
    replacements = {"--eval_interval": str(probe_steps), "--eval_episodes": "1"}
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
    per_worker_cpu_safety_factor: float,
    minimum_cpu_cores_per_worker: float,
    max_workers: int | None,
    max_growth_factor: float,
    revalidation_samples: int,
    revalidation_sample_seconds: float,
) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    run_spec, _ = pilot.load_w0_run_spec(run_spec_path)
    grid, _ = pilot.load_w0_grid(grid_path)
    argv = [str(value) for value in run_spec["trainer_argv_template"]]
    source_paths = (
        "src/drpo/e7_ppo_w0_grid_pilot.py",
        "src/drpo/e7_ppo_w0_bootstrap.py",
        "src/drpo/e7_canonical_ppo_injection.py",
        "src/drpo/e7_canonical_injection.py",
        "src/drpo/e7_canonical_sweep.py",
    )
    return {
        "schema_version": 2,
        "adapter_id": ADAPTER_ID,
        "selector_policy_version": SELECTOR_POLICY_VERSION,
        "hard_fields": {
            "contract_sha256": _file_sha256(contract_path),
            "run_spec_sha256": _file_sha256(run_spec_path),
            "grid_sha256": _file_sha256(grid_path),
            "source_sha256": {path: _file_sha256(repo / path) for path in source_paths},
            "actor_update_mode": "ppo_clip",
            "batch_size": int(pilot._flag_value(argv, "--batch")),
            "optimizer_learning_rate": float(pilot._flag_value(argv, "--lr")),
            "updates_per_old_policy": int(grid["ppo"]["updates_per_old_policy"]),
            "diagnostics_interval": int(grid["ppo"]["diagnostics_interval"]),
            "thread_environment": {
                name: str(run_spec.get("environment", {}).get(name))
                for name in (
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                )
            },
            "representative_workload": {
                "dataset": REPRESENTATIVE_DATASET,
                "weight_at_zero": REPRESENTATIVE_W0,
                "exp_coefficient": REPRESENTATIVE_COEFFICIENT,
            },
        },
        "soft_fields": {
            "formal_evaluation_interval": int(
                pilot._flag_value(argv, "--eval_interval")
            ),
            "formal_evaluation_episodes": int(
                pilot._flag_value(argv, "--eval_episodes")
            ),
            "probe_terminal_evaluation_episodes": 1,
            "probe_steps": int(probe_steps),
            "probe_seconds": float(probe_seconds),
            "probe_seed_namespace": int(probe_seed),
        },
        "selection_policy": {
            "fallback_workers": int(fallback_workers),
            "cpu_fraction": float(cpu_fraction),
            "memory_headroom_fraction": float(memory_headroom_fraction),
            "per_worker_memory_safety_factor": float(per_worker_safety_factor),
            "per_worker_cpu_safety_factor": float(per_worker_cpu_safety_factor),
            "minimum_cpu_cores_per_worker": float(minimum_cpu_cores_per_worker),
            "max_workers": None if max_workers is None else int(max_workers),
            "max_growth_factor": float(max_growth_factor),
            "throughput_retention_fraction": float(throughput_retention_fraction),
            "revalidation_samples": int(revalidation_samples),
            "revalidation_sample_seconds": float(revalidation_sample_seconds),
            "load_average_role": "diagnostic_only",
        },
        "ignored_scientific_coordinates": [
            "development_seed_values",
            "weight_at_zero_grid",
            "exp_coefficient_grid",
            "training_horizon",
        ],
        "tuned_runtime_field": "active_subprocess_count",
        "scientific_matrix_changed": False,
    }


def _inputs(
    contract_path: str | Path,
    run_spec_path: str | Path,
    grid_path: str | Path,
) -> tuple[base.CanonicalContract, dict[str, Any], dict[str, Any], list[base.Branch]]:
    contract = base.CanonicalContract.load(Path(contract_path).expanduser().resolve())
    contract.verify_runtime()
    run_spec, _ = pilot.load_w0_run_spec(run_spec_path)
    grid, _ = pilot.load_w0_grid(grid_path)
    branches = pilot.build_w0_branches(contract, run_spec, grid)
    return contract, run_spec, grid, branches


def _representative(branches: list[base.Branch]) -> base.Branch:
    matches = [
        branch
        for branch in branches
        if branch.dataset.id == REPRESENTATIVE_DATASET
        and branch.seed == pilot.EXPECTED_SEEDS[0]
        and math.isclose(
            float(branch.template_values["weight_at_zero"]),
            REPRESENTATIVE_W0,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and math.isclose(
            float(branch.template_values["exp_coefficient"]),
            REPRESENTATIVE_COEFFICIENT,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ]
    if len(matches) != 1:
        raise RuntimeResourceError(
            f"expected one representative PPO w(0) branch, found {len(matches)}"
        )
    return matches[0]


def build_probe_command(
    *,
    contract_path: str | Path,
    run_spec_path: str | Path,
    grid_path: str | Path,
    probe_root: str | Path,
    probe_steps: int,
    probe_seed: int,
) -> tuple[list[str], Path, dict[str, str], int, Path]:
    contract, run_spec, _grid, branches = _inputs(
        contract_path, run_spec_path, grid_path
    )
    representative = _representative(branches)
    representative.dataset.verify()
    probe_branch = dataclasses.replace(
        representative,
        branch_id=f"resource_probe__seed{probe_seed}__{representative.branch_id}",
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
    command, _ = pilot.w0_branch_command(
        contract_path=Path(contract_path).expanduser().resolve(),
        contract=contract,
        branch=probe_branch,
        branch_dir=branch_dir,
        trainer_argv_template=_probe_trainer_template(
            run_spec, probe_steps=probe_steps
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


def _process_group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _terminate_process_group(
    process: subprocess.Popen[str], *, grace_seconds: float = 5.0
) -> tuple[int, bool]:
    process.poll()
    intervened = _process_group_alive(process.pid)
    if intervened:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + grace_seconds
        while _process_group_alive(process.pid) and time.monotonic() < deadline:
            time.sleep(0.05)
        if _process_group_alive(process.pid):
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    try:
        returncode = int(process.wait(timeout=max(1.0, grace_seconds)))
    except subprocess.TimeoutExpired:
        returncode = int(process.wait())
    return returncode, intervened


def _measure_representative_resources(
    *,
    command: Sequence[str],
    cwd: Path,
    environment: Mapping[str, str],
    log_path: Path,
    timeout_seconds: float,
    binding: cpu.CPUBinding,
    proc_stat_path: str | Path,
    poll_interval_seconds: float = 0.20,
) -> dict[str, Any]:
    if timeout_seconds <= 0 or poll_interval_seconds <= 0:
        raise RuntimeResourceError("resource probe durations must be positive")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started_utc = utc_now()
    start_counter = cpu.capture_cpu_counters(binding, proc_stat_path=proc_stat_path)
    peak_rss = 0
    peak_cpu_seconds = 0.0
    timed_out = False
    controller_terminated = False
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("COMMAND=" + " ".join(str(item) for item in command) + "\n")
        handle.flush()
        process = subprocess.Popen(
            [str(item) for item in command],
            cwd=str(cwd),
            env=dict(environment),
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        deadline = time.monotonic() + timeout_seconds
        while True:
            peak_rss = max(peak_rss, process_tree_rss(process.pid))
            peak_cpu_seconds = max(
                peak_cpu_seconds, cpu.process_tree_cpu_seconds(process.pid)
            )
            returncode = process.poll()
            if returncode is not None:
                break
            if time.monotonic() >= deadline:
                timed_out = True
                returncode, controller_terminated = _terminate_process_group(process)
                break
            time.sleep(poll_interval_seconds)
        if process.poll() is None or _process_group_alive(process.pid):
            returncode, intervened = _terminate_process_group(process)
            controller_terminated = controller_terminated or intervened
    end_counter = cpu.capture_cpu_counters(binding, proc_stat_path=proc_stat_path)
    interval = cpu.cpu_interval_measurement(start_counter, end_counter)
    if not timed_out and int(returncode) != 0:
        raise RuntimeResourceError(
            f"resource probe exited with return code {returncode}; see {log_path}"
        )
    if peak_rss <= 0:
        raise RuntimeResourceError("resource probe did not expose positive RSS")
    measured_cpu_cores = peak_cpu_seconds / interval.elapsed_seconds
    if measured_cpu_cores <= 0:
        raise RuntimeResourceError("resource probe did not expose positive CPU demand")
    if _process_group_alive(process.pid):
        raise RuntimeResourceError("resource probe left a live process group")
    return {
        "command": [str(item) for item in command],
        "started_utc": started_utc,
        "finished_utc": utc_now(),
        "elapsed_seconds": interval.elapsed_seconds,
        "peak_rss_bytes": peak_rss,
        "process_tree_cpu_seconds": peak_cpu_seconds,
        "measured_cpu_cores": measured_cpu_cores,
        "cpu_interval": interval.as_dict(),
        "returncode": int(returncode),
        "timed_out": timed_out,
        "controller_terminated": controller_terminated,
        "process_group_alive_after_cleanup": False,
        "log_path": str(log_path),
    }


def _run_probe_process(
    *,
    command: list[str],
    cwd: Path,
    environment: Mapping[str, str],
    log_path: Path,
    timeout_seconds: float,
    poll_interval_seconds: float = 0.20,
) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    timed_out = False
    controller_terminated = False
    peak_rss = 0
    peak_cpu_seconds = 0.0
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
        deadline = started + timeout_seconds
        while True:
            peak_rss = max(peak_rss, process_tree_rss(process.pid))
            peak_cpu_seconds = max(
                peak_cpu_seconds, cpu.process_tree_cpu_seconds(process.pid)
            )
            returncode = process.poll()
            if returncode is not None:
                break
            if time.monotonic() >= deadline:
                timed_out = True
                returncode, controller_terminated = _terminate_process_group(process)
                break
            time.sleep(poll_interval_seconds)
        if process.poll() is None or _process_group_alive(process.pid):
            returncode, intervened = _terminate_process_group(process)
            controller_terminated = controller_terminated or intervened
    return {
        "returncode": int(returncode),
        "timed_out": timed_out,
        "controller_terminated": controller_terminated,
        "process_group_alive_after_cleanup": _process_group_alive(process.pid),
        "elapsed_seconds": time.monotonic() - started,
        "peak_rss_bytes": peak_rss,
        "process_tree_cpu_seconds": peak_cpu_seconds,
        "log_path": str(log_path),
    }


def candidate_workers(safe_cap: int, fallback_workers: int) -> list[int]:
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


def _throughput_commands(
    *,
    contract_path: str | Path,
    run_spec_path: str | Path,
    grid_path: str | Path,
    candidate_root: Path,
    concurrency: int,
    probe_steps: int,
    probe_seed: int,
) -> list[tuple[list[str], Path, dict[str, str], Path]]:
    contract, run_spec, _grid, branches = _inputs(
        contract_path, run_spec_path, grid_path
    )
    representative = _representative(branches)
    representative.dataset.verify()
    candidate_root.mkdir(parents=True, exist_ok=True)
    values: list[tuple[list[str], Path, dict[str, str], Path]] = []
    template = _probe_trainer_template(run_spec, probe_steps=probe_steps)
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
        command, _ = pilot.w0_branch_command(
            contract_path=Path(contract_path).expanduser().resolve(),
            contract=contract,
            branch=branch,
            branch_dir=branch_dir,
            trainer_argv_template=template,
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
    return values


def benchmark_concurrency(
    *,
    contract_path: str | Path,
    run_spec_path: str | Path,
    grid_path: str | Path,
    probe_root: Path,
    concurrency: int,
    probe_steps: int,
    probe_seed: int,
    timeout_seconds: float,
    binding: cpu.CPUBinding,
    proc_stat_path: str | Path,
    cpu_fraction: float,
    cpu_safety_factor: float,
    usable_memory_bytes: int,
) -> dict[str, Any]:
    candidate_root = probe_root / f"workers-{concurrency:03d}"
    if candidate_root.exists():
        shutil.rmtree(candidate_root)
    commands = _throughput_commands(
        contract_path=contract_path,
        run_spec_path=run_spec_path,
        grid_path=grid_path,
        candidate_root=candidate_root,
        concurrency=concurrency,
        probe_steps=probe_steps,
        probe_seed=probe_seed,
    )
    start_counter = cpu.capture_cpu_counters(binding, proc_stat_path=proc_stat_path)
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
    end_counter = cpu.capture_cpu_counters(binding, proc_stat_path=proc_stat_path)
    interval = cpu.cpu_interval_measurement(start_counter, end_counter)
    completed = sum(
        row["returncode"] == 0
        and not row["timed_out"]
        and not row["controller_terminated"]
        and not row["process_group_alive_after_cleanup"]
        for row in results
    )
    aggregate_cpu_seconds = sum(
        float(row["process_tree_cpu_seconds"]) for row in results
    )
    measured_candidate_cpu_cores = aggregate_cpu_seconds / interval.elapsed_seconds
    aggregate_peak_rss = sum(int(row["peak_rss_bytes"]) for row in results)
    cpu_ok = False
    cpu_details: dict[str, Any]
    if measured_candidate_cpu_cores > 0:
        cpu_ok, cpu_details = cpu.candidate_cpu_capacity_ok(
            binding,
            interval,
            measured_candidate_cpu_cores=measured_candidate_cpu_cores,
            cpu_fraction=cpu_fraction,
            safety_factor=cpu_safety_factor,
        )
    else:
        cpu_details = {"ok": False, "reason": "candidate_cpu_demand_not_observed"}
    memory_ok = aggregate_peak_rss > 0 and aggregate_peak_rss <= usable_memory_bytes
    elapsed = interval.elapsed_seconds
    valid = completed == concurrency and elapsed > 0 and cpu_ok and memory_ok
    summary = {
        "concurrency": concurrency,
        "probe_steps_per_branch": probe_steps,
        "elapsed_seconds": elapsed,
        "completed": completed,
        "failed": concurrency - completed,
        "timed_out": sum(bool(row["timed_out"]) for row in results),
        "controller_terminated": sum(
            bool(row["controller_terminated"]) for row in results
        ),
        "orphan_process_groups": sum(
            bool(row["process_group_alive_after_cleanup"]) for row in results
        ),
        "aggregate_updates_per_second": (
            completed * probe_steps / elapsed if elapsed > 0 else 0.0
        ),
        "aggregate_process_tree_cpu_seconds": aggregate_cpu_seconds,
        "measured_candidate_cpu_cores": measured_candidate_cpu_cores,
        "aggregate_peak_rss_bytes": aggregate_peak_rss,
        "usable_memory_bytes": usable_memory_bytes,
        "cpu_interval": interval.as_dict(),
        "cpu_capacity": cpu_details,
        "cpu_capacity_ok": cpu_ok,
        "memory_capacity_ok": memory_ok,
        "process_results": results,
        "valid": valid,
    }
    atomic_write_json(candidate_root / "BENCHMARK_SUMMARY.json", summary)
    for slot_dir in candidate_root.glob("slot-*"):
        _cleanup_probe_payload(slot_dir)
    return summary


def select_from_throughput(
    benchmarks: list[Mapping[str, Any]], *, retention_fraction: float
) -> tuple[int, dict[str, Any]]:
    if not 0.5 <= retention_fraction <= 1.0:
        raise RuntimeResourceError(
            "throughput_retention_fraction must be in [0.5, 1.0]"
        )
    valid = [row for row in benchmarks if row.get("valid") is True]
    if not valid:
        raise RuntimeResourceError("no resource-valid concurrency candidate completed")
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
        "rule": "smallest_resource_valid_candidate_at_or_above_retained_peak",
    }


def _memory_capacity(
    machine: MachineSnapshot,
    *,
    peak_rss_bytes: int,
    memory_headroom_fraction: float,
    per_worker_safety_factor: float,
) -> tuple[int, int, int]:
    if peak_rss_bytes <= 0:
        raise RuntimeResourceError("representative peak RSS must be positive")
    reserved = max(1, math.ceil(peak_rss_bytes * per_worker_safety_factor))
    usable = max(
        0,
        math.floor(
            machine.effective_memory_available_bytes
            * (1.0 - memory_headroom_fraction)
        ),
    )
    limit = usable // reserved
    if limit < 1:
        raise RuntimeResourceError(
            "insufficient host memory for one worker after safety headroom"
        )
    return limit, reserved, usable


def _selection_digest_payload(document: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": document.get("schema_version"),
        "selector_policy_version": document.get("selector_policy_version"),
        "adapter_id": document.get("adapter_id"),
        "resource_fingerprint_sha256": document.get("resource_fingerprint_sha256"),
        "selector_implementation": document.get("selector_implementation"),
        "source_commit": document.get("source", {}).get("commit"),
        "cpu_binding": document.get("cpu_binding"),
        "selection": document.get("selection"),
        "scientific_matrix_changed": document.get("scientific_matrix_changed"),
    }


def _finalize_selection_document(
    document: dict[str, Any],
    *,
    binding: cpu.CPUBinding,
    repo_root: str | Path,
) -> dict[str, Any]:
    document["schema_version"] = SELECTION_SCHEMA_VERSION
    document["selector_policy_version"] = SELECTOR_POLICY_VERSION
    document["selector_implementation"] = _selector_implementation_identity(repo_root)
    document["cpu_binding"] = binding.as_dict()
    document["load_average_is_diagnostic_only"] = True
    document["selection_digest"] = canonical_json_sha256(
        _selection_digest_payload(document)
    )
    return document


def _validate_policy_inputs(
    *,
    per_worker_cpu_safety_factor: float,
    minimum_cpu_cores_per_worker: float,
    revalidation_samples: int,
    revalidation_sample_seconds: float,
) -> None:
    if per_worker_cpu_safety_factor < 1.0:
        raise RuntimeResourceError("per-worker CPU safety factor must be >= 1")
    if minimum_cpu_cores_per_worker <= 0:
        raise RuntimeResourceError("minimum CPU cores per worker must be positive")
    if revalidation_samples < 1:
        raise RuntimeResourceError("revalidation_samples must be positive")
    if revalidation_sample_seconds <= 0:
        raise RuntimeResourceError("revalidation_sample_seconds must be positive")


def _interval_from_dict(value: Mapping[str, Any]) -> cpu.CPUIntervalMeasurement:
    usage = value.get("quota_domain_usage_cores", {})
    if not isinstance(usage, Mapping):
        raise RuntimeResourceError("CPU interval quota usage must be a mapping")
    return cpu.CPUIntervalMeasurement(
        elapsed_seconds=float(value["elapsed_seconds"]),
        affinity_cpu_ids=tuple(int(item) for item in value["affinity_cpu_ids"]),
        system_busy_tick_delta=int(value["system_busy_tick_delta"]),
        system_total_tick_delta=int(value["system_total_tick_delta"]),
        system_busy_cores=float(value["system_busy_cores"]),
        quota_domain_usage_cores=tuple(
            (str(path), float(cores)) for path, cores in usage.items()
        ),
        started_monotonic_seconds=float(value["started_monotonic_seconds"]),
        finished_monotonic_seconds=float(value["finished_monotonic_seconds"]),
    )


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
    per_worker_cpu_safety_factor: float,
    minimum_cpu_cores_per_worker: float,
    max_workers: int | None,
    max_growth_factor: float,
    minimum_branches_for_probe: int,
    cgroup_root: str | Path = "/sys/fs/cgroup",
    proc_self_cgroup_path: str | Path = "/proc/self/cgroup",
    proc_stat_path: str | Path = "/proc/stat",
    revalidation_samples: int = 3,
    revalidation_sample_seconds: float = 1.0,
) -> dict[str, Any]:
    _validate_policy_inputs(
        per_worker_cpu_safety_factor=per_worker_cpu_safety_factor,
        minimum_cpu_cores_per_worker=minimum_cpu_cores_per_worker,
        revalidation_samples=revalidation_samples,
        revalidation_sample_seconds=revalidation_sample_seconds,
    )
    work = Path(work_dir).resolve()
    selection_path = work / "RUNTIME_SELECTION.json"
    if selection_path.exists():
        raise RuntimeResourceError(
            "RUNTIME_SELECTION.json already exists; use run to consume it or a new "
            "work directory to create another automatic selection"
        )
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
        per_worker_cpu_safety_factor=per_worker_cpu_safety_factor,
        minimum_cpu_cores_per_worker=minimum_cpu_cores_per_worker,
        max_workers=max_workers,
        max_growth_factor=max_growth_factor,
        revalidation_samples=revalidation_samples,
        revalidation_sample_seconds=revalidation_sample_seconds,
    )
    try:
        binding = cpu.discover_cpu_binding(
            cgroup_root=cgroup_root,
            proc_self_cgroup_path=proc_self_cgroup_path,
        )
    except cpu.CPUCapacityError as exc:
        raise RuntimeResourceError(str(exc)) from exc
    command, probe_dir, environment, total_tasks, command_cwd = build_probe_command(
        contract_path=contract_path,
        run_spec_path=run_spec_path,
        grid_path=grid_path,
        probe_root=work / "_runtime_resource_probe" / "w0_resources",
        probe_steps=probe_steps,
        probe_seed=probe_seed,
    )
    try:
        resource_probe = _measure_representative_resources(
            command=command,
            cwd=command_cwd,
            environment=environment,
            log_path=probe_dir / "stdout_stderr.log",
            timeout_seconds=probe_seconds,
            binding=binding,
            proc_stat_path=proc_stat_path,
        )
    except cpu.CPUCapacityError as exc:
        raise RuntimeResourceError(str(exc)) from exc
    finally:
        _cleanup_probe_payload(probe_dir)
    measured_worker_cpu = float(resource_probe["measured_cpu_cores"])
    try:
        reserved_worker_cpu = cpu.reserve_worker_cpu_cores(
            measured_worker_cpu,
            safety_factor=per_worker_cpu_safety_factor,
            minimum_cpu_cores_per_worker=minimum_cpu_cores_per_worker,
        )
        interval = _interval_from_dict(resource_probe["cpu_interval"])
        cpu_capacity = cpu.derive_worker_cpu_capacity(
            binding,
            interval,
            measured_probe_cpu_cores=measured_worker_cpu,
            reserved_cpu_cores_per_worker=reserved_worker_cpu,
            cpu_fraction=cpu_fraction,
        )
    except cpu.CPUCapacityError as exc:
        raise RuntimeResourceError(str(exc)) from exc
    if cpu_capacity.cpu_worker_limit < 1:
        raise RuntimeResourceError("measured CPU capacity cannot support one worker")
    memory_limit, reserved_memory, usable_memory = _memory_capacity(
        machine,
        peak_rss_bytes=int(resource_probe["peak_rss_bytes"]),
        memory_headroom_fraction=memory_headroom_fraction,
        per_worker_safety_factor=per_worker_safety_factor,
    )
    if fallback_workers < 1 or max_growth_factor < 1:
        raise RuntimeResourceError("fallback workers and growth factor must be positive")
    growth_limit = max(1, math.floor(fallback_workers * max_growth_factor))
    limits = [cpu_capacity.cpu_worker_limit, memory_limit, total_tasks, growth_limit]
    if max_workers is not None:
        if max_workers < 1:
            raise RuntimeResourceError("max_workers must be positive")
        limits.append(max_workers)
    safe_cap = min(limits)
    if safe_cap < 1:
        raise RuntimeResourceError("measured CPU/RAM capacity produced no worker slot")
    benchmarks: list[dict[str, Any]] = []
    throughput_rule: dict[str, Any] | None = None
    if total_tasks < minimum_branches_for_probe:
        selected_workers = min(fallback_workers, safe_cap)
        candidates: list[int] = []
        reason = "small_task_measured_capacity_without_throughput_grid"
        mode = "exempt"
    else:
        candidates = candidate_workers(safe_cap, fallback_workers)
        throughput_root = work / "_runtime_resource_probe" / "w0_throughput"
        for index, concurrency in enumerate(candidates):
            try:
                benchmark = benchmark_concurrency(
                    contract_path=contract_path,
                    run_spec_path=run_spec_path,
                    grid_path=grid_path,
                    probe_root=throughput_root,
                    concurrency=concurrency,
                    probe_steps=probe_steps,
                    probe_seed=probe_seed + index * 10_000,
                    timeout_seconds=probe_seconds,
                    binding=binding,
                    proc_stat_path=proc_stat_path,
                    cpu_fraction=cpu_fraction,
                    cpu_safety_factor=per_worker_cpu_safety_factor,
                    usable_memory_bytes=usable_memory,
                )
            except cpu.CPUCapacityError as exc:
                raise RuntimeResourceError(str(exc)) from exc
            benchmarks.append(benchmark)
            if not benchmark["valid"]:
                break
        selected_workers, throughput_rule = select_from_throughput(
            benchmarks, retention_fraction=throughput_retention_fraction
        )
        reason = "resource_valid_short_throughput_grid_under_measured_capacity"
        mode = "auto"
    selection_payload = {
        "selected_workers": selected_workers,
        "safe_capacity_ceiling": safe_cap,
        "cpu_limit": cpu_capacity.cpu_worker_limit,
        "memory_limit": memory_limit,
        "task_limit": total_tasks,
        "configured_limit": max_workers,
        "growth_limit": growth_limit,
        "fallback_workers": fallback_workers,
        "per_worker_peak_bytes": int(resource_probe["peak_rss_bytes"]),
        "per_worker_reserved_bytes": reserved_memory,
        "measured_cpu_cores_per_worker": measured_worker_cpu,
        "per_worker_reserved_cpu_cores": reserved_worker_cpu,
        "cpu_capacity": cpu_capacity.as_dict(),
        "reason": reason,
    }
    document = selection_document(
        adapter_id=ADAPTER_ID,
        resource_fingerprint=fingerprint,
        machine=machine,
        mode=mode,
        selection=selection_payload,
        probe={
            "single_branch_resource_probe": resource_probe,
            "candidate_workers": candidates,
            "throughput_benchmarks": benchmarks,
            "throughput_selection_rule": throughput_rule,
        },
        fallback={"workers": fallback_workers, "reason": "legacy_verified_schedule"},
        repo_root=repo_root,
        limitations=[
            "candidate_grid_not_continuous_global_optimization",
            "probe_uses_one_terminal_eval_episode_not_formal_10_episode_burst",
            "single_representative_ppo_workload_family",
            "load_average_is_diagnostic_only",
        ],
    )
    _finalize_selection_document(document, binding=binding, repo_root=repo_root)
    atomic_write_json(selection_path, document)
    return document


def _binding_from_dict(value: Mapping[str, Any]) -> cpu.CPUBinding:
    domains_value = value.get("quota_domains", [])
    if not isinstance(domains_value, list):
        raise RuntimeResourceError("CPU binding quota_domains must be a list")
    domains = tuple(
        cpu.CPUQuotaDomain(
            path=str(item["path"]),
            quota_cores=float(item["quota_cores"]),
            usage_path=str(item["usage_path"]),
            usage_kind=str(item["usage_kind"]),
            cgroup_version=str(item["cgroup_version"]),
        )
        for item in domains_value
        if isinstance(item, Mapping)
    )
    if len(domains) != len(domains_value):
        raise RuntimeResourceError("CPU binding contains malformed quota domains")
    return cpu.CPUBinding(
        affinity_cpu_ids=tuple(int(item) for item in value["affinity_cpu_ids"]),
        affinity_source=str(value["affinity_source"]),
        cgroup_version=(
            None if value.get("cgroup_version") is None else str(value["cgroup_version"])
        ),
        current_cgroup_path=(
            None
            if value.get("current_cgroup_path") is None
            else str(value["current_cgroup_path"])
        ),
        quota_domains=domains,
    )


def _read_ppid(path: Path) -> int | None:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    close = text.rfind(")")
    if close < 0:
        return None
    fields = text[close + 2 :].split()
    if len(fields) < 2:
        return None
    try:
        return int(fields[1])
    except ValueError:
        return None


def _ancestor_pids(pid: int, proc_root: Path) -> set[int]:
    values = {pid}
    current = pid
    for _ in range(128):
        parent = _read_ppid(proc_root / str(current) / "stat")
        if parent is None or parent <= 0 or parent in values:
            break
        values.add(parent)
        current = parent
    return values


def conflicting_workdir_processes(
    work_dir: str | Path, *, proc_root: str | Path = "/proc"
) -> list[dict[str, Any]]:
    root = Path(proc_root)
    target = str(Path(work_dir).resolve())
    excluded = _ancestor_pids(os.getpid(), root)
    conflicts: list[dict[str, Any]] = []
    try:
        entries = list(root.iterdir())
    except OSError:
        return conflicts
    for entry in entries:
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid in excluded:
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
        except OSError:
            continue
        command = raw.replace(b"\x00", b" ").decode("utf-8", errors="replace").strip()
        if target and target in command:
            conflicts.append({"pid": pid, "command": command})
    return conflicts


def _conservative_pressure(
    samples: Sequence[cpu.CPUIntervalMeasurement],
) -> tuple[float, dict[str, float]]:
    if not samples:
        raise RuntimeResourceError("at least one CPU revalidation sample is required")
    system_busy = max(sample.system_busy_cores for sample in samples)
    paths = samples[0].quota_usage_map().keys()
    if any(sample.quota_usage_map().keys() != paths for sample in samples[1:]):
        raise RuntimeResourceError("quota-domain set changed across revalidation samples")
    domain_busy = {
        path: max(sample.quota_usage_map()[path] for sample in samples)
        for path in paths
    }
    return system_busy, domain_busy


def revalidate_runtime(
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
    per_worker_cpu_safety_factor: float,
    minimum_cpu_cores_per_worker: float,
    max_workers: int | None,
    max_growth_factor: float,
    minimum_branches_for_probe: int,
    cgroup_root: str | Path = "/sys/fs/cgroup",
    proc_self_cgroup_path: str | Path = "/proc/self/cgroup",
    proc_stat_path: str | Path = "/proc/stat",
    proc_root: str | Path = "/proc",
    revalidation_samples: int = 3,
    revalidation_sample_seconds: float = 1.0,
) -> dict[str, Any]:
    del minimum_branches_for_probe
    _validate_policy_inputs(
        per_worker_cpu_safety_factor=per_worker_cpu_safety_factor,
        minimum_cpu_cores_per_worker=minimum_cpu_cores_per_worker,
        revalidation_samples=revalidation_samples,
        revalidation_sample_seconds=revalidation_sample_seconds,
    )
    work = Path(work_dir).resolve()
    selection_path = work / "RUNTIME_SELECTION.json"
    if not selection_path.is_file():
        raise RuntimeResourceError("missing immutable RUNTIME_SELECTION.json; run plan first")
    document = load_json(selection_path)
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
        per_worker_cpu_safety_factor=per_worker_cpu_safety_factor,
        minimum_cpu_cores_per_worker=minimum_cpu_cores_per_worker,
        max_workers=max_workers,
        max_growth_factor=max_growth_factor,
        revalidation_samples=revalidation_samples,
        revalidation_sample_seconds=revalidation_sample_seconds,
    )
    failures: list[str] = []
    if document.get("schema_version") != SELECTION_SCHEMA_VERSION:
        failures.append("selection_schema_version_mismatch")
    if document.get("selector_policy_version") != SELECTOR_POLICY_VERSION:
        failures.append("selector_policy_version_mismatch")
    if document.get("adapter_id") != ADAPTER_ID:
        failures.append("adapter_id_mismatch")
    if document.get("resource_fingerprint_sha256") != canonical_json_sha256(fingerprint):
        failures.append("resource_fingerprint_mismatch")
    if document.get("selector_implementation") != _selector_implementation_identity(
        repo_root
    ):
        failures.append("selector_implementation_mismatch")
    expected_digest = canonical_json_sha256(_selection_digest_payload(document))
    if document.get("selection_digest") != expected_digest:
        failures.append("selection_digest_mismatch")
    current_source = git_state(repo_root)
    stored_source = document.get("source")
    if not isinstance(stored_source, Mapping):
        failures.append("selection_source_missing")
    else:
        if stored_source.get("commit") != current_source.get("commit"):
            failures.append("source_commit_mismatch")
        if current_source.get("dirty") is True:
            failures.append("current_worktree_dirty")
    try:
        current_binding = cpu.discover_cpu_binding(
            cgroup_root=cgroup_root,
            proc_self_cgroup_path=proc_self_cgroup_path,
        )
    except cpu.CPUCapacityError as exc:
        current_binding = None
        failures.append(f"cpu_binding_unavailable:{exc}")
    stored_binding_value = document.get("cpu_binding")
    if not isinstance(stored_binding_value, Mapping):
        stored_binding = None
        failures.append("stored_cpu_binding_missing")
    else:
        try:
            stored_binding = _binding_from_dict(stored_binding_value)
        except (KeyError, TypeError, ValueError, RuntimeResourceError) as exc:
            stored_binding = None
            failures.append(f"stored_cpu_binding_invalid:{exc}")
    if (
        current_binding is not None
        and stored_binding is not None
        and current_binding.as_dict() != stored_binding.as_dict()
    ):
        failures.append("cpu_binding_changed")
    conflicts = conflicting_workdir_processes(work, proc_root=proc_root)
    if conflicts:
        failures.append("conflicting_workdir_processes")
    samples: list[cpu.CPUIntervalMeasurement] = []
    if current_binding is not None and not conflicts:
        try:
            for _ in range(revalidation_samples):
                samples.append(
                    cpu.sample_cpu_interval(
                        current_binding,
                        sample_seconds=revalidation_sample_seconds,
                        proc_stat_path=proc_stat_path,
                    )
                )
        except cpu.CPUCapacityError as exc:
            failures.append(f"cpu_revalidation_measurement_failed:{exc}")
    selection = document.get("selection")
    if not isinstance(selection, Mapping):
        failures.append("selection_payload_missing")
        selected_workers = 0
        reserved_cpu = 0.0
        reserved_memory = 0
    else:
        selected_workers = int(selection.get("selected_workers", 0) or 0)
        reserved_cpu = float(selection.get("per_worker_reserved_cpu_cores", 0.0) or 0.0)
        reserved_memory = int(selection.get("per_worker_reserved_bytes", 0) or 0)
        if selected_workers < 1 or reserved_cpu <= 0 or reserved_memory < 1:
            failures.append("selection_reservations_invalid")
    cpu_details: dict[str, Any] = {}
    if current_binding is not None and samples and selected_workers > 0 and reserved_cpu > 0:
        system_busy, domain_busy = _conservative_pressure(samples)
        planned_cpu = selected_workers * reserved_cpu
        affinity_budget = current_binding.affinity_capacity_cores * cpu_fraction
        affinity_projected = system_busy + planned_cpu
        affinity_ok = affinity_projected <= affinity_budget
        domain_rows: list[dict[str, Any]] = []
        domains_ok = True
        for domain in current_binding.quota_domains:
            budget = domain.quota_cores * cpu_fraction
            observed = domain_busy.get(domain.path)
            if observed is None:
                ok = False
                projected = None
            else:
                projected = observed + planned_cpu
                ok = projected <= budget
            domains_ok = domains_ok and ok
            domain_rows.append(
                {
                    "path": domain.path,
                    "budget_cores": budget,
                    "observed_busy_cores": observed,
                    "planned_worker_cpu_cores": planned_cpu,
                    "projected_total_busy_cores": projected,
                    "ok": ok,
                }
            )
        cpu_details = {
            "samples": [sample.as_dict() for sample in samples],
            "conservative_system_busy_cores": system_busy,
            "conservative_quota_domain_busy_cores": domain_busy,
            "planned_worker_cpu_cores": planned_cpu,
            "affinity_budget_cores": affinity_budget,
            "affinity_projected_total_busy_cores": affinity_projected,
            "affinity_ok": affinity_ok,
            "quota_domains": domain_rows,
            "ok": affinity_ok and domains_ok,
        }
        if not cpu_details["ok"]:
            failures.append("cpu_capacity_changed")
    usable_memory = math.floor(
        machine.effective_memory_available_bytes * (1.0 - memory_headroom_fraction)
    )
    planned_memory = selected_workers * reserved_memory
    memory_ok = selected_workers > 0 and planned_memory <= usable_memory
    if not memory_ok:
        failures.append("memory_capacity_changed")
    attempt_id = f"attempt-{time.time_ns()}"
    attempt_path = (
        work / "_runtime_resource_attempts" / attempt_id / "RUNTIME_REVALIDATION.json"
    )
    record = {
        "schema_version": 1,
        "attempt_id": attempt_id,
        "created_utc": utc_now(),
        "selection_path": str(selection_path),
        "selection_digest": document.get("selection_digest"),
        "selected_workers": selected_workers,
        "source": current_source,
        "current_machine_snapshot": machine.as_dict(),
        "current_cpu_binding": (
            None if current_binding is None else current_binding.as_dict()
        ),
        "stored_cpu_binding": (
            None if stored_binding is None else stored_binding.as_dict()
        ),
        "conflicting_processes": conflicts,
        "cpu_revalidation": cpu_details,
        "memory_revalidation": {
            "usable_memory_bytes": usable_memory,
            "planned_worker_memory_bytes": planned_memory,
            "ok": memory_ok,
        },
        "failures": failures,
        "decision": "ALLOW" if not failures else "BLOCK",
        "scientific_matrix_changed": False,
    }
    atomic_write_json(attempt_path, record)
    if failures:
        raise RuntimeResourceError(
            "RUNTIME_CAPACITY_CHANGED_REPLAN_REQUIRED: " + ",".join(failures)
        )
    result = copy.deepcopy(document)
    result["revalidation"] = {
        "decision": "ALLOW",
        "path": str(attempt_path),
        "attempt_id": attempt_id,
    }
    return result
