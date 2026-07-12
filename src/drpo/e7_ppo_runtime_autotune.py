"""PPO-specific CPU capacity adapter for EXT-H-E7-PPO-STABILITY-01.

This module applies the existing conservative runtime-resource V1 contract to
one representative PPO branch.  It selects a safe active subprocess count from
CPU availability and measured peak host memory.  It does not change the frozen
scientific matrix and does not claim to locate the globally throughput-optimal
worker count.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import shutil
import tempfile
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
            "probe_steps": int(probe_steps),
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
    peak = probe.get("peak_rss_bytes")
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
        trainer_argv_template=[str(item) for item in run_spec["trainer_argv_template"]],
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

    selection: CPUSelection = select_cpu_workers(
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
    document = selection_document(
        adapter_id=ADAPTER_ID,
        resource_fingerprint=fingerprint,
        machine=machine,
        mode="auto",
        selection=selection.as_dict(),
        probe=probe.as_dict(),
        fallback={"workers": fallback_workers, "reason": "legacy_verified_schedule"},
        repo_root=repo_root,
        limitations=[
            "capacity_guard_not_throughput_knee_search",
            "single_representative_ppo_branch_memory_probe",
            "selected_workers_is_safe_capacity_not_global_throughput_optimum",
        ],
    )
    atomic_write_json(selection_path, document)
    return document
