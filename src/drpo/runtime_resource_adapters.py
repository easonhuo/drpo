"""Thin E7/E8 adapters for DRPO runtime resource autotuning."""
from __future__ import annotations

import contextlib
import copy
import dataclasses
import hashlib
import math
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import yaml

from drpo import e7_ppo_w0_runtime_autotune as measured
from drpo import runtime_cpu_capacity as cpu
from drpo.runtime_resource_autotune import (
    GIB,
    GPUSelection,
    MachineSnapshot,
    RuntimeResourceError,
    atomic_write_json,
    canonical_json_sha256,
    load_json,
    measure_command_peak_memory,
    select_gpu_devices,
    selection_document,
)

E7_ADAPTER_ID = "e7_canonical_exp_horizon_cpu_v2"
E8_ADAPTER_ID = "e8_countdown_taper_cuda_v1"
E7_SELECTOR_POLICY_VERSION = 2
E7_SELECTION_SCHEMA_VERSION = 2


def _file_sha256(path: str | Path) -> str:
    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _e7_selector_implementation_identity(repo_root: str | Path) -> dict[str, str]:
    repo = Path(repo_root).resolve()
    return {
        "runtime_resource_adapters.py": _file_sha256(Path(__file__).resolve()),
        "runtime_cpu_capacity.py": _file_sha256(
            repo / "src/drpo/runtime_cpu_capacity.py"
        ),
        "e7_ppo_w0_runtime_autotune.py": _file_sha256(
            repo / "src/drpo/e7_ppo_w0_runtime_autotune.py"
        ),
    }


def e7_resource_fingerprint(
    *,
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
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    del throughput_retention_fraction
    source_hashes: dict[str, str] = {}
    if repo_root is not None:
        repo = Path(repo_root).resolve()
        for relative in (
            "src/drpo/runtime_resource_adapters.py",
            "src/drpo/runtime_cpu_capacity.py",
            "src/drpo/e7_canonical_exp_horizon_grid.py",
            "src/drpo/e7_canonical_sweep.py",
        ):
            source_hashes[relative] = _file_sha256(repo / relative)
    return {
        "schema_version": 2,
        "adapter_id": E7_ADAPTER_ID,
        "selector_policy_version": E7_SELECTOR_POLICY_VERSION,
        "contract_sha256": _file_sha256(contract_path),
        "run_spec_sha256": _file_sha256(run_spec_path),
        "grid_sha256": _file_sha256(grid_path),
        "source_sha256": source_hashes,
        "probe_steps": int(probe_steps),
        "probe_seconds": float(probe_seconds),
        "probe_seed_namespace": int(probe_seed),
        "selection_policy": {
            "fallback_workers": int(fallback_workers),
            "cpu_fraction": float(cpu_fraction),
            "memory_headroom_fraction": float(memory_headroom_fraction),
            "per_worker_memory_safety_factor": float(per_worker_safety_factor),
            "per_worker_cpu_safety_factor": float(per_worker_cpu_safety_factor),
            "minimum_cpu_cores_per_worker": float(minimum_cpu_cores_per_worker),
            "max_workers": None if max_workers is None else int(max_workers),
            "max_growth_factor": float(max_growth_factor),
            "throughput_search": False,
            "revalidation_samples": int(revalidation_samples),
            "revalidation_sample_seconds": float(revalidation_sample_seconds),
            "load_average_role": "diagnostic_only",
        },
        "tuned_runtime_field": "active_subprocess_count",
        "frozen_runtime_fields": [
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "dataloader_workers",
            "cpu_affinity",
            "numa_placement",
        ],
        "scientific_matrix_changed": False,
    }


def e8_resource_fingerprint(
    *,
    sweep_config_path: str | Path,
    base_config_path: str | Path,
    candidate_device_ids: Sequence[str],
    required_free_gpu_memory_bytes: int,
    required_host_memory_bytes_per_device: int,
    gpu_memory_headroom_fraction: float,
    host_memory_headroom_fraction: float,
    maximum_gpu_utilization_percent: float,
    max_devices: int | None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "adapter_id": E8_ADAPTER_ID,
        "sweep_config_sha256": _file_sha256(sweep_config_path),
        "base_config_sha256": _file_sha256(base_config_path),
        "candidate_device_ids": [str(value) for value in candidate_device_ids],
        "required_free_gpu_memory_bytes": int(required_free_gpu_memory_bytes),
        "required_host_memory_bytes_per_device": int(
            required_host_memory_bytes_per_device
        ),
        "gpu_memory_headroom_fraction": float(gpu_memory_headroom_fraction),
        "host_memory_headroom_fraction": float(host_memory_headroom_fraction),
        "maximum_gpu_utilization_percent": float(maximum_gpu_utilization_percent),
        "max_devices": None if max_devices is None else int(max_devices),
        "tuned_runtime_field": "active_gpu_device_slots",
        "max_processes_per_device": 1,
        "frozen_scientific_fields": [
            "micro_batch",
            "gradient_accumulation",
            "parameterization",
            "precision",
            "sequence_length",
            "generation_parameters",
        ],
        "scientific_matrix_changed": False,
    }


def cached_cpu_selection(*_args: Any, **_kwargs: Any) -> int | None:
    """Raw-load-average selections are incompatible with measured-CPU V2."""

    return None


def build_e7_probe_command(
    *,
    contract_path: str | Path,
    run_spec_path: str | Path,
    grid_path: str | Path,
    probe_root: str | Path,
    probe_steps: int,
    probe_seed: int,
) -> tuple[list[str], Path, dict[str, str], int, Path]:
    """Build one isolated representative E7 branch command."""
    if probe_steps < 1:
        raise RuntimeResourceError("probe_steps must be positive")
    from drpo import e7_canonical_exp_horizon_grid as joint
    from drpo import e7_canonical_sweep as base

    contract_source = Path(contract_path).expanduser().resolve()
    contract = base.CanonicalContract.load(contract_source)
    contract.verify_runtime()
    run_spec, _ = joint.load_exp_horizon_run_spec(str(Path(run_spec_path).resolve()))
    grid, _ = joint.load_exp_horizon_grid(str(Path(grid_path).resolve()))
    branches = joint.build_exp_horizon_branches(contract, run_spec, grid)
    if not branches:
        raise RuntimeResourceError("E7 adapter produced no branches")
    representative = branches[0]
    representative.dataset.verify()
    probe_branch = dataclasses.replace(
        representative,
        branch_id=(
            f"resource_probe__seed{probe_seed}__"
            f"{representative.branch_id.split('__', 2)[-1]}"
        ),
        seed=int(probe_seed),
        template_values={**representative.template_values, "steps": str(probe_steps)},
    )
    root = Path(probe_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    branch_dir = Path(tempfile.mkdtemp(prefix="attempt-", dir=root))
    command, _ = base.branch_command(
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


def _cleanup_e7_probe_payload(probe_dir: Path) -> None:
    """Keep small provenance/log files and remove generated model payload."""
    for relative in ("trainer_output", "checkpoints", "checkpoint"):
        target = probe_dir / relative
        if target.is_dir() and not target.is_symlink():
            shutil.rmtree(target)
        elif target.is_file() and not target.is_symlink():
            target.unlink()


def _finalize_e7_selection(
    document: dict[str, Any],
    *,
    binding: cpu.CPUBinding,
    repo_root: str | Path,
) -> dict[str, Any]:
    document["schema_version"] = E7_SELECTION_SCHEMA_VERSION
    document["selector_policy_version"] = E7_SELECTOR_POLICY_VERSION
    document["selector_implementation"] = _e7_selector_implementation_identity(repo_root)
    document["cpu_binding"] = binding.as_dict()
    document["load_average_is_diagnostic_only"] = True
    document["selection_digest"] = canonical_json_sha256(
        measured._selection_digest_payload(document)  # noqa: SLF001
    )
    return document


def select_e7_runtime(
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
    throughput_retention_fraction: float = 1.0,
) -> dict[str, Any]:
    del minimum_branches_for_probe
    work = Path(work_dir).resolve()
    selection_path = work / "RUNTIME_SELECTION.json"
    if selection_path.exists():
        raise RuntimeResourceError(
            "RUNTIME_SELECTION.json already exists; use run to consume it or a new "
            "work directory to create another automatic selection"
        )
    fingerprint = e7_resource_fingerprint(
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
    command, probe_dir, environment, total_tasks, command_cwd = build_e7_probe_command(
        contract_path=contract_path,
        run_spec_path=run_spec_path,
        grid_path=grid_path,
        probe_root=work / "_runtime_resource_probe" / "e7_resources",
        probe_steps=probe_steps,
        probe_seed=probe_seed,
    )
    try:
        resource_probe = measured._measure_representative_resources(  # noqa: SLF001
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
        _cleanup_e7_probe_payload(probe_dir)
    measured_worker_cpu = float(resource_probe["measured_cpu_cores"])
    try:
        reserved_worker_cpu = cpu.reserve_worker_cpu_cores(
            measured_worker_cpu,
            safety_factor=per_worker_cpu_safety_factor,
            minimum_cpu_cores_per_worker=minimum_cpu_cores_per_worker,
        )
        interval = measured._interval_from_dict(  # noqa: SLF001
            resource_probe["cpu_interval"]
        )
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
        raise RuntimeResourceError("measured CPU capacity cannot support one E7 worker")
    memory_limit, reserved_memory, _usable_memory = measured._memory_capacity(  # noqa: SLF001
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
    selected_workers = min(limits)
    if selected_workers < 1:
        raise RuntimeResourceError("measured CPU/RAM capacity produced no E7 worker")
    selection_payload = {
        "selected_workers": selected_workers,
        "safe_capacity_ceiling": selected_workers,
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
        "reason": "measured_cpu_ram_safe_capacity_without_throughput_grid",
    }
    document = selection_document(
        adapter_id=E7_ADAPTER_ID,
        resource_fingerprint=fingerprint,
        machine=machine,
        mode="auto",
        selection=selection_payload,
        probe={"single_branch_resource_probe": resource_probe},
        fallback={"workers": fallback_workers, "reason": "legacy_verified_schedule"},
        repo_root=repo_root,
        limitations=[
            "capacity_guard_not_throughput_knee_search",
            "single_representative_branch_resource_probe",
            "load_average_is_diagnostic_only",
        ],
    )
    _finalize_e7_selection(document, binding=binding, repo_root=repo_root)
    atomic_write_json(selection_path, document)
    return document


@contextlib.contextmanager
def _installed_e7_revalidation_adapter() -> Iterator[None]:
    previous = (
        measured.ADAPTER_ID,
        measured.resource_fingerprint,
        measured._selector_implementation_identity,  # noqa: SLF001
    )
    measured.ADAPTER_ID = E7_ADAPTER_ID
    measured.resource_fingerprint = e7_resource_fingerprint
    measured._selector_implementation_identity = (  # noqa: SLF001
        _e7_selector_implementation_identity
    )
    try:
        yield
    finally:
        (
            measured.ADAPTER_ID,
            measured.resource_fingerprint,
            measured._selector_implementation_identity,  # noqa: SLF001
        ) = previous


def revalidate_e7_runtime(**kwargs: Any) -> dict[str, Any]:
    kwargs.setdefault("throughput_retention_fraction", 1.0)
    with _installed_e7_revalidation_adapter():
        return measured.revalidate_runtime(**kwargs)


def validate_e8_scientific_config(config: Mapping[str, Any]) -> None:
    """Validate the frozen E8 config before selecting only runtime GPU slots."""
    from drpo import countdown_e8_oracle_offline_v2_taper_sweep as core

    core.validate_sweep_config(config)
    if int(config["execution"]["parallel_cells_per_gpu"]) != 1:
        raise RuntimeResourceError("E8 V1 requires one process per GPU")


def e8_parent_runtime_config(
    config: Mapping[str, Any], *, selected_gpu_count: int
) -> dict[str, Any]:
    """Create an in-memory parent-only config view for the selected GPU pool."""
    if selected_gpu_count < 1 or selected_gpu_count > 8:
        raise RuntimeResourceError("selected E8 GPU count must be in [1, 8]")
    validate_e8_scientific_config(config)
    runtime_config = copy.deepcopy(dict(config))
    execution = dict(runtime_config["execution"])
    execution["required_gpu_count"] = int(selected_gpu_count)
    execution["runtime_resource_override"] = {
        "adapter_id": E8_ADAPTER_ID,
        "field": "active_gpu_device_slots",
        "scientific_matrix_changed": False,
    }
    runtime_config["execution"] = execution
    return runtime_config


def validate_e8_parent_runtime_config(
    config: Mapping[str, Any],
    *,
    original_validator: Callable[[Mapping[str, Any]], None],
) -> None:
    selected_count = int(config["execution"]["required_gpu_count"])
    if selected_count < 1 or selected_count > 8:
        raise RuntimeResourceError("runtime E8 GPU count must be in [1, 8]")
    override = config["execution"].get("runtime_resource_override")
    if not isinstance(override, Mapping) or override.get("adapter_id") != E8_ADAPTER_ID:
        raise RuntimeResourceError("E8 runtime override marker is missing or invalid")
    normalized = copy.deepcopy(dict(config))
    execution = dict(normalized["execution"])
    execution["required_gpu_count"] = 8
    execution.pop("runtime_resource_override", None)
    normalized["execution"] = execution
    original_validator(normalized)


def _archive_runtime_selection(selection_path: Path) -> None:
    if not selection_path.is_file():
        return
    try:
        existing = load_json(selection_path)
    except Exception:  # noqa: BLE001
        existing = {"unreadable_previous_selection": True}
    history = selection_path.parent / "_runtime_resources" / "selection_history"
    history.mkdir(parents=True, exist_ok=True)
    atomic_write_json(history / f"RUNTIME_SELECTION.{time.time_ns()}.json", existing)


def select_e8_runtime(
    *,
    machine: MachineSnapshot,
    repo_root: str | Path,
    sweep_config_path: str | Path,
    base_config_path: str | Path,
    work_dir: str | Path,
    candidate_device_ids: Sequence[str],
    total_tasks: int,
    required_free_gpu_memory_gib: float,
    required_host_memory_gib_per_device: float,
    gpu_memory_headroom_fraction: float,
    host_memory_headroom_fraction: float,
    maximum_gpu_utilization_percent: float,
    max_devices: int | None,
) -> dict[str, Any]:
    if required_free_gpu_memory_gib <= 0:
        raise RuntimeResourceError("required_free_gpu_memory_gib must be positive")
    if required_host_memory_gib_per_device <= 0:
        raise RuntimeResourceError(
            "required_host_memory_gib_per_device must be positive"
        )
    if not 0.0 <= host_memory_headroom_fraction < 0.9:
        raise RuntimeResourceError("host_memory_headroom_fraction must be in [0, 0.9)")

    config = yaml.safe_load(Path(sweep_config_path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise RuntimeResourceError("E8 sweep config root must be a mapping")
    validate_e8_scientific_config(config)

    required_gpu_bytes = int(required_free_gpu_memory_gib * GIB)
    required_host_bytes = int(required_host_memory_gib_per_device * GIB)
    usable_host_bytes = int(
        machine.effective_memory_available_bytes * (1.0 - host_memory_headroom_fraction)
    )
    host_memory_slot_limit = usable_host_bytes // required_host_bytes
    if host_memory_slot_limit < 1:
        raise RuntimeResourceError(
            "insufficient host memory for one E8 GPU worker after safety headroom"
        )
    effective_max_devices = host_memory_slot_limit
    if max_devices is not None:
        effective_max_devices = min(effective_max_devices, max_devices)

    selection: GPUSelection = select_gpu_devices(
        machine,
        candidate_device_ids=candidate_device_ids,
        total_tasks=total_tasks,
        required_free_bytes_per_device=required_gpu_bytes,
        headroom_fraction=gpu_memory_headroom_fraction,
        maximum_utilization_percent=maximum_gpu_utilization_percent,
        max_devices=effective_max_devices,
    )
    selection_payload = selection.as_dict()
    selection_payload.update(
        {
            "host_memory_slot_limit": host_memory_slot_limit,
            "required_host_memory_bytes_per_device": required_host_bytes,
            "host_memory_headroom_fraction": host_memory_headroom_fraction,
        }
    )
    fingerprint = e8_resource_fingerprint(
        sweep_config_path=sweep_config_path,
        base_config_path=base_config_path,
        candidate_device_ids=candidate_device_ids,
        required_free_gpu_memory_bytes=required_gpu_bytes,
        required_host_memory_bytes_per_device=required_host_bytes,
        gpu_memory_headroom_fraction=gpu_memory_headroom_fraction,
        host_memory_headroom_fraction=host_memory_headroom_fraction,
        maximum_gpu_utilization_percent=maximum_gpu_utilization_percent,
        max_devices=max_devices,
    )
    work = Path(work_dir).resolve()
    document = selection_document(
        adapter_id=E8_ADAPTER_ID,
        resource_fingerprint=fingerprint,
        machine=machine,
        mode="auto",
        selection=selection_payload,
        probe={
            "kind": "nvidia_smi_visibility_utilization_and_free_memory_gate",
            "dynamic_training_probe": False,
            "phase_peak_validation_required_before_default_cutover": True,
        },
        fallback={
            "device_ids": [str(value) for value in candidate_device_ids],
            "reason": "original_frozen_eight_gpu_schedule",
        },
        repo_root=repo_root,
        limitations=[
            "one_process_per_gpu_only",
            "configured_vram_floor_not_dynamic_training_peak_probe",
            "opt_in_only_no_default_cutover",
        ],
    )
    selection_path = work / "RUNTIME_SELECTION.json"
    _archive_runtime_selection(selection_path)
    atomic_write_json(selection_path, document)
    return document
