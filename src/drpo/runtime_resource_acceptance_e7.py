"""Safe E7 validate-only and selected-count liveness actions."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

from drpo import runtime_cpu_capacity as cpu
from drpo.e7_ppo_w0_runtime_autotune import benchmark_concurrency, revalidate_runtime
from drpo.runtime_resource_acceptance import AcceptanceError
from drpo.runtime_resource_autotune import atomic_write_json, discover_machine, load_json


def selection_identity(work_dir: Path) -> tuple[int, str]:
    selection_document = load_json(work_dir / "RUNTIME_SELECTION.json")
    selection = selection_document.get("selection")
    if not isinstance(selection, Mapping):
        raise AcceptanceError("E7 runtime selection payload is missing")
    workers = int(selection.get("selected_workers", 0) or 0)
    digest = selection_document.get("selection_digest")
    if workers < 1 or not isinstance(digest, str) or not digest:
        raise AcceptanceError("E7 runtime selection identity is malformed")
    run_identity = load_json(work_dir / "RUN_IDENTITY.json")
    plan = run_identity.get("plan")
    binding = run_identity.get("runtime_resource_selection")
    if not isinstance(plan, Mapping) or int(plan.get("max_workers", 0) or 0) != workers:
        raise AcceptanceError("E7 RUN_IDENTITY worker count mismatch")
    if not isinstance(binding, Mapping):
        raise AcceptanceError("E7 RUN_IDENTITY lacks runtime selection binding")
    if int(binding.get("selected_workers", 0) or 0) != workers:
        raise AcceptanceError("E7 RUN_IDENTITY selected_workers mismatch")
    if binding.get("selection_digest") != digest:
        raise AcceptanceError("E7 RUN_IDENTITY selection digest mismatch")
    return workers, digest


def runtime_kwargs(
    profile: Mapping[str, Any], repo: Path, work_dir: Path, machine: Any
) -> dict[str, Any]:
    e7 = profile["e7"]
    return {
        "machine": machine,
        "repo_root": repo,
        "contract_path": e7["contract"],
        "run_spec_path": e7["run_spec"],
        "grid_path": e7["grid"],
        "work_dir": work_dir,
        "fallback_workers": int(e7["fallback_workers"]),
        "probe_steps": int(e7["probe_steps"]),
        "probe_seed": int(e7["probe_seed"]),
        "probe_seconds": float(e7["probe_seconds"]),
        "throughput_retention_fraction": float(e7["throughput_retention_fraction"]),
        "cpu_fraction": float(e7["cpu_fraction"]),
        "memory_headroom_fraction": float(e7["memory_headroom_fraction"]),
        "per_worker_safety_factor": float(e7["per_worker_safety_factor"]),
        "per_worker_cpu_safety_factor": float(e7["per_worker_cpu_safety_factor"]),
        "minimum_cpu_cores_per_worker": float(e7["minimum_cpu_cores_per_worker"]),
        "max_workers": e7["max_workers"],
        "max_growth_factor": float(e7["max_growth_factor"]),
        "minimum_branches_for_probe": int(e7["minimum_branches_for_probe"]),
        "cgroup_root": "/sys/fs/cgroup",
        "proc_self_cgroup_path": "/proc/self/cgroup",
        "proc_stat_path": "/proc/stat",
        "revalidation_samples": int(e7["revalidation_samples"]),
        "revalidation_sample_seconds": float(e7["revalidation_sample_seconds"]),
    }


def revalidate_only(
    profile: Mapping[str, Any], repo: Path, work_dir: Path, output: Path
) -> dict[str, Any]:
    workers, digest = selection_identity(work_dir)
    document = revalidate_runtime(
        **runtime_kwargs(profile, repo, work_dir, discover_machine()),
        proc_root="/proc",
    )
    if int(document["selection"]["selected_workers"]) != workers:
        raise AcceptanceError("E7 revalidation changed selected worker count")
    if str(document["selection_digest"]) != digest:
        raise AcceptanceError("E7 revalidation changed selection digest")
    payload = {
        "status": "PASS",
        "selected_workers": workers,
        "selection_digest": digest,
        "revalidation": document.get("revalidation"),
        "scientific_matrix_changed": False,
    }
    atomic_write_json(output, payload)
    return payload


def selected_liveness(
    profile: Mapping[str, Any], repo: Path, work_dir: Path, output: Path
) -> dict[str, Any]:
    workers, digest = selection_identity(work_dir)
    machine = discover_machine()
    document = revalidate_runtime(
        **runtime_kwargs(profile, repo, work_dir, machine),
        proc_root="/proc",
    )
    if int(document["selection"]["selected_workers"]) != workers:
        raise AcceptanceError("E7 liveness revalidation changed selected worker count")
    if str(document["selection_digest"]) != digest:
        raise AcceptanceError("E7 liveness revalidation changed selection digest")
    e7 = profile["e7"]
    usable_memory = math.floor(
        machine.effective_memory_available_bytes
        * (1.0 - float(e7["memory_headroom_fraction"]))
    )
    benchmark = benchmark_concurrency(
        contract_path=e7["contract"],
        run_spec_path=e7["run_spec"],
        grid_path=e7["grid"],
        probe_root=output.parent / "selected_liveness_probe",
        concurrency=workers,
        probe_steps=int(e7["liveness_steps"]),
        probe_seed=int(e7["liveness_seed"]),
        timeout_seconds=float(e7["liveness_timeout_seconds"]),
        binding=cpu.discover_cpu_binding(),
        proc_stat_path="/proc/stat",
        cpu_fraction=float(e7["cpu_fraction"]),
        cpu_safety_factor=float(e7["per_worker_cpu_safety_factor"]),
        usable_memory_bytes=usable_memory,
    )
    payload = {
        "status": "PASS" if benchmark.get("valid") is True else "FAIL",
        "selected_workers": workers,
        "selection_digest": digest,
        "revalidation": document.get("revalidation"),
        "benchmark": benchmark,
        "non_scientific_seed_namespace": int(e7["liveness_seed"]),
        "liveness_steps_per_worker": int(e7["liveness_steps"]),
        "full_scientific_matrix_started": False,
        "scientific_matrix_changed": False,
    }
    atomic_write_json(output, payload)
    if benchmark.get("valid") is not True:
        raise AcceptanceError("selected-count E7 liveness was not resource-valid")
    return payload
