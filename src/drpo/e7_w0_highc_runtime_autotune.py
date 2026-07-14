"""Automatic measured CPU/RAM selection adapter for the high-c actor ablation."""

from __future__ import annotations

import contextlib
import math
from pathlib import Path
from typing import Any, Iterator

from drpo import e7_ppo_w0_runtime_autotune as legacy
from drpo import e7_w0_highc_actor as pilot

ADAPTER_ID = "e7_w0_highc_actor_cpu_v2"
REPRESENTATIVE_DATASET = "walker2d-medium-v2"
REPRESENTATIVE_W0 = 1.0
REPRESENTATIVE_COEFFICIENT = 4.0
REPRESENTATIVE_ACTOR_MODE = "ppo_clip"

_ORIGINAL_SELECT_RUNTIME = legacy.select_runtime
_ORIGINAL_REVALIDATE_RUNTIME = legacy.revalidate_runtime
_ORIGINAL_REPRESENTATIVE = legacy._representative  # noqa: SLF001
_ORIGINAL_RESOURCE_FINGERPRINT = legacy.resource_fingerprint
_ORIGINAL_CLEANUP = legacy._cleanup_probe_payload  # noqa: SLF001
_ORIGINAL_SELECTOR_IMPLEMENTATION = legacy._selector_implementation_identity  # noqa: SLF001


def _selector_implementation_identity(repo_root: str | Path) -> dict[str, str]:
    values = dict(_ORIGINAL_SELECTOR_IMPLEMENTATION(repo_root))
    values["e7_w0_highc_runtime_autotune.py"] = legacy._file_sha256(  # noqa: SLF001
        Path(__file__).resolve()
    )
    return values


def _representative(branches: list[Any]) -> Any:
    matches = [
        branch
        for branch in branches
        if branch.dataset.id == REPRESENTATIVE_DATASET
        and branch.seed == pilot.EXPECTED_SEEDS[0]
        and branch.template_values.get("actor_update_mode") == REPRESENTATIVE_ACTOR_MODE
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
        raise legacy.RuntimeResourceError(
            f"expected one representative high-c PPO branch, found {len(matches)}"
        )
    return matches[0]


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
        "src/drpo/e7_w0_highc_actor.py",
        "src/drpo/e7_w0_highc_actor_bootstrap.py",
        "src/drpo/e7_w0_highc_actor_aggregate.py",
        "src/drpo/e7_w0_geometry_diagnostics.py",
        "src/drpo/e7_canonical_ppo_injection.py",
        "src/drpo/e7_canonical_injection.py",
        "src/drpo/e7_canonical_sweep.py",
    )
    return {
        "schema_version": 2,
        "adapter_id": ADAPTER_ID,
        "selector_policy_version": legacy.SELECTOR_POLICY_VERSION,
        "hard_fields": {
            "contract_sha256": legacy._file_sha256(contract_path),  # noqa: SLF001
            "run_spec_sha256": legacy._file_sha256(run_spec_path),  # noqa: SLF001
            "grid_sha256": legacy._file_sha256(grid_path),  # noqa: SLF001
            "source_sha256": {
                path: legacy._file_sha256(repo / path)  # noqa: SLF001
                for path in source_paths
            },
            "representative_actor_update_mode": REPRESENTATIVE_ACTOR_MODE,
            "batch_size": int(pilot._flag_value(argv, "--batch")),
            "optimizer_learning_rate": float(pilot._flag_value(argv, "--lr")),
            "updates_per_old_policy": int(grid["ppo"]["updates_per_old_policy"]),
            "diagnostics_interval": int(grid["ppo"]["diagnostics_interval"]),
            "sampled_values_per_update": int(
                grid["geometry_diagnostics"]["sampled_values_per_update"]
            ),
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
                "actor_update_mode": REPRESENTATIVE_ACTOR_MODE,
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
            "actor_update_mode_grid",
            "exp_coefficient_grid",
            "training_horizon",
        ],
        "tuned_runtime_field": "active_subprocess_count",
        "scientific_matrix_changed": False,
    }


def _cleanup_probe_payload(probe_dir: Path) -> None:
    _ORIGINAL_CLEANUP(probe_dir)
    for name in (
        "geometry_diagnostics.jsonl",
        "GEOMETRY_DIAGNOSTICS_LATEST.json",
    ):
        path = probe_dir / name
        if path.is_file() and not path.is_symlink():
            path.unlink()


@contextlib.contextmanager
def _installed_adapter() -> Iterator[None]:
    previous = (
        legacy.pilot,
        legacy.ADAPTER_ID,
        legacy.REPRESENTATIVE_DATASET,
        legacy.REPRESENTATIVE_W0,
        legacy.REPRESENTATIVE_COEFFICIENT,
        legacy._representative,  # noqa: SLF001
        legacy.resource_fingerprint,
        legacy._cleanup_probe_payload,  # noqa: SLF001
        legacy._selector_implementation_identity,  # noqa: SLF001
    )
    legacy.pilot = pilot
    legacy.ADAPTER_ID = ADAPTER_ID
    legacy.REPRESENTATIVE_DATASET = REPRESENTATIVE_DATASET
    legacy.REPRESENTATIVE_W0 = REPRESENTATIVE_W0
    legacy.REPRESENTATIVE_COEFFICIENT = REPRESENTATIVE_COEFFICIENT
    legacy._representative = _representative  # noqa: SLF001
    legacy.resource_fingerprint = resource_fingerprint
    legacy._cleanup_probe_payload = _cleanup_probe_payload  # noqa: SLF001
    legacy._selector_implementation_identity = (  # noqa: SLF001
        _selector_implementation_identity
    )
    try:
        yield
    finally:
        (
            legacy.pilot,
            legacy.ADAPTER_ID,
            legacy.REPRESENTATIVE_DATASET,
            legacy.REPRESENTATIVE_W0,
            legacy.REPRESENTATIVE_COEFFICIENT,
            legacy._representative,  # noqa: SLF001
            legacy.resource_fingerprint,
            legacy._cleanup_probe_payload,  # noqa: SLF001
            legacy._selector_implementation_identity,  # noqa: SLF001
        ) = previous


def select_runtime(**kwargs: Any) -> dict[str, Any]:
    with _installed_adapter():
        return _ORIGINAL_SELECT_RUNTIME(**kwargs)


def revalidate_runtime(**kwargs: Any) -> dict[str, Any]:
    with _installed_adapter():
        return _ORIGINAL_REVALIDATE_RUNTIME(**kwargs)
