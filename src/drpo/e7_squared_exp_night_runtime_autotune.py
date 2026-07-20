"""Automatic measured CPU/RAM selection for the shared squared-night runner."""

from __future__ import annotations

import contextlib
import contextvars
import math
from pathlib import Path
from typing import Any, Iterator

from drpo import e7_ppo_w0_runtime_autotune as legacy
from drpo import e7_squared_exp_night as pilot


SELECTOR_POLICY_VERSION = 3
PROBE_STEPS_LIMIT = 5_000
PROBE_SECONDS_LIMIT = 120.0
CANDIDATE_GROWTH_FACTOR = 1.75
CANDIDATE_POLICY = "low_first_geometric_v3"

_REQUESTED_PROBE_STEPS: contextvars.ContextVar[int | None] = contextvars.ContextVar(
    "e7_squared_exp_requested_probe_steps",
    default=None,
)
_REQUESTED_PROBE_SECONDS: contextvars.ContextVar[float | None] = contextvars.ContextVar(
    "e7_squared_exp_requested_probe_seconds",
    default=None,
)

_ORIGINAL_SELECT_RUNTIME = legacy.select_runtime
_ORIGINAL_REVALIDATE_RUNTIME = legacy.revalidate_runtime
_ORIGINAL_CLEANUP = legacy._cleanup_probe_payload  # noqa: SLF001
_ORIGINAL_SELECTOR_IMPLEMENTATION = (  # noqa: SLF001
    legacy._selector_implementation_identity
)
_ORIGINAL_BENCHMARK_CONCURRENCY = legacy.benchmark_concurrency


def _v3_adapter_id(profile: dict[str, Any]) -> str:
    adapter_id = str(profile["adapter_id"])
    if not adapter_id.endswith("_v2"):
        raise legacy.RuntimeResourceError(
            f"squared-night V3 requires a V2 predecessor adapter id: {adapter_id}"
        )
    return f"{adapter_id[:-3]}_v3"


def _selector_implementation_identity(repo_root: str | Path) -> dict[str, str]:
    values = dict(_ORIGINAL_SELECTOR_IMPLEMENTATION(repo_root))
    values["e7_squared_exp_night_runtime_autotune.py"] = legacy._file_sha256(  # noqa: SLF001
        Path(__file__).resolve()
    )
    return values


class _PilotProxy:
    EXPECTED_SEEDS = pilot.EXPECTED_SEEDS
    load_w0_run_spec = staticmethod(pilot.load_run_spec)
    load_w0_grid = staticmethod(pilot.load_grid)
    build_w0_branches = staticmethod(pilot.build_branches)
    w0_branch_command = staticmethod(pilot.branch_command)
    _flag_value = staticmethod(pilot._flag_value)  # noqa: SLF001


PROXY = _PilotProxy()


def _matches_profile(branch: Any, profile: dict[str, Any]) -> bool:
    values = branch.template_values
    if branch.dataset.id != profile["dataset"] or branch.seed != int(profile["seed"]):
        return False
    if values.get("actor_update_mode") != profile["actor_update_mode"]:
        return False
    if not math.isclose(
        float(values["weight_at_zero"]),
        float(profile["weight_at_zero"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        return False
    if not math.isclose(
        float(values["exp_coefficient"]),
        float(profile["exp_coefficient"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        return False
    expected_estimator = profile.get("advantage_estimator")
    return expected_estimator in {None, "one_step_td"} or values.get(
        "advantage_estimator"
    ) == expected_estimator


def _representative(branches: list[Any]) -> Any:
    profile = pilot.active_runtime_profile()
    matches = [branch for branch in branches if _matches_profile(branch, profile)]
    if len(matches) != 1:
        raise legacy.RuntimeResourceError(
            "expected one representative squared-night branch for "
            f"{profile}, found {len(matches)}"
        )
    return matches[0]


def _bounded_probe_policy(probe_steps: int, probe_seconds: float) -> dict[str, int | float]:
    if probe_steps < 1:
        raise legacy.RuntimeResourceError("probe_steps must be positive")
    if not math.isfinite(probe_seconds) or probe_seconds <= 0:
        raise legacy.RuntimeResourceError("probe_seconds must be finite and positive")
    return {
        "requested_probe_steps": int(probe_steps),
        "effective_probe_steps": min(int(probe_steps), PROBE_STEPS_LIMIT),
        "requested_probe_seconds": float(probe_seconds),
        "effective_probe_seconds": min(float(probe_seconds), PROBE_SECONDS_LIMIT),
    }


def _low_first_candidate_workers(safe_cap: int, fallback_workers: int) -> list[int]:
    """Return a bounded ascending grid that always has a one-worker foothold."""

    if safe_cap < 1:
        raise legacy.RuntimeResourceError("safe worker cap must be positive")
    values = {1, int(safe_cap)}
    candidate = 1
    while candidate < safe_cap:
        candidate = min(
            int(safe_cap),
            max(candidate + 1, math.ceil(candidate * CANDIDATE_GROWTH_FACTOR)),
        )
        values.add(candidate)
    if 1 <= fallback_workers <= safe_cap:
        values.add(int(fallback_workers))
    return sorted(values)


def _bounded_benchmark_concurrency(**kwargs: Any) -> dict[str, Any]:
    """Keep candidate measurement bounded even when legacy callers request longer probes."""

    policy = _bounded_probe_policy(
        int(kwargs.get("probe_steps", 0) or 0),
        float(kwargs.get("timeout_seconds", 0.0) or 0.0),
    )
    requested_steps = _REQUESTED_PROBE_STEPS.get()
    requested_seconds = _REQUESTED_PROBE_SECONDS.get()
    if requested_steps is None:
        requested_steps = int(policy["requested_probe_steps"])
    if requested_seconds is None:
        requested_seconds = float(policy["requested_probe_seconds"])
    effective_steps = int(policy["effective_probe_steps"])
    effective_seconds = float(policy["effective_probe_seconds"])
    bounded = dict(kwargs)
    bounded["probe_steps"] = effective_steps
    bounded["timeout_seconds"] = effective_seconds
    summary = dict(_ORIGINAL_BENCHMARK_CONCURRENCY(**bounded))
    summary.update(
        {
            "requested_probe_steps_per_branch": requested_steps,
            "effective_probe_steps_per_branch": effective_steps,
            "requested_timeout_seconds": requested_seconds,
            "effective_timeout_seconds": effective_seconds,
            "probe_policy_bounded_by_adapter": (
                effective_steps != requested_steps
                or not math.isclose(effective_seconds, requested_seconds)
            ),
            "candidate_policy": CANDIDATE_POLICY,
        }
    )
    candidate_root = Path(str(kwargs["probe_root"])) / (
        f"workers-{int(kwargs['concurrency']):03d}"
    )
    legacy.atomic_write_json(candidate_root / "BENCHMARK_SUMMARY.json", summary)
    return summary


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
    run_spec, _ = pilot.load_run_spec(run_spec_path)
    pilot.load_grid(grid_path)
    argv = [str(value) for value in run_spec["trainer_argv_template"]]
    profile = pilot.active_runtime_profile()
    source_paths = (
        "src/drpo/e7_squared_exp_kernel.py",
        "src/drpo/e7_ppo_kl_refresh.py",
        "src/drpo/e7_squared_exp_night.py",
        "src/drpo/e7_squared_exp_night_bootstrap.py",
        "src/drpo/e7_squared_exp_night_aggregate.py",
        "src/drpo/e7_w0_geometry_diagnostics.py",
        "src/drpo/e7_canonical_ppo_injection.py",
        "src/drpo/e7_canonical_injection.py",
        "src/drpo/e7_canonical_sweep.py",
    )
    hard_fields: dict[str, Any] = {
        "contract_sha256": legacy._file_sha256(contract_path),  # noqa: SLF001
        "run_spec_sha256": legacy._file_sha256(run_spec_path),  # noqa: SLF001
        "grid_sha256": legacy._file_sha256(grid_path),  # noqa: SLF001
        "source_sha256": {
            path: legacy._file_sha256(repo / path)  # noqa: SLF001
            for path in source_paths
        },
        "representative_actor_update_mode": profile["actor_update_mode"],
        "representative_advantage_estimator": profile["advantage_estimator"],
        "batch_size": int(pilot._flag_value(argv, "--batch")),  # noqa: SLF001
        "optimizer_learning_rate": float(
            pilot._flag_value(argv, "--lr")  # noqa: SLF001
        ),
        "diagnostics_interval": pilot.DIAGNOSTICS_INTERVAL,
        "sampled_values_per_update": pilot.SAMPLED_VALUES_PER_UPDATE,
        "thread_environment": {
            name: str(run_spec.get("environment", {}).get(name))
            for name in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
            )
        },
        "representative_workload": {
            key: value
            for key, value in profile.items()
            if key != "adapter_id"
        },
    }
    for key in (
        "clip_epsilon",
        "max_updates_per_old_policy",
        "target_kl",
        "gae_lambda",
    ):
        if key in profile:
            hard_fields[key] = profile[key]
    return {
        "schema_version": 2,
        "adapter_id": _v3_adapter_id(profile),
        "selector_policy_version": legacy.SELECTOR_POLICY_VERSION,
        "hard_fields": hard_fields,
        "soft_fields": {
            "formal_evaluation_interval": int(
                pilot._flag_value(argv, "--eval_interval")  # noqa: SLF001
            ),
            "formal_evaluation_episodes": int(
                pilot._flag_value(argv, "--eval_episodes")  # noqa: SLF001
            ),
            "probe_terminal_evaluation_episodes": 1,
            "effective_probe_steps": int(probe_steps),
            "probe_steps_limit": PROBE_STEPS_LIMIT,
            "effective_probe_seconds": float(probe_seconds),
            "probe_seconds_limit": PROBE_SECONDS_LIMIT,
            "probe_seed_namespace": int(probe_seed),
        },
        "selection_policy": {
            "fallback_workers": int(fallback_workers),
            "fallback_role": "bounded_candidate_only",
            "candidate_policy": CANDIDATE_POLICY,
            "candidate_growth_factor": CANDIDATE_GROWTH_FACTOR,
            "candidate_one_worker_foothold": True,
            "cpu_fraction": float(cpu_fraction),
            "memory_headroom_fraction": float(memory_headroom_fraction),
            "per_worker_memory_safety_factor": float(per_worker_safety_factor),
            "per_worker_cpu_safety_factor": float(per_worker_cpu_safety_factor),
            "minimum_cpu_cores_per_worker": float(minimum_cpu_cores_per_worker),
            "max_workers": None if max_workers is None else int(max_workers),
            "max_workers_role": "optional_absolute_ceiling",
            "max_growth_factor": float(max_growth_factor),
            "throughput_retention_fraction": float(throughput_retention_fraction),
            "revalidation_samples": int(revalidation_samples),
            "revalidation_sample_seconds": float(revalidation_sample_seconds),
            "load_average_role": "diagnostic_only",
        },
        "ignored_scientific_coordinates": [
            "development_seed_values",
            "actor_update_mode_grid",
            "advantage_estimator_grid",
            "squared_exp_coefficient_grid",
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
        "ppo_kl_diagnostics.jsonl",
        "PPO_KL_DIAGNOSTICS_LATEST.json",
    ):
        path = probe_dir / name
        if path.is_file() and not path.is_symlink():
            path.unlink()


@contextlib.contextmanager
def _installed_adapter() -> Iterator[None]:
    profile = pilot.active_runtime_profile()
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
        legacy.candidate_workers,
        legacy.benchmark_concurrency,
        legacy.SELECTOR_POLICY_VERSION,
    )
    legacy.pilot = PROXY
    legacy.ADAPTER_ID = _v3_adapter_id(profile)
    legacy.REPRESENTATIVE_DATASET = str(profile["dataset"])
    legacy.REPRESENTATIVE_W0 = float(profile["weight_at_zero"])
    legacy.REPRESENTATIVE_COEFFICIENT = float(profile["exp_coefficient"])
    legacy._representative = _representative  # noqa: SLF001
    legacy.resource_fingerprint = resource_fingerprint
    legacy._cleanup_probe_payload = _cleanup_probe_payload  # noqa: SLF001
    legacy._selector_implementation_identity = (  # noqa: SLF001
        _selector_implementation_identity
    )
    legacy.candidate_workers = _low_first_candidate_workers
    legacy.benchmark_concurrency = _bounded_benchmark_concurrency
    legacy.SELECTOR_POLICY_VERSION = SELECTOR_POLICY_VERSION
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
            legacy.candidate_workers,
            legacy.benchmark_concurrency,
            legacy.SELECTOR_POLICY_VERSION,
        ) = previous


@contextlib.contextmanager
def _requested_probe_context(probe_steps: int, probe_seconds: float) -> Iterator[None]:
    steps_token = _REQUESTED_PROBE_STEPS.set(int(probe_steps))
    seconds_token = _REQUESTED_PROBE_SECONDS.set(float(probe_seconds))
    try:
        yield
    finally:
        _REQUESTED_PROBE_STEPS.reset(steps_token)
        _REQUESTED_PROBE_SECONDS.reset(seconds_token)


def _bounded_runtime_kwargs(kwargs: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    policy = _bounded_probe_policy(
        int(kwargs.get("probe_steps", 0) or 0),
        float(kwargs.get("probe_seconds", 0.0) or 0.0),
    )
    bounded = dict(kwargs)
    bounded["probe_steps"] = int(policy["effective_probe_steps"])
    bounded["probe_seconds"] = float(policy["effective_probe_seconds"])
    return bounded, policy


def _attach_requested_probe_policy(
    document: dict[str, Any],
    *,
    work_dir: str | Path,
    policy: dict[str, Any],
) -> dict[str, Any]:
    document["requested_probe_policy"] = {
        "requested_probe_steps": int(policy["requested_probe_steps"]),
        "effective_probe_steps": int(policy["effective_probe_steps"]),
        "requested_probe_seconds": float(policy["requested_probe_seconds"]),
        "effective_probe_seconds": float(policy["effective_probe_seconds"]),
        "identity_affecting": False,
    }
    legacy.atomic_write_json(Path(work_dir) / "RUNTIME_SELECTION.json", document)
    return document


def select_runtime(**kwargs: Any) -> dict[str, Any]:
    bounded, policy = _bounded_runtime_kwargs(kwargs)
    with _requested_probe_context(
        int(policy["requested_probe_steps"]),
        float(policy["requested_probe_seconds"]),
    ):
        with _installed_adapter():
            document = _ORIGINAL_SELECT_RUNTIME(**bounded)
    return _attach_requested_probe_policy(
        document,
        work_dir=str(kwargs["work_dir"]),
        policy=policy,
    )


def revalidate_runtime(**kwargs: Any) -> dict[str, Any]:
    bounded, policy = _bounded_runtime_kwargs(kwargs)
    with _requested_probe_context(
        int(policy["requested_probe_steps"]),
        float(policy["requested_probe_seconds"]),
    ):
        with _installed_adapter():
            return _ORIGINAL_REVALIDATE_RUNTIME(**bounded)
