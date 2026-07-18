#!/usr/bin/env python3
"""Execute a claimed RunSpec, package artifacts, and optionally deliver results."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

from runspec_delivery_policy import (
    RESULT_TOO_LARGE,
    is_result_too_large_error,
    record_result_too_large,
    validate_simple_size_policy,
)
from runspec_lib import (
    CLAIMED_DIR,
    DONE_DIR,
    FAILED_DIR,
    RUNNING_DIR,
    STATE_DIRNAME,
    RunSpecError,
    handle_cli_error,
    json_main,
    move_state,
    read_yaml,
    state_path,
)
from runspec_recovery import run_entrypoint_with_recovery, validate_recovery_policy
from runspec_registration import validate_registration_block
from runspec_results_delivery import validate_delivery_block
from runspec_safety import package_artifacts_safe, validate_provenance


def add_runtime_resource_args(parser: argparse.ArgumentParser) -> None:
    """Add opt-in runtime placement arguments shared by RunSpec entrypoints."""

    parser.add_argument(
        "--cpu-pool",
        default=None,
        help="Linux CPU-list syntax, for example 0-31,64-95",
    )
    parser.add_argument("--resource-cpu-fraction", type=float, default=0.85)
    parser.add_argument(
        "--minimum-available-cpu-cores",
        type=float,
        default=None,
        help="Required launch floor whenever --cpu-pool is declared",
    )
    parser.add_argument("--resource-wait-timeout-seconds", type=float, default=-1.0)
    parser.add_argument("--resource-poll-seconds", type=float, default=300.0)
    parser.add_argument("--resource-sample-seconds", type=float, default=1.0)
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Optional runtime worker ceiling exported as DRPO_RUNTIME_MAX_WORKERS",
    )


def _finite_number(value: object, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise RunSpecError(f"{name} must be numeric") from exc
    if not math.isfinite(number):
        raise RunSpecError(f"{name} must be finite")
    return number


def normalize_runtime_resource_request(
    raw: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Validate an opt-in resource request without changing scientific coordinates."""

    if raw is None:
        return None
    cpu_pool = str(raw.get("cpu_pool") or "").strip() or None
    max_workers_raw = raw.get("max_workers")
    minimum_raw = raw.get("minimum_available_cpu_cores")
    if cpu_pool is None and max_workers_raw is None:
        if minimum_raw is not None:
            raise RunSpecError("minimum_available_cpu_cores requires cpu_pool")
        return None

    cpu_fraction = _finite_number(raw.get("cpu_fraction", 0.85), "cpu_fraction")
    timeout = _finite_number(
        raw.get("wait_timeout_seconds", -1.0),
        "wait_timeout_seconds",
    )
    poll = _finite_number(raw.get("poll_seconds", 300.0), "poll_seconds")
    sample = _finite_number(raw.get("sample_seconds", 1.0), "sample_seconds")
    if not 0 < cpu_fraction <= 1:
        raise RunSpecError("cpu_fraction must be in (0, 1]")
    if poll <= 0:
        raise RunSpecError("poll_seconds must be positive")
    if sample <= 0:
        raise RunSpecError("sample_seconds must be positive")

    max_workers: int | None = None
    if max_workers_raw is not None:
        if not isinstance(max_workers_raw, int) or isinstance(max_workers_raw, bool):
            raise RunSpecError("max_workers must be an integer")
        if max_workers_raw < 1:
            raise RunSpecError("max_workers must be positive")
        max_workers = int(max_workers_raw)

    minimum: float | None = None
    if cpu_pool is not None:
        if minimum_raw is None:
            raise RunSpecError(
                "minimum_available_cpu_cores is required when cpu_pool is declared"
            )
        minimum = _finite_number(
            minimum_raw,
            "minimum_available_cpu_cores",
        )
        if minimum <= 0:
            raise RunSpecError("minimum_available_cpu_cores must be positive")
    elif minimum_raw is not None:
        raise RunSpecError("minimum_available_cpu_cores requires cpu_pool")

    return {
        "schema_version": 1,
        "cpu_pool": cpu_pool,
        "cpu_fraction": cpu_fraction,
        "minimum_available_cpu_cores": minimum,
        "wait_timeout_seconds": timeout,
        "poll_seconds": poll,
        "sample_seconds": sample,
        "max_workers": max_workers,
        "scientific_matrix_changed": False,
    }


def runtime_resource_request_from_args(
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    return normalize_runtime_resource_request(
        {
            "cpu_pool": args.cpu_pool,
            "cpu_fraction": args.resource_cpu_fraction,
            "minimum_available_cpu_cores": args.minimum_available_cpu_cores,
            "wait_timeout_seconds": args.resource_wait_timeout_seconds,
            "poll_seconds": args.resource_poll_seconds,
            "sample_seconds": args.resource_sample_seconds,
            "max_workers": args.max_workers,
        }
    )


def _write_immutable_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = dict(payload)
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RunSpecError(
                f"cannot read existing runtime resource identity: {path}"
            ) from exc
        if existing != normalized:
            raise RunSpecError(
                f"runtime resource identity changed for this run: {path}"
            )
        return path
    serialized = json.dumps(normalized, indent=2, sort_keys=True) + "\n"
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        return _write_immutable_json(path, normalized)
    return path


def _capacity_observation(
    *,
    binding: Any,
    interval: Any,
    cpu_fraction: float,
) -> dict[str, Any]:
    affinity_budget = len(binding.affinity_cpu_ids) * cpu_fraction
    affinity_available = max(0.0, affinity_budget - interval.system_busy_cores)
    quota_usage = interval.quota_usage_map()
    available_limits = [affinity_available]
    quota_rows: list[dict[str, Any]] = []
    for domain in binding.quota_domains:
        if domain.path not in quota_usage:
            raise RunSpecError(f"CPU quota usage is missing for domain: {domain.path}")
        budget = float(domain.quota_cores) * cpu_fraction
        observed = float(quota_usage[domain.path])
        available = max(0.0, budget - observed)
        available_limits.append(available)
        quota_rows.append(
            {
                "path": domain.path,
                "quota_cores": float(domain.quota_cores),
                "budget_cores": budget,
                "observed_busy_cores": observed,
                "available_cores": available,
            }
        )
    return {
        "affinity_cpu_ids": list(binding.affinity_cpu_ids),
        "affinity_cpu_count": len(binding.affinity_cpu_ids),
        "cpu_fraction": cpu_fraction,
        "affinity_budget_cores": affinity_budget,
        "observed_affinity_busy_cores": float(interval.system_busy_cores),
        "affinity_available_cores": affinity_available,
        "quota_domains": quota_rows,
        "available_cpu_cores": min(available_limits),
        "sample_seconds": float(interval.elapsed_seconds),
    }


def _wait_for_cpu_pool_capacity(
    *,
    log_root: Path,
    request: Mapping[str, Any],
) -> dict[str, Any]:
    from drpo import runtime_cpu_capacity as cpu
    from drpo.runtime_capacity_wait import wait_for_runtime_plan
    from drpo.runtime_resource_autotune import RuntimeResourceError, atomic_write_json

    try:
        binding = cpu.discover_cpu_binding()
    except cpu.CPUCapacityError as exc:
        raise RunSpecError(str(exc)) from exc
    policy_budgets = [len(binding.affinity_cpu_ids) * float(request["cpu_fraction"])]
    policy_budgets.extend(
        float(domain.quota_cores) * float(request["cpu_fraction"])
        for domain in binding.quota_domains
    )
    maximum_available = min(policy_budgets)
    minimum = float(request["minimum_available_cpu_cores"])
    if minimum > maximum_available:
        raise RunSpecError(
            "minimum_available_cpu_cores exceeds the selected pool/cgroup policy budget: "
            f"minimum={minimum:g} maximum={maximum_available:g}"
        )

    latest_path = log_root / "RUNTIME_CPU_CAPACITY_LATEST.json"

    def plan_once() -> dict[str, Any]:
        try:
            interval = cpu.sample_cpu_interval(
                binding,
                sample_seconds=float(request["sample_seconds"]),
            )
        except cpu.CPUCapacityError as exc:
            raise RuntimeResourceError(str(exc)) from exc
        observation = _capacity_observation(
            binding=binding,
            interval=interval,
            cpu_fraction=float(request["cpu_fraction"]),
        )
        atomic_write_json(latest_path, observation)
        available = float(observation["available_cpu_cores"])
        if available < minimum:
            raise RuntimeResourceError(
                "measured CPU capacity cannot support one worker: "
                f"available_cpu_cores={available:.6f} "
                f"minimum_available_cpu_cores={minimum:.6f}"
            )
        return {"cpu_capacity": observation}

    result = wait_for_runtime_plan(
        plan_once=plan_once,
        work_dir=log_root,
        wait_timeout_seconds=float(request["wait_timeout_seconds"]),
        poll_seconds=float(request["poll_seconds"]),
    )
    return {
        "latest_observation_path": str(latest_path),
        "cpu_capacity": result["cpu_capacity"],
        "wait": result["plan_capacity_wait"],
    }


def prepare_runtime_resources(
    repo: Path,
    *,
    run_id: str,
    request: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Apply one immutable resource request before any RunSpec entrypoint starts."""

    normalized = normalize_runtime_resource_request(request)
    if normalized is None:
        return None

    log_root = repo / STATE_DIRNAME / "logs" / run_id
    request_path = _write_immutable_json(
        log_root / "RUNTIME_RESOURCE_REQUEST.json",
        normalized,
    )
    pool_identity: str | None = None
    pool_payload: dict[str, Any] | None = None
    capacity: dict[str, Any] | None = None
    if normalized["cpu_pool"] is not None:
        from drpo.runtime_resource_pool import (
            activate_resource_pool,
            write_pool_identity,
        )

        pool = activate_resource_pool(
            cpu_pool=str(normalized["cpu_pool"]),
            gpu_pool=None,
            gpu_enforcement="none",
        )
        identity = write_pool_identity(log_root / "RESOURCE_POOL.json", pool)
        pool_identity = str(identity)
        pool_payload = pool.as_dict()
        capacity = _wait_for_cpu_pool_capacity(log_root=log_root, request=normalized)

    if normalized["max_workers"] is not None:
        os.environ["DRPO_RUNTIME_MAX_WORKERS"] = str(normalized["max_workers"])

    report = {
        "schema_version": 1,
        "run_id": run_id,
        "request_path": str(request_path),
        "pool_identity_path": pool_identity,
        "resource_pool": pool_payload,
        "capacity": capacity,
        "max_workers": normalized["max_workers"],
        "worker_environment": (
            None
            if normalized["max_workers"] is None
            else {"DRPO_RUNTIME_MAX_WORKERS": str(normalized["max_workers"])}
        ),
        "scientific_matrix_changed": False,
        "running_workers_resized": False,
    }
    from drpo.runtime_resource_autotune import atomic_write_json

    ready_path = log_root / "RUNTIME_RESOURCE_READY.json"
    atomic_write_json(ready_path, report)
    report["path"] = str(ready_path)
    return report


def execute_claimed_runspec(
    repo: Path,
    claimed: Path,
    *,
    runtime_resources: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], int]:
    spec = read_yaml(claimed)
    registration = validate_registration_block(spec)
    spec["registration"] = registration
    validate_provenance(repo, spec)
    validate_recovery_policy(repo, spec)
    validate_simple_size_policy(spec)
    delivery = validate_delivery_block(spec, str(spec.get("lane") or ""))
    publish = spec.get("publish") or {}
    if isinstance(publish, dict) and publish.get("enabled") is True:
        from publish_runspec_result import validate_publish_block

        validate_publish_block(spec, str(spec.get("lane") or ""))
    running = move_state(
        repo,
        claimed,
        RUNNING_DIR,
        {"run_id": spec["run_id"], "state": "running"},
    )
    resource_report: dict[str, Any] | None = None
    try:
        resource_report = prepare_runtime_resources(
            repo,
            run_id=spec["run_id"],
            request=runtime_resources,
        )
        run_result = run_entrypoint_with_recovery(repo, running)
        manifest = package_artifacts_safe(repo, running)
    except Exception as exc:  # noqa: BLE001
        failed = move_state(
            repo,
            running,
            FAILED_DIR,
            {"run_id": spec["run_id"], "state": "failed", "error": str(exc)},
        )
        raise RuntimeError(
            f"RunSpec execution failed; state={failed.relative_to(repo).as_posix()}: {exc}"
        ) from exc

    done = move_state(
        repo,
        running,
        DONE_DIR,
        {
            "run_id": spec["run_id"],
            "state": "done",
            "artifact_zip": manifest["zip_path"],
            "artifact_zip_sha256": manifest["zip_sha256"],
            "attempts": run_result.get("attempts", 1),
            "recovery_used": bool(run_result.get("recovery_used", False)),
            "recovery_report": run_result.get("recovery_report"),
            "runtime_resource_report": (
                None if resource_report is None else resource_report.get("path")
            ),
        },
    )
    payload: dict[str, Any] = {
        "status": "PASS",
        "run_id": spec["run_id"],
        "state_path": done.relative_to(repo).as_posix(),
        "artifact_zip": manifest["zip_path"],
        "returncode": run_result["returncode"],
        "attempts": run_result.get("attempts", 1),
        "recovery_used": bool(run_result.get("recovery_used", False)),
        "recovery_report": run_result.get("recovery_report"),
        "runtime_resources": resource_report,
        "registration_mode": registration["mode"],
        "registration_closure_required": registration["closure_required"],
        "delivery_status": "not_requested",
        "publish_status": "not_requested",
    }
    if delivery["enabled"] and delivery["auto"]:
        try:
            from runspec_results_delivery import deliver_completed_run

            report = deliver_completed_run(repo, spec["run_id"])
            payload["delivery_status"] = report["status"]
            payload["results_repository"] = report["repository"]
            payload["results_branch"] = report["branch"]
            payload["results_commit"] = report["results_commit"]
            payload["result_path"] = report["result_path"]
            payload["manifest_sha256"] = report["manifest_sha256"]
        except Exception as exc:  # noqa: BLE001
            if is_result_too_large_error(exc):
                report = record_result_too_large(repo, spec, manifest, exc)
                payload["delivery_status"] = RESULT_TOO_LARGE
                payload["delivery_error"] = report["reason"]
                payload["delivery_upload_attempted"] = False
                payload["local_artifact_zip"] = report["artifact_zip"]
                payload["local_artifact_zip_sha256"] = report["artifact_zip_sha256"]
                return payload, 0
            payload["status"] = "PARTIAL"
            payload["delivery_status"] = "FAIL"
            payload["delivery_error"] = str(exc)
            return payload, 2
    if (
        isinstance(publish, dict)
        and publish.get("enabled") is True
        and publish.get("auto") is True
    ):
        try:
            from publish_runspec_result import publish_completed_run

            report = publish_completed_run(repo, spec["run_id"])
            payload["publish_status"] = "PASS"
            payload["published_commit"] = report["published_commit"]
            payload["pr_url"] = report["pr_url"]
        except Exception as exc:  # noqa: BLE001
            payload["status"] = "PARTIAL"
            payload["publish_status"] = "FAIL"
            payload["publish_error"] = str(exc)
            return payload, 2
    return payload, 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--runspec", help="Claimed RunSpec path")
    group.add_argument("--run-id", help="Claimed run_id under .runspec_state/claimed")
    parser.add_argument("--json", action="store_true")
    add_runtime_resource_args(parser)
    args = parser.parse_args()
    repo = Path(args.repo_root).resolve()
    try:
        claimed = (
            Path(args.runspec).resolve()
            if args.runspec
            else state_path(repo, CLAIMED_DIR, args.run_id)
        )
        payload, code = execute_claimed_runspec(
            repo,
            claimed,
            runtime_resources=runtime_resource_request_from_args(args),
        )
    except Exception as exc:  # noqa: BLE001
        return handle_cli_error(exc, json_output=args.json)
    if args.json:
        json_main(payload)
    elif code == 0 and payload.get("delivery_status") == RESULT_TOO_LARGE:
        print(
            f"RunSpec execution: PASS run_id={payload['run_id']} "
            f"delivery={RESULT_TOO_LARGE} artifact={payload['local_artifact_zip']}"
        )
    elif code == 0:
        print(f"RunSpec execution: PASS run_id={payload['run_id']}")
    else:
        error = payload.get("delivery_error") or payload.get("publish_error")
        print(
            f"RunSpec execution: PASS but result handoff: FAIL run_id={payload['run_id']} "
            f"error={error}"
        )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
