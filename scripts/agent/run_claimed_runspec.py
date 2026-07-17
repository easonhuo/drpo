#!/usr/bin/env python3
"""Execute a claimed RunSpec, package artifacts, and optionally deliver results."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

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
    handle_cli_error,
    json_main,
    move_state,
    read_yaml,
    state_path,
    write_yaml,
)
from runspec_recovery import run_entrypoint_with_recovery, validate_recovery_policy
from runspec_registration import validate_registration_block
from runspec_results_delivery import validate_delivery_block
from runspec_safety import package_artifacts_safe, validate_provenance

DEFAULT_AUTO_DELIVERY_LANES = frozenset({"e7", "e8"})
DEFAULT_RESULTS_REPOSITORY = "easonhuo/drpo-results"


def apply_default_results_delivery(spec: dict[str, Any]) -> None:
    """Default E7/E8 RunSpecs to scoped results-repository delivery.

    An explicit delivery block remains authoritative, including an explicit
    ``enabled: false`` local-only choice.
    """

    lane = str(spec.get("lane") or "")
    if "delivery" in spec or lane not in DEFAULT_AUTO_DELIVERY_LANES:
        return
    spec["delivery"] = {
        "enabled": True,
        "auto": True,
        "mode": "results_repo",
        "repository": DEFAULT_RESULTS_REPOSITORY,
        "branch": f"ingest/{lane}",
        "export_profile": "manifest_text_v1",
        "max_total_size_mb": 30,
        "max_file_size_mb": 10,
    }


def execute_claimed_runspec(repo: Path, claimed: Path) -> tuple[dict[str, Any], int]:
    spec = read_yaml(claimed)
    apply_default_results_delivery(spec)
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
    # Persist the effective contract before entering RUNNING so completion,
    # retries, and later delivery audits see the same normalized delivery policy.
    write_yaml(claimed, spec)
    running = move_state(
        repo,
        claimed,
        RUNNING_DIR,
        {"run_id": spec["run_id"], "state": "running"},
    )
    try:
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
    args = parser.parse_args()
    repo = Path(args.repo_root).resolve()
    try:
        claimed = (
            Path(args.runspec).resolve()
            if args.runspec
            else state_path(repo, CLAIMED_DIR, args.run_id)
        )
        payload, code = execute_claimed_runspec(repo, claimed)
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
