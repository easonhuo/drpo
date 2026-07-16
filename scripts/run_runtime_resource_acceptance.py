#!/usr/bin/env python3
"""Run the complete DRPO runtime-resource engineering acceptance harness."""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
from pathlib import Path
from types import FrameType
from typing import Any, Mapping, Sequence

from drpo.runtime_resource_acceptance import (
    AcceptanceError,
    StageResult,
    add_worktree,
    compact_timestamp,
    ensure_commit,
    load_profile,
    overall_status,
    package_acceptance,
    remove_worktree,
    stage_result,
    utc_now,
    verify_checkout,
)
from drpo.runtime_resource_acceptance_capacity import normalize_capacity_block
from drpo.runtime_resource_acceptance_e7 import revalidate_only, selected_liveness
from drpo.runtime_resource_acceptance_gpu_stages import (
    concurrent_stage,
    gpu_stage,
    thread_scan_stage,
)
from drpo.runtime_resource_acceptance_local_stages import (
    e7_stage,
    process_inventory,
    resource_pool_stage,
    topology_stage,
)
from drpo.runtime_resource_autotune import atomic_write_json

STAGES = (
    "stage0_topology",
    "stage1_resource_pool",
    "stage2_e7_cpu_v2",
    "stage3_gpu_placement",
    "stage4_e8_thread_scan",
    "stage5_concurrent_pool",
)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--profile", required=True)
    value.add_argument("--validate-profile", action="store_true")
    value.add_argument(
        "--internal-e7-action", choices=("validate", "liveness"), help=argparse.SUPPRESS
    )
    value.add_argument("--e7-work-dir", help=argparse.SUPPRESS)
    value.add_argument("--internal-output", help=argparse.SUPPRESS)
    return value


def _append(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), sort_keys=True) + "\n")


def _blocked(root: Path, name: str, reason: str) -> StageResult:
    return stage_result(root, name, "BLOCKED", utc_now(), {"reason": reason})


def _interrupt(signum: int, _frame: FrameType | None) -> None:
    raise AcceptanceError(f"acceptance interrupted by signal {signum}")


def _install_signal_handlers() -> None:
    signal.signal(signal.SIGINT, _interrupt)
    signal.signal(signal.SIGTERM, _interrupt)


def _report(
    root: Path,
    checkout: Mapping[str, Any],
    profile: Mapping[str, Any],
    results: Sequence[StageResult],
    final_audit: Mapping[str, Any],
) -> dict[str, Any]:
    summary = {
        "schema_version": 1,
        "claim": "GOV-RUNTIME-RESOURCE-ACCEPTANCE-HARNESS-01",
        "created_utc": utc_now(),
        "overall_status": overall_status(results),
        "harness_checkout": dict(checkout),
        "gpu_selection_commit": profile["gpu_selection_commit"],
        "profile_sha256": profile["profile_sha256"],
        "scientific_result": False,
        "full_scientific_sweep_started": False,
        "stages": [result.as_dict() for result in results],
        "separate_failure_classes": {
            "task_or_process_failure": [
                result.name for result in results if result.status == "FAIL"
            ],
            "safe_capacity_unavailable": [
                result.name
                for result in results
                if result.status == "BLOCKED"
                and result.details.get("capacity_unavailable") is True
            ],
            "resource_boundary_or_oom": [
                result.name
                for result in results
                if "oom" in json.dumps(result.details).lower()
            ],
            "nan_inf_numerical_failure": [
                result.name
                for result in results
                if bool(result.details.get("nan_inf_matches"))
            ],
            "orphan_process_failure": bool(
                final_audit.get("residual_processes")
            ),
        },
        "final_process_audit": dict(final_audit),
    }
    atomic_write_json(root / "ACCEPTANCE_SUMMARY.json", summary)
    lines = [
        "# DRPO runtime-resource server acceptance",
        "",
        f"- Overall: **{summary['overall_status']}**",
        f"- Harness commit: `{checkout.get('commit')}`",
        f"- GPU selection commit: `{profile['gpu_selection_commit']}`",
        "- Scientific result: no",
        "- Full scientific sweep started: no",
        "",
        "## Stages",
        "",
    ]
    lines.extend(f"- `{result.name}`: **{result.status}**" for result in results)
    capacity_blocks = [
        result
        for result in results
        if result.status == "BLOCKED" and result.details.get("capacity_unavailable") is True
    ]
    if capacity_blocks:
        lines.extend(["", "## Safe-capacity blocks", ""])
        lines.extend(
            f"- `{result.name}`: {', '.join(result.details.get('capacity_failures', []))}"
            for result in capacity_blocks
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "This package contains engineering evidence only. It cannot establish task ",
            "performance, method ranking, convergence, steady state, controlled ",
            "mechanism identification, or OOD generalization.",
        ]
    )
    (root / "SERVER_ACCEPTANCE_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return summary


def _internal(args: argparse.Namespace, repo: Path) -> int:
    if not args.e7_work_dir or not args.internal_output:
        raise AcceptanceError("internal E7 action requires work-dir and output")
    profile = load_profile(args.profile, repo_root=repo)
    work = Path(args.e7_work_dir).resolve()
    output = Path(args.internal_output).resolve()
    if args.internal_e7_action == "validate":
        payload = revalidate_only(profile, repo, work, output)
    else:
        payload = selected_liveness(profile, repo, work, output)
    print(json.dumps(payload, sort_keys=True), flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    repo = Path(__file__).resolve().parents[1]
    if args.internal_e7_action:
        return _internal(args, repo)
    profile_path = Path(args.profile).expanduser().resolve()
    profile = load_profile(profile_path, repo_root=repo)
    if args.validate_profile:
        print(json.dumps(profile, indent=2, sort_keys=True))
        return 0
    checkout = verify_checkout(repo, profile.get("expected_harness_commit"))
    output_parent = Path(profile["output_parent"])
    output_parent.mkdir(parents=True, exist_ok=True)
    root = output_parent / f"drpo_runtime_acceptance_{compact_timestamp()}"
    root.mkdir(mode=0o750)
    for name in STAGES:
        (root / name).mkdir()
    atomic_write_json(root / "PROFILE.normalized.json", profile)
    ledger = root / "COMMANDS_EXECUTED.jsonl"
    gpu_worktree = root / "worktrees" / "gpu_selection"
    results: list[StageResult] = []
    worktree_removed = False
    _install_signal_handlers()
    try:
        ensure_commit(
            repo,
            profile["gpu_selection_commit"],
            profile["gpu_selection_ref"],
        )
        add_worktree(repo, gpu_worktree, profile["gpu_selection_commit"])
        results.append(topology_stage(root, repo, gpu_worktree, profile))
        if results[-1].status != "PASS" and not profile["continue_after_failure"]:
            results.extend(_blocked(root, name, "stage0 did not PASS") for name in STAGES[1:])
        else:
            results.append(resource_pool_stage(root, repo, profile, ledger))
            if results[-1].status != "PASS" and not profile["continue_after_failure"]:
                results.extend(
                    _blocked(root, name, "stage1 did not PASS") for name in STAGES[2:]
                )
            else:
                results.append(
                    normalize_capacity_block(
                        root,
                        e7_stage(root, repo, profile_path, profile, ledger),
                    )
                )
                results.append(
                    normalize_capacity_block(
                        root,
                        gpu_stage(root, repo, gpu_worktree, profile, ledger),
                    )
                )
                results.append(
                    thread_scan_stage(
                        root, repo, gpu_worktree, profile, ledger, results[3]
                    )
                )
                results.append(
                    concurrent_stage(
                        root,
                        repo,
                        gpu_worktree,
                        profile_path,
                        profile,
                        ledger,
                        results[2],
                        results[3],
                    )
                )
    except BaseException as exc:
        _append(
            ledger,
            {"fatal_utc": utc_now(), "error_type": type(exc).__name__, "error": str(exc)},
        )
        while len(results) < len(STAGES):
            results.append(
                _blocked(
                    root,
                    STAGES[len(results)],
                    f"harness fatal error: {type(exc).__name__}: {exc}",
                )
            )
    finally:
        if gpu_worktree.exists():
            remove_worktree(repo, gpu_worktree)
            worktree_removed = True
        try:
            checkout_after = verify_checkout(repo)
        except AcceptanceError as exc:
            checkout_after = {"dirty": True, "error": str(exc)}
        ancestors = {os.getpid(), os.getppid()}
        residual = [
            row
            for row in process_inventory()
            if str(root) in str(row.get("command", ""))
            and int(row["pid"]) not in ancestors
        ]
        final_audit = {
            "created_utc": utc_now(),
            "residual_processes": residual,
            "gpu_worktree_removed": worktree_removed,
            "repository_after": checkout_after,
            "repository_modified": bool(checkout_after.get("dirty", True)),
        }
        atomic_write_json(root / "FINAL_PROCESS_AUDIT.json", final_audit)
        while len(results) < len(STAGES):
            results.append(_blocked(root, STAGES[len(results)], "not reached"))
        _report(root, checkout, profile, results, final_audit)
        package = package_acceptance(root)
        print(
            json.dumps(
                {
                    "acceptance_root": str(root),
                    "overall_status": overall_status(results),
                    "package": package,
                },
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )
    return 0 if overall_status(results) in {"PASS", "INCONCLUSIVE"} else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AcceptanceError as exc:
        print(f"RUNTIME_ACCEPTANCE_ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
