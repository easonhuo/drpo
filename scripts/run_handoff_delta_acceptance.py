#!/usr/bin/env python3
"""Run fixed Stage 3 Fast, Standard, or Full acceptance tiers."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

import handoff_delta_shadow as shadow


def run(command: list[str], cwd: Path, timeout_seconds: float) -> dict[str, object]:
    started = time.perf_counter()
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            text=True,
            capture_output=True,
            check=False,
            timeout=max(timeout_seconds, 0.1),
        )
        return {
            "command": command,
            "returncode": result.returncode,
            "timed_out": False,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "command": command,
            "returncode": 124,
            "timed_out": True,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or f"Timed out after {timeout_seconds:.3f}s",
        }


def observation_fingerprint(update_ids: list[str]) -> str:
    return shadow.sha256_text("\n".join(sorted(update_ids)) + "\n")


def real_observation_ids(observations: list[dict[str, object]]) -> list[str]:
    """Return canonical lexicographic coverage IDs for durable Full reports."""
    return sorted(
        str(item["update_id"])
        for item in observations
        if item.get("kind") == "real"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--tier", choices=("fast", "standard", "full"), required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--reason",
        action="append",
        default=[],
        help="Registered Full Acceptance trigger; may be repeated.",
    )
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    policy = yaml.safe_load((repo / "docs/handoff_delta_policy.yaml").read_text())
    if args.tier == "fast":
        commands = [
            [sys.executable, "scripts/handoff_delta_shadow.py", "auto-check", "--repo-root", "."]
        ]
        limit = float(policy["fast_gate"]["hard_timeout_seconds"])
    elif args.tier == "standard":
        commands = [
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "tests/test_handoff_delta_shadow.py",
                "-k",
                "not full_acceptance",
            ],
            [sys.executable, "scripts/handoff_delta_shadow.py", "auto-check", "--repo-root", "."],
        ]
        limit = float(policy["standard_regression"]["target_seconds"])
    else:
        if not args.report:
            parser.error("--tier full requires --report so acceptance evidence is durable")
        commands = [
            [sys.executable, "-m", "pytest", "-q", "tests/test_handoff_delta_shadow.py"],
            [
                sys.executable,
                "scripts/handoff_delta_shadow.py",
                "corpus-check",
                "--repo-root",
                ".",
            ],
            [
                sys.executable,
                "scripts/handoff_delta_shadow.py",
                "auto-check",
                "--repo-root",
                ".",
                "--allow-full-due",
            ],
        ]
        limit = float(policy["full_acceptance"]["target_seconds"])

    started = time.perf_counter()
    outcomes = []
    for command in commands:
        elapsed = time.perf_counter() - started
        remaining = limit - elapsed
        if remaining <= 0:
            outcomes.append(
                {
                    "command": command,
                    "returncode": 124,
                    "timed_out": True,
                    "elapsed_seconds": 0.0,
                    "stdout": "",
                    "stderr": "Acceptance tier exhausted its total time budget before this command.",
                }
            )
            break
        outcomes.append(run(command, repo, remaining))
    total = time.perf_counter() - started
    passed = all(item["returncode"] == 0 for item in outcomes) and total <= limit
    report: dict[str, object] = {
        "schema_version": 1,
        "report_schema_version": 2,
        "policy_id": policy["policy_id"],
        "tier": args.tier,
        "status": "PASS" if passed else "FAIL",
        "validation_worktree_head": shadow.git_text(repo, "rev-parse", "HEAD"),
        "reasons": args.reason,
        "elapsed_seconds": round(total, 3),
        "target_seconds": limit,
        "outcomes": outcomes,
    }
    if args.tier == "full" and passed:
        observations = shadow.observation_records(repo)
        real_ids = real_observation_ids(observations)
        report["coverage"] = {
            "bootstrap_observation_count": sum(
                item["kind"] == "bootstrap" for item in observations
            ),
            "successful_real_observation_count": len(real_ids),
            "covered_update_ids": real_ids,
            "observation_fingerprint": observation_fingerprint(real_ids),
        }
        report["corpus_audit"] = {
            "observation_count": len(observations),
            "all_stored_reports_revalidated": True,
        }
    if args.report:
        report_path = args.report
        if not report_path.is_absolute():
            report_path = repo / report_path
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
