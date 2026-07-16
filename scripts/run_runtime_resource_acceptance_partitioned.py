#!/usr/bin/env python3
"""Run acceptance only inside a proven exclusive cgroup v2 CPU partition."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any, Mapping

from drpo.runtime_resource_acceptance import StageResult, stage_result
from drpo.runtime_resource_acceptance_isolation import (
    ancestor_pids,
    audit_resource_isolation,
    process_inventory,
)
from drpo.runtime_resource_autotune import atomic_write_json

BASE_SCRIPT = Path(__file__).with_name("run_runtime_resource_acceptance.py")
SPEC = importlib.util.spec_from_file_location(
    "runtime_resource_acceptance_base",
    BASE_SCRIPT,
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load base acceptance runner: {BASE_SCRIPT}")
base = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(base)

ORIGINAL_TOPOLOGY_STAGE = base.topology_stage
PERSISTENT_EXTERNAL_PATTERNS = (
    "researchbench",
    "collector_v2.py",
    "aide",
    "joblib.externals.loky",
    "loky.process_executor",
)


def _isolated_pids(audit: Mapping[str, Any]) -> set[int]:
    return {
        int(row["pid"])
        for row in audit.get("isolated_external_matches", [])
        if isinstance(row, Mapping) and "pid" in row
    }


def partition_aware_topology_stage(
    root: Path,
    repo: Path,
    gpu_worktree: Path,
    profile: Mapping[str, Any],
) -> StageResult:
    """Require exclusive cpuset proof and reconcile outside permanent workloads."""

    original = ORIGINAL_TOPOLOGY_STAGE(root, repo, gpu_worktree, profile)
    directory = root / "stage0_topology"
    inventory = process_inventory()
    pools = profile["resource_pools"]
    reserved = set(pools["e7_cpu_ids"]) | set(pools["e8_cpu_ids"])
    patterns = list(profile["conflict_process_patterns"])
    patterns.extend(PERSISTENT_EXTERNAL_PATTERNS)
    audit = audit_resource_isolation(
        inventory=inventory,
        conflict_patterns=patterns,
        reserved_cpu_ids=reserved,
        excluded_pids=ancestor_pids(),
    )
    atomic_write_json(directory / "EXCLUSIVE_PARTITION_AUDIT.json", audit)

    details = {
        **dict(original.details),
        "resource_isolation": audit,
        "persistent_external_patterns": list(PERSISTENT_EXTERNAL_PATTERNS),
        "original_status": original.status,
    }
    if original.status == "FAIL":
        return stage_result(
            root,
            "stage0_topology",
            "FAIL",
            original.started_utc,
            details,
        )

    proof_ready = (
        audit.get("exclusive_partition_proven") is True
        and audit.get("ready") is True
    )
    original_conflicts = original.details.get("conflicts", [])
    isolated_pids = _isolated_pids(audit)
    original_conflict_pids = {
        int(row["pid"])
        for row in original_conflicts
        if isinstance(row, Mapping) and "pid" in row
    }
    original_only_isolated = (
        bool(original_conflict_pids)
        and original_conflict_pids <= isolated_pids
    )

    if not proof_ready:
        details["partition_gate"] = "BLOCKED"
        details["partition_gate_reason"] = (
            audit.get("partition_error")
            or "exclusive partition contains conflicting or contaminating processes"
        )
        return stage_result(
            root,
            "stage0_topology",
            "BLOCKED",
            original.started_utc,
            details,
        )

    if original.status == "BLOCKED" and not original_only_isolated:
        details["partition_gate"] = "PASS"
        return stage_result(
            root,
            "stage0_topology",
            "BLOCKED",
            original.started_utc,
            details,
        )

    details["partition_gate"] = "PASS"
    details["outside_permanent_workloads_ignored"] = len(
        audit.get("isolated_external_matches", [])
    )
    return stage_result(
        root,
        "stage0_topology",
        "PASS",
        original.started_utc,
        details,
    )


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--profile", required=True)
    value.add_argument(
        "--check-only",
        action="store_true",
        help="print exclusive-partition evidence without starting acceptance",
    )
    return value


def _check_only(profile_path: Path) -> int:
    repo = Path(__file__).resolve().parents[1]
    profile = base.load_profile(profile_path, repo_root=repo)
    pools = profile["resource_pools"]
    reserved = set(pools["e7_cpu_ids"]) | set(pools["e8_cpu_ids"])
    patterns = list(profile["conflict_process_patterns"])
    patterns.extend(PERSISTENT_EXTERNAL_PATTERNS)
    audit = audit_resource_isolation(
        inventory=process_inventory(),
        conflict_patterns=patterns,
        reserved_cpu_ids=reserved,
        excluded_pids=ancestor_pids(),
    )
    print(json.dumps(audit, indent=2, sort_keys=True), flush=True)
    return 0 if (
        audit.get("exclusive_partition_proven") is True
        and audit.get("ready") is True
    ) else 2


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    profile_path = Path(args.profile).expanduser().resolve()
    if args.check_only:
        return _check_only(profile_path)

    base.topology_stage = partition_aware_topology_stage
    return int(base.main(["--profile", str(profile_path)]))


if __name__ == "__main__":
    raise SystemExit(main())
