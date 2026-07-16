#!/usr/bin/env python3
"""Run runtime-resource acceptance under shared-host dynamic capacity semantics."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Mapping, Sequence

from drpo.runtime_resource_acceptance import StageResult
from drpo.runtime_resource_acceptance_shared_host import (
    shared_host_e7_stage,
    shared_host_gpu_stage,
    shared_host_report,
    shared_host_topology_stage,
)

BASE_SCRIPT = Path(__file__).with_name("run_runtime_resource_acceptance.py")
SPEC = importlib.util.spec_from_file_location(
    "runtime_resource_acceptance_base_shared_host",
    BASE_SCRIPT,
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load base acceptance runner: {BASE_SCRIPT}")
base = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(base)

ORIGINAL_TOPOLOGY_STAGE = base.topology_stage
ORIGINAL_E7_STAGE = base.e7_stage
ORIGINAL_GPU_STAGE = base.gpu_stage
ORIGINAL_REPORT = base._report


def topology_stage(
    root: Path,
    repo: Path,
    gpu_worktree: Path,
    profile: Mapping[str, Any],
) -> StageResult:
    return shared_host_topology_stage(
        ORIGINAL_TOPOLOGY_STAGE,
        root,
        repo,
        gpu_worktree,
        profile,
    )


def e7_stage(
    root: Path,
    repo: Path,
    profile_path: Path,
    profile: Mapping[str, Any],
    ledger: Path,
) -> StageResult:
    return shared_host_e7_stage(
        ORIGINAL_E7_STAGE,
        root,
        repo,
        profile_path,
        profile,
        ledger,
    )


def gpu_stage(
    root: Path,
    repo: Path,
    gpu_repo: Path,
    profile: Mapping[str, Any],
    ledger: Path,
) -> StageResult:
    return shared_host_gpu_stage(
        ORIGINAL_GPU_STAGE,
        root,
        repo,
        gpu_repo,
        profile,
        ledger,
    )


def report(
    root: Path,
    checkout: Mapping[str, Any],
    profile: Mapping[str, Any],
    results: Sequence[StageResult],
    final_audit: Mapping[str, Any],
) -> dict[str, Any]:
    return shared_host_report(
        ORIGINAL_REPORT,
        root,
        checkout,
        profile,
        results,
        final_audit,
    )


def main(argv: list[str] | None = None) -> int:
    base.topology_stage = topology_stage
    base.e7_stage = e7_stage
    base.gpu_stage = gpu_stage
    base._report = report
    return int(base.main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
