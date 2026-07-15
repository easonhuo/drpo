"""Topology, resource-pool, and E7 acceptance stages."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from drpo.runtime_resource_acceptance import (
    AcceptanceError,
    StageResult,
    git_output,
    sha256_file,
    stage_result,
    utc_now,
    verify_checkout,
)
from drpo.runtime_resource_acceptance_commands import (
    candidate_above_one,
    e7_plan_command,
    internal_e7_command,
    last_json_line,
    numerical_matches,
    pool_command,
)
from drpo.runtime_resource_acceptance_e7 import selection_identity
from drpo.runtime_resource_acceptance_process import run_command
from drpo.runtime_resource_autotune import atomic_write_json, load_json
from drpo.runtime_resource_pool import format_cpu_pool


def _capture(command: Sequence[str], cwd: Path) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            [str(item) for item in command],
            cwd=str(cwd),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=30,
            check=False,
        )
        return {
            "command": [str(item) for item in command],
            "returncode": completed.returncode,
            "output": completed.stdout,
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "command": [str(item) for item in command],
            "returncode": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _ancestors() -> set[int]:
    values = {os.getpid()}
    current = os.getpid()
    for _ in range(128):
        path = Path("/proc") / str(current) / "stat"
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            break
        close = text.rfind(")")
        fields = text[close + 2 :].split() if close >= 0 else []
        if len(fields) < 2:
            break
        try:
            parent = int(fields[1])
        except ValueError:
            break
        if parent <= 0 or parent in values:
            break
        values.add(parent)
        current = parent
    return values


def process_inventory() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (
                (entry / "cmdline")
                .read_bytes()
                .replace(b"\x00", b" ")
                .decode("utf-8", errors="replace")
                .strip()
            )
        except OSError:
            continue
        if not command:
            continue
        pid = int(entry.name)
        try:
            affinity = sorted(int(value) for value in os.sched_getaffinity(pid))
        except (OSError, PermissionError, ProcessLookupError):
            affinity = []
        rows.append({"pid": pid, "affinity_cpu_ids": affinity, "command": command})
    return sorted(rows, key=lambda row: int(row["pid"]))


def topology_stage(
    root: Path,
    repo: Path,
    gpu_worktree: Path,
    profile: Mapping[str, Any],
) -> StageResult:
    started = utc_now()
    directory = root / "stage0_topology"
    try:
        checkout = verify_checkout(repo, profile.get("expected_harness_commit"))
        external: list[Path] = []
        if profile["e7"]["enabled"]:
            external.extend(Path(profile["e7"][key]) for key in ("contract", "run_spec", "grid"))
        if profile["e8"]["enabled"]:
            external.extend(
                Path(profile["e8"][key])
                for key in (
                    "model_path", "bank", "val", "test", "global_calibration",
                    "base_config", "sweep_config",
                )
            )
        missing = [str(path) for path in external if not path.exists()]
        if missing:
            raise AcceptanceError(f"external inputs missing: {missing}")
        inherited = set(int(value) for value in os.sched_getaffinity(0))
        pools = profile["resource_pools"]
        unavailable = {
            "e7": sorted(set(pools["e7_cpu_ids"]) - inherited),
            "e8": sorted(set(pools["e8_cpu_ids"]) - inherited),
        }
        if unavailable["e7"] or unavailable["e8"]:
            raise AcceptanceError(f"CPU pool unavailable: {unavailable}")
        inventory = process_inventory()
        excluded = _ancestors()
        patterns = [item.lower() for item in profile["conflict_process_patterns"]]
        conflicts = [
            row
            for row in inventory
            if int(row["pid"]) not in excluded
            and any(pattern in str(row["command"]).lower() for pattern in patterns)
        ]
        atomic_write_json(directory / "PROCESS_INVENTORY.json", inventory)
        atomic_write_json(directory / "LIVE_CONFLICTS.json", conflicts)
        commands = [
            ["uname", "-a"], ["date", "-u"], ["nproc"], ["lscpu"],
            ["lscpu", "-e=CPU,NODE,SOCKET,CORE,ONLINE"],
            ["numactl", "--hardware"], ["nvidia-smi"],
            ["nvidia-smi", "topo", "-m"], ["cat", "/proc/loadavg"],
            ["cat", "/proc/meminfo"], ["cat", "/proc/self/cgroup"],
        ]
        atomic_write_json(
            directory / "TOPOLOGY_COMMANDS.json",
            [_capture(command, repo) for command in commands],
        )
        details = {
            "checkout": checkout,
            "gpu_worktree_commit": git_output(gpu_worktree, "rev-parse", "HEAD"),
            "inherited_affinity": sorted(inherited),
            "resource_pools": pools,
            "conflicts": conflicts,
        }
        atomic_write_json(directory / "TOPOLOGY.json", details)
        status = "BLOCKED" if conflicts else "PASS"
        return stage_result(root, "stage0_topology", status, started, details)
    except AcceptanceError as exc:
        return stage_result(
            root, "stage0_topology", "BLOCKED", started, {"error": str(exc)}
        )
    except BaseException as exc:
        return stage_result(
            root,
            "stage0_topology",
            "FAIL",
            started,
            {"error_type": type(exc).__name__, "error": str(exc)},
        )


def resource_pool_stage(
    root: Path, repo: Path, profile: Mapping[str, Any], ledger: Path
) -> StageResult:
    started = utc_now()
    directory = root / "stage1_resource_pool"
    pools = profile["resource_pools"]
    harmless = [
        sys.executable,
        "-c",
        (
            "import json,os;print(json.dumps({"
            "'affinity':sorted(os.sched_getaffinity(0)),"
            "'digest':os.environ.get('DRPO_RESOURCE_POOL_DIGEST'),"
            "'gpu_pool':os.environ.get('DRPO_GPU_POOL')}))"
        ),
    ]
    results: dict[str, Any] = {}
    try:
        e7_identity = directory / "RESOURCE_POOL_E7.json"
        e8_identity = directory / "RESOURCE_POOL_E8.json"
        cases = {
            "e7_dry_1": pool_command(
                repo, cpu_pool=pools["e7_cpu_pool"], identity=e7_identity,
                command=harmless, dry_run=True,
            ),
            "e7_dry_2": pool_command(
                repo, cpu_pool=pools["e7_cpu_pool"], identity=e7_identity,
                command=harmless, dry_run=True,
            ),
            "e7_child": pool_command(
                repo, cpu_pool=pools["e7_cpu_pool"], identity=e7_identity,
                command=harmless,
            ),
            "e8_child": pool_command(
                repo, cpu_pool=pools["e8_cpu_pool"], identity=e8_identity,
                command=[*harmless, "--gpus", ",".join(pools["e8_gpu_ids"])],
                gpu_ids=pools["e8_gpu_ids"],
            ),
        }
        for name, command in cases.items():
            result = run_command(
                command,
                cwd=repo,
                environment=os.environ.copy(),
                timeout_seconds=30,
                log_path=directory / f"{name}.log",
                samples_path=directory / f"{name}.jsonl",
                command_ledger=ledger,
            )
            results[name] = result.as_dict()
            if not result.ok:
                raise AcceptanceError(f"resource-pool command failed: {name}")
        e7_child = last_json_line(directory / "e7_child.log")
        e8_child = last_json_line(directory / "e8_child.log")
        if e7_child.get("affinity") != pools["e7_cpu_ids"]:
            raise AcceptanceError("E7 child affinity mismatch")
        if e8_child.get("affinity") != pools["e8_cpu_ids"]:
            raise AcceptanceError("E8 child affinity mismatch")
        alternate = list(pools["e7_cpu_ids"])
        if len(alternate) > 1:
            alternate.pop()
        else:
            alternate.append(int(pools["e8_cpu_ids"][0]))
        cpu_mismatch = run_command(
            pool_command(
                repo, cpu_pool=format_cpu_pool(alternate), identity=e7_identity,
                command=harmless, dry_run=True,
            ),
            cwd=repo,
            environment=os.environ.copy(),
            timeout_seconds=30,
            log_path=directory / "cpu_mismatch.log",
            samples_path=None,
            command_ledger=ledger,
        )
        gpu_ids = list(pools["e8_gpu_ids"])
        alternate_gpu = gpu_ids[:-1] if len(gpu_ids) > 1 else [gpu_ids[0], "999999"]
        gpu_mismatch = run_command(
            pool_command(
                repo, cpu_pool=pools["e8_cpu_pool"], identity=e8_identity,
                command=[*harmless, "--gpus", ",".join(alternate_gpu)],
                gpu_ids=alternate_gpu, dry_run=True,
            ),
            cwd=repo,
            environment=os.environ.copy(),
            timeout_seconds=30,
            log_path=directory / "gpu_mismatch.log",
            samples_path=None,
            command_ledger=ledger,
        )
        if cpu_mismatch.returncode == 0 or gpu_mismatch.returncode == 0:
            raise AcceptanceError("resource-pool mismatch did not fail closed")
        details = {
            "commands": results,
            "e7_child": e7_child,
            "e8_child": e8_child,
            "cpu_mismatch_returncode": cpu_mismatch.returncode,
            "gpu_mismatch_returncode": gpu_mismatch.returncode,
        }
        return stage_result(root, "stage1_resource_pool", "PASS", started, details)
    except BaseException as exc:
        return stage_result(
            root,
            "stage1_resource_pool",
            "FAIL",
            started,
            {"error_type": type(exc).__name__, "error": str(exc), "commands": results},
        )


def e7_stage(
    root: Path,
    repo: Path,
    profile_path: Path,
    profile: Mapping[str, Any],
    ledger: Path,
) -> StageResult:
    started = utc_now()
    directory = root / "stage2_e7_cpu_v2"
    if not profile["e7"]["enabled"]:
        return stage_result(
            root, "stage2_e7_cpu_v2", "NOT_RUN", started, {"reason": "E7 disabled"}
        )
    work = directory / "work"
    identity = directory / "RESOURCE_POOL.json"
    pool = profile["resource_pools"]["e7_cpu_pool"]
    try:
        plan = run_command(
            pool_command(
                repo, cpu_pool=pool, identity=identity,
                command=e7_plan_command(repo, profile, work),
            ),
            cwd=repo,
            environment=os.environ.copy(),
            timeout_seconds=float(profile["e7"]["plan_timeout_seconds"]),
            log_path=directory / "plan.log",
            samples_path=directory / "plan_samples.jsonl",
            command_ledger=ledger,
        )
        if not plan.ok:
            raise AcceptanceError("E7 plan failed")
        workers, digest = selection_identity(work)
        selection = load_json(work / "RUNTIME_SELECTION.json")
        selection_hash = sha256_file(work / "RUNTIME_SELECTION.json")
        for action, output in (
            ("validate", directory / "REVALIDATION_ONLY.json"),
            ("liveness", directory / "SELECTED_LIVENESS.json"),
        ):
            timeout = (
                float(profile["e7"]["liveness_timeout_seconds"]) + 180
                if action == "liveness"
                else 120
            )
            result = run_command(
                pool_command(
                    repo, cpu_pool=pool, identity=identity,
                    command=internal_e7_command(repo, profile_path, action, work, output),
                ),
                cwd=repo,
                environment=os.environ.copy(),
                timeout_seconds=timeout,
                log_path=directory / f"{action}.log",
                samples_path=directory / f"{action}_samples.jsonl",
                command_ledger=ledger,
            )
            if not result.ok:
                raise AcceptanceError(f"E7 {action} failed")
        if sha256_file(work / "RUNTIME_SELECTION.json") != selection_hash:
            raise AcceptanceError("E7 immutable selection changed")
        numerical = numerical_matches(list(directory.rglob("*")))
        if numerical:
            raise AcceptanceError("E7 logs contain NaN/Inf indicators")
        above_one = candidate_above_one(selection)
        status = "PASS" if above_one else "INCONCLUSIVE"
        return stage_result(
            root,
            "stage2_e7_cpu_v2",
            status,
            started,
            {
                "selected_workers": workers,
                "selection_digest": digest,
                "selection_sha256": selection_hash,
                "candidate_above_one_observed": above_one,
                "nan_inf_matches": numerical,
                "full_scientific_matrix_started": False,
            },
        )
    except BaseException as exc:
        return stage_result(
            root,
            "stage2_e7_cpu_v2",
            "FAIL",
            started,
            {"error_type": type(exc).__name__, "error": str(exc)},
        )
