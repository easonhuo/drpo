"""Read-only cgroup v2 exclusive-partition evidence for runtime acceptance."""
from __future__ import annotations

import os
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

from drpo.runtime_resource_pool import parse_cpu_pool


def read_unified_cgroup_path(path: str | Path) -> str | None:
    """Read the cgroup v2 path from one ``/proc/<pid>/cgroup`` file."""

    try:
        lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    for line in lines:
        fields = line.split(":", 2)
        if len(fields) == 3 and fields[0] == "0" and fields[1] == "":
            value = fields[2].removesuffix(" (deleted)").strip()
            return value if value.startswith("/") else None
    return None


def _within(child: str, parent: str) -> bool:
    child_parts = PurePosixPath(child).parts
    parent_parts = PurePosixPath(parent).parts
    return child_parts[: len(parent_parts)] == parent_parts


def _cpu_ids(path: Path) -> list[int]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    return list(parse_cpu_pool(text))


def ancestor_pids(
    *,
    start_pid: int | None = None,
    proc_root: str | Path = "/proc",
) -> set[int]:
    """Return the current process and its visible parent chain."""

    root = Path(proc_root)
    values: set[int] = set()
    current = os.getpid() if start_pid is None else int(start_pid)
    for _ in range(128):
        if current <= 0 or current in values:
            break
        values.add(current)
        try:
            text = (root / str(current) / "stat").read_text(encoding="utf-8")
        except OSError:
            break
        close = text.rfind(")")
        fields = text[close + 2 :].split() if close >= 0 else []
        if len(fields) < 2:
            break
        try:
            current = int(fields[1])
        except ValueError:
            break
    return values


def process_inventory(
    *,
    proc_root: str | Path = "/proc",
) -> list[dict[str, Any]]:
    """Collect command, affinity, and cgroup v2 identity for visible processes."""

    root = Path(proc_root)
    rows: list[dict[str, Any]] = []
    for entry in root.iterdir():
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
        rows.append(
            {
                "pid": pid,
                "affinity_cpu_ids": affinity,
                "cgroup_v2_path": read_unified_cgroup_path(entry / "cgroup"),
                "command": command,
            }
        )
    return sorted(rows, key=lambda row: int(row["pid"]))


def _partition_candidate(
    *,
    current_cgroup_path: str,
    cgroup_mount: Path,
    reserved_cpu_ids: set[int],
) -> dict[str, Any]:
    current = cgroup_mount / current_cgroup_path.lstrip("/")
    mount = cgroup_mount.resolve()
    try:
        current.resolve().relative_to(mount)
    except (OSError, ValueError):
        return {
            "exclusive_partition_proven": False,
            "partition_error": "current cgroup is outside the cgroup v2 mount",
        }

    cursor = current
    while True:
        state_path = cursor / "cpuset.cpus.partition"
        if state_path.is_file():
            state = state_path.read_text(encoding="utf-8").strip()
            if state.startswith("root invalid") or state.startswith("isolated invalid"):
                return {
                    "exclusive_partition_proven": False,
                    "partition_path": "/" + cursor.relative_to(mount).as_posix(),
                    "partition_state": state,
                    "partition_error": "cpuset partition is invalid",
                }
            if state in {"root", "isolated"}:
                effective = _cpu_ids(cursor / "cpuset.cpus.effective")
                exclusive = _cpu_ids(cursor / "cpuset.cpus.exclusive.effective")
                missing_effective = sorted(reserved_cpu_ids - set(effective))
                missing_exclusive = sorted(reserved_cpu_ids - set(exclusive))
                partition_path = "/" + cursor.relative_to(mount).as_posix()
                if partition_path == "/.":
                    partition_path = "/"
                return {
                    "exclusive_partition_proven": not (
                        missing_effective or missing_exclusive
                    ),
                    "partition_path": partition_path,
                    "partition_state": state,
                    "partition_effective_cpu_ids": effective,
                    "partition_exclusive_cpu_ids": exclusive,
                    "missing_effective_cpu_ids": missing_effective,
                    "missing_exclusive_cpu_ids": missing_exclusive,
                    "partition_error": (
                        None
                        if not missing_effective and not missing_exclusive
                        else "reserved CPUs are not fully exclusive in the partition"
                    ),
                }
        if cursor == mount:
            break
        cursor = cursor.parent
    return {
        "exclusive_partition_proven": False,
        "partition_error": "no valid cgroup v2 cpuset partition root contains the harness",
    }


def audit_resource_isolation(
    *,
    inventory: Sequence[Mapping[str, Any]],
    conflict_patterns: Iterable[str],
    reserved_cpu_ids: Iterable[int],
    excluded_pids: Iterable[int],
    current_cgroup_path: str | None = None,
    cgroup_mount: str | Path = "/sys/fs/cgroup",
) -> dict[str, Any]:
    """Classify permanent external workloads using exclusive-partition evidence.

    When a valid cgroup v2 cpuset partition contains every reserved E7/E8 CPU,
    matched processes outside that partition are evidence-only, not conflicts.
    The function never creates cgroups or migrates, kills, renices, or rebinds tasks.
    """

    reserved = {int(value) for value in reserved_cpu_ids}
    excluded = {int(value) for value in excluded_pids}
    patterns = [str(value).lower() for value in conflict_patterns]
    if current_cgroup_path is None:
        current_cgroup_path = read_unified_cgroup_path("/proc/self/cgroup")

    base: dict[str, Any] = {
        "mode": "shared_host_process_conflicts",
        "current_cgroup_path": current_cgroup_path,
        "reserved_cpu_ids": sorted(reserved),
        "exclusive_partition_proven": False,
        "isolated_external_matches": [],
        "partition_contaminants": [],
        "conflicts": [],
    }
    if current_cgroup_path is None:
        base["partition_error"] = "cannot resolve current cgroup v2 path"
        partition = base
    else:
        partition = {
            **base,
            **_partition_candidate(
                current_cgroup_path=current_cgroup_path,
                cgroup_mount=Path(cgroup_mount),
                reserved_cpu_ids=reserved,
            ),
        }

    matched_rows: list[dict[str, Any]] = []
    for raw in inventory:
        row = dict(raw)
        if int(row.get("pid", -1)) in excluded:
            continue
        command = str(row.get("command", "")).lower()
        if any(pattern in command for pattern in patterns):
            matched_rows.append(row)

    if not partition["exclusive_partition_proven"]:
        partition["external_matches"] = matched_rows
        partition["conflicts"] = matched_rows
        partition["ready"] = not matched_rows
        return partition

    partition["mode"] = "cgroup_v2_exclusive_partition"
    partition_path = str(partition["partition_path"])
    contaminants: list[dict[str, Any]] = []
    isolated: list[dict[str, Any]] = []
    conflicts_by_pid: dict[int, dict[str, Any]] = {}

    for raw in inventory:
        row = dict(raw)
        pid = int(row.get("pid", -1))
        if pid in excluded:
            continue
        cgroup_path = row.get("cgroup_v2_path")
        inside = isinstance(cgroup_path, str) and _within(cgroup_path, partition_path)
        overlap = sorted(reserved & {int(value) for value in row.get("affinity_cpu_ids", [])})
        matched = any(
            pattern in str(row.get("command", "")).lower() for pattern in patterns
        )
        if inside:
            enriched = {**row, "isolation_reason": "process_inside_acceptance_partition"}
            contaminants.append(enriched)
            conflicts_by_pid[pid] = enriched
        elif matched and (cgroup_path is None or overlap):
            enriched = {
                **row,
                "reserved_cpu_overlap": overlap,
                "isolation_reason": (
                    "unresolved_cgroup_membership"
                    if cgroup_path is None
                    else "outside_process_affinity_overlaps_reserved_partition"
                ),
            }
            conflicts_by_pid[pid] = enriched
        elif matched:
            isolated.append(
                {
                    **row,
                    "reserved_cpu_overlap": overlap,
                    "isolation_reason": "outside_exclusive_partition",
                }
            )

    partition["external_matches"] = matched_rows
    partition["isolated_external_matches"] = isolated
    partition["partition_contaminants"] = contaminants
    partition["conflicts"] = [
        conflicts_by_pid[pid] for pid in sorted(conflicts_by_pid)
    ]
    partition["ready"] = not partition["conflicts"]
    return partition
