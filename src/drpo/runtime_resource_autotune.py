"""Small, auditable runtime resource selection helpers for DRPO workloads.

This module owns runtime-capacity decisions only. It never changes scientific
matrices, seeds, model settings, batch sizes, horizons, or evaluation protocols.
The first production scope is deliberately narrow:

* E7: choose active subprocess count from CPU and measured host-memory capacity.
* E8: choose idle, visible GPU slots with host-RAM and VRAM safety gates.

The selectors are conservative capacity guards, not a general distributed
scheduler and not a throughput-optimality claim.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


GIB = 1024**3
MIB = 1024**2


class RuntimeResourceError(RuntimeError):
    """Raised when a safe runtime schedule cannot be selected."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json_sha256(payload: Any) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def atomic_write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)


def load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeResourceError(f"JSON root must be an object: {path}")
    return value


def git_state(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()

    def run(*args: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(root), *args],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=15,
        ).strip()

    try:
        return {
            "commit": run("rev-parse", "HEAD"),
            "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(run("status", "--porcelain")),
        }
    except (OSError, subprocess.SubprocessError):
        return {"commit": "unknown", "branch": "unknown", "dirty": True}


@dataclasses.dataclass(frozen=True)
class GPUDevice:
    index: str
    name: str
    memory_total_bytes: int
    memory_free_bytes: int
    utilization_percent: float

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class MachineSnapshot:
    logical_cpu_count: int
    memory_total_bytes: int
    memory_available_bytes: int
    effective_memory_limit_bytes: int
    effective_memory_current_bytes: int
    effective_memory_available_bytes: int
    swap_total_bytes: int
    swap_free_bytes: int
    cgroup_version: str | None
    load_average_1m: float
    gpus: tuple[GPUDevice, ...]

    def static_identity(self) -> dict[str, Any]:
        return {
            "logical_cpu_count": self.logical_cpu_count,
            "effective_memory_limit_bytes": self.effective_memory_limit_bytes,
            "gpus": [
                {
                    "index": gpu.index,
                    "name": gpu.name,
                    "memory_total_bytes": gpu.memory_total_bytes,
                }
                for gpu in self.gpus
            ],
        }

    def as_dict(self) -> dict[str, Any]:
        value = dataclasses.asdict(self)
        value["gpus"] = [gpu.as_dict() for gpu in self.gpus]
        value["swap_used_bytes"] = max(0, self.swap_total_bytes - self.swap_free_bytes)
        return value


@dataclasses.dataclass(frozen=True)
class CommandProbeResult:
    command: tuple[str, ...]
    started_utc: str
    finished_utc: str
    elapsed_seconds: float
    peak_rss_bytes: int
    returncode: int
    timed_out: bool
    log_path: str

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class CPUSelection:
    selected_workers: int
    cpu_limit: int
    memory_limit: int | None
    task_limit: int
    configured_limit: int | None
    growth_limit: int
    fallback_workers: int
    per_worker_peak_bytes: int | None
    per_worker_reserved_bytes: int | None
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class GPUSelection:
    selected_device_ids: tuple[str, ...]
    rejected_devices: tuple[dict[str, Any], ...]
    required_free_bytes_per_device: int
    headroom_fraction: float
    maximum_utilization_percent: float
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "selected_device_ids": list(self.selected_device_ids),
            "rejected_devices": [dict(item) for item in self.rejected_devices],
            "required_free_bytes_per_device": self.required_free_bytes_per_device,
            "headroom_fraction": self.headroom_fraction,
            "maximum_utilization_percent": self.maximum_utilization_percent,
            "reason": self.reason,
        }


def _read_meminfo(path: Path) -> dict[str, int]:
    values: dict[str, int] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise RuntimeResourceError(f"cannot read memory information: {path}") from exc
    for line in lines:
        if ":" not in line:
            continue
        key, raw = line.split(":", 1)
        tokens = raw.strip().split()
        if not tokens:
            continue
        try:
            number = int(tokens[0])
        except ValueError:
            continue
        multiplier = 1024 if len(tokens) > 1 and tokens[1].lower() == "kb" else 1
        values[key] = number * multiplier
    return values


def _read_optional_int(path: Path) -> int | None:
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not text or text == "max":
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _cgroup_memory(cgroup_root: Path) -> tuple[str | None, int | None, int | None]:
    v2_limit = _read_optional_int(cgroup_root / "memory.max")
    v2_current = _read_optional_int(cgroup_root / "memory.current")
    if (
        (cgroup_root / "cgroup.controllers").exists()
        or v2_limit is not None
        or v2_current is not None
    ):
        return "v2", v2_limit, v2_current

    v1_root = cgroup_root / "memory"
    v1_limit = _read_optional_int(v1_root / "memory.limit_in_bytes")
    v1_current = _read_optional_int(v1_root / "memory.usage_in_bytes")
    if v1_limit is not None or v1_current is not None:
        return "v1", v1_limit, v1_current
    return None, None, None


def discover_gpus(*, nvidia_smi: str = "nvidia-smi") -> tuple[GPUDevice, ...]:
    command = [
        nvidia_smi,
        "--query-gpu=index,name,memory.total,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(
            command,
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return ()

    devices: list[GPUDevice] = []
    for line in output.splitlines():
        parts = [item.strip() for item in line.split(",")]
        if len(parts) != 5:
            continue
        index, name, total_mib, free_mib, utilization = parts
        try:
            devices.append(
                GPUDevice(
                    index=index,
                    name=name,
                    memory_total_bytes=int(float(total_mib) * MIB),
                    memory_free_bytes=int(float(free_mib) * MIB),
                    utilization_percent=float(utilization),
                )
            )
        except ValueError:
            continue
    return tuple(devices)


def discover_machine(
    *,
    meminfo_path: str | Path = "/proc/meminfo",
    loadavg_path: str | Path = "/proc/loadavg",
    cgroup_root: str | Path = "/sys/fs/cgroup",
    nvidia_smi: str = "nvidia-smi",
) -> MachineSnapshot:
    meminfo = _read_meminfo(Path(meminfo_path))
    total = int(meminfo.get("MemTotal", 0))
    available = int(meminfo.get("MemAvailable", meminfo.get("MemFree", 0)))
    if total <= 0 or available < 0:
        raise RuntimeResourceError("invalid /proc/meminfo values")

    if hasattr(os, "sched_getaffinity"):
        logical_cpu_count = len(os.sched_getaffinity(0))
    else:  # pragma: no cover - Linux production path uses sched_getaffinity
        logical_cpu_count = int(os.cpu_count() or 1)
    logical_cpu_count = max(1, logical_cpu_count)

    try:
        load_average_1m = float(
            Path(loadavg_path).read_text(encoding="utf-8").split()[0]
        )
    except (OSError, ValueError, IndexError):
        load_average_1m = 0.0

    cgroup_version, cgroup_limit, cgroup_current = _cgroup_memory(Path(cgroup_root))
    effective_limit = total
    if cgroup_limit is not None and 0 < cgroup_limit < effective_limit:
        effective_limit = cgroup_limit

    host_current = max(0, total - available)
    effective_current = host_current
    if cgroup_current is not None and cgroup_limit is not None and cgroup_limit <= total:
        effective_current = max(0, cgroup_current)
    effective_available = max(0, min(available, effective_limit - effective_current))

    return MachineSnapshot(
        logical_cpu_count=logical_cpu_count,
        memory_total_bytes=total,
        memory_available_bytes=available,
        effective_memory_limit_bytes=effective_limit,
        effective_memory_current_bytes=effective_current,
        effective_memory_available_bytes=effective_available,
        swap_total_bytes=int(meminfo.get("SwapTotal", 0)),
        swap_free_bytes=int(meminfo.get("SwapFree", 0)),
        cgroup_version=cgroup_version,
        load_average_1m=max(0.0, load_average_1m),
        gpus=discover_gpus(nvidia_smi=nvidia_smi),
    )


def select_cpu_workers(
    snapshot: MachineSnapshot,
    *,
    total_tasks: int,
    fallback_workers: int,
    per_worker_peak_bytes: int | None,
    cpu_fraction: float = 0.85,
    memory_headroom_fraction: float = 0.15,
    per_worker_safety_factor: float = 1.20,
    max_workers: int | None = None,
    max_growth_factor: float = 3.0,
) -> CPUSelection:
    if total_tasks < 1:
        raise RuntimeResourceError("total_tasks must be positive")
    if fallback_workers < 1:
        raise RuntimeResourceError("fallback_workers must be positive")
    if not 0.05 <= cpu_fraction <= 1.0:
        raise RuntimeResourceError("cpu_fraction must be in [0.05, 1.0]")
    if not 0.0 <= memory_headroom_fraction < 0.9:
        raise RuntimeResourceError("memory_headroom_fraction must be in [0, 0.9)")
    if per_worker_safety_factor < 1.0:
        raise RuntimeResourceError("per_worker_safety_factor must be >= 1")
    if max_growth_factor < 1.0:
        raise RuntimeResourceError("max_growth_factor must be >= 1")
    if max_workers is not None and max_workers < 1:
        raise RuntimeResourceError("max_workers must be positive")

    cpu_budget = snapshot.logical_cpu_count * cpu_fraction
    cpu_limit = max(1, math.floor(cpu_budget - snapshot.load_average_1m))
    configured_limit = max_workers
    growth_limit = max(1, math.floor(fallback_workers * max_growth_factor))
    task_limit = total_tasks
    limits = [cpu_limit, task_limit, growth_limit]
    if configured_limit is not None:
        limits.append(configured_limit)

    memory_limit: int | None = None
    reserved_per_worker: int | None = None
    if per_worker_peak_bytes is not None:
        if per_worker_peak_bytes <= 0:
            raise RuntimeResourceError("per_worker_peak_bytes must be positive")
        reserved_per_worker = max(1, math.ceil(per_worker_peak_bytes * per_worker_safety_factor))
        usable = max(
            0,
            math.floor(
                snapshot.effective_memory_available_bytes
                * (1.0 - memory_headroom_fraction)
            ),
        )
        memory_limit = usable // reserved_per_worker
        if memory_limit < 1:
            raise RuntimeResourceError(
                "insufficient host memory for one worker after safety headroom"
            )
        limits.append(memory_limit)
        reason = "capacity_model_with_measured_worker_memory"
    else:
        limits.append(fallback_workers)
        reason = "no_worker_memory_probe_use_conservative_fallback"

    selected = max(1, min(limits))
    return CPUSelection(
        selected_workers=selected,
        cpu_limit=cpu_limit,
        memory_limit=memory_limit,
        task_limit=task_limit,
        configured_limit=configured_limit,
        growth_limit=growth_limit,
        fallback_workers=fallback_workers,
        per_worker_peak_bytes=per_worker_peak_bytes,
        per_worker_reserved_bytes=reserved_per_worker,
        reason=reason,
    )


def select_gpu_devices(
    snapshot: MachineSnapshot,
    *,
    candidate_device_ids: Sequence[str],
    total_tasks: int,
    required_free_bytes_per_device: int,
    headroom_fraction: float = 0.12,
    maximum_utilization_percent: float = 20.0,
    max_devices: int | None = None,
) -> GPUSelection:
    if total_tasks < 1:
        raise RuntimeResourceError("total_tasks must be positive")
    if required_free_bytes_per_device <= 0:
        raise RuntimeResourceError("required_free_bytes_per_device must be positive")
    if not 0.0 <= headroom_fraction < 0.9:
        raise RuntimeResourceError("headroom_fraction must be in [0, 0.9)")
    if not 0.0 <= maximum_utilization_percent <= 100.0:
        raise RuntimeResourceError("maximum_utilization_percent must be in [0, 100]")
    normalized = [str(value).strip() for value in candidate_device_ids if str(value).strip()]
    if not normalized:
        raise RuntimeResourceError("candidate_device_ids must be non-empty")
    if len(normalized) != len(set(normalized)):
        raise RuntimeResourceError("candidate_device_ids must be unique")
    if max_devices is not None and max_devices < 1:
        raise RuntimeResourceError("max_devices must be positive")

    inventory = {gpu.index: gpu for gpu in snapshot.gpus}
    threshold = math.ceil(required_free_bytes_per_device / (1.0 - headroom_fraction))
    selected: list[str] = []
    rejected: list[dict[str, Any]] = []
    for device_id in normalized:
        gpu = inventory.get(device_id)
        if gpu is None:
            rejected.append({"device_id": device_id, "reason": "not_visible"})
            continue
        if gpu.utilization_percent > maximum_utilization_percent:
            rejected.append(
                {
                    "device_id": device_id,
                    "reason": "device_busy",
                    "utilization_percent": gpu.utilization_percent,
                    "maximum_utilization_percent": maximum_utilization_percent,
                }
            )
            continue
        if gpu.memory_free_bytes < threshold:
            rejected.append(
                {
                    "device_id": device_id,
                    "reason": "insufficient_free_memory",
                    "memory_free_bytes": gpu.memory_free_bytes,
                    "required_with_headroom_bytes": threshold,
                }
            )
            continue
        selected.append(device_id)

    limit = min(total_tasks, len(selected))
    if max_devices is not None:
        limit = min(limit, max_devices)
    selected = selected[:limit]
    if not selected:
        raise RuntimeResourceError(
            "no GPU satisfies visibility, utilization, and free-memory gates"
        )
    return GPUSelection(
        selected_device_ids=tuple(selected),
        rejected_devices=tuple(rejected),
        required_free_bytes_per_device=required_free_bytes_per_device,
        headroom_fraction=headroom_fraction,
        maximum_utilization_percent=maximum_utilization_percent,
        reason="visible_idle_unique_devices_with_required_free_memory",
    )


def _read_process_ppid(stat_path: Path) -> int | None:
    try:
        text = stat_path.read_text(encoding="utf-8")
    except OSError:
        return None
    close = text.rfind(")")
    if close < 0:
        return None
    fields = text[close + 2 :].split()
    if len(fields) < 2:
        return None
    try:
        return int(fields[1])
    except ValueError:
        return None


def _read_process_rss(status_path: Path) -> int:
    try:
        lines = status_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return 0
    for line in lines:
        if line.startswith("VmRSS:"):
            tokens = line.split()
            if len(tokens) >= 2:
                try:
                    return int(tokens[1]) * 1024
                except ValueError:
                    return 0
    return 0


def process_tree_rss(root_pid: int, *, proc_root: str | Path = "/proc") -> int:
    root = Path(proc_root)
    parent_map: dict[int, int] = {}
    try:
        entries = list(root.iterdir())
    except OSError:
        return 0
    for entry in entries:
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        ppid = _read_process_ppid(entry / "stat")
        if ppid is not None:
            parent_map[pid] = ppid

    descendants = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, ppid in parent_map.items():
            if ppid in descendants and pid not in descendants:
                descendants.add(pid)
                changed = True
    return sum(_read_process_rss(root / str(pid) / "status") for pid in descendants)


def _terminate_process_group(process: subprocess.Popen[Any], grace_seconds: float) -> int:
    if process.poll() is not None:
        return int(process.returncode or 0)
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        return int(process.wait(timeout=grace_seconds))
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        return int(process.wait())


def measure_command_peak_memory(
    command: Sequence[str],
    *,
    cwd: str | Path,
    environment: Mapping[str, str] | None,
    log_path: str | Path,
    sample_seconds: float,
    sample_interval_seconds: float = 0.20,
    terminate_grace_seconds: float = 5.0,
    accept_timeout: bool = True,
) -> CommandProbeResult:
    if not command:
        raise RuntimeResourceError("probe command must be non-empty")
    if sample_seconds <= 0:
        raise RuntimeResourceError("sample_seconds must be positive")
    if sample_interval_seconds <= 0:
        raise RuntimeResourceError("sample_interval_seconds must be positive")

    target_log = Path(log_path)
    target_log.parent.mkdir(parents=True, exist_ok=True)
    started_utc = utc_now()
    started = time.monotonic()
    peak_rss = 0
    timed_out = False
    with target_log.open("w", encoding="utf-8") as handle:
        handle.write("COMMAND=" + " ".join(str(item) for item in command) + "\n")
        handle.flush()
        try:
            process = subprocess.Popen(
                [str(item) for item in command],
                cwd=str(Path(cwd)),
                env=None if environment is None else dict(environment),
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
        except OSError as exc:
            raise RuntimeResourceError(f"cannot start resource probe: {exc}") from exc

        deadline = started + sample_seconds
        while True:
            peak_rss = max(peak_rss, process_tree_rss(process.pid))
            returncode = process.poll()
            if returncode is not None:
                break
            if time.monotonic() >= deadline:
                timed_out = True
                returncode = _terminate_process_group(process, terminate_grace_seconds)
                break
            time.sleep(sample_interval_seconds)

    elapsed = time.monotonic() - started
    returncode = int(returncode)
    if timed_out and not accept_timeout:
        raise RuntimeResourceError("resource probe exceeded its time budget")
    if not timed_out and returncode != 0:
        raise RuntimeResourceError(
            f"resource probe exited before timeout with return code {returncode}; "
            f"see {target_log}"
        )
    if peak_rss <= 0:
        raise RuntimeResourceError("resource probe did not expose a positive process-tree RSS")
    return CommandProbeResult(
        command=tuple(str(item) for item in command),
        started_utc=started_utc,
        finished_utc=utc_now(),
        elapsed_seconds=elapsed,
        peak_rss_bytes=peak_rss,
        returncode=returncode,
        timed_out=timed_out,
        log_path=str(target_log),
    )


def selection_document(
    *,
    adapter_id: str,
    resource_fingerprint: Mapping[str, Any],
    machine: MachineSnapshot,
    mode: str,
    selection: Mapping[str, Any],
    probe: Mapping[str, Any] | None,
    fallback: Mapping[str, Any],
    repo_root: str | Path,
    limitations: Sequence[str] = (),
) -> dict[str, Any]:
    if mode not in {"auto", "cached", "fixed", "exempt"}:
        raise RuntimeResourceError(f"unsupported selection mode: {mode}")
    fingerprint_payload = dict(resource_fingerprint)
    return {
        "schema_version": 1,
        "adapter_id": adapter_id,
        "created_utc": utc_now(),
        "source": git_state(repo_root),
        "mode": mode,
        "resource_fingerprint": fingerprint_payload,
        "resource_fingerprint_sha256": canonical_json_sha256(fingerprint_payload),
        "machine_snapshot": machine.as_dict(),
        "machine_static_sha256": canonical_json_sha256(machine.static_identity()),
        "selection": dict(selection),
        "probe": None if probe is None else dict(probe),
        "fallback": dict(fallback),
        "limitations": list(limitations),
        "scientific_matrix_changed": False,
    }
