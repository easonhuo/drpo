"""Measured Linux CPU capacity helpers for DRPO runtime autotuning.

This module observes CPU affinity, cgroup quota domains, host execution occupancy,
and process-tree CPU demand. It contains no workload, scheduler, or scientific
logic. Load average is deliberately excluded from capacity arithmetic.
"""
from __future__ import annotations

import dataclasses
import math
import os
import time
from pathlib import Path
from typing import Any, Callable, Sequence


class CPUCapacityError(RuntimeError):
    """Raised when CPU capacity cannot be measured safely."""


@dataclasses.dataclass(frozen=True)
class CPUQuotaDomain:
    path: str
    quota_cores: float
    usage_path: str
    usage_kind: str
    cgroup_version: str

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class CPUBinding:
    affinity_cpu_ids: tuple[int, ...]
    affinity_source: str
    cgroup_version: str | None
    current_cgroup_path: str | None
    quota_domains: tuple[CPUQuotaDomain, ...]

    @property
    def affinity_capacity_cores(self) -> int:
        return len(self.affinity_cpu_ids)

    @property
    def effective_cpu_capacity_cores(self) -> float:
        values = [float(self.affinity_capacity_cores)]
        values.extend(domain.quota_cores for domain in self.quota_domains)
        return min(values)

    def as_dict(self) -> dict[str, Any]:
        return {
            "affinity_cpu_ids": list(self.affinity_cpu_ids),
            "affinity_capacity_cores": self.affinity_capacity_cores,
            "affinity_source": self.affinity_source,
            "cgroup_version": self.cgroup_version,
            "current_cgroup_path": self.current_cgroup_path,
            "quota_domains": [domain.as_dict() for domain in self.quota_domains],
            "effective_cpu_capacity_cores": self.effective_cpu_capacity_cores,
        }


@dataclasses.dataclass(frozen=True)
class CPUCounterSnapshot:
    monotonic_seconds: float
    affinity_cpu_ids: tuple[int, ...]
    system_busy_ticks: int
    system_total_ticks: int
    quota_usage_seconds: tuple[tuple[str, float], ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "monotonic_seconds": self.monotonic_seconds,
            "affinity_cpu_ids": list(self.affinity_cpu_ids),
            "system_busy_ticks": self.system_busy_ticks,
            "system_total_ticks": self.system_total_ticks,
            "quota_usage_seconds": {
                path: seconds for path, seconds in self.quota_usage_seconds
            },
        }


@dataclasses.dataclass(frozen=True)
class CPUIntervalMeasurement:
    elapsed_seconds: float
    affinity_cpu_ids: tuple[int, ...]
    system_busy_tick_delta: int
    system_total_tick_delta: int
    system_busy_cores: float
    quota_domain_usage_cores: tuple[tuple[str, float], ...]
    started_monotonic_seconds: float
    finished_monotonic_seconds: float

    def quota_usage_map(self) -> dict[str, float]:
        return dict(self.quota_domain_usage_cores)

    def as_dict(self) -> dict[str, Any]:
        return {
            "elapsed_seconds": self.elapsed_seconds,
            "affinity_cpu_ids": list(self.affinity_cpu_ids),
            "system_busy_tick_delta": self.system_busy_tick_delta,
            "system_total_tick_delta": self.system_total_tick_delta,
            "system_busy_cores": self.system_busy_cores,
            "quota_domain_usage_cores": self.quota_usage_map(),
            "started_monotonic_seconds": self.started_monotonic_seconds,
            "finished_monotonic_seconds": self.finished_monotonic_seconds,
        }


@dataclasses.dataclass(frozen=True)
class CPUWorkerCapacity:
    reserved_cpu_cores_per_worker: float
    measured_cpu_cores_per_worker: float
    affinity_budget_cores: float
    affinity_external_busy_cores: float
    affinity_worker_budget_cores: float
    quota_domain_budgets: tuple[tuple[str, float], ...]
    quota_domain_external_busy_cores: tuple[tuple[str, float], ...]
    worker_cpu_budget_cores: float
    cpu_worker_limit: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "reserved_cpu_cores_per_worker": self.reserved_cpu_cores_per_worker,
            "measured_cpu_cores_per_worker": self.measured_cpu_cores_per_worker,
            "affinity_budget_cores": self.affinity_budget_cores,
            "affinity_external_busy_cores": self.affinity_external_busy_cores,
            "affinity_worker_budget_cores": self.affinity_worker_budget_cores,
            "quota_domain_budgets": dict(self.quota_domain_budgets),
            "quota_domain_external_busy_cores": dict(
                self.quota_domain_external_busy_cores
            ),
            "worker_cpu_budget_cores": self.worker_cpu_budget_cores,
            "cpu_worker_limit": self.cpu_worker_limit,
        }


def _affinity_cpu_ids() -> tuple[tuple[int, ...], str]:
    if hasattr(os, "sched_getaffinity"):
        values = tuple(sorted(int(value) for value in os.sched_getaffinity(0)))
        if not values:
            raise CPUCapacityError("sched_getaffinity returned an empty CPU set")
        return values, "sched_getaffinity"
    count = int(os.cpu_count() or 0)
    if count < 1:
        raise CPUCapacityError("no process-visible CPU count is available")
    return tuple(range(count)), "os_cpu_count_fallback"


def _parse_self_cgroup(path: Path) -> tuple[str | None, str | None]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None, None
    v2_paths: list[str] = []
    v1_paths: list[str] = []
    for line in lines:
        fields = line.split(":", 2)
        if len(fields) != 3:
            raise CPUCapacityError(f"malformed cgroup membership line: {line!r}")
        _hierarchy, controllers, relative = fields
        if not relative.startswith("/"):
            raise CPUCapacityError("cgroup membership path must be absolute")
        if controllers == "":
            v2_paths.append(relative)
            continue
        names = {item.strip() for item in controllers.split(",") if item.strip()}
        if "cpu" in names:
            v1_paths.append(relative)
    if v2_paths and v1_paths:
        raise CPUCapacityError("contradictory cgroup v1 and v2 CPU membership")
    if len(set(v2_paths)) > 1 or len(set(v1_paths)) > 1:
        raise CPUCapacityError("multiple conflicting CPU cgroup membership paths")
    if v2_paths:
        return "v2", v2_paths[0]
    if v1_paths:
        return "v1", v1_paths[0]
    return None, None


def _safe_join(root: Path, relative: str) -> Path:
    root_resolved = root.resolve()
    candidate = (root_resolved / relative.lstrip("/")).resolve()
    try:
        candidate.relative_to(root_resolved)
    except ValueError as exc:
        raise CPUCapacityError("cgroup path escapes configured mount root") from exc
    return candidate


def _ancestor_paths(active: Path, root: Path) -> tuple[Path, ...]:
    active = active.resolve()
    root = root.resolve()
    try:
        active.relative_to(root)
    except ValueError as exc:
        raise CPUCapacityError("active cgroup is outside configured mount root") from exc
    values: list[Path] = []
    current = active
    while True:
        values.append(current)
        if current == root:
            break
        current = current.parent
    return tuple(values)


def _pid_is_listed(path: Path, pid: int) -> bool:
    try:
        values = path.read_text(encoding="utf-8").split()
    except OSError:
        return False
    return str(int(pid)) in values


def _resolve_active_cgroup(
    *,
    root: Path,
    relative: str,
    membership_filename: str,
    version_label: str,
) -> Path:
    root = root.resolve()
    if not root.is_dir():
        raise CPUCapacityError(f"configured {version_label} cgroup root does not exist")
    candidate = _safe_join(root, relative)
    if candidate.is_dir():
        return candidate
    # In a cgroup namespace, /proc/self/cgroup can expose a host-relative path that is
    # not present below the namespaced mount root. Accept the mount root only when it
    # explicitly lists this process as a member; otherwise a missing active path is
    # contradictory evidence and must fail closed.
    if relative != "/" and _pid_is_listed(root / membership_filename, os.getpid()):
        return root
    raise CPUCapacityError(f"current {version_label} cgroup path cannot be resolved")


def _parse_v2_quota(path: Path) -> float | None:
    try:
        tokens = path.read_text(encoding="utf-8").strip().split()
    except OSError as exc:
        raise CPUCapacityError(f"cannot read cgroup v2 quota: {path}") from exc
    if len(tokens) != 2:
        raise CPUCapacityError(f"malformed cgroup v2 quota: {path}")
    quota_raw, period_raw = tokens
    try:
        period = int(period_raw)
    except ValueError as exc:
        raise CPUCapacityError(f"malformed cgroup v2 period: {path}") from exc
    if period <= 0:
        raise CPUCapacityError(f"non-positive cgroup v2 period: {path}")
    if quota_raw == "max":
        return None
    try:
        quota = int(quota_raw)
    except ValueError as exc:
        raise CPUCapacityError(f"malformed cgroup v2 quota: {path}") from exc
    if quota <= 0:
        raise CPUCapacityError(f"non-positive cgroup v2 quota: {path}")
    return quota / period


def _parse_v1_quota(directory: Path) -> float | None:
    quota_path = directory / "cpu.cfs_quota_us"
    period_path = directory / "cpu.cfs_period_us"
    if quota_path.exists() != period_path.exists():
        raise CPUCapacityError(f"contradictory cgroup v1 quota files: {directory}")
    if not quota_path.exists():
        return None
    try:
        quota = int(quota_path.read_text(encoding="utf-8").strip())
        period = int(period_path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError) as exc:
        raise CPUCapacityError(f"malformed cgroup v1 quota: {directory}") from exc
    if period <= 0:
        raise CPUCapacityError(f"non-positive cgroup v1 period: {directory}")
    if quota < 0:
        return None
    if quota == 0:
        raise CPUCapacityError(f"zero cgroup v1 quota: {directory}")
    return quota / period


def _v2_binding(root: Path, relative: str) -> tuple[Path, tuple[CPUQuotaDomain, ...]]:
    active = _resolve_active_cgroup(
        root=root,
        relative=relative,
        membership_filename="cgroup.procs",
        version_label="cgroup v2",
    )
    domains: list[CPUQuotaDomain] = []
    saw_quota_file = False
    for directory in _ancestor_paths(active, root):
        quota_path = directory / "cpu.max"
        if not quota_path.exists():
            continue
        saw_quota_file = True
        quota = _parse_v2_quota(quota_path)
        if quota is None:
            continue
        usage_path = directory / "cpu.stat"
        if not usage_path.is_file():
            raise CPUCapacityError(f"finite cgroup v2 quota lacks cpu.stat: {directory}")
        domains.append(
            CPUQuotaDomain(
                path=str(directory),
                quota_cores=float(quota),
                usage_path=str(usage_path),
                usage_kind="v2_usage_usec",
                cgroup_version="v2",
            )
        )
    controllers_path = root / "cgroup.controllers"
    controllers = ""
    if controllers_path.is_file():
        try:
            controllers = controllers_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise CPUCapacityError("cannot read cgroup.controllers") from exc
    if not saw_quota_file and "cpu" in controllers.split():
        raise CPUCapacityError("CPU controller is visible but cpu.max is unavailable")
    return active, tuple(domains)


def _v1_binding(root: Path, relative: str) -> tuple[Path, tuple[CPUQuotaDomain, ...]]:
    controller_root = root / "cpu" if (root / "cpu").exists() else root
    active = _resolve_active_cgroup(
        root=controller_root,
        relative=relative,
        membership_filename="tasks",
        version_label="cgroup v1 CPU",
    )
    domains: list[CPUQuotaDomain] = []
    for directory in _ancestor_paths(active, controller_root):
        quota = _parse_v1_quota(directory)
        if quota is None:
            continue
        usage_path = directory / "cpuacct.usage"
        if not usage_path.is_file():
            raise CPUCapacityError(f"finite cgroup v1 quota lacks cpuacct.usage: {directory}")
        domains.append(
            CPUQuotaDomain(
                path=str(directory),
                quota_cores=float(quota),
                usage_path=str(usage_path),
                usage_kind="v1_cpuacct_nanoseconds",
                cgroup_version="v1",
            )
        )
    return active, tuple(domains)


def discover_cpu_binding(
    *,
    cgroup_root: str | Path = "/sys/fs/cgroup",
    proc_self_cgroup_path: str | Path = "/proc/self/cgroup",
) -> CPUBinding:
    affinity_ids, affinity_source = _affinity_cpu_ids()
    root = Path(cgroup_root)
    version, relative = _parse_self_cgroup(Path(proc_self_cgroup_path))
    if version is None or relative is None:
        return CPUBinding(
            affinity_cpu_ids=affinity_ids,
            affinity_source=affinity_source,
            cgroup_version=None,
            current_cgroup_path=None,
            quota_domains=(),
        )
    if version == "v2":
        active, domains = _v2_binding(root, relative)
    elif version == "v1":
        active, domains = _v1_binding(root, relative)
    else:  # pragma: no cover - parser constrains values
        raise CPUCapacityError(f"unsupported cgroup version: {version}")
    return CPUBinding(
        affinity_cpu_ids=affinity_ids,
        affinity_source=affinity_source,
        cgroup_version=version,
        current_cgroup_path=str(active),
        quota_domains=domains,
    )


def _read_proc_stat_ticks(path: Path, cpu_ids: Sequence[int]) -> tuple[int, int]:
    selected = {int(value) for value in cpu_ids}
    observed: set[int] = set()
    busy_total = 0
    tick_total = 0
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise CPUCapacityError(f"cannot read CPU accounting: {path}") from exc
    for line in lines:
        fields = line.split()
        if not fields or fields[0] == "cpu" or not fields[0].startswith("cpu"):
            continue
        suffix = fields[0][3:]
        if not suffix.isdigit():
            continue
        cpu_id = int(suffix)
        if cpu_id not in selected:
            continue
        if cpu_id in observed:
            raise CPUCapacityError(f"duplicate CPU row in {path}: cpu{cpu_id}")
        try:
            values = [int(value) for value in fields[1:]]
        except ValueError as exc:
            raise CPUCapacityError(f"malformed CPU row in {path}: cpu{cpu_id}") from exc
        if len(values) < 8:
            raise CPUCapacityError(f"short CPU row in {path}: cpu{cpu_id}")
        user, nice, system, idle, iowait, irq, softirq, steal = values[:8]
        total = user + nice + system + idle + iowait + irq + softirq + steal
        busy = user + nice + system + irq + softirq + steal
        if min(total, busy) < 0:
            raise CPUCapacityError(f"negative CPU accounting in {path}: cpu{cpu_id}")
        observed.add(cpu_id)
        busy_total += busy
        tick_total += total
    missing = selected - observed
    if missing:
        raise CPUCapacityError(f"missing affinity CPU rows: {sorted(missing)}")
    return busy_total, tick_total


def _read_domain_usage_seconds(domain: CPUQuotaDomain) -> float:
    path = Path(domain.usage_path)
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise CPUCapacityError(f"cannot read cgroup CPU usage: {path}") from exc
    if domain.usage_kind == "v2_usage_usec":
        values: dict[str, str] = {}
        for line in text.splitlines():
            tokens = line.split()
            if len(tokens) == 2:
                values[tokens[0]] = tokens[1]
        if "usage_usec" not in values:
            raise CPUCapacityError(f"cpu.stat lacks usage_usec: {path}")
        try:
            value = int(values["usage_usec"]) / 1_000_000.0
        except ValueError as exc:
            raise CPUCapacityError(f"malformed usage_usec: {path}") from exc
    elif domain.usage_kind == "v1_cpuacct_nanoseconds":
        try:
            value = int(text) / 1_000_000_000.0
        except ValueError as exc:
            raise CPUCapacityError(f"malformed cpuacct.usage: {path}") from exc
    else:  # pragma: no cover - constructors constrain values
        raise CPUCapacityError(f"unsupported domain usage kind: {domain.usage_kind}")
    if value < 0:
        raise CPUCapacityError(f"negative cgroup CPU usage: {path}")
    return value


def capture_cpu_counters(
    binding: CPUBinding,
    *,
    proc_stat_path: str | Path = "/proc/stat",
    monotonic: Callable[[], float] = time.monotonic,
) -> CPUCounterSnapshot:
    current_ids, _source = _affinity_cpu_ids()
    if current_ids != binding.affinity_cpu_ids:
        raise CPUCapacityError("CPU affinity changed during measurement")
    busy, total = _read_proc_stat_ticks(Path(proc_stat_path), binding.affinity_cpu_ids)
    usage = tuple(
        (domain.path, _read_domain_usage_seconds(domain))
        for domain in binding.quota_domains
    )
    return CPUCounterSnapshot(
        monotonic_seconds=float(monotonic()),
        affinity_cpu_ids=binding.affinity_cpu_ids,
        system_busy_ticks=busy,
        system_total_ticks=total,
        quota_usage_seconds=usage,
    )


def cpu_interval_measurement(
    start: CPUCounterSnapshot,
    end: CPUCounterSnapshot,
) -> CPUIntervalMeasurement:
    if start.affinity_cpu_ids != end.affinity_cpu_ids:
        raise CPUCapacityError("CPU affinity changed between counter snapshots")
    elapsed = end.monotonic_seconds - start.monotonic_seconds
    busy_delta = end.system_busy_ticks - start.system_busy_ticks
    total_delta = end.system_total_ticks - start.system_total_ticks
    if elapsed <= 0:
        raise CPUCapacityError("non-positive CPU measurement interval")
    if busy_delta < 0 or total_delta <= 0 or busy_delta > total_delta:
        raise CPUCapacityError("invalid /proc/stat counter delta")
    start_usage = dict(start.quota_usage_seconds)
    end_usage = dict(end.quota_usage_seconds)
    if start_usage.keys() != end_usage.keys():
        raise CPUCapacityError("quota-domain set changed during measurement")
    usage_cores: list[tuple[str, float]] = []
    for path in start_usage:
        delta = end_usage[path] - start_usage[path]
        if delta < 0:
            raise CPUCapacityError(f"negative quota-domain CPU usage delta: {path}")
        usage_cores.append((path, delta / elapsed))
    count = len(start.affinity_cpu_ids)
    system_busy_cores = count * busy_delta / total_delta
    return CPUIntervalMeasurement(
        elapsed_seconds=elapsed,
        affinity_cpu_ids=start.affinity_cpu_ids,
        system_busy_tick_delta=busy_delta,
        system_total_tick_delta=total_delta,
        system_busy_cores=system_busy_cores,
        quota_domain_usage_cores=tuple(usage_cores),
        started_monotonic_seconds=start.monotonic_seconds,
        finished_monotonic_seconds=end.monotonic_seconds,
    )


def sample_cpu_interval(
    binding: CPUBinding,
    *,
    sample_seconds: float = 1.0,
    proc_stat_path: str | Path = "/proc/stat",
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> CPUIntervalMeasurement:
    if sample_seconds <= 0:
        raise CPUCapacityError("sample_seconds must be positive")
    start = capture_cpu_counters(
        binding, proc_stat_path=proc_stat_path, monotonic=monotonic
    )
    sleep(sample_seconds)
    end = capture_cpu_counters(
        binding, proc_stat_path=proc_stat_path, monotonic=monotonic
    )
    return cpu_interval_measurement(start, end)


def _read_process_stat(path: Path) -> tuple[int, int] | None:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    close = text.rfind(")")
    if close < 0:
        return None
    fields = text[close + 2 :].split()
    if len(fields) < 15:
        return None
    try:
        ppid = int(fields[1])
        ticks = sum(int(value) for value in fields[11:15])
    except ValueError:
        return None
    return ppid, ticks


def process_tree_cpu_seconds(
    root_pid: int,
    *,
    proc_root: str | Path = "/proc",
    ticks_per_second: float | None = None,
) -> float:
    root = Path(proc_root)
    try:
        entries = list(root.iterdir())
    except OSError:
        return 0.0
    parent_map: dict[int, int] = {}
    cpu_ticks: dict[int, int] = {}
    for entry in entries:
        if not entry.name.isdigit():
            continue
        parsed = _read_process_stat(entry / "stat")
        if parsed is None:
            continue
        pid = int(entry.name)
        parent_map[pid], cpu_ticks[pid] = parsed
    descendants = {int(root_pid)}
    changed = True
    while changed:
        changed = False
        for pid, ppid in parent_map.items():
            if ppid in descendants and pid not in descendants:
                descendants.add(pid)
                changed = True
    if ticks_per_second is None:
        try:
            ticks_per_second = float(os.sysconf("SC_CLK_TCK"))
        except (OSError, ValueError, TypeError):
            ticks_per_second = 100.0
    if ticks_per_second <= 0:
        raise CPUCapacityError("clock ticks per second must be positive")
    return sum(cpu_ticks.get(pid, 0) for pid in descendants) / ticks_per_second


def reserve_worker_cpu_cores(
    measured_cpu_cores_per_worker: float,
    *,
    safety_factor: float,
    minimum_cpu_cores_per_worker: float,
) -> float:
    if measured_cpu_cores_per_worker <= 0 or not math.isfinite(
        measured_cpu_cores_per_worker
    ):
        raise CPUCapacityError("measured worker CPU demand must be finite and positive")
    if safety_factor < 1 or not math.isfinite(safety_factor):
        raise CPUCapacityError("worker CPU safety factor must be finite and >= 1")
    if minimum_cpu_cores_per_worker <= 0 or not math.isfinite(
        minimum_cpu_cores_per_worker
    ):
        raise CPUCapacityError("minimum worker CPU demand must be finite and positive")
    return max(
        minimum_cpu_cores_per_worker,
        measured_cpu_cores_per_worker * safety_factor,
    )


def derive_worker_cpu_capacity(
    binding: CPUBinding,
    interval: CPUIntervalMeasurement,
    *,
    measured_probe_cpu_cores: float,
    reserved_cpu_cores_per_worker: float,
    cpu_fraction: float,
) -> CPUWorkerCapacity:
    if not 0.05 <= cpu_fraction <= 1.0:
        raise CPUCapacityError("cpu_fraction must be in [0.05, 1.0]")
    if measured_probe_cpu_cores <= 0 or not math.isfinite(measured_probe_cpu_cores):
        raise CPUCapacityError("probe CPU demand must be finite and positive")
    if reserved_cpu_cores_per_worker <= 0 or not math.isfinite(
        reserved_cpu_cores_per_worker
    ):
        raise CPUCapacityError("reserved worker CPU demand must be finite and positive")
    if interval.affinity_cpu_ids != binding.affinity_cpu_ids:
        raise CPUCapacityError("CPU interval does not match capacity binding")
    affinity_budget = binding.affinity_capacity_cores * cpu_fraction
    affinity_external = max(0.0, interval.system_busy_cores - measured_probe_cpu_cores)
    affinity_worker_budget = max(0.0, affinity_budget - affinity_external)
    domain_usage = interval.quota_usage_map()
    domain_budgets: list[tuple[str, float]] = []
    domain_external: list[tuple[str, float]] = []
    available = [affinity_worker_budget]
    for domain in binding.quota_domains:
        if domain.path not in domain_usage:
            raise CPUCapacityError(f"missing measured quota domain: {domain.path}")
        budget = domain.quota_cores * cpu_fraction
        external = max(0.0, domain_usage[domain.path] - measured_probe_cpu_cores)
        remaining = max(0.0, budget - external)
        domain_budgets.append((domain.path, budget))
        domain_external.append((domain.path, external))
        available.append(remaining)
    worker_budget = min(available)
    limit = math.floor(worker_budget / reserved_cpu_cores_per_worker)
    return CPUWorkerCapacity(
        reserved_cpu_cores_per_worker=reserved_cpu_cores_per_worker,
        measured_cpu_cores_per_worker=measured_probe_cpu_cores,
        affinity_budget_cores=affinity_budget,
        affinity_external_busy_cores=affinity_external,
        affinity_worker_budget_cores=affinity_worker_budget,
        quota_domain_budgets=tuple(domain_budgets),
        quota_domain_external_busy_cores=tuple(domain_external),
        worker_cpu_budget_cores=worker_budget,
        cpu_worker_limit=limit,
    )


def candidate_cpu_capacity_ok(
    binding: CPUBinding,
    interval: CPUIntervalMeasurement,
    *,
    measured_candidate_cpu_cores: float,
    cpu_fraction: float,
    safety_factor: float = 1.0,
) -> tuple[bool, dict[str, Any]]:
    if measured_candidate_cpu_cores <= 0 or not math.isfinite(
        measured_candidate_cpu_cores
    ):
        raise CPUCapacityError("candidate CPU demand must be finite and positive")
    if safety_factor < 1 or not math.isfinite(safety_factor):
        raise CPUCapacityError("candidate CPU safety factor must be finite and >= 1")
    demand = measured_candidate_cpu_cores * safety_factor
    affinity_budget = binding.affinity_capacity_cores * cpu_fraction
    affinity_external = max(0.0, interval.system_busy_cores - measured_candidate_cpu_cores)
    affinity_total = affinity_external + demand
    affinity_ok = affinity_total <= affinity_budget
    domain_usage = interval.quota_usage_map()
    domains: list[dict[str, Any]] = []
    domain_ok = True
    for domain in binding.quota_domains:
        if domain.path not in domain_usage:
            raise CPUCapacityError(f"missing measured quota domain: {domain.path}")
        budget = domain.quota_cores * cpu_fraction
        external = max(0.0, domain_usage[domain.path] - measured_candidate_cpu_cores)
        projected = external + demand
        ok = projected <= budget
        domain_ok = domain_ok and ok
        domains.append(
            {
                "path": domain.path,
                "budget_cores": budget,
                "measured_external_busy_cores": external,
                "projected_total_busy_cores": projected,
                "ok": ok,
            }
        )
    return affinity_ok and domain_ok, {
        "measured_candidate_cpu_cores": measured_candidate_cpu_cores,
        "candidate_cpu_safety_factor": safety_factor,
        "reserved_candidate_cpu_cores": demand,
        "affinity_budget_cores": affinity_budget,
        "affinity_external_busy_cores": affinity_external,
        "affinity_projected_total_busy_cores": affinity_total,
        "affinity_ok": affinity_ok,
        "quota_domains": domains,
        "ok": affinity_ok and domain_ok,
    }
