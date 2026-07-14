from __future__ import annotations

import os
from pathlib import Path

import pytest

from drpo import runtime_cpu_capacity as cpu


def write_v1_domain(directory: Path, *, quota: int, period: int, usage_ns: int) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "cpu.cfs_quota_us").write_text(f"{quota}\n", encoding="utf-8")
    (directory / "cpu.cfs_period_us").write_text(f"{period}\n", encoding="utf-8")
    (directory / "cpuacct.usage").write_text(f"{usage_ns}\n", encoding="utf-8")


def test_v1_cpu_controller_discovers_current_and_ancestor_quotas(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "cgroup"
    controller = root / "cpu"
    write_v1_domain(controller, quota=-1, period=100_000, usage_ns=1)
    write_v1_domain(controller / "team", quota=400_000, period=100_000, usage_ns=2)
    write_v1_domain(
        controller / "team" / "job",
        quota=200_000,
        period=100_000,
        usage_ns=3,
    )
    membership = tmp_path / "self.cgroup"
    membership.write_text("2:cpu,cpuacct:/team/job\n", encoding="utf-8")
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: {0, 1, 2, 3, 4})

    binding = cpu.discover_cpu_binding(
        cgroup_root=root,
        proc_self_cgroup_path=membership,
    )

    assert binding.cgroup_version == "v1"
    assert binding.current_cgroup_path == str(controller / "team" / "job")
    assert [domain.quota_cores for domain in binding.quota_domains] == [2.0, 4.0]
    assert all(domain.usage_kind == "v1_cpuacct_nanoseconds" for domain in binding.quota_domains)
    assert binding.effective_cpu_capacity_cores == 2.0


def test_v1_direct_controller_root_is_supported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "cpu,cpuacct"
    write_v1_domain(root, quota=150_000, period=100_000, usage_ns=4)
    membership = tmp_path / "self.cgroup"
    membership.write_text("3:cpu,cpuacct:/\n", encoding="utf-8")
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: {0, 1, 2, 3})

    binding = cpu.discover_cpu_binding(
        cgroup_root=root,
        proc_self_cgroup_path=membership,
    )

    assert binding.current_cgroup_path == str(root)
    assert binding.effective_cpu_capacity_cores == 1.5


def test_v1_unresolved_membership_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "cgroup"
    controller = root / "cpu"
    write_v1_domain(controller, quota=-1, period=100_000, usage_ns=1)
    membership = tmp_path / "self.cgroup"
    membership.write_text("2:cpu:/missing\n", encoding="utf-8")
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: {0, 1})

    with pytest.raises(cpu.CPUCapacityError, match="cannot be resolved"):
        cpu.discover_cpu_binding(
            cgroup_root=root,
            proc_self_cgroup_path=membership,
        )


def test_v1_usage_interval_uses_nanoseconds() -> None:
    start = cpu.CPUCounterSnapshot(
        monotonic_seconds=10.0,
        affinity_cpu_ids=(0, 1),
        system_busy_ticks=100,
        system_total_ticks=200,
        quota_usage_seconds=(("/cg", 1.0),),
    )
    end = cpu.CPUCounterSnapshot(
        monotonic_seconds=12.0,
        affinity_cpu_ids=(0, 1),
        system_busy_ticks=120,
        system_total_ticks=240,
        quota_usage_seconds=(("/cg", 4.0),),
    )

    interval = cpu.cpu_interval_measurement(start, end)

    assert interval.quota_usage_map()["/cg"] == 1.5
