from __future__ import annotations

import os
from pathlib import Path

import pytest

from drpo import runtime_cpu_capacity as cpu


def write_v2(root: Path) -> tuple[Path, Path]:
    root.mkdir()
    (root / "cgroup.controllers").write_text("cpu memory\n", encoding="utf-8")
    leaf = root / "team" / "job"
    leaf.mkdir(parents=True)
    (root / "cpu.max").write_text("max 100000\n", encoding="utf-8")
    (root / "cpu.stat").write_text("usage_usec 1000000\n", encoding="utf-8")
    (root / "team" / "cpu.max").write_text("800000 100000\n", encoding="utf-8")
    (root / "team" / "cpu.stat").write_text("usage_usec 700000\n", encoding="utf-8")
    (leaf / "cpu.max").write_text("400000 100000\n", encoding="utf-8")
    (leaf / "cpu.stat").write_text("usage_usec 500000\n", encoding="utf-8")
    membership = root.parent / "self.cgroup"
    membership.write_text("0::/team/job\n", encoding="utf-8")
    return leaf, membership


def test_v2_binding_discovers_finite_current_and_ancestor_domains(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "cgroup"
    leaf, membership = write_v2(root)
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: {0, 1, 2, 3, 4, 5})
    binding = cpu.discover_cpu_binding(
        cgroup_root=root, proc_self_cgroup_path=membership
    )
    assert binding.current_cgroup_path == str(leaf)
    assert binding.affinity_capacity_cores == 6
    assert [domain.quota_cores for domain in binding.quota_domains] == [4.0, 8.0]
    assert binding.effective_cpu_capacity_cores == 4.0


def test_unresolved_membership_path_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "cg"
    root.mkdir()
    (root / "cgroup.controllers").write_text("cpu\n", encoding="utf-8")
    (root / "cpu.max").write_text("max 100000\n", encoding="utf-8")
    (root / "cpu.stat").write_text("usage_usec 1\n", encoding="utf-8")
    membership = tmp_path / "self"
    membership.write_text("0::/missing/path\n", encoding="utf-8")
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: {0, 1})
    with pytest.raises(cpu.CPUCapacityError, match="cannot be resolved"):
        cpu.discover_cpu_binding(cgroup_root=root, proc_self_cgroup_path=membership)


def test_namespaced_root_is_accepted_only_when_it_lists_current_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "cg"
    root.mkdir()
    (root / "cgroup.controllers").write_text("cpu\n", encoding="utf-8")
    (root / "cpu.max").write_text("200000 100000\n", encoding="utf-8")
    (root / "cpu.stat").write_text("usage_usec 1\n", encoding="utf-8")
    (root / "cgroup.procs").write_text(f"{os.getpid()}\n", encoding="utf-8")
    membership = tmp_path / "self"
    membership.write_text("0::/host/path/not/exposed\n", encoding="utf-8")
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: {0, 1, 2, 3})
    binding = cpu.discover_cpu_binding(
        cgroup_root=root, proc_self_cgroup_path=membership
    )
    assert binding.current_cgroup_path == str(root)
    assert binding.effective_cpu_capacity_cores == 2.0


def test_v2_fractional_quota_is_not_rounded_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "cg"
    root.mkdir()
    (root / "cgroup.controllers").write_text("cpu\n", encoding="utf-8")
    (root / "cpu.max").write_text("150000 100000\n", encoding="utf-8")
    (root / "cpu.stat").write_text("usage_usec 1\n", encoding="utf-8")
    membership = tmp_path / "self"
    membership.write_text("0::/\n", encoding="utf-8")
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: {0, 1, 2, 3})
    binding = cpu.discover_cpu_binding(
        cgroup_root=root, proc_self_cgroup_path=membership
    )
    assert binding.effective_cpu_capacity_cores == 1.5


def test_malformed_quota_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "cg"
    root.mkdir()
    (root / "cgroup.controllers").write_text("cpu\n", encoding="utf-8")
    (root / "cpu.max").write_text("100000 0\n", encoding="utf-8")
    (root / "cpu.stat").write_text("usage_usec 1\n", encoding="utf-8")
    membership = tmp_path / "self"
    membership.write_text("0::/\n", encoding="utf-8")
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: {0})
    with pytest.raises(cpu.CPUCapacityError, match="period"):
        cpu.discover_cpu_binding(cgroup_root=root, proc_self_cgroup_path=membership)


def test_proc_stat_excludes_iowait_and_guest(tmp_path: Path) -> None:
    stat = tmp_path / "stat"
    stat.write_text(
        "cpu 0 0 0 0 0 0 0 0 0 0\n"
        "cpu0 10 2 3 20 30 4 5 6 999 999\n"
        "cpu1 1 1 1 10 10 1 1 1 999 999\n",
        encoding="utf-8",
    )
    busy, total = cpu._read_proc_stat_ticks(stat, [0, 1])  # noqa: SLF001
    assert busy == (10 + 2 + 3 + 4 + 5 + 6) + (1 + 1 + 1 + 1 + 1 + 1)
    assert total == (10 + 2 + 3 + 20 + 30 + 4 + 5 + 6) + (
        1 + 1 + 1 + 10 + 10 + 1 + 1 + 1
    )


def test_interval_and_capacity_apply_affinity_and_each_quota_domain() -> None:
    binding = cpu.CPUBinding(
        affinity_cpu_ids=tuple(range(8)),
        affinity_source="test",
        cgroup_version="v2",
        current_cgroup_path="/cg/leaf",
        quota_domains=(
            cpu.CPUQuotaDomain("/cg/leaf", 6.0, "/x", "v2_usage_usec", "v2"),
            cpu.CPUQuotaDomain("/cg", 10.0, "/y", "v2_usage_usec", "v2"),
        ),
    )
    interval = cpu.CPUIntervalMeasurement(
        elapsed_seconds=1.0,
        affinity_cpu_ids=binding.affinity_cpu_ids,
        system_busy_tick_delta=50,
        system_total_tick_delta=100,
        system_busy_cores=4.0,
        quota_domain_usage_cores=(("/cg/leaf", 3.0), ("/cg", 5.0)),
        started_monotonic_seconds=0.0,
        finished_monotonic_seconds=1.0,
    )
    result = cpu.derive_worker_cpu_capacity(
        binding,
        interval,
        measured_probe_cpu_cores=1.0,
        reserved_cpu_cores_per_worker=1.25,
        cpu_fraction=0.8,
    )
    assert result.cpu_worker_limit == 2
    assert result.worker_cpu_budget_cores == pytest.approx(2.8)


def test_tight_quota_does_not_subtract_unrelated_host_load_twice() -> None:
    binding = cpu.CPUBinding(
        affinity_cpu_ids=tuple(range(64)),
        affinity_source="test",
        cgroup_version="v2",
        current_cgroup_path="/leaf",
        quota_domains=(
            cpu.CPUQuotaDomain("/leaf", 4.0, "/x", "v2_usage_usec", "v2"),
        ),
    )
    interval = cpu.CPUIntervalMeasurement(
        elapsed_seconds=1.0,
        affinity_cpu_ids=binding.affinity_cpu_ids,
        system_busy_tick_delta=800,
        system_total_tick_delta=1000,
        system_busy_cores=51.2,
        quota_domain_usage_cores=(("/leaf", 1.0),),
        started_monotonic_seconds=0.0,
        finished_monotonic_seconds=1.0,
    )
    result = cpu.derive_worker_cpu_capacity(
        binding,
        interval,
        measured_probe_cpu_cores=0.5,
        reserved_cpu_cores_per_worker=1.0,
        cpu_fraction=0.85,
    )
    assert result.worker_cpu_budget_cores == pytest.approx(2.9)
    assert result.cpu_worker_limit == 2


def test_candidate_validation_rejects_sibling_quota_pressure() -> None:
    binding = cpu.CPUBinding(
        affinity_cpu_ids=tuple(range(32)),
        affinity_source="test",
        cgroup_version="v2",
        current_cgroup_path="/parent/leaf",
        quota_domains=(
            cpu.CPUQuotaDomain("/parent/leaf", 20.0, "/x", "v2_usage_usec", "v2"),
            cpu.CPUQuotaDomain("/parent", 12.0, "/y", "v2_usage_usec", "v2"),
        ),
    )
    interval = cpu.CPUIntervalMeasurement(
        elapsed_seconds=1.0,
        affinity_cpu_ids=binding.affinity_cpu_ids,
        system_busy_tick_delta=50,
        system_total_tick_delta=100,
        system_busy_cores=16.0,
        quota_domain_usage_cores=(("/parent/leaf", 4.0), ("/parent", 11.0)),
        started_monotonic_seconds=0.0,
        finished_monotonic_seconds=1.0,
    )
    ok, details = cpu.candidate_cpu_capacity_ok(
        binding,
        interval,
        measured_candidate_cpu_cores=4.0,
        cpu_fraction=0.85,
        safety_factor=1.1,
    )
    assert ok is False
    assert details["quota_domains"][1]["ok"] is False


def test_zero_domain_usage_delta_is_valid_idle_evidence() -> None:
    start = cpu.CPUCounterSnapshot(0.0, (0,), 10, 100, (("/cg", 1.0),))
    end = cpu.CPUCounterSnapshot(1.0, (0,), 20, 200, (("/cg", 1.0),))
    result = cpu.cpu_interval_measurement(start, end)
    assert result.quota_usage_map()["/cg"] == 0.0


def write_proc_stat(
    path: Path, ppid: int, ticks: tuple[int, int, int, int]
) -> None:
    fields = ["S", str(ppid)] + ["0"] * 9 + [str(value) for value in ticks]
    path.parent.mkdir(parents=True)
    path.write_text("1 (worker name) " + " ".join(fields) + "\n", encoding="utf-8")


def test_process_tree_cpu_includes_descendants_and_waited_children(
    tmp_path: Path,
) -> None:
    proc = tmp_path / "proc"
    proc.mkdir()
    write_proc_stat(proc / "10" / "stat", 1, (100, 50, 25, 25))
    write_proc_stat(proc / "11" / "stat", 10, (40, 10, 0, 0))
    assert cpu.process_tree_cpu_seconds(10, proc_root=proc, ticks_per_second=100) == 2.5
