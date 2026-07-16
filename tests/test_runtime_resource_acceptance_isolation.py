from __future__ import annotations

from pathlib import Path

from drpo import runtime_resource_acceptance_isolation as isolation


def _partition(tmp_path: Path, cpus: str = "0-3") -> Path:
    mount = tmp_path / "cgroup"
    partition = mount / "drpo"
    session = partition / "session"
    session.mkdir(parents=True)
    (partition / "cpuset.cpus.partition").write_text("root\n", encoding="utf-8")
    (partition / "cpuset.cpus.effective").write_text(f"{cpus}\n", encoding="utf-8")
    (partition / "cpuset.cpus.exclusive.effective").write_text(
        f"{cpus}\n",
        encoding="utf-8",
    )
    return mount


def test_permanent_external_process_outside_partition_is_not_conflict(
    tmp_path: Path,
) -> None:
    mount = _partition(tmp_path)
    inventory = [
        {
            "pid": 200,
            "command": "python ResearchBench/rb_generate.py",
            "affinity_cpu_ids": [4, 5],
            "cgroup_v2_path": "/external/researchbench",
        }
    ]

    audit = isolation.audit_resource_isolation(
        inventory=inventory,
        conflict_patterns=["ResearchBench"],
        reserved_cpu_ids=[0, 1, 2, 3],
        excluded_pids={100},
        current_cgroup_path="/drpo/session",
        cgroup_mount=mount,
    )

    assert audit["exclusive_partition_proven"] is True
    assert audit["ready"] is True
    assert audit["conflicts"] == []
    assert [row["pid"] for row in audit["isolated_external_matches"]] == [200]


def test_process_inside_acceptance_partition_blocks(tmp_path: Path) -> None:
    mount = _partition(tmp_path)
    inventory = [
        {
            "pid": 201,
            "command": "python unrelated_worker.py",
            "affinity_cpu_ids": [0],
            "cgroup_v2_path": "/drpo/other",
        }
    ]

    audit = isolation.audit_resource_isolation(
        inventory=inventory,
        conflict_patterns=["ResearchBench"],
        reserved_cpu_ids=[0, 1, 2, 3],
        excluded_pids={100},
        current_cgroup_path="/drpo/session",
        cgroup_mount=mount,
    )

    assert audit["exclusive_partition_proven"] is True
    assert audit["ready"] is False
    assert audit["partition_contaminants"][0]["pid"] == 201
    assert audit["conflicts"][0]["pid"] == 201


def test_outside_affinity_overlap_blocks_even_with_partition(tmp_path: Path) -> None:
    mount = _partition(tmp_path)
    inventory = [
        {
            "pid": 202,
            "command": "python -m joblib.externals.loky.backend.popen_loky_posix",
            "affinity_cpu_ids": [2, 4],
            "cgroup_v2_path": "/external/aide",
        }
    ]

    audit = isolation.audit_resource_isolation(
        inventory=inventory,
        conflict_patterns=["joblib.externals.loky"],
        reserved_cpu_ids=[0, 1, 2, 3],
        excluded_pids={100},
        current_cgroup_path="/drpo/session",
        cgroup_mount=mount,
    )

    assert audit["ready"] is False
    assert audit["conflicts"][0]["reserved_cpu_overlap"] == [2]


def test_partition_must_exclusively_cover_reserved_pools(tmp_path: Path) -> None:
    mount = _partition(tmp_path, cpus="0-1")

    audit = isolation.audit_resource_isolation(
        inventory=[],
        conflict_patterns=[],
        reserved_cpu_ids=[0, 1, 2, 3],
        excluded_pids=set(),
        current_cgroup_path="/drpo/session",
        cgroup_mount=mount,
    )

    assert audit["exclusive_partition_proven"] is False
    assert audit["missing_exclusive_cpu_ids"] == [2, 3]
    assert audit["partition_error"]


def test_without_partition_matching_process_remains_conflict(tmp_path: Path) -> None:
    mount = tmp_path / "cgroup"
    (mount / "shared").mkdir(parents=True)
    inventory = [
        {
            "pid": 203,
            "command": "python /root/llm4mle/collector_v2.py",
            "affinity_cpu_ids": [0, 1],
            "cgroup_v2_path": "/shared",
        }
    ]

    audit = isolation.audit_resource_isolation(
        inventory=inventory,
        conflict_patterns=["collector_v2.py"],
        reserved_cpu_ids=[0, 1],
        excluded_pids=set(),
        current_cgroup_path="/shared",
        cgroup_mount=mount,
    )

    assert audit["exclusive_partition_proven"] is False
    assert audit["ready"] is False
    assert audit["conflicts"][0]["pid"] == 203
