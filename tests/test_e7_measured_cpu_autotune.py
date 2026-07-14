from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from drpo import e7_ppo_w0_runtime_autotune as autotune
from drpo import runtime_cpu_capacity as cpu
from drpo.runtime_resource_autotune import GIB, MachineSnapshot, RuntimeResourceError


def machine(*, cpus: int = 8, available_gib: int = 32, load: float = 999.0) -> MachineSnapshot:
    total = 64 * GIB
    available = available_gib * GIB
    return MachineSnapshot(
        logical_cpu_count=cpus,
        memory_total_bytes=total,
        memory_available_bytes=available,
        effective_memory_limit_bytes=total,
        effective_memory_current_bytes=total - available,
        effective_memory_available_bytes=available,
        swap_total_bytes=0,
        swap_free_bytes=0,
        cgroup_version="v2",
        load_average_1m=load,
        gpus=(),
    )


def binding() -> cpu.CPUBinding:
    return cpu.CPUBinding(
        affinity_cpu_ids=tuple(range(8)),
        affinity_source="sched_getaffinity",
        cgroup_version=None,
        current_cgroup_path=None,
        quota_domains=(),
    )


def interval(system_busy: float) -> cpu.CPUIntervalMeasurement:
    return cpu.CPUIntervalMeasurement(
        elapsed_seconds=1.0,
        affinity_cpu_ids=tuple(range(8)),
        system_busy_tick_delta=int(system_busy * 100),
        system_total_tick_delta=800,
        system_busy_cores=system_busy,
        quota_domain_usage_cores=(),
        started_monotonic_seconds=0.0,
        finished_monotonic_seconds=1.0,
    )


def selection_document(tmp_path: Path) -> dict:
    document = {
        "schema_version": autotune.SELECTION_SCHEMA_VERSION,
        "selector_policy_version": autotune.SELECTOR_POLICY_VERSION,
        "adapter_id": autotune.ADAPTER_ID,
        "mode": "auto",
        "source": {"commit": "abc", "branch": "dev", "dirty": False},
        "resource_fingerprint": {"x": 1},
        "resource_fingerprint_sha256": autotune.canonical_json_sha256({"x": 1}),
        "selector_implementation": {"selector": "sha"},
        "cpu_binding": binding().as_dict(),
        "selection": {
            "selected_workers": 2,
            "per_worker_reserved_cpu_cores": 1.0,
            "per_worker_reserved_bytes": 1 * GIB,
        },
        "scientific_matrix_changed": False,
    }
    document["selection_digest"] = autotune.canonical_json_sha256(
        autotune._selection_digest_payload(document)  # noqa: SLF001
    )
    path = tmp_path / "work" / "RUNTIME_SELECTION.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(document), encoding="utf-8")
    return document


def revalidation_kwargs(tmp_path: Path) -> dict:
    return {
        "machine": machine(),
        "repo_root": tmp_path,
        "contract_path": tmp_path / "contract",
        "run_spec_path": tmp_path / "run_spec",
        "grid_path": tmp_path / "grid",
        "work_dir": tmp_path / "work",
        "fallback_workers": 2,
        "probe_steps": 10,
        "probe_seed": 99,
        "probe_seconds": 1.0,
        "throughput_retention_fraction": 0.97,
        "cpu_fraction": 0.85,
        "memory_headroom_fraction": 0.15,
        "per_worker_safety_factor": 1.2,
        "per_worker_cpu_safety_factor": 1.25,
        "minimum_cpu_cores_per_worker": 1.0,
        "max_workers": None,
        "max_growth_factor": 3.0,
        "minimum_branches_for_probe": 8,
        "revalidation_samples": 3,
        "revalidation_sample_seconds": 0.01,
        "cgroup_root": tmp_path / "cgroup",
        "proc_self_cgroup_path": tmp_path / "self.cgroup",
        "proc_stat_path": tmp_path / "stat",
        "proc_root": tmp_path / "proc",
    }


def patch_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(autotune, "resource_fingerprint", lambda **_kwargs: {"x": 1})
    monkeypatch.setattr(
        autotune, "_selector_implementation_identity", lambda _root: {"selector": "sha"}
    )
    monkeypatch.setattr(
        autotune, "git_state", lambda _root: {"commit": "abc", "branch": "dev", "dirty": False}
    )
    monkeypatch.setattr(cpu, "discover_cpu_binding", lambda **_kwargs: binding())
    monkeypatch.setattr(autotune, "conflicting_workdir_processes", lambda *_args, **_kwargs: [])


def test_plan_refuses_to_replace_existing_selection(tmp_path: Path) -> None:
    path = tmp_path / "work" / "RUNTIME_SELECTION.json"
    path.parent.mkdir(parents=True)
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeResourceError, match="already exists"):
        autotune.select_runtime(
            **{key: value for key, value in revalidation_kwargs(tmp_path).items() if key != "proc_root"}
        )


def test_run_consumes_frozen_selection_without_probe_or_benchmark(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = selection_document(tmp_path)
    patch_identity(monkeypatch)
    values = iter([interval(1.0), interval(2.0), interval(1.5)])
    monkeypatch.setattr(cpu, "sample_cpu_interval", lambda *_args, **_kwargs: next(values))
    monkeypatch.setattr(
        autotune,
        "build_probe_command",
        lambda **_kwargs: pytest.fail("run must not build a probe"),
    )
    monkeypatch.setattr(
        autotune,
        "benchmark_concurrency",
        lambda **_kwargs: pytest.fail("run must not benchmark concurrency"),
    )

    result = autotune.revalidate_runtime(**revalidation_kwargs(tmp_path))

    assert result["selection"]["selected_workers"] == 2
    record = json.loads(Path(result["revalidation"]["path"]).read_text())
    assert record["decision"] == "ALLOW"
    assert record["cpu_revalidation"]["conservative_system_busy_cores"] == 2.0
    assert json.loads((tmp_path / "work" / "RUNTIME_SELECTION.json").read_text()) == original


def test_high_load_average_does_not_change_frozen_worker_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selection_document(tmp_path)
    patch_identity(monkeypatch)
    monkeypatch.setattr(cpu, "sample_cpu_interval", lambda *_args, **_kwargs: interval(1.0))
    result = autotune.revalidate_runtime(**revalidation_kwargs(tmp_path))
    assert result["selection"]["selected_workers"] == 2
    assert result["machine_snapshot"]["load_average_1m"] == 999.0


def test_true_cpu_pressure_blocks_without_downshift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selection_document(tmp_path)
    patch_identity(monkeypatch)
    monkeypatch.setattr(cpu, "sample_cpu_interval", lambda *_args, **_kwargs: interval(6.5))

    with pytest.raises(RuntimeResourceError, match="cpu_capacity_changed"):
        autotune.revalidate_runtime(**revalidation_kwargs(tmp_path))

    records = list(
        (tmp_path / "work" / "_runtime_resource_attempts").glob(
            "*/RUNTIME_REVALIDATION.json"
        )
    )
    assert len(records) == 1
    record = json.loads(records[0].read_text())
    assert record["decision"] == "BLOCK"
    assert record["selected_workers"] == 2
    assert "cpu_capacity_changed" in record["failures"]


def test_old_selection_schema_is_invalidated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    document = selection_document(tmp_path)
    document["schema_version"] = 1
    (tmp_path / "work" / "RUNTIME_SELECTION.json").write_text(
        json.dumps(document), encoding="utf-8"
    )
    patch_identity(monkeypatch)
    monkeypatch.setattr(cpu, "sample_cpu_interval", lambda *_args, **_kwargs: interval(1.0))
    with pytest.raises(RuntimeResourceError, match="selection_schema_version_mismatch"):
        autotune.revalidate_runtime(**revalidation_kwargs(tmp_path))


def test_fast_but_resource_invalid_candidate_is_not_selected() -> None:
    selected, rule = autotune.select_from_throughput(
        [
            {"concurrency": 8, "valid": True, "aggregate_updates_per_second": 80.0},
            {"concurrency": 16, "valid": False, "aggregate_updates_per_second": 200.0},
        ],
        retention_fraction=0.97,
    )
    assert selected == 8
    assert rule["peak_aggregate_updates_per_second"] == 80.0


def test_selection_digest_changes_with_selected_workers() -> None:
    first = selection_document(Path("/tmp") / "digest-a")
    second = copy.deepcopy(first)
    second["selection"]["selected_workers"] = 3
    assert autotune.canonical_json_sha256(
        autotune._selection_digest_payload(first)  # noqa: SLF001
    ) != autotune.canonical_json_sha256(
        autotune._selection_digest_payload(second)  # noqa: SLF001
    )
