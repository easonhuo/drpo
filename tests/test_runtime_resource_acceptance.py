from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

from drpo import runtime_resource_acceptance as acceptance
from drpo import runtime_resource_acceptance_commands as commands
from drpo import runtime_resource_acceptance_process as process

SCRIPT = Path("scripts/run_runtime_resource_acceptance.py")
SPEC = importlib.util.spec_from_file_location("runtime_resource_acceptance_cli", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
cli = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cli)


def _profile(tmp_path: Path, repo: Path) -> Path:
    external = tmp_path / "external"
    external.mkdir()
    model = external / "model"
    model.mkdir()
    values: dict[str, str] = {}
    for name in (
        "contract",
        "run_spec",
        "grid",
        "bank",
        "val",
        "global_calibration",
        "base_config",
        "sweep_config",
    ):
        path = external / f"{name}.json"
        path.write_text("{}\n", encoding="utf-8")
        values[name] = str(path)
    payload = {
        "schema_version": 1,
        "output_parent": str(tmp_path / "outputs"),
        "expected_harness_commit": None,
        "gpu_selection_commit": "a" * 40,
        "gpu_selection_ref": "dev/gpu-shadow",
        "continue_after_failure": False,
        "conflict_process_patterns": ["run_e7_"],
        "resource_pools": {
            "e7_cpu_pool": "0-1",
            "e8_cpu_pool": "2-3",
            "e8_gpu_ids": ["0", "1"],
        },
        "e7": {
            "enabled": True,
            "contract": values["contract"],
            "run_spec": values["run_spec"],
            "grid": values["grid"],
            "fallback_workers": 2,
            "probe_steps": 5,
            "probe_seed": 9001,
            "probe_seconds": 2.0,
            "plan_timeout_seconds": 20.0,
            "throughput_retention_fraction": 0.97,
            "cpu_fraction": 0.85,
            "memory_headroom_fraction": 0.15,
            "per_worker_safety_factor": 1.2,
            "per_worker_cpu_safety_factor": 1.25,
            "minimum_cpu_cores_per_worker": 1.0,
            "max_workers": 4,
            "max_growth_factor": 3.0,
            "minimum_branches_for_probe": 2,
            "revalidation_samples": 3,
            "revalidation_sample_seconds": 0.01,
            "liveness_steps": 2,
            "liveness_seed": 9002,
            "liveness_timeout_seconds": 10.0,
        },
        "e8": {
            "enabled": True,
            "model_path": str(model),
            "bank": values["bank"],
            "val": values["val"],
            "test": "/dev/null",
            "global_calibration": values["global_calibration"],
            "base_config": values["base_config"],
            "sweep_config": values["sweep_config"],
            "required_free_gpu_memory_gib": 8.0,
            "required_host_memory_gib_per_worker": 4.0,
            "gpu_memory_headroom_fraction": 0.12,
            "host_memory_headroom_fraction": 0.15,
            "per_worker_host_memory_safety_factor": 1.25,
            "per_worker_vram_safety_factor": 1.25,
            "cpu_fraction": 0.85,
            "per_worker_cpu_safety_factor": 1.5,
            "minimum_cpu_cores_per_worker": 1.0,
            "maximum_gpu_utilization_percent": 20.0,
            "max_devices": 2,
            "max_slots_per_gpu": 2,
            "single_probe_seconds": 2.0,
            "validation_probe_seconds": 2.0,
            "probe_budget_seconds": 4.0,
            "probe_free_floor_gib": 4.0,
            "selection_timeout_seconds": 10.0,
            "thread_candidates": [None, 4, 8, 16],
        },
        "concurrent": {
            "enabled": True,
            "timeout_seconds": 20.0,
            "sample_interval_seconds": 0.05,
        },
    }
    target = tmp_path / "profile.json"
    target.write_text(json.dumps(payload), encoding="utf-8")
    return target


def test_profile_validation_is_closed_and_normalizes_pools(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    target = _profile(tmp_path, repo)
    profile = acceptance.load_profile(target, repo_root=repo)
    assert profile["resource_pools"]["e7_cpu_ids"] == [0, 1]
    assert profile["resource_pools"]["e8_cpu_ids"] == [2, 3]
    assert profile["resource_pools"]["e8_gpu_ids"] == ["0", "1"]
    raw = json.loads(target.read_text())
    raw["unknown"] = True
    target.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(acceptance.AcceptanceError, match="unknown fields"):
        acceptance.load_profile(target, repo_root=repo)


def test_profile_rejects_placeholders_and_overlapping_pools(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    target = _profile(tmp_path, repo)
    raw = json.loads(target.read_text())
    raw["e7"]["contract"] = "REPLACE_WITH_ABSOLUTE_PATH/contract.json"
    target.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(acceptance.AcceptanceError, match="absolute path"):
        acceptance.load_profile(target, repo_root=repo)
    target = _profile(tmp_path, repo)
    raw = json.loads(target.read_text())
    raw["resource_pools"]["e8_cpu_pool"] = "1-3"
    target.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(acceptance.AcceptanceError, match="overlap"):
        acceptance.load_profile(target, repo_root=repo)


def test_gpu_selection_command_never_contains_test_split(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    profile = acceptance.load_profile(_profile(tmp_path, repo), repo_root=repo)
    command = commands.gpu_selection_command(
        repo,
        profile,
        work_dir=tmp_path / "gpu_work",
        gpu_ids=("0",),
        max_devices=1,
        max_slots=1,
    )
    assert "--selection-only" in command
    assert "--test" not in command
    assert "/dev/null" not in command


def test_overall_status_does_not_majority_vote() -> None:
    now = acceptance.utc_now()

    def result(status: str) -> acceptance.StageResult:
        return acceptance.StageResult("x", status, now, now, {})

    assert acceptance.overall_status([result("PASS"), result("FAIL")]) == "FAIL"
    assert acceptance.overall_status([result("PASS"), result("BLOCKED")]) == "BLOCKED"
    assert (
        acceptance.overall_status([result("PASS"), result("INCONCLUSIVE")])
        == "INCONCLUSIVE"
    )


def test_owned_timeout_cleans_only_its_process_group(tmp_path: Path) -> None:
    result = process.run_command(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        cwd=tmp_path,
        environment=os.environ.copy(),
        timeout_seconds=0.1,
        log_path=tmp_path / "timeout.log",
        samples_path=tmp_path / "samples.jsonl",
        sample_interval_seconds=0.02,
        command_ledger=tmp_path / "commands.jsonl",
    )
    assert result.timed_out is True
    assert result.controller_intervened is True
    assert result.process_group_alive_after_cleanup is False


def test_package_is_text_only_and_rejects_model_payload(tmp_path: Path) -> None:
    root = tmp_path / "evidence"
    root.mkdir()
    (root / "report.json").write_text("{}\n", encoding="utf-8")
    package = acceptance.package_acceptance(root)
    assert Path(package["path"]).is_file()
    forbidden = root / "checkpoint.pt"
    forbidden.write_bytes(b"weights")
    with pytest.raises(acceptance.AcceptanceError, match="file type"):
        acceptance.package_acceptance(root)


def test_affinity_audit_rejects_cross_pool_process(tmp_path: Path) -> None:
    from drpo.runtime_resource_acceptance_gpu_stages import _affinity_violations

    samples = tmp_path / "samples.jsonl"
    samples.write_text(
        json.dumps(
            {
                "commands": {
                    "e7": {"processes": [{"pid": 1, "affinity_cpu_ids": [0, 1]}]},
                    "e8": {"processes": [{"pid": 2, "affinity_cpu_ids": [1, 2]}]},
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )
    violations = _affinity_violations(samples, {"e7": {0, 1}, "e8": {2, 3}})
    assert len(violations) == 1
    assert violations[0]["command"] == "e8"
