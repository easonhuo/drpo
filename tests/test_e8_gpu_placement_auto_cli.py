from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import pytest

from drpo import countdown_e8_oracle_offline_v2_taper_resource_probe as resource_probe
from drpo import runtime_gpu_placement_autotune_v2 as placement
from drpo.runtime_resource_autotune import GIB, RuntimeResourceError

SCRIPT = Path("scripts/run_countdown_e8_oracle_offline_v2_taper_auto.py")
SPEC = importlib.util.spec_from_file_location("e8_gpu_placement_auto", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
auto = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(auto)


def required_args(*, include_test: bool = True) -> list[str]:
    values = [
        "--model_path",
        "model",
        "--work_dir",
        "work",
        "--bank",
        "bank.jsonl",
        "--val",
        "val.jsonl",
    ]
    if include_test:
        values.extend(["--test", "test.jsonl"])
    values.extend(
        [
            "--global_calibration",
            "calibration.json",
            "--base_config",
            "base.yaml",
            "--sweep_config",
            "sweep.yaml",
        ]
    )
    return values


def test_legacy_host_memory_option_remains_a_compatible_alias() -> None:
    parser = auto.build_parser()
    legacy = parser.parse_args(
        required_args() + ["--required-host-memory-gib-per-gpu", "7"]
    )
    current = parser.parse_args(
        required_args() + ["--required-host-memory-gib-per-worker", "6"]
    )
    assert legacy.required_host_memory_gib_per_worker == 7
    assert current.required_host_memory_gib_per_worker == 6


def test_gpu_placement_probe_defaults_are_bounded_and_phase_aware() -> None:
    args = auto.build_parser().parse_args(required_args())
    assert args.probe_budget_seconds == 600
    assert args.single_probe_seconds == 240
    assert args.validation_probe_seconds == 300
    assert args.max_slots_per_gpu == 8
    assert args.per_worker_host_memory_safety_factor == 1.25
    assert args.per_worker_vram_safety_factor == 1.25
    assert args.cpu_fraction == 0.85
    assert args.per_worker_cpu_safety_factor == 1.5
    assert args.minimum_cpu_cores_per_worker == 1.0
    assert args.selection_only is False
    assert placement.PROBE_CONTRACT_VERSION == 2
    assert placement.SELECTOR_POLICY_VERSION == 2
    assert "evaluation_peak_completed" in placement.DEFAULT_REQUIRED_PHASES


def test_selection_only_is_explicit_opt_in_and_does_not_require_test() -> None:
    args = auto.build_parser().parse_args(
        required_args(include_test=False) + ["--selection-only"]
    )
    assert args.selection_only is True
    assert args.test is None


def test_selection_only_fingerprint_does_not_read_test_split(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    files: dict[str, Path] = {}
    for name in ("bank", "val", "global_calibration", "base_config", "sweep_config"):
        path = tmp_path / name
        path.write_text(name, encoding="utf-8")
        files[name] = path
    args = argparse.Namespace(
        selection_only=True,
        model_path=str(tmp_path / "model"),
        bank=str(files["bank"]),
        val=str(files["val"]),
        test=str(tmp_path / "must_not_be_read"),
        global_calibration=str(files["global_calibration"]),
        base_config=str(files["base_config"]),
        sweep_config=str(files["sweep_config"]),
    )
    original = auto.legacy_runtime.sha256_file

    def checked(path: str | Path) -> str:
        assert Path(path) != Path(args.test)
        return original(path)

    monkeypatch.setattr(auto.legacy_runtime, "sha256_file", checked)
    fingerprint = auto._workload_fingerprint(args)  # noqa: SLF001
    assert fingerprint["test_split_access"] == "not_accessed_selection_only"
    assert fingerprint["test_sha256"] is None


def test_full_run_fingerprint_requires_test_split() -> None:
    args = argparse.Namespace(selection_only=False, test=None)
    with pytest.raises(RuntimeResourceError, match="requires --test"):
        auto._workload_fingerprint(args)  # noqa: SLF001


def test_selection_only_stops_before_slot_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_run(*_args: object, **_kwargs: object) -> int:
        raise AssertionError("selection-only must not call the scientific slot runtime")

    monkeypatch.setattr(auto.slot_runtime, "run", unexpected_run)
    result = auto._finish_after_selection(  # noqa: SLF001
        argparse.Namespace(selection_only=True),
        argparse.Namespace(),
        work_dir=tmp_path,
    )
    assert result == 0


def test_normal_mode_still_delegates_selected_placement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    def fake_run(runtime_args: argparse.Namespace, *, placement_path: Path) -> int:
        observed["runtime_args"] = runtime_args
        observed["placement_path"] = placement_path
        return 17

    monkeypatch.setattr(auto.slot_runtime, "run", fake_run)
    runtime_args = argparse.Namespace(marker="normal")
    result = auto._finish_after_selection(  # noqa: SLF001
        argparse.Namespace(selection_only=False),
        runtime_args,
        work_dir=tmp_path,
    )
    assert result == 17
    assert observed["runtime_args"] is runtime_args
    assert observed["placement_path"] == tmp_path / "RUNTIME_SELECTION.json"


def test_resource_probe_command_uses_dedicated_probe_not_scientific_worker(
    tmp_path: Path,
) -> None:
    args = argparse.Namespace(
        model_path="model",
        bank="bank.jsonl",
        val="val.jsonl",
        base_config="base.yaml",
        sweep_config="sweep.yaml",
    )
    cell = auto.legacy_runtime.core.Cell("exponential", 0.5, 0)
    command = resource_probe.resource_probe_command(
        args=args,
        cell=cell,
        output_dir=tmp_path / "probe",
        calibration=tmp_path / "calibration.json",
    )
    assert Path(command[1]).name == (
        "countdown_e8_oracle_offline_v2_taper_resource_probe.py"
    )
    assert "worker" not in command
    assert "--output_dir" in command
    assert "--test" not in command


def test_phase_peak_runner_uses_worker_reported_cuda_peak(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = tmp_path / "worker_00.phases.json"
    state.write_text(
        json.dumps({"maximum_reported_worker_vram_bytes": 20 * GIB}),
        encoding="utf-8",
    )
    result = placement.GPUConcurrencyProbeResult(
        concurrency=1,
        device_id="0",
        started_utc="a",
        finished_utc="b",
        elapsed_seconds=1.0,
        success=True,
        sample_window_completed=True,
        global_deadline_reached=False,
        oom_detected=False,
        worker_returncodes=(0,),
        workers_exited_cleanly=True,
        controller_terminated_workers=False,
        initial_free_vram_bytes=90 * GIB,
        minimum_free_vram_bytes=85 * GIB,
        peak_incremental_vram_bytes=5 * GIB,
        peak_host_rss_bytes=2 * GIB,
        aggregate_cpu_seconds=1.0,
        average_cpu_cores=1.0,
        system_average_busy_cores=2.0,
        baseline_system_busy_cores=1.0,
        required_phases=tuple(placement.DEFAULT_REQUIRED_PHASES),
        completed_phases_by_worker=(tuple(placement.DEFAULT_REQUIRED_PHASES),),
        phase_contract_satisfied=True,
        log_paths=(),
        phase_evidence_paths=(str(state),),
        reason="phase_complete_clean_exit_probe_passed",
    )
    monkeypatch.setattr(auto, "probe_same_gpu_concurrency", lambda **_kwargs: result)
    updated = auto._phase_peak_probe_runner()  # noqa: SLF001
    assert updated.peak_incremental_vram_bytes == 20 * GIB
