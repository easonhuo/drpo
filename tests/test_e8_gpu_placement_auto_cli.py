from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

from drpo import countdown_e8_oracle_offline_v2_taper_resource_probe as resource_probe
from drpo import runtime_gpu_placement_autotune as placement

SCRIPT = Path("scripts/run_countdown_e8_oracle_offline_v2_taper_auto.py")
SPEC = importlib.util.spec_from_file_location("e8_gpu_placement_auto", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
auto = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(auto)


def required_args() -> list[str]:
    return [
        "--model_path",
        "model",
        "--work_dir",
        "work",
        "--bank",
        "bank.jsonl",
        "--val",
        "val.jsonl",
        "--test",
        "test.jsonl",
        "--global_calibration",
        "calibration.json",
        "--base_config",
        "base.yaml",
        "--sweep_config",
        "sweep.yaml",
    ]


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
    assert args.single_probe_seconds == 120
    assert args.validation_probe_seconds == 180
    assert args.max_slots_per_gpu == 8
    assert args.per_worker_host_memory_safety_factor == 1.25
    assert args.per_worker_vram_safety_factor == 1.25
    assert args.cpu_fraction == 0.85
    assert placement.PROBE_CONTRACT_VERSION == 2
    assert "evaluation_peak_completed" in placement.DEFAULT_REQUIRED_PHASES


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
    assert Path(command[1]).name == "countdown_e8_oracle_offline_v2_taper_resource_probe.py"
    assert "worker" not in command
    assert "--output_dir" in command
