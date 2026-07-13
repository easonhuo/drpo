from __future__ import annotations

import importlib.util
from pathlib import Path

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


def test_gpu_placement_probe_defaults_are_bounded() -> None:
    args = auto.build_parser().parse_args(required_args())
    assert args.probe_budget_seconds == 600
    assert args.max_slots_per_gpu == 8
    assert args.per_worker_host_memory_safety_factor == 1.25
    assert args.per_worker_vram_safety_factor == 1.25
    assert args.cpu_fraction == 0.85
