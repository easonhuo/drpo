from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import yaml


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_countdown_pilot.py"
SPEC = importlib.util.spec_from_file_location("countdown_one_click", MODULE_PATH)
assert SPEC and SPEC.loader
launcher = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = launcher
SPEC.loader.exec_module(launcher)


def test_one_click_launcher_requires_only_model_and_work_dir() -> None:
    args = launcher.build_parser().parse_args([
        "--model_path", "/models/Qwen2.5-0.5B-Instruct",
        "--work_dir", "/runs/countdown-pilot",
    ])
    assert args.gpus == "auto"
    assert args.artifact_output is None
    assert args.allow_dirty is False
    assert launcher.EXPERIMENT_ID == "EXT-C-E8-V4.4-OFFLINE-BANK"


def test_registry_points_to_the_one_click_v4_5_offline_bank_runner() -> None:
    registry = Path(__file__).resolve().parents[1] / "experiments" / "registry.yaml"
    payload = yaml.safe_load(registry.read_text())
    entry = next(x for x in payload["experiments"] if x.get("id") == "EXT-C-E8-V4.4-OFFLINE-BANK")
    assert entry["status"] == "not_run"
    assert entry["parameterization"]["runner_version"] == "4.5.0-offline-negative-bank"
    assert entry["one_click_entrypoint"] == "scripts/run_countdown_pilot.py"
    assert entry["orchestration"]["reason_build_offline_not_sharded"] == (
        "preserve_one_deterministic_rng_stream_pattern_quota_and_fixed_bank_protocol"
    )
