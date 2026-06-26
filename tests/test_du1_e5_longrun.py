from __future__ import annotations

import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "src" / "drpo" / "du1_e5_longrun_rerun.py"
VALIDATOR = REPO_ROOT / "scripts" / "validate_formal_execution_channel.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("du1_e5_longrun_rerun", RUNNER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_direct_softmax_reconstructs_locked_reference() -> None:
    module = load_runner()
    protocol = module.Protocol()
    summaries = {}
    for name, initial in module.DIRECT_CASES.items():
        _, summaries[name] = module.run_direct_case(name, initial, protocol)
    comparison = module.direct_reference_comparison(summaries)
    assert comparison["all_direct_reference_checks_passed"] is True
    assert summaries["high_probability_negative"]["entropy_pattern"] == "rise_then_fall"
    assert summaries["low_probability_negative"]["entropy_pattern"] == "nonincreasing_or_flat"
    assert max(row["max_score"] for row in summaries.values()) <= math.sqrt(2.0) + 1e-12


def test_du1_preflight_and_long_run_roles() -> None:
    module = load_runner()
    protocol = module.Protocol()
    data = module.build_seed_data(10, protocol)
    report = module.preflight(data, protocol)
    assert report["all_checks_passed"] is True
    assert report["near_probability_exceeds_far"] is True
    assert report["action_ids_are_permuted"] is True

    rows = []
    for method in module.METHODS:
        _, summary = module.run_method(data, method, protocol)
        rows.append(summary)
    module.add_task_classification(rows, protocol)
    by_method = {row["method"]: row for row in rows}

    assert by_method["baseline"]["task_collapse"] is True
    assert by_method["baseline"]["support_collapse"] is True
    assert by_method["near_zero"]["task_collapse"] is True
    assert by_method["near_zero"]["support_collapse"] is True
    assert by_method["far_zero"]["terminal_class"] == "stable_bounded"
    assert by_method["far_zero"]["support_collapse"] is False
    assert by_method["far_cap"]["terminal_class"] == "stable_bounded"
    assert by_method["global_scale"]["task_collapse"] is False
    assert by_method["global_scale"]["support_collapse"] is True
    assert by_method["positive_only"]["terminal_class"] == "stable_bounded"
    assert all(row["nan_inf_numerical_failure"] is False for row in rows)
    assert all(row["historical_joint_class_matches"] is True for row in rows)


def test_smoke_cli_writes_expected_outputs(tmp_path: Path) -> None:
    output = tmp_path / "run"
    proc = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--mode",
            "smoke",
            "--seeds",
            "10",
            "--max-steps",
            "400",
            "--direct-steps",
            "1000",
            "--output-root",
            str(output),
            "--skip-plots",
        ],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert proc.returncode == 0, proc.stderr
    for relative in (
        "config.json",
        "preflight.json",
        "RUN_COMPLETE.json",
        "terminal_audit.json",
        "REPORT.md",
        "manifest.json",
        "direct_softmax/summary.json",
        "causal/per_seed_summary.csv",
        "causal/aggregate_summary.json",
        "historical_reference_comparison.json",
    ):
        assert (output / relative).is_file(), relative
    audit = json.loads((output / "terminal_audit.json").read_text())
    assert audit["raw_runs_complete"] is True
    assert audit["task_support_nan_inf_reported_separately"] is True
    assert audit["scientific_status_candidate"] == "pilot_or_inconclusive"


def test_registry_and_canonical_formal_channel() -> None:
    registry = yaml.safe_load((REPO_ROOT / "experiments" / "registry.yaml").read_text())
    experiments = {entry["id"]: entry for entry in registry["experiments"]}
    entry = experiments["D-U1-E5-LONGRUN-RERUN"]
    assert entry["status"] == "not_run"
    assert entry["execution_class"] == "formal"
    assert entry["formal_execution"]["activation_state"] == "active"
    assert entry["formal_execution"]["entrypoint"] == "src/drpo/du1_e5_longrun_rerun.py"
    assert entry["formal_execution"]["runner_archive_policy"] == {"mode": "forbid"}
    assert entry["protocol"]["du1_causal"]["formal_held_out_seeds"] == list(range(10, 30))
    assert entry["protocol"]["du1_causal"]["maximum_steps"] == 20000

    proc = subprocess.run(
        [sys.executable, str(VALIDATOR), "--repo-root", str(REPO_ROOT)],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert proc.returncode == 0, proc.stderr
