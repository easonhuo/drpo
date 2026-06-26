from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"


def load_runner():
    src_root = str(SRC_ROOT)
    if src_root not in sys.path:
        sys.path.insert(0, src_root)
    return importlib.import_module("drpo.drpo_cu1_e1_e4_oneclick")


def _row(
    step: int,
    displacement: float,
    reward: float,
    raw_gradient: float,
    adam_update: float,
) -> dict[str, object]:
    return {
        "step": step,
        "stage": "full_state_audit" if step in {2400, 3200, 4000} else "adam_training",
        "normalized_extrapolation_displacement": displacement,
        "reward": reward,
        "full_data_raw_total_gradient_norm": raw_gradient,
        "adam_parameter_update_norm": adam_update,
        "finite_parameters": True,
        "log_sigma_output_finite": True,
        "sigma_output_finite": True,
    }


def test_registry_and_handoff_freeze_e4_convergence_protocol() -> None:
    registry = yaml.safe_load((REPO_ROOT / "experiments" / "registry.yaml").read_text())
    experiments = {row["id"]: row for row in registry["experiments"]}
    conv = experiments["C-U1-E4-CONV-01"]
    taper = experiments["C-U1-E4-TAPER-01"]

    assert conv["status"] == "not_run"
    assert conv["scope"]["alphas"] == [0.75, 1.0, 1.25]
    assert conv["scope"]["positive_only_additional_run"] is False
    assert conv["training"]["max_steps"] == 4000
    assert conv["training"]["full_state_audit_steps"] == [400, 800, 1600, 2400, 3200, 4000]
    assert conv["training"]["terminal_window_1"] == [2000, 3000]
    assert conv["training"]["terminal_window_2"] == [3000, 4000]
    assert conv["terminal_classification"]["residual_threshold_2e_3_is_hard_gate"] is False
    assert conv["aggregate_acceptance"]["minimum_expected_state_seeds_per_alpha"] == 18
    assert taper["execution_gate"]["depends_on_delivered_experiment"] == "C-U1-E4-CONV-01"

    handoff = (REPO_ROOT / "docs" / "handoff.md").read_text()
    assert "v33（C-U1 E4 长程终态确认协议冻结版）" in handoff
    assert "从追加运行范围中移除 `alpha=0`" in handoff
    assert "`W1=2000--3000` 与 `W2=3000--4000`" in handoff
    assert "`C-U1-E4-TAPER-01` 继续阻塞" in handoff


def test_runner_protocol_values_and_stage_are_frozen() -> None:
    runner = load_runner()
    args = runner.parse_args(
        ["--stage", "e4_convergence", "--output-root", "outputs/test-e4-conv"]
    )
    assert args.stage == "e4_convergence"
    assert runner.P.e4_convergence_alphas == (0.75, 1.0, 1.25)
    assert runner.P.e4_convergence_steps == 4000
    assert runner.P.e4_convergence_audit_steps == (400, 800, 1600, 2400, 3200, 4000)
    assert runner.P.e4_convergence_window_1 == (2000, 3000)
    assert runner.P.e4_convergence_window_2 == (3000, 4000)
    assert runner.P.e4_convergence_consensus_min == 18


def test_terminal_classifier_distinguishes_beneficial_over_and_runaway() -> None:
    runner = load_runner()
    steps = list(range(2000, 4001, 200))

    beneficial = [
        _row(step, 1.00 + 0.005 * (step - 2000) / 2000, 0.992, 1.0, 0.1)
        for step in steps
    ]
    assert (
        runner.classify_e4_convergence_trajectory(beneficial)["terminal_state"]
        == "stable_beneficial_extrapolation"
    )

    over = [
        _row(step, 1.95 + 0.005 * (step - 2000) / 2000, 0.65, 1.0, 0.1)
        for step in steps
    ]
    assert (
        runner.classify_e4_convergence_trajectory(over)["terminal_state"]
        == "stable_over_extrapolation"
    )

    runaway = []
    for step in steps:
        progress = (step - 2000) / 2000
        runaway.append(_row(step, 1.5 + 0.4 * progress, 0.5, 1.0 + 2.0 * progress, 0.1))
    assert (
        runner.classify_e4_convergence_trajectory(runaway)["terminal_state"]
        == "finite_continuing_runaway"
    )


def test_aggregate_gate_allows_only_expected_or_inconclusive() -> None:
    runner = load_runner()
    rows = []
    expected = {
        0.75: "stable_beneficial_extrapolation",
        1.0: "stable_beneficial_extrapolation",
        1.25: "stable_over_extrapolation",
    }
    for alpha, state in expected.items():
        rows.extend(
            {"alpha": alpha, "terminal_state": state, "seed": seed}
            for seed in range(50, 68)
        )
        rows.extend(
            {"alpha": alpha, "terminal_state": "terminal_state_inconclusive", "seed": seed}
            for seed in range(68, 70)
        )
    audit = runner.e4_convergence_terminal_audit(rows)
    assert audit["scientific_acceptance_all_passed"] is True

    rows[0] = {"alpha": 0.75, "terminal_state": "finite_continuing_runaway", "seed": 50}
    audit = runner.e4_convergence_terminal_audit(rows)
    assert audit["scientific_acceptance_all_passed"] is False
    assert audit["per_alpha"]["0.75"]["explicit_opposite_count"] == 1


def test_smoke_stage_writes_audits_and_checkpoint(tmp_path: Path) -> None:
    output = tmp_path / "e4-convergence-smoke"
    env = os.environ.copy()
    env["DRPO_CU1_SMOKE"] = "1"
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "src" / "drpo" / "drpo_cu1_e1_e4_oneclick.py"),
            "--stage",
            "e4_convergence",
            "--output-root",
            str(output),
        ],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        timeout=180,
    )
    complete = json.loads((output / "RUN_COMPLETE.json").read_text())
    assert complete["stage"] == "e4_convergence"
    assert complete["e4_convergence_rows"] == 3
    assert (output / "e4_convergence" / "per_seed.csv").is_file()
    assert (output / "e4_convergence" / "terminal_audit.json").is_file()
    assert (output / "e4_convergence" / "TERMINAL_AUDIT.md").is_file()
    checkpoints = list((output / "checkpoints").glob("*.zip"))
    assert len(checkpoints) == 1
