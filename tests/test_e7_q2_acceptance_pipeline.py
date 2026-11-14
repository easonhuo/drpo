from __future__ import annotations

import argparse
import importlib.util
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from drpo.e7_hopper_q2 import (
    METHODS,
    SquashedGaussianPolicy,
    actor_batch_loss,
    classify_actor_terminal,
    load_config,
    normalized_window_drift,
    parameter_update_statistics,
    spearman,
    train_actor_stage,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "e7_hopper_q2_medium_replay_v2.yaml"


def test_rebuttal_plan_lists_24_cells_and_uses_profile_output_contract(tmp_path):
    spec = importlib.util.spec_from_file_location("e7_launcher", ROOT / "scripts/run_e7_hopper_q2.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    dataset = tmp_path / "data.hdf5"
    dataset.touch()
    work = tmp_path / "planned_only"
    args = module.build_parser().parse_args([
        "--experiment-profile", "rebuttal_pq_is", "--plan-only",
        "--dataset-path", str(dataset), "--work-dir", str(work),
    ])
    command, plan = module.build_command(args)
    assert plan["cell_count"] == 24
    assert len({tuple(r.values()) for r in plan["matrix"]}) == 24
    assert {r["method"] for r in plan["matrix"]} == {"positive_only", "signed", "drpo", "is"}
    assert "summary/MATRIX.csv" in command and "aggregate_summary.json" not in command
    assert "cells/seed_*/*/*/training_metrics.csv" in command
    assert "--expected-commit" in command and "--require-origin-main-match" not in command
    assert plan["actor_steps"] == 1_000_000 and not work.exists()
    args.smoke = True
    command, smoke = module.build_command(args)
    assert smoke["cell_count"] == 8 and smoke["actor_steps"] == 10
    assert smoke["run_class"] == "pilot" and "--smoke" in command
    defaults = module.build_parser().parse_args([])
    assert defaults.experiment_profile == "q2" and defaults.config == module.DEFAULT_CONFIG


@pytest.mark.parametrize("advantage_mode", ["frozen", "refreshed"])
@pytest.mark.parametrize("method", ["drpo", "is"])
def test_rebuttal_interrupted_resume_reproduces_uninterrupted_updates(tmp_path, monkeypatch, advantage_mode, method):
    import h5py

    from drpo import e7_hopper_q2 as module

    settings, base = module.load_rebuttal_config(ROOT / "configs/e7_hopper_pq_is_rebuttal_v1.yaml", True)
    settings.update(actor_steps=4, critic_steps=2, critic_eval_interval=1, actor_batch_size=8,
                    critic_batch_size=8, diagnostic_size=8, diagnostic_batch_size=8,
                    diagnostic_interval=1, checkpoint_interval=1, log_interval=1,
                    evaluation_interval=1, variance_steps=[0, 4], variance_batches=2,
                    variance_batch_size=8, late_steps=[3, 4], max_transitions=None)
    base = replace(base, hidden_sizes=(4,), critic_batch_size=8, actor_batch_size=8)
    dataset = tmp_path / "toy.hdf5"
    rng = np.random.default_rng(11)
    with h5py.File(dataset, "w") as f:
        f["observations"] = rng.normal(size=(80, 3)).astype(np.float32)
        f["next_observations"] = rng.normal(size=(80, 3)).astype(np.float32)
        f["actions"] = np.tanh(rng.normal(size=(80, 2))).astype(np.float32)
        f["rewards"] = rng.normal(size=80).astype(np.float32)
        f["terminals"] = np.arange(80) % 4 == 3
        f["timeouts"] = np.zeros(80, bool)
        f["infos/action_log_probs"] = np.full(80, -2., np.float32)
    full, interrupted, resumed = (tmp_path / name for name in ("full", "interrupted", "resumed"))
    module.prepare_rebuttal_context(dataset, full / "prepare", settings, base, torch.device("cpu"), "a" * 40)
    shutil.copytree(full / "prepare", interrupted / "prepare")
    shutil.copytree(full / "prepare", resumed / "prepare")
    monkeypatch.setattr(module, "load_rebuttal_config", lambda *args: (settings, base))

    def fake_rollout(**kwargs):
        score = float(sum(p.detach().double().sum() for p in kwargs["policy"].parameters()))
        return {"rollout_return_mean": score, "rollout_return_std": 0., "normalized_return": score,
                "rollout_episodes": 1, "rollout_status": "available"}

    monkeypatch.setattr(module, "evaluate_d4rl_rollouts", fake_rollout)
    args = argparse.Namespace(config="toy", smoke=True, device="cpu", seed=200,
                              advantage_mode=advantage_mode, method=method,
                              dataset_path=str(dataset), work_dir=str(full), resume_from=None)
    assert module.train_rebuttal_branch(args) == 0
    save = module._atomic_rebuttal_checkpoint

    def interrupt_after_checkpoint(path, state):
        save(path, state)
        if path.name == "checkpoint.pt" and state["step"] == 2:
            raise InterruptedError("simulated interruption after durable checkpoint")

    monkeypatch.setattr(module, "_atomic_rebuttal_checkpoint", interrupt_after_checkpoint)
    args.work_dir = str(interrupted)
    with pytest.raises(InterruptedError):
        module.train_rebuttal_branch(args)
    source = module._rebuttal_cell(interrupted, 200, advantage_mode, method) / "checkpoint.pt"
    checksum = module.sha256_file(source)
    monkeypatch.setattr(module, "_atomic_rebuttal_checkpoint", save)
    args.work_dir, args.resume_from = str(resumed), str(interrupted)
    assert module.train_rebuttal_branch(args) == 0
    assert module.sha256_file(source) == checksum
    first = torch.load(module._rebuttal_cell(full, 200, advantage_mode, method) / "checkpoint.pt", weights_only=False)
    second = torch.load(module._rebuttal_cell(resumed, 200, advantage_mode, method) / "checkpoint.pt", weights_only=False)
    for group in ("actor", "critic"):
        for key in first[group]:
            torch.testing.assert_close(first[group][key], second[group][key], atol=0, rtol=0)
    for group in ("actor_optimizer", "critic_optimizer"):
        assert first[group]["param_groups"] == second[group]["param_groups"]
        for key, state in first[group]["state"].items():
            for name, value in state.items():
                torch.testing.assert_close(value, second[group]["state"][key][name], atol=0, rtol=0)
    assert first["training_rng"] == second["training_rng"]
    assert first["critic_updates"] == second["critic_updates"] == (4 if advantage_mode == "refreshed" else 0)
    assert first["history"] == second["history"]
    endpoint = tmp_path / "endpoint_export_recovery"
    shutil.copytree(full / "prepare", endpoint / "prepare")
    args.work_dir, args.resume_from = str(endpoint), str(full)
    assert module.train_rebuttal_branch(args) == 0
    exported = module._rebuttal_cell(endpoint, 200, advantage_mode, method)
    for name in ("diagnostics.csv", "training_metrics.csv", "rollouts.csv", "checkpoint_index.json"):
        assert (exported / name).stat().st_size > 0
    recovered = torch.load(exported / "checkpoint.pt", weights_only=False)
    assert recovered["step"] == 4 and recovered["history"] == first["history"]


def test_rebuttal_aggregation_retains_failures_and_uses_actual_n_and_sample_sd(tmp_path, monkeypatch):
    from drpo import e7_hopper_q2 as module

    settings, _ = module.load_rebuttal_config(ROOT / "configs/e7_hopper_pq_is_rebuttal_v1.yaml")
    for key in module.rebuttal_matrix(settings["seeds"]):
        fail = key["method"] == "signed" and key["seed"] == 202
        value = 1 + 2 * (key["seed"] - 200) + int(key["method"] != "positive_only")
        row = {**key, "status": "numerical_failure" if fail else "completed",
               "late_mean": None if fail else value, "terminal": None if fail else value,
               "completed_actor_steps": 100 if fail else settings["actor_steps"],
               "late_count": 0 if fail else len(settings["late_steps"])}
        cell = module._rebuttal_cell(tmp_path, key["seed"], key["advantage_mode"], key["method"])
        module._write_rebuttal_json(cell / "summary.json", row)
    monkeypatch.setattr(module, "_plot_rebuttal", lambda *args: None)
    audit = module.aggregate_rebuttal(tmp_path, settings)
    assert audit["terminal_cells"] == 24 and audit["numerical_failure_cells"] == 2
    rows = module._read_rebuttal_csv(tmp_path / "summary/METHOD_SUMMARY.csv")
    pos = next(r for r in rows if r["advantage_mode"] == "frozen" and r["method"] == "positive_only")
    signed = next(r for r in rows if r["advantage_mode"] == "frozen" and r["method"] == "signed")
    assert float(pos["late_mean_mean"]) == 3 and float(pos["late_mean_seed_sample_sd"]) == 2
    assert signed["late_mean_n"] == "2" and float(signed["late_mean_mean"]) == 3


def test_protocol_v43_loads_fixed_longrun_budgets_and_dynamic_global_control() -> None:
    config = load_config(CONFIG)
    assert "dynamic_budget_matched_global" in METHODS
    assert "budget_matched_global" not in METHODS
    assert config.critic_validation_r2_min == 0.50
    assert config.critic_update_tolerance == 0.0002
    assert config.actor_update_tolerance == 0.0002
    assert config.actor_state_drift_tolerance == 0.01
    assert config.formal.critic_max_steps == 100_000
    assert config.formal.positive_max_steps == 100_000
    assert config.formal.branch_max_steps == 200_000
    assert config.formal.critic_eval_interval == 2_000
    assert config.formal.actor_eval_interval == 5_000
    assert config.formal.rollout_eval_interval == 25_000
    assert config.formal.rollout_episodes == 5
    assert config.formal.final_rollout_episodes == 20


def test_parameter_update_statistics_are_relative_to_model_scale() -> None:
    layer = torch.nn.Linear(3, 2, bias=False)
    before = [parameter.detach().clone() for parameter in layer.parameters()]
    with torch.no_grad():
        for parameter in layer.parameters():
            parameter.add_(0.01)
    stats = parameter_update_statistics(before, layer.parameters(), elapsed_steps=10)
    assert stats["raw_per_step"] > 0.0
    assert stats["rms_per_step"] > 0.0
    assert stats["relative_per_step"] > 0.0
    assert stats["relative_per_step"] < stats["raw_per_step"] * 10.0


def test_spearman_handles_monotone_values_and_ties() -> None:
    first = np.asarray([1.0, 2.0, 2.0, 4.0])
    second = np.asarray([10.0, 20.0, 20.0, 40.0])
    assert abs(spearman(first, second) - 1.0) < 1e-12


def test_dynamic_global_matches_far_cap_output_score_proxy_per_batch() -> None:
    torch.manual_seed(3)
    policy = SquashedGaussianPolicy(3, 2, (8,), -5.0, 2.0, 1e-6)
    obs = torch.randn(12, 3)
    actions = torch.tanh(torch.randn(12, 2))
    advantages = torch.tensor(
        [1.0, 0.5, -0.2, -0.4, -0.6, -0.8, -1.0, -1.2, -0.3, -0.7, 0.2, -0.9]
    )
    _, diagnostics = actor_batch_loss(
        policy,
        obs,
        actions,
        advantages,
        "dynamic_budget_matched_global",
        far_threshold=1.0,
        global_scale=0.123,
        far_cap_score=1.5,
    )
    assert 0.0 <= diagnostics["dynamic_global_scale"] <= 1.0
    assert diagnostics["negative_influence_proxy_before"] > 0.0
    assert abs(
        diagnostics["negative_influence_proxy_after"]
        - diagnostics["negative_influence_proxy_target"]
    ) <= 1e-5 * max(diagnostics["negative_influence_proxy_target"], 1.0)


def test_actor_terminal_uses_relative_update_not_raw_gradient_gate() -> None:
    config = load_config(CONFIG)
    config = replace(
        config,
        audit_windows=3,
        actor_relative_slope_tolerance=1e-4,
        actor_state_drift_tolerance=1e-4,
        actor_update_tolerance=1e-3,
        actor_gradient_tolerance=1e-12,
    )
    rows = []
    for step in (100, 200, 300):
        rows.append(
            {
                "step": step,
                "loss": 1.0,
                "positive_nll": 1.0,
                "gradient_norm": 100.0,
                "update_norm": 0.1,
                "relative_update_norm": 1e-5,
                "sigma_mean": 1.0,
                "mean_abs": 0.2,
                "phantom_distance_mean": 2.0,
                "mean_boundary_fraction": 0.0,
                "log_std_min_fraction": 0.0,
                "log_std_max_fraction": 0.0,
                "normalized_return": 30.0,
                "rollout_status": "available",
            }
        )
    audit = classify_actor_terminal(
        rows,
        config,
        candidate_step=100,
        extension_complete=True,
        fixed_budget_completed=True,
    )
    assert audit["state"] == "finite_terminal"
    assert audit["fixed_budget_completed"] is True
    assert audit["terminal_audit_controls_stopping"] is False
    assert not audit["numerical_nonfinite"]


def test_normalized_window_drift_uses_training_step_and_state_scale() -> None:
    rows = [
        {"step": 100, "value": 2.00},
        {"step": 200, "value": 2.01},
        {"step": 300, "value": 2.02},
    ]
    drift = normalized_window_drift(rows, "value", 3)
    assert 0.009 < drift < 0.011

def test_actor_training_consumes_fixed_budget_after_early_terminal_candidate(
    tmp_path: Path,
) -> None:
    torch.manual_seed(5)
    config = replace(
        load_config(CONFIG),
        hidden_sizes=(4,),
        actor_batch_size=8,
        audit_windows=2,
        actor_state_drift_tolerance=1e9,
        actor_update_tolerance=1e9,
    )
    obs = np.zeros((24, 3), dtype=np.float32)
    actions = np.zeros((24, 2), dtype=np.float32)
    advantages = np.ones(24, dtype=np.float32)
    policy = SquashedGaussianPolicy(3, 2, (4,), -5.0, 2.0, 1e-6)
    _, audit = train_actor_stage(
        policy=policy,
        method="positive_only",
        obs=obs,
        actions=actions,
        advantages=advantages,
        train_indices=np.arange(24, dtype=np.int64),
        audit_indices=np.arange(8, dtype=np.int64),
        fixed_negative_indices=np.arange(8, dtype=np.int64),
        config=config,
        min_steps=1,
        max_steps=4,
        eval_interval=1,
        seed=11,
        device=torch.device("cpu"),
        output_dir=tmp_path / "actor",
    )
    assert audit["candidate_step"] == 1
    assert audit["extension_complete"] is True
    assert audit["final_step"] == 4
    assert audit["fixed_budget_completed"] is True
    assert audit["stopping_rule"] == "fixed_optimizer_steps"
    assert audit["terminal_audit_controls_stopping"] is False


def test_actor_numerical_failure_stops_before_optimizer_step(tmp_path: Path) -> None:
    torch.manual_seed(9)
    config = replace(
        load_config(CONFIG),
        hidden_sizes=(4,),
        actor_batch_size=8,
        audit_windows=2,
    )
    obs = np.zeros((16, 3), dtype=np.float32)
    actions = np.zeros((16, 2), dtype=np.float32)
    advantages = np.full(16, np.nan, dtype=np.float32)
    policy = SquashedGaussianPolicy(3, 2, (4,), -5.0, 2.0, 1e-6)
    before = {
        name: parameter.detach().clone()
        for name, parameter in policy.named_parameters()
    }
    _, audit = train_actor_stage(
        policy=policy,
        method="signed",
        obs=obs,
        actions=actions,
        advantages=advantages,
        train_indices=np.arange(16, dtype=np.int64),
        audit_indices=np.arange(8, dtype=np.int64),
        fixed_negative_indices=np.arange(8, dtype=np.int64),
        config=config,
        min_steps=1,
        max_steps=4,
        eval_interval=1,
        seed=13,
        device=torch.device("cpu"),
        output_dir=tmp_path / "actor_nonfinite",
    )
    assert audit["fixed_budget_completed"] is False
    assert audit["early_stop_reason"] == "nonfinite_train_loss"
    assert audit["numerical_nonfinite"] is True
    for name, parameter in policy.named_parameters():
        assert torch.equal(parameter.detach(), before[name])
