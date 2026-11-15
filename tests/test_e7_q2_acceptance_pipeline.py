from __future__ import annotations

import argparse
import importlib.util
import json
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


def test_rebuttal_plan_lists_36_cells_30_jobs_and_uses_profile_output_contract(tmp_path):
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
    from drpo import e7_hopper_q2 as runner

    assert plan["cell_count"] == 36
    assert len({tuple(r.values()) for r in plan["matrix"]}) == 36
    assert sum(r["advantage_mode"] == "frozen" for r in plan["matrix"]) == 24
    assert sum(r["method"] in runner.REBUTTAL_METHODS for r in plan["matrix"]) == 24
    assert all(r["advantage_mode"] == "frozen" for r in plan["matrix"] if r["method"] in runner.Q2_METHODS)
    settings, _ = runner.load_rebuttal_config(plan["config"])
    assert plan["jobs"] == runner.rebuttal_jobs(settings)
    assert plan["job_count"] == 30 and plan["pair_count"] == 6
    scheduled = [tuple(cell.values()) for job in plan["jobs"] for cell in job["cells"]]
    assert len(scheduled) == len(set(scheduled)) == 36
    assert "summary/MATRIX.csv" in command and "aggregate_summary.json" not in command
    assert "cells/seed_*/*/*/training_metrics.csv" in command
    assert "--expected-commit" in command and "--require-origin-main-match" not in command
    assert plan["actor_steps"] == 1_000_000 and not work.exists()
    args.smoke = True
    command, smoke = module.build_command(args)
    assert smoke["cell_count"] == 12 and smoke["actor_steps"] == 10
    assert smoke["job_count"] == 10 and smoke["pair_count"] == 2
    assert smoke["run_class"] == "pilot" and "--smoke" in command
    defaults = module.build_parser().parse_args([])
    assert defaults.experiment_profile == "q2" and defaults.config == module.DEFAULT_CONFIG


def _rebuttal_worker_fixture(tmp_path, monkeypatch, advantage_mode="refreshed", method="is"):
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
    full = tmp_path / "full"
    module.prepare_rebuttal_context(dataset, full / "prepare", settings, base, torch.device("cpu"), "a" * 40)
    monkeypatch.setattr(module, "load_rebuttal_config", lambda *args: (settings, base))

    def fake_rollout(**kwargs):
        score = float(sum(p.detach().double().sum() for p in kwargs["policy"].parameters()))
        return {"rollout_return_mean": score, "rollout_return_std": 0., "normalized_return": score,
                "rollout_episodes": 1, "rollout_status": "available"}

    monkeypatch.setattr(module, "evaluate_d4rl_rollouts", fake_rollout)
    args = argparse.Namespace(config="toy", smoke=True, device="cpu", seed=200,
                              advantage_mode=advantage_mode, method=method,
                              dataset_path=str(dataset), work_dir=str(full), resume_from=None)
    return module, args, full


def test_q2_pair_runtime_failure_records_each_cell_method(tmp_path, monkeypatch):
    module, args, root = _rebuttal_worker_fixture(tmp_path, monkeypatch, "frozen", "far_zero")

    def fail_rollout(**kwargs):
        raise RuntimeError("simulated rollout export failure")

    monkeypatch.setattr(module, "evaluate_d4rl_rollouts", fail_rollout)
    with pytest.raises(RuntimeError, match="simulated rollout export failure"):
        module.train_rebuttal_q2_pair(args)
    for method in ("far_zero", "global_matched_far_zero"):
        cell = module._rebuttal_cell(root, 200, "frozen", method)
        for name in ("failure.json", "status.json"):
            row = json.loads((cell / name).read_text())
            assert row["method"] == method
            assert row["status"] == "runtime_failure"
            assert "simulated rollout export failure" in row["traceback"]


def test_q2_coordinator_nonzero_exit_keeps_global_cell_identity(tmp_path, monkeypatch):
    module, args, root = _rebuttal_worker_fixture(tmp_path, monkeypatch, "frozen", "far_zero")
    settings, _ = module.load_rebuttal_config(args.config, True)
    args.config = str(ROOT / "configs/e7_hopper_pq_is_rebuttal_v1.yaml")
    args.repo_root, args.gpu_ids, args.max_workers = str(ROOT), "0", 2
    args.large_file_persistence = "ephemeral"
    monkeypatch.setattr(module, "preflight_rollout_environment", lambda **kwargs: {"status": "PASS"})
    monkeypatch.setattr(module, "_plot_rebuttal", lambda *args: None)

    class FakeWorker:
        def __init__(self, command, **kwargs):
            method = command[command.index("--method") + 1]
            mode = command[command.index("--advantage-mode") + 1]
            pair = command[command.index("--job-kind") + 1] == "q2_pair"
            methods = next(p for p in settings["q2"]["pairs"] if p[0] == method) if pair else [method]
            self.returncode = -11 if method == "far_zero" else 0
            for name in methods:
                cell = module._rebuttal_cell(root, 200, mode, name)
                module._write_rebuttal_json(cell / "summary.json", {
                    "seed": 200, "advantage_mode": mode, "method": name, "status": "completed",
                    "late_mean": 7., "terminal": 7., "completed_actor_steps": settings["actor_steps"],
                    "late_count": len(settings["late_steps"]),
                })
                if self.returncode:
                    module._write_rebuttal_json(cell / "failure.json", {"method": name, "traceback": "preserve worker evidence"})

        def poll(self):
            return self.returncode

    monkeypatch.setattr(module.subprocess, "Popen", FakeWorker)
    # git_output uses subprocess.run, so resolve repository identity before the
    # fake worker substitutes Popen underneath it.
    monkeypatch.setattr(module, "git_output", lambda *args: "a" * 40)
    assert module.run_rebuttal_experiment(args) == 2
    for method in ("far_zero", "global_matched_far_zero"):
        cell = module._rebuttal_cell(root, 200, "frozen", method)
        for name in ("failure.json", "status.json"):
            row = json.loads((cell / name).read_text())
            assert row["method"] == method and row["returncode"] == -11
            assert row["traceback"] == "preserve worker evidence"
    audit = json.loads((root / "summary/TERMINAL_AUDIT.json").read_text())
    assert audit["runtime_failure_cells"] == 2 and audit["completed_cells"] == 10
    assert (root / "RUN_PARTIAL.json").is_file() and not (root / "RUN_COMPLETE.json").exists()


@pytest.mark.parametrize("advantage_mode", ["frozen", "refreshed"])
@pytest.mark.parametrize("method", ["drpo", "is"])
def test_rebuttal_interrupted_resume_reproduces_uninterrupted_updates(tmp_path, monkeypatch, advantage_mode, method):
    module, args, full = _rebuttal_worker_fixture(tmp_path, monkeypatch, advantage_mode, method)
    interrupted, resumed = (tmp_path / name for name in ("interrupted", "resumed"))
    shutil.copytree(full / "prepare", interrupted / "prepare")
    shutil.copytree(full / "prepare", resumed / "prepare")
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


@pytest.mark.parametrize("failure_in_checkpoint", [False, True])
def test_rebuttal_resume_does_not_train_past_an_observed_numerical_failure(tmp_path, monkeypatch, failure_in_checkpoint):
    module, args, failed = _rebuttal_worker_fixture(tmp_path, monkeypatch)
    actor_loss = module.rebuttal_actor_loss
    write_json = module._write_rebuttal_json
    save_checkpoint = module._atomic_rebuttal_checkpoint
    calls = 0

    def fail_third_loss(logp, coefficient):
        nonlocal calls
        calls += 1
        return logp.sum() * float("nan") if calls == 3 else actor_loss(logp, coefficient)

    def interrupt_before_summary(path, value):
        if path.name == "summary.json":
            raise InterruptedError("numerical failure saved, summary not yet exported")
        write_json(path, value)

    def keep_initial_checkpoint(path, state):
        save_checkpoint(path, state)
        if path.name == "checkpoint.pt" and state["step"] == 0:
            torch.save(state, path.with_name("initial_checkpoint.pt"))

    monkeypatch.setattr(module, "rebuttal_actor_loss", fail_third_loss)
    monkeypatch.setattr(module, "_write_rebuttal_json", interrupt_before_summary)
    monkeypatch.setattr(module, "_atomic_rebuttal_checkpoint", keep_initial_checkpoint)
    with pytest.raises(InterruptedError):
        module.train_rebuttal_branch(args)
    old_cell = module._rebuttal_cell(failed, 200, "refreshed", "is")
    if failure_in_checkpoint:
        # Checkpoint-only recovery must preserve the numerical stop as well.
        (old_cell / "numerical_failure.json").unlink()
    else:
        # The failure marker survived, but the newer failure checkpoint did not.
        shutil.copy2(old_cell / "initial_checkpoint.pt", old_cell / "checkpoint.pt")
    old_state = torch.load(old_cell / "checkpoint.pt", weights_only=False)
    resumed = tmp_path / "resumed_failure"
    shutil.copytree(failed / "prepare", resumed / "prepare")
    monkeypatch.setattr(module, "_write_rebuttal_json", write_json)
    monkeypatch.setattr(module, "_atomic_rebuttal_checkpoint", save_checkpoint)
    args.work_dir, args.resume_from = str(resumed), str(failed)
    assert module.train_rebuttal_branch(args) == 0
    cell = module._rebuttal_cell(resumed, 200, "refreshed", "is")
    summary = json.loads((cell / "summary.json").read_text())
    assert summary["status"] == "numerical_failure"
    assert summary["completed_actor_steps"] == old_state["step"] and summary["late_mean"] is None
    assert summary["numerical_failure_observed_actor_steps"] == 2
    assert calls == 3
    state = torch.load(cell / "checkpoint.pt", weights_only=False)
    assert state["training_rng"] == old_state["training_rng"]
    for group in ("actor", "critic"):
        for key, value in old_state[group].items():
            torch.testing.assert_close(state[group][key], value, rtol=0, atol=0)


def test_rebuttal_failed_worker_exit_overrides_a_previously_written_summary(tmp_path, monkeypatch):
    from drpo import e7_hopper_q2 as module

    settings, _ = module.load_rebuttal_config(ROOT / "configs/e7_hopper_pq_is_rebuttal_v1.yaml", True)
    for key in module.rebuttal_matrix(settings["seeds"]):
        cell = module._rebuttal_cell(tmp_path, key["seed"], key["advantage_mode"], key["method"])
        module._write_rebuttal_json(cell / "summary.json", {
            **key, "status": "completed", "completed_actor_steps": settings["actor_steps"],
            "late_mean": 7., "terminal": 7., "late_count": 1,
        })
    cell = module._rebuttal_cell(tmp_path, 200, "refreshed", "is")
    module._write_rebuttal_json(cell / "failure.json", {"status": "runtime_failure", "returncode": -11})
    monkeypatch.setattr(module, "_plot_rebuttal", lambda *args: None)
    audit = module.aggregate_rebuttal(tmp_path, settings)
    assert audit["runtime_failure_cells"] == 1 and audit["completed_cells"] == 11
    assert audit["status"] == "PARTIAL"
    matrix = module._read_rebuttal_csv(tmp_path / "summary/MATRIX.csv")
    row = next(r for r in matrix if r["advantage_mode"] == "refreshed" and r["method"] == "is")
    assert row["status"] == "runtime_failure" and row["late_mean"] == "7.0"
    summary = module._read_rebuttal_csv(tmp_path / "summary/METHOD_SUMMARY.csv")
    row = next(r for r in summary if r["advantage_mode"] == "refreshed" and r["method"] == "is")
    assert row["late_mean_n"] == "0" and row["late_mean_mean"] == ""


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
    assert audit["terminal_cells"] == 36 and audit["numerical_failure_cells"] == 2
    rows = module._read_rebuttal_csv(tmp_path / "summary/METHOD_SUMMARY.csv")
    pos = next(r for r in rows if r["advantage_mode"] == "frozen" and r["method"] == "positive_only")
    signed = next(r for r in rows if r["advantage_mode"] == "frozen" and r["method"] == "signed")
    assert float(pos["late_mean_mean"]) == 3 and float(pos["late_mean_seed_sample_sd"]) == 2
    assert signed["late_mean_n"] == "2" and float(signed["late_mean_mean"]) == 3
    q2 = module._read_rebuttal_csv(tmp_path / "summary/Q2_PAIRED_COMPARISONS.csv")
    aggregate = [r for r in q2 if r["row_type"] == "aggregate"]
    assert len(aggregate) == 8
    far_vs_signed = next(r for r in aggregate if r["method"] == "far_zero" and r["reference"] == "signed" and r["metric"] == "late_mean")
    assert far_vs_signed["paired_n"] == "2" and float(far_vs_signed["mean_difference"]) == 0
    q2_methods = module._read_rebuttal_csv(tmp_path / "summary/Q2_METHOD_SUMMARY.csv")
    assert len(q2_methods) == 8
    assert next(r for r in q2_methods if r["method"] == "signed")["late_mean_n"] == "2"


@pytest.mark.parametrize("intervention", ["far_zero", "far_cap"])
def test_rebuttal_q2_pair_resume_preserves_both_trajectories_and_partial_budget_block(tmp_path, monkeypatch, intervention):
    module, args, full = _rebuttal_worker_fixture(tmp_path, monkeypatch, "frozen", intervention)
    settings, _ = module.load_rebuttal_config(args.config, True)
    settings["log_interval"] = 3  # checkpoint at step 2 holds an unfinished block
    interrupted, resumed = (tmp_path / name for name in ("interrupted", "resumed"))
    for directory in (interrupted, resumed):
        shutil.copytree(full / "prepare", directory / "prepare")
    assert module.train_rebuttal_q2_pair(args) == 0
    save = module._atomic_rebuttal_checkpoint

    def interrupt(path, state):
        save(path, state)
        if path.name == "checkpoint.pt" and state["step"] == 2:
            raise InterruptedError("pair checkpoint saved")

    monkeypatch.setattr(module, "_atomic_rebuttal_checkpoint", interrupt)
    args.work_dir = str(interrupted)
    with pytest.raises(InterruptedError):
        module.train_rebuttal_q2_pair(args)
    old = module._q2_pair_path(interrupted, 200, intervention) / "checkpoint.pt"
    digest = module.sha256_file(old)
    state = torch.load(old, weights_only=False)
    assert state["step"] == state["budget_block"]["count"] == 2
    monkeypatch.setattr(module, "_atomic_rebuttal_checkpoint", save)
    args.work_dir, args.resume_from = str(resumed), str(interrupted)
    assert module.train_rebuttal_q2_pair(args) == 0
    assert module.sha256_file(old) == digest
    first = torch.load(module._q2_pair_path(full, 200, intervention) / "checkpoint.pt", weights_only=False)
    second = torch.load(module._q2_pair_path(resumed, 200, intervention) / "checkpoint.pt", weights_only=False)
    for k in ("actors", "optimizers", "global_rng"):
        assert _nested_equal(first[k], second[k])
    for k in ("training_rng", "histories", "budgets", "clipped", "step", "calibration"):
        assert first[k] == second[k], k
    assert sum(row["updates"] for row in first["budgets"]) == 4
    assert all(row["q2_absolute_error_max"] < 1e-5 for row in first["budgets"])
    # Pair batch stream remains identical to the original independent seed map.
    context = module._load_rebuttal_context(full / "prepare", Path(args.dataset_path), settings, module.load_rebuttal_config(args.config, True)[1])
    rng = np.random.default_rng(np.random.SeedSequence([200, 41591]))
    for _ in range(4):
        module.sample_indices(rng, context["train_ids"], settings["actor_batch_size"])
    assert rng.bit_generator.state == first["training_rng"]
    job = next(job for job in module.rebuttal_jobs(settings) if job["method"] == intervention)
    reused = tmp_path / "reused"
    assert module._reuse_rebuttal_job(full, reused, job, settings, "a" * 40)
    # A copied complete pair can still recover an interrupted endpoint export
    # through its indexed synchronized checkpoint, without independent halves.
    endpoint = tmp_path / "endpoint"
    shutil.copytree(full / "prepare", endpoint / "prepare")
    args.work_dir, args.resume_from = str(endpoint), str(reused)
    assert module.train_rebuttal_q2_pair(args) == 0
    recovered = torch.load(module._q2_pair_path(endpoint, 200, intervention) / "checkpoint.pt", weights_only=False)
    assert recovered["histories"] == first["histories"]
    assert _nested_equal(recovered["actors"], first["actors"])
    failed_cell = module._rebuttal_cell(full, 200, "frozen", f"global_matched_{intervention}")
    module._write_rebuttal_json(failed_cell / "failure.json", {"returncode": -11})
    assert not module._reuse_rebuttal_job(full, tmp_path / "must_retry_pair", job, settings, "a" * 40)


def _nested_equal(a, b):
    if isinstance(a, torch.Tensor):
        return torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return np.array_equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_nested_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_nested_equal(x, y) for x, y in zip(a, b))
    return a == b


@pytest.mark.parametrize("status", ["undefined_no_initial_negative", "nonfinite_calibration"])
def test_q2_undefined_calibration_is_reported_without_manufacturing_negative_data(tmp_path, monkeypatch, status):
    module, args, full = _rebuttal_worker_fixture(tmp_path, monkeypatch, "frozen", "far_zero")
    path = full / "prepare/q2_calibration_seed_200.json"
    calibration = json.loads(path.read_text())
    calibration.update(status=status, far_threshold=None, far_cap=None)
    module._write_rebuttal_json(path, calibration)
    assert module.train_rebuttal_q2_pair(args) == 0
    rows = [json.loads((module._rebuttal_cell(full, 200, "frozen", method) / "summary.json").read_text()) for method in ("far_zero", "global_matched_far_zero")]
    expected = "calibration_undefined" if status == "undefined_no_initial_negative" else "numerical_failure"
    assert all(row["status"] == expected and row["completed_actor_steps"] == 0 for row in rows)
    assert all(row["late_mean"] is None for row in rows)


@pytest.mark.parametrize("checkpoint_only", [False, True])
def test_rebuttal_q2_pair_numerical_stop_survives_interrupted_export(tmp_path, monkeypatch, checkpoint_only):
    module, args, full = _rebuttal_worker_fixture(tmp_path, monkeypatch, "frozen", "far_cap")
    clip, write, save = module._clip_rebuttal_gradients, module._write_rebuttal_json, module._atomic_rebuttal_checkpoint
    calls = 0

    def fail_third_step(parameters, limit):
        nonlocal calls
        calls += 1
        norm = clip(parameters, limit)
        return norm * float("nan") if calls == 5 else norm

    def interrupt_export(path, value):
        if path.name == "summary.json":
            raise InterruptedError("numerical pair stop saved")
        write(path, value)

    def initial_snapshot(path, state):
        save(path, state)
        if path.name == "checkpoint.pt" and state["step"] == 0:
            torch.save(state, path.with_name("initial.pt"))

    monkeypatch.setattr(module, "_clip_rebuttal_gradients", fail_third_step)
    monkeypatch.setattr(module, "_write_rebuttal_json", interrupt_export)
    monkeypatch.setattr(module, "_atomic_rebuttal_checkpoint", initial_snapshot)
    with pytest.raises(InterruptedError):
        module.train_rebuttal_q2_pair(args)
    pair = module._q2_pair_path(full, 200, "far_cap")
    if checkpoint_only:
        (pair / "numerical_failure.json").unlink()
    else:
        shutil.copy2(pair / "initial.pt", pair / "checkpoint.pt")
    before = torch.load(pair / "checkpoint.pt", weights_only=False)
    resumed = tmp_path / "resumed_stop"
    shutil.copytree(full / "prepare", resumed / "prepare")
    args.work_dir, args.resume_from = str(resumed), str(full)
    monkeypatch.setattr(module, "_write_rebuttal_json", write)
    monkeypatch.setattr(module, "_atomic_rebuttal_checkpoint", save)
    assert module.train_rebuttal_q2_pair(args) == 0
    assert calls == 6  # both gradients checked, neither optimizer advanced
    state = torch.load(module._q2_pair_path(resumed, 200, "far_cap") / "checkpoint.pt", weights_only=False)
    assert _nested_equal(state["actors"], before["actors"])
    assert _nested_equal(state["optimizers"], before["optimizers"])
    assert state["step"] == before["step"]
    assert state["numerical_failure"]["observed_completed_actor_steps"] == 2
    rows = [json.loads((module._rebuttal_cell(resumed, 200, "frozen", method) / "summary.json").read_text()) for method in ("far_cap", "global_matched_far_cap")]
    assert [r["status"] for r in rows] == ["numerical_failure", "paired_stop"]
    assert all(r["late_mean"] is None for r in rows)


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
