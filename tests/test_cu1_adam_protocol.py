from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = REPO_ROOT / "src" / "drpo" / "drpo_cu1_e1_e4_oneclick.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("drpo_cu1_adam_runner_test", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_registered_adam_stages_and_optimizer() -> None:
    runner = load_runner()

    args = runner.parse_args(["--stage", "e3", "--output-root", "outputs/test-e3"])
    assert args.stage == "e3"
    assert args.output_root == Path("outputs/test-e3")

    actor = runner.GaussianActor().to(runner.DEVICE)
    optimizer = runner.make_adam(actor.all_parameters(), runner.P.e3_learn_lr)
    assert isinstance(optimizer, torch.optim.Adam)
    assert optimizer.defaults["betas"] == (0.9, 0.999)
    assert optimizer.defaults["eps"] == 1e-8

    for function in (
        runner.run_intervention,
        runner.run_local_scan_seed,
        runner.run_control_seed,
        runner.run_variance_robustness,
    ):
        source = inspect.getsource(function)
        assert "torch.optim.SGD" not in source

    assert "torch.optim.LBFGS" not in inspect.getsource(runner.run_intervention)
    assert "torch.optim.LBFGS" not in inspect.getsource(runner.run_local_scan_seed)
    assert "load_initialization_state(seed)" in inspect.getsource(runner.run_variance_robustness)


def test_full_state_support_audit_and_event_precedence() -> None:
    runner = load_runner()
    n_states = 2048
    split = SimpleNamespace(s=torch.zeros(n_states, runner.P.state_dim, device=runner.DEVICE))

    class DummyActor(torch.nn.Module):
        def __init__(self, values: torch.Tensor) -> None:
            super().__init__()
            self.register_buffer("values", values)

        def forward(self, states: torch.Tensor):
            mu = torch.zeros(len(states), runner.P.action_dim, device=states.device)
            return mu, self.values[: len(states)]

    # Put the first contraction outside the old 1024-state prefix. The audit
    # must still see it because the current protocol covers every state.
    contraction_values = torch.zeros(n_states, device=runner.DEVICE)
    contraction_values[1500] = -(runner.P.log_sigma_event_boundary + 1.0)
    contraction = runner.support_diagnostics(DummyActor(contraction_values), split)
    assert contraction["log_sigma_min_all_states"] < -runner.P.log_sigma_event_boundary
    assert contraction["support_contraction_boundary"] is True
    assert runner.support_event_type(contraction) == "support_contraction"

    # A finite positive crossing is unexpected, not a scientific variance branch.
    positive_values = torch.zeros(n_states, device=runner.DEVICE)
    positive_values[1500] = runner.P.log_sigma_event_boundary + 1.0
    positive = runner.support_diagnostics(DummyActor(positive_values), split)
    assert positive["unexpected_support_expansion_boundary"] is True
    assert runner.support_event_type(positive) == "unexpected_support_expansion"

    # exp(log_sigma) overflow is a numerical output failure and takes precedence
    # over the finite positive-boundary label.
    overflow_values = torch.zeros(n_states, device=runner.DEVICE)
    overflow_values[1500] = 1000.0
    overflow = runner.support_diagnostics(DummyActor(overflow_values), split)
    assert overflow["log_sigma_output_finite_all_states"] is True
    assert overflow["sigma_output_finite_all_states"] is False
    assert runner.support_event_type(overflow) == "nonfinite_sigma_output"


def test_registry_and_handoff_register_adam_reruns_as_not_run() -> None:
    registry = yaml.safe_load((REPO_ROOT / "experiments" / "registry.yaml").read_text())
    experiments = {row["id"]: row for row in registry["experiments"]}

    e3 = experiments["C-U1-E3-ADAM-RERUN"]
    e4 = experiments["C-U1-E4-ADAM-RERUN"]
    assert e3["status"] == "not_run"
    assert e4["status"] == "not_run"
    assert e3["optimizer"]["name"] == "Adam"
    assert e4["optimizer"]["name"] == "Adam"
    assert e4["depends_on_delivered_experiment"] == "C-U1-E3-ADAM-RERUN"
    assert e3["data"]["terminology"] == "held_out_context_generalization"

    handoff = (REPO_ROOT / "docs" / "handoff.md").read_text()
    assert "v30（Gaussian 二次临界界、C-U1 共享实现与统一 Adam 对齐版）" in handoff
    assert "v29 增量记录：C-U1 E3/E4 统一 Adam 与方差坍缩口径修正" in handoff
    assert "C-U1-E3-ADAM-RERUN" in handoff
    assert "C-U1-E4-ADAM-RERUN" in handoff
    assert "方差爆炸" in handoff  # only in the explicit prohibition/correction record
    assert "不得写成“方差爆炸”" in handoff
