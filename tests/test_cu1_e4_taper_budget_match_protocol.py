from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "drpo"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import cu1_taper_budget_match_formal as budget  # noqa: E402
import drpo_cu1_e1_e4_oneclick as experiment  # noqa: E402


def _entry(experiment_id: str) -> dict:
    registry = yaml.safe_load((ROOT / "experiments" / "registry.yaml").read_text())
    return next(row for row in registry["experiments"] if row["id"] == experiment_id)


def test_budget_protocol_freezes_primary_coordinate_and_seed_firewall() -> None:
    protocol = budget.PROTOCOL
    assert protocol.development_seeds == tuple(range(5))
    assert protocol.formal_seeds == tuple(range(110, 130))
    assert protocol.target_retention == 0.75
    assert protocol.maximum_steps == 8000
    assert protocol.matched_methods == (
        "reciprocal_linear",
        "reciprocal_quadratic",
        "exponential",
        "squared_distance_exponential",
        "global_stepwise_scale",
    )
    assert budget.method_names(protocol) == (
        "positive_only",
        "unweighted_boundary",
        *protocol.matched_methods,
    )


def test_gradient_rescaling_matches_target_without_changing_direction() -> None:
    torch.manual_seed(0)
    actor = experiment.GaussianActor().to(experiment.DEVICE)
    params = list(actor.parameters())
    positive = tuple(torch.randn_like(parameter) for parameter in params)
    negative = tuple(torch.randn_like(parameter) for parameter in params)
    raw_norm = float(experiment.norm_tuple(negative).item())
    target = 0.37 * raw_norm
    scale = target / raw_norm
    budget._set_combined_gradient(actor, positive, negative, scale)
    realized_negative = experiment.scale_tuple(negative, scale)
    assert float(experiment.norm_tuple(realized_negative).item()) == pytest.approx(
        target, rel=1e-6, abs=1e-8
    )
    for original, scaled in zip(negative, realized_negative):
        assert torch.dot(original.reshape(-1), scaled.reshape(-1)).item() > 0


def test_registry_activates_budget_only_and_preserves_later_gates() -> None:
    row = _entry(budget.EXPERIMENT_ID)
    assert row["execution_gate"]["state"] == "ready"
    assert row["implementation_state"] == "implemented"
    assert row["formal_execution"]["activation_state"] == "active"
    assert row["formal_execution"]["entrypoint"] == (
        "src/drpo/cu1_taper_budget_match_formal.py"
    )
    contract = row["budget_contract"]
    assert contract["primary_mode"] == "stepwise_raw_negative_gradient_l2_before_Adam"
    assert contract["reference_method"] == "reciprocal_linear_at_near_retention_0_75"
    assert contract["Adam_parameter_update_norm_matched"] is False
    assert contract["Adam_parameter_update_norm_logged"] is True
    assert row["protocol"]["formal_paired_seeds"] == list(range(110, 130))

    convergence = _entry("C-U1-E4-TAPER-CONV-01")
    confirmation = _entry("C-U1-E4-TAPER-CONFIRM-01")
    assert convergence["execution_gate"]["state"] == "blocked"
    assert convergence["terminal_contract"]["maximum_total_steps"] == 32000
    assert confirmation["execution_gate"]["state"] == "blocked"
    assert confirmation["confirmation_contract"]["exact_untouched_seeds"] == list(
        range(130, 150)
    )


def test_budget_engineering_smoke_writes_complete_non_scientific_artifacts(
    tmp_path: Path,
) -> None:
    output = tmp_path / "smoke"
    env = os.environ.copy()
    env.update({"PYTHONPATH": str(SRC), "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
    subprocess.run(
        [
            sys.executable,
            str(SRC / "cu1_taper_budget_match_formal.py"),
            "--output-dir",
            str(output),
            "--base-commit",
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "--smoke",
        ],
        cwd=ROOT,
        env=env,
        check=True,
        timeout=180,
    )
    complete = json.loads((output / "RUN_COMPLETE.json").read_text())
    audit = json.loads((output / "terminal_audit.json").read_text())
    budget_audit = json.loads((output / "budget_audit.json").read_text())
    freeze = json.loads((output / "formal_protocol_freeze.json").read_text())
    assert complete["formal_run_started"] is False
    assert complete["result_status"] == "engineering_smoke"
    assert complete["coverage_checks_passed"] is True
    assert audit["scientific_status"] == "not run / 尚未运行"
    assert budget_audit["maximum_relative_error"] <= 1e-6
    assert budget_audit["Adam_parameter_update_norm_matched"] is False
    assert freeze["primary_budget_coordinate"] == (
        "stepwise_raw_negative_gradient_l2_before_Adam"
    )
    assert freeze["OOD_claim_allowed"] is False
