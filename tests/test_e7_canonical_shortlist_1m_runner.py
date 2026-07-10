from __future__ import annotations

import json
from pathlib import Path

import pytest

from drpo.e7_canonical_injection import CanonicalContract
from drpo.e7_canonical_shortlist_audit import audit_branch
from drpo.e7_canonical_shortlist_protocol import (
    EXPECTED_REPORTING_ALIASES,
    apply_reporting_aliases,
    validate_fixed_protocol,
)
from drpo.e7_canonical_sweep import Branch, DatasetSpec, build_branches, load_grid


CONFIG_PATH = Path("configs/e7_canonical_two_dataset_shortlist_1m_v1.json")


def _contract(tmp_path: Path) -> CanonicalContract:
    return CanonicalContract.from_mapping(
        {
            "contract_version": "e7-canonical-contract-v1",
            "canonical_source_root": str(tmp_path),
            "python_tree_sha256": "0" * 64,
            "agents_relpath": "agents.py",
            "agents_sha256": "1" * 64,
            "trainer_relpath": "train_sna2c_variant.py",
            "trainer_sha256": "2" * 64,
            "module_name": "agents",
            "target_class": "SNA2C_IQLV_ExpRankAgent",
            "agent_flavor": "signed_td_v_v1",
            "expected_canonical_alpha": 0.11,
        }
    )


def _run_spec() -> dict[str, object]:
    return {
        "run_kind": "pilot",
        "profile": "taper-pilot",
        "datasets": [
            {
                "id": "hopper-medium-replay-v2",
                "path": "/tmp/hopper-medium-replay-v2.hdf5",
                "sha256": "a" * 64,
            },
            {
                "id": "hopper-medium-expert-v2",
                "path": "/tmp/hopper-medium-expert-v2.hdf5",
                "sha256": "b" * 64,
            },
        ],
        "seeds": [200, 201, 202, 203],
        "trainer_argv_template": [
            "--dataset",
            "{dataset_id}",
            "--hdf5",
            "{dataset_path}",
            "--variant",
            "iqlv_exp_rank",
            "--alpha",
            "0.11",
            "--tau",
            "0.5",
            "--temp",
            "5.0",
            "--steps",
            "1000000",
            "--batch",
            "256",
            "--lr",
            "0.0003",
            "--eval_interval",
            "50000",
            "--eval_episodes",
            "10",
            "--seed",
            "{seed}",
            "--out_dir",
            "{output_dir}",
            "--ckpt_dir",
            "{output_dir}/ckpts",
            "--ckpt_interval",
            "50000",
            "--last_pct",
            "0.1",
        ],
        "passthrough_variants": [
            {"id": "original_exp_rank_mr", "template_values": {}}
        ],
        "environment": {
            "OMP_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
            "OPENBLAS_NUM_THREADS": "2",
        },
    }


def test_fixed_protocol_validator_rejects_scientific_drift(tmp_path: Path) -> None:
    grid, _ = load_grid(CONFIG_PATH)
    run_spec = _run_spec()
    result = validate_fixed_protocol(_contract(tmp_path), run_spec, grid)
    assert result["status"] == "PASS"

    drifted = dict(run_spec)
    drifted["seeds"] = [200]
    with pytest.raises(ValueError, match="seeds changed"):
        validate_fixed_protocol(_contract(tmp_path), drifted, grid)


def test_reporting_aliases_are_consumed_in_branch_identity(tmp_path: Path) -> None:
    grid, _ = load_grid(CONFIG_PATH)
    generic = build_branches(_contract(tmp_path), _run_spec(), grid)
    remapped = apply_reporting_aliases(generic, EXPECTED_REPORTING_ALIASES)

    assert len(remapped) == 56
    suffixes = {branch.branch_id.split("__", 2)[2] for branch in remapped}
    assert suffixes == set(EXPECTED_REPORTING_ALIASES.values())
    assert all("scale" not in branch.branch_id for branch in remapped)


def _write_summary(branch_dir: Path, scores: list[float]) -> None:
    trainer_output = branch_dir / "trainer_output"
    trainer_output.mkdir(parents=True)
    steps = list(range(50_000, 1_000_001, 50_000))
    (trainer_output / "synthetic_summary.json").write_text(
        json.dumps({"history": {"steps": steps, "score": scores}})
    )
    (branch_dir / "COMPLETED.json").write_text("{}")


def test_terminal_audit_reports_late_window_without_claiming_convergence(
    tmp_path: Path,
) -> None:
    dataset = DatasetSpec("hopper-medium-replay-v2", "/tmp/data.hdf5", "a" * 64)
    branch = Branch(
        branch_id="hopper-medium-replay-v2__seed200__positive_only",
        branch_kind="injected",
        dataset=dataset,
        seed=200,
        template_values={},
        negative_control=None,
    )
    branch_dir = tmp_path / "branches" / branch.branch_id
    scores = [float(value) for value in range(1, 21)]
    _write_summary(branch_dir, scores)

    row = audit_branch(
        tmp_path,
        branch,
        [750_000, 800_000, 850_000, 900_000, 950_000, 1_000_000],
    )
    assert row["late_window_mean"] == pytest.approx(17.5)
    assert row["final_score"] == 20.0
    assert row["best_step"] == 1_000_000
    assert row["terminal_classification"] == "fixed_horizon_inconclusive"
    assert row["late_fraction_above_registered_threshold"] is None
    numerical = row["event_separation"]["nan_inf_numerical_failure"]
    assert numerical["status"] == "absent"


def test_terminal_audit_rejects_nonfinite_scores(tmp_path: Path) -> None:
    dataset = DatasetSpec("hopper-medium-replay-v2", "/tmp/data.hdf5", "a" * 64)
    branch = Branch(
        branch_id="hopper-medium-replay-v2__seed200__positive_only",
        branch_kind="injected",
        dataset=dataset,
        seed=200,
        template_values={},
        negative_control=None,
    )
    branch_dir = tmp_path / "branches" / branch.branch_id
    scores = [float(value) for value in range(1, 21)]
    scores[-1] = float("nan")
    _write_summary(branch_dir, scores)

    with pytest.raises(RuntimeError, match="non-finite evaluation score"):
        audit_branch(
            tmp_path,
            branch,
            [750_000, 800_000, 850_000, 900_000, 950_000, 1_000_000],
        )
