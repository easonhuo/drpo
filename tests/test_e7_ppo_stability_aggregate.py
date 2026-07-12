from __future__ import annotations

import csv
import json
from pathlib import Path

from drpo.e7_ppo_stability_aggregate import aggregate


DATASETS = (
    "hopper-medium-expert-v2",
    "walker2d-medium-v2",
    "walker2d-medium-replay-v2",
)
SEEDS = (200, 201, 202, 203)
CONTROLS = (
    ("positive_only", None),
    ("exponential", 0.5),
    ("exponential", 1.0),
    ("exponential", 1.5),
)
MODES = ("a2c", "ppo_clip")


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def _diagnostic_record() -> dict:
    return {
        "status": "complete",
        "update": 1_000_000,
        "old_policy_refresh_count": 249_999,
        "pre_update": {
            "samples": 1024,
            "ratio_mean": 1.01,
            "ratio_min": 0.70,
            "ratio_max": 1.35,
            "abs_log_ratio_mean": 0.05,
            "ratio_outside_fraction": 0.25,
            "objective_clip_fraction": 0.20,
            "positive_samples": 512,
            "positive_objective_clip_fraction": 0.15,
            "negative_samples": 512,
            "negative_objective_clip_fraction": 0.25,
        },
        "actor_gradient_norm": 2.0,
        "actor_parameter_update_norm": 0.02,
        "actor_relative_parameter_update_norm": 0.001,
        "sampled_post_update": {
            "ratio_to_old_outside_fraction": 0.30,
            "single_step_ratio_max": 1.10,
        },
    }


def test_aggregate_requires_and_summarizes_complete_96_branch_matrix(
    tmp_path: Path,
) -> None:
    branches = tmp_path / "branches"
    branch_count = 0
    for dataset_index, dataset in enumerate(DATASETS):
        for seed in SEEDS:
            for method, coefficient in CONTROLS:
                for mode in MODES:
                    branch_id = (
                        f"{dataset}__seed{seed}__{method}__{coefficient}__{mode}"
                    )
                    branch_dir = branches / branch_id
                    branch_dir.mkdir(parents=True)
                    _write_json(branch_dir / "COMPLETED.json", {"return_code": 0})
                    control = {
                        "method": method,
                        "negative_scale": 0.0 if method == "positive_only" else 1.0,
                        "canonical_alpha": 0.11,
                        "reference_distance": 2.0,
                        "exponential_coefficient": (
                            0.5 if coefficient is None else coefficient
                        ),
                    }
                    _write_json(
                        branch_dir / "branch_config.json",
                        {
                            "branch_id": branch_id,
                            "dataset_id": dataset,
                            "seed": seed,
                            "template_values": {"actor_update_mode": mode},
                            "negative_control": control,
                        },
                    )
                    final = 60.0 + dataset_index + (seed - 200) + (
                        1.0 if mode == "ppo_clip" else 0.0
                    )
                    _write_json(
                        branch_dir / "trainer_output" / "run_summary.json",
                        {
                            "history": {
                                "steps": [50_000, 1_000_000],
                                "score": [final + 5.0, final],
                            }
                        },
                    )
                    if mode == "ppo_clip":
                        (branch_dir / "ppo_diagnostics.jsonl").write_text(
                            json.dumps(_diagnostic_record()) + "\n"
                        )
                    branch_count += 1

    assert branch_count == 96
    result = aggregate(tmp_path)
    assert result["summary"]["branch_count"] == 96
    assert result["summary"]["ppo_branch_count"] == 48
    assert result["terminal_audit"]["status"] == "pass"
    assert result["terminal_audit"]["support_or_variance_boundary_events"] is None
    assert len(result["summary"]["groups"]) == 24
    assert len(result["summary"]["paired_ppo_vs_a2c"]) == 12
    assert all(
        row["ppo_minus_a2c_final_mean"] == 1.0
        for row in result["summary"]["paired_ppo_vs_a2c"]
    )

    with (tmp_path / "aggregate" / "per_branch_summary.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 96
