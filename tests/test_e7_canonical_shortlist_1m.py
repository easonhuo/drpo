from __future__ import annotations

import math
from pathlib import Path

from drpo.e7_canonical_injection import CanonicalContract
from drpo.e7_canonical_sweep import build_branches, expand_injected_controls, load_grid


CONFIG_PATH = Path("configs/e7_canonical_two_dataset_shortlist_1m_v1.json")
LAUNCHER_PATH = Path("scripts/run_e7_canonical_two_dataset_shortlist_1m.sh")


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
        "trainer_argv_template": [],
        "passthrough_variants": [
            {
                "id": "original_exp_rank_mr",
                "template_values": {},
            }
        ],
    }


def test_shortlist_has_seven_fixed_methods_and_matched_global_control() -> None:
    grid, _ = load_grid(CONFIG_PATH)
    controls = expand_injected_controls(grid)

    assert len(controls) == grid["branch_count_per_dataset_seed"] == 6
    by_identity = {
        (control.method, control.negative_scale): control for control in controls
    }

    assert by_identity[("positive_only", 0.0)].effective_alpha == 0.0
    assert math.isclose(
        by_identity[("canonical_signed", 1.0)].effective_alpha,
        0.11,
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    assert math.isclose(
        by_identity[("global", 0.1)].effective_alpha,
        0.011,
        rel_tol=0.0,
        abs_tol=1e-12,
    )

    for method in ("reciprocal_linear", "reciprocal_quadratic", "exponential"):
        assert math.isclose(
            by_identity[(method, 0.1)].effective_alpha,
            0.011,
            rel_tol=0.0,
            abs_tol=1e-12,
        )

    aliases = grid["reporting_aliases"]
    assert aliases["canonical_signed__scale1"] == "global_neg_0p11"
    assert aliases["global__scale0p1"] == "global_neg_0p011"


def test_shortlist_expands_to_two_datasets_four_seeds_and_56_parallel_jobs(
    tmp_path: Path,
) -> None:
    grid, _ = load_grid(CONFIG_PATH)
    branches = build_branches(_contract(tmp_path), _run_spec(), grid)

    assert grid["expected_total_branches"] == 56
    assert len(branches) == 2 * 4 * 7 == 56
    assert sum(branch.branch_kind == "passthrough" for branch in branches) == 8
    assert sum(branch.branch_kind == "injected" for branch in branches) == 48
    assert len({branch.branch_id for branch in branches}) == 56


def test_shortlist_launcher_freezes_1m_budget_and_parallel_execution() -> None:
    text = LAUNCHER_PATH.read_text()

    assert "--grid configs/e7_canonical_two_dataset_shortlist_1m_v1.json" in text
    assert "--steps 1000000" in text
    assert "--eval-interval 50000" in text
    assert "--eval-episodes 10" in text
    assert "--ckpt-interval 50000" in text
    assert 'MAX_WORKERS="${E7_MAX_WORKERS:-40}"' in text
    assert '(( MAX_WORKERS < 2 ))' in text
    assert '--max-workers "$MAX_WORKERS"' in text
