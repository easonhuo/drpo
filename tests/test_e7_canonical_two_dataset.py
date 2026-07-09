from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

from drpo.e7_canonical_sweep import build_branches, expand_injected_controls, load_grid
from drpo.e7_canonical_two_dataset import write_run_spec
from drpo.e7_canonical_injection import CanonicalContract


def _args(tmp_path: Path) -> Namespace:
    replay = tmp_path / "hopper-medium-replay-v2.hdf5"
    expert = tmp_path / "hopper-medium-expert-v2.hdf5"
    replay.write_bytes(b"replay")
    expert.write_bytes(b"expert")
    return Namespace(
        hopper_medium_replay_hdf5=str(replay),
        hopper_medium_expert_hdf5=str(expert),
        hopper_medium_replay_sha256=None,
        hopper_medium_expert_sha256=None,
        seeds=[200, 201],
        alpha=0.11,
        tau=0.5,
        temp=5.0,
        steps=1_000_000,
        batch=256,
        lr=3e-4,
        eval_interval=50_000,
        eval_episodes=10,
        eval_max_steps=None,
        ckpt_interval=50_000,
        last_pct=0.10,
        device=0,
        omp_threads=2,
    )


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


def test_two_dataset_run_spec_preserves_experank_mr_defaults(tmp_path: Path) -> None:
    output = tmp_path / "run_spec.json"
    payload = write_run_spec(_args(tmp_path), output)
    assert output.is_file()
    assert payload["datasets"][0]["id"] == "hopper-medium-replay-v2"
    assert payload["datasets"][1]["id"] == "hopper-medium-expert-v2"
    assert payload["seeds"] == [200, 201]
    template = payload["trainer_argv_template"]
    assert template[template.index("--variant") + 1] == "iqlv_exp_rank"
    assert template[template.index("--alpha") + 1] == "0.11"
    assert template[template.index("--tau") + 1] == "0.5"
    assert template[template.index("--temp") + 1] == "5.0"
    assert {row["id"] for row in payload["passthrough_variants"]} == {
        "original_exp_rank_mr"
    }
    reloaded = json.loads(output.read_text())
    assert reloaded == payload


def test_two_dataset_grid_branch_count_includes_passthrough(tmp_path: Path) -> None:
    run_spec = write_run_spec(_args(tmp_path), tmp_path / "run_spec.json")
    grid, _ = load_grid("configs/e7_canonical_two_dataset_taper_grid_v1.json")
    controls = expand_injected_controls(grid)
    assert len(controls) == grid["branch_count_per_dataset_seed"] == 14
    branches = build_branches(_contract(tmp_path), run_spec, grid)
    assert len(branches) == 2 * 2 * (14 + 1)
    assert sum(branch.branch_kind == "passthrough" for branch in branches) == 4
    assert any("original_exp_rank_mr" in branch.branch_id for branch in branches)
