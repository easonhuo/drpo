from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

from drpo.e7_canonical_sweep import build_branches, expand_injected_controls, load_grid
from drpo.e7_canonical_two_dataset import default_canonical_root, write_run_spec
from drpo.e7_canonical_injection import CanonicalContract


def _args(tmp_path: Path, *, profile: str = "reproduce") -> Namespace:
    replay = tmp_path / "hopper-medium-replay-v2.hdf5"
    expert = tmp_path / "hopper-medium-expert-v2.hdf5"
    replay.write_bytes(b"replay")
    expert.write_bytes(b"expert")
    return Namespace(
        profile=profile,
        data_dir=None,
        hopper_medium_replay_hdf5=str(replay),
        hopper_medium_expert_hdf5=str(expert),
        hopper_medium_replay_sha256=None,
        hopper_medium_expert_sha256=None,
        seeds=[200, 201, 202, 203],
        alpha=0.11,
        tau=0.5,
        temp=5.0,
        steps=None,
        batch=256,
        lr=3e-4,
        eval_interval=None,
        eval_episodes=None,
        eval_max_steps=None,
        ckpt_interval=None,
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
    assert payload["profile"] == "reproduce"
    assert payload["datasets"][0]["id"] == "hopper-medium-replay-v2"
    assert payload["datasets"][1]["id"] == "hopper-medium-expert-v2"
    assert payload["seeds"] == [200, 201, 202, 203]
    template = payload["trainer_argv_template"]
    assert template[template.index("--variant") + 1] == "iqlv_exp_rank"
    assert template[template.index("--alpha") + 1] == "0.11"
    assert template[template.index("--tau") + 1] == "0.5"
    assert template[template.index("--temp") + 1] == "5.0"
    assert template[template.index("--steps") + 1] == "1000000"
    assert {row["id"] for row in payload["passthrough_variants"]} == {
        "original_exp_rank_mr"
    }
    reloaded = json.loads(output.read_text())
    assert reloaded == payload


def test_smoke_profile_shortens_only_liveness_budget(tmp_path: Path) -> None:
    payload = write_run_spec(_args(tmp_path, profile="smoke"), tmp_path / "run_spec.json")
    template = payload["trainer_argv_template"]
    assert payload["run_kind"] == "smoke"
    assert template[template.index("--steps") + 1] == "20000"
    assert template[template.index("--eval_interval") + 1] == "10000"
    assert template[template.index("--eval_episodes") + 1] == "1"


def test_reproduce_grid_is_original_passthrough_only(tmp_path: Path) -> None:
    run_spec = write_run_spec(_args(tmp_path), tmp_path / "run_spec.json")
    grid, _ = load_grid("configs/e7_canonical_two_dataset_grid_reproduce_v2.json")
    controls = expand_injected_controls(grid)
    assert len(controls) == grid["branch_count_per_dataset_seed"] == 0
    branches = build_branches(_contract(tmp_path), run_spec, grid)
    assert len(branches) == 2 * 4
    assert all(branch.branch_kind == "passthrough" for branch in branches)
    assert all("original_exp_rank_mr" in branch.branch_id for branch in branches)


def test_small_taper_grid_branch_count_includes_passthrough(tmp_path: Path) -> None:
    run_spec = write_run_spec(
        _args(tmp_path, profile="taper-pilot"), tmp_path / "run_spec.json"
    )
    grid, _ = load_grid("configs/e7_canonical_two_dataset_grid_small_taper_v2.json")
    controls = expand_injected_controls(grid)
    assert len(controls) == grid["branch_count_per_dataset_seed"] == 6
    branches = build_branches(_contract(tmp_path), run_spec, grid)
    assert len(branches) == 2 * 4 * (6 + 1)
    assert sum(branch.branch_kind == "passthrough" for branch in branches) == 8
    assert any("exponential" in branch.branch_id for branch in branches)


def test_vendored_canonical_source_is_default_and_present() -> None:
    root = default_canonical_root()
    assert root.is_dir()
    assert (root / "agents.py").is_file()
    assert (root / "train_sna2c_variant.py").is_file()
    assert (root / "d4rl_common" / "train_loop.py").is_file()
    assert (root / "refs" / "d4rl_infos.py").is_file()
