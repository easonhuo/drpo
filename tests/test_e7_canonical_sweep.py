from __future__ import annotations

from pathlib import Path

from drpo.e7_canonical_injection import CanonicalContract
from drpo.e7_canonical_sweep import (
    build_branches,
    expand_injected_controls,
    load_grid,
)


def grid_payload() -> dict:
    return {
        "experiment_id": "EXT-H-E7-BENCH-01",
        "run_kind": "pilot",
        "canonical_alpha": 0.11,
        "reference_distance": 2.0,
        "coefficients": {
            "reciprocal_linear": 0.4362580032734791,
            "reciprocal_quadratic": 0.5520268617673281,
            "exponential": 0.374162511054291,
        },
        "negative_scale_grid": {
            "global": [0.0001, 0.3],
            "reciprocal_linear": [0.001, 1.0],
            "reciprocal_quadratic": [0.001, 1.0],
            "exponential": [0.001, 1.0],
        },
    }


def contract(tmp_path: Path) -> CanonicalContract:
    return CanonicalContract.from_mapping(
        {
            "contract_version": "e7-canonical-contract-v1",
            "canonical_source_root": str(tmp_path),
            "python_tree_sha256": "0" * 64,
            "agents_relpath": "agents.py",
            "agents_sha256": "1" * 64,
            "trainer_relpath": "trainer.py",
            "trainer_sha256": "2" * 64,
            "module_name": "agents",
            "target_class": "SNA2C_IQLV_DistAgent",
            "agent_flavor": "signed_td_v_v1",
            "expected_canonical_alpha": 0.11,
        }
    )


def test_grid_has_anchors_small_scales_and_no_quartic() -> None:
    controls = expand_injected_controls(grid_payload())
    identities = {(item.method, item.negative_scale) for item in controls}
    assert ("positive_only", 0.0) in identities
    assert ("canonical_signed", 1.0) in identities
    assert ("global", 0.0001) in identities
    assert all("quartic" not in method for method, _ in identities)


def test_build_branches_parallel_unit_is_dataset_seed_method_scale(tmp_path: Path) -> None:
    dataset_file = tmp_path / "dataset.hdf5"
    dataset_file.write_bytes(b"fixture")
    import hashlib

    digest = hashlib.sha256(b"fixture").hexdigest()
    run_spec = {
        "run_kind": "pilot",
        "datasets": [{"id": "hopper", "path": str(dataset_file), "sha256": digest}],
        "seeds": [200, 201],
        "trainer_argv_template": [],
        "injected_template_values": {"agent_selector": "SNA2C"},
        "passthrough_variants": [
            {"id": "iql", "template_values": {"agent_selector": "IQL"}}
        ],
    }
    controls = expand_injected_controls(grid_payload())
    branches = build_branches(contract(tmp_path), run_spec, grid_payload())
    assert len(branches) == 2 * (len(controls) + 1)
    assert len({branch.branch_id for branch in branches}) == len(branches)
    assert sum(branch.branch_kind == "passthrough" for branch in branches) == 2


def test_repository_grid_declares_31_injected_branches() -> None:
    path = Path("configs/e7_canonical_weight_grid_v1.json")
    raw, _ = load_grid(path)
    controls = expand_injected_controls(raw)
    assert len(controls) == 31
    assert raw["branch_count_per_dataset_seed"] == 31
