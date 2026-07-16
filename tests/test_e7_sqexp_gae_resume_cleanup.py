from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from drpo import e7_canonical_sweep as base
from drpo import e7_sqexp_gae_contract as protocol


def test_branch_command_clears_stale_failed_marker_before_resume(
    tmp_path: Path,
) -> None:
    branch = base.Branch(
        branch_id="hopper-medium-expert-v2__seed200__td__positive_only__w0_0__a2c__steps1m",
        branch_kind="injected",
        dataset=base.DatasetSpec(
            id="hopper-medium-expert-v2",
            path="/tmp/hopper-medium-expert-v2.hdf5",
            sha256="0" * 64,
        ),
        seed=200,
        template_values={
            "steps": "1000000",
            "diagnostics_interval": "1000",
            "sampled_values_per_update": "16",
            "advantage_estimator": "td",
            "actor_update_mode": "a2c",
            "weight_method": "positive_only",
            "weight_at_zero": "0",
            "exp_coefficient": "0",
            "reference_distance": "2",
        },
        negative_control=None,
    )
    branch_dir = tmp_path / "work" / "branches" / branch.branch_id
    branch_dir.mkdir(parents=True)
    stale_failure = branch_dir / "FAILED.json"
    stale_failure.write_text('{"return_code": 2}')

    command, _ = protocol.branch_command(
        contract_path=tmp_path / "contract.json",
        contract=SimpleNamespace(source_root=tmp_path),
        branch=branch,
        branch_dir=branch_dir,
        trainer_argv_template=["--seed", "{seed}", "--steps", "{steps}"],
    )

    assert not stale_failure.exists()
    assert (branch_dir / "branch_config.json").is_file()
    assert "--seed" in command
    assert "200" in command
