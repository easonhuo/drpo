from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from drpo import e7_squared_exp_night as night


GRID = Path("configs/e7_squared_exp_night_v1.json")


def test_branch_command_uses_supported_canonical_variant(tmp_path: Path) -> None:
    grid, _ = night.load_grid(GRID)
    digest = "0" * 64
    run_spec = {
        "datasets": [
            {
                "id": dataset,
                "path": f"/tmp/{dataset}.hdf5",
                "sha256": digest,
            }
            for dataset in night.EXPECTED_DATASETS
        ],
        "seeds": list(night.EXPECTED_SEEDS),
    }
    contract = SimpleNamespace(
        expected_canonical_alpha=0.11,
        source_root=tmp_path,
    )
    branch = night.build_branches(contract, run_spec, grid)[0]
    command, _ = night.branch_command(
        contract_path=tmp_path / "contract.json",
        contract=contract,
        branch=branch,
        branch_dir=tmp_path / "branch",
        trainer_argv_template=[
            "--dataset",
            "{dataset_path}",
            "--variant",
            "{variant}",
            "--steps",
            "{steps}",
            "--out_dir",
            "{output_dir}",
        ],
    )
    variant_index = command.index("--variant")
    assert command[variant_index + 1] == "iqlv_exp_rank"
    assert "iqlv_squared_exp_night" not in command
