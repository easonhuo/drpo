"""Thin 192-branch wrapper over the existing canonical E7 sweep."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Sequence

from drpo import e7_canonical_sweep as base
from drpo import e7_squared_exp_night as night
from drpo.e7_sqexp_gae_audit import terminal_audit
from drpo.e7_sqexp_gae_matrix import build_branches
from drpo.e7_sqexp_gae_protocol import (
    EXPERIMENT_ID,
    RUNNER_VERSION,
    SCIENTIFIC_STATUS,
    load_grid,
    load_run_spec,
)
from drpo.e7_squared_exp_kernel import FORMULA

PREPARED_ROOT_ENV = "E7_SQEXP_GAE_PREPARED_ROOT"
_ORIGINAL_WRITE_PLAN = base.write_plan


def _prepared_root(work_dir: Path) -> Path:
    configured = os.environ.get(PREPARED_ROOT_ENV)
    return Path(configured).expanduser().resolve() if configured else work_dir / "prepared"


def branch_command(
    *,
    contract_path: Path,
    contract: base.CanonicalContract,
    branch: base.Branch,
    branch_dir: Path,
    trainer_argv_template: Sequence[str],
) -> tuple[list[str], dict[str, Any]]:
    values = branch.template_values
    work_dir = branch_dir.parent.parent
    manifest = (
        _prepared_root(work_dir)
        / branch.dataset.id
        / f"seed{branch.seed}"
        / "ADVANTAGE_MANIFEST.json"
    )
    if not manifest.is_file():
        raise FileNotFoundError(f"missing preserved shared-critic artifact: {manifest}")
    context = {
        "canonical_root": str(contract.source_root),
        "dataset_id": branch.dataset.id,
        "dataset_path": str(Path(branch.dataset.path).expanduser().resolve()),
        "dataset_sha256": branch.dataset.sha256,
        "seed": branch.seed,
        "output_dir": str(branch_dir / "trainer_output"),
        "branch_id": branch.branch_id,
        "variant": "iqlv_exp_rank",
        **values,
    }
    trainer_args = [
        base._format_value(str(item), context)  # noqa: SLF001
        for item in trainer_argv_template
    ]
    branch_config = {
        "experiment_id": EXPERIMENT_ID,
        "branch_id": branch.branch_id,
        "branch_kind": branch.branch_kind,
        "canonical_root": context["canonical_root"],
        "dataset_id": branch.dataset.id,
        "dataset_path": context["dataset_path"],
        "dataset_sha256": branch.dataset.sha256,
        "seed": branch.seed,
        "template_values": values,
        "advantage_manifest": str(manifest),
        "weight_control": {
            "method": values["weight_method"],
            "weight_at_zero": float(values["weight_at_zero"]),
            "exp_coefficient": float(values["exp_coefficient"]),
            "reference_distance": night.REFERENCE_DISTANCE,
            "formula": FORMULA,
        },
    }
    config_path = branch_dir / "branch_config.json"
    base.atomic_write_json(config_path, branch_config)
    return (
        [
            sys.executable,
            "-m",
            "drpo.e7_sqexp_gae_minimal",
            "--contract",
            str(contract_path),
            "--branch-config",
            str(config_path),
            "--branch-manifest",
            str(branch_dir / "branch_manifest.json"),
            "--",
            *trainer_args,
        ],
        branch_config,
    )


def write_plan(**kwargs: Any) -> dict[str, Any]:
    work_dir = Path(kwargs["work_dir"]).expanduser().resolve()
    prepared_root = _prepared_root(work_dir)
    pairs = {(branch.dataset.id, branch.seed) for branch in kwargs["branches"]}
    missing = [
        str(prepared_root / dataset / f"seed{seed}" / "ADVANTAGE_MANIFEST.json")
        for dataset, seed in sorted(pairs)
        if not (
            prepared_root / dataset / f"seed{seed}" / "ADVANTAGE_MANIFEST.json"
        ).is_file()
    ]
    if missing:
        raise FileNotFoundError("missing preserved prepared artifacts: " + ", ".join(missing))
    return _ORIGINAL_WRITE_PLAN(**kwargs)


def main(argv: list[str] | None = None) -> int:
    previous = (
        base.EXPERIMENT_ID,
        base.SCIENTIFIC_STATUS,
        base.RUNNER_VERSION,
        base.load_grid,
        base.load_run_spec,
        base.build_branches,
        base.branch_command,
        base.write_plan,
    )
    base.EXPERIMENT_ID = EXPERIMENT_ID
    base.SCIENTIFIC_STATUS = SCIENTIFIC_STATUS
    base.RUNNER_VERSION = RUNNER_VERSION
    base.load_grid = load_grid
    base.load_run_spec = load_run_spec
    base.build_branches = build_branches
    base.branch_command = branch_command
    base.write_plan = write_plan
    delegated = list(sys.argv[1:] if argv is None else argv)
    try:
        result = int(base.main(delegated))
        if delegated and delegated[0] == "run":
            work_index = delegated.index("--work-dir")
            terminal_audit(Path(delegated[work_index + 1]).expanduser().resolve())
        return result
    finally:
        (
            base.EXPERIMENT_ID,
            base.SCIENTIFIC_STATUS,
            base.RUNNER_VERSION,
            base.load_grid,
            base.load_run_spec,
            base.build_branches,
            base.branch_command,
            base.write_plan,
        ) = previous


if __name__ == "__main__":
    raise SystemExit(main())
