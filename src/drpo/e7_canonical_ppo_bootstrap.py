"""Run the unchanged canonical E7 trainer with A2C or PPO actor injection."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import runpy
import sys
from pathlib import Path
from typing import Any

from drpo.e7_canonical_injection import (
    CanonicalContract,
    NegativeControl,
    canonical_environment_manifest,
    load_verified_canonical_module,
    patch_canonical_module,
)
from drpo.e7_canonical_ppo_injection import (
    PPOActorControl,
    patch_canonical_module_ppo,
)


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify the canonical D4RL source, inject the registered negative "
            "control and actor update, then run the original trainer unchanged."
        )
    )
    parser.add_argument("--contract", required=True)
    parser.add_argument("--branch-config", required=True)
    parser.add_argument("--branch-manifest", required=True)
    parser.add_argument(
        "trainer_args",
        nargs=argparse.REMAINDER,
        help="arguments after -- are passed to the canonical trainer",
    )
    return parser


def _ppo_paths(branch_manifest: Path) -> tuple[Path, Path]:
    branch_dir = branch_manifest.parent
    return (
        branch_dir / "ppo_diagnostics.jsonl",
        branch_dir / "PPO_DIAGNOSTICS_LATEST.json",
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    contract = CanonicalContract.load(args.contract)
    branch_config_path = Path(args.branch_config).expanduser().resolve()
    branch = json.loads(branch_config_path.read_text())
    if str(branch.get("branch_kind")) != "injected":
        raise ValueError("PPO stability bootstrap only supports injected branches")

    template_values = {
        str(key): str(value)
        for key, value in branch.get("template_values", {}).items()
    }
    actor_update_mode = template_values.get("actor_update_mode")
    if actor_update_mode not in {"a2c", "ppo_clip"}:
        raise ValueError(
            "template_values.actor_update_mode must be 'a2c' or 'ppo_clip'"
        )

    control = NegativeControl.from_mapping(branch["negative_control"])
    if control.canonical_alpha != contract.expected_canonical_alpha:
        raise RuntimeError(
            "branch canonical_alpha does not match the source contract: "
            f"{control.canonical_alpha} != {contract.expected_canonical_alpha}"
        )

    module, source_checks = load_verified_canonical_module(contract)
    branch_manifest_path = Path(args.branch_manifest).expanduser().resolve()
    ppo_payload: dict[str, Any] | None = None
    diagnostics_jsonl: Path | None = None
    diagnostics_latest: Path | None = None

    if actor_update_mode == "a2c":
        patch_canonical_module(module, contract, control)
    else:
        ppo_control = PPOActorControl.from_mapping(
            {
                "clip_epsilon": template_values["clip_epsilon"],
                "updates_per_old_policy": template_values[
                    "updates_per_old_policy"
                ],
                "diagnostics_interval": template_values["diagnostics_interval"],
                "total_steps": template_values["steps"],
            }
        )
        diagnostics_jsonl, diagnostics_latest = _ppo_paths(branch_manifest_path)
        diagnostics_jsonl.unlink(missing_ok=True)
        diagnostics_latest.unlink(missing_ok=True)
        patch_canonical_module_ppo(
            module,
            contract.target_class,
            negative_control=control,
            ppo_control=ppo_control,
            return_mode=contract.return_mode,
            diagnostics_jsonl=diagnostics_jsonl,
            diagnostics_latest=diagnostics_latest,
        )
        ppo_payload = dataclasses.asdict(ppo_control)

    trainer_args = list(args.trainer_args)
    if trainer_args and trainer_args[0] == "--":
        trainer_args = trainer_args[1:]
    manifest: dict[str, Any] = {
        "status": "started",
        "branch": branch,
        "source_checks": source_checks,
        "negative_control": branch["negative_control"],
        "actor_update_mode": actor_update_mode,
        "ppo_control": ppo_payload,
        "trainer_path": str(contract.trainer_path),
        "trainer_args": trainer_args,
        "environment": canonical_environment_manifest(),
    }
    atomic_write_json(branch_manifest_path, manifest)

    old_argv = sys.argv[:]
    old_cwd = Path.cwd()
    try:
        os.chdir(contract.source_root)
        sys.argv = [str(contract.trainer_path), *trainer_args]
        try:
            runpy.run_path(str(contract.trainer_path), run_name="__main__")
        except SystemExit as exc:
            if exc.code not in (None, 0):
                raise

        if actor_update_mode == "ppo_clip":
            assert diagnostics_jsonl is not None
            assert diagnostics_latest is not None
            if not diagnostics_jsonl.is_file() or not diagnostics_latest.is_file():
                raise RuntimeError("PPO branch completed without diagnostics outputs")
            latest = json.loads(diagnostics_latest.read_text())
            expected_steps = int(template_values["steps"])
            if latest.get("status") != "complete":
                raise RuntimeError("PPO diagnostics did not reach complete status")
            if int(latest.get("update", -1)) != expected_steps:
                raise RuntimeError(
                    "PPO diagnostics update count mismatch: "
                    f"{latest.get('update')} != {expected_steps}"
                )
            manifest["ppo_diagnostics"] = {
                "jsonl": str(diagnostics_jsonl),
                "latest": str(diagnostics_latest),
                "final": latest,
            }
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["error_type"] = type(exc).__name__
        manifest["error"] = str(exc)
        atomic_write_json(branch_manifest_path, manifest)
        raise
    finally:
        sys.argv = old_argv
        os.chdir(old_cwd)

    manifest["status"] = "completed"
    atomic_write_json(branch_manifest_path, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
