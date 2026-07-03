"""Execute an unchanged canonical D4RL trainer with a verified class injection."""

from __future__ import annotations

import argparse
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


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify the old D4RL source tree, optionally inject one negative-"
            "control branch, and run its original trainer unchanged."
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


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    contract = CanonicalContract.load(args.contract)
    branch = json.loads(Path(args.branch_config).read_text())
    branch_kind = str(branch["branch_kind"])
    if branch_kind not in {"injected", "passthrough"}:
        raise ValueError(f"unsupported branch_kind={branch_kind!r}")

    module, source_checks = load_verified_canonical_module(contract)
    control_payload: dict[str, Any] | None = None
    if branch_kind == "injected":
        control = NegativeControl.from_mapping(branch["negative_control"])
        if control.canonical_alpha != contract.expected_canonical_alpha:
            raise RuntimeError(
                "branch canonical_alpha does not match the source contract: "
                f"{control.canonical_alpha} != {contract.expected_canonical_alpha}"
            )
        patch_canonical_module(module, contract, control)
        control_payload = branch["negative_control"]

    trainer_args = list(args.trainer_args)
    if trainer_args and trainer_args[0] == "--":
        trainer_args = trainer_args[1:]
    manifest_path = Path(args.branch_manifest).expanduser().resolve()
    manifest = {
        "status": "started",
        "branch": branch,
        "source_checks": source_checks,
        "negative_control": control_payload,
        "trainer_path": str(contract.trainer_path),
        "trainer_args": trainer_args,
        "environment": canonical_environment_manifest(),
    }
    atomic_write_json(manifest_path, manifest)

    old_argv = sys.argv[:]
    old_cwd = Path.cwd()
    try:
        os.chdir(contract.source_root)
        sys.argv = [str(contract.trainer_path), *trainer_args]
        try:
            runpy.run_path(str(contract.trainer_path), run_name="__main__")
        except SystemExit as exc:
            # Many unchanged training scripts end with ``raise SystemExit(main())``.
            # A zero exit is a normal completion, not a failed branch.
            if exc.code not in (None, 0):
                raise
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["error_type"] = type(exc).__name__
        manifest["error"] = str(exc)
        atomic_write_json(manifest_path, manifest)
        raise
    finally:
        sys.argv = old_argv
        os.chdir(old_cwd)

    manifest["status"] = "completed"
    atomic_write_json(manifest_path, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
