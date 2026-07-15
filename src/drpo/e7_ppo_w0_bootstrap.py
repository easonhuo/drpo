"""Run canonical E7 PPO training from a direct-w(0) branch config.

The public branch contract contains only ``w(0)`` and the exponential coefficient
``c``. Conversion to the frozen canonical alpha representation is private and is
never persisted in the branch manifest or final diagnostics.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import runpy
import sys
from pathlib import Path
from typing import Any, Mapping

from drpo.e7_canonical_injection import (
    CanonicalContract,
    NegativeControl,
    canonical_environment_manifest,
    load_verified_canonical_module,
)
from drpo.e7_canonical_ppo_injection import PPOActorControl, patch_canonical_module_ppo

EXPERIMENT_ID = "EXT-H-E7-PPO-W0-EXP-GRID-01"
REFERENCE_DISTANCE = 2.0


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--branch-config", required=True)
    parser.add_argument("--branch-manifest", required=True)
    parser.add_argument("trainer_args", nargs=argparse.REMAINDER)
    return parser


def _ppo_paths(branch_manifest: Path) -> tuple[Path, Path]:
    branch_dir = branch_manifest.parent
    return (
        branch_dir / "ppo_diagnostics.jsonl",
        branch_dir / "PPO_DIAGNOSTICS_LATEST.json",
    )


def _validate_weight_control(raw: Mapping[str, Any]) -> dict[str, Any]:
    forbidden = {"negative_scale", "canonical_alpha", "effective_alpha"}
    present = sorted(forbidden & set(raw))
    if present:
        raise ValueError(
            "direct-w(0) branch forbids legacy scale/alpha fields: " + ", ".join(present)
        )
    method = str(raw.get("method"))
    w0 = float(raw.get("weight_at_zero"))
    coefficient = float(raw.get("exp_coefficient"))
    reference_distance = float(raw.get("reference_distance"))
    if method not in {"positive_only", "exponential"}:
        raise ValueError("weight_control.method must be positive_only or exponential")
    if not math.isfinite(w0) or not 0.0 <= w0 <= 1.0:
        raise ValueError("weight_at_zero must be finite and in [0, 1]")
    if not math.isfinite(coefficient) or coefficient < 0.0:
        raise ValueError("exp_coefficient must be finite and non-negative")
    if not math.isclose(reference_distance, REFERENCE_DISTANCE, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("reference_distance must remain the frozen value 2.0")
    if method == "positive_only" and (w0 != 0.0 or coefficient != 0.0):
        raise ValueError("positive_only requires w(0)=0 and c=0 in storage")
    if method == "exponential" and w0 <= 0.0:
        raise ValueError("exponential requires w(0)>0")
    return {
        "method": method,
        "weight_at_zero": w0,
        "exp_coefficient": coefficient,
        "reference_distance": reference_distance,
        "formula": "w(d)=w(0)*exp(-c*(d/2))",
    }


def _internal_control(
    public: Mapping[str, Any],
    *,
    canonical_alpha: float,
) -> NegativeControl:
    w0 = float(public["weight_at_zero"])
    method = str(public["method"])
    return NegativeControl(
        method=method,
        negative_scale=(0.0 if method == "positive_only" else w0 / canonical_alpha),
        canonical_alpha=canonical_alpha,
        reference_distance=REFERENCE_DISTANCE,
        exponential_coefficient=float(public["exp_coefficient"]),
    )


def _public_record(record: Mapping[str, Any], public: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(record)
    value.pop("negative_control", None)
    value["weight_control"] = dict(public)
    return value


def _sanitize_diagnostics(
    jsonl_path: Path,
    latest_path: Path,
    public: Mapping[str, Any],
) -> None:
    if jsonl_path.is_file():
        records = [
            _public_record(json.loads(line), public)
            for line in jsonl_path.read_text().splitlines()
            if line.strip()
        ]
        temporary = jsonl_path.with_suffix(jsonl_path.suffix + ".tmp")
        temporary.write_text(
            "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
        )
        os.replace(temporary, jsonl_path)
    if latest_path.is_file():
        latest = _public_record(json.loads(latest_path.read_text()), public)
        atomic_write_json(latest_path, latest)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    contract = CanonicalContract.load(args.contract)
    branch_config_path = Path(args.branch_config).expanduser().resolve()
    branch = json.loads(branch_config_path.read_text())
    if branch.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("branch experiment_id mismatch")
    if str(branch.get("branch_kind")) != "injected":
        raise ValueError("w(0) PPO bootstrap only supports injected branches")
    if "negative_control" in branch:
        raise ValueError("w(0) branch config must not contain negative_control")

    public_control = _validate_weight_control(branch["weight_control"])
    internal_control = _internal_control(
        public_control,
        canonical_alpha=contract.expected_canonical_alpha,
    )
    if not math.isclose(
        internal_control.effective_alpha,
        float(public_control["weight_at_zero"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError("private compatibility conversion changed w(0)")

    template_values = {
        str(key): str(value)
        for key, value in branch.get("template_values", {}).items()
    }
    if template_values.get("actor_update_mode") != "ppo_clip":
        raise ValueError("w(0) screening pilot is PPO-only")
    ppo_control = PPOActorControl.from_mapping(
        {
            "clip_epsilon": template_values["clip_epsilon"],
            "updates_per_old_policy": template_values["updates_per_old_policy"],
            "diagnostics_interval": template_values["diagnostics_interval"],
            "total_steps": template_values["steps"],
        }
    )

    module, source_checks = load_verified_canonical_module(contract)
    branch_manifest_path = Path(args.branch_manifest).expanduser().resolve()
    diagnostics_jsonl, diagnostics_latest = _ppo_paths(branch_manifest_path)
    diagnostics_jsonl.unlink(missing_ok=True)
    diagnostics_latest.unlink(missing_ok=True)
    patch_canonical_module_ppo(
        module,
        contract.target_class,
        negative_control=internal_control,
        ppo_control=ppo_control,
        return_mode=contract.return_mode,
        diagnostics_jsonl=diagnostics_jsonl,
        diagnostics_latest=diagnostics_latest,
    )

    trainer_args = list(args.trainer_args)
    if trainer_args and trainer_args[0] == "--":
        trainer_args = trainer_args[1:]
    manifest: dict[str, Any] = {
        "status": "started",
        "experiment_id": EXPERIMENT_ID,
        "branch": branch,
        "source_checks": source_checks,
        "weight_control": public_control,
        "actor_update_mode": "ppo_clip",
        "ppo_control": dataclasses.asdict(ppo_control),
        "trainer_path": str(contract.trainer_path),
        "trainer_args": trainer_args,
        "environment": canonical_environment_manifest(),
        "legacy_scale_persisted": False,
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

        if not diagnostics_jsonl.is_file() or not diagnostics_latest.is_file():
            raise RuntimeError("PPO branch completed without diagnostics outputs")
        _sanitize_diagnostics(diagnostics_jsonl, diagnostics_latest, public_control)
        latest = json.loads(diagnostics_latest.read_text())
        expected_steps = int(template_values["steps"])
        if latest.get("status") != "complete":
            raise RuntimeError("PPO diagnostics did not reach complete status")
        if int(latest.get("update", -1)) != expected_steps:
            raise RuntimeError(
                "PPO diagnostics update count mismatch: "
                f"{latest.get('update')} != {expected_steps}"
            )
        if "negative_control" in latest or "weight_control" not in latest:
            raise RuntimeError("PPO diagnostics did not preserve the public w(0) contract")
        manifest["ppo_diagnostics"] = {
            "jsonl": str(diagnostics_jsonl),
            "latest": str(diagnostics_latest),
            "final": latest,
        }
    except BaseException as exc:
        _sanitize_diagnostics(diagnostics_jsonl, diagnostics_latest, public_control)
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
