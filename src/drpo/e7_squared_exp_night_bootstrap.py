"""Run one canonical E7 squared-remoteness night-suite branch."""

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
    patch_canonical_module,
)
from drpo.e7_canonical_ppo_injection import (
    PPOActorControl,
    patch_canonical_module_ppo,
)
from drpo.e7_ppo_kl_refresh import (
    PPOKLEarlyRefreshControl,
    patch_canonical_module_ppo_kl_refresh,
)
from drpo.e7_squared_exp_kernel import FORMULA, install_squared_exponential_kernel
from drpo.e7_w0_geometry_diagnostics import (
    GeometryDiagnostics,
    install_controlled_advantage_observer,
)


EXPERIMENT_ID = "EXT-H-E7-SQUARED-EXP-NIGHT-01"
REFERENCE_DISTANCE = 2.0
ACTOR_MODES = {"a2c", "ppo_clip_k4", "ppo_clip_kl_k16"}


def _atomic_json(path: Path, payload: Any) -> None:
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


def _branch_paths(branch_manifest: Path) -> dict[str, Path]:
    root = branch_manifest.parent
    return {
        "ppo_jsonl": root / "ppo_diagnostics.jsonl",
        "ppo_latest": root / "PPO_DIAGNOSTICS_LATEST.json",
        "kl_jsonl": root / "ppo_kl_diagnostics.jsonl",
        "kl_latest": root / "PPO_KL_DIAGNOSTICS_LATEST.json",
        "geometry_jsonl": root / "geometry_diagnostics.jsonl",
        "geometry_latest": root / "GEOMETRY_DIAGNOSTICS_LATEST.json",
    }


def _validate_weight_control(raw: Mapping[str, Any]) -> dict[str, Any]:
    forbidden = {"negative_scale", "canonical_alpha", "effective_alpha"}
    present = sorted(forbidden & set(raw))
    if present:
        raise ValueError(
            "direct-w(0) branch forbids legacy scale/alpha fields: "
            + ", ".join(present)
        )
    method = str(raw.get("method"))
    weight_at_zero = float(raw.get("weight_at_zero"))
    coefficient = float(raw.get("exp_coefficient"))
    reference_distance = float(raw.get("reference_distance"))
    formula = str(raw.get("formula"))
    if method not in {"positive_only", "squared_exponential"}:
        raise ValueError("method must be positive_only or squared_exponential")
    if not math.isfinite(weight_at_zero) or not 0.0 <= weight_at_zero <= 1.0:
        raise ValueError("weight_at_zero must be finite and in [0, 1]")
    if not math.isfinite(coefficient) or coefficient < 0.0:
        raise ValueError("exp_coefficient must be finite and non-negative")
    if not math.isclose(
        reference_distance,
        REFERENCE_DISTANCE,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("reference_distance must remain 2.0")
    if formula != FORMULA:
        raise ValueError("branch formula is not the squared-remoteness contract")
    if method == "positive_only" and (
        weight_at_zero != 0.0 or coefficient != 0.0
    ):
        raise ValueError("Positive-only requires stored w(0)=0,c=0")
    if method == "squared_exponential" and weight_at_zero != 1.0:
        raise ValueError("squared EXP branches require w(0)=1")
    return {
        "method": method,
        "weight_at_zero": weight_at_zero,
        "exp_coefficient": coefficient,
        "reference_distance": reference_distance,
        "formula": FORMULA,
    }


def _internal_control(
    public: Mapping[str, Any],
    *,
    canonical_alpha: float,
) -> NegativeControl:
    public_method = str(public["method"])
    weight_at_zero = float(public["weight_at_zero"])
    return NegativeControl(
        method=(
            "positive_only" if public_method == "positive_only" else "exponential"
        ),
        negative_scale=(
            0.0
            if public_method == "positive_only"
            else weight_at_zero / canonical_alpha
        ),
        canonical_alpha=canonical_alpha,
        reference_distance=REFERENCE_DISTANCE,
        exponential_coefficient=float(public["exp_coefficient"]),
    )


def _public_record(
    record: Mapping[str, Any],
    public: Mapping[str, Any],
) -> dict[str, Any]:
    value = dict(record)
    value.pop("negative_control", None)
    value["weight_control"] = dict(public)
    return value


def _sanitize_ppo_diagnostics(
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
            "".join(
                json.dumps(record, sort_keys=True) + "\n" for record in records
            )
        )
        os.replace(temporary, jsonl_path)
    if latest_path.is_file():
        _atomic_json(
            latest_path,
            _public_record(json.loads(latest_path.read_text()), public),
        )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    contract = CanonicalContract.load(args.contract)
    branch_config_path = Path(args.branch_config).expanduser().resolve()
    branch = json.loads(branch_config_path.read_text())
    if branch.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("branch experiment_id mismatch")
    if str(branch.get("branch_kind")) != "injected":
        raise ValueError("night-suite bootstrap supports injected branches only")
    if "negative_control" in branch:
        raise ValueError("public branch config must not contain negative_control")

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
    actor_mode = template_values.get("actor_update_mode")
    if actor_mode not in ACTOR_MODES:
        raise ValueError(f"unsupported actor_update_mode={actor_mode!r}")
    expected_steps = int(template_values["steps"])
    runtime_probe = os.environ.get("DRPO_RUNTIME_RESOURCE_PROBE") == "1"
    if runtime_probe:
        if expected_steps <= 0:
            raise ValueError("resource probe steps must be positive")
    elif expected_steps != 1_000_000:
        raise ValueError("night-suite branches must run exactly 1,000,000 updates")
    diagnostics_interval = int(template_values["diagnostics_interval"])
    sampled_values_per_update = int(template_values["sampled_values_per_update"])

    module, source_checks = load_verified_canonical_module(contract)
    branch_manifest_path = Path(args.branch_manifest).expanduser().resolve()
    paths = _branch_paths(branch_manifest_path)
    for path in paths.values():
        path.unlink(missing_ok=True)

    geometry = GeometryDiagnostics(
        public_control=public_control,
        actor_update_mode=("a2c" if actor_mode == "a2c" else "ppo_clip"),
        interval=diagnostics_interval,
        total_steps=expected_steps,
        sampled_values_per_update=sampled_values_per_update,
        jsonl_path=paths["geometry_jsonl"],
        latest_path=paths["geometry_latest"],
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
        "actor_update_mode": actor_mode,
        "trainer_path": str(contract.trainer_path),
        "trainer_args": trainer_args,
        "environment": canonical_environment_manifest(),
        "legacy_scale_persisted": False,
        "gae_used": False,
        "runtime_resource_probe": runtime_probe,
    }
    _atomic_json(branch_manifest_path, manifest)

    old_argv = sys.argv[:]
    old_cwd = Path.cwd()
    try:
        with (
            install_squared_exponential_kernel(),
            install_controlled_advantage_observer(geometry),
        ):
            if actor_mode == "a2c":
                patch_canonical_module(module, contract, internal_control)
                manifest["ppo_control"] = None
                manifest["kl_control"] = None
            elif actor_mode == "ppo_clip_k4":
                ppo_control = PPOActorControl.from_mapping(
                    {
                        "clip_epsilon": 0.2,
                        "updates_per_old_policy": 4,
                        "diagnostics_interval": diagnostics_interval,
                        "total_steps": expected_steps,
                    }
                )
                patch_canonical_module_ppo(
                    module,
                    contract.target_class,
                    negative_control=internal_control,
                    ppo_control=ppo_control,
                    return_mode=contract.return_mode,
                    diagnostics_jsonl=paths["ppo_jsonl"],
                    diagnostics_latest=paths["ppo_latest"],
                )
                manifest["ppo_control"] = dataclasses.asdict(ppo_control)
                manifest["kl_control"] = None
            else:
                ppo_control = PPOActorControl.from_mapping(
                    {
                        "clip_epsilon": 0.2,
                        "updates_per_old_policy": 16,
                        "diagnostics_interval": diagnostics_interval,
                        "total_steps": expected_steps,
                    }
                )
                kl_control = PPOKLEarlyRefreshControl(
                    target_kl=0.01,
                    diagnostics_interval=diagnostics_interval,
                )
                patch_canonical_module_ppo_kl_refresh(
                    module,
                    contract.target_class,
                    negative_control=internal_control,
                    ppo_control=ppo_control,
                    kl_control=kl_control,
                    return_mode=contract.return_mode,
                    ppo_diagnostics_jsonl=paths["ppo_jsonl"],
                    ppo_diagnostics_latest=paths["ppo_latest"],
                    kl_diagnostics_jsonl=paths["kl_jsonl"],
                    kl_diagnostics_latest=paths["kl_latest"],
                )
                manifest["ppo_control"] = dataclasses.asdict(ppo_control)
                manifest["kl_control"] = dataclasses.asdict(kl_control)

            os.chdir(contract.source_root)
            sys.argv = [str(contract.trainer_path), *trainer_args]
            try:
                runpy.run_path(str(contract.trainer_path), run_name="__main__")
            except SystemExit as exc:
                if exc.code not in (None, 0):
                    raise

        geometry_final = geometry.validate_complete()
        manifest["geometry_diagnostics"] = {
            "jsonl": str(paths["geometry_jsonl"]),
            "latest": str(paths["geometry_latest"]),
            "final": geometry_final,
        }
        if actor_mode != "a2c":
            _sanitize_ppo_diagnostics(
                paths["ppo_jsonl"],
                paths["ppo_latest"],
                public_control,
            )
            if not paths["ppo_jsonl"].is_file() or not paths[
                "ppo_latest"
            ].is_file():
                raise RuntimeError("PPO branch completed without diagnostics")
            latest = json.loads(paths["ppo_latest"].read_text())
            if latest.get("status") != "complete" or int(
                latest.get("update", -1)
            ) != expected_steps:
                raise RuntimeError("PPO diagnostics final update mismatch")
            manifest["ppo_diagnostics"] = {
                "jsonl": str(paths["ppo_jsonl"]),
                "latest": str(paths["ppo_latest"]),
                "final": latest,
            }
        if actor_mode == "ppo_clip_kl_k16":
            if not paths["kl_jsonl"].is_file() or not paths[
                "kl_latest"
            ].is_file():
                raise RuntimeError(
                    "KL-refresh branch completed without KL diagnostics"
                )
            kl_latest = json.loads(paths["kl_latest"].read_text())
            if kl_latest.get("status") != "complete" or int(
                kl_latest.get("update", -1)
            ) != expected_steps:
                raise RuntimeError("KL diagnostics final update mismatch")
            manifest["kl_diagnostics"] = {
                "jsonl": str(paths["kl_jsonl"]),
                "latest": str(paths["kl_latest"]),
                "final": kl_latest,
            }
    except BaseException as exc:
        _sanitize_ppo_diagnostics(
            paths["ppo_jsonl"],
            paths["ppo_latest"],
            public_control,
        )
        manifest["status"] = "failed"
        manifest["error_type"] = type(exc).__name__
        manifest["error"] = str(exc)
        _atomic_json(branch_manifest_path, manifest)
        raise
    finally:
        sys.argv = old_argv
        os.chdir(old_cwd)

    manifest["status"] = "completed"
    _atomic_json(branch_manifest_path, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
