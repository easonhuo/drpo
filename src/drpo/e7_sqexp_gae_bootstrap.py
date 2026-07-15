"""Run one EXT-H-E7-SQEXP-GAE-01 actor branch."""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

from drpo.e7_canonical_injection import (
    CanonicalContract,
    NegativeControl,
    load_verified_canonical_module,
)
from drpo.e7_canonical_ppo_injection import PPOActorControl
from drpo.e7_offline_gae import atomic_write_json, sha256_file
from drpo.e7_precomputed_advantage_injection import (
    patch_canonical_module_precomputed_a2c,
    patch_canonical_module_precomputed_ppo,
)
from drpo.e7_squared_exp_kernel import FORMULA, install_squared_exponential_kernel
from drpo.e7_sqexp_gae_trainer import main as trainer_main
from drpo.e7_w0_geometry_diagnostics import (
    GeometryDiagnostics,
    install_controlled_advantage_observer,
)


EXPERIMENT_ID = "EXT-H-E7-SQEXP-GAE-01"
REFERENCE_DISTANCE = 2.0
ACTOR_MODES = {"a2c", "ppo_clip_k4"}
ADVANTAGE_ESTIMATORS = {"td", "gae"}


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
        "geometry_jsonl": root / "geometry_diagnostics.jsonl",
        "geometry_latest": root / "GEOMETRY_DIAGNOSTICS_LATEST.json",
    }


def _validate_weight_control(raw: Mapping[str, Any]) -> dict[str, Any]:
    forbidden = {"negative_scale", "canonical_alpha", "effective_alpha"}
    present = sorted(forbidden & set(raw))
    if present:
        raise ValueError(
            "public GAE branch forbids legacy scale/alpha fields: "
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
    if not math.isclose(reference_distance, REFERENCE_DISTANCE, abs_tol=1e-12):
        raise ValueError("reference_distance must remain 2")
    if formula != FORMULA:
        raise ValueError("branch formula is not squared remoteness")
    if method == "positive_only" and (
        weight_at_zero != 0.0 or coefficient != 0.0
    ):
        raise ValueError("Positive-only requires w(0)=0,c=0")
    if method == "squared_exponential" and weight_at_zero != 1.0:
        raise ValueError("squared EXP requires w(0)=1")
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
    method = str(public["method"])
    weight_at_zero = float(public["weight_at_zero"])
    return NegativeControl(
        method="positive_only" if method == "positive_only" else "exponential",
        negative_scale=(
            0.0 if method == "positive_only" else weight_at_zero / canonical_alpha
        ),
        canonical_alpha=canonical_alpha,
        reference_distance=REFERENCE_DISTANCE,
        exponential_coefficient=float(public["exp_coefficient"]),
    )


def _sanitize_ppo_diagnostics(
    jsonl_path: Path,
    latest_path: Path,
    public: Mapping[str, Any],
) -> None:
    def public_record(record: Mapping[str, Any]) -> dict[str, Any]:
        result = dict(record)
        result.pop("negative_control", None)
        result["weight_control"] = dict(public)
        return result

    if jsonl_path.is_file():
        records = [
            public_record(json.loads(line))
            for line in jsonl_path.read_text().splitlines()
            if line.strip()
        ]
        temporary = jsonl_path.with_suffix(jsonl_path.suffix + ".tmp")
        temporary.write_text(
            "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
        )
        os.replace(temporary, jsonl_path)
    if latest_path.is_file():
        atomic_write_json(
            latest_path,
            public_record(json.loads(latest_path.read_text())),
        )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    contract = CanonicalContract.load(args.contract)
    branch_config_path = Path(args.branch_config).expanduser().resolve()
    branch = json.loads(branch_config_path.read_text())
    if branch.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("branch experiment_id mismatch")
    if branch.get("branch_kind") != "injected":
        raise ValueError("GAE bootstrap supports injected branches only")
    public_control = _validate_weight_control(branch["weight_control"])
    internal_control = _internal_control(
        public_control,
        canonical_alpha=contract.expected_canonical_alpha,
    )
    if not math.isclose(
        internal_control.effective_alpha,
        float(public_control["weight_at_zero"]),
        abs_tol=1e-12,
    ):
        raise RuntimeError("compatibility conversion changed w(0)")

    template = {
        str(key): str(value)
        for key, value in branch.get("template_values", {}).items()
    }
    actor_mode = template.get("actor_update_mode")
    estimator = template.get("advantage_estimator")
    if actor_mode not in ACTOR_MODES:
        raise ValueError(f"unsupported actor mode {actor_mode!r}")
    if estimator not in ADVANTAGE_ESTIMATORS:
        raise ValueError(f"unsupported advantage estimator {estimator!r}")
    expected_steps = int(template["steps"])
    runtime_probe = os.environ.get("DRPO_RUNTIME_RESOURCE_PROBE") == "1"
    if not runtime_probe and expected_steps != 1_000_000:
        raise ValueError("GAE branches must run exactly 1,000,000 updates")
    if runtime_probe and expected_steps <= 0:
        raise ValueError("resource probe steps must be positive")
    diagnostics_interval = int(template["diagnostics_interval"])
    sampled_values_per_update = int(template["sampled_values_per_update"])

    prepared_manifest_path = Path(
        branch["advantage_manifest"]
    ).expanduser().resolve()
    if not prepared_manifest_path.is_file():
        raise FileNotFoundError(prepared_manifest_path)
    prepared = json.loads(prepared_manifest_path.read_text())
    if prepared.get("status") != "complete":
        raise RuntimeError("prepared advantage artifact is incomplete")
    if prepared.get("dataset_id") != branch["dataset_id"]:
        raise RuntimeError("prepared artifact dataset mismatch")
    if int(prepared.get("seed", -1)) != int(branch["seed"]):
        raise RuntimeError("prepared artifact seed mismatch")

    module, source_checks = load_verified_canonical_module(contract)
    branch_manifest_path = Path(args.branch_manifest).expanduser().resolve()
    paths = _branch_paths(branch_manifest_path)
    for path in paths.values():
        path.unlink(missing_ok=True)

    geometry = GeometryDiagnostics(
        public_control=public_control,
        actor_update_mode="a2c" if actor_mode == "a2c" else "ppo_clip",
        interval=diagnostics_interval,
        total_steps=expected_steps,
        sampled_values_per_update=sampled_values_per_update,
        jsonl_path=paths["geometry_jsonl"],
        latest_path=paths["geometry_latest"],
    )
    trainer_args = list(args.trainer_args)
    if trainer_args and trainer_args[0] == "--":
        trainer_args = trainer_args[1:]
    if (
        "--advantage-manifest" in trainer_args
        or "--advantage-estimator" in trainer_args
    ):
        raise ValueError("source trainer args must not inject advantage flags")
    trainer_args.extend(
        [
            "--advantage-manifest",
            str(prepared_manifest_path),
            "--advantage-estimator",
            estimator,
        ]
    )

    manifest: dict[str, Any] = {
        "status": "started",
        "experiment_id": EXPERIMENT_ID,
        "branch": branch,
        "source_checks": source_checks,
        "weight_control": public_control,
        "actor_update_mode": actor_mode,
        "advantage_estimator": estimator,
        "advantage_manifest": str(prepared_manifest_path),
        "advantage_manifest_sha256": sha256_file(prepared_manifest_path),
        "critic_frozen": True,
        "behavior_trajectory_not_on_policy": True,
        "trainer_module": "drpo.e7_sqexp_gae_trainer",
        "trainer_args": trainer_args,
        "gae_used": estimator == "gae",
        "runtime_resource_probe": runtime_probe,
    }
    atomic_write_json(branch_manifest_path, manifest)

    try:
        with (
            install_squared_exponential_kernel(),
            install_controlled_advantage_observer(geometry),
        ):
            if actor_mode == "a2c":
                patch_canonical_module_precomputed_a2c(
                    module,
                    contract.target_class,
                    negative_control=internal_control,
                    return_mode=contract.return_mode,
                )
                manifest["ppo_control"] = None
            else:
                ppo_control = PPOActorControl.from_mapping(
                    {
                        "clip_epsilon": 0.2,
                        "updates_per_old_policy": 4,
                        "diagnostics_interval": diagnostics_interval,
                        "total_steps": expected_steps,
                    }
                )
                patch_canonical_module_precomputed_ppo(
                    module,
                    contract.target_class,
                    negative_control=internal_control,
                    ppo_control=ppo_control,
                    return_mode=contract.return_mode,
                    diagnostics_jsonl=paths["ppo_jsonl"],
                    diagnostics_latest=paths["ppo_latest"],
                )
                manifest["ppo_control"] = dataclasses.asdict(ppo_control)
            trainer_main(trainer_args)

        manifest["geometry_diagnostics"] = {
            "jsonl": str(paths["geometry_jsonl"]),
            "latest": str(paths["geometry_latest"]),
            "final": geometry.validate_complete(),
        }
        if actor_mode == "ppo_clip_k4":
            _sanitize_ppo_diagnostics(
                paths["ppo_jsonl"], paths["ppo_latest"], public_control
            )
            if not paths["ppo_latest"].is_file():
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
    except BaseException as exc:
        _sanitize_ppo_diagnostics(
            paths["ppo_jsonl"], paths["ppo_latest"], public_control
        )
        manifest["status"] = "failed"
        manifest["error_type"] = type(exc).__name__
        manifest["error"] = str(exc)
        atomic_write_json(branch_manifest_path, manifest)
        raise

    manifest["status"] = "completed"
    atomic_write_json(branch_manifest_path, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
