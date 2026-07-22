"""Run one branch of the E7 actor/high-c decision pilot."""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import math
import os
import runpy
import sys
from pathlib import Path
from typing import Any, Mapping

from drpo import e7_squared_exp_night_bootstrap as common
from drpo.e7_canonical_injection import (
    CanonicalContract,
    NegativeControl,
    canonical_environment_manifest,
    load_verified_canonical_module,
    patch_canonical_module,
)
from drpo.e7_canonical_ppo_injection import PPOActorControl
from drpo.e7_ppo_kl_refresh import (
    PPOKLEarlyRefreshControl,
    patch_canonical_module_ppo_kl_refresh,
)
from drpo.e7_squared_exp_kernel import install_squared_exponential_kernel
from drpo.e7_w0_geometry_diagnostics import (
    GeometryDiagnostics,
    install_controlled_advantage_observer,
)


EXPERIMENT_ID = "EXT-H-E7-SQEXP-ACTOR-DECISION-01"
REFERENCE_DISTANCE = 2.0
ACTOR_MODES = {"a2c", "ppo_clip_kl_k4"}
LINEAR_FORMULA = "w(d)=w(0)*exp(-c*(d/2))"
SQUARED_FORMULA = "w(d)=w(0)*exp(-c*(d/2)^2)"
POSITIVE_ONLY_FORMULA = "w(d)=0"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--branch-config", required=True)
    parser.add_argument("--branch-manifest", required=True)
    parser.add_argument("trainer_args", nargs=argparse.REMAINDER)
    return parser


def _validate_weight_control(raw: Mapping[str, Any]) -> dict[str, Any]:
    forbidden = {"negative_scale", "canonical_alpha", "effective_alpha"}
    present = sorted(forbidden & set(raw))
    if present:
        raise ValueError("public weight control forbids legacy fields: " + ", ".join(present))
    control_id = str(raw.get("id"))
    family = str(raw.get("family"))
    w0 = float(raw.get("weight_at_zero"))
    coefficient = float(raw.get("exp_coefficient"))
    reference_distance = float(raw.get("reference_distance"))
    formula = str(raw.get("formula"))
    expected: dict[str, tuple[str, float, float, str]] = {
        "positive_only": ("positive_only", 0.0, 0.0, POSITIVE_ONLY_FORMULA),
        "linear_c12": ("linear_exponential", 1.0, 12.0, LINEAR_FORMULA),
        "squared_c4": ("squared_exponential", 1.0, 4.0, SQUARED_FORMULA),
        "squared_c8": ("squared_exponential", 1.0, 8.0, SQUARED_FORMULA),
        "squared_c16": ("squared_exponential", 1.0, 16.0, SQUARED_FORMULA),
        "squared_c32": ("squared_exponential", 1.0, 32.0, SQUARED_FORMULA),
        "squared_c64": ("squared_exponential", 1.0, 64.0, SQUARED_FORMULA),
        "squared_c128": ("squared_exponential", 1.0, 128.0, SQUARED_FORMULA),
    }
    if control_id not in expected:
        raise ValueError(f"unsupported control: {control_id}")
    expected_family, expected_w0, expected_c, expected_formula = expected[control_id]
    if family != expected_family:
        raise ValueError(f"{control_id} family changed")
    if not math.isclose(w0, expected_w0, abs_tol=1e-12):
        raise ValueError(f"{control_id} w(0) changed")
    if not math.isclose(coefficient, expected_c, abs_tol=1e-12):
        raise ValueError(f"{control_id} coefficient changed")
    if not math.isclose(reference_distance, REFERENCE_DISTANCE, abs_tol=1e-12):
        raise ValueError(f"{control_id} reference distance changed")
    if formula != expected_formula:
        raise ValueError(f"{control_id} formula changed")
    return {
        "id": control_id,
        "family": family,
        "weight_at_zero": w0,
        "exp_coefficient": coefficient,
        "reference_distance": reference_distance,
        "formula": formula,
    }


def _internal_control(public: Mapping[str, Any], *, canonical_alpha: float) -> NegativeControl:
    family = str(public["family"])
    w0 = float(public["weight_at_zero"])
    return NegativeControl(
        method="positive_only" if family == "positive_only" else "exponential",
        negative_scale=0.0 if family == "positive_only" else w0 / canonical_alpha,
        canonical_alpha=canonical_alpha,
        reference_distance=REFERENCE_DISTANCE,
        exponential_coefficient=float(public["exp_coefficient"]),
    )


def _kernel_context(family: str) -> contextlib.AbstractContextManager[Any]:
    if family == "squared_exponential":
        return install_squared_exponential_kernel()
    return contextlib.nullcontext()


def _validate_actor(raw: Mapping[str, Any]) -> dict[str, Any]:
    actor_id = str(raw.get("id"))
    if actor_id not in ACTOR_MODES:
        raise ValueError(f"unsupported actor mode: {actor_id}")
    expected = (
        {
            "id": "a2c",
            "clip_epsilon": None,
            "max_updates_per_old_policy": None,
            "analytic_kl_early_refresh": False,
            "target_kl": None,
            "kl_penalty": False,
        }
        if actor_id == "a2c"
        else {
            "id": "ppo_clip_kl_k4",
            "clip_epsilon": 0.2,
            "max_updates_per_old_policy": 4,
            "analytic_kl_early_refresh": True,
            "target_kl": 0.01,
            "kl_penalty": False,
        }
    )
    for key, value in expected.items():
        actual = raw.get(key)
        if isinstance(value, float):
            if actual is None or not math.isclose(float(actual), value, abs_tol=1e-12):
                raise ValueError(f"{actor_id} {key} changed")
        elif actual != value:
            raise ValueError(f"{actor_id} {key} changed")
    return expected


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    contract = CanonicalContract.load(args.contract)
    branch_config_path = Path(args.branch_config).expanduser().resolve()
    branch = json.loads(branch_config_path.read_text())
    if branch.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("branch experiment_id mismatch")
    if str(branch.get("branch_kind")) != "injected":
        raise ValueError("actor-decision bootstrap supports injected branches only")
    if "negative_control" in branch:
        raise ValueError("public branch config must not contain negative_control")

    public_control = _validate_weight_control(branch["weight_control"])
    actor = _validate_actor(branch["actor_update"])
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
    if template_values.get("actor_update_mode") != actor["id"]:
        raise ValueError("template actor mode does not match branch actor mode")
    expected_steps = int(template_values["steps"])
    runtime_probe = os.environ.get("DRPO_RUNTIME_RESOURCE_PROBE") == "1"
    if runtime_probe:
        if expected_steps <= 0:
            raise ValueError("resource probe steps must be positive")
    elif expected_steps != 1_000_000:
        raise ValueError("actor-decision branches must run exactly 1,000,000 updates")
    diagnostics_interval = int(template_values["diagnostics_interval"])
    if diagnostics_interval != 10_000:
        raise ValueError("diagnostics interval changed")
    sampled_values_per_update = int(template_values["sampled_values_per_update"])

    module, source_checks = load_verified_canonical_module(contract)
    branch_manifest_path = Path(args.branch_manifest).expanduser().resolve()
    paths = common._branch_paths(branch_manifest_path)  # noqa: SLF001
    for path in paths.values():
        path.unlink(missing_ok=True)

    geometry = GeometryDiagnostics(
        public_control=public_control,
        actor_update_mode="a2c" if actor["id"] == "a2c" else "ppo_clip",
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
        "actor_update": actor,
        "trainer_path": str(contract.trainer_path),
        "trainer_args": trainer_args,
        "environment": canonical_environment_manifest(),
        "legacy_scale_persisted": False,
        "gae_used": False,
        "runtime_resource_probe": runtime_probe,
        "kl_event_jsonl_enabled": False,
    }
    common._atomic_json(branch_manifest_path, manifest)  # noqa: SLF001

    old_argv = sys.argv[:]
    old_cwd = Path.cwd()
    try:
        with (
            _kernel_context(str(public_control["family"])),
            install_controlled_advantage_observer(geometry),
        ):
            if actor["id"] == "a2c":
                patch_canonical_module(module, contract, internal_control)
                manifest["ppo_control"] = None
                manifest["kl_control"] = None
            else:
                ppo_control = PPOActorControl.from_mapping(
                    {
                        "clip_epsilon": 0.2,
                        "updates_per_old_policy": 4,
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
                    kl_diagnostics_jsonl=None,
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
        if actor["id"] != "a2c":
            common._sanitize_ppo_diagnostics(  # noqa: SLF001
                paths["ppo_jsonl"],
                paths["ppo_latest"],
                public_control,
            )
            if not paths["ppo_jsonl"].is_file() or not paths["ppo_latest"].is_file():
                raise RuntimeError("PPO branch completed without diagnostics")
            ppo_latest = json.loads(paths["ppo_latest"].read_text())
            if ppo_latest.get("status") != "complete" or int(
                ppo_latest.get("update", -1)
            ) != expected_steps:
                raise RuntimeError("PPO diagnostics final update mismatch")
            manifest["ppo_diagnostics"] = {
                "jsonl": str(paths["ppo_jsonl"]),
                "latest": str(paths["ppo_latest"]),
                "final": ppo_latest,
            }
            if paths["kl_jsonl"].exists():
                raise RuntimeError("per-trigger KL JSONL must remain disabled")
            if not paths["kl_latest"].is_file():
                raise RuntimeError("PPO-KL branch completed without KL latest diagnostics")
            kl_latest = json.loads(paths["kl_latest"].read_text())
            if kl_latest.get("status") != "complete" or int(
                kl_latest.get("update", -1)
            ) != expected_steps:
                raise RuntimeError("KL diagnostics final update mismatch")
            if not math.isclose(float(kl_latest["target_kl"]), 0.01, abs_tol=1e-12):
                raise RuntimeError("KL target mismatch")
            manifest["kl_diagnostics"] = {
                "jsonl": None,
                "latest": str(paths["kl_latest"]),
                "final": kl_latest,
            }
    except BaseException as exc:
        common._sanitize_ppo_diagnostics(  # noqa: SLF001
            paths["ppo_jsonl"],
            paths["ppo_latest"],
            public_control,
        )
        manifest["status"] = "failed"
        manifest["error_type"] = type(exc).__name__
        manifest["error"] = str(exc)
        common._atomic_json(branch_manifest_path, manifest)  # noqa: SLF001
        raise
    finally:
        sys.argv = old_argv
        os.chdir(old_cwd)

    manifest["status"] = "complete"
    common._atomic_json(branch_manifest_path, manifest)  # noqa: SLF001
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
