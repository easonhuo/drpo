"""Run one Stage A squared-EXP KL-threshold tuning branch."""

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

from drpo import e7_squared_exp_night_bootstrap as common
from drpo.e7_canonical_injection import (
    CanonicalContract,
    canonical_environment_manifest,
    load_verified_canonical_module,
)
from drpo.e7_canonical_ppo_injection import (
    PPOActorControl,
    patch_canonical_module_ppo,
)
from drpo.e7_ppo_kl_refresh import (
    PPOKLEarlyRefreshControl,
    patch_canonical_module_ppo_kl_refresh,
)
from drpo.e7_squared_exp_kernel import install_squared_exponential_kernel
from drpo.e7_w0_geometry_diagnostics import (
    GeometryDiagnostics,
    install_controlled_advantage_observer,
)


EXPERIMENT_ID = "EXT-H-E7-SQUARED-EXP-KL-TUNE-01"
STAGE_ID = "stage_a_kl_threshold_and_reference_lifecycle_screen"
REFERENCE_DISTANCE = 2.0
SUPPORTED_LIFECYCLES = {
    "ppo_clip_k4",
    "ppo_clip_k16",
    "ppo_clip_kl_k16_t0p003",
    "ppo_clip_kl_k16_t0p01",
    "ppo_clip_kl_k16_t0p03",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--branch-config", required=True)
    parser.add_argument("--branch-manifest", required=True)
    parser.add_argument("trainer_args", nargs=argparse.REMAINDER)
    return parser


def _validate_lifecycle(raw: Mapping[str, Any]) -> dict[str, Any]:
    lifecycle_id = str(raw.get("id"))
    if lifecycle_id not in SUPPORTED_LIFECYCLES:
        raise ValueError(f"unsupported reference lifecycle: {lifecycle_id}")
    clip_epsilon = float(raw.get("clip_epsilon"))
    updates = int(raw.get("max_updates_per_old_policy", -1))
    analytic = bool(raw.get("analytic_kl_early_refresh"))
    target = raw.get("target_kl")
    if not math.isclose(clip_epsilon, 0.2, abs_tol=1e-12):
        raise ValueError("clip epsilon must remain 0.2")
    expected_updates = 4 if lifecycle_id == "ppo_clip_k4" else 16
    if updates != expected_updates:
        raise ValueError(f"{lifecycle_id} reference window changed")
    expected_analytic = lifecycle_id.startswith("ppo_clip_kl_")
    if analytic is not expected_analytic:
        raise ValueError(f"{lifecycle_id} KL refresh flag changed")
    expected_targets = {
        "ppo_clip_kl_k16_t0p003": 0.003,
        "ppo_clip_kl_k16_t0p01": 0.01,
        "ppo_clip_kl_k16_t0p03": 0.03,
    }
    if expected_analytic:
        expected_target = expected_targets[lifecycle_id]
        if target is None or not math.isclose(
            float(target), expected_target, abs_tol=1e-12
        ):
            raise ValueError(f"{lifecycle_id} target_kl changed")
        target_value: float | None = expected_target
    else:
        if target is not None:
            raise ValueError(f"{lifecycle_id} must not define target_kl")
        target_value = None
    return {
        "id": lifecycle_id,
        "clip_epsilon": clip_epsilon,
        "max_updates_per_old_policy": updates,
        "analytic_kl_early_refresh": analytic,
        "target_kl": target_value,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    contract = CanonicalContract.load(args.contract)
    branch_config_path = Path(args.branch_config).expanduser().resolve()
    branch = json.loads(branch_config_path.read_text())
    if branch.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("branch experiment_id mismatch")
    if branch.get("stage_id") != STAGE_ID:
        raise ValueError("branch stage_id mismatch")
    if str(branch.get("branch_kind")) != "injected":
        raise ValueError("Stage A bootstrap supports injected branches only")
    if "negative_control" in branch:
        raise ValueError("public branch config must not contain negative_control")

    public_control = common._validate_weight_control(branch["weight_control"])  # noqa: SLF001
    internal_control = common._internal_control(  # noqa: SLF001
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
    lifecycle = _validate_lifecycle(branch["reference_lifecycle"])

    template_values = {
        str(key): str(value)
        for key, value in branch.get("template_values", {}).items()
    }
    if template_values.get("actor_update_mode") != lifecycle["id"]:
        raise ValueError("template lifecycle does not match branch lifecycle")
    expected_steps = int(template_values["steps"])
    runtime_probe = os.environ.get("DRPO_RUNTIME_RESOURCE_PROBE") == "1"
    if runtime_probe:
        if expected_steps <= 0:
            raise ValueError("resource probe steps must be positive")
    elif expected_steps != 1_000_000:
        raise ValueError("Stage A branches must run exactly 1,000,000 updates")
    diagnostics_interval = int(template_values["diagnostics_interval"])
    sampled_values_per_update = int(template_values["sampled_values_per_update"])

    module, source_checks = load_verified_canonical_module(contract)
    branch_manifest_path = Path(args.branch_manifest).expanduser().resolve()
    paths = common._branch_paths(branch_manifest_path)  # noqa: SLF001
    for path in paths.values():
        path.unlink(missing_ok=True)

    geometry = GeometryDiagnostics(
        public_control=public_control,
        actor_update_mode="ppo_clip",
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
        "stage_id": STAGE_ID,
        "branch": branch,
        "source_checks": source_checks,
        "weight_control": public_control,
        "reference_lifecycle": lifecycle,
        "trainer_path": str(contract.trainer_path),
        "trainer_args": trainer_args,
        "environment": canonical_environment_manifest(),
        "legacy_scale_persisted": False,
        "gae_used": False,
        "runtime_resource_probe": runtime_probe,
    }
    common._atomic_json(branch_manifest_path, manifest)  # noqa: SLF001

    ppo_control = PPOActorControl.from_mapping(
        {
            "clip_epsilon": lifecycle["clip_epsilon"],
            "updates_per_old_policy": lifecycle[
                "max_updates_per_old_policy"
            ],
            "diagnostics_interval": diagnostics_interval,
            "total_steps": expected_steps,
        }
    )

    old_argv = sys.argv[:]
    old_cwd = Path.cwd()
    try:
        with (
            install_squared_exponential_kernel(),
            install_controlled_advantage_observer(geometry),
        ):
            if lifecycle["analytic_kl_early_refresh"]:
                kl_control = PPOKLEarlyRefreshControl(
                    target_kl=float(lifecycle["target_kl"]),
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
                manifest["kl_control"] = dataclasses.asdict(kl_control)
            else:
                patch_canonical_module_ppo(
                    module,
                    contract.target_class,
                    negative_control=internal_control,
                    ppo_control=ppo_control,
                    return_mode=contract.return_mode,
                    diagnostics_jsonl=paths["ppo_jsonl"],
                    diagnostics_latest=paths["ppo_latest"],
                )
                manifest["kl_control"] = None
            manifest["ppo_control"] = dataclasses.asdict(ppo_control)

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

        if lifecycle["analytic_kl_early_refresh"]:
            if not paths["kl_jsonl"].is_file() or not paths["kl_latest"].is_file():
                raise RuntimeError("adaptive-KL branch completed without KL diagnostics")
            kl_latest = json.loads(paths["kl_latest"].read_text())
            if kl_latest.get("status") != "complete" or int(
                kl_latest.get("update", -1)
            ) != expected_steps:
                raise RuntimeError("KL diagnostics final update mismatch")
            if not math.isclose(
                float(kl_latest["target_kl"]),
                float(lifecycle["target_kl"]),
                abs_tol=1e-12,
            ):
                raise RuntimeError("KL diagnostics target mismatch")
            manifest["kl_diagnostics"] = {
                "jsonl": str(paths["kl_jsonl"]),
                "latest": str(paths["kl_latest"]),
                "final": kl_latest,
            }
        elif paths["kl_jsonl"].exists() or paths["kl_latest"].exists():
            raise RuntimeError("fixed-reference branch unexpectedly wrote KL diagnostics")
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
