"""Run one EXT-H-E7-SQEXP-GAE-01 branch with frozen trajectory advantages."""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import importlib
import json
import math
import os
import runpy
import sys
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np

from drpo.e7_canonical_injection import (
    CanonicalContract,
    canonical_environment_manifest,
    load_verified_canonical_module,
    patch_canonical_module,
    sha256_file,
)
from drpo.e7_canonical_ppo_injection import (
    PPOActorControl,
    patch_canonical_module_ppo,
)
from drpo.e7_frozen_advantage import build_external_advantage_agent_class
from drpo.e7_squared_exp_kernel import install_squared_exponential_kernel
from drpo.e7_squared_exp_night_bootstrap import (
    _atomic_json,
    _branch_paths,
    _internal_control,
    _sanitize_ppo_diagnostics,
    _validate_weight_control,
)
from drpo.e7_trajectory_advantage import (
    array_sha256,
    verify_array_manifest,
)
from drpo.e7_w0_geometry_diagnostics import (
    GeometryDiagnostics,
    install_controlled_advantage_observer,
)


EXPERIMENT_ID = "EXT-H-E7-SQEXP-GAE-01"
ACTOR_MODES = {"a2c", "ppo_clip_k4"}
ADVANTAGE_MODES = {"one_step_td", "gae_lambda_0p95"}
EXPECTED_GAMMA = 0.99
EXPECTED_GAE_LAMBDA = 0.95
EXPECTED_STEPS = 1_000_000


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--branch-config", required=True)
    parser.add_argument("--branch-manifest", required=True)
    parser.add_argument("trainer_args", nargs=argparse.REMAINDER)
    return parser


def _flag_value(argv: list[str], flag: str) -> str | None:
    positions = [index for index, token in enumerate(argv) if token == flag]
    if not positions:
        return None
    if len(positions) != 1 or positions[0] + 1 >= len(argv):
        raise ValueError(f"trainer args must contain at most one complete {flag}")
    return argv[positions[0] + 1]


def _validate_trainer_args(argv: list[str]) -> None:
    ret_weight = _flag_value(argv, "--ret_weight_mode")
    if ret_weight not in {None, "none"}:
        raise ValueError(
            "trajectory-advantage branches require uniform transition sampling; "
            "ret_weight_mode must remain none"
        )
    steps = _flag_value(argv, "--steps")
    runtime_probe = os.environ.get("DRPO_RUNTIME_RESOURCE_PROBE") == "1"
    if steps is None:
        raise ValueError("trainer args are missing --steps")
    if runtime_probe:
        if int(steps) <= 0:
            raise ValueError("resource-probe steps must be positive")
    elif int(steps) != EXPECTED_STEPS:
        raise ValueError("GAE pilot branches must run exactly 1,000,000 updates")


def _load_prepared_artifact(
    artifact_dir: Path,
    *,
    branch: Mapping[str, Any],
    advantage_mode: str,
) -> tuple[np.ndarray, dict[str, Any], Path]:
    manifest_path = artifact_dir / "ADVANTAGE_MANIFEST.json"
    arrays_path = artifact_dir / "advantages.npz"
    critic_path = artifact_dir / "critic_checkpoint.pt"
    for required in (manifest_path, arrays_path, critic_path):
        if not required.is_file():
            raise FileNotFoundError(f"prepared GAE artifact is missing: {required}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "COMPLETE":
        raise ValueError("prepared GAE artifact is not COMPLETE")
    if manifest.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("prepared artifact experiment_id mismatch")
    if manifest.get("dataset_id") != branch.get("dataset_id"):
        raise ValueError("prepared artifact dataset_id mismatch")
    if int(manifest.get("seed", -1)) != int(branch.get("seed", -2)):
        raise ValueError("prepared artifact seed mismatch")
    if manifest.get("dataset_sha256") != branch.get("dataset_sha256"):
        raise ValueError("prepared artifact dataset SHA-256 mismatch")
    if not math.isclose(
        float(manifest.get("gamma")), EXPECTED_GAMMA, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("prepared artifact gamma changed")
    if not math.isclose(
        float(manifest.get("gae_lambda")),
        EXPECTED_GAE_LAMBDA,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("prepared artifact lambda changed")
    if sha256_file(arrays_path) != manifest.get("advantages_file_sha256"):
        raise ValueError("prepared advantages.npz SHA-256 mismatch")
    if sha256_file(critic_path) != manifest.get("critic_checkpoint_sha256"):
        raise ValueError("prepared critic checkpoint SHA-256 mismatch")
    with np.load(arrays_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    verify_array_manifest(arrays, manifest)
    advantage = np.ascontiguousarray(arrays[advantage_mode].astype(np.float32))
    if array_sha256(advantage) != manifest[f"{advantage_mode}_sha256"]:
        raise ValueError("selected advantage array SHA-256 mismatch")
    return advantage, manifest, critic_path


@contextlib.contextmanager
def _install_precomputed_advantage_return(
    *,
    contract: CanonicalContract,
    advantage: np.ndarray,
    prepared_manifest: Mapping[str, Any],
) -> Iterator[None]:
    """Patch the trainer's MC-return helper before its from-import executes."""

    root_text = str(contract.source_root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    train_loop = importlib.import_module("d4rl_common.train_loop")
    original = train_loop.compute_mc_returns
    calls = 0

    def injected(rews: Any, terms: Any, touts: Any, gamma: float = 0.99) -> np.ndarray:
        nonlocal calls
        calls += 1
        if calls != 1:
            raise RuntimeError("trainer requested the prepared advantage array more than once")
        rewards = np.ascontiguousarray(np.asarray(rews, dtype=np.float32))
        terminals = np.ascontiguousarray(np.asarray(terms, dtype=np.bool_))
        timeouts = np.ascontiguousarray(np.asarray(touts, dtype=np.bool_))
        if len(rewards) != len(advantage):
            raise ValueError("trainer dataset length differs from prepared advantage length")
        expected_hashes = {
            "normalized_reward_sha256": array_sha256(rewards),
            "terminal_sha256": array_sha256(terminals),
            "timeout_sha256": array_sha256(timeouts),
        }
        for key, actual in expected_hashes.items():
            if actual != prepared_manifest.get(key):
                raise ValueError(
                    f"trainer ordered dataset does not match prepared artifact: {key}"
                )
        if not math.isclose(
            float(gamma), EXPECTED_GAMMA, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError("trainer gamma changed from prepared artifact")
        return advantage.copy()

    train_loop.compute_mc_returns = injected
    completed = False
    try:
        yield
        completed = True
    finally:
        train_loop.compute_mc_returns = original
    if completed and calls != 1:
        raise RuntimeError(
            f"trainer advantage injection call count was {calls}, expected exactly one"
        )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    contract = CanonicalContract.load(args.contract)
    branch_config_path = Path(args.branch_config).expanduser().resolve()
    branch = json.loads(branch_config_path.read_text())
    if branch.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("branch experiment_id mismatch")
    if str(branch.get("branch_kind")) != "injected":
        raise ValueError("GAE bootstrap supports injected branches only")
    if "negative_control" in branch:
        raise ValueError("public branch config must not contain negative_control")

    public_control = _validate_weight_control(branch["weight_control"])
    internal_control = _internal_control(
        public_control,
        canonical_alpha=contract.expected_canonical_alpha,
    )
    values = {str(key): str(value) for key, value in branch["template_values"].items()}
    actor_mode = values.get("actor_update_mode")
    advantage_mode = values.get("advantage_mode")
    if actor_mode not in ACTOR_MODES:
        raise ValueError(f"unsupported actor_update_mode={actor_mode!r}")
    if advantage_mode not in ADVANTAGE_MODES:
        raise ValueError(f"unsupported advantage_mode={advantage_mode!r}")
    expected_steps = int(values["steps"])
    diagnostics_interval = int(values["diagnostics_interval"])
    sampled_values_per_update = int(values["sampled_values_per_update"])

    trainer_args = list(args.trainer_args)
    if trainer_args and trainer_args[0] == "--":
        trainer_args = trainer_args[1:]
    _validate_trainer_args(trainer_args)

    artifact_dir = Path(branch["advantage_artifact_dir"]).expanduser().resolve()
    advantage, prepared_manifest, critic_path = _load_prepared_artifact(
        artifact_dir,
        branch=branch,
        advantage_mode=advantage_mode,
    )
    module, source_checks = load_verified_canonical_module(contract)
    branch_manifest_path = Path(args.branch_manifest).expanduser().resolve()
    paths = _branch_paths(branch_manifest_path)
    paths["support_jsonl"] = branch_manifest_path.parent / "actor_support_diagnostics.jsonl"
    paths["support_latest"] = (
        branch_manifest_path.parent / "ACTOR_SUPPORT_DIAGNOSTICS_LATEST.json"
    )
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
    manifest: dict[str, Any] = {
        "status": "started",
        "experiment_id": EXPERIMENT_ID,
        "branch": branch,
        "source_checks": source_checks,
        "weight_control": public_control,
        "actor_update_mode": actor_mode,
        "advantage_mode": advantage_mode,
        "trajectory_advantage_used": True,
        "gae_used": advantage_mode == "gae_lambda_0p95",
        "gamma": EXPECTED_GAMMA,
        "gae_lambda": (
            EXPECTED_GAE_LAMBDA if advantage_mode == "gae_lambda_0p95" else 0.0
        ),
        "critic_frozen": True,
        "critic_checkpoint": str(critic_path),
        "critic_checkpoint_sha256": prepared_manifest["critic_checkpoint_sha256"],
        "prepared_advantage_manifest": str(artifact_dir / "ADVANTAGE_MANIFEST.json"),
        "prepared_ordered_trajectory_identity": prepared_manifest[
            "ordered_trajectory_identity"
        ],
        "trainer_path": str(contract.trainer_path),
        "trainer_args": trainer_args,
        "environment": canonical_environment_manifest(),
        "legacy_scale_persisted": False,
        "runtime_resource_probe": os.environ.get("DRPO_RUNTIME_RESOURCE_PROBE") == "1",
    }
    _atomic_json(branch_manifest_path, manifest)

    old_argv = sys.argv[:]
    old_cwd = Path.cwd()
    try:
        if actor_mode == "a2c":
            injected = patch_canonical_module(module, contract, internal_control)
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
            injected = patch_canonical_module_ppo(
                module,
                contract.target_class,
                negative_control=internal_control,
                ppo_control=ppo_control,
                return_mode=contract.return_mode,
                diagnostics_jsonl=paths["ppo_jsonl"],
                diagnostics_latest=paths["ppo_latest"],
            )
            manifest["ppo_control"] = dataclasses.asdict(ppo_control)
        external = build_external_advantage_agent_class(
            injected,
            critic_checkpoint=critic_path,
            advantage_metadata={
                "advantage_source": advantage_mode,
                "gamma": EXPECTED_GAMMA,
                "gae_lambda": (
                    EXPECTED_GAE_LAMBDA
                    if advantage_mode == "gae_lambda_0p95"
                    else 0.0
                ),
                "ordered_trajectory_identity": prepared_manifest[
                    "ordered_trajectory_identity"
                ],
                "critic_checkpoint_sha256": prepared_manifest[
                    "critic_checkpoint_sha256"
                ],
            },
            return_mode=contract.return_mode,
            diagnostics_interval=diagnostics_interval,
            total_steps=expected_steps,
            support_diagnostics_jsonl=paths["support_jsonl"],
            support_diagnostics_latest=paths["support_latest"],
        )
        setattr(module, contract.target_class, external)
        _atomic_json(branch_manifest_path, manifest)

        with (
            install_squared_exponential_kernel(),
            install_controlled_advantage_observer(geometry),
            _install_precomputed_advantage_return(
                contract=contract,
                advantage=advantage,
                prepared_manifest=prepared_manifest,
            ),
        ):
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
        if not paths["support_jsonl"].is_file() or not paths["support_latest"].is_file():
            raise RuntimeError("branch completed without support diagnostics")
        support_latest = json.loads(paths["support_latest"].read_text())
        if support_latest.get("status") != "complete" or int(
            support_latest.get("update", -1)
        ) != expected_steps:
            raise RuntimeError("support diagnostics final update mismatch")
        manifest["support_diagnostics"] = {
            "jsonl": str(paths["support_jsonl"]),
            "latest": str(paths["support_latest"]),
            "final": support_latest,
        }
        if actor_mode == "ppo_clip_k4":
            _sanitize_ppo_diagnostics(
                paths["ppo_jsonl"],
                paths["ppo_latest"],
                public_control,
            )
            if not paths["ppo_jsonl"].is_file() or not paths["ppo_latest"].is_file():
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
