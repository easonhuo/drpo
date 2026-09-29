"""Run one branch through the shared E7 squared-remoteness bootstrap."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import runpy
import sys
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np
import torch

from drpo import e7_squared_exp_night as suite
from drpo.e7_canonical_injection import (
    CanonicalContract,
    NegativeControl,
    canonical_environment_manifest,
    load_verified_canonical_module,
    patch_canonical_module,
    sha256_file,
)
from drpo.e7_canonical_ppo_injection import (
    PPOActorControl,
    patch_canonical_module_ppo,
)
from drpo.e7_ppo_kl_refresh import (
    PPOKLEarlyRefreshControl,
    patch_canonical_module_ppo_kl_refresh,
)
from drpo.e7_squared_exp_kernel import (
    FORMULA,
    THRESHOLDED_FORMULA,
    install_squared_exponential_kernel,
)
from drpo.e7_w0_geometry_diagnostics import (
    GeometryDiagnostics,
    install_controlled_advantage_observer,
)


_REQUIRED_HDF5_FIELDS = (
    "observations",
    "actions",
    "rewards",
    "terminals",
    "timeouts",
    "next_observations",
)


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
    if forbidden & set(raw):
        raise ValueError("public branch config contains legacy scale/alpha fields")
    formula = str(raw.get("formula"))
    if formula == THRESHOLDED_FORMULA:
        control = {
            "method": str(raw.get("method")),
            "weight_at_zero": float(raw.get("weight_at_zero")),
            "reference_distance": float(raw.get("reference_distance")),
            "formula": formula,
            "coordinate": str(raw.get("coordinate")),
            "remoteness_threshold": float(raw.get("remoteness_threshold")),
            "remoteness_scale": float(raw.get("remoteness_scale")),
            "taper_lambda": float(raw.get("taper_lambda")),
            "derived_exp_coefficient": float(raw.get("derived_exp_coefficient")),
        }
        finite = all(
            math.isfinite(float(control[name]))
            for name in (
                "weight_at_zero",
                "reference_distance",
                "remoteness_threshold",
                "remoteness_scale",
                "taper_lambda",
                "derived_exp_coefficient",
            )
        )
        if not finite:
            raise ValueError("thresholded taper parameters must be finite")
        if control["method"] not in {
            "positive_only",
            "thresholded_exponential",
            "uncontrolled",
        }:
            raise ValueError("unknown P1 thresholded control")
        if control["coordinate"] != "normalized_squared_standardized_distance":
            raise ValueError("P1 remoteness coordinate changed")
        if control["reference_distance"] != suite.REFERENCE_DISTANCE:
            raise ValueError("reference_distance changed")
        if control["remoteness_threshold"] < 0.0:
            raise ValueError("remoteness_threshold must be non-negative")
        if control["remoteness_scale"] <= 0.0 or control["taper_lambda"] <= 0.0:
            raise ValueError("remoteness_scale and taper_lambda must be positive")
        method = control["method"]
        expected = control["taper_lambda"] / control["remoteness_scale"]
        if method == "thresholded_exponential":
            if control["weight_at_zero"] != 1.0 or not math.isclose(
                control["derived_exp_coefficient"],
                expected,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError("thresholded EXP requires w(0)=1 and lambda/c")
        elif control["derived_exp_coefficient"] != 0.0:
            raise ValueError("anchor controls require zero derived coefficient")
        if method == "positive_only" and control["weight_at_zero"] != 0.0:
            raise ValueError("Positive-only requires zero negative weight")
        if method == "uncontrolled" and control["weight_at_zero"] != 1.0:
            raise ValueError("uncontrolled requires unit negative weight")
        return control

    control = {
        "method": str(raw.get("method")),
        "weight_at_zero": float(raw.get("weight_at_zero")),
        "exp_coefficient": float(raw.get("exp_coefficient")),
        "reference_distance": float(raw.get("reference_distance")),
        "formula": formula,
    }
    method, w0, coefficient = (
        control["method"],
        control["weight_at_zero"],
        control["exp_coefficient"],
    )
    if method not in {"positive_only", "squared_exponential"}:
        raise ValueError("unknown squared-remoteness control")
    if not math.isfinite(w0) or not 0.0 <= w0 <= 1.0:
        raise ValueError("weight_at_zero must be finite and in [0,1]")
    if not math.isfinite(coefficient) or coefficient < 0.0:
        raise ValueError("exp_coefficient must be finite and non-negative")
    if control["reference_distance"] != suite.REFERENCE_DISTANCE or formula != FORMULA:
        raise ValueError("squared-remoteness public contract changed")
    if method == "positive_only" and (w0 != 0.0 or coefficient != 0.0):
        raise ValueError("Positive-only requires w(0)=0,c=0")
    if method == "squared_exponential" and w0 != 1.0:
        raise ValueError("squared EXP requires w(0)=1")
    return control


def _internal_control(public: Mapping[str, Any], alpha: float) -> NegativeControl:
    method = str(public["method"])
    if method == "positive_only":
        return NegativeControl(
            method="positive_only",
            negative_scale=0.0,
            canonical_alpha=alpha,
            reference_distance=suite.REFERENCE_DISTANCE,
        )
    if method == "uncontrolled":
        return NegativeControl(
            method="global",
            negative_scale=1.0 / alpha,
            canonical_alpha=alpha,
            reference_distance=suite.REFERENCE_DISTANCE,
        )
    coefficient = float(
        public.get("derived_exp_coefficient", public.get("exp_coefficient"))
    )
    return NegativeControl(
        method="exponential",
        negative_scale=float(public["weight_at_zero"]) / alpha,
        canonical_alpha=alpha,
        reference_distance=suite.REFERENCE_DISTANCE,
        exponential_coefficient=coefficient,
    )


def _public_record(record: Mapping[str, Any], public: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(record)
    value.pop("negative_control", None)
    value["weight_control"] = dict(public)
    return value


def _sanitize_ppo_diagnostics(
    jsonl_path: Path, latest_path: Path, public: Mapping[str, Any]
) -> None:
    if jsonl_path.is_file():
        records = [
            _public_record(json.loads(line), public)
            for line in jsonl_path.read_text().splitlines()
            if line.strip()
        ]
        temporary = jsonl_path.with_suffix(jsonl_path.suffix + ".tmp")
        temporary.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in records)
        )
        os.replace(temporary, jsonl_path)
    if latest_path.is_file():
        _atomic_json(
            latest_path,
            _public_record(json.loads(latest_path.read_text()), public),
        )


def _validate_ordered_hdf5(path: str | Path) -> Path:
    source = Path(path).expanduser().resolve()
    with h5py.File(source, "r") as handle:
        missing = [name for name in _REQUIRED_HDF5_FIELDS if name not in handle]
        if missing:
            raise ValueError(f"ordered GAE replay is missing HDF5 fields: {missing}")
        lengths = {int(handle[name].shape[0]) for name in _REQUIRED_HDF5_FIELDS}
        if lengths == {0} or len(lengths) != 1:
            raise ValueError("ordered GAE HDF5 fields must be non-empty and aligned")
        terminal = np.asarray(handle["terminals"][:], dtype=np.bool_).reshape(-1)
        timeout = np.asarray(handle["timeouts"][:], dtype=np.bool_).reshape(-1)
        if bool((terminal & timeout).any()):
            raise ValueError("terminal and timeout flags must not overlap")
    return source


def compute_snapshot_tables(
    rewards: np.ndarray,
    values: np.ndarray,
    next_values: np.ndarray,
    terminals: np.ndarray,
    timeouts: np.ndarray,
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    arrays = [
        np.asarray(x).reshape(-1)
        for x in (rewards, values, next_values, terminals, timeouts)
    ]
    reward, value, next_value = [x.astype(np.float64) for x in arrays[:3]]
    terminal, timeout = [x.astype(np.bool_) for x in arrays[3:]]
    if not reward.size or len({x.shape for x in arrays}) != 1:
        raise ValueError("snapshot arrays must be non-empty and aligned")
    if bool((terminal & timeout).any()):
        raise ValueError("terminal and timeout flags must not overlap")
    if not all(np.isfinite(x).all() for x in (reward, value, next_value)):
        raise ValueError("snapshot values must be finite")
    if not 0.0 <= gamma <= 1.0 or not 0.0 <= gae_lambda <= 1.0:
        raise ValueError("gamma and gae_lambda must be in [0,1]")
    td = reward + gamma * next_value * (~terminal) - value
    continuation = ~(terminal | timeout)
    continuation[-1] = False
    gae = np.empty_like(td)
    running = 0.0
    for index in range(td.size - 1, -1, -1):
        running = td[index] + gamma * gae_lambda * continuation[index] * running
        gae[index] = running
    return td.astype(np.float32), gae.astype(np.float32)


def _critic_hash(critic: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(critic.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _critic_values(critic: torch.nn.Module, observations: np.ndarray) -> np.ndarray:
    device = next(critic.parameters()).device
    output = np.empty(len(observations), dtype=np.float32)
    was_training = critic.training
    critic.eval()
    try:
        with torch.no_grad():
            for start in range(0, len(observations), 8192):
                states = torch.as_tensor(
                    observations[start : start + 8192],
                    dtype=torch.float32,
                    device=device,
                )
                values = critic(states).squeeze(-1).detach().cpu().numpy()
                output[start : start + len(values)] = values
    finally:
        critic.train(was_training)
    if not np.isfinite(output).all():
        raise FloatingPointError("critic snapshot produced non-finite values")
    return output


class TrajectorySnapshotAdvantage:
    """Periodic TD/GAE lookup; the canonical update still owns both optimizers."""

    def __init__(
        self,
        replay: Mapping[str, np.ndarray],
        estimator: str,
        batch_size: int = 256,
    ) -> None:
        if estimator not in {"td", "gae"}:
            raise ValueError(f"unsupported estimator={estimator!r}")
        if len({len(value) for value in replay.values()}) != 1:
            raise ValueError("ordered replay arrays must be aligned")
        self.replay, self.estimator = dict(replay), estimator
        self.refresh_interval = math.ceil(len(replay["rewards"]) / batch_size)
        self.update_count, self.table, self.agent = 0, None, None
        self.snapshot_hashes: list[str] = []

    def _refresh(self, agent: Any) -> None:
        td, gae = compute_snapshot_tables(
            self.replay["rewards"],
            _critic_values(agent.critic, self.replay["observations"]),
            _critic_values(agent.critic, self.replay["next_observations"]),
            self.replay["terminals"],
            self.replay["timeouts"],
            gamma=float(agent.gamma),
            gae_lambda=suite.GAE_LAMBDA,
        )
        self.table = torch.from_numpy(td if self.estimator == "td" else gae)
        self.snapshot_hashes.append(_critic_hash(agent.critic))

    def __call__(self, agent: Any, context: Any, default: torch.Tensor) -> torch.Tensor:
        self.agent = agent
        self.update_count += 1
        if self.table is None or (self.update_count - 1) % self.refresh_interval == 0:
            self._refresh(agent)
        raw = torch.as_tensor(context, dtype=torch.float32).reshape(-1)
        ids = raw.round().long().cpu()
        if not bool(torch.isfinite(raw).all()) or not torch.equal(raw, ids.float()):
            raise ValueError("transition IDs must be finite exact integers")
        if (
            ids.numel() != default.numel()
            or int(ids.min()) < 0
            or int(ids.max()) >= len(self.table)
        ):
            raise ValueError("transition IDs are outside or misaligned with ordered replay")
        return self.table.index_select(0, ids)

    def summary(self) -> dict[str, Any]:
        if self.agent is None or not self.snapshot_hashes:
            raise RuntimeError("trajectory snapshot provider was never used")
        final = _critic_hash(self.agent.critic)
        return {
            "estimator": self.estimator,
            "gae_lambda": suite.GAE_LAMBDA,
            "snapshot_count": len(self.snapshot_hashes),
            "snapshot_refresh_interval": self.refresh_interval,
            "snapshot_hashes": list(self.snapshot_hashes),
            "first_snapshot_critic_sha256": self.snapshot_hashes[0],
            "latest_snapshot_critic_sha256": self.snapshot_hashes[-1],
            "final_critic_sha256": final,
            "critic_evolution_observed": final != self.snapshot_hashes[0],
        }


def _ordered_replay(branch: Mapping[str, Any]) -> tuple[dict[str, np.ndarray], Any]:
    source = _validate_ordered_hdf5(branch["dataset_path"])
    if sha256_file(source) != branch["dataset_sha256"]:
        raise RuntimeError("ordered dataset hash mismatch")
    root = str(Path(branch["canonical_root"]).expanduser().resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    import d4rl_common.train_loop as train_loop

    raw = train_loop.load_hdf5(source, dataset_name=str(branch["dataset_id"]))
    replay = {
        "observations": np.asarray(raw["obs"], dtype=np.float32),
        "actions": np.asarray(raw["acts"], dtype=np.float32),
        "rewards": np.asarray(raw["rews"], dtype=np.float32).reshape(-1),
        "next_observations": np.asarray(raw["next_obs"], dtype=np.float32),
        "terminals": np.asarray(raw["terms"], dtype=np.bool_).reshape(-1),
        "timeouts": np.asarray(raw["touts"], dtype=np.bool_).reshape(-1),
    }
    return replay, train_loop


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    contract = CanonicalContract.load(args.contract)
    branch = json.loads(Path(args.branch_config).expanduser().read_text())
    experiment_id = str(branch.get("experiment_id"))
    if experiment_id not in {suite.EXPERIMENT_ID, suite.GAE_EXPERIMENT_ID}:
        raise ValueError("branch experiment_id mismatch")
    if branch.get("profile_id") not in {
        None,
        suite.TUNING_PROFILE_ID,
        suite.P3_PROFILE_ID,
    }:
        raise ValueError("branch tuning profile mismatch")
    if branch.get("branch_kind") != "injected" or "negative_control" in branch:
        raise ValueError("bootstrap requires a public injected branch")
    public = _validate_weight_control(branch["weight_control"])
    control = _internal_control(public, contract.expected_canonical_alpha)
    values = {
        str(key): str(value) for key, value in branch["template_values"].items()
    }
    actor_mode, expected_steps = values["actor_update_mode"], int(values["steps"])
    runtime_probe = os.environ.get("DRPO_RUNTIME_RESOURCE_PROBE") == "1"
    bounded = runtime_probe or values.get("execution_mode") == "liveness"
    if expected_steps <= 0 or (not bounded and expected_steps != suite.EXPECTED_STEPS):
        raise ValueError("branch optimizer-step budget changed")

    trainer_args = list(args.trainer_args)
    if trainer_args and trainer_args[0] == "--":
        trainer_args.pop(0)
    provider = None
    train_loop = original_returns = None
    if experiment_id == suite.GAE_EXPERIMENT_ID:
        if (
            actor_mode != "a2c"
            or suite._flag_value(trainer_args, "--batch") != "256"  # noqa: SLF001
        ):
            raise ValueError("GAE successor requires canonical A2C batch 256")
        if (
            "--ret_weight_mode" in trainer_args
            and suite._flag_value(trainer_args, "--ret_weight_mode")  # noqa: SLF001
            != "none"
        ):
            raise ValueError("transition IDs require ret_weight_mode=none")
        replay, train_loop = _ordered_replay(branch)
        provider = TrajectorySnapshotAdvantage(
            replay,
            values["advantage_estimator"],
        )
        original_returns = train_loop.compute_mc_returns
        train_loop.compute_mc_returns = lambda rewards, *_args, **_kwargs: np.arange(
            len(rewards), dtype=np.float32
        )

    module, source_checks = load_verified_canonical_module(contract)
    manifest_path = Path(args.branch_manifest).expanduser().resolve()
    paths = _branch_paths(manifest_path)
    for path in paths.values():
        path.unlink(missing_ok=True)
    geometry = GeometryDiagnostics(
        public_control=public,
        actor_update_mode="a2c" if actor_mode == "a2c" else "ppo_clip",
        interval=int(values["diagnostics_interval"]),
        total_steps=expected_steps,
        sampled_values_per_update=int(values["sampled_values_per_update"]),
        jsonl_path=paths["geometry_jsonl"],
        latest_path=paths["geometry_latest"],
    )
    manifest = {
        "status": "started",
        "experiment_id": experiment_id,
        "branch": branch,
        "source_checks": source_checks,
        "weight_control": public,
        "actor_update_mode": actor_mode,
        "trainer_path": str(contract.trainer_path),
        "trainer_args": trainer_args,
        "environment": canonical_environment_manifest(),
        "legacy_scale_persisted": False,
        "gae_used": bool(provider and provider.estimator == "gae"),
        "runtime_resource_probe": runtime_probe,
    }
    _atomic_json(manifest_path, manifest)

    old_argv, old_cwd = sys.argv[:], Path.cwd()
    try:
        with (
            install_squared_exponential_kernel(
                remoteness_threshold=float(public.get("remoteness_threshold", 0.0))
            ),
            install_controlled_advantage_observer(geometry),
        ):
            if actor_mode == "a2c":
                patch_canonical_module(
                    module,
                    contract,
                    control,
                    advantage_provider=provider,
                )
                manifest.update(ppo_control=None, kl_control=None)
            else:
                if provider is not None:
                    raise ValueError("GAE provider cannot enter the PPO path")
                updates = 4 if actor_mode == "ppo_clip_k4" else 16
                ppo = PPOActorControl.from_mapping(
                    {
                        "clip_epsilon": 0.2,
                        "updates_per_old_policy": updates,
                        "diagnostics_interval": int(values["diagnostics_interval"]),
                        "total_steps": expected_steps,
                    }
                )
                if actor_mode == "ppo_clip_k4":
                    patch_canonical_module_ppo(
                        module,
                        contract.target_class,
                        negative_control=control,
                        ppo_control=ppo,
                        return_mode=contract.return_mode,
                        diagnostics_jsonl=paths["ppo_jsonl"],
                        diagnostics_latest=paths["ppo_latest"],
                    )
                    manifest.update(
                        ppo_control=dataclasses.asdict(ppo),
                        kl_control=None,
                    )
                elif actor_mode == "ppo_clip_kl_k16":
                    kl = PPOKLEarlyRefreshControl(
                        target_kl=0.01,
                        diagnostics_interval=int(values["diagnostics_interval"]),
                    )
                    patch_canonical_module_ppo_kl_refresh(
                        module,
                        contract.target_class,
                        negative_control=control,
                        ppo_control=ppo,
                        kl_control=kl,
                        return_mode=contract.return_mode,
                        ppo_diagnostics_jsonl=paths["ppo_jsonl"],
                        ppo_diagnostics_latest=paths["ppo_latest"],
                        kl_diagnostics_jsonl=paths["kl_jsonl"],
                        kl_diagnostics_latest=paths["kl_latest"],
                    )
                    manifest.update(
                        ppo_control=dataclasses.asdict(ppo),
                        kl_control=dataclasses.asdict(kl),
                    )
                else:
                    raise ValueError(f"unsupported actor_update_mode={actor_mode!r}")
            os.chdir(contract.source_root)
            sys.argv = [str(contract.trainer_path), *trainer_args]
            try:
                runpy.run_path(str(contract.trainer_path), run_name="__main__")
            except SystemExit as exc:
                if exc.code not in (None, 0):
                    raise

        manifest["geometry_diagnostics"] = {
            "jsonl": str(paths["geometry_jsonl"]),
            "latest": str(paths["geometry_latest"]),
            "final": geometry.validate_complete(),
        }
        if provider is not None:
            snapshot = provider.summary()
            if not runtime_probe and (
                snapshot["snapshot_count"] < 2
                or not snapshot["critic_evolution_observed"]
            ):
                raise RuntimeError("GAE branch did not prove snapshots and critic evolution")
            manifest.update(
                advantage_estimator=provider.estimator,
                critic_updated_during_actor_training=True,
                prepared_advantage_artifact_used=False,
                transition_id_channel="ep_ret_exact_float32_index",
                trajectory_snapshot=snapshot,
            )
        if actor_mode != "a2c":
            _sanitize_ppo_diagnostics(
                paths["ppo_jsonl"], paths["ppo_latest"], public
            )
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
            latest = json.loads(paths["kl_latest"].read_text())
            if latest.get("status") != "complete" or int(
                latest.get("update", -1)
            ) != expected_steps:
                raise RuntimeError("KL diagnostics final update mismatch")
            manifest["kl_diagnostics"] = {
                "jsonl": str(paths["kl_jsonl"]),
                "latest": str(paths["kl_latest"]),
                "final": latest,
            }
    except BaseException as exc:
        _sanitize_ppo_diagnostics(
            paths["ppo_jsonl"], paths["ppo_latest"], public
        )
        manifest.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        _atomic_json(manifest_path, manifest)
        raise
    finally:
        if train_loop is not None:
            train_loop.compute_mc_returns = original_returns
        sys.argv, _ = old_argv, os.chdir(old_cwd)

    manifest["status"] = "completed"
    _atomic_json(manifest_path, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
