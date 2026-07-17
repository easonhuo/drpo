"""Run one canonical E7 squared-remoteness branch through the shared bootstrap."""

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
from drpo.e7_squared_exp_kernel import FORMULA, install_squared_exponential_kernel
from drpo.e7_w0_geometry_diagnostics import (
    GeometryDiagnostics,
    install_controlled_advantage_observer,
)


EXPERIMENT_ID = "EXT-H-E7-SQUARED-EXP-NIGHT-01"
GAE_EXPERIMENT_ID = "EXT-H-E7-SQEXP-GAE-01"
REFERENCE_DISTANCE = 2.0
ACTOR_MODES = {"a2c", "ppo_clip_k4", "ppo_clip_kl_k16"}
GAE_ESTIMATORS = {"td", "gae"}
GAE_LAMBDA = 0.95
GAE_BATCH_SIZE = 256
_FLOAT32_EXACT_INTEGER_LIMIT = 2**24
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
            "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
        )
        os.replace(temporary, jsonl_path)
    if latest_path.is_file():
        _atomic_json(
            latest_path,
            _public_record(json.loads(latest_path.read_text()), public),
        )


def _trainer_flag(argv: list[str], flag: str) -> str | None:
    positions = [index for index, token in enumerate(argv) if token == flag]
    if len(positions) > 1:
        raise ValueError(f"trainer args contain duplicate {flag}")
    if not positions:
        return None
    index = positions[0]
    if index + 1 >= len(argv):
        raise ValueError(f"trainer arg {flag} has no value")
    return str(argv[index + 1])


def _validate_gae_trainer_args(
    argv: list[str],
    *,
    expected_steps: int,
    runtime_probe: bool,
    liveness: bool,
) -> None:
    if _trainer_flag(argv, "--variant") != "iqlv_exp_rank":
        raise ValueError("GAE pilot requires canonical iqlv_exp_rank")
    if _trainer_flag(argv, "--batch") != str(GAE_BATCH_SIZE):
        raise ValueError("GAE pilot requires canonical batch size 256")
    if _trainer_flag(argv, "--ret_weight_mode") not in {None, "none"}:
        raise ValueError("transition-ID channel requires uniform ret_weight_mode=none")
    steps = _trainer_flag(argv, "--steps")
    if steps is None or int(steps) != expected_steps or expected_steps <= 0:
        raise ValueError("GAE trainer --steps does not match the branch contract")
    if not runtime_probe and not liveness and expected_steps != 1_000_000:
        raise ValueError("full GAE branches require exactly 1,000,000 updates")


def _validate_ordered_hdf5(path: str | Path) -> Path:
    source = Path(path).expanduser().resolve()
    with h5py.File(source, "r") as handle:
        missing = [name for name in _REQUIRED_HDF5_FIELDS if name not in handle]
        if missing:
            raise ValueError(f"ordered GAE replay is missing HDF5 fields: {missing}")
        lengths = {int(handle[name].shape[0]) for name in _REQUIRED_HDF5_FIELDS}
        if lengths == {0} or len(lengths) != 1:
            raise ValueError("ordered GAE HDF5 fields must be non-empty and aligned")
        terminals = np.asarray(handle["terminals"][:], dtype=np.bool_).reshape(-1)
        timeouts = np.asarray(handle["timeouts"][:], dtype=np.bool_).reshape(-1)
        if bool((terminals & timeouts).any()):
            raise ValueError("terminal and timeout flags must not overlap")
    return source


def _transition_ids(transition_count: int) -> np.ndarray:
    if transition_count <= 0 or transition_count > _FLOAT32_EXACT_INTEGER_LIMIT:
        raise ValueError("transition count is outside the exact float32 ID range")
    return np.arange(transition_count, dtype=np.float32)


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
    """Compute matched one-step TD and GAE tables from one critic snapshot."""

    reward, value, next_value = (
        np.asarray(item, dtype=np.float64).reshape(-1)
        for item in (rewards, values, next_values)
    )
    terminal, timeout = (
        np.asarray(item, dtype=np.bool_).reshape(-1)
        for item in (terminals, timeouts)
    )
    shapes = {item.shape for item in (reward, value, next_value, terminal, timeout)}
    if not reward.size or len(shapes) != 1:
        raise ValueError("snapshot arrays must be non-empty and aligned")
    if bool((terminal & timeout).any()):
        raise ValueError("terminal and timeout flags must not overlap")
    if not all(np.isfinite(item).all() for item in (reward, value, next_value)):
        raise ValueError("snapshot values must be finite")
    if not 0.0 <= gamma <= 1.0 or not 0.0 <= gae_lambda <= 1.0:
        raise ValueError("gamma and gae_lambda must be in [0, 1]")

    td = reward + gamma * next_value * (~terminal) - value
    continuation = ~(terminal | timeout)
    continuation[-1] = False
    gae = np.empty_like(td)
    running = 0.0
    for index in range(td.size - 1, -1, -1):
        running = td[index] + gamma * gae_lambda * continuation[index] * running
        gae[index] = running
    return td.astype(np.float32), gae.astype(np.float32)


def _state_dict_sha256(state: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        for item in (name, str(value.dtype), str(tuple(value.shape))):
            digest.update(item.encode("utf-8"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _critic_values(
    critic: torch.nn.Module,
    observations: np.ndarray,
    *,
    device: torch.device,
    chunk_size: int = 8192,
) -> np.ndarray:
    output = np.empty(len(observations), dtype=np.float32)
    was_training = critic.training
    critic.eval()
    try:
        with torch.no_grad():
            for start in range(0, len(observations), chunk_size):
                states = torch.as_tensor(
                    observations[start : start + chunk_size],
                    dtype=torch.float32,
                    device=device,
                )
                values = critic(states).squeeze(-1).detach().cpu().float().numpy()
                output[start : start + len(values)] = values
    finally:
        critic.train(was_training)
    if not np.isfinite(output).all():
        raise FloatingPointError("critic snapshot produced non-finite values")
    return output


class TrajectorySnapshotAdvantage:
    """Periodic actor-advantage lookup that leaves the canonical update intact."""

    def __init__(
        self,
        *,
        replay: Mapping[str, np.ndarray],
        estimator: str,
        gae_lambda: float = GAE_LAMBDA,
        batch_size: int = GAE_BATCH_SIZE,
    ) -> None:
        if estimator not in GAE_ESTIMATORS:
            raise ValueError(f"unsupported estimator={estimator!r}")
        if not 0.0 <= gae_lambda <= 1.0:
            raise ValueError("gae_lambda must be in [0, 1]")
        lengths = {len(value) for value in replay.values()}
        if lengths == {0} or len(lengths) != 1:
            raise ValueError("ordered replay arrays must be non-empty and aligned")
        self.replay = dict(replay)
        self.estimator = estimator
        self.gae_lambda = float(gae_lambda)
        self.refresh_interval = math.ceil(len(self.replay["rewards"]) / batch_size)
        self.update_count = 0
        self.table: torch.Tensor | None = None
        self.snapshot_hashes: list[str] = []
        self.last_snapshot_update = 0
        self.agent: Any | None = None

    def _refresh(self, agent: Any) -> None:
        device = next(agent.actor.parameters()).device
        values = _critic_values(agent.critic, self.replay["observations"], device=device)
        next_values = _critic_values(
            agent.critic, self.replay["next_observations"], device=device
        )
        td, gae = compute_snapshot_tables(
            self.replay["rewards"],
            values,
            next_values,
            self.replay["terminals"],
            self.replay["timeouts"],
            gamma=float(agent.gamma),
            gae_lambda=self.gae_lambda,
        )
        selected = td if self.estimator == "td" else gae
        self.table = torch.from_numpy(selected.copy())
        self.snapshot_hashes.append(_state_dict_sha256(agent.critic.state_dict()))
        self.last_snapshot_update = self.update_count

    def __call__(
        self,
        agent: Any,
        transition_context: Any,
        default_advantages: torch.Tensor,
    ) -> torch.Tensor:
        self.agent = agent
        self.update_count += 1
        if self.table is None or (self.update_count - 1) % self.refresh_interval == 0:
            self._refresh(agent)
        raw_ids = torch.as_tensor(transition_context, dtype=torch.float32).reshape(-1)
        rounded = raw_ids.round()
        if not bool(torch.isfinite(raw_ids).all()) or not torch.equal(raw_ids, rounded):
            raise ValueError("transition IDs must be finite exact integers")
        transition_ids = rounded.long().cpu()
        if transition_ids.numel() != default_advantages.numel():
            raise ValueError("transition IDs are not aligned with actor batch")
        assert self.table is not None
        if int(transition_ids.min()) < 0 or int(transition_ids.max()) >= self.table.numel():
            raise ValueError("transition ID is outside ordered replay")
        return self.table.index_select(0, transition_ids)

    def summary(self) -> dict[str, Any]:
        if self.agent is None:
            raise RuntimeError("trajectory snapshot provider was never used")
        first = self.snapshot_hashes[0] if self.snapshot_hashes else None
        final = _state_dict_sha256(self.agent.critic.state_dict())
        return {
            "estimator": self.estimator,
            "gae_lambda": self.gae_lambda,
            "snapshot_count": len(self.snapshot_hashes),
            "snapshot_refresh_interval": self.refresh_interval,
            "snapshot_hashes": list(self.snapshot_hashes),
            "first_snapshot_critic_sha256": first,
            "latest_snapshot_critic_sha256": (
                self.snapshot_hashes[-1] if self.snapshot_hashes else None
            ),
            "final_critic_sha256": final,
            "critic_evolution_observed": bool(first and final != first),
            "last_snapshot_update": self.last_snapshot_update,
        }


def _load_ordered_replay(
    *,
    canonical_root: str | Path,
    dataset_path: str | Path,
    dataset_id: str,
) -> tuple[dict[str, np.ndarray], Any]:
    source = _validate_ordered_hdf5(dataset_path)
    root = str(Path(canonical_root).expanduser().resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    import d4rl_common.train_loop as train_loop

    raw = train_loop.load_hdf5(source, dataset_name=dataset_id)
    replay = {
        "observations": np.asarray(raw["obs"], dtype=np.float32),
        "actions": np.asarray(raw["acts"], dtype=np.float32),
        "rewards": np.asarray(raw["rews"], dtype=np.float32).reshape(-1),
        "next_observations": np.asarray(raw["next_obs"], dtype=np.float32),
        "terminals": np.asarray(raw["terms"], dtype=np.bool_).reshape(-1),
        "timeouts": np.asarray(raw["touts"], dtype=np.bool_).reshape(-1),
    }
    if bool((replay["terminals"] & replay["timeouts"]).any()):
        raise ValueError("terminal and timeout flags must not overlap")
    return replay, train_loop


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    contract = CanonicalContract.load(args.contract)
    branch_config_path = Path(args.branch_config).expanduser().resolve()
    branch = json.loads(branch_config_path.read_text())
    experiment_id = str(branch.get("experiment_id"))
    if experiment_id not in {EXPERIMENT_ID, GAE_EXPERIMENT_ID}:
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
    liveness = template_values.get("execution_mode") == "liveness"
    if runtime_probe or liveness:
        if expected_steps <= 0:
            raise ValueError("bounded probe steps must be positive")
    elif expected_steps != 1_000_000:
        raise ValueError("night-suite branches must run exactly 1,000,000 updates")
    diagnostics_interval = int(template_values["diagnostics_interval"])
    sampled_values_per_update = int(template_values["sampled_values_per_update"])

    trainer_args = list(args.trainer_args)
    if trainer_args and trainer_args[0] == "--":
        trainer_args = trainer_args[1:]

    advantage_provider: TrajectorySnapshotAdvantage | None = None
    train_loop: Any | None = None
    original_compute_mc_returns: Any | None = None
    if experiment_id == GAE_EXPERIMENT_ID:
        if actor_mode != "a2c":
            raise ValueError("joint GAE pilot supports canonical A2C only")
        estimator = str(template_values.get("advantage_estimator"))
        _validate_gae_trainer_args(
            trainer_args,
            expected_steps=expected_steps,
            runtime_probe=runtime_probe,
            liveness=liveness,
        )
        dataset_path = Path(branch["dataset_path"]).expanduser().resolve()
        if sha256_file(dataset_path) != branch["dataset_sha256"]:
            raise RuntimeError("ordered dataset hash mismatch")
        replay, train_loop = _load_ordered_replay(
            canonical_root=branch["canonical_root"],
            dataset_path=dataset_path,
            dataset_id=str(branch["dataset_id"]),
        )
        advantage_provider = TrajectorySnapshotAdvantage(
            replay=replay,
            estimator=estimator,
        )
        original_compute_mc_returns = train_loop.compute_mc_returns

        def transition_id_returns(rewards: Any, *_: Any, **__: Any) -> np.ndarray:
            if len(rewards) != len(replay["rewards"]):
                raise RuntimeError(
                    "trainer replay length changed before transition-ID injection"
                )
            return _transition_ids(len(rewards))

        train_loop.compute_mc_returns = transition_id_returns

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

    manifest: dict[str, Any] = {
        "status": "started",
        "experiment_id": experiment_id,
        "branch": branch,
        "source_checks": source_checks,
        "weight_control": public_control,
        "actor_update_mode": actor_mode,
        "trainer_path": str(contract.trainer_path),
        "trainer_args": trainer_args,
        "environment": canonical_environment_manifest(),
        "legacy_scale_persisted": False,
        "gae_used": bool(
            advantage_provider is not None and advantage_provider.estimator == "gae"
        ),
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
                patch_canonical_module(
                    module,
                    contract,
                    internal_control,
                    advantage_provider=advantage_provider,
                )
                manifest["ppo_control"] = None
                manifest["kl_control"] = None
            elif actor_mode == "ppo_clip_k4":
                if advantage_provider is not None:
                    raise ValueError("GAE provider cannot enter the PPO path")
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
                if advantage_provider is not None:
                    raise ValueError("GAE provider cannot enter the PPO path")
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
        if advantage_provider is not None:
            snapshot = advantage_provider.summary()
            if not runtime_probe and (
                int(snapshot["snapshot_count"]) < 2
                or snapshot["critic_evolution_observed"] is not True
            ):
                raise RuntimeError(
                    "joint-critic branch did not prove two snapshots and critic evolution"
                )
            manifest.update(
                advantage_estimator=advantage_provider.estimator,
                critic_updated_during_actor_training=True,
                prepared_advantage_artifact_used=False,
                transition_id_channel="ep_ret_exact_float32_index",
                trajectory_snapshot=snapshot,
            )
        if actor_mode != "a2c":
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
        if actor_mode == "ppo_clip_kl_k16":
            if not paths["kl_jsonl"].is_file() or not paths["kl_latest"].is_file():
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
        if train_loop is not None and original_compute_mc_returns is not None:
            train_loop.compute_mc_returns = original_compute_mc_returns
        sys.argv = old_argv
        os.chdir(old_cwd)

    manifest["status"] = "completed"
    _atomic_json(branch_manifest_path, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
