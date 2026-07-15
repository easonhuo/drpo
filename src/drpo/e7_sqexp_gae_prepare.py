"""Prepare one shared frozen-critic TD/GAE artifact for E7."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from drpo.e7_canonical_injection import (
    CanonicalContract,
    load_verified_canonical_module,
)
from drpo.e7_offline_gae import (
    advantage_diagnostics,
    atomic_write_json,
    audit_ordered_trajectories,
    compute_gae_numpy,
    compute_gae_torch,
    compute_td_advantage,
    sha256_file,
)


SCHEMA_VERSION = 1
PREPARER_VERSION = "1.0.0-e7-sqexp-gae"
DEFAULT_GAMMA = 0.99
DEFAULT_GAE_LAMBDA = 0.95
DEFAULT_CRITIC_STEPS = 100_000
DEFAULT_BATCH_SIZE = 256
DEFAULT_LEARNING_RATE = 3e-4
DEFAULT_EXPECTILE = 0.5


def _load_dataset(path: Path, dataset_id: str) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as source:
        required = {
            "observations",
            "actions",
            "rewards",
            "next_observations",
            "terminals",
            "timeouts",
        }
        missing = sorted(required - set(source.keys()))
        if missing:
            raise ValueError(
                "GAE preparation requires explicit ordered D4RL arrays; "
                f"missing={missing}"
            )
        data = {
            "obs": source["observations"][:].astype(np.float32),
            "acts": source["actions"][:].astype(np.float32),
            "rews": source["rewards"][:].astype(np.float32),
            "next_obs": source["next_observations"][:].astype(np.float32),
            "terms": source["terminals"][:].astype(np.bool_),
            "touts": source["timeouts"][:].astype(np.bool_),
        }
    n = int(data["obs"].shape[0])
    for name, array in data.items():
        if int(array.shape[0]) != n:
            raise ValueError(f"dataset array {name} has inconsistent length")
    prefix = dataset_id.replace("_", "-").split("-")[0]
    if prefix in {"hopper", "halfcheetah", "walker2d"}:
        from d4rl_common.normalize import reward_norm_locomotion

        data["rews"] = reward_norm_locomotion(
            data["rews"], data["terms"], data["touts"]
        ).astype(np.float32)
    return data


def _critic_values(
    critic: torch.nn.Module,
    observations: np.ndarray,
    *,
    device: torch.device,
    chunk_size: int = 65_536,
) -> np.ndarray:
    result = np.empty(observations.shape[0], dtype=np.float32)
    critic.eval()
    with torch.no_grad():
        for start in range(0, observations.shape[0], chunk_size):
            stop = min(start + chunk_size, observations.shape[0])
            states = torch.from_numpy(observations[start:stop]).to(
                device=device, dtype=torch.float32
            )
            result[start:stop] = (
                critic(states).squeeze(-1).detach().cpu().numpy().astype(np.float32)
            )
    if not np.isfinite(result).all():
        raise FloatingPointError("critic values contain NaN/Inf")
    return result


def _train_shared_critic(
    critic: torch.nn.Module,
    data: dict[str, np.ndarray],
    *,
    seed: int,
    steps: int,
    batch_size: int,
    learning_rate: float,
    gamma: float,
    expectile: float,
    device: torch.device,
) -> dict[str, Any]:
    critic.train()
    optimizer = torch.optim.Adam(critic.parameters(), lr=learning_rate)
    obs = torch.from_numpy(data["obs"]).to(device=device)
    rewards = torch.from_numpy(data["rews"]).to(device=device)
    next_obs = torch.from_numpy(data["next_obs"]).to(device=device)
    terminals = torch.from_numpy(data["terms"]).to(device=device)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    n = int(obs.shape[0])
    last_loss = float("nan")
    loss_sum = 0.0
    for step in range(1, steps + 1):
        indices = torch.randint(
            0, n, (batch_size,), generator=generator, device=device
        )
        states = obs.index_select(0, indices)
        batch_rewards = rewards.index_select(0, indices)
        next_states = next_obs.index_select(0, indices)
        batch_terminals = terminals.index_select(0, indices)
        values = critic(states).squeeze(-1)
        with torch.no_grad():
            next_values = critic(next_states).squeeze(-1)
            targets = (
                batch_rewards
                + gamma * next_values * (~batch_terminals).to(dtype=torch.float32)
            )
        error = targets - values
        weights = torch.where(
            error > 0,
            torch.full_like(error, expectile),
            torch.full_like(error, 1.0 - expectile),
        )
        loss = (weights * error.square()).mean()
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"critic loss is non-finite at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach().cpu())
        loss_sum += last_loss
    parameters_finite = all(
        bool(torch.isfinite(parameter).all()) for parameter in critic.parameters()
    )
    if not parameters_finite:
        raise FloatingPointError("critic parameters contain NaN/Inf")
    return {
        "steps": steps,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "expectile": expectile,
        "gamma": gamma,
        "last_loss": last_loss,
        "mean_loss": loss_sum / steps,
        "parameters_finite": parameters_finite,
        "fixed_horizon_not_convergence": True,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--dataset-sha256", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--critic-steps", type=int, default=DEFAULT_CRITIC_STEPS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    parser.add_argument("--expectile", type=float, default=DEFAULT_EXPECTILE)
    parser.add_argument("--gae-lambda", type=float, default=DEFAULT_GAE_LAMBDA)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--continuity-atol", type=float, default=1e-5)
    parser.add_argument("--continuity-rtol", type=float, default=1e-5)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.critic_steps <= 0 or args.batch_size <= 0:
        raise ValueError("critic steps and batch size must be positive")
    if not 0.0 < args.learning_rate:
        raise ValueError("learning rate must be positive")
    if not 0.0 <= args.gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")
    if not 0.0 <= args.gae_lambda <= 1.0:
        raise ValueError("gae lambda must be in [0, 1]")
    if not 0.0 < args.expectile < 1.0:
        raise ValueError("expectile must be in (0, 1)")

    dataset_path = Path(args.dataset_path).expanduser().resolve()
    if not dataset_path.is_file():
        raise FileNotFoundError(dataset_path)
    actual_dataset_sha256 = sha256_file(dataset_path)
    if actual_dataset_sha256 != args.dataset_sha256.lower():
        raise RuntimeError(
            "dataset SHA-256 mismatch: "
            f"{actual_dataset_sha256} != {args.dataset_sha256.lower()}"
        )
    output_dir = Path(args.output_dir).expanduser().resolve()
    manifest_path = output_dir / "ADVANTAGE_MANIFEST.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text())
        if args.resume and existing.get("status") == "complete":
            print(json.dumps(existing, indent=2, sort_keys=True))
            return 0
        raise RuntimeError("prepared output already exists; use --resume or a new path")
    output_dir.mkdir(parents=True, exist_ok=True)

    contract = CanonicalContract.load(args.contract)
    module, source_checks = load_verified_canonical_module(contract)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is unavailable")
    module.DEVICE = device

    data = _load_dataset(dataset_path, args.dataset_id)
    audit = audit_ordered_trajectories(
        data["obs"],
        data["next_obs"],
        data["terms"],
        data["touts"],
        atol=args.continuity_atol,
        rtol=args.continuity_rtol,
    )
    atomic_write_json(output_dir / "TRAJECTORY_AUDIT.json", audit.to_dict())

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    if not hasattr(module, "Critic"):
        raise RuntimeError("canonical agent module does not expose Critic")
    critic = module.Critic(int(data["obs"].shape[1])).to(device)
    train_summary = _train_shared_critic(
        critic,
        data,
        seed=args.seed,
        steps=args.critic_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        expectile=args.expectile,
        device=device,
    )
    critic_path = output_dir / "critic_final.pt"
    torch.save(
        {
            "schema_version": SCHEMA_VERSION,
            "preparer_version": PREPARER_VERSION,
            "dataset_id": args.dataset_id,
            "dataset_sha256": actual_dataset_sha256,
            "seed": args.seed,
            "state_dict": critic.state_dict(),
            "state_dim": int(data["obs"].shape[1]),
            "train_summary": train_summary,
        },
        critic_path,
    )
    critic_sha256 = sha256_file(critic_path)

    values = _critic_values(critic, data["obs"], device=device)
    next_values = _critic_values(critic, data["next_obs"], device=device)
    td_advantages = compute_td_advantage(
        data["rews"],
        values,
        next_values,
        data["terms"],
        gamma=args.gamma,
    )
    gae_advantages = compute_gae_numpy(
        data["rews"],
        values,
        next_values,
        data["terms"],
        data["touts"],
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
    )
    lambda_zero = compute_gae_numpy(
        data["rews"],
        values,
        next_values,
        data["terms"],
        data["touts"],
        gamma=args.gamma,
        gae_lambda=0.0,
    )
    lambda_zero_max_abs_error = float(
        np.max(np.abs(lambda_zero.astype(np.float64) - td_advantages))
    )
    if lambda_zero_max_abs_error > 1e-6:
        raise RuntimeError(
            "lambda=0 regression failed: "
            f"max_abs_error={lambda_zero_max_abs_error}"
        )

    torch_gae = compute_gae_torch(
        torch.from_numpy(data["rews"].astype(np.float64)),
        torch.from_numpy(values.astype(np.float64)),
        torch.from_numpy(next_values.astype(np.float64)),
        torch.from_numpy(data["terms"]),
        torch.from_numpy(data["touts"]),
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
    ).cpu().numpy()
    numpy_torch_max_abs_error = float(
        np.max(np.abs(torch_gae - gae_advantages.astype(np.float64)))
    )
    if numpy_torch_max_abs_error > 1e-6:
        raise RuntimeError(
            "NumPy/Torch GAE cross-check failed: "
            f"max_abs_error={numpy_torch_max_abs_error}"
        )

    advantage_path = output_dir / "advantages.npz"
    np.savez_compressed(
        advantage_path,
        td=td_advantages,
        gae=gae_advantages,
        values=values,
        next_values=next_values,
    )
    advantage_sha256 = sha256_file(advantage_path)
    diagnostics = advantage_diagnostics(td_advantages, gae_advantages)
    diagnostics["lambda_zero_max_abs_error"] = lambda_zero_max_abs_error
    diagnostics["numpy_torch_max_abs_error"] = numpy_torch_max_abs_error
    diagnostics["open_tail_excluded_fraction"] = (
        audit.open_tail_length / audit.transition_count
        if audit.transition_count
        else 0.0
    )
    atomic_write_json(output_dir / "ADVANTAGE_DIAGNOSTICS.json", diagnostics)
    atomic_write_json(output_dir / "CRITIC_TRAIN_SUMMARY.json", train_summary)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "preparer_version": PREPARER_VERSION,
        "status": "complete",
        "dataset_id": args.dataset_id,
        "dataset_path": str(dataset_path),
        "dataset_sha256": actual_dataset_sha256,
        "seed": args.seed,
        "transition_count": int(data["obs"].shape[0]),
        "state_dim": int(data["obs"].shape[1]),
        "action_dim": int(data["acts"].shape[1]),
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "critic": {
            "path": str(critic_path),
            "sha256": critic_sha256,
            "train_summary": train_summary,
        },
        "advantages": {
            "path": str(advantage_path),
            "sha256": advantage_sha256,
            "keys": ["td", "gae", "values", "next_values"],
            "diagnostics": diagnostics,
        },
        "trajectory_audit": audit.to_dict(),
        "source_checks": source_checks,
        "fixed_horizon_not_convergence": True,
        "behavior_trajectory_not_on_policy": True,
    }
    atomic_write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
