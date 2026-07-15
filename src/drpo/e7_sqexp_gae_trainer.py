"""Canonical E7 actor loop for prepared TD/GAE advantages."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from drpo.e7_offline_gae import sha256_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--hdf5", required=True)
    parser.add_argument("--variant", choices=["iqlv_exp_rank"], required=True)
    parser.add_argument("--alpha", type=float, default=0.11)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--temp", type=float, default=5.0)
    parser.add_argument("--steps", type=int, default=1_000_000)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eval_interval", type=int, default=50_000)
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--ckpt_dir", default=None)
    parser.add_argument("--ckpt_interval", type=int, default=50_000)
    parser.add_argument("--last_pct", type=float, default=0.1)
    parser.add_argument("--advantage-manifest", required=True)
    parser.add_argument("--advantage-estimator", choices=["td", "gae"], required=True)
    return parser


def _load_prepared(
    manifest_path: Path,
    *,
    dataset: str,
    dataset_path: Path,
    seed: int,
    estimator: str,
    transition_count: int,
) -> tuple[np.ndarray, dict[str, Any], Path]:
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "complete":
        raise RuntimeError("advantage artifact is not complete")
    if manifest.get("dataset_id") != dataset:
        raise RuntimeError("advantage artifact dataset mismatch")
    if int(manifest.get("seed", -1)) != seed:
        raise RuntimeError("advantage artifact seed mismatch")
    dataset_sha256 = sha256_file(dataset_path)
    if manifest.get("dataset_sha256") != dataset_sha256:
        raise RuntimeError("advantage artifact dataset SHA-256 mismatch")
    if int(manifest.get("transition_count", -1)) != transition_count:
        raise RuntimeError("advantage artifact transition count mismatch")
    advantage_path = Path(manifest["advantages"]["path"]).expanduser().resolve()
    if not advantage_path.is_file():
        raise FileNotFoundError(advantage_path)
    if sha256_file(advantage_path) != manifest["advantages"]["sha256"]:
        raise RuntimeError("advantage NPZ SHA-256 mismatch")
    with np.load(advantage_path, allow_pickle=False) as arrays:
        if estimator not in arrays:
            raise RuntimeError(f"advantage NPZ has no estimator {estimator!r}")
        advantages = arrays[estimator].astype(np.float32, copy=True)
    if advantages.shape != (transition_count,):
        raise RuntimeError("prepared advantage shape mismatch")
    if not np.isfinite(advantages).all():
        raise FloatingPointError("prepared advantages contain NaN/Inf")
    critic_path = Path(manifest["critic"]["path"]).expanduser().resolve()
    if not critic_path.is_file():
        raise FileNotFoundError(critic_path)
    if sha256_file(critic_path) != manifest["critic"]["sha256"]:
        raise RuntimeError("critic checkpoint SHA-256 mismatch")
    return advantages, manifest, critic_path


def _parameter_snapshot(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in module.named_parameters()
    }


def _max_parameter_change(
    module: torch.nn.Module,
    reference: Mapping[str, torch.Tensor],
) -> float:
    current = dict(module.named_parameters())
    if set(current) != set(reference):
        raise RuntimeError("critic parameter names changed during actor training")
    maximum = 0.0
    for name, parameter in current.items():
        maximum = max(
            maximum,
            float((parameter.detach().cpu() - reference[name]).abs().max()),
        )
    return maximum


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.steps <= 0 or args.batch <= 0:
        raise ValueError("steps and batch must be positive")
    if args.eval_interval <= 0 or args.steps % args.eval_interval != 0:
        raise ValueError("eval_interval must be positive and divide steps")
    if not 0.0 < args.last_pct <= 1.0:
        raise ValueError("last_pct must be in (0, 1]")

    import agents
    from d4rl_common import normalize_score
    from d4rl_common.train_loop import evaluate, load_hdf5

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is unavailable")
    agents.DEVICE = device
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    dataset_path = Path(args.hdf5).expanduser().resolve()
    data = load_hdf5(dataset_path, dataset_name=args.dataset)
    n = int(data["obs"].shape[0])
    state_dim = int(data["obs"].shape[1])
    action_dim = int(data["acts"].shape[1])
    advantages, prepared_manifest, critic_path = _load_prepared(
        Path(args.advantage_manifest).expanduser().resolve(),
        dataset=args.dataset,
        dataset_path=dataset_path,
        seed=args.seed,
        estimator=args.advantage_estimator,
        transition_count=n,
    )

    agent_class = getattr(agents, "SNA2C_IQLV_ExpRankAgent")
    agent = agent_class(
        state_dim,
        action_dim,
        lr=args.lr,
        alpha=args.alpha,
        tau=args.tau,
        T=args.temp,
    )
    try:
        checkpoint = torch.load(critic_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(critic_path, map_location=device)
    if int(checkpoint.get("state_dim", -1)) != state_dim:
        raise RuntimeError("critic checkpoint state dimension mismatch")
    agent.critic.load_state_dict(checkpoint["state_dict"])
    agent.critic.eval()
    for parameter in agent.critic.parameters():
        parameter.requires_grad_(False)
    actor_parameter_ids = {id(parameter) for parameter in agent.actor.parameters()}
    critic_parameter_ids = {id(parameter) for parameter in agent.critic.parameters()}
    if actor_parameter_ids & critic_parameter_ids:
        raise RuntimeError("actor and critic parameter sets overlap")
    critic_reference = _parameter_snapshot(agent.critic)

    tensors = {
        "obs": torch.from_numpy(data["obs"]).to(device=device, dtype=torch.float32),
        "acts": torch.from_numpy(data["acts"]).to(device=device, dtype=torch.float32),
        "rews": torch.from_numpy(data["rews"]).to(device=device, dtype=torch.float32),
        "next_obs": torch.from_numpy(data["next_obs"]).to(
            device=device, dtype=torch.float32
        ),
        "terms": torch.from_numpy(data["terms"]).to(device=device, dtype=torch.bool),
        "adv": torch.from_numpy(advantages).to(device=device, dtype=torch.float32),
    }
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)

    output_dir = Path(args.out_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    branch_id = os.environ.get("DRPO_E7_BRANCH_ID", "unbound_branch")
    algo_label = f"SQEXP-{args.advantage_estimator.upper()}-{branch_id}"
    run_name = (
        f"{args.dataset}_sqexp_{args.advantage_estimator}_seed{args.seed}"
        .replace("/", "_")
    )
    history: dict[str, list[float] | list[int]] = {
        "steps": [],
        "score": [],
        "actor_log_std_min": [],
        "actor_log_std_max": [],
        "actor_sigma_min": [],
        "actor_sigma_max": [],
        "lower_log_std_boundary_fraction": [],
        "upper_log_std_boundary_fraction": [],
        "critic_max_abs_parameter_change": [],
    }
    support_probe_count = min(n, 4096)
    support_probe_states = tensors["obs"][:support_probe_count]
    log_path = output_dir / f"{run_name}.log"
    checkpoint_dir = (
        None if args.ckpt_dir is None else Path(args.ckpt_dir).expanduser().resolve()
    )
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_start = int(args.steps * (1.0 - args.last_pct))
    start_time = time.time()

    for step in range(1, args.steps + 1):
        indices = torch.randint(
            0, n, (args.batch,), generator=generator, device=device
        )
        agent.update(
            tensors["obs"].index_select(0, indices),
            tensors["acts"].index_select(0, indices),
            tensors["rews"].index_select(0, indices),
            tensors["next_obs"].index_select(0, indices),
            tensors["terms"].index_select(0, indices),
            tensors["adv"].index_select(0, indices),
        )
        if (
            checkpoint_dir is not None
            and step >= checkpoint_start
            and step % args.ckpt_interval == 0
        ):
            torch.save(
                {
                    "actor": agent.actor.state_dict(),
                    "step": step,
                    "dataset": args.dataset,
                    "seed": args.seed,
                    "advantage_estimator": args.advantage_estimator,
                    "critic_sha256": prepared_manifest["critic"]["sha256"],
                },
                checkpoint_dir / f"step_{step:07d}.pt",
            )
        if step % args.eval_interval == 0:
            critic_change = _max_parameter_change(agent.critic, critic_reference)
            if critic_change != 0.0:
                raise RuntimeError(
                    f"frozen critic changed during actor training: {critic_change}"
                )
            raw = evaluate(
                agent, args.dataset, n=args.eval_episodes, seed=args.seed
            )
            score = normalize_score(args.dataset, raw)
            with torch.no_grad():
                _, probe_log_std = agent.actor(support_probe_states)
                probe_log_std = probe_log_std.detach()
                probe_sigma = probe_log_std.exp()
                log_std_min = float(probe_log_std.min().cpu())
                log_std_max = float(probe_log_std.max().cpu())
                sigma_min = float(probe_sigma.min().cpu())
                sigma_max = float(probe_sigma.max().cpu())
                lower_fraction = float(
                    (probe_log_std <= -5.0 + 1e-6).float().mean().cpu()
                )
                upper_fraction = float(
                    (probe_log_std >= 2.0 - 1e-6).float().mean().cpu()
                )
                if not all(
                    np.isfinite(value)
                    for value in (
                        log_std_min,
                        log_std_max,
                        sigma_min,
                        sigma_max,
                        lower_fraction,
                        upper_fraction,
                        critic_change,
                    )
                ):
                    raise FloatingPointError(
                        "actor support or critic diagnostics contain NaN/Inf"
                    )
            history["steps"].append(step)
            history["score"].append(float(score))
            history["actor_log_std_min"].append(log_std_min)
            history["actor_log_std_max"].append(log_std_max)
            history["actor_sigma_min"].append(sigma_min)
            history["actor_sigma_max"].append(sigma_max)
            history["lower_log_std_boundary_fraction"].append(lower_fraction)
            history["upper_log_std_boundary_fraction"].append(upper_fraction)
            history["critic_max_abs_parameter_change"].append(critic_change)
            elapsed = time.time() - start_time
            speed = step / elapsed if elapsed > 0 else 0.0
            line = (
                f"[{args.dataset}|{args.advantage_estimator}|s{args.seed}] "
                f"step={step}/{args.steps} speed={speed:.1f}/s "
                f"raw={raw:.6f} norm={score:.6f} critic_delta={critic_change:.3g}"
            )
            print(line, flush=True)
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    if not history["score"]:
        raise RuntimeError("actor run completed without evaluation")
    final_critic_change = _max_parameter_change(agent.critic, critic_reference)
    if final_critic_change != 0.0:
        raise RuntimeError(
            f"frozen critic changed at terminal audit: {final_critic_change}"
        )
    summary = {
        "schema_version": 1,
        "dataset": args.dataset,
        "variant": args.variant,
        "seed": args.seed,
        "steps": args.steps,
        "batch": args.batch,
        "learning_rate": args.lr,
        "alpha": args.alpha,
        "tau": args.tau,
        "temperature": args.temp,
        "advantage_estimator": args.advantage_estimator,
        "advantage_manifest": str(
            Path(args.advantage_manifest).expanduser().resolve()
        ),
        "advantage_npz_sha256": prepared_manifest["advantages"]["sha256"],
        "critic_sha256": prepared_manifest["critic"]["sha256"],
        "critic_frozen": True,
        "critic_immutability_verified": True,
        "critic_max_abs_parameter_change": final_critic_change,
        "actor_critic_parameter_sets_disjoint": True,
        "behavior_trajectory_not_on_policy": True,
        "final_norm": float(history["score"][-1]),
        "final_score": float(history["score"][-1]),
        "score_type": "norm",
        "history": history,
        "support_probe_count": support_probe_count,
        "support_boundary_diagnostics_recorded": True,
        "support_boundary_event_rule_registered": False,
        "task_performance_collapse_rule_registered": False,
        "nan_inf_numerical_failure": False,
        "fixed_horizon_not_convergence": True,
        "algo_label": algo_label,
    }
    summary_path = output_dir / f"{run_name}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"Summary -> {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
