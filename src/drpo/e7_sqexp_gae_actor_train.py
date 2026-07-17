"""One frozen-advantage actor branch execution."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any
import numpy as np
import torch
from drpo.e7_sqexp_gae_contract import (
    EXPERIMENT_ID, SCIENTIFIC_STATUS, ActorBranch, DatasetSpec, FrozenProtocol,
    array_sha256, atomic_json, canonical_hash, critic_identity, prepared_advantage_identity,
    sha256_file, utc_now, write_csv,
)
from drpo.e7_sqexp_gae_models import CanonicalActor, OldPolicyCadence, actor_objective, normalize_observations
from drpo.e7_sqexp_gae_preparation import _device, _load_data, _seed
from drpo.e7_sqexp_gae_evaluation import rollout

def train_actor_branch(
    *,
    branch: ActorBranch,
    dataset: DatasetSpec,
    protocol: FrozenProtocol,
    critic_dir: Path,
    output_dir: Path,
    device_name: str,
    source_run_spec_sha256: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = critic_dir / "prepared_advantage_manifest.json"
    prepared_path = critic_dir / "prepared_advantages.npz"
    checkpoint_path = critic_dir / "frozen_critic.pt"
    manifest = json.loads(manifest_path.read_text())
    critic_sha_before = sha256_file(checkpoint_path)
    if critic_sha_before != manifest["critic_checkpoint_sha256"]:
        raise RuntimeError("critic checkpoint identity changed before actor launch")
    if sha256_file(prepared_path) != manifest["prepared_file_sha256"]:
        raise RuntimeError("prepared advantage file identity changed")
    expected_critic = critic_identity(
        dataset=dataset,
        seed=branch.seed,
        protocol=protocol,
        source_run_spec_sha256=source_run_spec_sha256,
    )
    if manifest.get("critic_identity_sha256") != expected_critic:
        raise RuntimeError("actor branch critic identity does not match frozen protocol")
    if manifest.get("source_run_spec_sha256") != source_run_spec_sha256:
        raise RuntimeError("actor branch source RunSpec identity changed")
    if manifest.get("dataset_id") != dataset.id or int(manifest.get("seed", -1)) != branch.seed:
        raise RuntimeError("actor branch points to a different dataset/seed critic")
    data = _load_data(dataset)
    with np.load(prepared_path) as saved:
        advantages = np.asarray(saved[branch.estimator], dtype=np.float32)
        obs_mean = np.asarray(saved["obs_mean"], dtype=np.float32)
        obs_std = np.asarray(saved["obs_std"], dtype=np.float32)
    estimator_manifest = manifest["estimators"][branch.estimator]
    actor_array_sha = array_sha256(advantages)
    if actor_array_sha != estimator_manifest["array_sha256"]:
        raise RuntimeError("actor-facing advantage array hash mismatch")
    expected_prepared_identity = prepared_advantage_identity(
        critic_identity_sha256=expected_critic,
        critic_checkpoint_sha256=critic_sha_before,
        estimator=branch.estimator,
        protocol=protocol,
        arrays_sha256=actor_array_sha,
    )
    if estimator_manifest.get("prepared_advantage_identity_sha256") != expected_prepared_identity:
        raise RuntimeError("prepared advantage identity does not match frozen critic/estimator")
    if advantages.dtype != np.float32:
        raise TypeError("actor-facing advantages must be float32")
    if len(advantages) != data.size:
        raise ValueError("prepared advantage length does not match dataset")
    split_npz = np.load(critic_dir / "episode_split.npz")
    train_indices = np.asarray(split_npz["train"], dtype=np.int64)
    observations = normalize_observations(data.observations, obs_mean, obs_std)
    _seed(branch.seed)
    device = _device(device_name)
    actor = CanonicalActor(observations.shape[1], data.actions.shape[1]).to(device)
    old_actor = copy.deepcopy(actor).to(device) if branch.actor_mode == "ppo_clip_k4" else None
    cadence = OldPolicyCadence(protocol.ppo_old_policy_cadence)
    optimizer = torch.optim.Adam(actor.parameters(), lr=protocol.learning_rate)
    rng = np.random.default_rng(branch.seed + 5000)
    audit_rng = np.random.default_rng(branch.seed + 7000)
    audit_indices = audit_rng.choice(
        train_indices, size=min(16384, len(train_indices)), replace=False
    )
    rows: list[dict[str, Any]] = []
    failure_reason: str | None = None
    initial_state_hash = canonical_hash(
        {key: array_sha256(value.detach().cpu().numpy()) for key, value in actor.state_dict().items()}
    )
    for step in range(1, protocol.actor_steps + 1):
        if old_actor is not None and cadence.should_refresh_before(step):
            cadence.refresh(old_actor, actor, step)
        batch = rng.choice(train_indices, size=protocol.batch_size, replace=True)
        obs_t = torch.as_tensor(observations[batch], device=device)
        action_t = torch.as_tensor(data.actions[batch], dtype=torch.float32, device=device)
        advantage_t = torch.as_tensor(advantages[batch], dtype=torch.float32, device=device)
        loss, diagnostics = actor_objective(
            actor=actor,
            old_actor=old_actor,
            observations=obs_t,
            actions=action_t,
            advantages=advantage_t,
            actor_mode=branch.actor_mode,
            control_id=branch.control_id,
            clip_epsilon=protocol.ppo_clip_epsilon,
        )
        optimizer.zero_grad(set_to_none=True)
        if not math.isfinite(float(loss.detach().cpu())):
            failure_reason = "nonfinite_actor_loss"
            break
        loss.backward()
        grad_norm = math.sqrt(
            sum(float(parameter.grad.detach().square().sum().cpu()) for parameter in actor.parameters() if parameter.grad is not None)
        )
        if not math.isfinite(grad_norm):
            failure_reason = "nonfinite_actor_gradient"
            break
        optimizer.step()
        if step % protocol.evaluation_interval == 0 or step == protocol.actor_steps:
            rollout_metrics = rollout(
                actor,
                dataset,
                obs_mean,
                obs_std,
                protocol.evaluation_episodes,
                branch.seed * 1_000_000 + step,
                device,
            )
            with torch.no_grad():
                audit_obs = torch.as_tensor(observations[audit_indices], device=device)
                audit_mean, audit_log_std = actor(audit_obs)
                mean_boundary_fraction = float(
                    (audit_mean.abs() >= 0.99).any(dim=-1).float().mean().cpu()
                )
                log_std_min = float(audit_log_std.min().cpu())
                log_std_max = float(audit_log_std.max().cpu())
            row = {
                "step": step,
                "raw_gradient_norm": grad_norm,
                "mean_boundary_fraction": mean_boundary_fraction,
                "log_std_min": log_std_min,
                "log_std_max": log_std_max,
                **diagnostics,
                **rollout_metrics,
            }
            rows.append(row)
    final_step = int(rows[-1]["step"]) if rows else 0
    fixed_budget_completed = failure_reason is None and final_step == protocol.actor_steps
    critic_sha_after = sha256_file(checkpoint_path)
    critic_unchanged = critic_sha_after == critic_sha_before
    if not critic_unchanged:
        raise RuntimeError("frozen critic checkpoint changed during actor training")
    task_state = "not_adjudicated_no_frozen_task_collapse_threshold"
    final_mean_boundary_fraction = (
        float(rows[-1]["mean_boundary_fraction"]) if rows else float("nan")
    )
    support_event = bool(
        (rows and final_mean_boundary_fraction >= 0.10)
        or actor.log_std.detach().min().cpu() <= -5.0 + 1e-7
        or actor.log_std.detach().max().cpu() >= 2.0 - 1e-7
    )
    numerical_nonfinite = failure_reason is not None
    terminal_state = (
        "nan_inf_numerical_failure"
        if numerical_nonfinite
        else "support_or_variance_boundary_event"
        if support_event
        else "finite_fixed_horizon_terminal"
    )
    late_rows = [row for row in rows if int(row["step"]) >= protocol.late_window_start]
    terminal = {
        "experiment_id": EXPERIMENT_ID,
        "branch_id": branch.id,
        "scientific_status": SCIENTIFIC_STATUS,
        "fixed_budget_completed": fixed_budget_completed,
        "fixed_horizon_is_convergence": False,
        "terminal_state": terminal_state,
        "task_performance_collapse": task_state,
        "support_or_variance_boundary_event": support_event,
        "support_boundary_threshold": 0.99,
        "support_boundary_fraction_threshold": 0.10,
        "final_mean_boundary_fraction": final_mean_boundary_fraction,
        "variance_boundary": {"log_std_min": -5.0, "log_std_max": 2.0},
        "nan_inf_numerical_failure": numerical_nonfinite,
        "failure_reason": failure_reason,
        "critic_checkpoint_sha256_before": critic_sha_before,
        "critic_checkpoint_sha256_after": critic_sha_after,
        "critic_immutability_verified": critic_unchanged,
        "actor_and_critic_parameters_disjoint": True,
        "critic_loaded_into_actor_optimizer": False,
        "prepared_advantage_identity_sha256": estimator_manifest[
            "prepared_advantage_identity_sha256"
        ],
        "advantage_dtype": "float32",
        "advantage_normalization": "none",
        "advantage_clipping": "none",
        "ppo_reference_refresh_count": cadence.refresh_count,
        "ppo_reference_first_refresh_step": cadence.first_refresh_step,
        "ppo_reference_last_refresh_step": cadence.last_refresh_step,
        "ppo_reference_cadence": protocol.ppo_old_policy_cadence,
        "actor_initial_state_sha256": initial_state_hash,
        "late_window_start": protocol.late_window_start,
        "late_window_rows": len(late_rows),
        "late_window_normalized_return_mean": (
            float(np.mean([row["normalized_return"] for row in late_rows]))
            if late_rows
            else float("nan")
        ),
        "best_normalized_return": (
            float(np.nanmax([row["normalized_return"] for row in rows])) if rows else float("nan")
        ),
        "final_normalized_return": (
            float(rows[-1]["normalized_return"]) if rows else float("nan")
        ),
        "completed_utc": utc_now(),
    }
    torch.save(
        {
            "model": actor.state_dict(),
            "branch_id": branch.id,
            "step": final_step,
            "fixed_budget_completed": fixed_budget_completed,
            "failure_reason": failure_reason,
        },
        output_dir / "terminal_actor.pt",
    )
    write_csv(output_dir / "metrics.csv", rows)
    atomic_json(output_dir / "terminal_audit.json", terminal)
    atomic_json(output_dir / "summary.json", terminal)
    atomic_json(output_dir / "WORKER_COMPLETE.json", terminal)
    return terminal


