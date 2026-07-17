"""Shared frozen critic training and TD/GAE materialization."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from drpo.e7_sqexp_gae_contract import (
    ESTIMATORS, EXPERIMENT_ID, RUNNER_VERSION, SCIENTIFIC_STATUS,
    DatasetSpec, FrozenProtocol, array_sha256, atomic_json,
    compute_td_and_gae, critic_identity, prepared_advantage_identity,
    sha256_file, trajectory_end_mask, utc_now, validate_advantage_numerics,
    write_csv,
)
from drpo.e7_sqexp_gae_models import (
    CanonicalCritic, expectile_critic_loss, fit_observation_normalizer,
    normalize_observations,
)

def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = torch.device(name)
    if result.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return result


def _seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_data(dataset: DatasetSpec, max_transitions: int | None = None) -> Any:
    from drpo import e7_bench

    path = Path(dataset.path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if actual != dataset.sha256:
        raise ValueError(f"dataset SHA mismatch for {dataset.id}: {actual}")
    bench_spec = e7_bench.DatasetSpec(
        id=dataset.id,
        relative_path=path.name,
        sha256=dataset.sha256,
        format=dataset.format,
        env_id=dataset.env_id,
        dataset_family="d4rl_v2",
        score_protocol=dataset.score_protocol,
        reference_min_score=dataset.reference_min_score,
        reference_max_score=dataset.reference_max_score,
        formal_cell_eligible=False,
        provenance_note="EXT-H-E7-SQEXP-GAE-01 development source RunSpec",
    )
    return e7_bench.load_dataset(path, bench_spec, max_transitions)


def _split_episodes(episode_ids: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    from drpo import e7_hopper_q2 as q2

    return q2.split_episode_indices(episode_ids, seed, 0.8, 0.1)


def train_frozen_critic_and_prepare(
    *,
    dataset: DatasetSpec,
    seed: int,
    protocol: FrozenProtocol,
    source_run_spec_sha256: str,
    output_dir: Path,
    device_name: str,
) -> dict[str, Any]:
    """Train one critic, freeze it, then materialize TD and GAE arrays."""
    output_dir.mkdir(parents=True, exist_ok=True)
    identity = critic_identity(
        dataset=dataset,
        seed=seed,
        protocol=protocol,
        source_run_spec_sha256=source_run_spec_sha256,
    )
    _seed(seed)
    device = _device(device_name)
    data = _load_data(dataset)
    # Fail closed on ordered-trajectory and boundary semantics before the 100k critic budget.
    trajectory_end_mask(data.episode_ids, data.terminals, data.timeouts)
    split = _split_episodes(data.episode_ids, seed)
    obs_mean, obs_std = fit_observation_normalizer(data.observations, split["train"])
    observations = normalize_observations(data.observations, obs_mean, obs_std)
    next_observations = normalize_observations(data.next_observations, obs_mean, obs_std)
    critic = CanonicalCritic(observations.shape[1]).to(device)
    optimizer = torch.optim.Adam(critic.parameters(), lr=protocol.learning_rate)
    rng = np.random.default_rng(seed + 1000)
    rows: list[dict[str, Any]] = []
    failure_reason: str | None = None
    train_indices = np.asarray(split["train"], dtype=np.int64)
    for step in range(1, protocol.critic_steps + 1):
        batch = rng.choice(train_indices, size=protocol.batch_size, replace=True)
        obs_t = torch.as_tensor(observations[batch], device=device)
        next_t = torch.as_tensor(next_observations[batch], device=device)
        reward_t = torch.as_tensor(data.rewards[batch], dtype=torch.float32, device=device)
        terminal_t = torch.as_tensor(data.terminals[batch], dtype=torch.bool, device=device)
        with torch.no_grad():
            target = reward_t + protocol.gamma * critic(next_t) * (~terminal_t).float()
        value = critic(obs_t)
        error = target - value
        loss = expectile_critic_loss(error, protocol.expectile_tau)
        optimizer.zero_grad(set_to_none=True)
        loss_value = float(loss.detach().cpu())
        if not math.isfinite(loss_value):
            failure_reason = "nonfinite_critic_loss"
            break
        loss.backward()
        grad = math.sqrt(
            sum(float(parameter.grad.detach().square().sum().cpu()) for parameter in critic.parameters() if parameter.grad is not None)
        )
        if not math.isfinite(grad):
            failure_reason = "nonfinite_critic_gradient"
            break
        optimizer.step()
        if step % 2000 == 0 or step == protocol.critic_steps:
            rows.append({"step": step, "expectile_loss": loss_value, "raw_gradient_norm": grad})
    completed = failure_reason is None and bool(rows) and int(rows[-1]["step"]) == protocol.critic_steps
    checkpoint = output_dir / "frozen_critic.pt"
    torch.save(
        {
            "model": {key: value.detach().cpu() for key, value in critic.state_dict().items()},
            "obs_mean": obs_mean,
            "obs_std": obs_std,
            "seed": seed,
            "dataset_id": dataset.id,
            "critic_identity_sha256": identity,
            "fixed_budget_completed": completed,
            "failure_reason": failure_reason,
        },
        checkpoint,
    )
    checkpoint_sha = sha256_file(checkpoint)
    if not completed:
        atomic_json(
            output_dir / "WORKER_FAILED.json",
            {
                "worker": "critic_prepare",
                "critic_identity_sha256": identity,
                "fixed_budget_completed": False,
                "failure_reason": failure_reason or "critic_budget_incomplete",
            },
        )
        raise RuntimeError(f"critic job incomplete: {dataset.id} seed={seed}")
    critic.eval()
    values: list[np.ndarray] = []
    next_values: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(observations), 65536):
            stop = min(len(observations), start + 65536)
            values.append(critic(torch.as_tensor(observations[start:stop], device=device)).cpu().numpy())
            next_values.append(critic(torch.as_tensor(next_observations[start:stop], device=device)).cpu().numpy())
    value = np.concatenate(values).astype(np.float64)
    next_value = np.concatenate(next_values).astype(np.float64)
    arrays = compute_td_and_gae(
        rewards=data.rewards,
        values=value,
        next_values=next_value,
        terminals=data.terminals,
        timeouts=data.timeouts,
        episode_ids=data.episode_ids,
        gamma=protocol.gamma,
        gae_lambda=protocol.gae_lambda,
    )
    numerical = validate_advantage_numerics(
        arrays,
        episode_ids=data.episode_ids,
        terminals=data.terminals,
        timeouts=data.timeouts,
        gamma=protocol.gamma,
        gae_lambda=protocol.gae_lambda,
    )
    np.savez_compressed(output_dir / "episode_split.npz", **split)
    np.savez_compressed(
        output_dir / "prepared_advantages.npz",
        one_step_td=arrays.td_float32,
        behavior_gae=arrays.gae_float32,
        value=value.astype(np.float32),
        next_value=next_value.astype(np.float32),
        episode_ids=np.asarray(data.episode_ids, dtype=np.int64),
        terminals=np.asarray(data.terminals, dtype=np.bool_),
        timeouts=np.asarray(data.timeouts, dtype=np.bool_),
        obs_mean=obs_mean,
        obs_std=obs_std,
    )
    prepared_path = output_dir / "prepared_advantages.npz"
    prepared_sha = sha256_file(prepared_path)
    estimators: dict[str, Any] = {}
    with np.load(prepared_path) as saved:
        for estimator in ESTIMATORS:
            estimator_array = np.asarray(saved[estimator], dtype=np.float32)
            arrays_hash = array_sha256(estimator_array)
            estimators[estimator] = {
                "array_sha256": arrays_hash,
                "prepared_advantage_identity_sha256": prepared_advantage_identity(
                    critic_identity_sha256=identity,
                    critic_checkpoint_sha256=checkpoint_sha,
                    estimator=estimator,
                    protocol=protocol,
                    arrays_sha256=arrays_hash,
                ),
                "dtype": str(estimator_array.dtype),
                "normalization": "none",
                "clipping": "none",
            }
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "worker": "critic_prepare",
        "scientific_status": SCIENTIFIC_STATUS,
        "dataset_id": dataset.id,
        "dataset_sha256": dataset.sha256,
        "seed": seed,
        "critic_identity_sha256": identity,
        "source_run_spec_sha256": source_run_spec_sha256,
        "critic_checkpoint_sha256": checkpoint_sha,
        "prepared_file_sha256": prepared_sha,
        "fixed_budget_completed": True,
        "critic_frozen_before_actor": True,
        "actor_parameter_overlap": False,
        "advantage_normalization": "none",
        "advantage_clipping": "none",
        "boundary_contract": {
            "true_terminal": "no_bootstrap_and_stop_carry",
            "timeout": "bootstrap_and_stop_carry",
            "final_stored_nonterminal": "bootstrap_and_stop_carry",
            "terminal_timeout_overlap_allowed": False,
        },
        "numerical_audit": numerical,
        "estimators": estimators,
        "completed_utc": utc_now(),
    }
    write_csv(output_dir / "critic_metrics.csv", rows)
    atomic_json(output_dir / "prepared_advantage_manifest.json", manifest)
    atomic_json(output_dir / "WORKER_COMPLETE.json", manifest)
    return manifest


