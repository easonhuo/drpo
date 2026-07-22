#!/usr/bin/env python3
"""Repository entrypoint for E7 benchmark and D4RL-9 Figure-1 diagnostics.

Existing E7 benchmark commands are delegated unchanged to ``drpo.e7_bench``.
The ``figure1-d4rl9`` command runs only the learned-critic -> Positive-only ->
negative-transition gradient diagnostic and saves every selected plot point.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import dataclasses
import json
import math
import multiprocessing
import os
import sys
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

FIG1_ID = "EXT-H-E7-FIG1-D4RL9-GRADIENT-COVERAGE-01"
FIG1_VERSION = "1.0.0-d4rl9-point-provenance"
D4RL9 = (
    "halfcheetah-medium-v2",
    "halfcheetah-medium-replay-v2",
    "halfcheetah-medium-expert-v2",
    "hopper-medium-v2",
    "hopper-medium-replay-v2",
    "hopper-medium-expert-v2",
    "walker2d-medium-v2",
    "walker2d-medium-replay-v2",
    "walker2d-medium-expert-v2",
)
POINT_FIELDS = (
    "dataset_id",
    "dataset_sha256",
    "seed",
    "distance_bin",
    "advantage_bin",
    "match_group_id",
    "target_abs_advantage",
    "relative_advantage_error",
    "transition_index",
    "episode_id",
    "split",
    "advantage",
    "abs_advantage",
    "standardized_distance",
    "mean_score_norm",
    "raw_log_scale_score_norm",
    "corrected_q_xi",
    "joint_output_score_norm",
    "log_scale_to_mean_ratio",
    "raw_action_distance",
    "pre_squash_distance",
    "full_parameter_gradient_norm",
    "distance_bin_left",
    "distance_bin_right",
    "advantage_bin_left",
    "advantage_bin_right",
    "critic_checkpoint_sha256",
    "positive_actor_checkpoint_sha256",
    "runspec_sha256",
    "base_config_sha256",
    "source_commit_sha",
    "runner_sha256",
)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    return value


def _list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    return value


def load_runspec(path: str | Path) -> dict[str, Any]:
    from drpo import e7_hopper_q2 as q2
    import yaml

    source = Path(path).expanduser().resolve()
    raw = _mapping(yaml.safe_load(source.read_text()), "runspec")
    frozen = {
        "experiment_id": FIG1_ID,
        "runner_version": FIG1_VERSION,
        "execution_class": "pilot",
        "scientific_status": "pilot",
        "formal_launch_allowed": False,
    }
    for key, expected in frozen.items():
        if raw.get(key) != expected:
            raise ValueError(f"{key} must be {expected!r}")
    datasets = _list(raw.get("datasets"), "datasets")
    if tuple(_mapping(item, "dataset").get("id") for item in datasets) != D4RL9:
        raise ValueError("runspec must retain the exact ordered D4RL-v2 3x3 cells")
    for dataset in datasets:
        if dataset.get("format") != "legacy_d4rl_hdf5":
            raise ValueError(f"{dataset['id']} must use legacy_d4rl_hdf5")
        expected = dataset.get("expected_sha256")
        if expected not in (None, "auto_capture_before_training"):
            if len(str(expected)) != 64:
                raise ValueError(f"invalid expected_sha256 for {dataset['id']}")
    protocol = _mapping(raw.get("protocol"), "protocol")
    if tuple(protocol.get("seeds", ())) != tuple(range(100, 110)):
        raise ValueError("seeds must remain 100..109")
    if protocol.get("canonical_critic_seed") != 100:
        raise ValueError("canonical_critic_seed must remain 100")
    budget = _mapping(protocol.get("budget"), "protocol.budget")
    required_budget = {
        "max_transitions": None,
        "critic_steps": 100000,
        "critic_eval_interval": 2000,
        "positive_actor_steps": 100000,
        "positive_actor_eval_interval": 5000,
        "audit_sample_size": 16384,
    }
    for key, expected in required_budget.items():
        if budget.get(key) != expected:
            raise ValueError(f"protocol.budget.{key} must remain {expected!r}")
    probe = _mapping(protocol.get("probe"), "protocol.probe")
    required_probe = {
        "distance_bins": 7,
        "advantage_bins": 8,
        "gradient_points_per_seed": 64,
        "matching_rule": "nearest_abs_advantage_targets_shared_across_distance_bins",
        "relative_advantage_tolerance": 0.05,
        "save_every_selected_point": True,
    }
    for key, expected in required_probe.items():
        if probe.get(key) != expected:
            raise ValueError(f"protocol.probe.{key} must remain {expected!r}")
    if protocol.get("train_method_branches") is not False:
        raise ValueError("this diagnostic must not train E7 method branches")
    execution = _mapping(raw.get("execution"), "execution")
    if execution.get("require_clean_worktree") is not True:
        raise ValueError("execution.require_clean_worktree must remain true")
    if int(execution.get("dataset_workers", 0)) < 1:
        raise ValueError("execution.dataset_workers must be positive")
    raw["_path"] = str(source)
    raw["_sha256"] = q2.sha256_file(source)
    return raw


def _dataset_spec(item: dict[str, Any], sha256: str) -> Any:
    from drpo import e7_bench

    return e7_bench.DatasetSpec(
        id=str(item["id"]),
        relative_path=str(item["relative_path"]),
        sha256=sha256,
        format=str(item["format"]),
        env_id=str(item["env_id"]),
        dataset_family="d4rl_v2",
        score_protocol="gradient_diagnostic_no_rollout",
        reference_min_score=None,
        reference_max_score=None,
        formal_cell_eligible=True,
        provenance_note=str(item["provenance_note"]),
    )


def lock_datasets(runspec: dict[str, Any], dataset_root: str | Path) -> dict[str, Any]:
    from drpo import e7_bench
    from drpo import e7_hopper_q2 as q2

    root = Path(dataset_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"dataset root does not exist: {root}")
    rows = []
    for item in runspec["datasets"]:
        path = (root / str(item["relative_path"])).resolve()
        path.relative_to(root)
        if not path.is_file():
            raise FileNotFoundError(f"missing D4RL dataset: {path}")
        sha256 = q2.sha256_file(path)
        expected = item.get("expected_sha256")
        if expected not in (None, "auto_capture_before_training") and expected != sha256:
            raise RuntimeError(
                f"dataset SHA mismatch for {item['id']}: expected {expected}, got {sha256}"
            )
        spec = _dataset_spec(item, sha256)
        sample = e7_bench.load_dataset(path, spec, max_transitions=32)
        rows.append(
            {
                **{key: item[key] for key in ("id", "relative_path", "format", "env_id")},
                "resolved_path": str(path),
                "sha256": sha256,
                "size_bytes": path.stat().st_size,
                "observation_dim": int(sample.observations.shape[1]),
                "action_dim": int(sample.actions.shape[1]),
                "provenance_note": item["provenance_note"],
            }
        )
    payload = {
        "experiment_id": FIG1_ID,
        "runner_version": FIG1_VERSION,
        "runspec_sha256": runspec["_sha256"],
        "capture_stage": "all_nine_files_before_any_training",
        "datasets": rows,
    }
    payload["dataset_lock_sha256"] = _payload_sha256(payload)
    return payload


def _payload_sha256(payload: Any) -> str:
    import hashlib

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _adapt_config(runspec: dict[str, Any], dataset: dict[str, Any]) -> tuple[Any, Any]:
    from drpo import e7_hopper_q2 as q2

    base_path = (ROOT / str(runspec["base_config_path"])).resolve()
    base = q2.load_config(base_path)
    budget = runspec["protocol"]["budget"]
    probe = runspec["protocol"]["probe"]
    mode = q2.ModeConfig(
        max_transitions=None,
        seeds=tuple(runspec["protocol"]["seeds"]),
        canonical_critic_seed=100,
        critic_max_steps=budget["critic_steps"],
        critic_min_steps=budget["critic_steps"] // 10,
        critic_eval_interval=budget["critic_eval_interval"],
        positive_max_steps=budget["positive_actor_steps"],
        positive_min_steps=budget["positive_actor_steps"] // 10,
        actor_eval_interval=budget["positive_actor_eval_interval"],
        branch_max_steps=0,
        branch_min_steps=0,
        matched_pairs=probe["gradient_points_per_seed"],
        audit_sample_size=budget["audit_sample_size"],
        rollout_episodes=0,
        final_rollout_episodes=0,
        rollout_eval_interval=0,
    )
    return dataclasses.replace(
        base,
        experiment_id=FIG1_ID,
        dataset_basename=Path(dataset["resolved_path"]).name,
        dataset_sha256=dataset["sha256"],
        rollout_dataset_id=dataset["id"],
        env_id=dataset["env_id"],
        normalized_score_percent=False,
        normalized_score_reference_min=0.0,
        normalized_score_reference_max=1.0,
        formal_rollout_required=False,
        pilot_rollout_required=False,
        pilot=mode,
        formal=mode,
    ), mode


def _bin_ids(values: Any, edges: Any) -> Any:
    import numpy as np

    return np.clip(np.searchsorted(edges[1:-1], values, side="right"), 0, len(edges) - 2)


def select_matched_points(
    negative_indices: Any,
    distances: Any,
    advantages: Any,
    *,
    distance_bins: int,
    advantage_bins: int,
    point_budget: int,
    minimum_per_bin: int,
    tolerance: float,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Match each distance bin to shared nearest absolute-advantage targets."""
    import numpy as np

    negative_indices = np.asarray(negative_indices, dtype=np.int64)
    magnitude = np.abs(np.asarray(advantages, dtype=np.float64))
    neg_distance = np.asarray(distances[negative_indices], dtype=np.float64)
    neg_magnitude = magnitude[negative_indices]
    distance_edges = np.unique(
        np.quantile(neg_distance, np.linspace(0.0, 1.0, distance_bins + 1))
    )
    advantage_edges = np.unique(
        np.quantile(neg_magnitude, np.linspace(0.0, 1.0, advantage_bins + 1))
    )
    if len(distance_edges) != distance_bins + 1 or len(advantage_edges) < 3:
        raise RuntimeError("distance or advantage quantiles collapsed")
    distance_id = _bin_ids(neg_distance, distance_edges)
    pools = [negative_indices[distance_id == index] for index in range(distance_bins)]
    if any(len(pool) == 0 for pool in pools):
        raise RuntimeError("one or more distance bins are empty")
    common_left = max(float(magnitude[pool].min()) for pool in pools)
    common_right = min(float(magnitude[pool].max()) for pool in pools)
    anchors = pools[0][
        (magnitude[pools[0]] >= common_left) & (magnitude[pools[0]] <= common_right)
    ]
    if not common_left < common_right or len(anchors) == 0:
        raise RuntimeError("distance bins lack common absolute-advantage support")
    position = {int(index): offset for offset, index in enumerate(negative_indices)}
    advantage_id = _bin_ids(neg_magnitude, advantage_edges)
    strata = [[] for _ in range(len(advantage_edges) - 1)]
    rng = np.random.default_rng(seed)
    for anchor in anchors:
        strata[int(advantage_id[position[int(anchor)]])].append(int(anchor))
    for members in strata:
        rng.shuffle(members)
    ordered = []
    while any(strata):
        for members in strata:
            if members:
                ordered.append(members.pop())
    target_groups = max(1, point_budget // distance_bins)
    used = [set() for _ in pools]
    groups = []
    for anchor in ordered:
        if len(groups) >= target_groups:
            break
        target = float(magnitude[anchor])
        group = [anchor]
        errors = [0.0]
        for distance_bin in range(1, distance_bins):
            available = [
                int(index)
                for index in pools[distance_bin]
                if int(index) not in used[distance_bin]
            ]
            if not available:
                group = []
                break
            candidate = min(
                available, key=lambda index: abs(float(magnitude[index]) - target)
            )
            error = abs(float(magnitude[candidate]) - target) / max(target, 1e-8)
            if error > tolerance:
                group = []
                break
            group.append(candidate)
            errors.append(error)
        if not group:
            continue
        for distance_bin, index in enumerate(group):
            used[distance_bin].add(index)
        groups.append((target, group, errors))
    if len(groups) < minimum_per_bin:
        raise RuntimeError(
            f"only {len(groups)} matched groups found; minimum is {minimum_per_bin}"
        )
    rows = []
    for group_id, (target, group, errors) in enumerate(groups):
        advantage_bin = int(_bin_ids(np.asarray([target]), advantage_edges)[0])
        for distance_bin, (index, error) in enumerate(zip(group, errors)):
            rows.append((index, distance_bin, advantage_bin, group_id, target, error))
    rows.sort(key=lambda row: (row[1], row[3]))
    selection = {
        "indices": np.asarray([row[0] for row in rows], dtype=np.int64),
        "distance_bin": np.asarray([row[1] for row in rows], dtype=np.int64),
        "advantage_bin": np.asarray([row[2] for row in rows], dtype=np.int64),
        "match_group_id": np.asarray([row[3] for row in rows], dtype=np.int64),
        "target_abs_advantage": np.asarray([row[4] for row in rows], dtype=np.float64),
        "relative_advantage_error": np.asarray([row[5] for row in rows], dtype=np.float64),
        "distance_edges": distance_edges,
        "advantage_edges": advantage_edges,
    }
    errors = selection["relative_advantage_error"]
    audit = {
        "matching_rule": "nearest_abs_advantage_targets_shared_across_distance_bins",
        "relative_advantage_tolerance": tolerance,
        "requested_point_budget": point_budget,
        "points_per_distance_bin": len(groups),
        "total_points": len(rows),
        "common_abs_advantage_support": [common_left, common_right],
        "mean_relative_advantage_error": float(errors.mean()),
        "max_relative_advantage_error": float(errors.max()),
        "distance_edges": distance_edges,
        "advantage_edges": advantage_edges,
    }
    return selection, audit


def _configure_threads(count: int) -> None:
    import torch

    count = max(1, int(count))
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = str(count)
    torch.set_num_threads(count)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def _critic_context(
    dataset: dict[str, Any],
    runspec: dict[str, Any],
    data: Any,
    config: Any,
    mode: Any,
    device: Any,
    output_dir: Path,
    resume: bool,
    identity: dict[str, str],
) -> dict[str, Any]:
    from drpo import e7_hopper_q2 as q2
    import numpy as np

    marker = output_dir / "CRITIC_COMPLETE.json"
    if marker.is_file():
        if not resume:
            raise RuntimeError(f"critic exists for {dataset['id']}; pass --resume")
        complete = json.loads(marker.read_text())
        if complete["identity_sha256"] != _payload_sha256(identity):
            raise RuntimeError(f"critic identity mismatch for {dataset['id']}")
        split_npz = np.load(output_dir / "episode_split.npz")
        split = {key: split_npz[key].astype(np.int64) for key in split_npz.files}
        norm = np.load(output_dir / "observation_normalizer.npz")
        obs_norm = q2.Normalizer(
            norm["mean"].astype(np.float32), norm["std"].astype(np.float32)
        )
        advantages = np.load(
            output_dir / "frozen_advantage" / "frozen_advantages.npz"
        )["advantage"]
        return {
            "split": split,
            "obs_norm": obs_norm,
            "advantages": advantages,
            "complete": complete,
        }
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"partial critic output without marker: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    q2.seed_everything(mode.canonical_critic_seed)
    split = q2.split_episode_indices(
        data.episode_ids,
        mode.canonical_critic_seed,
        config.train_fraction,
        config.validation_fraction,
    )
    obs_norm = q2.Normalizer.fit(data.observations[split["train"]])
    returns = q2.discounted_returns(
        data.rewards, data.terminals, data.timeouts, config.gamma
    )
    critic, target_norm, audit = q2.train_critic(
        data=data,
        split=split,
        obs_norm=obs_norm,
        returns=returns,
        config=config,
        mode=mode,
        seed=mode.canonical_critic_seed,
        device=device,
        output_dir=output_dir / "critic",
    )
    if not audit["fixed_budget_completed"]:
        raise RuntimeError(f"critic budget incomplete for {dataset['id']}")
    advantages, advantage_manifest = q2.freeze_advantages(
        critic=critic,
        data=data,
        obs_norm=obs_norm,
        target_norm=target_norm,
        gamma=config.gamma,
        standardize=config.advantage_standardize,
        standardization_indices=split["train"],
        device=device,
        output_dir=output_dir / "frozen_advantage",
    )
    np.savez_compressed(output_dir / "episode_split.npz", **split)
    np.savez_compressed(
        output_dir / "observation_normalizer.npz", mean=obs_norm.mean, std=obs_norm.std
    )
    complete = {
        "identity_sha256": _payload_sha256(identity),
        "critic_checkpoint": audit["checkpoint"],
        "advantage_manifest": advantage_manifest,
        "critic_fixed_budget_completed": True,
        "completed_utc": q2.utc_now(),
    }
    q2.atomic_write_json(marker, complete)
    return {
        "split": split,
        "obs_norm": obs_norm,
        "advantages": advantages,
        "complete": complete,
    }


def _components(
    policy: Any, obs: Any, actions: Any, indices: Any, device: Any
) -> dict[str, Any]:
    from drpo import e7_hopper_q2 as q2
    import torch

    with torch.no_grad():
        values = policy.score_components(
            q2.tensor(obs[indices], device), q2.tensor(actions[indices], device)
        )
    keys = (
        "radius",
        "mean_score_norm",
        "raw_log_scale_score_norm",
        "corrected_q_xi",
        "joint_output_score_norm",
        "log_scale_to_mean_ratio",
        "raw_action_distance",
        "pre_squash_distance",
    )
    return {key: values[key].detach().cpu().numpy() for key in keys}


def _seed_curve(
    points: Sequence[dict[str, Any]], bins: int
) -> list[dict[str, Any]]:
    grouped = defaultdict(list)
    for row in points:
        grouped[int(row["distance_bin"])].append(row)
    if sorted(grouped) != list(range(bins)):
        raise RuntimeError("probe is missing a distance bin")
    rows = []
    for distance_bin in range(bins):
        local = grouped[distance_bin]
        rows.append(
            {
                "distance_bin": distance_bin,
                "distance_mean": sum(
                    row["standardized_distance"] for row in local
                )
                / len(local),
                "gradient_mean": sum(
                    row["full_parameter_gradient_norm"] for row in local
                )
                / len(local),
                "abs_advantage_mean": sum(row["abs_advantage"] for row in local)
                / len(local),
                "n_samples": len(local),
            }
        )
    bases = (
        rows[0]["distance_mean"],
        rows[0]["gradient_mean"],
        rows[0]["abs_advantage_mean"],
    )
    for row in rows:
        row["relative_distance"] = row["distance_mean"] / max(bases[0], 1e-12)
        row["relative_gradient"] = row["gradient_mean"] / max(bases[1], 1e-12)
        row["relative_abs_advantage"] = row["abs_advantage_mean"] / max(
            bases[2], 1e-12
        )
    return rows


def _run_seed(
    seed: int,
    dataset: dict[str, Any],
    runspec: dict[str, Any],
    data: Any,
    config: Any,
    mode: Any,
    context: dict[str, Any],
    device: Any,
    output_dir: Path,
    resume: bool,
    source_commit: str,
    runner_sha: str,
) -> dict[str, Any]:
    from drpo import e7_hopper_q2 as q2
    import numpy as np
    import torch

    marker = output_dir / "SEED_COMPLETE.json"
    identity = {
        "dataset_sha256": dataset["sha256"],
        "seed": seed,
        "critic_sha256": context["complete"]["critic_checkpoint"]["sha256"],
        "runspec_sha256": runspec["_sha256"],
        "source_commit": source_commit,
        "runner_sha256": runner_sha,
    }
    if marker.is_file():
        complete = json.loads(marker.read_text())
        if not resume or complete["identity_sha256"] != _payload_sha256(identity):
            raise RuntimeError(f"seed output conflict for {dataset['id']} seed {seed}")
        return complete
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"partial seed output without marker: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    q2.seed_everything(seed)
    split, obs_norm, advantages = (
        context["split"],
        context["obs_norm"],
        context["advantages"],
    )
    obs = obs_norm.transform(data.observations)
    train = split["train"]
    positive = train[advantages[train] > 0]
    negative = train[advantages[train] < 0]
    validation = split["validation"]
    val_pos = validation[advantages[validation] > 0]
    val_neg = validation[advantages[validation] < 0]
    if min(len(positive), len(negative), len(val_pos), len(val_neg)) == 0:
        raise RuntimeError("frozen advantage signs are insufficient for the diagnostic")
    rng = np.random.default_rng(seed + 321)
    half = max(1, mode.audit_sample_size // 2)
    audit_indices = np.concatenate(
        (
            rng.choice(val_pos, min(half, len(val_pos)), replace=False),
            rng.choice(val_neg, min(half, len(val_neg)), replace=False),
        )
    ).astype(np.int64)
    rng.shuffle(audit_indices)
    fixed_negative = rng.choice(
        negative, min(mode.audit_sample_size, len(negative)), replace=False
    )
    policy = q2.SquashedGaussianPolicy(
        obs.shape[1],
        data.actions.shape[1],
        config.hidden_sizes,
        config.log_std_min,
        config.log_std_max,
        config.action_clip_epsilon,
        config.activation,
        config.init_scheme,
        config.init_gain,
    ).to(device)
    policy, actor_audit = q2.train_actor_stage(
        policy=policy,
        method="positive_only",
        obs=obs,
        actions=data.actions,
        advantages=advantages,
        train_indices=positive,
        audit_indices=audit_indices,
        fixed_negative_indices=fixed_negative,
        config=config,
        min_steps=mode.positive_min_steps,
        max_steps=mode.positive_max_steps,
        eval_interval=mode.actor_eval_interval,
        seed=seed + 500000,
        device=device,
        output_dir=output_dir / "positive_only",
        rollout_evaluator=None,
        rollout_eval_interval=0,
        heartbeat=None,
    )
    if actor_audit["numerical_nonfinite"] or not actor_audit["fixed_budget_completed"]:
        raise RuntimeError(f"Positive-only budget failed for {dataset['id']} seed {seed}")
    actor_path = output_dir / "positive_only" / "terminal_actor.pt"
    distances = np.full(data.size, np.nan, dtype=np.float32)
    policy.eval()
    with torch.no_grad():
        for offset in range(0, len(negative), 65536):
            index = negative[offset : offset + 65536]
            distances[index] = policy.standardized_distance(
                q2.tensor(obs[index], device), q2.tensor(data.actions[index], device)
            ).cpu().numpy()
    probe = runspec["protocol"]["probe"]
    selection, selection_audit = select_matched_points(
        negative,
        distances,
        advantages,
        distance_bins=probe["distance_bins"],
        advantage_bins=probe["advantage_bins"],
        point_budget=probe["gradient_points_per_seed"],
        minimum_per_bin=probe["minimum_points_per_distance_bin"],
        tolerance=probe["relative_advantage_tolerance"],
        seed=seed,
    )
    selected = selection["indices"]
    components = _components(policy, obs, data.actions, selected, device)
    gradients = q2.per_sample_gradient_norm(
        policy, obs, data.actions, advantages, selected, device
    )
    actor_sha = q2.sha256_file(actor_path)
    base_sha = q2.sha256_file(ROOT / runspec["base_config_path"])
    points = []
    for position, transition in enumerate(selected):
        distance_bin = int(selection["distance_bin"][position])
        advantage_bin = int(selection["advantage_bin"][position])
        row = {
            "dataset_id": dataset["id"],
            "dataset_sha256": dataset["sha256"],
            "seed": seed,
            "distance_bin": distance_bin,
            "advantage_bin": advantage_bin,
            "match_group_id": int(selection["match_group_id"][position]),
            "target_abs_advantage": float(
                selection["target_abs_advantage"][position]
            ),
            "relative_advantage_error": float(
                selection["relative_advantage_error"][position]
            ),
            "transition_index": int(transition),
            "episode_id": int(data.episode_ids[transition]),
            "split": "critic_train_episode_split",
            "advantage": float(advantages[transition]),
            "abs_advantage": float(abs(advantages[transition])),
            "full_parameter_gradient_norm": float(gradients[position]),
            "distance_bin_left": float(selection["distance_edges"][distance_bin]),
            "distance_bin_right": float(
                selection["distance_edges"][distance_bin + 1]
            ),
            "advantage_bin_left": float(
                selection["advantage_edges"][advantage_bin]
            ),
            "advantage_bin_right": float(
                selection["advantage_edges"][advantage_bin + 1]
            ),
            "critic_checkpoint_sha256": context["complete"]["critic_checkpoint"][
                "sha256"
            ],
            "positive_actor_checkpoint_sha256": actor_sha,
            "runspec_sha256": runspec["_sha256"],
            "base_config_sha256": base_sha,
            "source_commit_sha": source_commit,
            "runner_sha256": runner_sha,
        }
        for key, values in components.items():
            row["standardized_distance" if key == "radius" else key] = float(
                values[position]
            )
        if set(row) != set(POINT_FIELDS):
            missing = sorted(set(POINT_FIELDS) - set(row))
            extra = sorted(set(row) - set(POINT_FIELDS))
            raise RuntimeError(
                f"point schema mismatch: missing={missing}, extra={extra}"
            )
        points.append(row)
    point_path = output_dir / "gradient_probe_points.csv"
    q2.write_csv(point_path, points)
    curve = _seed_curve(points, probe["distance_bins"])
    for row in curve:
        row.update({"dataset_id": dataset["id"], "seed": seed})
    q2.write_csv(output_dir / "seed_curve.csv", curve)
    selection_audit.update(
        {
            "point_file_sha256": q2.sha256_file(point_path),
            "abs_advantage_curve_max_min_ratio": max(
                row["abs_advantage_mean"] for row in curve
            )
            / max(min(row["abs_advantage_mean"] for row in curve), 1e-12),
        }
    )
    q2.atomic_write_json(output_dir / "probe_selection_audit.json", selection_audit)
    complete = {
        "identity_sha256": _payload_sha256(identity),
        "dataset_id": dataset["id"],
        "seed": seed,
        "point_rows": len(points),
        "positive_actor_fixed_budget_completed": True,
        "positive_actor_support_boundary_event": actor_audit[
            "support_boundary_event"
        ],
        "positive_actor_numerical_nonfinite": actor_audit["numerical_nonfinite"],
        "task_performance_collapse": "not_evaluated_in_gradient_only_diagnostic",
        "completed_utc": q2.utc_now(),
    }
    q2.atomic_write_json(marker, complete)
    return complete


def _dataset_worker(payload: dict[str, Any]) -> dict[str, Any]:
    from drpo import e7_bench
    from drpo import e7_hopper_q2 as q2

    dataset = payload["dataset"]
    output_dir = Path(payload["work_dir"]) / "datasets" / dataset["id"]
    try:
        _configure_threads(payload["cpus_per_worker"])
        runspec = load_runspec(payload["runspec_path"])
        spec = _dataset_spec(dataset, dataset["sha256"])
        data = e7_bench.load_dataset(
            Path(dataset["resolved_path"]), spec, max_transitions=None
        )
        config, mode = _adapt_config(runspec, dataset)
        device = q2.choose_device(payload["device"])
        critic_identity = {
            "dataset_sha256": dataset["sha256"],
            "runspec_sha256": runspec["_sha256"],
            "source_commit": payload["source_commit"],
            "runner_sha256": payload["runner_sha256"],
            "critic_seed": 100,
            "critic_steps": mode.critic_max_steps,
        }
        context = _critic_context(
            dataset,
            runspec,
            data,
            config,
            mode,
            device,
            output_dir / "critic_context",
            payload["resume"],
            critic_identity,
        )
        completions = [
            _run_seed(
                seed,
                dataset,
                runspec,
                data,
                config,
                mode,
                context,
                device,
                output_dir / f"seed_{seed}",
                payload["resume"],
                payload["source_commit"],
                payload["runner_sha256"],
            )
            for seed in runspec["protocol"]["seeds"]
        ]
        result = {
            "dataset_id": dataset["id"],
            "dataset_sha256": dataset["sha256"],
            "transitions": data.size,
            "episodes": int(data.episode_ids.max()) + 1,
            "seed_count": len(completions),
            "completed_utc": q2.utc_now(),
        }
        q2.atomic_write_json(output_dir / "DATASET_COMPLETE.json", result)
        return result
    except Exception as exc:
        output_dir.mkdir(parents=True, exist_ok=True)
        q2.atomic_write_json(
            output_dir / "DATASET_FAILED.json",
            {
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "failed_utc": q2.utc_now(),
            },
        )
        raise


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _mean_ci95(values: Iterable[float]) -> tuple[float, float, float]:
    import numpy as np

    values = np.asarray(list(values), dtype=np.float64)
    mean = float(values.mean())
    if len(values) == 1:
        return mean, mean, mean
    half = 1.96 * float(values.std(ddof=1)) / math.sqrt(len(values))
    return mean, mean - half, mean + half


def _plot(rows: Sequence[dict[str, Any]], output_dir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    titles = {
        dataset: dataset.replace("-v2", "").replace("-", " ") for dataset in D4RL9
    }
    figure, axes = plt.subplots(3, 3, figsize=(13.2, 10.2))
    handles = labels = None
    for axis, dataset in zip(axes.flat, D4RL9):
        local = sorted(
            (row for row in rows if row["dataset_id"] == dataset),
            key=lambda row: row["distance_bin"],
        )
        x = [row["relative_distance"] for row in local]
        for key, low, high, marker, linestyle, label in (
            (
                "relative_gradient_mean",
                "relative_gradient_ci95_low",
                "relative_gradient_ci95_high",
                "o",
                "-",
                "implemented actor-gradient",
            ),
            (
                "relative_abs_advantage_mean",
                "relative_abs_advantage_ci95_low",
                "relative_abs_advantage_ci95_high",
                "s",
                "--",
                "matched |advantage|",
            ),
        ):
            axis.plot(
                x,
                [row[key] for row in local],
                marker=marker,
                linestyle=linestyle,
                label=label,
            )
            axis.fill_between(
                x,
                [row[low] for row in local],
                [row[high] for row in local],
                alpha=0.15,
            )
        axis.axhline(1.0, linewidth=0.8, alpha=0.6)
        axis.set(
            title=titles[dataset],
            xlabel="relative standardized distance",
            ylabel="relative quantity",
        )
        axis.grid(alpha=0.2)
        if handles is None:
            handles, labels = axis.get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    paths = []
    for suffix in ("pdf", "png", "svg"):
        path = output_dir / f"figure1_d4rl9_appendix.{suffix}"
        figure.savefig(path, dpi=220, bbox_inches="tight")
        paths.append(str(path))
    plt.close(figure)
    return paths


def aggregate(work_dir: Path, runspec: dict[str, Any]) -> dict[str, Any]:
    from drpo import e7_hopper_q2 as q2

    aggregate_dir = work_dir / "aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    points, curves = [], []
    for dataset in D4RL9:
        for seed in runspec["protocol"]["seeds"]:
            seed_dir = work_dir / "datasets" / dataset / f"seed_{seed}"
            points.extend(_read_csv(seed_dir / "gradient_probe_points.csv"))
            curves.extend(_read_csv(seed_dir / "seed_curve.csv"))
    q2.write_csv(aggregate_dir / "d4rl9_gradient_probe_points.csv", points)
    q2.write_csv(aggregate_dir / "d4rl9_seed_curves.csv", curves)
    grouped = defaultdict(list)
    counts = defaultdict(int)
    for row in curves:
        grouped[(row["dataset_id"], int(row["distance_bin"]))].append(row)
    for row in points:
        counts[(row["dataset_id"], int(row["distance_bin"]))] += 1
    plot_rows = []
    for dataset in D4RL9:
        for distance_bin in range(runspec["protocol"]["probe"]["distance_bins"]):
            local = grouped[(dataset, distance_bin)]
            if len(local) != len(runspec["protocol"]["seeds"]):
                raise RuntimeError(
                    f"seed count mismatch for {dataset} bin {distance_bin}"
                )
            distance = _mean_ci95(float(row["relative_distance"]) for row in local)
            gradient = _mean_ci95(float(row["relative_gradient"]) for row in local)
            advantage = _mean_ci95(
                float(row["relative_abs_advantage"]) for row in local
            )
            plot_rows.append(
                {
                    "dataset_id": dataset,
                    "distance_bin": distance_bin,
                    "relative_distance": distance[0],
                    "relative_distance_ci95_low": distance[1],
                    "relative_distance_ci95_high": distance[2],
                    "relative_gradient_mean": gradient[0],
                    "relative_gradient_ci95_low": gradient[1],
                    "relative_gradient_ci95_high": gradient[2],
                    "relative_abs_advantage_mean": advantage[0],
                    "relative_abs_advantage_ci95_low": advantage[1],
                    "relative_abs_advantage_ci95_high": advantage[2],
                    "n_seeds": len(local),
                    "n_samples": counts[(dataset, distance_bin)],
                }
            )
    q2.write_csv(aggregate_dir / "d4rl9_plot_data.csv", plot_rows)
    plots = _plot(plot_rows, aggregate_dir)
    summaries = []
    for dataset in D4RL9:
        local = [row for row in plot_rows if row["dataset_id"] == dataset]
        farthest = max(local, key=lambda row: row["distance_bin"])
        advantages = [row["relative_abs_advantage_mean"] for row in local]
        summaries.append(
            {
                "dataset_id": dataset,
                "farthest_relative_distance": farthest["relative_distance"],
                "farthest_relative_gradient": farthest["relative_gradient_mean"],
                "farthest_relative_abs_advantage": farthest[
                    "relative_abs_advantage_mean"
                ],
                "abs_advantage_curve_max_min_ratio": max(advantages)
                / max(min(advantages), 1e-12),
                "n_seeds": farthest["n_seeds"],
                "n_points": sum(row["n_samples"] for row in local),
            }
        )
    q2.write_csv(aggregate_dir / "dataset_summary.csv", summaries)
    result = {
        "dataset_count": 9,
        "seed_count_per_dataset": len(runspec["protocol"]["seeds"]),
        "gradient_point_rows": len(points),
        "plot_data_rows": len(plot_rows),
        "plot_paths": plots,
        "scientific_status": "pilot",
        "completed_utc": q2.utc_now(),
    }
    q2.atomic_write_json(aggregate_dir / "AGGREGATE_COMPLETE.json", result)
    return result


def _plan(runspec: dict[str, Any], dataset_root: str | Path) -> dict[str, Any]:
    return {
        "experiment_id": FIG1_ID,
        "scientific_status_cap": "pilot",
        "dataset_root": str(Path(dataset_root).expanduser().resolve()),
        "datasets": list(D4RL9),
        "seeds": runspec["protocol"]["seeds"],
        "critic_jobs": 9,
        "positive_actor_probe_jobs": 90,
        "method_branch_jobs": 0,
        "dataset_workers": runspec["execution"]["dataset_workers"],
        "device_pool": runspec["execution"]["device_pool"],
        "probe": runspec["protocol"]["probe"],
    }


def run_figure1(args: argparse.Namespace) -> int:
    from drpo import e7_hopper_q2 as q2

    runspec = load_runspec(args.runspec)
    if args.plan:
        print(json.dumps(_plan(runspec, args.dataset_root), indent=2))
        return 0
    git_state = q2.collect_git_state(ROOT)
    if not git_state.get("available") or str(
        git_state.get("status_porcelain", "")
    ).strip():
        raise RuntimeError("Figure-1 execution requires a clean Git checkout")
    source_commit = str(git_state["head"])
    runner_sha = q2.sha256_file(Path(__file__).resolve())
    work_dir = Path(args.work_dir).expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    lock = lock_datasets(runspec, args.dataset_root)
    lock_path = work_dir / "DATASET_LOCK.json"
    if lock_path.is_file():
        previous = json.loads(lock_path.read_text())
        if (
            not args.resume
            or previous["dataset_lock_sha256"] != lock["dataset_lock_sha256"]
        ):
            raise RuntimeError("work directory dataset lock conflict")
    q2.atomic_write_json(lock_path, lock)
    q2.atomic_write_json(
        work_dir / "RUNSPEC_RESOLVED.json",
        {
            **{key: value for key, value in runspec.items() if not key.startswith("_")},
            "runspec_sha256": runspec["_sha256"],
            "dataset_lock_sha256": lock["dataset_lock_sha256"],
            "source_commit_sha": source_commit,
            "runner_sha256": runner_sha,
        },
    )
    if args.validate_only:
        q2.atomic_write_json(
            work_dir / "PREFLIGHT_COMPLETE.json",
            {
                "training_started": False,
                "dataset_lock_sha256": lock["dataset_lock_sha256"],
                "completed_utc": q2.utc_now(),
            },
        )
        return 0
    if (work_dir / "RUN_COMPLETE.json").is_file():
        if args.resume:
            return 0
        raise RuntimeError(
            "RUN_COMPLETE.json exists; pass --resume or use a new work directory"
        )
    devices = (
        [value.strip() for value in args.device_pool.split(",") if value.strip()]
        if args.device_pool
        else list(runspec["execution"]["device_pool"])
    )
    workers = min(args.workers or runspec["execution"]["dataset_workers"], 9)
    cpus = args.cpus_per_worker or runspec["execution"]["cpus_per_worker"]
    if not devices or workers < 1 or cpus < 1:
        raise ValueError("device pool, workers, and cpus-per-worker must be positive")
    payloads = [
        {
            "dataset": dataset,
            "runspec_path": runspec["_path"],
            "work_dir": str(work_dir),
            "resume": args.resume,
            "device": devices[index % len(devices)],
            "cpus_per_worker": cpus,
            "source_commit": source_commit,
            "runner_sha256": runner_sha,
        }
        for index, dataset in enumerate(lock["datasets"])
    ]
    manifest = {
        "experiment_id": FIG1_ID,
        "runner_version": FIG1_VERSION,
        "scientific_status_cap": "pilot",
        "runspec_sha256": runspec["_sha256"],
        "dataset_lock_sha256": lock["dataset_lock_sha256"],
        "source_commit_sha": source_commit,
        "runner_sha256": runner_sha,
        "git_state": git_state,
        "execution_plan": _plan(runspec, args.dataset_root),
        "started_utc": q2.utc_now(),
    }
    q2.atomic_write_json(work_dir / "RUN_MANIFEST.json", manifest)
    try:
        if workers == 1:
            results = [_dataset_worker(payload) for payload in payloads]
        else:
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=workers,
                mp_context=multiprocessing.get_context("spawn"),
            ) as executor:
                futures = [executor.submit(_dataset_worker, payload) for payload in payloads]
                results = [future.result() for future in futures]
        complete = {
            **manifest,
            "dataset_results": results,
            "aggregate": aggregate(work_dir, runspec),
            "raw_complete": True,
            "task_performance_collapse": "not_evaluated",
            "support_or_variance_boundary": "preserved_per_positive_only_seed_audit",
            "nan_inf_numerical_failure": "fail_closed",
            "completed_utc": q2.utc_now(),
        }
        q2.atomic_write_json(work_dir / "RUN_COMPLETE.json", complete)
        return 0
    except Exception as exc:
        q2.atomic_write_json(
            work_dir / "RUN_FAILED.json",
            {
                **manifest,
                "raw_complete": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "failed_utc": q2.utc_now(),
            },
        )
        raise


def self_test() -> int:
    import numpy as np

    rng = np.random.default_rng(7)
    count = 7000
    distances = np.exp(rng.normal(size=count))
    advantages = -np.exp(rng.normal(scale=0.4, size=count))
    selection, audit = select_matched_points(
        np.arange(count),
        distances,
        advantages,
        distance_bins=7,
        advantage_bins=8,
        point_budget=64,
        minimum_per_bin=4,
        tolerance=0.05,
        seed=100,
    )
    assert len(selection["indices"]) == 7 * audit["points_per_distance_bin"]
    assert audit["max_relative_advantage_error"] <= 0.05
    assert len(set(np.bincount(selection["distance_bin"]))) == 1
    synthetic = []
    for position, index in enumerate(selection["indices"]):
        synthetic.append(
            {
                "distance_bin": int(selection["distance_bin"][position]),
                "standardized_distance": float(distances[index]),
                "full_parameter_gradient_norm": float(distances[index] ** 1.2),
                "abs_advantage": float(abs(advantages[index])),
            }
        )
    curve = _seed_curve(synthetic, 7)
    assert curve[0]["relative_distance"] == curve[0]["relative_gradient"] == 1.0
    print(
        json.dumps(
            {"self_test": "passed", "selected_points": len(synthetic)}, indent=2
        )
    )
    return 0


def figure1_main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(
        description="D4RL-9 Figure-1 gradient coverage diagnostic"
    )
    parser.add_argument(
        "--runspec", default="configs/e7_figure1_d4rl9_runspec.yaml"
    )
    parser.add_argument("--dataset-root")
    parser.add_argument("--work-dir")
    parser.add_argument("--device-pool")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--cpus-per-worker", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        return self_test()
    if not args.dataset_root:
        parser.error("--dataset-root is required")
    if not args.work_dir and not args.plan:
        parser.error("--work-dir is required unless --plan is used")
    return run_figure1(args)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "figure1-d4rl9":
        return figure1_main(arguments[1:])
    from drpo.e7_bench import main as bench_main

    return bench_main(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
