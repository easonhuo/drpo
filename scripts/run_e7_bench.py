#!/usr/bin/env python3
"""E7 benchmark entrypoint plus the D4RL-9 Figure-1 gradient diagnostic.

The ``figure1-d4rl9`` subcommand preserves the existing E7-Q2 protocol:
learned critic -> frozen advantages -> Positive-only actor -> advantage-matched
near/far negative probes.  Every selected gradient point is persisted before
per-dataset aggregation and appendix plotting.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import dataclasses
import hashlib
import json
import math
import multiprocessing
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Iterable, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

EXPERIMENT_ID = "EXT-H-E7-FIG1-D4RL9-GRADIENT-COVERAGE-01"
RUNNER_VERSION = "1.1.1-e7q2-protocol-preserving"
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
ENV_DIMS = {
    "HalfCheetah-v4": (17, 6),
    "Hopper-v4": (11, 3),
    "Walker2d-v4": (17, 6),
}
POINT_COMPONENTS = (
    "radius",
    "mean_score_norm",
    "raw_log_scale_score_norm",
    "corrected_q_xi",
    "joint_output_score_norm",
    "log_scale_to_mean_ratio",
    "raw_action_distance",
    "pre_squash_distance",
    "full_parameter_gradient_norm",
)


def payload_hash(payload: Any) -> str:
    body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(body).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def load_runspec(path: str | Path) -> dict[str, Any]:
    import yaml

    from drpo import e7_hopper_q2 as q2

    source = Path(path).expanduser().resolve()
    raw = yaml.safe_load(source.read_text())
    if not isinstance(raw, dict):
        raise ValueError("runspec must be a mapping")
    frozen = {
        "experiment_id": EXPERIMENT_ID,
        "runner_version": RUNNER_VERSION,
        "execution_class": "pilot",
        "scientific_status": "pilot",
        "formal_launch_allowed": False,
    }
    for key, expected in frozen.items():
        if raw.get(key) != expected:
            raise ValueError(f"{key} must remain {expected!r}")
    datasets = raw.get("datasets")
    if not isinstance(datasets, list) or tuple(x.get("id") for x in datasets) != D4RL9:
        raise ValueError("runspec must retain the ordered D4RL-v2 3x3 cells")
    for item in datasets:
        env_id = item.get("env_id")
        if item.get("format") != "legacy_d4rl_hdf5" or env_id not in ENV_DIMS:
            raise ValueError(f"invalid dataset contract for {item.get('id')}")
        if (item.get("observation_dim"), item.get("action_dim")) != ENV_DIMS[env_id]:
            raise ValueError(f"dimension contract mismatch for {item['id']}")
        expected = item.get("expected_sha256")
        if expected not in (None, "auto_capture_before_training") and len(str(expected)) != 64:
            raise ValueError(f"invalid expected_sha256 for {item['id']}")
    protocol = raw.get("protocol", {})
    if tuple(protocol.get("seeds", ())) != tuple(range(100, 110)):
        raise ValueError("seeds must remain 100..109")
    if protocol.get("canonical_critic_seed") != 100:
        raise ValueError("canonical_critic_seed must remain 100")
    if protocol.get("train_method_branches") is not False:
        raise ValueError("method branches are forbidden in this diagnostic")
    budget = protocol.get("budget", {})
    required_budget = {
        "max_transitions": None,
        "critic_steps": 100000,
        "critic_eval_interval": 2000,
        "positive_actor_steps": 100000,
        "positive_actor_eval_interval": 5000,
        "audit_sample_size": 16384,
    }
    probe = protocol.get("probe", {})
    required_probe = {
        "matching_function": "drpo.e7_hopper_q2.match_near_far_indices",
        "near_quantile": 0.25,
        "far_quantile": 0.75,
        "advantage_matching_bins": 20,
        "relative_advantage_tolerance": 0.05,
        "matched_pair_pool_per_seed": 256,
        "gradient_pairs_per_seed": 64,
        "far_distance_bins": 7,
        "save_every_selected_point": True,
    }
    for section, actual, expected in (
        ("budget", budget, required_budget),
        ("probe", probe, required_probe),
    ):
        for key, value in expected.items():
            if actual.get(key) != value:
                raise ValueError(f"protocol.{section}.{key} must remain {value!r}")
    execution = raw.get("execution", {})
    if execution.get("require_clean_worktree") is not True:
        raise ValueError("clean-worktree execution must remain enabled")
    if min(int(execution.get("dataset_workers", 0)), int(execution.get("cpus_per_worker", 0))) < 1:
        raise ValueError("dataset_workers and cpus_per_worker must be positive")
    base_config = (ROOT / str(raw.get("base_config_path", ""))).resolve()
    try:
        base_config.relative_to(ROOT)
    except ValueError as exc:
        raise ValueError("base_config_path must remain inside the repository") from exc
    if not base_config.is_file():
        raise FileNotFoundError(f"base config does not exist: {base_config}")
    q2.load_config(base_config)
    raw["_path"] = str(source)
    raw["_sha256"] = q2.sha256_file(source)
    raw["_base_config_sha256"] = q2.sha256_file(base_config)
    return raw


def dataset_spec(item: dict[str, Any], sha256: str) -> Any:
    from drpo import e7_bench

    return e7_bench.DatasetSpec(
        id=item["id"],
        relative_path=item["relative_path"],
        sha256=sha256,
        format=item["format"],
        env_id=item["env_id"],
        dataset_family="d4rl_v2",
        score_protocol="gradient_diagnostic_no_rollout",
        reference_min_score=None,
        reference_max_score=None,
        formal_cell_eligible=True,
        provenance_note=item["provenance_note"],
    )


def lock_datasets(runspec: dict[str, Any], root_value: str | Path) -> dict[str, Any]:
    from drpo import e7_bench
    from drpo import e7_hopper_q2 as q2

    root = Path(root_value).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"dataset root does not exist: {root}")
    locked = []
    for item in runspec["datasets"]:
        path = (root / item["relative_path"]).resolve()
        path.relative_to(root)
        if not path.is_file():
            raise FileNotFoundError(f"missing D4RL dataset: {path}")
        digest = q2.sha256_file(path)
        expected = item.get("expected_sha256")
        if expected not in (None, "auto_capture_before_training") and digest != expected:
            raise RuntimeError(f"dataset SHA mismatch for {item['id']}: {digest}")
        sample = e7_bench.load_dataset(path, dataset_spec(item, digest), max_transitions=32)
        dims = (int(sample.observations.shape[1]), int(sample.actions.shape[1]))
        expected_dims = (item["observation_dim"], item["action_dim"])
        if dims != expected_dims:
            raise RuntimeError(f"dataset dimensions mismatch for {item['id']}: {dims}")
        locked.append(
            {
                **item,
                "resolved_path": str(path),
                "sha256": digest,
                "size_bytes": path.stat().st_size,
            }
        )
    result = {
        "experiment_id": EXPERIMENT_ID,
        "runner_version": RUNNER_VERSION,
        "runspec_sha256": runspec["_sha256"],
        "capture_stage": "all_nine_files_before_any_training",
        "datasets": locked,
    }
    result["dataset_lock_sha256"] = payload_hash(result)
    return result


def adapted_config(runspec: dict[str, Any], dataset: dict[str, Any]) -> tuple[Any, Any]:
    from drpo import e7_hopper_q2 as q2

    base = q2.load_config(ROOT / runspec["base_config_path"])
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
        matched_pairs=probe["matched_pair_pool_per_seed"],
        audit_sample_size=budget["audit_sample_size"],
        rollout_episodes=0,
        final_rollout_episodes=0,
        rollout_eval_interval=0,
    )
    config = dataclasses.replace(
        base,
        experiment_id=EXPERIMENT_ID,
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
    )
    return config, mode


def configure_threads(count: int) -> None:
    import torch

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


def heartbeat(path: Path, stage: str, step: int, **extra: Any) -> None:
    from drpo import e7_hopper_q2 as q2

    q2.atomic_write_json(path, {"utc": q2.utc_now(), "stage": stage, "step": step, **extra})


def point_rows_from_probe(
    probe_rows: Sequence[dict[str, str]],
    *,
    dataset: dict[str, Any],
    seed: int,
    data: Any,
    gradient_pairs: int,
    critic_sha: str,
    actor_sha: str,
    runspec_sha: str,
    base_config_sha: str,
    source_commit: str,
    runner_sha: str,
    probe: dict[str, Any],
) -> list[dict[str, Any]]:
    points = []
    for pair in probe_rows[:gradient_pairs]:
        pair_id = int(pair["pair_id"])
        for role, paired_role in (("near", "far"), ("far", "near")):
            transition = int(pair[f"{role}_index"])
            advantage = float(pair[f"{role}_advantage"])
            paired_abs = float(pair[f"{paired_role}_abs_advantage"])
            current_abs = abs(advantage)
            row = {
                "dataset_id": dataset["id"],
                "dataset_sha256": dataset["sha256"],
                "seed": seed,
                "pair_id": pair_id,
                "pair_role": role,
                "transition_index": transition,
                "episode_id": int(data.episode_ids[transition]),
                "split": "critic_train_episode_split",
                "advantage": advantage,
                "abs_advantage": current_abs,
                "paired_abs_advantage": paired_abs,
                "pair_relative_advantage_error": (
                    abs(current_abs - paired_abs) / max(paired_abs, 1e-8)
                ),
                "near_quantile": probe["near_quantile"],
                "far_quantile": probe["far_quantile"],
                "advantage_matching_bins": probe["advantage_matching_bins"],
                "matching_relative_tolerance": probe["relative_advantage_tolerance"],
                "critic_checkpoint_sha256": critic_sha,
                "positive_actor_checkpoint_sha256": actor_sha,
                "runspec_sha256": runspec_sha,
                "base_config_sha256": base_config_sha,
                "source_commit_sha": source_commit,
                "runner_sha256": runner_sha,
            }
            for key in POINT_COMPONENTS:
                value = pair.get(f"{role}_{key}")
                if value in (None, ""):
                    raise RuntimeError(f"missing {role}_{key} for pair {pair_id}")
                row["standardized_distance" if key == "radius" else key] = float(value)
            points.append(row)
    if len(points) != 2 * gradient_pairs:
        raise RuntimeError("gradient point count mismatch")
    return points


def run_seed(
    *,
    seed: int,
    dataset: dict[str, Any],
    runspec: dict[str, Any],
    data: Any,
    config: Any,
    mode: Any,
    canonical: Any,
    device: Any,
    output_dir: Path,
    resume: bool,
    source_commit: str,
    runner_sha: str,
) -> dict[str, Any]:
    import numpy as np
    import torch

    from drpo import e7_hopper_q2 as q2

    critic_sha = canonical.artifact_manifest["files"]["critic_checkpoint"]["sha256"]
    identity = {
        "dataset_sha256": dataset["sha256"],
        "seed": seed,
        "critic_sha256": critic_sha,
        "runspec_sha256": runspec["_sha256"],
        "source_commit_sha": source_commit,
        "runner_sha256": runner_sha,
    }
    marker = output_dir / "SEED_COMPLETE.json"
    if marker.is_file():
        complete = json.loads(marker.read_text())
        if resume and complete.get("identity_sha256") == payload_hash(identity):
            return complete
        raise RuntimeError(f"seed output conflict for {dataset['id']} seed {seed}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"partial seed output without marker: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    q2.seed_everything(seed)
    split = canonical.split
    advantages = canonical.advantages
    obs = canonical.obs_norm.transform(data.observations)
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
    fixed_negative = rng.choice(negative, min(mode.audit_sample_size, len(negative)), replace=False)
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
        heartbeat=lambda stage, step: heartbeat(
            output_dir / "HEARTBEAT.json", stage, step, dataset_id=dataset["id"], seed=seed
        ),
    )
    if actor_audit["numerical_nonfinite"] or not actor_audit["fixed_budget_completed"]:
        raise RuntimeError(f"Positive-only budget failed for {dataset['id']} seed {seed}")
    distances = np.full(data.size, np.nan, dtype=np.float32)
    policy.eval()
    with torch.no_grad():
        for offset in range(0, len(negative), 65536):
            index = negative[offset : offset + 65536]
            distances[index] = policy.standardized_distance(
                q2.tensor(obs[index], device), q2.tensor(data.actions[index], device)
            ).cpu().numpy()
    probe = runspec["protocol"]["probe"]
    near, far, matching_summary = q2.match_near_far_indices(
        advantages,
        distances,
        negative,
        probe["near_quantile"],
        probe["far_quantile"],
        probe["advantage_matching_bins"],
        probe["matched_pair_pool_per_seed"],
        probe["relative_advantage_tolerance"],
        seed,
    )
    if len(near) < probe["gradient_pairs_per_seed"]:
        raise RuntimeError(f"only {len(near)} matched pairs for {dataset['id']} seed {seed}")
    q2.write_csv(
        output_dir / "matching_pairs.csv",
        [
            {
                "pair_id": i,
                "selected_for_gradient": i < probe["gradient_pairs_per_seed"],
                "near_index": int(n),
                "far_index": int(f),
                "near_abs_advantage": float(abs(advantages[n])),
                "far_abs_advantage": float(abs(advantages[f])),
                "near_standardized_distance": float(distances[n]),
                "far_standardized_distance": float(distances[f]),
            }
            for i, (n, f) in enumerate(zip(near, far))
        ],
    )
    q2.atomic_write_json(output_dir / "matching_summary.json", matching_summary)
    probe_dir = output_dir / "probe_raw"
    probe_summary = q2.create_gradient_probe(
        policy=policy,
        obs=obs,
        actions=data.actions,
        advantages=advantages,
        near_indices=near,
        far_indices=far,
        population_indices=fixed_negative,
        max_gradient_pairs=probe["gradient_pairs_per_seed"],
        distance_bins=probe["far_distance_bins"],
        device=device,
        output_dir=probe_dir,
    )
    actor_sha = actor_audit["checkpoint"]["sha256"]
    points = point_rows_from_probe(
        read_csv(probe_dir / "matched_near_far_components.csv"),
        dataset=dataset,
        seed=seed,
        data=data,
        gradient_pairs=probe["gradient_pairs_per_seed"],
        critic_sha=critic_sha,
        actor_sha=actor_sha,
        runspec_sha=runspec["_sha256"],
        base_config_sha=q2.sha256_file(ROOT / runspec["base_config_path"]),
        source_commit=source_commit,
        runner_sha=runner_sha,
        probe=probe,
    )
    point_path = output_dir / "gradient_probe_points.csv"
    q2.write_csv(point_path, points)
    probe_summary.update(
        {
            "saved_point_rows": len(points),
            "point_file_sha256": q2.sha256_file(point_path),
            "protocol_preserved_from": "EXT-H-E7-Q2",
        }
    )
    q2.atomic_write_json(output_dir / "gradient_probe_summary.json", probe_summary)
    complete = {
        "identity_sha256": payload_hash(identity),
        "dataset_id": dataset["id"],
        "seed": seed,
        "matching_pair_pool": len(near),
        "gradient_pairs": probe["gradient_pairs_per_seed"],
        "point_rows": len(points),
        "positive_actor_fixed_budget_completed": True,
        "positive_actor_support_boundary_event": actor_audit["support_boundary_event"],
        "positive_actor_numerical_nonfinite": actor_audit["numerical_nonfinite"],
        "task_performance_collapse": "not_evaluated_in_gradient_only_diagnostic",
        "completed_utc": q2.utc_now(),
    }
    q2.atomic_write_json(marker, complete)
    return complete


def bootstrap_ci(values: Iterable[float], *, seed: int, draws: int) -> tuple[float, float, float]:
    import numpy as np

    array = np.asarray(list(values), dtype=np.float64)
    if len(array) == 0:
        raise ValueError("bootstrap requires values")
    mean = float(array.mean())
    if len(array) == 1:
        return mean, mean, mean
    rng = np.random.default_rng(seed)
    means = rng.choice(array, size=(draws, len(array)), replace=True).mean(axis=1)
    low, high = np.quantile(means, (0.025, 0.975))
    return mean, float(low), float(high)


def build_plot_rows(
    points: Sequence[dict[str, Any]],
    *,
    dataset_id: str,
    seeds: Sequence[int],
    far_bins: int,
    bootstrap_draws: int,
    bootstrap_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    import numpy as np

    rows = []
    for source in points:
        row = dict(source)
        for key in ("seed", "pair_id", "transition_index", "episode_id"):
            row[key] = int(row[key])
        for key in ("abs_advantage", "standardized_distance", "full_parameter_gradient_norm"):
            row[key] = float(row[key])
        rows.append(row)
    expected_seeds = tuple(int(seed) for seed in seeds)
    if tuple(sorted({row["seed"] for row in rows})) != expected_seeds:
        raise RuntimeError(f"seed set mismatch for {dataset_id}")
    far_distance = np.asarray(
        [row["standardized_distance"] for row in rows if row["pair_role"] == "far"]
    )
    edges = np.unique(np.quantile(far_distance, np.linspace(0.0, 1.0, far_bins + 1)))
    if len(edges) != far_bins + 1:
        raise RuntimeError(f"far distance quantiles collapsed for {dataset_id}")
    binned = []
    for source in rows:
        row = dict(source)
        if row["pair_role"] == "near":
            row.update(distance_bin=-1, distance_bin_left=None, distance_bin_right=None)
        elif row["pair_role"] == "far":
            bin_id = int(
                np.clip(
                    np.searchsorted(
                        edges[1:-1], row["standardized_distance"], side="right"
                    ),
                    0,
                    far_bins - 1,
                )
            )
            row.update(
                distance_bin=bin_id,
                distance_bin_left=float(edges[bin_id]),
                distance_bin_right=float(edges[bin_id + 1]),
            )
        else:
            raise RuntimeError(f"unknown pair role: {row['pair_role']}")
        binned.append(row)
    seed_curves = []
    for seed in expected_seeds:
        local = [row for row in binned if row["seed"] == seed]
        near = [row for row in local if row["pair_role"] == "near"]
        far = [row for row in local if row["pair_role"] == "far"]
        if len(near) != len(far) or {x["pair_id"] for x in near} != {x["pair_id"] for x in far}:
            raise RuntimeError(f"unbalanced pairs for {dataset_id} seed {seed}")
        bases = (
            np.mean([x["standardized_distance"] for x in near]),
            np.mean([x["full_parameter_gradient_norm"] for x in near]),
            np.mean([x["abs_advantage"] for x in near]),
        )
        seed_curves.append(
            {
                "dataset_id": dataset_id,
                "seed": seed,
                "distance_bin": -1,
                "relative_distance": 1.0,
                "relative_gradient": 1.0,
                "relative_abs_advantage": 1.0,
                "n_samples": len(near),
            }
        )
        for bin_id in range(far_bins):
            selected = [x for x in far if x["distance_bin"] == bin_id]
            if selected:
                seed_curves.append(
                    {
                        "dataset_id": dataset_id,
                        "seed": seed,
                        "distance_bin": bin_id,
                        "relative_distance": float(
                            np.mean([x["standardized_distance"] for x in selected])
                            / max(bases[0], 1e-12)
                        ),
                        "relative_gradient": float(
                            np.mean(
                                [x["full_parameter_gradient_norm"] for x in selected]
                            )
                            / max(bases[1], 1e-12)
                        ),
                        "relative_abs_advantage": float(
                            np.mean([x["abs_advantage"] for x in selected])
                            / max(bases[2], 1e-12)
                        ),
                        "n_samples": len(selected),
                    }
                )
    plot_rows = []
    for bin_id in (-1, *range(far_bins)):
        local = [x for x in seed_curves if x["distance_bin"] == bin_id]
        if not local:
            raise RuntimeError(f"empty aggregate bin {bin_id} for {dataset_id}")
        if bin_id == -1:
            distance = gradient = advantage = (1.0, 1.0, 1.0)
        else:
            base = bootstrap_seed + 1000 * bin_id
            distance = bootstrap_ci(
                (x["relative_distance"] for x in local),
                seed=base + 11,
                draws=bootstrap_draws,
            )
            gradient = bootstrap_ci(
                (x["relative_gradient"] for x in local),
                seed=base + 23,
                draws=bootstrap_draws,
            )
            advantage = bootstrap_ci(
                (x["relative_abs_advantage"] for x in local),
                seed=base + 37,
                draws=bootstrap_draws,
            )
        plot_rows.append(
            {
                "dataset_id": dataset_id,
                "distance_bin": bin_id,
                "relative_distance": distance[0],
                "relative_gradient_mean": gradient[0],
                "relative_gradient_ci_low": gradient[1],
                "relative_gradient_ci_high": gradient[2],
                "relative_abs_advantage_mean": advantage[0],
                "relative_abs_advantage_ci_low": advantage[1],
                "relative_abs_advantage_ci_high": advantage[2],
                "n_seeds": len(local),
                "n_samples": sum(int(x["n_samples"]) for x in local),
            }
        )
    near = [x for x in binned if x["pair_role"] == "near"]
    far = [x for x in binned if x["pair_role"] == "far"]
    summary = {
        "dataset_id": dataset_id,
        "distance_edges": edges.tolist(),
        "matched_pairs_per_seed": len(near) // len(expected_seeds),
        "point_rows": len(binned),
        "farthest_relative_distance": plot_rows[-1]["relative_distance"],
        "farthest_relative_gradient": plot_rows[-1]["relative_gradient_mean"],
        "farthest_relative_abs_advantage": plot_rows[-1]["relative_abs_advantage_mean"],
        "absolute_advantage_far_near_ratio_all_points": float(
            np.mean([x["abs_advantage"] for x in far])
            / max(np.mean([x["abs_advantage"] for x in near]), 1e-12)
        ),
        "n_seeds": len(expected_seeds),
    }
    return binned, seed_curves, plot_rows, summary


def aggregate_dataset(
    output_dir: Path, dataset: dict[str, Any], runspec: dict[str, Any]
) -> dict[str, Any]:
    from drpo import e7_hopper_q2 as q2

    points = []
    for seed in runspec["protocol"]["seeds"]:
        points.extend(read_csv(output_dir / f"seed_{seed}" / "gradient_probe_points.csv"))
    aggregation = runspec["protocol"]["aggregation"]
    seed_value = int(payload_hash(dataset["id"])[:8], 16) + aggregation["bootstrap_seed"]
    binned, seed_curves, plot_rows, summary = build_plot_rows(
        points,
        dataset_id=dataset["id"],
        seeds=runspec["protocol"]["seeds"],
        far_bins=runspec["protocol"]["probe"]["far_distance_bins"],
        bootstrap_draws=aggregation["bootstrap_draws"],
        bootstrap_seed=seed_value,
    )
    q2.write_csv(output_dir / "dataset_gradient_probe_points.csv", binned)
    q2.write_csv(output_dir / "dataset_seed_curves.csv", seed_curves)
    q2.write_csv(output_dir / "dataset_plot_data.csv", plot_rows)
    summary.update(
        dataset_sha256=dataset["sha256"],
        scientific_status="pilot",
        completed_utc=q2.utc_now(),
    )
    q2.atomic_write_json(output_dir / "DATASET_AGGREGATE_COMPLETE.json", summary)
    return summary


def dataset_worker(payload: dict[str, Any]) -> dict[str, Any]:
    from drpo import e7_bench
    from drpo import e7_hopper_q2 as q2

    dataset = payload["dataset"]
    output_dir = Path(payload["work_dir"]) / "datasets" / dataset["id"]
    try:
        configure_threads(payload["cpus_per_worker"])
        runspec = load_runspec(payload["runspec_path"])
        data = e7_bench.load_dataset(
            Path(dataset["resolved_path"]),
            dataset_spec(dataset, dataset["sha256"]),
            max_transitions=None,
        )
        config, mode = adapted_config(runspec, dataset)
        device = q2.choose_device(payload["device"])
        canonical_root = output_dir / "canonical_critic"
        if canonical_root.exists() and not payload["resume"]:
            raise RuntimeError(f"canonical critic exists for {dataset['id']}; pass --resume")
        manifest = {
            "basename": Path(dataset["resolved_path"]).name,
            "sha256": dataset["sha256"],
            "size_bytes": dataset["size_bytes"],
        }
        canonical_binding = {
            "experiment_id": EXPERIMENT_ID,
            "implementation_origin": "EXT-H-E7-Q2",
            "runspec_sha256": runspec["_sha256"],
            "base_config_sha256": runspec["_base_config_sha256"],
            "source_commit_sha": payload["source_commit"],
            "runner_sha256": payload["runner_sha256"],
            "dataset_id": dataset["id"],
            "dataset_sha256": dataset["sha256"],
            "canonical_critic_seed": mode.canonical_critic_seed,
            "critic_steps": mode.critic_max_steps,
            "critic_eval_interval": mode.critic_eval_interval,
            "model": {
                "hidden_sizes": list(config.hidden_sizes),
                "activation": config.activation,
                "init_scheme": config.init_scheme,
                "init_gain": config.init_gain,
            },
            "critic": {
                "gamma": config.gamma,
                "learning_rate": config.critic_lr,
                "batch_size": config.critic_batch_size,
                "weight_decay": config.weight_decay,
            },
        }
        canonical_binding_path = output_dir / "CANONICAL_CRITIC_INPUT_BINDING.json"
        if canonical_binding_path.is_file():
            previous_binding = json.loads(canonical_binding_path.read_text())
            if previous_binding != canonical_binding:
                raise RuntimeError(
                    "canonical critic input binding mismatch for "
                    f"{dataset['id']}"
                )
        q2.atomic_write_json(canonical_binding_path, canonical_binding)
        canonical = q2.prepare_canonical_critic_context(
            data=data,
            config=config,
            mode=mode,
            mode_name="pilot",
            config_path=canonical_binding_path,
            dataset_manifest=manifest,
            device=device,
            artifact_root=canonical_root,
            reuse_root=None,
            heartbeat=lambda stage, step: heartbeat(
                output_dir / "HEARTBEAT.json", stage, step, dataset_id=dataset["id"]
            ),
        )
        q2.atomic_write_json(
            output_dir / "CANONICAL_CRITIC_BINDING.json",
            {
                **canonical_binding,
                "helper_manifest_experiment_id": canonical.artifact_manifest[
                    "identity"
                ]["experiment_id"],
                "helper_manifest_runner_version": canonical.artifact_manifest[
                    "identity"
                ]["runner_version"],
                "canonical_critic_checkpoint_sha256": canonical.artifact_manifest[
                    "files"
                ]["critic_checkpoint"]["sha256"],
            },
        )
        completions = [
            run_seed(
                seed=seed,
                dataset=dataset,
                runspec=runspec,
                data=data,
                config=config,
                mode=mode,
                canonical=canonical,
                device=device,
                output_dir=output_dir / f"seed_{seed}",
                resume=payload["resume"],
                source_commit=payload["source_commit"],
                runner_sha=payload["runner_sha256"],
            )
            for seed in runspec["protocol"]["seeds"]
        ]
        result = {
            "dataset_id": dataset["id"],
            "dataset_sha256": dataset["sha256"],
            "transitions": data.size,
            "episodes": int(data.episode_ids.max()) + 1,
            "seed_count": len(completions),
            "aggregate": aggregate_dataset(output_dir, dataset, runspec),
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


def plot_appendix(rows: Sequence[dict[str, Any]], output_dir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(3, 3, figsize=(13.2, 10.2))
    handles = labels = None
    for axis, dataset_id in zip(axes.flat, D4RL9):
        local = sorted(
            (x for x in rows if x["dataset_id"] == dataset_id),
            key=lambda x: int(x["distance_bin"]),
        )
        x = [float(row["relative_distance"]) for row in local]
        for key, low, high, marker, linestyle, label in (
            (
                "relative_gradient_mean",
                "relative_gradient_ci_low",
                "relative_gradient_ci_high",
                "o",
                "-",
                "implemented actor-gradient",
            ),
            (
                "relative_abs_advantage_mean",
                "relative_abs_advantage_ci_low",
                "relative_abs_advantage_ci_high",
                "s",
                "--",
                "matched |advantage|",
            ),
        ):
            axis.plot(
                x,
                [float(row[key]) for row in local],
                marker=marker,
                linestyle=linestyle,
                label=label,
            )
            axis.fill_between(
                x,
                [float(row[low]) for row in local],
                [float(row[high]) for row in local],
                alpha=0.15,
            )
        axis.axhline(1.0, linewidth=0.8, alpha=0.6)
        axis.set(
            title=dataset_id.replace("-v2", "").replace("-", " "),
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


def aggregate_run(work_dir: Path, runspec: dict[str, Any]) -> dict[str, Any]:
    from drpo import e7_hopper_q2 as q2

    output_dir = work_dir / "aggregate"
    output_dir.mkdir(parents=True, exist_ok=True)
    points, curves, plot_rows, summaries = [], [], [], []
    for dataset_id in D4RL9:
        source = work_dir / "datasets" / dataset_id
        points.extend(read_csv(source / "dataset_gradient_probe_points.csv"))
        curves.extend(read_csv(source / "dataset_seed_curves.csv"))
        plot_rows.extend(read_csv(source / "dataset_plot_data.csv"))
        summaries.append(json.loads((source / "DATASET_AGGREGATE_COMPLETE.json").read_text()))
    q2.write_csv(output_dir / "d4rl9_gradient_probe_points.csv", points)
    q2.write_csv(output_dir / "d4rl9_seed_curves.csv", curves)
    q2.write_csv(output_dir / "d4rl9_plot_data.csv", plot_rows)
    q2.write_csv(output_dir / "dataset_summary.csv", summaries)
    result = {
        "dataset_count": 9,
        "seed_count_per_dataset": len(runspec["protocol"]["seeds"]),
        "gradient_point_rows": len(points),
        "plot_data_rows": len(plot_rows),
        "plot_paths": plot_appendix(plot_rows, output_dir),
        "scientific_status": "pilot",
        "completed_utc": q2.utc_now(),
    }
    q2.atomic_write_json(output_dir / "AGGREGATE_COMPLETE.json", result)
    return result


def plan(runspec: dict[str, Any], dataset_root: str | Path) -> dict[str, Any]:
    return {
        "experiment_id": EXPERIMENT_ID,
        "scientific_status_cap": "pilot",
        "dataset_root": str(Path(dataset_root).expanduser().resolve()),
        "datasets": list(D4RL9),
        "seeds": runspec["protocol"]["seeds"],
        "critic_jobs": 9,
        "positive_actor_probe_jobs": 90,
        "method_branch_jobs": 0,
        "dataset_workers": runspec["execution"]["dataset_workers"],
        "cpus_per_worker": runspec["execution"]["cpus_per_worker"],
        "device_pool": runspec["execution"]["device_pool"],
        "probe": runspec["protocol"]["probe"],
    }


def run_figure1(args: argparse.Namespace) -> int:
    from drpo import e7_hopper_q2 as q2

    runspec = load_runspec(args.runspec)
    if args.plan:
        print(json.dumps(plan(runspec, args.dataset_root), indent=2))
        return 0
    git_state = q2.collect_git_state(ROOT)
    if not git_state.get("available") or str(git_state.get("status_porcelain", "")).strip():
        raise RuntimeError("Figure-1 execution requires a clean Git checkout")
    source_commit = str(git_state["head"])
    runner_sha = q2.sha256_file(Path(__file__).resolve())
    work_dir = Path(args.work_dir).expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    lock = lock_datasets(runspec, args.dataset_root)
    lock_path = work_dir / "DATASET_LOCK.json"
    if lock_path.is_file():
        previous = json.loads(lock_path.read_text())
        if not args.resume or previous.get("dataset_lock_sha256") != lock["dataset_lock_sha256"]:
            raise RuntimeError("work directory dataset lock conflict")
    q2.atomic_write_json(lock_path, lock)
    resolved = {
        **{key: value for key, value in runspec.items() if not key.startswith("_")},
        "runspec_sha256": runspec["_sha256"],
        "dataset_lock_sha256": lock["dataset_lock_sha256"],
        "base_config_sha256": runspec["_base_config_sha256"],
        "source_commit_sha": source_commit,
        "runner_sha256": runner_sha,
    }
    q2.atomic_write_json(work_dir / "RUNSPEC_RESOLVED.json", resolved)
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
    complete_path = work_dir / "RUN_COMPLETE.json"
    identity = {
        "runspec_sha256": runspec["_sha256"],
        "dataset_lock_sha256": lock["dataset_lock_sha256"],
        "base_config_sha256": runspec["_base_config_sha256"],
        "source_commit_sha": source_commit,
        "runner_sha256": runner_sha,
    }
    if complete_path.is_file():
        existing = json.loads(complete_path.read_text())
        if args.resume and all(existing.get(key) == value for key, value in identity.items()):
            return 0
        raise RuntimeError("RUN_COMPLETE.json has a conflicting identity; use a new work directory")
    devices = (
        [item.strip() for item in args.device_pool.split(",") if item.strip()]
        if args.device_pool
        else list(runspec["execution"]["device_pool"])
    )
    workers = min(args.workers or runspec["execution"]["dataset_workers"], 9)
    cpus = args.cpus_per_worker or runspec["execution"]["cpus_per_worker"]
    if not devices or min(workers, cpus) < 1:
        raise ValueError("device pool, workers, and cpus-per-worker must be positive")
    available = os.cpu_count() or 1
    if workers * cpus > available:
        raise RuntimeError(f"refusing CPU oversubscription: {workers} x {cpus} > {available}")
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
    existing_manifest_path = work_dir / "RUN_MANIFEST.json"
    if existing_manifest_path.is_file():
        existing_manifest = json.loads(existing_manifest_path.read_text())
        if not args.resume or any(
            existing_manifest.get(key) != value for key, value in identity.items()
        ):
            raise RuntimeError(
                "work directory run identity mismatch; use --resume or a new work directory"
            )
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "runner_version": RUNNER_VERSION,
        "scientific_status_cap": "pilot",
        **identity,
        "git_state": git_state,
        "execution_plan": plan(runspec, args.dataset_root),
        "started_utc": q2.utc_now(),
    }
    q2.atomic_write_json(work_dir / "RUN_MANIFEST.json", manifest)
    try:
        if workers == 1:
            results = [dataset_worker(payload) for payload in payloads]
        else:
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=workers, mp_context=multiprocessing.get_context("spawn")
            ) as executor:
                futures = [executor.submit(dataset_worker, payload) for payload in payloads]
                results = []
                try:
                    for future in concurrent.futures.as_completed(futures):
                        results.append(future.result())
                except Exception:
                    for pending in futures:
                        pending.cancel()
                    raise
        by_id = {row["dataset_id"]: row for row in results}
        complete = {
            **manifest,
            "dataset_results": [by_id[item] for item in D4RL9],
            "aggregate": aggregate_run(work_dir, runspec),
            "raw_complete": True,
            "task_performance_collapse": "not_evaluated",
            "support_or_variance_boundary": "preserved_per_positive_only_seed_audit",
            "nan_inf_numerical_failure": "fail_closed",
            "completed_utc": q2.utc_now(),
        }
        q2.atomic_write_json(complete_path, complete)
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
    points = []
    for seed in range(100, 110):
        advantage = np.exp(rng.normal(scale=0.25, size=64))
        near = np.exp(rng.normal(scale=0.08, size=64))
        far = np.linspace(2.8, 6.0, 64) * np.exp(rng.normal(scale=0.025, size=64))
        for pair_id in range(64):
            for role, distance, gradient, magnitude in (
                ("near", near[pair_id], near[pair_id] * advantage[pair_id], advantage[pair_id]),
                (
                    "far",
                    far[pair_id],
                    far[pair_id] * advantage[pair_id],
                    advantage[pair_id] * (1 + rng.normal(scale=0.002)),
                ),
            ):
                points.append(
                    {
                        "dataset_id": "synthetic",
                        "seed": seed,
                        "pair_id": pair_id,
                        "pair_role": role,
                        "transition_index": seed * 1000 + pair_id,
                        "episode_id": pair_id,
                        "abs_advantage": magnitude,
                        "standardized_distance": distance,
                        "full_parameter_gradient_norm": gradient,
                    }
                )
    binned, _, plot_rows, summary = build_plot_rows(
        points,
        dataset_id="synthetic",
        seeds=tuple(range(100, 110)),
        far_bins=7,
        bootstrap_draws=1000,
        bootstrap_seed=42,
    )
    assert len(binned) == 1280 and len(plot_rows) == 8
    assert plot_rows[0]["distance_bin"] == -1 and plot_rows[0]["n_samples"] == 640
    assert sum(row["n_samples"] for row in plot_rows[1:]) == 640
    assert plot_rows[-1]["relative_gradient_mean"] > plot_rows[1]["relative_gradient_mean"]
    assert math.isclose(summary["absolute_advantage_far_near_ratio_all_points"], 1.0, abs_tol=0.01)
    print(
        json.dumps(
            {
                "self_test": "passed",
                "selected_point_rows": len(binned),
                "plot_rows": len(plot_rows),
            },
            indent=2,
        )
    )
    return 0


def figure1_main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(description="D4RL-9 Figure-1 gradient coverage diagnostic")
    parser.add_argument("--runspec", default="configs/e7_figure1_d4rl9_runspec.yaml")
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
