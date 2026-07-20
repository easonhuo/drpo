"""Hopper E7-Q2 Positive-only preparation and six-branch orchestration."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from drpo_reference.common.io import atomic_json, write_csv
from drpo_reference.common.seeding import seed_all

from .hopper_actor import Heartbeat, train_actor_stage
from .hopper_metrics import (
    create_gradient_probe,
    match_near_far_indices,
    resolve_global_scale,
)
from .hopper_models import SquashedGaussianPolicy
from .hopper_optim import tensor
from .hopper_protocol import METHODS, HopperProtocol

StageRunner = Callable[..., tuple[SquashedGaussianPolicy, dict[str, Any]]]


@dataclass
class PreparedActor:
    """Prepared policy and frozen branch inputs for one Hopper actor seed."""

    policy: SquashedGaussianPolicy
    checkpoint_path: Path
    checkpoint_sha256: str
    positive_audit: dict[str, Any]
    train_indices: np.ndarray
    positive_train_indices: np.ndarray
    negative_train_indices: np.ndarray
    audit_indices: np.ndarray
    fixed_negative_indices: np.ndarray
    near_indices: np.ndarray
    far_indices: np.ndarray
    matching_summary: dict[str, Any]
    gradient_summary: dict[str, Any]
    far_threshold: float
    far_cap_score: float
    global_budget: dict[str, Any]
    reload_identity: bool


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _state_digest(policy: SquashedGaussianPolicy) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(policy.state_dict().items()):
        digest.update(name.encode("utf-8"))
        array = value.detach().cpu().contiguous().numpy()
        digest.update(str(array.dtype).encode("utf-8"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _state_dicts_equal(
    first: dict[str, torch.Tensor],
    second: dict[str, torch.Tensor],
) -> bool:
    if set(first) != set(second):
        return False
    return all(torch.equal(first[name], second[name]) for name in first)


def _validate_inputs(
    observations: np.ndarray,
    actions: np.ndarray,
    advantages: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
) -> None:
    if observations.ndim != 2 or actions.ndim != 2:
        raise ValueError("observations and actions must be rank-two arrays")
    if advantages.ndim != 1:
        raise ValueError("advantages must be a rank-one array")
    if not (len(observations) == len(actions) == len(advantages)):
        raise ValueError("observations, actions, and advantages disagree")
    population = len(advantages)
    for name, indices in (
        ("train_indices", train_indices),
        ("validation_indices", validation_indices),
    ):
        if indices.ndim != 1 or len(indices) == 0:
            raise ValueError(f"{name} must be a non-empty rank-one array")
        if len(np.unique(indices)) != len(indices):
            raise ValueError(f"{name} contains duplicate indices")
        if int(np.min(indices)) < 0 or int(np.max(indices)) >= population:
            raise ValueError(f"{name} contains an out-of-range index")
    if np.intersect1d(train_indices, validation_indices).size:
        raise ValueError("train and validation indices must be disjoint")
    if not np.isfinite(observations).all():
        raise ValueError("observations contain NaN or Inf")
    if not np.isfinite(actions).all():
        raise ValueError("actions contain NaN or Inf")
    if not np.isfinite(advantages).all():
        raise ValueError("advantages contain NaN or Inf")


def _require_new_or_empty(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Hopper suite output must be new or empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def make_policy(
    protocol: HopperProtocol,
    observation_dim: int,
    action_dim: int,
    device: torch.device | str,
) -> SquashedGaussianPolicy:
    """Construct the one canonical actor architecture for E7-Q2."""

    return SquashedGaussianPolicy(
        observation_dim,
        action_dim,
        protocol.hidden_sizes,
        protocol.log_std_min,
        protocol.log_std_max,
        protocol.action_clip_epsilon,
        protocol.activation,
        protocol.init_scheme,
        protocol.init_gain,
    ).to(torch.device(device))


def clone_policy(
    policy: SquashedGaussianPolicy,
    protocol: HopperProtocol,
    observation_dim: int,
    action_dim: int,
    device: torch.device | str,
) -> SquashedGaussianPolicy:
    """Clone one prepared actor without sharing parameter storage."""

    clone = make_policy(
        protocol,
        observation_dim,
        action_dim,
        device,
    )
    clone.load_state_dict(policy.state_dict())
    return clone


def _load_prepared_checkpoint(
    checkpoint_path: Path,
    protocol: HopperProtocol,
    observation_dim: int,
    action_dim: int,
    device: torch.device,
) -> tuple[SquashedGaussianPolicy, dict[str, Any]]:
    payload = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    if payload.get("method") != "positive_only":
        raise RuntimeError("prepared checkpoint is not Positive-only")
    if payload.get("checkpoint_role") != "fixed_budget_final_checkpoint":
        raise RuntimeError("prepared checkpoint role is not fixed-budget final")
    if not bool(payload.get("fixed_budget_completed")):
        raise RuntimeError("Positive-only preparation did not complete its budget")
    if int(payload.get("fixed_budget_steps", -1)) != protocol.positive_steps:
        raise RuntimeError("prepared checkpoint has the wrong step budget")
    policy = make_policy(
        protocol,
        observation_dim,
        action_dim,
        device,
    )
    policy.load_state_dict(payload["model"])
    return policy, payload


def prepare_positive_only_actor(
    *,
    observations: np.ndarray,
    actions: np.ndarray,
    advantages: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    protocol: HopperProtocol,
    seed: int,
    device: torch.device | str,
    output_dir: Path,
    heartbeat: Heartbeat | None = None,
    stage_runner: StageRunner = train_actor_stage,
) -> PreparedActor:
    """Train, persist, reload, and calibrate one shared actor start."""

    _validate_inputs(
        observations,
        actions,
        advantages,
        train_indices,
        validation_indices,
    )
    resolved_device = torch.device(device)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_all(seed)

    positive_train = train_indices[advantages[train_indices] > 0]
    negative_train = train_indices[advantages[train_indices] < 0]
    if len(positive_train) < 10 or len(negative_train) < 10:
        raise RuntimeError("frozen advantages do not yield enough positive and negative samples")
    validation_positive = validation_indices[advantages[validation_indices] > 0]
    validation_negative = validation_indices[advantages[validation_indices] < 0]
    if len(validation_positive) == 0 or len(validation_negative) == 0:
        raise RuntimeError("validation split must contain positive and negative advantages")

    generator = np.random.default_rng(seed + 321)
    half = max(1, protocol.audit_sample_size // 2)
    audit_positive = generator.choice(
        validation_positive,
        size=min(half, len(validation_positive)),
        replace=False,
    )
    audit_negative = generator.choice(
        validation_negative,
        size=min(half, len(validation_negative)),
        replace=False,
    )
    audit_indices = np.concatenate([audit_positive, audit_negative]).astype(np.int64)
    generator.shuffle(audit_indices)
    fixed_negative_indices = generator.choice(
        negative_train,
        size=min(protocol.audit_sample_size, len(negative_train)),
        replace=False,
    ).astype(np.int64)

    initial_policy = make_policy(
        protocol,
        observations.shape[1],
        actions.shape[1],
        resolved_device,
    )
    trained_policy, positive_audit = stage_runner(
        policy=initial_policy,
        method="positive_only",
        observations=observations,
        actions=actions,
        advantages=advantages,
        train_indices=positive_train,
        audit_indices=audit_indices,
        fixed_negative_indices=fixed_negative_indices,
        protocol=protocol,
        min_steps=protocol.positive_min_steps,
        max_steps=protocol.positive_steps,
        eval_interval=protocol.actor_eval_interval,
        seed=seed + 500_000,
        device=resolved_device,
        output_dir=output_dir / "positive_only_initialization",
        heartbeat=heartbeat,
    )
    if positive_audit.get("numerical_nonfinite"):
        raise RuntimeError("Positive-only preparation ended with NaN or Inf")
    if not positive_audit.get("fixed_budget_completed"):
        raise RuntimeError("Positive-only preparation did not finish its budget")

    checkpoint_path = output_dir / "positive_only_initialization" / "terminal_actor.pt"
    trained_state = {
        name: value.detach().cpu().clone() for name, value in trained_policy.state_dict().items()
    }
    reloaded_policy, checkpoint_payload = _load_prepared_checkpoint(
        checkpoint_path,
        protocol,
        observations.shape[1],
        actions.shape[1],
        resolved_device,
    )
    reloaded_state = {
        name: value.detach().cpu().clone() for name, value in reloaded_policy.state_dict().items()
    }
    reload_identity = _state_dicts_equal(trained_state, reloaded_state)
    if not reload_identity:
        raise RuntimeError("Positive-only checkpoint reload changed actor state")

    all_negative_distances = np.full(
        len(advantages),
        np.nan,
        dtype=np.float32,
    )
    with torch.no_grad():
        for offset in range(0, len(negative_train), 65_536):
            indices = negative_train[offset : offset + 65_536]
            all_negative_distances[indices] = (
                reloaded_policy.standardized_distance(
                    tensor(observations[indices], resolved_device),
                    tensor(actions[indices], resolved_device),
                )
                .cpu()
                .numpy()
            )

    near_indices, far_indices, matching_summary = match_near_far_indices(
        advantages,
        all_negative_distances,
        negative_train,
        protocol.near_quantile,
        protocol.far_quantile,
        protocol.advantage_bins,
        protocol.matched_pairs,
        protocol.advantage_match_relative_tolerance,
        seed,
    )
    probe_dir = output_dir / "probes"
    pair_rows = [
        {
            "pair_id": pair_id,
            "near_index": int(near_index),
            "far_index": int(far_index),
            "near_abs_advantage": float(abs(advantages[near_index])),
            "far_abs_advantage": float(abs(advantages[far_index])),
            "near_distance": float(all_negative_distances[near_index]),
            "far_distance": float(all_negative_distances[far_index]),
        }
        for pair_id, (near_index, far_index) in enumerate(zip(near_indices, far_indices))
    ]
    write_csv(probe_dir / "matching_pairs.csv", pair_rows)
    atomic_json(probe_dir / "matching_summary.json", matching_summary)
    gradient_summary = create_gradient_probe(
        policy=reloaded_policy,
        observations=observations,
        actions=actions,
        advantages=advantages,
        near_indices=near_indices,
        far_indices=far_indices,
        population_indices=fixed_negative_indices,
        max_gradient_pairs=min(
            protocol.gradient_probe_pairs,
            len(near_indices),
        ),
        distance_bins=protocol.distance_bins,
        device=resolved_device,
        output_dir=probe_dir,
    )

    far_threshold = float((matching_summary["near_cut"] + matching_summary["far_cut"]) / 2.0)
    near_negative_pool = negative_train[
        all_negative_distances[negative_train] <= matching_summary["near_cut"]
    ]
    if len(near_negative_pool) == 0:
        raise RuntimeError("no near negatives are available for Far-cap")
    with torch.no_grad():
        near_joint_scores = (
            reloaded_policy.output_score_norm(
                tensor(observations[near_negative_pool], resolved_device),
                tensor(actions[near_negative_pool], resolved_device),
            )
            .cpu()
            .numpy()
        )
    far_cap_score = float(
        np.quantile(
            near_joint_scores,
            protocol.far_cap_reference_quantile,
        )
    )
    far_cap_definition = {
        "reference_population": "near_negative_pool",
        "reference_quantile": protocol.far_cap_reference_quantile,
        "far_cap_score": far_cap_score,
        "far_threshold": far_threshold,
        "detached": True,
    }
    atomic_json(
        probe_dir / "far_cap_definition.json",
        far_cap_definition,
    )
    global_budget = resolve_global_scale(
        policy=reloaded_policy,
        observations=observations,
        actions=actions,
        advantages=advantages,
        negative_indices=negative_train,
        far_threshold=far_threshold,
        far_cap_score=far_cap_score,
        audit_size=protocol.global_budget_audit_size,
        seed=seed,
        device=resolved_device,
    )
    atomic_json(
        probe_dir / "global_budget_match.json",
        global_budget,
    )

    checkpoint_sha256 = _sha256_file(checkpoint_path)
    prepared_manifest = {
        "seed": seed,
        "positive_training_seed": seed + 500_000,
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": checkpoint_sha256,
            "size_bytes": checkpoint_path.stat().st_size,
            "checkpoint_role": checkpoint_payload["checkpoint_role"],
            "fixed_budget_steps": checkpoint_payload["fixed_budget_steps"],
        },
        "checkpoint_reload_identity": reload_identity,
        "prepared_state_sha256": _state_digest(reloaded_policy),
        "train_count": int(len(train_indices)),
        "positive_train_count": int(len(positive_train)),
        "negative_train_count": int(len(negative_train)),
        "audit_indices": audit_indices.tolist(),
        "fixed_negative_indices": fixed_negative_indices.tolist(),
        "near_indices": near_indices.tolist(),
        "far_indices": far_indices.tolist(),
        "far_threshold": far_threshold,
        "far_cap_score": far_cap_score,
        "global_scale": float(global_budget["global_scale"]),
        "matching": matching_summary,
        "gradient_probe": gradient_summary,
        "positive_audit": positive_audit,
    }
    atomic_json(
        output_dir / "prepared_actor_manifest.json",
        prepared_manifest,
    )
    return PreparedActor(
        policy=reloaded_policy,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
        positive_audit=positive_audit,
        train_indices=train_indices.copy(),
        positive_train_indices=positive_train.copy(),
        negative_train_indices=negative_train.copy(),
        audit_indices=audit_indices,
        fixed_negative_indices=fixed_negative_indices,
        near_indices=near_indices,
        far_indices=far_indices,
        matching_summary=matching_summary,
        gradient_summary=gradient_summary,
        far_threshold=far_threshold,
        far_cap_score=far_cap_score,
        global_budget=global_budget,
        reload_identity=reload_identity,
    )


def run_hopper_six_branch_suite(
    *,
    observations: np.ndarray,
    actions: np.ndarray,
    advantages: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    protocol: HopperProtocol,
    seed: int,
    device: torch.device | str,
    output_dir: Path,
    heartbeat: Heartbeat | None = None,
    stage_runner: StageRunner = train_actor_stage,
    continue_on_branch_failure: bool = True,
) -> dict[str, Any]:
    """Run all six methods from one reloaded Positive-only checkpoint."""

    _require_new_or_empty(output_dir)
    resolved_device = torch.device(device)
    prepared = prepare_positive_only_actor(
        observations=observations,
        actions=actions,
        advantages=advantages,
        train_indices=train_indices,
        validation_indices=validation_indices,
        protocol=protocol,
        seed=seed,
        device=resolved_device,
        output_dir=output_dir,
        heartbeat=heartbeat,
        stage_runner=stage_runner,
    )
    prepared_digest = _state_digest(prepared.policy)
    branch_audits: dict[str, Any] = {}
    branch_failures: dict[str, Any] = {}
    branch_initial_digests: dict[str, str] = {}

    for method in METHODS:
        policy = clone_policy(
            prepared.policy,
            protocol,
            observations.shape[1],
            actions.shape[1],
            resolved_device,
        )
        initial_digest = _state_digest(policy)
        branch_initial_digests[method] = initial_digest
        if initial_digest != prepared_digest:
            raise RuntimeError(f"branch {method} did not clone the prepared actor")
        try:
            _, audit = stage_runner(
                policy=policy,
                method=method,
                observations=observations,
                actions=actions,
                advantages=advantages,
                train_indices=prepared.train_indices,
                audit_indices=prepared.audit_indices,
                fixed_negative_indices=(prepared.fixed_negative_indices),
                protocol=protocol,
                min_steps=protocol.branch_min_steps,
                max_steps=protocol.branch_steps,
                eval_interval=protocol.actor_eval_interval,
                seed=seed,
                device=resolved_device,
                output_dir=output_dir / "methods" / method,
                far_threshold=prepared.far_threshold,
                global_scale=float(prepared.global_budget["global_scale"]),
                far_cap_score=prepared.far_cap_score,
                heartbeat=heartbeat,
            )
            branch_audits[method] = audit
        except Exception as exc:
            failure = {
                "method": method,
                "exception_type": type(exc).__name__,
                "message": str(exc),
                "branch_initial_state_sha256": initial_digest,
                "prepared_state_sha256": prepared_digest,
                "failure_isolated": True,
            }
            branch_failures[method] = failure
            atomic_json(
                output_dir / "methods" / method / "branch_failure.json",
                failure,
            )
            if not continue_on_branch_failure:
                raise

    completed_methods = [method for method in METHODS if method in branch_audits]
    failed_methods = [method for method in METHODS if method in branch_failures]
    summary = {
        "seed": seed,
        "execution_profile": protocol.execution_profile,
        "formal_evidence_allowed": False,
        "method_order": list(METHODS),
        "positive_training_seed": seed + 500_000,
        "branch_training_seed": seed,
        "prepared_checkpoint": {
            "path": str(prepared.checkpoint_path),
            "sha256": prepared.checkpoint_sha256,
            "reload_identity": prepared.reload_identity,
            "state_sha256": prepared_digest,
        },
        "all_branch_initial_states_identical": bool(
            len(set(branch_initial_digests.values())) == 1
            and set(branch_initial_digests) == set(METHODS)
        ),
        "branch_initial_state_sha256": branch_initial_digests,
        "positive_only_initialization": prepared.positive_audit,
        "matching": prepared.matching_summary,
        "gradient_probe": prepared.gradient_summary,
        "initial_global_budget_diagnostic": prepared.global_budget,
        "global_budget": prepared.global_budget,
        "far_threshold": prepared.far_threshold,
        "far_cap_score": prepared.far_cap_score,
        "methods": branch_audits,
        "branch_failures": branch_failures,
        "completed_methods": completed_methods,
        "failed_methods": failed_methods,
        "all_methods_completed": not failed_methods,
        "suite_status": "complete" if not failed_methods else "partial_failure",
        "rollout_included": False,
        "public_runner_included": False,
        "scientific_status_changed": False,
    }
    atomic_json(output_dir / "suite_summary.json", summary)
    completion = {
        "seed": seed,
        "suite_status": summary["suite_status"],
        "expected_methods": list(METHODS),
        "completed_methods": completed_methods,
        "failed_methods": failed_methods,
        "prepared_checkpoint_sha256": prepared.checkpoint_sha256,
        "all_branch_initial_states_identical": summary["all_branch_initial_states_identical"],
        "terminal_audits_complete": bool(
            not failed_methods
            and all(audit.get("terminal_audit_complete") for audit in branch_audits.values())
        ),
        "formal_evidence_allowed": False,
    }
    atomic_json(output_dir / "SUITE_COMPLETE.json", completion)
    return summary
