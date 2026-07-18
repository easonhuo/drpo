"""Public Hopper E7-Q2 runner and multi-seed aggregation.

This module composes the already migrated Hopper data, critic, actor-suite, and
rollout layers. It does not discover historical results, package archives, or
upgrade an engineering run into a scientific claim.
"""

from __future__ import annotations

import hashlib
import json
import math
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch

from drpo_reference.common import atomic_json, seed_all, write_csv
from drpo_reference.external.hopper_actor import train_actor_stage
from drpo_reference.external.hopper_critic import train_critic
from drpo_reference.external.hopper_data import (
    Normalizer,
    OfflineData,
    discounted_returns,
    load_hopper_hdf5,
    split_episode_indices,
)
from drpo_reference.external.hopper_protocol import METHODS, HopperProtocol, smoke_protocol
from drpo_reference.external.hopper_rollout import (
    evaluate_d4rl_rollouts,
    preflight_from_protocol,
)
from drpo_reference.external.hopper_suite import run_hopper_six_branch_suite

EXPERIMENT_ID = "EXT-H-E7-Q2"
PUBLIC_RUNNER_VERSION = "1.0.0"


@dataclass(frozen=True)
class HopperExecutionPlan:
    """Resolved public-runner identity without changing frozen budgets."""

    protocol: HopperProtocol
    seeds: tuple[int, ...]
    execution_kind: str
    formal_evidence_eligible: bool
    method_ranking_claim_allowed: bool = False


@dataclass
class CanonicalCriticContext:
    """One verified critic/frozen-advantage artifact shared by all actor seeds."""

    root: Path
    split: dict[str, np.ndarray]
    observation_normalizer: Normalizer
    advantage_arrays: dict[str, np.ndarray]
    critic_audit: dict[str, Any]
    artifact_manifest: dict[str, Any]
    reused: bool


SuiteRunner = Callable[..., dict[str, Any]]
PreflightRunner = Callable[..., dict[str, Any]]
RolloutRunner = Callable[..., dict[str, Any]]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_fingerprint(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_new_or_empty(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Hopper output root must be new or empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _resolve_device(requested: str | torch.device) -> torch.device:
    if isinstance(requested, torch.device):
        return requested
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def resolve_hopper_execution(
    *,
    seeds: Sequence[int] | None = None,
    smoke: bool = False,
) -> HopperExecutionPlan:
    """Resolve formal, formal-subset, or smoke execution without claim drift."""

    protocol = smoke_protocol() if smoke else HopperProtocol()
    selected = tuple(protocol.formal_seeds if seeds is None else (int(seed) for seed in seeds))
    if not selected:
        raise ValueError("at least one Hopper seed is required")
    if len(set(selected)) != len(selected):
        raise ValueError("Hopper seed list contains duplicates")

    if smoke:
        return HopperExecutionPlan(
            protocol=protocol,
            seeds=selected,
            execution_kind="smoke_non_evidence",
            formal_evidence_eligible=False,
        )

    registered = HopperProtocol().formal_seeds
    registered_set = set(registered)
    unknown = [seed for seed in selected if seed not in registered_set]
    if unknown:
        raise ValueError(
            "non-smoke Hopper seeds must be a subset of the registered formal seeds; "
            f"unknown seeds: {unknown}"
        )
    registered_order = tuple(seed for seed in registered if seed in set(selected))
    if selected != registered_order:
        raise ValueError("Hopper seed subsets must preserve registered seed order")
    full = selected == registered
    return HopperExecutionPlan(
        protocol=protocol,
        seeds=selected,
        execution_kind=("formal" if full else "formal_subset_non_evidence"),
        formal_evidence_eligible=full,
    )


def validate_dataset_identity(
    dataset_path: Path,
    protocol: HopperProtocol,
) -> dict[str, Any]:
    """Verify the exact registered HDF5 basename and SHA-256 before loading."""

    path = Path(dataset_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Hopper dataset does not exist: {path}")
    if path.name != protocol.dataset_basename:
        raise ValueError(
            "Hopper dataset basename mismatch: "
            f"expected {protocol.dataset_basename}, got {path.name}"
        )
    digest = _sha256_file(path)
    if digest != protocol.dataset_sha256:
        raise ValueError(
            "Hopper dataset SHA-256 mismatch: "
            f"expected {protocol.dataset_sha256}, got {digest}"
        )
    return {
        "path": str(path),
        "basename": path.name,
        "sha256": digest,
        "size_bytes": path.stat().st_size,
        "identity_verified": True,
    }


def _canonical_identity(
    *,
    protocol: HopperProtocol,
    dataset_manifest: dict[str, Any],
    data: OfflineData,
) -> dict[str, Any]:
    return {
        "experiment_id": protocol.experiment_id,
        "protocol_sha256": _json_fingerprint(asdict(protocol)),
        "dataset_basename": dataset_manifest["basename"],
        "dataset_sha256": dataset_manifest["sha256"],
        "transitions": data.size,
        "observation_dim": int(data.observations.shape[1]),
        "action_dim": int(data.actions.shape[1]),
        "canonical_critic_seed": protocol.canonical_critic_seed,
        "train_fraction": protocol.train_fraction,
        "validation_fraction": protocol.validation_fraction,
        "critic_steps": protocol.critic_steps,
    }


def _canonical_file_names() -> tuple[str, ...]:
    return (
        "canonical_critic.pt",
        "final_training_critic.pt",
        "critic_metrics.csv",
        "critic_terminal_audit.json",
        "frozen_advantages.npz",
        "split_indices.npz",
        "observation_normalizer.npz",
    )


def _hash_canonical_files(root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name in _canonical_file_names():
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"canonical critic artifact is missing {name}")
        hashes[name] = _sha256_file(path)
    return hashes


def _validate_loaded_context(
    *,
    context: CanonicalCriticContext,
    data: OfflineData,
) -> None:
    expected_keys = {"train", "validation", "test"}
    if set(context.split) != expected_keys:
        raise RuntimeError("canonical split artifact must contain train/validation/test")
    populations = []
    for name in ("train", "validation", "test"):
        indices = np.asarray(context.split[name], dtype=np.int64)
        if indices.ndim != 1 or len(indices) == 0:
            raise RuntimeError(f"canonical {name} split is empty or malformed")
        if int(indices.min()) < 0 or int(indices.max()) >= data.size:
            raise RuntimeError(f"canonical {name} split contains an invalid index")
        if len(np.unique(indices)) != len(indices):
            raise RuntimeError(f"canonical {name} split contains duplicate indices")
        populations.append(indices)
    combined = np.concatenate(populations)
    if len(np.unique(combined)) != len(combined):
        raise RuntimeError("canonical train/validation/test splits overlap")
    if set(combined.tolist()) != set(range(data.size)):
        raise RuntimeError("canonical split artifact does not cover the dataset")

    if context.observation_normalizer.mean.shape != (data.observations.shape[1],):
        raise RuntimeError("canonical observation normalizer has the wrong dimension")
    advantage = context.advantage_arrays.get("advantage")
    if advantage is None or np.asarray(advantage).shape != (data.size,):
        raise RuntimeError("canonical frozen advantages have the wrong shape")
    if not np.isfinite(np.asarray(advantage)).all():
        raise RuntimeError("canonical frozen advantages contain NaN or Inf")
    if not bool(context.critic_audit.get("fixed_budget_completed")):
        raise RuntimeError("canonical critic did not complete its fixed budget")
    if not bool(context.critic_audit.get("critic_accepted_for_frozen_advantage")):
        raise RuntimeError("canonical critic is not accepted for frozen advantages")


def prepare_canonical_critic_context(
    *,
    data: OfflineData,
    protocol: HopperProtocol,
    dataset_manifest: dict[str, Any],
    device: str | torch.device,
    artifact_root: Path,
    reuse_root: Path | None = None,
) -> CanonicalCriticContext:
    """Train exactly once or strictly verify and reuse one canonical artifact."""

    target_device = _resolve_device(device)
    expected_identity = _canonical_identity(
        protocol=protocol,
        dataset_manifest=dataset_manifest,
        data=data,
    )

    if reuse_root is not None:
        root = Path(reuse_root).expanduser().resolve()
        manifest_path = root / "canonical_critic_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"canonical critic reuse manifest does not exist: {manifest_path}"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not bool(manifest.get("complete")):
            raise RuntimeError("canonical critic reuse artifact is incomplete")
        if manifest.get("identity") != expected_identity:
            raise RuntimeError("canonical critic reuse identity does not match this run")
        actual_hashes = _hash_canonical_files(root)
        if manifest.get("files") != actual_hashes:
            raise RuntimeError("canonical critic reuse file hashes do not match")
        split_payload = np.load(root / "split_indices.npz")
        normalizer_payload = np.load(root / "observation_normalizer.npz")
        advantages_payload = np.load(root / "frozen_advantages.npz")
        context = CanonicalCriticContext(
            root=root,
            split={name: split_payload[name].astype(np.int64) for name in split_payload.files},
            observation_normalizer=Normalizer(
                mean=normalizer_payload["mean"].astype(np.float32),
                std=normalizer_payload["std"].astype(np.float32),
            ),
            advantage_arrays={
                name: advantages_payload[name] for name in advantages_payload.files
            },
            critic_audit=json.loads(
                (root / "critic_terminal_audit.json").read_text(encoding="utf-8")
            ),
            artifact_manifest=manifest,
            reused=True,
        )
        _validate_loaded_context(context=context, data=data)
        return context

    root = Path(artifact_root)
    _require_new_or_empty(root)
    split = split_episode_indices(
        data.episode_ids,
        protocol.canonical_critic_seed,
        protocol.train_fraction,
        protocol.validation_fraction,
    )
    observation_normalizer = Normalizer.fit(data.observations[split["train"]])
    returns = discounted_returns(
        data.rewards,
        data.terminals,
        data.timeouts,
        protocol.gamma,
    )
    seed_all(protocol.canonical_critic_seed)
    critic_run = train_critic(
        data=data,
        split=split,
        observation_normalizer=observation_normalizer,
        returns=returns,
        protocol=protocol,
        seed=protocol.canonical_critic_seed,
        device=target_device,
        output_dir=root,
    )
    np.savez_compressed(root / "split_indices.npz", **split)
    np.savez_compressed(
        root / "observation_normalizer.npz",
        mean=observation_normalizer.mean,
        std=observation_normalizer.std,
    )
    file_hashes = _hash_canonical_files(root)
    manifest = {
        "schema_version": 1,
        "identity": expected_identity,
        "complete": True,
        "critic_training_count": 1,
        "shared_across_all_actor_seeds": True,
        "files": file_hashes,
    }
    atomic_json(root / "canonical_critic_manifest.json", manifest)
    context = CanonicalCriticContext(
        root=root,
        split={name: values.copy() for name, values in split.items()},
        observation_normalizer=observation_normalizer,
        advantage_arrays={
            name: np.asarray(values).copy()
            for name, values in critic_run.advantages.items()
        },
        critic_audit=dict(critic_run.audit),
        artifact_manifest=manifest,
        reused=False,
    )
    _validate_loaded_context(context=context, data=data)
    return context


def _finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def flatten_seed_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """Flatten one suite summary without conflating distinct failure classes."""

    probe = summary.get("gradient_probe", {})
    global_budget = summary.get("global_budget", {})
    positive = summary.get("positive_only_initialization", {})
    row: dict[str, Any] = {
        "seed": int(summary["seed"]),
        "suite_status": summary.get("suite_status"),
        "all_methods_completed": bool(summary.get("all_methods_completed")),
        "all_branch_initial_states_identical": bool(
            summary.get("all_branch_initial_states_identical")
        ),
        "prepared_checkpoint_reload_identity": bool(
            summary.get("prepared_checkpoint", {}).get("reload_identity")
        ),
        "full_parameter_gradient_far_near_ratio": _finite_or_none(
            probe.get("full_parameter_gradient_far_near_ratio")
        ),
        "standardized_distance_far_near_ratio": _finite_or_none(
            probe.get("standardized_distance_far_near_ratio")
        ),
        "corrected_q_xi_loglog_slope_vs_radius": _finite_or_none(
            probe.get("corrected_q_xi_loglog_slope_vs_radius")
        ),
        "analytic_autograd_relative_error_max": _finite_or_none(
            probe.get("analytic_autograd_relative_error_max")
        ),
        "natural_far_field_present": bool(probe.get("natural_far_field_present")),
        "initial_global_scale": _finite_or_none(global_budget.get("global_scale")),
        "positive_only_fixed_budget_completed": bool(
            positive.get("fixed_budget_completed")
        ),
        "positive_only_terminal_audit_complete": bool(
            positive.get("terminal_audit_complete")
        ),
        "positive_only_terminal_state": positive.get("state"),
        "positive_only_task_performance_status": positive.get(
            "task_performance_status"
        ),
        "positive_only_normalized_return": _finite_or_none(
            positive.get("final_normalized_return")
        ),
    }
    methods = summary.get("methods", {})
    failures = summary.get("branch_failures", {})
    for method in METHODS:
        audit = methods.get(method)
        row[f"{method}_branch_failed"] = method in failures
        if audit is None:
            row[f"{method}_fixed_budget_completed"] = False
            row[f"{method}_terminal_audit_complete"] = False
            row[f"{method}_terminal_state"] = None
            row[f"{method}_task_performance_status"] = "branch_failed"
            row[f"{method}_task_performance_collapse"] = None
            row[f"{method}_support_boundary_event"] = None
            row[f"{method}_nan_inf_event"] = None
            row[f"{method}_normalized_return"] = None
            continue
        row[f"{method}_fixed_budget_completed"] = bool(
            audit.get("fixed_budget_completed")
        )
        row[f"{method}_terminal_audit_complete"] = bool(
            audit.get("terminal_audit_complete")
        )
        row[f"{method}_terminal_state"] = audit.get("state")
        row[f"{method}_task_performance_status"] = audit.get(
            "task_performance_status"
        )
        row[f"{method}_task_performance_collapse"] = audit.get(
            "task_performance_collapse"
        )
        row[f"{method}_support_boundary_event"] = audit.get(
            "support_boundary_event"
        )
        row[f"{method}_nan_inf_event"] = audit.get("numerical_nonfinite")
        row[f"{method}_normalized_return"] = _finite_or_none(
            audit.get("final_normalized_return")
        )
    return row


def _stats(values: Sequence[float]) -> dict[str, float] | None:
    finite = np.asarray([value for value in values if math.isfinite(value)], dtype=np.float64)
    if len(finite) == 0:
        return None
    return {
        "count": int(len(finite)),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0,
        "median": float(np.median(finite)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
    }


def aggregate_seed_summaries(summaries: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate paired seed records while keeping all event classes separate."""

    if not summaries:
        return {
            "seeds_completed": 0,
            "seed_ids": [],
            "method_ranking_claim_allowed": False,
        }
    rows = [flatten_seed_summary(summary) for summary in summaries]
    aggregate: dict[str, Any] = {
        "seeds_completed": len(rows),
        "seed_ids": [int(row["seed"]) for row in rows],
        "method_ranking_claim_allowed": False,
        "scientific_review_required": True,
    }

    numeric_keys = (
        "full_parameter_gradient_far_near_ratio",
        "standardized_distance_far_near_ratio",
        "corrected_q_xi_loglog_slope_vs_radius",
        "analytic_autograd_relative_error_max",
        "initial_global_scale",
        "positive_only_normalized_return",
    )
    for key in numeric_keys:
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        statistics = _stats(values)
        if statistics is not None:
            aggregate[key] = statistics

    aggregate["terminal_state_counts"] = {
        method: {
            state: sum(
                row.get(f"{method}_terminal_state") == state for row in rows
            )
            for state in sorted(
                {
                    str(row[f"{method}_terminal_state"])
                    for row in rows
                    if row.get(f"{method}_terminal_state") is not None
                }
            )
        }
        for method in METHODS
    }
    aggregate["reporting_separation"] = {
        method: {
            "task_performance_available_count": sum(
                row.get(f"{method}_task_performance_status") == "available"
                for row in rows
            ),
            "task_performance_unavailable_count": sum(
                row.get(f"{method}_task_performance_status") != "available"
                for row in rows
            ),
            "task_performance_collapse_count": sum(
                row.get(f"{method}_task_performance_collapse") is True
                for row in rows
            ),
            "support_or_variance_boundary_count": sum(
                row.get(f"{method}_support_boundary_event") is True for row in rows
            ),
            "nan_inf_numerical_count": sum(
                row.get(f"{method}_nan_inf_event") is True for row in rows
            ),
            "branch_failure_count": sum(
                bool(row.get(f"{method}_branch_failed")) for row in rows
            ),
        }
        for method in METHODS
    }

    paired_returns: dict[str, list[dict[str, Any]]] = {}
    for method in METHODS:
        key = f"{method}_normalized_return"
        paired_returns[method] = [
            {"seed": int(row["seed"]), "normalized_return": row[key]}
            for row in rows
            if row.get(key) is not None
        ]
    aggregate["paired_normalized_returns"] = paired_returns

    signed_by_seed = {
        int(row["seed"]): float(row["signed_normalized_return"])
        for row in rows
        if row.get("signed_normalized_return") is not None
    }
    deltas: dict[str, Any] = {}
    for method in METHODS:
        values = [
            float(row[f"{method}_normalized_return"]) - signed_by_seed[int(row["seed"])]
            for row in rows
            if row.get(f"{method}_normalized_return") is not None
            and int(row["seed"]) in signed_by_seed
        ]
        statistics = _stats(values)
        if statistics is not None:
            deltas[method] = statistics
    aggregate["paired_normalized_return_delta_vs_signed"] = deltas
    return aggregate


def build_root_terminal_audit(
    *,
    summaries: Sequence[dict[str, Any]],
    plan: HopperExecutionPlan,
    canonical: CanonicalCriticContext,
    rollout_preflight: dict[str, Any],
    dataset_manifest: dict[str, Any],
    seed_failures: dict[int, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the root terminal audit without authorizing a scientific claim."""

    failures = seed_failures or {}
    aggregate = aggregate_seed_summaries(summaries)
    completed_seed_ids = tuple(int(summary["seed"]) for summary in summaries)
    seed_count_complete = completed_seed_ids == plan.seeds and not failures
    all_methods_completed = bool(
        summaries and all(summary.get("all_methods_completed") for summary in summaries)
    )
    all_branch_initial_states_identical = bool(
        summaries
        and all(
            summary.get("all_branch_initial_states_identical") for summary in summaries
        )
    )
    positive_fixed_budget = bool(
        summaries
        and all(
            summary.get("positive_only_initialization", {}).get(
                "fixed_budget_completed"
            )
            and summary.get("positive_only_initialization", {}).get(
                "terminal_audit_complete"
            )
            for summary in summaries
        )
    )
    method_terminal_audits_complete = bool(
        summaries
        and all(
            all(
                audit.get("terminal_audit_complete")
                for audit in summary.get("methods", {}).values()
            )
            and set(summary.get("methods", {})) == set(METHODS)
            for summary in summaries
        )
    )
    all_actor_budgets_completed = bool(
        summaries
        and all(
            all(
                audit.get("fixed_budget_completed")
                for audit in summary.get("methods", {}).values()
            )
            and set(summary.get("methods", {})) == set(METHODS)
            for summary in summaries
        )
    )
    rollout_available = bool(
        rollout_preflight.get("status") == "passed"
        and summaries
        and all(
            summary.get("positive_only_initialization", {}).get(
                "task_performance_status"
            )
            == "available"
            and all(
                audit.get("task_performance_status") == "available"
                for audit in summary.get("methods", {}).values()
            )
            and set(summary.get("methods", {})) == set(METHODS)
            for summary in summaries
        )
    )
    canonical_complete = bool(
        canonical.artifact_manifest.get("complete")
        and canonical.critic_audit.get("fixed_budget_completed")
        and canonical.critic_audit.get("critic_accepted_for_frozen_advantage")
    )
    mechanism_records_complete = bool(
        summaries
        and all(
            summary.get("matching")
            and summary.get("gradient_probe")
            and summary.get("global_budget")
            for summary in summaries
        )
    )
    engineering_complete = bool(
        dataset_manifest.get("identity_verified")
        and seed_count_complete
        and canonical_complete
        and rollout_preflight.get("status") == "passed"
        and positive_fixed_budget
        and all_methods_completed
        and all_actor_budgets_completed
        and method_terminal_audits_complete
        and all_branch_initial_states_identical
        and mechanism_records_complete
        and (rollout_available if plan.protocol.rollout_required else True)
    )
    paired_complete = bool(
        plan.formal_evidence_eligible and seed_count_complete and len(plan.seeds) > 1
    )
    formal_prerequisites = bool(
        plan.formal_evidence_eligible
        and engineering_complete
        and paired_complete
        and rollout_available
    )
    return {
        "experiment_id": EXPERIMENT_ID,
        "public_runner_version": PUBLIC_RUNNER_VERSION,
        "execution_kind": plan.execution_kind,
        "seeds_expected": list(plan.seeds),
        "seeds_completed": list(completed_seed_ids),
        "seed_failures": {str(seed): value for seed, value in failures.items()},
        "seed_count_complete": seed_count_complete,
        "dataset_identity_verified": bool(dataset_manifest.get("identity_verified")),
        "one_shared_canonical_critic": True,
        "canonical_critic_reused": canonical.reused,
        "canonical_critic_fixed_budget_completed": bool(
            canonical.critic_audit.get("fixed_budget_completed")
        ),
        "canonical_critic_accepted_for_frozen_advantage": bool(
            canonical.critic_audit.get("critic_accepted_for_frozen_advantage")
        ),
        "positive_only_fixed_budget_all_seeds": positive_fixed_budget,
        "all_methods_completed": all_methods_completed,
        "all_actor_fixed_budgets_completed": all_actor_budgets_completed,
        "terminal_audit_records_complete": method_terminal_audits_complete,
        "all_branch_initial_states_identical": all_branch_initial_states_identical,
        "mechanism_records_complete": mechanism_records_complete,
        "rollout_preflight_passed": rollout_preflight.get("status") == "passed",
        "rollout_available_all_required_checkpoints": rollout_available,
        "paired_seed_evidence_complete": paired_complete,
        "engineering_pipeline_complete": engineering_complete,
        "formal_evidence_prerequisites_complete": formal_prerequisites,
        "formal_scientific_gate_passed": False,
        "formal_scientific_gate_reason": "post_run_scientific_review_required",
        "method_ranking_claim_allowed": False,
        "root_completion_marker_allowed": engineering_complete,
        "reporting_separation": aggregate.get("reporting_separation", {}),
        "scientific_status": (
            "raw_complete_pending_review"
            if plan.formal_evidence_eligible and engineering_complete
            else plan.execution_kind
        ),
    }


def _runtime_dataset_summary(data: OfflineData) -> dict[str, Any]:
    return {
        "transitions_loaded": data.size,
        "episodes_loaded": int(len(np.unique(data.episode_ids))),
        "observation_dim": int(data.observations.shape[1]),
        "action_dim": int(data.actions.shape[1]),
        "reward_mean": float(np.mean(data.rewards)),
        "reward_std": float(np.std(data.rewards)),
        "terminal_fraction": float(np.mean(data.terminals)),
        "timeout_fraction": float(np.mean(data.timeouts)),
    }


def _make_rollout_evaluator(
    *,
    seed: int,
    seed_dir: Path,
    protocol: HopperProtocol,
    observation_normalizer: Normalizer,
    device: torch.device,
    rollout_runner: RolloutRunner,
) -> Callable[[Any, int, str], dict[str, Any]]:
    def evaluate(policy: Any, step: int, stage: str) -> dict[str, Any]:
        is_preparation = stage == "positive_only_initialization"
        terminal_step = protocol.positive_steps if is_preparation else protocol.branch_steps
        episodes = (
            protocol.final_rollout_episodes
            if step >= terminal_step
            else protocol.rollout_episodes
        )
        diagnostics = seed_dir / "rollouts" / stage / f"step_{step:08d}.json"
        return rollout_runner(
            policy=policy,
            obs_norm=observation_normalizer,
            backend=protocol.rollout_backend,
            dataset_id=protocol.rollout_dataset_id,
            env_id=protocol.env_id,
            episodes=episodes,
            seed=seed * 100_000 + step,
            device=device,
            normalized_score_percent=protocol.normalized_score_percent,
            reference_min_score=protocol.normalized_score_reference_min,
            reference_max_score=protocol.normalized_score_reference_max,
            required=protocol.rollout_required,
            diagnostics_path=diagnostics,
        )

    return evaluate


def run_hopper(
    *,
    dataset_path: Path,
    output_root: Path,
    seeds: Sequence[int] | None = None,
    smoke: bool = False,
    device: str | torch.device = "auto",
    critic_artifact: Path | None = None,
    suite_runner: SuiteRunner = run_hopper_six_branch_suite,
    preflight_runner: PreflightRunner = preflight_from_protocol,
    rollout_runner: RolloutRunner = evaluate_d4rl_rollouts,
) -> dict[str, Any]:
    """Run the public Hopper pipeline; no method ranking is authorized here."""

    plan = resolve_hopper_execution(seeds=seeds, smoke=smoke)
    protocol = plan.protocol
    output = Path(output_root).expanduser().resolve()
    _require_new_or_empty(output)
    selected_device = _resolve_device(device)
    run_manifest: dict[str, Any] = {
        "experiment_id": EXPERIMENT_ID,
        "public_runner_version": PUBLIC_RUNNER_VERSION,
        "started_utc": utc_now(),
        "state": "running",
        "execution_kind": plan.execution_kind,
        "formal_evidence_eligible": plan.formal_evidence_eligible,
        "method_ranking_claim_allowed": False,
        "seeds": list(plan.seeds),
        "device": str(selected_device),
        "protocol": asdict(protocol),
        "critic_artifact_request": (
            str(Path(critic_artifact).expanduser().resolve())
            if critic_artifact is not None
            else None
        ),
    }
    atomic_json(output / "scientific_run_manifest.json", run_manifest)
    summaries: list[dict[str, Any]] = []
    seed_failures: dict[int, dict[str, Any]] = {}
    canonical: CanonicalCriticContext | None = None
    preflight: dict[str, Any] = {"status": "not_started"}
    dataset_manifest: dict[str, Any] = {"identity_verified": False}

    try:
        dataset_manifest = validate_dataset_identity(dataset_path, protocol)
        atomic_json(output / "DATASET_MANIFEST.json", dataset_manifest)
        data = load_hopper_hdf5(dataset_path)
        atomic_json(output / "DATASET_RUNTIME_SUMMARY.json", _runtime_dataset_summary(data))
        preflight = preflight_runner(
            protocol=protocol,
            expected_observation_dim=int(data.observations.shape[1]),
            expected_action_dim=int(data.actions.shape[1]),
            seed=protocol.canonical_critic_seed,
            output_dir=output / "rollout_preflight",
            required=protocol.rollout_required,
        )
        atomic_json(output / "ROLLOUT_PREFLIGHT.json", preflight)
        canonical = prepare_canonical_critic_context(
            data=data,
            protocol=protocol,
            dataset_manifest=dataset_manifest,
            device=selected_device,
            artifact_root=output / "canonical_critic",
            reuse_root=critic_artifact,
        )
        atomic_json(
            output / "CANONICAL_CRITIC_REFERENCE.json",
            {
                "root": str(canonical.root),
                "reused": canonical.reused,
                "identity": canonical.artifact_manifest["identity"],
                "critic_training_count": canonical.artifact_manifest.get(
                    "critic_training_count"
                ),
                "shared_across_all_actor_seeds": True,
                "files": canonical.artifact_manifest["files"],
            },
        )
        normalized_observations = canonical.observation_normalizer.transform(
            data.observations
        )
        advantages = np.asarray(
            canonical.advantage_arrays["advantage"], dtype=np.float32
        )

        for index, seed in enumerate(plan.seeds, start=1):
            seed_dir = output / "seeds" / f"seed_{seed}"
            atomic_json(
                output / "scientific_heartbeat.json",
                {
                    "utc": utc_now(),
                    "state": "running",
                    "seed": seed,
                    "seed_index": index,
                    "seed_total": len(plan.seeds),
                    "seeds_completed": len(summaries),
                },
            )

            def heartbeat(stage: str, step: int, *, current_seed: int = seed) -> None:
                atomic_json(
                    output / "scientific_heartbeat.json",
                    {
                        "utc": utc_now(),
                        "state": "running",
                        "seed": current_seed,
                        "seed_index": index,
                        "seed_total": len(plan.seeds),
                        "stage": stage,
                        "step": step,
                        "seeds_completed": len(summaries),
                    },
                )

            evaluator = _make_rollout_evaluator(
                seed=seed,
                seed_dir=seed_dir,
                protocol=protocol,
                observation_normalizer=canonical.observation_normalizer,
                device=selected_device,
                rollout_runner=rollout_runner,
            )

            def stage_runner_with_rollout(**kwargs: Any) -> tuple[Any, dict[str, Any]]:
                method = str(kwargs["method"])
                stage_output = Path(kwargs["output_dir"])
                stage_label = (
                    "positive_only_initialization"
                    if stage_output.name == "positive_only_initialization"
                    else f"methods/{method}"
                )

                def labelled_rollout(
                    policy: Any,
                    step: int,
                    _method: str,
                ) -> dict[str, Any]:
                    return evaluator(policy, step, stage_label)

                return train_actor_stage(
                    **kwargs,
                    rollout_evaluator=labelled_rollout,
                    rollout_eval_interval=protocol.rollout_eval_interval,
                )

            try:
                summary = suite_runner(
                    observations=normalized_observations,
                    actions=data.actions,
                    advantages=advantages,
                    train_indices=canonical.split["train"],
                    validation_indices=canonical.split["validation"],
                    protocol=protocol,
                    seed=seed,
                    device=selected_device,
                    output_dir=seed_dir,
                    heartbeat=heartbeat,
                    stage_runner=stage_runner_with_rollout,
                    continue_on_branch_failure=True,
                )
                summary["rollout_included"] = True
                summary["public_runner_included"] = True
                atomic_json(seed_dir / "suite_summary.json", summary)
                suite_completion_path = seed_dir / "SUITE_COMPLETE.json"
                if suite_completion_path.is_file():
                    suite_completion = json.loads(
                        suite_completion_path.read_text(encoding="utf-8")
                    )
                else:
                    suite_completion = {
                        "seed": seed,
                        "suite_status": summary.get("suite_status"),
                    }
                suite_completion.update(
                    {
                        "rollout_included": True,
                        "public_runner_included": True,
                        "formal_evidence_allowed": False,
                    }
                )
                atomic_json(suite_completion_path, suite_completion)
                summaries.append(summary)
            except Exception as exc:
                failure = {
                    "seed": seed,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "failure_isolated": True,
                }
                seed_failures[seed] = failure
                atomic_json(seed_dir / "seed_failure.json", failure)

            write_csv(
                output / "per_seed_summary.csv",
                [flatten_seed_summary(summary) for summary in summaries],
            )
            partial = aggregate_seed_summaries(summaries)
            partial["execution_kind"] = plan.execution_kind
            partial["seed_failures"] = {
                str(seed_id): value for seed_id, value in seed_failures.items()
            }
            atomic_json(output / "aggregate_summary.json", partial)

        audit = build_root_terminal_audit(
            summaries=summaries,
            plan=plan,
            canonical=canonical,
            rollout_preflight=preflight,
            dataset_manifest=dataset_manifest,
            seed_failures=seed_failures,
        )
        atomic_json(output / "terminal_audit.json", audit)
        if not audit["root_completion_marker_allowed"]:
            incomplete = {
                "experiment_id": EXPERIMENT_ID,
                "completed_utc": utc_now(),
                "result_status": "incomplete_engineering_run",
                "method_ranking_claim_allowed": False,
                "terminal_audit": str(output / "terminal_audit.json"),
            }
            atomic_json(output / "RUN_INCOMPLETE.json", incomplete)
            raise RuntimeError(
                "Hopper public run is incomplete; see terminal_audit.json and "
                "RUN_INCOMPLETE.json"
            )

        completion = {
            "experiment_id": EXPERIMENT_ID,
            "public_runner_version": PUBLIC_RUNNER_VERSION,
            "completed_utc": utc_now(),
            "execution_kind": plan.execution_kind,
            "seeds_completed": list(plan.seeds),
            "result_status": audit["scientific_status"],
            "formal_result_claim": False,
            "method_ranking_claim_allowed": False,
            "formal_evidence_prerequisites_complete": audit[
                "formal_evidence_prerequisites_complete"
            ],
        }
        atomic_json(output / "RUN_COMPLETE.json", completion)
        atomic_json(
            output / "scientific_heartbeat.json",
            {
                "utc": utc_now(),
                "state": "completed",
                "seeds_completed": len(summaries),
            },
        )
        run_manifest.update(
            {
                "state": "raw_complete",
                "completed_utc": utc_now(),
                "seeds_completed": len(summaries),
            }
        )
        atomic_json(output / "scientific_run_manifest.json", run_manifest)
        return {
            "completion": completion,
            "terminal_audit": audit,
            "aggregate": aggregate_seed_summaries(summaries),
        }
    except Exception as exc:
        failure = {
            "experiment_id": EXPERIMENT_ID,
            "failed_utc": utc_now(),
            "execution_kind": plan.execution_kind,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "seeds_completed": [int(summary["seed"]) for summary in summaries],
            "rollout_preflight_status": preflight.get("status"),
            "canonical_critic_available": canonical is not None,
            "dataset_identity_verified": bool(dataset_manifest.get("identity_verified")),
            "method_ranking_claim_allowed": False,
        }
        atomic_json(output / "SCIENTIFIC_RUN_FAILED.json", failure)
        run_manifest.update({"state": "failed", "failed_utc": utc_now()})
        atomic_json(output / "scientific_run_manifest.json", run_manifest)
        raise
