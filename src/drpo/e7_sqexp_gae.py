"""Minimal canonical wrapper for the 192-branch E7 TD/GAE development pilot."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from drpo import e7_canonical_sweep as base
from drpo import e7_squared_exp_night as night
from drpo import e7_squared_exp_night_bootstrap as canonical
from drpo.e7_canonical_injection import _agent_device, _as_tensor, sha256_file
from drpo.e7_squared_exp_kernel import FORMULA

EXPERIMENT_ID = "EXT-H-E7-SQEXP-GAE-01"
SCIENTIFIC_STATUS = "frozen_critic_trajectory_gae_development_pilot_only"
RUNNER_VERSION = "3.0.0-single-owner-canonical-wrapper"
EXPECTED_DATASETS = night.EXPECTED_DATASETS
EXPECTED_SEEDS = (200, 201, 202, 203)
HELD_OUT_SEEDS = night.HELD_OUT_SEEDS
ACTOR_MODES = ("a2c", "ppo_clip_k4")
ESTIMATORS = ("td", "gae")
COEFFICIENTS = (64.0, 128.0, 256.0)
EXPECTED_BRANCHES = 192
EXPECTED_STEPS = 1_000_000
PREPARED_ROOT_ENV = "E7_SQEXP_GAE_PREPARED_ROOT"
_ORIGINAL_WRITE_PLAN = base.write_plan


def compute_gae_from_td(
    td: np.ndarray,
    terminals: np.ndarray,
    timeouts: np.ndarray,
    *,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> np.ndarray:
    """Accumulate TD residuals without crossing terminal, timeout, or dataset tail."""
    residual = np.asarray(td, dtype=np.float64).reshape(-1)
    terminal = np.asarray(terminals, dtype=np.bool_).reshape(-1)
    timeout = np.asarray(timeouts, dtype=np.bool_).reshape(-1)
    if not residual.size or not (residual.shape == terminal.shape == timeout.shape):
        raise ValueError("TD, terminal, and timeout vectors must be non-empty and aligned")
    if bool((terminal & timeout).any()):
        raise ValueError("terminal and timeout flags overlap")
    if (
        not np.isfinite(residual).all()
        or not 0.0 <= gamma <= 1.0
        or not 0.0 <= gae_lambda <= 1.0
    ):
        raise ValueError("invalid GAE input")
    result = np.empty_like(residual)
    running = 0.0
    continuation = ~(terminal | timeout)
    for index in range(residual.size - 1, -1, -1):
        running = residual[index] + gamma * gae_lambda * continuation[index] * running
        result[index] = running
    return result.astype(np.float32)


def _state_digest(state: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _load_grid(path: str | Path) -> tuple[dict[str, Any], str]:
    source = Path(path)
    raw = json.loads(source.read_text())
    required = {
        "experiment_id": EXPERIMENT_ID,
        "run_kind": "pilot",
        "status": "not_run",
        "scientific_status": SCIENTIFIC_STATUS,
        "predecessor_experiment_id": night.EXPERIMENT_ID,
        "datasets": list(EXPECTED_DATASETS),
        "development_seeds": list(EXPECTED_SEEDS),
        "held_out_seeds": list(HELD_OUT_SEEDS),
        "steps": EXPECTED_STEPS,
        "actor_update_modes": list(ACTOR_MODES),
        "advantage_modes": ["one_step_td", "gae_lambda_0p95"],
        "expected_total_branches": EXPECTED_BRANCHES,
        "screening_only": True,
        "formal_evidence_allowed": False,
    }
    changed = [key for key, value in required.items() if raw.get(key) != value]
    weight = raw.get("weight_control", {})
    if (
        changed
        or weight.get("formula") != FORMULA
        or tuple(float(value) for value in weight.get("exp_coefficients", ()))
        != COEFFICIENTS
        or float(raw.get("trajectory_advantage", {}).get("gae_lambda", -1.0)) != 0.95
        or raw.get("shared_frozen_critic", {}).get("updated_during_actor_training")
        is not False
    ):
        raise ValueError(f"frozen GAE grid changed: {changed or ['nested_contract']}")
    return raw, sha256_file(source)


def _load_run_spec(path: str | Path) -> tuple[dict[str, Any], str]:
    run_spec, digest = night.load_run_spec(path)
    run_spec["seeds"] = list(EXPECTED_SEEDS)
    return run_spec, digest


def _label(value: float) -> str:
    return f"{value:.8g}".replace("-", "m").replace(".", "p")


def _build_branches(
    contract: base.CanonicalContract,
    run_spec: Mapping[str, Any],
    grid: Mapping[str, Any],
) -> list[base.Branch]:
    del grid
    if not math.isclose(contract.expected_canonical_alpha, 0.11, abs_tol=1e-12):
        raise ValueError("canonical source alpha changed")
    datasets = [base.DatasetSpec.from_mapping(item) for item in run_spec["datasets"]]
    if tuple(item.id for item in datasets) != EXPECTED_DATASETS:
        raise ValueError("GAE dataset order changed")
    controls = [("positive_only", 0.0, 0.0)] + [
        ("squared_exponential", 1.0, coefficient) for coefficient in COEFFICIENTS
    ]
    branches: list[base.Branch] = []
    for estimator in ESTIMATORS:
        for actor_mode in ACTOR_MODES:
            for method, weight_at_zero, coefficient in controls:
                control = (
                    "positive_only"
                    if method == "positive_only"
                    else f"sqexp_c{_label(coefficient)}"
                )
                for dataset in datasets:
                    for seed in EXPECTED_SEEDS:
                        branches.append(
                            base.Branch(
                                branch_id=(
                                    f"{dataset.id}__seed{seed}__{estimator}__{control}__"
                                    f"{actor_mode}__steps1m"
                                ),
                                branch_kind="injected",
                                dataset=dataset,
                                seed=seed,
                                template_values={
                                    "steps": str(EXPECTED_STEPS),
                                    "actor_update_mode": actor_mode,
                                    "advantage_estimator": estimator,
                                    "weight_method": method,
                                    "weight_at_zero": f"{weight_at_zero:.17g}",
                                    "exp_coefficient": f"{coefficient:.17g}",
                                    "reference_distance": f"{night.REFERENCE_DISTANCE:.17g}",
                                    "diagnostics_interval": "1000",
                                    "sampled_values_per_update": "16",
                                },
                                negative_control=None,
                            )
                        )
    ids = [branch.branch_id for branch in branches]
    if len(ids) != EXPECTED_BRANCHES or len(ids) != len(set(ids)):
        raise RuntimeError(f"expected {EXPECTED_BRANCHES} unique branches")
    return branches


def _prepared_root(work_dir: Path) -> Path:
    configured = os.environ.get(PREPARED_ROOT_ENV)
    return Path(configured).expanduser().resolve() if configured else work_dir / "prepared"


def _manifest_for(branch: base.Branch, work_dir: Path) -> Path:
    return (
        _prepared_root(work_dir)
        / branch.dataset.id
        / f"seed{branch.seed}"
        / "ADVANTAGE_MANIFEST.json"
    )


def _branch_command(
    *,
    contract_path: Path,
    contract: base.CanonicalContract,
    branch: base.Branch,
    branch_dir: Path,
    trainer_argv_template: Sequence[str],
) -> tuple[list[str], dict[str, Any]]:
    manifest = _manifest_for(branch, branch_dir.parent.parent)
    if not manifest.is_file():
        raise FileNotFoundError(f"missing preserved shared-critic artifact: {manifest}")
    values = branch.template_values
    context = {
        "canonical_root": str(contract.source_root),
        "dataset_id": branch.dataset.id,
        "dataset_path": str(Path(branch.dataset.path).expanduser().resolve()),
        "dataset_sha256": branch.dataset.sha256,
        "seed": branch.seed,
        "output_dir": str(branch_dir / "trainer_output"),
        "branch_id": branch.branch_id,
        "variant": "iqlv_exp_rank",
        **values,
    }
    trainer_args = [
        base._format_value(str(item), context)  # noqa: SLF001
        for item in trainer_argv_template
    ]
    branch_config = {
        "experiment_id": EXPERIMENT_ID,
        "branch_id": branch.branch_id,
        "branch_kind": branch.branch_kind,
        "canonical_root": context["canonical_root"],
        "dataset_id": branch.dataset.id,
        "dataset_path": context["dataset_path"],
        "dataset_sha256": branch.dataset.sha256,
        "seed": branch.seed,
        "template_values": values,
        "advantage_manifest": str(manifest),
        "weight_control": {
            "method": values["weight_method"],
            "weight_at_zero": float(values["weight_at_zero"]),
            "exp_coefficient": float(values["exp_coefficient"]),
            "reference_distance": night.REFERENCE_DISTANCE,
            "formula": FORMULA,
        },
    }
    config_path = branch_dir / "branch_config.json"
    base.atomic_write_json(config_path, branch_config)
    command = [
        sys.executable,
        "-m",
        "drpo.e7_sqexp_gae",
        "bootstrap",
        "--contract",
        str(contract_path),
        "--branch-config",
        str(config_path),
        "--branch-manifest",
        str(branch_dir / "branch_manifest.json"),
        "--",
        *trainer_args,
    ]
    return command, branch_config


def _write_plan(**kwargs: Any) -> dict[str, Any]:
    work_dir = Path(kwargs["work_dir"]).expanduser().resolve()
    missing = [
        str(_manifest_for(branch, work_dir))
        for branch in kwargs["branches"]
        if not _manifest_for(branch, work_dir).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "missing preserved prepared artifacts: "
            + ", ".join(sorted(set(missing)))
        )
    return _ORIGINAL_WRITE_PLAN(**kwargs)


def _load_prepared(
    branch: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any], str, dict[str, Any]]:
    manifest_path = Path(branch["advantage_manifest"]).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text())
    estimator = str(branch["template_values"]["advantage_estimator"])
    if estimator not in ESTIMATORS or manifest.get("status") != "complete":
        raise RuntimeError("unsupported estimator or incomplete prepared manifest")
    expected = branch["dataset_id"], int(branch["seed"]), branch["dataset_sha256"]
    observed = (
        manifest.get("dataset_id"),
        int(manifest.get("seed", -1)),
        manifest.get("dataset_sha256"),
    )
    if observed != expected:
        raise RuntimeError(f"prepared identity mismatch: {observed} != {expected}")
    arrays = Path(manifest["advantages"]["path"]).expanduser().resolve()
    critic = Path(manifest["critic"]["path"]).expanduser().resolve()
    if sha256_file(arrays) != manifest["advantages"]["sha256"]:
        raise RuntimeError("prepared advantage hash mismatch")
    if sha256_file(critic) != manifest["critic"]["sha256"]:
        raise RuntimeError("prepared critic hash mismatch")
    with np.load(arrays, allow_pickle=False) as payload:
        stored_td = payload["td"].astype(np.float32, copy=True)
        stored_gae = payload["gae"].astype(np.float32, copy=True)
        values = payload["values"].astype(np.float32, copy=True)
        next_values = payload["next_values"].astype(np.float32, copy=True)
    dataset = Path(branch["dataset_path"]).expanduser().resolve()
    if sha256_file(dataset) != branch["dataset_sha256"]:
        raise RuntimeError("ordered dataset hash mismatch")
    canonical_root = str(Path(branch["canonical_root"]).expanduser().resolve())
    inserted = canonical_root not in sys.path
    if inserted:
        sys.path.insert(0, canonical_root)
    try:
        from d4rl_common.train_loop import load_hdf5

        ordered = load_hdf5(dataset, dataset_name=str(branch["dataset_id"]))
        rewards = ordered["rews"]
        terminals = ordered["terms"]
        timeouts = ordered["touts"]
    finally:
        if inserted:
            sys.path.remove(canonical_root)
    gamma = float(manifest.get("gamma", 0.99))
    td64 = rewards.astype(np.float64) + gamma * (~terminals) * next_values - values
    td = td64.astype(np.float32)
    if not np.array_equal(td, stored_td):
        raise RuntimeError("current one-step TD disagrees with prepared artifact")
    gae = compute_gae_from_td(
        td64,
        terminals,
        timeouts,
        gamma=gamma,
        gae_lambda=float(manifest.get("gae_lambda", 0.95)),
    )
    if not np.allclose(gae, stored_gae, atol=1e-6, rtol=1e-6):
        raise RuntimeError("current GAE implementation disagrees with prepared artifact")
    advantage = td if estimator == "td" else gae
    if advantage.ndim != 1 or not np.isfinite(advantage).all():
        raise RuntimeError("prepared advantage must be one finite vector")
    try:
        checkpoint = torch.load(critic, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(critic, map_location="cpu")
    return advantage, checkpoint["state_dict"], estimator, {
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "gae_recomputed_from_td_and_boundaries": True,
        "gae_matches_prepared_artifact": True,
    }


def _prepared_agent(parent: type, state_dict: dict[str, Any], instances: list[Any]) -> type:
    class PreparedAgent(parent):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.critic.load_state_dict(state_dict, strict=True)
            self._drpo_critic_initial_sha256 = _state_digest(self.critic.state_dict())
            self.c_opt.step = lambda *args, **kwargs: None
            instances.append(self)

        def update(self, s: Any, a: Any, r: Any, ns: Any, d: Any, ep_ret: Any = None) -> Any:
            del r
            device = _agent_device(self)
            states = _as_tensor(s, device=device)
            next_states = _as_tensor(ns, device=device)
            dones = _as_tensor(d, device=device, dtype=torch.bool).reshape(-1)
            advantage = _as_tensor(ep_ret, device=device).reshape(-1).detach()
            if advantage.shape[0] != states.shape[0] or not bool(torch.isfinite(advantage).all()):
                raise RuntimeError("prepared advantage batch is invalid")
            with torch.no_grad():
                value = self.critic(states).squeeze(-1)
                next_value = self.critic(next_states).squeeze(-1)
                reward = advantage + value - float(self.gamma) * next_value * (~dones)
                recovered = reward + float(self.gamma) * next_value * (~dones) - value
                if not torch.allclose(recovered, advantage, atol=1e-6, rtol=1e-6):
                    raise RuntimeError("adapter changed the prepared advantage")
            return super().update(s, a, reward, ns, d, ep_ret)

    return PreparedAgent


def _bootstrap(argv: list[str]) -> int:
    args = canonical.build_parser().parse_args(argv)
    branch = json.loads(Path(args.branch_config).expanduser().read_text())
    if branch.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("GAE branch experiment_id mismatch")
    if branch["template_values"]["actor_update_mode"] not in ACTOR_MODES:
        raise ValueError("GAE pilot supports only canonical A2C and PPO-K4")
    advantage, state_dict, estimator, provenance = _load_prepared(branch)
    instances: list[Any] = []
    old_id = canonical.EXPERIMENT_ID
    old_a2c = canonical.patch_canonical_module
    old_ppo = canonical.patch_canonical_module_ppo
    old_returns: Any = None

    def install(module: Any, target: str) -> None:
        nonlocal old_returns
        setattr(module, target, _prepared_agent(getattr(module, target), state_dict, instances))
        import d4rl_common.train_loop as loop

        old_returns = loop.compute_mc_returns

        def prepared_returns(rewards: np.ndarray, *_: Any) -> np.ndarray:
            if len(rewards) != len(advantage):
                raise RuntimeError("prepared advantage length mismatch")
            return advantage.copy()

        loop.compute_mc_returns = prepared_returns

    def patch_a2c(module: Any, contract: Any, control: Any) -> type:
        result = old_a2c(module, contract, control)
        install(module, contract.target_class)
        return result

    def patch_ppo(module: Any, target: str, **kwargs: Any) -> type:
        result = old_ppo(module, target, **kwargs)
        install(module, target)
        return result

    canonical.EXPERIMENT_ID = EXPERIMENT_ID
    canonical.patch_canonical_module = patch_a2c
    canonical.patch_canonical_module_ppo = patch_ppo
    manifest_path = Path(args.branch_manifest).expanduser().resolve()
    try:
        result = canonical.main(argv)
        if len(instances) != 1:
            raise RuntimeError(f"expected one canonical agent, found {len(instances)}")
        initial_sha = instances[0]._drpo_critic_initial_sha256
        final_sha = _state_digest(instances[0].critic.state_dict())
        payload = json.loads(manifest_path.read_text())
        payload.update(
            {
                "advantage_estimator": estimator,
                "gae_used": estimator == "gae",
                "advantage_provenance": provenance,
                "critic_initial_state_sha256": initial_sha,
                "critic_final_state_sha256": final_sha,
                "critic_immutability_verified": initial_sha == final_sha,
            }
        )
        if initial_sha != final_sha:
            raise RuntimeError("frozen critic changed during actor training")
        canonical._atomic_json(manifest_path, payload)  # noqa: SLF001
        return result
    except BaseException as exc:
        if manifest_path.is_file():
            payload = json.loads(manifest_path.read_text())
            payload.update(
                {
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            canonical._atomic_json(manifest_path, payload)  # noqa: SLF001
        raise
    finally:
        canonical.EXPERIMENT_ID = old_id
        canonical.patch_canonical_module = old_a2c
        canonical.patch_canonical_module_ppo = old_ppo
        if old_returns is not None:
            import d4rl_common.train_loop as loop

            loop.compute_mc_returns = old_returns


def _read_branch(branch_dir: Path) -> dict[str, Any]:
    summaries = list((branch_dir / "trainer_output").glob("*_summary.json"))
    if len(summaries) != 1 or not (branch_dir / "COMPLETED.json").is_file():
        raise RuntimeError(f"branch output is incomplete: {branch_dir.name}")
    branch = json.loads((branch_dir / "branch_config.json").read_text())
    manifest = json.loads((branch_dir / "branch_manifest.json").read_text())
    history = json.loads(summaries[0].read_text())["history"]
    score_key = next(key for key in history if key != "steps")
    steps = [int(value) for value in history["steps"]]
    scores = [float(value) for value in history[score_key]]
    if steps[-1] != EXPECTED_STEPS or not all(math.isfinite(value) for value in scores):
        raise RuntimeError(f"non-finite or incomplete branch: {branch_dir.name}")
    values = branch["template_values"]
    return {
        "branch_id": branch["branch_id"],
        "dataset": branch["dataset_id"],
        "seed": int(branch["seed"]),
        "advantage_estimator": values["advantage_estimator"],
        "actor_update_mode": values["actor_update_mode"],
        "exp_coefficient": (
            None
            if values["weight_method"] == "positive_only"
            else float(values["exp_coefficient"])
        ),
        "best_score": max(scores),
        "final_score": scores[-1],
        "critic_immutability_verified": bool(
            manifest.get("critic_immutability_verified")
        ),
        "task_performance_collapse_event": "not_adjudicated_no_registered_threshold",
        "support_or_variance_boundary_event": "not_instrumented_in_this_pilot",
        "nan_inf_numerical_failure": False,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _aggregate(work_dir: Path) -> dict[str, Any]:
    branch_root = work_dir / "branches"
    branch_dirs = sorted(path for path in branch_root.iterdir() if path.is_dir())
    if len(branch_dirs) != EXPECTED_BRANCHES:
        raise RuntimeError(f"expected {EXPECTED_BRANCHES} branches, found {len(branch_dirs)}")
    rows = [_read_branch(path) for path in branch_dirs]
    if not all(row["critic_immutability_verified"] for row in rows):
        raise RuntimeError("one or more branches changed the frozen critic")

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                row["dataset"],
                row["advantage_estimator"],
                row["actor_update_mode"],
                row["exp_coefficient"],
            )
        ].append(row)
    groups: list[dict[str, Any]] = []
    for key, values in sorted(grouped.items(), key=lambda item: repr(item[0])):
        seeds = tuple(sorted(int(row["seed"]) for row in values))
        if seeds != EXPECTED_SEEDS:
            raise RuntimeError(f"paired seed set changed for {key}: {seeds}")
        finals = [float(row["final_score"]) for row in values]
        groups.append(
            {
                "dataset": key[0],
                "advantage_estimator": key[1],
                "actor_update_mode": key[2],
                "exp_coefficient": key[3],
                "seeds": list(seeds),
                "final_mean": statistics.fmean(finals),
                "final_seed_std": statistics.stdev(finals),
                "best_mean": statistics.fmean(
                    float(row["best_score"]) for row in values
                ),
            }
        )
    index = {
        (
            row["dataset"],
            row["advantage_estimator"],
            row["actor_update_mode"],
            row["exp_coefficient"],
        ): float(row["final_mean"])
        for row in groups
    }
    comparisons: list[dict[str, Any]] = []
    for dataset in EXPECTED_DATASETS:
        for coefficient in (None, *COEFFICIENTS):
            for estimator in ESTIMATORS:
                comparisons.append(
                    {
                        "dataset": dataset,
                        "advantage_estimator": estimator,
                        "exp_coefficient": coefficient,
                        "ppo_k4_minus_a2c_final": (
                            index[(dataset, estimator, "ppo_clip_k4", coefficient)]
                            - index[(dataset, estimator, "a2c", coefficient)]
                        ),
                    }
                )
            for actor_mode in ACTOR_MODES:
                comparisons.append(
                    {
                        "dataset": dataset,
                        "actor_update_mode": actor_mode,
                        "exp_coefficient": coefficient,
                        "gae_minus_td_final": (
                            index[(dataset, "gae", actor_mode, coefficient)]
                            - index[(dataset, "td", actor_mode, coefficient)]
                        ),
                    }
                )

    aggregate_dir = work_dir / "aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(aggregate_dir / "branch_results.csv", rows)
    _write_csv(aggregate_dir / "group_summary.csv", groups)
    _write_csv(aggregate_dir / "comparisons.csv", comparisons)
    audit = {
        "status": "PASS",
        "experiment_id": EXPERIMENT_ID,
        "raw_complete": True,
        "branch_count_observed": len(rows),
        "expected_branch_count": EXPECTED_BRANCHES,
        "critic_immutability_verified": True,
        "held_out_seeds_touched": False,
        "task_performance_collapse_separate": True,
        "support_or_variance_boundary_separate": True,
        "nan_inf_separate": True,
        "convergence_claim_allowed": False,
        "steady_state_ranking_allowed": False,
        "universal_gae_superiority_claim_allowed": False,
        "formal_evidence_allowed": False,
    }
    base.atomic_write_json(aggregate_dir / "terminal_audit.json", audit)
    summary = {
        "experiment_id": EXPERIMENT_ID,
        "status": "PASS",
        "branch_count": len(rows),
        "group_count": len(groups),
        "comparison_count": len(comparisons),
        "terminal_audit": str(aggregate_dir / "terminal_audit.json"),
    }
    base.atomic_write_json(aggregate_dir / "aggregate_summary.json", summary)
    return summary


def _runner(argv: list[str] | None = None) -> int:
    previous = (
        base.EXPERIMENT_ID,
        base.SCIENTIFIC_STATUS,
        base.RUNNER_VERSION,
        base.load_grid,
        base.load_run_spec,
        base.build_branches,
        base.branch_command,
        base.write_plan,
    )
    base.EXPERIMENT_ID = EXPERIMENT_ID
    base.SCIENTIFIC_STATUS = SCIENTIFIC_STATUS
    base.RUNNER_VERSION = RUNNER_VERSION
    base.load_grid = _load_grid
    base.load_run_spec = _load_run_spec
    base.build_branches = _build_branches
    base.branch_command = _branch_command
    base.write_plan = _write_plan
    delegated = list(sys.argv[1:] if argv is None else argv)
    try:
        result = int(base.main(delegated))
        if delegated and delegated[0] == "run":
            index = delegated.index("--work-dir")
            _aggregate(Path(delegated[index + 1]).expanduser().resolve())
        return result
    finally:
        (
            base.EXPERIMENT_ID,
            base.SCIENTIFIC_STATUS,
            base.RUNNER_VERSION,
            base.load_grid,
            base.load_run_spec,
            base.build_branches,
            base.branch_command,
            base.write_plan,
        ) = previous


def main(argv: list[str] | None = None) -> int:
    delegated = list(sys.argv[1:] if argv is None else argv)
    if delegated and delegated[0] == "bootstrap":
        return _bootstrap(delegated[1:])
    return _runner(delegated)


if __name__ == "__main__":
    raise SystemExit(main())
