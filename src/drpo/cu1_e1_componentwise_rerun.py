#!/usr/bin/env python3
"""Rerun frozen C-U1 E1/E2 and decompose Gaussian output-score growth.

Experiment ID: C-U1-E1-COMP-01.

This diagnostic deliberately stays in Gaussian output space. It does not study
neural-network pullback. It reuses the frozen C-U1 E1/E2 trainer and seeds from
``drpo_cu1_e1_e4_oneclick.py`` and adds component-wise measurements:

* mean score: ||∂ log π / ∂μ|| = d / σ²;
* log-scale score: ∂ log π / ∂logσ = d² / σ² - D;
* corrected log-scale term: (∂ log π / ∂logσ + D) σ² = d².

The exact distance-law checks normalize out learned σ before estimating slopes.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch

import drpo_cu1_e1_e4_oneclick as core

try:
    from . import cu1_core as shared_core
except ImportError:  # direct script execution
    import cu1_core as shared_core

EXPERIMENT_ID = "C-U1-E1-COMP-01"
SCRIPT_VERSION = "2026.06.25-componentwise-v3-shared-core"
BASE_COMMIT = "a9e0d860a6f03d1be12280885002c24ba2f1b66a"
EPS = 1e-12


def atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mean_ci(values: Sequence[float], seed: int = 20260625, n_boot: int = 4000) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=float)
    if len(array) == 0:
        return float("nan"), float("nan"), float("nan")
    if len(array) == 1:
        value = float(array[0])
        return value, value, value
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(n_boot, len(array)))
    means = array[indices].mean(axis=1)
    return float(array.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def linear_slope(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if mask.sum() < 2:
        return float("nan")
    return float(np.polyfit(np.log(x[mask]), np.log(y[mask]), 1)[0])


def relative_error(observed: torch.Tensor, expected: torch.Tensor) -> torch.Tensor:
    denominator = torch.clamp(expected.abs(), min=1e-8)
    return (observed - expected).abs() / denominator


gaussian_output_components = shared_core.gaussian_output_components


def componentwise_seed(
    seed: int,
    actor: core.GaussianActor,
    env: core.Environment,
    output_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    n = min(core.P.probe_states, len(env.train.s))
    ids = torch.arange(n, device=core.DEVICE)
    states = env.train.s[ids]
    actions = env.train.negative_actions[ids]
    advantages = env.train.negative_advantages[ids]

    actor.eval()
    with torch.no_grad():
        mu, log_std = actor(states)
        components = gaussian_output_components(mu, log_std, actions, core.P.action_dim)
        sigma = components["sigma"]
        raw_distance = components["raw_distance"]
        standardized2 = components["standardized2"]
        mean_score = components["mean_score"]
        log_scale_score = components["log_scale_score"]
        corrected_log_scale = components["corrected_log_scale"]
        joint_score = components["joint_score"]
        normalized_mean = components["normalized_mean"]
        normalized_quadratic = components["normalized_quadratic"]
        expected_linear = raw_distance
        expected_quadratic = raw_distance.square()

    # Output-tensor autograd validation. Expanding mu/log_std independently for
    # each contour action prevents summation over actions from obscuring the
    # per-action score.
    mu_probe = mu.detach()[:, None, :].expand(-1, actions.shape[1], -1).clone().requires_grad_(True)
    log_std_probe = log_std.detach()[:, None].expand(-1, actions.shape[1]).clone().requires_grad_(True)
    inv_std = torch.exp(-log_std_probe)[..., None]
    z = (actions.detach() - mu_probe) * inv_std
    log_prob = (
        -0.5 * z.square().sum(-1)
        - core.P.action_dim * log_std_probe
        - 0.5 * core.P.action_dim * math.log(2.0 * math.pi)
    )
    auto_mu, auto_log_scale = torch.autograd.grad(log_prob.sum(), (mu_probe, log_std_probe))
    analytic_mu = (actions.detach() - mu_probe.detach()) * torch.exp(-2.0 * log_std_probe.detach())[..., None]
    analytic_log_scale = z.detach().square().sum(-1) - core.P.action_dim
    auto_mu_norm = torch.linalg.vector_norm(auto_mu, dim=-1)

    flat_d = raw_distance.detach().cpu().numpy().reshape(-1)
    flat_linear = normalized_mean.detach().cpu().numpy().reshape(-1)
    flat_quadratic = normalized_quadratic.detach().cpu().numpy().reshape(-1)

    near_index, far_index = 0, 4
    near_mean = mean_score[:, near_index]
    far_mean = mean_score[:, far_index]
    near_log = log_scale_score[:, near_index].abs()
    far_log = log_scale_score[:, far_index].abs()
    near_joint = joint_score[:, near_index]
    far_joint = joint_score[:, far_index]
    near_distance = raw_distance[:, near_index]
    far_distance = raw_distance[:, far_index]

    linear_identity_error = relative_error(normalized_mean, expected_linear)
    quadratic_identity_error = relative_error(normalized_quadratic, expected_quadratic)
    mu_autograd_error = relative_error(auto_mu, analytic_mu)
    log_autograd_error = relative_error(auto_log_scale, analytic_log_scale)
    joint_reconstruction = torch.sqrt(auto_mu_norm.square() + auto_log_scale.square())
    joint_error = relative_error(joint_reconstruction, joint_score)

    contour_rows: list[dict[str, Any]] = []
    for state_index in range(n):
        for contour_index in range(actions.shape[1]):
            contour_rows.append(
                {
                    "seed": seed,
                    "state_index": state_index,
                    "contour_index": contour_index,
                    "advantage": float(advantages[state_index, contour_index].item()),
                    "sigma": float(sigma[state_index].item()),
                    "raw_distance": float(raw_distance[state_index, contour_index].item()),
                    "standardized_distance": float(torch.sqrt(standardized2[state_index, contour_index]).item()),
                    "mean_score_norm": float(mean_score[state_index, contour_index].item()),
                    "log_scale_score_signed": float(log_scale_score[state_index, contour_index].item()),
                    "corrected_log_scale_term": float(corrected_log_scale[state_index, contour_index].item()),
                    "joint_output_score_norm": float(joint_score[state_index, contour_index].item()),
                    "variance_normalized_mean_term": float(normalized_mean[state_index, contour_index].item()),
                    "variance_normalized_quadratic_term": float(normalized_quadratic[state_index, contour_index].item()),
                }
            )

    summary = {
        "seed": seed,
        "probe_states": n,
        "contour_actions_per_state": int(actions.shape[1]),
        "advantage_max_range_per_state": float((advantages.max(1).values - advantages.min(1).values).abs().max().item()),
        "raw_distance_far_near_ratio": float((far_distance / (near_distance + EPS)).mean().item()),
        "mean_score_far_near_ratio": float((far_mean / (near_mean + EPS)).mean().item()),
        "log_scale_score_abs_far_near_ratio": float((far_log / (near_log + EPS)).mean().item()),
        "corrected_log_scale_far_near_ratio": float(
            (corrected_log_scale[:, far_index] / (corrected_log_scale[:, near_index] + EPS)).mean().item()
        ),
        "joint_output_score_far_near_ratio": float((far_joint / (near_joint + EPS)).mean().item()),
        "normalized_mean_distance_loglog_slope": linear_slope(flat_d, flat_linear),
        "normalized_corrected_log_scale_distance_loglog_slope": linear_slope(flat_d, flat_quadratic),
        "linear_identity_max_relative_error": float(linear_identity_error.max().item()),
        "quadratic_identity_max_relative_error": float(quadratic_identity_error.max().item()),
        "mu_autograd_max_relative_error": float(mu_autograd_error.max().item()),
        "log_scale_autograd_max_relative_error": float(log_autograd_error.max().item()),
        "joint_reconstruction_max_relative_error": float(joint_error.max().item()),
    }
    atomic_json(output_root / "componentwise" / f"seed_{seed}.json", summary)
    write_csv(output_root / "componentwise" / f"seed_{seed}_raw.csv", contour_rows)
    return summary, contour_rows


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    excluded = {"seed", "probe_states", "contour_actions_per_state"}
    metrics = [key for key in rows[0] if key not in excluded]
    output: list[dict[str, Any]] = []
    for metric in metrics:
        values = [float(row[metric]) for row in rows if math.isfinite(float(row[metric]))]
        mean, low, high = mean_ci(values)
        output.append(
            {
                "metric": metric,
                "mean": mean,
                "ci_low": low,
                "ci_high": high,
                "min": min(values),
                "max": max(values),
                "n": len(values),
            }
        )
    return output


def build_terminal_audit(
    e1_rows: list[dict[str, Any]],
    e2_rows: list[dict[str, Any]],
    component_rows: list[dict[str, Any]],
    provenance: str,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, value: Any, threshold: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), "value": value, "threshold": threshold})

    add("twenty_seeds_completed", len(component_rows) == 20, len(component_rows), 20)
    finite = all(
        math.isfinite(float(value))
        for row in component_rows
        for key, value in row.items()
        if key not in {"seed", "probe_states", "contour_actions_per_state"}
    )
    add("all_component_metrics_finite", finite, finite, True)
    max_positive_grad = max(float(row["total_gradient_norm"]) for row in e2_rows)
    add("e2_terminal_positive_gradient_norm", max_positive_grad < 1e-3, max_positive_grad, "<1e-3")
    all_stable = all(row.get("status") == "stable_plateau_2x_confirmed" for row in e2_rows)
    add("e2_terminal_status", all_stable, all_stable, "all stable_plateau_2x_confirmed")

    mean_slope_error = max(abs(float(row["normalized_mean_distance_loglog_slope"]) - 1.0) for row in component_rows)
    quad_slope_error = max(
        abs(float(row["normalized_corrected_log_scale_distance_loglog_slope"]) - 2.0)
        for row in component_rows
    )
    add("mean_branch_linear_slope", mean_slope_error < 1e-4, mean_slope_error, "max |slope-1| <1e-4")
    add("log_scale_branch_quadratic_slope", quad_slope_error < 1e-4, quad_slope_error, "max |slope-2| <1e-4")

    max_auto = max(
        max(float(row["mu_autograd_max_relative_error"]), float(row["log_scale_autograd_max_relative_error"]))
        for row in component_rows
    )
    add("analytic_autograd_agreement", max_auto < 1e-5, max_auto, "<1e-5")

    e1_by_seed = {int(row["seed"]): row for row in e1_rows}
    joint_delta = max(
        abs(
            float(row["joint_output_score_far_near_ratio"])
            - float(e1_by_seed[int(row["seed"])]["output_score_far_near_ratio"])
        )
        for row in component_rows
    )
    add("existing_e1_joint_ratio_reconstructed", joint_delta < 1e-5, joint_delta, "<1e-5")
    advantage_error = max(abs(float(row["advantage_far_near_ratio"]) - 1.0) for row in e1_rows)
    add("equal_advantage_gate", advantage_error < 1e-6, advantage_error, "<1e-6")

    scientific_pass = all(check["passed"] for check in checks)
    formal_eligible = provenance == "clean_committed_exact_base"
    return {
        "experiment_id": EXPERIMENT_ID,
        "scientific_checks_passed": scientific_pass,
        "formal_provenance_eligible": formal_eligible,
        "provenance_classification": provenance,
        "result_status": "finite_step_validated" if scientific_pass and formal_eligible else "pilot",
        "checks": checks,
    }


def render_report(
    output_root: Path,
    aggregate_rows: list[dict[str, Any]],
    terminal_audit: dict[str, Any],
    e2_rows: list[dict[str, Any]],
) -> None:
    by_name = {row["metric"]: row for row in aggregate_rows}

    def fmt(metric: str, digits: int = 6) -> str:
        row = by_name[metric]
        return f"{row['mean']:.{digits}f} [{row['ci_low']:.{digits}f}, {row['ci_high']:.{digits}f}]"

    e2_reward = mean_ci([float(row["reward"]) for row in e2_rows])
    e2_sigma = mean_ci([float(row["sigma_mean"]) for row in e2_rows])
    e2_grad = mean_ci([float(row["total_gradient_norm"]) for row in e2_rows])
    report = f"""# {EXPERIMENT_ID} result report

## Status

- Scientific status: **{terminal_audit['result_status']}**.
- Scientific checks passed: `{terminal_audit['scientific_checks_passed']}`.
- Formal provenance eligible: `{terminal_audit['formal_provenance_eligible']}`.
- Scope: Gaussian output-space component decomposition only; neural-network pullback is excluded.

## Frozen E2 rerun

- Seeds: 10--29.
- Held-out-context reward: `{e2_reward[0]:.6f} [{e2_reward[1]:.6f}, {e2_reward[2]:.6f}]`.
- Learned sigma: `{e2_sigma[0]:.6f} [{e2_sigma[1]:.6f}, {e2_sigma[2]:.6f}]`.
- Final full-data positive-gradient norm: `{e2_grad[0]:.6e} [{e2_grad[1]:.6e}, {e2_grad[2]:.6e}]`.

## Component-wise E1 decomposition

- Equal-advantage range error: `{fmt('advantage_max_range_per_state', 10)}`.
- Raw-distance far/near ratio: `{fmt('raw_distance_far_near_ratio')}`.
- Mean-score far/near ratio: `{fmt('mean_score_far_near_ratio')}`.
- Absolute log-scale-score far/near ratio: `{fmt('log_scale_score_abs_far_near_ratio')}`.
- Corrected log-scale term far/near ratio: `{fmt('corrected_log_scale_far_near_ratio')}`.
- Joint output-score far/near ratio: `{fmt('joint_output_score_far_near_ratio')}`.
- Variance-normalized mean branch log-log slope: `{fmt('normalized_mean_distance_loglog_slope', 8)}` (theory: 1).
- Variance-normalized corrected log-scale branch log-log slope: `{fmt('normalized_corrected_log_scale_distance_loglog_slope', 8)}` (theory: 2).
- Maximum analytic/autograd errors are recorded in `componentwise_aggregate.csv` and the terminal audit.

## Interpretation boundary

The rerun verifies the Gaussian identities

`||score_mu|| * sigma^2 = d`

and

`(score_log_sigma + D) * sigma^2 = d^2`.

Thus the mean branch is linear in raw distance after controlling for learned variance, while the corrected log-scale branch is exactly quadratic. The raw log-scale score itself includes the additive `-D` term, so it should be described as asymptotically quadratic rather than as an exact pure monomial at every distance. The result supports nonlinear far-field control as a mechanism motivation but does not prove that exponential tapering must outperform linear or global controls.
"""
    (output_root / "RESULTS.md").write_text(report, encoding="utf-8")


def collect_files(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS.txt":
            rows.append(
                {
                    "path": str(path.relative_to(root)),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    return rows


def write_checksums(root: Path) -> None:
    lines = [f"{row['sha256']}  {row['path']}" for row in collect_files(root)]
    (root / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "drpo_cu1_e1_componentwise_results",
    )
    parser.add_argument(
        "--provenance",
        choices=("clean_committed_exact_base", "explicit_dirty_pilot", "reconstructed_source_pilot"),
        default="clean_committed_exact_base",
    )
    parser.add_argument("--base-commit", default=BASE_COMMIT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    core.ROOT = output_root

    started = time.time()
    run_manifest = {
        "experiment_id": EXPERIMENT_ID,
        "script_version": SCRIPT_VERSION,
        "base_commit": args.base_commit,
        "provenance_classification": args.provenance,
        "protocol": asdict(core.P),
        "device": str(core.DEVICE),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "python_version": sys.version,
        "platform": platform.platform(),
        "pid": os.getpid(),
        "start_time_unix": started,
    }
    atomic_json(output_root / "run_manifest.json", run_manifest)

    e1_rows: list[dict[str, Any]] = []
    e2_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    all_raw_rows: list[dict[str, Any]] = []

    for index, seed in enumerate(core.P.e1_e2_seeds, start=1):
        print(json.dumps({"event": "seed_start", "seed": seed, "index": index, "total": len(core.P.e1_e2_seeds)}), flush=True)
        actor, env, _, e2 = core.train_positive(seed)
        e1 = core.run_e1_seed(seed, actor, env)
        component, raw = componentwise_seed(seed, actor, env, output_root)
        e1_rows.append(e1)
        e2_rows.append(e2)
        component_rows.append(component)
        all_raw_rows.extend(raw)
        write_csv(output_root / "e1_per_seed.csv", e1_rows)
        write_csv(output_root / "e2_per_seed.csv", e2_rows)
        write_csv(output_root / "componentwise_per_seed.csv", component_rows)
        atomic_json(
            output_root / "progress.json",
            {
                "completed_seeds": [int(row["seed"]) for row in component_rows],
                "last_seed": seed,
                "completed": len(component_rows),
                "total": len(core.P.e1_e2_seeds),
                "updated_unix": time.time(),
            },
        )
        print(json.dumps({"event": "seed_complete", "seed": seed, "e2_grad": e2["total_gradient_norm"], "joint_ratio": component["joint_output_score_far_near_ratio"]}), flush=True)

    write_csv(output_root / "componentwise_raw_all_seeds.csv", all_raw_rows)
    aggregate_rows = aggregate(component_rows)
    write_csv(output_root / "componentwise_aggregate.csv", aggregate_rows)
    terminal_audit = build_terminal_audit(e1_rows, e2_rows, component_rows, args.provenance)
    atomic_json(output_root / "TERMINAL_AUDIT.json", terminal_audit)
    render_report(output_root, aggregate_rows, terminal_audit, e2_rows)

    completed = {
        "experiment_id": EXPERIMENT_ID,
        "exit_status": "success" if terminal_audit["scientific_checks_passed"] else "audit_failed",
        "result_status": terminal_audit["result_status"],
        "base_commit": args.base_commit,
        "provenance_classification": args.provenance,
        "seeds_completed": [int(row["seed"]) for row in component_rows],
        "duration_seconds": time.time() - started,
        "terminal_audit_passed": terminal_audit["scientific_checks_passed"],
    }
    atomic_json(output_root / "RUN_COMPLETE.json", completed)
    write_checksums(output_root)
    print(json.dumps(completed), flush=True)
    return 0 if terminal_audit["scientific_checks_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
