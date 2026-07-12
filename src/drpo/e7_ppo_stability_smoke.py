"""Real-data smoke gate for the canonical E7 PPO-stability pilot.

The smoke run is deliberately excluded from scientific aggregation.  It executes
one matched A2C/PPO pair on a development seed and writes a machine-readable gate
that the full pilot must validate before launch.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping

from drpo import e7_canonical_ppo_stability as pilot
from drpo import e7_canonical_ppo_stability_entry as entry
from drpo import e7_canonical_sweep as base
from drpo.e7_canonical_injection import sha256_file

EXPERIMENT_ID = pilot.EXPERIMENT_ID
SMOKE_RUN_KIND = "real_data_liveness_smoke_only"
SMOKE_DATASET = "walker2d-medium-v2"
SMOKE_SEED = 200
SMOKE_EXP_COEFFICIENT = 1.5
SMOKE_STEPS = 20_000
SMOKE_DIAGNOSTICS_INTERVAL = 1_000
SMOKE_MODES = ("a2c", "ppo_clip")
SMOKE_BRANCH_COUNT = 2

PROTECTED_IMPLEMENTATION_PATHS = (
    "configs/e7_canonical_ppo_stability_v1.json",
    "src/drpo/e7_canonical_ppo_injection.py",
    "src/drpo/e7_canonical_ppo_bootstrap.py",
    "src/drpo/e7_canonical_ppo_stability.py",
    "src/drpo/e7_canonical_ppo_stability_entry.py",
    "src/drpo/e7_ppo_stability_aggregate.py",
    "src/drpo/e7_ppo_stability_smoke.py",
    "src/drpo/e7_ppo_runtime_autotune.py",
    "scripts/run_e7_ppo_stability_smoke.py",
    "scripts/run_e7_ppo_stability_smoke_one_click.sh",
    "scripts/run_e7_ppo_stability_pilot_auto.py",
    "scripts/run_e7_ppo_stability_pilot_auto_one_click.sh",
)


class SmokeGateError(RuntimeError):
    """Raised when the real-data smoke gate cannot be established or reused."""


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _current_commit(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=15,
        ).strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise SmokeGateError("cannot resolve repository commit for smoke gate") from exc


def _fingerprints(repo_root: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for relative in PROTECTED_IMPLEMENTATION_PATHS:
        path = repo_root / relative
        if not path.is_file():
            raise SmokeGateError(f"protected smoke implementation file is missing: {relative}")
        values[relative] = sha256_file(path)
    return values


def _load_inputs(
    *,
    contract_path: Path,
    run_spec_path: Path,
    grid_path: Path,
) -> tuple[
    base.CanonicalContract,
    dict[str, Any],
    dict[str, Any],
    str,
    str,
    list[base.Branch],
]:
    contract = base.CanonicalContract.load(contract_path)
    contract.verify_runtime()

    previous = pilot._BASE_LOAD_RUN_SPEC  # noqa: SLF001
    pilot._BASE_LOAD_RUN_SPEC = entry._load_source_run_spec  # noqa: SLF001
    try:
        run_spec, run_spec_sha256 = pilot.load_ppo_run_spec(run_spec_path)
    finally:
        pilot._BASE_LOAD_RUN_SPEC = previous  # noqa: SLF001
    grid, grid_sha256 = pilot.load_ppo_grid(grid_path)
    branches = pilot.build_ppo_branches(contract, run_spec, grid)
    return contract, run_spec, grid, run_spec_sha256, grid_sha256, branches


def _smoke_trainer_template(run_spec: Mapping[str, Any]) -> list[str]:
    argv = [str(value) for value in run_spec["trainer_argv_template"]]
    positions = [index for index, token in enumerate(argv) if token == "--eval_interval"]
    if len(positions) != 1 or positions[0] + 1 >= len(argv):
        raise SmokeGateError(
            "smoke trainer template must contain exactly one --eval_interval"
        )
    current = argv[positions[0] + 1]
    if current != "50000":
        raise SmokeGateError(
            f"canonical smoke source eval interval changed: {current} != 50000"
        )
    argv[positions[0] + 1] = str(SMOKE_STEPS)
    return argv


def _select_smoke_branches(branches: Iterable[base.Branch]) -> list[base.Branch]:
    selected: list[base.Branch] = []
    for mode in SMOKE_MODES:
        matches = [
            branch
            for branch in branches
            if branch.dataset.id == SMOKE_DATASET
            and branch.seed == SMOKE_SEED
            and branch.template_values.get("actor_update_mode") == mode
            and branch.negative_control is not None
            and branch.negative_control.method == "exponential"
            and math.isclose(
                branch.negative_control.exponential_coefficient,
                SMOKE_EXP_COEFFICIENT,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ]
        if len(matches) != 1:
            raise SmokeGateError(
                f"expected one smoke branch for actor_update_mode={mode}, found {len(matches)}"
            )
        original = matches[0]
        selected.append(
            dataclasses.replace(
                original,
                branch_id=f"smoke20k__{original.branch_id}",
                template_values={
                    **original.template_values,
                    "steps": str(SMOKE_STEPS),
                    "diagnostics_interval": str(SMOKE_DIAGNOSTICS_INTERVAL),
                },
            )
        )
    return selected


def _trainer_summary(branch_dir: Path) -> dict[str, Any]:
    candidates = sorted((branch_dir / "trainer_output").glob("*_summary.json"))
    if len(candidates) != 1:
        raise SmokeGateError(
            f"expected one trainer summary under {branch_dir}, found {len(candidates)}"
        )
    value = json.loads(candidates[0].read_text())
    if not isinstance(value, dict):
        raise SmokeGateError(f"trainer summary is not a mapping: {candidates[0]}")
    return value


def _history_check(summary: Mapping[str, Any]) -> dict[str, Any]:
    history = summary.get("history")
    if not isinstance(history, dict):
        raise SmokeGateError("trainer summary is missing history")
    steps = [int(value) for value in history.get("steps", [])]
    score_keys = [key for key in history if key != "steps"]
    if len(score_keys) != 1:
        raise SmokeGateError("trainer smoke history must contain exactly one score series")
    scores = [float(value) for value in history[score_keys[0]]]
    if not steps or len(steps) != len(scores):
        raise SmokeGateError("trainer smoke history steps/scores mismatch")
    if steps[-1] != SMOKE_STEPS:
        raise SmokeGateError(
            f"trainer smoke final step mismatch: {steps[-1]} != {SMOKE_STEPS}"
        )
    if not all(math.isfinite(value) for value in scores):
        raise SmokeGateError("trainer smoke contains non-finite task scores")
    return {
        "evaluation_count": len(steps),
        "final_step": steps[-1],
        "final_score": scores[-1],
        "best_score": max(scores),
        "best_step": steps[max(range(len(scores)), key=scores.__getitem__)],
    }


def _fraction(value: Any, field: str) -> float:
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise SmokeGateError(f"invalid PPO fraction {field}={value!r}")
    return number


def _finite_number(value: Any, field: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise SmokeGateError(f"non-finite PPO diagnostic {field}={value!r}")
    return number


def _validate_ppo_diagnostics(branch_dir: Path) -> dict[str, Any]:
    latest_path = branch_dir / "PPO_DIAGNOSTICS_LATEST.json"
    jsonl_path = branch_dir / "ppo_diagnostics.jsonl"
    if not latest_path.is_file() or not jsonl_path.is_file():
        raise SmokeGateError("PPO smoke did not produce both diagnostics files")
    latest = json.loads(latest_path.read_text())
    if latest.get("status") != "complete":
        raise SmokeGateError("PPO smoke diagnostics did not reach complete status")
    if int(latest.get("update", -1)) != SMOKE_STEPS:
        raise SmokeGateError("PPO smoke diagnostics final update mismatch")

    expected_refreshes = (SMOKE_STEPS - 1) // pilot.EXPECTED_UPDATES_PER_OLD_POLICY
    refreshes = int(latest.get("old_policy_refresh_count", -1))
    if refreshes != expected_refreshes:
        raise SmokeGateError(
            f"old-policy refresh count mismatch: {refreshes} != {expected_refreshes}"
        )

    by_position = latest.get("pre_update_by_block_position")
    if not isinstance(by_position, dict):
        raise SmokeGateError("PPO smoke is missing block-position diagnostics")
    expected_positions = {
        str(position)
        for position in range(1, pilot.EXPECTED_UPDATES_PER_OLD_POLICY + 1)
    }
    if set(by_position) != expected_positions:
        raise SmokeGateError(
            f"PPO block positions changed: {sorted(by_position)} != {sorted(expected_positions)}"
        )

    position_one = by_position["1"]
    ratio_one = _finite_number(position_one.get("ratio_mean"), "position1.ratio_mean")
    if abs(ratio_one - 1.0) > 1e-5:
        raise SmokeGateError(
            f"old-policy block position 1 ratio is not one: {ratio_one}"
        )

    later_abs_log_ratio = [
        _finite_number(
            by_position[str(position)].get("abs_log_ratio_mean"),
            f"position{position}.abs_log_ratio_mean",
        )
        for position in range(2, pilot.EXPECTED_UPDATES_PER_OLD_POLICY + 1)
    ]
    if max(later_abs_log_ratio) <= 1e-9:
        raise SmokeGateError("PPO ratio never departed from one within the old-policy block")

    pre = latest.get("pre_update")
    if not isinstance(pre, dict):
        raise SmokeGateError("PPO smoke is missing pre-update diagnostics")
    ratio_min = _finite_number(pre.get("ratio_min"), "ratio_min")
    ratio_max = _finite_number(pre.get("ratio_max"), "ratio_max")
    if ratio_min <= 0.0 or ratio_max <= 0.0:
        raise SmokeGateError("PPO likelihood ratios must remain positive")
    ratio_outside = _fraction(pre.get("ratio_outside_fraction"), "ratio_outside_fraction")
    objective_clip = _fraction(
        pre.get("objective_clip_fraction"),
        "objective_clip_fraction",
    )
    positive_clip = _fraction(
        pre.get("positive_objective_clip_fraction"),
        "positive_objective_clip_fraction",
    )
    negative_clip = _fraction(
        pre.get("negative_objective_clip_fraction"),
        "negative_objective_clip_fraction",
    )

    for field in (
        "actor_gradient_norm",
        "actor_parameter_update_norm",
        "actor_relative_parameter_update_norm",
    ):
        value = _finite_number(latest.get(field), field)
        if value < 0.0:
            raise SmokeGateError(f"PPO norm must be non-negative: {field}={value}")

    sampled_post = latest.get("sampled_post_update")
    if not isinstance(sampled_post, dict):
        raise SmokeGateError("PPO smoke is missing sampled post-update diagnostics")
    for field, value in sampled_post.items():
        _finite_number(value, f"sampled_post_update.{field}")

    lines = [line for line in jsonl_path.read_text().splitlines() if line.strip()]
    expected_records = SMOKE_STEPS // SMOKE_DIAGNOSTICS_INTERVAL
    if len(lines) != expected_records:
        raise SmokeGateError(
            f"PPO diagnostics record count mismatch: {len(lines)} != {expected_records}"
        )

    return {
        "diagnostic_records": len(lines),
        "old_policy_refresh_count": refreshes,
        "position_1_ratio_mean": ratio_one,
        "later_position_abs_log_ratio_mean": later_abs_log_ratio,
        "ratio_min": ratio_min,
        "ratio_max": ratio_max,
        "ratio_outside_fraction": ratio_outside,
        "objective_clip_fraction": objective_clip,
        "positive_objective_clip_fraction": positive_clip,
        "negative_objective_clip_fraction": negative_clip,
        "latest_path": str(latest_path),
        "jsonl_path": str(jsonl_path),
    }


def validate_smoke_gate(
    *,
    repo_root: str | Path,
    smoke_dir: str | Path,
) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    root = Path(smoke_dir).resolve()
    gate_path = root / "SMOKE_GATE.json"
    if not gate_path.is_file():
        raise SmokeGateError(f"smoke gate is missing: {gate_path}")
    gate = json.loads(gate_path.read_text())
    if gate.get("status") != "pass":
        raise SmokeGateError(f"smoke gate is not pass: {gate.get('status')!r}")
    if gate.get("experiment_id") != EXPERIMENT_ID:
        raise SmokeGateError("smoke gate experiment identity mismatch")
    if gate.get("run_kind") != SMOKE_RUN_KIND:
        raise SmokeGateError("smoke gate run kind mismatch")
    expected = _fingerprints(repo)
    if gate.get("protected_implementation_sha256") != expected:
        raise SmokeGateError(
            "protected PPO implementation changed after the smoke gate; rerun smoke"
        )
    return gate


def run_smoke(
    *,
    repo_root: str | Path,
    contract_path: str | Path,
    run_spec_path: str | Path,
    grid_path: str | Path,
    work_dir: str | Path,
    resume: bool,
) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    contract_source = Path(contract_path).expanduser().resolve()
    run_spec_source = Path(run_spec_path).expanduser().resolve()
    grid_source = Path(grid_path).expanduser().resolve()
    work = Path(work_dir).resolve()
    work.mkdir(parents=True, exist_ok=True)

    (
        contract,
        run_spec,
        _grid,
        run_spec_sha256,
        grid_sha256,
        branches,
    ) = _load_inputs(
        contract_path=contract_source,
        run_spec_path=run_spec_source,
        grid_path=grid_source,
    )
    selected = _select_smoke_branches(branches)
    for branch in selected:
        branch.dataset.verify()

    previous = (
        base.EXPERIMENT_ID,
        base.SCIENTIFIC_STATUS,
        base.RUNNER_VERSION,
        base.branch_command,
    )
    base.EXPERIMENT_ID = EXPERIMENT_ID
    base.SCIENTIFIC_STATUS = SMOKE_RUN_KIND
    base.RUNNER_VERSION = f"{pilot.RUNNER_VERSION}-smoke-v1"
    base.branch_command = pilot.ppo_branch_command
    results: list[dict[str, Any]] = []
    try:
        for branch in selected:
            results.append(
                base.execute_branch(
                    contract_path=contract_source,
                    contract=contract,
                    branch=branch,
                    work_dir=work,
                    grid_sha256=grid_sha256,
                    run_spec_sha256=run_spec_sha256,
                    trainer_argv_template=_smoke_trainer_template(run_spec),
                    base_environment={
                        str(key): str(value)
                        for key, value in run_spec.get("environment", {}).items()
                    },
                    resume=resume,
                )
            )
    finally:
        (
            base.EXPERIMENT_ID,
            base.SCIENTIFIC_STATUS,
            base.RUNNER_VERSION,
            base.branch_command,
        ) = previous

    results.sort(key=lambda row: row["branch_id"])
    failed = [row for row in results if row["status"] == "failed"]
    report: dict[str, Any] = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "run_kind": SMOKE_RUN_KIND,
        "status": "fail",
        "repository_commit": _current_commit(repo),
        "contract_path": str(contract_source),
        "run_spec_path": str(run_spec_source),
        "grid_path": str(grid_source),
        "work_dir": str(work),
        "smoke_protocol": {
            "dataset": SMOKE_DATASET,
            "seed": SMOKE_SEED,
            "exp_coefficient": SMOKE_EXP_COEFFICIENT,
            "steps": SMOKE_STEPS,
            "diagnostics_interval": SMOKE_DIAGNOSTICS_INTERVAL,
            "actor_update_modes": list(SMOKE_MODES),
            "scientific_aggregation_allowed": False,
        },
        "branch_results": results,
        "checks": {},
        "protected_implementation_sha256": _fingerprints(repo),
    }
    try:
        if failed:
            raise SmokeGateError(f"{len(failed)} smoke branches failed")
        if len(results) != SMOKE_BRANCH_COUNT:
            raise SmokeGateError(
                f"smoke branch count mismatch: {len(results)} != {SMOKE_BRANCH_COUNT}"
            )
        branch_checks: dict[str, Any] = {}
        for branch in selected:
            branch_dir = work / "branches" / branch.branch_id
            if not (branch_dir / "COMPLETED.json").is_file():
                raise SmokeGateError(f"smoke branch is not completed: {branch.branch_id}")
            mode = branch.template_values["actor_update_mode"]
            branch_checks[mode] = {
                "trajectory": _history_check(_trainer_summary(branch_dir)),
            }
            if mode == "ppo_clip":
                branch_checks[mode]["ppo_diagnostics"] = _validate_ppo_diagnostics(
                    branch_dir
                )
        report["checks"] = {
            "branch_count": len(results),
            "failed_branches": 0,
            "matched_a2c_ppo_pair": True,
            "held_out_seeds_touched": False,
            "branches": branch_checks,
        }
        report["status"] = "pass"
    except Exception as exc:
        report["error_type"] = type(exc).__name__
        report["error"] = str(exc)
        _atomic_json(work / "SMOKE_GATE.json", report)
        _atomic_json(work / "RUN_SUMMARY.json", report)
        raise

    _atomic_json(work / "SMOKE_GATE.json", report)
    _atomic_json(work / "RUN_SUMMARY.json", report)
    return report
