"""Frozen protocol and provenance checks for the E7 canonical 1M shortlist."""

from __future__ import annotations

import dataclasses
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

import drpo.e7_canonical_sweep as sweep
from drpo.e7_canonical_injection import CanonicalContract

EXPERIMENT_ID = "EXT-H-E7-BENCH-01"
SCIENTIFIC_STATUS = "canonical_agent_two_dataset_fixed_shortlist_1m_pilot_only"
RUNNER_VERSION = "1.0.0-fixed-shortlist-1m"
EXPECTED_DATASETS = ("hopper-medium-replay-v2", "hopper-medium-expert-v2")
EXPECTED_DATASET_SHA256 = {
    "hopper-medium-replay-v2": (
        "e121c5f7c9857a307baa9edc6a2c3b48e85fedb9ac316ecddd0f48ca7ef4e39b"
    ),
    "hopper-medium-expert-v2": (
        "9d51ad87f8c905be3880d84c6140bcdb7fbf39a19e046a237f238ba34fec9e26"
    ),
}
EXPECTED_SEEDS = (200, 201, 202, 203)
EXPECTED_REPORTING_ALIASES = {
    "baseline__original_exp_rank_mr": "original_exp_rank_mr",
    "positive_only__scale0": "positive_only",
    "canonical_signed__scale1": "global_neg_0p11",
    "global__scale0p1": "global_neg_0p011",
    "reciprocal_linear__scale0p1": "reciprocal_linear_max0p011",
    "reciprocal_quadratic__scale0p1": "reciprocal_quadratic_max0p011",
    "exponential__scale0p1": "exponential_max0p011",
}
EXPECTED_FIXED_PROTOCOL = {
    "datasets": list(EXPECTED_DATASETS),
    "dataset_sha256": dict(EXPECTED_DATASET_SHA256),
    "seeds": list(EXPECTED_SEEDS),
    "steps": 1_000_000,
    "eval_interval": 50_000,
    "eval_episodes": 10,
    "checkpoint_interval": 50_000,
    "checkpoint_retention_fraction": 0.1,
    "parallel_workers_default": 40,
    "omp_threads_per_worker": 2,
    "primary_late_window_steps": [
        750_000,
        800_000,
        850_000,
        900_000,
        950_000,
        1_000_000,
    ],
}
EXPECTED_TERMINAL_AUDIT = {
    "registered_score_threshold": None,
    "stationarity_rule_registered": False,
    "default_terminal_classification": "fixed_horizon_inconclusive",
    "ranking_requires_future_registered_terminal_rule": True,
}
EXPECTED_TRAINER_FLAGS = {
    "--dataset": "{dataset_id}",
    "--hdf5": "{dataset_path}",
    "--variant": "iqlv_exp_rank",
    "--alpha": "0.11",
    "--tau": "0.5",
    "--temp": "5.0",
    "--steps": "1000000",
    "--batch": "256",
    "--lr": "0.0003",
    "--eval_interval": "50000",
    "--eval_episodes": "10",
    "--seed": "{seed}",
    "--out_dir": "{output_dir}",
    "--ckpt_dir": "{output_dir}/ckpts",
    "--ckpt_interval": "50000",
    "--last_pct": "0.1",
}


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def expected_canonical_root() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "src"
        / "drpo"
        / "e7_canonical_vendor"
        / "d4rl"
    ).resolve()


def _parse_flag_value_pairs(argv: Sequence[str]) -> dict[str, str]:
    if len(argv) % 2:
        raise ValueError("trainer_argv_template must contain flag/value pairs")
    parsed: dict[str, str] = {}
    for index in range(0, len(argv), 2):
        flag = str(argv[index])
        value = str(argv[index + 1])
        if not flag.startswith("--"):
            raise ValueError(f"unexpected trainer token at position {index}: {flag!r}")
        if flag in parsed:
            raise ValueError(f"duplicate trainer flag in frozen protocol: {flag}")
        parsed[flag] = value
    return parsed


def validate_fixed_protocol(
    contract: CanonicalContract,
    run_spec: Mapping[str, Any],
    grid: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the complete scientific matrix before any branch is launched."""

    if grid.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError(f"grid experiment_id must be {EXPERIMENT_ID}")
    if grid.get("run_kind") != "pilot":
        raise ValueError("the fixed shortlist must use run_kind='pilot'")
    if grid.get("scientific_status") != SCIENTIFIC_STATUS:
        raise ValueError("fixed shortlist scientific_status mismatch")
    if grid.get("fixed_protocol") != EXPECTED_FIXED_PROTOCOL:
        raise ValueError("fixed_protocol differs from the registered 1M matrix")
    if grid.get("terminal_audit") != EXPECTED_TERMINAL_AUDIT:
        raise ValueError("terminal_audit differs from the registered audit boundary")
    if grid.get("reporting_aliases") != EXPECTED_REPORTING_ALIASES:
        raise ValueError("reporting_aliases differ from the registered shortlist")
    if int(grid.get("branch_count_per_dataset_seed", -1)) != 6:
        raise ValueError("branch_count_per_dataset_seed must remain 6")
    if int(grid.get("expected_total_branches", -1)) != 56:
        raise ValueError("expected_total_branches must remain 56")

    if not math.isclose(
        float(grid.get("canonical_alpha")), 0.11, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("canonical_alpha must remain 0.11")
    if not math.isclose(
        float(grid.get("reference_distance")), 2.0, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("reference_distance must remain 2.0")
    expected_coefficients = {
        "reciprocal_linear": 0.4362580032734791,
        "reciprocal_quadratic": 0.5520268617673281,
        "exponential": 0.374162511054291,
    }
    coefficients = grid.get("coefficients")
    if not isinstance(coefficients, Mapping):
        raise ValueError("grid coefficients must be a mapping")
    if set(coefficients) != set(expected_coefficients):
        raise ValueError("grid coefficient keys changed")
    for name, expected in expected_coefficients.items():
        if not math.isclose(
            float(coefficients[name]), expected, rel_tol=0.0, abs_tol=1e-15
        ):
            raise ValueError(f"coefficient changed for {name}")

    if grid.get("anchors") != {
        "positive_only": {"negative_scale": 0.0},
        "canonical_signed": {"negative_scale": 1.0},
    }:
        raise ValueError("shortlist anchors changed")
    if grid.get("negative_scale_grid") != {
        "global": [0.1],
        "reciprocal_linear": [0.1],
        "reciprocal_quadratic": [0.1],
        "exponential": [0.1],
    }:
        raise ValueError("negative_scale_grid changed")

    if run_spec.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("run_spec experiment_id changed")
    if run_spec.get("wrapper") != "canonical-agent-two-dataset-v2-self-contained":
        raise ValueError("run_spec wrapper changed")
    if run_spec.get("scope") != "two_dataset_canonical_agent_validation_only":
        raise ValueError("run_spec scope changed")
    if run_spec.get("run_kind") != "pilot":
        raise ValueError("run_spec.run_kind must remain pilot")
    if run_spec.get("profile") != "taper-pilot":
        raise ValueError("run_spec.profile must remain taper-pilot")
    if run_spec.get("injected_template_values") not in (None, {}):
        raise ValueError("injected_template_values are not allowed in the fixed shortlist")

    datasets = run_spec.get("datasets", [])
    dataset_ids = [str(item["id"]) for item in datasets]
    if dataset_ids != list(EXPECTED_DATASETS):
        raise ValueError(f"run_spec datasets changed: {dataset_ids}")
    for item in datasets:
        dataset_id = str(item["id"])
        actual_sha256 = str(item.get("sha256", "")).lower()
        expected_sha256 = EXPECTED_DATASET_SHA256[dataset_id]
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"dataset SHA-256 changed for {dataset_id}: "
                f"{actual_sha256} != {expected_sha256}"
            )

    seeds = [int(value) for value in run_spec.get("seeds", [])]
    if seeds != list(EXPECTED_SEEDS):
        raise ValueError(f"run_spec seeds changed: {seeds}")

    passthrough = run_spec.get("passthrough_variants", [])
    passthrough_ids = [str(item.get("id")) for item in passthrough]
    if passthrough_ids != ["original_exp_rank_mr"]:
        raise ValueError("passthrough baseline changed")
    if passthrough[0].get("template_values", {}) != {}:
        raise ValueError("passthrough baseline template values changed")

    trainer_flags = _parse_flag_value_pairs(run_spec.get("trainer_argv_template", []))
    if trainer_flags != EXPECTED_TRAINER_FLAGS:
        added = sorted(set(trainer_flags) - set(EXPECTED_TRAINER_FLAGS))
        removed = sorted(set(EXPECTED_TRAINER_FLAGS) - set(trainer_flags))
        changed = sorted(
            key
            for key in set(trainer_flags) & set(EXPECTED_TRAINER_FLAGS)
            if trainer_flags[key] != EXPECTED_TRAINER_FLAGS[key]
        )
        raise ValueError(
            "trainer arguments differ from the frozen protocol: "
            f"added={added}, removed={removed}, changed={changed}"
        )

    expected_environment = {
        "OMP_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "2",
    }
    if run_spec.get("environment") != expected_environment:
        raise ValueError("trainer thread environment differs from the frozen protocol")

    if contract.source_root != expected_canonical_root():
        raise ValueError(
            "canonical source root changed: "
            f"{contract.source_root} != {expected_canonical_root()}"
        )
    if contract.agents_relpath != "agents.py":
        raise ValueError("canonical agents_relpath changed")
    if contract.trainer_relpath != "train_sna2c_variant.py":
        raise ValueError("canonical trainer_relpath changed")
    if contract.module_name != "agents":
        raise ValueError("canonical module_name changed")
    if contract.target_class != "SNA2C_IQLV_ExpRankAgent":
        raise ValueError("canonical target class changed")
    if contract.agent_flavor != "signed_td_v_v1":
        raise ValueError("canonical agent flavor changed")
    if contract.return_mode != "zero_float":
        raise ValueError("canonical return mode changed")
    if not math.isclose(
        contract.expected_canonical_alpha, 0.11, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("canonical contract alpha changed")

    controls = sweep.expand_injected_controls(grid)
    if len(controls) != 6:
        raise ValueError("fixed shortlist must contain six injected controls")

    return {
        "status": "PASS",
        "experiment_id": EXPERIMENT_ID,
        "scientific_status": SCIENTIFIC_STATUS,
        "datasets": dataset_ids,
        "dataset_sha256": dict(EXPECTED_DATASET_SHA256),
        "seeds": seeds,
        "reporting_ids": list(EXPECTED_REPORTING_ALIASES.values()),
        "expected_branches": 56,
        "canonical_source_root": str(expected_canonical_root()),
    }


def _internal_alias_key(branch: sweep.Branch) -> str:
    parts = branch.branch_id.split("__")
    if len(parts) < 4:
        raise ValueError(f"unexpected generic branch id: {branch.branch_id}")
    return "__".join(parts[2:])


def apply_reporting_aliases(
    branches: Sequence[sweep.Branch], aliases: Mapping[str, str]
) -> list[sweep.Branch]:
    """Replace generic method/scale suffixes with frozen reporting IDs."""

    remapped: list[sweep.Branch] = []
    used_alias_keys: set[str] = set()
    for branch in branches:
        key = _internal_alias_key(branch)
        if key not in aliases:
            raise ValueError(f"no reporting alias registered for {key}")
        reporting_id = str(aliases[key])
        used_alias_keys.add(key)
        branch_id = f"{branch.dataset.id}__seed{branch.seed}__{reporting_id}"
        remapped.append(dataclasses.replace(branch, branch_id=branch_id))

    if used_alias_keys != set(aliases):
        unused = sorted(set(aliases) - used_alias_keys)
        raise ValueError(f"unused reporting aliases: {unused}")
    branch_ids = [branch.branch_id for branch in remapped]
    if len(branch_ids) != len(set(branch_ids)):
        raise ValueError("reporting aliases produced duplicate branch IDs")
    return remapped


def _git(repo_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "git "
            f"{' '.join(args)} failed: "
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )
    return completed.stdout.strip()


def capture_repository_provenance(*, require_clean_main: bool) -> dict[str, Any]:
    """Capture and optionally enforce clean current-origin/main provenance."""

    repo_root = Path(__file__).resolve().parents[2]
    head = _git(repo_root, "rev-parse", "HEAD")
    status = _git(repo_root, "status", "--porcelain=v1", "--untracked-files=all")
    branch = _git(repo_root, "rev-parse", "--abbrev-ref", "HEAD")
    origin_url = _git(repo_root, "remote", "get-url", "origin")
    remote_main: str | None = None
    remote_error: str | None = None
    try:
        response = _git(repo_root, "ls-remote", "origin", "refs/heads/main")
        fields = response.split()
        if len(fields) != 2 or fields[1] != "refs/heads/main":
            raise RuntimeError(f"unexpected ls-remote response: {response!r}")
        remote_main = fields[0]
    except RuntimeError as exc:
        remote_error = str(exc)

    payload = {
        "repo_root": str(repo_root),
        "head_commit": head,
        "branch": branch,
        "worktree_clean": not bool(status),
        "origin_url": origin_url,
        "origin_main_commit": remote_main,
        "origin_main_resolution_error": remote_error,
        "require_clean_main": require_clean_main,
    }
    if require_clean_main:
        if status:
            raise RuntimeError("fixed shortlist launch requires a clean worktree")
        if remote_main is None:
            raise RuntimeError(
                "cannot authoritatively resolve origin/main for fixed shortlist launch"
            )
        if head != remote_main:
            raise RuntimeError(
                "fixed shortlist launch requires HEAD == origin/main: "
                f"{head} != {remote_main}"
            )
    return payload
