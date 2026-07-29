#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

CONTRACT="${CONTRACT:-/root/d4rl2/configs/e7_canonical_contract_9task.json}"
SOURCE_RUN_SPEC="${SOURCE_RUN_SPEC:-/root/d4rl2/configs/e7_canonical_9task_full_grid_run_spec_v1.json}"
MASTER_CONFIG="${MASTER_CONFIG:-configs/e7_canonical_d4rl9_exp_tau_tuning_v1.json}"
WORK_ROOT="${WORK_ROOT:-outputs/e7/d4rl9_exp_tau_tuning_run_001}"
RUNTIME_ADAPTER="${WORK_ROOT}/generated_runtime/e7_exp_tau_runtime_adapter.py"

validate_master() {
  python3 - "${MASTER_CONFIG}" <<'PY'
import json
import math
import sys
from pathlib import Path

path = Path(sys.argv[1])
raw = json.loads(path.read_text())
expected_datasets = [
    "hopper-medium-v2",
    "hopper-medium-replay-v2",
    "hopper-medium-expert-v2",
    "walker2d-medium-v2",
    "walker2d-medium-replay-v2",
    "walker2d-medium-expert-v2",
    "halfcheetah-medium-v2",
    "halfcheetah-medium-replay-v2",
    "halfcheetah-medium-expert-v2",
]
expected_scalars = {
    "experiment_id": "EXT-H-E7-D4RL9-EXP-TAU-TUNE-01",
    "source_experiment_id": "EXT-H-E7-BENCH-01",
    "predecessor_experiment_id": "EXT-H-E7-D4RL9-EXP-ALPHA-C-JOINT-TUNE-01",
    "run_kind": "pilot",
    "scientific_status": "d4rl9_taskwise_exponential_taper_onset_tau_tuning_pilot_only",
    "runner_version": "1.0.0-d4rl9-exp-taper-onset-tau",
    "taper_coordinate": "normalized_rms_standardized_distance_u_equals_distance_over_reference_distance",
    "taper_formula": "exp(-exponential_coefficient * max(u - taper_onset_tau, 0))",
    "fixed_max_workers": 60,
    "parallel_units": 15,
    "workers_per_unit": 4,
    "operating_points_per_task": 3,
    "candidate_count_per_task": 15,
    "expected_unit_count": 135,
    "expected_total_branches": 540,
    "primary_selection_metric": "late_window_mean_750k_to_1m",
}
for key, expected in expected_scalars.items():
    if raw.get(key) != expected:
        raise SystemExit(f"{key} changed: expected {expected!r}, got {raw.get(key)!r}")
if raw.get("expected_datasets") != expected_datasets:
    raise SystemExit("expected_datasets changed")
if raw.get("source_run_spec_seeds") != [200, 201]:
    raise SystemExit("source_run_spec_seeds changed")
if raw.get("tuning_seeds") != [200, 201, 202, 203]:
    raise SystemExit("tuning_seeds changed")
if raw.get("held_out_seeds") != [204, 205, 206, 207]:
    raise SystemExit("held_out_seeds changed")
if raw.get("late_window_steps") != [750000, 800000, 850000, 900000, 950000, 1000000]:
    raise SystemExit("late_window_steps changed")
for key, expected in (("canonical_alpha", 0.11), ("reference_distance", 2.0)):
    if not math.isclose(float(raw.get(key)), expected, rel_tol=0.0, abs_tol=1e-12):
        raise SystemExit(f"{key} must remain {expected}")
taus = [float(value) for value in raw.get("taper_onset_tau_grid", [])]
if taus != [0.0, 0.125, 0.25, 0.375, 0.5]:
    raise SystemExit(f"taper_onset_tau_grid changed: {taus}")
if any(not math.isfinite(value) or value < 0.0 for value in taus):
    raise SystemExit("taper_onset_tau values must be finite and nonnegative")
points = raw.get("taskwise_operating_points", {})
if list(points) != expected_datasets:
    raise SystemExit("taskwise_operating_points coverage/order changed")
unit_count = 0
for dataset_id in expected_datasets:
    rows = points[dataset_id]
    if len(rows) != 3:
        raise SystemExit(f"{dataset_id}: expected exactly three operating points")
    identities = set()
    roles = set()
    for row in rows:
        role = str(row["role"])
        scale = float(row["negative_scale"])
        coefficient = float(row["exponential_coefficient"])
        if role in roles:
            raise SystemExit(f"{dataset_id}: duplicate role {role}")
        roles.add(role)
        if not math.isfinite(scale) or scale < 0.0:
            raise SystemExit(f"{dataset_id}: invalid negative_scale")
        if not math.isfinite(coefficient) or coefficient < 0.0:
            raise SystemExit(f"{dataset_id}: invalid exponential_coefficient")
        identity = (float(format(scale, '.12g')), float(format(coefficient, '.12g')))
        if identity in identities:
            raise SystemExit(f"{dataset_id}: duplicate operating point")
        identities.add(identity)
        unit_count += len(taus)
if unit_count != 135:
    raise SystemExit(f"expected 135 units, got {unit_count}")
if unit_count * 4 != 540:
    raise SystemExit("expected 540 branches")
for u in (0.0, 0.125, 0.5, 1.0, 2.0, 4.0):
    old = math.exp(-0.425 * u)
    new = math.exp(-0.425 * max(u - 0.0, 0.0))
    if old != new:
        raise SystemExit("taper_onset_tau=0 formula is not exact")
print(json.dumps({
    "status": "PASS",
    "experiment_id": raw["experiment_id"],
    "tasks": 9,
    "operating_points_per_task": 3,
    "tau_values": 5,
    "units": 135,
    "branches": 540,
}, sort_keys=True))
PY
}

write_runtime_adapter() {
  mkdir -p "$(dirname "${RUNTIME_ADAPTER}")"
  cat > "${RUNTIME_ADAPTER}" <<'PY'
#!/usr/bin/env python3
"""Generated runtime adapter for the E7 DRPO taper-onset tau development pilot.

This file is generated under the output root by the tracked one-click launcher.
It reuses the verified canonical trainer and the repository's canonical injection
layer, changing only the detached Exponential shape from exp(-c*u) to
exp(-c*max(u-taper_onset_tau, 0)). The trainer's own ``--tau`` flag is untouched.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import runpy
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch

from drpo import e7_canonical_injection as injection


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(payload: Any) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def format_template(value: str, context: Mapping[str, Any]) -> str:
    try:
        return value.format_map(context)
    except KeyError as exc:
        raise ValueError(f"unknown trainer template placeholder: {exc.args[0]}") from exc


def thresholded_taper_factor(
    distance: torch.Tensor,
    control: injection.NegativeControl,
    taper_onset_tau: float,
) -> torch.Tensor:
    if not math.isfinite(taper_onset_tau) or taper_onset_tau < 0.0:
        raise ValueError("taper_onset_tau must be finite and nonnegative")
    if control.method != "exponential":
        return injection.taper_factor(distance, control)
    u = distance / control.reference_distance
    active_u = torch.clamp(u - taper_onset_tau, min=0.0)
    exponent = torch.clamp(
        -control.exponential_coefficient * active_u,
        min=-40.0,
        max=0.0,
    )
    return torch.exp(exponent)


def self_test() -> int:
    control = injection.NegativeControl(
        method="exponential",
        negative_scale=0.01,
        canonical_alpha=0.11,
        reference_distance=2.0,
        exponential_coefficient=0.425,
    )
    distance = torch.tensor([0.0, 0.25, 0.5, 1.0, 2.0, 4.0])
    old = injection.taper_factor(distance, control)
    zero = thresholded_taper_factor(distance, control, 0.0)
    if not torch.equal(old, zero):
        raise SystemExit("tau=0 is not bitwise identical to the predecessor taper")
    tau = 0.25
    shifted = thresholded_taper_factor(distance, control, tau)
    u = distance / control.reference_distance
    expected = torch.exp(-0.425 * torch.clamp(u - tau, min=0.0))
    if not torch.equal(shifted, expected):
        raise SystemExit("thresholded taper formula mismatch")
    if not bool(torch.equal(shifted[u <= tau], torch.ones_like(shifted[u <= tau]))):
        raise SystemExit("near-field plateau is not exactly one")
    try:
        thresholded_taper_factor(distance, control, -0.1)
    except ValueError:
        pass
    else:
        raise SystemExit("negative taper_onset_tau did not fail closed")
    print(json.dumps({"status": "PASS", "tau_zero_exact": True}, sort_keys=True))
    return 0


def run_branch(config_path: Path, contract_path: Path, manifest_path: Path) -> int:
    branch = json.loads(config_path.read_text())
    contract = injection.CanonicalContract.load(contract_path)
    control_raw = dict(branch["negative_control"])
    onset = float(control_raw.pop("taper_onset_tau"))
    control = injection.NegativeControl.from_mapping(control_raw)
    original_taper = injection.taper_factor

    def runtime_taper(distance: torch.Tensor, active_control: injection.NegativeControl) -> torch.Tensor:
        if active_control.method != "exponential":
            return original_taper(distance, active_control)
        return thresholded_taper_factor(distance, active_control, onset)

    injection.taper_factor = runtime_taper
    module, source_checks = injection.load_verified_canonical_module(contract)
    injection.patch_canonical_module(module, contract, control)
    trainer_args = [str(value) for value in branch["trainer_args"]]
    manifest = {
        "status": "started",
        "branch": branch,
        "source_checks": source_checks,
        "negative_control": branch["negative_control"],
        "taper_onset_tau": onset,
        "taper_coordinate": branch["taper_coordinate"],
        "taper_formula": branch["taper_formula"],
        "trainer_path": str(contract.trainer_path),
        "trainer_args": trainer_args,
        "environment": injection.canonical_environment_manifest(),
    }
    atomic_write_json(manifest_path, manifest)
    old_argv = sys.argv[:]
    old_cwd = Path.cwd()
    try:
        os.chdir(contract.source_root)
        sys.argv = [str(contract.trainer_path), *trainer_args]
        try:
            runpy.run_path(str(contract.trainer_path), run_name="__main__")
        except SystemExit as exc:
            if exc.code not in (None, 0):
                raise
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["error_type"] = type(exc).__name__
        manifest["error"] = str(exc)
        atomic_write_json(manifest_path, manifest)
        raise
    finally:
        sys.argv = old_argv
        os.chdir(old_cwd)
        injection.taper_factor = original_taper
    manifest["status"] = "completed"
    atomic_write_json(manifest_path, manifest)
    return 0


def execute_seed(
    *,
    unit: Mapping[str, Any],
    seed: int,
    contract_path: Path,
    unit_work_dir: Path,
    resume: bool,
) -> dict[str, Any]:
    label = lambda value: format(float(value), ".12g").replace("-", "m").replace(".", "p")
    branch_id = (
        f"{unit['dataset']['id']}__seed{seed}__exponential__"
        f"s{label(unit['negative_scale'])}__c{label(unit['exponential_coefficient'])}__"
        f"tau{label(unit['taper_onset_tau'])}"
    )
    branch_dir = unit_work_dir / "branches" / branch_id
    branch_dir.mkdir(parents=True, exist_ok=True)
    output_dir = branch_dir / "trainer_output"
    context = {
        "canonical_root": str(injection.CanonicalContract.load(contract_path).source_root),
        "dataset_id": unit["dataset"]["id"],
        "dataset_path": str(Path(unit["dataset"]["path"]).expanduser().resolve()),
        "dataset_sha256": unit["dataset"]["sha256"],
        "seed": seed,
        "output_dir": str(output_dir),
        "branch_id": branch_id,
        **{str(key): str(value) for key, value in unit.get("injected_template_values", {}).items()},
    }
    trainer_args = [format_template(str(item), context) for item in unit["trainer_argv_template"]]
    branch_config = {
        "experiment_id": unit["experiment_id"],
        "scientific_status": unit["scientific_status"],
        "runner_version": unit["runner_version"],
        "branch_id": branch_id,
        "branch_kind": "injected",
        "candidate_id": unit["candidate_id"],
        "candidate_index": unit["candidate_index"],
        "operating_point_index": unit["operating_point_index"],
        "operating_point_role": unit["operating_point_role"],
        "dataset_id": unit["dataset"]["id"],
        "dataset_sha256": unit["dataset"]["sha256"],
        "seed": seed,
        "negative_control": {
            "method": "exponential",
            "negative_scale": unit["negative_scale"],
            "canonical_alpha": unit["canonical_alpha"],
            "reference_distance": unit["reference_distance"],
            "reciprocal_linear_coefficient": 0.5,
            "reciprocal_quadratic_coefficient": 0.5,
            "exponential_coefficient": unit["exponential_coefficient"],
            "taper_onset_tau": unit["taper_onset_tau"],
        },
        "taper_coordinate": unit["taper_coordinate"],
        "taper_formula": unit["taper_formula"],
        "trainer_args": trainer_args,
    }
    identity_payload = {
        **branch_config,
        "contract_sha256": sha256_file(contract_path),
        "unit_config_sha256": unit["unit_config_sha256"],
        "runtime_adapter_sha256": unit["runtime_adapter_sha256"],
    }
    identity = canonical_json_sha256(identity_payload)
    identity_path = branch_dir / "BRANCH_IDENTITY.json"
    completed_path = branch_dir / "COMPLETED.json"
    if identity_path.is_file():
        existing = json.loads(identity_path.read_text())
        if existing.get("identity_sha256") != identity:
            raise RuntimeError(f"branch identity mismatch: {branch_id}")
        if completed_path.is_file() and resume:
            done = json.loads(completed_path.read_text())
            return {"branch_id": branch_id, "status": "skipped", **done}
        if not resume:
            raise RuntimeError(f"branch exists; pass --resume: {branch_id}")
    else:
        if any(branch_dir.iterdir()):
            raise RuntimeError(f"non-empty branch directory has no identity: {branch_dir}")
        atomic_write_json(identity_path, {**identity_payload, "identity_sha256": identity, "created_utc": utc_now()})
    config_path = branch_dir / "branch_config.json"
    manifest_path = branch_dir / "branch_manifest.json"
    atomic_write_json(config_path, branch_config)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "branch",
        "--contract",
        str(contract_path),
        "--branch-config",
        str(config_path),
        "--branch-manifest",
        str(manifest_path),
    ]
    environment = os.environ.copy()
    environment.update({str(key): str(value) for key, value in unit.get("environment", {}).items()})
    log_path = branch_dir / "stdout.log"
    with log_path.open("w") as handle:
        process = subprocess.Popen(
            command,
            cwd=str(Path(unit["repo_root"])),
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return_code = int(process.wait())
    completed = {
        "branch_id": branch_id,
        "return_code": return_code,
        "finished_utc": utc_now(),
        "identity_sha256": identity,
        "stdout_log": str(log_path),
    }
    atomic_write_json(completed_path, completed)
    if return_code != 0:
        raise RuntimeError(f"branch failed with return code {return_code}: {branch_id}")
    return {"branch_id": branch_id, "status": "completed", **completed}


def run_unit(unit_config_path: Path, contract_path: Path, work_dir: Path, resume: bool) -> int:
    unit = json.loads(unit_config_path.read_text())
    unit["unit_config_sha256"] = sha256_file(unit_config_path)
    unit["runtime_adapter_sha256"] = sha256_file(Path(__file__).resolve())
    dataset_path = Path(unit["dataset"]["path"]).expanduser().resolve()
    if not dataset_path.is_file():
        raise FileNotFoundError(f"dataset missing: {dataset_path}")
    actual_dataset_sha = sha256_file(dataset_path)
    if actual_dataset_sha != str(unit["dataset"]["sha256"]).lower():
        raise RuntimeError(
            f"dataset SHA mismatch for {unit['dataset']['id']}: "
            f"expected {unit['dataset']['sha256']}, got {actual_dataset_sha}"
        )
    seeds = [int(value) for value in unit["seeds"]]
    if seeds != [200, 201, 202, 203]:
        raise RuntimeError(f"development seed identity changed: {seeds}")
    work_dir.mkdir(parents=True, exist_ok=True)
    results = []
    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(
                execute_seed,
                unit=unit,
                seed=seed,
                contract_path=contract_path,
                unit_work_dir=work_dir,
                resume=resume,
            ): seed
            for seed in seeds
        }
        for future in concurrent.futures.as_completed(futures):
            seed = futures[future]
            try:
                results.append(future.result())
            except BaseException as exc:
                failures.append({"seed": seed, "error_type": type(exc).__name__, "error": str(exc)})
    summary = {
        "experiment_id": unit["experiment_id"],
        "unit_id": unit["unit_id"],
        "candidate_id": unit["candidate_id"],
        "branch_count": len(seeds),
        "completed": sum(int(item.get("return_code", -1)) == 0 for item in results),
        "failed": len(failures),
        "results": sorted(results, key=lambda item: item["branch_id"]),
        "failures": failures,
        "finished_utc": utc_now(),
    }
    atomic_write_json(work_dir / "RUN_SUMMARY.json", summary)
    if failures:
        raise RuntimeError(f"unit failed: {unit['unit_id']}")
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    sub = root.add_subparsers(dest="command", required=True)
    sub.add_parser("self-test")
    unit = sub.add_parser("unit")
    unit.add_argument("--contract", required=True)
    unit.add_argument("--unit-config", required=True)
    unit.add_argument("--work-dir", required=True)
    unit.add_argument("--resume", action="store_true")
    branch = sub.add_parser("branch")
    branch.add_argument("--contract", required=True)
    branch.add_argument("--branch-config", required=True)
    branch.add_argument("--branch-manifest", required=True)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "self-test":
        return self_test()
    if args.command == "unit":
        return run_unit(
            Path(args.unit_config).resolve(),
            Path(args.contract).resolve(),
            Path(args.work_dir).resolve(),
            bool(args.resume),
        )
    if args.command == "branch":
        return run_branch(
            Path(args.branch_config).resolve(),
            Path(args.contract).resolve(),
            Path(args.branch_manifest).resolve(),
        )
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
PY
  chmod 700 "${RUNTIME_ADAPTER}"
}

prepare_matrix() {
  python3 - "${MASTER_CONFIG}" "${SOURCE_RUN_SPEC}" "${WORK_ROOT}" "${RUNTIME_ADAPTER}" "${REPO_ROOT}" <<'PY'
import copy
import hashlib
import json
import sys
from pathlib import Path

master_path = Path(sys.argv[1]).resolve()
source_path = Path(sys.argv[2]).resolve()
work_root = Path(sys.argv[3]).resolve()
runtime_adapter = Path(sys.argv[4]).resolve()
repo_root = Path(sys.argv[5]).resolve()
master = json.loads(master_path.read_text())
source = json.loads(source_path.read_text())
datasets = list(master["expected_datasets"])
tuning_seeds = list(master["tuning_seeds"])
held_out = set(master["held_out_seeds"])
if source.get("experiment_id") != master["source_experiment_id"]:
    raise SystemExit("source run-spec experiment_id changed")
if source.get("run_kind") not in {"pilot", "smoke"}:
    raise SystemExit("source run-spec run_kind must remain pilot/smoke")
if list(source.get("seeds", [])) != list(master["source_run_spec_seeds"]):
    raise SystemExit("source run-spec seed identity changed")
source_dataset_ids = [str(item["id"]) for item in source.get("datasets", [])]
if source_dataset_ids != datasets:
    raise SystemExit(f"source run-spec datasets changed: {source_dataset_ids}")
if held_out.intersection(tuning_seeds):
    raise SystemExit("held-out and tuning seeds overlap")
argv = [str(item) for item in source["trainer_argv_template"]]
def flag_value(flag):
    positions = [index for index, token in enumerate(argv) if token == flag]
    if len(positions) != 1 or positions[0] + 1 >= len(argv):
        raise SystemExit(f"source trainer template must contain exactly one {flag}")
    return argv[positions[0] + 1]
for flag, expected in master["fixed_trainer_flags"].items():
    actual = flag_value(flag)
    if actual != str(expected):
        raise SystemExit(f"source trainer flag changed: {flag} expected {expected}, got {actual}")
for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    if str(source.get("environment", {}).get(name)) != "1":
        raise SystemExit(f"source environment {name} must remain 1")

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def label(value):
    return format(float(value), ".12g").replace("-", "m").replace(".", "p")

generated = work_root / "generated"
generated.mkdir(parents=True, exist_ok=True)
dataset_by_id = {str(item["id"]): copy.deepcopy(item) for item in source["datasets"]}
units = []
matrix = []
taus = [float(value) for value in master["taper_onset_tau_grid"]]
for dataset_id in datasets:
    points = master["taskwise_operating_points"][dataset_id]
    candidate_index = 0
    for operating_point_index, point in enumerate(points, start=1):
        scale = float(point["negative_scale"])
        coefficient = float(point["exponential_coefficient"])
        role = str(point["role"])
        for tau in taus:
            candidate_index += 1
            candidate_id = (
                f"{dataset_id}__cand{candidate_index:02d}__op{operating_point_index}__"
                f"s{label(scale)}__c{label(coefficient)}__tau{label(tau)}"
            )
            unit_dir = generated / candidate_id
            unit_dir.mkdir(parents=True, exist_ok=True)
            unit_config_path = unit_dir / "unit_config.json"
            unit_work_dir = work_root / "units" / candidate_id
            unit = {
                "schema_version": 1,
                "experiment_id": master["experiment_id"],
                "scientific_status": master["scientific_status"],
                "runner_version": master["runner_version"],
                "unit_id": candidate_id,
                "candidate_id": candidate_id,
                "candidate_index": candidate_index,
                "operating_point_index": operating_point_index,
                "operating_point_role": role,
                "dataset": dataset_by_id[dataset_id],
                "seeds": tuning_seeds,
                "negative_scale": scale,
                "exponential_coefficient": coefficient,
                "taper_onset_tau": tau,
                "canonical_alpha": master["canonical_alpha"],
                "reference_distance": master["reference_distance"],
                "effective_near_coefficient": master["canonical_alpha"] * scale,
                "taper_coordinate": master["taper_coordinate"],
                "taper_formula": master["taper_formula"],
                "trainer_argv_template": source["trainer_argv_template"],
                "injected_template_values": source.get("injected_template_values", {}),
                "environment": source.get("environment", {}),
                "repo_root": str(repo_root),
            }
            unit_config_path.write_text(json.dumps(unit, indent=2) + "\n")
            plan_unit = {
                **{key: unit[key] for key in (
                    "unit_id", "candidate_id", "candidate_index", "operating_point_index",
                    "operating_point_role", "negative_scale", "exponential_coefficient",
                    "taper_onset_tau", "effective_near_coefficient"
                )},
                "dataset_id": dataset_id,
                "unit_config": str(unit_config_path),
                "work_dir": str(unit_work_dir),
                "expected_branches": 4,
            }
            units.append(plan_unit)
            matrix.append({
                key: plan_unit[key]
                for key in (
                    "candidate_id", "candidate_index", "operating_point_index",
                    "operating_point_role", "dataset_id", "negative_scale",
                    "exponential_coefficient", "taper_onset_tau", "effective_near_coefficient"
                )
            })
    if candidate_index != 15:
        raise SystemExit(f"{dataset_id}: expected 15 candidates, got {candidate_index}")
if len(units) != 135:
    raise SystemExit(f"expected 135 units, got {len(units)}")
plan = {
    "schema_version": 1,
    "experiment_id": master["experiment_id"],
    "source_experiment_id": master["source_experiment_id"],
    "predecessor_experiment_id": master["predecessor_experiment_id"],
    "scientific_status": master["scientific_status"],
    "runner_version": master["runner_version"],
    "unit_count": len(units),
    "expected_branch_count": len(units) * 4,
    "canonical_alpha": master["canonical_alpha"],
    "reference_distance": master["reference_distance"],
    "taper_onset_tau_grid": taus,
    "tuning_seeds": tuning_seeds,
    "held_out_seeds": master["held_out_seeds"],
    "source_run_spec_path": str(source_path),
    "source_run_spec_sha256": sha256(source_path),
    "master_config_path": str(master_path),
    "master_config_sha256": sha256(master_path),
    "runtime_adapter_path": str(runtime_adapter),
    "runtime_adapter_sha256": sha256(runtime_adapter),
    "units": units,
}
(work_root / "EXECUTION_PLAN.json").write_text(json.dumps(plan, indent=2) + "\n")
(work_root / "CANDIDATE_MATRIX.json").write_text(json.dumps({
    "schema_version": 1,
    "experiment_id": master["experiment_id"],
    "candidate_count_per_task": 15,
    "candidate_matrix": matrix,
}, indent=2) + "\n")
with (work_root / "UNITS.tsv").open("w") as handle:
    for unit in units:
        handle.write(f"{unit['unit_config']}\t{unit['work_dir']}\n")
print(json.dumps({
    "status": "READY",
    "units": len(units),
    "branches": len(units) * 4,
    "execution_plan": str(work_root / "EXECUTION_PLAN.json"),
}, sort_keys=True))
PY
}

run_unit() {
  local unit_config="$1"
  local unit_work_dir="$2"
  PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
  python3 "${RUNTIME_ADAPTER}" unit \
    --contract "${CONTRACT}" \
    --unit-config "${unit_config}" \
    --work-dir "${unit_work_dir}" \
    --resume
}

run_matrix() {
  local parallel_units
  parallel_units="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["parallel_units"])' "${MASTER_CONFIG}")"
  local -a pids=()
  local status=0
  local batch_count=0
  while IFS=$'\t' read -r unit_config unit_work_dir; do
    run_unit "${unit_config}" "${unit_work_dir}" &
    pids+=("$!")
    batch_count=$((batch_count + 1))
    if [[ "${batch_count}" -eq "${parallel_units}" ]]; then
      for pid in "${pids[@]}"; do
        if ! wait "${pid}"; then status=1; fi
      done
      pids=()
      batch_count=0
      if [[ "${status}" -ne 0 ]]; then return "${status}"; fi
    fi
  done < "${WORK_ROOT}/UNITS.tsv"
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then status=1; fi
  done
  return "${status}"
}

audit_matrix() {
  python3 - "${MASTER_CONFIG}" "${WORK_ROOT}" <<'PY'
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

master = json.loads(Path(sys.argv[1]).read_text())
work_root = Path(sys.argv[2]).resolve()
plan = json.loads((work_root / "EXECUTION_PLAN.json").read_text())
datasets = list(master["expected_datasets"])
tuning_seeds = list(master["tuning_seeds"])
late_steps = list(master["late_window_steps"])
expected_history_steps = list(range(50000, 1000001, 50000))

def read_json(path):
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise SystemExit(f"JSON root must be an object: {path}")
    return value

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def mean(values): return float(statistics.fmean(values))
def sample_std(values): return float(statistics.stdev(values)) if len(values) > 1 else 0.0
def population_std(values): return float(statistics.pstdev(values)) if len(values) > 1 else 0.0
def slope(xs, ys):
    xm, ym = mean(xs), mean(ys)
    denominator = sum((x - xm) ** 2 for x in xs)
    return 0.0 if denominator == 0.0 else float(sum((x-xm)*(y-ym) for x,y in zip(xs,ys))/denominator)

rows = []
for unit in plan["units"]:
    unit_work_dir = Path(unit["work_dir"])
    run_summary = read_json(unit_work_dir / "RUN_SUMMARY.json")
    if int(run_summary.get("branch_count", -1)) != 4 or int(run_summary.get("completed", -1)) != 4 or int(run_summary.get("failed", -1)) != 0:
        raise SystemExit(f"incomplete unit: {unit['unit_id']}")
    branch_dirs = sorted((unit_work_dir / "branches").glob("*"))
    if len(branch_dirs) != 4:
        raise SystemExit(f"expected four branches: {unit['unit_id']}")
    for branch_dir in branch_dirs:
        completed = read_json(branch_dir / "COMPLETED.json")
        if int(completed.get("return_code", -1)) != 0:
            raise SystemExit(f"nonzero branch return code: {branch_dir}")
        branch_config = read_json(branch_dir / "branch_config.json")
        control = branch_config["negative_control"]
        expected_control = {
            "method": "exponential",
            "negative_scale": float(unit["negative_scale"]),
            "canonical_alpha": 0.11,
            "reference_distance": 2.0,
            "exponential_coefficient": float(unit["exponential_coefficient"]),
            "taper_onset_tau": float(unit["taper_onset_tau"]),
        }
        for key, expected in expected_control.items():
            actual = control.get(key)
            if isinstance(expected, float):
                if not math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-12):
                    raise SystemExit(f"control mismatch {key}: {branch_dir}")
            elif actual != expected:
                raise SystemExit(f"control mismatch {key}: {branch_dir}")
        if branch_config.get("taper_formula") != master["taper_formula"]:
            raise SystemExit(f"taper formula mismatch: {branch_dir}")
        summaries = sorted((branch_dir / "trainer_output").glob("*_summary.json"))
        if len(summaries) != 1:
            raise SystemExit(f"expected one trainer summary: {branch_dir}")
        summary_path = summaries[0]
        summary = read_json(summary_path)
        for key, expected in {
            "dataset": unit["dataset_id"], "variant": "iqlv_exp_rank",
            "steps": 1000000, "score_type": "norm", "goal_conditioned": False,
        }.items():
            if summary.get(key) != expected:
                raise SystemExit(f"trainer metadata mismatch {key}: {summary_path}")
        seed = int(summary["seed"])
        if seed not in tuning_seeds:
            raise SystemExit(f"unexpected seed: {summary_path}")
        for key, expected in (("alpha", 0.11), ("tau", 0.5)):
            if not math.isclose(float(summary[key]), expected, rel_tol=0.0, abs_tol=1e-12):
                raise SystemExit(f"frozen trainer scalar changed {key}: {summary_path}")
        history = summary.get("history")
        if not isinstance(history, dict):
            raise SystemExit(f"missing history: {summary_path}")
        steps = [int(value) for value in history.get("steps", [])]
        if steps != expected_history_steps:
            raise SystemExit(f"evaluation cadence mismatch: {summary_path}")
        metric_keys = [key for key in history if key != "steps"]
        if len(metric_keys) != 1:
            raise SystemExit(f"expected one metric series: {summary_path}")
        scores = [float(value) for value in history[metric_keys[0]]]
        if len(scores) != len(steps) or any(not math.isfinite(value) for value in scores):
            raise SystemExit(f"non-finite/misaligned history: {summary_path}")
        by_step = dict(zip(steps, scores))
        late_scores = [by_step[step] for step in late_steps]
        best_index = max(range(len(scores)), key=scores.__getitem__)
        best_score = scores[best_index]
        late_mean = mean(late_scores)
        rows.append({
            "unit_id": unit["unit_id"], "candidate_id": unit["candidate_id"],
            "candidate_index": int(unit["candidate_index"]),
            "operating_point_index": int(unit["operating_point_index"]),
            "operating_point_role": unit["operating_point_role"],
            "branch_id": branch_config["branch_id"], "dataset_id": unit["dataset_id"],
            "seed": seed, "method": "exponential",
            "negative_scale": float(unit["negative_scale"]),
            "exponential_coefficient": float(unit["exponential_coefficient"]),
            "taper_onset_tau": float(unit["taper_onset_tau"]),
            "canonical_alpha": 0.11,
            "effective_near_coefficient": float(unit["effective_near_coefficient"]),
            "late_window_mean": late_mean,
            "late_window_std": population_std(late_scores),
            "late_window_min": min(late_scores), "late_window_max": max(late_scores),
            "late_window_scores": late_scores,
            "final_score": scores[-1], "best_score": best_score,
            "best_step": steps[best_index],
            "best_to_final_drop": best_score - scores[-1],
            "best_to_late_mean_drop": best_score - late_mean,
            "terminal_slope_per_100k_steps": slope([float(step) for step in late_steps], late_scores) * 100000.0,
            "completed_manifest": str((branch_dir / "COMPLETED.json").relative_to(work_root)),
            "trainer_summary": str(summary_path.relative_to(work_root)),
            "terminal_classification": "fixed_horizon_inconclusive",
        })
if len(rows) != 540:
    raise SystemExit(f"expected 540 branches, got {len(rows)}")

grouped = {}
for row in rows:
    grouped.setdefault((row["dataset_id"], row["candidate_id"]), []).append(row)
candidate_groups = []
for (dataset_id, candidate_id), members in grouped.items():
    members = sorted(members, key=lambda item: int(item["seed"]))
    seeds = [int(item["seed"]) for item in members]
    if seeds != tuning_seeds:
        raise SystemExit(f"seed coverage mismatch for {candidate_id}: {seeds}")
    late_means = [float(item["late_window_mean"]) for item in members]
    instantaneous = [
        sample_std([float(member["late_window_scores"][index]) for member in members])
        for index in range(len(late_steps))
    ]
    first = members[0]
    candidate_groups.append({
        "dataset_id": dataset_id, "candidate_id": candidate_id,
        "candidate_index": int(first["candidate_index"]),
        "operating_point_index": int(first["operating_point_index"]),
        "operating_point_role": first["operating_point_role"],
        "method": "exponential",
        "parameter_names": ["negative_scale", "exponential_coefficient", "taper_onset_tau"],
        "negative_scale": float(first["negative_scale"]),
        "exponential_coefficient": float(first["exponential_coefficient"]),
        "taper_onset_tau": float(first["taper_onset_tau"]),
        "canonical_alpha": 0.11,
        "effective_near_coefficient": float(first["effective_near_coefficient"]),
        "seeds": seeds, "seed_count": 4,
        "late_window_mean_across_seeds": mean(late_means),
        "late_window_std_across_seeds": sample_std(late_means),
        "late_window_min_across_seeds": min(late_means),
        "late_window_max_across_seeds": max(late_means),
        "late_temporal_std_mean_across_seeds": mean([float(item["late_window_std"]) for item in members]),
        "late_instantaneous_seed_std_mean": mean(instantaneous),
        "late_instantaneous_seed_std_by_step": dict(zip([str(step) for step in late_steps], instantaneous)),
        "final_mean_across_seeds": mean([float(item["final_score"]) for item in members]),
        "final_std_across_seeds": sample_std([float(item["final_score"]) for item in members]),
        "best_mean_across_seeds": mean([float(item["best_score"]) for item in members]),
        "best_to_final_drop_mean": mean([float(item["best_to_final_drop"]) for item in members]),
        "best_to_late_mean_drop_mean": mean([float(item["best_to_late_mean_drop"]) for item in members]),
        "terminal_slope_per_100k_mean": mean([float(item["terminal_slope_per_100k_steps"]) for item in members]),
        "terminal_classification": "fixed_horizon_inconclusive",
    })
candidate_groups.sort(key=lambda item: (datasets.index(item["dataset_id"]), int(item["candidate_index"])))
if len(candidate_groups) != 135:
    raise SystemExit(f"expected 135 groups, got {len(candidate_groups)}")

by_dataset = {dataset_id: [] for dataset_id in datasets}
for item in candidate_groups: by_dataset[item["dataset_id"]].append(item)
selections = []
tau_curves = []
for dataset_id in datasets:
    candidates = by_dataset[dataset_id]
    if len(candidates) != 15:
        raise SystemExit(f"expected 15 candidates for {dataset_id}")
    ranked = sorted(candidates, key=lambda row: (
        -float(row["late_window_mean_across_seeds"]),
        -float(row["late_window_min_across_seeds"]),
        float(row["best_to_late_mean_drop_mean"]),
        float(row["taper_onset_tau"]),
        float(row["negative_scale"]),
        float(row["exponential_coefficient"]),
        int(row["candidate_index"]),
    ))
    winner = dict(ranked[0])
    winner["selection_rank"] = 1
    winner["candidate_count"] = 15
    winner["selection_rule"] = list(master["selection_rule"])
    selections.append(winner)
    for op_index in (1, 2, 3):
        curve = [item for item in candidates if int(item["operating_point_index"]) == op_index]
        curve.sort(key=lambda item: float(item["taper_onset_tau"]))
        if [float(item["taper_onset_tau"]) for item in curve] != [0.0, 0.125, 0.25, 0.375, 0.5]:
            raise SystemExit(f"tau curve incomplete: {dataset_id} op{op_index}")
        tau_curves.append({
            "dataset_id": dataset_id,
            "operating_point_index": op_index,
            "operating_point_role": curve[0]["operating_point_role"],
            "negative_scale": curve[0]["negative_scale"],
            "exponential_coefficient": curve[0]["exponential_coefficient"],
            "tau_points": curve,
        })

event_separation = {
    "task_performance_collapse": "not_classified_without_registered_threshold",
    "support_or_variance_boundary_event": "not_available_in_unchanged_canonical_trainer_summary",
    "nan_inf_numerical_failure": "not_observed_in_zero_exit_branches_and_finite_evaluation_histories",
}
payload = {
    "schema_version": 1, "experiment_id": master["experiment_id"],
    "source_experiment_id": master["source_experiment_id"],
    "predecessor_experiment_id": master["predecessor_experiment_id"],
    "scientific_status": master["scientific_status"], "runner_version": master["runner_version"],
    "status": "PASS", "expected_branch_count": 540, "audited_branch_count": len(rows),
    "datasets": datasets, "tuning_seeds": tuning_seeds,
    "held_out_seeds_untouched": master["held_out_seeds"],
    "methods": ["exponential"], "canonical_alpha": 0.11, "reference_distance": 2.0,
    "taper_coordinate": master["taper_coordinate"], "taper_formula": master["taper_formula"],
    "taper_onset_tau_grid": master["taper_onset_tau_grid"],
    "trainer_expectile_tau_frozen": 0.5,
    "primary_metric": master["primary_selection_metric"], "late_window_steps": late_steps,
    "selection_scope": "per_dataset_frozen_sc_pair_and_taper_onset_tau",
    "candidate_count_per_task": 15, "candidate_groups": candidate_groups,
    "taskwise_selections": selections, "tau_curves": tau_curves,
    "cross_method_ranking_allowed": False, "formal_d4rl9_table_population_allowed": False,
    "steady_state_claim_allowed": False, "fixed_horizon_is_not_convergence": True,
    "confirmation_required_before_method_ranking": True, "branches": rows,
    "execution_plan_binding": {"path": "EXECUTION_PLAN.json", "sha256": sha256(work_root / "EXECUTION_PLAN.json")},
    "candidate_matrix_binding": {"path": "CANDIDATE_MATRIX.json", "sha256": sha256(work_root / "CANDIDATE_MATRIX.json")},
    "runtime_adapter_binding": {"path": str(Path(plan["runtime_adapter_path"]).relative_to(work_root)), "sha256": plan["runtime_adapter_sha256"]},
    "event_separation_summary": event_separation,
}
(work_root / "TERMINAL_AUDIT.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
(work_root / "TASKWISE_SELECTION.json").write_text(json.dumps({
    "schema_version": 1, "experiment_id": master["experiment_id"],
    "scientific_status": master["scientific_status"],
    "primary_metric": master["primary_selection_metric"],
    "tuning_seeds": tuning_seeds, "held_out_seeds_untouched": master["held_out_seeds"],
    "selection_scope": "per_dataset_frozen_sc_pair_and_taper_onset_tau",
    "candidate_count_per_task": 15, "taskwise_selections": selections,
    "confirmation_required_before_method_ranking": True,
}, indent=2, sort_keys=True) + "\n")
(work_root / "TASKWISE_TAU_CURVES.json").write_text(json.dumps({
    "schema_version": 1, "experiment_id": master["experiment_id"],
    "taper_onset_tau_grid": master["taper_onset_tau_grid"],
    "tau_curves": tau_curves,
}, indent=2, sort_keys=True) + "\n")
print(json.dumps({
    "status": "PASS", "audited_branches": len(rows), "candidate_groups": len(candidate_groups),
    "terminal_audit": str(work_root / "TERMINAL_AUDIT.json"),
    "taskwise_selection": str(work_root / "TASKWISE_SELECTION.json"),
    "tau_curves": str(work_root / "TASKWISE_TAU_CURVES.json"),
}, sort_keys=True))
PY
}

validate_master
if [[ "${1:-}" == "--validate-only" ]]; then
  exit 0
fi

for required in "${SOURCE_RUN_SPEC}" "${MASTER_CONFIG}"; do
  if [[ ! -f "${required}" ]]; then
    echo "missing required file: ${required}" >&2
    exit 2
  fi
done
mkdir -p "${WORK_ROOT}"
write_runtime_adapter
PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" python3 "${RUNTIME_ADAPTER}" self-test
prepare_matrix
if [[ "${1:-}" == "--prepare-only" ]]; then
  echo "Preparation-only validation completed."
  exit 0
fi
if [[ ! -f "${CONTRACT}" ]]; then
  echo "missing required contract: ${CONTRACT}" >&2
  exit 2
fi

echo "=== Running DRPO taper-onset tau matrix ==="
run_matrix
echo "=== Auditing DRPO taper-onset tau matrix ==="
audit_matrix

echo "DRPO taper-onset tau development sweep completed."
echo "Terminal audit: ${WORK_ROOT}/TERMINAL_AUDIT.json"
echo "Task-wise selection: ${WORK_ROOT}/TASKWISE_SELECTION.json"
echo "Tau curves: ${WORK_ROOT}/TASKWISE_TAU_CURVES.json"
