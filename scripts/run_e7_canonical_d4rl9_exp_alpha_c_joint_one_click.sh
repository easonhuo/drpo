#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

CONTRACT="${CONTRACT:-/root/d4rl2/configs/e7_canonical_contract_9task.json}"
SOURCE_RUN_SPEC="${SOURCE_RUN_SPEC:-/root/d4rl2/configs/e7_canonical_9task_full_grid_run_spec_v1.json}"
MASTER_CONFIG="${MASTER_CONFIG:-configs/e7_canonical_d4rl9_exp_alpha_c_joint_v1.json}"
WORK_ROOT="${WORK_ROOT:-outputs/e7/d4rl9_exp_alpha_c_joint_run_001}"

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
checks = {
    "experiment_id": "EXT-H-E7-D4RL9-EXP-ALPHA-C-JOINT-TUNE-01",
    "source_experiment_id": "EXT-H-E7-BENCH-01",
    "predecessor_experiment_id": "EXT-H-E7-D4RL9-EXP-3WAVE-TUNE-01",
    "run_kind": "pilot",
    "scientific_status": "d4rl9_taskwise_exponential_alpha_c_joint_tuning_pilot_only",
    "runner_version": "1.0.0-d4rl9-exp-alpha-c-joint",
    "candidate_count_per_task": 15,
    "expected_unit_count": 135,
    "expected_total_branches": 540,
    "fixed_max_workers": 60,
    "parallel_units": 15,
    "workers_per_unit": 4,
    "primary_selection_metric": "late_window_mean_750k_to_1m",
    "rounding_significant_digits": 12,
}
for key, expected in checks.items():
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
    actual = float(raw.get(key))
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
        raise SystemExit(f"{key} must remain {expected}")
if set(raw.get("taskwise_search", {})) != set(expected_datasets):
    raise SystemExit("taskwise_search dataset coverage changed")
if list(raw["taskwise_search"]) != expected_datasets:
    raise SystemExit("taskwise_search dataset order changed")

digits = int(raw["rounding_significant_digits"])
def rounded(value):
    return float(format(float(value), f".{digits}g"))

def pair_key(scale, coefficient):
    return (rounded(scale), rounded(coefficient))

def expand(dataset_id, spec):
    mode = str(spec.get("mode"))
    c_star = float(spec.get("c_star"))
    if not math.isfinite(c_star) or c_star <= 0.0:
        raise SystemExit(f"{dataset_id}: c_star must be finite and positive")
    pairs = []
    if mode == "alpha_primary_local_c":
        scales = [float(value) for value in spec.get("scale_grid", [])]
        multipliers = [float(value) for value in spec.get("c_multipliers", [])]
        if len(scales) != 4 or len(set(scales)) != 4:
            raise SystemExit(f"{dataset_id}: scale_grid must contain four unique values")
        if len(multipliers) != 3 or len(set(multipliers)) != 3:
            raise SystemExit(f"{dataset_id}: c_multipliers must contain three unique values")
        if any(value <= 0.0 or not math.isfinite(value) for value in scales):
            raise SystemExit(f"{dataset_id}: scale_grid must be finite and positive")
        if any(value <= 0.0 or not math.isfinite(value) for value in multipliers):
            raise SystemExit(f"{dataset_id}: c_multipliers must be finite and positive")
        positive_only_c = float(spec.get("positive_only_c"))
        pairs.append(pair_key(0.0, positive_only_c))
        for scale in scales:
            for multiplier in multipliers:
                pairs.append(pair_key(scale, c_star * multiplier))
        extras = list(spec.get("extra_pairs", []))
        if len(extras) != 2:
            raise SystemExit(f"{dataset_id}: extra_pairs must contain exactly two points")
        for item in extras:
            scale = float(item["negative_scale"])
            coefficient = (
                float(item["exponential_coefficient"])
                if "exponential_coefficient" in item
                else c_star * float(item["c_multiplier"])
            )
            pairs.append(pair_key(scale, coefficient))
    elif mode == "c_boundary_joint_scale":
        explicit = list(spec.get("explicit_pairs", []))
        for item in explicit:
            pairs.append(
                pair_key(
                    float(item["negative_scale"]),
                    float(item["exponential_coefficient"]),
                )
            )
    else:
        raise SystemExit(f"{dataset_id}: unsupported mode={mode!r}")
    if len(pairs) != 15:
        raise SystemExit(f"{dataset_id}: expected 15 candidates, got {len(pairs)}")
    if len(set(pairs)) != len(pairs):
        raise SystemExit(f"{dataset_id}: duplicate (negative_scale, c) candidate")
    for scale, coefficient in pairs:
        if scale < 0.0 or coefficient < 0.0:
            raise SystemExit(f"{dataset_id}: scale and c must be nonnegative")
        if not math.isfinite(scale) or not math.isfinite(coefficient):
            raise SystemExit(f"{dataset_id}: scale and c must be finite")
    if pair_key(0.0, c_star) not in pairs:
        raise SystemExit(f"{dataset_id}: exact zero-negative anchor is missing")
    if pair_key(1.0, c_star) not in pairs:
        raise SystemExit(f"{dataset_id}: predecessor winner anchor (s=1,c=c_star) is missing")
    return pairs

matrix = {}
for dataset_id in expected_datasets:
    matrix[dataset_id] = expand(dataset_id, raw["taskwise_search"][dataset_id])
walker_pairs = set(matrix["walker2d-medium-replay-v2"])
if pair_key(1.0, 0.0) not in walker_pairs:
    raise SystemExit("walker2d-medium-replay-v2 must include the c=0 global-shape anchor")
if sum(len(values) for values in matrix.values()) != 135:
    raise SystemExit("expanded candidate matrix must contain 135 task-pair units")
print(json.dumps({
    "status": "PASS",
    "experiment_id": raw["experiment_id"],
    "tasks": len(matrix),
    "candidates_per_task": 15,
    "units": 135,
    "branches": 540,
}, sort_keys=True))
PY
}

if [[ "${1:-}" == "--validate-only" ]]; then
  validate_master
  exit 0
fi

for required in "${CONTRACT}" "${SOURCE_RUN_SPEC}" "${MASTER_CONFIG}"; do
  if [[ ! -f "${required}" ]]; then
    echo "missing required file: ${required}" >&2
    exit 2
  fi
done

validate_master
mkdir -p "${WORK_ROOT}"

prepare_matrix() {
  python3 - "${MASTER_CONFIG}" "${SOURCE_RUN_SPEC}" "${WORK_ROOT}" <<'PY'
import copy
import hashlib
import json
import math
import sys
from pathlib import Path

master_path = Path(sys.argv[1]).resolve()
source_run_spec_path = Path(sys.argv[2]).resolve()
work_root = Path(sys.argv[3]).resolve()
master = json.loads(master_path.read_text())
source = json.loads(source_run_spec_path.read_text())
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
passthrough = source.get("passthrough_variants", [])
if passthrough not in ([], [{"id": "original_exp_rank_mr", "template_values": {}}]):
    ids = [str(item.get("id")) for item in passthrough]
    if ids not in ([], ["original_exp_rank_mr"]):
        raise SystemExit(f"unexpected passthrough variants: {ids}")

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

digits = int(master["rounding_significant_digits"])
def rounded(value):
    return float(format(float(value), f".{digits}g"))

def pair_key(scale, coefficient):
    return (rounded(scale), rounded(coefficient))

def expand(dataset_id, spec):
    c_star = float(spec["c_star"])
    pairs = []
    if spec["mode"] == "alpha_primary_local_c":
        pairs.append(pair_key(0.0, float(spec["positive_only_c"])))
        for scale in spec["scale_grid"]:
            for multiplier in spec["c_multipliers"]:
                pairs.append(pair_key(scale, c_star * float(multiplier)))
        for item in spec["extra_pairs"]:
            coefficient = (
                float(item["exponential_coefficient"])
                if "exponential_coefficient" in item
                else c_star * float(item["c_multiplier"])
            )
            pairs.append(pair_key(item["negative_scale"], coefficient))
    elif spec["mode"] == "c_boundary_joint_scale":
        pairs = [
            pair_key(item["negative_scale"], item["exponential_coefficient"])
            for item in spec["explicit_pairs"]
        ]
    else:
        raise SystemExit(f"unsupported search mode for {dataset_id}")
    if len(pairs) != 15 or len(set(pairs)) != 15:
        raise SystemExit(f"{dataset_id}: candidate expansion is not 15 unique pairs")
    return pairs

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def label(value):
    return format(float(value), ".12g").replace("-", "m").replace(".", "p")

generated_dir = work_root / "generated"
generated_dir.mkdir(parents=True, exist_ok=True)
dataset_by_id = {str(item["id"]): item for item in source["datasets"]}
units = []
matrix = []
for dataset_id in datasets:
    spec = master["taskwise_search"][dataset_id]
    pairs = expand(dataset_id, spec)
    for candidate_index, (negative_scale, coefficient) in enumerate(pairs, start=1):
        candidate_id = f"{dataset_id}__cand{candidate_index:02d}__s{label(negative_scale)}__c{label(coefficient)}"
        unit_generated = generated_dir / candidate_id
        unit_generated.mkdir(parents=True, exist_ok=True)
        unit_work_dir = work_root / "units" / candidate_id
        run_spec = copy.deepcopy(source)
        run_spec["datasets"] = [copy.deepcopy(dataset_by_id[dataset_id])]
        run_spec["seeds"] = tuning_seeds
        run_spec["passthrough_variants"] = []
        grid = {
            "experiment_id": master["experiment_id"],
            "run_kind": "pilot",
            "canonical_alpha": master["canonical_alpha"],
            "reference_distance": master["reference_distance"],
            "coefficients": {
                "reciprocal_linear": 0.5,
                "reciprocal_quadratic": 0.5,
                "exponential": coefficient,
            },
            "anchors": {},
            "negative_scale_grid": {"exponential": [negative_scale]},
            "dataset_id": dataset_id,
            "candidate_index": candidate_index,
            "candidate_id": candidate_id,
            "parameter_names": ["negative_scale", "exponential_coefficient"],
            "negative_scale": negative_scale,
            "exponential_coefficient": coefficient,
            "effective_near_coefficient": master["canonical_alpha"] * negative_scale,
        }
        run_spec_path = unit_generated / "run_spec.json"
        grid_path = unit_generated / "grid.json"
        run_spec_path.write_text(json.dumps(run_spec, indent=2) + "\n")
        grid_path.write_text(json.dumps(grid, indent=2) + "\n")
        unit = {
            "unit_id": candidate_id,
            "candidate_id": candidate_id,
            "candidate_index": candidate_index,
            "dataset_id": dataset_id,
            "negative_scale": negative_scale,
            "exponential_coefficient": coefficient,
            "effective_near_coefficient": master["canonical_alpha"] * negative_scale,
            "run_spec": str(run_spec_path),
            "grid": str(grid_path),
            "work_dir": str(unit_work_dir),
            "expected_branches": len(tuning_seeds),
        }
        units.append(unit)
        matrix.append({
            key: unit[key]
            for key in (
                "candidate_id",
                "candidate_index",
                "dataset_id",
                "negative_scale",
                "exponential_coefficient",
                "effective_near_coefficient",
            )
        })
if len(units) != 135:
    raise SystemExit(f"expected 135 units, got {len(units)}")
payload = {
    "schema_version": 1,
    "experiment_id": master["experiment_id"],
    "source_experiment_id": master["source_experiment_id"],
    "predecessor_experiment_id": master["predecessor_experiment_id"],
    "scientific_status": master["scientific_status"],
    "runner_version": master["runner_version"],
    "candidate_count_per_task": master["candidate_count_per_task"],
    "unit_count": len(units),
    "expected_branch_count": len(units) * len(tuning_seeds),
    "canonical_alpha": master["canonical_alpha"],
    "reference_distance": master["reference_distance"],
    "tuning_seeds": tuning_seeds,
    "held_out_seeds": master["held_out_seeds"],
    "source_run_spec_path": str(source_run_spec_path),
    "source_run_spec_sha256": sha256(source_run_spec_path),
    "master_config_path": str(master_path),
    "master_config_sha256": sha256(master_path),
    "units": units,
}
(work_root / "EXECUTION_PLAN.json").write_text(json.dumps(payload, indent=2) + "\n")
(work_root / "CANDIDATE_MATRIX.json").write_text(json.dumps({
    "schema_version": 1,
    "experiment_id": master["experiment_id"],
    "candidate_count_per_task": 15,
    "candidate_matrix": matrix,
}, indent=2) + "\n")
with (work_root / "UNITS.tsv").open("w") as handle:
    for unit in units:
        handle.write(f"{unit['run_spec']}\t{unit['grid']}\t{unit['work_dir']}\n")
print(json.dumps({
    "status": "READY",
    "units": len(units),
    "branches": len(units) * len(tuning_seeds),
    "execution_plan": str(work_root / "EXECUTION_PLAN.json"),
    "candidate_matrix": str(work_root / "CANDIDATE_MATRIX.json"),
}, sort_keys=True))
PY
}

run_unit() {
  local run_spec="$1"
  local grid="$2"
  local unit_work_dir="$3"
  local experiment_id scientific_status runner_version workers_per_unit
  experiment_id="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["experiment_id"])' "${MASTER_CONFIG}")"
  scientific_status="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["scientific_status"])' "${MASTER_CONFIG}")"
  runner_version="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["runner_version"])' "${MASTER_CONFIG}")"
  workers_per_unit="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["workers_per_unit"])' "${MASTER_CONFIG}")"
  DRPO_E7_EXPERIMENT_ID="${experiment_id}" \
  DRPO_E7_SCIENTIFIC_STATUS="${scientific_status}" \
  DRPO_E7_RUNNER_VERSION="${runner_version}" \
  PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
  python3 -c '
import os
import sys
from drpo import e7_canonical_sweep as runner
runner.EXPERIMENT_ID = os.environ["DRPO_E7_EXPERIMENT_ID"]
runner.SCIENTIFIC_STATUS = os.environ["DRPO_E7_SCIENTIFIC_STATUS"]
runner.RUNNER_VERSION = os.environ["DRPO_E7_RUNNER_VERSION"]
raise SystemExit(runner.main(sys.argv[1:]))
' run \
    --contract "${CONTRACT}" \
    --run-spec "${run_spec}" \
    --grid "${grid}" \
    --work-dir "${unit_work_dir}" \
    --max-workers "${workers_per_unit}" \
    --resume
}

run_matrix() {
  local parallel_units
  parallel_units="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["parallel_units"])' "${MASTER_CONFIG}")"
  local -a pids=()
  local status=0
  local batch_count=0
  while IFS=$'\t' read -r run_spec grid unit_work_dir; do
    run_unit "${run_spec}" "${grid}" "${unit_work_dir}" &
    pids+=("$!")
    batch_count=$((batch_count + 1))
    if [[ "${batch_count}" -eq "${parallel_units}" ]]; then
      for pid in "${pids[@]}"; do
        if ! wait "${pid}"; then
          status=1
        fi
      done
      pids=()
      batch_count=0
      if [[ "${status}" -ne 0 ]]; then
        return "${status}"
      fi
    fi
  done < "${WORK_ROOT}/UNITS.tsv"
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      status=1
    fi
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

def mean(values):
    return float(statistics.fmean(values))

def sample_std(values):
    return float(statistics.stdev(values)) if len(values) > 1 else 0.0

def population_std(values):
    return float(statistics.pstdev(values)) if len(values) > 1 else 0.0

def slope(xs, ys):
    x_mean = mean(xs)
    y_mean = mean(ys)
    denominator = sum((x - x_mean) ** 2 for x in xs)
    if denominator == 0.0:
        return 0.0
    return float(sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / denominator)

rows = []
for unit in plan["units"]:
    unit_work_dir = Path(unit["work_dir"])
    run_summary_path = unit_work_dir / "RUN_SUMMARY.json"
    if not run_summary_path.is_file():
        raise SystemExit(f"missing unit RUN_SUMMARY: {run_summary_path}")
    run_summary = read_json(run_summary_path)
    if (
        int(run_summary.get("branch_count", -1)) != 4
        or int(run_summary.get("completed", -1)) != 4
        or int(run_summary.get("failed", -1)) != 0
    ):
        raise SystemExit(f"incomplete unit: {unit['unit_id']}")
    branch_dirs = sorted((unit_work_dir / "branches").glob("*"))
    if len(branch_dirs) != 4:
        raise SystemExit(f"expected four branch directories for {unit['unit_id']}")
    negative_scale = float(unit["negative_scale"])
    coefficient = float(unit["exponential_coefficient"])
    for branch_dir in branch_dirs:
        completed = read_json(branch_dir / "COMPLETED.json")
        if int(completed.get("return_code", -1)) != 0:
            raise SystemExit(f"nonzero branch return code: {branch_dir}")
        branch_config = read_json(branch_dir / "branch_config.json")
        control = branch_config.get("negative_control")
        expected_control = {
            "method": "exponential",
            "negative_scale": negative_scale,
            "canonical_alpha": 0.11,
            "reference_distance": 2.0,
            "exponential_coefficient": coefficient,
        }
        for key, expected in expected_control.items():
            actual = control.get(key)
            if isinstance(expected, float):
                if not math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-12):
                    raise SystemExit(f"control mismatch {key}: {branch_dir}")
            elif actual != expected:
                raise SystemExit(f"control mismatch {key}: {branch_dir}")
        summaries = sorted((branch_dir / "trainer_output").glob("*_summary.json"))
        if len(summaries) != 1:
            raise SystemExit(f"expected one trainer summary: {branch_dir}")
        summary_path = summaries[0]
        summary = read_json(summary_path)
        expected_metadata = {
            "dataset": unit["dataset_id"],
            "variant": "iqlv_exp_rank",
            "steps": 1000000,
            "score_type": "norm",
            "goal_conditioned": False,
        }
        for key, expected in expected_metadata.items():
            if summary.get(key) != expected:
                raise SystemExit(f"trainer metadata mismatch {key}: {summary_path}")
        seed = int(summary["seed"])
        if seed not in tuning_seeds:
            raise SystemExit(f"unexpected seed {seed}: {summary_path}")
        for key, expected in (("alpha", 0.11), ("tau", 0.5)):
            if not math.isclose(float(summary[key]), expected, rel_tol=0.0, abs_tol=1e-12):
                raise SystemExit(f"trainer scalar mismatch {key}: {summary_path}")
        history = summary.get("history")
        if not isinstance(history, dict):
            raise SystemExit(f"trainer summary has no history: {summary_path}")
        steps = [int(value) for value in history.get("steps", [])]
        if steps != expected_history_steps:
            raise SystemExit(f"evaluation cadence mismatch: {summary_path}")
        metric_keys = [key for key in history if key != "steps"]
        if len(metric_keys) != 1:
            raise SystemExit(f"expected one metric series: {summary_path}")
        scores = [float(value) for value in history[metric_keys[0]]]
        if len(scores) != len(steps) or any(not math.isfinite(value) for value in scores):
            raise SystemExit(f"non-finite or misaligned history: {summary_path}")
        score_by_step = dict(zip(steps, scores))
        late_scores = [score_by_step[step] for step in late_steps]
        best_index = max(range(len(scores)), key=scores.__getitem__)
        best_score = scores[best_index]
        late_mean = mean(late_scores)
        rows.append({
            "unit_id": unit["unit_id"],
            "candidate_id": unit["candidate_id"],
            "candidate_index": int(unit["candidate_index"]),
            "branch_id": branch_config["branch_id"],
            "dataset_id": unit["dataset_id"],
            "seed": seed,
            "method": "exponential",
            "negative_scale": negative_scale,
            "exponential_coefficient": coefficient,
            "canonical_alpha": 0.11,
            "effective_near_coefficient": 0.11 * negative_scale,
            "late_window_mean": late_mean,
            "late_window_std": population_std(late_scores),
            "late_window_min": min(late_scores),
            "late_window_max": max(late_scores),
            "final_score": scores[-1],
            "best_score": best_score,
            "best_step": steps[best_index],
            "best_to_final_drop": best_score - scores[-1],
            "best_to_late_mean_drop": best_score - late_mean,
            "terminal_slope_per_100k_steps": slope(
                [float(step) for step in late_steps], late_scores
            ) * 100000.0,
            "completed_manifest": str((branch_dir / "COMPLETED.json").relative_to(work_root)),
            "trainer_summary": str(summary_path.relative_to(work_root)),
            "terminal_classification": "fixed_horizon_inconclusive",
        })
if len(rows) != 540:
    raise SystemExit(f"expected 540 audited branches, got {len(rows)}")

grouped = {}
for row in rows:
    key = (row["dataset_id"], row["candidate_id"])
    grouped.setdefault(key, []).append(row)
candidate_groups = []
for (dataset_id, candidate_id), members in grouped.items():
    members = sorted(members, key=lambda item: int(item["seed"]))
    seeds = [int(item["seed"]) for item in members]
    if seeds != tuning_seeds:
        raise SystemExit(f"seed coverage mismatch for {candidate_id}: {seeds}")
    late = [float(item["late_window_mean"]) for item in members]
    first = members[0]
    candidate_groups.append({
        "dataset_id": dataset_id,
        "candidate_id": candidate_id,
        "candidate_index": int(first["candidate_index"]),
        "method": "exponential",
        "parameter_names": ["negative_scale", "exponential_coefficient"],
        "negative_scale": float(first["negative_scale"]),
        "exponential_coefficient": float(first["exponential_coefficient"]),
        "canonical_alpha": 0.11,
        "effective_near_coefficient": float(first["effective_near_coefficient"]),
        "seeds": seeds,
        "seed_count": len(seeds),
        "late_window_mean_across_seeds": mean(late),
        "late_window_std_across_seeds": sample_std(late),
        "late_window_min_across_seeds": min(late),
        "late_window_max_across_seeds": max(late),
        "final_mean_across_seeds": mean([float(item["final_score"]) for item in members]),
        "best_mean_across_seeds": mean([float(item["best_score"]) for item in members]),
        "best_to_final_drop_mean": mean([float(item["best_to_final_drop"]) for item in members]),
        "best_to_late_mean_drop_mean": mean([float(item["best_to_late_mean_drop"]) for item in members]),
        "terminal_slope_per_100k_mean": mean([float(item["terminal_slope_per_100k_steps"]) for item in members]),
        "terminal_classification": "fixed_horizon_inconclusive",
    })
candidate_groups.sort(
    key=lambda item: (
        datasets.index(str(item["dataset_id"])),
        int(item["candidate_index"]),
    )
)
if len(candidate_groups) != 135:
    raise SystemExit(f"expected 135 candidate groups, got {len(candidate_groups)}")

by_dataset = {dataset_id: [] for dataset_id in datasets}
for item in candidate_groups:
    by_dataset[str(item["dataset_id"])].append(item)
selections = []
for dataset_id in datasets:
    candidates = by_dataset[dataset_id]
    if len(candidates) != 15:
        raise SystemExit(f"expected 15 candidates for {dataset_id}, got {len(candidates)}")
    ranked = sorted(
        candidates,
        key=lambda row: (
            -float(row["late_window_mean_across_seeds"]),
            -float(row["late_window_min_across_seeds"]),
            float(row["best_to_late_mean_drop_mean"]),
            float(row["negative_scale"]),
            float(row["exponential_coefficient"]),
            int(row["candidate_index"]),
        ),
    )
    winner = dict(ranked[0])
    winner["selection_rank"] = 1
    winner["candidate_count"] = len(candidates)
    winner["selection_rule"] = list(master["selection_rule"])
    selections.append(winner)

event_separation = {
    "task_performance_collapse": "not_classified_without_registered_threshold",
    "support_or_variance_boundary_event": "not_available_in_unchanged_canonical_trainer_summary",
    "nan_inf_numerical_failure": "not_observed_in_zero_exit_branches_and_finite_evaluation_histories",
}
payload = {
    "schema_version": 1,
    "experiment_id": master["experiment_id"],
    "source_experiment_id": master["source_experiment_id"],
    "predecessor_experiment_id": master["predecessor_experiment_id"],
    "scientific_status": master["scientific_status"],
    "runner_version": master["runner_version"],
    "status": "PASS",
    "expected_branch_count": 540,
    "audited_branch_count": len(rows),
    "datasets": datasets,
    "tuning_seeds": tuning_seeds,
    "held_out_seeds_untouched": master["held_out_seeds"],
    "methods": ["exponential"],
    "canonical_alpha": 0.11,
    "reference_distance": 2.0,
    "primary_metric": master["primary_selection_metric"],
    "late_window_steps": late_steps,
    "selection_scope": "per_dataset_exponential_negative_scale_and_c_pair",
    "candidate_count_per_task": 15,
    "candidate_groups": candidate_groups,
    "taskwise_selections": selections,
    "positive_only_equivalent_anchor_included": True,
    "walker2d_replay_c_zero_global_shape_anchor_included": True,
    "cross_method_ranking_allowed": False,
    "formal_d4rl9_table_population_allowed": False,
    "steady_state_claim_allowed": False,
    "fixed_horizon_is_not_convergence": True,
    "confirmation_required_before_method_ranking": True,
    "branches": rows,
    "execution_plan_binding": {
        "path": "EXECUTION_PLAN.json",
        "sha256": sha256(work_root / "EXECUTION_PLAN.json"),
    },
    "candidate_matrix_binding": {
        "path": "CANDIDATE_MATRIX.json",
        "sha256": sha256(work_root / "CANDIDATE_MATRIX.json"),
    },
    "event_separation_summary": event_separation,
}
(work_root / "TERMINAL_AUDIT.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n"
)
selection_payload = {
    "schema_version": 1,
    "experiment_id": master["experiment_id"],
    "scientific_status": master["scientific_status"],
    "primary_metric": master["primary_selection_metric"],
    "tuning_seeds": tuning_seeds,
    "held_out_seeds_untouched": master["held_out_seeds"],
    "selection_scope": "per_dataset_exponential_negative_scale_and_c_pair",
    "candidate_count_per_task": 15,
    "taskwise_selections": selections,
    "confirmation_required_before_method_ranking": True,
}
(work_root / "TASKWISE_SELECTION.json").write_text(
    json.dumps(selection_payload, indent=2, sort_keys=True) + "\n"
)
print(json.dumps({
    "status": "PASS",
    "audited_branches": len(rows),
    "candidate_groups": len(candidate_groups),
    "terminal_audit": str(work_root / "TERMINAL_AUDIT.json"),
    "taskwise_selection": str(work_root / "TASKWISE_SELECTION.json"),
}, sort_keys=True))
PY
}

echo "=== Preparing Exp alpha-c joint matrix ==="
prepare_matrix
if [[ "${1:-}" == "--prepare-only" ]]; then
  echo "Preparation-only validation completed."
  exit 0
fi
echo "=== Running Exp alpha-c joint matrix ==="
run_matrix
echo "=== Auditing Exp alpha-c joint matrix ==="
audit_matrix

echo "Exp alpha-c joint development sweep completed."
echo "Terminal audit: ${WORK_ROOT}/TERMINAL_AUDIT.json"
echo "Task-wise selection: ${WORK_ROOT}/TASKWISE_SELECTION.json"
