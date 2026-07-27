#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

CONTRACT="${CONTRACT:-/root/d4rl2/configs/e7_canonical_contract_9task.json}"
SOURCE_RUN_SPEC="${SOURCE_RUN_SPEC:-/root/d4rl2/configs/e7_canonical_9task_full_grid_run_spec_v1.json}"
MASTER_CONFIG="${MASTER_CONFIG:-configs/e7_canonical_d4rl9_exp_three_wave_v1.json}"
WORK_ROOT="${WORK_ROOT:-outputs/e7/d4rl9_exp_three_wave_run_001}"

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
    "experiment_id": "EXT-H-E7-D4RL9-EXP-3WAVE-TUNE-01",
    "source_experiment_id": "EXT-H-E7-BENCH-01",
    "run_kind": "pilot",
    "scientific_status": "d4rl9_taskwise_exponential_three_wave_tuning_pilot_only",
    "runner_version": "1.0.0-d4rl9-exp-three-wave",
    "wave_count": 3,
    "candidates_per_task_per_wave": 5,
    "branches_per_wave": 180,
    "expected_total_branches": 540,
    "fixed_max_workers": 60,
    "parallel_units": 15,
    "workers_per_unit": 4,
    "primary_selection_metric": "late_window_mean_750k_to_1m",
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
for key, expected in (
    ("canonical_alpha", 0.11),
    ("negative_scale", 1.0),
    ("reference_distance", 2.0),
):
    actual = float(raw.get(key))
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
        raise SystemExit(f"{key} must remain {expected}")
taskwise = raw.get("taskwise_wave1_c_grid")
if not isinstance(taskwise, dict) or list(taskwise) != expected_datasets:
    raise SystemExit("taskwise_wave1_c_grid dataset order changed")
for dataset_id, values in taskwise.items():
    parsed = [float(value) for value in values]
    if len(parsed) != 5 or len(set(parsed)) != 5:
        raise SystemExit(f"{dataset_id} requires five unique Wave-1 c values")
    if any(value <= 0.0 or not math.isfinite(value) for value in parsed):
        raise SystemExit(f"{dataset_id} has invalid Wave-1 c values")
rules = raw.get("adaptive_rules", {})
for key in ("wave2_multiplier_pool", "wave3_multiplier_pool", "fallback_multiplier_pool"):
    values = [float(value) for value in rules.get(key, [])]
    if len(values) < 5 or any(value <= 0.0 or not math.isfinite(value) for value in values):
        raise SystemExit(f"adaptive_rules.{key} is invalid")
if int(rules.get("significant_digits", -1)) != 12:
    raise SystemExit("adaptive_rules.significant_digits must remain 12")
print(json.dumps({
    "status": "PASS",
    "experiment_id": raw["experiment_id"],
    "wave_count": raw["wave_count"],
    "expected_total_branches": raw["expected_total_branches"],
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

prepare_wave() {
  local wave="$1"
  python3 - "${MASTER_CONFIG}" "${SOURCE_RUN_SPEC}" "${WORK_ROOT}" "${wave}" <<'PY'
import copy
import hashlib
import json
import math
import sys
from pathlib import Path

master_path = Path(sys.argv[1]).resolve()
source_run_spec_path = Path(sys.argv[2]).resolve()
work_root = Path(sys.argv[3]).resolve()
wave = int(sys.argv[4])

master = json.loads(master_path.read_text())
source = json.loads(source_run_spec_path.read_text())
if source.get("experiment_id") != master["source_experiment_id"]:
    raise SystemExit("source run-spec experiment_id changed")
if source.get("run_kind") not in {"pilot", "smoke"}:
    raise SystemExit("source run-spec run_kind must remain pilot/smoke")
datasets = list(master["expected_datasets"])
tuning_seeds = list(master["tuning_seeds"])
held_out = set(master["held_out_seeds"])
if wave not in (1, 2, 3):
    raise SystemExit("wave must be 1, 2, or 3")
if list(source.get("seeds", [])) != list(master["source_run_spec_seeds"]):
    raise SystemExit("source run-spec seed identity changed")
source_dataset_ids = [str(item["id"]) for item in source.get("datasets", [])]
if source_dataset_ids != datasets:
    raise SystemExit(f"source run-spec datasets changed: {source_dataset_ids}")
if held_out.intersection(tuning_seeds):
    raise SystemExit("held-out and tuning seeds overlap")
if source.get("passthrough_variants", []) not in ([], [{"id": "original_exp_rank_mr", "template_values": {}}]):
    ids = [str(item.get("id")) for item in source.get("passthrough_variants", [])]
    if ids not in ([], ["original_exp_rank_mr"]):
        raise SystemExit(f"unexpected passthrough variants: {ids}")

argv = [str(item) for item in source["trainer_argv_template"]]
def flag_value(flag):
    positions = [i for i, token in enumerate(argv) if token == flag]
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

def rounded(value):
    digits = int(master["adaptive_rules"]["significant_digits"])
    return float(format(float(value), f".{digits}g"))

def same_value(a, b):
    return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=1e-15)

def generate_next(winner, existing, pool):
    rules = master["adaptive_rules"]
    lower = float(rules["min_c"])
    upper = float(rules["max_c"])
    candidates = []
    pools = [list(pool), list(rules["fallback_multiplier_pool"])]
    for multipliers in pools:
        for multiplier in multipliers:
            value = rounded(float(winner) * float(multiplier))
            if not (lower <= value <= upper) or not math.isfinite(value):
                continue
            if any(same_value(value, old) for old in existing):
                continue
            if any(same_value(value, old) for old in candidates):
                continue
            candidates.append(value)
            if len(candidates) == 5:
                return sorted(candidates)
    raise SystemExit(f"could not generate five new c values around winner={winner}")

parent_binding = None
if wave == 1:
    taskwise = {
        dataset_id: [float(value) for value in master["taskwise_wave1_c_grid"][dataset_id]]
        for dataset_id in datasets
    }
else:
    previous_dir = work_root / f"wave_{wave - 1:02d}"
    previous_audit = previous_dir / "TERMINAL_AUDIT.json"
    if not previous_audit.is_file():
        raise SystemExit(f"missing previous terminal audit: {previous_audit}")
    previous = json.loads(previous_audit.read_text())
    expected_previous = {
        "experiment_id": master["experiment_id"],
        "scientific_status": master["scientific_status"],
        "status": "PASS",
        "wave_index": wave - 1,
        "candidate_count_per_task": 5 * (wave - 1),
        "held_out_seeds_untouched": master["held_out_seeds"],
    }
    mismatches = {
        key: {"expected": expected, "actual": previous.get(key)}
        for key, expected in expected_previous.items()
        if previous.get(key) != expected
    }
    if mismatches:
        raise SystemExit(f"previous terminal audit identity mismatch: {mismatches}")
    selections = {
        str(item["dataset_id"]): item
        for item in previous["taskwise_selections"]
    }
    if list(selections) != datasets:
        raise SystemExit("previous taskwise selection dataset order changed")
    existing_by_dataset = {dataset_id: [] for dataset_id in datasets}
    for item in previous["candidate_groups"]:
        existing_by_dataset[str(item["dataset_id"])].append(float(item["parameter_value"]))
    pool_key = "wave2_multiplier_pool" if wave == 2 else "wave3_multiplier_pool"
    taskwise = {}
    for dataset_id in datasets:
        winner = float(selections[dataset_id]["parameter_value"])
        taskwise[dataset_id] = generate_next(
            winner,
            existing_by_dataset[dataset_id],
            master["adaptive_rules"][pool_key],
        )
    parent_binding = {
        "path": str(previous_audit.relative_to(Path.cwd().resolve())),
        "sha256": sha256(previous_audit),
        "wave_index": wave - 1,
        "candidate_count_per_task": 5 * (wave - 1),
    }

for dataset_id, values in taskwise.items():
    if len(values) != 5 or len(set(values)) != 5:
        raise SystemExit(f"Wave {wave} requires five unique values for {dataset_id}")
    if any(value <= 0.0 or not math.isfinite(value) for value in values):
        raise SystemExit(f"Wave {wave} has invalid values for {dataset_id}")

wave_dir = work_root / f"wave_{wave:02d}"
generated_dir = wave_dir / "generated"
generated_dir.mkdir(parents=True, exist_ok=True)
wave_grid = {
    "schema_version": 1,
    "experiment_id": master["experiment_id"],
    "scientific_status": master["scientific_status"],
    "runner_version": master["runner_version"],
    "wave_index": wave,
    "canonical_alpha": master["canonical_alpha"],
    "negative_scale": master["negative_scale"],
    "reference_distance": master["reference_distance"],
    "tuning_seeds": tuning_seeds,
    "held_out_seeds": master["held_out_seeds"],
    "primary_selection_metric": master["primary_selection_metric"],
    "taskwise_c_grid": taskwise,
    "parent_terminal_audit_binding": parent_binding,
    "generation_rule": (
        "frozen_taskwise_wave1_grid"
        if wave == 1
        else f"deterministic_{'wave2' if wave == 2 else 'wave3'}_multiplier_pool"
    ),
}
(wave_dir / "WAVE_GRID.json").write_text(json.dumps(wave_grid, indent=2) + "\n")

dataset_by_id = {str(item["id"]): item for item in source["datasets"]}
units = []
def label(value):
    return format(float(value), ".12g").replace("-", "m").replace(".", "p")
for dataset_id in datasets:
    for c_value in taskwise[dataset_id]:
        c_label = label(c_value)
        unit_id = f"{dataset_id}__c{c_label}"
        unit_generated = generated_dir / unit_id
        unit_generated.mkdir(parents=True, exist_ok=True)
        unit_work_dir = wave_dir / "units" / unit_id
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
                "exponential": c_value,
            },
            "anchors": {},
            "negative_scale_grid": {"exponential": [master["negative_scale"]]},
            "wave_index": wave,
            "dataset_id": dataset_id,
            "parameter_name": "exponential_coefficient",
            "parameter_value": c_value,
        }
        run_spec_path = unit_generated / "run_spec.json"
        grid_path = unit_generated / "grid.json"
        run_spec_path.write_text(json.dumps(run_spec, indent=2) + "\n")
        grid_path.write_text(json.dumps(grid, indent=2) + "\n")
        units.append({
            "unit_id": unit_id,
            "wave_index": wave,
            "dataset_id": dataset_id,
            "parameter_name": "exponential_coefficient",
            "parameter_value": c_value,
            "run_spec": str(run_spec_path),
            "grid": str(grid_path),
            "work_dir": str(unit_work_dir),
            "expected_branches": len(tuning_seeds),
        })
if len(units) != 45:
    raise SystemExit(f"expected 45 units in Wave {wave}, got {len(units)}")
units_payload = {
    "schema_version": 1,
    "experiment_id": master["experiment_id"],
    "wave_index": wave,
    "unit_count": len(units),
    "expected_branch_count": len(units) * len(tuning_seeds),
    "units": units,
}
(wave_dir / "UNITS.json").write_text(json.dumps(units_payload, indent=2) + "\n")
with (wave_dir / "UNITS.tsv").open("w") as handle:
    for unit in units:
        handle.write(
            f"{unit['run_spec']}\t{unit['grid']}\t{unit['work_dir']}\n"
        )
print(json.dumps({
    "status": "READY",
    "wave": wave,
    "units": len(units),
    "branches": len(units) * len(tuning_seeds),
    "wave_grid": str(wave_dir / "WAVE_GRID.json"),
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

run_wave() {
  local wave="$1"
  local units_tsv="${WORK_ROOT}/wave_$(printf '%02d' "${wave}")/UNITS.tsv"
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
  done < "${units_tsv}"
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      status=1
    fi
  done
  return "${status}"
}

audit_wave() {
  local wave="$1"
  python3 - "${MASTER_CONFIG}" "${WORK_ROOT}" "${wave}" <<'PY'
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

master = json.loads(Path(sys.argv[1]).read_text())
work_root = Path(sys.argv[2]).resolve()
wave = int(sys.argv[3])
wave_dir = work_root / f"wave_{wave:02d}"
units_payload = json.loads((wave_dir / "UNITS.json").read_text())
wave_grid = json.loads((wave_dir / "WAVE_GRID.json").read_text())
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
for unit in units_payload["units"]:
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
    c_value = float(unit["parameter_value"])
    for branch_dir in branch_dirs:
        completed = read_json(branch_dir / "COMPLETED.json")
        if int(completed.get("return_code", -1)) != 0:
            raise SystemExit(f"nonzero branch return code: {branch_dir}")
        branch_config = read_json(branch_dir / "branch_config.json")
        control = branch_config.get("negative_control")
        expected_control = {
            "method": "exponential",
            "negative_scale": 1.0,
            "canonical_alpha": 0.11,
            "reference_distance": 2.0,
            "exponential_coefficient": c_value,
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
            "branch_id": branch_config["branch_id"],
            "dataset_id": unit["dataset_id"],
            "seed": seed,
            "method": "exponential",
            "parameter_name": "exponential_coefficient",
            "parameter_value": c_value,
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
if len(rows) != 180:
    raise SystemExit(f"expected 180 audited branches, got {len(rows)}")

grouped = {}
for row in rows:
    key = (row["dataset_id"], float(row["parameter_value"]))
    grouped.setdefault(key, []).append(row)
round_groups = []
for (dataset_id, c_value), members in sorted(grouped.items()):
    members = sorted(members, key=lambda item: int(item["seed"]))
    seeds = [int(item["seed"]) for item in members]
    if seeds != tuning_seeds:
        raise SystemExit(f"seed coverage mismatch for {dataset_id}/c={c_value}: {seeds}")
    late = [float(item["late_window_mean"]) for item in members]
    round_groups.append({
        "dataset_id": dataset_id,
        "method": "exponential",
        "parameter_name": "exponential_coefficient",
        "parameter_value": c_value,
        "candidate_wave": wave,
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
if len(round_groups) != 45:
    raise SystemExit(f"expected 45 Wave-{wave} candidate groups, got {len(round_groups)}")

previous_groups = []
previous_binding = None
if wave > 1:
    previous_path = work_root / f"wave_{wave - 1:02d}" / "TERMINAL_AUDIT.json"
    previous = read_json(previous_path)
    previous_groups = [dict(item) for item in previous["candidate_groups"]]
    previous_binding = {
        "path": str(previous_path.relative_to(Path.cwd().resolve())),
        "sha256": sha256(previous_path),
        "wave_index": wave - 1,
        "candidate_count_per_task": 5 * (wave - 1),
    }

seen = set()
combined = []
for item in previous_groups + round_groups:
    key = (str(item["dataset_id"]), float(item["parameter_value"]))
    if key in seen:
        raise SystemExit(f"duplicate combined candidate: {key}")
    seen.add(key)
    combined.append(dict(item))
combined.sort(key=lambda item: (datasets.index(str(item["dataset_id"])), float(item["parameter_value"])))

def select(groups, expected_count):
    by_dataset = {dataset_id: [] for dataset_id in datasets}
    for item in groups:
        by_dataset[str(item["dataset_id"])].append(item)
    selections = []
    for dataset_id in datasets:
        candidates = by_dataset[dataset_id]
        if len(candidates) != expected_count:
            raise SystemExit(
                f"expected {expected_count} candidates for {dataset_id}, got {len(candidates)}"
            )
        ranked = sorted(
            candidates,
            key=lambda row: (
                -float(row["late_window_mean_across_seeds"]),
                -float(row["late_window_min_across_seeds"]),
                float(row["best_to_late_mean_drop_mean"]),
                float(row["parameter_value"]),
            ),
        )
        winner = dict(ranked[0])
        winner["selection_rank"] = 1
        winner["candidate_count"] = len(candidates)
        winner["selection_rule"] = [
            "maximize late_window_mean_across_seeds",
            "then maximize late_window_min_across_seeds",
            "then minimize best_to_late_mean_drop_mean",
            "then choose the smaller numeric c",
        ]
        selections.append(winner)
    return selections

round_selections = select(round_groups, 5)
combined_selections = select(combined, 5 * wave)
payload = {
    "schema_version": 1,
    "experiment_id": master["experiment_id"],
    "source_experiment_id": master["source_experiment_id"],
    "scientific_status": master["scientific_status"],
    "runner_version": master["runner_version"],
    "status": "PASS",
    "wave_index": wave,
    "expected_branch_count": 180,
    "audited_branch_count": len(rows),
    "cumulative_expected_branch_count": 180 * wave,
    "datasets": datasets,
    "tuning_seeds": tuning_seeds,
    "held_out_seeds_untouched": master["held_out_seeds"],
    "methods": ["exponential"],
    "canonical_alpha": 0.11,
    "negative_scale": 1.0,
    "reference_distance": 2.0,
    "primary_metric": master["primary_selection_metric"],
    "late_window_steps": late_steps,
    "selection_scope": "per_dataset_exponential_c",
    "candidate_count_per_task": 5 * wave,
    "cross_method_ranking_allowed": False,
    "formal_d4rl9_table_population_allowed": False,
    "steady_state_claim_allowed": False,
    "fixed_horizon_is_not_convergence": True,
    "positive_only_rerun_included": False,
    "global_rerun_included": False,
    "reciprocal_rerun_included": False,
    "branches": rows,
    "round_candidate_groups": round_groups,
    "round_taskwise_selections": round_selections,
    "candidate_groups": combined,
    "taskwise_selections": combined_selections,
    "parent_terminal_audit_binding": previous_binding,
    "confirmation_required_before_method_ranking": True,
    "event_separation_summary": {
        "task_performance_collapse": "not_classified_without_registered_threshold",
        "support_or_variance_boundary_event": "not_available_in_unchanged_canonical_trainer_summary",
        "nan_inf_numerical_failure": "not_observed_in_zero_exit_branches_and_finite_evaluation_histories",
    },
}
audit_path = wave_dir / "TERMINAL_AUDIT.json"
audit_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
selection_payload = {
    "schema_version": 1,
    "experiment_id": master["experiment_id"],
    "scientific_status": master["scientific_status"],
    "wave_index": wave,
    "primary_metric": master["primary_selection_metric"],
    "tuning_seeds": tuning_seeds,
    "held_out_seeds_untouched": master["held_out_seeds"],
    "selection_scope": "per_dataset_exponential_c",
    "candidate_count_per_task": 5 * wave,
    "taskwise_selections": combined_selections,
    "confirmation_required_before_method_ranking": True,
}
(wave_dir / "TASKWISE_SELECTION.json").write_text(
    json.dumps(selection_payload, indent=2, sort_keys=True) + "\n"
)
if wave == 3:
    final_payload = {
        "schema_version": 1,
        "experiment_id": master["experiment_id"],
        "source_experiment_id": master["source_experiment_id"],
        "scientific_status": master["scientific_status"],
        "runner_version": master["runner_version"],
        "status": "PASS",
        "wave_count": 3,
        "expected_branch_count": 540,
        "audited_branch_count": 540,
        "datasets": datasets,
        "tuning_seeds": tuning_seeds,
        "held_out_seeds_untouched": master["held_out_seeds"],
        "methods": ["exponential"],
        "canonical_alpha": 0.11,
        "negative_scale": 1.0,
        "reference_distance": 2.0,
        "primary_metric": master["primary_selection_metric"],
        "selection_scope": "per_dataset_exponential_c",
        "candidate_count_per_task": 15,
        "candidate_groups": combined,
        "taskwise_selections": combined_selections,
        "wave_terminal_audits": [
            {
                "wave_index": index,
                "path": str(
                    (work_root / f"wave_{index:02d}" / "TERMINAL_AUDIT.json")
                    .relative_to(Path.cwd().resolve())
                ),
                "sha256": sha256(
                    work_root / f"wave_{index:02d}" / "TERMINAL_AUDIT.json"
                ),
            }
            for index in (1, 2, 3)
        ],
        "cross_method_ranking_allowed": False,
        "formal_d4rl9_table_population_allowed": False,
        "steady_state_claim_allowed": False,
        "fixed_horizon_is_not_convergence": True,
        "confirmation_required_before_method_ranking": True,
        "event_separation_summary": payload["event_separation_summary"],
    }
    (work_root / "TERMINAL_AUDIT.json").write_text(
        json.dumps(final_payload, indent=2, sort_keys=True) + "\n"
    )
    (work_root / "TASKWISE_SELECTION.json").write_text(
        json.dumps(selection_payload, indent=2, sort_keys=True) + "\n"
    )
print(json.dumps({
    "status": "PASS",
    "wave": wave,
    "audited_branches": len(rows),
    "candidate_count_per_task": 5 * wave,
    "terminal_audit": str(audit_path),
}, sort_keys=True))
PY
}

for wave in 1 2 3; do
  echo "=== Preparing Exp Wave ${wave}/3 ==="
  prepare_wave "${wave}"
  echo "=== Running Exp Wave ${wave}/3 ==="
  run_wave "${wave}"
  echo "=== Auditing Exp Wave ${wave}/3 ==="
  audit_wave "${wave}"
done

echo "Exp three-wave development sweep completed."
echo "Terminal audit: ${WORK_ROOT}/TERMINAL_AUDIT.json"
echo "Task-wise selection: ${WORK_ROOT}/TASKWISE_SELECTION.json"
