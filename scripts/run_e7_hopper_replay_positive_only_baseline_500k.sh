#!/usr/bin/env bash
set -euo pipefail

COMMAND="${1:-run}"
case "${COMMAND}" in
  validate|plan|run) ;;
  *)
    echo "usage: $0 [validate|plan|run]" >&2
    exit 2
    ;;
esac

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

GRID="${E7_HR_POSBASE_GRID:-configs/e7_hopper_replay_positive_only_baseline_500k.json}"
CONTRACT="${E7_CANONICAL_CONTRACT:-/root/d4rl2/configs/e7_canonical_contract_9task.json}"
RUN_SPEC="${E7_CANONICAL_RUN_SPEC:-/root/d4rl2/configs/e7_canonical_9task_full_grid_run_spec_v1.json}"
WORK_DIR="${E7_HR_POSBASE_WORK_DIR:-outputs/e7/hopper_replay_positive_only_baseline_500k_001}"
MAX_WORKERS="${E7_HR_POSBASE_MAX_WORKERS:-66}"
RUNTIME_DIR="${WORK_DIR}/.posbase_runtime"
BOOTSTRAP_WRAPPER="${RUNTIME_DIR}/bootstrap_wrapper.py"

if [[ -n "$(git status --porcelain)" ]]; then
  echo "refusing to run from a dirty checkout" >&2
  exit 2
fi
if [[ ! -f "${GRID}" ]]; then
  echo "missing Hopper replay Positive-only baseline grid: ${GRID}" >&2
  exit 2
fi
if [[ "${COMMAND}" != "validate" ]]; then
  for required in "${CONTRACT}" "${RUN_SPEC}"; do
    if [[ ! -f "${required}" ]]; then
      echo "missing required file: ${required}" >&2
      exit 2
    fi
  done
fi
if ! [[ "${MAX_WORKERS}" =~ ^[1-9][0-9]*$ ]] || (( MAX_WORKERS > 66 )); then
  echo "E7_HR_POSBASE_MAX_WORKERS must be an integer in [1,66]" >&2
  exit 2
fi

mkdir -p "${RUNTIME_DIR}"
cat >"${BOOTSTRAP_WRAPPER}" <<'PYWRAP'
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import torch

EXPERIMENT_ID = "EXT-H-E7-SQEXP-GAE-HOPPER-REPLAY-POSBASE-500K-01"
PROFILE_ID = "hopper_replay_positive_only_baseline_500k_v1"
EXPECTED_STEPS = 500_000
EXPECTED_BATCH = 512
EXPECTED_LR = 3.0e-4
ALLOWED_BASELINES = {float(value) for value in range(11)}
ALLOWED_METHOD = "positive_only"


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _optimizer_lrs(optimizer: Any) -> list[float]:
    groups = getattr(optimizer, "param_groups", None)
    if not isinstance(groups, list) or not groups:
        raise RuntimeError("canonical optimizer has no parameter groups")
    values = [float(group["lr"]) for group in groups]
    if not all(math.isfinite(value) and value > 0.0 for value in values):
        raise RuntimeError("canonical optimizer has invalid learning rates")
    return values


def _install_optimizer_audit(module: Any, target_class: str) -> dict[str, Any]:
    original_class = getattr(module, target_class)
    original_init = original_class.__init__
    state: dict[str, Any] = {"applied_count": 0}

    def wrapped_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        if state["applied_count"] != 0:
            raise RuntimeError("canonical agent was instantiated more than once")
        actor_lrs = _optimizer_lrs(self.a_opt)
        critic_lrs = _optimizer_lrs(self.c_opt)
        for label, values in (("actor", actor_lrs), ("critic", critic_lrs)):
            if not all(
                math.isclose(value, EXPECTED_LR, rel_tol=0.0, abs_tol=1e-12)
                for value in values
            ):
                raise RuntimeError(f"canonical {label} learning rate changed: {values}")
        state.update(
            applied_count=1,
            actor_lr=actor_lrs,
            critic_lr=critic_lrs,
            actor_lr_unchanged=True,
            critic_lr_unchanged=True,
        )

    original_class.__init__ = wrapped_init
    return state


def _snapshot_stats(
    raw: torch.Tensor,
    baseline: float,
    *,
    provider_update: int,
    include_quantiles: bool = False,
) -> dict[str, Any]:
    values = raw.detach().float().cpu().reshape(-1)
    shifted = values - baseline
    if values.numel() == 0 or shifted.numel() != values.numel():
        raise RuntimeError("advantage shift changed sample cardinality")
    active = shifted > 0
    active_count = int(active.sum())
    raw_std = float(values.std(unbiased=False))
    payload: dict[str, Any] = {
        "provider_update": int(provider_update),
        "samples": int(values.numel()),
        "raw_mean": float(values.mean()),
        "raw_std": raw_std,
        "raw_positive_fraction": float((values > 0).float().mean()),
        "shifted_mean": float(shifted.mean()),
        "shifted_std": float(shifted.std(unbiased=False)),
        "effective_positive_fraction": float(active.float().mean()),
        "effective_positive_count": active_count,
        "crossed_positive_to_nonpositive_fraction": float(
            ((values > 0) & (shifted <= 0)).float().mean()
        ),
        "baseline_over_raw_std": None if raw_std <= 0.0 else float(baseline / raw_std),
        "selected_raw_advantage_mean": (
            None if active_count == 0 else float(values[active].mean())
        ),
        "selected_shifted_advantage_mean": (
            None if active_count == 0 else float(shifted[active].mean())
        ),
    }
    if include_quantiles:
        quantiles = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9], dtype=values.dtype)
        raw_q = torch.quantile(values, quantiles)
        payload["raw_quantiles"] = {
            "p10": float(raw_q[0]),
            "p25": float(raw_q[1]),
            "p50": float(raw_q[2]),
            "p75": float(raw_q[3]),
            "p90": float(raw_q[4]),
        }
        if active_count > 0:
            selected_q = torch.quantile(values[active], quantiles)
            payload["selected_raw_quantiles"] = {
                "p10": float(selected_q[0]),
                "p25": float(selected_q[1]),
                "p50": float(selected_q[2]),
                "p75": float(selected_q[3]),
                "p90": float(selected_q[4]),
            }
        else:
            payload["selected_raw_quantiles"] = None
    return payload


def _self_test() -> int:
    raw = torch.tensor([-2.0, -0.1, 0.1, 0.4, 1.2, 3.0])
    stats = _snapshot_stats(raw, 1.0, provider_update=1, include_quantiles=True)
    shifted = raw - 1.0
    if shifted.numel() != raw.numel():
        raise RuntimeError("self-test deleted replay samples")
    if not torch.allclose(shifted, torch.tensor([-3.0, -1.1, -0.9, -0.6, 0.2, 2.0])):
        raise RuntimeError("self-test produced the wrong shifted advantages")
    if not math.isclose(stats["effective_positive_fraction"], 2.0 / 6.0, abs_tol=1e-7):
        raise RuntimeError("self-test produced the wrong Positive-only active fraction")
    print(json.dumps({"status": "PASS", "positive_only_baseline": stats}, sort_keys=True))
    return 0


def _run_bootstrap(argv: list[str]) -> int:
    from drpo import e7_squared_exp_night as suite
    from drpo import e7_squared_exp_night_bootstrap as bootstrap

    suite.GAE_EXPERIMENT_ID = EXPERIMENT_ID
    suite.TUNING_PROFILE_ID = PROFILE_ID
    suite.EXPECTED_STEPS = EXPECTED_STEPS

    try:
        branch_index = argv.index("--branch-config") + 1
    except (ValueError, IndexError) as exc:
        raise ValueError("bootstrap wrapper requires --branch-config") from exc
    branch_config_path = Path(argv[branch_index]).expanduser().resolve()
    branch_dir = branch_config_path.parent
    snapshots_path = branch_dir / "ADVANTAGE_SHIFT_SNAPSHOTS.jsonl"
    snapshots_path.unlink(missing_ok=True)

    branch = json.loads(branch_config_path.read_text(encoding="utf-8"))
    if branch.get("experiment_id") != EXPERIMENT_ID or branch.get("profile_id") != PROFILE_ID:
        raise ValueError("Hopper replay Positive-only baseline bootstrap identity mismatch")
    values = branch.get("template_values", {})
    batch_size = int(values.get("batch_size", -1))
    baseline = float(values.get("advantage_baseline", float("nan")))
    method = str(values.get("weight_method"))
    if batch_size != EXPECTED_BATCH:
        raise ValueError(f"unsupported Positive-only baseline batch_size={batch_size}")
    if baseline not in ALLOWED_BASELINES:
        raise ValueError(f"unsupported fixed advantage baseline={baseline}")
    if method != ALLOWED_METHOD:
        raise ValueError(f"unsupported method={method}; this profile is Positive-only only")

    original_flag_value = suite._flag_value  # noqa: SLF001

    def compatible_flag_value(tokens: list[str], flag: str) -> str:
        value = original_flag_value(tokens, flag)
        return "256" if flag == "--batch" else value

    suite._flag_value = compatible_flag_value  # type: ignore[assignment]  # noqa: SLF001

    original_provider = bootstrap.TrajectorySnapshotAdvantage

    class ShiftedBatchAwareProvider(original_provider):
        def __init__(self, replay: Any, estimator: str, batch_size_arg: int = 256) -> None:
            del batch_size_arg
            super().__init__(replay, estimator, batch_size=EXPECTED_BATCH)
            self.fixed_baseline = baseline
            self.shift_snapshot_stats: list[dict[str, Any]] = []

        def _refresh(self, agent: Any) -> None:
            super()._refresh(agent)
            if self.table is None:
                raise RuntimeError("trajectory snapshot table was not materialized")
            stats = _snapshot_stats(
                self.table,
                self.fixed_baseline,
                provider_update=self.update_count,
            )
            self.shift_snapshot_stats.append(stats)
            _append_jsonl(snapshots_path, stats)

        def __call__(self, agent: Any, context: Any, default: torch.Tensor) -> torch.Tensor:
            raw = super().__call__(agent, context, default)
            shifted = raw - self.fixed_baseline
            if shifted.shape != raw.shape or shifted.numel() != raw.numel():
                raise RuntimeError("advantage shift changed the sampled batch")
            return shifted

        def summary(self) -> dict[str, Any]:
            payload = super().summary()
            if not self.shift_snapshot_stats or self.table is None:
                raise RuntimeError("Positive-only baseline statistics were never recorded")
            latest = _snapshot_stats(
                self.table,
                self.fixed_baseline,
                provider_update=self.update_count,
                include_quantiles=True,
            )
            fractions = [
                float(record["effective_positive_fraction"])
                for record in self.shift_snapshot_stats
            ]
            payload["advantage_shift"] = {
                "formula": "shifted_advantage = raw_trajectory_snapshot_gae - fixed_baseline",
                "actor_rule": "max(shifted_advantage, 0) times log_probability",
                "fixed_baseline": self.fixed_baseline,
                "baseline_units": "raw_trajectory_snapshot_gae",
                "shift_before_positive_only_mask": True,
                "replay_sample_deletion": False,
                "replay_sample_masking": False,
                "negative_shifted_advantage_actor_contribution": 0.0,
                "snapshot_stat_count": len(self.shift_snapshot_stats),
                "snapshot_series_file": snapshots_path.name,
                "first_snapshot": self.shift_snapshot_stats[0],
                "latest_snapshot": latest,
                "effective_positive_fraction_min": min(fractions),
                "effective_positive_fraction_max": max(fractions),
                "effective_positive_fraction_mean": sum(fractions) / len(fractions),
            }
            return payload

    bootstrap.TrajectorySnapshotAdvantage = ShiftedBatchAwareProvider

    original_patch = bootstrap.patch_canonical_module
    state_holder: dict[str, Any] = {}

    def patched_module(*args: Any, **kwargs: Any) -> Any:
        result = original_patch(*args, **kwargs)
        module = args[0]
        contract = args[1]
        state_holder["optimizer"] = _install_optimizer_audit(module, contract.target_class)
        return result

    bootstrap.patch_canonical_module = patched_module
    result = int(bootstrap.main(argv))
    optimizer = state_holder.get("optimizer")
    if not isinstance(optimizer, dict) or optimizer.get("applied_count") != 1:
        raise RuntimeError("optimizer audit was not applied exactly once")

    manifest_path = branch_dir / "branch_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    snapshot = manifest.get("trajectory_snapshot", {})
    shift = snapshot.get("advantage_shift")
    if not isinstance(shift, dict) or float(shift.get("fixed_baseline", float("nan"))) != baseline:
        raise RuntimeError("completed manifest is missing the exact Positive-only baseline")
    if not snapshots_path.is_file():
        raise RuntimeError("completed branch is missing snapshot-series diagnostics")
    control = {
        "experiment_id": EXPERIMENT_ID,
        "profile_id": PROFILE_ID,
        "control_id": values.get("control_id"),
        "weight_method": method,
        "fixed_baseline": baseline,
        "baseline_units": "raw_trajectory_snapshot_gae",
        "shift_before_positive_only_mask": True,
        "replay_sample_deletion": False,
        "replay_sample_masking": False,
        "negative_shifted_advantage_actor_contribution": 0.0,
        "batch_size": EXPECTED_BATCH,
        "optimizer_control": optimizer,
        "snapshot_advantage_shift": shift,
    }
    _atomic_json(branch_dir / "ADVANTAGE_SHIFT_CONTROL.json", control)
    _atomic_json(branch_dir / "OPTIMIZER_CONTROL.json", optimizer)
    manifest["advantage_shift_control"] = control
    manifest["trajectory_snapshot_batch_size"] = EXPECTED_BATCH
    _atomic_json(manifest_path, manifest)
    return result


def main() -> int:
    if sys.argv[1:] == ["--self-test"]:
        return _self_test()
    return _run_bootstrap(sys.argv[1:])


raise SystemExit(main())
PYWRAP

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
python -m py_compile "${BOOTSTRAP_WRAPPER}"
python "${BOOTSTRAP_WRAPPER}" --self-test >/dev/null

python - \
  "${COMMAND}" \
  "${CONTRACT}" \
  "${RUN_SPEC}" \
  "${GRID}" \
  "${WORK_DIR}" \
  "${MAX_WORKERS}" \
  "${BOOTSTRAP_WRAPPER}" <<'PY'
from __future__ import annotations

import copy
import json
import math
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from drpo import e7_canonical_sweep as base
from drpo import e7_squared_exp_night as suite
from drpo import e7_squared_exp_night_aggregate as agg
from drpo.e7_canonical_injection import sha256_file

EXPERIMENT_ID = "EXT-H-E7-SQEXP-GAE-HOPPER-REPLAY-POSBASE-500K-01"
PROFILE_ID = "hopper_replay_positive_only_baseline_500k_v1"
SCIENTIFIC_STATUS = "hopper_replay_positive_only_fixed_baseline_response_500k_pilot_only"
RUNNER_VERSION = "5.7.0-hopper-replay-positive-only-baseline-500k"
FULL_RUN_ENV = "DRPO_E7_HOPPER_REPLAY_POSBASE_500K_FULL_RUN"
DATASET_ID = "hopper-medium-replay-v2"
SEEDS = (200, 201, 202, 203, 208, 209)
HELD_OUT_SEEDS = (204, 205, 206, 207)
BASELINES = tuple(float(value) for value in range(11))
CONTROL_ORDER = tuple(f"pos_b{int(value)}" for value in BASELINES)
CONTROL_SPECS = dict(zip(CONTROL_ORDER, BASELINES, strict=True))
STEPS = 500_000
EVAL_INTERVAL = 20_000
EVAL_EPISODES = 10
BATCH_SIZE = 512
ACTOR_LR = 3.0e-4
CRITIC_LR = 3.0e-4
REFERENCE_DISTANCE = 2.0
REMOTENESS_SCALE = 0.08
REMOTENESS_THRESHOLD = 0.0
TAPER_LAMBDA = 1.0
EXPECTED_BRANCHES = 66
EXPECTED_EVAL_STEPS = tuple(range(EVAL_INTERVAL, STEPS + 1, EVAL_INTERVAL))
EARLY_START, EARLY_END = 20_000, 200_000
MIDDLE_START, MIDDLE_END = 220_000, 340_000
LATE_START, LATE_END = 360_000, 500_000
TREND_START, TREND_END = 200_000, 500_000

COMMAND, CONTRACT, RUN_SPEC, GRID, WORK_DIR, MAX_WORKERS, WRAPPER = sys.argv[1:]
WORK_ROOT = Path(WORK_DIR)
WRAPPER_PATH = Path(WRAPPER).resolve()

ORIGINAL_CONFIGURE = suite.configure_execution
ORIGINAL_LOAD_GRID = suite.load_grid
ORIGINAL_LOAD_RUN_SPEC = suite.load_run_spec
ORIGINAL_GAE_BRANCHES = suite._gae_branches  # noqa: SLF001
ORIGINAL_BRANCH_COMMAND = suite.branch_command
ORIGINAL_AGGREGATE = suite.aggregate_results
ORIGINAL_IS_TUNING = suite._is_tuning  # noqa: SLF001
ORIGINAL_IS_P3 = suite._is_p3  # noqa: SLF001
ORIGINAL_ACTIVE_STATUS = suite.active_scientific_status
ORIGINAL_ACTIVE_COUNT = suite.active_expected_branch_count


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise RuntimeError("cannot average an empty sequence")
    return statistics.fmean(values)


def _median(values: Sequence[float]) -> float:
    if not values:
        raise RuntimeError("cannot take the median of an empty sequence")
    return statistics.median(values)


def _std(values: Sequence[float]) -> float | None:
    return statistics.stdev(values) if len(values) >= 2 else None


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    agg._write_csv(path, rows)  # noqa: SLF001


def _slope_per_100k(
    steps: Sequence[int], scores: Sequence[float], start: int, end: int
) -> float | None:
    pairs = [
        (float(step), float(score))
        for step, score in zip(steps, scores, strict=True)
        if start <= step <= end and math.isfinite(score)
    ]
    if len(pairs) < 2:
        return None
    x_mean = _mean([x for x, _ in pairs])
    y_mean = _mean([y for _, y in pairs])
    denominator = sum((x - x_mean) ** 2 for x, _ in pairs)
    if denominator <= 0.0:
        return None
    slope = sum((x - x_mean) * (y - y_mean) for x, y in pairs) / denominator
    return slope * 100_000.0


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise RuntimeError(f"missing required JSONL file: {path}")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise RuntimeError(f"empty required JSONL file: {path}")
    return rows


def _load_config(path: str | Path) -> tuple[dict[str, Any], str]:
    source = Path(path)
    raw = json.loads(source.read_text(encoding="utf-8"))
    required = {
        "experiment_id": EXPERIMENT_ID,
        "profile_id": PROFILE_ID,
        "parent_experiment_id": "EXT-H-E7-BENCH-01",
        "predecessor_experiment_id": "EXT-H-E7-SQEXP-GAE-HOPPER-REPLAY-ADVSHIFT-200K-01",
        "run_kind": "pilot",
        "status": "not_run",
        "scientific_status": SCIENTIFIC_STATUS,
        "dataset": DATASET_ID,
        "development_seeds": list(SEEDS),
        "held_out_seeds": list(HELD_OUT_SEEDS),
        "steps": STEPS,
        "evaluation_interval": EVAL_INTERVAL,
        "evaluation_episodes": EVAL_EPISODES,
        "actor_update_mode": "a2c",
        "advantage_mode": "gae_lambda_0p95",
        "batch_size": BATCH_SIZE,
        "actor_lr": ACTOR_LR,
        "critic_lr": CRITIC_LR,
        "baselines": [int(value) for value in BASELINES],
        "expected_branches": EXPECTED_BRANCHES,
        "screening_only": True,
        "formal_evidence_allowed": False,
    }
    changed = [key for key, value in required.items() if raw.get(key) != value]
    if changed:
        raise ValueError(f"Hopper replay Positive-only baseline grid changed: {changed}")
    expected_controls = [
        {
            "id": control_id,
            "weight_method": "positive_only",
            "advantage_baseline": int(CONTROL_SPECS[control_id]),
        }
        for control_id in CONTROL_ORDER
    ]
    if raw.get("controls") != expected_controls:
        raise ValueError("Positive-only baseline control matrix changed")
    if raw.get("advantage_shift") != {
        "formula": "shifted_advantage = raw_trajectory_snapshot_gae - fixed_baseline",
        "baseline_units": "raw_trajectory_snapshot_gae",
        "shift_applied_before_positive_only_mask": True,
        "replay_sample_deletion": False,
        "replay_sample_masking": False,
        "actor_contribution_rule": "max(shifted_advantage, 0) times log_probability",
        "negative_shifted_advantage_actor_contribution": 0,
    }:
        raise ValueError("Positive-only baseline shift semantics changed")
    return raw, sha256_file(source)


def _reset_profile_hooks() -> None:
    suite.configure_execution = ORIGINAL_CONFIGURE
    suite.load_grid = ORIGINAL_LOAD_GRID
    suite.load_run_spec = ORIGINAL_LOAD_RUN_SPEC
    suite._gae_branches = ORIGINAL_GAE_BRANCHES  # noqa: SLF001
    suite.branch_command = ORIGINAL_BRANCH_COMMAND
    suite.aggregate_results = ORIGINAL_AGGREGATE
    suite._is_tuning = ORIGINAL_IS_TUNING  # noqa: SLF001
    suite._is_p3 = ORIGINAL_IS_P3  # noqa: SLF001
    suite.active_scientific_status = ORIGINAL_ACTIVE_STATUS
    suite.active_expected_branch_count = ORIGINAL_ACTIVE_COUNT


def _install_profile() -> None:
    _reset_profile_hooks()
    suite.GAE_EXPERIMENT_ID = EXPERIMENT_ID
    suite.TUNING_PROFILE_ID = PROFILE_ID
    suite.TUNING_SEEDS = SEEDS
    suite.TUNING_EXPECTED_BRANCHES = EXPECTED_BRANCHES
    suite.TUNING_RUNNER_VERSION = RUNNER_VERSION
    suite.TUNING_FULL_RUN_ENV = FULL_RUN_ENV
    suite.EXPECTED_STEPS = STEPS
    agg.GAE_EXPERIMENT_ID = EXPERIMENT_ID

    def is_tuning() -> bool:
        return (
            suite._ACTIVE_EXPERIMENT_ID == EXPERIMENT_ID  # noqa: SLF001
            and suite._ACTIVE_PROFILE_ID == PROFILE_ID  # noqa: SLF001
        )

    suite._is_tuning = is_tuning  # type: ignore[assignment]  # noqa: SLF001
    suite._is_p3 = lambda: False  # type: ignore[assignment]  # noqa: SLF001
    suite.active_scientific_status = lambda: SCIENTIFIC_STATUS
    suite.active_expected_branch_count = lambda: EXPECTED_BRANCHES

    def configure_execution(
        grid_path: str | Path,
        *,
        liveness_pair: bool = False,
        liveness_steps: int | None = None,
    ) -> None:
        if liveness_pair or liveness_steps is not None:
            raise ValueError("the 500k Positive-only baseline profile has no liveness submatrix")
        loaded, _ = _load_config(grid_path)
        suite._ACTIVE_EXPERIMENT_ID = str(loaded["experiment_id"])  # noqa: SLF001
        suite._ACTIVE_PROFILE_ID = str(loaded["profile_id"])  # noqa: SLF001
        suite._LIVENESS_STEPS = None  # noqa: SLF001

    suite.configure_execution = configure_execution
    suite.load_grid = _load_config

    def load_run_spec(path: str | Path) -> tuple[dict[str, Any], str]:
        loaded, digest = ORIGINAL_LOAD_RUN_SPEC(path)
        loaded = copy.deepcopy(loaded)
        by_id = {str(item["id"]): item for item in loaded["datasets"]}
        if DATASET_ID not in by_id:
            raise ValueError("canonical run spec is missing hopper-medium-replay-v2")
        loaded["datasets"] = [copy.deepcopy(by_id[DATASET_ID])]
        loaded["seeds"] = list(SEEDS)
        argv = [str(value) for value in loaded["trainer_argv_template"]]
        expected_source = {
            "--batch": "256",
            "--lr": "0.0003",
            "--eval_interval": "50000",
            "--eval_episodes": "10",
            "--steps": "{steps}",
        }
        for flag, expected in expected_source.items():
            if suite._flag_value(argv, flag) != expected:  # noqa: SLF001
                raise ValueError(f"canonical source {flag} changed before 500k injection")
        argv[argv.index("--batch") + 1] = "{batch_size}"
        argv[argv.index("--eval_interval") + 1] = "{evaluation_interval}"
        loaded["trainer_argv_template"] = argv
        return loaded, digest

    suite.load_run_spec = load_run_spec

    def gae_branches(
        run_spec: Mapping[str, Any], grid: Mapping[str, Any]
    ) -> list[base.Branch]:
        del grid
        datasets = [base.DatasetSpec.from_mapping(item) for item in run_spec["datasets"]]
        if len(datasets) != 1 or datasets[0].id != DATASET_ID:
            raise ValueError("Positive-only baseline profile must contain only hopper-medium-replay-v2")
        dataset = datasets[0]
        branches: list[base.Branch] = []
        for control_id in CONTROL_ORDER:
            baseline = CONTROL_SPECS[control_id]
            for seed in SEEDS:
                branches.append(
                    base.Branch(
                        branch_id=(
                            f"{dataset.id}__seed{seed}__{control_id}__gae__a2c__steps500k"
                        ),
                        branch_kind="injected",
                        dataset=dataset,
                        seed=seed,
                        template_values={
                            "steps": str(STEPS),
                            "stage": "positive_only_baseline_500k_response",
                            "actor_update_mode": "a2c",
                            "advantage_estimator": "gae",
                            "weight_method": "positive_only",
                            "weight_at_zero": "0",
                            "exp_coefficient": "0",
                            "reference_distance": f"{REFERENCE_DISTANCE:.17g}",
                            "remoteness_threshold": f"{REMOTENESS_THRESHOLD:.17g}",
                            "remoteness_scale": f"{REMOTENESS_SCALE:.17g}",
                            "taper_lambda": f"{TAPER_LAMBDA:.17g}",
                            "advantage_baseline": f"{baseline:.17g}",
                            "advantage_baseline_units": "raw_trajectory_snapshot_gae",
                            "control_id": control_id,
                            "batch_size": str(BATCH_SIZE),
                            "evaluation_interval": str(EVAL_INTERVAL),
                            "diagnostics_interval": str(suite.DIAGNOSTICS_INTERVAL),
                            "sampled_values_per_update": str(suite.SAMPLED_VALUES_PER_UPDATE),
                            "execution_mode": "full",
                        },
                        negative_control=None,
                    )
                )
        if len(branches) != EXPECTED_BRANCHES or len(
            {branch.branch_id for branch in branches}
        ) != EXPECTED_BRANCHES:
            raise RuntimeError("Positive-only baseline branch matrix is not exact")
        if {branch.seed for branch in branches} != set(SEEDS):
            raise RuntimeError("Positive-only baseline seed set changed")
        if {branch.seed for branch in branches} & set(HELD_OUT_SEEDS):
            raise RuntimeError("held-out seeds entered the Positive-only baseline matrix")
        return branches

    suite._gae_branches = gae_branches  # type: ignore[assignment]  # noqa: SLF001

    def branch_command(**kwargs: Any) -> tuple[list[str], dict[str, Any]]:
        command, branch_config = ORIGINAL_BRANCH_COMMAND(**kwargs)
        if command[1:3] != ["-m", "drpo.e7_squared_exp_night_bootstrap"]:
            raise RuntimeError("unexpected canonical bootstrap command shape")
        command = [command[0], str(WRAPPER_PATH), *command[3:]]
        branch_config["profile_id"] = PROFILE_ID
        values = branch_config["template_values"]
        branch_config["advantage_shift"] = {
            "formula": "shifted_advantage = raw_trajectory_snapshot_gae - fixed_baseline",
            "fixed_baseline": float(values["advantage_baseline"]),
            "baseline_units": "raw_trajectory_snapshot_gae",
            "shift_before_positive_only_mask": True,
            "replay_sample_deletion": False,
            "replay_sample_masking": False,
            "negative_shifted_advantage_actor_contribution": 0.0,
        }
        branch_dir = Path(kwargs["branch_dir"])
        base.atomic_write_json(branch_dir / "branch_config.json", branch_config)
        return command, branch_config

    suite.branch_command = branch_command
    suite.aggregate_results = lambda work_dir: _aggregate(Path(work_dir))


def _aggregate(work: Path) -> dict[str, Any]:
    branch_dirs, experiment_id = agg._branch_dirs(work)  # noqa: SLF001
    if experiment_id != EXPERIMENT_ID or len(branch_dirs) != EXPECTED_BRANCHES:
        raise RuntimeError("Positive-only baseline aggregate identity or branch count mismatch")
    rows: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    snapshot_curves: list[dict[str, Any]] = []
    for branch_dir in branch_dirs:
        if not (branch_dir / "COMPLETED.json").is_file():
            raise RuntimeError(f"incomplete Positive-only baseline branch: {branch_dir.name}")
        branch = json.loads((branch_dir / "branch_config.json").read_text())
        manifest = json.loads((branch_dir / "branch_manifest.json").read_text())
        values = branch["template_values"]
        if branch.get("experiment_id") != EXPERIMENT_ID or branch.get("profile_id") != PROFILE_ID:
            raise RuntimeError("Positive-only baseline branch identity mismatch")
        seed = int(branch["seed"])
        if seed not in SEEDS or seed in HELD_OUT_SEEDS:
            raise RuntimeError("Positive-only baseline branch used a forbidden seed")
        control_id = str(values["control_id"])
        baseline = float(values["advantage_baseline"])
        if CONTROL_SPECS.get(control_id) != baseline or str(values["weight_method"]) != "positive_only":
            raise RuntimeError("Positive-only baseline branch control changed")
        if int(values["batch_size"]) != BATCH_SIZE:
            raise RuntimeError("Positive-only baseline branch batch size changed")

        summary_path = agg._only(  # noqa: SLF001
            (branch_dir / "trainer_output").glob("*_summary.json"), "trainer summary"
        )
        steps, scores = agg._read_history(json.loads(summary_path.read_text()))  # noqa: SLF001
        if tuple(steps) != EXPECTED_EVAL_STEPS:
            raise RuntimeError(
                f"Positive-only baseline evaluation trajectory changed for {branch_dir.name}: {steps}"
            )
        if not all(math.isfinite(score) for score in scores):
            raise RuntimeError(f"non-finite score in {branch_dir.name}")

        early = [score for step, score in zip(steps, scores, strict=True) if EARLY_START <= step <= EARLY_END]
        middle = [score for step, score in zip(steps, scores, strict=True) if MIDDLE_START <= step <= MIDDLE_END]
        late = [score for step, score in zip(steps, scores, strict=True) if LATE_START <= step <= LATE_END]
        trend = _slope_per_100k(steps, scores, TREND_START, TREND_END)
        best_index = max(range(len(scores)), key=scores.__getitem__)

        shift_control = manifest.get("advantage_shift_control")
        if not isinstance(shift_control, dict):
            raise RuntimeError("branch manifest is missing Positive-only baseline control")
        if (
            float(shift_control.get("fixed_baseline", float("nan"))) != baseline
            or shift_control.get("replay_sample_deletion") is not False
            or shift_control.get("replay_sample_masking") is not False
            or float(shift_control.get("negative_shifted_advantage_actor_contribution", float("nan"))) != 0.0
        ):
            raise RuntimeError("branch manifest Positive-only baseline semantics changed")
        optimizer = shift_control.get("optimizer_control", {})
        if optimizer.get("actor_lr_unchanged") is not True or optimizer.get("critic_lr_unchanged") is not True:
            raise RuntimeError("branch optimizer audit failed")
        snapshot_shift = shift_control.get("snapshot_advantage_shift", {})
        first_shift = snapshot_shift.get("first_snapshot", {})
        latest_shift = snapshot_shift.get("latest_snapshot", {})
        snapshot_rows = _load_jsonl(branch_dir / "ADVANTAGE_SHIFT_SNAPSHOTS.jsonl")
        if len(snapshot_rows) != int(snapshot_shift.get("snapshot_stat_count", -1)):
            raise RuntimeError("snapshot-series count does not match branch manifest")
        effective_fractions = [float(item["effective_positive_fraction"]) for item in snapshot_rows]
        for item in snapshot_rows:
            snapshot_curves.append(
                {
                    "branch_id": branch["branch_id"],
                    "seed": seed,
                    "control_id": control_id,
                    "advantage_baseline": baseline,
                    **item,
                }
            )

        row = {
            "branch_id": branch["branch_id"],
            "experiment_id": EXPERIMENT_ID,
            "profile_id": PROFILE_ID,
            "dataset": DATASET_ID,
            "seed": seed,
            "control_id": control_id,
            "weight_method": "positive_only",
            "advantage_baseline": baseline,
            "advantage_baseline_units": "raw_trajectory_snapshot_gae",
            "trainer_steps": STEPS,
            "evaluation_interval": EVAL_INTERVAL,
            "evaluation_episodes": EVAL_EPISODES,
            "batch_size": BATCH_SIZE,
            "actor_lr": ACTOR_LR,
            "critic_lr": CRITIC_LR,
            "early_window_mean_20k_200k": _mean(early),
            "middle_window_mean_220k_340k": _mean(middle),
            "late_window_mean_360k_500k": _mean(late),
            "late_window_std_360k_500k": _std(late),
            "trend_slope_per_100k_200k_500k": trend,
            "best_score": scores[best_index],
            "best_step": steps[best_index],
            "final_score_500k": scores[-1],
            "best_to_final_drop": scores[best_index] - scores[-1],
            "snapshot_stat_count": len(snapshot_rows),
            "first_effective_positive_fraction": first_shift.get("effective_positive_fraction"),
            "latest_effective_positive_fraction": latest_shift.get("effective_positive_fraction"),
            "min_effective_positive_fraction": min(effective_fractions),
            "max_effective_positive_fraction": max(effective_fractions),
            "mean_effective_positive_fraction": _mean(effective_fractions),
            "latest_raw_advantage_mean": latest_shift.get("raw_mean"),
            "latest_raw_advantage_std": latest_shift.get("raw_std"),
            "latest_baseline_over_raw_std": latest_shift.get("baseline_over_raw_std"),
            "latest_selected_raw_advantage_mean": latest_shift.get("selected_raw_advantage_mean"),
            "latest_selected_shifted_advantage_mean": latest_shift.get("selected_shifted_advantage_mean"),
            "task_performance_collapse_event": "not_adjudicated_no_registered_threshold",
            "support_or_variance_boundary_event": "not_instrumented_in_this_pilot",
            "rollout_failure_event": False,
            "nan_inf_numerical_failure": False,
        }
        rows.append(row)
        for step, score in zip(steps, scores, strict=True):
            curves.append(
                {
                    "branch_id": branch["branch_id"],
                    "dataset": DATASET_ID,
                    "seed": seed,
                    "control_id": control_id,
                    "advantage_baseline": baseline,
                    "step": step,
                    "normalized_score": score,
                }
            )

    observed_cells = {(row["control_id"], int(row["seed"])) for row in rows}
    expected_cells = {(control, seed) for control in CONTROL_ORDER for seed in SEEDS}
    if observed_cells != expected_cells:
        raise RuntimeError("Positive-only baseline aggregate cell set is incomplete")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["control_id"])].append(row)
    summaries: list[dict[str, Any]] = []
    for control_id in CONTROL_ORDER:
        values = sorted(grouped[control_id], key=lambda row: int(row["seed"]))
        if len(values) != len(SEEDS):
            raise RuntimeError(f"control {control_id} does not contain six seeds")
        late_values = [float(row["late_window_mean_360k_500k"]) for row in values]
        final_values = [float(row["final_score_500k"]) for row in values]
        slope_values = [
            float(row["trend_slope_per_100k_200k_500k"])
            for row in values
            if row["trend_slope_per_100k_200k_500k"] is not None
        ]
        drop_values = [float(row["best_to_final_drop"]) for row in values]
        effective_values = [float(row["latest_effective_positive_fraction"]) for row in values]
        worst_two = sorted(late_values)[:2]
        summaries.append(
            {
                "control_id": control_id,
                "weight_method": "positive_only",
                "advantage_baseline": CONTROL_SPECS[control_id],
                "seed_ids": list(SEEDS),
                "late_mean": _mean(late_values),
                "late_median": _median(late_values),
                "late_std": _std(late_values),
                "worst_two_seed_late_mean": _mean(worst_two),
                "final_mean": _mean(final_values),
                "final_median": _median(final_values),
                "final_std": _std(final_values),
                "trend_slope_mean_per_100k": _mean(slope_values),
                "trend_slope_median_per_100k": _median(slope_values),
                "nonnegative_trend_seed_count": sum(value >= 0.0 for value in slope_values),
                "mean_best_to_final_drop": _mean(drop_values),
                "latest_effective_positive_fraction_mean": _mean(effective_values),
                "latest_effective_positive_fraction_median": _median(effective_values),
            }
        )

    index = {(str(row["control_id"]), int(row["seed"])): row for row in rows}
    paired: list[dict[str, Any]] = []
    for candidate in CONTROL_ORDER[1:]:
        for seed in SEEDS:
            left = index[(candidate, seed)]
            right = index[("pos_b0", seed)]
            paired.append(
                {
                    "candidate": candidate,
                    "candidate_baseline": left["advantage_baseline"],
                    "comparator": "pos_b0",
                    "seed": seed,
                    "late_difference": float(left["late_window_mean_360k_500k"])
                    - float(right["late_window_mean_360k_500k"]),
                    "final_difference": float(left["final_score_500k"])
                    - float(right["final_score_500k"]),
                    "trend_slope_difference_per_100k": float(
                        left["trend_slope_per_100k_200k_500k"]
                    )
                    - float(right["trend_slope_per_100k_200k_500k"]),
                }
            )

    aggregate_dir = work / "aggregate"
    _write_csv(aggregate_dir / "branch_results.csv", rows)
    _write_csv(aggregate_dir / "training_curves_long.csv", curves)
    _write_csv(aggregate_dir / "advantage_snapshot_curves_long.csv", snapshot_curves)
    _write_csv(aggregate_dir / "control_summary.csv", summaries)
    _write_csv(aggregate_dir / "paired_vs_pos_b0.csv", paired)
    audit = {
        "status": "PASS",
        "experiment_id": EXPERIMENT_ID,
        "profile_id": PROFILE_ID,
        "raw_complete": True,
        "branch_count_observed": len(rows),
        "expected_branch_count": EXPECTED_BRANCHES,
        "controls": list(CONTROL_ORDER),
        "baselines": list(BASELINES),
        "seeds": list(SEEDS),
        "held_out_seeds_touched": False,
        "all_replay_samples_retained": True,
        "advantage_shift_before_positive_only_mask": True,
        "negative_shifted_advantage_actor_contribution": 0.0,
        "fixed_baseline_units": "raw_trajectory_snapshot_gae",
        "complete_snapshot_series_retained": True,
        "task_performance_collapse_status": "not_adjudicated_no_registered_threshold",
        "support_or_variance_boundary_status": "not_instrumented_in_this_pilot",
        "rollout_failure_count": sum(bool(row["rollout_failure_event"]) for row in rows),
        "nan_inf_numerical_failure_count": sum(
            bool(row["nan_inf_numerical_failure"]) for row in rows
        ),
        "fixed_500k_horizon_is_not_convergence": True,
        "steady_state_ranking_allowed": False,
        "formal_evidence_allowed": False,
    }
    _atomic_json(aggregate_dir / "terminal_audit.json", audit)
    summary = {
        "status": "PASS",
        "experiment_id": EXPERIMENT_ID,
        "profile_id": PROFILE_ID,
        "branch_count": len(rows),
        "control_count": len(summaries),
        "interpretation": (
            "500k Positive-only fixed-baseline response curve; finite-horizon pilot only, "
            "not convergence, steady-state ranking, or formal evidence"
        ),
        "files": {
            "branch_results": str(aggregate_dir / "branch_results.csv"),
            "training_curves": str(aggregate_dir / "training_curves_long.csv"),
            "advantage_snapshot_curves": str(
                aggregate_dir / "advantage_snapshot_curves_long.csv"
            ),
            "control_summary": str(aggregate_dir / "control_summary.csv"),
            "paired_vs_pos_b0": str(aggregate_dir / "paired_vs_pos_b0.csv"),
            "terminal_audit": str(aggregate_dir / "terminal_audit.json"),
        },
    }
    _atomic_json(aggregate_dir / "aggregate_summary.json", summary)
    _atomic_json(work / "RUN_SUMMARY.json", summary)
    return summary


def main() -> int:
    _, digest = _load_config(GRID)
    if COMMAND == "validate":
        payload = {
            "status": "PASS",
            "experiment_id": EXPERIMENT_ID,
            "profile_id": PROFILE_ID,
            "grid_sha256": digest,
            "dataset": DATASET_ID,
            "controls": list(CONTROL_ORDER),
            "baselines": list(BASELINES),
            "seeds": list(SEEDS),
            "expected_branches": EXPECTED_BRANCHES,
            "steps": STEPS,
            "evaluation_interval": EVAL_INTERVAL,
            "all_replay_samples_retained": True,
            "negative_shifted_advantage_actor_contribution": 0.0,
            "held_out_seeds_touched": False,
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    _install_profile()
    args = [
        "plan",
        "--contract",
        CONTRACT,
        "--run-spec",
        RUN_SPEC,
        "--grid",
        GRID,
        "--work-dir",
        str(WORK_ROOT),
        "--max-workers",
        MAX_WORKERS,
    ]
    suite.main(args)
    if COMMAND == "plan":
        return 0
    os.environ[FULL_RUN_ENV] = "1"
    args[0] = "run"
    args.append("--resume")
    suite.main(args)
    return 0


raise SystemExit(main())
PY
