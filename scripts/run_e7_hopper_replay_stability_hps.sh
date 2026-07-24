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

GRID="${E7_HR_STAB_GRID:-configs/e7_hopper_replay_stability_hps.json}"
CONTRACT="${E7_CANONICAL_CONTRACT:-/root/d4rl2/configs/e7_canonical_contract_9task.json}"
RUN_SPEC="${E7_CANONICAL_RUN_SPEC:-/root/d4rl2/configs/e7_canonical_9task_full_grid_run_spec_v1.json}"
WORK_DIR="${E7_HR_STAB_WORK_DIR:-outputs/e7/hopper_replay_stability_hps_001}"
MAX_WORKERS="${E7_HR_STAB_MAX_WORKERS:-48}"
RUNTIME_DIR="${WORK_DIR}/.hps_runtime"
BOOTSTRAP_WRAPPER="${RUNTIME_DIR}/bootstrap_wrapper.py"

if [[ -n "$(git status --porcelain)" ]]; then
  echo "refusing to run from a dirty checkout" >&2
  exit 2
fi
if [[ ! -f "${GRID}" ]]; then
  echo "missing Hopper replay stability grid: ${GRID}" >&2
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
if ! [[ "${MAX_WORKERS}" =~ ^[1-9][0-9]*$ ]] || (( MAX_WORKERS > 48 )); then
  echo "E7_HR_STAB_MAX_WORKERS must be an integer in [1,48]" >&2
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

EXPERIMENT_ID = "EXT-H-E7-SQEXP-GAE-HOPPER-REPLAY-STAB-HPS-01"
PROFILE_ID = "hopper_replay_stability_hps_v1"
BASE_LR = 3.0e-4
ALLOWED_MULTIPLIERS = {1.0, 0.5, 0.25}
ALLOWED_BATCH_SIZES = {256, 512}


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _optimizer_lrs(optimizer: Any) -> list[float]:
    groups = getattr(optimizer, "param_groups", None)
    if not isinstance(groups, list) or not groups:
        raise RuntimeError("canonical optimizer has no parameter groups")
    values = [float(group["lr"]) for group in groups]
    if not all(math.isfinite(value) and value > 0.0 for value in values):
        raise RuntimeError("canonical optimizer has invalid learning rates")
    return values


def _install_actor_lr_multiplier(
    module: Any,
    target_class: str,
    multiplier: float,
) -> dict[str, Any]:
    if multiplier not in ALLOWED_MULTIPLIERS:
        raise ValueError(f"unsupported actor_lr_multiplier={multiplier}")
    original_class = getattr(module, target_class)
    original_init = original_class.__init__
    state: dict[str, Any] = {"applied_count": 0}

    def wrapped_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        if state["applied_count"] != 0:
            raise RuntimeError("canonical agent was instantiated more than once")
        actor_before = _optimizer_lrs(self.a_opt)
        critic_before = _optimizer_lrs(self.c_opt)
        if not all(math.isclose(value, BASE_LR, rel_tol=0.0, abs_tol=1e-12) for value in actor_before):
            raise RuntimeError(f"canonical actor LR changed: {actor_before}")
        if not all(math.isclose(value, BASE_LR, rel_tol=0.0, abs_tol=1e-12) for value in critic_before):
            raise RuntimeError(f"canonical critic LR changed: {critic_before}")
        for group in self.a_opt.param_groups:
            group["lr"] = float(group["lr"]) * multiplier
        actor_after = _optimizer_lrs(self.a_opt)
        critic_after = _optimizer_lrs(self.c_opt)
        expected_actor = BASE_LR * multiplier
        if not all(
            math.isclose(value, expected_actor, rel_tol=0.0, abs_tol=1e-12)
            for value in actor_after
        ):
            raise RuntimeError("actor learning-rate multiplier was not applied exactly")
        if critic_after != critic_before:
            raise RuntimeError("critic learning rate changed while scaling the actor")
        state.update(
            applied_count=1,
            actor_lr_multiplier=multiplier,
            canonical_base_lr=BASE_LR,
            actor_lr_before=actor_before,
            actor_lr_after=actor_after,
            critic_lr_before=critic_before,
            critic_lr_after=critic_after,
            critic_lr_unchanged=True,
        )

    original_class.__init__ = wrapped_init
    return state


def _self_test() -> int:
    class Agent:
        def __init__(self) -> None:
            actor = torch.nn.Parameter(torch.tensor(1.0))
            critic = torch.nn.Parameter(torch.tensor(1.0))
            self.a_opt = torch.optim.Adam([actor], lr=BASE_LR)
            self.c_opt = torch.optim.Adam([critic], lr=BASE_LR)

    from types import SimpleNamespace

    module = SimpleNamespace(Agent=Agent)
    state = _install_actor_lr_multiplier(module, "Agent", 0.5)
    module.Agent()
    if state.get("applied_count") != 1:
        raise RuntimeError("actor LR self-test did not instantiate exactly one agent")
    if state.get("actor_lr_after") != [BASE_LR * 0.5]:
        raise RuntimeError("actor LR self-test produced the wrong actor LR")
    if state.get("critic_lr_after") != [BASE_LR]:
        raise RuntimeError("actor LR self-test changed the critic LR")
    print(json.dumps({"status": "PASS", "optimizer_control": state}, sort_keys=True))
    return 0


def _run_bootstrap(argv: list[str]) -> int:
    from drpo import e7_squared_exp_night as suite
    from drpo import e7_squared_exp_night_bootstrap as bootstrap

    suite.GAE_EXPERIMENT_ID = EXPERIMENT_ID
    suite.TUNING_PROFILE_ID = PROFILE_ID

    try:
        branch_index = argv.index("--branch-config") + 1
    except (ValueError, IndexError) as exc:
        raise ValueError("bootstrap wrapper requires --branch-config") from exc
    branch_config_path = Path(argv[branch_index]).expanduser().resolve()
    branch = json.loads(branch_config_path.read_text(encoding="utf-8"))
    if branch.get("experiment_id") != EXPERIMENT_ID or branch.get("profile_id") != PROFILE_ID:
        raise ValueError("Hopper replay stability bootstrap identity mismatch")
    values = branch.get("template_values", {})
    batch_size = int(values.get("batch_size", -1))
    multiplier = float(values.get("actor_lr_multiplier", float("nan")))
    if batch_size not in ALLOWED_BATCH_SIZES:
        raise ValueError(f"unsupported stability batch_size={batch_size}")
    if multiplier not in ALLOWED_MULTIPLIERS:
        raise ValueError(f"unsupported stability actor_lr_multiplier={multiplier}")

    original_flag_value = suite._flag_value  # noqa: SLF001

    def compatible_flag_value(tokens: list[str], flag: str) -> str:
        value = original_flag_value(tokens, flag)
        return "256" if flag == "--batch" else value

    suite._flag_value = compatible_flag_value  # type: ignore[assignment]  # noqa: SLF001

    original_provider = bootstrap.TrajectorySnapshotAdvantage

    class BatchAwareProvider(original_provider):
        def __init__(self, replay: Any, estimator: str, batch_size_arg: int = 256) -> None:
            del batch_size_arg
            super().__init__(replay, estimator, batch_size=batch_size)

    bootstrap.TrajectorySnapshotAdvantage = BatchAwareProvider

    original_patch = bootstrap.patch_canonical_module
    state_holder: dict[str, Any] = {}

    def patched_module(*args: Any, **kwargs: Any) -> Any:
        result = original_patch(*args, **kwargs)
        module = args[0]
        contract = args[1]
        state_holder["optimizer"] = _install_actor_lr_multiplier(
            module,
            contract.target_class,
            multiplier,
        )
        return result

    bootstrap.patch_canonical_module = patched_module
    result = int(bootstrap.main(argv))
    state = state_holder.get("optimizer")
    if not isinstance(state, dict) or state.get("applied_count") != 1:
        raise RuntimeError("actor optimizer control was not applied exactly once")
    state = {
        **state,
        "batch_size": batch_size,
        "trajectory_snapshot_batch_size": batch_size,
        "profile_id": PROFILE_ID,
        "experiment_id": EXPERIMENT_ID,
    }
    optimizer_path = branch_config_path.parent / "OPTIMIZER_CONTROL.json"
    _atomic_json(optimizer_path, state)
    manifest_path = branch_config_path.parent / "branch_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["optimizer_control"] = state
    manifest["trajectory_snapshot_batch_size"] = batch_size
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
from drpo.e7_squared_exp_kernel import THRESHOLDED_FORMULA

EXPERIMENT_ID = "EXT-H-E7-SQEXP-GAE-HOPPER-REPLAY-STAB-HPS-01"
PROFILE_ID = "hopper_replay_stability_hps_v1"
SCIENTIFIC_STATUS = (
    "hopper_replay_joint_critic_gae_stability_hyperparameter_screening_pilot_only"
)
RUNNER_VERSION = "5.5.0-hopper-replay-stability-hps"
FULL_RUN_ENV = "DRPO_E7_HOPPER_REPLAY_STABILITY_HPS_FULL_RUN"
DATASET_ID = "hopper-medium-replay-v2"
PHASE_A = "phase_a"
PHASE_B = "phase_b"
EXPECTED_PHASE_A_BRANCHES = 48
EXPECTED_PHASE_B_BRANCHES = 10
EXPECTED_TOTAL_BRANCHES = 58
HELD_OUT_SEEDS = (204, 205, 206, 207)

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


def _std(values: Sequence[float]) -> float | None:
    return statistics.stdev(values) if len(values) >= 2 else None


def _config_id(c_value: float, lr_multiplier: float, batch_size: int) -> str:
    c_label = f"{c_value:g}".replace(".", "p")
    lr_label = f"{lr_multiplier:g}".replace(".", "p")
    return f"c{c_label}__alr{lr_label}__b{batch_size}"


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _load_config(path: str | Path) -> tuple[dict[str, Any], str]:
    source = Path(path)
    raw = json.loads(source.read_text(encoding="utf-8"))
    required = {
        "experiment_id": EXPERIMENT_ID,
        "profile_id": PROFILE_ID,
        "parent_experiment_id": "EXT-H-E7-BENCH-01",
        "predecessor_experiment_id": "EXT-H-E7-SQEXP-GAE-TASKC-MS-01",
        "run_kind": "pilot",
        "status": "not_run",
        "scientific_status": SCIENTIFIC_STATUS,
        "dataset": DATASET_ID,
        "held_out_seeds": list(HELD_OUT_SEEDS),
        "steps": suite.EXPECTED_STEPS,
        "evaluation_interval": 50_000,
        "evaluation_episodes": 10,
        "actor_update_mode": "a2c",
        "advantage_mode": "gae_lambda_0p95",
        "expected_total_new_branches": EXPECTED_TOTAL_BRANCHES,
        "screening_only": True,
        "formal_evidence_allowed": False,
    }
    changed = [key for key, value in required.items() if raw.get(key) != value]
    if changed:
        raise ValueError(f"Hopper replay stability grid changed: {changed}")
    phase_a = raw.get(PHASE_A, {})
    if phase_a != {
        "remoteness_scales": [0.08, 0.125],
        "actor_lr_multipliers": [1.0, 0.5, 0.25],
        "batch_sizes": [256, 512],
        "seeds": [200, 202, 203, 208],
        "expected_branches": EXPECTED_PHASE_A_BRANCHES,
        "selected_config_count": 2,
    }:
        raise ValueError("Phase A stability matrix changed")
    phase_b = raw.get(PHASE_B, {})
    if phase_b != {
        "seeds": [200, 201, 202, 203, 208],
        "fresh_rerun": True,
        "expected_branches": EXPECTED_PHASE_B_BRANCHES,
    }:
        raise ValueError("Phase B stability matrix changed")
    if set(phase_a["seeds"]) & set(HELD_OUT_SEEDS) or set(phase_b["seeds"]) & set(
        HELD_OUT_SEEDS
    ):
        raise ValueError("held-out seeds entered the stability matrix")
    weight = raw.get("weight_control", {})
    expected_weight = {
        "formula": THRESHOLDED_FORMULA,
        "coordinate": "normalized_squared_standardized_distance",
        "weight_at_zero": 1.0,
        "reference_distance": 2.0,
        "taper_lambda": 1.0,
        "remoteness_threshold": 0.0,
    }
    if weight != expected_weight:
        raise ValueError("Hopper replay stability taper contract changed")
    optimizer = raw.get("optimizer_control", {})
    if optimizer != {
        "optimizer_family": "Adam",
        "canonical_base_lr": 0.0003,
        "critic_lr_frozen": 0.0003,
        "actor_lr_formula": "canonical_base_lr * actor_lr_multiplier",
    }:
        raise ValueError("Hopper replay stability optimizer contract changed")
    return raw, sha256_file(source)


def _phase_configs(raw: Mapping[str, Any], phase: str) -> list[dict[str, Any]]:
    if phase == PHASE_A:
        phase_a = raw[PHASE_A]
        configs = [
            {
                "remoteness_scale": float(c_value),
                "actor_lr_multiplier": float(multiplier),
                "batch_size": int(batch_size),
            }
            for c_value in phase_a["remoteness_scales"]
            for multiplier in phase_a["actor_lr_multipliers"]
            for batch_size in phase_a["batch_sizes"]
        ]
        if len(configs) != 12:
            raise RuntimeError("Phase A must contain exactly twelve configurations")
        return configs
    manifest_path = WORK_ROOT / PHASE_A / "aggregate" / "phase_a_selection_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("experiment_id") != EXPERIMENT_ID or manifest.get("status") != "PASS":
        raise RuntimeError("Phase A selection manifest identity/status mismatch")
    configs = manifest.get("selected_configs")
    if not isinstance(configs, list) or len(configs) != 2:
        raise RuntimeError("Phase A must select exactly two configurations")
    normalized = [
        {
            "remoteness_scale": float(item["remoteness_scale"]),
            "actor_lr_multiplier": float(item["actor_lr_multiplier"]),
            "batch_size": int(item["batch_size"]),
        }
        for item in configs
    ]
    config_ids = {
        _config_id(
            float(item["remoteness_scale"]),
            float(item["actor_lr_multiplier"]),
            int(item["batch_size"]),
        )
        for item in normalized
    }
    if len(config_ids) != 2:
        raise RuntimeError("Phase B selected configurations are not unique")
    return normalized


def _phase_seeds(raw: Mapping[str, Any], phase: str) -> tuple[int, ...]:
    return tuple(int(value) for value in raw[phase]["seeds"])


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


def _install_phase_profile(
    raw: Mapping[str, Any],
    phase: str,
    configs: list[dict[str, Any]],
) -> None:
    _reset_profile_hooks()
    seeds = _phase_seeds(raw, phase)
    expected_branches = (
        EXPECTED_PHASE_A_BRANCHES if phase == PHASE_A else EXPECTED_PHASE_B_BRANCHES
    )
    suite.GAE_EXPERIMENT_ID = EXPERIMENT_ID
    suite.TUNING_PROFILE_ID = PROFILE_ID
    suite.TUNING_SEEDS = seeds
    suite.TUNING_EXPECTED_BRANCHES = expected_branches
    suite.TUNING_RUNNER_VERSION = RUNNER_VERSION
    suite.TUNING_FULL_RUN_ENV = FULL_RUN_ENV
    agg.GAE_EXPERIMENT_ID = EXPERIMENT_ID
    agg.TUNING_PROFILE_ID = PROFILE_ID
    agg.TUNING_SEEDS = seeds

    def is_tuning() -> bool:
        return (
            suite._ACTIVE_EXPERIMENT_ID == EXPERIMENT_ID  # noqa: SLF001
            and suite._ACTIVE_PROFILE_ID == PROFILE_ID  # noqa: SLF001
        )

    suite._is_tuning = is_tuning  # type: ignore[assignment]  # noqa: SLF001
    suite._is_p3 = lambda: False  # type: ignore[assignment]  # noqa: SLF001
    suite.active_scientific_status = lambda: SCIENTIFIC_STATUS
    suite.active_expected_branch_count = lambda: expected_branches

    def configure_execution(
        grid_path: str | Path,
        *,
        liveness_pair: bool = False,
        liveness_steps: int | None = None,
    ) -> None:
        if liveness_pair or liveness_steps is not None:
            raise ValueError("the Hopper replay stability profile has no liveness submatrix")
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
        loaded["seeds"] = list(seeds)
        argv = [str(value) for value in loaded["trainer_argv_template"]]
        if suite._flag_value(argv, "--batch") != "256":  # noqa: SLF001
            raise ValueError("canonical source batch changed before stability injection")
        if suite._flag_value(argv, "--lr") != "0.0003":  # noqa: SLF001
            raise ValueError("canonical source learning rate changed")
        argv[argv.index("--batch") + 1] = "{batch_size}"
        loaded["trainer_argv_template"] = argv
        return loaded, digest

    suite.load_run_spec = load_run_spec

    def gae_branches(
        run_spec: Mapping[str, Any],
        grid: Mapping[str, Any],
    ) -> list[base.Branch]:
        del grid
        datasets = [base.DatasetSpec.from_mapping(item) for item in run_spec["datasets"]]
        if len(datasets) != 1 or datasets[0].id != DATASET_ID:
            raise ValueError("stability profile must contain only hopper-medium-replay-v2")
        dataset = datasets[0]
        branches: list[base.Branch] = []
        for config in configs:
            c_value = float(config["remoteness_scale"])
            multiplier = float(config["actor_lr_multiplier"])
            batch_size = int(config["batch_size"])
            config_id = _config_id(c_value, multiplier, batch_size)
            coefficient = 1.0 / c_value
            for seed in seeds:
                branches.append(
                    base.Branch(
                        branch_id=(
                            f"{dataset.id}__{phase}__seed{seed}__{config_id}__"
                            "gae__a2c__steps1m"
                        ),
                        branch_kind="injected",
                        dataset=dataset,
                        seed=seed,
                        template_values={
                            "steps": str(suite.EXPECTED_STEPS),
                            "stage": phase,
                            "actor_update_mode": "a2c",
                            "advantage_estimator": "gae",
                            "weight_method": "thresholded_exponential",
                            "weight_at_zero": "1",
                            "exp_coefficient": f"{coefficient:.17g}",
                            "reference_distance": "2",
                            "remoteness_threshold": "0",
                            "remoteness_scale": f"{c_value:.17g}",
                            "taper_lambda": "1",
                            "actor_lr_multiplier": f"{multiplier:.17g}",
                            "batch_size": str(batch_size),
                            "config_id": config_id,
                            "diagnostics_interval": str(suite.DIAGNOSTICS_INTERVAL),
                            "sampled_values_per_update": str(
                                suite.SAMPLED_VALUES_PER_UPDATE
                            ),
                            "execution_mode": "full",
                        },
                        negative_control=None,
                    )
                )
        if len(branches) != expected_branches or len(
            {branch.branch_id for branch in branches}
        ) != expected_branches:
            raise RuntimeError(f"{phase} branch matrix is not exact")
        if {branch.seed for branch in branches} != set(seeds):
            raise RuntimeError(f"{phase} seed set changed")
        if {branch.seed for branch in branches} & set(HELD_OUT_SEEDS):
            raise RuntimeError("held-out seeds entered the stability branch matrix")
        return branches

    suite._gae_branches = gae_branches  # type: ignore[assignment]  # noqa: SLF001

    def branch_command(**kwargs: Any) -> tuple[list[str], dict[str, Any]]:
        command, branch_config = ORIGINAL_BRANCH_COMMAND(**kwargs)
        if command[1:3] != ["-m", "drpo.e7_squared_exp_night_bootstrap"]:
            raise RuntimeError("unexpected canonical bootstrap command shape")
        command = [command[0], str(WRAPPER_PATH), *command[3:]]
        branch_config["profile_id"] = PROFILE_ID
        branch_config["stability_phase"] = phase
        branch_dir = Path(kwargs["branch_dir"])
        base.atomic_write_json(branch_dir / "branch_config.json", branch_config)
        return command, branch_config

    suite.branch_command = branch_command
    suite.aggregate_results = lambda work_dir: _aggregate_phase(
        Path(work_dir), raw, phase, configs, seeds
    )


def _ranking_key(row: Mapping[str, Any]) -> tuple[float, ...]:
    return (
        float(row["late_median"]),
        float(row["worst2_late_mean"]),
        float(row["final_median"]),
        -float(row["mean_best_to_final_drop"]),
        -float(row["actor_lr_multiplier"]),
        float(row["batch_size"]),
        -float(row["remoteness_scale"]),
    )


def _aggregate_phase(
    work: Path,
    raw: Mapping[str, Any],
    phase: str,
    configs: list[dict[str, Any]],
    seeds: tuple[int, ...],
) -> dict[str, Any]:
    branch_dirs, experiment_id = agg._branch_dirs(work)  # noqa: SLF001
    expected = EXPECTED_PHASE_A_BRANCHES if phase == PHASE_A else EXPECTED_PHASE_B_BRANCHES
    if experiment_id != EXPERIMENT_ID or len(branch_dirs) != expected:
        raise RuntimeError(f"{phase} aggregate identity or branch count mismatch")
    rows: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    for branch_dir in branch_dirs:
        row = agg._gae_branch_row(branch_dir)  # noqa: SLF001
        branch = json.loads((branch_dir / "branch_config.json").read_text())
        manifest = json.loads((branch_dir / "branch_manifest.json").read_text())
        values = branch["template_values"]
        optimizer = manifest.get("optimizer_control")
        if not isinstance(optimizer, dict) or optimizer.get("applied_count") != 1:
            raise RuntimeError(f"optimizer control missing for {branch_dir.name}")
        multiplier = float(values["actor_lr_multiplier"])
        batch_size = int(values["batch_size"])
        if not math.isclose(
            float(optimizer["actor_lr_after"][0]),
            0.0003 * multiplier,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RuntimeError(f"actor LR audit failed for {branch_dir.name}")
        if optimizer.get("critic_lr_unchanged") is not True or any(
            not math.isclose(float(value), 0.0003, rel_tol=0.0, abs_tol=1e-12)
            for value in optimizer.get("critic_lr_after", [])
        ):
            raise RuntimeError(f"critic LR audit failed for {branch_dir.name}")
        if int(optimizer.get("batch_size", -1)) != batch_size:
            raise RuntimeError(f"batch audit failed for {branch_dir.name}")
        if int(row["snapshot_refresh_interval"]) != batch_size:
            raise RuntimeError(
                f"trajectory-snapshot refresh interval mismatch for {branch_dir.name}"
            )
        if int(manifest.get("trajectory_snapshot_batch_size", -1)) != batch_size:
            raise RuntimeError(
                f"trajectory-snapshot batch provenance mismatch for {branch_dir.name}"
            )
        row.update(
            stability_phase=phase,
            config_id=str(values["config_id"]),
            actor_lr_multiplier=multiplier,
            actor_lr=0.0003 * multiplier,
            critic_lr=0.0003,
            batch_size=batch_size,
        )
        rows.append(row)
        summary_path = agg._only(  # noqa: SLF001
            (branch_dir / "trainer_output").glob("*_summary.json"),
            "trainer summary",
        )
        steps, scores = agg._read_history(json.loads(summary_path.read_text()))  # noqa: SLF001
        for step, score in zip(steps, scores, strict=True):
            curves.append(
                {
                    "phase": phase,
                    "config_id": values["config_id"],
                    "remoteness_scale": float(values["remoteness_scale"]),
                    "actor_lr_multiplier": multiplier,
                    "batch_size": batch_size,
                    "seed": int(branch["seed"]),
                    "step": step,
                    "normalized_return": score,
                }
            )

    observed_configs = {str(row["config_id"]) for row in rows}
    expected_configs = {
        _config_id(
            float(item["remoteness_scale"]),
            float(item["actor_lr_multiplier"]),
            int(item["batch_size"]),
        )
        for item in configs
    }
    if observed_configs != expected_configs:
        raise RuntimeError(f"{phase} configuration set changed")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["config_id"])].append(row)
    summaries: list[dict[str, Any]] = []
    for config_id, values in sorted(grouped.items()):
        observed_seeds = tuple(sorted(int(row["seed"]) for row in values))
        if observed_seeds != tuple(sorted(seeds)):
            raise RuntimeError(f"{phase} seed set changed for {config_id}")
        late = [float(row["late_window_mean_800k_1m"]) for row in values]
        final = [float(row["final_score"]) for row in values]
        drops = [float(row["best_to_final_drop"]) for row in values]
        ordered = sorted(
            values,
            key=lambda row: (
                float(row["late_window_mean_800k_1m"]),
                float(row["final_score"]),
                -int(row["seed"]),
            ),
            reverse=True,
        )
        top3 = ordered[: min(3, len(ordered))]
        sorted_late = sorted(late)
        summary = {
            "phase": phase,
            "config_id": config_id,
            "remoteness_scale": float(values[0]["remoteness_scale"]),
            "actor_lr_multiplier": float(values[0]["actor_lr_multiplier"]),
            "actor_lr": float(values[0]["actor_lr"]),
            "critic_lr": 0.0003,
            "batch_size": int(values[0]["batch_size"]),
            "seeds": list(observed_seeds),
            "late_mean": _mean(late),
            "late_std": _std(late),
            "late_median": statistics.median(late),
            "worst2_late_mean": _mean(sorted_late[:2]),
            "final_mean": _mean(final),
            "final_std": _std(final),
            "final_median": statistics.median(final),
            "mean_best_to_final_drop": _mean(drops),
            "top3_of5_late_mean": (
                _mean([float(row["late_window_mean_800k_1m"]) for row in top3])
                if phase == PHASE_B
                else None
            ),
            "top3_seed_ids": (
                [int(row["seed"]) for row in top3] if phase == PHASE_B else None
            ),
            "nan_inf_numerical_failures": sum(
                bool(row["nan_inf_numerical_failure"]) for row in values
            ),
            "rollout_failures": sum(bool(row["rollout_failure_event"]) for row in values),
        }
        summaries.append(summary)

    ranked = sorted(summaries, key=_ranking_key, reverse=True)
    for rank, row in enumerate(ranked, start=1):
        row["stability_rank"] = rank
    aggregate_dir = work / "aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    agg._write_csv(aggregate_dir / "branch_results.csv", rows)  # noqa: SLF001
    agg._write_csv(aggregate_dir / "training_curves_long.csv", curves)  # noqa: SLF001
    agg._write_csv(aggregate_dir / "config_summary.csv", ranked)  # noqa: SLF001

    if phase == PHASE_A:
        selected = [
            {
                "config_id": row["config_id"],
                "remoteness_scale": row["remoteness_scale"],
                "actor_lr_multiplier": row["actor_lr_multiplier"],
                "batch_size": row["batch_size"],
                "phase_a_rank": row["stability_rank"],
            }
            for row in ranked[:2]
        ]
        manifest_name = "phase_a_selection_manifest.json"
        manifest_payload = {
            "status": "PASS",
            "experiment_id": EXPERIMENT_ID,
            "profile_id": PROFILE_ID,
            "phase": PHASE_A,
            "selection_rule": raw["selection_protocol"]["ranking"],
            "selected_configs": selected,
            "ranked_configs": ranked,
            "held_out_seeds_touched": False,
        }
    else:
        manifest_name = "phase_b_confirmation_manifest.json"
        manifest_payload = {
            "status": "PASS",
            "experiment_id": EXPERIMENT_ID,
            "profile_id": PROFILE_ID,
            "phase": PHASE_B,
            "selection_rule": raw["selection_protocol"]["ranking"],
            "confirmed_configs": ranked,
            "selected_config": ranked[0],
            "held_out_seeds_touched": False,
        }
    _atomic_json(aggregate_dir / manifest_name, manifest_payload)
    audit = {
        "status": "PASS",
        "experiment_id": EXPERIMENT_ID,
        "profile_id": PROFILE_ID,
        "phase": phase,
        "branch_count_observed": len(rows),
        "expected_branch_count": expected,
        "config_count": len(summaries),
        "seeds": list(seeds),
        "critic_lr_unchanged": True,
        "actor_lr_multiplier_applied": True,
        "batch_size_applied": True,
        "held_out_seeds_touched": False,
        "task_performance_collapse_status": "not_adjudicated_no_registered_threshold",
        "support_or_variance_boundary_status": "not_instrumented_in_this_pilot",
        "rollout_failure_count": sum(bool(row["rollout_failure_event"]) for row in rows),
        "nan_inf_numerical_failure_count": sum(
            bool(row["nan_inf_numerical_failure"]) for row in rows
        ),
        "fixed_horizon_is_not_convergence": True,
        "formal_evidence_allowed": False,
    }
    _atomic_json(aggregate_dir / "terminal_audit.json", audit)
    summary = {
        "status": "PASS",
        "experiment_id": EXPERIMENT_ID,
        "profile_id": PROFILE_ID,
        "phase": phase,
        "branch_count": len(rows),
        "config_count": len(summaries),
        "selection_manifest": str(aggregate_dir / manifest_name),
        "terminal_audit": str(aggregate_dir / "terminal_audit.json"),
    }
    _atomic_json(aggregate_dir / "aggregate_summary.json", summary)
    return summary


def _run_phase(
    raw: Mapping[str, Any],
    phase: str,
    configs: list[dict[str, Any]],
    command: str,
) -> None:
    _install_phase_profile(raw, phase, configs)
    phase_work = WORK_ROOT / phase
    args = [
        "plan",
        "--contract",
        CONTRACT,
        "--run-spec",
        RUN_SPEC,
        "--grid",
        GRID,
        "--work-dir",
        str(phase_work),
        "--max-workers",
        MAX_WORKERS,
    ]
    suite.main(args)
    if command == "plan":
        return
    os.environ[FULL_RUN_ENV] = "1"
    args[0] = "run"
    args.append("--resume")
    suite.main(args)


def _finalize(raw: Mapping[str, Any]) -> None:
    phase_a_audit = json.loads(
        (WORK_ROOT / PHASE_A / "aggregate" / "terminal_audit.json").read_text()
    )
    phase_b_audit = json.loads(
        (WORK_ROOT / PHASE_B / "aggregate" / "terminal_audit.json").read_text()
    )
    confirmation = json.loads(
        (
            WORK_ROOT
            / PHASE_B
            / "aggregate"
            / "phase_b_confirmation_manifest.json"
        ).read_text()
    )
    if phase_a_audit.get("status") != "PASS" or phase_b_audit.get("status") != "PASS":
        raise RuntimeError("one or more stability phases failed terminal audit")
    aggregate_dir = WORK_ROOT / "aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    selected = confirmation["selected_config"]
    _atomic_json(aggregate_dir / "final_selected_config.json", selected)
    audit = {
        "status": "PASS",
        "experiment_id": EXPERIMENT_ID,
        "profile_id": PROFILE_ID,
        "raw_complete": True,
        "phase_a_branch_count": int(phase_a_audit["branch_count_observed"]),
        "phase_b_branch_count": int(phase_b_audit["branch_count_observed"]),
        "branch_count_observed": int(phase_a_audit["branch_count_observed"])
        + int(phase_b_audit["branch_count_observed"]),
        "expected_branch_count": EXPECTED_TOTAL_BRANCHES,
        "phase_b_fresh_rerun": True,
        "selected_config": selected,
        "critic_lr_unchanged": True,
        "actor_lr_multiplier_applied": True,
        "batch_size_applied": True,
        "held_out_seeds_touched": False,
        "task_performance_collapse_status": "not_adjudicated_no_registered_threshold",
        "support_or_variance_boundary_status": "not_instrumented_in_this_pilot",
        "rollout_failure_count": int(phase_a_audit["rollout_failure_count"])
        + int(phase_b_audit["rollout_failure_count"]),
        "nan_inf_numerical_failure_count": int(
            phase_a_audit["nan_inf_numerical_failure_count"]
        )
        + int(phase_b_audit["nan_inf_numerical_failure_count"]),
        "fixed_horizon_is_not_convergence": True,
        "steady_state_ranking_allowed": False,
        "formal_evidence_allowed": False,
    }
    if audit["branch_count_observed"] != EXPECTED_TOTAL_BRANCHES:
        raise RuntimeError("combined stability branch count is not 58")
    _atomic_json(aggregate_dir / "terminal_audit.json", audit)
    summary = {
        "status": "PASS",
        "experiment_id": EXPERIMENT_ID,
        "profile_id": PROFILE_ID,
        "branch_count": EXPECTED_TOTAL_BRANCHES,
        "selected_config": selected,
        "files": {
            "phase_a_selection": str(
                WORK_ROOT / PHASE_A / "aggregate" / "phase_a_selection_manifest.json"
            ),
            "phase_b_confirmation": str(
                WORK_ROOT
                / PHASE_B
                / "aggregate"
                / "phase_b_confirmation_manifest.json"
            ),
            "final_selected_config": str(aggregate_dir / "final_selected_config.json"),
            "terminal_audit": str(aggregate_dir / "terminal_audit.json"),
        },
    }
    _atomic_json(WORK_ROOT / "RUN_SUMMARY.json", summary)
    _atomic_json(aggregate_dir / "aggregate_summary.json", summary)


def main() -> int:
    raw, digest = _load_config(GRID)
    if COMMAND == "validate":
        payload = {
            "status": "PASS",
            "experiment_id": EXPERIMENT_ID,
            "profile_id": PROFILE_ID,
            "grid_sha256": digest,
            "phase_a_branches": EXPECTED_PHASE_A_BRANCHES,
            "phase_b_branches": EXPECTED_PHASE_B_BRANCHES,
            "total_branches": EXPECTED_TOTAL_BRANCHES,
            "held_out_seeds_touched": False,
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    phase_a_configs = _phase_configs(raw, PHASE_A)
    _run_phase(raw, PHASE_A, phase_a_configs, COMMAND)
    planned = {
        "experiment_id": EXPERIMENT_ID,
        "profile_id": PROFILE_ID,
        "phase_a_branches": EXPECTED_PHASE_A_BRANCHES,
        "phase_b_branches_after_selection": EXPECTED_PHASE_B_BRANCHES,
        "expected_total_branches": EXPECTED_TOTAL_BRANCHES,
        "phase_b_fresh_rerun": True,
    }
    _atomic_json(WORK_ROOT / "PLANNED_PHASES.json", planned)
    if COMMAND == "plan":
        return 0
    phase_b_configs = _phase_configs(raw, PHASE_B)
    _run_phase(raw, PHASE_B, phase_b_configs, COMMAND)
    _finalize(raw)
    return 0


raise SystemExit(main())
PY
