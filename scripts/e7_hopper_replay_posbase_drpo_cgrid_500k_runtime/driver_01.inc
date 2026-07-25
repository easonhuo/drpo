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

EXPERIMENT_ID = "EXT-H-E7-SQEXP-GAE-HOPPER-REPLAY-POSBASE-DRPO-CGRID-500K-01"
PROFILE_ID = "hopper_replay_posbase_drpo_cgrid_500k_v1"
SCIENTIFIC_STATUS = "hopper_replay_positive_baseline_by_drpo_c_interaction_500k_pilot_only"
RUNNER_VERSION = "5.8.0-hopper-replay-posbase-drpo-cgrid-500k"
FULL_RUN_ENV = "DRPO_E7_HOPPER_REPLAY_POSBASE_DRPO_CGRID_500K_FULL_RUN"
DATASET_ID = "hopper-medium-replay-v2"
SEEDS = (200, 201, 202, 203, 208, 209, 210, 211, 212)
HELD_OUT_SEEDS = (204, 205, 206, 207)
BASELINES = (0.0, 4.0, 5.0, 6.0)
REMOTENESS_SCALES = (0.06, 0.08, 0.10, 0.125)
STEPS = 500_000
EVAL_INTERVAL = 20_000
EVAL_EPISODES = 10
BATCH_SIZE = 512
ACTOR_LR = 3.0e-4
CRITIC_LR = 3.0e-4
REFERENCE_DISTANCE = 2.0
REMOTENESS_THRESHOLD = 0.0
TAPER_LAMBDA = 1.0
EXPECTED_CONTROLS = 20
EXPECTED_BRANCHES = 180
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


def _label(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def _positive_control_id(baseline: float) -> str:
    return f"po_b{int(baseline)}"


def _drpo_control_id(baseline: float, scale: float) -> str:
    return f"drpo_b{int(baseline)}_c{_label(scale)}"


def _control_specs() -> list[dict[str, Any]]:
    controls = [
        {
            "id": _positive_control_id(baseline),
            "weight_method": "positive_only",
            "advantage_baseline": baseline,
            "remoteness_scale": None,
        }
        for baseline in BASELINES
    ]
    controls.extend(
        {
            "id": _drpo_control_id(baseline, scale),
            "weight_method": "thresholded_exponential",
            "advantage_baseline": baseline,
            "remoteness_scale": scale,
        }
        for baseline in BASELINES
        for scale in REMOTENESS_SCALES
    )
    if len(controls) != EXPECTED_CONTROLS or len({item["id"] for item in controls}) != EXPECTED_CONTROLS:
        raise RuntimeError("Positive-only/DRPO control matrix is not exact")
    return controls


CONTROL_SPECS = _control_specs()
CONTROL_ORDER = tuple(item["id"] for item in CONTROL_SPECS)
CONTROL_BY_ID = {item["id"]: item for item in CONTROL_SPECS}


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
