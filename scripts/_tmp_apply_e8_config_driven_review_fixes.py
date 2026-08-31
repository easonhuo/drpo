from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise SystemExit(
            f"{path}: expected one replacement target, found {count}: {old[:120]!r}"
        )
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


py = ROOT / "src/drpo/e8_multitask_exp_tuning.py"
runner = ROOT / "scripts/run_e8_multitask_exp_coldstart.sh"
bootstrap = ROOT / "scripts/bootstrap_e8_multitask_exp_coldstart.sh"
tests = ROOT / "tests/test_e8_multitask_p0.py"

replace_once(
    py,
    "PAPER_SEED_OFFSETS = (4000, 5000)\n\n\n@dataclass(frozen=True)\nclass Cell:",
    '''PAPER_SEED_OFFSETS = (4000, 5000)
COUNTDOWN_DIAGNOSTIC_SENTINELS = (
    0.105360516,
    0.430782916,
    0.916290732,
    1.897119985,
    2.302585093,
    2.995732274,
)
LOCKED_COUNTDOWN_COEFFICIENTS = frozenset(
    PAPER_ROUND1_COEFFICIENTS + PAPER_EXTENSION_COEFFICIENTS
)
FROZEN_COLDSTART_SWEEP_IDENTITIES: dict[str, dict[str, Any]] = {
    COLDSTART_EXPERIMENT_ID: {
        "parameterization": "paper_coefficient_c",
        "countdown_seed_offsets": (4000, 5000),
        "countdown_include_positive_only": True,
        "transfer_positive_only_seed_offsets": (4000, 5000, 6000, 7000),
        "task_transfer_seed_offset": 4000,
        "tuning_seed": 4000,
        "expected_cells": 208,
        "include_global_endpoint": False,
        "countdown_values": COUNTDOWN_DIAGNOSTIC_SENTINELS,
        "task_grid_hashes": {
            "word_sorting": "d24edbd6099f1d4f081318b305e62b39834db7ab94af6b4279d706f94e8d6de3",
            "spiral_matrix": "805966b9e3e1774e748d1d96ca64667e23276782681e15c1e072e5536a02199a",
            "mini_sudoku": "2340c4729b70ae5bafb7b6bf07049f38056b18368749bba3b967b4bbd950e29c",
            "maze": "399732f8572faf670a0486cd5f838d3956bfa56cabbb5ae79b8521fbbfc45d33",
            "word_ladder": "4fff6be5b923b9071ae0ee949e734087330463072d9be02015a661e51a2689e4",
            "knights_knaves": "37f675c933e909ac1ca8a6464c8ed72497758b31d1c8a82c855cad10961c4cab",
            "graph_color": "65c02a38a4339888d2e485c277fa1668a5e0309fab857e3ae2b2b2c0dc862f9d",
            "wikisql": "e42e516dbbe0bfd0f9fd00f5dec22949f88503253a2235c8c81bd908af4779a0",
        },
    },
    LAMBDA_COMPLETION_EXPERIMENT_ID: {
        "parameterization": "paper_lambda_c1",
        "countdown_seed_offsets": (),
        "countdown_include_positive_only": True,
        "transfer_positive_only_seed_offsets": (8000, 9000),
        "task_transfer_seed_offset": 4000,
        "tuning_seed": 4000,
        "expected_cells": 199,
        "include_global_endpoint": False,
        "countdown_values": COUNTDOWN_DIAGNOSTIC_SENTINELS,
        "task_grid_hashes": {
            "word_sorting": "99b23e7907c99405e4986f0f8445fa9a1dae157be433a9a40546864a18165313",
            "spiral_matrix": "aa89682224279902104d19fe88782882897c14dd704c3f016cc1afeafd597f6a",
            "mini_sudoku": "7344872db672909c35d18debbbaa97da549cc1f0e37902d8073a8d1d45f465fd",
            "maze": "6862dfaa292a5cfcdfc0e5ed78c232d6c812f138a7073f1b5045b46ae2258562",
            "word_ladder": "99b23e7907c99405e4986f0f8445fa9a1dae157be433a9a40546864a18165313",
            "knights_knaves": "6862dfaa292a5cfcdfc0e5ed78c232d6c812f138a7073f1b5045b46ae2258562",
            "graph_color": "c703de6803fd89b0a3d9247e210214f96f1b1aa734db05dd11be1ba4b29dd75c",
            "wikisql": "1a56632b1143e245a5ac8607785220bf7380165a3314096e8fdf648aec23df5f",
        },
    },
    LAMBDA_CURVE_COMPLETION_EXPERIMENT_ID: {
        "parameterization": "paper_lambda_c1",
        "countdown_seed_offsets": (),
        "countdown_include_positive_only": True,
        "transfer_positive_only_seed_offsets": (),
        "task_transfer_seed_offset": 4000,
        "tuning_seed": 4000,
        "expected_cells": 140,
        "include_global_endpoint": False,
        "countdown_values": COUNTDOWN_DIAGNOSTIC_SENTINELS,
        "task_grid_hashes": {
            "word_sorting": "70e407c4531b74ac7665fcad26eebb58333a215ad5fd671477b1ee9c441ea62d",
            "spiral_matrix": "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
            "mini_sudoku": "bcab21853d90c917d038aa44f96e45ca66cd4511aee35515291092dbf2e78619",
            "maze": "2575a4b0d8b70346a1794f6bb03fa0c44b5ec78b08df65e4298fc48d68cae21d",
            "word_ladder": "70e407c4531b74ac7665fcad26eebb58333a215ad5fd671477b1ee9c441ea62d",
            "knights_knaves": "ac60739bf2dc004cd1a13949b6c340a06a345176f82812f6bd42ad297a01b9b4",
            "graph_color": "ebbbf01c8321f21b20fc79484f7cd01e1b195e6f137bf4fcb754048a94946b67",
            "wikisql": "622e405617b7fd6762fdd984f5e4ef2aadbd377b3998bef461caa941d4ba4673",
        },
    },
}


@dataclass(frozen=True)
class Cell:''',
)

replace_once(
    py,
    '''def experiment_id(config: Mapping[str, Any]) -> str:
    value = config.get("experiment_id")
    allowed = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-"
    if (
        not isinstance(value, str)
        or not 1 <= len(value) <= 128
        or not value[0].isalnum()
        or any(character not in allowed for character in value)
    ):
        raise ValueError(
            "experiment_id must match [A-Za-z0-9][A-Za-z0-9._-]{0,127}"
        )
    return value


def sweep_profile''',
    '''def experiment_id(config: Mapping[str, Any]) -> str:
    value = config.get("experiment_id")
    allowed = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-"
    if (
        not isinstance(value, str)
        or not 1 <= len(value) <= 128
        or not value[0].isalnum()
        or any(character not in allowed for character in value)
    ):
        raise ValueError(
            "experiment_id must match [A-Za-z0-9][A-Za-z0-9._-]{0,127}"
        )
    return value


def _validate_frozen_coldstart_sweep_identity(config: Mapping[str, Any]) -> None:
    expected = FROZEN_COLDSTART_SWEEP_IDENTITIES.get(experiment_id(config))
    if expected is None:
        return
    sweep = config["sweep"]
    observed = {
        "parameterization": str(sweep.get("parameterization", "")),
        "countdown_seed_offsets": tuple(
            _configured_seed(value, "Countdown seed offset")
            for value in sweep.get("countdown_seed_offsets", ())
        ),
        "countdown_include_positive_only": sweep.get(
            "countdown_include_positive_only", True
        ),
        "transfer_positive_only_seed_offsets": tuple(
            _configured_seed(value, "Transfer Positive-only seed offset")
            for value in sweep.get("transfer_positive_only_seed_offsets", ())
        ),
        "task_transfer_seed_offset": _configured_seed(
            sweep.get("task_transfer_seed_offset"), "Transfer Exp seed offset"
        ),
        "tuning_seed": _configured_seed(
            sweep.get("tuning_seed"), "Cold-start tuning seed"
        ),
        "expected_cells": int(sweep.get("expected_cells", -1)),
        "include_global_endpoint": sweep.get("include_global_endpoint", False),
        "countdown_values": _task_lambdas(config, "countdown"),
        "task_grid_hashes": dict(sweep.get("task_grid_hashes", {})),
    }
    if observed != expected:
        raise ValueError(
            f"Frozen cold-start experiment identity drifted for {experiment_id(config)}"
        )


def sweep_profile''',
)

replace_once(
    py,
    '''        countdown_sentinels = (
            0.105360516,
            0.430782916,
            0.916290732,
            1.897119985,
            2.302585093,
            2.995732274,
        )
        if _tuple_floats(
            sweep.get("countdown_sentinel_coefficients", ())
        ) != countdown_sentinels:
            raise ValueError("Countdown sentinel coefficients drifted")
        countdown_values = _task_lambdas(config, "countdown")
        if countdown_values and countdown_values != countdown_sentinels:
            raise ValueError(
                "Configured Countdown lambdas must be empty or equal the six diagnostic sentinels"
            )
        countdown_seeds = tuple(
            _configured_seed(value, "Countdown seed offset")
            for value in sweep.get("countdown_seed_offsets", ())
        )
        if len(countdown_seeds) != len(set(countdown_seeds)):
            raise ValueError("Countdown seed offsets must be unique")
        if countdown_seeds and not countdown_values:
            raise ValueError("Countdown seeds require configured Countdown lambda sentinels")

        transfer_positive_seeds = tuple(
            _configured_seed(value, "Transfer Positive-only seed offset")
            for value in sweep.get("transfer_positive_only_seed_offsets", ())
        )
''',
    '''        if _tuple_floats(
            sweep.get("countdown_sentinel_coefficients", ())
        ) != COUNTDOWN_DIAGNOSTIC_SENTINELS:
            raise ValueError("Countdown sentinel coefficients drifted")
        countdown_values = _task_lambdas(config, "countdown")
        if (
            not countdown_values
            or len(countdown_values) != len(set(countdown_values))
            or any(value not in LOCKED_COUNTDOWN_COEFFICIENTS for value in countdown_values)
        ):
            raise ValueError(
                "Countdown task_lambda must be a non-empty unique subset of the locked paper grids"
            )
        countdown_seeds = tuple(
            _configured_seed(value, "Countdown seed offset")
            for value in sweep.get("countdown_seed_offsets", ())
        )
        if len(countdown_seeds) != len(set(countdown_seeds)):
            raise ValueError("Countdown seed offsets must be unique")
        if any(value not in PAPER_SEED_OFFSETS for value in countdown_seeds):
            raise ValueError(
                "Countdown seed offsets must be a subset of the locked paper seed offsets"
            )
        countdown_include_positive_only = sweep.get(
            "countdown_include_positive_only", True
        )
        if not isinstance(countdown_include_positive_only, bool):
            raise ValueError("countdown_include_positive_only must be boolean")
        include_global_endpoint = sweep.get("include_global_endpoint", False)
        if not isinstance(include_global_endpoint, bool):
            raise ValueError("include_global_endpoint must be boolean")

        transfer_positive_seeds = tuple(
            _configured_seed(value, "Transfer Positive-only seed offset")
            for value in sweep.get("transfer_positive_only_seed_offsets", ())
        )
''',
)

replace_once(
    py,
    '''        expanded_cells = len(countdown_seeds) * (2 + len(countdown_values)) + sum(
            len(transfer_positive_seeds) + len(_task_lambdas(config, task))
            for task in active_transfer_tasks
        )
        if int(sweep.get("expected_cells", -1)) != expanded_cells:
            raise ValueError("Cold-start expected_cells must match the configured matrix")

        initialization = config.get("initialization", {})
''',
    '''        countdown_cells_per_seed = (
            1 + int(countdown_include_positive_only) + len(countdown_values)
        )
        expanded_cells = len(countdown_seeds) * countdown_cells_per_seed + sum(
            len(transfer_positive_seeds)
            + int(include_global_endpoint)
            + len(_task_lambdas(config, task))
            for task in active_transfer_tasks
        )
        if expanded_cells <= 0:
            raise ValueError("Cold-start sweep must schedule at least one scientific cell")
        if int(sweep.get("expected_cells", -1)) != expanded_cells:
            raise ValueError("Cold-start expected_cells must match the configured matrix")
        _validate_frozen_coldstart_sweep_identity(config)

        initialization = config.get("initialization", {})
''',
)

replace_once(
    py,
    '''        countdown_coefficients = _task_lambdas(config, "countdown")
        for seed_offset in tuple(int(value) for value in config["sweep"]["countdown_seed_offsets"]):
            cells.append(
                Cell("countdown", METHOD_POSITIVE_ONLY, None, seed_offset, "countdown_sentinel")
            )
            cells.append(
                Cell("countdown", METHOD_GLOBAL, 1.0, seed_offset, "countdown_sentinel", 0.0)
            )
''',
    '''        countdown_coefficients = _task_lambdas(config, "countdown")
        countdown_include_positive_only = bool(
            config["sweep"].get("countdown_include_positive_only", True)
        )
        include_global_endpoint = bool(config["sweep"].get("include_global_endpoint", False))
        for seed_offset in tuple(int(value) for value in config["sweep"]["countdown_seed_offsets"]):
            if countdown_include_positive_only:
                cells.append(
                    Cell(
                        "countdown",
                        METHOD_POSITIVE_ONLY,
                        None,
                        seed_offset,
                        "countdown_sentinel",
                    )
                )
            cells.append(
                Cell("countdown", METHOD_GLOBAL, 1.0, seed_offset, "countdown_sentinel", 0.0)
            )
''',
)

replace_once(
    py,
    '''            cells.extend(
                Cell(task, METHOD_POSITIVE_ONLY, None, seed_offset, "task_transfer")
                for seed_offset in positive_seeds
            )
            cells.extend(
                Cell(
                    task,
                    METHOD_EXPONENTIAL,
''',
    '''            cells.extend(
                Cell(task, METHOD_POSITIVE_ONLY, None, seed_offset, "task_transfer")
                for seed_offset in positive_seeds
            )
            if include_global_endpoint:
                cells.append(
                    Cell(task, METHOD_GLOBAL, 1.0, exp_seed, "task_transfer", 0.0)
                )
            cells.extend(
                Cell(
                    task,
                    METHOD_EXPONENTIAL,
''',
)

replace_once(
    runner,
    'EXPERIMENT_ID="${E8_COLDSTART_EXPERIMENT_ID:-EXT-C-E8-MULTITASK-EXP-COLDSTART-01}"\nCONFIG_PATH=',
    'EXPERIMENT_ID_OVERRIDE="${E8_COLDSTART_EXPERIMENT_ID:-}"\nEXPERIMENT_ID=""\nCONFIG_PATH=',
)

replace_once(
    runner,
    '''resolve_config_repo_path

# Source provenance follows the selected config, not an experiment-ID branch.
''',
    '''resolve_config_repo_path

resolve_experiment_id() {
  local matches=()
  mapfile -t matches < <(
    sed -nE 's/^experiment_id:[[:space:]]*([A-Za-z0-9][A-Za-z0-9._-]{0,127})[[:space:]]*$/\\1/p' "${CONFIG_PATH}"
  )
  [[ "${#matches[@]}" -eq 1 ]] || \
    fail "config must contain exactly one well-formed top-level experiment_id: ${CONFIG_REPO_PATH}"
  local config_experiment_id="${matches[0]}"
  if [[ -n "${EXPERIMENT_ID_OVERRIDE}" && "${EXPERIMENT_ID_OVERRIDE}" != "${config_experiment_id}" ]]; then
    fail "experiment_id mismatch: env=${EXPERIMENT_ID_OVERRIDE} config=${config_experiment_id}"
  fi
  EXPERIMENT_ID="${config_experiment_id}"
  export E8_COLDSTART_EXPERIMENT_ID="${EXPERIMENT_ID}"
}

resolve_experiment_id

# Source provenance follows the selected config, not an experiment-ID branch.
''',
)

replace_once(
    runner,
    '''liveness() {
  check_source
  activate_runtime
  CUDA_VISIBLE_DEVICES=0 LOCAL_RANK=0 run_module liveness \
    --task countdown \
    --lambda 0.916290732 \
    --base-model-path "${MODEL_DIR}"
}
''',
    '''liveness() {
  check_source
  activate_runtime
  local liveness_lambda
  liveness_lambda="$(python - "${CONFIG_PATH}" <<'PY_LIVENESS'
import math
import sys
from pathlib import Path

import yaml

config = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
values = [float(value) for value in config["sweep"]["task_lambda"]["countdown"]]
if not values:
    raise SystemExit("Countdown liveness anchor grid is empty")
preferred = 0.916290732
selected = next(
    (value for value in values if math.isclose(value, preferred, rel_tol=0.0, abs_tol=1.0e-15)),
    values[0],
)
print(f"{selected:.17g}")
PY_LIVENESS
)"
  CUDA_VISIBLE_DEVICES=0 LOCAL_RANK=0 run_module liveness \
    --task countdown \
    --lambda "${liveness_lambda}" \
    --base-model-path "${MODEL_DIR}"
}
''',
)

replace_once(
    bootstrap,
    'EXPERIMENT_ID="${E8_COLDSTART_EXPERIMENT_ID:-EXT-C-E8-MULTITASK-EXP-COLDSTART-01}"\nEXPECTED_REPOSITORY=',
    'EXPERIMENT_ID_OVERRIDE="${E8_COLDSTART_EXPERIMENT_ID:-}"\nEXPERIMENT_ID="${EXPERIMENT_ID_OVERRIDE:-UNRESOLVED_FROM_CONFIG}"\nEXPECTED_REPOSITORY=',
)

replace_once(
    bootstrap,
    '''  grep -Fqx "experiment_id=${EXPERIMENT_ID}" "${STATE_FILE}" || \
    fail "existing bootstrap state belongs to another experiment"
  grep -Fqx "mode=${MODE}" "${STATE_FILE}" || \
''',
    '''  state_experiment_id="$(
    sed -nE 's/^experiment_id=([A-Za-z0-9][A-Za-z0-9._-]{0,127})$/\\1/p' "${STATE_FILE}"
  )"
  [[ -n "${state_experiment_id}" ]] || fail "existing bootstrap state has no readable experiment identity"
  if [[ -n "${EXPERIMENT_ID_OVERRIDE}" \
        && "${state_experiment_id}" != "${EXPERIMENT_ID_OVERRIDE}" \
        && "${state_experiment_id}" != "UNRESOLVED_FROM_CONFIG" ]]; then
    fail "existing bootstrap state belongs to another experiment"
  fi
  if [[ -z "${EXPERIMENT_ID_OVERRIDE}" && "${state_experiment_id}" != "UNRESOLVED_FROM_CONFIG" ]]; then
    EXPERIMENT_ID="${state_experiment_id}"
  fi
  grep -Fqx "mode=${MODE}" "${STATE_FILE}" || \
''',
)

replace_once(
    bootstrap,
    '''[[ -f "${CHECKOUT}/scripts/run_e8_multitask_exp_coldstart.sh" ]] || \
  fail "target commit does not contain the reviewed experiment entrypoint"

BOOTSTRAP_STATUS="prepared"
write_state "${BOOTSTRAP_STATUS}"
''',
    '''[[ -f "${CHECKOUT}/scripts/run_e8_multitask_exp_coldstart.sh" ]] || \
  fail "target commit does not contain the reviewed experiment entrypoint"

CURRENT_STAGE="resolve_config_identity"
CONFIG_REPO_PATH="${E8_COLDSTART_CONFIG:-configs/e8_multitask_exp_coldstart.yaml}"
[[ "${CONFIG_REPO_PATH}" != /* \
   && "${CONFIG_REPO_PATH}" != ../* \
   && "${CONFIG_REPO_PATH}" != *"/../"* \
   && "${CONFIG_REPO_PATH}" != *"/.." ]] || \
  fail "bootstrap config must be a repository-relative path without parent traversal"
CONFIG_PATH="${CHECKOUT}/${CONFIG_REPO_PATH}"
[[ -f "${CONFIG_PATH}" ]] || fail "target commit is missing config: ${CONFIG_REPO_PATH}"
mapfile -t config_experiment_ids < <(
  sed -nE 's/^experiment_id:[[:space:]]*([A-Za-z0-9][A-Za-z0-9._-]{0,127})[[:space:]]*$/\\1/p' "${CONFIG_PATH}"
)
[[ "${#config_experiment_ids[@]}" -eq 1 ]] || \
  fail "config must contain exactly one well-formed top-level experiment_id: ${CONFIG_REPO_PATH}"
CONFIG_EXPERIMENT_ID="${config_experiment_ids[0]}"
if [[ "${EXPERIMENT_ID}" != "UNRESOLVED_FROM_CONFIG" \
      && "${EXPERIMENT_ID}" != "${CONFIG_EXPERIMENT_ID}" ]]; then
  fail "experiment_id mismatch: bootstrap=${EXPERIMENT_ID} config=${CONFIG_EXPERIMENT_ID}"
fi
EXPERIMENT_ID="${CONFIG_EXPERIMENT_ID}"
export E8_COLDSTART_EXPERIMENT_ID="${EXPERIMENT_ID}"

BOOTSTRAP_STATUS="prepared"
write_state "${BOOTSTRAP_STATUS}"
''',
)

replace_once(
    tests,
    '    sweep["countdown_seed_offsets"] = []\n    sweep["task_lambda"]["countdown"] = []\n    sweep["task_lambda"]["word_sorting"] = [13.0, 15.0, 18.0]\n',
    '    sweep["countdown_seed_offsets"] = []\n    sweep["task_lambda"]["word_sorting"] = [13.0, 15.0, 18.0]\n',
)

replace_once(
    tests,
    '''    both_empty = copy.deepcopy(completion)
    both_empty["sweep"]["task_lambda"]["countdown"] = []
    exp_tuning.validate_config(both_empty)
    assert len(exp_tuning.build_cells(both_empty)) == 199

    cold = exp_tuning.load_config(Path("configs/e8_multitask_exp_coldstart.yaml"))
    seeds_without_values = copy.deepcopy(cold)
    seeds_without_values["sweep"]["task_lambda"]["countdown"] = []
    with pytest.raises(ValueError, match="Countdown seeds require"):
        exp_tuning.validate_config(seeds_without_values)
''',
    '''    both_empty = copy.deepcopy(completion)
    both_empty["experiment_id"] = "EXT-C-E8-MULTITASK-EXP-EMPTY-COUNTDOWN-UNSEEN-TEST"
    both_empty["sweep"]["task_lambda"]["countdown"] = []
    with pytest.raises(ValueError, match="locked paper grids"):
        exp_tuning.validate_config(both_empty)

    cold = exp_tuning.load_config(Path("configs/e8_multitask_exp_coldstart.yaml"))
    seeds_without_values = copy.deepcopy(cold)
    seeds_without_values["experiment_id"] = "EXT-C-E8-MULTITASK-EXP-COUNTDOWN-SEEDS-NO-GRID-TEST"
    seeds_without_values["sweep"]["task_lambda"]["countdown"] = []
    with pytest.raises(ValueError, match="locked paper grids"):
        exp_tuning.validate_config(seeds_without_values)
''',
)

marker = '''def test_generic_coldstart_runner_has_no_successor_id_control_flow() -> None:
    runner = Path("scripts/run_e8_multitask_exp_coldstart.sh").read_text(encoding="utf-8")
    assert "SUCCESSOR_SOURCE_ARGS" not in runner
    assert "CONFIG_SOURCE_ARGS" in runner
    assert "${CONFIG_REPO_PATH}" in runner
    assert "EXT-C-E8-MULTITASK-EXP-LAMBDA-COMPLETION-01" not in runner
    assert "EXT-C-E8-MULTITASK-EXP-LAMBDA-CURVE-COMPLETION-02" not in runner


'''
addition = marker + '''def test_frozen_coldstart_ids_reject_scientific_matrix_drift() -> None:
    from drpo import e8_multitask_exp_tuning as exp_tuning

    cases = (
        Path("configs/e8_multitask_exp_coldstart.yaml"),
        Path("configs/e8_multitask_exp_lambda_completion.yaml"),
        Path("configs/e8_multitask_exp_lambda_curve_completion.yaml"),
    )
    for path in cases:
        config = exp_tuning.load_config(path)
        bad = copy.deepcopy(config)
        bad["sweep"]["task_transfer_seed_offset"] += 1
        with pytest.raises(ValueError, match="Frozen cold-start experiment identity drifted"):
            exp_tuning.validate_config(bad)

        bad = copy.deepcopy(config)
        bad["sweep"]["include_global_endpoint"] = True
        bad["sweep"]["expected_cells"] += sum(
            bool(bad["sweep"]["task_lambda"][task])
            for task in bad["suite"]["p0_tasks"]
        )
        with pytest.raises(ValueError, match="Frozen cold-start experiment identity drifted"):
            exp_tuning.validate_config(bad)


def test_generic_coldstart_global_endpoint_and_countdown_controls_are_config_driven() -> None:
    from drpo import e8_multitask_exp_tuning as exp_tuning

    config = exp_tuning.load_config(
        Path("configs/e8_multitask_exp_lambda_curve_completion.yaml")
    )
    config["experiment_id"] = "EXT-C-E8-MULTITASK-EXP-GLOBAL-ENDPOINT-UNSEEN-TEST"
    sweep = config["sweep"]
    sweep["include_global_endpoint"] = True
    active_transfer_tasks = [
        task for task in config["suite"]["p0_tasks"] if sweep["task_lambda"][task]
    ]
    sweep["expected_cells"] += len(active_transfer_tasks)
    exp_tuning.validate_config(config)
    cells = exp_tuning.build_cells(config)
    for task in active_transfer_tasks:
        globals_for_task = [
            cell
            for cell in cells
            if cell.task == task and cell.method == exp_tuning.METHOD_GLOBAL
        ]
        assert len(globals_for_task) == 1
        assert globals_for_task[0].lambda_value == 0.0
        assert globals_for_task[0].seed == sweep["task_transfer_seed_offset"]

    countdown_only = exp_tuning.load_config(Path("configs/e8_multitask_exp_coldstart.yaml"))
    countdown_only["experiment_id"] = "EXT-C-E8-MULTITASK-EXP-COUNTDOWN-CONTROLS-UNSEEN-TEST"
    sweep = countdown_only["sweep"]
    sweep["countdown_include_positive_only"] = False
    sweep["transfer_positive_only_seed_offsets"] = []
    for task in countdown_only["suite"]["p0_tasks"]:
        sweep["task_lambda"][task] = []
        sweep["task_grid_hashes"][task] = exp_tuning.stable_hash([])
    sweep["expected_cells"] = len(sweep["countdown_seed_offsets"]) * (
        1 + len(sweep["task_lambda"]["countdown"])
    )
    exp_tuning.validate_config(countdown_only)
    countdown_cells = exp_tuning.build_cells(countdown_only)
    assert countdown_cells
    assert not any(
        cell.method == exp_tuning.METHOD_POSITIVE_ONLY for cell in countdown_cells
    )
    assert sum(cell.method == exp_tuning.METHOD_GLOBAL for cell in countdown_cells) == 2


def test_generic_countdown_grid_and_seed_are_limited_to_locked_worker_domain() -> None:
    from drpo import e8_multitask_exp_tuning as exp_tuning

    config = exp_tuning.load_config(Path("configs/e8_multitask_exp_lambda_completion.yaml"))
    config["experiment_id"] = "EXT-C-E8-MULTITASK-EXP-COUNTDOWN-GRID-UNSEEN-TEST"
    config["sweep"]["task_lambda"]["countdown"] = [3.506557897]
    exp_tuning.validate_config(config)

    bad = copy.deepcopy(config)
    bad["sweep"]["task_lambda"]["countdown"] = [13.0]
    with pytest.raises(ValueError, match="locked paper grids"):
        exp_tuning.validate_config(bad)

    bad = copy.deepcopy(config)
    bad["sweep"]["countdown_seed_offsets"] = [3000]
    bad["sweep"]["expected_cells"] += 3
    with pytest.raises(ValueError, match="locked paper seed offsets"):
        exp_tuning.validate_config(bad)


def test_generic_coldstart_rejects_zero_scientific_cells() -> None:
    from drpo import e8_multitask_exp_tuning as exp_tuning

    config = exp_tuning.load_config(
        Path("configs/e8_multitask_exp_lambda_curve_completion.yaml")
    )
    config["experiment_id"] = "EXT-C-E8-MULTITASK-EXP-ZERO-CELLS-UNSEEN-TEST"
    sweep = config["sweep"]
    sweep["countdown_seed_offsets"] = []
    sweep["transfer_positive_only_seed_offsets"] = []
    for task in config["suite"]["p0_tasks"]:
        sweep["task_lambda"][task] = []
        sweep["task_grid_hashes"][task] = exp_tuning.stable_hash([])
    sweep["expected_cells"] = 0
    with pytest.raises(ValueError, match="at least one scientific cell"):
        exp_tuning.validate_config(config)


def test_runner_and_bootstrap_derive_experiment_id_from_config() -> None:
    runner = Path("scripts/run_e8_multitask_exp_coldstart.sh").read_text(encoding="utf-8")
    bootstrap = Path("scripts/bootstrap_e8_multitask_exp_coldstart.sh").read_text(encoding="utf-8")
    assert 'EXPERIMENT_ID_OVERRIDE="${E8_COLDSTART_EXPERIMENT_ID:-}"' in runner
    assert 'EXPERIMENT_ID="${config_experiment_id}"' in runner
    assert "experiment_id mismatch: env=" in runner
    assert 'EXPERIMENT_ID="${EXPERIMENT_ID_OVERRIDE:-UNRESOLVED_FROM_CONFIG}"' in bootstrap
    assert 'CONFIG_EXPERIMENT_ID="${config_experiment_ids[0]}"' in bootstrap
    assert "experiment_id mismatch: bootstrap=" in bootstrap
    assert '--lambda "${liveness_lambda}"' in runner


'''
replace_once(tests, marker, addition)

print("Applied E8 config-driven review fixes.")
