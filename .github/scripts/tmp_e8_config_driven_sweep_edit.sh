#!/usr/bin/env bash
set -euo pipefail
cd "${GITHUB_WORKSPACE}"

git config user.name "github-actions[bot]"
git config user.email "41898282+github-actions[bot]@users.noreply.github.com"

python3 - <<'PY'
from pathlib import Path
import re

source_path = Path("src/drpo/e8_multitask_exp_tuning.py")
test_path = Path("tests/test_e8_multitask_p0.py")
source = source_path.read_text(encoding="utf-8")
tests = test_path.read_text(encoding="utf-8")


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)


def replace_between(text: str, start: str, end: str, replacement: str, label: str) -> str:
    start_count = text.count(start)
    if start_count != 1:
        raise SystemExit(f"{label}: start matches={start_count}")
    left = text.index(start)
    try:
        right = text.index(end, left)
    except ValueError as exc:
        raise SystemExit(f"{label}: end marker missing") from exc
    return text[:left] + replacement + text[right:]


source, removed = re.subn(
    r"^SUPPORTED_EXPERIMENT_IDS = .*\n",
    "",
    source,
    count=1,
    flags=re.MULTILINE,
)
if removed != 1:
    raise SystemExit(f"SUPPORTED_EXPERIMENT_IDS removal count={removed}")

source = replace_once(
    source,
    '''def experiment_id(config: Mapping[str, Any]) -> str:\n    value = str(config.get("experiment_id", ""))\n    if value not in SUPPORTED_EXPERIMENT_IDS:\n        raise ValueError(f"Unsupported experiment_id: {value}")\n    return value\n''',
    '''def experiment_id(config: Mapping[str, Any]) -> str:\n    value = config.get("experiment_id")\n    if not isinstance(value, str) or not value.strip():\n        raise ValueError("experiment_id must be a non-empty string")\n    return value\n''',
    "experiment id metadata",
)

source = replace_once(
    source,
    '    current_experiment = experiment_id(config)\n',
    '    experiment_id(config)\n',
    "validate id without dispatch",
)

source = replace_once(
    source,
    '''        if (\n            current_experiment != DENSE_EXPERIMENT_ID\n            or len(tasks) != 7\n            or len(set(tasks)) != 7\n            or set(tasks) != _dense_tasks()\n        ):\n''',
    '''        if (\n            len(tasks) != 7\n            or len(set(tasks)) != 7\n            or set(tasks) != _dense_tasks()\n        ):\n''',
    "dense id dispatch",
)

source = replace_once(
    source,
    '''        if (\n            current_experiment not in (COLDSTART_EXPERIMENT_ID, LAMBDA_COMPLETION_EXPERIMENT_ID, LAMBDA_CURVE_COMPLETION_EXPERIMENT_ID)\n            or len(tasks) != 9\n            or len(set(tasks)) != 9\n            or set(tasks) != expected_tasks\n        ):\n''',
    '''        if (\n            len(tasks) != 9\n            or len(set(tasks)) != 9\n            or set(tasks) != expected_tasks\n        ):\n''',
    "coldstart id dispatch",
)

cold_start = '    else:\n        task_lambda = sweep.get("task_lambda")\n'
cold_end = '        initialization = config.get("initialization", {})\n'
cold_replacement = '''    else:\n        task_lambda = sweep.get("task_lambda")\n        if not isinstance(task_lambda, Mapping) or set(task_lambda) != set(tasks):\n            raise ValueError("Cold-start task_lambda must contain the exact nine tasks")\n        parameterization = str(sweep.get("parameterization", ""))\n        if parameterization not in ("paper_coefficient_c", "paper_lambda_c1"):\n            raise ValueError(\n                "Cold-start parameterization must be paper_coefficient_c or paper_lambda_c1"\n            )\n\n        countdown_sentinels = (\n            0.105360516,\n            0.430782916,\n            0.916290732,\n            1.897119985,\n            2.302585093,\n            2.995732274,\n        )\n        if _tuple_floats(\n            sweep.get("countdown_sentinel_coefficients", ())\n        ) != countdown_sentinels:\n            raise ValueError("Countdown sentinel coefficients drifted")\n        countdown_values = _task_lambdas(config, "countdown")\n        if countdown_values and countdown_values != countdown_sentinels:\n            raise ValueError(\n                "Configured Countdown lambdas must be empty or equal the six diagnostic sentinels"\n            )\n        countdown_seeds = tuple(\n            int(value) for value in sweep.get("countdown_seed_offsets", ())\n        )\n        if len(countdown_seeds) != len(set(countdown_seeds)):\n            raise ValueError("Countdown seed offsets must be unique")\n        if countdown_seeds and not countdown_values:\n            raise ValueError("Countdown seeds require configured Countdown lambda sentinels")\n\n        transfer_positive_seeds = tuple(\n            int(value) for value in sweep.get("transfer_positive_only_seed_offsets", ())\n        )\n        if len(transfer_positive_seeds) != len(set(transfer_positive_seeds)):\n            raise ValueError("Transfer Positive-only seed offsets must be unique")\n        try:\n            int(sweep["task_transfer_seed_offset"])\n            int(sweep["tuning_seed"])\n        except (KeyError, TypeError, ValueError) as exc:\n            raise ValueError("Cold-start tuning seeds must be configured integers") from exc\n\n        transfer_tasks = set(tasks) - {"countdown"}\n        grid_hashes = sweep.get("task_grid_hashes")\n        provenance = sweep.get("task_grid_provenance")\n        if not isinstance(grid_hashes, Mapping) or set(grid_hashes) != transfer_tasks:\n            raise ValueError("Transfer task-grid hashes must cover the exact eight tasks")\n        if not isinstance(provenance, Mapping) or set(provenance) != transfer_tasks:\n            raise ValueError("Transfer task-grid provenance must cover the exact eight tasks")\n        if any(not str(provenance[task]).strip() for task in transfer_tasks):\n            raise ValueError("Transfer task-grid provenance entries must be non-empty")\n        active_transfer_tasks: list[str] = []\n        for task in transfer_tasks:\n            values = _task_lambdas(config, str(task))\n            if len(values) != len(set(values)):\n                raise ValueError(f"{task} Exp coefficient grid contains duplicates")\n            if stable_hash(list(values)) != str(grid_hashes[task]):\n                raise ValueError(f"{task} coefficient grid does not match its locked hash")\n            if values:\n                active_transfer_tasks.append(str(task))\n\n        expanded_cells = len(countdown_seeds) * (2 + len(countdown_values)) + sum(\n            len(transfer_positive_seeds) + len(_task_lambdas(config, task))\n            for task in active_transfer_tasks\n        )\n        if int(sweep.get("expected_cells", -1)) != expanded_cells:\n            raise ValueError("Cold-start expected_cells must match the configured matrix")\n\n'''
source = replace_between(
    source,
    cold_start,
    cold_end,
    cold_replacement,
    "coldstart matrix validation",
)

source = replace_once(
    source,
    '''    expected_waves = (\n        math.ceil(int(config["sweep"]["expected_cells"]) / expected_capacity)\n        if profile == SWEEP_PROFILE_COLDSTART\n        else (7 if profile == SWEEP_PROFILE_DENSE else 5)\n    )\n    if int(execution["slots_per_gpu"]) != 2 or int(execution["expected_waves"]) != expected_waves:\n        raise ValueError(\n            f"The frozen topology is two slots per GPU and {expected_waves} {'nominal batches' if _is_coldstart(config) else 'waves'}"\n        )\n''',
    '''    if int(execution["slots_per_gpu"]) != 2:\n        raise ValueError("The frozen topology is two slots per GPU")\n    if not _is_coldstart(config):\n        expected_waves = 7 if profile == SWEEP_PROFILE_DENSE else 5\n        if int(execution["expected_waves"]) != expected_waves:\n            raise ValueError(\n                f"The frozen topology requires {expected_waves} waves for this profile"\n            )\n''',
    "derive coldstart waves",
)

source = replace_once(
    source,
    '        lambda_only = experiment_id(config) in (LAMBDA_COMPLETION_EXPERIMENT_ID, LAMBDA_CURVE_COMPLETION_EXPERIMENT_ID)\n',
    '        lambda_only = config["sweep"]["parameterization"] == "paper_lambda_c1"\n',
    "lambda parameterization from config",
)

source = replace_once(
    source,
    '''        for task in tasks:\n            if task == "countdown":\n                continue\n            cells.extend(\n                Cell(task, METHOD_POSITIVE_ONLY, None, seed_offset, "task_transfer")\n                for seed_offset in positive_seeds\n            )\n            cells.extend(\n                Cell(\n                    task,\n                    METHOD_EXPONENTIAL,\n                    None if lambda_only else math.exp(-coefficient),\n                    exp_seed,\n                    "task_transfer",\n                    coefficient,\n                )\n                for coefficient in _task_lambdas(config, task)\n            )\n''',
    '''        for task in tasks:\n            if task == "countdown":\n                continue\n            coefficients = _task_lambdas(config, task)\n            if not coefficients:\n                continue\n            cells.extend(\n                Cell(task, METHOD_POSITIVE_ONLY, None, seed_offset, "task_transfer")\n                for seed_offset in positive_seeds\n            )\n            cells.extend(\n                Cell(\n                    task,\n                    METHOD_EXPONENTIAL,\n                    None if lambda_only else math.exp(-coefficient),\n                    exp_seed,\n                    "task_transfer",\n                    coefficient,\n                )\n                for coefficient in coefficients\n            )\n''',
    "empty task lambda means zero task cells",
)

source = replace_once(
    source,
    '''            not waves\n            or len(waves) != int(config["execution"]["expected_waves"])\n            or any(len(wave) != capacity for wave in waves[:-1])\n''',
    '''            not waves\n            or any(len(wave) != capacity for wave in waves[:-1])\n''',
    "wave geometry from cells",
)

source = replace_once(
    source,
    '            "status": "NOT_RUN" if countdown_not_run else ("PASS" if not identity_failures and len(countdown_cells) == 16 else "FAIL"),\n',
    '            "status": "NOT_RUN" if countdown_not_run else ("PASS" if not identity_failures else "FAIL"),\n',
    "countdown diagnostic geometry",
)

source = replace_once(
    source,
    '''        positive = next((row for row in grouped if row["method"] == METHOD_POSITIVE_ONLY), None)\n        if positive is None and experiment_id(config) != LAMBDA_CURVE_COMPLETION_EXPERIMENT_ID:\n            raise RuntimeError("Cold-start aggregate requires Positive-only reference rows")\n        positive_score = None if positive is None else float(positive["late_window_pass8_mean"])\n''',
    '''        positive = next((row for row in grouped if row["method"] == METHOD_POSITIVE_ONLY), None)\n        positive_score = None if positive is None else float(positive["late_window_pass8_mean"])\n''',
    "aggregate PO from built cells",
)

if "SUPPORTED_EXPERIMENT_IDS" in source or "current_experiment" in source:
    raise SystemExit("experiment-instance dispatch remains")
for name in (
    "DENSE_EXPERIMENT_ID",
    "COLDSTART_EXPERIMENT_ID",
    "LAMBDA_COMPLETION_EXPERIMENT_ID",
    "LAMBDA_CURVE_COMPLETION_EXPERIMENT_ID",
):
    if source.count(name) != 1:
        raise SystemExit(f"{name} must be metadata-only, count={source.count(name)}")

tests = replace_once(tests, "import csv\n", "import copy\nimport csv\n", "copy import")
tests = replace_once(
    tests,
    '''    config["sweep"]["parameterization"] = "paper_coefficient_c"\n    with pytest.raises(ValueError, match="parameterization"):\n        exp_tuning.validate_config(config)\n''',
    '''    config["sweep"]["parameterization"] = "unsupported_parameterization"\n    with pytest.raises(ValueError, match="parameterization"):\n        exp_tuning.validate_config(config)\n''',
    "invalid parameterization test",
)

marker = 'def test_lambda_curve_completion_aggregate_accepts_zero_positive_only(tmp_path: Path) -> None:\n'
generic_test = '''def test_coldstart_sweep_instance_is_config_only() -> None:\n    from drpo import e8_multitask_exp_tuning as exp_tuning\n\n    config = exp_tuning.load_config(\n        Path("configs/e8_multitask_exp_lambda_curve_completion.yaml")\n    )\n    config["experiment_id"] = "EXT-C-E8-MULTITASK-EXP-CONFIG-ONLY-UNSEEN-TEST"\n    sweep = config["sweep"]\n    sweep["task_transfer_seed_offset"] = 5000\n    sweep["tuning_seed"] = 5000\n    sweep["transfer_positive_only_seed_offsets"] = [8000, 9000]\n    sweep["countdown_seed_offsets"] = []\n    sweep["task_lambda"]["countdown"] = []\n    sweep["task_lambda"]["word_sorting"] = [13.0, 15.0, 18.0]\n    sweep["task_lambda"]["maze"] = []\n\n    for task in config["suite"]["p0_tasks"]:\n        sweep["task_grid_hashes"][task] = exp_tuning.stable_hash(\n            list(sweep["task_lambda"][task])\n        )\n    active_tasks = [\n        task\n        for task in config["suite"]["p0_tasks"]\n        if sweep["task_lambda"][task]\n    ]\n    expected_cells = sum(\n        len(sweep["transfer_positive_only_seed_offsets"])\n        + len(sweep["task_lambda"][task])\n        for task in active_tasks\n    )\n    sweep["expected_cells"] = expected_cells\n    config["execution"].pop("expected_waves", None)\n\n    exp_tuning.validate_config(config)\n    cells = exp_tuning.build_cells(config)\n    assert exp_tuning.experiment_id(config) == config["experiment_id"]\n    assert len(cells) == expected_cells\n    assert {cell.task for cell in cells} == set(active_tasks)\n    assert not any(\n        cell.task in {"countdown", "spiral_matrix", "maze"} for cell in cells\n    )\n    for task in active_tasks:\n        task_cells = [cell for cell in cells if cell.task == task]\n        positives = [\n            cell for cell in task_cells if cell.method == exp_tuning.METHOD_POSITIVE_ONLY\n        ]\n        exponentials = [\n            cell for cell in task_cells if cell.method == exp_tuning.METHOD_EXPONENTIAL\n        ]\n        assert [cell.seed for cell in positives] == [8000, 9000]\n        assert {cell.seed for cell in exponentials} == {5000}\n        assert [cell.lambda_value for cell in exponentials] == sweep["task_lambda"][task]\n        assert all(cell.rho is None for cell in exponentials)\n\n    capacity = int(config["execution"]["max_concurrent_cells"])\n    expected_wave_sizes = [capacity] * (expected_cells // capacity)\n    if expected_cells % capacity:\n        expected_wave_sizes.append(expected_cells % capacity)\n    assert [len(wave) for wave in exp_tuning.build_waves(config)] == expected_wave_sizes\n\n    bad = copy.deepcopy(config)\n    bad["sweep"]["expected_cells"] += 1\n    with pytest.raises(ValueError, match="expected_cells"):\n        exp_tuning.validate_config(bad)\n\n\n'''
if marker not in tests:
    raise SystemExit("generic test insertion marker missing")
tests = tests.replace(marker, generic_test + marker, 1)

aggregate_start = tests.index(marker)
aggregate_tail = tests[aggregate_start:]
old = '''    config = exp_tuning.load_config(Path("configs/e8_multitask_exp_lambda_curve_completion.yaml"))\n    cells = exp_tuning.build_cells(config)\n'''
new = '''    config = exp_tuning.load_config(Path("configs/e8_multitask_exp_lambda_curve_completion.yaml"))\n    config["experiment_id"] = "EXT-C-E8-MULTITASK-EXP-ZERO-PO-UNSEEN-TEST"\n    exp_tuning.validate_config(config)\n    cells = exp_tuning.build_cells(config)\n'''
if old not in aggregate_tail:
    raise SystemExit("aggregate unseen-id marker missing")
aggregate_tail = aggregate_tail.replace(old, new, 1)
tests = tests[:aggregate_start] + aggregate_tail

source_path.write_text(source, encoding="utf-8")
test_path.write_text(tests, encoding="utf-8")
PY

python3 -m py_compile src/drpo/e8_multitask_exp_tuning.py tests/test_e8_multitask_p0.py
git diff --check
PYTHONPATH=src python3 -m pytest -q tests/test_e8_multitask_p0.py -k 'lambda_completion_matrix_is_config_driven_and_lambda_only or lambda_completion_preserves_restored_coldstart_behavior or lambda_curve_completion_matrix_is_config_driven or coldstart_sweep_instance_is_config_only or lambda_curve_completion_aggregate_accepts_zero_positive_only'

python3 - <<'PY'
from pathlib import Path
text = Path("src/drpo/e8_multitask_exp_tuning.py").read_text(encoding="utf-8")
assert "SUPPORTED_EXPERIMENT_IDS" not in text
assert "current_experiment" not in text
for name in (
    "DENSE_EXPERIMENT_ID",
    "COLDSTART_EXPERIMENT_ID",
    "LAMBDA_COMPLETION_EXPERIMENT_ID",
    "LAMBDA_CURVE_COMPLETION_EXPERIMENT_ID",
):
    assert text.count(name) == 1, (name, text.count(name))
PY

rm .github/workflows/tmp-e8-config-driven-sweep-edit.yml
rm .github/scripts/tmp_e8_config_driven_sweep_edit.sh

git add src/drpo/e8_multitask_exp_tuning.py tests/test_e8_multitask_p0.py .github/workflows/tmp-e8-config-driven-sweep-edit.yml .github/scripts/tmp_e8_config_driven_sweep_edit.sh
git diff --cached --check
git commit -m "E8: make cold-start sweeps config-driven"
git push origin HEAD:dev/e8-config-driven-sweep-01
