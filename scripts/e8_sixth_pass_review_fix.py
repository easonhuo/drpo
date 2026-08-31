#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise SystemExit(
            f"{path}: expected one target, found {count}: {old[:120]!r}"
        )
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


py = Path("src/drpo/e8_multitask_exp_tuning.py")
tests = Path("tests/test_e8_multitask_p0.py")

replace_once(
    py,
    '''def _task_lambdas(config: Mapping[str, Any], task: str) -> tuple[float, ...]:
    if not _uses_task_lambdas(config):
        raise ValueError("Task-local lambdas are not defined for this profile")
    values = _tuple_floats(config["sweep"]["task_lambda"][task])
    if any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError(f"{task} lambda values must be finite and positive")
    return values
''',
    '''def _task_lambdas(config: Mapping[str, Any], task: str) -> tuple[float, ...]:
    if not _uses_task_lambdas(config):
        raise ValueError("Task-local lambdas are not defined for this profile")
    raw_values = config["sweep"]["task_lambda"][task]
    if isinstance(raw_values, (str, bytes)) or not isinstance(raw_values, Sequence):
        raise ValueError(f"{task} lambda grid must be a sequence of numeric scalars")
    values: list[float] = []
    for value in raw_values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                f"{task} lambda values must be numeric, finite, and positive"
            )
        numeric = float(value)
        if not math.isfinite(numeric) or numeric <= 0.0:
            raise ValueError(
                f"{task} lambda values must be numeric, finite, and positive"
            )
        values.append(numeric)
    return tuple(values)
''',
)

replace_once(
    py,
    '''    for key, expected in expected_split.items():
        if int(split[key]) != expected:
            raise ValueError(f"{key} must remain {expected}")
    if bool(split.get("test_access_allowed", True)):
''',
    '''    for key, expected in expected_split.items():
        if int(split[key]) != expected:
            raise ValueError(f"{key} must remain {expected}")
    if _is_coldstart(config):
        hash_seed = _configured_seed(
            split.get("hash_seed"), "Cold-start split hash_seed"
        )
        if hash_seed != 2026072901:
            raise ValueError("Cold-start split hash_seed must remain 2026072901")
    if bool(split.get("test_access_allowed", True)):
''',
)

replace_once(
    py,
    '''        task_lambda = sweep.get("task_lambda")
        if not isinstance(task_lambda, Mapping) or set(task_lambda) != set(tasks):
            raise ValueError("Cold-start task_lambda must contain the exact nine tasks")
        parameterization = str(sweep.get("parameterization", ""))
''',
    '''        task_lambda = sweep.get("task_lambda")
        if not isinstance(task_lambda, Mapping) or set(task_lambda) != set(tasks):
            raise ValueError("Cold-start task_lambda must contain the exact nine tasks")
        if sweep.get("method") != "exponential":
            raise ValueError("Cold-start sweep.method must remain exponential")
        parameterization = str(sweep.get("parameterization", ""))
''',
)

replace_once(
    py,
    '''        initialization = config.get("initialization", {})
        if (
            initialization.get("source") != "base_model"
            or int(initialization.get("optimizer_updates", -1)) != 0
            or bool(initialization.get("external_adapter_allowed", True))
        ):
            raise ValueError("Cold-start must use a zero-update base-model LoRA initialization")
''',
    '''        initialization = config.get("initialization", {})
        initialization_updates = _configured_seed(
            initialization.get("optimizer_updates"),
            "Cold-start initialization optimizer_updates",
        )
        initialization_seed = _configured_seed(
            initialization.get("seed"), "Cold-start initialization seed"
        )
        if (
            initialization.get("source") != "base_model"
            or initialization_updates != 0
            or initialization.get("external_adapter_allowed") is not False
            or initialization.get("deterministic_fresh_lora") is not True
            or initialization_seed != 2026070803
        ):
            raise ValueError(
                "Cold-start fresh-LoRA initialization contract drifted"
            )
''',
)

text = tests.read_text(encoding="utf-8")
stale = '''    changed["initialization"]["external_adapter_allowed"] = True
    with pytest.raises(ValueError, match="zero-update"):
        exp_tuning.validate_config(changed)
'''
updated = '''    changed["initialization"]["external_adapter_allowed"] = True
    with pytest.raises(ValueError, match="fresh-LoRA initialization contract drifted"):
        exp_tuning.validate_config(changed)
'''
if text.count(stale) != 1:
    raise SystemExit(
        f"expected one stale initialization assertion, found {text.count(stale)}"
    )
text = text.replace(stale, updated, 1)
marker = '''def test_generic_coldstart_global_endpoint_and_countdown_controls_are_config_driven() -> None:
'''
extra = '''def test_generic_coldstart_rejects_scientific_input_drift() -> None:
    from drpo import e8_multitask_exp_tuning as exp_tuning

    config = exp_tuning.load_config(
        Path("configs/e8_multitask_exp_lambda_curve_completion.yaml")
    )
    config["experiment_id"] = "EXT-C-E8-MULTITASK-EXP-SCIENCE-LOCK-UNSEEN-TEST"

    bad = copy.deepcopy(config)
    bad["sweep"]["method"] = "quadratic"
    with pytest.raises(ValueError, match="sweep.method must remain exponential"):
        exp_tuning.validate_config(bad)

    bad = copy.deepcopy(config)
    bad["split"]["hash_seed"] += 1
    with pytest.raises(ValueError, match="split hash_seed must remain 2026072901"):
        exp_tuning.validate_config(bad)

    bad = copy.deepcopy(config)
    bad["initialization"]["seed"] += 1
    with pytest.raises(ValueError, match="fresh-LoRA initialization contract drifted"):
        exp_tuning.validate_config(bad)

    bad = copy.deepcopy(config)
    bad["initialization"]["deterministic_fresh_lora"] = False
    with pytest.raises(ValueError, match="fresh-LoRA initialization contract drifted"):
        exp_tuning.validate_config(bad)

    for invalid in (True, "13.0"):
        bad = copy.deepcopy(config)
        bad["sweep"]["task_lambda"]["word_sorting"][0] = invalid
        with pytest.raises(ValueError, match="lambda values must be numeric, finite, and positive"):
            exp_tuning.validate_config(bad)


def test_task_lambda_grid_rejects_scalar_string_container() -> None:
    from drpo import e8_multitask_exp_tuning as exp_tuning

    config = exp_tuning.load_config(
        Path("configs/e8_multitask_exp_lambda_curve_completion.yaml")
    )
    config["experiment_id"] = "EXT-C-E8-MULTITASK-EXP-LAMBDA-TYPE-UNSEEN-TEST"
    config["sweep"]["task_lambda"]["word_sorting"] = "13.0"
    with pytest.raises(ValueError, match="lambda grid must be a sequence of numeric scalars"):
        exp_tuning.validate_config(config)


'''
if text.count(marker) != 1 or "test_generic_coldstart_rejects_scientific_input_drift" in text:
    raise SystemExit("cannot insert sixth-pass scientific input tests")
tests.write_text(text.replace(marker, extra + marker, 1), encoding="utf-8")
