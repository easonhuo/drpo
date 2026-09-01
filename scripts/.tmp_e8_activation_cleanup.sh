#!/usr/bin/env bash
set -euo pipefail

python - <<'PY'
from pathlib import Path

path = Path('src/drpo/e8_multitask_exp_tuning.py')
text = path.read_text(encoding='utf-8')

old_activation = '    modules = _activate_paper_grid_modules(modules, grid_path)\n'
new_activation = '    modules = _activate_paper_grid_modules(modules, grid_source_path)\n'
count = text.count(old_activation)
if count != 1:
    raise SystemExit(f'activation call: expected 1 match, found {count}')
text = text.replace(old_activation, new_activation, 1)

start = text.index('def validate_config(config: Mapping[str, Any]) -> None:\n')
end = text.index('\ndef coefficient_from_rho(rho: float) -> float:\n', start)
new_validate = '''def validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != 1:
        raise ValueError("Expected schema_version: 1")
    current_experiment = experiment_id(config)
    profile = sweep_profile(config)
    if profile not in (SWEEP_PROFILE_RHO, SWEEP_PROFILE_DENSE, SWEEP_PROFILE_COLDSTART):
        raise ValueError(f"Unsupported sweep profile: {profile}")
    experiment_config.validate_profile_experiment_id(config)
    if profile == SWEEP_PROFILE_COLDSTART:
        return

    expected_parent = EXPERIMENT_ID if profile == SWEEP_PROFILE_DENSE else P0_EXPERIMENT_ID
    if config.get("parent", {}).get("experiment_id") != expected_parent:
        raise ValueError("Unexpected parent experiment")

    tasks = tuple(config.get("suite", {}).get("tasks", ()))
    if profile == SWEEP_PROFILE_RHO:
        if len(tasks) != 9 or len(set(tasks)) != 9 or set(tasks) != set(TASK_NAMES):
            raise ValueError("The rho suite must be Countdown plus the exact eight P0 tasks")
        if set(config["suite"].get("p0_tasks", ())) != set(TASK_NAMES) - {"countdown"}:
            raise ValueError("suite.p0_tasks must be the exact eight P0 tasks")
        if tuple(config["suite"].get("external_tasks", ())) != ("countdown",):
            raise ValueError("Countdown must be the only external task")
    else:
        if (
            current_experiment != DENSE_EXPERIMENT_ID
            or len(tasks) != 7
            or len(set(tasks)) != 7
            or set(tasks) != _dense_tasks()
        ):
            raise ValueError(
                "The dense suite must be the exact seven non-Countdown, non-Spiral tasks"
            )
        if tuple(config["suite"].get("p0_tasks", ())) != tasks:
            raise ValueError("Dense suite.p0_tasks must preserve the exact task order")
        if tuple(config["suite"].get("external_tasks", ())) != ():
            raise ValueError("Dense refinement has no external Countdown task")

    reference = config["reference"]
    if reference["checkpoint_kind"] != "train_only_task_positive_warmstart_100":
        raise ValueError("reference.checkpoint_kind must be train_only_task_positive_warmstart_100")
    if int(reference["optimizer_updates"]) != 100:
        raise ValueError("Reference initialization must use 100 updates")
    if int(reference["validation_rows_seen"]) != 0 or int(reference["test_rows_seen"]) != 0:
        raise ValueError("Train-only reference preparation must not see validation or test rows")

    split = config["split"]
    if _is_engineering_self_test(config):
        expected_split = {
            "p0_train_rows": 2,
            "p0_validation_rows": 1,
            "p0_test_rows": 1,
            "countdown_train_rows": 2,
            "countdown_validation_rows": 1,
        }
    else:
        expected_split = {
            "p0_train_rows": 5000,
            "p0_validation_rows": 500,
            "p0_test_rows": 500,
        }
        if profile == SWEEP_PROFILE_RHO:
            expected_split.update(
                {"countdown_train_rows": 5000, "countdown_validation_rows": 500}
            )
    for key, expected in expected_split.items():
        if int(split[key]) != expected:
            raise ValueError(f"{key} must remain {expected}")
    if bool(split.get("test_access_allowed", True)):
        raise ValueError("Tuning must forbid test access")

    training = config["training"]
    if int(training["optimizer_updates"]) != 1200:
        raise ValueError("The tuning horizon must remain 1200 updates")
    if int(training["micro_batch"]) != 1 or int(training["gradient_accumulation"]) != 8:
        raise ValueError("The method-training effective prompt batch must remain 8")
    if not math.isclose(
        float(training["learning_rate"]), 5.0e-5, rel_tol=0.0, abs_tol=1.0e-12
    ):
        raise ValueError("The method-training learning rate must remain 5e-5")
    if not math.isclose(
        float(training["warmup_ratio"]), 0.03, rel_tol=0.0, abs_tol=1.0e-12
    ):
        raise ValueError("The method-training warmup ratio must remain 0.03")
    if int(training["evaluation_every_updates"]) != 100:
        raise ValueError("Evaluation cadence must remain 100 updates")
    if not math.isclose(float(training["weight_decay"]), 0.01) or not math.isclose(
        float(training["max_grad_norm"]), 1.0
    ):
        raise ValueError("The old optimizer weight-decay/gradient-clip contract changed")
    if bool(training.get("early_stopping", True)):
        raise ValueError("Early stopping is forbidden")
    if tuple(int(value) for value in training["late_window_updates"]) != (
        800,
        900,
        1000,
        1100,
        1200,
    ):
        raise ValueError("Unexpected late-window updates")

    evaluation = config["evaluation"]
    if int(evaluation["greedy_prompt_rows"]) != 500:
        raise ValueError("Greedy validation must use 500 prompts")
    if int(evaluation["passk_prompt_rows"]) != 128 or int(evaluation["pass_k"]) != 8:
        raise ValueError("Pass@8 validation must use the frozen 128-prompt subset")

    negative = config["negative_sampling"]
    if int(negative["negatives_per_prompt"]) != 16:
        raise ValueError("Every training prompt must retain exactly 16 negatives")
    if _tuple_floats(negative["near_far_mix"]) != (0.5, 0.5):
        raise ValueError("Near/far branch mass must remain 0.5/0.5")
    if not bool(negative["selection_stop_gradient"]):
        raise ValueError("Current near/far selection must be stop-gradient")
    if bool(negative["weight_sum_normalization"]):
        raise ValueError("Weight-sum normalization is forbidden")

    calibration = config["remoteness_calibration"]
    if not math.isclose(
        float(calibration["target_negative_to_positive_gradient_ratio"]),
        1.0 / 32.0,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("Initial negative-gradient target must remain 1/32 of positive")

    sweep = config["sweep"]
    if profile == SWEEP_PROFILE_RHO:
        if _tuple_floats(sweep["coarse_rho"]) != (0.9, 0.6, 0.35, 0.125):
            raise ValueError("Unexpected coarse rho grid")
        if _tuple_floats(sweep["refinement_rho"]) != (0.75, 0.5, 0.25):
            raise ValueError("Unexpected refinement rho grid")
        if _tuple_floats(sweep["all_rho"]) != (
            0.9,
            0.75,
            0.6,
            0.5,
            0.35,
            0.25,
            0.125,
        ):
            raise ValueError("Unexpected full rho grid")
        if int(sweep["positive_only_per_task"]) != 1 or int(sweep["expected_cells"]) != 72:
            raise ValueError("The rho matrix must be 7 Exp plus 1 Positive-only per task")
    else:
        task_lambda = sweep.get("task_lambda")
        bridges = sweep.get("bridge_lambda")
        if not isinstance(task_lambda, Mapping) or set(task_lambda) != set(tasks):
            raise ValueError("Dense task_lambda must contain the exact seven tasks")
        if not isinstance(bridges, Mapping) or set(bridges) != set(tasks):
            raise ValueError("Dense bridge_lambda must contain the exact seven tasks")
        for task in tasks:
            values = _task_lambdas(config, task)
            if len(values) != 16 or len(set(values)) != 16:
                raise ValueError(f"{task} must contain 16 unique lambda values")
            bridge = float(bridges[task])
            if bridge not in values:
                raise ValueError(f"{task} bridge lambda must be one of its 16 cells")
        if int(sweep["positive_only_per_task"]) != 0 or int(sweep["expected_cells"]) != 112:
            raise ValueError("The dense matrix must be 16 Exp cells for each of seven tasks")
        if int(sweep["tuning_seed"]) != 2026072904:
            raise ValueError("Dense shape discovery must preserve the predecessor tuning seed")

    execution = config["execution"]
    expected_capacity = 16
    if int(execution["max_concurrent_cells"]) != expected_capacity:
        raise ValueError(f"The scheduler must expose exactly {expected_capacity} slots")
    if tuple(int(value) for value in execution["gpu_ids"]) != tuple(range(8)):
        raise ValueError("The default GPU pool must remain 0--7")
    expected_waves = 7 if profile == SWEEP_PROFILE_DENSE else 5
    if int(execution["slots_per_gpu"]) != 2 or int(execution["expected_waves"]) != expected_waves:
        raise ValueError(f"The frozen topology is two slots per GPU and {expected_waves} waves")
'''
text = text[:start] + new_validate + text[end:]
path.write_text(text, encoding='utf-8')

# Regression: the generic runtime grid must never be used for profile activation.
test_path = Path('tests/test_e8_multitask_p0.py')
test_text = test_path.read_text(encoding='utf-8')
marker = 'def test_runtime_activation_uses_canonical_grid_before_generic_bridge() -> None:'
if marker not in test_text:
    test_text += '''\n\ndef test_runtime_activation_uses_canonical_grid_before_generic_bridge() -> None:\n    import inspect\n\n    from drpo import e8_multitask_exp_tuning as exp_tuning\n\n    source = inspect.getsource(exp_tuning._train_canonical_cold_cell)\n    assert "_activate_paper_grid_modules(modules, grid_source_path)" in source\n    assert "_activate_paper_grid_modules(modules, grid_path)" not in source\n\n\ndef test_coldstart_validation_has_single_config_authority_exit() -> None:\n    import inspect\n\n    from drpo import e8_multitask_exp_tuning as exp_tuning\n\n    source = inspect.getsource(exp_tuning.validate_config)\n    assert "experiment_config.validate_profile_experiment_id(config)" in source\n    assert "if profile == SWEEP_PROFILE_COLDSTART:" in source\n    assert "old_lora_contract" not in source\n    assert "Countdown sentinel coefficients drifted" not in source\n    assert "Cold-start task-interface length/evaluation contract drifted" not in source\n'''
    test_path.write_text(test_text, encoding='utf-8')
PY
