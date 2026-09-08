from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one replacement, found {count}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


config_path = "src/drpo/e8_experiment_config.py"
replace_once(
    config_path,
    '''def task_delta_vs(config: Mapping[str, Any], task: str) -> tuple[float, ...]:
    raw = _sequence(
        config["sweep"]["task_delta_v"][task],
        f"{task} delta_v grid",
    )
    return tuple(_number(value, f"{task} delta_v value") for value in raw)
''',
    '''def task_delta_vs(config: Mapping[str, Any], task: str) -> tuple[float, ...]:
    raw = _sequence(
        config["sweep"]["task_delta_v"][task],
        f"{task} delta_v grid",
    )
    values = tuple(_number(value, f"{task} delta_v value") for value in raw)
    if any(value < -1.0 for value in values):
        raise ValueError(
            f"{task} delta_v values must be >= -1 so canonical alpha=1+delta_v "
            "remains non-negative"
        )
    return values
''',
)

replace_once(
    config_path,
    '''    if len(set(positive_seeds)) != len(positive_seeds):
        raise ValueError("Transfer Positive-only seed offsets must be unique")
    _integer(sweep.get("task_transfer_seed_offset"), "task_transfer_seed_offset")
    _integer(sweep.get("tuning_seed"), "tuning_seed")
''',
    '''    if len(set(positive_seeds)) != len(positive_seeds):
        raise ValueError("Transfer Positive-only seed offsets must be unique")
    if method == COLDSTART_METHOD_DPO and (positive_seeds or include_global):
        raise ValueError(
            "Current multitask DPO capability does not implement Positive-only or Global "
            "control cells inside a DPO config"
        )
    task_transfer_seed = _integer(
        sweep.get("task_transfer_seed_offset"), "task_transfer_seed_offset"
    )
    tuning_seed = _integer(sweep.get("tuning_seed"), "tuning_seed")
    if tuning_seed != task_transfer_seed:
        raise ValueError(
            "Cold-start tuning_seed must match task_transfer_seed_offset so the reviewed "
            "seed is not silently ignored"
        )
''',
)

replace_once(
    config_path,
    '''        if values:
            transfer_cell_count += len(positive_seeds) + int(include_global) + len(values)

    expanded = (
''',
    '''        if values:
            transfer_cell_count += len(positive_seeds) + int(include_global) + len(values)

    if method == COLDSTART_METHOD_DPO:
        dpo = _mapping(config.get("dpo"), "dpo")
        liveness_task = str(dpo["liveness_task"])
        liveness_beta = float(dpo["liveness_beta"])
        liveness_values = read_values(config, liveness_task)
        if not any(
            math.isclose(liveness_beta, value, rel_tol=0.0, abs_tol=1.0e-12)
            for value in liveness_values
        ):
            raise ValueError(
                "DPO liveness_beta must be one configured beta point for dpo.liveness_task"
            )

    expanded = (
''',
)

runner_path = "src/drpo/e8_multitask_exp_tuning.py"
replace_once(
    runner_path,
    '''def _dpo_shared_sft_adapter(config: Mapping[str, Any]) -> Path | None:
    dpo = config["dpo"]
    mode = str(dpo["initialization_mode"])
    if mode == "base_model_fresh_lora":
        return None
    env_name = str(dpo["shared_sft_adapter_env"])
    value = os.environ.get(env_name)
    if not value:
        raise RuntimeError(f"Shared-SFT DPO requires environment variable {env_name}")
    path = Path(value).resolve()
    if not (path / "adapter_config.json").is_file() or not any(
        (path / name).is_file() for name in ("adapter_model.safetensors", "adapter_model.bin")
    ):
        raise FileNotFoundError(f"Shared-SFT DPO adapter is incomplete: {path}")
    return path
''',
    '''def _dpo_shared_sft_adapter(config: Mapping[str, Any]) -> Path | None:
    dpo = config["dpo"]
    mode = str(dpo["initialization_mode"])
    if mode == "base_model_fresh_lora":
        return None
    env_name = str(dpo["shared_sft_adapter_env"])
    value = os.environ.get(env_name)
    if not value:
        raise RuntimeError(f"Shared-SFT DPO requires environment variable {env_name}")
    path = Path(value).resolve()
    adapter_config_path = path / "adapter_config.json"
    if not adapter_config_path.is_file() or not any(
        (path / name).is_file() for name in ("adapter_model.safetensors", "adapter_model.bin")
    ):
        raise FileNotFoundError(f"Shared-SFT DPO adapter is incomplete: {path}")
    adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
    if not isinstance(adapter_config, dict):
        raise TypeError("Shared-SFT DPO adapter_config.json must contain a mapping")
    model_config = config["model"]
    compatible = (
        str(adapter_config.get("peft_type", "")).upper() == "LORA"
        and int(adapter_config.get("r", -1)) == int(model_config["lora_rank"])
        and int(adapter_config.get("lora_alpha", -1)) == int(model_config["lora_alpha"])
        and math.isclose(
            float(adapter_config.get("lora_dropout", -1.0)),
            float(model_config["lora_dropout"]),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
    )
    if not compatible:
        raise ValueError(
            "Shared-SFT DPO adapter LoRA configuration does not match reviewed model.* "
            "LoRA values"
        )
    return path
''',
)

replace_once(
    runner_path,
    '''        "transfer_exp_single_seed_response_shape_localization": _is_coldstart(config),
''',
    '''        "transfer_exp_single_seed_response_shape_localization": (
            _is_coldstart(config) and _coldstart_method(config) == METHOD_EXPONENTIAL
        ),
''',
)

tests = Path("tests/test_e8_multitask_p0.py")
text = tests.read_text(encoding="utf-8")
marker = "\ndef test_fifth_review_rejects_asymre_delta_below_runtime_domain() -> None:\n"
if marker in text:
    raise RuntimeError("fifth-review tests already present")
append = r'''


def test_fifth_review_rejects_asymre_delta_below_runtime_domain() -> None:
    from drpo import e8_multitask_exp_tuning as exp_tuning

    config = _asymre_capability_test_config()
    config["sweep"]["task_delta_v"]["word_sorting"] = [-1.01]
    with pytest.raises(ValueError, match="delta_v values must be >= -1"):
        exp_tuning.validate_config(config)


def test_fifth_review_rejects_dpo_controls_that_dispatch_cannot_consume() -> None:
    from drpo import e8_multitask_exp_tuning as exp_tuning

    positive = _dpo_capability_test_config()
    positive["sweep"]["transfer_positive_only_seed_offsets"] = [5000]
    with pytest.raises(ValueError, match="does not implement Positive-only or Global"):
        exp_tuning.validate_config(positive)

    global_control = _dpo_capability_test_config()
    global_control["sweep"]["include_global_endpoint"] = True
    with pytest.raises(ValueError, match="does not implement Positive-only or Global"):
        exp_tuning.validate_config(global_control)


def test_fifth_review_requires_dpo_liveness_beta_on_liveness_task_grid() -> None:
    from drpo import e8_multitask_exp_tuning as exp_tuning

    config = _dpo_capability_test_config()
    config["dpo"]["liveness_beta"] = 0.3
    with pytest.raises(ValueError, match="liveness_beta must be one configured beta point"):
        exp_tuning.validate_config(config)


def test_fifth_review_rejects_silently_ignored_coldstart_tuning_seed() -> None:
    from drpo import e8_multitask_exp_tuning as exp_tuning

    config = _asymre_capability_test_config()
    config["sweep"]["tuning_seed"] = 5000
    with pytest.raises(ValueError, match="tuning_seed must match task_transfer_seed_offset"):
        exp_tuning.validate_config(config)


def test_fifth_review_shared_sft_adapter_binds_reviewed_lora_shape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from drpo import e8_multitask_exp_tuning as exp_tuning

    config = _dpo_capability_test_config(shared_sft=True)
    adapter = tmp_path / "shared"
    adapter.mkdir()
    adapter_config = {
        "peft_type": "LORA",
        "r": config["model"]["lora_rank"],
        "lora_alpha": config["model"]["lora_alpha"],
        "lora_dropout": config["model"]["lora_dropout"],
    }
    (adapter / "adapter_config.json").write_text(
        json.dumps(adapter_config), encoding="utf-8"
    )
    (adapter / "adapter_model.safetensors").write_bytes(b"identity-only-test")
    monkeypatch.setenv("E8_DPO_SHARED_SFT_ADAPTER", str(adapter))
    assert exp_tuning._dpo_shared_sft_adapter(config) == adapter.resolve()

    adapter_config["r"] = int(config["model"]["lora_rank"]) + 1
    (adapter / "adapter_config.json").write_text(
        json.dumps(adapter_config), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="LoRA configuration does not match"):
        exp_tuning._dpo_shared_sft_adapter(config)


def test_fifth_review_terminal_audit_does_not_mislabel_non_exp_methods() -> None:
    import inspect

    from drpo import e8_multitask_exp_tuning as exp_tuning

    source = inspect.getsource(exp_tuning.cmd_audit)
    assert "transfer_exp_single_seed_response_shape_localization" in source
    assert "_coldstart_method(config) == METHOD_EXPONENTIAL" in source
'''
tests.write_text(text + append, encoding="utf-8")
