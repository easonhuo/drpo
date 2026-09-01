#!/usr/bin/env bash
set -euo pipefail

python - <<'PY'
from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)


# ---------------------------------------------------------------------------
# 1. Config authority: expose exactly one effective runtime and reject
#    accepted-but-unconsumable combinations.
# ---------------------------------------------------------------------------
path = Path("src/drpo/e8_experiment_config.py")
text = path.read_text(encoding="utf-8")

old = '''    if late[-1] > training["optimizer_updates"]:
        raise ValueError("training.late_window_updates may not exceed optimizer_updates")

    evaluation = config["evaluation"]
'''
new = '''    if late[-1] > training["optimizer_updates"]:
        raise ValueError("training.late_window_updates may not exceed optimizer_updates")
    optimizer_updates = _integer(
        training["optimizer_updates"], "training.optimizer_updates", positive=True
    )
    evaluation_every = _integer(
        training["evaluation_every_updates"],
        "training.evaluation_every_updates",
        positive=True,
    )
    if any(
        update != optimizer_updates and update % evaluation_every != 0
        for update in late
    ):
        raise ValueError(
            "training.late_window_updates must fall on configured evaluation updates "
            "or the terminal optimizer update"
        )

    model = config["model"]
    for field in ("lora_rank", "lora_alpha", "max_length", "max_new_tokens"):
        _integer(model[field], f"model.{field}", positive=True)
    dropout = float(model["lora_dropout"])
    if not 0.0 <= dropout < 1.0:
        raise ValueError("model.lora_dropout must be in [0, 1)")

    evaluation = config["evaluation"]
'''
text = replace_once(text, old, new, "scalar-runtime-alignment")

anchor = '\ndef _validate_sweep(config: Mapping[str, Any], tasks: tuple[str, ...]) -> None:\n'
if text.count(anchor) != 1:
    raise SystemExit("effective-runtime anchor mismatch")
addition = r'''

def is_historical_coldstart_config(config: Mapping[str, Any]) -> bool:
    return experiment_id(config) in HISTORICAL_CONFIG_IDENTITIES


def effective_coldstart_runtime(config: Mapping[str, Any], task: str) -> dict[str, Any]:
    """Resolve the values that the canonical cold-start runtime must actually consume."""

    tasks = tuple(str(value) for value in config.get("suite", {}).get("tasks", ()))
    if task not in tasks:
        raise ValueError(f"Unknown cold-start task: {task}")
    runtime = _mapping(config["task_runtime"][task], f"task_runtime.{task}")
    model = _mapping(config["model"], "model")
    training = _mapping(config["training"], "training")
    evaluation = _mapping(config["evaluation"], "evaluation")
    auxiliary = [int(value) for value in runtime["auxiliary_pass_ks"]]
    pass_k = int(evaluation["pass_k"])
    return {
        "initialization_seed": int(config["initialization"]["seed"]),
        "model": {
            "parameterization": str(model["parameterization"]),
            "dtype": str(model["dtype"]),
            "lora_rank": int(model["lora_rank"]),
            "lora_alpha": int(model["lora_alpha"]),
            "lora_dropout": float(model["lora_dropout"]),
            "gradient_checkpointing": bool(model["gradient_checkpointing"]),
            "max_length": int(runtime["max_length"]),
            "max_new_tokens": int(runtime["max_new_tokens"]),
        },
        "training": {
            "optimizer_updates": int(training["optimizer_updates"]),
            "micro_batch": int(training["micro_batch"]),
            "gradient_accumulation": int(training["gradient_accumulation"]),
            "learning_rate": float(training["learning_rate"]),
            "weight_decay": float(training["weight_decay"]),
            "warmup_ratio": float(training["warmup_ratio"]),
            "max_grad_norm": float(training["max_grad_norm"]),
            "evaluation_every_updates": int(training["evaluation_every_updates"]),
            "late_window_updates": [int(value) for value in training["late_window_updates"]],
        },
        "evaluation": {
            "examples": max(
                int(runtime["greedy_prompt_rows"]), int(runtime["passk_prompt_rows"])
            ),
            "batch_size": int(runtime["evaluation_batch_size"]),
            "greedy_prompt_rows": int(runtime["greedy_prompt_rows"]),
            "passk_prompt_rows": int(runtime["passk_prompt_rows"]),
            "pass_k": pass_k,
            "auxiliary_pass_ks": auxiliary,
            "pass_ks": [pass_k, *auxiliary],
            "sampling_temperature": float(evaluation["sampling_temperature"]),
            "top_p": float(evaluation["top_p"]),
            "generation_seed": int(evaluation["generation_seed"]),
            "max_new_tokens": int(runtime["max_new_tokens"]),
        },
    }


def _validate_runtime_authority_consistency(
    config: Mapping[str, Any], tasks: tuple[str, ...]
) -> None:
    runtime = config["task_runtime"]
    countdown = runtime["countdown"]
    model = config["model"]
    evaluation = config["evaluation"]
    expected_pairs = (
        (model["max_length"], countdown["max_length"], "model.max_length"),
        (model["max_new_tokens"], countdown["max_new_tokens"], "model.max_new_tokens"),
        (
            evaluation["max_new_tokens"],
            countdown["max_new_tokens"],
            "evaluation.max_new_tokens",
        ),
        (
            evaluation["batch_size"],
            countdown["evaluation_batch_size"],
            "evaluation.batch_size",
        ),
        (
            evaluation["greedy_prompt_rows"],
            countdown["greedy_prompt_rows"],
            "evaluation.greedy_prompt_rows",
        ),
        (
            evaluation["passk_prompt_rows"],
            countdown["passk_prompt_rows"],
            "evaluation.passk_prompt_rows",
        ),
    )
    for configured, task_value, label in expected_pairs:
        if int(configured) != int(task_value):
            raise ValueError(f"{label} must match task_runtime.countdown")
    if tuple(int(value) for value in evaluation["auxiliary_pass_ks"]) != tuple(
        int(value) for value in countdown["auxiliary_pass_ks"]
    ):
        raise ValueError("evaluation.auxiliary_pass_ks must match task_runtime.countdown")
    if int(countdown["greedy_prompt_rows"]) != int(countdown["passk_prompt_rows"]):
        raise ValueError(
            "Countdown canonical evaluator currently requires equal greedy/pass-k prompt budgets"
        )
    if int(evaluation["pass_k"]) != 8:
        raise ValueError("Cold-start canonical reporting currently implements pass_k=8")
    for task in tasks:
        auxiliary = tuple(int(value) for value in runtime[task]["auxiliary_pass_ks"])
        if len(set(auxiliary)) != len(auxiliary) or any(value != 64 for value in auxiliary):
            raise ValueError(
                f"task_runtime.{task}.auxiliary_pass_ks currently supports only optional pass@64"
            )
'''
text = text.replace(anchor, addition + anchor, 1)

text = replace_once(
    text,
    '''    _validate_task_runtime(config, tasks)\n    _validate_sweep(config, tasks)\n''',
    '''    _validate_task_runtime(config, tasks)\n    _validate_runtime_authority_consistency(config, tasks)\n    _validate_sweep(config, tasks)\n''',
    "runtime-consistency-call",
)
path.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# 2. Wrapper: materialize generic runtime base/grid files while preserving the
#    historical three IDs and all byte-locked canonical paper sources.
# ---------------------------------------------------------------------------
path = Path("src/drpo/e8_multitask_exp_tuning.py")
text = path.read_text(encoding="utf-8")
start = text.index("def _task_base_config(\n")
end = text.index("\ndef _evenly_spaced_rank_indices", start)
replacement = r'''def _leaf_values(value: Any, prefix: str = "") -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {prefix: value}
    result: dict[str, Any] = {}
    for key, item in value.items():
        child = f"{prefix}.{key}" if prefix else str(key)
        result.update(_leaf_values(item, child))
    return result


def _changed_leaf_paths(original: Mapping[str, Any], derived: Mapping[str, Any]) -> list[str]:
    left = _leaf_values(original)
    right = _leaf_values(derived)
    return sorted(key for key in set(left) | set(right) if left.get(key) != right.get(key))


def _atomic_yaml(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(yaml.safe_dump(dict(value), sort_keys=False), encoding="utf-8")
    temporary.replace(path)


def _task_base_config(
    config: Mapping[str, Any],
    *,
    task: str,
    canonical_paths: Mapping[str, Path],
    task_root: Path,
) -> tuple[Path, list[str]]:
    """Materialize effective base runtime without editing the canonical source."""

    base_path = canonical_paths["base_config"]
    original = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    if not isinstance(original, dict):
        raise TypeError("Paper base config root must be a mapping")
    historical = experiment_config.is_historical_coldstart_config(config)
    if historical and task == "countdown":
        return base_path, []

    derived = copy.deepcopy(original)
    effective = experiment_config.effective_coldstart_runtime(config, task)
    runtime = config["task_runtime"][task]
    if historical:
        # Preserve the exact wrapper behavior of the three closed historical IDs.
        derived["model"]["max_length"] = int(runtime["max_length"])
        derived["model"]["max_new_tokens"] = int(runtime["max_new_tokens"])
        derived["evaluation"]["batch_size"] = int(runtime["evaluation_batch_size"])
        derived["evaluation"]["pass_ks"] = [8] + [
            int(value) for value in runtime["auxiliary_pass_ks"]
        ]
    else:
        model = effective["model"]
        training = effective["training"]
        evaluation = effective["evaluation"]
        derived["model"].update(
            {
                "max_length": int(model["max_length"]),
                "max_new_tokens": int(model["max_new_tokens"]),
                "dtype": str(model["dtype"]),
                "lora_rank": int(model["lora_rank"]),
                "lora_alpha": int(model["lora_alpha"]),
                "lora_dropout": float(model["lora_dropout"]),
                "gradient_checkpointing": bool(model["gradient_checkpointing"]),
            }
        )
        derived["offline_training"].update(
            {
                "seed": int(effective["initialization_seed"]),
                "steps": int(training["optimizer_updates"]),
                "micro_batch": int(training["micro_batch"]),
                "gradient_accumulation": int(training["gradient_accumulation"]),
                "learning_rate": float(training["learning_rate"]),
                "weight_decay": float(training["weight_decay"]),
                "warmup_ratio": float(training["warmup_ratio"]),
                "maximum_gradient_norm": float(training["max_grad_norm"]),
                "eval_every": int(training["evaluation_every_updates"]),
            }
        )
        derived["evaluation"].update(
            {
                "examples": int(evaluation["examples"]),
                "batch_size": int(evaluation["batch_size"]),
                "pass_ks": [int(value) for value in evaluation["pass_ks"]],
                "seed": int(evaluation["generation_seed"]),
                "sampling_temperature": float(evaluation["sampling_temperature"]),
                "top_p": float(evaluation["top_p"]),
                "greedy_prompt_rows": int(evaluation["greedy_prompt_rows"]),
                "passk_prompt_rows": int(evaluation["passk_prompt_rows"]),
            }
        )
    path = task_root / "paper_base_task_interface.yaml"
    _atomic_yaml(path, derived)
    return path, _changed_leaf_paths(original, derived)


def _task_grid_configs(
    config: Mapping[str, Any],
    *,
    canonical_paths: Mapping[str, Path],
    task_root: Path,
) -> dict[str, dict[str, Any]]:
    """Return historical grids unchanged or generic derived runtime-grid copies."""

    result: dict[str, dict[str, Any]] = {}
    historical = experiment_config.is_historical_coldstart_config(config)
    training = config["training"]
    for name in ("round1_grid", "extension_grid"):
        source = canonical_paths[name]
        original = yaml.safe_load(source.read_text(encoding="utf-8"))
        if not isinstance(original, dict):
            raise TypeError(f"Paper grid root must be a mapping: {source}")
        if historical:
            runtime_path = source
            changed: list[str] = []
        else:
            derived = copy.deepcopy(original)
            derived["training"]["steps"] = int(training["optimizer_updates"])
            derived["training"]["eval_every"] = int(training["evaluation_every_updates"])
            runtime_path = task_root / f"paper_{name}_runtime.yaml"
            _atomic_yaml(runtime_path, derived)
            changed = _changed_leaf_paths(original, derived)
        result[name] = {
            "path": runtime_path,
            "source": source,
            "changed_fields": changed,
        }
    return result
'''
text = text[:start] + replacement + text[end:]

old = '''        task_base_config, changed_fields = _task_base_config(
            config,
            task=task,
            canonical_paths=canonical_paths,
            task_root=task_root,
        )
        canonical_record = {
            "train": str(train_path.resolve()),
            "validation": str(validation_path.resolve()),
            "sealed_test": str(sealed_test_path.resolve()),
            "base_config": str(task_base_config.resolve()),
            "base_config_sha256": sha256_file(task_base_config),
            "round1_grid": str(canonical_paths["round1_grid"]),
            "round1_grid_sha256": sha256_file(canonical_paths["round1_grid"]),
            "extension_grid": str(canonical_paths["extension_grid"]),
            "extension_grid_sha256": sha256_file(canonical_paths["extension_grid"]),
            "task_interface_changed_fields": changed_fields,
'''
new = '''        task_base_config, changed_fields = _task_base_config(
            config,
            task=task,
            canonical_paths=canonical_paths,
            task_root=task_root,
        )
        runtime_grids = _task_grid_configs(
            config,
            canonical_paths=canonical_paths,
            task_root=task_root,
        )
        canonical_record = {
            "train": str(train_path.resolve()),
            "validation": str(validation_path.resolve()),
            "sealed_test": str(sealed_test_path.resolve()),
            "base_config": str(task_base_config.resolve()),
            "base_config_sha256": sha256_file(task_base_config),
            "round1_grid": str(runtime_grids["round1_grid"]["path"].resolve()),
            "round1_grid_sha256": sha256_file(runtime_grids["round1_grid"]["path"]),
            "extension_grid": str(runtime_grids["extension_grid"]["path"].resolve()),
            "extension_grid_sha256": sha256_file(runtime_grids["extension_grid"]["path"]),
            "task_interface_changed_fields": changed_fields,
'''
text = replace_once(text, old, new, "canonical-grid-materialization")

old = '''        record["canonical_coldstart"] = canonical_record
        records[task] = canonical_record
'''
new = '''        if not experiment_config.is_historical_coldstart_config(config):
            canonical_record["effective_runtime"] = experiment_config.effective_coldstart_runtime(
                config, task
            )
            canonical_record["runtime_grid_sources"] = {
                name: {
                    "source": str(runtime_grids[name]["source"].resolve()),
                    "source_sha256": sha256_file(runtime_grids[name]["source"]),
                    "changed_fields": list(runtime_grids[name]["changed_fields"]),
                }
                for name in ("round1_grid", "extension_grid")
            }
        record["canonical_coldstart"] = canonical_record
        records[task] = canonical_record
'''
text = replace_once(text, old, new, "canonical-runtime-provenance")

# Process-local bridges live in the wrapper, not the byte-locked old core.
anchor = '\ndef _train_canonical_cold_cell(\n'
if text.count(anchor) != 1:
    raise SystemExit("canonical-train anchor mismatch")
bridge = r'''

def _runtime_bridge_contract(effective: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "fresh_lora": {
            "rank": int(effective["model"]["lora_rank"]),
            "alpha": int(effective["model"]["lora_alpha"]),
            "dropout": float(effective["model"]["lora_dropout"]),
        },
        "gradient_checkpointing": bool(effective["model"]["gradient_checkpointing"]),
        "optimizer_weight_decay": float(effective["training"]["weight_decay"]),
        "sampling_temperature": float(effective["evaluation"]["sampling_temperature"]),
        "top_p": float(effective["evaluation"]["top_p"]),
    }


@contextmanager
def _legacy_arena_runtime_bridge(arena: Any, effective: Mapping[str, Any]) -> Any:
    """Temporarily parameterize legacy arena interface literals; never touch loss math."""

    original_lora_config = arena.LoraConfig
    original_load_model = arena.load_model
    original_generate_outputs = arena.generate_outputs
    contract = _runtime_bridge_contract(effective)

    def configured_lora_config(*args: Any, **kwargs: Any) -> Any:
        kwargs["r"] = int(contract["fresh_lora"]["rank"])
        kwargs["lora_alpha"] = int(contract["fresh_lora"]["alpha"])
        kwargs["lora_dropout"] = float(contract["fresh_lora"]["dropout"])
        return original_lora_config(*args, **kwargs)

    def configured_load_model(*args: Any, **kwargs: Any) -> Any:
        values = list(args)
        if len(values) > 5:
            values[5] = bool(contract["gradient_checkpointing"])
        else:
            kwargs["gradient_checkpointing"] = bool(contract["gradient_checkpointing"])
        return original_load_model(*values, **kwargs)

    def configured_generate_outputs(
        model: Any,
        tokenizer: Any,
        prompts: list[str],
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
        top_p: float,
        num_return_sequences: int = 1,
    ) -> Any:
        if do_sample:
            temperature = float(contract["sampling_temperature"])
            top_p = float(contract["top_p"])
        return original_generate_outputs(
            model,
            tokenizer,
            prompts,
            max_new_tokens,
            do_sample,
            temperature,
            top_p,
            num_return_sequences,
        )

    arena.LoraConfig = configured_lora_config
    arena.load_model = configured_load_model
    arena.generate_outputs = configured_generate_outputs
    try:
        yield contract
    finally:
        arena.LoraConfig = original_lora_config
        arena.load_model = original_load_model
        arena.generate_outputs = original_generate_outputs


def _validated_runtime_grid(
    candidate: Mapping[str, Any],
    *,
    canonical_grid: Mapping[str, Any],
    effective: Mapping[str, Any],
    strict_validator: Any,
) -> None:
    allowed = {"training.steps", "training.eval_every"}
    changed = set(_changed_leaf_paths(canonical_grid, candidate))
    forbidden = sorted(changed - allowed)
    if forbidden:
        raise ValueError(f"Derived runtime grid changed non-runtime fields: {forbidden}")
    training = candidate.get("training", {})
    if int(training.get("steps", -1)) != int(effective["training"]["optimizer_updates"]):
        raise ValueError("Derived runtime grid steps do not match effective runtime")
    if int(training.get("eval_every", -1)) != int(
        effective["training"]["evaluation_every_updates"]
    ):
        raise ValueError("Derived runtime grid eval_every does not match effective runtime")
    strict = copy.deepcopy(dict(candidate))
    strict["training"]["steps"] = canonical_grid["training"]["steps"]
    strict["training"]["eval_every"] = canonical_grid["training"]["eval_every"]
    strict_validator(strict)


@contextmanager
def _legacy_paper_runtime_bridge(
    modules: Mapping[str, Any],
    effective: Mapping[str, Any],
    *,
    grid_path: Path,
    grid_source_path: Path,
) -> Any:
    """Bridge configured runtime scalars into byte-locked paper interfaces."""

    scan_trainer = modules["scan_trainer"]
    paper_common = modules["paper_common"]
    canonical_grid = yaml.safe_load(grid_source_path.read_text(encoding="utf-8"))
    if not isinstance(canonical_grid, dict):
        raise TypeError("Canonical paper grid root must be a mapping")
    candidate_grid = yaml.safe_load(grid_path.read_text(encoding="utf-8"))
    if not isinstance(candidate_grid, dict):
        raise TypeError("Derived paper grid root must be a mapping")

    validator_targets: list[tuple[Any, str, Any]] = []
    seen: set[tuple[int, str]] = set()
    for module_name in ("paper_common", "scan_common", "scan_trainer", "scan_runtime"):
        module = modules[module_name]
        if not hasattr(module, "validate_grid_config"):
            continue
        key = (id(module), "validate_grid_config")
        if key in seen:
            continue
        seen.add(key)
        validator_targets.append((module, "validate_grid_config", module.validate_grid_config))
    strict_validator = paper_common.validate_grid_config

    def configured_validator(value: Mapping[str, Any]) -> None:
        _validated_runtime_grid(
            value,
            canonical_grid=canonical_grid,
            effective=effective,
            strict_validator=strict_validator,
        )

    optimizer_holder = scan_trainer.torch.optim
    original_adamw = optimizer_holder.AdamW

    def configured_adamw(*args: Any, **kwargs: Any) -> Any:
        kwargs["weight_decay"] = float(effective["training"]["weight_decay"])
        return original_adamw(*args, **kwargs)

    with _legacy_arena_runtime_bridge(modules["arena"], effective) as contract:
        optimizer_holder.AdamW = configured_adamw
        for module, name, _ in validator_targets:
            setattr(module, name, configured_validator)
        try:
            configured_validator(candidate_grid)
            yield contract
        finally:
            optimizer_holder.AdamW = original_adamw
            for module, name, original in validator_targets:
                setattr(module, name, original)
'''
text = text.replace(anchor, bridge + anchor, 1)

# Reference-remoteness policy must instantiate the configured LoRA dimensions.
old = '''        model = arena.load_model(
            str(Path(base_model_path).resolve()),
            adapter_path=None,
            trainable_adapter=True,
            load_in_4bit=bool(base_config["model"].get("load_in_4bit", False)),
            dtype=str(base_config["model"].get("dtype", "auto")),
            gradient_checkpointing=False,
            parameterization="lora",
        )
        model.eval()
'''
new = '''        reference_effective = experiment_config.effective_coldstart_runtime(
            config, pending[0]
        )
        with _legacy_arena_runtime_bridge(arena, reference_effective):
            model = arena.load_model(
                str(Path(base_model_path).resolve()),
                adapter_path=None,
                trainable_adapter=True,
                load_in_4bit=bool(base_config["model"].get("load_in_4bit", False)),
                dtype=str(base_config["model"].get("dtype", "auto")),
                gradient_checkpointing=False,
                parameterization="lora",
            )
        model.eval()
'''
text = replace_once(text, old, new, "reference-lora-runtime")

old = '''    grid_path = _paper_grid_for_cell(record, cell)
    modules = _activate_paper_grid_modules(modules, grid_path)
    arena = modules["arena"]
    runtime = modules["paper_runtime"]
    base_config = yaml.safe_load(base_config_path.read_text(encoding="utf-8"))
'''
new = '''    grid_path = _paper_grid_for_cell(record, cell)
    grid_source_name = (
        "round1_grid"
        if cell.task != "countdown"
        else _paper_grid_name(0.0 if cell.lambda_value is None else float(cell.lambda_value))
    )
    grid_source_path = _canonical_paths(config)[grid_source_name]
    modules = _activate_paper_grid_modules(modules, grid_path)
    arena = modules["arena"]
    runtime = modules["paper_runtime"]
    effective_runtime = experiment_config.effective_coldstart_runtime(config, cell.task)
    base_config = yaml.safe_load(base_config_path.read_text(encoding="utf-8"))
'''
text = replace_once(text, old, new, "cell-effective-runtime")

old = '''            "paper_grid_config": str(grid_path.resolve()),
            "paper_grid_config_sha256": sha256_file(grid_path),
            "paper_base_config": str(base_config_path.resolve()),
'''
new = '''            "paper_grid_config": str(grid_path.resolve()),
            "paper_grid_config_sha256": sha256_file(grid_path),
            "paper_grid_source": str(grid_source_path.resolve()),
            "paper_grid_source_sha256": sha256_file(grid_source_path),
            "paper_base_config": str(base_config_path.resolve()),
'''
text = replace_once(text, old, new, "cell-grid-source-provenance")

# Generic-only cell provenance; historical identity shape stays unchanged except for
# the two source-path fields above, which are identical to the active historical grid.
marker = '            "task_runtime_contract": dict(config["task_runtime"][cell.task]),\n        }\n    )\n'
pos = text.index(marker, text.index("def _train_canonical_cold_cell")) + len(marker)
insert = '''    if not experiment_config.is_historical_coldstart_config(config):
        identity["effective_runtime"] = effective_runtime
        identity["legacy_runtime_bridge"] = _runtime_bridge_contract(effective_runtime)
'''
text = text[:pos] + insert + text[pos:]

# Allow configured optional pass@64 for both Countdown and transfer evaluators.
old = '''                def transfer_evaluate(**kwargs: Any) -> dict[str, Any]:
                    kwargs["pass64_enabled"] = False
                    return original_trainer_evaluate(**kwargs)

                scan_trainer._evaluate_validation = transfer_evaluate
'''
new = '''                def configured_evaluate(**kwargs: Any) -> dict[str, Any]:
                    kwargs["pass64_enabled"] = 64 in set(
                        int(value)
                        for value in effective_runtime["evaluation"]["auxiliary_pass_ks"]
                    )
                    return original_trainer_evaluate(**kwargs)

                scan_trainer._evaluate_validation = configured_evaluate
'''
text = replace_once(text, old, new, "transfer-pass64-runtime")

# Countdown has no custom evaluator, but its pass@64 enablement must also be config-backed.
old = '''        try:
            if evaluator is not None:
                arena.evaluate_rows = evaluator
'''
new = '''        try:
            if evaluator is None:
                def configured_evaluate(**kwargs: Any) -> dict[str, Any]:
                    kwargs["pass64_enabled"] = 64 in set(
                        int(value)
                        for value in effective_runtime["evaluation"]["auxiliary_pass_ks"]
                    )
                    return original_trainer_evaluate(**kwargs)

                scan_trainer._evaluate_validation = configured_evaluate
            if evaluator is not None:
                arena.evaluate_rows = evaluator
'''
text = replace_once(text, old, new, "countdown-pass64-runtime")

old = '''    with task_interface():
        if cell.task == "countdown":
'''
new = '''    with _legacy_paper_runtime_bridge(
        modules,
        effective_runtime,
        grid_path=grid_path,
        grid_source_path=grid_source_path,
    ), task_interface():
        if cell.task == "countdown":
'''
text = replace_once(text, old, new, "canonical-runtime-context")
path.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# 3. Preflight uses the same effective-runtime resolver.
# ---------------------------------------------------------------------------
path = Path("scripts/preflight_e8_multitask_config.py")
text = path.read_text(encoding="utf-8")
old = '''    counts = Counter(cell.task for cell in cells)
    return {
'''
new = '''    counts = Counter(cell.task for cell in cells)
    active_tasks = [task for task in value["suite"]["tasks"] if counts[task]]
    effective_runtime = (
        {
            task: experiment_config.effective_coldstart_runtime(value, task)
            for task in active_tasks
        }
        if experiment_config.sweep_profile(value) == experiment_config.SWEEP_PROFILE_COLDSTART
        else {}
    )
    return {
'''
text = replace_once(text, old, new, "preflight-effective-runtime-setup")
text = replace_once(
    text,
    '''        "active_tasks": [task for task in value["suite"]["tasks"] if counts[task]],
        "cells_per_task": {task: counts[task] for task in value["suite"]["tasks"]},
''',
    '''        "active_tasks": active_tasks,
        "cells_per_task": {task: counts[task] for task in value["suite"]["tasks"]},
        "effective_runtime": effective_runtime,
''',
    "preflight-effective-runtime-payload",
)
text = replace_once(
    text,
    '        "note": "Validation/preflight only; no model, optimizer, GPU, or scientific metric executed.",\n',
    '        "note": "Resolved effective runtime only; no model, optimizer, GPU, or scientific metric executed.",\n',
    "preflight-note",
)
path.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# 4. Replace validator-only acceptance with actual runtime/materialization tests.
# ---------------------------------------------------------------------------
path = Path("tests/test_e8_multitask_p0.py")
text = path.read_text(encoding="utf-8")
start = text.index("def test_new_coldstart_config_controls_scientific_scalars_without_core_edits(")
end = text.index("\ndef test_profile_experiment_id_scope_is_fail_closed", start)
replacement = r'''def test_new_coldstart_config_controls_materialized_runtime_without_core_edits(
    tmp_path: Path,
) -> None:
    from drpo import e8_experiment_config as experiment_config
    from drpo import e8_multitask_exp_tuning as exp_tuning

    config = exp_tuning.load_config(Path("configs/e8_multitask_exp_lambda_curve_completion.yaml"))
    config["experiment_id"] = "EXT-C-E8-MULTITASK-EXP-CONFIG-AUTHORITY-TEST"
    config["training"].update(
        {
            "optimizer_updates": 1500,
            "micro_batch": 2,
            "gradient_accumulation": 4,
            "learning_rate": 7.0e-5,
            "weight_decay": 0.02,
            "warmup_ratio": 0.05,
            "max_grad_norm": 0.8,
            "evaluation_every_updates": 125,
            "late_window_updates": [1000, 1125, 1250, 1375, 1500],
        }
    )
    config["evaluation"].update(
        {"sampling_temperature": 0.7, "top_p": 0.9, "generation_seed": 2026090101}
    )
    config["split"]["hash_seed"] = 2026090102
    config["initialization"]["seed"] = 2026090103
    config["model"].update({"lora_rank": 16, "lora_alpha": 32, "lora_dropout": 0.02})
    config["task_runtime"]["word_sorting"]["max_length"] = 640

    exp_tuning.validate_config(config)
    effective = experiment_config.effective_coldstart_runtime(config, "word_sorting")
    assert effective["initialization_seed"] == 2026090103
    assert effective["model"]["lora_rank"] == 16
    assert effective["model"]["lora_alpha"] == 32
    assert effective["model"]["lora_dropout"] == pytest.approx(0.02)
    assert effective["model"]["max_length"] == 640
    assert effective["training"]["optimizer_updates"] == 1500
    assert effective["training"]["micro_batch"] == 2
    assert effective["training"]["gradient_accumulation"] == 4
    assert effective["training"]["learning_rate"] == pytest.approx(7.0e-5)
    assert effective["training"]["weight_decay"] == pytest.approx(0.02)
    assert effective["training"]["evaluation_every_updates"] == 125
    assert effective["evaluation"]["sampling_temperature"] == pytest.approx(0.7)
    assert effective["evaluation"]["top_p"] == pytest.approx(0.9)
    assert effective["evaluation"]["generation_seed"] == 2026090101

    canonical_paths = exp_tuning._canonical_paths(config)
    task_root = tmp_path / "word_sorting"
    task_root.mkdir()
    base_path, changed = exp_tuning._task_base_config(
        config,
        task="word_sorting",
        canonical_paths=canonical_paths,
        task_root=task_root,
    )
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    assert base["model"]["lora_rank"] == 16
    assert base["model"]["lora_alpha"] == 32
    assert base["model"]["lora_dropout"] == pytest.approx(0.02)
    assert base["model"]["max_length"] == 640
    assert base["offline_training"]["seed"] == 2026090103
    assert base["offline_training"]["steps"] == 1500
    assert base["offline_training"]["micro_batch"] == 2
    assert base["offline_training"]["gradient_accumulation"] == 4
    assert base["offline_training"]["learning_rate"] == pytest.approx(7.0e-5)
    assert base["offline_training"]["weight_decay"] == pytest.approx(0.02)
    assert base["offline_training"]["eval_every"] == 125
    assert base["evaluation"]["seed"] == 2026090101
    assert base["evaluation"]["sampling_temperature"] == pytest.approx(0.7)
    assert base["evaluation"]["top_p"] == pytest.approx(0.9)
    assert "offline_training.learning_rate" in changed

    grids = exp_tuning._task_grid_configs(
        config,
        canonical_paths=canonical_paths,
        task_root=task_root,
    )
    round1 = yaml.safe_load(grids["round1_grid"]["path"].read_text(encoding="utf-8"))
    assert round1["training"]["steps"] == 1500
    assert round1["training"]["eval_every"] == 125
    assert set(grids["round1_grid"]["changed_fields"]) <= {
        "training.steps",
        "training.eval_every",
    }


def test_legacy_runtime_bridge_forwards_configured_interface_values(tmp_path: Path) -> None:
    from drpo import e8_experiment_config as experiment_config
    from drpo import e8_multitask_exp_tuning as exp_tuning

    config = exp_tuning.load_config(Path("configs/e8_multitask_exp_lambda_curve_completion.yaml"))
    config["experiment_id"] = "EXT-C-E8-MULTITASK-EXP-RUNTIME-BRIDGE-TEST"
    config["training"].update(
        {
            "optimizer_updates": 1500,
            "weight_decay": 0.02,
            "evaluation_every_updates": 125,
            "late_window_updates": [1000, 1125, 1250, 1375, 1500],
        }
    )
    config["evaluation"].update({"sampling_temperature": 0.7, "top_p": 0.9})
    config["model"].update({"lora_rank": 16, "lora_alpha": 32, "lora_dropout": 0.02})
    exp_tuning.validate_config(config)
    effective = experiment_config.effective_coldstart_runtime(config, "word_sorting")
    canonical_paths = exp_tuning._canonical_paths(config)
    task_root = tmp_path / "word_sorting"
    task_root.mkdir()
    grids = exp_tuning._task_grid_configs(
        config,
        canonical_paths=canonical_paths,
        task_root=task_root,
    )
    grid_path = grids["round1_grid"]["path"]
    grid_source = grids["round1_grid"]["source"]

    calls: dict[str, object] = {}

    def fake_lora_config(*args, **kwargs):
        calls["lora"] = dict(kwargs)
        return (args, kwargs)

    def fake_load_model(*args, **kwargs):
        calls["load_model"] = dict(kwargs)
        return "model"

    def fake_generate_outputs(
        model,
        tokenizer,
        prompts,
        max_new_tokens,
        do_sample,
        temperature,
        top_p,
        num_return_sequences=1,
    ):
        del model, tokenizer, prompts, max_new_tokens
        calls["sampling"] = (do_sample, temperature, top_p, num_return_sequences)
        return [["x"]]

    arena = SimpleNamespace(
        LoraConfig=fake_lora_config,
        load_model=fake_load_model,
        generate_outputs=fake_generate_outputs,
    )
    strict_calls: list[tuple[int, int]] = []

    def strict_validator(value):
        strict_calls.append(
            (int(value["training"]["steps"]), int(value["training"]["eval_every"]))
        )
        assert value["training"]["steps"] == 1200
        assert value["training"]["eval_every"] == 100

    optim = SimpleNamespace()

    def original_adamw(*args, **kwargs):
        calls["adamw"] = dict(kwargs)
        return (args, kwargs)

    optim.AdamW = original_adamw
    paper_common = SimpleNamespace(validate_grid_config=strict_validator)
    scan_common = SimpleNamespace(validate_grid_config=strict_validator)
    scan_runtime = SimpleNamespace(validate_grid_config=strict_validator)
    scan_trainer = SimpleNamespace(
        validate_grid_config=strict_validator,
        torch=SimpleNamespace(optim=optim),
    )
    modules = {
        "arena": arena,
        "paper_common": paper_common,
        "scan_common": scan_common,
        "scan_runtime": scan_runtime,
        "scan_trainer": scan_trainer,
    }

    with exp_tuning._legacy_paper_runtime_bridge(
        modules,
        effective,
        grid_path=grid_path,
        grid_source_path=grid_source,
    ):
        arena.LoraConfig(r=32, lora_alpha=64, lora_dropout=0.05)
        arena.load_model("m", None, True, False, "auto", True, parameterization="lora")
        arena.generate_outputs(None, None, [], 80, True, 0.8, 0.95, 8)
        scan_trainer.torch.optim.AdamW([], lr=5.0e-5, weight_decay=0.01)
        candidate = yaml.safe_load(grid_path.read_text(encoding="utf-8"))
        paper_common.validate_grid_config(candidate)
        bad = copy.deepcopy(candidate)
        bad["training"]["early_stop"] = not bool(bad["training"]["early_stop"])
        with pytest.raises(ValueError, match="non-runtime fields"):
            paper_common.validate_grid_config(bad)

    assert calls["lora"]["r"] == 16
    assert calls["lora"]["lora_alpha"] == 32
    assert calls["lora"]["lora_dropout"] == pytest.approx(0.02)
    assert calls["load_model"]["gradient_checkpointing"] is True
    assert calls["sampling"] == (True, 0.7, 0.9, 8)
    assert calls["adamw"]["weight_decay"] == pytest.approx(0.02)
    assert strict_calls and set(strict_calls) == {(1200, 100)}
    assert arena.LoraConfig is fake_lora_config
    assert arena.load_model is fake_load_model
    assert arena.generate_outputs is fake_generate_outputs
    assert scan_trainer.torch.optim.AdamW is original_adamw
'''
text = text[:start] + replacement + text[end:]

needle = '''    bad = copy.deepcopy(config)
    bad["sweep"]["method"] = "quadratic"
    with pytest.raises(ValueError, match="sweep.method"):
        exp_tuning.validate_config(bad)
'''
addition = needle + '''
    bad = copy.deepcopy(config)
    bad["evaluation"]["pass_k"] = 4
    with pytest.raises(ValueError, match="pass_k=8"):
        exp_tuning.validate_config(bad)

    bad = copy.deepcopy(config)
    bad["training"]["evaluation_every_updates"] = 125
    with pytest.raises(ValueError, match="late_window_updates"):
        exp_tuning.validate_config(bad)
'''
text = replace_once(text, needle, addition, "capability-tests")

old = '''    assert summary["wave_sizes"] == [16] * 13
    assert summary["scientific_status"] == "not_run"
'''
new = '''    assert summary["wave_sizes"] == [16] * 13
    assert summary["scientific_status"] == "not_run"
    assert summary["effective_runtime"]["countdown"]["training"]["optimizer_updates"] == 1200
    assert summary["effective_runtime"]["countdown"]["evaluation"][
        "sampling_temperature"
    ] == pytest.approx(0.8)
'''
text = replace_once(text, old, new, "preflight-effective-runtime-test")
path.write_text(text, encoding="utf-8")
PY
