"""Unified nine-task Structured Generation training and evaluation runtime."""

from __future__ import annotations

import copy
import importlib
import json
import os
import random
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from drpo_reference.categorical.structured_generation import (
    STRUCTURED_GENERATION_METHODS,
    TASK_NAMES,
    TaskAdapter,
    TaskInstance,
    build_adapters,
    build_task_bank,
    collate_training_items,
    completion_stats,
    dpo_objective,
    encode_training_row,
    evaluate_outputs,
    format_chat_prompt,
    move_tensor_batch_to_device,
    positive_only_objective,
    split_bank,
)
from drpo_reference.common.io import atomic_json

POLICY_ADAPTER = "default"
REFERENCE_ADAPTER = "reference"


@dataclass(frozen=True)
class HFStack:
    AutoModelForCausalLM: Any
    AutoTokenizer: Any
    BitsAndBytesConfig: Any
    LoraConfig: Any
    PeftModel: Any
    get_peft_model: Any
    get_cosine_schedule_with_warmup: Any
    prepare_model_for_kbit_training: Any


def _load_hf_stack() -> HFStack:
    try:
        transformers = importlib.import_module("transformers")
        peft = importlib.import_module("peft")
    except ImportError as exc:
        raise RuntimeError(
            "Structured Generation runtime dependencies are missing; install "
            "the paper_code[structured-generation] extra."
        ) from exc
    return HFStack(
        AutoModelForCausalLM=transformers.AutoModelForCausalLM,
        AutoTokenizer=transformers.AutoTokenizer,
        BitsAndBytesConfig=transformers.BitsAndBytesConfig,
        LoraConfig=peft.LoraConfig,
        PeftModel=peft.PeftModel,
        get_peft_model=peft.get_peft_model,
        get_cosine_schedule_with_warmup=transformers.get_cosine_schedule_with_warmup,
        prepare_model_for_kbit_training=peft.prepare_model_for_kbit_training,
    )


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value))
    if isinstance(value, list):
        return [_expand(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    return value


def load_structured_generation_config(path: str | Path) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise TypeError("Structured Generation config root must be a mapping")
    return _expand(config)


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _torch_dtype(value: str) -> torch.dtype | str:
    if value == "auto":
        return "auto"
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if value not in mapping:
        raise ValueError(f"Unsupported dtype: {value}")
    return mapping[value]


def _load_tokenizer(stack: HFStack, model_path: str) -> Any:
    tokenizer = stack.AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.eos_token_id is None or tokenizer.eos_token is None:
        raise RuntimeError("Tokenizer must define an EOS token")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def _load_policy_model(
    stack: HFStack,
    config: Mapping[str, Any],
    *,
    initial_adapter: str | None = None,
) -> Any:
    model_cfg = config["model"]
    device = _resolve_device(str(model_cfg.get("device", "auto")))
    load_in_4bit = bool(model_cfg.get("load_in_4bit", False))
    if load_in_4bit and device.type != "cuda":
        raise RuntimeError("4-bit loading requires CUDA")
    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": _torch_dtype(str(model_cfg.get("dtype", "auto"))),
    }
    if load_in_4bit:
        kwargs["device_map"] = {"": device.index or 0}
        kwargs["quantization_config"] = stack.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=(
                torch.bfloat16
                if str(model_cfg.get("dtype", "auto")) == "auto"
                else _torch_dtype(str(model_cfg["dtype"]))
            ),
            bnb_4bit_use_double_quant=True,
        )
    model = stack.AutoModelForCausalLM.from_pretrained(str(model_cfg["path"]), **kwargs)
    if load_in_4bit:
        model = stack.prepare_model_for_kbit_training(model)
    else:
        model = model.to(device)
    if initial_adapter:
        model = stack.PeftModel.from_pretrained(model, initial_adapter, is_trainable=True)
    else:
        model = stack.get_peft_model(
            model,
            stack.LoraConfig(
                r=int(model_cfg["lora_rank"]),
                lora_alpha=int(model_cfg["lora_alpha"]),
                lora_dropout=float(model_cfg["lora_dropout"]),
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=list(model_cfg["lora_target_modules"]),
            ),
        )
    if bool(model_cfg.get("gradient_checkpointing", True)):
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    model.config.use_cache = False
    return model


def _adapter_parameters(model: Any, adapter_name: str) -> list[torch.nn.Parameter]:
    token = f".{adapter_name}."
    parameters = [parameter for name, parameter in model.named_parameters() if token in name]
    if not parameters:
        raise RuntimeError(f"No parameters found for adapter {adapter_name!r}")
    return parameters


def _copy_adapter_parameters(model: Any, source: str, destination: str) -> None:
    source_token = f".{source}."
    destination_token = f".{destination}."
    source_parameters = {
        name.replace(source_token, ".<adapter>."): parameter
        for name, parameter in model.named_parameters()
        if source_token in name
    }
    destination_parameters = {
        name.replace(destination_token, ".<adapter>."): parameter
        for name, parameter in model.named_parameters()
        if destination_token in name
    }
    if source_parameters.keys() != destination_parameters.keys():
        raise RuntimeError("Policy/reference adapter structures differ")
    with torch.no_grad():
        for key, source_parameter in source_parameters.items():
            destination_parameters[key].copy_(source_parameter)


def _add_reference_adapter(
    model: Any,
) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    if not hasattr(model, "add_adapter") or not hasattr(model, "set_adapter"):
        raise RuntimeError("Reference-based methods require PEFT multi-adapter support")
    if POLICY_ADAPTER not in model.peft_config:
        raise RuntimeError("Policy adapter is missing")
    model.add_adapter(
        REFERENCE_ADAPTER,
        copy.deepcopy(model.peft_config[POLICY_ADAPTER]),
    )
    _copy_adapter_parameters(model, POLICY_ADAPTER, REFERENCE_ADAPTER)
    return (
        _adapter_parameters(model, POLICY_ADAPTER),
        _adapter_parameters(model, REFERENCE_ADAPTER),
    )


def _activate_policy(
    model: Any,
    policy_parameters: Sequence[torch.nn.Parameter],
    reference_parameters: Sequence[torch.nn.Parameter] = (),
) -> None:
    if hasattr(model, "set_adapter"):
        model.set_adapter(POLICY_ADAPTER)
    for parameter in policy_parameters:
        parameter.requires_grad_(True)
    for parameter in reference_parameters:
        parameter.requires_grad_(False)


def _activate_reference(
    model: Any,
    reference_parameters: Sequence[torch.nn.Parameter],
    *,
    trainable: bool,
) -> None:
    model.set_adapter(REFERENCE_ADAPTER)
    for parameter in reference_parameters:
        parameter.requires_grad_(trainable)


def _release_model(model: Any | None) -> None:
    if model is not None:
        del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _task_value(
    spec: Mapping[str, Any],
    key: str,
    task: str,
    default: Any = None,
) -> Any:
    value = spec.get(key, default)
    if isinstance(value, Mapping):
        if task in value:
            return value[task]
        return value.get("default", default)
    return value


def _task_runtime(config: Mapping[str, Any], task: str) -> dict[str, Any]:
    common = dict(config.get("task_runtime", {}).get("default", {}))
    common.update(config.get("task_runtime", {}).get(task, {}))
    return common


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def _batch_indices(size: int, batch_size: int, seed: int) -> Iterator[list[int]]:
    rng = random.Random(seed)
    order = list(range(size))
    while True:
        rng.shuffle(order)
        for start in range(0, size - batch_size + 1, batch_size):
            yield order[start : start + batch_size]


def _packed_batch(
    rows: Sequence[Mapping[str, Any]],
    indices: Sequence[int],
    tokenizer: Any,
    max_length: int,
    device: torch.device,
) -> dict[str, Any]:
    items = [encode_training_row(rows[index], tokenizer, max_length) for index in indices]
    packed = collate_training_items(items, int(tokenizer.pad_token_id))
    return {
        "positive": move_tensor_batch_to_device(packed["positive"], device),
        "negative": move_tensor_batch_to_device(packed["negative"], device),
        "negative_row_index": packed["negative_row_index"].to(device),
        "negative_counts": packed["negative_counts"].to(device),
    }


def _generate_batches(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    *,
    batch_size: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    num_return_sequences: int,
) -> list[list[str]]:
    outputs: list[list[str]] = []
    device = next(model.parameters()).device
    previous_padding = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        for start in range(0, len(prompts), batch_size):
            chunk = list(prompts[start : start + batch_size])
            rendered = [format_chat_prompt(tokenizer, prompt) for prompt in chunk]
            batch = tokenizer(
                rendered,
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            )
            tensors = {key: value.to(device) for key, value in batch.items()}
            prompt_length = int(tensors["input_ids"].shape[1])
            kwargs: dict[str, Any] = {
                **tensors,
                "max_new_tokens": int(max_new_tokens),
                "do_sample": bool(do_sample),
                "num_return_sequences": int(num_return_sequences),
                "pad_token_id": int(tokenizer.pad_token_id),
                "eos_token_id": int(tokenizer.eos_token_id),
                "use_cache": True,
            }
            if do_sample:
                kwargs.update(
                    {
                        "temperature": float(temperature),
                        "top_p": float(top_p),
                    }
                )
            generated = model.generate(**kwargs)
            decoded = tokenizer.batch_decode(
                generated[:, prompt_length:],
                skip_special_tokens=True,
            )
            for index in range(len(chunk)):
                first = index * num_return_sequences
                outputs.append(decoded[first : first + num_return_sequences])
    finally:
        tokenizer.padding_side = previous_padding
    return outputs


def evaluate_model(
    model: Any,
    tokenizer: Any,
    *,
    adapter: TaskAdapter,
    instances: Mapping[str, TaskInstance],
    rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    task: str,
    seed: int,
) -> dict[str, Any]:
    runtime = _task_runtime(config, task)
    evaluation = config["evaluation"]
    selected = list(rows[: int(evaluation["examples"])])
    prompts = [str(row["prompt"]) for row in selected]
    was_training = bool(model.training)
    cache = getattr(model.config, "use_cache", False)
    checkpointing = bool(getattr(model, "is_gradient_checkpointing", False))
    model.eval()
    if checkpointing and hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    model.config.use_cache = True
    _seed_all(seed)
    try:
        with torch.no_grad():
            greedy = _generate_batches(
                model,
                tokenizer,
                prompts,
                batch_size=int(runtime["evaluation_batch_size"]),
                max_new_tokens=int(runtime["max_new_tokens"]),
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                num_return_sequences=1,
            )
            sampled = _generate_batches(
                model,
                tokenizer,
                prompts,
                batch_size=int(runtime["evaluation_batch_size"]),
                max_new_tokens=int(runtime["max_new_tokens"]),
                do_sample=True,
                temperature=float(evaluation["temperature"]),
                top_p=float(evaluation["top_p"]),
                num_return_sequences=int(evaluation["pass_k"]),
            )
    finally:
        model.config.use_cache = cache
        if checkpointing and hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        model.train(was_training)
    return evaluate_outputs(
        adapter,
        instances,
        selected,
        [group[0] for group in greedy],
        sampled,
    )


def _method_optimizers(
    stack: HFStack,
    model: Any,
    config: Mapping[str, Any],
    method: str,
) -> tuple[
    list[torch.nn.Parameter],
    list[torch.nn.Parameter],
    torch.optim.Optimizer,
    Any,
]:
    training = config["training"]
    steps = int(training["optimizer_updates"])
    if method == "dpo":
        policy, reference = _add_reference_adapter(model)
    else:
        policy = [parameter for parameter in model.parameters() if parameter.requires_grad]
        reference = []
    optimizer = torch.optim.AdamW(
        policy,
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    warmup = int(steps * float(training["warmup_ratio"]))
    scheduler = stack.get_cosine_schedule_with_warmup(
        optimizer,
        warmup,
        steps,
    )
    _activate_policy(model, policy, reference)
    return policy, reference, optimizer, scheduler


def _cell_loss(
    model: Any,
    packed: Mapping[str, Any],
    *,
    method: str,
    method_spec: Mapping[str, Any],
    task: str,
    policy_parameters: Sequence[torch.nn.Parameter],
    reference_parameters: Sequence[torch.nn.Parameter],
) -> torch.Tensor:
    positive = packed["positive"]
    negative = packed["negative"]
    row_index = packed["negative_row_index"]
    counts = packed["negative_counts"]

    if method == "dpo":
        _activate_reference(
            model,
            reference_parameters,
            trainable=False,
        )
        model.eval()
        with torch.no_grad():
            reference_positive = completion_stats(model, positive)
            reference_negative = completion_stats(model, negative)
        _activate_policy(
            model,
            policy_parameters,
            reference_parameters,
        )
        model.eval()
        policy_positive = completion_stats(model, positive)
        policy_negative = completion_stats(model, negative)
        return dpo_objective(
            policy_positive["sum_logprob"],
            policy_negative["sum_logprob"],
            reference_positive["sum_logprob"],
            reference_negative["sum_logprob"],
            row_index,
            counts,
            beta=float(
                _task_value(
                    method_spec,
                    "beta",
                    task,
                )
            ),
        )

    positive_stats = completion_stats(model, positive)
    if method == "positive_only":
        return positive_only_objective(positive_stats["mean_logprob"])
    raise ValueError(f"Unsupported Structured Generation method: {method}")


def _train_cell(
    stack: HFStack,
    tokenizer: Any,
    *,
    config: Mapping[str, Any],
    task: str,
    method: str,
    method_spec: Mapping[str, Any],
    adapter: TaskAdapter,
    instances: Mapping[str, TaskInstance],
    train_rows: Sequence[Mapping[str, Any]],
    validation_rows: Sequence[Mapping[str, Any]],
    test_rows: Sequence[Mapping[str, Any]],
    seed: int,
) -> dict[str, Any]:
    training = config["training"]
    runtime = _task_runtime(config, task)
    initial_adapter = None
    if method == "dpo":
        configured_adapter = _task_value(
            method_spec,
            "initial_adapter",
            task,
        )
        if not configured_adapter:
            raise ValueError("DPO requires the short-SFT initial adapter")
        initial_adapter = str(configured_adapter)

    _seed_all(int(config["initialization_seed"]))
    model = _load_policy_model(
        stack,
        config,
        initial_adapter=initial_adapter,
    )
    policy, reference, optimizer, scheduler = _method_optimizers(
        stack,
        model,
        config,
        method,
    )

    _seed_all(seed)
    if method == "dpo":
        model.eval()
    else:
        model.train()

    batch_stream = _batch_indices(
        len(train_rows),
        int(training["micro_batch"]),
        seed,
    )
    accumulation = int(training["gradient_accumulation"])
    steps = int(training["optimizer_updates"])
    device = next(model.parameters()).device
    trajectory: list[dict[str, Any]] = []
    completed_update = 0

    for update in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        loss_total = 0.0

        for _ in range(accumulation):
            packed = _packed_batch(
                train_rows,
                next(batch_stream),
                tokenizer,
                int(runtime["max_length"]),
                device,
            )
            loss = _cell_loss(
                model,
                packed,
                method=method,
                method_spec=method_spec,
                task=task,
                policy_parameters=policy,
                reference_parameters=reference,
            )
            (loss / accumulation).backward()
            loss_total += float(loss.detach()) / accumulation

        policy_gradient = torch.nn.utils.clip_grad_norm_(
            policy,
            float(training["max_grad_norm"]),
        )
        optimizer.step()
        scheduler.step()
        completed_update = update

        should_evaluate = update % int(training["evaluation_every_updates"]) == 0 or update == steps
        if should_evaluate:
            _activate_policy(
                model,
                policy,
                reference,
            )
            metrics = evaluate_model(
                model,
                tokenizer,
                adapter=adapter,
                instances=instances,
                rows=validation_rows,
                config=config,
                task=task,
                seed=(int(config["evaluation"]["seed"]) + update * 1009 + seed),
            )
            trajectory.append(
                {
                    "update": update,
                    "loss": loss_total,
                    "policy_gradient_l2": float(policy_gradient),
                    **metrics,
                }
            )
            if method == "dpo":
                model.eval()
            else:
                model.train()

    _activate_policy(
        model,
        policy,
        reference,
    )
    terminal_validation = evaluate_model(
        model,
        tokenizer,
        adapter=adapter,
        instances=instances,
        rows=validation_rows,
        config=config,
        task=task,
        seed=int(config["evaluation"]["seed"]) + seed,
    )
    terminal_test = evaluate_model(
        model,
        tokenizer,
        adapter=adapter,
        instances=instances,
        rows=test_rows,
        config=config,
        task=task,
        seed=int(config["evaluation"]["seed"]) + seed + 1,
    )

    result = {
        "task": task,
        "method": method,
        "seed": seed,
        "optimizer_updates_requested": steps,
        "optimizer_updates_completed": completed_update,
        "method_parameters": dict(method_spec),
        "validation_trajectory": trajectory,
        "terminal_validation": terminal_validation,
        "terminal_test": terminal_test,
        "task_performance": (
            None if terminal_test is None else {"pass_at_k": terminal_test["pass_at_k"]}
        ),
        "valid_or_structure_diagnostic": {
            "greedy_valid_rate": terminal_test["greedy_valid_rate"],
            "sampled_valid_rate": terminal_test["sampled_valid_rate"],
        },
    }
    _release_model(model)
    return result


def run_structured_generation(
    *,
    config_path: str | Path,
    output_root: str | Path,
    tasks: Sequence[str] | None = None,
    methods: Sequence[str] | None = None,
) -> dict[str, Any]:
    config = load_structured_generation_config(config_path)
    requested_tasks = tuple(tasks or config["tasks"]["names"])
    requested_methods = tuple(
        methods
        or [name for name, spec in config["methods"].items() if bool(spec.get("enabled", True))]
    )
    unknown_tasks = sorted(set(requested_tasks) - set(TASK_NAMES))
    unknown_methods = sorted(set(requested_methods) - set(STRUCTURED_GENERATION_METHODS))
    if unknown_tasks:
        raise ValueError(f"Unknown Structured Generation tasks: {unknown_tasks}")
    if unknown_methods:
        raise ValueError(f"Unknown Structured Generation methods: {unknown_methods}")

    output = Path(output_root).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_root = Path(str(config["sources_root"])).expanduser().resolve()
    adapter_config = copy.deepcopy(config)
    adapter_config["tasks"]["names"] = list(requested_tasks)
    adapters = build_adapters(
        adapter_config,
        source_root,
    )

    stack = _load_hf_stack()
    tokenizer = _load_tokenizer(
        stack,
        str(config["model"]["path"]),
    )
    data = config["data"]
    all_runs: list[dict[str, Any]] = []

    for task in requested_tasks:
        task_seed = int(data["generation_seed"]) + TASK_NAMES.index(task) * 100_003
        rows, instances = build_task_bank(
            adapters[task],
            candidate_rows=int(data["candidate_rows"]),
            accepted_rows=(
                int(data["train_rows"]) + int(data["validation_rows"]) + int(data["test_rows"])
            ),
            negatives_per_prompt=int(data["negatives_per_prompt"]),
            seed=task_seed,
        )
        partitions = split_bank(
            rows,
            train_rows=int(data["train_rows"]),
            validation_rows=int(data["validation_rows"]),
            test_rows=int(data["test_rows"]),
            seed=int(data["split_seed"]),
        )
        for split, split_rows in partitions.items():
            _write_jsonl(
                output / "data" / task / f"{split}.jsonl",
                split_rows,
            )

        for method in requested_methods:
            method_spec = config["methods"][method]
            result = _train_cell(
                stack,
                tokenizer,
                config=config,
                task=task,
                method=method,
                method_spec=method_spec,
                adapter=adapters[task],
                instances=instances,
                train_rows=partitions["train"],
                validation_rows=partitions["validation"],
                test_rows=partitions["test"],
                seed=int(config["training_seed"]),
            )
            atomic_json(
                output / "results" / task / f"{method}.json",
                result,
            )
            all_runs.append(result)

    summary = {
        "tasks": list(requested_tasks),
        "methods": list(requested_methods),
        "runs": all_runs,
    }
    atomic_json(
        output / "SUMMARY.json",
        summary,
    )
    return summary


__all__ = [
    "HFStack",
    "evaluate_model",
    "load_structured_generation_config",
    "run_structured_generation",
]
