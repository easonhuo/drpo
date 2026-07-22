#!/usr/bin/env python3
"""Training core for the E8 alpha=1 c-only scan development pilot."""
from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch.utils.data import DataLoader

from drpo.countdown_e8_alpha1_c_scan_common import (
    EXPERIMENT_ID,
    TOPR_POLICY_ADAPTER,
    TOPR_REFERENCE_ADAPTER,
    VERSION,
    Cell,
    ContinuousUniqueBankDataset,
    _identity,
    arena,
    atomic_json,
    branch_balanced_reference_loss,
    continuous_exp_weights,
    joint_topr_diagnostics,
    joint_topr_negative_weights,
    load_yaml,
    make_continuous_unique_bank_collator,
    mean_unique_negative_term,
    unique_negative_expressions,
    validate_grid_config,
    weight_diagnostics,
)


def _identity_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return json.dumps(left, sort_keys=True) == json.dumps(right, sort_keys=True)


def _best_row(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any] | None:
    candidates = [row for row in rows if row.get(key) is not None]
    if not candidates:
        return None
    winner = max(candidates, key=lambda row: float(row[key]))
    return {"step": int(winner["step"]), "value": float(winner[key])}


def _evaluate_validation(
    *,
    model: Any,
    tokenizer: Any,
    val_rows: list[dict[str, Any]],
    known_structures: set[str],
    model_cfg: Mapping[str, Any],
    eval_cfg: Mapping[str, Any],
    step: int,
    pass64_every: int,
    pass64_enabled: bool,
    examples_override: int | None,
) -> dict[str, Any]:
    examples = int(eval_cfg["examples"]) if examples_override is None else examples_override
    selected = val_rows[:examples]
    pass8 = arena.evaluate_rows(
        model,
        tokenizer,
        selected,
        int(eval_cfg["batch_size"]),
        int(model_cfg["max_new_tokens"]),
        8,
        int(eval_cfg["seed"]),
        known_structures,
    )
    row: dict[str, Any] = {
        "step": int(step),
        "val_greedy": float(pass8["greedy_success"]),
        "val_pass_at_8": float(pass8["pass_at_k"]),
        "val_valid_rate": float(pass8["valid_rate"]),
        "val_pass_at_64": None,
        "validation_examples": len(selected),
    }
    if pass64_enabled and step % pass64_every == 0:
        pass64 = arena.evaluate_rows(
            model,
            tokenizer,
            selected,
            int(eval_cfg["batch_size"]),
            int(model_cfg["max_new_tokens"]),
            64,
            int(eval_cfg["seed"]),
            known_structures,
        )
        row["val_pass_at_64"] = float(pass64["pass_at_k"])
    return row


def _parameter_update_norm(
    before: Sequence[torch.Tensor], parameters: Sequence[torch.nn.Parameter]
) -> float:
    total = torch.zeros((), dtype=torch.float64)
    for saved, parameter in zip(before, parameters, strict=True):
        delta = parameter.detach().float().cpu() - saved
        total += delta.double().square().sum()
    return float(torch.sqrt(total).item())


def _adapter_parameters(model: Any, adapter_name: str) -> list[torch.nn.Parameter]:
    token = f".{adapter_name}."
    parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if token in name
    ]
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
        missing_source = sorted(destination_parameters.keys() - source_parameters.keys())
        missing_destination = sorted(source_parameters.keys() - destination_parameters.keys())
        raise RuntimeError(
            "TOPR adapter structures differ: "
            f"missing_source={missing_source[:3]}, "
            f"missing_destination={missing_destination[:3]}"
        )
    with torch.no_grad():
        for key in sorted(source_parameters):
            destination_parameters[key].copy_(source_parameters[key])


def _capture_rng_state() -> tuple[torch.Tensor, list[torch.Tensor] | None]:
    cpu_state = torch.get_rng_state().clone()
    cuda_states = (
        [state.clone() for state in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available()
        else None
    )
    return cpu_state, cuda_states


def _restore_rng_state(
    state: tuple[torch.Tensor, list[torch.Tensor] | None]
) -> None:
    cpu_state, cuda_states = state
    torch.set_rng_state(cpu_state)
    if cuda_states is not None:
        torch.cuda.set_rng_state_all(cuda_states)


def _joint_optimizer_step_with_last_finite_guard(
    policy_optimizer: torch.optim.Optimizer,
    reference_optimizer: torch.optim.Optimizer,
    policy_parameters: Sequence[torch.nn.Parameter],
    reference_parameters: Sequence[torch.nn.Parameter],
) -> bool:
    """Step both adapters atomically at the parameter level.

    If either adapter becomes non-finite, both parameter sets are restored to
    their joint pre-step values and the caller stops the run. Optimizer-state
    mutation is retained only in the failing process, which terminates without
    resuming from that state.
    """
    all_parameters = [*policy_parameters, *reference_parameters]
    snapshots = [parameter.detach().clone() for parameter in all_parameters]
    reference_optimizer.step()
    policy_optimizer.step()
    finite = all(bool(torch.isfinite(parameter).all()) for parameter in all_parameters)
    if finite:
        return True
    with torch.no_grad():
        for parameter, saved in zip(all_parameters, snapshots, strict=True):
            parameter.copy_(saved)
    return False


def _empty_standard_diagnostics(
    packed: Mapping[str, torch.Tensor], *, weight: float
) -> dict[str, float | None]:
    return {
        "negative_surprisal_mean": None,
        "u_mean": None,
        "u_p10": None,
        "u_p50": None,
        "u_p90": None,
        "weight_mean": weight,
        "weight_p10": weight,
        "weight_p50": weight,
        "weight_p90": weight,
        "unique_negative_count_mean": float(packed["unique_counts"].float().mean()),
        "raw_bank_count_mean": float(packed["raw_bank_counts"].float().mean()),
        "duplicates_removed_mean": float(
            (packed["raw_bank_counts"] - packed["unique_counts"])
            .float()
            .mean()
        ),
    }


def train_cell(
    *,
    cell: Cell,
    model_path: Path,
    bank: Path,
    val: Path,
    base_config_path: Path,
    grid_config_path: Path,
    output_dir: Path,
    repo: Path,
    smoke: bool = False,
) -> dict[str, Any]:
    base_config = load_yaml(base_config_path)
    grid_config = load_yaml(grid_config_path)
    validate_grid_config(grid_config)
    train_cfg = base_config["offline_training"]
    model_cfg = base_config["model"]
    eval_cfg = base_config["evaluation"]
    fixed_training = grid_config["training"]
    liveness = grid_config["execution"]["liveness"]
    steps = int(liveness["steps"]) if smoke else int(fixed_training["steps"])
    eval_every = steps if smoke else int(fixed_training["eval_every"])
    pass64_every = 10**9 if smoke else int(fixed_training["pass64_every"])
    examples_override = int(liveness["validation_examples"]) if smoke else None
    seed = int(train_cfg["seed"]) + int(cell.seed_offset)
    arena.seed_all(seed)
    is_topr = cell.method == "joint_fitted_reference_topr"

    output_dir.mkdir(parents=True, exist_ok=True)
    identity = _identity(
        repo=repo,
        model_path=model_path,
        bank=bank,
        val=val,
        base_config=base_config_path,
        grid_config=grid_config_path,
        cell=cell,
        smoke=smoke,
    )
    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        existing = json.loads(summary_path.read_text())
        if _identity_equal(existing.get("run_identity", {}), identity):
            return existing
        raise RuntimeError(f"Stale result identity at {summary_path}; use a new work_dir")

    tokenizer = arena.load_tokenizer(str(model_path))
    train_rows = arena.read_jsonl(bank)
    val_rows = arena.read_jsonl(val)
    unique_counts = [len(unique_negative_expressions(row)) for row in train_rows]
    raw_counts = [len(row.get("negative_bank", [])) for row in train_rows]
    if min(unique_counts) < 1:
        raise RuntimeError("At least one training row has no unique negative")
    dataset = ContinuousUniqueBankDataset(
        train_rows, tokenizer, int(model_cfg["max_length"])
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=int(train_cfg["micro_batch"]),
        shuffle=True,
        generator=generator,
        collate_fn=make_continuous_unique_bank_collator(tokenizer.pad_token_id),
        num_workers=int(train_cfg["num_workers"]),
    )
    iterator = iter(loader)
    model = arena.load_model(
        str(model_path),
        adapter_path=None,
        trainable_adapter=True,
        load_in_4bit=bool(model_cfg.get("load_in_4bit", False)),
        dtype=str(model_cfg.get("dtype", "auto")),
        gradient_checkpointing=True,
        parameterization="lora",
    )
    device = next(model.parameters()).device

    policy_optimizer: torch.optim.Optimizer | None = None
    reference_optimizer: torch.optim.Optimizer | None = None
    policy_scheduler: Any = None
    reference_scheduler: Any = None
    reference_parameters: list[torch.nn.Parameter] = []
    if is_topr:
        if not hasattr(model, "add_adapter") or not hasattr(model, "set_adapter"):
            raise RuntimeError("Joint fitted-reference TOPR requires PEFT multi-adapter support")
        if TOPR_POLICY_ADAPTER not in model.peft_config:
            raise RuntimeError("Loaded LoRA model has no default policy adapter")
        if TOPR_REFERENCE_ADAPTER in model.peft_config:
            raise RuntimeError("Reference adapter already exists before TOPR initialization")
        model.add_adapter(
            TOPR_REFERENCE_ADAPTER,
            copy.deepcopy(model.peft_config[TOPR_POLICY_ADAPTER]),
        )
        _copy_adapter_parameters(model, TOPR_POLICY_ADAPTER, TOPR_REFERENCE_ADAPTER)
        model.set_adapter(TOPR_POLICY_ADAPTER)
        trainable = _adapter_parameters(model, TOPR_POLICY_ADAPTER)
        reference_parameters = _adapter_parameters(model, TOPR_REFERENCE_ADAPTER)
        policy_optimizer = torch.optim.AdamW(
            trainable, lr=float(train_cfg["learning_rate"]), weight_decay=0.01
        )
        reference_optimizer = torch.optim.AdamW(
            reference_parameters,
            lr=float(train_cfg["learning_rate"])
            * float(grid_config["reference_policy"]["learning_rate_multiplier"]),
            weight_decay=0.01,
        )
        warmup_steps = max(1, int(steps * float(train_cfg["warmup_ratio"])))
        policy_scheduler = arena.get_cosine_schedule_with_warmup(
            policy_optimizer, warmup_steps, steps
        )
        reference_scheduler = arena.get_cosine_schedule_with_warmup(
            reference_optimizer, warmup_steps, steps
        )
        optimizer = None
        scheduler = None
    else:
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable, lr=float(train_cfg["learning_rate"]), weight_decay=0.01
        )
        scheduler = arena.get_cosine_schedule_with_warmup(
            optimizer,
            max(1, int(steps * float(train_cfg["warmup_ratio"]))),
            steps,
        )

    best_dir = output_dir / "best_pass8_adapter"
    terminal_dir = output_dir / "terminal_adapter"
    last_finite_dir = output_dir / "last_finite_adapter"
    validation_diagnostics_path = output_dir / "validation_diagnostics.jsonl"
    training_diagnostics_path = output_dir / "training_diagnostics.jsonl"
    metric_rows: list[dict[str, Any]] = []
    checkpoint_records: list[dict[str, Any]] = []
    best_pass8 = -float("inf")
    best_step = 0
    numerical_failure: str | None = None
    terminal_step = 0
    last_finite_step = 0
    stop_reason = "smoke_complete" if smoke else "max_steps"
    initial_ratio_max_abs: float | None = None
    known_structures = {
        row.get("oracle_structure") or arena.expression_structure(row["oracle"])
        for row in train_rows
    }

    def activate_policy() -> None:
        if is_topr:
            model.set_adapter(TOPR_POLICY_ADAPTER)

    def evaluate(step: int) -> dict[str, Any]:
        activate_policy()
        row = _evaluate_validation(
            model=model,
            tokenizer=tokenizer,
            val_rows=val_rows,
            known_structures=known_structures,
            model_cfg=model_cfg,
            eval_cfg=eval_cfg,
            step=step,
            pass64_every=pass64_every,
            pass64_enabled=not smoke,
            examples_override=examples_override,
        )
        row.update(
            {
                "method": cell.method,
                "alpha": cell.alpha,
                "c": cell.c,
                "delta_v": cell.delta_v if cell.method == "asymre" else None,
                "positive_coefficient": (
                    1.0 - cell.delta_v if cell.method == "asymre" else 1.0
                ),
                "negative_repulsion_coefficient": (
                    None
                    if is_topr
                    else 1.0 + cell.delta_v
                    if cell.method == "asymre"
                    else cell.alpha
                ),
                "distance_control": cell.method not in {
                    "asymre",
                    "joint_fitted_reference_topr",
                },
                "behavior_relative_ratio_control": is_topr,
                "evaluated_adapter": TOPR_POLICY_ADAPTER if is_topr else None,
                "test_data_used": False,
            }
        )
        metric_rows.append(row)
        with validation_diagnostics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        return row

    initial = evaluate(0)
    best_pass8 = float(initial["val_pass_at_8"])
    checkpoint_records.append(
        arena.save_local_model_checkpoint(model, tokenizer, best_dir, "best_pass8", 0)
    )
    model.train()
    activate_policy()

    standard_diagnostic_keys = (
        "negative_surprisal_mean",
        "u_mean",
        "u_p10",
        "u_p50",
        "u_p90",
        "weight_mean",
        "weight_p10",
        "weight_p50",
        "weight_p90",
        "unique_negative_count_mean",
        "raw_bank_count_mean",
        "duplicates_removed_mean",
    )
    topr_diagnostic_keys = (
        "log_ratio_mean",
        "log_ratio_p10",
        "log_ratio_p50",
        "log_ratio_p90",
        "clipped_at_one_fraction",
        "reference_loss",
        "reference_positive_lp",
        "reference_negative_lp",
    )
    diagnostic_keys = standard_diagnostic_keys + topr_diagnostic_keys

    for update_step in range(1, steps + 1):
        if is_topr:
            assert policy_optimizer is not None
            assert reference_optimizer is not None
            policy_optimizer.zero_grad(set_to_none=True)
            reference_optimizer.zero_grad(set_to_none=True)
        else:
            assert optimizer is not None
            optimizer.zero_grad(set_to_none=True)

        accum: dict[str, float | None] = {
            "loss": 0.0,
            "positive_lp": 0.0,
            "weighted_negative_lp": 0.0,
            **{key: 0.0 for key in diagnostic_keys},
        }
        diagnostic_observations = {key: 0 for key in diagnostic_keys}
        abort_update = False
        for accumulation_index in range(int(train_cfg["gradient_accumulation"])):
            try:
                packed = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                packed = next(iterator)
            positive_batch = arena.move_to_device(packed["positive"], device)
            bank_batch = arena.move_to_device(packed["bank"], device)

            if is_topr:
                rng_state = _capture_rng_state()
                model.set_adapter(TOPR_REFERENCE_ADAPTER)
                reference_positive_stats = arena.completion_stats(model, positive_batch)
                reference_bank_stats = arena.completion_stats(model, bank_batch)
                reference_positive_lp = reference_positive_stats["seq_lp"].mean()
                reference_negative_lp = mean_unique_negative_term(
                    reference_bank_stats["seq_lp"],
                    torch.ones_like(reference_bank_stats["seq_lp"]),
                    packed["bank_row_index"],
                    packed["unique_counts"],
                )
                reference_loss = branch_balanced_reference_loss(
                    reference_positive_lp,
                    reference_bank_stats["seq_lp"],
                    packed["bank_row_index"],
                    packed["unique_counts"],
                )
                if not bool(torch.isfinite(reference_loss)):
                    numerical_failure = (
                        f"nonfinite_reference_loss_at_step_{update_step}"
                    )
                    stop_reason = numerical_failure
                    abort_update = True
                    break
                (
                    reference_loss / int(train_cfg["gradient_accumulation"])
                ).backward()

                model.set_adapter(TOPR_POLICY_ADAPTER)
                _restore_rng_state(rng_state)
                positive_stats = arena.completion_stats(model, positive_batch)
                bank_stats = arena.completion_stats(model, bank_batch)
                positive_lp = positive_stats["seq_lp"].mean()
                weights, log_ratio = joint_topr_negative_weights(
                    bank_stats, reference_bank_stats
                )
                if update_step == 1 and accumulation_index == 0:
                    initial_ratio_max_abs = float(log_ratio.abs().max().item())
                    tolerance = float(
                        fixed_training["initial_ratio_max_abs_tolerance"]
                    )
                    if initial_ratio_max_abs > tolerance:
                        numerical_failure = "initial_policy_reference_ratio_mismatch"
                        stop_reason = numerical_failure
                        abort_update = True
                        break
                weighted_negative_lp = mean_unique_negative_term(
                    bank_stats["seq_lp"],
                    weights,
                    packed["bank_row_index"],
                    packed["unique_counts"],
                )
                raw_loss = -(positive_lp - weighted_negative_lp)
                diagnostics: dict[str, float | None] = {
                    **_empty_standard_diagnostics(packed, weight=0.0),
                    **joint_topr_diagnostics(
                        log_ratio,
                        weights,
                        packed["unique_counts"],
                        packed["raw_bank_counts"],
                    ),
                    "reference_loss": float(reference_loss.detach()),
                    "reference_positive_lp": float(reference_positive_lp.detach()),
                    "reference_negative_lp": float(reference_negative_lp.detach()),
                }
                diagnostics["weight_mean"] = float(weights.float().mean())
                diagnostics["weight_p10"] = float(torch.quantile(weights.float(), 0.10))
                diagnostics["weight_p50"] = float(torch.quantile(weights.float(), 0.50))
                diagnostics["weight_p90"] = float(torch.quantile(weights.float(), 0.90))
            else:
                positive_stats = arena.completion_stats(model, positive_batch)
                positive_lp = positive_stats["seq_lp"].mean()
                if cell.alpha == 0.0:
                    weighted_negative_lp = torch.zeros(
                        (), device=device, dtype=positive_lp.dtype
                    )
                    diagnostics = _empty_standard_diagnostics(packed, weight=0.0)
                else:
                    bank_stats = arena.completion_stats(model, bank_batch)
                    weights = continuous_exp_weights(
                        bank_stats["seq_lp"], alpha=cell.alpha, c=cell.c
                    )
                    weighted_negative_lp = mean_unique_negative_term(
                        bank_stats["seq_lp"],
                        weights,
                        packed["bank_row_index"],
                        packed["unique_counts"],
                    )
                    diagnostics = weight_diagnostics(
                        bank_stats["seq_lp"],
                        weights,
                        packed["unique_counts"],
                        packed["raw_bank_counts"],
                    )
                raw_loss = -(
                    ((1.0 - cell.delta_v) if cell.method == "asymre" else 1.0)
                    * positive_lp
                    - weighted_negative_lp
                )

            if not bool(torch.isfinite(raw_loss)):
                numerical_failure = f"nonfinite_loss_at_step_{update_step}"
                stop_reason = numerical_failure
                abort_update = True
                break
            (raw_loss / int(train_cfg["gradient_accumulation"])).backward()
            divisor = float(train_cfg["gradient_accumulation"])
            accum["loss"] = float(accum["loss"] or 0.0) + float(
                raw_loss.detach()
            ) / divisor
            accum["positive_lp"] = float(accum["positive_lp"] or 0.0) + float(
                positive_lp.detach()
            ) / divisor
            accum["weighted_negative_lp"] = float(
                accum["weighted_negative_lp"] or 0.0
            ) + float(weighted_negative_lp.detach()) / divisor
            for key in diagnostic_keys:
                value = diagnostics.get(key)
                if value is None:
                    continue
                accum[key] = float(accum[key] or 0.0) + float(value) / divisor
                diagnostic_observations[key] += 1
        if abort_update:
            break

        maximum_gradient_norm = float(train_cfg["maximum_gradient_norm"])
        if is_topr:
            activate_policy()
            policy_raw_grad_norm = torch.nn.utils.clip_grad_norm_(
                trainable, maximum_gradient_norm
            )
            model.set_adapter(TOPR_REFERENCE_ADAPTER)
            reference_raw_grad_norm = torch.nn.utils.clip_grad_norm_(
                reference_parameters, maximum_gradient_norm
            )
            activate_policy()
            if not bool(torch.isfinite(policy_raw_grad_norm)):
                numerical_failure = f"nonfinite_policy_gradient_at_step_{update_step}"
                stop_reason = numerical_failure
                break
            if not bool(torch.isfinite(reference_raw_grad_norm)):
                numerical_failure = f"nonfinite_reference_gradient_at_step_{update_step}"
                stop_reason = numerical_failure
                break
        else:
            raw_grad_norm = torch.nn.utils.clip_grad_norm_(
                trainable, maximum_gradient_norm
            )
            if not bool(torch.isfinite(raw_grad_norm)):
                numerical_failure = f"nonfinite_gradient_at_step_{update_step}"
                stop_reason = numerical_failure
                break

        sample_update_norm = (
            update_step % int(train_cfg["log_every"]) == 0 or update_step == steps
        )
        policy_before = (
            [parameter.detach().float().cpu().clone() for parameter in trainable]
            if sample_update_norm
            else []
        )
        reference_before = (
            [
                parameter.detach().float().cpu().clone()
                for parameter in reference_parameters
            ]
            if is_topr and sample_update_norm
            else []
        )
        if is_topr:
            assert policy_optimizer is not None
            assert reference_optimizer is not None
            if not _joint_optimizer_step_with_last_finite_guard(
                policy_optimizer,
                reference_optimizer,
                trainable,
                reference_parameters,
            ):
                numerical_failure = f"nonfinite_parameters_at_step_{update_step}"
                stop_reason = numerical_failure
                break
            policy_update_norm = (
                _parameter_update_norm(policy_before, trainable)
                if sample_update_norm
                else None
            )
            reference_update_norm = (
                _parameter_update_norm(reference_before, reference_parameters)
                if sample_update_norm
                else None
            )
            policy_scheduler.step()
            reference_scheduler.step()
            activate_policy()
        else:
            assert optimizer is not None
            if not arena.optimizer_step_with_last_finite_guard(optimizer, trainable):
                numerical_failure = f"nonfinite_parameters_at_step_{update_step}"
                stop_reason = numerical_failure
                break
            policy_update_norm = (
                _parameter_update_norm(policy_before, trainable)
                if sample_update_norm
                else None
            )
            reference_update_norm = None
            scheduler.step()

        terminal_step = update_step
        last_finite_step = update_step
        for key in diagnostic_keys:
            if diagnostic_observations[key] == 0:
                accum[key] = None
        if is_topr:
            accum["raw_gradient_norm"] = float(policy_raw_grad_norm)
            accum["policy_raw_gradient_norm"] = float(policy_raw_grad_norm)
            accum["reference_raw_gradient_norm"] = float(reference_raw_grad_norm)
            accum["optimizer_update_norm"] = policy_update_norm
            accum["policy_optimizer_update_norm"] = policy_update_norm
            accum["reference_optimizer_update_norm"] = reference_update_norm
            accum["initial_ratio_max_abs"] = initial_ratio_max_abs
        else:
            accum["raw_gradient_norm"] = float(raw_grad_norm)
            accum["optimizer_update_norm"] = policy_update_norm

        if update_step % int(train_cfg["log_every"]) == 0 or update_step == steps:
            training_record = {
                "cell": cell.name,
                "step": update_step,
                "method": cell.method,
                "alpha": cell.alpha,
                "c": cell.c,
                "delta_v": cell.delta_v if cell.method == "asymre" else None,
                "positive_coefficient": (
                    1.0 - cell.delta_v if cell.method == "asymre" else 1.0
                ),
                "negative_repulsion_coefficient": (
                    None
                    if is_topr
                    else 1.0 + cell.delta_v
                    if cell.method == "asymre"
                    else cell.alpha
                ),
                "distance_control": cell.method not in {
                    "asymre",
                    "joint_fitted_reference_topr",
                },
                "behavior_relative_ratio_control": is_topr,
                "test_data_used": False,
                **accum,
            }
            with training_diagnostics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(training_record, ensure_ascii=False) + "\n")
            print(json.dumps(training_record), flush=True)
        if update_step % eval_every == 0 or update_step == steps:
            row = evaluate(update_step)
            value = float(row["val_pass_at_8"])
            if value > best_pass8:
                best_pass8 = value
                best_step = update_step
                checkpoint_records = [
                    record
                    for record in checkpoint_records
                    if record["kind"] != "best_pass8"
                ]
                checkpoint_records.append(
                    arena.save_local_model_checkpoint(
                        model, tokenizer, best_dir, "best_pass8", update_step
                    )
                )
            model.train()
            activate_policy()

    activate_policy()
    if numerical_failure:
        checkpoint_records.append(
            arena.save_local_model_checkpoint(
                model, tokenizer, last_finite_dir, "last_finite", last_finite_step
            )
        )
        terminal_kind = "last_finite"
    else:
        checkpoint_records.append(
            arena.save_local_model_checkpoint(
                model, tokenizer, terminal_dir, "terminal", terminal_step
            )
        )
        terminal_kind = "terminal"

    fieldnames = sorted({key for row in metric_rows for key in row})
    with (output_dir / "metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(arena.csv_safe_row(row) for row in metric_rows)

    metric_bests = {
        key: _best_row(metric_rows, key)
        for key in ("val_greedy", "val_pass_at_8", "val_pass_at_64", "val_valid_rate")
    }
    if is_topr:
        objective_formula = (
            "policy=positive_mean_lp-mean_unique_negative("
            "stopgrad(min(pi/mu,1))*negative_mean_lp);"
            "reference=0.5*positive_mean_lp+0.5*mean_unique_negative(negative_mean_lp)"
        )
        weight_formula = "exp(min(sum_logpi-sum_logmu,0))"
        checkpoint_policy = (
            "server-local dual LoRA only; default=policy, reference=joint fitted reference"
        )
    elif cell.method == "asymre":
        objective_formula = "(1-delta_v)*positive_lp-(1+delta_v)*negative_lp"
        weight_formula = str(grid_config["remoteness"]["weight"])
        checkpoint_policy = "server-local LoRA only"
    else:
        objective_formula = "positive_lp-weighted_negative_lp"
        weight_formula = str(grid_config["remoteness"]["weight"])
        checkpoint_policy = "server-local LoRA only"

    manifest = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "version": VERSION,
        "result_status": "smoke_only" if smoke else "pilot",
        "registration_state": str(grid_config["registration_state"]),
        "cell": cell.name,
        "method": cell.method,
        "method_identity": (
            "joint_fitted_reference_topr_not_canonical_topr" if is_topr else None
        ),
        "alpha": cell.alpha,
        "c": cell.c,
        "delta_v": cell.delta_v if cell.method == "asymre" else None,
        "positive_coefficient": (
            1.0 - cell.delta_v if cell.method == "asymre" else 1.0
        ),
        "negative_repulsion_coefficient": (
            None
            if is_topr
            else 1.0 + cell.delta_v
            if cell.method == "asymre"
            else cell.alpha
        ),
        "distance_control": cell.method not in {
            "asymre",
            "joint_fitted_reference_topr",
        },
        "behavior_relative_ratio_control": is_topr,
        "objective_formula": objective_formula,
        "weight_formula": weight_formula,
        "ratio_coordinate": (
            "full_completion_summed_log_probability" if is_topr else None
        ),
        "task_loss_log_probability": "mean_completion_token_log_probability",
        "reference_policy": (
            {
                "adapter": TOPR_REFERENCE_ADAPTER,
                "training_mode": "joint_branch_balanced_bank_sft",
                "positive_branch_mass": 0.5,
                "negative_branch_mass": 0.5,
                "negative_within_branch": "uniform_over_unique_negatives_per_prompt",
                "update_frequency_per_policy_step": 1,
                "receives_ratio_gradient": False,
            }
            if is_topr
            else None
        ),
        "policy_adapter": TOPR_POLICY_ADAPTER if is_topr else None,
        "initial_ratio_max_abs": initial_ratio_max_abs,
        "initial_ratio_max_abs_tolerance": (
            float(fixed_training["initial_ratio_max_abs_tolerance"])
            if is_topr
            else None
        ),
        "u_definition": (
            None
            if cell.method in {"asymre", "joint_fitted_reference_topr"}
            else "current_sequence_surprisal/2"
        ),
        "unique_negative_denominator": True,
        "weight_sum_normalization": False,
        "extreme_selection_used": False,
        "hidden_scale_used": False,
        "seed": seed,
        "seed_offset": cell.seed_offset,
        "steps_requested": steps,
        "terminal_step": terminal_step,
        "last_finite_step": last_finite_step,
        "best_pass8_step": best_step,
        "best_pass8_value": best_pass8,
        "stop_reason": stop_reason,
        "numerical_failure": numerical_failure,
        "terminal_checkpoint_kind": terminal_kind,
        "checkpoint_policy": checkpoint_policy,
        "checkpoints": checkpoint_records,
        "run_identity": identity,
        "bank_audit": {
            "rows": len(train_rows),
            "raw_bank_count_min": min(raw_counts),
            "raw_bank_count_max": max(raw_counts),
            "unique_negative_count_min": min(unique_counts),
            "unique_negative_count_max": max(unique_counts),
            "rows_with_duplicates_removed": sum(
                raw > unique
                for raw, unique in zip(raw_counts, unique_counts, strict=True)
            ),
        },
        "metric_bests": metric_bests,
        "terminal_metrics": dict(metric_rows[-1]),
        "diagnostic_files": {
            "training": str(training_diagnostics_path),
            "validation": str(validation_diagnostics_path),
        },
        "test_data_used": False,
        "reporting_separation": {
            "task_performance": "validation trajectories and metric-specific best values",
            "support_or_structure_boundary": (
                "valid_rate diagnostic only; no formal boundary threshold registered"
            ),
            "nan_inf_numerical_failure": numerical_failure,
        },
        "fixed_horizon_is_convergence": False,
        "method_ranking_claim_allowed": False,
    }
    atomic_json(output_dir / "manifest.json", manifest)
    atomic_json(summary_path, manifest)
    return manifest
