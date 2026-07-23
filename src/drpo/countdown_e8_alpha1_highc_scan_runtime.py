#!/usr/bin/env python3
"""Runtime adapter for the E8 paper-aligned taper-family scans."""
from __future__ import annotations

import argparse
import copy
import csv
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from drpo import countdown_e8_alpha1_highc_scan_common as highc


EVALUATION_SEMANTICS: dict[str, Any] = {
    "evaluation_split_file": "val.jsonl",
    "evaluation_split_role": "structurally_disjoint_held_out_evaluation",
    "evaluation_enters_training_loss": False,
    "training_structure_overlap": "none",
    "training_problem_key_overlap": "none",
    "paper_facing_checkpoint_policy": "late_window_and_terminal",
    "paper_facing_summary": ["late_window_pass_at_8", "terminal_pass_at_8"],
    "best_checkpoint_role": "supplementary_only",
    "separate_test_jsonl_used": False,
    "separate_test_jsonl_required_for_existing_curve_validity": False,
}


def _grid_config_from_argv(argv: list[str]) -> str | None:
    for index, token in enumerate(argv):
        if token == "--grid_config" and index + 1 < len(argv):
            return argv[index + 1]
    return None


def _load_dpo_profile_adapter() -> Any:
    path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "run_countdown_e8_oracle_offline_v2_alpha1_highc_scan_auto.py"
    )
    spec = importlib.util.spec_from_file_location("_e8_highc_launcher_profile", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load DPO profile adapter: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_grid_config = _grid_config_from_argv(sys.argv[1:])
_dpo_profile = _load_dpo_profile_adapter()
_dpo_profile._install_canonical_dpo_profile(_grid_config)
CANONICAL_DPO_EXPERIMENT_ID = _dpo_profile.CANONICAL_DPO_EXPERIMENT_ID
if _grid_config is None:
    highc.activate()
else:
    highc.activate_for_grid_config(_grid_config)

from drpo import countdown_e8_alpha1_c_scan_runtime as _base_runtime  # noqa: E402
from drpo import countdown_e8_alpha1_c_scan_trainer as _trainer  # noqa: E402


_ORIGINAL_TRAIN_CELL = _base_runtime.train_cell
_ORIGINAL_PLAN = _base_runtime.plan
_ORIGINAL_AGGREGATE = _base_runtime._aggregate
_ORIGINAL_RUN = _base_runtime.run


def _semantic_payload(config: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = dict(EVALUATION_SEMANTICS)
    if config is not None:
        evaluation = config.get("evaluation", {})
        payload.update(
            {
                "evaluation_split_file": str(
                    evaluation.get("split_file", payload["evaluation_split_file"])
                ),
                "evaluation_split_role": str(
                    evaluation.get("split_role", payload["evaluation_split_role"])
                ),
                "evaluation_enters_training_loss": bool(
                    evaluation.get(
                        "enters_training_loss",
                        payload["evaluation_enters_training_loss"],
                    )
                ),
                "paper_facing_checkpoint_policy": str(
                    evaluation.get(
                        "paper_facing_checkpoint_policy",
                        payload["paper_facing_checkpoint_policy"],
                    )
                ),
                "best_checkpoint_role": str(
                    evaluation.get(
                        "best_checkpoint_role", payload["best_checkpoint_role"]
                    )
                ),
            }
        )
    return payload


def _augment_json(path: Path, payload: dict[str, Any]) -> None:
    if not path.is_file():
        return
    current = json.loads(path.read_text(encoding="utf-8"))
    current.update(payload)
    highc.atomic_json(path, current)


def _prompt_balanced_mean(
    values: torch.Tensor,
    row_index: torch.Tensor,
    unique_counts: torch.Tensor,
) -> torch.Tensor:
    ones = torch.ones_like(values)
    return highc.mean_unique_negative_term(values, ones, row_index, unique_counts)


def _quantile(values: torch.Tensor, q: float) -> float:
    return float(torch.quantile(values.detach().float().cpu(), q).item())


def _canonical_dpo_train_cell(
    *,
    cell: Any,
    model_path: Path,
    bank: Path,
    val: Path,
    base_config_path: Path,
    grid_config_path: Path,
    output_dir: Path,
    repo: Path,
    smoke: bool = False,
) -> dict[str, Any]:
    if cell.method != "canonical_dpo":
        raise ValueError(f"Canonical DPO trainer received non-DPO cell: {cell}")
    base_config = highc.load_yaml(base_config_path)
    grid_config = highc.load_yaml(grid_config_path)
    highc.validate_grid_config(grid_config)
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
    dpo_beta = float(cell.c)
    _trainer.arena.seed_all(seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    identity = highc._identity(
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
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
        if json.dumps(existing.get("run_identity", {}), sort_keys=True) == json.dumps(
            identity, sort_keys=True
        ):
            return existing
        raise RuntimeError(f"Stale result identity at {summary_path}; use a new work_dir")

    tokenizer = _trainer.arena.load_tokenizer(str(model_path))
    train_rows = _trainer.arena.read_jsonl(bank)
    val_rows = _trainer.arena.read_jsonl(val)
    unique_counts_list = [
        len(highc.unique_negative_expressions(row)) for row in train_rows
    ]
    raw_counts = [len(row.get("negative_bank", [])) for row in train_rows]
    if min(unique_counts_list) < 1:
        raise RuntimeError("At least one training row has no unique negative")
    dataset = highc.ContinuousUniqueBankDataset(
        train_rows, tokenizer, int(model_cfg["max_length"])
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=int(train_cfg["micro_batch"]),
        shuffle=True,
        generator=generator,
        collate_fn=highc.make_continuous_unique_bank_collator(tokenizer.pad_token_id),
        num_workers=int(train_cfg["num_workers"]),
    )
    iterator = iter(loader)
    model = _trainer.arena.load_model(
        str(model_path),
        adapter_path=None,
        trainable_adapter=True,
        load_in_4bit=bool(model_cfg.get("load_in_4bit", False)),
        dtype=str(model_cfg.get("dtype", "auto")),
        gradient_checkpointing=True,
        parameterization="lora",
    )
    if not hasattr(model, "add_adapter") or not hasattr(model, "set_adapter"):
        raise RuntimeError("Canonical DPO requires PEFT multi-adapter support")
    policy_adapter = str(grid_config["model"]["policy_adapter"])
    reference_adapter = str(grid_config["model"]["reference_adapter"])
    if policy_adapter not in model.peft_config:
        raise RuntimeError("Loaded LoRA model has no policy adapter")
    if reference_adapter in model.peft_config:
        raise RuntimeError("Reference adapter exists before DPO initialization")
    model.add_adapter(
        reference_adapter,
        copy.deepcopy(model.peft_config[policy_adapter]),
    )
    _trainer._copy_adapter_parameters(model, policy_adapter, reference_adapter)
    policy_parameters = _trainer._adapter_parameters(model, policy_adapter)
    reference_parameters = _trainer._adapter_parameters(model, reference_adapter)

    def activate_reference() -> None:
        model.set_adapter(reference_adapter)
        for parameter in reference_parameters:
            parameter.requires_grad_(False)

    def activate_policy() -> None:
        model.set_adapter(policy_adapter)
        for parameter in policy_parameters:
            parameter.requires_grad_(True)
        for parameter in reference_parameters:
            parameter.requires_grad_(False)

    activate_policy()
    if any(parameter.requires_grad for parameter in reference_parameters):
        raise RuntimeError("Canonical DPO reference adapter is unexpectedly trainable")
    optimizer = torch.optim.AdamW(
        policy_parameters,
        lr=float(train_cfg["learning_rate"]),
        weight_decay=0.01,
    )
    scheduler = _trainer.arena.get_cosine_schedule_with_warmup(
        optimizer,
        max(1, int(steps * float(train_cfg["warmup_ratio"]))),
        steps,
    )
    device = next(model.parameters()).device
    model.eval()

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
    initial_pair_margin_max_abs: float | None = None
    known_structures = {
        row.get("oracle_structure") or _trainer.arena.expression_structure(row["oracle"])
        for row in train_rows
    }

    def evaluate(step: int) -> dict[str, Any]:
        activate_policy()
        model.eval()
        row = _trainer._evaluate_validation(
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
                "method": "canonical_dpo",
                "family": "canonical_dpo",
                "dpo_beta": dpo_beta,
                "alpha": 1.0,
                "c": dpo_beta,
                "distance_control": False,
                "behavior_relative_ratio_control": False,
                "reference_role": "exact_frozen_initial_policy",
                "evaluated_adapter": policy_adapter,
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
        _trainer.arena.save_local_model_checkpoint(
            model, tokenizer, best_dir, "best_pass8", 0
        )
    )

    diagnostic_keys = (
        "policy_chosen_sum_lp",
        "policy_rejected_sum_lp",
        "reference_chosen_sum_lp",
        "reference_rejected_sum_lp",
        "pair_margin_mean",
        "pair_margin_p10",
        "pair_margin_p50",
        "pair_margin_p90",
        "preference_accuracy",
        "logit_saturation_fraction",
        "unique_negative_count_mean",
        "raw_bank_count_mean",
        "duplicates_removed_mean",
    )

    for update_step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        accum: dict[str, float | None] = {
            "loss": 0.0,
            **{key: 0.0 for key in diagnostic_keys},
        }
        abort_update = False
        for accumulation_index in range(int(train_cfg["gradient_accumulation"])):
            try:
                packed = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                packed = next(iterator)
            positive_batch = _trainer.arena.move_to_device(packed["positive"], device)
            bank_batch = _trainer.arena.move_to_device(packed["bank"], device)
            row_index = packed["bank_row_index"].to(device)
            unique_counts = packed["unique_counts"].to(device)

            activate_reference()
            model.eval()
            with torch.no_grad():
                reference_positive_stats = _trainer.arena.completion_stats(
                    model, positive_batch
                )
                reference_bank_stats = _trainer.arena.completion_stats(model, bank_batch)
                reference_chosen = highc.full_sequence_log_probability(
                    reference_positive_stats
                ).detach()
                reference_rejected = highc.full_sequence_log_probability(
                    reference_bank_stats
                ).detach()

            activate_policy()
            model.eval()
            policy_positive_stats = _trainer.arena.completion_stats(model, positive_batch)
            policy_bank_stats = _trainer.arena.completion_stats(model, bank_batch)
            policy_chosen = highc.full_sequence_log_probability(policy_positive_stats)
            policy_rejected = highc.full_sequence_log_probability(policy_bank_stats)

            pair_margin = (
                policy_chosen[row_index]
                - policy_rejected
                - reference_chosen[row_index]
                + reference_rejected
            )
            dpo_logits = dpo_beta * pair_margin
            pair_losses = F.softplus(-dpo_logits)
            raw_loss = _prompt_balanced_mean(pair_losses, row_index, unique_counts)

            if update_step == 1 and accumulation_index == 0:
                initial_pair_margin_max_abs = float(pair_margin.detach().abs().max())
                tolerance = float(
                    fixed_training["initial_pair_margin_max_abs_tolerance"]
                )
                if initial_pair_margin_max_abs > tolerance:
                    numerical_failure = "initial_policy_reference_pair_margin_mismatch"
                    stop_reason = numerical_failure
                    abort_update = True
                    break
            if not bool(torch.isfinite(raw_loss)):
                numerical_failure = f"nonfinite_loss_at_step_{update_step}"
                stop_reason = numerical_failure
                abort_update = True
                break

            (raw_loss / int(train_cfg["gradient_accumulation"])).backward()
            divisor = float(train_cfg["gradient_accumulation"])
            policy_rejected_mean = _prompt_balanced_mean(
                policy_rejected, row_index, unique_counts
            )
            reference_rejected_mean = _prompt_balanced_mean(
                reference_rejected, row_index, unique_counts
            )
            diagnostics = {
                "policy_chosen_sum_lp": float(policy_chosen.detach().mean()),
                "policy_rejected_sum_lp": float(policy_rejected_mean.detach()),
                "reference_chosen_sum_lp": float(reference_chosen.mean()),
                "reference_rejected_sum_lp": float(reference_rejected_mean),
                "pair_margin_mean": float(pair_margin.detach().mean()),
                "pair_margin_p10": _quantile(pair_margin, 0.10),
                "pair_margin_p50": _quantile(pair_margin, 0.50),
                "pair_margin_p90": _quantile(pair_margin, 0.90),
                "preference_accuracy": float(
                    (pair_margin.detach() > 0.0).float().mean()
                ),
                "logit_saturation_fraction": float(
                    (dpo_logits.detach().abs() >= 10.0).float().mean()
                ),
                "unique_negative_count_mean": float(
                    packed["unique_counts"].float().mean()
                ),
                "raw_bank_count_mean": float(
                    packed["raw_bank_counts"].float().mean()
                ),
                "duplicates_removed_mean": float(
                    (packed["raw_bank_counts"] - packed["unique_counts"])
                    .float()
                    .mean()
                ),
            }
            accum["loss"] = float(accum["loss"] or 0.0) + float(
                raw_loss.detach()
            ) / divisor
            for key, value in diagnostics.items():
                accum[key] = float(accum[key] or 0.0) + value / divisor
        if abort_update:
            break

        raw_grad_norm = torch.nn.utils.clip_grad_norm_(
            policy_parameters, float(train_cfg["maximum_gradient_norm"])
        )
        if not bool(torch.isfinite(raw_grad_norm)):
            numerical_failure = f"nonfinite_gradient_at_step_{update_step}"
            stop_reason = numerical_failure
            break
        sample_update_norm = (
            update_step % int(train_cfg["log_every"]) == 0 or update_step == steps
        )
        policy_before = (
            [
                parameter.detach().float().cpu().clone()
                for parameter in policy_parameters
            ]
            if sample_update_norm
            else []
        )
        if not _trainer.arena.optimizer_step_with_last_finite_guard(
            optimizer, policy_parameters
        ):
            numerical_failure = f"nonfinite_parameters_at_step_{update_step}"
            stop_reason = numerical_failure
            break
        scheduler.step()
        policy_update_norm = (
            _trainer._parameter_update_norm(policy_before, policy_parameters)
            if sample_update_norm
            else None
        )
        terminal_step = update_step
        last_finite_step = update_step
        accum["raw_gradient_norm"] = float(raw_grad_norm)
        accum["optimizer_update_norm"] = policy_update_norm
        accum["initial_pair_margin_max_abs"] = initial_pair_margin_max_abs

        if update_step % int(train_cfg["log_every"]) == 0 or update_step == steps:
            training_record = {
                "cell": cell.name,
                "step": update_step,
                "method": "canonical_dpo",
                "family": "canonical_dpo",
                "dpo_beta": dpo_beta,
                "distance_control": False,
                "behavior_relative_ratio_control": False,
                "reference_role": "exact_frozen_initial_policy",
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
                    _trainer.arena.save_local_model_checkpoint(
                        model, tokenizer, best_dir, "best_pass8", update_step
                    )
                )
            model.eval()
            activate_policy()

    activate_policy()
    model.eval()
    if numerical_failure:
        checkpoint_records.append(
            _trainer.arena.save_local_model_checkpoint(
                model, tokenizer, last_finite_dir, "last_finite", last_finite_step
            )
        )
        terminal_kind = "last_finite"
    else:
        checkpoint_records.append(
            _trainer.arena.save_local_model_checkpoint(
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
        writer.writerows(_trainer.arena.csv_safe_row(row) for row in metric_rows)

    metric_bests = {
        key: _trainer._best_row(metric_rows, key)
        for key in ("val_greedy", "val_pass_at_8", "val_pass_at_64", "val_valid_rate")
    }
    manifest = {
        "schema_version": 1,
        "experiment_id": CANONICAL_DPO_EXPERIMENT_ID,
        "version": highc.VERSION,
        "result_status": "smoke_only" if smoke else "pilot",
        "registration_state": str(grid_config["registration_state"]),
        "cell": cell.name,
        "method": "canonical_dpo",
        "method_identity": "canonical_sigmoid_dpo_frozen_initial_reference",
        "family": "canonical_dpo",
        "alpha": 1.0,
        "c": dpo_beta,
        "dpo_beta": dpo_beta,
        "objective_formula": (
            "mean_prompt(mean_unique_negative(-logsigmoid(beta*((logpi_chosen-"
            "logpi_rejected)-(logref_chosen-logref_rejected)))))"
        ),
        "sequence_log_probability": "full_completion_summed_log_probability",
        "chosen_completion": "oracle_completion",
        "rejected_completions": "all_unique_verifier_wrong_completions",
        "pair_aggregation": "mean_within_prompt_then_mean_prompts",
        "label_smoothing": 0.0,
        "distance_control": False,
        "behavior_relative_ratio_control": False,
        "reference_policy": {
            "adapter": reference_adapter,
            "role": "exact_frozen_initial_policy",
            "copied_from_policy_before_update_1": True,
            "trainable": False,
            "update_frequency_per_policy_step": 0,
            "receives_gradient": False,
            "dropout_disabled": True,
        },
        "policy_adapter": policy_adapter,
        "initial_pair_margin_max_abs": initial_pair_margin_max_abs,
        "initial_pair_margin_max_abs_tolerance": float(
            fixed_training["initial_pair_margin_max_abs_tolerance"]
        ),
        "unique_negative_denominator": True,
        "weight_sum_normalization": False,
        "extreme_selection_used": False,
        "hard_negative_mining_used": False,
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
        "checkpoint_policy": (
            "server-local dual LoRA; default=trained policy, reference=frozen exact initialization"
        ),
        "checkpoints": checkpoint_records,
        "run_identity": identity,
        "bank_audit": {
            "rows": len(train_rows),
            "raw_bank_count_min": min(raw_counts),
            "raw_bank_count_max": max(raw_counts),
            "unique_negative_count_min": min(unique_counts_list),
            "unique_negative_count_max": max(unique_counts_list),
            "rows_with_duplicates_removed": sum(
                raw > unique
                for raw, unique in zip(raw_counts, unique_counts_list, strict=True)
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
            "task_performance": "validation trajectories, late-window, and terminal metrics",
            "support_or_structure_boundary": (
                "valid-rate diagnostic only; no formal boundary threshold registered"
            ),
            "nan_inf_numerical_failure": numerical_failure,
        },
        "fixed_horizon_is_convergence": False,
        "method_ranking_claim_allowed": False,
    }
    highc.atomic_json(output_dir / "manifest.json", manifest)
    highc.atomic_json(summary_path, manifest)
    return manifest


def _train_cell_with_evaluation_semantics(*args: Any, **kwargs: Any) -> dict[str, Any]:
    cell = kwargs.get("cell")
    if cell is not None and cell.method == "canonical_dpo":
        summary = _canonical_dpo_train_cell(*args, **kwargs)
    else:
        summary = _ORIGINAL_TRAIN_CELL(*args, **kwargs)
    output_dir = Path(kwargs["output_dir"]).resolve()
    config = highc.load_yaml(kwargs["grid_config_path"])
    summary.update(_semantic_payload(config))
    summary["best_checkpoint_saved"] = True
    reporting = dict(summary.get("reporting_separation", {}))
    reporting["task_performance"] = (
        "predeclared late-window and terminal held-out metrics; "
        "metric-specific best values are supplementary only"
    )
    summary["reporting_separation"] = reporting
    highc.atomic_json(output_dir / "manifest.json", summary)
    highc.atomic_json(output_dir / "summary.json", summary)
    return summary


_base_runtime.train_cell = _train_cell_with_evaluation_semantics


def _worker_command(args, cell, output_dir: Path) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "worker",
        "--model_path",
        args.model_path,
        "--bank",
        args.bank,
        "--val",
        args.val,
        "--base_config",
        args.base_config,
        "--grid_config",
        args.grid_config,
        "--output_dir",
        str(output_dir),
        "--family",
        str(cell.family),
        "--alpha",
        str(cell.alpha),
        "--c",
        str(float(cell.c)),
        "--seed_offset",
        str(cell.seed_offset),
    ]


_base_runtime._worker_command = _worker_command


def plan(args: argparse.Namespace) -> int:
    result = _ORIGINAL_PLAN(args)
    config = highc.load_yaml(args.grid_config)
    path = Path(args.work_dir).resolve() / "SWEEP_PLAN.json"
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload.update(_semantic_payload(config))
        for row in payload.get("cells", []):
            if row.get("method") == "asymre":
                row["delta_v"] = round(float(row["alpha"]) - 1.0, 12)
            if row.get("method") == "canonical_dpo":
                row["dpo_beta"] = float(row["c"])
                row["reference_role"] = "exact_frozen_initial_policy"
        highc.atomic_json(path, payload)
    return result


_base_runtime.plan = plan


def _augment_aggregate_csv(path: Path) -> None:
    if not path.is_file():
        return
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return
    for row in rows:
        row.update(
            {
                "evaluation_split_role": str(
                    EVALUATION_SEMANTICS["evaluation_split_role"]
                ),
                "evaluation_enters_training_loss": "false",
                "paper_facing_checkpoint_policy": str(
                    EVALUATION_SEMANTICS["paper_facing_checkpoint_policy"]
                ),
                "best_checkpoint_role": str(
                    EVALUATION_SEMANTICS["best_checkpoint_role"]
                ),
                "separate_test_jsonl_used": "false",
            }
        )
        if row.get("method") == "asymre":
            row["delta_v"] = str(round(float(row["alpha"]) - 1.0, 12))
        if row.get("method") == "canonical_dpo":
            row["dpo_beta"] = row.get("c", "")
            row["reference_role"] = "exact_frozen_initial_policy"
    fieldnames = list(rows[0])
    for row in rows[1:]:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _aggregate(
    work_dir: Path,
    cells: Sequence[Any],
    *,
    registration_state: str,
) -> dict[str, Any]:
    audit = _ORIGINAL_AGGREGATE(
        work_dir, cells, registration_state=registration_state
    )
    audit.update(_semantic_payload())
    audit["task_performance_status"] = (
        "late_window_and_terminal_reported_not_adjudicated"
    )
    aggregate_dir = work_dir / "aggregate"
    _augment_aggregate_csv(aggregate_dir / "per_cell_summary.csv")
    highc.atomic_json(aggregate_dir / "terminal_audit.json", audit)
    return audit


_base_runtime._aggregate = _aggregate


def run(args: argparse.Namespace) -> int:
    result = _ORIGINAL_RUN(args)
    config = highc.load_yaml(args.grid_config)
    _augment_json(
        Path(args.work_dir).resolve() / "SWEEP_COMPLETE.json",
        _semantic_payload(config),
    )
    return result


_base_runtime.run = run


def smoke(args: argparse.Namespace) -> int:
    repo = Path(__file__).resolve().parents[2]
    model, bank, val, base_config, grid_config = _base_runtime._required_inputs(args)
    config = highc.load_yaml(grid_config)
    highc.validate_grid_config(config)
    liveness = config["execution"]["liveness"]
    family = str(liveness.get("representative_family", "exponential"))
    cell = highc.Cell(
        alpha=float(liveness["representative_alpha"]),
        coefficient=float(liveness["representative_c"]),
        seed_offset=int(highc.SEED_OFFSETS[0]),
        family=family,
    )
    output_dir = Path(args.work_dir).resolve() / "_liveness" / cell.name
    if output_dir.exists() and not (output_dir / "summary.json").exists():
        shutil.rmtree(output_dir)
    try:
        summary = _base_runtime.train_cell(
            cell=cell,
            model_path=model,
            bank=bank,
            val=val,
            base_config_path=base_config,
            grid_config_path=grid_config,
            output_dir=output_dir,
            repo=repo,
            smoke=True,
        )
        passed = summary.get("numerical_failure") is None and int(
            summary.get("terminal_step", -1)
        ) == int(liveness["steps"])
        gate = {
            "schema_version": 1,
            "experiment_id": highc.EXPERIMENT_ID,
            "registration_state": str(config["registration_state"]),
            "status": "PASS" if passed else "FAIL",
            "scientific_evidence": False,
            "cell": cell.name,
            "summary": str(output_dir / "summary.json"),
            "run_identity": summary.get("run_identity"),
            "test_data_used": False,
            **_semantic_payload(config),
        }
    except BaseException as error:
        gate = {
            "schema_version": 1,
            "experiment_id": highc.EXPERIMENT_ID,
            "registration_state": str(config["registration_state"]),
            "status": "FAIL",
            "scientific_evidence": False,
            "error_type": type(error).__name__,
            "error": str(error),
            "test_data_used": False,
            **_semantic_payload(config),
        }
        highc.atomic_json(Path(args.work_dir).resolve() / "SMOKE_GATE.json", gate)
        raise
    highc.atomic_json(Path(args.work_dir).resolve() / "SMOKE_GATE.json", gate)
    return 0 if gate["status"] == "PASS" else 1


def worker(args: argparse.Namespace) -> int:
    cell = highc.Cell(
        alpha=float(args.alpha),
        coefficient=float(args.c),
        seed_offset=int(args.seed_offset),
        family=str(args.family),
    )
    config = highc.load_yaml(args.grid_config)
    expected = highc.build_cells(config)
    if cell not in expected:
        raise ValueError(f"Worker cell is outside the frozen grid: {cell}")
    summary = _base_runtime.train_cell(
        cell=cell,
        model_path=Path(args.model_path).resolve(),
        bank=Path(args.bank).resolve(),
        val=Path(args.val).resolve(),
        base_config_path=Path(args.base_config).resolve(),
        grid_config_path=Path(args.grid_config).resolve(),
        output_dir=Path(args.output_dir).resolve(),
        repo=Path(__file__).resolve().parents[2],
        smoke=False,
    )
    return 0 if summary.get("numerical_failure") is None else 1


def parser() -> argparse.ArgumentParser:
    command_parser = argparse.ArgumentParser(description=__doc__)
    command_parser.add_argument("--version", action="version", version=highc.VERSION)
    subparsers = command_parser.add_subparsers(dest="command", required=True)

    def common(subparser: argparse.ArgumentParser, *, include_work_dir: bool = True) -> None:
        subparser.add_argument("--model_path", required=True)
        subparser.add_argument("--bank", required=True)
        subparser.add_argument("--val", required=True)
        subparser.add_argument("--base_config", required=True)
        subparser.add_argument("--grid_config", required=True)
        if include_work_dir:
            subparser.add_argument("--work_dir", required=True)

    common(subparsers.add_parser("plan"))
    common(subparsers.add_parser("smoke"))
    run_parser = subparsers.add_parser("run")
    common(run_parser)
    run_parser.add_argument("--gpus", required=True)
    run_parser.add_argument("--runtime-slots-per-gpu", type=int, default=2)
    worker_parser = subparsers.add_parser("worker")
    common(worker_parser, include_work_dir=False)
    worker_parser.add_argument("--output_dir", required=True)
    worker_parser.add_argument("--family", required=True)
    worker_parser.add_argument("--alpha", type=float, required=True)
    worker_parser.add_argument("--c", type=float, required=True)
    worker_parser.add_argument("--seed_offset", type=int, required=True)
    return command_parser


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "plan":
        return plan(args)
    if args.command == "smoke":
        return smoke(args)
    if args.command == "run":
        return run(args)
    if args.command == "worker":
        return worker(args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
