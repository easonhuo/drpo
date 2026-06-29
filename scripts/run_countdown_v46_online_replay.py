#!/usr/bin/env python3
"""Run EXT-C-E8-V4.6 online off-policy replay pilot.

The registered 2x2 design separates data refresh from negative-gradient use:

* frozen_positive_only: frozen V4.4 bank, positive-only updates;
* frozen_dynamic: frozen V4.4 bank, V4.5-selected dynamic control;
* online_positive_only: policy-refreshed replay, positive-only updates;
* online_dynamic: policy-refreshed replay, V4.5-selected dynamic control.

Online workers keep one learner/optimizer alive across four collection phases.
After the first phase, each optimizer update uses exactly half fresh microbatches
and half stale microbatches from prior collector versions, so the learner is
online in data acquisition and off-policy in replay use. The test split is not
used until all training finishes. This remains a 0.5B external-validity pilot.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import queue
import random
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from threading import Thread
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

EXPERIMENT_ID = "EXT-C-E8-V4.6-ONLINE-OFFPOLICY-REPLAY"
PREDECESSOR_ID = "EXT-C-E8-V4.5-OFFLINE-BANK-TUNING"
FROZEN_SOURCE_ID = "EXT-C-E8-V4.4-OFFLINE-BANK"
METHODS = (
    "frozen_positive_only",
    "frozen_dynamic",
    "online_positive_only",
    "online_dynamic",
)
CONFIRM_SEEDS = (6234, 7234, 8234)
COLLECTION_PHASES = 4
REFRESH_ROWS = 1000
ROLLOUTS_PER_PROMPT = 12
RESAMPLE_ROUNDS = 4
ONLINE_BANK_SIZE = 16
REPLAY_WINDOW = 3
FRESH_MICROBATCHES = 4
STALE_MICROBATCHES = 4
GRAD_ACCUM = FRESH_MICROBATCHES + STALE_MICROBATCHES
SURPRISAL_THRESHOLD = 2.0


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True, stderr=subprocess.STDOUT
    ).strip()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    tmp.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_tree(root: Path) -> dict[str, str]:
    if not root.is_dir():
        raise RuntimeError(f"Frozen adapter directory does not exist: {root}")
    members = sorted(root.rglob("*"))
    symlinks = [path for path in members if path.is_symlink()]
    if symlinks:
        raise RuntimeError(f"Frozen adapter directory contains symlinks: {symlinks[0]}")
    files = [path for path in members if path.is_file()]
    if not files:
        raise RuntimeError(f"Frozen adapter directory is empty: {root}")
    return {str(path.relative_to(root)): _sha256(path) for path in files}


def split_update_budget(total_updates: int, phases: int) -> tuple[int, ...]:
    if total_updates < phases or phases < 1:
        raise ValueError("total_updates must be at least the positive phase count")
    base, extra = divmod(total_updates, phases)
    return tuple(base + int(index < extra) for index in range(phases))


def replay_age_plan(phase: int, replay_window: int = REPLAY_WINDOW) -> dict[str, Any]:
    if phase < 0:
        raise ValueError("phase must be non-negative")
    if replay_window < 2:
        raise ValueError("replay_window must be at least two")
    if phase == 0:
        return {
            "fresh_microbatches": GRAD_ACCUM,
            "stale_microbatches": 0,
            "eligible_stale_rounds": [],
            "off_policy": False,
        }
    oldest = max(0, phase - replay_window + 1)
    return {
        "fresh_microbatches": FRESH_MICROBATCHES,
        "stale_microbatches": STALE_MICROBATCHES,
        "eligible_stale_rounds": list(range(oldest, phase)),
        "off_policy": True,
    }


def deterministic_stale_rows(
    round_rows: Sequence[Sequence[dict[str, Any]]], phase: int, target: int, seed: int
) -> list[dict[str, Any]]:
    plan = replay_age_plan(phase)
    pool = [
        row
        for round_index in plan["eligible_stale_rounds"]
        for row in round_rows[round_index]
    ]
    if not pool:
        raise RuntimeError("Off-policy phase requires non-empty stale replay")
    rng = random.Random(seed + phase * 100003)
    order = list(range(len(pool)))
    rng.shuffle(order)
    return [dict(pool[order[index % len(order)]]) for index in range(target)]


def _trainable_digest(parameters: Sequence[torch.nn.Parameter]) -> str:
    digest = hashlib.sha256()
    for parameter in parameters:
        value = parameter.detach().cpu().contiguous()
        digest.update(str(tuple(value.shape)).encode())
        digest.update(str(value.dtype).encode())
        digest.update(value.float().numpy().tobytes())
    return digest.hexdigest()


def _discover_reference(v44: Path) -> Path:
    for candidate in (
        v44 / "sft_adapter" / "best_adapter",
        v44 / "reference_adapter",
    ):
        if (candidate / "adapter_config.json").is_file():
            return candidate
    raise RuntimeError("Could not find the frozen V4.4 reference adapter")


def _read_metrics_at_step(path: Path, step: int) -> dict[str, float]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    exact = [row for row in rows if int(float(row["step"])) == int(step)]
    if not exact:
        raise RuntimeError(f"No metrics row for step={step}: {path}")
    row = exact[-1]
    return {
        key: float(row[key])
        for key in (
            "greedy_success",
            "pass_at_k",
            "valid_rate",
            "heldout_pattern_family_coverage",
            "heldout_pattern_family_precision_micro",
        )
        if row.get(key) not in (None, "", "None")
    }


def _load_training_record(output_dir: Path, seed: int) -> dict[str, Any]:
    manifest = json.loads((output_dir / "manifest.json").read_text())
    best_step = int(manifest["best_step"])
    terminal_step = manifest.get("terminal_step")
    if terminal_step is None:
        terminal_step = manifest.get("last_finite_step", 0)
    return {
        "seed": seed,
        "best_step": best_step,
        "terminal_step": int(terminal_step),
        "numerical_failure": manifest.get("numerical_failure"),
        "stop_reason": manifest.get("stop_reason"),
        "best_metrics": _read_metrics_at_step(output_dir / "metrics.csv", best_step),
        "terminal_metrics": _read_metrics_at_step(
            output_dir / "metrics.csv", int(terminal_step)
        ),
    }


@dataclass(frozen=True)
class Task:
    name: str
    command: list[str]
    log_path: Path


def _run_task(task: Task, gpu_id: str, repo: Path) -> None:
    task.log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    with task.log_path.open("w") as log:
        log.write("COMMAND: " + " ".join(task.command) + "\n")
        log.write(f"CUDA_VISIBLE_DEVICES={gpu_id}\n")
        log.flush()
        result = subprocess.run(
            task.command,
            cwd=repo,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if result.returncode:
        raise RuntimeError(
            f"Task {task.name} failed with exit {result.returncode}; see {task.log_path}"
        )


def _run_group(tasks: list[Task], gpu_ids: list[str], repo: Path) -> None:
    pending: queue.Queue[Task] = queue.Queue()
    for task in tasks:
        pending.put(task)
    errors: list[BaseException] = []

    def worker(gpu_id: str) -> None:
        while not errors:
            try:
                task = pending.get_nowait()
            except queue.Empty:
                return
            try:
                _run_task(task, gpu_id, repo)
            except BaseException as exc:
                errors.append(exc)
            finally:
                pending.task_done()

    workers = [Thread(target=worker, args=(gpu,), daemon=False) for gpu in gpu_ids]
    for thread in workers:
        thread.start()
    for thread in workers:
        thread.join()
    if errors:
        raise errors[0]


def _next_batch(iterator: Any, loader: DataLoader) -> tuple[Any, Any]:
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def collect_online_replay_rows(
    arena: Any,
    model: Any,
    tokenizer: Any,
    source_rows: Sequence[dict[str, Any]],
    allowed_patterns: set[str],
    *,
    target_rows: int,
    collector_round: int,
    collector_step: int,
    collector_seed: int,
    collector_method: str,
    bank_size: int,
    rollouts: int,
    resample_rounds: int,
    batch_size: int,
    max_length: int = 256,
    max_new_tokens: int = 80,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Collect verifier-labelled replay from the current learner only.

    No synthetic negative is admitted. Correct generated expressions are used
    when available; otherwise the frozen oracle supplies the positive branch.
    """
    if target_rows < 1 or bank_size < 2:
        raise ValueError("target_rows and bank_size must be positive")
    rng = random.Random(collector_seed)
    ordered = list(source_rows)
    rng.shuffle(ordered)
    diagnostics: Counter[str] = Counter()
    output: list[dict[str, Any]] = []
    model.eval()

    for start in range(0, len(ordered), max(1, batch_size)):
        if len(output) >= target_rows:
            break
        chunk = ordered[start : start + max(1, batch_size)]
        groups = arena.generate_outputs(
            model,
            tokenizer,
            [row["prompt"] for row in chunk],
            max_new_tokens,
            True,
            0.8,
            0.95,
            rollouts,
        )
        for row, initial in zip(chunk, groups):
            if len(output) >= target_rows:
                break
            seen: set[str] = set()
            evaluated: list[dict[str, Any]] = []
            for attempt in range(resample_rounds + 1):
                texts = initial if attempt == 0 else arena.generate_outputs(
                    model,
                    tokenizer,
                    [row["prompt"]],
                    max_new_tokens,
                    True,
                    0.8,
                    0.95,
                    rollouts,
                )[0]
                evaluated.extend(
                    arena._score_new_candidates(
                        model,
                        tokenizer,
                        row,
                        texts,
                        seen,
                        allowed_patterns,
                        max_length,
                        max(1, batch_size * 2),
                        diagnostics,
                    )
                )
                wrong = [item for item in evaluated if not item["correct"]]
                if len(wrong) >= bank_size:
                    break
                diagnostics["resample_rounds_used"] += int(attempt < resample_rounds)

            wrong = [item for item in evaluated if not item["correct"]]
            if len(wrong) < bank_size:
                diagnostics["dropped_insufficient_generated_wrong"] += 1
                continue
            detailed = [arena.candidate_metadata(item, tokenizer) for item in wrong]
            detailed.sort(key=lambda item: (float(item["surprisal"]), item["text"]))
            near_item, far_item = detailed[0], detailed[-1]
            bank = arena.select_fixed_negative_bank(
                detailed, near_item, far_item, bank_size
            )
            oracle_structure = row.get("oracle_structure") or arena.expression_structure(
                row["oracle"]
            )
            correct = [
                item
                for item in evaluated
                if item["correct"] and item.get("structure") == oracle_structure
            ]
            if correct:
                positive_item = min(correct, key=lambda item: float(item["surprisal"]))
                positive = positive_item["expression"]
                positive_surprisal = float(positive_item["surprisal"])
                positive_source = "collector_generated_correct"
                diagnostics["generated_positive"] += 1
            else:
                positive = row["oracle"]
                positive_surprisal = arena.score_completions_batch(
                    model,
                    tokenizer,
                    [(row["prompt"], positive)],
                    max_length,
                    1,
                )[0]
                positive_source = "oracle_fallback"
                diagnostics["oracle_positive"] += 1
            output.append(
                {
                    **row,
                    "positive": positive,
                    "positive_base_surprisal": positive_surprisal,
                    "positive_source": positive_source,
                    "near_negative": near_item["text"],
                    "far_negative": far_item["text"],
                    "near_base_surprisal": float(near_item["surprisal"]),
                    "far_base_surprisal": float(far_item["surprisal"]),
                    "negative_bank_size": bank_size,
                    "negative_bank": arena.serialize_negative_bank(bank),
                    "online_replay": True,
                    "collector_round": collector_round,
                    "collector_step": collector_step,
                    "collector_seed": collector_seed,
                    "collector_method": collector_method,
                    "generated_only_negative_bank": True,
                    "pair_matched": True,
                    "pair_protocol": "online_generated_current_policy_extremes",
                }
            )

    manifest = {
        "collector_round": collector_round,
        "collector_step": collector_step,
        "collector_seed": collector_seed,
        "collector_method": collector_method,
        "requested_rows": target_rows,
        "saved_rows": len(output),
        "bank_size": bank_size,
        "rollouts_per_prompt": rollouts,
        "resample_rounds": resample_rounds,
        "generated_only_negative_bank": True,
        "generated_positive_fraction": diagnostics["generated_positive"]
        / max(len(output), 1),
        **dict(diagnostics),
    }
    if len(output) < target_rows:
        raise RuntimeError(
            f"Online collector produced only {len(output)}/{target_rows} replay rows"
        )
    return output, manifest


def _write_metrics(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True)
                    if isinstance(value, (dict, list, tuple))
                    else value
                    for key, value in row.items()
                }
            )


def _inside_online_worker(args: argparse.Namespace, repo: Path) -> int:
    sys.path.insert(0, str(repo / "src"))
    from drpo import countdown_qwen_arena_onefile as arena

    if args.worker_method not in {"online_positive_only", "online_dynamic"}:
        raise RuntimeError("Invalid online worker method")
    seed = int(args.seed)
    arena.seed_all(seed)
    output = Path(args.output_dir).resolve()
    arena.ensure_checkpoint_output_is_local_or_ignored(output)
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("Online worker output_dir must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    replay_dir = output / "replay"
    replay_dir.mkdir()

    tokenizer = arena.load_tokenizer(args.model_path)
    model = arena.load_model(
        args.model_path,
        args.reference_adapter,
        trainable_adapter=True,
        load_in_4bit=False,
        dtype="bf16",
        gradient_checkpointing=True,
    )
    device = next(model.parameters()).device
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    scheduler = arena.get_cosine_schedule_with_warmup(
        optimizer,
        max(1, int(args.total_steps * 0.03)),
        args.total_steps,
    )
    calibration = json.loads(Path(args.calibration_json).read_text())
    negative_scale = None
    if args.worker_method == "online_dynamic":
        negative_scale = arena.effective_negative_scale(
            float(calibration["bank_negative_scale"]), args.selected_alpha
        )
    source_rows = arena.read_jsonl(args.train_data)
    val_rows = arena.read_jsonl(args.val_data)
    structure_rows = arena.read_jsonl(args.train_data)
    known_structures = {
        row.get("oracle_structure") or arena.expression_structure(row["oracle"])
        for row in structure_rows
    }
    split_manifest = json.loads(Path(args.split_manifest).read_text())
    allowed_patterns = set(split_manifest["negative_training_allowed_patterns"])
    phase_budgets = split_update_budget(args.total_steps, args.collection_phases)

    best_dir = output / "best_adapter"
    terminal_dir = output / "terminal_adapter"
    last_finite_dir = output / "last_finite_adapter"
    checkpoint_records: list[dict[str, Any]] = []
    metrics_rows: list[dict[str, Any]] = []
    diagnostics_path = output / "dynamic_diagnostics.jsonl"
    all_round_rows: list[list[dict[str, Any]]] = []
    collector_manifests: list[dict[str, Any]] = []
    global_step = 0
    numerical_failure: str | None = None
    last_finite_step = 0
    best_step = 0

    initial_eval = arena.evaluate_rows(
        model,
        tokenizer,
        val_rows[:500],
        args.eval_batch,
        80,
        8,
        seed + 6000,
        known_structures,
    )
    best_value = float(initial_eval["greedy_success"])
    checkpoint_records.append(
        arena.save_local_model_checkpoint(model, tokenizer, best_dir, "best", 0)
    )
    metrics_rows.append(
        {
            "step": 0,
            "phase": -1,
            "method": args.worker_method,
            "effective_epoch": 0.0,
            "negative_scale": negative_scale or 0.0,
            **initial_eval,
        }
    )

    def evaluate_checkpoint(
        phase: int,
        fresh_rows: Sequence[dict[str, Any]],
        stale_rows: Sequence[dict[str, Any]],
        plan: dict[str, Any],
    ) -> None:
        nonlocal best_step, best_value
        model.eval()
        eval_metrics = arena.evaluate_rows(
            model,
            tokenizer,
            val_rows[:500],
            args.eval_batch,
            80,
            8,
            seed + 6000,
            known_structures,
        )
        replay_rows_for_diagnostics = list(fresh_rows) + list(stale_rows)
        diagnostics = arena.dynamic_negative_diagnostics(
            model,
            tokenizer,
            replay_rows_for_diagnostics,
            trainable,
            max_examples=32,
            gradient_examples=8,
            batch_size=8,
            max_length=256,
            exp_lambda=args.selected_lambda,
            surprisal_threshold=SURPRISAL_THRESHOLD,
            negative_scale=negative_scale,
        )
        replay_ages = [int(row["replay_age"]) for row in replay_rows_for_diagnostics]
        collector_versions = sorted(
            {int(row["collector_round"]) for row in replay_rows_for_diagnostics}
        )
        diagnostics.update(
            {
                "replay_age_mean": float(np.mean(replay_ages)),
                "replay_age_max": float(max(replay_ages)),
                "collector_versions_present": collector_versions,
            }
        )
        with diagnostics_path.open("a") as handle:
            handle.write(
                json.dumps(
                    {
                        "step": global_step,
                        "phase": phase,
                        "method": args.worker_method,
                        **diagnostics,
                    }
                )
                + "\n"
            )
        row = {
            "step": global_step,
            "phase": phase,
            "method": args.worker_method,
            "effective_epoch": global_step / max(args.eval_every, 1),
            "negative_scale": negative_scale or 0.0,
            "fresh_microbatches": plan["fresh_microbatches"],
            "stale_microbatches": plan["stale_microbatches"],
            **diagnostics,
            **eval_metrics,
        }
        metrics_rows.append(row)
        value = float(eval_metrics["greedy_success"])
        if value > best_value + 0.002:
            best_value = value
            best_step = global_step
            checkpoint_records[:] = [
                record for record in checkpoint_records if record["kind"] != "best"
            ]
            checkpoint_records.append(
                arena.save_local_model_checkpoint(
                    model, tokenizer, best_dir, "best", global_step
                )
            )
        model.train()

    for phase, phase_steps in enumerate(phase_budgets):
        collector_digest = _trainable_digest(trainable)
        rows, collector_manifest = collect_online_replay_rows(
            arena,
            model,
            tokenizer,
            source_rows,
            allowed_patterns,
            target_rows=args.refresh_rows,
            collector_round=phase,
            collector_step=global_step,
            collector_seed=seed + 10000 + phase,
            collector_method=args.worker_method,
            bank_size=args.bank_size,
            rollouts=args.rollouts,
            resample_rounds=args.resample_rounds,
            batch_size=args.rollout_batch,
        )
        collector_manifest["collector_policy_digest"] = collector_digest
        collector_manifests.append(collector_manifest)
        all_round_rows.append(rows)
        arena.write_jsonl(replay_dir / f"round_{phase}.jsonl", rows)
        _atomic_json(replay_dir / f"round_{phase}.manifest.json", collector_manifest)

        fresh_rows = [
            {**row, "replay_phase": phase, "replay_age": 0}
            for row in rows
        ]
        stale_rows: list[dict[str, Any]] = []
        plan = replay_age_plan(phase, args.replay_window)
        if phase > 0:
            stale_rows = [
                {
                    **row,
                    "replay_phase": phase,
                    "replay_age": phase - int(row["collector_round"]),
                }
                for row in deterministic_stale_rows(
                    all_round_rows,
                    phase,
                    args.refresh_rows,
                    seed,
                )
            ]
            arena.write_jsonl(replay_dir / f"phase_{phase}_stale.jsonl", stale_rows)
        arena.write_jsonl(replay_dir / f"phase_{phase}_fresh.jsonl", fresh_rows)
        _atomic_json(
            replay_dir / f"phase_{phase}_mix.json",
            {
                **plan,
                "phase": phase,
                "fresh_rows": len(fresh_rows),
                "stale_rows": len(stale_rows),
                "fresh_fraction_after_warmup": 0.5 if phase > 0 else 1.0,
                "collector_versions_present": sorted(
                    {int(row["collector_round"]) for row in fresh_rows + stale_rows}
                ),
            },
        )

        fresh_dataset = arena.OfflineDataset(fresh_rows, tokenizer, 256)
        fresh_generator = torch.Generator().manual_seed(seed + phase * 101)
        fresh_loader = DataLoader(
            fresh_dataset,
            batch_size=args.micro_batch,
            shuffle=True,
            generator=fresh_generator,
            collate_fn=arena.make_offline_collator(tokenizer.pad_token_id),
            num_workers=0,
        )
        fresh_iterator = iter(fresh_loader)
        stale_loader = None
        stale_iterator = None
        if stale_rows:
            stale_dataset = arena.OfflineDataset(stale_rows, tokenizer, 256)
            stale_generator = torch.Generator().manual_seed(seed + phase * 101 + 1)
            stale_loader = DataLoader(
                stale_dataset,
                batch_size=args.micro_batch,
                shuffle=True,
                generator=stale_generator,
                collate_fn=arena.make_offline_collator(tokenizer.pad_token_id),
                num_workers=0,
            )
            stale_iterator = iter(stale_loader)

        model.train()
        for _ in range(phase_steps):
            optimizer.zero_grad(set_to_none=True)
            micro_sources = (
                ["fresh"] * GRAD_ACCUM
                if phase == 0
                else ["fresh"] * FRESH_MICROBATCHES
                + ["stale"] * STALE_MICROBATCHES
            )
            random.Random(seed + global_step).shuffle(micro_sources)
            step_loss = 0.0
            near_weight = 0.0
            far_weight = 0.0
            for source in micro_sources:
                if source == "fresh":
                    packed, fresh_iterator = _next_batch(fresh_iterator, fresh_loader)
                else:
                    assert stale_loader is not None and stale_iterator is not None
                    packed, stale_iterator = _next_batch(stale_iterator, stale_loader)
                pos = arena.completion_stats(
                    model, arena.move_to_device(packed["positive"], device)
                )
                positive_lp = pos["seq_lp"].mean()
                if args.worker_method == "online_positive_only":
                    raw_loss = -positive_lp
                else:
                    bank_size = int(packed["bank_size"])
                    bank_batch = arena.move_to_device(packed["bank"], device)
                    near_batch, far_batch, _, _ = arena.current_bank_training_batches(
                        model,
                        bank_batch,
                        packed["positive"]["input_ids"].shape[0],
                        bank_size,
                    )
                    near = arena.completion_stats(model, near_batch)
                    far = arena.completion_stats(model, far_batch)
                    near_weights = arena.detached_token_surprisal_taper(
                        near, args.selected_lambda, SURPRISAL_THRESHOLD
                    )
                    far_weights = arena.detached_token_surprisal_taper(
                        far, args.selected_lambda, SURPRISAL_THRESHOLD
                    )
                    negative_lp = 0.5 * arena.weighted_sequence_logprob(
                        near, near_weights
                    ).mean() + 0.5 * arena.weighted_sequence_logprob(
                        far, far_weights
                    ).mean()
                    assert negative_scale is not None
                    raw_loss = -(positive_lp - negative_scale * negative_lp)
                    near_weight += float(
                        (near_weights.detach() * near["token_mask"]).sum()
                        / near["token_mask"].sum().clamp_min(1)
                    ) / GRAD_ACCUM
                    far_weight += float(
                        (far_weights.detach() * far["token_mask"]).sum()
                        / far["token_mask"].sum().clamp_min(1)
                    ) / GRAD_ACCUM
                if not bool(torch.isfinite(raw_loss)):
                    numerical_failure = f"nonfinite_loss_at_step_{global_step + 1}"
                    break
                (raw_loss / GRAD_ACCUM).backward()
                step_loss += float(raw_loss.detach()) / GRAD_ACCUM
            if numerical_failure:
                break
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            if not bool(torch.isfinite(grad_norm)):
                numerical_failure = f"nonfinite_gradient_at_step_{global_step + 1}"
                break
            if not arena.optimizer_step_with_last_finite_guard(
                optimizer, trainable
            ):
                numerical_failure = f"nonfinite_parameters_at_step_{global_step + 1}"
                break
            scheduler.step()
            global_step += 1
            last_finite_step = global_step
            if global_step % 20 == 0:
                print(
                    json.dumps(
                        {
                            "method": args.worker_method,
                            "phase": phase,
                            "step": global_step,
                            "loss": step_loss,
                            "near_weight": near_weight,
                            "far_weight": far_weight,
                            "fresh_microbatches": plan["fresh_microbatches"],
                            "stale_microbatches": plan["stale_microbatches"],
                        }
                    ),
                    flush=True,
                )
            if global_step % args.eval_every == 0 or global_step == args.total_steps:
                evaluate_checkpoint(phase, fresh_rows, stale_rows, plan)
        if numerical_failure:
            break

    if numerical_failure:
        checkpoint_records.append(
            arena.save_local_model_checkpoint(
                model,
                tokenizer,
                last_finite_dir,
                "last_finite",
                last_finite_step,
            )
        )
        terminal_step = None
        stop_reason = numerical_failure
    else:
        checkpoint_records.append(
            arena.save_local_model_checkpoint(
                model, tokenizer, terminal_dir, "terminal", global_step
            )
        )
        terminal_step = global_step
        stop_reason = "fixed_update_budget_complete"

    _write_metrics(output / "metrics.csv", metrics_rows)
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "method": args.worker_method,
        "seed": seed,
        "source_provenance": arena.source_provenance(),
        "best_step": best_step,
        "best_value": best_value,
        "terminal_step": terminal_step,
        "last_finite_step": last_finite_step,
        "numerical_failure": numerical_failure,
        "stop_reason": stop_reason,
        "total_steps": args.total_steps,
        "eval_every": args.eval_every,
        "completed_steps": global_step,
        "phase_budgets": phase_budgets,
        "collection_phases": args.collection_phases,
        "refresh_rows": args.refresh_rows,
        "rollouts": args.rollouts,
        "resample_rounds": args.resample_rounds,
        "bank_size": args.bank_size,
        "replay_window": args.replay_window,
        "fresh_microbatches_after_warmup": FRESH_MICROBATCHES,
        "stale_microbatches_after_warmup": STALE_MICROBATCHES,
        "selected_alpha": args.selected_alpha,
        "selected_lambda": args.selected_lambda,
        "negative_scale": negative_scale,
        "collector_manifests": collector_manifests,
        "checkpoints": checkpoint_records,
        "checkpoint_policy": "server-local adapters only",
        "result_status": "pilot",
    }
    _atomic_json(output / "manifest.json", manifest)
    _atomic_json(
        output / "checkpoint_manifest.json",
        {
            "local_only": True,
            "method": args.worker_method,
            "seed": seed,
            "checkpoints": checkpoint_records,
        },
    )
    return 1 if numerical_failure else 0


def _frozen_train_command(
    *,
    python: str,
    runner: Path,
    model: Path,
    reference: Path,
    offline: Path,
    val: Path,
    train: Path,
    output: Path,
    plan: dict[str, Any],
    calibration: Path,
    seed: int,
    selected_alpha: float,
    selected_lambda: float,
    positive_only: bool,
    total_steps: int,
    eval_every: int,
) -> list[str]:
    command = [
        python,
        str(runner),
        "train_method",
        "--model_path",
        str(model),
        "--dtype",
        "bf16",
        "--reference_adapter",
        str(reference),
        "--offline_data",
        str(offline),
        "--val_data",
        str(val),
        "--structure_reference_data",
        str(train),
        "--output_dir",
        str(output),
        "--method",
        "positive_only" if positive_only else "bank_dynamic_controlled_negative",
        "--steps",
        str(total_steps),
        "--min_steps",
        str(total_steps),
        "--early_stop_patience",
        "2",
        "--early_stop_delta",
        "0.002",
        "--selection_metric",
        "greedy_success",
        "--micro_batch",
        str(plan["micro_batch"]),
        "--grad_accum",
        str(GRAD_ACCUM),
        "--lr",
        "5e-5",
        "--warmup_ratio",
        "0.03",
        "--max_grad_norm",
        "1.0",
        "--eval_examples",
        "500",
        "--eval_batch",
        str(plan["eval_batch"]),
        "--pass_k",
        "8",
        "--eval_every",
        str(eval_every),
        "--eval_seed",
        str(seed + 6000),
        "--seed",
        str(seed),
        "--diagnostic_examples",
        "32",
        "--diagnostic_gradient_examples",
        "8",
        "--diagnostic_batch",
        "8",
        "--num_workers",
        "2",
        "--result_status",
        "pilot",
        "--exp_lambda",
        str(selected_lambda),
        "--surprisal_threshold",
        str(SURPRISAL_THRESHOLD),
    ]
    if not positive_only:
        command.extend(
            [
                "--negative_calibration_json",
                str(calibration),
                "--negative_scale_multiplier",
                str(selected_alpha),
            ]
        )
    return command


def _eval_command(
    *,
    python: str,
    runner: Path,
    model: Path,
    adapter: Path,
    data: Path,
    train: Path,
    output: Path,
    plan: dict[str, Any],
    seed: int,
) -> list[str]:
    return [
        python,
        str(runner),
        "evaluate",
        "--model_path",
        str(model),
        "--dtype",
        "bf16",
        "--adapter",
        str(adapter),
        "--data",
        str(data),
        "--structure_reference_data",
        str(train),
        "--batch_size",
        str(plan["eval_batch"]),
        "--pass_k",
        "8",
        "--seed",
        str(seed),
        "--output_json",
        str(output),
    ]


def _inside_run(args: argparse.Namespace, repo: Path) -> int:
    sys.path.insert(0, str(repo / "src"))
    from drpo import countdown_qwen_arena_onefile as arena

    root = Path(args.work_dir).resolve()
    if root.exists() and any(root.iterdir()):
        raise RuntimeError("V4.6 work_dir must be new or empty")
    root.mkdir(parents=True, exist_ok=True)
    logs = root / "logs"
    logs.mkdir()
    predecessor = Path(args.predecessor_work_dir).resolve()
    complete = json.loads((predecessor / "RUN_COMPLETE.json").read_text())
    audit = json.loads((predecessor / "terminal_audit.json").read_text())
    if complete.get("experiment_id") != PREDECESSOR_ID:
        raise RuntimeError("predecessor_work_dir is not a V4.5 result")
    if audit.get("experiment_id") != PREDECESSOR_ID:
        raise RuntimeError("V4.5 terminal audit has the wrong experiment_id")
    selected_alpha = float(complete["selected_alpha"])
    selected_lambda = float(complete["selected_lambda"])
    v45_config = json.loads((predecessor / "run_config.json").read_text())
    v44 = Path(v45_config["predecessor_work_dir"]).resolve()
    v44_complete = json.loads((v44 / "RUN_COMPLETE.json").read_text())
    if v44_complete.get("experiment_id") != FROZEN_SOURCE_ID:
        raise RuntimeError("V4.5 does not point to the registered V4.4 source")

    data_dir = v44 / "data"
    train = data_dir / "train.jsonl"
    val = data_dir / "val.jsonl"
    test = data_dir / "test.jsonl"
    split_manifest = data_dir / "split_manifest.json"
    offline = data_dir / "offline_6000.jsonl"
    calibration = v44 / "negative_budget_calibration.json"
    reference_validation_path = v44 / "reference_val_metrics.json"
    reference = _discover_reference(v44)
    required = [
        train, val, test, split_manifest, offline, calibration,
        reference_validation_path,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"V4.4 source is missing required files: {missing}")
    input_hashes = {str(path): _sha256(path) for path in required}
    reference_hashes = _hash_tree(reference)
    reference_validation = json.loads(reference_validation_path.read_text())

    model = Path(args.model_path).resolve()
    if not model.is_dir():
        raise RuntimeError(f"Model directory does not exist: {model}")
    gpu_ids = arena.resolve_gpu_ids(args.gpus, None)
    plan = arena.resolve_execution_plan(
        str(model),
        "0.5b",
        "bf16",
        gpu_index=arena._parent_gpu_index(gpu_ids[0]),
        gpu_visible=gpu_ids[0],
    )
    if plan["dtype"] != "bf16" or plan["load_in_4bit"]:
        raise RuntimeError("V4.6 is frozen to BF16 LoRA")
    offline_rows = len(arena.read_jsonl(offline))
    updates_per_epoch = math.ceil(offline_rows / max(plan["micro_batch"] * GRAD_ACCUM, 1))
    total_steps = 6 * updates_per_epoch
    eval_every = updates_per_epoch
    runner = repo / "src" / "drpo" / "countdown_qwen_arena_onefile.py"
    script = Path(__file__).resolve()
    head = _git(repo, "rev-parse", "HEAD")

    _atomic_json(
        root / "run_config.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "base_commit": head,
            "predecessor_experiment_id": PREDECESSOR_ID,
            "predecessor_work_dir": str(predecessor),
            "frozen_source_experiment_id": FROZEN_SOURCE_ID,
            "frozen_source_work_dir": str(v44),
            "model_path": str(model),
            "gpu_ids": gpu_ids,
            "methods": METHODS,
            "confirm_seeds": CONFIRM_SEEDS,
            "selected_alpha": selected_alpha,
            "selected_lambda": selected_lambda,
            "reference_validation": reference_validation,
            "collection_phases": COLLECTION_PHASES,
            "refresh_rows": REFRESH_ROWS,
            "rollouts_per_prompt": ROLLOUTS_PER_PROMPT,
            "resample_rounds": RESAMPLE_ROUNDS,
            "online_bank_size": ONLINE_BANK_SIZE,
            "replay_window": REPLAY_WINDOW,
            "fresh_microbatches_after_warmup": FRESH_MICROBATCHES,
            "stale_microbatches_after_warmup": STALE_MICROBATCHES,
            "total_optimizer_updates_per_cell": total_steps,
            "eval_every_optimizer_updates": eval_every,
            "phase_update_budgets": split_update_budget(total_steps, COLLECTION_PHASES),
            "input_hashes": input_hashes,
            "reference_adapter_hashes": reference_hashes,
            "plan": plan,
        },
    )

    training_tasks: list[Task] = []
    outputs: dict[tuple[str, int], Path] = {}
    for method in METHODS:
        for seed in CONFIRM_SEEDS:
            output = root / "training" / method / f"seed_{seed}"
            outputs[(method, seed)] = output
            if method.startswith("frozen_"):
                command = _frozen_train_command(
                    python=sys.executable,
                    runner=runner,
                    model=model,
                    reference=reference,
                    offline=offline,
                    val=val,
                    train=train,
                    output=output,
                    plan=plan,
                    calibration=calibration,
                    seed=seed,
                    selected_alpha=selected_alpha,
                    selected_lambda=selected_lambda,
                    positive_only=method == "frozen_positive_only",
                    total_steps=total_steps,
                    eval_every=eval_every,
                )
            else:
                command = [
                    sys.executable,
                    str(script),
                    "--online-worker",
                    "--worker-method",
                    method,
                    "--model_path",
                    str(model),
                    "--reference_adapter",
                    str(reference),
                    "--train_data",
                    str(train),
                    "--val_data",
                    str(val),
                    "--split_manifest",
                    str(split_manifest),
                    "--calibration_json",
                    str(calibration),
                    "--output_dir",
                    str(output),
                    "--seed",
                    str(seed),
                    "--micro_batch",
                    str(plan["micro_batch"]),
                    "--eval_batch",
                    str(plan["eval_batch"]),
                    "--rollout_batch",
                    str(plan["rollout_batch"]),
                    "--total_steps",
                    str(total_steps),
                    "--eval_every",
                    str(eval_every),
                    "--collection_phases",
                    str(COLLECTION_PHASES),
                    "--refresh_rows",
                    str(REFRESH_ROWS),
                    "--rollouts",
                    str(ROLLOUTS_PER_PROMPT),
                    "--resample_rounds",
                    str(RESAMPLE_ROUNDS),
                    "--bank_size",
                    str(ONLINE_BANK_SIZE),
                    "--replay_window",
                    str(REPLAY_WINDOW),
                    "--selected_alpha",
                    str(selected_alpha),
                    "--selected_lambda",
                    str(selected_lambda),
                    "--lr",
                    "5e-5",
                ]
            training_tasks.append(
                Task(
                    f"train_{method}_seed_{seed}",
                    command,
                    logs / f"train_{method}_seed_{seed}.log",
                )
            )
    _run_group(training_tasks, gpu_ids, repo)

    eval_tasks: list[Task] = []
    eval_records: list[tuple[str, int, str, Path]] = []
    for method in METHODS:
        for seed in CONFIRM_SEEDS:
            training_output = outputs[(method, seed)]
            for checkpoint in ("best", "terminal"):
                adapter = training_output / f"{checkpoint}_adapter"
                if not (adapter / "adapter_config.json").is_file():
                    raise RuntimeError(f"Missing {checkpoint} adapter: {adapter}")
                result = training_output / f"test_metrics_{checkpoint}.json"
                eval_tasks.append(
                    Task(
                        f"test_{method}_{seed}_{checkpoint}",
                        _eval_command(
                            python=sys.executable,
                            runner=runner,
                            model=model,
                            adapter=adapter,
                            data=test,
                            train=train,
                            output=result,
                            plan=plan,
                            seed=seed + 7000,
                        ),
                        logs / f"test_{method}_{seed}_{checkpoint}.log",
                    )
                )
                eval_records.append((method, seed, checkpoint, result))
    _run_group(eval_tasks, gpu_ids, repo)

    summary_rows: list[dict[str, Any]] = []
    for method, seed, checkpoint, result in eval_records:
        manifest = json.loads((outputs[(method, seed)] / "manifest.json").read_text())
        summary_rows.append(
            {
                "method": method,
                "seed": seed,
                "checkpoint": checkpoint,
                "selected_alpha": selected_alpha if method.endswith("dynamic") else 0.0,
                "selected_lambda": selected_lambda if method.endswith("dynamic") else None,
                "stop_reason": manifest.get("stop_reason"),
                "numerical_failure": manifest.get("numerical_failure"),
                **json.loads(result.read_text()),
            }
        )
    fields = sorted({key for row in summary_rows for key in row})
    with (root / "arena_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    paired: dict[str, list[float]] = {}
    for checkpoint in ("best", "terminal"):
        for metric in ("greedy_success", "pass_at_k", "valid_rate"):
            def values(method: str) -> dict[int, float]:
                return {
                    int(row["seed"]): float(row[metric])
                    for row in summary_rows
                    if row["method"] == method and row["checkpoint"] == checkpoint
                }
            cells = {method: values(method) for method in METHODS}
            paired[f"{checkpoint}_{metric}_online_refresh_positive"] = [
                cells["online_positive_only"][seed]
                - cells["frozen_positive_only"][seed]
                for seed in CONFIRM_SEEDS
            ]
            paired[f"{checkpoint}_{metric}_online_refresh_dynamic"] = [
                cells["online_dynamic"][seed] - cells["frozen_dynamic"][seed]
                for seed in CONFIRM_SEEDS
            ]
            paired[f"{checkpoint}_{metric}_negative_effect_frozen"] = [
                cells["frozen_dynamic"][seed]
                - cells["frozen_positive_only"][seed]
                for seed in CONFIRM_SEEDS
            ]
            paired[f"{checkpoint}_{metric}_negative_effect_online"] = [
                cells["online_dynamic"][seed]
                - cells["online_positive_only"][seed]
                for seed in CONFIRM_SEEDS
            ]
            paired[f"{checkpoint}_{metric}_interaction"] = [
                (
                    cells["online_dynamic"][seed]
                    - cells["online_positive_only"][seed]
                )
                - (
                    cells["frozen_dynamic"][seed]
                    - cells["frozen_positive_only"][seed]
                )
                for seed in CONFIRM_SEEDS
            ]

    all_manifests = [
        json.loads((outputs[(method, seed)] / "manifest.json").read_text())
        for method in METHODS
        for seed in CONFIRM_SEEDS
    ]
    completed_updates = [
        int(
            item.get("completed_steps")
            if item.get("completed_steps") is not None
            else (
                item.get("terminal_step")
                if item.get("terminal_step") is not None
                else item.get("last_finite_step", 0)
            )
        )
        for item in all_manifests
    ]
    exact_update_budget_complete = all(
        completed == total_steps
        for completed in completed_updates
    )
    ranking_eligible = bool(
        reference_validation.get("greedy_success", 0.0) >= 0.15
        and reference_validation.get("valid_rate", 0.0) >= 0.95
    )
    terminal_audit = {
        "experiment_id": EXPERIMENT_ID,
        "base_commit": head,
        "selected_alpha": selected_alpha,
        "selected_lambda": selected_lambda,
        "reference_validation": reference_validation,
        "formal_ranking_eligible": ranking_eligible,
        "task_performance": summary_rows,
        "paired_effects": paired,
        "online_off_policy_audit": {
            "collection_phases": COLLECTION_PHASES,
            "post_warmup_fresh_microbatches": FRESH_MICROBATCHES,
            "post_warmup_stale_microbatches": STALE_MICROBATCHES,
            "stale_fraction": 0.5,
            "replay_window": REPLAY_WINDOW,
            "generated_only_negative_bank": True,
            "collector_policy_digest_recorded_each_round": True,
            "replay_age_recorded_per_training_row": True,
            "evaluation_schedule_matched_across_cells": True,
            "optimizer_update_budget_expected": total_steps,
            "optimizer_update_budget_completed": completed_updates,
            "optimizer_update_budget_exactly_matched": exact_update_budget_complete,
        },
        "support_or_structure_boundary": {
            "valid_rate_reported_separately": True,
            "heldout_pattern_metrics_reported": True,
        },
        "numerical": {
            "all_runs_finite": all(not item.get("numerical_failure") for item in all_manifests),
            "nan_inf_reported_separately": True,
        },
        "interpretation_limit": (
            "pilot_only_no_formal_ranking"
            if not ranking_eligible
            else "multi_seed_pilot_requires_effect_review"
        ),
    }
    _atomic_json(root / "terminal_audit.json", terminal_audit)

    if {str(path): _sha256(path) for path in required} != input_hashes:
        raise RuntimeError("A frozen predecessor input changed during V4.6")
    if _hash_tree(reference) != reference_hashes:
        raise RuntimeError("The frozen reference adapter changed during V4.6")
    run_complete = {
        "experiment_id": EXPERIMENT_ID,
        "base_commit": head,
        "predecessor_experiment_id": PREDECESSOR_ID,
        "frozen_source_experiment_id": FROZEN_SOURCE_ID,
        "selected_alpha": selected_alpha,
        "selected_lambda": selected_lambda,
        "reference_validation": reference_validation,
        "methods": METHODS,
        "confirm_seeds": CONFIRM_SEEDS,
        "summary": summary_rows,
        "paired_effects": paired,
        "result_status": "pilot",
        "formal_ranking_eligible": ranking_eligible,
        "test_used_only_after_all_training_finished": True,
        "terminal_audit_present": True,
    }
    _atomic_json(root / "RUN_COMPLETE.json", run_complete)
    _atomic_json(root / "pipeline_status.json", {
        "experiment_id": EXPERIMENT_ID,
        "status": "terminal_audited",
        "completed_unix": time.time(),
    })
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Countdown V4.6 online replay pilot")
    parser.add_argument("--model_path")
    parser.add_argument("--predecessor_work_dir")
    parser.add_argument("--work_dir")
    parser.add_argument("--gpus", default="auto")
    parser.add_argument("--artifact_output", default=None)
    parser.add_argument("--allow_dirty", action="store_true")
    parser.add_argument("--inside_guard", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--online-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-method", help=argparse.SUPPRESS)
    parser.add_argument("--reference_adapter", help=argparse.SUPPRESS)
    parser.add_argument("--train_data", help=argparse.SUPPRESS)
    parser.add_argument("--val_data", help=argparse.SUPPRESS)
    parser.add_argument("--split_manifest", help=argparse.SUPPRESS)
    parser.add_argument("--calibration_json", help=argparse.SUPPRESS)
    parser.add_argument("--output_dir", help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--micro_batch", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--eval_batch", type=int, default=8, help=argparse.SUPPRESS)
    parser.add_argument("--rollout_batch", type=int, default=8, help=argparse.SUPPRESS)
    parser.add_argument("--total_steps", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--eval_every", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--collection_phases", type=int, default=COLLECTION_PHASES, help=argparse.SUPPRESS)
    parser.add_argument("--refresh_rows", type=int, default=REFRESH_ROWS, help=argparse.SUPPRESS)
    parser.add_argument("--rollouts", type=int, default=ROLLOUTS_PER_PROMPT, help=argparse.SUPPRESS)
    parser.add_argument("--resample_rounds", type=int, default=RESAMPLE_ROUNDS, help=argparse.SUPPRESS)
    parser.add_argument("--bank_size", type=int, default=ONLINE_BANK_SIZE, help=argparse.SUPPRESS)
    parser.add_argument("--replay_window", type=int, default=REPLAY_WINDOW, help=argparse.SUPPRESS)
    parser.add_argument("--selected_alpha", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--selected_lambda", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--lr", type=float, default=5e-5, help=argparse.SUPPRESS)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    script = Path(__file__).resolve()
    repo = Path(_git(script.parent, "rev-parse", "--show-toplevel")).resolve()
    if args.online_worker:
        required_worker = (
            "worker_method",
            "model_path",
            "reference_adapter",
            "train_data",
            "val_data",
            "split_manifest",
            "calibration_json",
            "output_dir",
            "seed",
            "total_steps",
            "eval_every",
            "selected_alpha",
            "selected_lambda",
        )
        missing = [name for name in required_worker if getattr(args, name) is None]
        if missing:
            raise SystemExit(f"Missing online-worker arguments: {missing}")
        return _inside_online_worker(args, repo)
    if args.inside_guard:
        return _inside_run(args, repo)
    for name in ("model_path", "predecessor_work_dir", "work_dir"):
        if not getattr(args, name):
            raise SystemExit(f"--{name} is required")
    head = _git(repo, "rev-parse", "HEAD")
    work = Path(args.work_dir).resolve()
    artifact = (
        Path(args.artifact_output).resolve()
        if args.artifact_output
        else work.parent / f"{work.name}_{EXPERIMENT_ID}_pilot.zip"
    )
    if artifact.exists():
        raise SystemExit(f"Artifact output already exists: {artifact}")
    guard = repo / "scripts" / "run_experiment_guard_hardened.py"
    command = [
        sys.executable,
        str(guard),
        "--experiment-id",
        EXPERIMENT_ID,
        "--repo-root",
        str(repo),
        "--output-root",
        str(work),
        "--artifact-output",
        str(artifact),
        "--run-class",
        "pilot",
        "--expected-commit",
        head,
        "--large-file-persistence",
        "persistent_local",
        "--required-output",
        "RUN_COMPLETE.json",
        "--required-output",
        "terminal_audit.json",
        "--required-output",
        "arena_summary.csv",
        "--source-file",
        "scripts/run_countdown_v46_online_replay.py",
        "--source-file",
        "src/drpo/countdown_qwen_arena_onefile.py",
        "--source-file",
        "docs/handoff.md",
        "--source-file",
        "experiments/registry.yaml",
        "--progress-glob",
        "logs/*.log",
        "--progress-glob",
        "training/**/metrics.csv",
        "--progress-glob",
        "training/**/replay/*.manifest.json",
    ]
    if args.allow_dirty:
        command.append("--allow-dirty")
    command.extend(
        [
            "--",
            sys.executable,
            str(script),
            "--inside_guard",
            "--model_path",
            str(Path(args.model_path).resolve()),
            "--predecessor_work_dir",
            str(Path(args.predecessor_work_dir).resolve()),
            "--work_dir",
            str(work),
            "--gpus",
            args.gpus,
        ]
    )
    print(f"Experiment: {EXPERIMENT_ID}")
    print(f"Git commit: {head}")
    print(f"Predecessor: {Path(args.predecessor_work_dir).resolve()}")
    print(f"Work dir: {work}")
    print(f"Artifact: {artifact}")
    return subprocess.run(command, cwd=repo).returncode


if __name__ == "__main__":
    raise SystemExit(main())
