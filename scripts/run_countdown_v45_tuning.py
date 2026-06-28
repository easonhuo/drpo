#!/usr/bin/env python3
"""Run the registered EXT-C-E8-V4.5 offline-bank alpha/lambda tuning pilot.

This successor reuses the immutable V4.4 reference adapter, data split, frozen
16-negative bank, and initial gradient calibration. It changes only two
registered control parameters in two validation-only stages:

1. negative-scale multiplier alpha in {0.5, 1.0, 1.5, 2.0} at lambda=0.7;
2. taper lambda in {0.3, 0.7, 1.2} at the selected alpha.

The test split is untouched until one alpha/lambda pair has been selected.
Final confirmation compares that pair with Positive-only on three held-out
training seeds. This remains a 0.5B pilot and cannot create a formal ranking
when the inherited reference gate is below 15% greedy success.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import queue
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from threading import Thread
from typing import Any, Iterable

EXPERIMENT_ID = "EXT-C-E8-V4.5-OFFLINE-BANK-TUNING"
PREDECESSOR_ID = "EXT-C-E8-V4.4-OFFLINE-BANK"
ALPHA_CANDIDATES = (0.5, 1.0, 1.5, 2.0)
LAMBDA_CANDIDATES = (0.3, 0.7, 1.2)
ALPHA_STAGE_LAMBDA = 0.7
TUNING_SEEDS = (1234, 2234)
CONFIRM_SEEDS = (3234, 4234, 5234)
VALID_RATE_FLOOR = 0.95


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
    """Hash every regular file in a frozen input directory deterministically."""
    if not root.is_dir():
        raise RuntimeError(f"Frozen input directory does not exist: {root}")
    members = sorted(root.rglob("*"))
    symlinks = [path for path in members if path.is_symlink()]
    if symlinks:
        raise RuntimeError(f"Frozen input directory contains symlinks: {symlinks[0]}")
    files = [path for path in members if path.is_file()]
    if not files:
        raise RuntimeError(f"Frozen input directory is empty: {root}")
    return {str(path.relative_to(root)): _sha256(path) for path in files}


def _float(value: Any) -> float:
    return float(value) if value not in (None, "", "None") else float("nan")


def _tag(value: float) -> str:
    return f"{value:.2f}".replace(".", "p")


def _read_metrics_at_step(path: Path, step: int) -> dict[str, float]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    exact = [row for row in rows if int(float(row["step"])) == int(step)]
    if not exact:
        raise RuntimeError(f"No metrics row for step={step}: {path}")
    row = exact[-1]
    return {
        key: _float(row.get(key))
        for key in (
            "greedy_success", "pass_at_k", "valid_rate",
            "heldout_pattern_family_coverage",
            "heldout_pattern_family_precision_micro",
        )
    }


def summarize_candidate(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(records)
    if not rows:
        raise ValueError("Candidate summary requires at least one seed record")
    eligible = all(
        not row.get("numerical_failure")
        and row["best_metrics"]["valid_rate"] >= VALID_RATE_FLOOR
        for row in rows
    )
    return {
        "eligible": eligible,
        "seeds": [int(row["seed"]) for row in rows],
        "mean_best_greedy_success": mean(row["best_metrics"]["greedy_success"] for row in rows),
        "mean_best_pass_at_k": mean(row["best_metrics"]["pass_at_k"] for row in rows),
        "mean_best_valid_rate": mean(row["best_metrics"]["valid_rate"] for row in rows),
        "mean_terminal_greedy_success": mean(row["terminal_metrics"]["greedy_success"] for row in rows),
        "mean_terminal_pass_at_k": mean(row["terminal_metrics"]["pass_at_k"] for row in rows),
        "mean_terminal_valid_rate": mean(row["terminal_metrics"]["valid_rate"] for row in rows),
        "per_seed": rows,
    }


def choose_candidate(
    summaries: dict[float, dict[str, Any]], *, conservative_tie: str
) -> float:
    eligible = {value: item for value, item in summaries.items() if item["eligible"]}
    if not eligible:
        raise RuntimeError("No tuning candidate passed the registered valid-rate/numerical gate")

    def key(item: tuple[float, dict[str, Any]]) -> tuple[float, ...]:
        value, summary = item
        conservative = -value if conservative_tie == "smaller" else value
        return (
            summary["mean_best_greedy_success"],
            summary["mean_best_pass_at_k"],
            summary["mean_terminal_greedy_success"],
            summary["mean_best_valid_rate"],
            conservative,
        )

    return max(eligible.items(), key=key)[0]


@dataclass(frozen=True)
class Task:
    name: str
    command: list[str]
    log_path: Path


def _run_task(task: Task, gpu_id: str, repo: Path) -> None:
    task.log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
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
    if not tasks:
        return
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
            except BaseException as exc:  # preserve first failure and stop scheduling
                errors.append(exc)
            finally:
                pending.task_done()

    workers = [Thread(target=worker, args=(gpu,), daemon=False) for gpu in gpu_ids]
    for worker_thread in workers:
        worker_thread.start()
    for worker_thread in workers:
        worker_thread.join()
    if errors:
        raise errors[0]


def _discover_reference(predecessor: Path) -> Path:
    candidates = [
        predecessor / "sft_adapter" / "best_adapter",
        predecessor / "reference_adapter",
    ]
    for candidate in candidates:
        if (candidate / "adapter_config.json").exists():
            return candidate
    raise RuntimeError("Could not find the frozen V4.4 LoRA reference adapter")


def _load_training_record(output_dir: Path, seed: int) -> dict[str, Any]:
    manifest = json.loads((output_dir / "manifest.json").read_text())
    metrics_path = output_dir / "metrics.csv"
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
        "negative_scale": manifest.get("negative_scale"),
        "negative_scale_multiplier": manifest.get("negative_scale_multiplier"),
        "best_metrics": _read_metrics_at_step(metrics_path, best_step),
        "terminal_metrics": _read_metrics_at_step(metrics_path, int(terminal_step)),
    }


def _train_command(
    *, python: str, runner: Path, model: Path, reference: Path, offline: Path,
    val: Path, train: Path, output: Path, plan: dict[str, Any], calibration: Path,
    seed: int, alpha: float | None, taper_lambda: float, positive_only: bool,
    steps: int, min_steps: int, eval_every: int,
) -> list[str]:
    command = [
        python, str(runner), "train_method",
        "--model_path", str(model), "--dtype", "bf16",
        "--reference_adapter", str(reference),
        "--offline_data", str(offline), "--val_data", str(val),
        "--structure_reference_data", str(train),
        "--output_dir", str(output),
        "--method", "positive_only" if positive_only else "bank_dynamic_controlled_negative",
        "--steps", str(steps), "--min_steps", str(min_steps),
        "--early_stop_patience", "2", "--early_stop_delta", "0.002",
        "--selection_metric", "greedy_success",
        "--micro_batch", str(plan["micro_batch"]), "--grad_accum", "8",
        "--lr", "5e-5", "--warmup_ratio", "0.03", "--max_grad_norm", "1.0",
        "--eval_examples", "500", "--eval_batch", str(plan["eval_batch"]),
        "--pass_k", "8", "--eval_every", str(eval_every),
        "--eval_seed", str(seed + 6000), "--seed", str(seed),
        "--diagnostic_examples", "32", "--diagnostic_gradient_examples", "8",
        "--diagnostic_batch", "8", "--num_workers", "2",
        "--result_status", "pilot", "--exp_lambda", str(taper_lambda),
        "--surprisal_threshold", "2.0",
    ]
    if not positive_only:
        assert alpha is not None
        command.extend([
            "--negative_calibration_json", str(calibration),
            "--negative_scale_multiplier", str(alpha),
        ])
    return command


def _eval_command(
    *, python: str, runner: Path, model: Path, adapter: Path, data: Path,
    train: Path, output: Path, plan: dict[str, Any], seed: int,
) -> list[str]:
    return [
        python, str(runner), "evaluate", "--model_path", str(model),
        "--dtype", "bf16", "--adapter", str(adapter), "--data", str(data),
        "--structure_reference_data", str(train), "--batch_size", str(plan["eval_batch"]),
        "--pass_k", "8", "--seed", str(seed), "--output_json", str(output),
    ]


def _inside_run(args: argparse.Namespace, repo: Path) -> int:
    sys.path.insert(0, str(repo / "src"))
    from drpo import countdown_qwen_arena_onefile as arena

    head = _git(repo, "rev-parse", "HEAD")
    predecessor = Path(args.predecessor_work_dir).resolve()
    root = Path(args.work_dir).resolve()
    model = Path(args.model_path).resolve()
    if not model.is_dir():
        raise RuntimeError(f"Model directory does not exist: {model}")
    if root.exists() and any(root.iterdir()):
        raise RuntimeError("V4.5 work_dir must be new or empty")
    root.mkdir(parents=True, exist_ok=True)
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    _atomic_json(root / "pipeline_status.json", {
        "experiment_id": EXPERIMENT_ID, "status": "running", "started_unix": time.time()
    })

    complete_path = predecessor / "RUN_COMPLETE.json"
    audit_path = predecessor / "terminal_audit.json"
    if not complete_path.is_file() or not audit_path.is_file():
        raise RuntimeError("V4.4 predecessor must contain RUN_COMPLETE.json and terminal_audit.json")
    predecessor_complete = json.loads(complete_path.read_text())
    predecessor_audit = json.loads(audit_path.read_text())
    if predecessor_complete.get("experiment_id") != PREDECESSOR_ID:
        raise RuntimeError("predecessor_work_dir is not an EXT-C-E8-V4.4 run")
    if predecessor_audit.get("experiment_id") != PREDECESSOR_ID:
        raise RuntimeError("V4.4 terminal audit has the wrong experiment_id")

    data_dir = predecessor / "data"
    train = data_dir / "train.jsonl"
    val = data_dir / "val.jsonl"
    test = data_dir / "test.jsonl"
    offline = data_dir / "offline_6000.jsonl"
    offline_manifest_path = Path(str(offline) + ".manifest.json")
    calibration = predecessor / "negative_budget_calibration.json"
    reference_validation_path = predecessor / "reference_val_metrics.json"
    reference = _discover_reference(predecessor)
    required = [
        train, val, test, offline, offline_manifest_path, calibration,
        reference_validation_path,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"V4.4 predecessor is missing required inputs: {missing}")
    offline_manifest = json.loads(offline_manifest_path.read_text())
    if offline_manifest.get("status") != "complete" or offline_manifest.get("negative_bank_size") != 16:
        raise RuntimeError("V4.5 requires the complete frozen V4.4 16-negative bank")

    input_hashes = {str(path): _sha256(path) for path in required}
    reference_adapter_hashes = _hash_tree(reference)
    predecessor_base_commit = (
        predecessor_complete.get("base_commit")
        or predecessor_audit.get("base_commit")
    )
    reference_validation = (
        predecessor_complete.get("reference_validation")
        or (predecessor_audit.get("reference_gates") or {}).get("lora_reference_validation")
        or json.loads(reference_validation_path.read_text())
    )
    gpu_ids = arena.resolve_gpu_ids(args.gpus, None)
    plan = arena.resolve_execution_plan(
        str(model), "0.5b", "bf16",
        gpu_index=arena._parent_gpu_index(gpu_ids[0]), gpu_visible=gpu_ids[0],
    )
    if plan["dtype"] != "bf16" or plan["load_in_4bit"]:
        raise RuntimeError("V4.5 is frozen to BF16 LoRA")
    offline_rows = len(arena.read_jsonl(offline))
    updates_per_epoch = math.ceil(offline_rows / max(plan["micro_batch"] * 8, 1))
    steps = 6 * updates_per_epoch
    min_steps = 2 * updates_per_epoch
    eval_every = updates_per_epoch
    runner = repo / "src" / "drpo" / "countdown_qwen_arena_onefile.py"

    run_config = {
        "experiment_id": EXPERIMENT_ID,
        "base_commit": head,
        "predecessor_experiment_id": PREDECESSOR_ID,
        "predecessor_base_commit": predecessor_base_commit,
        "predecessor_work_dir": str(predecessor),
        "model_path": str(model),
        "gpu_ids": gpu_ids,
        "alpha_candidates": ALPHA_CANDIDATES,
        "lambda_candidates": LAMBDA_CANDIDATES,
        "alpha_stage_lambda": ALPHA_STAGE_LAMBDA,
        "tuning_seeds": TUNING_SEEDS,
        "confirm_seeds": CONFIRM_SEEDS,
        "surprisal_threshold": 2.0,
        "selection": "validation_only_lexicographic_greedy_pass_terminal_valid_then_conservative",
        "test_access": "only_after_alpha_and_lambda_selection",
        "input_hashes": input_hashes,
        "reference_adapter_hashes": reference_adapter_hashes,
        "plan": plan,
    }
    _atomic_json(root / "run_config.json", run_config)

    # Stage A: alpha scan at frozen lambda=0.7.
    alpha_tasks: list[Task] = []
    alpha_outputs: dict[tuple[float, int], Path] = {}
    for alpha in ALPHA_CANDIDATES:
        for seed in TUNING_SEEDS:
            output = root / "alpha_sweep" / f"alpha_{_tag(alpha)}" / f"seed_{seed}"
            alpha_outputs[(alpha, seed)] = output
            alpha_tasks.append(Task(
                f"alpha_{alpha}_seed_{seed}",
                _train_command(
                    python=sys.executable, runner=runner, model=model, reference=reference,
                    offline=offline, val=val, train=train, output=output, plan=plan,
                    calibration=calibration, seed=seed, alpha=alpha,
                    taper_lambda=ALPHA_STAGE_LAMBDA, positive_only=False,
                    steps=steps, min_steps=min_steps, eval_every=eval_every,
                ),
                logs / f"alpha_{_tag(alpha)}_seed_{seed}.log",
            ))
    _run_group(alpha_tasks, gpu_ids, repo)
    alpha_summaries = {
        alpha: summarize_candidate(
            _load_training_record(alpha_outputs[(alpha, seed)], seed)
            for seed in TUNING_SEEDS
        )
        for alpha in ALPHA_CANDIDATES
    }
    selected_alpha = choose_candidate(alpha_summaries, conservative_tie="smaller")
    _atomic_json(root / "alpha_selection.json", {
        "selected_alpha": selected_alpha,
        "lambda_fixed": ALPHA_STAGE_LAMBDA,
        "summaries": {str(k): v for k, v in alpha_summaries.items()},
        "test_data_used": False,
    })

    # Stage B: lambda scan at selected alpha. Reuse lambda=0.7 alpha-stage runs.
    lambda_outputs: dict[tuple[float, int], Path] = {
        (ALPHA_STAGE_LAMBDA, seed): alpha_outputs[(selected_alpha, seed)]
        for seed in TUNING_SEEDS
    }
    lambda_tasks: list[Task] = []
    for taper_lambda in LAMBDA_CANDIDATES:
        if math.isclose(taper_lambda, ALPHA_STAGE_LAMBDA):
            continue
        for seed in TUNING_SEEDS:
            output = root / "lambda_sweep" / f"lambda_{_tag(taper_lambda)}" / f"seed_{seed}"
            lambda_outputs[(taper_lambda, seed)] = output
            lambda_tasks.append(Task(
                f"lambda_{taper_lambda}_seed_{seed}",
                _train_command(
                    python=sys.executable, runner=runner, model=model, reference=reference,
                    offline=offline, val=val, train=train, output=output, plan=plan,
                    calibration=calibration, seed=seed, alpha=selected_alpha,
                    taper_lambda=taper_lambda, positive_only=False,
                    steps=steps, min_steps=min_steps, eval_every=eval_every,
                ),
                logs / f"lambda_{_tag(taper_lambda)}_seed_{seed}.log",
            ))
    _run_group(lambda_tasks, gpu_ids, repo)
    lambda_summaries = {
        taper_lambda: summarize_candidate(
            _load_training_record(lambda_outputs[(taper_lambda, seed)], seed)
            for seed in TUNING_SEEDS
        )
        for taper_lambda in LAMBDA_CANDIDATES
    }
    selected_lambda = choose_candidate(lambda_summaries, conservative_tie="larger")
    _atomic_json(root / "lambda_selection.json", {
        "selected_alpha": selected_alpha,
        "selected_lambda": selected_lambda,
        "summaries": {str(k): v for k, v in lambda_summaries.items()},
        "test_data_used": False,
    })

    # Final untouched-seed confirmation: selected dynamic vs Positive-only.
    confirm_tasks: list[Task] = []
    confirm_outputs: dict[tuple[str, int], Path] = {}
    for method in ("positive_only", "selected_dynamic"):
        for seed in CONFIRM_SEEDS:
            output = root / "confirmation" / method / f"seed_{seed}"
            confirm_outputs[(method, seed)] = output
            confirm_tasks.append(Task(
                f"confirm_{method}_seed_{seed}",
                _train_command(
                    python=sys.executable, runner=runner, model=model, reference=reference,
                    offline=offline, val=val, train=train, output=output, plan=plan,
                    calibration=calibration, seed=seed,
                    alpha=None if method == "positive_only" else selected_alpha,
                    taper_lambda=selected_lambda,
                    positive_only=method == "positive_only",
                    steps=steps, min_steps=min_steps, eval_every=eval_every,
                ),
                logs / f"confirm_{method}_seed_{seed}.log",
            ))
    _run_group(confirm_tasks, gpu_ids, repo)

    eval_tasks: list[Task] = []
    eval_records: list[tuple[str, int, str, Path]] = []
    for method in ("positive_only", "selected_dynamic"):
        for seed in CONFIRM_SEEDS:
            output = confirm_outputs[(method, seed)]
            for checkpoint in ("best", "terminal"):
                adapter = output / f"{checkpoint}_adapter"
                result = output / f"test_metrics_{checkpoint}.json"
                if not (adapter / "adapter_config.json").exists():
                    raise RuntimeError(f"Missing {checkpoint} adapter: {adapter}")
                eval_tasks.append(Task(
                    f"test_{method}_{seed}_{checkpoint}",
                    _eval_command(
                        python=sys.executable, runner=runner, model=model, adapter=adapter,
                        data=test, train=train, output=result, plan=plan, seed=seed + 7000,
                    ),
                    logs / f"test_{method}_{seed}_{checkpoint}.log",
                ))
                eval_records.append((method, seed, checkpoint, result))
    reference_test = root / "reference_test_metrics.json"
    eval_tasks.append(Task(
        "test_reference",
        _eval_command(
            python=sys.executable, runner=runner, model=model, adapter=reference,
            data=test, train=train, output=reference_test, plan=plan, seed=7000,
        ),
        logs / "test_reference.log",
    ))
    _run_group(eval_tasks, gpu_ids, repo)

    summary_rows: list[dict[str, Any]] = []
    for method, seed, checkpoint, result in eval_records:
        manifest = json.loads((confirm_outputs[(method, seed)] / "manifest.json").read_text())
        summary_rows.append({
            "method": method, "seed": seed, "checkpoint": checkpoint,
            "selected_alpha": selected_alpha if method == "selected_dynamic" else 0.0,
            "selected_lambda": selected_lambda if method == "selected_dynamic" else None,
            "stop_reason": manifest.get("stop_reason"),
            "numerical_failure": manifest.get("numerical_failure"),
            **json.loads(result.read_text()),
        })
    summary_path = root / "arena_summary.csv"
    fields = sorted({key for row in summary_rows for key in row})
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    paired: dict[str, list[float]] = {}
    for checkpoint in ("best", "terminal"):
        for metric in ("greedy_success", "pass_at_k", "valid_rate"):
            diffs = []
            for seed in CONFIRM_SEEDS:
                dynamic = next(row for row in summary_rows if row["method"] == "selected_dynamic" and row["seed"] == seed and row["checkpoint"] == checkpoint)
                positive = next(row for row in summary_rows if row["method"] == "positive_only" and row["seed"] == seed and row["checkpoint"] == checkpoint)
                diffs.append(float(dynamic[metric]) - float(positive[metric]))
            paired[f"{checkpoint}_{metric}"] = diffs
    ranking_eligible = bool(
        reference_validation.get("greedy_success", 0.0) >= 0.15
        and reference_validation.get("valid_rate", 0.0) >= 0.95
    )
    terminal_audit = {
        "experiment_id": EXPERIMENT_ID,
        "base_commit": head,
        "selected_alpha": selected_alpha,
        "selected_lambda": selected_lambda,
        "formal_ranking_eligible": ranking_eligible,
        "task_performance": summary_rows,
        "paired_dynamic_minus_positive_only": paired,
        "support_or_structure_boundary": {
            "valid_rate_reported_separately": True,
            "heldout_pattern_metrics_reported": True,
        },
        "numerical": {
            "all_confirmation_runs_finite": all(not row["numerical_failure"] for row in summary_rows),
            "nan_inf_reported_separately": True,
        },
        "interpretation_limit": (
            "pilot_only_no_formal_ranking" if not ranking_eligible else "multi_seed_confirmation_requires_effect_review"
        ),
    }
    _atomic_json(root / "terminal_audit.json", terminal_audit)

    if {str(path): _sha256(path) for path in required} != input_hashes:
        raise RuntimeError("A frozen V4.4 predecessor file changed during V4.5")
    if _hash_tree(reference) != reference_adapter_hashes:
        raise RuntimeError("The frozen V4.4 reference adapter changed during V4.5")
    complete = {
        "experiment_id": EXPERIMENT_ID,
        "base_commit": head,
        "predecessor": {
            "experiment_id": PREDECESSOR_ID,
            "base_commit": predecessor_base_commit,
            "work_dir": str(predecessor),
            "input_hashes": input_hashes,
            "reference_adapter_hashes": reference_adapter_hashes,
        },
        "selected_alpha": selected_alpha,
        "selected_lambda": selected_lambda,
        "alpha_selection": json.loads((root / "alpha_selection.json").read_text()),
        "lambda_selection": json.loads((root / "lambda_selection.json").read_text()),
        "reference_test": json.loads(reference_test.read_text()),
        "summary": summary_rows,
        "terminal_audit_present": True,
        "result_status": "pilot",
        "formal_ranking_eligible": ranking_eligible,
        "test_used_only_after_selection": True,
    }
    _atomic_json(root / "RUN_COMPLETE.json", complete)
    _atomic_json(root / "run_complete.json", complete)
    _atomic_json(root / "pipeline_status.json", {
        "experiment_id": EXPERIMENT_ID,
        "status": "terminal_audited",
        "completed_unix": time.time(),
        "selected_alpha": selected_alpha,
        "selected_lambda": selected_lambda,
    })
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Countdown V4.5 alpha/lambda tuning")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--predecessor_work_dir", required=True)
    parser.add_argument("--work_dir", required=True)
    parser.add_argument("--gpus", default="auto")
    parser.add_argument("--artifact_output", default=None)
    parser.add_argument("--allow_dirty", action="store_true")
    parser.add_argument("--inside_guard", action="store_true", help=argparse.SUPPRESS)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    script = Path(__file__).resolve()
    repo = Path(_git(script.parent, "rev-parse", "--show-toplevel")).resolve()
    if args.inside_guard:
        return _inside_run(args, repo)

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
        sys.executable, str(guard),
        "--experiment-id", EXPERIMENT_ID,
        "--repo-root", str(repo),
        "--output-root", str(work),
        "--artifact-output", str(artifact),
        "--run-class", "pilot",
        "--expected-commit", head,
        "--large-file-persistence", "persistent_local",
        "--required-output", "RUN_COMPLETE.json",
        "--required-output", "terminal_audit.json",
        "--required-output", "arena_summary.csv",
        "--source-file", "scripts/run_countdown_v45_tuning.py",
        "--source-file", "src/drpo/countdown_qwen_arena_onefile.py",
        "--source-file", "docs/handoff.md",
        "--source-file", "experiments/registry.yaml",
        "--progress-glob", "logs/*.log",
        "--progress-glob", "**/metrics.csv",
    ]
    if args.allow_dirty:
        command.append("--allow-dirty")
    command.extend([
        "--", sys.executable, str(script),
        "--inside_guard",
        "--model_path", str(Path(args.model_path).resolve()),
        "--predecessor_work_dir", str(Path(args.predecessor_work_dir).resolve()),
        "--work_dir", str(work),
        "--gpus", args.gpus,
    ])
    print(f"Experiment: {EXPERIMENT_ID}")
    print(f"Git commit: {head}")
    print(f"Predecessor: {Path(args.predecessor_work_dir).resolve()}")
    print(f"Work dir: {work}")
    print(f"Artifact: {artifact}")
    return subprocess.run(command, cwd=repo).returncode


if __name__ == "__main__":
    raise SystemExit(main())
