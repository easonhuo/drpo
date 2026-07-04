#!/usr/bin/env python3
"""One-command Countdown full-bank surprisal-gradient diagnostic pipeline.

This wrapper fixes the common foot-gun where ``probe_gradients`` is pointed at
raw ``train.jsonl`` or at a tiny smoke ``offline_6000.jsonl``.  It first checks
whether a complete offline bank with ``near_negative``/``far_negative`` already
exists; if not, it builds that bank with the canonical Countdown runner and then
runs the per-response gradient probe on the built bank.

No method training is performed here.  ``build_offline`` only constructs frozen
near/far negatives and a fixed negative bank from an existing SFT/reference
adapter; ``probe_gradients`` computes diagnostics without an optimizer or
parameter update.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


REQUIRED_OFFLINE_KEYS = (
    "id",
    "prompt",
    "numbers",
    "target",
    "near_negative",
    "far_negative",
    "near_base_surprisal",
    "far_base_surprisal",
)
REQUIRED_PROBE_FIELDS = (
    "seed",
    "puzzle_id",
    "response_role",
    "response",
    "token_count",
    "valid_format",
    "uses_numbers",
    "correct",
    "verifier_category",
    "mean_token_surprisal",
    "stored_base_surprisal",
    "direct_logit_score",
    "negative_coefficient_abs",
    "trainable_parameter_gradient_norm",
)


@dataclass(frozen=True)
class OfflineStatus:
    path: Path
    exists: bool
    rows: int
    complete_rows: int
    has_required_schema: bool
    first_missing_key: str | None
    usable: bool
    reason: str


def _read_jsonl_prefix(path: Path, limit: int | None = None) -> Iterable[dict[str, Any]]:
    with path.open() as handle:
        for index, line in enumerate(handle):
            if limit is not None and index >= limit:
                break
            if line.strip():
                yield json.loads(line)


def inspect_offline_data(path: str | Path, target_examples: int) -> OfflineStatus:
    offline_path = Path(path).expanduser().resolve()
    if not offline_path.exists():
        return OfflineStatus(
            path=offline_path,
            exists=False,
            rows=0,
            complete_rows=0,
            has_required_schema=False,
            first_missing_key=None,
            usable=False,
            reason="missing",
        )
    rows = 0
    complete_rows = 0
    first_missing_key: str | None = None
    with offline_path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            rows += 1
            row = json.loads(line)
            missing = [key for key in REQUIRED_OFFLINE_KEYS if key not in row]
            if missing:
                if first_missing_key is None:
                    first_missing_key = missing[0]
                continue
            complete_rows += 1
    has_schema = rows > 0 and first_missing_key is None and complete_rows == rows
    enough_rows = target_examples <= 0 or rows >= target_examples
    usable = has_schema and enough_rows
    if not has_schema:
        reason = f"missing_key:{first_missing_key}" if first_missing_key else "empty_or_invalid"
    elif not enough_rows:
        reason = f"too_few_rows:{rows}<{target_examples}"
    else:
        reason = "usable"
    return OfflineStatus(
        path=offline_path,
        exists=True,
        rows=rows,
        complete_rows=complete_rows,
        has_required_schema=has_schema,
        first_missing_key=first_missing_key,
        usable=usable,
        reason=reason,
    )


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")


def _adapter_config(path: Path) -> Path:
    return path / "adapter_config.json"


def resolve_adapter(work_dir: Path, explicit_adapter: str | None) -> Path:
    if explicit_adapter:
        adapter = Path(explicit_adapter).expanduser().resolve()
        _require_file(_adapter_config(adapter), "adapter_config.json")
        return adapter
    candidates = [
        work_dir / "sft_adapter" / "best_adapter",
        work_dir / "reference_adapter",
        work_dir / "sft_adapter",
    ]
    for candidate in candidates:
        if _adapter_config(candidate).is_file():
            return candidate.resolve()
    tried = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Could not auto-detect an SFT/reference adapter. Pass --sft_adapter. "
        f"Tried: {tried}"
    )


def _run(command: list[str], *, dry_run: bool, env: dict[str, str] | None = None) -> None:
    print("+ " + " ".join(command), flush=True)
    if dry_run:
        return
    subprocess.run(command, check=True, env=env)


def _resolve_runner(repo_root: Path, explicit_runner: str | None) -> Path:
    runner = (
        Path(explicit_runner).expanduser().resolve()
        if explicit_runner
        else repo_root / "src" / "drpo" / "countdown_qwen_arena_onefile.py"
    )
    _require_file(runner, "Countdown runner")
    return runner


def build_offline_command(args: argparse.Namespace, runner: Path, adapter: Path, offline_data: Path) -> list[str]:
    command = [
        sys.executable,
        str(runner),
        "build_offline",
        "--model_path",
        str(Path(args.model_path).expanduser().resolve()),
        "--reference_adapter",
        str(adapter),
        "--input_data",
        str(Path(args.input_data).expanduser().resolve()),
        "--split_manifest",
        str(Path(args.split_manifest).expanduser().resolve()),
        "--output_data",
        str(offline_data),
        "--negative_bank_size",
        str(args.negative_bank_size),
        "--min_negative_candidates",
        str(args.min_negative_candidates),
        "--batch_size",
        str(args.build_batch_size),
        "--max_examples",
        str(args.target_examples),
        "--pair_resample_rounds",
        str(args.pair_resample_rounds),
        "--synthetic_rescue_candidates",
        str(args.synthetic_rescue_candidates),
        "--score_batch_size",
        str(args.score_batch_size),
        "--rollouts",
        str(args.rollouts),
        "--temperature",
        str(args.temperature),
        "--top_p",
        str(args.top_p),
        "--max_new_tokens",
        str(args.max_new_tokens),
        "--max_length",
        str(args.max_length),
        "--seed",
        str(args.seed),
    ]
    if args.balance_by_oracle_pattern:
        command.append("--balance_by_oracle_pattern")
    if args.nested_sizes:
        command.extend(["--nested_sizes", args.nested_sizes])
    if args.nested_output_dir:
        command.extend(["--nested_output_dir", str(Path(args.nested_output_dir).expanduser().resolve())])
    if args.load_in_4bit:
        command.append("--load_in_4bit")
    command.extend(["--dtype", args.dtype])
    return command


def probe_command(args: argparse.Namespace, runner: Path, adapter: Path, offline_data: Path, output_csv: Path) -> list[str]:
    command = [
        sys.executable,
        str(runner),
        "probe_gradients",
        "--model_path",
        str(Path(args.model_path).expanduser().resolve()),
        "--sft_adapter",
        str(adapter),
        "--offline_data",
        str(offline_data),
        "--output_csv",
        str(output_csv),
        "--gpu",
        str(args.gpu),
        "--seed",
        str(args.seed),
        "--max_length",
        str(args.max_length),
        "--max_stored_surprisal_delta",
        str(args.max_stored_surprisal_delta),
    ]
    if args.probe_max_examples > 0:
        command.extend(["--max_examples", str(args.probe_max_examples)])
    if args.load_in_4bit:
        command.append("--load_in_4bit")
    command.extend(["--dtype", args.dtype])
    return command


def verify_probe_csv(path: Path, expected_puzzles: int) -> dict[str, Any]:
    _require_file(path, "probe output CSV")
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = tuple(reader.fieldnames or ())
        rows = list(reader)
    missing_fields = [field for field in REQUIRED_PROBE_FIELDS if field not in fields]
    if missing_fields:
        raise RuntimeError(f"Probe CSV is missing fields: {missing_fields}")
    expected_responses = expected_puzzles * 2
    if len(rows) != expected_responses:
        raise RuntimeError(
            f"Probe CSV has {len(rows)} rows; expected {expected_responses} "
            f"for {expected_puzzles} puzzles x near/far"
        )
    category_counts: dict[str, int] = {}
    roles: dict[str, int] = {}
    max_delta = 0.0
    for row in rows:
        roles[row["response_role"]] = roles.get(row["response_role"], 0) + 1
        category = row["verifier_category"]
        category_counts[category] = category_counts.get(category, 0) + 1
        coefficient = float(row["negative_coefficient_abs"])
        if coefficient != 1.0:
            raise RuntimeError(f"negative_coefficient_abs is not 1.0: {coefficient}")
        surprisal = float(row["mean_token_surprisal"])
        stored = float(row["stored_base_surprisal"])
        gradient = float(row["trainable_parameter_gradient_norm"])
        score = float(row["direct_logit_score"])
        values = (surprisal, stored, gradient, score)
        if not all(math.isfinite(value) for value in values):
            raise RuntimeError(f"Non-finite numeric value in row: {row}")
        if gradient < 0.0:
            raise RuntimeError(f"Negative gradient norm in row: {row}")
        max_delta = max(max_delta, abs(surprisal - stored))
    if roles.get("near", 0) != expected_puzzles or roles.get("far", 0) != expected_puzzles:
        raise RuntimeError(f"Expected exactly one near and one far response per puzzle; roles={roles}")
    return {
        "csv": str(path),
        "puzzle_count": expected_puzzles,
        "response_count": len(rows),
        "roles": roles,
        "verifier_category_counts": category_counts,
        "max_abs_stored_surprisal_delta": max_delta,
    }


def cmd_main(args: argparse.Namespace) -> None:
    repo_root = Path(args.repo_root).expanduser().resolve()
    work_dir = Path(args.work_dir).expanduser().resolve()
    data_dir = work_dir / "data"
    runner = _resolve_runner(repo_root, args.runner)
    input_data = Path(args.input_data).expanduser().resolve() if args.input_data else data_dir / "train.jsonl"
    split_manifest = (
        Path(args.split_manifest).expanduser().resolve()
        if args.split_manifest
        else data_dir / "split_manifest.json"
    )
    offline_data = (
        Path(args.offline_data).expanduser().resolve()
        if args.offline_data
        else data_dir / f"offline_{args.target_examples}_full.jsonl"
    )
    output_csv = (
        Path(args.output_csv).expanduser().resolve()
        if args.output_csv
        else work_dir / f"countdown_gradient_samples_seed{args.seed}_full.csv"
    )
    pipeline_manifest = (
        Path(args.pipeline_manifest).expanduser().resolve()
        if args.pipeline_manifest
        else Path(str(output_csv) + ".pipeline_manifest.json")
    )

    _require_file(input_data, "raw Countdown train data")
    _require_file(split_manifest, "split manifest")
    adapter = resolve_adapter(work_dir, args.sft_adapter or args.reference_adapter)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    offline_data.parent.mkdir(parents=True, exist_ok=True)

    before = inspect_offline_data(offline_data, args.target_examples)
    build_needed = args.force_rebuild_offline or not before.usable
    if before.exists and not before.usable and args.offline_data and not args.force_rebuild_offline:
        raise RuntimeError(
            f"Explicit --offline_data is not usable ({before.reason}): {before.path}. "
            "Pass --force_rebuild_offline to overwrite it, or omit --offline_data so "
            "the wrapper writes the default offline_6000_full.jsonl path."
        )
    build_command: list[str] | None = None
    if build_needed:
        build_command = build_offline_command(args, runner, adapter, offline_data)
        _run(build_command, dry_run=args.dry_run)
    after = inspect_offline_data(offline_data, args.target_examples)
    if args.dry_run:
        expected_puzzles = args.target_examples if args.probe_max_examples <= 0 else min(args.probe_max_examples, args.target_examples)
    else:
        if not after.usable:
            raise RuntimeError(f"Offline data is still not usable after build decision: {after}")
        expected_puzzles = after.rows
        if args.probe_max_examples > 0:
            expected_puzzles = min(expected_puzzles, args.probe_max_examples)

    probe_cmd = probe_command(args, runner, adapter, offline_data, output_csv)
    _run(probe_cmd, dry_run=args.dry_run)
    probe_summary = None if args.dry_run else verify_probe_csv(output_csv, expected_puzzles)

    manifest = {
        "version": "countdown-gradient-full-pipeline-v1",
        "experiment_id": "EXT-C-E8-V4.4-OFFLINE-BANK",
        "task": "full-bank Countdown surprisal-gradient diagnostic export",
        "repo_root": str(repo_root),
        "runner": str(runner),
        "work_dir": str(work_dir),
        "model_path": str(Path(args.model_path).expanduser().resolve()),
        "adapter": str(adapter),
        "input_data": str(input_data),
        "split_manifest": str(split_manifest),
        "offline_data": str(offline_data),
        "output_csv": str(output_csv),
        "target_examples": args.target_examples,
        "offline_status_before": before.__dict__,
        "offline_status_after": after.__dict__,
        "offline_rebuilt": bool(build_needed),
        "build_command": build_command,
        "probe_command": probe_cmd,
        "probe_summary": probe_summary,
        "scientific_status": "full-bank pilot if this is the only independent SFT/offline seed",
        "training_updates_executed_by_this_wrapper": 0,
        "method_training_started": False,
        "plot_generated": False,
        "dry_run": bool(args.dry_run),
    }
    # Convert paths from dataclass dicts to strings for JSON stability.
    for key in ("offline_status_before", "offline_status_after"):
        if isinstance(manifest[key].get("path"), Path):
            manifest[key]["path"] = str(manifest[key]["path"])
    pipeline_manifest.parent.mkdir(parents=True, exist_ok=True)
    if not args.dry_run:
        pipeline_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a complete Countdown near/far offline bank when needed and then "
            "run probe_gradients on the complete bank."
        )
    )
    parser.add_argument("--model_path", required=True, help="Local Qwen model directory")
    parser.add_argument("--work_dir", required=True, help="Countdown run directory")
    parser.add_argument("--repo_root", default=".")
    parser.add_argument("--runner", default=None, help="Override countdown_qwen_arena_onefile.py path")
    parser.add_argument("--sft_adapter", default=None)
    parser.add_argument("--reference_adapter", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--input_data", default=None, help="Default: WORK_DIR/data/train.jsonl")
    parser.add_argument("--split_manifest", default=None, help="Default: WORK_DIR/data/split_manifest.json")
    parser.add_argument("--offline_data", default=None, help="Default: WORK_DIR/data/offline_TARGET_full.jsonl")
    parser.add_argument("--output_csv", default=None, help="Default: WORK_DIR/countdown_gradient_samples_seedSEED_full.csv")
    parser.add_argument("--pipeline_manifest", default=None)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--target_examples", type=int, default=6000)
    parser.add_argument("--probe_max_examples", type=int, default=0, help="0 means probe the full usable bank")
    parser.add_argument("--force_rebuild_offline", action="store_true")
    parser.add_argument("--dry_run", action="store_true")

    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--dtype", choices=["auto", "bf16", "fp16"], default="auto")
    parser.add_argument("--negative_bank_size", type=int, default=16)
    parser.add_argument("--min_negative_candidates", type=int, default=16)
    parser.add_argument("--build_batch_size", type=int, default=4)
    parser.add_argument("--pair_resample_rounds", type=int, default=8)
    parser.add_argument("--synthetic_rescue_candidates", type=int, default=64)
    parser.add_argument("--score_batch_size", type=int, default=16)
    parser.add_argument("--rollouts", type=int, default=12)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_new_tokens", type=int, default=80)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--max_stored_surprisal_delta", type=float, default=0.01)
    parser.add_argument("--balance_by_oracle_pattern", action="store_true", default=True)
    parser.add_argument("--no_balance_by_oracle_pattern", dest="balance_by_oracle_pattern", action="store_false")
    parser.add_argument("--nested_sizes", default="1500,3000,6000")
    parser.add_argument("--nested_output_dir", default=None)
    return parser


def main() -> None:
    cmd_main(build_parser().parse_args())


if __name__ == "__main__":
    main()
