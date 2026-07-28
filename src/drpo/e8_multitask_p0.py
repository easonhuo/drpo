"""P0 pipeline for frozen multitask banks and Figure-1-style diagnostics.

Stages
------
``prepare``
    Acquire pinned public sources and build model-independent oracle/negative banks.
``qualify``
    Audit oracle validity, uniqueness, negative coverage, and output diversity.
``diagnose``
    Measure current-policy mean-token surprisal and raw full-parameter gradient
    norm for frozen negative completions.  This stage does not train a policy.
``aggregate``
    Bin within each task, normalize within task, and aggregate tasks equally.

The ``all`` command is fail closed: it never loads a model when any requested
task fails bank qualification.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import tarfile
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

try:
    import torch
    import torch.nn.functional as F
except ImportError:  # Bank preparation and qualification do not require Torch.
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]

from drpo.e8_multitask_tasks import (
    REASONING_GYM_COMMIT,
    TASK_NAMES,
    WIKISQL_COMMIT,
    build_adapters,
    stable_hash,
)

EXPERIMENT_ID = "EXT-C-E8-MULTITASK-P0-01"
DEFAULT_CONFIG = Path("configs/e8_multitask_p0.yaml")
REASONING_GYM_URL = "https://github.com/open-thought/reasoning-gym.git"
WIKISQL_URL = "https://github.com/salesforce/WikiSQL.git"


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_config_hash(config: Mapping[str, Any]) -> str:
    return stable_hash(config)


def run_checked(args: Sequence[str], *, cwd: Path | None = None) -> str:
    completed = subprocess.run(
        list(args),
        cwd=cwd,
        check=True,
        text=True,
        capture_output=True,
    )
    return completed.stdout.strip()


def validate_work_dir(path: str | Path) -> Path:
    path = Path(path).resolve()
    if path == Path(path.anchor) or len(path.parts) < 3:
        raise ValueError(f"Refusing unsafe work directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise TypeError("Configuration root must be a mapping")
    if config.get("schema_version") != 1:
        raise ValueError("Expected schema_version: 1")
    if config.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError(f"Expected experiment_id: {EXPERIMENT_ID}")
    pins = config.get("sources", {})
    if pins.get("reasoning_gym", {}).get("commit") != REASONING_GYM_COMMIT:
        raise ValueError("Reasoning Gym commit does not match the audited pin")
    if pins.get("wikisql", {}).get("commit") != WIKISQL_COMMIT:
        raise ValueError("WikiSQL commit does not match the audited pin")
    names = tuple(config.get("tasks", {}).get("names", ()))
    if not names or len(names) != len(set(names)):
        raise ValueError("tasks.names must be a non-empty unique list")
    unknown = sorted(set(names) - set(TASK_NAMES))
    if unknown:
        raise ValueError(f"Unknown tasks: {unknown}")
    bank = config.get("bank", {})
    if int(bank.get("rows_per_task", 0)) <= 0:
        raise ValueError("bank.rows_per_task must be positive")
    if int(bank.get("candidate_rows_per_task", 0)) < int(bank["rows_per_task"]):
        raise ValueError("candidate_rows_per_task must be >= rows_per_task")
    if int(bank.get("negatives_per_prompt", 0)) <= 0:
        raise ValueError("bank.negatives_per_prompt must be positive")
    return config


def with_smoke_overrides(
    config: Mapping[str, Any],
    *,
    rows: int | None,
    negatives: int | None,
) -> dict[str, Any]:
    updated = copy.deepcopy(dict(config))
    if rows is not None:
        if rows <= 0:
            raise ValueError("smoke rows must be positive")
        updated["bank"]["rows_per_task"] = rows
        updated["bank"]["candidate_rows_per_task"] = max(rows * 3, rows)
    if negatives is not None:
        if negatives <= 0:
            raise ValueError("smoke negatives must be positive")
        updated["bank"]["negatives_per_prompt"] = negatives
        qualification = updated["qualification"]
        qualification["minimum_negatives_per_prompt"] = min(
            int(qualification["minimum_negatives_per_prompt"]),
            negatives,
        )
    updated["runtime_override"] = {
        "smoke_rows": rows,
        "smoke_negatives": negatives,
    }
    return updated


def git_head(checkout: Path) -> str:
    return run_checked(["git", "rev-parse", "HEAD"], cwd=checkout)


def acquire_git_checkout(
    *,
    destination: Path,
    url: str,
    commit: str,
    skip_download: bool,
) -> dict[str, Any]:
    if destination.exists() and not (destination / ".git").is_dir():
        raise RuntimeError(f"Source destination exists but is not a git checkout: {destination}")
    newly_cloned = False
    if not destination.exists():
        if skip_download:
            raise FileNotFoundError(f"Missing pinned checkout: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        run_checked(
            [
                "git",
                "clone",
                "--filter=blob:none",
                "--no-checkout",
                url,
                str(destination),
            ]
        )
        newly_cloned = True
    current = git_head(destination)
    if newly_cloned:
        run_checked(["git", "checkout", "--detach", commit], cwd=destination)
    elif current != commit:
        if skip_download:
            raise RuntimeError(
                f"Source pin mismatch at {destination}: expected {commit}, found {current}"
            )
        try:
            run_checked(["git", "checkout", "--detach", commit], cwd=destination)
        except subprocess.CalledProcessError:
            run_checked(["git", "fetch", "origin", commit], cwd=destination)
            run_checked(["git", "checkout", "--detach", commit], cwd=destination)
    resolved = git_head(destination)
    if resolved != commit:
        raise RuntimeError(f"Could not resolve exact source commit {commit}: {resolved}")
    status = run_checked(["git", "status", "--porcelain"], cwd=destination)
    if status:
        raise RuntimeError(f"Pinned source checkout is dirty: {destination}")
    return {
        "url": url,
        "commit": resolved,
        "checkout": str(destination),
        "clean": True,
    }


def safe_extract_tar(archive: Path, destination: Path) -> None:
    destination = destination.resolve()
    with tarfile.open(archive, mode="r:bz2") as bundle:
        members = bundle.getmembers()
        for member in members:
            target = (destination / member.name).resolve()
            if destination not in target.parents and target != destination:
                raise RuntimeError(f"Unsafe archive member: {member.name}")
            if member.issym() or member.islnk():
                raise RuntimeError(f"Links are not allowed in source archives: {member.name}")
        if sys.version_info >= (3, 12):
            bundle.extractall(destination, members=members, filter="data")
        else:
            bundle.extractall(destination, members=members)


def acquire_sources(
    config: Mapping[str, Any],
    work_dir: Path,
    *,
    skip_download: bool,
) -> dict[str, Any]:
    sources_root = work_dir / "sources"
    sources_root.mkdir(parents=True, exist_ok=True)
    source_config = config["sources"]
    reasoning_gym = acquire_git_checkout(
        destination=sources_root / "reasoning-gym",
        url=str(source_config["reasoning_gym"]["url"]),
        commit=str(source_config["reasoning_gym"]["commit"]),
        skip_download=skip_download,
    )
    wikisql_root = sources_root / "wikisql"
    wikisql = acquire_git_checkout(
        destination=wikisql_root,
        url=str(source_config["wikisql"]["url"]),
        commit=str(source_config["wikisql"]["commit"]),
        skip_download=skip_download,
    )
    archive = wikisql_root / str(source_config["wikisql"]["archive"])
    if not archive.is_file():
        raise FileNotFoundError(f"WikiSQL archive is absent from pinned checkout: {archive}")
    if not (wikisql_root / "data" / "train.jsonl").is_file():
        if skip_download:
            raise FileNotFoundError("WikiSQL archive is present but data has not been extracted")
        safe_extract_tar(archive, wikisql_root)
    wikisql["archive"] = str(archive)
    wikisql["archive_sha256"] = sha256_file(archive)
    manifest = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "reasoning_gym": reasoning_gym,
        "wikisql": wikisql,
    }
    atomic_json(work_dir / "source_manifest.json", manifest)
    return manifest


def bank_path(work_dir: Path, task: str) -> Path:
    return work_dir / "banks" / f"{task}.jsonl"


def build_banks(
    config: Mapping[str, Any],
    work_dir: Path,
    *,
    force: bool,
) -> dict[str, Any]:
    sources_root = work_dir / "sources"
    adapters = build_adapters(config, sources_root)
    target_rows = int(config["bank"]["rows_per_task"])
    candidate_rows = int(config["bank"]["candidate_rows_per_task"])
    negative_count = int(config["bank"]["negatives_per_prompt"])
    base_seed = int(config["bank"]["generation_seed"])
    task_audits: dict[str, Any] = {}
    for task_index, (task, adapter) in enumerate(adapters.items()):
        output = bank_path(work_dir, task)
        audit_path = work_dir / "banks" / f"{task}.build_audit.json"
        if output.is_file() and audit_path.is_file() and not force:
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            if (
                audit.get("config_hash") == stable_config_hash(config)
                and audit.get("bank_sha256") == sha256_file(output)
                and audit.get("accepted_rows") == target_rows
            ):
                task_audits[task] = audit
                continue
            raise RuntimeError(
                f"Existing bank identity mismatch for {task}; pass --force to rebuild"
            )

        task_seed = base_seed + task_index * 100_003
        temporary = output.parent / f".{output.name}.building"
        output.parent.mkdir(parents=True, exist_ok=True)
        if temporary.exists():
            temporary.unlink()
        accepted = 0
        dropped: Counter[str] = Counter()
        class_counts: Counter[str] = Counter()
        with temporary.open("w", encoding="utf-8") as handle:
            for instance in adapter.generate_instances(candidate_rows, task_seed):
                row, row_audit = adapter.build_bank_row(
                    instance,
                    negative_count=negative_count,
                    seed=task_seed,
                )
                if row is None:
                    dropped[str(row_audit["reason"])] += 1
                    continue
                row["experiment_id"] = EXPERIMENT_ID
                row["generation_seed"] = task_seed
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                accepted += 1
                class_counts.update(item["error_class"] for item in row["negatives"])
                if accepted >= target_rows:
                    break
        temporary.replace(output)
        audit = {
            "schema_version": 1,
            "experiment_id": EXPERIMENT_ID,
            "task": task,
            "config_hash": stable_config_hash(config),
            "target_rows": target_rows,
            "candidate_rows": candidate_rows,
            "accepted_rows": accepted,
            "dropped_by_reason": dict(sorted(dropped.items())),
            "negative_count": accepted * negative_count,
            "negative_error_class_counts": dict(sorted(class_counts.items())),
            "bank_path": str(output),
            "bank_sha256": sha256_file(output),
            "complete": accepted == target_rows,
        }
        atomic_json(audit_path, audit)
        task_audits[task] = audit

    manifest = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "config_hash": stable_config_hash(config),
        "model_independent_bank": True,
        "forbidden_bank_fields": [
            "near",
            "far",
            "surprisal",
            "policy_probability",
            "taper_weight",
        ],
        "tasks": task_audits,
        "complete": all(audit["complete"] for audit in task_audits.values()),
    }
    atomic_json(work_dir / "bank_manifest.json", manifest)
    return manifest


def cmd_prepare(
    config: Mapping[str, Any],
    work_dir: Path,
    *,
    force: bool,
    skip_download: bool,
) -> dict[str, Any]:
    acquire_sources(config, work_dir, skip_download=skip_download)
    return build_banks(config, work_dir, force=force)


def qualify_task(
    task: str,
    rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    qualification = config["qualification"]
    target_rows = int(config["bank"]["rows_per_task"])
    minimum_negatives = int(qualification["minimum_negatives_per_prompt"])
    minimum_classes = int(qualification["minimum_error_classes_per_prompt"])
    maximum_chars = int(qualification["maximum_completion_characters"])
    prompt_audits: list[dict[str, Any]] = []
    task_classes: Counter[str] = Counter()
    oracle_passes = 0
    format_valid_count = 0
    negative_count = 0
    duplicate_count = 0
    for row in rows:
        oracle_ok = bool(row.get("oracle_verification", {}).get("correct"))
        oracle_passes += int(oracle_ok)
        negatives = list(row.get("negatives", ()))
        canonical = [str(item.get("canonical_completion", "")) for item in negatives]
        duplicates = len(canonical) - len(set(canonical))
        duplicate_count += duplicates
        classes = {
            str(item.get("error_class")) for item in negatives if bool(item.get("format_valid"))
        }
        task_classes.update(str(item.get("error_class")) for item in negatives)
        format_valid_count += sum(bool(item.get("format_valid")) for item in negatives)
        negative_count += len(negatives)
        too_long = sum(int(item.get("response_chars", 0)) > maximum_chars for item in negatives)
        wrongly_correct = sum(bool(item.get("binary_correct")) for item in negatives)
        passed = (
            oracle_ok
            and len(negatives) >= minimum_negatives
            and len(classes) >= minimum_classes
            and duplicates == 0
            and too_long == 0
            and wrongly_correct == 0
        )
        prompt_audits.append(
            {
                "prompt_id": row.get("prompt_id"),
                "passed": passed,
                "oracle_ok": oracle_ok,
                "negative_count": len(negatives),
                "format_valid_error_classes": sorted(classes),
                "duplicates": duplicates,
                "too_long": too_long,
                "wrongly_correct": wrongly_correct,
            }
        )

    prompt_passes = sum(bool(item["passed"]) for item in prompt_audits)
    denominator = max(len(rows), 1)
    negative_denominator = max(negative_count, 1)
    metrics = {
        "row_count": len(rows),
        "target_row_count": target_rows,
        "oracle_verification_rate": oracle_passes / denominator,
        "prompt_pass_fraction": prompt_passes / denominator,
        "format_valid_negative_fraction": format_valid_count / negative_denominator,
        "duplicate_negative_count": duplicate_count,
        "task_error_classes": sorted(task_classes),
        "negative_count": negative_count,
    }
    gates = {
        "target_rows_complete": len(rows) == target_rows,
        "oracle_verification_rate": metrics["oracle_verification_rate"]
        >= float(qualification["minimum_oracle_verification_rate"]),
        "prompt_pass_fraction": metrics["prompt_pass_fraction"]
        >= float(qualification["minimum_prompt_pass_fraction"]),
        "format_valid_negative_fraction": metrics["format_valid_negative_fraction"]
        >= float(qualification["minimum_format_valid_negative_fraction"]),
        "task_error_class_count": len(task_classes)
        >= int(qualification["minimum_error_classes_per_task"]),
        "no_duplicate_negatives": duplicate_count == 0,
    }
    return {
        "task": task,
        "passed": all(gates.values()),
        "metrics": metrics,
        "gates": gates,
        "prompt_failures": [item for item in prompt_audits if not item["passed"]],
    }


def cmd_qualify(config: Mapping[str, Any], work_dir: Path) -> dict[str, Any]:
    task_results: dict[str, Any] = {}
    for task in config["tasks"]["names"]:
        path = bank_path(work_dir, task)
        if not path.is_file():
            task_results[task] = {
                "task": task,
                "passed": False,
                "error": "missing_bank",
                "path": str(path),
            }
            continue
        task_results[task] = qualify_task(task, read_jsonl(path), config)
    failed = [task for task, result in task_results.items() if not result["passed"]]
    audit = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "config_hash": stable_config_hash(config),
        "passed": not failed,
        "failed_tasks": failed,
        "tasks": task_results,
        "scientific_status": "not_run",
        "note": "Bank qualification is an engineering/data gate, not experimental evidence.",
    }
    atomic_json(work_dir / "qualification_audit.json", audit)
    return audit


@dataclass(frozen=True)
class EncodedCompletion:
    input_ids: list[int]
    labels: list[int]


def format_chat_prompt(tokenizer: Any, prompt: str) -> str:
    messages = [
        {
            "role": "system",
            "content": "Answer with only the requested final output and no explanation.",
        },
        {"role": "user", "content": prompt},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
    return f"System: {messages[0]['content']}\nUser: {prompt}\nAssistant:"


def encode_prompt_completion(
    tokenizer: Any,
    prompt: str,
    completion: str,
    max_length: int,
) -> EncodedCompletion:
    prefix = format_chat_prompt(tokenizer, prompt)
    eos = getattr(tokenizer, "eos_token", None) or ""
    prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(prefix + completion + eos, add_special_tokens=False)["input_ids"]
    full_ids = list(full_ids[:max_length])
    prefix_length = min(len(prefix_ids), len(full_ids))
    labels = [-100] * prefix_length + full_ids[prefix_length:]
    if not full_ids or all(value == -100 for value in labels):
        raise ValueError("Completion was truncated; increase diagnostics.max_length")
    return EncodedCompletion(input_ids=full_ids, labels=labels)


def completion_stats(
    model: Any,
    encoded: EncodedCompletion,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    input_ids = torch.tensor([encoded.input_ids], dtype=torch.long, device=device)
    labels = torch.tensor([encoded.labels], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    output = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    logits = output.logits[:, :-1, :].float()
    shifted_labels = labels[:, 1:]
    mask = shifted_labels.ne(-100)
    safe_labels = shifted_labels.masked_fill(~mask, 0)
    log_probs = F.log_softmax(logits, dim=-1)
    probabilities = log_probs.exp()
    token_log_probs = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    lengths = mask.sum(-1).clamp_min(1)
    sequence_log_prob = (token_log_probs * mask).sum(-1) / lengths
    selected_probability = probabilities.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    probability_l2 = probabilities.square().sum(-1)
    token_score = torch.sqrt(
        torch.clamp(1.0 - 2.0 * selected_probability + probability_l2, min=0.0)
    )
    score = (token_score * mask).sum(-1) / lengths
    return {
        "seq_lp": sequence_log_prob,
        "score": score,
        "lengths": lengths,
    }


def model_identity(model_path: str, adapter_path: str | None) -> dict[str, Any]:
    def describe(path_value: str | None) -> dict[str, Any] | None:
        if path_value is None:
            return None
        path = Path(path_value).expanduser()
        result: dict[str, Any] = {"requested": path_value}
        if not path.exists():
            result["kind"] = "remote_or_unresolved_identifier"
            return result
        path = path.resolve()
        result.update({"kind": "local_path", "resolved": str(path)})
        names = (
            "config.json",
            "adapter_config.json",
            "tokenizer_config.json",
            "generation_config.json",
        )
        result["identity_files"] = {
            name: sha256_file(path / name) for name in names if (path / name).is_file()
        }
        return result

    return {"model": describe(model_path), "adapter": describe(adapter_path)}


def load_diagnostic_model(
    model_path: str,
    adapter_path: str | None,
    *,
    dtype: str,
) -> tuple[Any, Any]:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("diagnose requires transformers") from exc
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    if dtype == "auto":
        torch_dtype = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else (torch.float16 if torch.cuda.is_available() else torch.float32)
        )
    else:
        mapping = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }
        if dtype not in mapping:
            raise ValueError(f"Unknown diagnostic dtype: {dtype}")
        torch_dtype = mapping[dtype]
    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": torch_dtype,
    }
    if torch.cuda.is_available():
        kwargs["device_map"] = {"": int(os.environ.get("LOCAL_RANK", "0"))}
    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    if adapter_path:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise RuntimeError("Loading adapter_path requires peft") from exc
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
        if hasattr(model, "merge_and_unload"):
            model = model.merge_and_unload()
    if not torch.cuda.is_available():
        model.to(torch.device("cpu"))
    for parameter in model.parameters():
        if not parameter.is_floating_point():
            raise RuntimeError("Full-parameter diagnostic requires floating-point parameters")
        parameter.requires_grad_(True)
    model.config.use_cache = False
    model.eval()
    return model, tokenizer


def select_diagnostic_points(
    rows: Sequence[Mapping[str, Any]],
    *,
    task: str,
    limit: int,
    seed: int,
) -> list[dict[str, Any]]:
    prompts: list[tuple[str, Mapping[str, Any], list[Mapping[str, Any]]]] = []
    for row in rows:
        negatives = sorted(
            row["negatives"],
            key=lambda item: stable_hash(
                {
                    "task": task,
                    "prompt_id": row["prompt_id"],
                    "negative_id": item["negative_id"],
                    "seed": seed,
                }
            ),
        )
        prompts.append((str(row["prompt_id"]), row, negatives))
    prompts.sort(key=lambda item: stable_hash({"task": task, "prompt_id": item[0], "seed": seed}))
    points: list[dict[str, Any]] = []
    depth = 0
    while len(points) < limit:
        added = 0
        for _, row, negatives in prompts:
            if depth >= len(negatives):
                continue
            negative = negatives[depth]
            points.append(
                {
                    "task": task,
                    "prompt_id": row["prompt_id"],
                    "prompt": row["prompt"],
                    "negative": negative,
                }
            )
            added += 1
            if len(points) >= limit:
                break
        if added == 0:
            break
        depth += 1
    return points


def diagnose_point(
    model: Any,
    tokenizer: Any,
    point: Mapping[str, Any],
    *,
    max_length: int,
    matched_absolute_advantage: float,
) -> dict[str, Any]:
    negative = point["negative"]
    encoded = encode_prompt_completion(
        tokenizer,
        str(point["prompt"]),
        str(negative["completion"]),
        max_length,
    )
    device = next(model.parameters()).device
    model.zero_grad(set_to_none=True)
    stats = completion_stats(model, encoded, device)
    loss = matched_absolute_advantage * stats["seq_lp"].mean()
    loss.backward()
    squared_norm = torch.zeros((), dtype=torch.float64, device=device)
    parameter_count = 0
    gradient_parameter_count = 0
    for parameter in model.parameters():
        parameter_count += parameter.numel()
        if parameter.grad is None:
            continue
        gradient_parameter_count += parameter.numel()
        squared_norm += parameter.grad.detach().double().square().sum()
    gradient_norm = float(torch.sqrt(squared_norm).cpu())
    model.zero_grad(set_to_none=True)
    return {
        "task": point["task"],
        "prompt_id": point["prompt_id"],
        "negative_id": negative["negative_id"],
        "error_class": negative["error_class"],
        "verifier_score": float(negative["verifier_score"]),
        "response_chars": int(negative["response_chars"]),
        "response_tokens": int(stats["lengths"].item()),
        "mean_token_surprisal": float((-stats["seq_lp"]).item()),
        "mean_direct_logit_score_norm": float(stats["score"].item()),
        "matched_absolute_advantage": matched_absolute_advantage,
        "raw_full_parameter_gradient_norm": gradient_norm,
        "implemented_actor_gradient_norm": gradient_norm,
        "parameter_count": parameter_count,
        "gradient_parameter_count": gradient_parameter_count,
        "checkpoint_kind": "supplied_policy",
    }


def cmd_diagnose(
    config: Mapping[str, Any],
    work_dir: Path,
    *,
    model_path: str,
    adapter_path: str | None,
    requested_tasks: Sequence[str] | None = None,
    model_and_tokenizer: tuple[Any, Any] | None = None,
) -> dict[str, Any]:
    audit_path = work_dir / "qualification_audit.json"
    if not audit_path.is_file():
        raise RuntimeError("Run qualify before diagnose")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    tasks = list(requested_tasks or config["tasks"]["names"])
    unqualified = [
        task for task in tasks if not audit.get("tasks", {}).get(task, {}).get("passed", False)
    ]
    if unqualified:
        raise RuntimeError(f"Refusing to diagnose unqualified tasks: {unqualified}")

    diagnostic_config = config["diagnostics"]
    identity = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "config_hash": stable_config_hash(config),
        "tasks": tasks,
        "model_identity": model_identity(model_path, adapter_path),
        "banks": {task: sha256_file(bank_path(work_dir, task)) for task in tasks},
        "diagnostics": diagnostic_config,
    }
    identity["identity_hash"] = stable_hash(identity)
    state_path = work_dir / "diagnostics" / "diagnostic_identity.json"
    if state_path.is_file():
        existing_identity = json.loads(state_path.read_text(encoding="utf-8"))
        if existing_identity.get("identity_hash") != identity["identity_hash"]:
            raise RuntimeError("Diagnostic resume identity mismatch")
    else:
        atomic_json(state_path, identity)

    if model_and_tokenizer is None:
        model, tokenizer = load_diagnostic_model(
            model_path,
            adapter_path,
            dtype=str(diagnostic_config["dtype"]),
        )
    else:
        model, tokenizer = model_and_tokenizer
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        model.eval()

    point_limit = int(diagnostic_config["points_per_task"])
    seed = int(diagnostic_config["sampling_seed"])
    completed: dict[str, Any] = {}
    for task in tasks:
        output = work_dir / "diagnostics" / f"{task}.jsonl"
        existing_rows = read_jsonl(output) if output.is_file() else []
        existing_keys = {(str(row["prompt_id"]), str(row["negative_id"])) for row in existing_rows}
        points = select_diagnostic_points(
            read_jsonl(bank_path(work_dir, task)),
            task=task,
            limit=point_limit,
            seed=seed,
        )
        for point in points:
            key = (
                str(point["prompt_id"]),
                str(point["negative"]["negative_id"]),
            )
            if key in existing_keys:
                continue
            result = diagnose_point(
                model,
                tokenizer,
                point,
                max_length=int(diagnostic_config["max_length"]),
                matched_absolute_advantage=float(diagnostic_config["matched_absolute_advantage"]),
            )
            result.update(
                {
                    "experiment_id": EXPERIMENT_ID,
                    "diagnostic_seed": seed,
                    "model_identity_hash": stable_hash(identity["model_identity"]),
                }
            )
            append_jsonl(output, result)
            existing_keys.add(key)
        final_rows = read_jsonl(output)
        completed[task] = {
            "expected_points": len(points),
            "actual_points": len(final_rows),
            "complete": len(final_rows) == len(points),
            "path": str(output),
            "sha256": sha256_file(output),
        }
    manifest = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "identity_hash": identity["identity_hash"],
        "tasks": completed,
        "complete": all(item["complete"] for item in completed.values()),
        "scientific_status": "not_run",
        "claim_boundary": (
            "Occurrence/implemented-gradient diagnostic only; not causal identification, "
            "training effectiveness, convergence, or method ranking."
        ),
    }
    atomic_json(work_dir / "diagnostics" / "diagnostic_manifest.json", manifest)
    return manifest


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def assign_task_bins(
    task: str,
    points: Sequence[Mapping[str, Any]],
    *,
    bins: int,
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], int]]:
    if len(points) < bins:
        raise RuntimeError(f"{task} has {len(points)} points, fewer than {bins} bins")
    surprisals = np.asarray(
        [float(point["mean_token_surprisal"]) for point in points],
        dtype=float,
    )
    median = float(np.median(surprisals))
    q25, q75 = np.percentile(surprisals, [25.0, 75.0])
    scale = max(float(q75 - q25), 1.0e-12)
    order = np.argsort(surprisals, kind="stable")
    split_indices = np.array_split(order, bins)
    assignments: dict[tuple[str, str], int] = {}
    rows: list[dict[str, Any]] = []
    base_gradient: float | None = None
    base_advantage: float | None = None
    for bin_index, indices in enumerate(split_indices):
        selected = [points[int(index)] for index in indices]
        gradient = float(
            np.mean([float(point["implemented_actor_gradient_norm"]) for point in selected])
        )
        advantage = float(
            np.mean([float(point["matched_absolute_advantage"]) for point in selected])
        )
        if bin_index == 0:
            base_gradient = gradient
            base_advantage = advantage
        assert base_gradient is not None and base_advantage is not None
        if base_gradient <= 0 or base_advantage <= 0:
            raise RuntimeError(f"{task} has a non-positive normalization anchor")
        for point in selected:
            assignments[(str(point["prompt_id"]), str(point["negative_id"]))] = bin_index
        rows.append(
            {
                "task": task,
                "bin_index": bin_index,
                "point_count": len(selected),
                "prompt_count": len({str(point["prompt_id"]) for point in selected}),
                "mean_token_surprisal": float(
                    np.mean([float(point["mean_token_surprisal"]) for point in selected])
                ),
                "mean_within_task_robust_standardized_distance": float(
                    np.mean(
                        [
                            (float(point["mean_token_surprisal"]) - median) / scale
                            for point in selected
                        ]
                    )
                ),
                "relative_implemented_actor_gradient": gradient / base_gradient,
                "relative_matched_absolute_advantage": advantage / base_advantage,
                "mean_direct_logit_score_norm": float(
                    np.mean([float(point["mean_direct_logit_score_norm"]) for point in selected])
                ),
                "mean_response_tokens": float(
                    np.mean([float(point["response_tokens"]) for point in selected])
                ),
            }
        )
    return rows, assignments


def percentile_interval(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        return math.nan, math.nan
    low, high = np.percentile(np.asarray(values, dtype=float), [2.5, 97.5])
    return float(low), float(high)


def bootstrap_task_equal_aggregate(
    points_by_task: Mapping[str, Sequence[Mapping[str, Any]]],
    assignments: Mapping[str, Mapping[tuple[str, str], int]],
    *,
    bins: int,
    replicates: int,
    seed: int,
) -> dict[tuple[int, str], tuple[float, float]]:
    rng = random.Random(seed)
    metric_names = (
        "implemented_actor_gradient_norm",
        "matched_absolute_advantage",
    )
    samples: dict[tuple[int, str], list[float]] = defaultdict(list)
    grouped: dict[str, dict[str, list[Mapping[str, Any]]]] = {}
    for task, points in points_by_task.items():
        prompt_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for point in points:
            prompt_groups[str(point["prompt_id"])].append(point)
        grouped[task] = prompt_groups

    for _ in range(replicates):
        task_metrics: dict[str, dict[str, list[float]]] = {}
        valid = True
        for task, prompt_groups in grouped.items():
            prompt_ids = sorted(prompt_groups)
            selected: list[Mapping[str, Any]] = []
            for _prompt in prompt_ids:
                chosen = rng.choice(prompt_ids)
                selected.extend(prompt_groups[chosen])
            values_by_metric: dict[str, list[float]] = {}
            for metric in metric_names:
                bin_values: list[float] = []
                for bin_index in range(bins):
                    values = [
                        float(point[metric])
                        for point in selected
                        if assignments[task][(str(point["prompt_id"]), str(point["negative_id"]))]
                        == bin_index
                    ]
                    if not values:
                        valid = False
                        break
                    bin_values.append(float(np.mean(values)))
                if not valid or bin_values[0] <= 0:
                    break
                values_by_metric[metric] = [value / bin_values[0] for value in bin_values]
            if not valid:
                break
            task_metrics[task] = values_by_metric
        if not valid:
            continue
        for bin_index in range(bins):
            for metric in metric_names:
                samples[(bin_index, metric)].append(
                    float(
                        np.mean(
                            [task_metrics[task][metric][bin_index] for task in sorted(task_metrics)]
                        )
                    )
                )
    return {key: percentile_interval(values) for key, values in samples.items()}


def cmd_aggregate(
    config: Mapping[str, Any],
    work_dir: Path,
    *,
    requested_tasks: Sequence[str] | None = None,
) -> dict[str, Any]:
    tasks = list(requested_tasks or config["tasks"]["names"])
    points_by_task: dict[str, list[dict[str, Any]]] = {}
    per_task_rows: list[dict[str, Any]] = []
    assignments: dict[str, dict[tuple[str, str], int]] = {}
    bins = int(config["aggregation"]["quantile_bins"])
    for task in tasks:
        path = work_dir / "diagnostics" / f"{task}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"Missing diagnostic points for {task}: {path}")
        points = read_jsonl(path)
        points_by_task[task] = points
        rows, task_assignments = assign_task_bins(task, points, bins=bins)
        per_task_rows.extend(rows)
        assignments[task] = task_assignments
    write_csv(work_dir / "aggregate" / "per_task_bins.csv", per_task_rows)

    intervals = bootstrap_task_equal_aggregate(
        points_by_task,
        assignments,
        bins=bins,
        replicates=int(config["aggregation"]["bootstrap_replicates"]),
        seed=int(config["aggregation"]["bootstrap_seed"]),
    )
    aggregate_rows: list[dict[str, Any]] = []
    for bin_index in range(bins):
        rows = [row for row in per_task_rows if row["bin_index"] == bin_index]
        gradient_values = [float(row["relative_implemented_actor_gradient"]) for row in rows]
        advantage_values = [float(row["relative_matched_absolute_advantage"]) for row in rows]
        gradient_interval = intervals.get(
            (bin_index, "implemented_actor_gradient_norm"),
            (math.nan, math.nan),
        )
        advantage_interval = intervals.get(
            (bin_index, "matched_absolute_advantage"),
            (math.nan, math.nan),
        )
        aggregate_rows.append(
            {
                "bin_index": bin_index,
                "task_count": len(rows),
                "mean_within_task_robust_standardized_distance": float(
                    np.mean(
                        [
                            float(row["mean_within_task_robust_standardized_distance"])
                            for row in rows
                        ]
                    )
                ),
                "relative_implemented_actor_gradient": float(np.mean(gradient_values)),
                "relative_implemented_actor_gradient_ci95_low": gradient_interval[0],
                "relative_implemented_actor_gradient_ci95_high": gradient_interval[1],
                "relative_matched_absolute_advantage": float(np.mean(advantage_values)),
                "relative_matched_absolute_advantage_ci95_low": advantage_interval[0],
                "relative_matched_absolute_advantage_ci95_high": advantage_interval[1],
                "mean_direct_logit_score_norm": float(
                    np.mean([float(row["mean_direct_logit_score_norm"]) for row in rows])
                ),
                "mean_response_tokens": float(
                    np.mean([float(row["mean_response_tokens"]) for row in rows])
                ),
            }
        )
    write_csv(work_dir / "aggregate" / "task_equal_aggregate.csv", aggregate_rows)
    summary = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "tasks": tasks,
        "task_equal_weighting": True,
        "within_task_quantile_bins": bins,
        "normalization_anchor": "lowest_surprisal_bin_within_each_task",
        "remoteness_coordinate": "within_task_median_iqr_standardized_surprisal",
        "bootstrap_unit": "prompt_within_task",
        "bootstrap_replicates": int(config["aggregation"]["bootstrap_replicates"]),
        "categorical_theory_boundary": (
            "Selected-logit score is bounded; growth followed by saturation is allowed. "
            "Monotone unbounded explosion is not a pass criterion."
        ),
        "scientific_status": "not_run",
        "claim_boundary": (
            "Descriptive occurrence diagnostic only; D-U1 remains the categorical "
            "causal-identification environment."
        ),
        "outputs": {
            "per_task_bins": str(work_dir / "aggregate" / "per_task_bins.csv"),
            "task_equal_aggregate": str(work_dir / "aggregate" / "task_equal_aggregate.csv"),
        },
    }
    atomic_json(work_dir / "aggregate" / "aggregate_summary.json", summary)
    return summary


def cmd_all(
    config: Mapping[str, Any],
    work_dir: Path,
    *,
    force: bool,
    skip_download: bool,
    model_path: str | None,
    adapter_path: str | None,
) -> dict[str, Any]:
    bank_manifest = cmd_prepare(
        config,
        work_dir,
        force=force,
        skip_download=skip_download,
    )
    qualification = cmd_qualify(config, work_dir)
    if not qualification["passed"]:
        raise RuntimeError(
            "Bank qualification failed before model loading: "
            + ", ".join(qualification["failed_tasks"])
        )
    if not model_path:
        raise ValueError("--model-path is required after all qualification gates pass")
    diagnostic = cmd_diagnose(
        config,
        work_dir,
        model_path=model_path,
        adapter_path=adapter_path,
    )
    aggregate = cmd_aggregate(config, work_dir)
    return {
        "bank_manifest": bank_manifest,
        "qualification": qualification,
        "diagnostic": diagnostic,
        "aggregate": aggregate,
    }


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--work-dir", required=True)
    parser.add_argument(
        "--smoke-rows",
        type=int,
        help="Override rows per task for a non-scientific integration smoke.",
    )
    parser.add_argument(
        "--smoke-negatives",
        type=int,
        help="Override negatives per prompt for a non-scientific integration smoke.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--force", action="store_true")
    prepare.add_argument("--skip-download", action="store_true")

    subparsers.add_parser("qualify")

    diagnose = subparsers.add_parser("diagnose")
    diagnose.add_argument("--model-path", required=True)
    diagnose.add_argument("--adapter-path")
    diagnose.add_argument("--tasks", nargs="+")

    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--tasks", nargs="+")

    all_parser = subparsers.add_parser("all")
    all_parser.add_argument("--force", action="store_true")
    all_parser.add_argument("--skip-download", action="store_true")
    all_parser.add_argument("--model-path")
    all_parser.add_argument("--adapter-path")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = make_parser().parse_args(argv)
    config = with_smoke_overrides(
        load_config(args.config),
        rows=args.smoke_rows,
        negatives=args.smoke_negatives,
    )
    work_dir = validate_work_dir(args.work_dir)
    if args.command == "prepare":
        result = cmd_prepare(
            config,
            work_dir,
            force=bool(args.force),
            skip_download=bool(args.skip_download),
        )
    elif args.command == "qualify":
        result = cmd_qualify(config, work_dir)
    elif args.command == "diagnose":
        result = cmd_diagnose(
            config,
            work_dir,
            model_path=args.model_path,
            adapter_path=args.adapter_path,
            requested_tasks=args.tasks,
        )
    elif args.command == "aggregate":
        result = cmd_aggregate(
            config,
            work_dir,
            requested_tasks=args.tasks,
        )
    elif args.command == "all":
        result = cmd_all(
            config,
            work_dir,
            force=bool(args.force),
            skip_download=bool(args.skip_download),
            model_path=args.model_path,
            adapter_path=args.adapter_path,
        )
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
