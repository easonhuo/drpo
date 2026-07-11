#!/usr/bin/env python3
"""Large paired pilot sweep for paper taper families on the frozen E8 V2 bank.

This experiment is external-validity evidence. It keeps the V2 model-independent
bank, current-policy near/far reselection, base Qwen2.5-0.5B-Instruct initialization,
and the registered offline training protocol. It tunes only the active paper
families: reciprocal-linear, reciprocal-quadratic, and exponential. Global is
not rerun; its successful x1/32 initialization budget is used only as the frozen
calibration target.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

try:
    from drpo import countdown_e8_base_rl_replay as base_runner
    from drpo import countdown_qwen_arena_onefile as arena
except ImportError:  # pragma: no cover - direct execution from src/drpo
    import countdown_e8_base_rl_replay as base_runner  # type: ignore
    import countdown_qwen_arena_onefile as arena  # type: ignore

EXPERIMENT_ID = "EXT-C-E8-ORACLE-OFFLINE-V2-TAPER-SWEEP-0.5B-01"
VERSION = "0.1.0"
DEFAULT_SWEEP_CONFIG = "configs/countdown_e8_oracle_offline_v2_taper_sweep_0p5b.yaml"
DEFAULT_BASE_CONFIG = "configs/countdown_e8_base_rl_replay_0p5b.yaml"
METHODS = ("reciprocal_linear", "reciprocal_quadratic", "exponential")


@dataclass(frozen=True)
class Cell:
    method: str
    rho: float
    seed_offset: int

    @property
    def coefficient(self) -> float:
        return coefficient_from_rho(self.method, self.rho)

    @property
    def name(self) -> str:
        rho_tag = f"{self.rho:.5f}".rstrip("0").rstrip(".").replace(".", "p")
        return f"base_{self.method}_rho{rho_tag}_seed{self.seed_offset}"


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(target)


def _git_state(repo: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(repo), *args], text=True, stderr=subprocess.DEVNULL
        ).strip()

    try:
        return {
            "commit": run("rev-parse", "HEAD"),
            "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(run("status", "--porcelain")),
        }
    except Exception:  # noqa: BLE001
        return {"commit": "unknown", "branch": "unknown", "dirty": True}


def load_yaml(path: str | Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return value


def coefficient_from_rho(method: str, rho: float) -> float:
    if method not in METHODS:
        raise ValueError(f"Unsupported taper method: {method}")
    if not math.isfinite(rho) or not 0.0 < rho < 1.0:
        raise ValueError("rho must be finite and strictly between zero and one")
    if method in {"reciprocal_linear", "reciprocal_quadratic"}:
        return 1.0 / rho - 1.0
    return -math.log(rho)


def taper_weight(method: str, distance: torch.Tensor, coefficient: float) -> torch.Tensor:
    if method == "reciprocal_linear":
        return 1.0 / (1.0 + float(coefficient) * distance)
    if method == "reciprocal_quadratic":
        return 1.0 / (1.0 + float(coefficient) * distance.square())
    if method == "exponential":
        return torch.exp(-float(coefficient) * distance)
    raise ValueError(f"Unsupported taper method: {method}")


def normalized_distance(
    seq_lp: torch.Tensor, *, tau: float, surprisal_scale: float
) -> torch.Tensor:
    if not math.isfinite(tau) or tau < 0:
        raise ValueError("tau must be finite and non-negative")
    if not math.isfinite(surprisal_scale) or surprisal_scale <= 0:
        raise ValueError("surprisal_scale must be finite and positive")
    excess = torch.relu(-seq_lp.detach() - float(tau))
    return torch.sqrt(excess / float(surprisal_scale))


def validate_sweep_config(config: Mapping[str, Any]) -> None:
    if config.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("Sweep config experiment_id mismatch")
    if tuple(config["sweep"]["methods"]) != METHODS:
        raise ValueError("Active methods must be Linear, Quadratic, and Exp only")
    rhos = tuple(float(x) for x in config["sweep"]["rho_values"])
    if len(rhos) != 8 or len(set(rhos)) != len(rhos):
        raise ValueError("The frozen pilot grid requires eight unique rho values")
    if any(not 0.0 < rho < 1.0 for rho in rhos):
        raise ValueError("Every rho must be strictly between zero and one")
    offsets = tuple(int(x) for x in config["sweep"]["seed_offsets"])
    if offsets != (0, 1000, 2000):
        raise ValueError("The pilot tuning seed offsets are frozen to 0,1000,2000")
    if int(config["execution"]["required_gpu_count"]) != 8:
        raise ValueError("This one-click sweep is frozen to an eight-GPU pool")
    if float(config["calibration"]["reference_global_multiplier"]) != 1.0 / 32.0:
        raise ValueError("The frozen reference Global multiplier must be x1/32")
    if int(config["calibration"]["remoteness_prompt_rows"]) < 32:
        raise ValueError("Remoteness calibration requires at least 32 prompts")
    if int(config["calibration"]["gradient_prompt_rows"]) < 4:
        raise ValueError("Gradient calibration requires at least four prompts")


def build_cells(config: Mapping[str, Any]) -> tuple[Cell, ...]:
    validate_sweep_config(config)
    return tuple(
        Cell(method, float(rho), int(seed_offset))
        for method in METHODS
        for rho in config["sweep"]["rho_values"]
        for seed_offset in config["sweep"]["seed_offsets"]
    )


def _method_key(method: str, rho: float) -> str:
    return f"{method}|{rho:.12g}"


def _gradient_geometry(
    near_grads: Sequence[torch.Tensor | None],
    far_grads: Sequence[torch.Tensor | None],
) -> tuple[float, float, float]:
    near_sq = torch.zeros((), dtype=torch.float64)
    far_sq = torch.zeros((), dtype=torch.float64)
    dot = torch.zeros((), dtype=torch.float64)
    for near_grad, far_grad in zip(near_grads, far_grads):
        near_cpu = near_grad.detach().double().cpu() if near_grad is not None else None
        far_cpu = far_grad.detach().double().cpu() if far_grad is not None else None
        if near_cpu is not None:
            near_sq += near_cpu.square().sum()
        if far_cpu is not None:
            far_sq += far_cpu.square().sum()
        if near_cpu is not None and far_cpu is not None:
            dot += (near_cpu * far_cpu).sum()
    return float(near_sq), float(far_sq), float(dot)


def _balanced_rows(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    selected = arena.balanced_diagnostic_rows(rows, count)
    if len(selected) != count:
        raise RuntimeError(f"Requested {count} calibration rows, received {len(selected)}")
    return list(selected)


def _single_row_bank_batches(
    row: dict[str, Any], tokenizer: Any, max_length: int
) -> tuple[dict[str, torch.Tensor], int]:
    dataset = arena.OfflineDataset([row], tokenizer, max_length)
    packed = arena.make_offline_collator(tokenizer.pad_token_id)([dataset[0]])
    return packed, int(packed["bank_size"])


def calibrate(
    *,
    model_path: Path,
    bank_path: Path,
    global_calibration_path: Path,
    base_config_path: Path,
    sweep_config_path: Path,
    output_path: Path,
    repo: Path,
) -> dict[str, Any]:
    sweep_config = load_yaml(sweep_config_path)
    validate_sweep_config(sweep_config)
    base_config = load_yaml(base_config_path)
    calibration_cfg = sweep_config["calibration"]
    arena.seed_all(int(calibration_cfg["seed"]))

    global_calibration = json.loads(global_calibration_path.read_text())
    required_global = (
        "positive_rms_gradient_norm",
        "bank_uncontrolled_rms_gradient_norm",
        "bank_negative_scale",
        "bank_global_gamma",
    )
    missing = [key for key in required_global if key not in global_calibration]
    if missing:
        raise RuntimeError(f"Global calibration is missing keys: {missing}")
    reference_multiplier = float(calibration_cfg["reference_global_multiplier"])
    target_from_positive = (
        float(global_calibration["positive_rms_gradient_norm"])
        * float(global_calibration["bank_global_gamma"])
        * reference_multiplier
    )
    target_from_uncontrolled = (
        float(global_calibration["bank_uncontrolled_rms_gradient_norm"])
        * float(global_calibration["bank_negative_scale"])
        * float(global_calibration["bank_global_gamma"])
        * reference_multiplier
    )
    if not math.isclose(target_from_positive, target_from_uncontrolled, rel_tol=1e-5):
        raise RuntimeError("Global x1/32 target has inconsistent calibration identities")
    target_rms = target_from_positive

    tokenizer = arena.load_tokenizer(str(model_path))
    model = arena.load_model(
        str(model_path),
        adapter_path=None,
        trainable_adapter=True,
        load_in_4bit=bool(base_config["model"].get("load_in_4bit", False)),
        dtype=str(base_config["model"].get("dtype", "auto")),
        gradient_checkpointing=False,
        parameterization="lora",
    )
    model.eval()
    device = next(model.parameters()).device
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    rows = arena.read_jsonl(bank_path)
    remoteness_rows = _balanced_rows(rows, int(calibration_cfg["remoteness_prompt_rows"]))

    near_surprisals: list[float] = []
    far_surprisals: list[float] = []
    with torch.no_grad():
        for row in remoteness_rows:
            packed, bank_size = _single_row_bank_batches(
                row, tokenizer, int(base_config["model"]["max_length"])
            )
            bank_batch = arena.move_to_device(packed["bank"], device)
            bank_stats = arena.completion_stats(model, bank_batch)
            near, far, _, _ = arena.select_current_bank_extremes(bank_stats, 1, bank_size)
            near_surprisals.append(float(-near["seq_lp"].item()))
            far_surprisals.append(float(-far["seq_lp"].item()))

    near_median = float(np.median(near_surprisals))
    far_median = float(np.median(far_surprisals))
    tau = float(np.median(np.asarray(near_surprisals + far_surprisals, dtype=float)))
    scale = far_median - near_median
    minimum_scale = float(calibration_cfg["minimum_surprisal_scale"])
    if not math.isfinite(scale) or scale < minimum_scale:
        raise RuntimeError(
            f"Degenerate V2 near/far remoteness scale: {scale} < {minimum_scale}"
        )

    geometry_rows = remoteness_rows[: int(calibration_cfg["gradient_prompt_rows"])]
    geometries: list[dict[str, float]] = []
    for row in geometry_rows:
        packed, bank_size = _single_row_bank_batches(
            row, tokenizer, int(base_config["model"]["max_length"])
        )
        bank_batch = arena.move_to_device(packed["bank"], device)
        near_batch, far_batch, _, _ = arena.current_bank_training_batches(
            model, bank_batch, 1, bank_size
        )
        model.zero_grad(set_to_none=True)
        near_stats = arena.completion_stats(model, near_batch)
        near_lp = near_stats["seq_lp"].mean()
        near_grads = torch.autograd.grad(near_lp, trainable, allow_unused=True)
        near_surprisal = float(-near_lp.detach())

        model.zero_grad(set_to_none=True)
        far_stats = arena.completion_stats(model, far_batch)
        far_lp = far_stats["seq_lp"].mean()
        far_grads = torch.autograd.grad(far_lp, trainable, allow_unused=True)
        far_surprisal = float(-far_lp.detach())
        near_sq, far_sq, dot = _gradient_geometry(near_grads, far_grads)
        geometries.append(
            {
                "near_surprisal": near_surprisal,
                "far_surprisal": far_surprisal,
                "near_sq": near_sq,
                "far_sq": far_sq,
                "dot": dot,
            }
        )
        model.zero_grad(set_to_none=True)

    method_calibration: dict[str, Any] = {}
    for method in METHODS:
        for rho_value in sweep_config["sweep"]["rho_values"]:
            rho = float(rho_value)
            coefficient = coefficient_from_rho(method, rho)
            per_row_norms: list[float] = []
            near_weights: list[float] = []
            far_weights: list[float] = []
            for row in geometries:
                near_lp_tensor = torch.tensor([-row["near_surprisal"]], dtype=torch.float32)
                far_lp_tensor = torch.tensor([-row["far_surprisal"]], dtype=torch.float32)
                near_d = normalized_distance(near_lp_tensor, tau=tau, surprisal_scale=scale)
                far_d = normalized_distance(far_lp_tensor, tau=tau, surprisal_scale=scale)
                near_weight = float(taper_weight(method, near_d, coefficient).item())
                far_weight = float(taper_weight(method, far_d, coefficient).item())
                a = 0.5 * near_weight
                b = 0.5 * far_weight
                norm_sq = (
                    a * a * row["near_sq"]
                    + b * b * row["far_sq"]
                    + 2.0 * a * b * row["dot"]
                )
                per_row_norms.append(math.sqrt(max(norm_sq, 0.0)))
                near_weights.append(near_weight)
                far_weights.append(far_weight)
            unscaled_rms = float(np.sqrt(np.mean(np.square(per_row_norms))))
            if not math.isfinite(unscaled_rms) or unscaled_rms <= 0:
                raise RuntimeError(f"Invalid unscaled taper gradient RMS for {method}, rho={rho}")
            negative_scale = target_rms / unscaled_rms
            if not math.isfinite(negative_scale) or negative_scale <= 0:
                raise RuntimeError(f"Invalid calibrated negative scale for {method}, rho={rho}")
            method_calibration[_method_key(method, rho)] = {
                "method": method,
                "rho": rho,
                "coefficient": coefficient,
                "unscaled_weighted_negative_rms": unscaled_rms,
                "target_negative_rms": target_rms,
                "negative_scale": negative_scale,
                "mean_near_weight": float(np.mean(near_weights)),
                "mean_far_weight": float(np.mean(far_weights)),
                "mean_far_over_near_weight": float(
                    np.mean(np.asarray(far_weights) / np.maximum(near_weights, 1e-30))
                ),
            }

    result = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "runner_version": VERSION,
        "source": _git_state(repo),
        "identity": {
            "model_path": str(model_path),
            "bank_sha256": _sha256_file(bank_path),
            "global_calibration_sha256": _sha256_file(global_calibration_path),
            "base_config_sha256": _sha256_file(base_config_path),
            "sweep_config_sha256": _sha256_file(sweep_config_path),
        },
        "reference_global": {
            "multiplier": reference_multiplier,
            "target_negative_rms": target_rms,
            "positive_rms_gradient_norm": global_calibration["positive_rms_gradient_norm"],
            "bank_uncontrolled_rms_gradient_norm": global_calibration[
                "bank_uncontrolled_rms_gradient_norm"
            ],
            "bank_negative_scale": global_calibration["bank_negative_scale"],
            "bank_global_gamma": global_calibration["bank_global_gamma"],
            "rerun_in_sweep": False,
        },
        "remoteness": {
            "definition": "u=sqrt(relu(sequence_surprisal-tau)/scale)",
            "tau": tau,
            "scale": scale,
            "near_median_surprisal": near_median,
            "far_median_surprisal": far_median,
            "rows": len(remoteness_rows),
        },
        "gradient_rows": len(geometries),
        "methods": method_calibration,
        "task_metrics_used": False,
        "test_data_used": False,
        "frozen_before_training": True,
    }
    _atomic_json(output_path, result)
    return result


def _calibration_identity_matches(
    calibration: Mapping[str, Any],
    *,
    bank: Path,
    global_calibration: Path,
    base_config: Path,
    sweep_config: Path,
) -> bool:
    identity = calibration.get("identity", {})
    return (
        calibration.get("experiment_id") == EXPERIMENT_ID
        and identity.get("bank_sha256") == _sha256_file(bank)
        and identity.get("global_calibration_sha256") == _sha256_file(global_calibration)
        and identity.get("base_config_sha256") == _sha256_file(base_config)
        and identity.get("sweep_config_sha256") == _sha256_file(sweep_config)
    )


def _checkpoint_diagnostics(
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    *,
    method: str,
    coefficient: float,
    tau: float,
    scale: float,
    max_length: int,
    examples: int,
) -> dict[str, float]:
    selected = _balanced_rows(rows, min(examples, len(rows)))
    device = next(model.parameters()).device
    near_s: list[float] = []
    far_s: list[float] = []
    near_w: list[float] = []
    far_w: list[float] = []
    was_training = bool(model.training)
    model.eval()
    with torch.no_grad():
        for row in selected:
            packed, bank_size = _single_row_bank_batches(row, tokenizer, max_length)
            stats = arena.completion_stats(model, arena.move_to_device(packed["bank"], device))
            near, far, _, _ = arena.select_current_bank_extremes(stats, 1, bank_size)
            near_distance = normalized_distance(near["seq_lp"], tau=tau, surprisal_scale=scale)
            far_distance = normalized_distance(far["seq_lp"], tau=tau, surprisal_scale=scale)
            near_s.append(float(-near["seq_lp"].item()))
            far_s.append(float(-far["seq_lp"].item()))
            near_w.append(float(taper_weight(method, near_distance, coefficient).item()))
            far_w.append(float(taper_weight(method, far_distance, coefficient).item()))
    model.train(was_training)
    return {
        "bank_current_near_surprisal_mean": float(np.mean(near_s)),
        "bank_current_far_surprisal_mean": float(np.mean(far_s)),
        "bank_current_near_weight_mean": float(np.mean(near_w)),
        "bank_current_far_weight_mean": float(np.mean(far_w)),
        "bank_current_far_over_near_weight_mean": float(
            np.mean(np.asarray(far_w) / np.maximum(near_w, 1e-30))
        ),
    }


def _run_identity(
    *,
    repo: Path,
    model_path: Path,
    bank: Path,
    val: Path,
    test: Path,
    base_config: Path,
    sweep_config: Path,
    calibration: Path,
    cell: Cell,
) -> dict[str, Any]:
    return {
        "experiment_id": EXPERIMENT_ID,
        "runner_version": VERSION,
        "source": _git_state(repo),
        "model_path": str(model_path),
        "bank_sha256": _sha256_file(bank),
        "val_sha256": _sha256_file(val),
        "test_sha256": _sha256_file(test),
        "base_config_sha256": _sha256_file(base_config),
        "sweep_config_sha256": _sha256_file(sweep_config),
        "calibration_sha256": _sha256_file(calibration),
        "cell": {
            "name": cell.name,
            "method": cell.method,
            "rho": cell.rho,
            "coefficient": cell.coefficient,
            "seed_offset": cell.seed_offset,
        },
    }


def _identity_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return json.dumps(left, sort_keys=True) == json.dumps(right, sort_keys=True)


def train_cell(
    *,
    cell: Cell,
    model_path: Path,
    bank: Path,
    val: Path,
    test: Path,
    base_config_path: Path,
    sweep_config_path: Path,
    calibration_path: Path,
    output_dir: Path,
    repo: Path,
) -> dict[str, Any]:
    base_config = load_yaml(base_config_path)
    sweep_config = load_yaml(sweep_config_path)
    validate_sweep_config(sweep_config)
    calibration = json.loads(calibration_path.read_text())
    calibration_row = calibration["methods"][_method_key(cell.method, cell.rho)]
    tau = float(calibration["remoteness"]["tau"])
    scale = float(calibration["remoteness"]["scale"])
    negative_scale = float(calibration_row["negative_scale"])
    coefficient = float(calibration_row["coefficient"])
    train_cfg = base_config["offline_training"]
    model_cfg = base_config["model"]
    eval_cfg = base_config["evaluation"]
    seed = int(train_cfg["seed"]) + cell.seed_offset
    arena.seed_all(seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    identity = _run_identity(
        repo=repo,
        model_path=model_path,
        bank=bank,
        val=val,
        test=test,
        base_config=base_config_path,
        sweep_config=sweep_config_path,
        calibration=calibration_path,
        cell=cell,
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
    dataset = arena.OfflineDataset(train_rows, tokenizer, int(model_cfg["max_length"]))
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=int(train_cfg["micro_batch"]),
        shuffle=True,
        generator=generator,
        collate_fn=arena.make_offline_collator(tokenizer.pad_token_id),
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
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=float(train_cfg["learning_rate"]), weight_decay=0.01
    )
    scheduler = arena.get_cosine_schedule_with_warmup(
        optimizer,
        max(1, int(int(train_cfg["steps"]) * float(train_cfg["warmup_ratio"]))),
        int(train_cfg["steps"]),
    )

    best_dir = output_dir / "best_adapter"
    terminal_dir = output_dir / "terminal_adapter"
    last_finite_dir = output_dir / "last_finite_adapter"
    checkpoint_records: list[dict[str, Any]] = []
    metrics_rows: list[dict[str, Any]] = []
    diagnostics_path = output_dir / "dynamic_diagnostics.jsonl"
    best_value = -float("inf")
    best_step = 0
    stale_checks = 0
    terminal_step: int | None = None
    numerical_failure: str | None = None
    stop_reason = "max_steps"
    last_finite_step = 0

    known_structures = {
        row.get("oracle_structure") or arena.expression_structure(row["oracle"])
        for row in train_rows
    }

    def evaluate(step: int) -> dict[str, Any]:
        metrics = arena.evaluate_rows(
            model,
            tokenizer,
            val_rows[: int(eval_cfg["examples"])],
            int(eval_cfg["batch_size"]),
            int(model_cfg["max_new_tokens"]),
            int(eval_cfg["pass_ks"][0]),
            int(eval_cfg["seed"]),
            known_structures,
        )
        diagnostics = _checkpoint_diagnostics(
            model,
            tokenizer,
            train_rows,
            method=cell.method,
            coefficient=coefficient,
            tau=tau,
            scale=scale,
            max_length=int(model_cfg["max_length"]),
            examples=int(train_cfg["diagnostic_examples"]),
        )
        row = {
            "step": step,
            "method": cell.method,
            "rho": cell.rho,
            "coefficient": coefficient,
            "negative_scale": negative_scale,
            **diagnostics,
            **metrics,
        }
        with diagnostics_path.open("a") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        metrics_rows.append(row)
        return row

    initial = evaluate(0)
    best_value = float(initial[str(train_cfg["selection_metric"])])
    checkpoint_records.append(
        arena.save_local_model_checkpoint(model, tokenizer, best_dir, "best", 0)
    )
    model.train()

    for update_step in range(1, int(train_cfg["steps"]) + 1):
        optimizer.zero_grad(set_to_none=True)
        accum = {
            "loss": 0.0,
            "positive_lp": 0.0,
            "negative_lp": 0.0,
            "near_weight": 0.0,
            "far_weight": 0.0,
        }
        abort_update = False
        for _ in range(int(train_cfg["gradient_accumulation"])):
            try:
                packed = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                packed = next(iterator)
            positive = arena.completion_stats(
                model, arena.move_to_device(packed["positive"], device)
            )
            positive_lp = positive["seq_lp"].mean()
            bank_size = int(packed.get("bank_size", 0))
            if bank_size < 2 or "bank" not in packed:
                raise RuntimeError("V2 taper training requires a populated negative bank")
            near_batch, far_batch, _, _ = arena.current_bank_training_batches(
                model,
                arena.move_to_device(packed["bank"], device),
                packed["positive"]["input_ids"].shape[0],
                bank_size,
            )
            near = arena.completion_stats(model, near_batch)
            far = arena.completion_stats(model, far_batch)
            near_distance = normalized_distance(
                near["seq_lp"], tau=tau, surprisal_scale=scale
            )
            far_distance = normalized_distance(
                far["seq_lp"], tau=tau, surprisal_scale=scale
            )
            near_weights = taper_weight(cell.method, near_distance, coefficient).detach()
            far_weights = taper_weight(cell.method, far_distance, coefficient).detach()
            negative_lp = 0.5 * (near_weights * near["seq_lp"]).mean() + 0.5 * (
                far_weights * far["seq_lp"]
            ).mean()
            raw_loss = -(positive_lp - negative_scale * negative_lp)
            if not bool(torch.isfinite(raw_loss)):
                numerical_failure = f"nonfinite_loss_at_step_{update_step}"
                stop_reason = numerical_failure
                abort_update = True
                break
            (raw_loss / int(train_cfg["gradient_accumulation"])).backward()
            divisor = float(train_cfg["gradient_accumulation"])
            accum["loss"] += float(raw_loss.detach()) / divisor
            accum["positive_lp"] += float(positive_lp.detach()) / divisor
            accum["negative_lp"] += float(negative_lp.detach()) / divisor
            accum["near_weight"] += float(near_weights.mean()) / divisor
            accum["far_weight"] += float(far_weights.mean()) / divisor
        if abort_update:
            break
        grad_norm = torch.nn.utils.clip_grad_norm_(
            trainable, float(train_cfg["maximum_gradient_norm"])
        )
        if not bool(torch.isfinite(grad_norm)):
            numerical_failure = f"nonfinite_gradient_at_step_{update_step}"
            stop_reason = numerical_failure
            break
        if not arena.optimizer_step_with_last_finite_guard(optimizer, trainable):
            numerical_failure = f"nonfinite_parameters_at_step_{update_step}"
            stop_reason = numerical_failure
            break
        scheduler.step()
        terminal_step = update_step
        last_finite_step = update_step
        if update_step % int(train_cfg["log_every"]) == 0:
            print(json.dumps({"cell": cell.name, "step": update_step, **accum}), flush=True)
        if update_step % int(train_cfg["eval_every"]) == 0 or update_step == int(
            train_cfg["steps"]
        ):
            row = evaluate(update_step)
            value = float(row[str(train_cfg["selection_metric"])])
            if value > best_value + float(train_cfg["early_stop_delta"]):
                best_value = value
                best_step = update_step
                stale_checks = 0
                checkpoint_records = [
                    record for record in checkpoint_records if record["kind"] != "best"
                ]
                checkpoint_records.append(
                    arena.save_local_model_checkpoint(
                        model, tokenizer, best_dir, "best", update_step
                    )
                )
            else:
                stale_checks += 1
            model.train()
            if (
                update_step >= int(train_cfg["min_steps"])
                and stale_checks >= int(train_cfg["early_stop_patience"])
            ):
                stop_reason = "early_stop_patience"
                break

    if numerical_failure:
        checkpoint_records.append(
            arena.save_local_model_checkpoint(
                model, tokenizer, last_finite_dir, "last_finite", last_finite_step
            )
        )
        terminal_checkpoint = last_finite_dir
        terminal_kind = "last_finite"
    else:
        checkpoint_records.append(
            arena.save_local_model_checkpoint(
                model,
                tokenizer,
                terminal_dir,
                "terminal",
                int(terminal_step or 0),
            )
        )
        terminal_checkpoint = terminal_dir
        terminal_kind = "terminal"

    with (output_dir / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics_rows[0].keys()))
        writer.writeheader()
        writer.writerows(arena.csv_safe_row(row) for row in metrics_rows)

    common_eval_seed_offset = cell.seed_offset
    data_paths = {
        "train": bank,
        "validation": val,
        "test": test,
        "split_manifest": bank,
    }
    best_evaluation = base_runner.evaluate_adapter_checkpoint(
        model_path,
        best_dir,
        data_paths,
        base_config,
        seed_offset=common_eval_seed_offset,
    )
    terminal_evaluation = base_runner.evaluate_adapter_checkpoint(
        model_path,
        terminal_checkpoint,
        data_paths,
        base_config,
        seed_offset=common_eval_seed_offset,
    )
    manifest = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "runner_version": VERSION,
        "result_status": sweep_config["result_status"],
        "cell": cell.name,
        "method": cell.method,
        "rho": cell.rho,
        "coefficient": coefficient,
        "negative_scale": negative_scale,
        "budget_target": calibration_row["target_negative_rms"],
        "seed": seed,
        "seed_offset": cell.seed_offset,
        "best_step": best_step,
        "best_value": best_value,
        "terminal_step": terminal_step,
        "last_finite_step": last_finite_step,
        "terminal_checkpoint_kind": terminal_kind,
        "stop_reason": stop_reason,
        "numerical_failure": numerical_failure,
        "checkpoint_policy": "server-local LoRA only",
        "checkpoints": checkpoint_records,
        "run_identity": identity,
    }
    _atomic_json(output_dir / "manifest.json", manifest)
    summary = {
        **manifest,
        "best_evaluation": best_evaluation,
        "terminal_evaluation": terminal_evaluation,
        "reporting_separation": {
            "task_performance": "reported_in_best_and_terminal_evaluation",
            "support_or_structure_boundary": (
                "valid_rate_diagnostic_only_not_formal_support_audit"
            ),
            "nan_inf_numerical_failure": numerical_failure,
        },
    }
    _atomic_json(summary_path, summary)
    return summary


def _worker_command(args: argparse.Namespace, cell: Cell, output_dir: Path) -> list[str]:
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
        "--test",
        args.test,
        "--base_config",
        args.base_config,
        "--sweep_config",
        args.sweep_config,
        "--calibration",
        args.calibration,
        "--output_dir",
        str(output_dir),
        "--method",
        cell.method,
        "--rho",
        str(cell.rho),
        "--seed_offset",
        str(cell.seed_offset),
    ]


def run_sweep(args: argparse.Namespace) -> int:
    repo = Path(__file__).resolve().parents[2]
    sweep_config_path = Path(args.sweep_config).resolve()
    base_config_path = Path(args.base_config).resolve()
    config = load_yaml(sweep_config_path)
    validate_sweep_config(config)
    model = Path(args.model_path).resolve()
    bank = Path(args.bank).resolve()
    val = Path(args.val).resolve()
    test = Path(args.test).resolve()
    global_calibration = Path(args.global_calibration).resolve()
    work_dir = Path(args.work_dir).resolve()
    calibration_path = work_dir / "calibration" / "taper_budget_calibration.json"
    args.calibration = str(calibration_path)
    logs_dir = work_dir / "logs"
    methods_dir = work_dir / "methods"
    for path in (
        model,
        bank,
        val,
        test,
        global_calibration,
        base_config_path,
        sweep_config_path,
    ):
        if not path.exists():
            raise SystemExit(f"Missing required input: {path}")
    work_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    methods_dir.mkdir(parents=True, exist_ok=True)

    gpu_pool = [item.strip() for item in args.gpus.split(",") if item.strip()]
    required = int(config["execution"]["required_gpu_count"])
    if len(gpu_pool) != required or len(set(gpu_pool)) != required:
        raise SystemExit(f"Expected exactly {required} unique GPUs, received {gpu_pool}")

    if calibration_path.exists():
        calibration = json.loads(calibration_path.read_text())
        if not _calibration_identity_matches(
            calibration,
            bank=bank,
            global_calibration=global_calibration,
            base_config=base_config_path,
            sweep_config=sweep_config_path,
        ):
            raise RuntimeError("Existing taper calibration identity is stale; use a new work_dir")
    else:
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = args.calibration_gpu
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "calibrate",
            "--model_path",
            str(model),
            "--bank",
            str(bank),
            "--global_calibration",
            str(global_calibration),
            "--base_config",
            str(base_config_path),
            "--sweep_config",
            str(sweep_config_path),
            "--output",
            str(calibration_path),
        ]
        result = subprocess.run(command, cwd=repo, env=env, check=False)
        if result.returncode != 0:
            return int(result.returncode)

    cells = build_cells(config)
    plan = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "runner_version": VERSION,
        "source": _git_state(repo),
        "gpu_pool": gpu_pool,
        "cell_count": len(cells),
        "methods": list(METHODS),
        "rho_values": [float(x) for x in config["sweep"]["rho_values"]],
        "seed_offsets": [int(x) for x in config["sweep"]["seed_offsets"]],
        "global_rerun": False,
        "cells": [cell.__dict__ | {"name": cell.name} for cell in cells],
    }
    _atomic_json(work_dir / "SWEEP_PLAN.json", plan)

    pending: list[Cell] = []
    for cell in cells:
        summary = methods_dir / cell.name / "summary.json"
        if not summary.exists():
            pending.append(cell)
            continue
        current_identity = _run_identity(
            repo=repo,
            model_path=model,
            bank=bank,
            val=val,
            test=test,
            base_config=base_config_path,
            sweep_config=sweep_config_path,
            calibration=calibration_path,
            cell=cell,
        )
        stored = json.loads(summary.read_text()).get("run_identity", {})
        if not _identity_equal(stored, current_identity):
            raise RuntimeError(f"Stale completed cell {cell.name}; use a new work_dir")

    results: dict[str, Any] = {}
    lock = threading.Lock()
    available = list(gpu_pool)
    available_lock = threading.Lock()

    def take_gpu() -> str:
        with available_lock:
            return available.pop(0)

    def give_gpu(gpu: str) -> None:
        with available_lock:
            available.append(gpu)

    def run_one(cell: Cell) -> tuple[str, int, str]:
        gpu = take_gpu()
        log_path = logs_dir / f"{cell.name}.log"
        try:
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = gpu
            command = _worker_command(args, cell, methods_dir / cell.name)
            with log_path.open("w") as handle:
                handle.write(f"GPU={gpu}\nCOMMAND={' '.join(command)}\n")
                handle.flush()
                process = subprocess.Popen(
                    command,
                    cwd=repo,
                    env=env,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                returncode = process.wait()
            return cell.name, int(returncode), str(log_path)
        finally:
            give_gpu(gpu)

    with ThreadPoolExecutor(max_workers=len(gpu_pool)) as executor:
        futures = {executor.submit(run_one, cell): cell for cell in pending}
        for future in as_completed(futures):
            name, returncode, log_path = future.result()
            with lock:
                results[name] = {
                    "returncode": returncode,
                    "status": "OK" if returncode == 0 else "FAIL",
                    "log": log_path,
                }
                status = {
                    "experiment_id": EXPERIMENT_ID,
                    "expected_cells": len(cells),
                    "completed_this_invocation": len(results),
                    "remaining_after_invocation": len(pending) - len(results),
                    "results": dict(sorted(results.items())),
                }
                _atomic_json(work_dir / "SWEEP_STATUS.json", status)
            print(f"[{name}] returncode={returncode} log={log_path}", flush=True)

    all_summaries = list(methods_dir.glob("*/summary.json"))
    failed = sorted(name for name, result in results.items() if result["returncode"] != 0)
    complete = len(all_summaries) == len(cells) and not failed
    final = {
        "experiment_id": EXPERIMENT_ID,
        "runner_version": VERSION,
        "result_status": config["result_status"],
        "expected_cells": len(cells),
        "summary_count": len(all_summaries),
        "failed_cells": failed,
        "all_expected_cells_present": complete,
        "source": _git_state(repo),
        "calibration_sha256": _sha256_file(calibration_path),
    }
    _atomic_json(work_dir / "SWEEP_COMPLETE.json", final)
    return 0 if complete else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="E8 V2 paper taper family sweep")
    parser.add_argument("--version", action="version", version=VERSION)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run")
    run.add_argument("--model_path", required=True)
    run.add_argument("--work_dir", required=True)
    run.add_argument("--bank", required=True)
    run.add_argument("--val", required=True)
    run.add_argument("--test", required=True)
    run.add_argument("--global_calibration", required=True)
    run.add_argument("--base_config", default=DEFAULT_BASE_CONFIG)
    run.add_argument("--sweep_config", default=DEFAULT_SWEEP_CONFIG)
    run.add_argument("--calibration_gpu", default="0")
    run.add_argument("--gpus", default="0,1,2,3,4,5,6,7")

    calibration = sub.add_parser("calibrate")
    calibration.add_argument("--model_path", required=True)
    calibration.add_argument("--bank", required=True)
    calibration.add_argument("--global_calibration", required=True)
    calibration.add_argument("--base_config", required=True)
    calibration.add_argument("--sweep_config", required=True)
    calibration.add_argument("--output", required=True)

    worker = sub.add_parser("worker")
    worker.add_argument("--model_path", required=True)
    worker.add_argument("--bank", required=True)
    worker.add_argument("--val", required=True)
    worker.add_argument("--test", required=True)
    worker.add_argument("--base_config", required=True)
    worker.add_argument("--sweep_config", required=True)
    worker.add_argument("--calibration", required=True)
    worker.add_argument("--output_dir", required=True)
    worker.add_argument("--method", choices=METHODS, required=True)
    worker.add_argument("--rho", type=float, required=True)
    worker.add_argument("--seed_offset", type=int, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    repo = Path(__file__).resolve().parents[2]
    if args.command == "run":
        return run_sweep(args)
    if args.command == "calibrate":
        calibrate(
            model_path=Path(args.model_path).resolve(),
            bank_path=Path(args.bank).resolve(),
            global_calibration_path=Path(args.global_calibration).resolve(),
            base_config_path=Path(args.base_config).resolve(),
            sweep_config_path=Path(args.sweep_config).resolve(),
            output_path=Path(args.output).resolve(),
            repo=repo,
        )
        return 0
    if args.command == "worker":
        train_cell(
            cell=Cell(args.method, float(args.rho), int(args.seed_offset)),
            model_path=Path(args.model_path).resolve(),
            bank=Path(args.bank).resolve(),
            val=Path(args.val).resolve(),
            test=Path(args.test).resolve(),
            base_config_path=Path(args.base_config).resolve(),
            sweep_config_path=Path(args.sweep_config).resolve(),
            calibration_path=Path(args.calibration).resolve(),
            output_dir=Path(args.output_dir).resolve(),
            repo=repo,
        )
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
