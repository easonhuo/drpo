#!/usr/bin/env python3
"""Paper-aligned Countdown E8 round-1 lambda scan.

Every unique negative completion participates in the loss.  For each completion,
``D`` is the detached mean-token surprisal produced by ``arena.completion_stats``.
The only negative weight is

    alpha * exp(-lambda * relu((D - tau) / scale_c)).

Round 1 freezes ``alpha=1`` for taper cells, derives ``tau`` and ``scale_c`` from
a frozen pre-training bank sample, and scans only ``lambda``.  No extra square,
current-near/current-far training classes, gradient-budget matching, hidden
negative scale, or weight-sum normalization is permitted.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset

try:
    from drpo import countdown_qwen_arena_onefile as arena
except ImportError:  # pragma: no cover - direct source execution
    import countdown_qwen_arena_onefile as arena  # type: ignore

EXPERIMENT_ID = "EXT-C-E8-ORACLE-OFFLINE-V2-PAPER-ALIGNED-LAMBDA-ROUND1-0.5B-01"
VERSION = "1.0.0-paper-aligned-linear-excess-surprisal"
DEFAULT_GRID_CONFIG = "configs/countdown_e8_paper_aligned_lambda_round1_0p5b.yaml"
DEFAULT_BASE_CONFIG = "configs/countdown_e8_base_rl_replay_0p5b.yaml"
FIXED_ALPHA = 1.0
LAMBDA_VALUES = (
    0.105360516,  # 90% retention at one normalized excess-surprisal unit
    0.287682072,  # 75%
    0.693147181,  # 50%
    1.386294361,  # 25%
    2.302585093,  # 10%
)
SEED_OFFSETS = (4000, 5000, 6000)
EXPECTED_POINTS = 6  # Positive-only plus five lambda values.
EXPECTED_CELLS = 18
FORBIDDEN_CONFIG_KEYS = {
    "rho",
    "rho_values",
    "c_values",
    "reference_distance",
    "negative_scale",
    "bank_negative_scale",
    "global_gamma",
    "gradient_rms_matching",
    "weight_sum_normalization",
}


@dataclass(frozen=True)
class Cell:
    alpha: float
    lambda_value: float
    seed_offset: int

    @property
    def method(self) -> str:
        return "positive_only" if self.alpha == 0.0 else "paper_aligned_exp"

    @property
    def name(self) -> str:
        return (
            f"base_{self.method}_alpha{_number_tag(self.alpha)}_"
            f"lambda{_number_tag(self.lambda_value)}_seed{self.seed_offset}"
        )


def _number_tag(value: float) -> str:
    return f"{value:.9g}".replace("-", "m").replace(".", "p")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(target)


def load_yaml(path: str | Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return value


def _walk_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, Mapping):
        for key, nested in value.items():
            keys.add(str(key))
            keys.update(_walk_keys(nested))
    elif isinstance(value, list):
        for nested in value:
            keys.update(_walk_keys(nested))
    return keys


def validate_grid_config(config: Mapping[str, Any]) -> None:
    if config.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("Paper-aligned E8 lambda experiment_id mismatch")
    if config.get("result_status") != "pilot":
        raise ValueError("Round 1 must remain a development pilot")
    if config.get("registration_state") != "registered_pilot":
        raise ValueError("Round 1 must remain a registered pilot")
    present = sorted(FORBIDDEN_CONFIG_KEYS & _walk_keys(config))
    if present:
        raise ValueError("Forbidden legacy/scaling coordinates: " + ", ".join(present))

    bank = config.get("bank", {})
    if bank.get("use_all_unique_negatives") is not True:
        raise ValueError("Every unique negative must participate")
    if bank.get("explicit_near_far_training_classes") is not False:
        raise ValueError("Explicit near/far training classes are forbidden")
    if bank.get("extreme_selection_forbidden") is not True:
        raise ValueError("Current-bank extreme selection must be forbidden")

    remoteness = config.get("remoteness", {})
    expected_formula = "alpha*exp(-lambda*relu((D-tau)/scale_c))"
    if remoteness.get("D_definition") != "negative_mean_completion_token_log_probability":
        raise ValueError("D must be mean-token surprisal")
    if remoteness.get("weight") != expected_formula:
        raise ValueError(f"Weight must be exactly {expected_formula}")
    if remoteness.get("extra_square_forbidden") is not True:
        raise ValueError("The extra surprisal square must be forbidden")
    if remoteness.get("eos_included_in_completion_mean") is not True:
        raise ValueError("Round 1 freezes EOS inclusion in mean-token surprisal")
    if remoteness.get("detached") is not True:
        raise ValueError("Remoteness weights must be stop-gradient")

    calibration = config.get("calibration", {})
    if calibration.get("tau_rule") != "frozen_pretraining_bank_sample_median_surprisal":
        raise ValueError("tau rule changed")
    if calibration.get("scale_rule") != (
        "frozen_pretraining_upper_half_median_minus_lower_half_median"
    ):
        raise ValueError("surprisal scale rule changed")
    if int(calibration.get("prompt_rows", -1)) != 256:
        raise ValueError("Round-1 calibration requires 256 prompts")
    if float(calibration.get("minimum_surprisal_scale", 0.0)) <= 0.0:
        raise ValueError("minimum_surprisal_scale must be positive")
    if not 0.0 < float(calibration.get("minimum_active_fraction", 0.0)) < 1.0:
        raise ValueError("minimum_active_fraction must be in (0,1)")

    sweep = config.get("sweep", {})
    if not math.isclose(
        float(sweep.get("fixed_alpha", -1.0)), FIXED_ALPHA, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("Round 1 fixes taper alpha=1")
    lambdas = tuple(float(value) for value in sweep.get("lambda_values", ()))
    offsets = tuple(int(value) for value in sweep.get("seed_offsets", ()))
    if lambdas != LAMBDA_VALUES:
        raise ValueError(f"lambda grid changed: {lambdas}")
    if offsets != SEED_OFFSETS:
        raise ValueError(f"development seed offsets changed: {offsets}")
    if int(sweep.get("unique_parameter_points", -1)) != EXPECTED_POINTS:
        raise ValueError("Round 1 requires six unique parameter points")
    if int(sweep.get("cells", -1)) != EXPECTED_CELLS:
        raise ValueError("Round 1 requires eighteen cells")
    if sweep.get("positive_only_included") is not True:
        raise ValueError("Positive-only paired confirmation is required")
    if sweep.get("global_rerun") is not False:
        raise ValueError("Global must be reused, not rerun, in round 1")

    training = config.get("training", {})
    if int(training.get("steps", -1)) != 1200:
        raise ValueError("The fixed development horizon is 1200 steps")
    if training.get("early_stop") is not False:
        raise ValueError("Early stopping is forbidden")
    if int(training.get("eval_every", -1)) != 100:
        raise ValueError("Greedy/Pass@8 cadence must remain 100")
    if int(training.get("pass64_every", -1)) != 200:
        raise ValueError("Pass@64 cadence must remain 200")
    if training.get("denominator") != "unique_negative_count_per_prompt":
        raise ValueError("Loss denominator must be unique negative count per prompt")
    if training.get("normalize_by_weight_sum") is not False:
        raise ValueError("Weight-sum normalization is forbidden")
    if training.get("hidden_negative_scale") is not False:
        raise ValueError("Hidden negative scaling is forbidden")
    if training.get("gradient_budget_matching") is not False:
        raise ValueError("Gradient-budget matching is forbidden")

    execution = config.get("execution", {})
    if int(execution.get("default_gpu_slots", -1)) != 2:
        raise ValueError("Round 1 defaults to exactly two auto-selected GPU slots")
    if int(execution.get("parallel_cells_per_gpu", -1)) != 1:
        raise ValueError("Only one process per selected GPU is permitted")
    if execution.get("identity_checked_resume") is not True:
        raise ValueError("Identity-checked resume is required")

    evaluation = config.get("evaluation", {})
    if evaluation.get("validation_only_during_tuning") is not True:
        raise ValueError("Round-1 tuning must remain validation-only")
    if evaluation.get("test_access_forbidden") is not True:
        raise ValueError("Test access must remain forbidden")


def parameter_points(config: Mapping[str, Any]) -> tuple[tuple[float, float], ...]:
    validate_grid_config(config)
    points = [(0.0, 0.0)] + [(FIXED_ALPHA, value) for value in LAMBDA_VALUES]
    if len(points) != EXPECTED_POINTS or len(set(points)) != EXPECTED_POINTS:
        raise AssertionError("Round-1 parameter grid must contain six unique points")
    return tuple(points)


def build_cells(config: Mapping[str, Any]) -> tuple[Cell, ...]:
    cells = tuple(
        Cell(alpha=alpha, lambda_value=lambda_value, seed_offset=seed_offset)
        for alpha, lambda_value in parameter_points(config)
        for seed_offset in SEED_OFFSETS
    )
    if len(cells) != EXPECTED_CELLS or len({cell.name for cell in cells}) != EXPECTED_CELLS:
        raise AssertionError("Round 1 must produce eighteen unique cells")
    return cells


def _expression_from_bank_item(item: Any) -> str:
    if isinstance(item, Mapping):
        if "expression" not in item:
            raise ValueError("Negative-bank mapping has no expression field")
        return str(item["expression"])
    return str(item)


def unique_negative_expressions(row: Mapping[str, Any]) -> list[str]:
    """Return first-occurrence unique negatives under the existing cleaner."""
    unique: list[str] = []
    seen: set[str] = set()
    for item in row.get("negative_bank", []):
        cleaned = arena.clean_expression(_expression_from_bank_item(item))
        if cleaned in seen:
            continue
        seen.add(cleaned)
        unique.append(cleaned)
    if not unique:
        raise ValueError(f"Row {row.get('id', '<unknown>')} has no unique negatives")
    return unique


class ContinuousUniqueBankDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], tokenizer: Any, max_length: int):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        negatives = unique_negative_expressions(row)
        return {
            "positive": arena.encode_prompt_completion(
                self.tokenizer, row["prompt"], row["positive"], self.max_length
            ),
            "bank": [
                arena.encode_prompt_completion(
                    self.tokenizer, row["prompt"], expression, self.max_length
                )
                for expression in negatives
            ],
            "unique_count": len(negatives),
            "raw_bank_count": len(row.get("negative_bank", [])),
        }


def make_continuous_unique_bank_collator(pad_id: int):
    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        positives = arena.pad_encoded([item["positive"] for item in batch], pad_id)
        flattened = [negative for item in batch for negative in item["bank"]]
        if not flattened:
            raise ValueError("Paper-aligned lambda batch has no unique negatives")
        row_index = [
            row for row, item in enumerate(batch) for _ in range(int(item["unique_count"]))
        ]
        counts = [int(item["unique_count"]) for item in batch]
        return {
            "positive": positives,
            "bank": arena.pad_encoded(flattened, pad_id),
            "bank_row_index": torch.tensor(row_index, dtype=torch.long),
            "unique_counts": torch.tensor(counts, dtype=torch.long),
            "raw_bank_counts": torch.tensor(
                [int(item["raw_bank_count"]) for item in batch], dtype=torch.long
            ),
        }

    return collate


def mean_token_surprisal(seq_lp: torch.Tensor) -> torch.Tensor:
    """Return detached mean-token surprisal D=-seq_lp."""
    return (-seq_lp.detach()).clamp_min(0.0)


def normalized_excess_surprisal(
    seq_lp: torch.Tensor, *, tau: float, scale_c: float
) -> torch.Tensor:
    if not math.isfinite(tau) or tau < 0.0:
        raise ValueError("tau must be finite and non-negative")
    if not math.isfinite(scale_c) or scale_c <= 0.0:
        raise ValueError("scale_c must be finite and positive")
    return torch.relu((mean_token_surprisal(seq_lp) - float(tau)) / float(scale_c))


def paper_aligned_lambda_weights(
    seq_lp: torch.Tensor,
    *,
    alpha: float,
    lambda_value: float,
    tau: float,
    scale_c: float,
) -> torch.Tensor:
    if not math.isfinite(alpha) or alpha < 0.0:
        raise ValueError("alpha must be finite and non-negative")
    if not math.isfinite(lambda_value) or lambda_value < 0.0:
        raise ValueError("lambda_value must be finite and non-negative")
    z = normalized_excess_surprisal(seq_lp, tau=tau, scale_c=scale_c)
    return (float(alpha) * torch.exp(-float(lambda_value) * z)).detach()


def calibration_from_surprisals(
    surprisals: Sequence[float], *, minimum_scale: float, minimum_active_fraction: float
) -> dict[str, float]:
    values = np.asarray([float(value) for value in surprisals], dtype=np.float64)
    if values.size < 4 or not np.isfinite(values).all() or (values < 0.0).any():
        raise ValueError("Calibration requires at least four finite non-negative surprisals")
    ordered = np.sort(values)
    tau = float(np.median(ordered))
    lower = ordered[: ordered.size // 2]
    upper = ordered[(ordered.size + 1) // 2 :]
    if lower.size == 0 or upper.size == 0:
        raise ValueError("Calibration halves are empty")
    lower_median = float(np.median(lower))
    upper_median = float(np.median(upper))
    scale_c = upper_median - lower_median
    if not math.isfinite(scale_c) or scale_c < float(minimum_scale):
        raise ValueError(f"Degenerate surprisal scale: {scale_c} < minimum {minimum_scale}")
    active_fraction = float(np.mean(ordered > tau))
    if active_fraction < float(minimum_active_fraction):
        raise ValueError(f"Active-tail fraction {active_fraction} < {minimum_active_fraction}")
    return {
        "tau": tau,
        "scale_c": scale_c,
        "lower_half_median": lower_median,
        "upper_half_median": upper_median,
        "active_fraction": active_fraction,
        "sample_count": float(values.size),
    }


def mean_unique_negative_term(
    seq_lp: torch.Tensor,
    weights: torch.Tensor,
    row_index: torch.Tensor,
    unique_counts: torch.Tensor,
) -> torch.Tensor:
    """Average by K_p for each prompt, then prompts; never normalize by weights."""
    if seq_lp.ndim != 1 or weights.shape != seq_lp.shape:
        raise ValueError("seq_lp and weights must be matching vectors")
    if row_index.shape != seq_lp.shape:
        raise ValueError("row_index must match flattened negative vector")
    if unique_counts.ndim != 1 or unique_counts.numel() < 1:
        raise ValueError("unique_counts must be a non-empty vector")
    if bool((unique_counts <= 0).any()):
        raise ValueError("Every prompt must have at least one unique negative")
    row_index = row_index.to(device=seq_lp.device)
    counts = unique_counts.to(device=seq_lp.device, dtype=seq_lp.dtype)
    sums = torch.zeros(unique_counts.numel(), device=seq_lp.device, dtype=seq_lp.dtype)
    sums.scatter_add_(0, row_index, weights * seq_lp)
    return (sums / counts).mean()


def _quantile(values: torch.Tensor, q: float) -> float:
    return float(torch.quantile(values.detach().float().cpu(), q).item())


def weight_diagnostics(
    seq_lp: torch.Tensor,
    weights: torch.Tensor,
    unique_counts: torch.Tensor,
    raw_bank_counts: torch.Tensor,
    *,
    tau: float,
    scale_c: float,
) -> dict[str, float]:
    surprisal = mean_token_surprisal(seq_lp)
    z = normalized_excess_surprisal(seq_lp, tau=tau, scale_c=scale_c)
    return {
        "negative_surprisal_mean": float(surprisal.mean()),
        "negative_surprisal_p10": _quantile(surprisal, 0.10),
        "negative_surprisal_p50": _quantile(surprisal, 0.50),
        "negative_surprisal_p90": _quantile(surprisal, 0.90),
        "normalized_excess_mean": float(z.mean()),
        "normalized_excess_p10": _quantile(z, 0.10),
        "normalized_excess_p50": _quantile(z, 0.50),
        "normalized_excess_p90": _quantile(z, 0.90),
        "active_tail_fraction": float((z > 0.0).float().mean()),
        "weight_mean": float(weights.mean()),
        "weight_p10": _quantile(weights, 0.10),
        "weight_p50": _quantile(weights, 0.50),
        "weight_p90": _quantile(weights, 0.90),
        "unique_negative_count_mean": float(unique_counts.float().mean()),
        "raw_bank_count_mean": float(raw_bank_counts.float().mean()),
        "duplicates_removed_mean": float((raw_bank_counts - unique_counts).float().mean()),
    }


def git_state(repo: Path) -> dict[str, Any]:
    def run(*arguments: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(repo), *arguments],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()

    try:
        return {
            "commit": run("rev-parse", "HEAD"),
            "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(run("status", "--porcelain")),
        }
    except Exception:  # noqa: BLE001
        return {"commit": "unknown", "branch": "unknown", "dirty": True}


def _identity(
    *,
    repo: Path,
    model_path: Path,
    bank: Path,
    val: Path,
    base_config: Path,
    grid_config: Path,
    calibration: Path,
    cell: Cell,
    smoke: bool,
) -> dict[str, Any]:
    source_paths = {
        "common": Path(__file__).resolve(),
        "trainer": Path(__file__)
        .resolve()
        .with_name("countdown_e8_paper_aligned_lambda_trainer.py"),
        "runtime": Path(__file__)
        .resolve()
        .with_name("countdown_e8_paper_aligned_lambda_runtime.py"),
        "auto_launcher": repo / "scripts" / "run_countdown_e8_paper_aligned_lambda_auto.py",
        "one_click": repo / "scripts" / "run_countdown_e8_paper_aligned_lambda_one_click.sh",
        "resume_one_click": repo
        / "scripts"
        / "run_countdown_e8_paper_aligned_lambda_resume_one_click.sh",
    }
    missing_sources = [name for name, path in source_paths.items() if not path.is_file()]
    if missing_sources:
        raise RuntimeError(
            "Paper-aligned lambda identity is missing protected sources: "
            + ", ".join(sorted(missing_sources))
        )
    return {
        "experiment_id": EXPERIMENT_ID,
        "version": VERSION,
        "source": git_state(repo),
        "source_sha256": {name: sha256_file(path) for name, path in sorted(source_paths.items())},
        "model_path": str(model_path),
        "bank_sha256": sha256_file(bank),
        "validation_sha256": sha256_file(val),
        "base_config_sha256": sha256_file(base_config),
        "grid_config_sha256": sha256_file(grid_config),
        "calibration_sha256": sha256_file(calibration),
        "formula": "alpha*exp(-lambda*relu((D-tau)/scale_c))",
        "D_definition": "negative_mean_completion_token_log_probability_with_eos",
        "cell": {
            "name": cell.name,
            "method": cell.method,
            "alpha": cell.alpha,
            "lambda": cell.lambda_value,
            "seed_offset": cell.seed_offset,
        },
        "smoke": bool(smoke),
        "test_data_used": False,
    }


def _identity_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return json.dumps(left, sort_keys=True) == json.dumps(right, sort_keys=True)
