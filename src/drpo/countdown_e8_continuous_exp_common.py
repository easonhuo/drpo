#!/usr/bin/env python3
"""Code-first E8 V2 continuous EXP alpha-by-c development pilot.

Every unique negative completion participates in the loss.  The only negative
weight is ``alpha * exp(-c * u**2)`` with ``u = current_sequence_surprisal / 2``.
No current-near/current-far selection, gradient-budget matching, or hidden scale
is permitted in this module.
"""
from __future__ import annotations

import hashlib
import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml
from torch.utils.data import Dataset

try:
    from drpo import countdown_qwen_arena_onefile as arena
except ImportError:  # pragma: no cover - direct source execution
    import countdown_qwen_arena_onefile as arena  # type: ignore

EXPERIMENT_ID = "EXT-C-E8-ORACLE-OFFLINE-V2-CONTINUOUS-EXP-GRID-0.5B-01"
VERSION = "0.1.0-dev-code-first"
DEFAULT_GRID_CONFIG = (
    "configs/countdown_e8_oracle_offline_v2_continuous_exp_grid_0p5b.yaml"
)
DEFAULT_BASE_CONFIG = "configs/countdown_e8_base_rl_replay_0p5b.yaml"
ALPHA_VALUES = (0.0, 0.025, 0.05, 0.11, 0.25, 0.5, 1.0)
C_VALUES = (0.0, 0.25, 0.5, 1.0, 1.5)
SEED_OFFSETS = (3000, 4000)
REFERENCE_DISTANCE = 2.0
EXPECTED_POINTS = 31
EXPECTED_CELLS = 62
FORBIDDEN_CONFIG_KEYS = {
    "rho",
    "rho_values",
    "lambda",
    "negative_scale",
    "bank_negative_scale",
    "global_gamma",
    "gradient_rms_matching",
}


@dataclass(frozen=True)
class Cell:
    alpha: float
    c: float
    seed_offset: int

    @property
    def method(self) -> str:
        if self.alpha == 0.0:
            return "positive_only"
        if self.c == 0.0:
            return "global"
        return "continuous_exp"

    @property
    def name(self) -> str:
        return (
            f"base_{self.method}_alpha{_number_tag(self.alpha)}_"
            f"c{_number_tag(self.c)}_seed{self.seed_offset}"
        )


def _number_tag(value: float) -> str:
    return f"{value:.8g}".replace("-", "m").replace(".", "p")


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
        raise ValueError("Continuous EXP grid experiment_id mismatch")
    if config.get("result_status") != "pilot":
        raise ValueError("Continuous EXP grid must remain a pilot")
    if config.get("registration_state") != "dev_code_first_unregistered":
        raise ValueError("Code-first pilot must remain explicitly unregistered")
    present = sorted(FORBIDDEN_CONFIG_KEYS & _walk_keys(config))
    if present:
        raise ValueError("Forbidden legacy/calibration coordinates: " + ", ".join(present))

    bank = config.get("bank", {})
    if bank.get("use_all_unique_negatives") is not True:
        raise ValueError("Every unique negative must participate")
    if bank.get("explicit_near_far_training_classes") is not False:
        raise ValueError("Explicit near/far training classes are forbidden")
    if bank.get("extreme_selection_forbidden") is not True:
        raise ValueError("Current-bank extreme selection must be forbidden")

    remoteness = config.get("remoteness", {})
    if remoteness.get("coordinate") != "u=d/2":
        raise ValueError("The frozen remoteness coordinate must be u=d/2")
    if remoteness.get("weight") != "alpha*exp(-c*u^2)":
        raise ValueError("The frozen weight must be alpha*exp(-c*u^2)")
    if not math.isclose(
        float(remoteness.get("reference_distance", -1.0)),
        REFERENCE_DISTANCE,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("reference_distance must remain 2.0")
    if remoteness.get("detached") is not True:
        raise ValueError("Remoteness weights must be stop-gradient")

    sweep = config.get("sweep", {})
    alphas = tuple(float(value) for value in sweep.get("alpha_values", ()))
    coefficients = tuple(float(value) for value in sweep.get("c_values", ()))
    offsets = tuple(int(value) for value in sweep.get("seed_offsets", ()))
    if alphas != ALPHA_VALUES:
        raise ValueError(f"alpha grid changed: {alphas}")
    if coefficients != C_VALUES:
        raise ValueError(f"c grid changed: {coefficients}")
    if offsets != SEED_OFFSETS:
        raise ValueError(f"development seed offsets changed: {offsets}")
    if int(sweep.get("unique_parameter_points", -1)) != EXPECTED_POINTS:
        raise ValueError("The pilot requires 31 unique parameter points")
    if int(sweep.get("cells", -1)) != EXPECTED_CELLS:
        raise ValueError("The pilot requires 62 cells")
    if sweep.get("alpha_zero_deduplicated") is not True:
        raise ValueError("alpha=0 must be deduplicated across c")

    training = config.get("training", {})
    if int(training.get("steps", -1)) != 1200:
        raise ValueError("The fixed development horizon is 1200 steps")
    if training.get("early_stop") is not False:
        raise ValueError("Early stopping is forbidden")
    if int(training.get("eval_every", -1)) != 100:
        raise ValueError("Greedy/Pass@8 evaluation cadence must remain 100")
    if int(training.get("pass64_every", -1)) != 200:
        raise ValueError("Pass@64 evaluation cadence must remain 200")
    if training.get("denominator") != "unique_negative_count_per_prompt":
        raise ValueError("Loss denominator must be unique negative count per prompt")
    if training.get("normalize_by_weight_sum") is not False:
        raise ValueError("Weight-sum normalization is forbidden")
    if training.get("hidden_negative_scale") is not False:
        raise ValueError("Hidden negative scaling is forbidden")
    if training.get("gradient_budget_matching") is not False:
        raise ValueError("Gradient-budget matching is forbidden")

    execution = config.get("execution", {})
    if int(execution.get("parallel_cells_per_gpu", -1)) != 1:
        raise ValueError("The pilot permits one process per selected GPU")
    if execution.get("identity_checked_resume") is not True:
        raise ValueError("Identity-checked resume is required")

    evaluation = config.get("evaluation", {})
    if evaluation.get("validation_only_during_tuning") is not True:
        raise ValueError("Tuning must remain validation-only")
    if evaluation.get("test_access_forbidden") is not True:
        raise ValueError("Test access must remain forbidden")


def parameter_points(config: Mapping[str, Any]) -> tuple[tuple[float, float], ...]:
    validate_grid_config(config)
    points: list[tuple[float, float]] = [(0.0, 0.0)]
    for alpha in ALPHA_VALUES:
        if alpha == 0.0:
            continue
        for coefficient in C_VALUES:
            points.append((alpha, coefficient))
    if len(points) != EXPECTED_POINTS or len(set(points)) != EXPECTED_POINTS:
        raise AssertionError("Continuous EXP parameter grid is not 31 unique points")
    return tuple(points)


def build_cells(config: Mapping[str, Any]) -> tuple[Cell, ...]:
    points = parameter_points(config)
    cells = tuple(
        Cell(alpha=alpha, c=coefficient, seed_offset=seed_offset)
        for alpha, coefficient in points
        for seed_offset in SEED_OFFSETS
    )
    if len(cells) != EXPECTED_CELLS or len({cell.name for cell in cells}) != EXPECTED_CELLS:
        raise AssertionError("Continuous EXP pilot must produce 62 unique cells")
    return cells


def _expression_from_bank_item(item: Any) -> str:
    if isinstance(item, Mapping):
        if "expression" not in item:
            raise ValueError("Negative-bank mapping has no expression field")
        return str(item["expression"])
    return str(item)


def unique_negative_expressions(row: Mapping[str, Any]) -> list[str]:
    """Return first-occurrence unique negatives under the existing expression cleaner."""
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
            raise ValueError("Continuous EXP batch has no unique negatives")
        row_index = [
            row
            for row, item in enumerate(batch)
            for _ in range(int(item["unique_count"]))
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


def continuous_remoteness(
    seq_lp: torch.Tensor, *, reference_distance: float = REFERENCE_DISTANCE
) -> torch.Tensor:
    if not math.isfinite(reference_distance) or reference_distance <= 0.0:
        raise ValueError("reference_distance must be finite and positive")
    distance = (-seq_lp.detach()).clamp_min(0.0)
    return distance / float(reference_distance)


def continuous_exp_weights(
    seq_lp: torch.Tensor,
    *,
    alpha: float,
    c: float,
    reference_distance: float = REFERENCE_DISTANCE,
) -> torch.Tensor:
    if not math.isfinite(alpha) or alpha < 0.0:
        raise ValueError("alpha must be finite and non-negative")
    if not math.isfinite(c) or c < 0.0:
        raise ValueError("c must be finite and non-negative")
    u = continuous_remoteness(seq_lp, reference_distance=reference_distance)
    return (float(alpha) * torch.exp(-float(c) * u.square())).detach()


def mean_unique_negative_term(
    seq_lp: torch.Tensor,
    weights: torch.Tensor,
    row_index: torch.Tensor,
    unique_counts: torch.Tensor,
) -> torch.Tensor:
    """Average by K_p for each prompt, then average prompts; never normalize by weights."""
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
    sums = torch.zeros(
        unique_counts.numel(), device=seq_lp.device, dtype=seq_lp.dtype
    )
    sums.scatter_add_(0, row_index, weights * seq_lp)
    return (sums / counts).mean()


def _quantile(values: torch.Tensor, q: float) -> float:
    return float(torch.quantile(values.detach().float().cpu(), q).item())


def weight_diagnostics(
    seq_lp: torch.Tensor,
    weights: torch.Tensor,
    unique_counts: torch.Tensor,
    raw_bank_counts: torch.Tensor,
) -> dict[str, float]:
    u = continuous_remoteness(seq_lp)
    return {
        "negative_surprisal_mean": float((-seq_lp.detach()).mean()),
        "u_mean": float(u.mean()),
        "u_p10": _quantile(u, 0.10),
        "u_p50": _quantile(u, 0.50),
        "u_p90": _quantile(u, 0.90),
        "weight_mean": float(weights.mean()),
        "weight_p10": _quantile(weights, 0.10),
        "weight_p50": _quantile(weights, 0.50),
        "weight_p90": _quantile(weights, 0.90),
        "unique_negative_count_mean": float(unique_counts.float().mean()),
        "raw_bank_count_mean": float(raw_bank_counts.float().mean()),
        "duplicates_removed_mean": float(
            (raw_bank_counts - unique_counts).float().mean()
        ),
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
    cell: Cell,
    smoke: bool,
) -> dict[str, Any]:
    return {
        "experiment_id": EXPERIMENT_ID,
        "version": VERSION,
        "source": git_state(repo),
        "source_sha256": sha256_file(__file__),
        "model_path": str(model_path),
        "bank_sha256": sha256_file(bank),
        "validation_sha256": sha256_file(val),
        "base_config_sha256": sha256_file(base_config),
        "grid_config_sha256": sha256_file(grid_config),
        "cell": {
            "name": cell.name,
            "method": cell.method,
            "alpha": cell.alpha,
            "c": cell.c,
            "seed_offset": cell.seed_offset,
        },
        "smoke": bool(smoke),
        "test_data_used": False,
    }


def _identity_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return json.dumps(left, sort_keys=True) == json.dumps(right, sort_keys=True)
