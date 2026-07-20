"""Stable reviewer-facing primitives for the Countdown sequence task.

This module intentionally stops below the experiment-entry layer. It contains
protocol-independent expression verification, prompt/completion masking,
autoregressive completion statistics, frozen-bank batching, the detached
paper-aligned linear-surprisal objective, and the registered E8-TAPER active-tail
objective/calibration primitives. It does not select a model path, coefficient
file, seed set, training budget, checkpoint, or test protocol.
"""

from __future__ import annotations

import ast
import math
import random
import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from statistics import median
from typing import Any

import torch
import torch.nn.functional as F

COUNTDOWN_CORE_VERSION = "0.4.0-active-tail-objective-core"
COUNTDOWN_REFERENCE_DISTANCE = 2.0
COUNTDOWN_ACTIVE_TAIL_METHODS = (
    "positive_only",
    "uncontrolled_negative",
    "global_matched",
    "reciprocal_linear",
    "exponential",
    "squared_distance_exponential",
)
COUNTDOWN_ACTIVE_TAIL_TAU_RULE = "calibration_common_half_median_surprisal"
IGNORE_INDEX = -100
MAX_EXPRESSION_LENGTH = 200
SYSTEM_PROMPT = (
    "You solve Countdown arithmetic puzzles. Use every supplied number exactly "
    "once, use only +, -, *, / and parentheses, and return only one arithmetic "
    "expression. Do not include explanations."
)


@dataclass(frozen=True)
class EncodedCompletion:
    """One causal-LM sequence with loss restricted to completion tokens."""

    input_ids: list[int]
    labels: list[int]

    def __post_init__(self) -> None:
        if not self.input_ids or len(self.input_ids) != len(self.labels):
            raise ValueError("input_ids and labels must be aligned and non-empty")
        if all(label == IGNORE_INDEX for label in self.labels):
            raise ValueError("encoded sequence contains no completion token")


@dataclass(frozen=True)
class CountdownTrainingItem:
    """One positive completion and its first-occurrence unique negative bank."""

    positive: EncodedCompletion
    bank: tuple[EncodedCompletion, ...]
    unique_count: int
    raw_bank_count: int

    def __post_init__(self) -> None:
        if not self.bank:
            raise ValueError("Countdown training item has no unique negative")
        if self.unique_count != len(self.bank):
            raise ValueError("unique_count must equal the encoded bank length")
        if self.raw_bank_count < self.unique_count:
            raise ValueError("raw_bank_count cannot be smaller than unique_count")


def clean_expression(text: str) -> str:
    """Extract the arithmetic expression using the canonical Countdown rules."""

    cleaned = re.sub(r"<think>.*?</think>", "", str(text), flags=re.S | re.I)
    answer_match = re.search(r"<answer>(.*?)</answer>", cleaned, flags=re.S | re.I)
    if answer_match:
        cleaned = answer_match.group(1)
    cleaned = cleaned.replace("```python", "").replace("```", "").strip()
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    cleaned = lines[-1] if lines else ""
    cleaned = re.sub(
        r"^(answer|expression)\s*[:=]\s*",
        "",
        cleaned,
        flags=re.I,
    )
    cleaned = cleaned.rstrip(". \t")
    if "=" in cleaned:
        cleaned = cleaned.split("=", 1)[0].strip()
    return cleaned


class ExpressionVerifier(ast.NodeVisitor):
    """Evaluate legal integer-leaf arithmetic while recording used numbers."""

    def __init__(self) -> None:
        self.numbers: list[int] = []

    def visit_Expression(self, node: ast.Expression) -> Fraction:
        return self.visit(node.body)

    def visit_Constant(self, node: ast.Constant) -> Fraction:
        if isinstance(node.value, bool) or not isinstance(node.value, int):
            raise ValueError("only integer literals are allowed")
        self.numbers.append(int(node.value))
        return Fraction(int(node.value), 1)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Fraction:
        raise ValueError("unary operators are not allowed")

    def visit_BinOp(self, node: ast.BinOp) -> Fraction:
        left = self.visit(node.left)
        right = self.visit(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            if right == 0:
                raise ValueError("division by zero")
            return left / right
        raise ValueError("unsupported operator")

    def generic_visit(self, node: ast.AST) -> Any:
        raise ValueError(f"unsupported syntax: {type(node).__name__}")


def verify_expression(
    text: str,
    numbers: Sequence[int],
    target: int,
) -> dict[str, Any]:
    """Return canonical mutually separable format/usage/correctness fields."""

    expression = clean_expression(text)
    result: dict[str, Any] = {
        "expression": expression,
        "valid_format": False,
        "uses_numbers": False,
        "correct": False,
        "value": None,
    }
    if not expression or len(expression) > MAX_EXPRESSION_LENGTH:
        return result
    try:
        visitor = ExpressionVerifier()
        value = visitor.visit(ast.parse(expression, mode="eval"))
        result["valid_format"] = True
        result["uses_numbers"] = Counter(visitor.numbers) == Counter(
            int(number) for number in numbers
        )
        result["value"] = float(value)
        result["correct"] = bool(
            result["uses_numbers"] and value == Fraction(int(target), 1)
        )
    except Exception:
        pass
    return result


def verifier_category(check: Mapping[str, Any]) -> str:
    if bool(check.get("correct")):
        return "correct"
    if not bool(check.get("valid_format")):
        return "invalid_format"
    if not bool(check.get("uses_numbers")):
        return "number_mismatch"
    return "arithmetic_wrong"


def chat_prompt(tokenizer: Any, user_prompt: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    try:
        return str(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        )
    except TypeError:
        return str(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )


def encode_prompt_completion(
    tokenizer: Any,
    prompt: str,
    completion: str,
    max_length: int,
) -> EncodedCompletion:
    """Encode one sample and mask every prompt token from the LM objective."""

    if max_length <= 0:
        raise ValueError("max_length must be positive")
    eos_token = getattr(tokenizer, "eos_token", None)
    if not isinstance(eos_token, str) or not eos_token:
        raise ValueError("tokenizer must provide a non-empty eos_token")
    prefix = chat_prompt(tokenizer, prompt)
    completion_text = clean_expression(completion) + eos_token
    prefix_ids = list(tokenizer(prefix, add_special_tokens=False)["input_ids"])
    full_ids = list(
        tokenizer(prefix + completion_text, add_special_tokens=False)["input_ids"]
    )[:max_length]
    prefix_length = min(len(prefix_ids), len(full_ids))
    labels = [IGNORE_INDEX] * prefix_length + full_ids[prefix_length:]
    return EncodedCompletion(full_ids, labels)


def pad_encoded(
    items: Sequence[EncodedCompletion],
    pad_id: int,
) -> dict[str, torch.Tensor]:
    if not items:
        raise ValueError("at least one encoded completion is required")
    maximum = max(len(item.input_ids) for item in items)
    input_ids: list[list[int]] = []
    labels: list[list[int]] = []
    masks: list[list[int]] = []
    for item in items:
        length = len(item.input_ids)
        padding = maximum - length
        input_ids.append(item.input_ids + [int(pad_id)] * padding)
        labels.append(item.labels + [IGNORE_INDEX] * padding)
        masks.append([1] * length + [0] * padding)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(masks, dtype=torch.long),
    }


def unique_negative_expressions(row: Mapping[str, Any]) -> list[str]:
    """Return first-occurrence unique expressions from a frozen negative bank."""

    unique: list[str] = []
    seen: set[str] = set()
    for item in row.get("negative_bank", []):
        if isinstance(item, Mapping):
            if "expression" not in item:
                raise ValueError("negative-bank mapping has no expression field")
            expression = str(item["expression"])
        else:
            expression = str(item)
        cleaned = clean_expression(expression)
        if cleaned in seen:
            continue
        seen.add(cleaned)
        unique.append(cleaned)
    if not unique:
        raise ValueError("row has no unique negative expression")
    return unique


def encode_countdown_training_row(
    row: Mapping[str, Any],
    tokenizer: Any,
    max_length: int,
) -> CountdownTrainingItem:
    """Encode the frozen positive and every first-occurrence unique negative."""

    prompt = row.get("prompt")
    positive = row.get("positive")
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("Countdown training row requires a non-empty prompt")
    if not isinstance(positive, str) or not positive:
        raise ValueError("Countdown training row requires a non-empty positive")
    negatives = unique_negative_expressions(row)
    raw_bank = row.get("negative_bank", [])
    if not isinstance(raw_bank, Sequence) or isinstance(raw_bank, (str, bytes)):
        raise ValueError("negative_bank must be a sequence")
    return CountdownTrainingItem(
        positive=encode_prompt_completion(tokenizer, prompt, positive, max_length),
        bank=tuple(
            encode_prompt_completion(tokenizer, prompt, expression, max_length)
            for expression in negatives
        ),
        unique_count=len(negatives),
        raw_bank_count=len(raw_bank),
    )


def collate_countdown_training_items(
    items: Sequence[CountdownTrainingItem],
    pad_id: int,
) -> dict[str, Any]:
    """Flatten the unique banks while preserving per-prompt denominators."""

    if not items:
        raise ValueError("at least one Countdown training item is required")
    flattened = [negative for item in items for negative in item.bank]
    row_index = [
        row
        for row, item in enumerate(items)
        for _ in range(item.unique_count)
    ]
    if len(flattened) != len(row_index):
        raise AssertionError("flattened bank and row index became misaligned")
    return {
        "positive": pad_encoded([item.positive for item in items], pad_id),
        "bank": pad_encoded(flattened, pad_id),
        "bank_row_index": torch.tensor(row_index, dtype=torch.long),
        "unique_counts": torch.tensor(
            [item.unique_count for item in items], dtype=torch.long
        ),
        "raw_bank_counts": torch.tensor(
            [item.raw_bank_count for item in items], dtype=torch.long
        ),
    }


def move_tensor_batch_to_device(
    batch: Mapping[str, torch.Tensor],
    device: torch.device | str,
) -> dict[str, torch.Tensor]:
    return {name: tensor.to(device) for name, tensor in batch.items()}


def completion_statistics_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compute mean completion log-probability, entropy, and logit score."""

    if logits.ndim != 3 or labels.ndim != 2:
        raise ValueError("logits must be rank-3 and labels rank-2")
    if logits.shape[:2] != labels.shape or logits.shape[1] < 2:
        raise ValueError("logits and labels must have aligned sequence axes")
    shifted_logits = logits[:, :-1, :].float()
    shifted_labels = labels[:, 1:]
    token_mask = shifted_labels.ne(IGNORE_INDEX)
    lengths = token_mask.sum(dim=-1)
    if bool((lengths <= 0).any()):
        raise ValueError("every sequence must contain a completion token")
    safe_labels = shifted_labels.masked_fill(~token_mask, 0)
    log_probabilities = F.log_softmax(shifted_logits, dim=-1)
    probabilities = log_probabilities.exp()
    token_log_probability = log_probabilities.gather(
        -1,
        safe_labels.unsqueeze(-1),
    ).squeeze(-1)
    float_mask = token_mask.to(token_log_probability.dtype)
    sequence_log_probability = (
        token_log_probability * float_mask
    ).sum(dim=-1) / lengths
    token_entropy = -(probabilities * log_probabilities).sum(dim=-1)
    entropy = (token_entropy * float_mask).sum(dim=-1) / lengths
    selected_probability = probabilities.gather(
        -1,
        safe_labels.unsqueeze(-1),
    ).squeeze(-1)
    probability_squared_norm = probabilities.square().sum(dim=-1)
    token_score = torch.sqrt(
        torch.clamp(
            1.0 - 2.0 * selected_probability + probability_squared_norm,
            min=0.0,
        )
    )
    score = (token_score * float_mask).sum(dim=-1) / lengths
    return {
        "seq_lp": sequence_log_probability,
        "entropy": entropy,
        "score": score,
        "token_lp": token_log_probability,
        "token_mask": token_mask,
        "token_score": token_score,
        "lengths": lengths,
    }


def completion_stats(
    model: Any,
    batch: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    required = {"input_ids", "attention_mask", "labels"}
    missing = sorted(required - set(batch))
    if missing:
        raise ValueError(f"completion batch is missing keys: {missing}")
    output = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        use_cache=False,
    )
    return completion_statistics_from_logits(output.logits, batch["labels"])


def weighted_sequence_logprob(
    stats: Mapping[str, torch.Tensor],
    token_weights: torch.Tensor,
) -> torch.Tensor:
    token_log_probability = stats["token_lp"]
    token_mask = stats["token_mask"]
    lengths = stats["lengths"]
    if token_weights.shape != token_log_probability.shape:
        raise ValueError("token_weights must match token log-probability shape")
    mask = token_mask.to(token_weights.dtype)
    return (
        token_log_probability * token_weights * mask
    ).sum(dim=-1) / lengths


def normalized_sequence_surprisal(
    sequence_log_probability: torch.Tensor,
    *,
    reference_distance: float = COUNTDOWN_REFERENCE_DISTANCE,
) -> torch.Tensor:
    if not math.isfinite(reference_distance) or reference_distance <= 0.0:
        raise ValueError("reference_distance must be finite and positive")
    if not bool(torch.isfinite(sequence_log_probability).all()):
        raise ValueError("sequence log-probability must be finite")
    return (-sequence_log_probability.detach()).clamp_min(0.0) / float(
        reference_distance
    )


def paper_aligned_linear_weights(
    sequence_log_probability: torch.Tensor,
    *,
    alpha: float,
    coefficient: float,
    reference_distance: float = COUNTDOWN_REFERENCE_DISTANCE,
) -> torch.Tensor:
    """Return detached ``alpha * exp(-coefficient * surprisal / 2)`` weights."""

    if not math.isfinite(alpha) or alpha < 0.0:
        raise ValueError("alpha must be finite and non-negative")
    if not math.isfinite(coefficient) or coefficient < 0.0:
        raise ValueError("coefficient must be finite and non-negative")
    coordinate = normalized_sequence_surprisal(
        sequence_log_probability,
        reference_distance=reference_distance,
    )
    return (float(alpha) * torch.exp(-float(coefficient) * coordinate)).detach()


def normalized_active_tail_remoteness(
    sequence_log_probability: torch.Tensor,
    *,
    tau: float,
    surprisal_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return detached normalized excess surprisal ``S`` and ``d=sqrt(S)``."""

    if not math.isfinite(tau) or tau < 0.0:
        raise ValueError("tau must be finite and non-negative")
    if not math.isfinite(surprisal_scale) or surprisal_scale <= 0.0:
        raise ValueError("surprisal_scale must be finite and positive")
    if not bool(torch.isfinite(sequence_log_probability).all()):
        raise ValueError("sequence log-probability must be finite")
    excess = torch.relu(-sequence_log_probability.detach() - float(tau))
    normalized_excess = excess / float(surprisal_scale)
    distance = torch.sqrt(normalized_excess)
    return normalized_excess.detach(), distance.detach()


def active_tail_taper_weights(
    method: str,
    distance: torch.Tensor,
    *,
    coefficient: float,
) -> torch.Tensor:
    """Return v79 detached weights on the true distance coordinate ``d``."""

    if method not in COUNTDOWN_ACTIVE_TAIL_METHODS:
        raise ValueError(f"unknown Countdown active-tail method: {method}")
    if not math.isfinite(coefficient) or coefficient < 0.0:
        raise ValueError("coefficient must be finite and non-negative")
    if not bool(torch.isfinite(distance).all()) or bool((distance < 0).any()):
        raise ValueError("distance must be finite and non-negative")
    if method == "positive_only":
        weights = torch.zeros_like(distance)
    elif method == "uncontrolled_negative":
        weights = torch.ones_like(distance)
    elif method == "global_matched":
        weights = torch.full_like(distance, float(coefficient))
    elif method == "reciprocal_linear":
        weights = 1.0 / (1.0 + float(coefficient) * distance)
    elif method == "exponential":
        weights = torch.exp(-float(coefficient) * distance)
    else:
        weights = torch.exp(-float(coefficient) * distance.square())
    return weights.detach()


def calibration_surprisal_scale(
    surprisals: Sequence[float],
    *,
    minimum: float,
) -> tuple[float, dict[str, float]]:
    """Return the upper-half minus lower-half median calibration scale."""

    if not math.isfinite(minimum) or minimum <= 0.0:
        raise ValueError("minimum must be finite and positive")
    values = sorted(float(value) for value in surprisals)
    if len(values) < 4 or any(not math.isfinite(value) for value in values):
        raise ValueError("calibration requires at least four finite surprisals")
    midpoint = len(values) // 2
    common_median = float(median(values[:midpoint]))
    rare_median = float(median(values[midpoint:]))
    scale = rare_median - common_median
    if not math.isfinite(scale) or scale < float(minimum):
        raise ValueError(
            "calibration surprisal spread is too small: "
            f"scale={scale}, minimum={minimum}"
        )
    return scale, {
        "common_half_median_surprisal": common_median,
        "rare_half_median_surprisal": rare_median,
        "scale": scale,
    }


def resolve_active_tail_tau(
    value: float | str,
    scale_diagnostics: Mapping[str, float],
) -> tuple[float, str]:
    """Resolve the v79 threshold without reading confirmation or test metrics."""

    if value == COUNTDOWN_ACTIVE_TAIL_TAU_RULE:
        tau = float(scale_diagnostics["common_half_median_surprisal"])
        rule = COUNTDOWN_ACTIVE_TAIL_TAU_RULE
    else:
        tau = float(value)
        rule = "fixed_numeric_surprisal_threshold"
    if not math.isfinite(tau) or tau < 0.0:
        raise ValueError("resolved tau must be finite and non-negative")
    return tau, rule


def active_distance_diagnostics(
    surprisals: Sequence[float],
    *,
    tau: float,
    surprisal_scale: float,
) -> dict[str, float | int]:
    """Summarize the nonzero active tail on an independent calibration split."""

    values = torch.tensor(list(surprisals), dtype=torch.float64)
    if values.numel() < 1 or not bool(torch.isfinite(values).all()):
        raise ValueError("surprisals must be a non-empty finite sequence")
    normalized, distance = normalized_active_tail_remoteness(
        -values,
        tau=tau,
        surprisal_scale=surprisal_scale,
    )
    active = normalized > 0
    return {
        "samples": int(values.numel()),
        "active_distance_count": int(active.sum().item()),
        "active_distance_fraction": float(active.float().mean().item()),
        "normalized_excess_mean": float(normalized.mean().item()),
        "distance_mean": float(distance.mean().item()),
        "distance_max": float(distance.max().item()),
    }


def validate_active_tail_calibration(
    *,
    active_distance_fraction: float,
    uncontrolled_norm: float,
    target_unscaled: float,
    coefficients: Mapping[str, float],
    minimum_active_distance_fraction: float,
    nondegenerate_target_max_ratio: float,
    minimum_taper_lambda: float,
) -> dict[str, Any]:
    """Fail closed when calibrated methods collapse into uncontrolled clones."""

    values = (
        active_distance_fraction,
        uncontrolled_norm,
        target_unscaled,
        minimum_active_distance_fraction,
        nondegenerate_target_max_ratio,
        minimum_taper_lambda,
    )
    if any(not math.isfinite(float(value)) for value in values):
        raise ValueError("calibration scalars must be finite")
    if uncontrolled_norm <= 0.0 or target_unscaled < 0.0:
        raise ValueError(
            "gradient norms must be non-negative and uncontrolled positive"
        )
    if minimum_active_distance_fraction <= 0.0:
        raise ValueError("minimum_active_distance_fraction must be positive")
    if not 0.0 < nondegenerate_target_max_ratio < 1.0:
        raise ValueError("nondegenerate_target_max_ratio must lie in (0, 1)")
    if minimum_taper_lambda <= 0.0:
        raise ValueError("minimum_taper_lambda must be positive")

    target_ratio = float(target_unscaled / uncontrolled_norm)
    failures: list[str] = []
    if target_ratio >= nondegenerate_target_max_ratio:
        failures.append("reference target is too close to uncontrolled")
    if active_distance_fraction < minimum_active_distance_fraction:
        failures.append("active-distance fraction is too small")
    if float(coefficients.get("global_matched", 1.0)) >= (
        nondegenerate_target_max_ratio
    ):
        failures.append("global_matched is degenerate or near-uncontrolled")
    for method in ("reciprocal_linear", "squared_distance_exponential"):
        if float(coefficients.get(method, 0.0)) <= minimum_taper_lambda:
            failures.append(f"{method} lambda is degenerate")
    payload = {
        "status": "pass" if not failures else "fail",
        "target_unscaled_to_uncontrolled_ratio": target_ratio,
        "nondegenerate_target_max_ratio": float(nondegenerate_target_max_ratio),
        "minimum_taper_lambda": float(minimum_taper_lambda),
        "minimum_active_distance_fraction": float(minimum_active_distance_fraction),
        "failures": failures,
    }
    if failures:
        raise RuntimeError(
            "Countdown active-tail calibration degenerated: " + "; ".join(failures)
        )
    return payload


def make_prompt_balanced_sampler_plan(
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    total_samples: int,
) -> list[dict[str, int]]:
    """Uniform prompt cycles plus within-prompt negative sampling."""

    if not rows:
        raise ValueError("sampler plan requires a non-empty replay pool")
    if total_samples <= 0:
        raise ValueError("total_samples must be positive")
    candidate_counts: list[int] = []
    for row in rows:
        candidates = row.get("negatives", row.get("negative_bank", []))
        if not isinstance(candidates, Sequence) or isinstance(
            candidates, (str, bytes)
        ):
            raise ValueError("every replay row must expose a candidate sequence")
        if len(candidates) < 1:
            raise ValueError("every replay row must have at least one negative")
        candidate_counts.append(len(candidates))

    rng = random.Random(int(seed))
    order: list[int] = []
    while len(order) < total_samples:
        cycle = list(range(len(rows)))
        rng.shuffle(cycle)
        order.extend(cycle)
    return [
        {
            "prompt_index": int(row_index),
            "negative_index": int(rng.randrange(candidate_counts[row_index])),
        }
        for row_index in order[:total_samples]
    ]


def calibrate_monotone_coefficient(
    norm_fn: Callable[[float], float],
    target: float,
    *,
    maximum: float,
    steps: int,
    tolerance: float,
) -> tuple[float, float, float]:
    """Match a gradient-norm target with a bracket scan plus bisection."""

    if not math.isfinite(target) or target <= 0.0:
        raise ValueError("target must be finite and positive")
    if not math.isfinite(maximum) or maximum <= 0.0:
        raise ValueError("maximum must be finite and positive")
    if steps < 0:
        raise ValueError("steps must be non-negative")
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance must be finite and non-negative")

    grid = [0.0]
    value = min(1.0e-4, maximum)
    while value < maximum:
        grid.append(value)
        value *= 2.0
    if grid[-1] != maximum:
        grid.append(float(maximum))
    observations = [(coefficient, float(norm_fn(coefficient))) for coefficient in grid]
    if any(not math.isfinite(norm) or norm < 0.0 for _, norm in observations):
        raise RuntimeError("calibration norm function returned a non-finite value")
    if observations[0][1] < target:
        raise RuntimeError("taper norm at coefficient zero is already below target")

    candidates = list(observations)
    brackets: list[tuple[float, float, float, float]] = []
    for (left, left_norm), (right, right_norm) in zip(
        observations, observations[1:]
    ):
        left_delta = left_norm - target
        right_delta = right_norm - target
        if left_delta == 0.0:
            brackets.append((left, left, left_norm, left_norm))
        elif left_delta * right_delta <= 0.0:
            brackets.append((left, right, left_norm, right_norm))
    if not brackets:
        closest = min(
            candidates,
            key=lambda item: abs(math.log(max(item[1], 1.0e-30) / target)),
        )
        relative_error = abs(closest[1] - target) / target
        if relative_error <= tolerance:
            return float(closest[0]), float(closest[1]), float(relative_error)
        raise RuntimeError("could not bracket calibration target")

    for left, right, left_norm, right_norm in brackets:
        if left == right:
            continue
        left_delta = left_norm - target
        for _ in range(steps):
            middle = 0.5 * (left + right)
            middle_norm = float(norm_fn(middle))
            if not math.isfinite(middle_norm) or middle_norm < 0.0:
                raise RuntimeError(
                    "calibration norm function returned a non-finite value"
                )
            candidates.append((middle, middle_norm))
            middle_delta = middle_norm - target
            if left_delta * middle_delta <= 0.0:
                right, right_norm = middle, middle_norm
            else:
                left, left_norm, left_delta = middle, middle_norm, middle_delta

    coefficient, matched = min(
        candidates,
        key=lambda item: abs(math.log(max(item[1], 1.0e-30) / target)),
    )
    relative_error = abs(matched - target) / target
    if relative_error > tolerance:
        raise RuntimeError(
            f"calibration relative error {relative_error:.6f} exceeds {tolerance:.6f}"
        )
    return float(coefficient), float(matched), float(relative_error)


def mean_unique_negative_term(
    sequence_log_probability: torch.Tensor,
    weights: torch.Tensor,
    row_index: torch.Tensor,
    unique_counts: torch.Tensor,
) -> torch.Tensor:
    """Average by unique negatives per prompt, never by the weight sum."""

    if (
        sequence_log_probability.ndim != 1
        or weights.shape != sequence_log_probability.shape
    ):
        raise ValueError(
            "sequence_log_probability and weights must be matching vectors"
        )
    if row_index.shape != sequence_log_probability.shape:
        raise ValueError("row_index must match the flattened negative vector")
    if unique_counts.ndim != 1 or unique_counts.numel() < 1:
        raise ValueError("unique_counts must be a non-empty vector")
    if bool((unique_counts <= 0).any()):
        raise ValueError("every prompt must have at least one unique negative")
    if bool((row_index < 0).any()) or bool(
        (row_index >= unique_counts.numel()).any()
    ):
        raise ValueError("row_index contains an invalid prompt index")
    indices = row_index.to(device=sequence_log_probability.device)
    counts = unique_counts.to(
        device=sequence_log_probability.device,
        dtype=sequence_log_probability.dtype,
    )
    sums = torch.zeros(
        unique_counts.numel(),
        device=sequence_log_probability.device,
        dtype=sequence_log_probability.dtype,
    )
    sums.scatter_add_(0, indices, weights * sequence_log_probability)
    return (sums / counts).mean()


def countdown_training_objective(
    positive_sequence_log_probability: torch.Tensor,
    *,
    alpha: float,
    coefficient: float,
    negative_sequence_log_probability: torch.Tensor | None = None,
    row_index: torch.Tensor | None = None,
    unique_counts: torch.Tensor | None = None,
    reference_distance: float = COUNTDOWN_REFERENCE_DISTANCE,
) -> dict[str, Any]:
    """Historical linear-surprisal objective retained for Round-1 compatibility."""

    _validate_positive_log_probability(positive_sequence_log_probability)
    if not math.isfinite(alpha) or alpha < 0.0:
        raise ValueError("alpha must be finite and non-negative")
    if not math.isfinite(coefficient) or coefficient < 0.0:
        raise ValueError("coefficient must be finite and non-negative")

    positive_lp = positive_sequence_log_probability.mean()
    empty = positive_sequence_log_probability.new_empty((0,))
    weighted_negative_lp = positive_lp.new_zeros(())
    weights = empty
    coordinate = empty
    negative_evaluated = False
    if alpha > 0.0:
        negative, indices, counts = _require_negative_inputs(
            negative_sequence_log_probability,
            row_index,
            unique_counts,
        )
        weights = paper_aligned_linear_weights(
            negative,
            alpha=alpha,
            coefficient=coefficient,
            reference_distance=reference_distance,
        )
        coordinate = normalized_sequence_surprisal(
            negative,
            reference_distance=reference_distance,
        )
        weighted_negative_lp = mean_unique_negative_term(
            negative,
            weights,
            indices,
            counts,
        )
        negative_evaluated = True

    loss = -(positive_lp - weighted_negative_lp)
    _require_finite_scalar(loss, "Countdown objective")
    return {
        "loss": loss,
        "positive_lp": positive_lp,
        "weighted_negative_lp": weighted_negative_lp,
        "weights": weights,
        "coordinate": coordinate,
        "negative_evaluated": negative_evaluated,
    }


def countdown_objective_from_model(
    model: Any,
    packed: Mapping[str, Any],
    *,
    alpha: float,
    coefficient: float,
    reference_distance: float = COUNTDOWN_REFERENCE_DISTANCE,
) -> dict[str, Any]:
    """Evaluate the historical objective and skip bank forward for Positive-only."""

    positive_batch = _require_tensor_mapping(packed, "positive")
    positive_stats = completion_stats(model, positive_batch)
    negative_stats: dict[str, torch.Tensor] | None = None
    if alpha > 0.0:
        bank_batch = _require_tensor_mapping(packed, "bank")
        row_index, unique_counts = _require_packed_bank_indices(packed)
        negative_stats = completion_stats(model, bank_batch)
        terms = countdown_training_objective(
            positive_stats["seq_lp"],
            alpha=alpha,
            coefficient=coefficient,
            negative_sequence_log_probability=negative_stats["seq_lp"],
            row_index=row_index,
            unique_counts=unique_counts,
            reference_distance=reference_distance,
        )
    else:
        terms = countdown_training_objective(
            positive_stats["seq_lp"],
            alpha=alpha,
            coefficient=coefficient,
            reference_distance=reference_distance,
        )
    return {
        **terms,
        "positive_stats": positive_stats,
        "negative_stats": negative_stats,
    }


def _validate_positive_log_probability(values: torch.Tensor) -> None:
    if values.ndim != 1 or values.numel() < 1:
        raise ValueError("positive sequence log-probability must be a non-empty vector")
    if not bool(torch.isfinite(values).all()):
        raise ValueError("positive sequence log-probability must be finite")


def _require_negative_inputs(
    sequence_log_probability: torch.Tensor | None,
    row_index: torch.Tensor | None,
    unique_counts: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if sequence_log_probability is None or row_index is None or unique_counts is None:
        raise ValueError(
            "negative log-probabilities, row_index, and unique_counts are required"
        )
    if not bool(torch.isfinite(sequence_log_probability).all()):
        raise ValueError("negative sequence log-probability must be finite")
    return sequence_log_probability, row_index, unique_counts


def _require_tensor_mapping(
    packed: Mapping[str, Any],
    name: str,
) -> Mapping[str, torch.Tensor]:
    value = packed.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"packed Countdown batch has no {name} tensor mapping")
    if not all(isinstance(tensor, torch.Tensor) for tensor in value.values()):
        raise ValueError(f"packed Countdown {name} mapping contains a non-tensor")
    return value


def _require_packed_bank_indices(
    packed: Mapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    row_index = packed.get("bank_row_index")
    unique_counts = packed.get("unique_counts")
    if not isinstance(row_index, torch.Tensor) or not isinstance(
        unique_counts, torch.Tensor
    ):
        raise ValueError("packed Countdown batch has invalid bank indices")
    return row_index, unique_counts


def _require_finite_scalar(value: torch.Tensor, name: str) -> None:
    if value.ndim != 0 or not bool(torch.isfinite(value)):
        raise FloatingPointError(f"{name} is non-finite")


def active_tail_objective_from_precomputed_weights(
    positive_sequence_log_probability: torch.Tensor,
    *,
    method: str,
    shared_negative_scale: float,
    negative_sequence_log_probability: torch.Tensor | None = None,
    weights: torch.Tensor | None = None,
    normalized_excess: torch.Tensor | None = None,
    distance: torch.Tensor | None = None,
    row_index: torch.Tensor | None = None,
    unique_counts: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Build the v79 objective from deterministic detached negative weights."""

    _validate_positive_log_probability(positive_sequence_log_probability)
    if method not in COUNTDOWN_ACTIVE_TAIL_METHODS:
        raise ValueError(f"unknown Countdown active-tail method: {method}")
    if not math.isfinite(shared_negative_scale) or shared_negative_scale < 0.0:
        raise ValueError("shared_negative_scale must be finite and non-negative")

    positive_lp = positive_sequence_log_probability.mean()
    empty = positive_sequence_log_probability.new_empty((0,))
    weighted_negative_lp = positive_lp.new_zeros(())
    effective_negative_lp = positive_lp.new_zeros(())
    actual_weights = empty
    actual_normalized = empty
    actual_distance = empty
    negative_evaluated = False

    if method != "positive_only":
        negative, indices, counts = _require_negative_inputs(
            negative_sequence_log_probability,
            row_index,
            unique_counts,
        )
        if weights is None or normalized_excess is None or distance is None:
            raise ValueError("active-tail objective requires precomputed remoteness")
        if weights.shape != negative.shape:
            raise ValueError(
                "active-tail weights must match negative log-probabilities"
            )
        if (
            normalized_excess.shape != negative.shape
            or distance.shape != negative.shape
        ):
            raise ValueError(
                "active-tail remoteness must match negative log-probabilities"
            )
        if (
            weights.requires_grad
            or normalized_excess.requires_grad
            or distance.requires_grad
        ):
            raise ValueError("active-tail weights and remoteness must be detached")
        if not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()):
            raise ValueError("active-tail weights must be finite and non-negative")
        weighted_negative_lp = mean_unique_negative_term(
            negative,
            weights,
            indices,
            counts,
        )
        effective_negative_lp = float(shared_negative_scale) * weighted_negative_lp
        actual_weights = weights
        actual_normalized = normalized_excess
        actual_distance = distance
        negative_evaluated = True

    loss = -(positive_lp - effective_negative_lp)
    _require_finite_scalar(loss, "Countdown active-tail objective")
    return {
        "method": method,
        "loss": loss,
        "positive_lp": positive_lp,
        "weighted_negative_lp": weighted_negative_lp,
        "effective_weighted_negative_lp": effective_negative_lp,
        "shared_negative_scale": float(shared_negative_scale),
        "weights": actual_weights,
        "normalized_excess": actual_normalized,
        "distance": actual_distance,
        "negative_evaluated": negative_evaluated,
        "weights_detached": True,
    }


def active_tail_training_objective(
    positive_sequence_log_probability: torch.Tensor,
    *,
    method: str,
    coefficient: float,
    shared_negative_scale: float,
    tau: float,
    surprisal_scale: float,
    negative_sequence_log_probability: torch.Tensor | None = None,
    row_index: torch.Tensor | None = None,
    unique_counts: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Build the active-tail objective directly from sequence log-probabilities."""

    if method == "positive_only":
        return active_tail_objective_from_precomputed_weights(
            positive_sequence_log_probability,
            method=method,
            shared_negative_scale=shared_negative_scale,
        )
    negative, indices, counts = _require_negative_inputs(
        negative_sequence_log_probability,
        row_index,
        unique_counts,
    )
    normalized, distance = normalized_active_tail_remoteness(
        negative,
        tau=tau,
        surprisal_scale=surprisal_scale,
    )
    weights = active_tail_taper_weights(
        method,
        distance,
        coefficient=coefficient,
    )
    return active_tail_objective_from_precomputed_weights(
        positive_sequence_log_probability,
        method=method,
        shared_negative_scale=shared_negative_scale,
        negative_sequence_log_probability=negative,
        weights=weights,
        normalized_excess=normalized,
        distance=distance,
        row_index=indices,
        unique_counts=counts,
    )


def deterministic_active_tail_weights_from_model(
    model: Any,
    negative_batch: Mapping[str, torch.Tensor],
    *,
    method: str,
    coefficient: float,
    tau: float,
    surprisal_scale: float,
) -> dict[str, torch.Tensor]:
    """Compute learner-relative weights in eval/no-grad and restore model mode."""

    if method == "positive_only":
        raise ValueError("Positive-only must skip the negative-bank forward")
    was_training = bool(model.training)
    model.eval()
    try:
        with torch.no_grad():
            stats = completion_stats(model, negative_batch)
            normalized, distance = normalized_active_tail_remoteness(
                stats["seq_lp"],
                tau=tau,
                surprisal_scale=surprisal_scale,
            )
            weights = active_tail_taper_weights(
                method,
                distance,
                coefficient=coefficient,
            )
    finally:
        model.train(was_training)
    return {
        "sequence_log_probability": stats["seq_lp"].detach(),
        "normalized_excess": normalized.detach(),
        "distance": distance.detach(),
        "weights": weights.detach(),
    }


def active_tail_objective_from_model(
    model: Any,
    packed: Mapping[str, Any],
    *,
    method: str,
    coefficient: float,
    shared_negative_scale: float,
    tau: float,
    surprisal_scale: float,
) -> dict[str, Any]:
    """Use eval/no-grad weights and a second gradient-bearing negative forward."""

    positive_batch = _require_tensor_mapping(packed, "positive")
    positive_stats = completion_stats(model, positive_batch)
    if method == "positive_only":
        terms = active_tail_objective_from_precomputed_weights(
            positive_stats["seq_lp"],
            method=method,
            shared_negative_scale=shared_negative_scale,
        )
        return {
            **terms,
            "positive_stats": positive_stats,
            "weight_stats": None,
            "negative_stats": None,
            "negative_forward_count": 0,
        }

    bank_batch = _require_tensor_mapping(packed, "bank")
    row_index, unique_counts = _require_packed_bank_indices(packed)
    weight_stats = deterministic_active_tail_weights_from_model(
        model,
        bank_batch,
        method=method,
        coefficient=coefficient,
        tau=tau,
        surprisal_scale=surprisal_scale,
    )
    negative_stats = completion_stats(model, bank_batch)
    terms = active_tail_objective_from_precomputed_weights(
        positive_stats["seq_lp"],
        method=method,
        shared_negative_scale=shared_negative_scale,
        negative_sequence_log_probability=negative_stats["seq_lp"],
        weights=weight_stats["weights"],
        normalized_excess=weight_stats["normalized_excess"],
        distance=weight_stats["distance"],
        row_index=row_index,
        unique_counts=unique_counts,
    )
    return {
        **terms,
        "positive_stats": positive_stats,
        "weight_stats": weight_stats,
        "negative_stats": negative_stats,
        "negative_forward_count": 2,
    }


def gradient_l2_from_loss(
    loss: torch.Tensor,
    parameters: Sequence[torch.nn.Parameter],
) -> float:
    """Return full-parameter raw gradient L2 without mutating ``parameter.grad``."""

    trainable = [parameter for parameter in parameters if parameter.requires_grad]
    if not trainable:
        raise ValueError("gradient norm requires at least one trainable parameter")
    _require_finite_scalar(loss, "calibration loss")
    gradients = torch.autograd.grad(loss, trainable, allow_unused=True)
    total = torch.zeros((), dtype=torch.float64)
    for gradient in gradients:
        if gradient is not None:
            if not bool(torch.isfinite(gradient).all()):
                raise FloatingPointError("calibration gradient is non-finite")
            total += gradient.detach().double().cpu().square().sum()
    return float(torch.sqrt(total).item())


def active_tail_objective_gradient_l2(
    model: Any,
    packed: Mapping[str, Any],
    parameters: Sequence[torch.nn.Parameter],
    *,
    objective: str,
    method: str,
    coefficient: float,
    tau: float,
    surprisal_scale: float,
) -> float:
    """Measure a deterministic positive or unscaled negative gradient norm."""

    if objective not in {"positive", "negative"}:
        raise ValueError("objective must be 'positive' or 'negative'")
    was_training = bool(model.training)
    model.zero_grad(set_to_none=True)
    model.eval()
    try:
        if objective == "positive":
            positive_batch = _require_tensor_mapping(packed, "positive")
            stats = completion_stats(model, positive_batch)
            loss = -stats["seq_lp"].mean()
        else:
            if method == "positive_only":
                raise ValueError("Positive-only has no negative calibration objective")
            bank_batch = _require_tensor_mapping(packed, "bank")
            row_index, unique_counts = _require_packed_bank_indices(packed)
            stats = completion_stats(model, bank_batch)
            normalized, distance = normalized_active_tail_remoteness(
                stats["seq_lp"],
                tau=tau,
                surprisal_scale=surprisal_scale,
            )
            weights = active_tail_taper_weights(
                method,
                distance,
                coefficient=coefficient,
            )
            loss = mean_unique_negative_term(
                stats["seq_lp"],
                weights,
                row_index,
                unique_counts,
            )
        return gradient_l2_from_loss(loss, parameters)
    finally:
        model.zero_grad(set_to_none=True)
        model.train(was_training)


def calibrate_active_tail_model(
    model: Any,
    packed: Mapping[str, Any],
    parameters: Sequence[torch.nn.Parameter],
    *,
    tau: float,
    surprisal_scale: float,
    inherited_exponential_coefficient: float,
    maximum_coefficient: float,
    bisection_steps: int,
    relative_l2_tolerance: float,
    minimum_active_distance_fraction: float,
    nondegenerate_target_max_ratio: float,
    minimum_taper_lambda: float,
) -> dict[str, Any]:
    """Calibrate v79 coefficients from one independent model-backed batch.

    The caller owns split construction, model identity, batching, and persistence.
    This function reads no confirmation/test metric and does not select a final
    experiment coordinate.
    """

    if not math.isfinite(inherited_exponential_coefficient) or (
        inherited_exponential_coefficient <= 0.0
    ):
        raise ValueError("inherited_exponential_coefficient must be positive")
    bank_batch = _require_tensor_mapping(packed, "bank")
    weight_stats = deterministic_active_tail_weights_from_model(
        model,
        bank_batch,
        method="uncontrolled_negative",
        coefficient=1.0,
        tau=tau,
        surprisal_scale=surprisal_scale,
    )
    active_fraction = float(
        (weight_stats["normalized_excess"] > 0).float().mean().item()
    )

    common = dict(
        model=model,
        packed=packed,
        parameters=parameters,
        tau=tau,
        surprisal_scale=surprisal_scale,
    )
    positive_norm = active_tail_objective_gradient_l2(
        objective="positive",
        method="positive_only",
        coefficient=0.0,
        **common,
    )
    uncontrolled_norm = active_tail_objective_gradient_l2(
        objective="negative",
        method="uncontrolled_negative",
        coefficient=1.0,
        **common,
    )
    target_unscaled = active_tail_objective_gradient_l2(
        objective="negative",
        method="exponential",
        coefficient=inherited_exponential_coefficient,
        **common,
    )
    if any(
        not math.isfinite(value) or value <= 0.0
        for value in (positive_norm, uncontrolled_norm, target_unscaled)
    ):
        raise RuntimeError("calibration norms must all be finite and positive")

    shared_negative_scale = positive_norm / uncontrolled_norm
    coefficients: dict[str, float] = {
        "positive_only": 0.0,
        "uncontrolled_negative": 1.0,
        "global_matched": target_unscaled / uncontrolled_norm,
        "exponential": float(inherited_exponential_coefficient),
    }
    matched_norms: dict[str, float] = {
        "positive_only": 0.0,
        "uncontrolled_negative": uncontrolled_norm,
        "global_matched": coefficients["global_matched"] * uncontrolled_norm,
        "exponential": target_unscaled,
    }
    errors: dict[str, float] = {
        "global_matched": abs(
            matched_norms["global_matched"] - target_unscaled
        )
        / target_unscaled,
        "exponential": 0.0,
    }
    for method in ("reciprocal_linear", "squared_distance_exponential"):
        coefficient, matched, relative_error = calibrate_monotone_coefficient(
            lambda value, method=method: active_tail_objective_gradient_l2(
                objective="negative",
                method=method,
                coefficient=value,
                **common,
            ),
            target_unscaled,
            maximum=maximum_coefficient,
            steps=bisection_steps,
            tolerance=relative_l2_tolerance,
        )
        coefficients[method] = coefficient
        matched_norms[method] = matched
        errors[method] = relative_error

    guard = validate_active_tail_calibration(
        active_distance_fraction=active_fraction,
        uncontrolled_norm=uncontrolled_norm,
        target_unscaled=target_unscaled,
        coefficients=coefficients,
        minimum_active_distance_fraction=minimum_active_distance_fraction,
        nondegenerate_target_max_ratio=nondegenerate_target_max_ratio,
        minimum_taper_lambda=minimum_taper_lambda,
    )
    return {
        "positive_gradient_l2": positive_norm,
        "uncontrolled_negative_gradient_l2": uncontrolled_norm,
        "shared_negative_scale": shared_negative_scale,
        "target_unscaled_negative_gradient_l2": target_unscaled,
        "target_effective_negative_gradient_l2": (
            shared_negative_scale * target_unscaled
        ),
        "method_coefficients": coefficients,
        "matched_unscaled_negative_gradient_l2": matched_norms,
        "relative_matching_error": errors,
        "active_distance_fraction": active_fraction,
        "calibration_degeneracy_guard": guard,
        "confirmation_or_test_metrics_used": False,
        "frozen_before_method_training": True,
    }


def _quantile(values: torch.Tensor, probability: float) -> float:
    if values.numel() < 1:
        raise ValueError("quantile input must be non-empty")
    return float(torch.quantile(values.detach().float().cpu(), probability).item())


def countdown_weight_diagnostics(
    sequence_log_probability: torch.Tensor,
    weights: torch.Tensor,
    unique_counts: torch.Tensor,
    raw_bank_counts: torch.Tensor,
    *,
    reference_distance: float = COUNTDOWN_REFERENCE_DISTANCE,
) -> dict[str, float]:
    """Return the stable bank/weight diagnostics used by the scan trainer."""

    if weights.shape != sequence_log_probability.shape:
        raise ValueError("weights must match sequence log-probabilities")
    if unique_counts.shape != raw_bank_counts.shape:
        raise ValueError("unique_counts and raw_bank_counts must align")
    if bool((raw_bank_counts < unique_counts).any()):
        raise ValueError("raw bank count cannot be smaller than unique count")
    coordinate = normalized_sequence_surprisal(
        sequence_log_probability,
        reference_distance=reference_distance,
    )
    return {
        "negative_surprisal_mean": float((-sequence_log_probability.detach()).mean()),
        "u_mean": float(coordinate.mean()),
        "u_p10": _quantile(coordinate, 0.10),
        "u_p50": _quantile(coordinate, 0.50),
        "u_p90": _quantile(coordinate, 0.90),
        "weight_mean": float(weights.detach().mean()),
        "weight_p10": _quantile(weights, 0.10),
        "weight_p50": _quantile(weights, 0.50),
        "weight_p90": _quantile(weights, 0.90),
        "unique_negative_count_mean": float(unique_counts.float().mean()),
        "raw_bank_count_mean": float(raw_bank_counts.float().mean()),
        "duplicates_removed_mean": float(
            (raw_bank_counts - unique_counts).float().mean()
        ),
    }


def parameter_update_norm(
    before: Sequence[torch.Tensor],
    parameters: Sequence[torch.nn.Parameter],
) -> float:
    """Measure the L2 parameter change after an optimizer step."""

    if len(before) != len(parameters):
        raise ValueError("parameter snapshots and live parameters must align")
    total = torch.zeros((), dtype=torch.float64)
    for saved, parameter in zip(before, parameters, strict=True):
        delta = parameter.detach().float().cpu() - saved.detach().float().cpu()
        total += delta.double().square().sum()
    return float(torch.sqrt(total).item())


def evaluate_response_batches(
    rows: Sequence[Mapping[str, Any]],
    greedy_outputs: Sequence[str],
    sampled_outputs: Sequence[Sequence[str]],
) -> dict[str, Any]:
    """Aggregate verifier-based Greedy, Pass@k, validity, and failure categories."""

    if (
        not rows
        or len(rows) != len(greedy_outputs)
        or len(rows) != len(sampled_outputs)
    ):
        raise ValueError("rows, greedy_outputs, and sampled_outputs must align")
    greedy_success: list[float] = []
    valid: list[float] = []
    pass_at_k: list[float] = []
    categories: Counter[str] = Counter()
    sample_counts: set[int] = set()
    for row, greedy_text, samples in zip(rows, greedy_outputs, sampled_outputs):
        numbers = row.get("numbers")
        target = row.get("target")
        if not isinstance(numbers, Sequence) or isinstance(numbers, (str, bytes)):
            raise ValueError("each Countdown row must contain a number sequence")
        if not isinstance(target, int):
            raise ValueError("each Countdown row must contain an integer target")
        sample_list = list(samples)
        if not sample_list:
            raise ValueError("each Countdown row must have at least one sampled output")
        sample_counts.add(len(sample_list))
        greedy_check = verify_expression(greedy_text, numbers, target)
        categories[verifier_category(greedy_check)] += 1
        greedy_success.append(float(greedy_check["correct"]))
        valid.append(
            float(greedy_check["valid_format"] and greedy_check["uses_numbers"])
        )
        pass_at_k.append(
            float(
                any(
                    verify_expression(sample, numbers, target)["correct"]
                    for sample in sample_list
                )
            )
        )
    if len(sample_counts) != 1:
        raise ValueError("sample count k must be constant across rows")
    count = float(len(rows))
    return {
        "n_eval": int(count),
        "pass_k": sample_counts.pop(),
        "greedy_success": sum(greedy_success) / count,
        "pass_at_k": sum(pass_at_k) / count,
        "valid_rate": sum(valid) / count,
        "greedy_verifier_categories": dict(sorted(categories.items())),
        "formal_result_claim": False,
        "final_countdown_protocol_frozen": False,
    }


__all__ = [
    "COUNTDOWN_ACTIVE_TAIL_METHODS",
    "COUNTDOWN_ACTIVE_TAIL_TAU_RULE",
    "COUNTDOWN_CORE_VERSION",
    "COUNTDOWN_REFERENCE_DISTANCE",
    "CountdownTrainingItem",
    "EncodedCompletion",
    "ExpressionVerifier",
    "SYSTEM_PROMPT",
    "active_distance_diagnostics",
    "active_tail_objective_from_model",
    "active_tail_objective_from_precomputed_weights",
    "active_tail_objective_gradient_l2",
    "active_tail_taper_weights",
    "active_tail_training_objective",
    "calibrate_active_tail_model",
    "calibrate_monotone_coefficient",
    "calibration_surprisal_scale",
    "chat_prompt",
    "clean_expression",
    "collate_countdown_training_items",
    "completion_statistics_from_logits",
    "completion_stats",
    "countdown_objective_from_model",
    "countdown_training_objective",
    "countdown_weight_diagnostics",
    "deterministic_active_tail_weights_from_model",
    "encode_countdown_training_row",
    "encode_prompt_completion",
    "evaluate_response_batches",
    "gradient_l2_from_loss",
    "make_prompt_balanced_sampler_plan",
    "mean_unique_negative_term",
    "move_tensor_batch_to_device",
    "normalized_active_tail_remoteness",
    "normalized_sequence_surprisal",
    "pad_encoded",
    "paper_aligned_linear_weights",
    "parameter_update_norm",
    "resolve_active_tail_tau",
    "unique_negative_expressions",
    "validate_active_tail_calibration",
    "verifier_category",
    "verify_expression",
    "weighted_sequence_logprob",
]
