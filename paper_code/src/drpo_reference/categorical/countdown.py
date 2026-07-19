"""Stable reviewer-facing primitives for the Countdown sequence task.

This module intentionally stops below the experiment-entry layer. It contains
protocol-independent expression verification, prompt/completion masking,
autoregressive completion statistics, frozen-bank batching, the detached
paper-aligned linear-surprisal objective, and response-level aggregation. It
does not select a model scale, coefficient, method matrix, seed set, training
budget, checkpoint, or test protocol.
"""

from __future__ import annotations

import ast
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

import torch
import torch.nn.functional as F

COUNTDOWN_CORE_VERSION = "0.2.0-stable-training-core"
COUNTDOWN_REFERENCE_DISTANCE = 2.0
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
    """Build the frozen linear-surprisal objective without selecting a protocol."""

    if (
        positive_sequence_log_probability.ndim != 1
        or positive_sequence_log_probability.numel() < 1
    ):
        raise ValueError("positive sequence log-probability must be a non-empty vector")
    if not bool(torch.isfinite(positive_sequence_log_probability).all()):
        raise ValueError("positive sequence log-probability must be finite")
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
        if (
            negative_sequence_log_probability is None
            or row_index is None
            or unique_counts is None
        ):
            raise ValueError(
                "nonzero alpha requires negative log-probabilities and bank indices"
            )
        weights = paper_aligned_linear_weights(
            negative_sequence_log_probability,
            alpha=alpha,
            coefficient=coefficient,
            reference_distance=reference_distance,
        )
        coordinate = normalized_sequence_surprisal(
            negative_sequence_log_probability,
            reference_distance=reference_distance,
        )
        weighted_negative_lp = mean_unique_negative_term(
            negative_sequence_log_probability,
            weights,
            row_index,
            unique_counts,
        )
        negative_evaluated = True

    loss = -(positive_lp - weighted_negative_lp)
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("Countdown objective is non-finite")
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
    """Evaluate the stable objective and skip the bank forward for Positive-only."""

    positive_batch = packed.get("positive")
    if not isinstance(positive_batch, Mapping):
        raise ValueError("packed Countdown batch has no positive tensor mapping")
    positive_stats = completion_stats(model, positive_batch)
    negative_stats: dict[str, torch.Tensor] | None = None
    if alpha > 0.0:
        bank_batch = packed.get("bank")
        row_index = packed.get("bank_row_index")
        unique_counts = packed.get("unique_counts")
        if not isinstance(bank_batch, Mapping):
            raise ValueError("packed Countdown batch has no bank tensor mapping")
        if not isinstance(row_index, torch.Tensor) or not isinstance(
            unique_counts, torch.Tensor
        ):
            raise ValueError("packed Countdown batch has invalid bank indices")
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
    "COUNTDOWN_CORE_VERSION",
    "COUNTDOWN_REFERENCE_DISTANCE",
    "CountdownTrainingItem",
    "EncodedCompletion",
    "ExpressionVerifier",
    "SYSTEM_PROMPT",
    "chat_prompt",
    "clean_expression",
    "collate_countdown_training_items",
    "completion_statistics_from_logits",
    "completion_stats",
    "countdown_objective_from_model",
    "countdown_training_objective",
    "countdown_weight_diagnostics",
    "encode_countdown_training_row",
    "encode_prompt_completion",
    "evaluate_response_batches",
    "mean_unique_negative_term",
    "move_tensor_batch_to_device",
    "normalized_sequence_surprisal",
    "pad_encoded",
    "paper_aligned_linear_weights",
    "parameter_update_norm",
    "unique_negative_expressions",
    "verifier_category",
    "verify_expression",
    "weighted_sequence_logprob",
]
