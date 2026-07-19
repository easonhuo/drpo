"""Stable reviewer-facing primitives for the Countdown sequence task.

This module intentionally stops below the experiment-entry layer.  It contains
only protocol-independent expression verification, prompt/completion masking,
autoregressive completion statistics, the detached paper-aligned linear
surprisal envelope, and response-level metric aggregation.  It does not select a
model scale, coefficient, method matrix, seed set, training budget, checkpoint,
or test protocol.
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

COUNTDOWN_CORE_VERSION = "0.1.0-stable-core"
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
    prefix_ids = list(
        tokenizer(prefix, add_special_tokens=False)["input_ids"]
    )
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

    if sequence_log_probability.ndim != 1 or weights.shape != sequence_log_probability.shape:
        raise ValueError("sequence_log_probability and weights must be matching vectors")
    if row_index.shape != sequence_log_probability.shape:
        raise ValueError("row_index must match the flattened negative vector")
    if unique_counts.ndim != 1 or unique_counts.numel() < 1:
        raise ValueError("unique_counts must be a non-empty vector")
    if bool((unique_counts <= 0).any()):
        raise ValueError("every prompt must have at least one unique negative")
    if bool((row_index < 0).any()) or bool((row_index >= unique_counts.numel()).any()):
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


def evaluate_response_batches(
    rows: Sequence[Mapping[str, Any]],
    greedy_outputs: Sequence[str],
    sampled_outputs: Sequence[Sequence[str]],
) -> dict[str, Any]:
    """Aggregate verifier-based Greedy, Pass@k, validity, and failure categories."""

    if not rows or len(rows) != len(greedy_outputs) or len(rows) != len(sampled_outputs):
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
    "EncodedCompletion",
    "ExpressionVerifier",
    "SYSTEM_PROMPT",
    "chat_prompt",
    "clean_expression",
    "completion_statistics_from_logits",
    "completion_stats",
    "encode_prompt_completion",
    "evaluate_response_batches",
    "mean_unique_negative_term",
    "normalized_sequence_surprisal",
    "pad_encoded",
    "paper_aligned_linear_weights",
    "unique_negative_expressions",
    "verifier_category",
    "verify_expression",
    "weighted_sequence_logprob",
]
