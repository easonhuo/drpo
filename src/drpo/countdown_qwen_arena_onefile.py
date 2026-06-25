#!/usr/bin/env python3
"""Countdown audited base-first external-validity arena for local Qwen Instruct models (v4.1).

One-command run
---------------
python3 src/drpo/countdown_qwen_arena_onefile.py run \
  --model_path /ABS/PATH/TO/QWEN-0.5B-INSTRUCT \
  --work_dir /ABS/PATH/TO/COUNTDOWN_RUN \
  --gpu 0 --preset auto --memory_mode bf16

The v4.1 protocol first evaluates the untouched base model. If the base checkpoint
passes the registered verifier/format gate, all compared methods start from one
shared untrained LoRA adapter and no Countdown SFT is performed. A minimal SFT
fallback is used only when the base gate fails.

The run has two responsibilities:
  1. a fixed-negative-advantage near/far mechanism probe on matched legal wrong
     expressions;
  2. a paired effect comparison: positive-only, controlled-negative,
     uncontrolled-negative, and a calibrated global-matched control.

All pilot methods use the same BF16 LoRA parameterization. Model/adaptor binaries
remain server-local; only manifests, metrics, and hashes belong in artifacts.

Countdown is external validity for the D-U1 categorical theory. It does not
replace D-U1 causal identification. Static checks and CPU self-tests are not
formal Qwen results.
"""
from __future__ import annotations

import argparse
import ast
import copy
import json
import math
import os
import random
import re
import csv
import hashlib
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from fractions import Fraction
from itertools import product
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
        get_cosine_schedule_with_warmup,
    )
except ImportError:  # help/data self-tests should still work before installation
    AutoModelForCausalLM = AutoTokenizer = BitsAndBytesConfig = None
    get_cosine_schedule_with_warmup = None

try:
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
except ImportError:
    LoraConfig = PeftModel = get_peft_model = prepare_model_for_kbit_training = None


def require_hf_stack() -> None:
    if AutoModelForCausalLM is None or LoraConfig is None:
        raise SystemExit(
            "Missing Hugging Face training dependencies. Install once with:\n"
            "  pip install -U 'transformers>=4.51' 'peft>=0.15' "
            "'accelerate>=1.2' 'bitsandbytes>=0.45' sentencepiece"
        )


OPS = ("+", "-", "*", "/")
SYSTEM_PROMPT = (
    "You solve Countdown arithmetic puzzles. Use every supplied number exactly "
    "once, use only +, -, *, / and parentheses, and return only one arithmetic "
    "expression. Do not include explanations."
)

VERSION = "4.1.0-audited-pilot"


def read_model_metadata(model_path: str) -> dict[str, Any]:
    """Read enough local config metadata to choose a safe single-GPU plan."""
    root = Path(model_path)
    cfg_path = root / "config.json"
    if not cfg_path.exists():
        return {"model_type": "unknown", "estimated_params_b": None}
    cfg = json.loads(cfg_path.read_text())
    hidden = cfg.get("hidden_size") or cfg.get("d_model")
    layers = cfg.get("num_hidden_layers") or cfg.get("n_layer")
    inter = cfg.get("intermediate_size")
    vocab = cfg.get("vocab_size")
    estimate = None
    if all(isinstance(x, int) and x > 0 for x in (hidden, layers, inter, vocab)):
        # Dense decoder-only estimate: embeddings + attention + MLP + norms.
        estimate = (vocab * hidden + layers * (4 * hidden * hidden + 3 * hidden * inter)) / 1e9
    name = root.name.lower()
    for tag, value in (("0.5b", 0.5), ("0.6b", 0.6), ("1.5b", 1.5),
                       ("1.8b", 1.8), ("3b", 3.0), ("4b", 4.0),
                       ("7b", 7.0), ("8b", 8.0), ("14b", 14.0),
                       ("32b", 32.0)):
        if tag in name:
            estimate = value
            break
    return {
        "model_type": cfg.get("model_type", "unknown"),
        "architectures": cfg.get("architectures", []),
        "estimated_params_b": estimate,
        "hidden_size": hidden,
        "num_hidden_layers": layers,
    }


def gpu_memory_gib(index: int = 0) -> float:
    if not torch.cuda.is_available():
        return 0.0
    props = torch.cuda.get_device_properties(index)
    return float(props.total_memory / (1024 ** 3))


def resolve_execution_plan(
    model_path: str, preset: str, memory_mode: str, gpu: str
) -> dict[str, Any]:
    """Resolve model-size preset, precision, quantization, and safe batch sizes."""
    if gpu != "auto":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for the formal Countdown arena")
    meta = read_model_metadata(model_path)
    params_b = meta.get("estimated_params_b") or 7.0
    mem = gpu_memory_gib(0)
    if preset == "auto":
        preset = "0.5b" if params_b <= 1.0 else ("3b" if params_b <= 4.5 else "7b")
    if memory_mode == "auto":
        # Conservative single-GPU rule. H20/A100-class cards use BF16 LoRA;
        # smaller cards use QLoRA. The real preflight still verifies backward/generation.
        bf16_need = 18.0 if params_b <= 1.0 else (34.0 if params_b <= 4.5 else 60.0)
        memory_mode = "bf16" if mem >= bf16_need else "qlora"
    if memory_mode == "bf16":
        load_in_4bit = False
        dtype = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    elif memory_mode == "qlora":
        load_in_4bit = True
        dtype = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    else:
        raise ValueError(f"Unknown memory_mode: {memory_mode}")
    if mem >= 80:
        micro_batch, eval_batch, rollout_batch, score_batch = 4, 32, 16, 64
    elif mem >= 40:
        micro_batch, eval_batch, rollout_batch, score_batch = 2, 16, 8, 32
    elif mem >= 24:
        micro_batch, eval_batch, rollout_batch, score_batch = 1, 8, 4, 16
    else:
        micro_batch, eval_batch, rollout_batch, score_batch = 1, 4, 2, 8
    return {
        "version": VERSION,
        "preset": preset,
        "memory_mode": memory_mode,
        "load_in_4bit": load_in_4bit,
        "dtype": dtype,
        "gpu_visible": os.environ.get("CUDA_VISIBLE_DEVICES", "all"),
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_memory_gib": round(mem, 2),
        "model_metadata": meta,
        "micro_batch": micro_batch,
        "eval_batch": eval_batch,
        "rollout_batch": rollout_batch,
        "score_batch": score_batch,
    }


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def clean_expression(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I)
    answer_match = re.search(r"<answer>(.*?)</answer>", text, flags=re.S | re.I)
    if answer_match:
        text = answer_match.group(1)
    text = text.replace("```python", "").replace("```", "").strip()
    # Keep only the first non-empty line, then strip common answer prefixes.
    lines = [x.strip() for x in text.splitlines() if x.strip()]
    text = lines[-1] if lines else ""
    text = re.sub(r"^(answer|expression)\s*[:=]\s*", "", text, flags=re.I)
    text = text.rstrip(". \t")
    # Some models write "expr = target". Keep the expression side.
    if "=" in text:
        text = text.split("=", 1)[0].strip()
    return text


class ExpressionVerifier(ast.NodeVisitor):
    def __init__(self) -> None:
        self.numbers: list[int] = []

    def visit_Expression(self, node: ast.Expression) -> Fraction:
        return self.visit(node.body)

    def visit_Constant(self, node: ast.Constant) -> Fraction:
        if isinstance(node.value, bool) or not isinstance(node.value, int):
            raise ValueError("Only integer literals are allowed")
        self.numbers.append(int(node.value))
        return Fraction(int(node.value), 1)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Fraction:
        # Do not allow unary minus because it creates numbers not supplied.
        raise ValueError("Unary operators are not allowed")

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


def verify_expression(text: str, numbers: Sequence[int], target: int) -> dict[str, Any]:
    expression = clean_expression(text)
    result: dict[str, Any] = {
        "expression": expression,
        "valid_format": False,
        "uses_numbers": False,
        "correct": False,
        "value": None,
    }
    if not expression or len(expression) > 200:
        return result
    try:
        tree = ast.parse(expression, mode="eval")
        visitor = ExpressionVerifier()
        value = visitor.visit(tree)
        result["valid_format"] = True
        result["uses_numbers"] = Counter(visitor.numbers) == Counter(int(x) for x in numbers)
        result["value"] = float(value)
        result["correct"] = bool(result["uses_numbers"] and value == Fraction(int(target), 1))
    except Exception:
        pass
    return result



_AST_OP = {
    ast.Add: "+",
    ast.Sub: "-",
    ast.Mult: "*",
    ast.Div: "/",
}


@dataclass
class _PatternNode:
    """Park-style generic signed arithmetic tree used only for structural audits."""

    op: str
    children: list["_PatternNode"] = field(default_factory=list)
    sign: str = "+"
    weight: int = 1

    def __post_init__(self) -> None:
        self.weight = 1 if not self.children else sum(child.weight for child in self.children)


def _flip_sign(sign: str) -> str:
    return "-" if sign == "+" else "+"


def _pattern_tree_from_ast(node: ast.AST, symbols: Iterable[str] | None = None) -> _PatternNode:
    symbol_iter = iter(symbols or (chr(ord("A") + i) for i in range(26)))

    def convert(current: ast.AST) -> _PatternNode:
        if isinstance(current, ast.Expression):
            return convert(current.body)
        if isinstance(current, (ast.Constant, ast.Name)):
            if isinstance(current, ast.Constant):
                if isinstance(current.value, bool) or not isinstance(current.value, int):
                    raise ValueError("only integer leaves are supported")
            return _PatternNode(next(symbol_iter))
        if isinstance(current, ast.BinOp) and type(current.op) in _AST_OP:
            return _PatternNode(
                _AST_OP[type(current.op)],
                [convert(current.left), convert(current.right)],
            )
        raise ValueError(f"unsupported structure node: {type(current).__name__}")

    return convert(node)


def _raw_pattern_shape(node: _PatternNode) -> str:
    if not node.children:
        return "L"
    return node.op + "(" + ",".join(child.sign + _raw_pattern_shape(child) for child in node.children) + ")"


def _generic_pattern_tree(node: _PatternNode) -> _PatternNode:
    """Flatten associative groups and encode subtraction/division as signed children.

    This follows the reproducible canonicalization machinery of Park et al. while
    deliberately avoiding their stronger claim that every subtree is a latent skill.
    """
    if not node.children:
        return node
    node.children = [_generic_pattern_tree(child) for child in node.children]
    if node.op in {"-", "/"}:
        node.children[1].sign = _flip_sign(node.children[1].sign)
    if node.op in {"+", "-"}:
        merged: list[_PatternNode] = []
        for child in node.children:
            if child.op in {"+", "-"}:
                for grandchild in child.children:
                    if child.sign == "-":
                        grandchild.sign = _flip_sign(grandchild.sign)
                    merged.append(grandchild)
            else:
                merged.append(child)
        node.children = merged
        node.op = "+"
    elif node.op in {"*", "/"}:
        merged = []
        for child in node.children:
            if child.op in {"*", "/"}:
                for grandchild in child.children:
                    if child.sign == "-":
                        grandchild.sign = _flip_sign(grandchild.sign)
                    merged.append(grandchild)
            else:
                merged.append(child)
        node.children = merged
        node.op = "*"
    node.weight = sum(child.weight for child in node.children)
    node.children.sort(
        key=lambda child: (
            child.sign,
            -child.weight,
            -sum(grandchild.sign == "+" for grandchild in child.children),
            _raw_pattern_shape(child),
        )
    )
    return node


def _canonical_pattern_string(tree: _PatternNode) -> str:
    symbols: dict[str, str] = {}

    def render(node: _PatternNode) -> str:
        if not node.children:
            if node.op not in symbols:
                symbols[node.op] = chr(ord("A") + len(symbols))
            return symbols[node.op]
        parts: list[str] = []
        for index, child in enumerate(node.children):
            child_text = render(child)
            if child.children and node.op == "*" and child.op == "+":
                child_text = f"({child_text})"
            if index > 0:
                if child.sign == "-":
                    parts.append("-" if node.op == "+" else "/")
                else:
                    parts.append(node.op)
            elif child.sign == "-":
                # The canonical sort places positive children first for all patterns
                # generated here. Keep this explicit guard for malformed inputs.
                parts.append("-" if node.op == "+" else "1/")
            parts.append(child_text)
        return "".join(parts)

    return render(tree)


def expression_structure(text: str) -> str:
    """Park-style canonical arithmetic pattern, ignoring literal values.

    Commutative and associative variants of addition/multiplication collapse to
    one pattern; subtraction and division remain direction-sensitive.
    """
    expression = clean_expression(text)
    if not expression:
        raise ValueError("empty expression")
    tree = _pattern_tree_from_ast(ast.parse(expression, mode="eval"))
    return _canonical_pattern_string(_generic_pattern_tree(tree))


def _tree_depth(node: ast.AST) -> int:
    if isinstance(node, ast.Expression):
        return _tree_depth(node.body)
    if isinstance(node, (ast.Constant, ast.Name)):
        return 0
    if isinstance(node, ast.BinOp):
        return 1 + max(_tree_depth(node.left), _tree_depth(node.right))
    raise ValueError(f"unsupported depth node: {type(node).__name__}")


def expression_tree_depth(text: str) -> int:
    expression = clean_expression(text)
    return _tree_depth(ast.parse(expression, mode="eval"))


def stable_fingerprint(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def source_provenance() -> dict[str, Any]:
    """Record the exact source file and Git state used by a local run."""
    source_file = Path(__file__).resolve()
    result: dict[str, Any] = {
        "source_file": str(source_file),
        "source_sha256": hashlib.sha256(source_file.read_bytes()).hexdigest(),
        "git_commit": None,
        "git_branch": None,
        "git_dirty": None,
    }
    try:
        repository = subprocess.check_output(
            ["git", "-C", str(source_file.parent), "rev-parse", "--show-toplevel"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        result["repository_root"] = str(Path(repository).resolve())
        result["git_commit"] = subprocess.check_output(
            ["git", "-C", repository, "rev-parse", "HEAD"], text=True
        ).strip()
        result["git_branch"] = subprocess.check_output(
            ["git", "-C", repository, "rev-parse", "--abbrev-ref", "HEAD"], text=True
        ).strip()
        status = subprocess.check_output(
            ["git", "-C", repository, "status", "--porcelain"], text=True
        )
        result["git_dirty"] = bool(status.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return result


_PATTERN_SHAPES: dict[int, tuple[str, ...]] = {
    3: ("(A{0}B){1}C", "A{0}(B{1}C)"),
    4: (
        "A{0}(B{1}(C{2}D))",
        "A{0}((B{2}C){1}D)",
        "(A{1}B){0}(C{2}D)",
        "(A{1}(B{2}C)){0}D",
        "((A{2}B){1}C){0}D",
    ),
}


def _all_raw_patterns(n_numbers: int) -> list[str]:
    if n_numbers not in _PATTERN_SHAPES:
        raise ValueError("Park-style structural protocol currently supports 3 or 4 numbers")
    return [
        shape.format(*ops)
        for shape in _PATTERN_SHAPES[n_numbers]
        for ops in product(OPS, repeat=n_numbers - 1)
    ]


def canonical_pattern_catalog(n_numbers: int = 4) -> dict[str, str]:
    """Map each canonical pattern to one deterministic executable template."""
    catalog: dict[str, str] = {}
    for raw in _all_raw_patterns(n_numbers):
        canonical = expression_structure(raw)
        catalog.setdefault(canonical, raw)
    return dict(sorted(catalog.items()))


def _canonical_subpatterns(raw_pattern: str, subtree_size: int) -> set[str]:
    root = ast.parse(raw_pattern, mode="eval").body
    found: set[str] = set()

    def leaf_count(node: ast.AST) -> int:
        if isinstance(node, ast.Name):
            return 1
        if isinstance(node, ast.BinOp):
            return leaf_count(node.left) + leaf_count(node.right)
        raise ValueError(type(node).__name__)

    def render(node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.BinOp) and type(node.op) in _AST_OP:
            return f"({render(node.left)} {_AST_OP[type(node.op)]} {render(node.right)})"
        raise ValueError(type(node).__name__)

    def visit(node: ast.AST) -> None:
        if leaf_count(node) == subtree_size:
            found.add(expression_structure(render(node)))
        if isinstance(node, ast.BinOp):
            visit(node.left)
            visit(node.right)

    visit(root)
    return found


def pattern_family_catalog() -> tuple[dict[str, str], dict[str, set[str]]]:
    """Return four-number templates and Park-style three-number derivative families."""
    templates: dict[str, str] = {}
    families: dict[str, set[str]] = defaultdict(set)
    for raw in _all_raw_patterns(4):
        canonical = expression_structure(raw)
        templates.setdefault(canonical, raw)
        for family_seed in _canonical_subpatterns(raw, 3):
            families[family_seed].add(canonical)
    return dict(sorted(templates.items())), {k: set(v) for k, v in sorted(families.items())}


def choose_disjoint_holdout_families(
    seed: int,
    val_count: int = 0,
    test_count: int = 0,
) -> tuple[str, str, set[str], set[str]]:
    templates, families = pattern_family_catalog()
    capacities = {
        pattern: len(_feasible_pattern_instances(pattern, template))
        for pattern, template in templates.items()
    }
    candidates: list[tuple[str, str]] = []
    keys = sorted(families)
    for index, left in enumerate(keys):
        left_patterns = {pattern for pattern in families[left] if capacities[pattern] > 0}
        if not left_patterns:
            continue
        for right in keys[index + 1 :]:
            right_patterns = {pattern for pattern in families[right] if capacities[pattern] > 0}
            if not right_patterns or not left_patterns.isdisjoint(right_patterns):
                continue
            left_capacity = sum(capacities[pattern] for pattern in left_patterns)
            right_capacity = sum(capacities[pattern] for pattern in right_patterns)
            # Either orientation may be used. Keep only pairs with enough headroom
            # for the requested held-out splits under the frozen 1..9 task range.
            if (
                left_capacity >= max(val_count, 1)
                and right_capacity >= max(test_count, 1)
            ) or (
                right_capacity >= max(val_count, 1)
                and left_capacity >= max(test_count, 1)
            ):
                candidates.append((left, right))
    if not candidates:
        raise RuntimeError("No disjoint held-out pattern families have enough feasible puzzles")
    candidates.sort(key=lambda pair: stable_fingerprint({"seed": seed, "pair": pair}))
    left, right = candidates[0]
    left_patterns = {pattern for pattern in families[left] if capacities[pattern] > 0}
    right_patterns = {pattern for pattern in families[right] if capacities[pattern] > 0}
    left_capacity = sum(capacities[pattern] for pattern in left_patterns)
    right_capacity = sum(capacities[pattern] for pattern in right_patterns)
    if left_capacity >= max(val_count, 1) and right_capacity >= max(test_count, 1):
        return left, right, left_patterns, right_patterns
    return right, left, right_patterns, left_patterns


def _instantiate_pattern(template: str, numbers: Sequence[int]) -> str:
    result = template
    for index, number in enumerate(numbers):
        result = result.replace(chr(ord("A") + index), str(int(number)))
    return result


@lru_cache(maxsize=None)
def _feasible_pattern_instances(
    canonical_pattern: str,
    template: str,
) -> tuple[tuple[tuple[tuple[int, ...], int], tuple[int, ...]], ...]:
    """Enumerate feasible 1..9 instances once for balanced pattern-first sampling."""
    del canonical_pattern  # Included in the cache key and for call-site clarity.
    code = compile(template, "<countdown-pattern>", "eval")
    unique: dict[tuple[tuple[int, ...], int], tuple[int, ...]] = {}
    for numbers in product(range(1, 10), repeat=4):
        env = {chr(ord("A") + index): Fraction(number) for index, number in enumerate(numbers)}
        try:
            value = eval(code, {"__builtins__": {}}, env)  # noqa: S307 - fixed internal templates only
        except ZeroDivisionError:
            continue
        if not isinstance(value, Fraction) or value.denominator != 1:
            continue
        target = int(value)
        if not (5 <= target <= 100):
            continue
        key = (tuple(sorted(numbers)), target)
        unique.setdefault(key, tuple(int(number) for number in numbers))
    return tuple(sorted(unique.items()))


def _select_capacity_balanced_patterns(
    patterns: Sequence[str],
    count: int,
    templates: dict[str, str],
    seed: int,
    min_patterns: int,
    forbidden_keys: set[tuple[tuple[int, ...], int]] | None = None,
) -> list[str]:
    """Choose the largest capacity-supported core with near-equal target quotas.

    Park et al. generate thousands of rows per pattern with numbers sampled from
    1..99. This project keeps the previously registered 1..9 number range, where
    several canonical patterns have only a handful of unique feasible prompts.
    Rather than claim balance while silently exhausting those patterns, select a
    deterministic capacity-supported subset and report the excluded patterns.
    """
    forbidden = forbidden_keys or set()

    def available_capacity(pattern: str) -> int:
        return sum(
            key not in forbidden
            for key, _ in _feasible_pattern_instances(pattern, templates[pattern])
        )

    ranked = sorted(
        patterns,
        key=lambda pattern: (
            -available_capacity(pattern),
            stable_fingerprint({"seed": seed, "pattern": pattern}),
        ),
    )
    maximum = min(len(ranked), max(count, 1))
    for n_patterns in range(maximum, min_patterns - 1, -1):
        quota_high = math.ceil(count / n_patterns)
        selected = ranked[:n_patterns]
        if all(available_capacity(pattern) >= quota_high for pattern in selected):
            return sorted(selected)
    raise RuntimeError(
        f"No capacity-supported balanced pattern subset for count={count}, "
        f"min_patterns={min_patterns}"
    )


def _allocate_balanced_pattern_rows(
    split: str,
    count: int,
    patterns: Sequence[str],
    templates: dict[str, str],
    seed: int,
    forbidden_keys: set[tuple[tuple[int, ...], int]],
    family_seed: str | None,
    allow_cross_pattern_key_reuse: bool,
) -> tuple[list[dict[str, Any]], Counter[str], set[tuple[tuple[int, ...], int]]]:
    rng = random.Random(seed)
    pools: dict[str, list[tuple[tuple[tuple[int, ...], int], tuple[int, ...]]]] = {}
    for pattern in patterns:
        pool = list(_feasible_pattern_instances(pattern, templates[pattern]))
        rng.shuffle(pool)
        pools[pattern] = pool
    cursors = {pattern: 0 for pattern in patterns}
    counts: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    split_keys: set[tuple[tuple[int, ...], int]] = set()
    active = set(patterns)
    while len(rows) < count:
        if not active:
            raise RuntimeError(f"Insufficient feasible puzzles for {split}: {len(rows)}/{count}")
        pattern = min(
            active,
            key=lambda item: (
                counts[item],
                stable_fingerprint({"seed": seed, "split": split, "pattern": item}),
            ),
        )
        pool = pools[pattern]
        cursor = cursors[pattern]
        selected: tuple[tuple[tuple[int, ...], int], tuple[int, ...]] | None = None
        while cursor < len(pool):
            candidate = pool[cursor]
            cursor += 1
            key = candidate[0]
            if key in forbidden_keys:
                continue
            if not allow_cross_pattern_key_reuse and key in split_keys:
                continue
            selected = candidate
            break
        cursors[pattern] = cursor
        if selected is None:
            active.remove(pattern)
            continue
        key, number_tuple = selected
        numbers = list(number_tuple)
        target = int(key[1])
        expression = _instantiate_pattern(templates[pattern], numbers)
        check = verify_expression(expression, numbers, target)
        if not check["correct"] or expression_structure(expression) != pattern:
            raise AssertionError(f"Internal pattern instance failed validation: {pattern}")
        split_keys.add(key)
        counts[pattern] += 1
        rows.append({
            "id": f"cd_{seed}_{split}_{len(rows):07d}",
            "numbers": numbers,
            "target": target,
            "prompt": make_prompt(numbers, target),
            "oracle": expression,
            "oracle_structure": pattern,
            "heldout_family_seed": family_seed,
            "split": split,
        })
    return rows, counts, split_keys


def generate_structural_splits(
    train_count: int,
    val_count: int,
    test_count: int,
    seed: int,
    n_numbers: int = 4,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Pattern-first generation with disjoint held-out derivative families.

    The structural machinery is Park-inspired. Because this project's frozen
    task range is much narrower (numbers 1..9 rather than 1..99), the generator
    uses deterministic capacity-supported pattern cores so each represented
    pattern receives an equal or near-equal quota. Training may reuse one prompt
    across distinct oracle patterns, as the reference implementation does across
    per-pattern files; prompt keys remain strictly disjoint across splits.
    """
    if n_numbers != 4:
        raise ValueError("The registered EXT-C/E8 protocol is frozen to four-number Countdown")
    templates, _ = pattern_family_catalog()
    val_family_seed, test_family_seed, val_family_patterns, test_family_patterns = (
        choose_disjoint_holdout_families(seed, val_count=val_count, test_count=test_count)
    )
    all_patterns = set(templates)
    feasible_patterns = {
        pattern
        for pattern, template in templates.items()
        if _feasible_pattern_instances(pattern, template)
    }
    train_candidates = feasible_patterns - val_family_patterns - test_family_patterns
    if not train_candidates or not val_family_patterns or not test_family_patterns:
        raise RuntimeError("Invalid pattern-family partition")

    val_patterns = set(_select_capacity_balanced_patterns(
        sorted(val_family_patterns), val_count, templates, seed + 202, min_patterns=3
    ))
    test_patterns = set(_select_capacity_balanced_patterns(
        sorted(test_family_patterns), test_count, templates, seed + 303, min_patterns=3
    ))
    forbidden: set[tuple[tuple[int, ...], int]] = set()
    val_rows, val_counts, val_keys = _allocate_balanced_pattern_rows(
        "val", val_count, sorted(val_patterns), templates, seed + 202,
        forbidden, val_family_seed, allow_cross_pattern_key_reuse=False,
    )
    forbidden.update(val_keys)
    test_rows, test_counts, test_keys = _allocate_balanced_pattern_rows(
        "test", test_count, sorted(test_patterns), templates, seed + 303,
        forbidden, test_family_seed, allow_cross_pattern_key_reuse=False,
    )
    forbidden.update(test_keys)
    train_patterns = set(_select_capacity_balanced_patterns(
        sorted(train_candidates), train_count, templates, seed + 101,
        min_patterns=16, forbidden_keys=forbidden,
    ))
    train_rows, train_counts, train_keys = _allocate_balanced_pattern_rows(
        "train", train_count, sorted(train_patterns), templates, seed + 101,
        forbidden, None, allow_cross_pattern_key_reuse=True,
    )

    train_set = {row["oracle_structure"] for row in train_rows}
    val_set = {row["oracle_structure"] for row in val_rows}
    test_set = {row["oracle_structure"] for row in test_rows}
    if train_set & val_set or train_set & test_set or val_set & test_set:
        raise AssertionError("canonical pattern split overlap")
    if train_keys & val_keys or train_keys & test_keys or val_keys & test_keys:
        raise AssertionError("problem-key leakage across splits")

    def balance_summary(counts: Counter[str]) -> dict[str, Any]:
        values = list(counts.values())
        return {
            "represented_patterns": len(values),
            "minimum": min(values) if values else 0,
            "maximum": max(values) if values else 0,
            "max_minus_min": (max(values) - min(values)) if values else 0,
        }

    manifest = {
        "protocol": "park_inspired_pattern_first_family_holdout_capacity_audited",
        "terminology": "held-out canonical pattern-family generalization",
        "numeric_range": {"numbers_inclusive": [1, 9], "integer_target_inclusive": [5, 100]},
        "seed": seed,
        "n_numbers": n_numbers,
        "train_examples": len(train_rows),
        "val_examples": len(val_rows),
        "test_examples": len(test_rows),
        "canonical_patterns_total": len(all_patterns),
        "feasible_patterns_under_numeric_range": len(feasible_patterns),
        "train_candidate_patterns": sorted(train_candidates),
        "train_patterns": sorted(train_patterns),
        "val_holdout_family_patterns": sorted(val_family_patterns),
        "test_holdout_family_patterns": sorted(test_family_patterns),
        "val_patterns": sorted(val_patterns),
        "test_patterns": sorted(test_patterns),
        "val_family_seed": val_family_seed,
        "test_family_seed": test_family_seed,
        "structure_sets_disjoint": True,
        "problem_keys_disjoint": True,
        "cross_split_problem_keys_disjoint": True,
        "within_split_unique_problem_keys": {
            "train": len(train_keys) == len(train_rows),
            "val": len(val_keys) == len(val_rows),
            "test": len(test_keys) == len(test_rows),
        },
        "training_cross_pattern_prompt_reuse": len(train_keys) < len(train_rows),
        "negative_training_allowed_patterns": sorted(train_patterns),
        "per_pattern_counts": {
            "train": dict(sorted(train_counts.items())),
            "val": dict(sorted(val_counts.items())),
            "test": dict(sorted(test_counts.items())),
        },
        "balance_summary": {
            "train": balance_summary(train_counts),
            "val": balance_summary(val_counts),
            "test": balance_summary(test_counts),
        },
        "capacity_audit": {
            "reason": (
                "The frozen 1..9 number range cannot support equal quotas for all 96 "
                "canonical patterns; deterministic capacity-supported cores are used."
            ),
            "excluded_feasible_train_patterns": sorted(train_candidates - train_patterns),
            "excluded_val_family_patterns": sorted(val_family_patterns - val_patterns),
            "excluded_test_family_patterns": sorted(test_family_patterns - test_patterns),
        },
        "review_safety_note": (
            "Uses canonicalization, pattern-first capacity-audited balancing, and family "
            "holdout only; does not treat a subtree as proof of a latent skill or call "
            "the split OOD."
        ),
    }
    return train_rows, val_rows, test_rows, manifest

def random_expression(rng: np.random.Generator, numbers: list[int]) -> tuple[str, Fraction]:
    pool: list[tuple[str, Fraction]] = [(str(n), Fraction(n, 1)) for n in numbers]
    rng.shuffle(pool)
    while len(pool) > 1:
        i, j = rng.choice(len(pool), size=2, replace=False)
        left = pool[max(i, j)]
        right = pool[min(i, j)]
        del pool[max(i, j)]
        del pool[min(i, j)]
        op = str(rng.choice(OPS))
        if op == "/" and right[1] == 0:
            op = "+"
        if op == "+":
            val = left[1] + right[1]
        elif op == "-":
            val = left[1] - right[1]
        elif op == "*":
            val = left[1] * right[1]
        else:
            val = left[1] / right[1]
        pool.append((f"({left[0]} {op} {right[0]})", val))
    return pool[0]


def make_prompt(numbers: Sequence[int], target: int) -> str:
    return (
        f"Numbers: {', '.join(map(str, numbers))}\n"
        f"Target: {target}\n"
        "Return only a valid expression using every number exactly once."
    )


def generate_examples(count: int, seed: int, n_numbers: int = 4) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    seen: set[tuple[tuple[int, ...], int]] = set()
    rows: list[dict[str, Any]] = []
    attempts = 0
    while len(rows) < count:
        attempts += 1
        if attempts > count * 500:
            raise RuntimeError("Could not generate enough unique problems")
        numbers = rng.integers(1, 10, size=n_numbers).tolist()
        expression, value = random_expression(rng, numbers.copy())
        if value.denominator != 1:
            continue
        target = int(value)
        if not (5 <= target <= 100):
            continue
        key = (tuple(sorted(numbers)), target)
        if key in seen:
            continue
        check = verify_expression(expression, numbers, target)
        if not check["correct"]:
            continue
        seen.add(key)
        rows.append(
            {
                "id": f"cd_{seed}_{len(rows):07d}",
                "numbers": numbers,
                "target": target,
                "prompt": make_prompt(numbers, target),
                "oracle": expression,
            }
        )
    return rows


def chat_prompt(tokenizer: Any, user_prompt: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )


def load_tokenizer(model_path: str) -> Any:
    require_hf_stack()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_model(
    model_path: str,
    adapter_path: str | None,
    trainable_adapter: bool,
    load_in_4bit: bool,
    dtype: str,
    gradient_checkpointing: bool,
) -> Any:
    require_hf_stack()
    if dtype == "auto":
        use_bf16 = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    else:
        use_bf16 = dtype == "bf16"
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": compute_dtype,
    }
    if torch.cuda.is_available():
        kwargs["device_map"] = {"": int(os.environ.get("LOCAL_RANK", "0"))}
    if load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    if load_in_4bit:
        model = prepare_model_for_kbit_training(model)
    if adapter_path:
        model = PeftModel.from_pretrained(
            model, adapter_path, is_trainable=trainable_adapter
        )
    elif trainable_adapter:
        config = LoraConfig(
            r=32,
            lora_alpha=64,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )
        model = get_peft_model(model, config)
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model.config.use_cache = False
    return model


@dataclass
class EncodedExample:
    input_ids: list[int]
    labels: list[int]


def encode_prompt_completion(
    tokenizer: Any, prompt: str, completion: str, max_length: int
) -> EncodedExample:
    prefix = chat_prompt(tokenizer, prompt)
    completion = clean_expression(completion) + tokenizer.eos_token
    prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(prefix + completion, add_special_tokens=False)["input_ids"]
    full_ids = full_ids[:max_length]
    prefix_len = min(len(prefix_ids), len(full_ids))
    labels = [-100] * prefix_len + full_ids[prefix_len:]
    if all(x == -100 for x in labels):
        raise ValueError("Completion was truncated; increase --max_length")
    return EncodedExample(full_ids, labels)


class SFTDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], tokenizer: Any, max_length: int):
        self.items = [
            encode_prompt_completion(tokenizer, r["prompt"], r["oracle"], max_length)
            for r in rows
        ]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> EncodedExample:
        return self.items[index]


class OfflineDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], tokenizer: Any, max_length: int):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        return {
            "positive": encode_prompt_completion(
                self.tokenizer, row["prompt"], row["positive"], self.max_length
            ),
            "near": encode_prompt_completion(
                self.tokenizer, row["prompt"], row["near_negative"], self.max_length
            ),
            "far": encode_prompt_completion(
                self.tokenizer, row["prompt"], row["far_negative"], self.max_length
            ),
        }


def pad_encoded(items: list[EncodedExample], pad_id: int) -> dict[str, torch.Tensor]:
    max_len = max(len(x.input_ids) for x in items)
    ids, labels, masks = [], [], []
    for item in items:
        n = len(item.input_ids)
        ids.append(item.input_ids + [pad_id] * (max_len - n))
        labels.append(item.labels + [-100] * (max_len - n))
        masks.append([1] * n + [0] * (max_len - n))
    return {
        "input_ids": torch.tensor(ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(masks, dtype=torch.long),
    }


def make_sft_collator(pad_id: int):
    def collate(batch: list[EncodedExample]) -> dict[str, torch.Tensor]:
        return pad_encoded(batch, pad_id)
    return collate


def make_offline_collator(pad_id: int):
    def collate(batch: list[dict[str, Any]]) -> dict[str, dict[str, torch.Tensor]]:
        return {
            key: pad_encoded([x[key] for x in batch], pad_id)
            for key in ("positive", "near", "far")
        }
    return collate


def move_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def completion_stats(model: Any, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        use_cache=False,
    )
    logits = out.logits[:, :-1, :].float()
    labels = batch["labels"][:, 1:]
    mask = labels.ne(-100)
    safe_labels = labels.masked_fill(~mask, 0)
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    token_lp = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    lengths = mask.sum(-1).clamp_min(1)
    seq_lp = (token_lp * mask).sum(-1) / lengths
    token_entropy = -(probs * log_probs).sum(-1)
    entropy = (token_entropy * mask).sum(-1) / lengths
    # Direct-logit score norm ||e_y - p||_2. It is bounded for categorical logits.
    py = probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    p2 = probs.square().sum(-1)
    token_score = torch.sqrt(torch.clamp(1.0 - 2.0 * py + p2, min=0.0))
    score = (token_score * mask).sum(-1) / lengths
    return {
        "seq_lp": seq_lp,
        "entropy": entropy,
        "score": score,
        "token_lp": token_lp,
        "token_mask": mask,
        "token_score": token_score,
        "lengths": lengths,
    }


def weighted_sequence_logprob(
    stats: dict[str, torch.Tensor], token_weights: torch.Tensor
) -> torch.Tensor:
    mask = stats["token_mask"].to(token_weights.dtype)
    return (stats["token_lp"] * token_weights * mask).sum(-1) / stats["lengths"]


@torch.no_grad()
def generate_outputs(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    num_return_sequences: int = 1,
) -> list[list[str]]:
    model.eval()
    rendered = [chat_prompt(tokenizer, p) for p in prompts]
    # Decoder-only batched generation must left-pad; training remains right-padded.
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    batch = tokenizer(rendered, return_tensors="pt", padding=True, add_special_tokens=False)
    tokenizer.padding_side = old_padding_side
    device = next(model.parameters()).device
    batch = {k: v.to(device) for k, v in batch.items()}
    generate_kwargs: dict[str, Any] = {
        **batch,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "num_return_sequences": num_return_sequences,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "use_cache": True,
    }
    if do_sample:
        generate_kwargs.update({"temperature": temperature, "top_p": top_p})
    output = model.generate(**generate_kwargs)
    prompt_len = batch["input_ids"].shape[1]
    decoded = tokenizer.batch_decode(output[:, prompt_len:], skip_special_tokens=True)
    grouped: list[list[str]] = []
    for i in range(len(prompts)):
        start = i * num_return_sequences
        grouped.append(decoded[start : start + num_return_sequences])
    return grouped


@torch.no_grad()
def evaluate_rows(
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    batch_size: int,
    max_new_tokens: int,
    pass_k: int,
    seed: int,
    known_structures: set[str] | None = None,
) -> dict[str, float]:
    seed_all(seed)
    was_training = bool(model.training)
    successes: list[float] = []
    valid: list[float] = []
    sampled_successes: list[float] = []
    greedy_unseen_successes: list[float] = []
    pass_unseen_successes: list[float] = []
    observed_correct_structures: set[str] = set()
    target_structures = {
        row.get("oracle_structure") or expression_structure(row["oracle"])
        for row in rows
    }
    heldout_targets = (
        target_structures - known_structures if known_structures is not None else set()
    )
    heldout_pattern_attempts = 0
    heldout_pattern_correct = 0

    for start_index in range(0, len(rows), batch_size):
        chunk = rows[start_index : start_index + batch_size]
        prompts = [row["prompt"] for row in chunk]
        greedy = generate_outputs(
            model, tokenizer, prompts, max_new_tokens, False, 1.0, 1.0, 1
        )
        samples = (
            generate_outputs(
                model, tokenizer, prompts, max_new_tokens, True, 0.8, 0.95, pass_k
            )
            if pass_k > 1
            else greedy
        )
        for row, greedy_outputs, sampled_outputs in zip(chunk, greedy, samples):
            greedy_check = verify_expression(
                greedy_outputs[0], row["numbers"], row["target"]
            )
            successes.append(float(greedy_check["correct"]))
            valid.append(float(greedy_check["valid_format"] and greedy_check["uses_numbers"]))
            greedy_unseen = False
            if greedy_check["valid_format"] and greedy_check["uses_numbers"]:
                try:
                    pattern = expression_structure(greedy_check["expression"])
                    if pattern in heldout_targets:
                        heldout_pattern_attempts += 1
                        heldout_pattern_correct += int(greedy_check["correct"])
                    if greedy_check["correct"]:
                        observed_correct_structures.add(pattern)
                        greedy_unseen = (
                            known_structures is not None and pattern not in known_structures
                        )
                except Exception:
                    pass
            greedy_unseen_successes.append(float(greedy_unseen))

            any_correct = False
            any_unseen_correct = False
            for output_text in sampled_outputs:
                check = verify_expression(output_text, row["numbers"], row["target"])
                if not (check["valid_format"] and check["uses_numbers"]):
                    continue
                try:
                    pattern = expression_structure(check["expression"])
                except Exception:
                    continue
                if pattern in heldout_targets:
                    heldout_pattern_attempts += 1
                    heldout_pattern_correct += int(check["correct"])
                if not check["correct"]:
                    continue
                any_correct = True
                observed_correct_structures.add(pattern)
                if known_structures is not None and pattern not in known_structures:
                    any_unseen_correct = True
            sampled_successes.append(float(any_correct))
            pass_unseen_successes.append(float(any_unseen_correct))

    heldout_correct_patterns = observed_correct_structures & heldout_targets
    metrics = {
        "greedy_success": float(np.mean(successes)),
        "pass_at_k": float(np.mean(sampled_successes)),
        "valid_rate": float(np.mean(valid)),
        "greedy_unseen_structure_success": float(np.mean(greedy_unseen_successes)),
        "pass_at_k_unseen_structure": float(np.mean(pass_unseen_successes)),
        "unique_correct_structures": float(len(observed_correct_structures)),
        "heldout_pattern_coverage": (
            float(len(heldout_correct_patterns) / len(heldout_targets))
            if heldout_targets else 0.0
        ),
        "heldout_pattern_precision": (
            float(heldout_pattern_correct / heldout_pattern_attempts)
            if heldout_pattern_attempts else 0.0
        ),
        "heldout_pattern_attempts": float(heldout_pattern_attempts),
        "heldout_patterns_observed_correct": float(len(heldout_correct_patterns)),
        "heldout_patterns_total": float(len(heldout_targets)),
        "n_eval": float(len(rows)),
    }
    if was_training:
        model.train()
    return metrics


def _existing_ancestor(path: Path) -> Path:
    current = path.resolve()
    while not current.exists() and current.parent != current:
        current = current.parent
    return current


def ensure_checkpoint_output_is_local_or_ignored(path: str | Path) -> None:
    """Prevent accidental Git tracking of adapters while allowing external paths."""
    target = Path(path).expanduser().resolve()
    ancestor = _existing_ancestor(target)
    try:
        root = Path(subprocess.check_output(
            ["git", "-C", str(ancestor), "rev-parse", "--show-toplevel"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()).resolve()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return
    try:
        relative = target.relative_to(root)
    except ValueError:
        return
    check = subprocess.run(
        ["git", "-C", str(root), "check-ignore", "--quiet", str(relative)],
        check=False,
    )
    if check.returncode != 0:
        raise RuntimeError(
            f"Checkpoint output {target} is inside the Git repository but is not ignored. "
            "Use an external server-local work_dir or an ignored outputs/runs path."
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_inventory(path: str | Path, kind: str, step: int) -> dict[str, Any]:
    root = Path(path).resolve()
    files = []
    for item in sorted(root.rglob("*")):
        if item.is_file():
            files.append({
                "relative_path": str(item.relative_to(root)),
                "size_bytes": item.stat().st_size,
                "sha256": _sha256_file(item),
            })
    adapter_config_path = root / "adapter_config.json"
    adapter_config = (
        json.loads(adapter_config_path.read_text())
        if adapter_config_path.exists()
        else None
    )
    return {
        "kind": kind,
        "step": int(step),
        "local_only": True,
        "path": str(root),
        "adapter_config": adapter_config,
        "files": files,
        "total_size_bytes": sum(item["size_bytes"] for item in files),
    }


def save_local_adapter_checkpoint(
    model: Any,
    tokenizer: Any,
    path: str | Path,
    kind: str,
    step: int,
) -> dict[str, Any]:
    destination = Path(path)
    ensure_checkpoint_output_is_local_or_ignored(destination)
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(destination)
    tokenizer.save_pretrained(destination)
    return checkpoint_inventory(destination, kind, step)


def _trainable_parameters_finite(parameters: Sequence[torch.nn.Parameter]) -> bool:
    return all(bool(torch.isfinite(parameter.detach()).all()) for parameter in parameters)


def cmd_generate(args: argparse.Namespace) -> None:
    train_rows, val_rows, test_rows, manifest = generate_structural_splits(
        args.train, args.val, args.test, args.seed, args.n_numbers
    )
    write_jsonl(args.train_out, train_rows)
    write_jsonl(args.val_out, val_rows)
    write_jsonl(args.test_out, test_rows)
    manifest_path = Path(args.manifest_out) if args.manifest_out else Path(args.train_out).parent / "split_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"generated": manifest, "manifest": str(manifest_path)}, indent=2))


def cmd_sft(args: argparse.Namespace) -> None:
    """Minimal LoRA SFT fallback with local best/terminal/last-finite audit."""
    seed_all(args.seed)
    tokenizer = load_tokenizer(args.model_path)
    rows = read_jsonl(args.train_data)
    val_rows = read_jsonl(args.val_data)
    dataset = SFTDataset(rows, tokenizer, args.max_length)
    loader = DataLoader(
        dataset,
        batch_size=args.micro_batch,
        shuffle=True,
        collate_fn=make_sft_collator(tokenizer.pad_token_id),
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    model = load_model(
        args.model_path,
        adapter_path=None,
        trainable_adapter=True,
        load_in_4bit=args.load_in_4bit,
        dtype=args.dtype,
        gradient_checkpointing=True,
    )
    device = next(model.parameters()).device
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    updates_per_epoch = math.ceil(len(loader) / args.grad_accum)
    total_updates = max(1, updates_per_epoch * args.epochs)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, max(1, int(total_updates * args.warmup_ratio)), total_updates
    )
    out_dir = Path(args.output_dir)
    ensure_checkpoint_output_is_local_or_ignored(out_dir)
    best_dir = out_dir / "best_adapter"
    terminal_dir = out_dir / "terminal_adapter"
    rolling_dir = out_dir / ".last_finite_adapter_work"
    last_finite_dir = out_dir / "last_finite_adapter"
    out_dir.mkdir(parents=True, exist_ok=True)
    best_value = -float("inf")
    best_epoch = -1
    eval_rows: list[dict[str, Any]] = []
    checkpoint_records: list[dict[str, Any]] = []
    numerical_failure: str | None = None
    stop_reason = "max_epochs"
    global_step = 0
    model.train()
    optimizer.zero_grad(set_to_none=True)
    rolling_record = save_local_adapter_checkpoint(
        model, tokenizer, rolling_dir, "rolling_last_finite", global_step
    )

    for epoch in range(args.epochs):
        running_loss = 0.0
        micro_count = 0
        for batch_index, batch in enumerate(loader):
            batch = move_to_device(batch, device)
            raw_loss = model(**batch, use_cache=False).loss
            if not bool(torch.isfinite(raw_loss)):
                numerical_failure = f"nonfinite_loss_at_update_{global_step + 1}"
                stop_reason = numerical_failure
                break
            (raw_loss / args.grad_accum).backward()
            running_loss += float(raw_loss.detach())
            micro_count += 1
            if (batch_index + 1) % args.grad_accum == 0 or batch_index + 1 == len(loader):
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
                if not bool(torch.isfinite(grad_norm)):
                    numerical_failure = f"nonfinite_gradient_at_update_{global_step + 1}"
                    stop_reason = numerical_failure
                    break
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if not _trainable_parameters_finite(trainable):
                    numerical_failure = f"nonfinite_parameters_at_update_{global_step}"
                    stop_reason = numerical_failure
                    break
                if global_step % args.log_every == 0:
                    print(json.dumps({
                        "stage": "sft",
                        "epoch": epoch + 1,
                        "update": global_step,
                        "loss": running_loss / max(micro_count, 1),
                        "lr": scheduler.get_last_lr()[0],
                    }))
                    running_loss = 0.0
                    micro_count = 0
        if numerical_failure:
            break
        metrics = evaluate_rows(
            model,
            tokenizer,
            val_rows[: args.eval_examples],
            args.eval_batch,
            args.max_new_tokens,
            args.pass_k,
            args.eval_seed,
        )
        row = {"epoch": epoch + 1, "update": global_step, **metrics}
        eval_rows.append(row)
        print("SFT_EVAL", json.dumps(row))
        rolling_record = save_local_adapter_checkpoint(
            model, tokenizer, rolling_dir, "rolling_last_finite", global_step
        )
        value = float(metrics[args.selection_metric])
        if value > best_value + args.selection_delta:
            best_value = value
            best_epoch = epoch + 1
            checkpoint_records = [
                record for record in checkpoint_records if record["kind"] != "best"
            ]
            checkpoint_records.append(save_local_adapter_checkpoint(
                model, tokenizer, best_dir, "best", global_step
            ))
        model.train()

    if numerical_failure:
        if last_finite_dir.exists():
            shutil.rmtree(last_finite_dir)
        shutil.move(str(rolling_dir), str(last_finite_dir))
        rolling_record["kind"] = "last_finite"
        rolling_record["path"] = str(last_finite_dir.resolve())
        checkpoint_records.append(rolling_record)
    else:
        checkpoint_records.append(save_local_adapter_checkpoint(
            model, tokenizer, terminal_dir, "terminal", global_step
        ))
        if rolling_dir.exists():
            shutil.rmtree(rolling_dir)
    if best_epoch < 0:
        best_value = float("nan")
    with (out_dir / "sft_metrics.csv").open("w", newline="") as handle:
        if eval_rows:
            writer = csv.DictWriter(handle, fieldnames=list(eval_rows[0].keys()))
            writer.writeheader()
            writer.writerows(eval_rows)
    manifest = {
        **vars(args),
        "source_provenance": source_provenance(),
        "best_epoch": best_epoch,
        "best_value": best_value,
        "terminal_step": global_step if not numerical_failure else None,
        "numerical_failure": numerical_failure,
        "stop_reason": stop_reason,
        "checkpoint_policy": "server-local adapters only; binaries must not enter Git/artifact packages",
        "checkpoints": checkpoint_records,
        "result_status": "pilot",
    }
    (out_dir / "sft_manifest.json").write_text(json.dumps(manifest, indent=2))
    (out_dir / "checkpoint_manifest.json").write_text(json.dumps({
        "local_only": True,
        "model_path": args.model_path,
        "source_provenance": source_provenance(),
        "checkpoints": checkpoint_records,
    }, indent=2))
    if numerical_failure:
        raise RuntimeError(f"SFT stopped with {numerical_failure}; last finite adapter was preserved")
    if best_epoch < 0:
        raise RuntimeError("SFT did not produce a valid best checkpoint")


def score_completion(
    model: Any, tokenizer: Any, prompt: str, completion: str, max_length: int
) -> float:
    encoded = encode_prompt_completion(tokenizer, prompt, completion, max_length)
    batch = pad_encoded([encoded], tokenizer.pad_token_id)
    batch = move_to_device(batch, next(model.parameters()).device)
    with torch.no_grad():
        return float((-completion_stats(model, batch)["seq_lp"])[0])


def score_completions_batch(
    model: Any,
    tokenizer: Any,
    prompt_completion: list[tuple[str, str]],
    max_length: int,
    batch_size: int = 16,
) -> list[float]:
    """Mean token surprisal for prompt/completion pairs, batched for speed."""
    out: list[float] = []
    device = next(model.parameters()).device
    for start in range(0, len(prompt_completion), batch_size):
        chunk = prompt_completion[start : start + batch_size]
        encoded = [
            encode_prompt_completion(tokenizer, prompt, completion, max_length)
            for prompt, completion in chunk
        ]
        batch = move_to_device(pad_encoded(encoded, tokenizer.pad_token_id), device)
        with torch.no_grad():
            values = -completion_stats(model, batch)["seq_lp"]
        out.extend(float(x) for x in values.detach().cpu())
    return out


def mutate_expression(expression: str, rng: random.Random) -> str:
    indices = [i for i, char in enumerate(expression) if char in OPS]
    if not indices:
        return expression + " + 1"
    index = rng.choice(indices)
    choices = [op for op in OPS if op != expression[index]]
    return expression[:index] + rng.choice(choices) + expression[index + 1 :]


def make_valid_wrong_expression(
    row: dict[str, Any],
    rng: random.Random,
    avoid: set[str] | None = None,
    allowed_patterns: set[str] | None = None,
    max_attempts: int = 400,
) -> str:
    """Create a legal reward-zero expression without leaking held-out families."""
    avoid = avoid or set()
    candidates = [mutate_expression(row["oracle"], rng)]
    np_rng = np.random.default_rng(rng.randrange(2**31))
    for _ in range(max_attempts):
        if candidates:
            expression = candidates.pop()
        else:
            expression, _ = random_expression(np_rng, list(row["numbers"]))
        check = verify_expression(expression, row["numbers"], row["target"])
        if not (
            check["valid_format"]
            and check["uses_numbers"]
            and not check["correct"]
            and check["expression"] not in avoid
        ):
            continue
        try:
            pattern = expression_structure(check["expression"])
        except Exception:
            continue
        if allowed_patterns is not None and pattern not in allowed_patterns:
            continue
        return check["expression"]
    raise RuntimeError(f"Could not construct an allowed valid wrong expression for {row['id']}")


def candidate_metadata(item: dict[str, Any], tokenizer: Any) -> dict[str, Any]:
    expression = item["expression"]
    value = item.get("value")
    return {
        **item,
        "text": expression,
        "structure": expression_structure(expression),
        "token_length": len(tokenizer(expression, add_special_tokens=False)["input_ids"]),
        "tree_depth": expression_tree_depth(expression),
        "value_error": (
            abs(float(value) - float(item["target"])) if value is not None else float("inf")
        ),
    }


def _pair_constraint_failures(
    near: dict[str, Any],
    far: dict[str, Any],
    min_surprisal_gap: float,
    max_token_length_diff: int,
    max_tree_depth_diff: int,
    max_value_error_ratio: float,
) -> set[str]:
    failures: set[str] = set()
    gap = float(far["surprisal"]) - float(near["surprisal"])
    if gap < min_surprisal_gap:
        failures.add("surprisal_gap")
    if abs(int(far["token_length"]) - int(near["token_length"])) > max_token_length_diff:
        failures.add("token_length")
    if abs(int(far["tree_depth"]) - int(near["tree_depth"])) > max_tree_depth_diff:
        failures.add("tree_depth")
    near_error = max(float(near["value_error"]), 1e-8)
    far_error = max(float(far["value_error"]), 1e-8)
    if max(near_error, far_error) / min(near_error, far_error) > max_value_error_ratio:
        failures.add("value_error")
    return failures


def select_matched_negative_pair(
    candidates: list[dict[str, Any]],
    min_surprisal_gap: float,
    max_token_length_diff: int,
    max_tree_depth_diff: int,
    max_value_error_ratio: float,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, bool]:
    """Return a genuinely matched pair; never force unmatched extrema into training."""
    ordered = sorted(candidates, key=lambda item: float(item["surprisal"]))
    best: tuple[float, dict[str, Any], dict[str, Any]] | None = None
    for index, near in enumerate(ordered):
        for far in ordered[index + 1 :]:
            failures = _pair_constraint_failures(
                near,
                far,
                min_surprisal_gap,
                max_token_length_diff,
                max_tree_depth_diff,
                max_value_error_ratio,
            )
            if failures:
                continue
            gap = float(far["surprisal"]) - float(near["surprisal"])
            if best is None or gap > best[0]:
                best = (gap, near, far)
    if best is None:
        return None, None, False
    return best[1], best[2], True


def _score_new_candidates(
    model: Any,
    tokenizer: Any,
    row: dict[str, Any],
    texts: Sequence[str],
    seen: set[str],
    allowed_patterns: set[str],
    max_length: int,
    score_batch_size: int,
    diagnostics: Counter[str],
) -> list[dict[str, Any]]:
    accepted: list[dict[str, Any]] = []
    for text in texts:
        check = verify_expression(text, row["numbers"], row["target"])
        expression = check["expression"]
        if not expression or expression in seen:
            diagnostics["duplicate_or_empty"] += 1
            continue
        seen.add(expression)
        if not (check["valid_format"] and check["uses_numbers"]):
            diagnostics["invalid_or_wrong_numbers"] += 1
            continue
        try:
            structure = expression_structure(expression)
        except Exception:
            diagnostics["structure_parse_failure"] += 1
            continue
        if structure not in allowed_patterns:
            diagnostics["heldout_pattern_rejected"] += 1
            continue
        accepted.append({**check, "target": row["target"], "structure": structure})
    if not accepted:
        return []
    surprisals = score_completions_batch(
        model,
        tokenizer,
        [(row["prompt"], item["expression"]) for item in accepted],
        max_length,
        score_batch_size,
    )
    for item, surprisal in zip(accepted, surprisals):
        item["surprisal"] = float(surprisal)
    return accepted


def _pair_failure_summary(
    candidates: list[dict[str, Any]],
    args: argparse.Namespace,
) -> Counter[str]:
    summary: Counter[str] = Counter()
    if len(candidates) < 2:
        summary["candidate_count"] += 1
        return summary
    ordered = sorted(candidates, key=lambda item: float(item["surprisal"]))
    all_failures: list[set[str]] = []
    for index, near in enumerate(ordered):
        for far in ordered[index + 1 :]:
            all_failures.append(_pair_constraint_failures(
                near,
                far,
                args.min_surprisal_gap,
                args.max_token_length_diff,
                args.max_tree_depth_diff,
                args.max_value_error_ratio,
            ))
    if not all_failures:
        summary["candidate_count"] += 1
    else:
        for name in sorted(set.union(*all_failures)):
            if all(name in failures for failures in all_failures):
                summary[name] += 1
    return summary


def cmd_build_offline(args: argparse.Namespace) -> None:
    seed_all(args.seed)
    if args.batch_size < 1:
        raise ValueError("--batch_size must be positive")
    tokenizer = load_tokenizer(args.model_path)
    reference_adapter = args.reference_adapter or getattr(args, "sft_adapter", None)
    model = load_model(
        args.model_path,
        reference_adapter,
        trainable_adapter=False,
        load_in_4bit=args.load_in_4bit,
        dtype=args.dtype,
        gradient_checkpointing=False,
    )
    model.eval()
    rows = read_jsonl(args.input_data)
    split_manifest = json.loads(Path(args.split_manifest).read_text())
    allowed_patterns = set(split_manifest["negative_training_allowed_patterns"])
    if not allowed_patterns:
        raise RuntimeError("Split manifest contains no allowed training patterns")
    target_examples = len(rows) if args.max_examples <= 0 else args.max_examples
    rng = random.Random(args.seed)
    output_rows: list[dict[str, Any]] = []
    diagnostics: Counter[str] = Counter()
    candidate_counts: list[int] = []

    # Initial rollouts are generated in batches for throughput. Only prompts that
    # fail a real matched-pair gate are resampled individually, so batching never
    # weakens the pair constraints or silently inserts an unmatched fallback.
    for start_index in range(0, len(rows), args.batch_size):
        if len(output_rows) >= target_examples:
            break
        chunk = rows[start_index : start_index + args.batch_size]
        initial_groups = generate_outputs(
            model,
            tokenizer,
            [row["prompt"] for row in chunk],
            args.max_new_tokens,
            True,
            args.temperature,
            args.top_p,
            args.rollouts,
        )
        diagnostics["initial_generation_batches"] += 1
        diagnostics["initial_generation_prompts"] += len(chunk)

        for row, initial_generated in zip(chunk, initial_groups):
            if len(output_rows) >= target_examples:
                break
            diagnostics["attempted_rows"] += 1
            seen: set[str] = set()
            evaluated: list[dict[str, Any]] = []
            matched_round: int | None = None
            near_item: dict[str, Any] | None = None
            far_item: dict[str, Any] | None = None

            for round_index in range(args.pair_resample_rounds + 1):
                if round_index == 0:
                    generated = initial_generated
                else:
                    generated = generate_outputs(
                        model,
                        tokenizer,
                        [row["prompt"]],
                        args.max_new_tokens,
                        True,
                        args.temperature,
                        args.top_p,
                        args.rollouts,
                    )[0]
                    diagnostics["resample_generation_calls"] += 1
                evaluated.extend(_score_new_candidates(
                    model,
                    tokenizer,
                    row,
                    generated,
                    seen,
                    allowed_patterns,
                    args.max_length,
                    args.score_batch_size,
                    diagnostics,
                ))

                valid_wrong = [item for item in evaluated if not item["correct"]]
                # Add legal synthetic candidates only inside the training pattern support.
                while len(valid_wrong) < args.min_negative_candidates:
                    try:
                        expression = make_valid_wrong_expression(
                            row,
                            rng,
                            avoid=seen,
                            allowed_patterns=allowed_patterns,
                        )
                    except RuntimeError:
                        diagnostics["synthetic_candidate_failure"] += 1
                        break
                    diagnostics["synthetic_negative"] += 1
                    evaluated.extend(_score_new_candidates(
                        model,
                        tokenizer,
                        row,
                        [expression],
                        seen,
                        allowed_patterns,
                        args.max_length,
                        args.score_batch_size,
                        diagnostics,
                    ))
                    valid_wrong = [item for item in evaluated if not item["correct"]]

                detailed = [candidate_metadata(item, tokenizer) for item in valid_wrong]
                near_item, far_item, matched = select_matched_negative_pair(
                    detailed,
                    args.min_surprisal_gap,
                    args.max_token_length_diff,
                    args.max_tree_depth_diff,
                    args.max_value_error_ratio,
                )
                if matched:
                    matched_round = round_index
                    break
                diagnostics["resample_rounds_used"] += int(
                    round_index < args.pair_resample_rounds
                )

            valid_wrong = [item for item in evaluated if not item["correct"]]
            candidate_counts.append(len(valid_wrong))
            if near_item is None or far_item is None or matched_round is None:
                diagnostics["dropped_unmatched"] += 1
                diagnostics.update({
                    f"drop_reason_{key}": value
                    for key, value in _pair_failure_summary(
                        [candidate_metadata(item, tokenizer) for item in valid_wrong], args
                    ).items()
                })
                continue

            correct = [item for item in evaluated if item["correct"]]
            oracle_structure = row.get("oracle_structure") or expression_structure(row["oracle"])
            same_structure_correct = [
                item for item in correct if item.get("structure") == oracle_structure
            ]
            if same_structure_correct:
                positive_item = min(
                    same_structure_correct, key=lambda item: float(item["surprisal"])
                )
                positive = positive_item["expression"]
                positive_surprisal = float(positive_item["surprisal"])
                diagnostics["sampled_positive"] += 1
            else:
                positive = row["oracle"]
                positive_surprisal = score_completions_batch(
                    model, tokenizer, [(row["prompt"], positive)], args.max_length, 1
                )[0]
                diagnostics["oracle_positive"] += 1

            diagnostics[
                "matched_initial" if matched_round == 0 else "matched_after_resample"
            ] += 1
            output_rows.append({
                **row,
                "positive": positive,
                "positive_base_surprisal": positive_surprisal,
                "near_negative": near_item["text"],
                "far_negative": far_item["text"],
                "near_structure": near_item["structure"],
                "far_structure": far_item["structure"],
                "near_base_surprisal": float(near_item["surprisal"]),
                "far_base_surprisal": float(far_item["surprisal"]),
                "surprisal_gap": float(
                    far_item["surprisal"] - near_item["surprisal"]
                ),
                "near_token_length": int(near_item["token_length"]),
                "far_token_length": int(far_item["token_length"]),
                "near_tree_depth": int(near_item["tree_depth"]),
                "far_tree_depth": int(far_item["tree_depth"]),
                "near_value_error": float(near_item["value_error"]),
                "far_value_error": float(far_item["value_error"]),
                "pair_matched": True,
                "matched_after_resample_round": matched_round,
            })
            if len(output_rows) % 25 == 0:
                print(f"built matched {len(output_rows)}/{target_examples}", flush=True)

    if len(output_rows) < target_examples:
        write_jsonl(args.output_data, output_rows)
        Path(str(args.output_data) + ".manifest.json").write_text(json.dumps({
            **vars(args),
            **dict(diagnostics),
            "reference_adapter": reference_adapter,
            "requested_examples": target_examples,
            "examples": len(output_rows),
            "status": "insufficient_matched_pairs",
        }, indent=2))
        raise RuntimeError(
            f"Only {len(output_rows)}/{target_examples} matched rows were constructed. "
            "Partial rows and diagnostics are preserved; increase input data or resampling."
        )
    if any(not row.get("pair_matched") for row in output_rows):
        raise AssertionError("Unmatched pairs must never enter the offline training data")
    heldout_leaks = [
        row for row in output_rows
        if row["near_structure"] not in allowed_patterns
        or row["far_structure"] not in allowed_patterns
    ]
    if heldout_leaks:
        raise AssertionError("Held-out canonical pattern leaked into training negatives")

    write_jsonl(args.output_data, output_rows)
    manifest = {
        **vars(args),
        "source_provenance": source_provenance(),
        **dict(diagnostics),
        "reference_adapter": reference_adapter,
        "requested_examples": target_examples,
        "examples": len(output_rows),
        "matched_pair_rate_in_saved_data": 1.0,
        "dropped_unmatched_rate": diagnostics["dropped_unmatched"]
        / max(diagnostics["attempted_rows"], 1),
        "mean_valid_wrong_candidates": (
            float(np.mean(candidate_counts)) if candidate_counts else 0.0
        ),
        "mean_surprisal_gap": float(np.mean([
            row["surprisal_gap"] for row in output_rows
        ])),
        "heldout_pattern_leaks": 0,
        "formal_interpretation": (
            "all saved rows are matched; dropped rows are reported, never silently trained"
        ),
    }
    Path(str(args.output_data) + ".manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )
    print(json.dumps(manifest, indent=2))

def cmd_train_method(args: argparse.Namespace) -> None:
    seed_all(args.seed)
    tokenizer = load_tokenizer(args.model_path)
    train_rows = read_jsonl(args.offline_data)
    if not train_rows or any(not row.get("pair_matched", False) for row in train_rows):
        raise RuntimeError("Method training requires a non-empty all-matched offline dataset")
    val_rows = read_jsonl(args.val_data)
    known_structures: set[str] | None = None
    if args.structure_reference_data:
        structure_rows = read_jsonl(args.structure_reference_data)
        known_structures = {
            row.get("oracle_structure") or expression_structure(row["oracle"])
            for row in structure_rows
        }
    dataset = OfflineDataset(train_rows, tokenizer, args.max_length)
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.micro_batch,
        shuffle=True,
        generator=generator,
        collate_fn=make_offline_collator(tokenizer.pad_token_id),
        num_workers=args.num_workers,
    )
    model = load_model(
        args.model_path,
        args.reference_adapter or args.sft_adapter,
        trainable_adapter=True,
        load_in_4bit=args.load_in_4bit,
        dtype=args.dtype,
        gradient_checkpointing=True,
    )
    device = next(model.parameters()).device
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, max(1, int(args.steps * args.warmup_ratio)), args.steps
    )
    calibrated_global_gamma: float | None = None
    if args.method == "global_matched":
        if not args.global_calibration_json:
            raise RuntimeError("global_matched requires --global_calibration_json")
        calibration = json.loads(Path(args.global_calibration_json).read_text())
        calibrated_global_gamma = float(calibration["global_gamma"])
        if not math.isfinite(calibrated_global_gamma) or calibrated_global_gamma <= 0:
            raise RuntimeError("Invalid calibrated global gamma")

    model.train()
    iterator = iter(loader)
    metrics_rows: list[dict[str, Any]] = []
    out_dir = Path(args.output_dir)
    ensure_checkpoint_output_is_local_or_ignored(out_dir)
    best_dir = out_dir / "best_adapter"
    terminal_dir = out_dir / "terminal_adapter"
    rolling_dir = out_dir / ".last_finite_adapter_work"
    last_finite_dir = out_dir / "last_finite_adapter"
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_records: list[dict[str, Any]] = []
    best_value = -float("inf")
    best_step = 0
    stale_checks = 0
    numerical_failure: str | None = None
    stop_reason = "max_steps"
    terminal_step: int | None = None
    last_gamma, last_weight = 1.0, 1.0

    initial_eval = evaluate_rows(
        model,
        tokenizer,
        val_rows[: args.eval_examples],
        args.eval_batch,
        args.max_new_tokens,
        args.pass_k,
        args.eval_seed,
        known_structures,
    )
    initial_gamma = calibrated_global_gamma if calibrated_global_gamma is not None else 1.0
    metrics_rows.append({
        "step": 0,
        "method": args.method,
        "gamma": initial_gamma,
        "weight": 1.0,
        **initial_eval,
    })
    best_value = float(initial_eval[args.selection_metric])
    checkpoint_records.append(save_local_adapter_checkpoint(
        model, tokenizer, best_dir, "best", 0
    ))
    rolling_record = save_local_adapter_checkpoint(
        model, tokenizer, rolling_dir, "rolling_last_finite", 0
    )

    stop_training = False
    for update_step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        log_accum: dict[str, float] = {
            "loss": 0.0,
            "gamma": 0.0,
            "weight": 0.0,
            "near_weight": 0.0,
            "far_weight": 0.0,
            "pos_lp": 0.0,
            "neg_lp": 0.0,
            "entropy": 0.0,
        }
        for _ in range(args.grad_accum):
            try:
                packed = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                packed = next(iterator)
            pos = completion_stats(model, move_to_device(packed["positive"], device))
            positive_lp = pos["seq_lp"].mean()

            if args.method == "positive_only":
                gamma = 0.0
                negative_lp = torch.zeros_like(positive_lp)
                near_weight_value = 0.0
                far_weight_value = 0.0
                mean_weight = 0.0
                raw_loss = -positive_lp
            else:
                near = completion_stats(model, move_to_device(packed["near"], device))
                far = completion_stats(model, move_to_device(packed["far"], device))
                near_seq_weights = torch.ones_like(near["seq_lp"])
                far_seq_weights = torch.ones_like(far["seq_lp"])
                far_token_weights = torch.ones_like(far["token_lp"])
                if args.method == "controlled_negative":
                    far_token_surprisal = -far["token_lp"].detach()
                    far_token_weights = torch.exp(
                        -args.exp_lambda
                        * F.relu(far_token_surprisal - args.surprisal_threshold)
                    )
                elif args.method in {"exp", "hybrid"}:
                    near_surprisal = -near["seq_lp"].detach()
                    far_surprisal = -far["seq_lp"].detach()
                    near_seq_weights = torch.exp(
                        -args.exp_lambda * F.relu(near_surprisal - args.surprisal_threshold)
                    )
                    far_seq_weights = torch.exp(
                        -args.exp_lambda * F.relu(far_surprisal - args.surprisal_threshold)
                    )

                gamma = 1.0
                if args.method == "global":
                    gamma = args.global_gamma
                elif args.method == "global_matched":
                    assert calibrated_global_gamma is not None
                    gamma = calibrated_global_gamma
                elif args.method in {"sbrc", "hybrid"}:
                    positive_budget = pos["score"].detach().mean()
                    negative_budget = args.alpha * (
                        args.near_mix * (near_seq_weights * near["score"].detach()).mean()
                        + args.far_mix * (far_seq_weights * far["score"].detach()).mean()
                    )
                    score_gamma = min(
                        1.0,
                        float(args.sbrc_kappa * positive_budget / (negative_budget + 1e-8)),
                    )
                    current_entropy = float(pos["entropy"].detach().mean())
                    entropy_gamma = (
                        1.0
                        if current_entropy >= args.entropy_floor
                        else max(0.0, current_entropy / max(args.entropy_floor, 1e-8))
                    )
                    gamma = min(score_gamma, entropy_gamma)

                if args.method == "controlled_negative":
                    near_lp = near["seq_lp"]
                    far_lp = weighted_sequence_logprob(far, far_token_weights)
                else:
                    near_lp = near_seq_weights * near["seq_lp"]
                    far_lp = far_seq_weights * far["seq_lp"]
                negative_lp = (
                    args.near_mix * near_lp.mean()
                    + args.far_mix * far_lp.mean()
                )
                raw_loss = -(positive_lp - args.alpha * gamma * negative_lp)
                if args.method == "entropy_bonus":
                    raw_loss = raw_loss - args.entropy_coef * pos["entropy"].mean()
                elif args.method == "target_entropy":
                    entropy_gap = F.relu(args.target_entropy - pos["entropy"].mean())
                    raw_loss = raw_loss + args.target_entropy_coef * entropy_gap.square()

                near_weight_value = float(near_seq_weights.detach().mean())
                far_weight_value = (
                    float(
                        (far_token_weights.detach() * far["token_mask"]).sum()
                        / far["token_mask"].sum().clamp_min(1)
                    )
                    if args.method == "controlled_negative"
                    else float(far_seq_weights.detach().mean())
                )
                mean_weight = (
                    args.near_mix * near_weight_value
                    + args.far_mix * far_weight_value
                )

            if not bool(torch.isfinite(raw_loss)):
                numerical_failure = f"nonfinite_loss_at_step_{update_step}"
                stop_reason = numerical_failure
                stop_training = True
                break
            (raw_loss / args.grad_accum).backward()
            log_accum["loss"] += float(raw_loss.detach()) / args.grad_accum
            log_accum["gamma"] += float(gamma) / args.grad_accum
            log_accum["weight"] += mean_weight / args.grad_accum
            log_accum["near_weight"] += near_weight_value / args.grad_accum
            log_accum["far_weight"] += far_weight_value / args.grad_accum
            log_accum["pos_lp"] += float(positive_lp.detach()) / args.grad_accum
            log_accum["neg_lp"] += float(negative_lp.detach()) / args.grad_accum
            log_accum["entropy"] += float(pos["entropy"].detach().mean()) / args.grad_accum
        if stop_training:
            break

        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
        if not bool(torch.isfinite(grad_norm)):
            numerical_failure = f"nonfinite_gradient_at_step_{update_step}"
            stop_reason = numerical_failure
            break
        optimizer.step()
        scheduler.step()
        if not _trainable_parameters_finite(trainable):
            numerical_failure = f"nonfinite_parameters_at_step_{update_step}"
            stop_reason = numerical_failure
            break
        last_gamma = log_accum["gamma"]
        last_weight = log_accum["weight"]

        if update_step % args.log_every == 0:
            print(json.dumps({"step": update_step, "method": args.method, **log_accum}))
        if update_step % args.eval_every == 0 or update_step == args.steps:
            metrics = evaluate_rows(
                model,
                tokenizer,
                val_rows[: args.eval_examples],
                args.eval_batch,
                args.max_new_tokens,
                args.pass_k,
                args.eval_seed,
                known_structures,
            )
            row = {
                "step": update_step,
                "method": args.method,
                "gamma": last_gamma,
                "weight": last_weight,
                **metrics,
            }
            metrics_rows.append(row)
            print("ARENA_EVAL", json.dumps(row))
            rolling_record = save_local_adapter_checkpoint(
                model, tokenizer, rolling_dir, "rolling_last_finite", update_step
            )
            value = float(metrics[args.selection_metric])
            if value > best_value + args.early_stop_delta:
                best_value = value
                best_step = update_step
                stale_checks = 0
                checkpoint_records = [
                    record for record in checkpoint_records if record["kind"] != "best"
                ]
                checkpoint_records.append(save_local_adapter_checkpoint(
                    model, tokenizer, best_dir, "best", update_step
                ))
            else:
                stale_checks += 1
            model.train()
            if update_step >= args.min_steps and stale_checks >= args.early_stop_patience:
                stop_reason = "early_stop_patience"
                terminal_step = update_step
                print(json.dumps({
                    "early_stop": True,
                    "step": update_step,
                    "best_step": best_step,
                    "best_value": best_value,
                }))
                stop_training = True
        if stop_training:
            break
        terminal_step = update_step

    if numerical_failure:
        if last_finite_dir.exists():
            shutil.rmtree(last_finite_dir)
        shutil.move(str(rolling_dir), str(last_finite_dir))
        rolling_record["kind"] = "last_finite"
        rolling_record["path"] = str(last_finite_dir.resolve())
        checkpoint_records.append(rolling_record)
        terminal_step = None
    else:
        if terminal_step is None:
            terminal_step = 0
        checkpoint_records.append(save_local_adapter_checkpoint(
            model, tokenizer, terminal_dir, "terminal", terminal_step
        ))
        if rolling_dir.exists():
            shutil.rmtree(rolling_dir)

    with (out_dir / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metrics_rows)
    manifest = {
        **vars(args),
        "source_provenance": source_provenance(),
        "best_step": best_step,
        "best_value": best_value,
        "terminal_step": terminal_step,
        "stop_reason": stop_reason,
        "numerical_failure": numerical_failure,
        "global_matched_gamma": calibrated_global_gamma,
        "checkpoint_policy": "server-local adapters only; binaries must not enter Git/artifact packages",
        "checkpoints": checkpoint_records,
        "result_status": "pilot",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (out_dir / "checkpoint_manifest.json").write_text(json.dumps({
        "local_only": True,
        "model_path": args.model_path,
        "reference_adapter": args.reference_adapter or args.sft_adapter,
        "method": args.method,
        "source_provenance": source_provenance(),
        "checkpoints": checkpoint_records,
    }, indent=2))


def _current_gradient_norm(parameters: Sequence[torch.nn.Parameter]) -> float:
    total = torch.zeros((), dtype=torch.float64)
    for parameter in parameters:
        if parameter.grad is not None:
            total += parameter.grad.detach().double().square().sum().cpu()
    return float(total.sqrt())


def cmd_calibrate_global(args: argparse.Namespace) -> None:
    """Freeze one gamma that matches controlled and global negative-gradient RMS budgets."""
    seed_all(args.seed)
    tokenizer = load_tokenizer(args.model_path)
    rows = read_jsonl(args.offline_data)
    if not rows or any(not row.get("pair_matched", False) for row in rows):
        raise RuntimeError("Global calibration requires all-matched offline rows")
    dataset = OfflineDataset(rows, tokenizer, args.max_length)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=make_offline_collator(tokenizer.pad_token_id),
        num_workers=0,
    )
    model = load_model(
        args.model_path,
        args.reference_adapter,
        trainable_adapter=True,
        load_in_4bit=args.load_in_4bit,
        dtype=args.dtype,
        gradient_checkpointing=False,
    )
    model.train()
    device = next(model.parameters()).device
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    controlled_norms: list[float] = []
    uncontrolled_norms: list[float] = []
    controlled_weights: list[float] = []

    for batch_index, packed in enumerate(loader):
        if batch_index >= args.calibration_batches:
            break
        near_batch = move_to_device(packed["near"], device)
        far_batch = move_to_device(packed["far"], device)

        model.zero_grad(set_to_none=True)
        near = completion_stats(model, near_batch)
        far = completion_stats(model, far_batch)
        far_surprisal = -far["token_lp"].detach()
        far_weights = torch.exp(
            -args.exp_lambda * F.relu(far_surprisal - args.surprisal_threshold)
        )
        controlled_lp = (
            args.near_mix * near["seq_lp"].mean()
            + args.far_mix * weighted_sequence_logprob(far, far_weights).mean()
        )
        (-controlled_lp).backward()
        controlled_norms.append(_current_gradient_norm(trainable))
        controlled_weights.append(float(
            args.near_mix
            + args.far_mix
            * (far_weights.detach() * far["token_mask"]).sum()
            / far["token_mask"].sum().clamp_min(1)
        ))

        model.zero_grad(set_to_none=True)
        near = completion_stats(model, near_batch)
        far = completion_stats(model, far_batch)
        uncontrolled_lp = (
            args.near_mix * near["seq_lp"].mean()
            + args.far_mix * far["seq_lp"].mean()
        )
        (-uncontrolled_lp).backward()
        uncontrolled_norms.append(_current_gradient_norm(trainable))

    model.zero_grad(set_to_none=True)
    if not controlled_norms or not uncontrolled_norms:
        raise RuntimeError("No calibration batches were processed")
    controlled_rms = float(np.sqrt(np.mean(np.square(controlled_norms))))
    uncontrolled_rms = float(np.sqrt(np.mean(np.square(uncontrolled_norms))))
    if uncontrolled_rms <= 0 or not math.isfinite(uncontrolled_rms):
        raise RuntimeError("Uncontrolled calibration gradient norm is invalid")
    gamma = controlled_rms / uncontrolled_rms
    if not math.isfinite(gamma) or gamma <= 0:
        raise RuntimeError("Calibrated global gamma is invalid")
    result = {
        "version": VERSION,
        "protocol": "fixed_calibration_split_rms_negative_gradient_budget_match",
        "reference_adapter": args.reference_adapter,
        "offline_data": args.offline_data,
        "seed": args.seed,
        "batches": len(controlled_norms),
        "batch_size": args.batch_size,
        "controlled_gradient_norms": controlled_norms,
        "uncontrolled_gradient_norms": uncontrolled_norms,
        "controlled_rms_gradient_norm": controlled_rms,
        "uncontrolled_rms_gradient_norm": uncontrolled_rms,
        "global_gamma": gamma,
        "mean_controlled_scalar_weight": float(np.mean(controlled_weights)),
        "frozen_before_method_training": True,
        "interpretation": (
            "global_matched applies the same fixed gamma to near and far negatives; "
            "controlled_negative uses selective far-token tapering."
        ),
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


def cmd_init_adapter(args: argparse.Namespace) -> None:
    """Create one shared, untrained LoRA adapter for base-first comparisons."""
    seed_all(args.seed)
    tokenizer = load_tokenizer(args.model_path)
    model = load_model(
        args.model_path,
        adapter_path=None,
        trainable_adapter=True,
        load_in_4bit=args.load_in_4bit,
        dtype=args.dtype,
        gradient_checkpointing=False,
    )
    output = Path(args.output_dir)
    ensure_checkpoint_output_is_local_or_ignored(output)
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output)
    tokenizer.save_pretrained(output)
    checkpoint = checkpoint_inventory(output, "shared_initial", 0)
    (output / "initialization_manifest.json").write_text(json.dumps({
        "version": VERSION,
        "seed": args.seed,
        "model_path": str(Path(args.model_path).resolve()),
        "training_updates": 0,
        "purpose": "shared zero-effect LoRA initialization for paired base-first methods",
        "local_only": True,
        "checkpoint": checkpoint,
    }, indent=2))
    print(json.dumps({"initialized_adapter": str(output), "training_updates": 0}, indent=2))


def _gradient_norm_for_negative_completion(
    model: Any,
    batch: dict[str, torch.Tensor],
    trainable: list[torch.nn.Parameter],
) -> tuple[float, float, float]:
    stats = completion_stats(model, batch)
    # For fixed A=-1, minimizing log pi(y|x) is the negative-advantage update.
    loss = stats["seq_lp"].mean()
    grads = torch.autograd.grad(loss, trainable, allow_unused=True)
    norm_sq = torch.zeros((), device=batch["input_ids"].device, dtype=torch.float32)
    for grad in grads:
        if grad is not None:
            norm_sq = norm_sq + grad.detach().float().square().sum()
    return (
        float(torch.sqrt(norm_sq)),
        float((-stats["seq_lp"].detach()).mean()),
        float(stats["score"].detach().mean()),
    )


def _snapshot_trainable(trainable: list[torch.nn.Parameter]) -> list[torch.Tensor]:
    return [p.detach().clone() for p in trainable]


def _restore_trainable(
    trainable: list[torch.nn.Parameter], snapshot: list[torch.Tensor]
) -> None:
    with torch.no_grad():
        for parameter, value in zip(trainable, snapshot):
            parameter.copy_(value)


def _negative_update_dynamics(
    model: Any,
    target_batch: dict[str, torch.Tensor],
    positive_batch: dict[str, torch.Tensor],
    trainable: list[torch.nn.Parameter],
    steps: int,
    lr: float,
) -> dict[str, float]:
    snapshot = _snapshot_trainable(trainable)
    model.eval()
    with torch.no_grad():
        before_target = float((-completion_stats(model, target_batch)["seq_lp"]).mean())
        before_positive = float((-completion_stats(model, positive_batch)["seq_lp"]).mean())
    for _ in range(steps):
        stats = completion_stats(model, target_batch)
        loss = stats["seq_lp"].mean()
        grads = torch.autograd.grad(loss, trainable, allow_unused=True)
        with torch.no_grad():
            for parameter, grad in zip(trainable, grads):
                if grad is not None:
                    parameter.add_(grad, alpha=-lr)
    with torch.no_grad():
        after_target = float((-completion_stats(model, target_batch)["seq_lp"]).mean())
        after_positive = float((-completion_stats(model, positive_batch)["seq_lp"]).mean())
    _restore_trainable(trainable, snapshot)
    return {
        "target_surprisal_before": before_target,
        "target_surprisal_after": after_target,
        "target_surprisal_delta": after_target - before_target,
        "positive_surprisal_before": before_positive,
        "positive_surprisal_after": after_positive,
        "positive_collateral_delta": after_positive - before_positive,
    }


def cmd_mechanism_probe(args: argparse.Namespace) -> None:
    """External-validity probe: fixed-A near/far influence in a real Transformer."""
    seed_all(args.seed)
    tokenizer = load_tokenizer(args.model_path)
    model = load_model(
        args.model_path,
        args.reference_adapter,
        trainable_adapter=True,
        load_in_4bit=args.load_in_4bit,
        dtype=args.dtype,
        gradient_checkpointing=False,
    )
    model.eval()
    device = next(model.parameters()).device
    trainable = [p for p in model.parameters() if p.requires_grad]
    rows = [row for row in read_jsonl(args.offline_data) if row.get("pair_matched")]
    if len(rows) < args.min_matched_pairs:
        raise RuntimeError(
            f"Only {len(rows)} matched near/far pairs; need at least {args.min_matched_pairs}."
        )
    rows = rows[: args.max_examples]
    per_pair: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        near = move_to_device(
            pad_encoded([
                encode_prompt_completion(tokenizer, row["prompt"], row["near_negative"], args.max_length)
            ], tokenizer.pad_token_id),
            device,
        )
        far = move_to_device(
            pad_encoded([
                encode_prompt_completion(tokenizer, row["prompt"], row["far_negative"], args.max_length)
            ], tokenizer.pad_token_id),
            device,
        )
        near_grad, near_surprisal, near_score = _gradient_norm_for_negative_completion(
            model, near, trainable
        )
        far_grad, far_surprisal, far_score = _gradient_norm_for_negative_completion(
            model, far, trainable
        )
        per_pair.append({
            "index": index,
            "id": row["id"],
            "near_surprisal": near_surprisal,
            "far_surprisal": far_surprisal,
            "surprisal_gap": far_surprisal - near_surprisal,
            "near_logit_score": near_score,
            "far_logit_score": far_score,
            "near_trainable_gradient_norm": near_grad,
            "far_trainable_gradient_norm": far_grad,
            "gradient_ratio_far_over_near": far_grad / max(near_grad, 1e-12),
        })
        print(f"probe {index + 1}/{len(rows)}", flush=True)

    dynamics_rows = rows[: min(args.dynamics_examples, len(rows))]
    def batch_for(key: str) -> dict[str, torch.Tensor]:
        encoded = [
            encode_prompt_completion(tokenizer, row["prompt"], row[key], args.max_length)
            for row in dynamics_rows
        ]
        return move_to_device(pad_encoded(encoded, tokenizer.pad_token_id), device)

    positive_batch = batch_for("positive")
    near_dynamics = _negative_update_dynamics(
        model, batch_for("near_negative"), positive_batch, trainable, args.probe_steps, args.probe_lr
    )
    far_dynamics = _negative_update_dynamics(
        model, batch_for("far_negative"), positive_batch, trainable, args.probe_steps, args.probe_lr
    )
    summary = {
        "version": VERSION,
        "claim": "EXT-C external validation of fixed-negative-advantage remoteness influence",
        "fixed_advantage": -1.0,
        "matched_pairs": len(per_pair),
        "mean_near_surprisal": float(np.mean([r["near_surprisal"] for r in per_pair])),
        "mean_far_surprisal": float(np.mean([r["far_surprisal"] for r in per_pair])),
        "mean_near_trainable_gradient_norm": float(np.mean([r["near_trainable_gradient_norm"] for r in per_pair])),
        "mean_far_trainable_gradient_norm": float(np.mean([r["far_trainable_gradient_norm"] for r in per_pair])),
        "median_gradient_ratio_far_over_near": float(np.median([r["gradient_ratio_far_over_near"] for r in per_pair])),
        "near_negative_dynamics": near_dynamics,
        "far_negative_dynamics": far_dynamics,
        "interpretation_boundary": (
            "Countdown is external validity. It does not replace D-U1 causal identification, "
            "and categorical logit score is bounded; probability/support suppression is reported separately."
        ),
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2))
    output_csv = Path(args.output_csv)
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_pair[0].keys()))
        writer.writeheader()
        writer.writerows(per_pair)
    print(json.dumps(summary, indent=2))

def cmd_preflight(args: argparse.Namespace) -> None:
    """Fail fast on tokenizer, LoRA targets, forward, backward, and generation."""
    model_dir = Path(args.model_path)
    if not model_dir.exists():
        raise FileNotFoundError(f"model_path does not exist: {model_dir}")
    print(json.dumps({
        "version": VERSION,
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_devices": torch.cuda.device_count(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_memory_gib": round(gpu_memory_gib(0), 2) if torch.cuda.is_available() else 0.0,
        "bf16_supported": bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
        "model_path": str(model_dir.resolve()),
        "model_metadata": read_model_metadata(args.model_path),
        "load_in_4bit": args.load_in_4bit,
        "dtype": args.dtype,
    }, indent=2))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    tokenizer = load_tokenizer(args.model_path)
    row = generate_examples(1, seed=args.seed, n_numbers=4)[0]
    rendered = chat_prompt(tokenizer, row["prompt"])
    tok = tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
    oracle_ok = verify_expression(row["oracle"], row["numbers"], row["target"])["correct"]
    print(json.dumps({"tokenizer_ok": True, "prompt_tokens": int(tok["input_ids"].shape[1]),
                      "verifier_oracle_ok": oracle_ok}))
    if args.tokenizer_only:
        return
    model = load_model(args.model_path, None, trainable_adapter=True,
                       load_in_4bit=args.load_in_4bit, dtype=args.dtype,
                       gradient_checkpointing=False)
    device = next(model.parameters()).device
    # 1) Forward + backward through a genuine completion-only SFT example.
    encoded = encode_prompt_completion(tokenizer, row["prompt"], row["oracle"], 256)
    batch = move_to_device(pad_encoded([encoded], tokenizer.pad_token_id), device)
    model.train()
    out = model(**batch, use_cache=False)
    out.loss.backward()
    grad_finite = all(
        bool(torch.isfinite(p.grad).all()) for p in model.parameters()
        if p.requires_grad and p.grad is not None
    )
    model.zero_grad(set_to_none=True)
    # 2) Real generation path and verifier. Correctness is not required pre-SFT.
    generated = generate_outputs(
        model, tokenizer, [row["prompt"]], max_new_tokens=40,
        do_sample=False, temperature=1.0, top_p=1.0, num_return_sequences=1,
    )[0][0]
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(json.dumps({
        "model_forward_ok": True,
        "model_backward_ok": grad_finite,
        "pre_sft_generation_ok": bool(generated.strip()),
        "sample_generation": generated[:160],
        "trainable_parameters": trainable,
        "total_parameters": total,
        "trainable_fraction": trainable / max(total, 1),
        "peak_memory_gib": round(torch.cuda.max_memory_allocated() / (1024 ** 3), 3),
    }, indent=2))
    if not grad_finite:
        raise RuntimeError("Non-finite gradients during preflight")


def cmd_evaluate(args: argparse.Namespace) -> None:
    seed_all(args.seed)
    tokenizer = load_tokenizer(args.model_path)
    model = load_model(
        args.model_path,
        args.adapter,
        trainable_adapter=False,
        load_in_4bit=args.load_in_4bit,
        dtype=args.dtype,
        gradient_checkpointing=False,
    )
    rows = read_jsonl(args.data)
    known_structures: set[str] | None = None
    if args.structure_reference_data:
        reference_rows = read_jsonl(args.structure_reference_data)
        known_structures = {
            row.get("oracle_structure") or expression_structure(row["oracle"])
            for row in reference_rows
        }
    metrics = evaluate_rows(
        model,
        tokenizer,
        rows,
        args.batch_size,
        args.max_new_tokens,
        args.pass_k,
        args.seed,
        known_structures,
    )
    print(json.dumps(metrics, indent=2))
    if getattr(args, "output_json", None):
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(metrics, indent=2))


def _run_stage(argv: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(Path(__file__).resolve()), *argv]
    print("\n[RUN]", " ".join(command), flush=True)
    with log_path.open("w") as log:
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
        code = proc.wait()
    if code != 0:
        raise RuntimeError(f"Stage failed ({code}). See {log_path}")


def cmd_run(args: argparse.Namespace) -> None:
    """One-command, resumable, base-first 0.5B Countdown audited pilot."""
    if args.gpu != "auto":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    plan = resolve_execution_plan(args.model_path, args.preset, args.memory_mode, args.gpu)
    root = Path(args.work_dir).resolve()
    ensure_checkpoint_output_is_local_or_ignored(root)
    root.mkdir(parents=True, exist_ok=True)
    if (
        (plan["memory_mode"] != "bf16" or plan["load_in_4bit"] or plan["dtype"] != "bf16")
        and not args.allow_non_bf16_smoke
    ):
        raise RuntimeError(
            "The registered pilot requires one shared BF16 LoRA parameterization for all methods. "
            "Use a BF16-capable GPU, or pass --allow_non_bf16_smoke for engineering smoke only."
        )
    params_b = plan["model_metadata"].get("estimated_params_b")
    registered_model = (
        plan["preset"] in {"0.5b", "small"}
        and params_b is not None
        and params_b <= 1.0
    )
    registered_parameterization = (
        plan["memory_mode"] == "bf16"
        and not plan["load_in_4bit"]
        and plan["dtype"] == "bf16"
    )
    if (not registered_model or not registered_parameterization) and not args.allow_non_bf16_smoke:
        raise RuntimeError(
            "EXT-C-E8-V4.1 is registered only for Qwen Instruct 0.5B with BF16 LoRA. "
            "Other model sizes or QLoRA/fp16 may be used only with "
            "--allow_non_bf16_smoke and remain engineering smoke, not pilot evidence."
        )
    run_status = (
        "pilot"
        if registered_model and registered_parameterization
        else "engineering_smoke"
    )

    data_dir = root / "data"
    logs = root / "logs"
    shared_adapter_dir = root / "reference_adapter"
    sft_dir = root / "sft_adapter"
    offline_file = data_dir / "offline.jsonl"
    split_manifest_file = data_dir / "split_manifest.json"
    methods_dir = root / "methods"
    calibration_json = root / "global_matched_calibration.json"

    methods = [method.strip() for method in args.methods.split(",") if method.strip()]
    allowed = {
        "positive_only",
        "controlled_negative",
        "uncontrolled_negative",
        "global_matched",
        # Historical development methods remain callable but are outside the v4.1 pilot default.
        "uncontrolled", "global", "exp", "entropy_bonus", "target_entropy", "sbrc", "hybrid",
    }
    unknown = set(methods) - allowed
    if unknown:
        raise ValueError(f"Unknown methods: {sorted(unknown)}")
    required_pilot = {
        "positive_only", "controlled_negative", "uncontrolled_negative", "global_matched"
    }
    if run_status == "pilot" and set(methods) != required_pilot:
        raise RuntimeError(
            "The registered v4.1 pilot comparison is frozen to exactly: "
            "positive_only, controlled_negative, uncontrolled_negative, global_matched."
        )

    run_spec = {
        "version": VERSION,
        "source_provenance": source_provenance(),
        "experiment_id": "EXT-C-E8-V4.1",
        "model_path": str(Path(args.model_path).resolve()),
        "preset": plan["preset"],
        "memory_mode": plan["memory_mode"],
        "parameterization": "shared_bf16_lora" if run_status == "pilot" else "nonformal_smoke",
        "methods": methods,
        "seed": args.seed,
        "min_base_success": args.min_base_success,
        "min_base_valid": args.min_base_valid,
        "min_sft_success": args.min_sft_success,
        "pair_resample_rounds": args.pair_resample_rounds,
        "result_status": run_status,
        "checkpoint_policy": "server-local only",
    }
    run_spec["fingerprint"] = stable_fingerprint(run_spec)
    run_config_path = root / "run_config.json"
    if run_config_path.exists() and not args.force:
        previous = json.loads(run_config_path.read_text())
        if previous.get("fingerprint") != run_spec["fingerprint"]:
            raise RuntimeError(
                "Existing work_dir was created with a different configuration. "
                "Use a new --work_dir or pass --force after reviewing the difference."
            )
    run_config_path.write_text(json.dumps(run_spec, indent=2))
    (root / "execution_plan.json").write_text(json.dumps(plan, indent=2))
    print("EXECUTION_PLAN", json.dumps(plan, indent=2))

    presets = {
        "0.5b": dict(
            train=6000, val=500, test=1000, offline=1500, rollouts=12,
            sft_epochs=3, sft_accum=16, method_steps=1200, method_accum=8,
            method_min_steps=400, patience=6, eval_examples=500,
            eval_every=100, pass_k=8, probe_examples=32, dynamics_examples=16,
            calibration_batches=16,
        ),
        "small": dict(
            train=6000, val=500, test=1000, offline=1500, rollouts=12,
            sft_epochs=3, sft_accum=16, method_steps=1200, method_accum=8,
            method_min_steps=400, patience=6, eval_examples=500,
            eval_every=100, pass_k=8, probe_examples=32, dynamics_examples=16,
            calibration_batches=16,
        ),
        "3b": dict(
            train=20000, val=1000, test=2000, offline=4000, rollouts=12,
            sft_epochs=3, sft_accum=32, method_steps=3000, method_accum=16,
            method_min_steps=1000, patience=8, eval_examples=1000,
            eval_every=100, pass_k=8, probe_examples=32, dynamics_examples=16,
            calibration_batches=16,
        ),
        "7b": dict(
            train=20000, val=1000, test=2000, offline=4000, rollouts=12,
            sft_epochs=2, sft_accum=32, method_steps=2500, method_accum=16,
            method_min_steps=800, patience=8, eval_examples=1000,
            eval_every=100, pass_k=8, probe_examples=24, dynamics_examples=12,
            calibration_batches=12,
        ),
    }
    preset = presets[plan["preset"]]
    train_file = data_dir / "train.jsonl"
    val_file = data_dir / "val.jsonl"
    test_file = data_dir / "test.jsonl"

    def model_flags_from_plan(current: dict[str, Any]) -> list[str]:
        flags = ["--model_path", args.model_path, "--dtype", current["dtype"]]
        if current["load_in_4bit"]:
            flags.append("--load_in_4bit")
        return flags

    model_flags = model_flags_from_plan(plan)
    if not args.skip_preflight:
        _run_stage(
            ["preflight", *model_flags, "--seed", str(args.seed)],
            logs / "00_preflight.log",
        )

    if args.force or not (train_file.exists() and val_file.exists() and test_file.exists()):
        _run_stage([
            "generate",
            "--train", str(preset["train"]),
            "--val", str(preset["val"]),
            "--test", str(preset["test"]),
            "--train_out", str(train_file),
            "--val_out", str(val_file),
            "--test_out", str(test_file),
            "--manifest_out", str(split_manifest_file),
            "--seed", str(args.seed),
        ], logs / "01_generate_pattern_family_split.log")

    base_val_json = root / "base_val_metrics.json"
    _run_stage([
        "evaluate", *model_flags,
        "--data", str(val_file),
        "--structure_reference_data", str(train_file),
        "--batch_size", str(plan["eval_batch"]),
        "--pass_k", str(preset["pass_k"]),
        "--output_json", str(base_val_json),
        "--seed", str(args.seed + 5000),
    ], logs / "02_base_eval.log")
    base_val = json.loads(base_val_json.read_text())
    base_passes = (
        base_val["greedy_success"] >= args.min_base_success
        and base_val["valid_rate"] >= args.min_base_valid
    )

    if base_passes:
        initialization_mode = "base_first_no_sft"
        if args.force and shared_adapter_dir.exists():
            shutil.rmtree(shared_adapter_dir)
        if args.force or not (shared_adapter_dir / "adapter_config.json").exists():
            _run_stage([
                "init_adapter", *model_flags,
                "--output_dir", str(shared_adapter_dir),
                "--seed", str(args.seed),
            ], logs / "03_init_shared_adapter.log")
        reference_dir = shared_adapter_dir
    else:
        if args.no_sft_fallback:
            raise RuntimeError(
                f"Base checkpoint failed gate: greedy={base_val['greedy_success']:.3f}, "
                f"valid={base_val['valid_rate']:.3f}. SFT fallback disabled."
            )
        initialization_mode = "minimal_sft_fallback"
        if args.force and sft_dir.exists():
            shutil.rmtree(sft_dir)
        if args.force or not (sft_dir / "best_adapter" / "adapter_config.json").exists():
            _run_stage([
                "sft", *model_flags,
                "--train_data", str(train_file),
                "--val_data", str(val_file),
                "--output_dir", str(sft_dir),
                "--epochs", str(preset["sft_epochs"]),
                "--micro_batch", str(plan["micro_batch"]),
                "--grad_accum", str(preset["sft_accum"]),
                "--eval_batch", str(plan["eval_batch"]),
                "--eval_examples", str(preset["eval_examples"]),
                "--pass_k", str(preset["pass_k"]),
                "--eval_seed", str(args.seed + 5000),
                "--seed", str(args.seed),
            ], logs / "03_sft_fallback.log")
        reference_dir = sft_dir / "best_adapter"
        sft_val_json = root / "sft_val_metrics.json"
        _run_stage([
            "evaluate", *model_flags,
            "--adapter", str(reference_dir),
            "--data", str(val_file),
            "--structure_reference_data", str(train_file),
            "--batch_size", str(plan["eval_batch"]),
            "--pass_k", str(preset["pass_k"]),
            "--output_json", str(sft_val_json),
            "--seed", str(args.seed + 5000),
        ], logs / "03b_sft_gate.log")
        sft_val = json.loads(sft_val_json.read_text())
        if sft_val["greedy_success"] < args.min_sft_success:
            raise RuntimeError(
                f"SFT fallback greedy_success={sft_val['greedy_success']:.3f} < "
                f"{args.min_sft_success:.3f}; stop before mechanism/method training."
            )

    reference_val_json = root / "reference_val_metrics.json"
    _run_stage([
        "evaluate", *model_flags,
        "--adapter", str(reference_dir),
        "--data", str(val_file),
        "--structure_reference_data", str(train_file),
        "--batch_size", str(plan["eval_batch"]),
        "--pass_k", str(preset["pass_k"]),
        "--output_json", str(reference_val_json),
        "--seed", str(args.seed + 5000),
    ], logs / "04_reference_eval.log")
    reference_val = json.loads(reference_val_json.read_text())

    if args.force or not offline_file.exists():
        _run_stage([
            "build_offline", *model_flags,
            "--reference_adapter", str(reference_dir),
            "--input_data", str(train_file),
            "--split_manifest", str(split_manifest_file),
            "--output_data", str(offline_file),
            "--max_examples", str(preset["offline"]),
            "--rollouts", str(preset["rollouts"]),
            "--batch_size", str(plan["rollout_batch"]),
            "--score_batch_size", str(plan["score_batch"]),
            "--pair_resample_rounds", str(args.pair_resample_rounds),
            "--seed", str(args.seed + 11),
        ], logs / "05_build_matched_offline.log")

    mechanism_json = root / "mechanism_probe.json"
    mechanism_csv = root / "mechanism_probe_pairs.csv"
    _run_stage([
        "mechanism_probe", *model_flags,
        "--reference_adapter", str(reference_dir),
        "--offline_data", str(offline_file),
        "--output_json", str(mechanism_json),
        "--output_csv", str(mechanism_csv),
        "--max_examples", str(preset["probe_examples"]),
        "--dynamics_examples", str(preset["dynamics_examples"]),
        "--min_matched_pairs", str(args.min_matched_pairs),
        "--seed", str(args.seed + 50),
    ], logs / "06_mechanism_probe.log")

    if "global_matched" in methods and (args.force or not calibration_json.exists()):
        _run_stage([
            "calibrate_global", *model_flags,
            "--reference_adapter", str(reference_dir),
            "--offline_data", str(offline_file),
            "--output_json", str(calibration_json),
            "--batch_size", str(plan["micro_batch"]),
            "--calibration_batches", str(preset["calibration_batches"]),
            "--seed", str(args.seed + 75),
        ], logs / "06b_global_matched_calibration.log")

    shared_method_seed = args.seed + 100
    for method in methods:
        output = methods_dir / method
        if args.force and output.exists():
            shutil.rmtree(output)
        complete_checkpoint = (
            output / "terminal_adapter" / "adapter_config.json"
        ).exists() or (
            output / "last_finite_adapter" / "adapter_config.json"
        ).exists()
        if args.force or not complete_checkpoint:
            command = [
                "train_method", *model_flags,
                "--reference_adapter", str(reference_dir),
                "--offline_data", str(offline_file),
                "--val_data", str(val_file),
                "--structure_reference_data", str(train_file),
                "--output_dir", str(output),
                "--method", method,
                "--steps", str(preset["method_steps"]),
                "--micro_batch", str(plan["micro_batch"]),
                "--grad_accum", str(preset["method_accum"]),
                "--min_steps", str(preset["method_min_steps"]),
                "--early_stop_patience", str(preset["patience"]),
                "--eval_examples", str(preset["eval_examples"]),
                "--eval_batch", str(plan["eval_batch"]),
                "--eval_every", str(preset["eval_every"]),
                "--pass_k", str(preset["pass_k"]),
                "--eval_seed", str(args.seed + 6000),
                "--seed", str(shared_method_seed),
            ]
            if method == "global_matched":
                command.extend(["--global_calibration_json", str(calibration_json)])
            _run_stage(command, logs / f"07_train_{method}.log")

    summary_rows: list[dict[str, Any]] = []

    def evaluate_checkpoint(
        label: str,
        adapter: Path | None,
        output_json: Path,
        log_path: Path,
        extra: dict[str, Any] | None = None,
    ) -> None:
        command = ["evaluate", *model_flags]
        if adapter is not None:
            command.extend(["--adapter", str(adapter)])
        command.extend([
            "--data", str(test_file),
            "--structure_reference_data", str(train_file),
            "--batch_size", str(plan["eval_batch"]),
            "--pass_k", str(preset["pass_k"]),
            "--output_json", str(output_json),
            "--seed", str(args.seed + 7000),
        ])
        _run_stage(command, log_path)
        summary_rows.append({"method": label, **(extra or {}), **json.loads(output_json.read_text())})

    evaluate_checkpoint(
        "raw_base_no_training", None, root / "base_test_metrics.json", logs / "08_test_raw_base.log"
    )
    evaluate_checkpoint(
        "shared_initial_checkpoint",
        reference_dir,
        root / "reference_test_metrics.json",
        logs / "08b_test_reference.log",
        {"initialization_mode": initialization_mode, "checkpoint_kind": "step_0"},
    )

    for method in methods:
        output = methods_dir / method
        manifest = json.loads((output / "manifest.json").read_text())
        for checkpoint_kind in ("best", "terminal", "last_finite"):
            adapter = output / f"{checkpoint_kind}_adapter"
            if not (adapter / "adapter_config.json").exists():
                continue
            result_json = output / f"test_metrics_{checkpoint_kind}.json"
            evaluate_checkpoint(
                method,
                adapter,
                result_json,
                logs / f"09_test_{method}_{checkpoint_kind}.log",
                {
                    "checkpoint_kind": checkpoint_kind,
                    "best_step": manifest.get("best_step"),
                    "terminal_step": manifest.get("terminal_step"),
                    "best_val": manifest.get("best_value"),
                    "stop_reason": manifest.get("stop_reason"),
                    "numerical_failure": manifest.get("numerical_failure"),
                    "global_matched_gamma": manifest.get("global_matched_gamma"),
                },
            )

    summary_path = root / "arena_summary.csv"
    fields = sorted({key for row in summary_rows for key in row})
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)
    complete = {
        "version": VERSION,
        "experiment_id": "EXT-C-E8-V4.1",
        "source_provenance": source_provenance(),
        "plan": plan,
        "initialization_mode": initialization_mode,
        "base_validation": base_val,
        "reference_validation": reference_val,
        "mechanism_probe": json.loads(mechanism_json.read_text()),
        "global_matched_calibration": (
            json.loads(calibration_json.read_text()) if calibration_json.exists() else None
        ),
        "summary": summary_rows,
        "result_status": run_status,
        "terminal_audit_present": all(
            any(
                row["method"] == method
                and row.get("checkpoint_kind") in {"terminal", "last_finite"}
                for row in summary_rows
            )
            for method in methods
        ),
        "full_finetune_confirmation": "not_run; required only after a LoRA pilot signal",
        "note": "A completed single-seed run remains a pilot, never a formal multi-seed result.",
    }
    (root / "run_complete.json").write_text(json.dumps(complete, indent=2))
    print("\nDONE. Summary:", summary_path)
    for row in summary_rows:
        print(json.dumps(row, ensure_ascii=False))


def cmd_selftest(args: argparse.Namespace) -> None:
    oracle = "((1 + 2) * (3 + 4))"
    check = verify_expression(oracle, [1, 2, 3, 4], 21)
    assert check["correct"]
    assert expression_structure("(1 + 2) + (3 + 4)") == expression_structure("1 + (2 + (3 + 4))")
    assert expression_structure("(1 + 2) * (3 + 4)") == expression_structure("(2 + 1) * (4 + 3)")
    assert len(canonical_pattern_catalog(4)) == 96
    row = {
        "id": "selftest",
        "numbers": [1, 2, 3, 4],
        "target": 21,
        "oracle": oracle,
    }
    allowed = set(canonical_pattern_catalog(4))
    wrong = make_valid_wrong_expression(row, random.Random(7), allowed_patterns=allowed)
    wrong_check = verify_expression(wrong, row["numbers"], row["target"])
    assert wrong_check["valid_format"] and wrong_check["uses_numbers"] and not wrong_check["correct"]
    near, far, matched = select_matched_negative_pair([
        {"surprisal": 1.0, "token_length": 7, "tree_depth": 2, "value_error": 4.0},
        {"surprisal": 2.0, "token_length": 8, "tree_depth": 2, "value_error": 5.0},
        {"surprisal": 5.0, "token_length": 8, "tree_depth": 3, "value_error": 6.0},
    ], 0.5, 2, 1, 4.0)
    assert matched and near is not None and far is not None and near["surprisal"] < far["surprisal"]
    no_near, no_far, no_match = select_matched_negative_pair([
        {"surprisal": 1.0, "token_length": 7, "tree_depth": 2, "value_error": 1.0},
        {"surprisal": 1.1, "token_length": 20, "tree_depth": 5, "value_error": 100.0},
    ], 0.5, 2, 1, 4.0)
    assert not no_match and no_near is None and no_far is None
    train, val, test, manifest = generate_structural_splits(40, 12, 12, seed=17)
    train_patterns = {item["oracle_structure"] for item in train}
    val_patterns = {item["oracle_structure"] for item in val}
    test_patterns = {item["oracle_structure"] for item in test}
    assert not (train_patterns & val_patterns or train_patterns & test_patterns or val_patterns & test_patterns)
    assert manifest["structure_sets_disjoint"]
    assert manifest["protocol"] == "park_inspired_pattern_first_family_holdout_capacity_audited"
    print(json.dumps({
        "selftest": "ok",
        "version": VERSION,
        "wrong_expression": wrong,
        "split_manifest": manifest,
    }, indent=2))


def common_model_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--model_path", required=True, help="Local Qwen model directory")
    ap.add_argument("--load_in_4bit", action="store_true")
    ap.add_argument("--dtype", choices=["auto", "bf16", "fp16"], default="auto")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    ap = sub.add_parser("selftest", help="Run CPU-only verifier, split, and pairing tests")
    ap.set_defaults(func=cmd_selftest)

    ap = sub.add_parser("preflight", help="Check local model, tokenizer, CUDA, quantization, and LoRA")
    common_model_args(ap)
    ap.add_argument("--tokenizer_only", action="store_true")
    ap.add_argument("--seed", type=int, default=1234)
    ap.set_defaults(func=cmd_preflight)

    ap = sub.add_parser("generate")
    ap.add_argument("--train", type=int, default=6000)
    ap.add_argument("--val", type=int, default=500)
    ap.add_argument("--test", type=int, default=1000)
    ap.add_argument("--n_numbers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--train_out", default="data/train.jsonl")
    ap.add_argument("--val_out", default="data/val.jsonl")
    ap.add_argument("--test_out", default="data/test.jsonl")
    ap.add_argument("--manifest_out", default=None)
    ap.set_defaults(func=cmd_generate)

    ap = sub.add_parser("init_adapter", help="Create one shared untrained LoRA adapter")
    common_model_args(ap)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--seed", type=int, default=1234)
    ap.set_defaults(func=cmd_init_adapter)

    ap = sub.add_parser("sft")
    common_model_args(ap)
    ap.add_argument("--train_data", required=True)
    ap.add_argument("--val_data", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--micro_batch", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--max_new_tokens", type=int, default=80)
    ap.add_argument("--eval_examples", type=int, default=500)
    ap.add_argument("--eval_batch", type=int, default=8)
    ap.add_argument("--pass_k", type=int, default=4)
    ap.add_argument("--eval_seed", type=int, default=5000)
    ap.add_argument("--selection_metric", choices=["greedy_success", "pass_at_k"], default="greedy_success")
    ap.add_argument("--selection_delta", type=float, default=0.0)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.set_defaults(func=cmd_sft)

    ap = sub.add_parser("build_offline")
    common_model_args(ap)
    ap.add_argument("--reference_adapter", default=None)
    ap.add_argument("--sft_adapter", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--input_data", required=True)
    ap.add_argument("--split_manifest", required=True)
    ap.add_argument("--output_data", required=True)
    ap.add_argument("--rollouts", type=int, default=12)
    ap.add_argument("--batch_size", type=int, default=4, help="Initial rollout generation batch size")
    ap.add_argument("--pair_resample_rounds", type=int, default=3)
    ap.add_argument("--min_negative_candidates", type=int, default=4)
    ap.add_argument("--score_batch_size", type=int, default=16)
    ap.add_argument("--max_examples", type=int, default=1500, help="0 means all")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_new_tokens", type=int, default=80)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--min_surprisal_gap", type=float, default=0.5)
    ap.add_argument("--max_token_length_diff", type=int, default=2)
    ap.add_argument("--max_tree_depth_diff", type=int, default=1)
    ap.add_argument("--max_value_error_ratio", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=11)
    ap.set_defaults(func=cmd_build_offline)

    ap = sub.add_parser("mechanism_probe")
    common_model_args(ap)
    ap.add_argument("--reference_adapter", required=True)
    ap.add_argument("--offline_data", required=True)
    ap.add_argument("--output_json", required=True)
    ap.add_argument("--output_csv", required=True)
    ap.add_argument("--max_examples", type=int, default=32)
    ap.add_argument("--dynamics_examples", type=int, default=16)
    ap.add_argument("--min_matched_pairs", type=int, default=16)
    ap.add_argument("--probe_steps", type=int, default=5)
    ap.add_argument("--probe_lr", type=float, default=5e-4)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--seed", type=int, default=1284)
    ap.set_defaults(func=cmd_mechanism_probe)

    ap = sub.add_parser("train_method")
    common_model_args(ap)
    ap.add_argument("--reference_adapter", default=None)
    ap.add_argument("--sft_adapter", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--offline_data", required=True)
    ap.add_argument("--val_data", required=True)
    ap.add_argument("--structure_reference_data", default=None)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument(
        "--method",
        choices=[
            "positive_only",
            "controlled_negative",
            "uncontrolled_negative",
            "global_matched",
            "uncontrolled",
            "global",
            "exp",
            "entropy_bonus",
            "target_entropy",
            "sbrc",
            "hybrid",
        ],
        required=True,
    )
    ap.add_argument("--steps", type=int, default=1200, help="Maximum optimizer updates")
    ap.add_argument("--min_steps", type=int, default=400, help="Do not early-stop before this many updates")
    ap.add_argument("--early_stop_patience", type=int, default=6, help="Validation checks without improvement")
    ap.add_argument("--early_stop_delta", type=float, default=0.002)
    ap.add_argument("--selection_metric", choices=["greedy_success", "pass_at_k"], default="greedy_success")
    ap.add_argument("--micro_batch", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--max_new_tokens", type=int, default=80)
    ap.add_argument("--eval_examples", type=int, default=500)
    ap.add_argument("--eval_batch", type=int, default=8)
    ap.add_argument("--pass_k", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=0.7)
    ap.add_argument("--near_mix", type=float, default=0.5)
    ap.add_argument("--far_mix", type=float, default=0.5)
    ap.add_argument("--global_gamma", type=float, default=0.55)
    ap.add_argument("--global_calibration_json", default=None)
    ap.add_argument("--exp_lambda", type=float, default=0.7)
    ap.add_argument("--surprisal_threshold", type=float, default=2.0)
    ap.add_argument("--entropy_coef", type=float, default=0.02)
    ap.add_argument("--target_entropy", type=float, default=1.8)
    ap.add_argument("--target_entropy_coef", type=float, default=0.05)
    ap.add_argument("--sbrc_kappa", type=float, default=0.92)
    ap.add_argument("--entropy_floor", type=float, default=1.0)
    ap.add_argument("--eval_every", type=int, default=100)
    ap.add_argument("--eval_seed", type=int, default=6000)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.set_defaults(func=cmd_train_method)

    ap = sub.add_parser("calibrate_global", help="Calibrate fixed global negative-gradient budget")
    common_model_args(ap)
    ap.add_argument("--reference_adapter", required=True)
    ap.add_argument("--offline_data", required=True)
    ap.add_argument("--output_json", required=True)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--calibration_batches", type=int, default=16)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--near_mix", type=float, default=0.5)
    ap.add_argument("--far_mix", type=float, default=0.5)
    ap.add_argument("--exp_lambda", type=float, default=0.7)
    ap.add_argument("--surprisal_threshold", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=1309)
    ap.set_defaults(func=cmd_calibrate_global)

    ap = sub.add_parser("evaluate")
    common_model_args(ap)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--data", required=True)
    ap.add_argument("--structure_reference_data", default=None)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=80)
    ap.add_argument("--pass_k", type=int, default=8)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--output_json", default=None)
    ap.set_defaults(func=cmd_evaluate)

    ap = sub.add_parser("run", aliases=["all"], help="Base-first 0.5B mechanism + effect arena")
    ap.add_argument("--model_path", required=True, help="Local Qwen Instruct model directory")
    ap.add_argument("--work_dir", required=True)
    ap.add_argument("--gpu", default="0", help="Single physical GPU id, or auto")
    ap.add_argument("--preset", choices=["auto", "0.5b", "small", "3b", "7b"], default="auto")
    ap.add_argument("--memory_mode", choices=["auto", "bf16", "qlora"], default="bf16")
    ap.add_argument("--methods", default="positive_only,controlled_negative,uncontrolled_negative,global_matched")
    ap.add_argument("--min_base_success", type=float, default=0.15)
    ap.add_argument("--min_base_valid", type=float, default=0.80)
    ap.add_argument("--min_sft_success", type=float, default=0.15)
    ap.add_argument("--min_matched_pairs", type=int, default=16)
    ap.add_argument("--pair_resample_rounds", type=int, default=3)
    ap.add_argument("--allow_non_bf16_smoke", action="store_true")
    ap.add_argument("--no_sft_fallback", action="store_true")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--skip_preflight", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.set_defaults(func=cmd_run)
    return parser

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if hasattr(args, "near_mix") and not math.isclose(args.near_mix + args.far_mix, 1.0):
        parser.error("--near_mix + --far_mix must equal 1")
    args.func(args)


if __name__ == "__main__":
    main()
