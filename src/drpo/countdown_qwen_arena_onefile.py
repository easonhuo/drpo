#!/usr/bin/env python3
"""Countdown dynamic-remoteness external-validity arena for local Qwen Instruct models (v4.4.0).

One-command run
---------------
python3 scripts/run_countdown_pilot.py \
  --model_path /ABS/PATH/TO/QWEN-0.5B-INSTRUCT \
  --work_dir /ABS/PATH/TO/COUNTDOWN_RUN

The v4.3 protocol first evaluates the untouched base model. If the base checkpoint
passes the registered verifier/format gate, all compared methods start from one
shared untrained LoRA adapter and no Countdown SFT is performed. A minimal SFT
fallback is used only when the base gate fails.

The run has two responsibilities:
  1. a fixed-negative-advantage near/far mechanism probe on matched legal wrong
     expressions;
  2. a focused paired comparison: positive-only, static controlled-negative,
     dynamic controlled-negative, and uncontrolled-negative. Static control tapers
     only the branch labeled far at data construction; dynamic control applies the
     same current-policy token-surprisal taper to both negative branches. The shared
     negative scale is calibrated once at the common initialization from a fixed
     training calibration split; it is not selected from task outcomes or test data.

All pilot methods use the same BF16 LoRA parameterization. Model/adaptor binaries
remain server-local; only manifests, metrics, and hashes belong in artifacts.

Countdown is external validity for the D-U1 categorical theory. It does not
replace D-U1 causal identification. Static checks and CPU self-tests are not
formal Qwen results.
"""
from __future__ import annotations

import argparse
import ast
import json
import time
import traceback
import math
import os
import random
import re
import csv
import hashlib
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
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

EXPERIMENT_ID = "EXT-C-E8-V4.3"
VERSION = "4.4.0-dynamic-negative-control"


def read_model_metadata(model_path: str) -> dict[str, Any]:
    """Read local model metadata for the registered 0.5B Instruct gate."""
    root = Path(model_path)
    cfg_path = root / "config.json"
    tok_cfg_path = root / "tokenizer_config.json"
    if not cfg_path.exists():
        return {
            "model_type": "unknown",
            "estimated_params_b": None,
            "has_chat_template": False,
            "identity_hints": [root.name],
            "registered_instruct_identity": False,
        }
    cfg = json.loads(cfg_path.read_text())
    tok_cfg = json.loads(tok_cfg_path.read_text()) if tok_cfg_path.exists() else {}
    hidden = cfg.get("hidden_size") or cfg.get("d_model")
    layers = cfg.get("num_hidden_layers") or cfg.get("n_layer")
    inter = cfg.get("intermediate_size")
    vocab = cfg.get("vocab_size")
    estimate = None
    if all(isinstance(x, int) and x > 0 for x in (hidden, layers, inter, vocab)):
        estimate = (vocab * hidden + layers * (4 * hidden * hidden + 3 * hidden * inter)) / 1e9
    identity_hints = [
        root.name,
        str(cfg.get("_name_or_path", "")),
        str(tok_cfg.get("name_or_path", "")),
    ]
    identity_text = " ".join(identity_hints).lower()
    for tag, value in (("0.5b", 0.5), ("0.6b", 0.6), ("1.5b", 1.5),
                       ("1.8b", 1.8), ("3b", 3.0), ("4b", 4.0),
                       ("7b", 7.0), ("8b", 8.0), ("14b", 14.0),
                       ("32b", 32.0)):
        if tag in identity_text:
            estimate = value
            break
    has_chat_template = bool(tok_cfg.get("chat_template"))
    registered_instruct_identity = bool(
        cfg.get("model_type") == "qwen2"
        and has_chat_template
        and ("qwen" in identity_text or "qwen2" in str(cfg.get("architectures", [])).lower())
    )
    return {
        "model_type": cfg.get("model_type", "unknown"),
        "architectures": cfg.get("architectures", []),
        "estimated_params_b": estimate,
        "hidden_size": hidden,
        "num_hidden_layers": layers,
        "has_chat_template": has_chat_template,
        "identity_hints": identity_hints,
        "registered_instruct_identity": registered_instruct_identity,
    }


def _visible_gpu_tokens(visible_env: str | None, device_count: int) -> list[str]:
    if visible_env and visible_env.strip() and visible_env.strip() != "-1":
        tokens = [x.strip() for x in visible_env.split(",") if x.strip()]
        if len(tokens) != device_count:
            # CUDA may accept UUIDs or masks that PyTorch normalizes differently. Use
            # local indices rather than guessing physical identities in that case.
            return [str(i) for i in range(device_count)]
        return tokens
    return [str(i) for i in range(device_count)]


def resolve_gpu_ids(
    gpu_spec: str = "auto",
    legacy_gpu: str | None = None,
    *,
    visible_env: str | None = None,
    device_count: int | None = None,
) -> list[str]:
    """Resolve a deterministic list of visible GPU tokens for child processes."""
    if legacy_gpu is not None:
        if gpu_spec != "auto":
            raise ValueError("Use only one of --gpus or legacy --gpu")
        gpu_spec = legacy_gpu
    if device_count is None:
        device_count = torch.cuda.device_count()
    if device_count <= 0:
        raise RuntimeError("A CUDA GPU is required for the Countdown pilot")
    if visible_env is None:
        visible_env = os.environ.get("CUDA_VISIBLE_DEVICES")
    available = _visible_gpu_tokens(visible_env, device_count)
    if gpu_spec == "auto":
        return available[:8]
    requested = [x.strip() for x in gpu_spec.split(",") if x.strip()]
    if not requested:
        raise ValueError("--gpus must be 'auto' or a non-empty comma-separated list")
    if len(set(requested)) != len(requested):
        raise ValueError(f"Duplicate GPU ids are not allowed: {requested}")
    unavailable = [x for x in requested if x not in available]
    if unavailable:
        raise ValueError(f"Requested GPUs are not visible: {unavailable}; visible={available}")
    return requested


def _parent_gpu_index(gpu_id: str) -> int:
    available = _visible_gpu_tokens(
        os.environ.get("CUDA_VISIBLE_DEVICES"), torch.cuda.device_count()
    )
    return available.index(gpu_id)


def gpu_memory_gib(index: int = 0) -> float:
    if not torch.cuda.is_available():
        return 0.0
    props = torch.cuda.get_device_properties(index)
    return float(props.total_memory / (1024 ** 3))


def resolve_execution_plan(
    model_path: str,
    preset: str,
    memory_mode: str,
    gpu_index: int = 0,
    gpu_visible: str = "0",
) -> dict[str, Any]:
    """Resolve model-size preset, precision, and safe single-process batches."""
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for the formal Countdown arena")
    meta = read_model_metadata(model_path)
    params_b = meta.get("estimated_params_b") or 7.0
    mem = gpu_memory_gib(gpu_index)
    if preset == "auto":
        preset = "0.5b" if params_b <= 1.0 else ("3b" if params_b <= 4.5 else "7b")
    if memory_mode == "auto":
        bf16_need = 18.0 if params_b <= 1.0 else (34.0 if params_b <= 4.5 else 60.0)
        memory_mode = "bf16" if mem >= bf16_need else "qlora"
    bf16_supported = torch.cuda.is_bf16_supported()
    if memory_mode == "bf16":
        load_in_4bit = False
        dtype = "bf16" if bf16_supported else "fp16"
    elif memory_mode == "qlora":
        load_in_4bit = True
        dtype = "bf16" if bf16_supported else "fp16"
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
        "gpu_visible": gpu_visible,
        "gpu_name": torch.cuda.get_device_name(gpu_index),
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


def serializable_namespace(args: argparse.Namespace) -> dict[str, Any]:
    """Return argparse values that are safe to persist in JSON manifests.

    ``set_defaults(func=...)`` stores a callable in the Namespace.  Serializing
    ``vars(args)`` directly therefore fails only after an expensive stage has
    completed.  Filter every callable centrally rather than patching individual
    manifests ad hoc.
    """
    payload: dict[str, Any] = {}
    for key, value in vars(args).items():
        if callable(value):
            continue
        payload[key] = str(value) if isinstance(value, Path) else value
    return payload


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def csv_safe_row(row: dict[str, Any]) -> dict[str, Any]:
    """Serialize nested diagnostics deterministically before CSV emission."""
    return {
        key: (
            json.dumps(value, ensure_ascii=False, sort_keys=True)
            if isinstance(value, (dict, list, tuple))
            else value
        )
        for key, value in row.items()
    }


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
    parameterization: str = "lora",
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
    if parameterization not in {"lora", "full"}:
        raise ValueError(f"Unknown parameterization: {parameterization}")
    if parameterization == "full" and load_in_4bit:
        raise ValueError("Full fine-tuning cannot use 4-bit loading")
    if parameterization == "full" and adapter_path:
        raise ValueError("Full fine-tuning starts from model_path and cannot load a LoRA adapter")
    if load_in_4bit:
        model = prepare_model_for_kbit_training(model)
    if parameterization == "full":
        for parameter in model.parameters():
            parameter.requires_grad_(bool(trainable_adapter))
    elif adapter_path:
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


def detached_token_surprisal_taper(
    stats: dict[str, torch.Tensor],
    exp_lambda: float,
    surprisal_threshold: float,
) -> torch.Tensor:
    """Return stop-gradient token weights from current policy surprisal.

    The offline completion identity stays fixed, but its policy-relative
    remoteness is recomputed at every optimizer step.  A token that was near at
    data construction therefore receives the same taper as an initially-far
    token once their current surprisals are equal.
    """
    token_surprisal = -stats["token_lp"].detach()
    return torch.exp(
        -exp_lambda * F.relu(token_surprisal - surprisal_threshold)
    )


def controlled_negative_token_weights(
    method: str,
    near: dict[str, torch.Tensor],
    far: dict[str, torch.Tensor],
    exp_lambda: float,
    surprisal_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return token weights for the static and dynamic control ablation.

    ``controlled_negative`` preserves the V4.2 static-label behavior: only the
    construction-time far branch is tapered. ``dynamic_controlled_negative``
    applies the same current-policy taper to both branches, so initial labels do
    not freeze their long-run treatment. Other methods receive unit token weights.
    """
    near_weights = torch.ones_like(near["token_lp"])
    far_weights = torch.ones_like(far["token_lp"])
    if method == "controlled_negative":
        far_weights = detached_token_surprisal_taper(
            far, exp_lambda, surprisal_threshold
        )
    elif method == "dynamic_controlled_negative":
        near_weights = detached_token_surprisal_taper(
            near, exp_lambda, surprisal_threshold
        )
        far_weights = detached_token_surprisal_taper(
            far, exp_lambda, surprisal_threshold
        )
    return near_weights, far_weights


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
) -> dict[str, Any]:
    seed_all(seed)
    was_training = bool(model.training)
    successes: list[float] = []
    valid: list[float] = []
    sampled_successes: list[float] = []
    greedy_unseen_presences: list[float] = []
    pass_unseen_presences: list[float] = []
    greedy_unseen_successes: list[float] = []
    pass_unseen_successes: list[float] = []
    observed_correct_structures: set[str] = set()
    greedy_correct_structures: set[str] = set()
    sampled_correct_structures: set[str] = set()
    target_structures = {
        row.get("oracle_structure") or expression_structure(row["oracle"])
        for row in rows
    }
    heldout_targets = (
        target_structures - known_structures if known_structures is not None else set()
    )
    greedy_pattern_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"attempts": 0, "correct": 0}
    )
    sampled_pattern_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"attempts": 0, "correct": 0}
    )

    def record_pattern(
        counts: dict[str, dict[str, int]], pattern: str, correct: bool
    ) -> None:
        counts[pattern]["attempts"] += 1
        counts[pattern]["correct"] += int(correct)

    def finalize_pattern_counts(
        counts: dict[str, dict[str, int]]
    ) -> dict[str, dict[str, float | int | None]]:
        output: dict[str, dict[str, float | int | None]] = {}
        for pattern in sorted(heldout_targets):
            attempts = int(counts.get(pattern, {}).get("attempts", 0))
            correct = int(counts.get(pattern, {}).get("correct", 0))
            output[pattern] = {
                "attempts": attempts,
                "correct": correct,
                "precision": float(correct / attempts) if attempts else None,
            }
        return output

    def aggregate_pattern_precision(
        counts: dict[str, dict[str, int]]
    ) -> tuple[float, float, int, int]:
        attempts = sum(int(item["attempts"]) for item in counts.values())
        correct = sum(int(item["correct"]) for item in counts.values())
        per_pattern = [
            float(item["correct"] / item["attempts"])
            for item in counts.values()
            if item["attempts"] > 0
        ]
        micro = float(correct / attempts) if attempts else 0.0
        macro = float(np.mean(per_pattern)) if per_pattern else 0.0
        return micro, macro, attempts, correct

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
            greedy_unseen_presence = False
            greedy_unseen = False
            if greedy_check["valid_format"] and greedy_check["uses_numbers"]:
                try:
                    pattern = expression_structure(greedy_check["expression"])
                    if pattern in heldout_targets:
                        record_pattern(
                            greedy_pattern_counts, pattern, bool(greedy_check["correct"])
                        )
                    greedy_unseen_presence = (
                        known_structures is not None and pattern not in known_structures
                    )
                    if greedy_check["correct"]:
                        observed_correct_structures.add(pattern)
                        greedy_correct_structures.add(pattern)
                        greedy_unseen = greedy_unseen_presence
                except Exception:
                    pass
            greedy_unseen_presences.append(float(greedy_unseen_presence))
            greedy_unseen_successes.append(float(greedy_unseen))

            any_correct = False
            any_unseen_presence = False
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
                    record_pattern(sampled_pattern_counts, pattern, bool(check["correct"]))
                if known_structures is not None and pattern not in known_structures:
                    any_unseen_presence = True
                if not check["correct"]:
                    continue
                any_correct = True
                observed_correct_structures.add(pattern)
                sampled_correct_structures.add(pattern)
                if known_structures is not None and pattern not in known_structures:
                    any_unseen_correct = True
            sampled_successes.append(float(any_correct))
            pass_unseen_presences.append(float(any_unseen_presence))
            pass_unseen_successes.append(float(any_unseen_correct))

    heldout_correct_patterns = observed_correct_structures & heldout_targets
    greedy_heldout_correct_patterns = greedy_correct_structures & heldout_targets
    sampled_heldout_correct_patterns = sampled_correct_structures & heldout_targets
    greedy_micro, greedy_macro, greedy_attempts, greedy_correct = (
        aggregate_pattern_precision(greedy_pattern_counts)
    )
    sampled_micro, sampled_macro, sampled_attempts, sampled_correct = (
        aggregate_pattern_precision(sampled_pattern_counts)
    )
    metrics = {
        "greedy_success": float(np.mean(successes)),
        "pass_at_k": float(np.mean(sampled_successes)),
        "valid_rate": float(np.mean(valid)),
        "greedy_unseen_structure_presence": float(np.mean(greedy_unseen_presences)),
        "greedy_unseen_structure_success": float(np.mean(greedy_unseen_successes)),
        "pass_at_k_unseen_structure_presence": float(np.mean(pass_unseen_presences)),
        "pass_at_k_unseen_structure": float(np.mean(pass_unseen_successes)),
        "pass_at_k_unseen_structure_success": float(np.mean(pass_unseen_successes)),
        "unique_correct_structures": float(len(observed_correct_structures)),
        "heldout_pattern_coverage": (
            float(len(heldout_correct_patterns) / len(heldout_targets))
            if heldout_targets else 0.0
        ),
        "greedy_heldout_pattern_coverage": (
            float(len(greedy_heldout_correct_patterns) / len(heldout_targets))
            if heldout_targets else 0.0
        ),
        "sampled_heldout_pattern_coverage": (
            float(len(sampled_heldout_correct_patterns) / len(heldout_targets))
            if heldout_targets else 0.0
        ),
        "greedy_heldout_pattern_precision_micro": greedy_micro,
        "greedy_heldout_pattern_precision_macro": greedy_macro,
        "sampled_heldout_pattern_precision_micro": sampled_micro,
        "sampled_heldout_pattern_precision_macro": sampled_macro,
        # Backward-compatible aliases now refer to the sampled-generation metric,
        # not a pooled greedy+sampled denominator.
        "heldout_pattern_precision": sampled_micro,
        "heldout_pattern_family_coverage": (
            float(len(heldout_correct_patterns) / len(heldout_targets))
            if heldout_targets else 0.0
        ),
        "heldout_pattern_family_precision_micro": sampled_micro,
        "heldout_pattern_family_precision_macro": sampled_macro,
        "heldout_pattern_attempts": float(sampled_attempts),
        "greedy_heldout_pattern_attempts": float(greedy_attempts),
        "greedy_heldout_pattern_correct": float(greedy_correct),
        "sampled_heldout_pattern_attempts": float(sampled_attempts),
        "sampled_heldout_pattern_correct": float(sampled_correct),
        "heldout_patterns_observed_correct": float(len(heldout_correct_patterns)),
        "correct_heldout_patterns": float(len(heldout_correct_patterns)),
        "heldout_patterns_total": float(len(heldout_targets)),
        "per_pattern_precision": {
            "greedy": finalize_pattern_counts(greedy_pattern_counts),
            "sampled": finalize_pattern_counts(sampled_pattern_counts),
        },
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
        "parameterization": "lora" if adapter_config is not None else "full_model",
        "adapter_config": adapter_config,
        "files": files,
        "total_size_bytes": sum(item["size_bytes"] for item in files),
    }


def save_local_model_checkpoint(
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
    if (destination / "adapter_config.json").exists():
        # LoRA evaluation always loads the frozen tokenizer from model_path.
        # Duplicating the multi-megabyte tokenizer in every adapter checkpoint
        # bloats durable evidence without improving reproducibility.
        (destination / "tokenizer_reference.json").write_text(json.dumps({
            "source": getattr(tokenizer, "name_or_path", None),
            "load_from_registered_model_path": True,
        }, indent=2))
    else:
        # A full-model diagnostic is evaluated directly from this directory and
        # therefore remains self-describing with its tokenizer metadata.
        tokenizer.save_pretrained(destination)
    return checkpoint_inventory(destination, kind, step)


def _trainable_parameters_finite(parameters: Sequence[torch.nn.Parameter]) -> bool:
    return all(bool(torch.isfinite(parameter.detach()).all()) for parameter in parameters)


def snapshot_trainable_parameters(
    parameters: Sequence[torch.nn.Parameter],
) -> list[torch.Tensor]:
    """Capture the current finite trainable state before one optimizer update.

    Only trainable adapter tensors are copied to CPU. This keeps the audit
    checkpoint exact without writing an adapter checkpoint on every step.
    """
    return [parameter.detach().cpu().clone() for parameter in parameters]


def restore_trainable_parameters(
    parameters: Sequence[torch.nn.Parameter], snapshot: Sequence[torch.Tensor]
) -> None:
    if len(parameters) != len(snapshot):
        raise ValueError("Trainable-parameter snapshot length mismatch")
    with torch.no_grad():
        for parameter, saved in zip(parameters, snapshot):
            if tuple(parameter.shape) != tuple(saved.shape):
                raise ValueError("Trainable-parameter snapshot shape mismatch")
            parameter.copy_(saved.to(device=parameter.device, dtype=parameter.dtype))


def optimizer_step_with_last_finite_guard(
    optimizer: torch.optim.Optimizer,
    parameters: Sequence[torch.nn.Parameter],
) -> bool:
    """Apply one update and restore the exact pre-step state on nonfinite output."""
    snapshot = snapshot_trainable_parameters(parameters)
    optimizer.step()
    if _trainable_parameters_finite(parameters):
        return True
    restore_trainable_parameters(parameters, snapshot)
    return False


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
    """Train the registered reference policy with LoRA or a full-FT diagnostic.

    The LoRA branch is the only branch eligible to initialize the four-method
    comparison.  The full-parameter branch is an isolated capacity diagnostic;
    it is evaluated and reported but never substituted into the LoRA ranking.
    """
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
        parameterization=args.parameterization,
    )
    device = next(model.parameters()).device
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("SFT has no trainable parameters")
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    updates_per_epoch = math.ceil(len(loader) / args.grad_accum)
    total_updates = max(1, updates_per_epoch * args.epochs)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, max(1, int(total_updates * args.warmup_ratio)), total_updates
    )
    out_dir = Path(args.output_dir)
    ensure_checkpoint_output_is_local_or_ignored(out_dir)
    suffix = "adapter" if args.parameterization == "lora" else "model"
    best_dir = out_dir / f"best_{suffix}"
    terminal_dir = out_dir / f"terminal_{suffix}"
    last_finite_dir = out_dir / f"last_finite_{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    best_value = -float("inf")
    best_epoch = -1
    stale_epochs = 0
    eval_rows: list[dict[str, Any]] = []
    checkpoint_records: list[dict[str, Any]] = []
    numerical_failure: str | None = None
    failure_detected_at_step: int | None = None
    last_finite_step: int | None = None
    stop_reason = "max_epochs"
    global_step = 0
    model.train()
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(args.epochs):
        running_loss = 0.0
        micro_count = 0
        for batch_index, batch in enumerate(loader):
            batch = move_to_device(batch, device)
            raw_loss = model(**batch, use_cache=False).loss
            if not bool(torch.isfinite(raw_loss)):
                numerical_failure = f"nonfinite_loss_at_update_{global_step + 1}"
                failure_detected_at_step = global_step + 1
                last_finite_step = global_step
                stop_reason = numerical_failure
                break
            (raw_loss / args.grad_accum).backward()
            running_loss += float(raw_loss.detach())
            micro_count += 1
            if (batch_index + 1) % args.grad_accum == 0 or batch_index + 1 == len(loader):
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
                if not bool(torch.isfinite(grad_norm)):
                    numerical_failure = f"nonfinite_gradient_at_update_{global_step + 1}"
                    failure_detected_at_step = global_step + 1
                    last_finite_step = global_step
                    stop_reason = numerical_failure
                    break
                candidate_step = global_step + 1
                if args.parameterization == "lora":
                    applied = optimizer_step_with_last_finite_guard(optimizer, trainable)
                else:
                    # Copying every full-model tensor to CPU before each optimizer
                    # step would dominate this diagnostic.  Full FT therefore fails
                    # closed on a nonfinite post-step state and keeps the most recent
                    # epoch checkpoint; it is never used for the main method ranking.
                    optimizer.step()
                    applied = _trainable_parameters_finite(trainable)
                if not applied:
                    numerical_failure = f"nonfinite_parameters_at_update_{candidate_step}"
                    failure_detected_at_step = candidate_step
                    last_finite_step = global_step
                    stop_reason = numerical_failure
                    break
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step = candidate_step
                if global_step % args.log_every == 0:
                    print(json.dumps({
                        "stage": "sft",
                        "parameterization": args.parameterization,
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
        row = {
            "epoch": epoch + 1,
            "update": global_step,
            "effective_train_epochs": epoch + 1,
            "parameterization": args.parameterization,
            **metrics,
        }
        eval_rows.append(row)
        print("SFT_EVAL", json.dumps(row))
        value = float(metrics[args.selection_metric])
        if value > best_value + args.selection_delta:
            best_value = value
            best_epoch = epoch + 1
            stale_epochs = 0
            checkpoint_records = [
                record for record in checkpoint_records if record["kind"] != "best"
            ]
            checkpoint_records.append(save_local_model_checkpoint(
                model, tokenizer, best_dir, "best", global_step
            ))
        else:
            stale_epochs += 1
        model.train()
        if epoch + 1 >= args.min_epochs and stale_epochs >= args.early_stop_patience:
            stop_reason = "early_stop_patience"
            break

    if numerical_failure:
        if args.parameterization == "lora":
            checkpoint_records.append(save_local_model_checkpoint(
                model,
                tokenizer,
                last_finite_dir,
                "last_finite",
                global_step if last_finite_step is None else last_finite_step,
            ))
    else:
        checkpoint_records.append(save_local_model_checkpoint(
            model, tokenizer, terminal_dir, "terminal", global_step
        ))
    if best_epoch < 0:
        best_value = float("nan")
    with (out_dir / "sft_metrics.csv").open("w", newline="") as handle:
        if eval_rows:
            writer = csv.DictWriter(handle, fieldnames=list(eval_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_safe_row(row) for row in eval_rows)
    manifest = {
        **serializable_namespace(args),
        "source_provenance": source_provenance(),
        "best_epoch": best_epoch,
        "best_value": best_value,
        "updates_per_effective_epoch": updates_per_epoch,
        "completed_effective_epochs": len(eval_rows),
        "terminal_step": global_step if not numerical_failure else None,
        "failure_detected_at_step": failure_detected_at_step,
        "last_finite_step": last_finite_step,
        "numerical_failure": numerical_failure,
        "stop_reason": stop_reason,
        "main_method_eligibility": args.parameterization == "lora",
        "full_ft_role": (
            "isolated_reference_capacity_diagnostic"
            if args.parameterization == "full" else None
        ),
        "last_finite_semantics": (
            "exact_pre_step_trainable_lora_snapshot"
            if args.parameterization == "lora"
            else "epoch_checkpoint_only; diagnostic fails closed on nonfinite parameters"
        ),
        "checkpoint_policy": "server-local model state only; binaries must not enter Git/artifact packages",
        "checkpoints": checkpoint_records,
        "result_status": args.result_status,
    }
    (out_dir / "sft_manifest.json").write_text(json.dumps(manifest, indent=2))
    (out_dir / "checkpoint_manifest.json").write_text(json.dumps({
        "local_only": True,
        "result_status": args.result_status,
        "model_path": args.model_path,
        "parameterization": args.parameterization,
        "source_provenance": source_provenance(),
        "checkpoints": checkpoint_records,
    }, indent=2))
    if numerical_failure:
        raise RuntimeError(f"SFT stopped with {numerical_failure}")
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


def balanced_pattern_quotas(patterns: Sequence[str], total: int) -> dict[str, int]:
    """Allocate a deterministic near-equal quota over canonical patterns."""
    ordered = sorted(set(patterns))
    if not ordered:
        raise ValueError("At least one pattern is required")
    if total < 0:
        raise ValueError("total must be non-negative")
    base, extra = divmod(total, len(ordered))
    return {
        pattern: base + int(index < extra)
        for index, pattern in enumerate(ordered)
    }


def build_nested_balanced_subsets(
    rows: Sequence[dict[str, Any]], sizes: Sequence[int]
) -> dict[int, list[dict[str, Any]]]:
    """Create deterministic, pattern-balanced nested prefixes from one full set."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        pattern = row.get("oracle_structure") or expression_structure(row["oracle"])
        grouped[pattern].append(row)
    patterns = sorted(grouped)
    output: dict[int, list[dict[str, Any]]] = {}
    previous_ids: set[str] = set()
    for size in sorted(set(int(value) for value in sizes)):
        if size <= 0 or size > len(rows):
            raise ValueError(f"Nested subset size {size} is outside 1..{len(rows)}")
        quotas = balanced_pattern_quotas(patterns, size)
        chosen_ids: set[str] = set()
        for pattern in patterns:
            if len(grouped[pattern]) < quotas[pattern]:
                raise RuntimeError(
                    f"Pattern {pattern} has {len(grouped[pattern])} rows but nested quota "
                    f"requires {quotas[pattern]}"
                )
            chosen_ids.update(str(row["id"]) for row in grouped[pattern][: quotas[pattern]])
        if not previous_ids.issubset(chosen_ids):
            raise AssertionError("Nested balanced subsets must be monotone")
        subset = [row for row in rows if str(row["id"]) in chosen_ids]
        if len(subset) != size:
            raise AssertionError("Nested balanced subset has the wrong size")
        output[size] = subset
        previous_ids = chosen_ids
    return output


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
    if target_examples > len(rows):
        raise RuntimeError(
            f"Requested {target_examples} matched rows from only {len(rows)} unique prompts"
        )
    train_patterns = sorted(set(split_manifest.get("train_patterns", allowed_patterns)))
    pattern_quotas = (
        balanced_pattern_quotas(train_patterns, target_examples)
        if args.balance_by_oracle_pattern else {}
    )
    accepted_by_pattern: Counter[str] = Counter()
    attempted_by_pattern: Counter[str] = Counter()
    dropped_by_pattern: Counter[str] = Counter()
    rng = random.Random(args.seed)
    output_rows: list[dict[str, Any]] = []
    diagnostics: Counter[str] = Counter()
    candidate_counts: list[int] = []

    def quota_open(row: dict[str, Any]) -> bool:
        if not pattern_quotas:
            return len(output_rows) < target_examples
        pattern = row.get("oracle_structure") or expression_structure(row["oracle"])
        if pattern not in pattern_quotas:
            raise AssertionError(f"Training row pattern is absent from the registered core: {pattern}")
        return accepted_by_pattern[pattern] < pattern_quotas[pattern]

    # Initial rollouts remain one deterministic RNG stream.  Pattern quotas are
    # enforced before generation so hard/easy patterns cannot silently rebalance
    # the saved data through differential matched-pair acceptance.
    for start_index in range(0, len(rows), args.batch_size):
        if len(output_rows) >= target_examples:
            break
        raw_chunk = rows[start_index : start_index + args.batch_size]
        chunk = [row for row in raw_chunk if quota_open(row)]
        diagnostics["skipped_quota_filled"] += len(raw_chunk) - len(chunk)
        if not chunk:
            continue
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
            if not quota_open(row):
                diagnostics["skipped_quota_filled"] += 1
                continue
            oracle_structure = row.get("oracle_structure") or expression_structure(row["oracle"])
            diagnostics["attempted_rows"] += 1
            attempted_by_pattern[oracle_structure] += 1
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

            if near_item is None or far_item is None:
                rescue_texts: list[str] = []
                for _ in range(args.synthetic_rescue_candidates):
                    try:
                        rescue_texts.append(make_valid_wrong_expression(
                            row, rng, avoid=seen.union(rescue_texts),
                            allowed_patterns=allowed_patterns,
                        ))
                    except RuntimeError:
                        diagnostics["synthetic_rescue_generation_failure"] += 1
                        break
                if rescue_texts:
                    diagnostics["synthetic_rescue_candidates"] += len(rescue_texts)
                    evaluated.extend(_score_new_candidates(
                        model, tokenizer, row, rescue_texts, seen, allowed_patterns,
                        args.max_length, args.score_batch_size, diagnostics,
                    ))
                    rescue_detailed = [
                        candidate_metadata(item, tokenizer)
                        for item in evaluated if not item["correct"]
                    ]
                    near_item, far_item, rescue_matched = select_matched_negative_pair(
                        rescue_detailed,
                        args.min_surprisal_gap,
                        args.max_token_length_diff,
                        args.max_tree_depth_diff,
                        args.max_value_error_ratio,
                    )
                    if rescue_matched:
                        matched_round = args.pair_resample_rounds + 1
                        diagnostics["matched_after_synthetic_rescue"] += 1

            valid_wrong = [item for item in evaluated if not item["correct"]]
            candidate_counts.append(len(valid_wrong))
            if near_item is None or far_item is None or matched_round is None:
                diagnostics["dropped_unmatched"] += 1
                dropped_by_pattern[oracle_structure] += 1
                diagnostics.update({
                    f"drop_reason_{key}": value
                    for key, value in _pair_failure_summary(
                        [candidate_metadata(item, tokenizer) for item in valid_wrong], args
                    ).items()
                })
                continue

            correct = [item for item in evaluated if item["correct"]]
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
            accepted_by_pattern[oracle_structure] += 1
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
                "surprisal_gap": float(far_item["surprisal"] - near_item["surprisal"]),
                "near_token_length": int(near_item["token_length"]),
                "far_token_length": int(far_item["token_length"]),
                "near_tree_depth": int(near_item["tree_depth"]),
                "far_tree_depth": int(far_item["tree_depth"]),
                "near_value_error": float(near_item["value_error"]),
                "far_value_error": float(far_item["value_error"]),
                "pair_matched": True,
                "matched_after_resample_round": matched_round,
            })
            if len(output_rows) % 100 == 0:
                completed = sum(
                    accepted_by_pattern[p] >= q for p, q in pattern_quotas.items()
                ) if pattern_quotas else None
                print(
                    f"built matched {len(output_rows)}/{target_examples}; "
                    f"pattern_quotas_complete={completed}",
                    flush=True,
                )

    missing_by_pattern = {
        pattern: quota - accepted_by_pattern[pattern]
        for pattern, quota in pattern_quotas.items()
        if accepted_by_pattern[pattern] < quota
    }
    partial_manifest = {
        **serializable_namespace(args),
        **dict(diagnostics),
        "reference_adapter": reference_adapter,
        "input_rows": len(rows),
        "requested_examples": target_examples,
        "examples": len(output_rows),
        "balance_by_oracle_pattern": bool(pattern_quotas),
        "per_pattern_quota": pattern_quotas,
        "per_pattern_attempted": dict(sorted(attempted_by_pattern.items())),
        "per_pattern_accepted": dict(sorted(accepted_by_pattern.items())),
        "per_pattern_dropped_unmatched": dict(sorted(dropped_by_pattern.items())),
        "missing_by_pattern": missing_by_pattern,
        "source_provenance": source_provenance(),
    }
    if len(output_rows) < target_examples or missing_by_pattern:
        write_jsonl(args.output_data, output_rows)
        Path(str(args.output_data) + ".manifest.json").write_text(json.dumps({
            **partial_manifest,
            "status": "insufficient_pattern_balanced_matched_pairs",
        }, indent=2))
        raise RuntimeError(
            f"Only {len(output_rows)}/{target_examples} balanced matched rows were "
            f"constructed; missing patterns={missing_by_pattern}. Partial evidence was preserved."
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
    nested_sizes = [
        int(value) for value in args.nested_sizes.split(",") if value.strip()
    ] if args.nested_sizes else []
    if target_examples not in nested_sizes:
        nested_sizes.append(target_examples)
    nested_outputs: dict[str, Any] = {}
    subsets = build_nested_balanced_subsets(output_rows, nested_sizes)
    nested_dir = Path(args.nested_output_dir) if args.nested_output_dir else Path(args.output_data).parent
    nested_dir.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output_data).resolve()
    for size, subset in subsets.items():
        subset_path = output_path if size == target_examples else (nested_dir / f"offline_{size}.jsonl").resolve()
        if subset_path != output_path:
            write_jsonl(subset_path, subset)
        counts = Counter(
            row.get("oracle_structure") or expression_structure(row["oracle"])
            for row in subset
        )
        nested_outputs[str(size)] = {
            "path": str(subset_path),
            "sha256": _sha256_file(subset_path),
            "rows": len(subset),
            "per_pattern_min": min(counts.values()),
            "per_pattern_max": max(counts.values()),
            "nested_in_next_larger_subset": True,
        }

    manifest = {
        **partial_manifest,
        "source_provenance": source_provenance(),
        "status": "complete",
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
        "nested_balanced_subsets": nested_outputs,
        "formal_interpretation": (
            "all saved rows are matched and oracle-pattern quotas are enforced; "
            "dropped prompts and per-pattern acceptance are reported"
        ),
    }
    Path(str(args.output_data) + ".manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )
    print(json.dumps(manifest, indent=2))

def balanced_diagnostic_rows(
    rows: Sequence[dict[str, Any]], max_examples: int
) -> list[dict[str, Any]]:
    """Select a deterministic round-robin sample across oracle patterns."""
    if max_examples <= 0 or max_examples >= len(rows):
        return list(rows)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        pattern = row.get("oracle_structure") or expression_structure(row["oracle"])
        grouped[pattern].append(row)
    selected: list[dict[str, Any]] = []
    index = 0
    patterns = sorted(grouped)
    while len(selected) < max_examples:
        progressed = False
        for pattern in patterns:
            if index < len(grouped[pattern]):
                selected.append(grouped[pattern][index])
                progressed = True
                if len(selected) >= max_examples:
                    break
        if not progressed:
            break
        index += 1
    return selected


def _gradient_norm_and_dot(
    left: Sequence[torch.Tensor | None], right: Sequence[torch.Tensor | None]
) -> tuple[float, float, float]:
    left_sq = torch.zeros((), dtype=torch.float64)
    right_sq = torch.zeros((), dtype=torch.float64)
    dot = torch.zeros((), dtype=torch.float64)
    for left_grad, right_grad in zip(left, right):
        if left_grad is not None:
            left_cpu = left_grad.detach().double().cpu()
            left_sq += left_cpu.square().sum()
        else:
            left_cpu = None
        if right_grad is not None:
            right_cpu = right_grad.detach().double().cpu()
            right_sq += right_cpu.square().sum()
        else:
            right_cpu = None
        if left_cpu is not None and right_cpu is not None:
            dot += (left_cpu * right_cpu).sum()
    left_norm = float(left_sq.sqrt())
    right_norm = float(right_sq.sqrt())
    cosine = float(dot / (left_sq.sqrt() * right_sq.sqrt()).clamp_min(1e-30))
    return left_norm, right_norm, cosine


def dynamic_negative_diagnostics(
    model: Any,
    tokenizer: Any,
    rows: Sequence[dict[str, Any]],
    trainable: Sequence[torch.nn.Parameter],
    *,
    max_examples: int,
    gradient_examples: int,
    batch_size: int,
    max_length: int,
    exp_lambda: float,
    surprisal_threshold: float,
    negative_scale: float | None,
) -> dict[str, float]:
    """Audit policy-relative remoteness and branch influence at one checkpoint."""
    diagnostic_rows = balanced_diagnostic_rows(rows, max_examples)
    if not diagnostic_rows:
        raise RuntimeError("Dynamic diagnostics require at least one offline row")
    was_training = bool(model.training)
    model.eval()
    dataset = OfflineDataset(list(diagnostic_rows), tokenizer, max_length)
    loader = DataLoader(
        dataset,
        batch_size=max(1, batch_size),
        shuffle=False,
        collate_fn=make_offline_collator(tokenizer.pad_token_id),
        num_workers=0,
    )
    device = next(model.parameters()).device
    positive_surprisal: list[float] = []
    near_surprisal: list[float] = []
    far_surprisal: list[float] = []
    near_weights: list[float] = []
    far_weights: list[float] = []
    near_crossings = 0
    far_crossings = 0
    total_sequences = 0
    with torch.no_grad():
        for packed in loader:
            pos = completion_stats(model, move_to_device(packed["positive"], device))
            near = completion_stats(model, move_to_device(packed["near"], device))
            far = completion_stats(model, move_to_device(packed["far"], device))
            pos_values = (-pos["seq_lp"]).detach().float().cpu().tolist()
            near_values = (-near["seq_lp"]).detach().float().cpu().tolist()
            far_values = (-far["seq_lp"]).detach().float().cpu().tolist()
            positive_surprisal.extend(float(value) for value in pos_values)
            near_surprisal.extend(float(value) for value in near_values)
            far_surprisal.extend(float(value) for value in far_values)
            near_crossings += sum(value > surprisal_threshold for value in near_values)
            far_crossings += sum(value > surprisal_threshold for value in far_values)
            total_sequences += len(near_values)
            near_token_weights = detached_token_surprisal_taper(
                near, exp_lambda, surprisal_threshold
            )
            far_token_weights = detached_token_surprisal_taper(
                far, exp_lambda, surprisal_threshold
            )
            near_weighted = (
                (near_token_weights * near["token_mask"]).sum(-1)
                / near["token_mask"].sum(-1).clamp_min(1)
            )
            far_weighted = (
                (far_token_weights * far["token_mask"]).sum(-1)
                / far["token_mask"].sum(-1).clamp_min(1)
            )
            near_weights.extend(
                float(value) for value in near_weighted.float().cpu().tolist()
            )
            far_weights.extend(
                float(value) for value in far_weighted.float().cpu().tolist()
            )

    gradient_rows = diagnostic_rows[: max(1, min(gradient_examples, len(diagnostic_rows)))]
    gradient_dataset = OfflineDataset(list(gradient_rows), tokenizer, max_length)
    gradient_batch = make_offline_collator(tokenizer.pad_token_id)(
        [gradient_dataset[index] for index in range(len(gradient_dataset))]
    )
    positive_batch = move_to_device(gradient_batch["positive"], device)
    near_batch = move_to_device(gradient_batch["near"], device)
    far_batch = move_to_device(gradient_batch["far"], device)

    model.zero_grad(set_to_none=True)
    positive = completion_stats(model, positive_batch)
    positive_grads = torch.autograd.grad(
        -positive["seq_lp"].mean(), trainable, allow_unused=True
    )
    model.zero_grad(set_to_none=True)
    near = completion_stats(model, near_batch)
    near_grads = torch.autograd.grad(
        near["seq_lp"].mean(), trainable, allow_unused=True
    )
    pos_norm, near_norm, pos_near_cosine = _gradient_norm_and_dot(
        positive_grads, near_grads
    )
    del near_grads

    model.zero_grad(set_to_none=True)
    near = completion_stats(model, near_batch)
    near_token_weights = detached_token_surprisal_taper(
        near, exp_lambda, surprisal_threshold
    )
    near_controlled_grads = torch.autograd.grad(
        weighted_sequence_logprob(near, near_token_weights).mean(),
        trainable,
        allow_unused=True,
    )
    _, near_controlled_norm, pos_near_controlled_cosine = _gradient_norm_and_dot(
        positive_grads, near_controlled_grads
    )
    del near_controlled_grads

    model.zero_grad(set_to_none=True)
    far = completion_stats(model, far_batch)
    far_raw_grads = torch.autograd.grad(
        far["seq_lp"].mean(), trainable, allow_unused=True
    )
    _, far_raw_norm, pos_far_raw_cosine = _gradient_norm_and_dot(
        positive_grads, far_raw_grads
    )
    del far_raw_grads

    model.zero_grad(set_to_none=True)
    far = completion_stats(model, far_batch)
    far_token_weights = detached_token_surprisal_taper(
        far, exp_lambda, surprisal_threshold
    )
    far_controlled_grads = torch.autograd.grad(
        weighted_sequence_logprob(far, far_token_weights).mean(),
        trainable,
        allow_unused=True,
    )
    _, far_controlled_norm, pos_far_controlled_cosine = _gradient_norm_and_dot(
        positive_grads, far_controlled_grads
    )
    model.zero_grad(set_to_none=True)
    scale = float(negative_scale) if negative_scale is not None else 1.0
    result = {
        "diagnostic_examples": float(len(diagnostic_rows)),
        "diagnostic_gradient_examples": float(len(gradient_rows)),
        "positive_surprisal_mean": float(np.mean(positive_surprisal)),
        "positive_surprisal_median": float(np.median(positive_surprisal)),
        "positive_surprisal_p90": float(np.quantile(positive_surprisal, 0.9)),
        "near_surprisal_mean": float(np.mean(near_surprisal)),
        "near_surprisal_median": float(np.median(near_surprisal)),
        "near_surprisal_p90": float(np.quantile(near_surprisal, 0.9)),
        "far_surprisal_mean": float(np.mean(far_surprisal)),
        "far_surprisal_median": float(np.median(far_surprisal)),
        "far_surprisal_p90": float(np.quantile(far_surprisal, 0.9)),
        "far_over_near_surprisal_ratio": float(
            np.mean(far_surprisal) / max(np.mean(near_surprisal), 1e-12)
        ),
        "near_dynamic_far_fraction": float(near_crossings / max(total_sequences, 1)),
        "far_above_threshold_fraction": float(far_crossings / max(total_sequences, 1)),
        "controlled_near_token_weight_mean": float(np.mean(near_weights)),
        "controlled_far_token_weight_mean": float(np.mean(far_weights)),
        "positive_gradient_norm": pos_norm,
        "near_negative_gradient_norm_raw": near_norm,
        "near_negative_gradient_norm_controlled": near_controlled_norm,
        "far_negative_gradient_norm_raw": far_raw_norm,
        "far_negative_gradient_norm_controlled": far_controlled_norm,
        "near_negative_gradient_norm_scaled": scale * near_norm,
        "far_negative_gradient_norm_scaled": scale * far_raw_norm,
        "far_over_near_gradient_norm_ratio": far_raw_norm / max(near_norm, 1e-30),
        "positive_near_update_cosine": pos_near_cosine,
        "positive_near_controlled_update_cosine": pos_near_controlled_cosine,
        "positive_far_raw_update_cosine": pos_far_raw_cosine,
        "positive_far_controlled_update_cosine": pos_far_controlled_cosine,
    }
    if was_training:
        model.train()
    return result


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
    effective_batch_size = args.micro_batch * args.grad_accum
    updates_per_effective_epoch = math.ceil(len(train_rows) / max(effective_batch_size, 1))
    max_effective_epochs = args.steps / max(updates_per_effective_epoch, 1)
    min_effective_epochs = args.min_steps / max(updates_per_effective_epoch, 1)
    diagnostic_rows = balanced_diagnostic_rows(train_rows, args.diagnostic_examples)

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
    calibrated_negative_scale: float | None = None
    calibration: dict[str, Any] | None = None
    if args.method != "positive_only":
        if args.negative_calibration_json:
            calibration = json.loads(Path(args.negative_calibration_json).read_text())
            calibrated_negative_scale = float(calibration["negative_scale"])
        elif args.negative_scale is not None:
            calibrated_negative_scale = float(args.negative_scale)
        else:
            raise RuntimeError(
                "Negative-advantage methods require --negative_calibration_json "
                "or an explicit --negative_scale for a separately registered run"
            )
        if (
            not math.isfinite(calibrated_negative_scale)
            or calibrated_negative_scale <= 0
        ):
            raise RuntimeError("Invalid calibrated negative scale")

    if args.method == "global_matched":
        if calibration is None:
            raise RuntimeError(
                "global_matched requires --negative_calibration_json containing global_gamma"
            )
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
    last_finite_dir = out_dir / "last_finite_adapter"
    out_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_path = out_dir / "dynamic_diagnostics.jsonl"
    if diagnostics_path.exists():
        diagnostics_path.unlink()
    checkpoint_records: list[dict[str, Any]] = []
    best_value = -float("inf")
    best_step = 0
    stale_checks = 0
    numerical_failure: str | None = None
    failure_detected_at_step: int | None = None
    last_finite_step: int | None = None
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
    initial_diagnostics = dynamic_negative_diagnostics(
        model, tokenizer, diagnostic_rows, trainable,
        max_examples=args.diagnostic_examples,
        gradient_examples=args.diagnostic_gradient_examples,
        batch_size=args.diagnostic_batch,
        max_length=args.max_length,
        exp_lambda=args.exp_lambda,
        surprisal_threshold=args.surprisal_threshold,
        negative_scale=calibrated_negative_scale,
    )
    initial_row = {
        "step": 0,
        "effective_epoch": 0.0,
        "method": args.method,
        "gamma": initial_gamma,
        "negative_scale": calibrated_negative_scale or 0.0,
        "weight": 1.0,
        **initial_diagnostics,
        **initial_eval,
    }
    metrics_rows.append(initial_row)
    with diagnostics_path.open("a") as handle:
        handle.write(json.dumps({"step": 0, "method": args.method, **initial_diagnostics}) + "\n")
    best_value = float(initial_eval[args.selection_metric])
    checkpoint_records.append(save_local_model_checkpoint(
        model, tokenizer, best_dir, "best", 0
    ))

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
                near_token_weights, far_token_weights = controlled_negative_token_weights(
                    args.method,
                    near,
                    far,
                    args.exp_lambda,
                    args.surprisal_threshold,
                )
                if args.method in {"exp", "hybrid"}:
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
                    negative_budget = calibrated_negative_scale * (
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

                if args.method in {"controlled_negative", "dynamic_controlled_negative"}:
                    near_lp = weighted_sequence_logprob(near, near_token_weights)
                    far_lp = weighted_sequence_logprob(far, far_token_weights)
                else:
                    near_lp = near_seq_weights * near["seq_lp"]
                    far_lp = far_seq_weights * far["seq_lp"]
                negative_lp = (
                    args.near_mix * near_lp.mean()
                    + args.far_mix * far_lp.mean()
                )
                raw_loss = -(positive_lp - calibrated_negative_scale * gamma * negative_lp)
                if args.method == "entropy_bonus":
                    raw_loss = raw_loss - args.entropy_coef * pos["entropy"].mean()
                elif args.method == "target_entropy":
                    entropy_gap = F.relu(args.target_entropy - pos["entropy"].mean())
                    raw_loss = raw_loss + args.target_entropy_coef * entropy_gap.square()

                if args.method in {"controlled_negative", "dynamic_controlled_negative"}:
                    near_weight_value = float(
                        (near_token_weights.detach() * near["token_mask"]).sum()
                        / near["token_mask"].sum().clamp_min(1)
                    )
                    far_weight_value = float(
                        (far_token_weights.detach() * far["token_mask"]).sum()
                        / far["token_mask"].sum().clamp_min(1)
                    )
                else:
                    near_weight_value = float(near_seq_weights.detach().mean())
                    far_weight_value = float(far_seq_weights.detach().mean())
                mean_weight = (
                    args.near_mix * near_weight_value
                    + args.far_mix * far_weight_value
                )

            if not bool(torch.isfinite(raw_loss)):
                numerical_failure = f"nonfinite_loss_at_step_{update_step}"
                failure_detected_at_step = update_step
                last_finite_step = update_step - 1
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
            failure_detected_at_step = update_step
            last_finite_step = update_step - 1
            stop_reason = numerical_failure
            break
        if not optimizer_step_with_last_finite_guard(optimizer, trainable):
            numerical_failure = f"nonfinite_parameters_at_step_{update_step}"
            failure_detected_at_step = update_step
            last_finite_step = update_step - 1
            stop_reason = numerical_failure
            break
        scheduler.step()
        terminal_step = update_step
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
            checkpoint_diagnostics = dynamic_negative_diagnostics(
                model, tokenizer, diagnostic_rows, trainable,
                max_examples=args.diagnostic_examples,
                gradient_examples=args.diagnostic_gradient_examples,
                batch_size=args.diagnostic_batch,
                max_length=args.max_length,
                exp_lambda=args.exp_lambda,
                surprisal_threshold=args.surprisal_threshold,
                negative_scale=calibrated_negative_scale,
            )
            row = {
                "step": update_step,
                "effective_epoch": update_step / max(updates_per_effective_epoch, 1),
                "method": args.method,
                "gamma": last_gamma,
                "weight": last_weight,
                **checkpoint_diagnostics,
                **metrics,
            }
            with diagnostics_path.open("a") as handle:
                handle.write(json.dumps({
                    "step": update_step, "method": args.method, **checkpoint_diagnostics
                }) + "\n")
            metrics_rows.append(row)
            print("ARENA_EVAL", json.dumps(row))
            value = float(metrics[args.selection_metric])
            if value > best_value + args.early_stop_delta:
                best_value = value
                best_step = update_step
                stale_checks = 0
                checkpoint_records = [
                    record for record in checkpoint_records if record["kind"] != "best"
                ]
                checkpoint_records.append(save_local_model_checkpoint(
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

    if numerical_failure:
        checkpoint_records.append(save_local_model_checkpoint(
            model,
            tokenizer,
            last_finite_dir,
            "last_finite",
            (
                last_finite_step
                if last_finite_step is not None
                else (terminal_step if terminal_step is not None else 0)
            ),
        ))
        terminal_step = None
    else:
        if terminal_step is None:
            terminal_step = 0
        checkpoint_records.append(save_local_model_checkpoint(
            model, tokenizer, terminal_dir, "terminal", terminal_step
        ))

    with (out_dir / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_safe_row(row) for row in metrics_rows)
    manifest = {
        **serializable_namespace(args),
        "source_provenance": source_provenance(),
        "best_step": best_step,
        "best_value": best_value,
        "terminal_step": terminal_step,
        "failure_detected_at_step": failure_detected_at_step,
        "last_finite_step": last_finite_step,
        "stop_reason": stop_reason,
        "numerical_failure": numerical_failure,
        "offline_rows": len(train_rows),
        "effective_batch_size": effective_batch_size,
        "updates_per_effective_epoch": updates_per_effective_epoch,
        "maximum_effective_epochs": max_effective_epochs,
        "minimum_effective_epochs_before_early_stop": min_effective_epochs,
        "completed_effective_epochs": (
            (terminal_step if terminal_step is not None else (last_finite_step or 0))
            / max(updates_per_effective_epoch, 1)
        ),
        "dynamic_diagnostics_path": str(diagnostics_path),
        "negative_scale": calibrated_negative_scale,
        "negative_scale_source": (
            "fixed_calibration_split_rms_gradient_match"
            if args.negative_calibration_json and args.method != "positive_only"
            else ("explicit_override" if args.method != "positive_only" else "not_applicable")
        ),
        "global_matched_gamma": calibrated_global_gamma,
        "checkpoint_policy": "server-local adapters only; binaries must not enter Git/artifact packages",
        "checkpoints": checkpoint_records,
        "result_status": args.result_status,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (out_dir / "checkpoint_manifest.json").write_text(json.dumps({
        "local_only": True,
        "result_status": args.result_status,
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


def calibration_scales_from_rms(
    positive_rms: float, controlled_rms: float, uncontrolled_rms: float
) -> tuple[float, float]:
    """Return shared negative scale and equal-budget global gamma.

    The shared negative scale matches the unscaled uncontrolled negative-gradient
    RMS to the positive-gradient RMS at the common initialization. The global
    gamma then matches the controlled and uncontrolled negative-gradient RMS.
    No task metric or test example enters either calculation.
    """
    values = (positive_rms, controlled_rms, uncontrolled_rms)
    if any((not math.isfinite(value) or value <= 0) for value in values):
        raise ValueError("Calibration RMS norms must be finite and positive")
    return positive_rms / uncontrolled_rms, controlled_rms / uncontrolled_rms


def cmd_calibrate_global(args: argparse.Namespace) -> None:
    """Freeze shared negative scale and global-matched gamma before training."""
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
    positive_norms: list[float] = []
    controlled_norms: list[float] = []
    uncontrolled_norms: list[float] = []
    controlled_weights: list[float] = []

    for batch_index, packed in enumerate(loader):
        if batch_index >= args.calibration_batches:
            break
        positive_batch = move_to_device(packed["positive"], device)
        near_batch = move_to_device(packed["near"], device)
        far_batch = move_to_device(packed["far"], device)

        model.zero_grad(set_to_none=True)
        positive = completion_stats(model, positive_batch)
        (-positive["seq_lp"].mean()).backward()
        positive_norms.append(_current_gradient_norm(trainable))

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
    if not positive_norms or not controlled_norms or not uncontrolled_norms:
        raise RuntimeError("No calibration batches were processed")
    positive_rms = float(np.sqrt(np.mean(np.square(positive_norms))))
    controlled_rms = float(np.sqrt(np.mean(np.square(controlled_norms))))
    uncontrolled_rms = float(np.sqrt(np.mean(np.square(uncontrolled_norms))))
    try:
        negative_scale, gamma = calibration_scales_from_rms(
            positive_rms, controlled_rms, uncontrolled_rms
        )
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc
    result = {
        "version": VERSION,
        "protocol": "fixed_calibration_split_rms_positive_negative_and_global_budget_match",
        "reference_adapter": args.reference_adapter,
        "offline_data": args.offline_data,
        "seed": args.seed,
        "batches": len(controlled_norms),
        "batch_size": args.batch_size,
        "positive_gradient_norms": positive_norms,
        "controlled_gradient_norms": controlled_norms,
        "uncontrolled_gradient_norms": uncontrolled_norms,
        "positive_rms_gradient_norm": positive_rms,
        "controlled_rms_gradient_norm": controlled_rms,
        "uncontrolled_rms_gradient_norm": uncontrolled_rms,
        "negative_scale": negative_scale,
        "global_gamma": gamma,
        "mean_controlled_scalar_weight": float(np.mean(controlled_weights)),
        "frozen_before_method_training": True,
        "interpretation": (
            "negative_scale is shared by every negative-advantage method and matches "
            "the initial uncontrolled negative-gradient RMS to the positive-gradient RMS. "
            "global_matched additionally applies one fixed gamma equally to near and far. "
            "controlled_negative preserves the V4.2 static far-only taper, whereas "
            "dynamic_controlled_negative applies the same current-policy token taper "
            "to both initially-near and initially-far branches."
        ),
        "task_metrics_used_for_selection": False,
        "test_data_used": False,
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


@dataclass(frozen=True)
class StageTask:
    name: str
    argv: list[str]
    log_path: Path
    gpu_id: str


class StageExecutionError(RuntimeError):
    def __init__(self, stage: str, code: int, log_path: Path, command: Sequence[str]):
        self.stage = stage
        self.code = code
        self.log_path = log_path
        self.command = list(command)
        tail = ""
        try:
            tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-40:])
        except OSError:
            pass
        super().__init__(
            f"Stage {stage!r} failed with exit code {code}. See {log_path}."
            + (f"\nLast log lines:\n{tail}" if tail else "")
        )


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    temporary.replace(path)


def _record_decision(root: Path, name: str, decision: Any, evidence: Any = None) -> None:
    path = root / "automatic_decisions.json"
    payload = json.loads(path.read_text()) if path.exists() else {"version": VERSION, "decisions": []}
    payload["decisions"].append({
        "name": name,
        "decision": decision,
        "evidence": evidence,
        "timestamp_unix": time.time(),
    })
    _atomic_json(path, payload)


def _run_stage(
    argv: list[str],
    log_path: Path,
    *,
    gpu_id: str | None = None,
    stage_name: str | None = None,
    stream_output: bool = True,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(Path(__file__).resolve()), *argv]
    stage = stage_name or argv[0]
    env = os.environ.copy()
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    print(f"\n[RUN:{stage}] GPU={gpu_id if gpu_id is not None else 'inherited'}", " ".join(command), flush=True)
    if stream_output:
        with log_path.open("w") as log:
            proc = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, env=env,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                print(line, end="")
                log.write(line)
            code = proc.wait()
    else:
        with log_path.open("w") as log:
            proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, text=True, env=env)
            code = proc.wait()
        print(f"[DONE:{stage}] GPU={gpu_id} exit={code} log={log_path}", flush=True)
    if code != 0:
        raise StageExecutionError(stage, code, log_path, command)


def _run_stage_group(tasks: Sequence[StageTask]) -> None:
    """Run one FIFO queue per GPU; different GPU queues execute concurrently."""
    if not tasks:
        return
    queues: dict[str, list[StageTask]] = defaultdict(list)
    for task in tasks:
        queues[task.gpu_id].append(task)

    def worker(gpu_id: str, queue: Sequence[StageTask]) -> None:
        for task in queue:
            _run_stage(
                task.argv,
                task.log_path,
                gpu_id=gpu_id,
                stage_name=task.name,
                stream_output=False,
            )

    failures: list[BaseException] = []
    with ThreadPoolExecutor(max_workers=len(queues)) as executor:
        futures = {
            executor.submit(worker, gpu_id, queue): gpu_id
            for gpu_id, queue in queues.items()
        }
        for future in as_completed(futures):
            try:
                future.result()
            except BaseException as exc:  # preserve the first exact stage failure
                failures.append(exc)
    if failures:
        raise failures[0]


def _evaluation_argv(
    model_flags: Sequence[str],
    data: Path,
    structure_reference: Path,
    output_json: Path,
    batch_size: int,
    pass_k: int,
    seed: int,
    adapter: Path | None = None,
) -> list[str]:
    command = ["evaluate", *model_flags]
    if adapter is not None:
        command.extend(["--adapter", str(adapter)])
    command.extend([
        "--data", str(data),
        "--structure_reference_data", str(structure_reference),
        "--batch_size", str(batch_size),
        "--pass_k", str(pass_k),
        "--output_json", str(output_json),
        "--seed", str(seed),
    ])
    return command


def cmd_run(args: argparse.Namespace) -> None:
    """Run the preregistered V4.3 dynamic-remoteness pilot."""
    gpu_ids = resolve_gpu_ids(args.gpus, args.gpu)
    primary_gpu = gpu_ids[0]
    primary_parent_index = _parent_gpu_index(primary_gpu)
    plan = resolve_execution_plan(
        args.model_path,
        args.preset,
        args.memory_mode,
        gpu_index=primary_parent_index,
        gpu_visible=primary_gpu,
    )
    plan["orchestration"] = {
        "gpu_ids": gpu_ids,
        "gpu_count": len(gpu_ids),
        "offline_builder": "single_rng_stream_balanced_by_oracle_pattern",
        "full_ft_reference_diagnostic": "isolated_from_lora_method_ranking",
        "mechanism_and_calibration": "parallel_when_two_or_more_gpus",
        "method_training": "one_method_per_gpu_fifo_queue",
        "checkpoint_evaluation": "all_gpu_fifo_queue",
    }
    root = Path(args.work_dir).resolve()
    ensure_checkpoint_output_is_local_or_ignored(root)
    root.mkdir(parents=True, exist_ok=True)
    _atomic_json(root / "pipeline_status.json", {
        "version": VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": "running",
        "started_unix": time.time(),
    })

    if (
        (plan["memory_mode"] != "bf16" or plan["load_in_4bit"] or plan["dtype"] != "bf16")
        and not args.allow_non_bf16_smoke
    ):
        raise RuntimeError(
            "The registered pilot requires the shared BF16 LoRA parameterization. "
            "Use --allow_non_bf16_smoke only for an explicitly nonformal engineering smoke."
        )
    params_b = plan["model_metadata"].get("estimated_params_b")
    registered_model = bool(
        plan["preset"] in {"0.5b", "small"}
        and params_b is not None
        and params_b <= 1.0
        and plan["model_metadata"].get("registered_instruct_identity")
    )
    registered_parameterization = bool(
        plan["memory_mode"] == "bf16"
        and not plan["load_in_4bit"]
        and plan["dtype"] == "bf16"
    )
    if (not registered_model or not registered_parameterization) and not args.allow_non_bf16_smoke:
        raise RuntimeError(
            f"{EXPERIMENT_ID} requires a local Qwen2.5 0.5B Instruct checkpoint with "
            "a chat template and BF16 LoRA. Model identity is verified from local metadata, "
            "not only from the directory name."
        )
    run_status = "pilot" if registered_model and registered_parameterization else "engineering_smoke"

    data_dir = root / "data"
    logs = root / "logs"
    shared_adapter_dir = root / "reference_adapter"
    sft_dir = root / "sft_adapter"
    full_ft_dir = root / "full_ft_reference_diagnostic"
    offline_file = data_dir / "offline_6000.jsonl"
    split_manifest_file = data_dir / "split_manifest.json"
    methods_dir = root / "methods"
    calibration_json = root / "negative_budget_calibration.json"

    methods = [method.strip() for method in args.methods.split(",") if method.strip()]
    allowed = {
        "positive_only", "controlled_negative", "dynamic_controlled_negative",
        "uncontrolled_negative", "global_matched", "uncontrolled", "global",
        "exp", "entropy_bonus", "target_entropy", "sbrc", "hybrid",
    }
    unknown = set(methods) - allowed
    if unknown:
        raise ValueError(f"Unknown methods: {sorted(unknown)}")
    required_pilot = {
        "positive_only",
        "controlled_negative",
        "dynamic_controlled_negative",
        "uncontrolled_negative",
    }
    if run_status == "pilot" and set(methods) != required_pilot:
        raise RuntimeError(
            "The registered V4.3 pilot is frozen to positive-only, static control, "
            "dynamic control, and uncontrolled negative updates."
        )

    presets = {
        "0.5b": dict(
            train=6000, val=500, test=1000, offline=6000, rollouts=12,
            sft_epochs=6, sft_min_epochs=3, sft_patience=2, sft_accum=16,
            full_ft_lr=2e-5, method_epochs=6, method_min_epochs=2,
            method_accum=8, patience=2, eval_examples=500, pass_k=8,
            probe_examples=32, dynamics_examples=16, calibration_batches=16,
            diagnostic_examples=32, diagnostic_gradient_examples=8, diagnostic_batch=8,
        ),
        "small": dict(
            train=6000, val=500, test=1000, offline=6000, rollouts=12,
            sft_epochs=6, sft_min_epochs=3, sft_patience=2, sft_accum=16,
            full_ft_lr=2e-5, method_epochs=6, method_min_epochs=2,
            method_accum=8, patience=2, eval_examples=500, pass_k=8,
            probe_examples=32, dynamics_examples=16, calibration_batches=16,
            diagnostic_examples=32, diagnostic_gradient_examples=8, diagnostic_batch=8,
        ),
        # Larger presets remain engineering-smoke conveniences.  They do not acquire
        # formal status without a separate registry entry and frozen protocol.
        "3b": dict(
            train=20000, val=1000, test=2000, offline=6000, rollouts=12,
            sft_epochs=4, sft_min_epochs=2, sft_patience=2, sft_accum=32,
            full_ft_lr=1e-5, method_epochs=6, method_min_epochs=2,
            method_accum=16, patience=2, eval_examples=1000, pass_k=8,
            probe_examples=32, dynamics_examples=16, calibration_batches=16,
            diagnostic_examples=32, diagnostic_gradient_examples=8, diagnostic_batch=8,
        ),
        "7b": dict(
            train=20000, val=1000, test=2000, offline=6000, rollouts=12,
            sft_epochs=3, sft_min_epochs=2, sft_patience=2, sft_accum=32,
            full_ft_lr=1e-5, method_epochs=6, method_min_epochs=2,
            method_accum=16, patience=2, eval_examples=1000, pass_k=8,
            probe_examples=24, dynamics_examples=12, calibration_batches=12,
            diagnostic_examples=24, diagnostic_gradient_examples=6, diagnostic_batch=6,
        ),
    }
    preset = presets[plan["preset"]]

    run_spec = {
        "version": VERSION,
        "source_provenance": source_provenance(),
        "experiment_id": EXPERIMENT_ID,
        "model_path": str(Path(args.model_path).resolve()),
        "preset": plan["preset"],
        "memory_mode": plan["memory_mode"],
        "main_parameterization": "shared_bf16_lora" if run_status == "pilot" else "nonformal_smoke",
        "full_ft_role": "isolated_reference_capacity_diagnostic_only",
        "methods": methods,
        "seed": args.seed,
        "data_protocol": {
            "generated_train_rows": preset["train"],
            "balanced_matched_rows": preset["offline"],
            "nested_subsets": [1500, 3000, 6000],
            "balance_key": "oracle_structure",
        },
        "gates": {
            "base": {"greedy_success": args.min_base_success, "valid_rate": args.min_base_valid},
            "mechanism": {
                "greedy_success": args.min_mechanism_success,
                "valid_rate": args.min_mechanism_valid,
            },
            "method_effect": {
                "greedy_success": args.min_sft_success,
                "valid_rate": args.min_sft_valid,
            },
        },
        "pair_resample_rounds": args.pair_resample_rounds,
        "result_status": run_status,
        "gpu_ids": gpu_ids,
        "checkpoint_policy": "server-local only",
        "negative_scale_protocol": (
            "fixed_training_calibration_split; match initial uncontrolled negative RMS "
            "gradient to positive RMS gradient; freeze across negative methods"
        ),
        "method_duration_protocol": {
            "max_effective_epochs": preset["method_epochs"],
            "min_effective_epochs": preset["method_min_epochs"],
            "validation_every_effective_epoch": True,
        },
    }
    run_spec["fingerprint"] = stable_fingerprint(run_spec)
    run_config_path = root / "run_config.json"
    if run_config_path.exists() and not args.force:
        previous = json.loads(run_config_path.read_text())
        if previous.get("fingerprint") != run_spec["fingerprint"]:
            raise RuntimeError("Existing work_dir has a different configuration; use a new directory.")
    _atomic_json(run_config_path, run_spec)
    _atomic_json(root / "execution_plan.json", plan)
    _record_decision(root, "gpu_selection", gpu_ids, plan["orchestration"])
    _record_decision(root, "model_identity_gate", run_status, plan["model_metadata"])
    _record_decision(
        root,
        "offline_builder_parallelism",
        "disabled",
        "balanced builder preserves one deterministic registered RNG stream",
    )
    _record_decision(
        root,
        "full_ft_role",
        "isolated_reference_capacity_diagnostic",
        "never substitutes for the LoRA reference in the focused four-method pilot",
    )
    print("EXECUTION_PLAN", json.dumps(plan, indent=2))

    train_file = data_dir / "train.jsonl"
    val_file = data_dir / "val.jsonl"
    test_file = data_dir / "test.jsonl"
    model_flags = ["--model_path", args.model_path, "--dtype", plan["dtype"]]
    if plan["load_in_4bit"]:
        model_flags.append("--load_in_4bit")

    if not args.skip_preflight:
        _run_stage(
            ["preflight", *model_flags, "--seed", str(args.seed)],
            logs / "00_preflight.log",
            gpu_id=primary_gpu,
        )
    if args.force or not (train_file.exists() and val_file.exists() and test_file.exists()):
        _run_stage([
            "generate", "--train", str(preset["train"]), "--val", str(preset["val"]),
            "--test", str(preset["test"]), "--train_out", str(train_file),
            "--val_out", str(val_file), "--test_out", str(test_file),
            "--manifest_out", str(split_manifest_file), "--seed", str(args.seed),
        ], logs / "01_generate_pattern_family_split.log", gpu_id=primary_gpu)

    base_val_json = root / "base_val_metrics.json"
    _run_stage(
        _evaluation_argv(
            model_flags, val_file, train_file, base_val_json,
            plan["eval_batch"], preset["pass_k"], args.seed + 5000,
        ),
        logs / "02_base_eval.log",
        gpu_id=primary_gpu,
    )
    base_val = json.loads(base_val_json.read_text())
    base_passes = bool(
        base_val["greedy_success"] >= args.min_base_success
        and base_val["valid_rate"] >= args.min_base_valid
    )
    _record_decision(root, "base_gate", "skip_sft" if base_passes else "run_extended_sft", {
        "greedy_success": base_val["greedy_success"],
        "valid_rate": base_val["valid_rate"],
        "thresholds": {
            "greedy_success": args.min_base_success,
            "valid_rate": args.min_base_valid,
        },
    })

    if base_passes:
        initialization_mode = "base_first_no_sft"
        if args.force and shared_adapter_dir.exists():
            shutil.rmtree(shared_adapter_dir)
        if args.force or not (shared_adapter_dir / "adapter_config.json").exists():
            _run_stage(
                ["init_adapter", *model_flags, "--output_dir", str(shared_adapter_dir),
                 "--seed", str(args.seed)],
                logs / "03_init_shared_adapter.log",
                gpu_id=primary_gpu,
            )
        reference_dir = shared_adapter_dir
    else:
        if args.no_sft_fallback:
            raise RuntimeError("Base gate failed and SFT fallback was explicitly disabled.")
        initialization_mode = "extended_lora_sft_fallback"
        if args.force and sft_dir.exists():
            shutil.rmtree(sft_dir)
        if args.force or not (sft_dir / "best_adapter" / "adapter_config.json").exists():
            _run_stage([
                "sft", *model_flags, "--parameterization", "lora",
                "--train_data", str(train_file), "--val_data", str(val_file),
                "--output_dir", str(sft_dir), "--epochs", str(preset["sft_epochs"]),
                "--min_epochs", str(preset["sft_min_epochs"]),
                "--early_stop_patience", str(preset["sft_patience"]),
                "--micro_batch", str(plan["micro_batch"]),
                "--grad_accum", str(preset["sft_accum"]),
                "--eval_batch", str(plan["eval_batch"]),
                "--eval_examples", str(preset["eval_examples"]),
                "--pass_k", str(preset["pass_k"]),
                "--eval_seed", str(args.seed + 5000),
                "--seed", str(args.seed), "--result_status", run_status,
            ], logs / "03_lora_sft_fallback.log", gpu_id=primary_gpu)
        reference_dir = sft_dir / "best_adapter"

    reference_val_json = root / "reference_val_metrics.json"
    _run_stage(
        _evaluation_argv(
            model_flags, val_file, train_file, reference_val_json,
            plan["eval_batch"], preset["pass_k"], args.seed + 5000, reference_dir,
        ),
        logs / "04_reference_eval.log",
        gpu_id=primary_gpu,
    )
    reference_val = json.loads(reference_val_json.read_text())
    mechanism_gate_passed = bool(
        reference_val["greedy_success"] >= args.min_mechanism_success
        and reference_val["valid_rate"] >= args.min_mechanism_valid
    )
    method_ranking_gate_passed = bool(
        reference_val["greedy_success"] >= args.min_sft_success
        and reference_val["valid_rate"] >= args.min_sft_valid
    )
    focused_dynamic_pilot_gate_passed = mechanism_gate_passed
    _record_decision(root, "reference_mechanism_gate", mechanism_gate_passed, {
        "metrics": reference_val,
        "thresholds": {
            "greedy_success": args.min_mechanism_success,
            "valid_rate": args.min_mechanism_valid,
        },
    })
    _record_decision(root, "reference_method_effect_gate", method_ranking_gate_passed, {
        "metrics": reference_val,
        "thresholds": {
            "greedy_success": args.min_sft_success,
            "valid_rate": args.min_sft_valid,
        },
        "failure_action": (
            "run the registered focused dynamic-control pilot but prohibit formal method ranking"
        ),
    })
    if not mechanism_gate_passed:
        raise RuntimeError(
            "The LoRA reference failed the mechanism-pilot capability/validity gate. "
            "The run stops rather than interpreting a floor-effect model."
        )

    # The balanced offline corpus and isolated full-FT reference diagnostic are
    # independent once the LoRA reference has been frozen, so they may safely run
    # in parallel on different GPUs.  The offline builder itself remains single-GPU.
    preparation_tasks: list[StageTask] = []
    if args.force or not offline_file.exists():
        preparation_tasks.append(StageTask(
            "build_balanced_offline_6000",
            [
                "build_offline", *model_flags, "--reference_adapter", str(reference_dir),
                "--input_data", str(train_file), "--split_manifest", str(split_manifest_file),
                "--output_data", str(offline_file), "--max_examples", str(preset["offline"]),
                "--balance_by_oracle_pattern", "--nested_sizes", "1500,3000,6000",
                "--nested_output_dir", str(data_dir),
                "--rollouts", str(preset["rollouts"]),
                "--batch_size", str(plan["rollout_batch"]),
                "--score_batch_size", str(plan["score_batch"]),
                "--pair_resample_rounds", str(args.pair_resample_rounds),
                "--min_negative_candidates", "8",
                "--synthetic_rescue_candidates", "64",
                "--seed", str(args.seed + 11),
            ],
            logs / "05_build_balanced_matched_offline.log",
            primary_gpu,
        ))
    full_ft_best = full_ft_dir / "best_model"
    if args.force and full_ft_dir.exists():
        shutil.rmtree(full_ft_dir)
    if args.force or not (full_ft_best / "config.json").exists():
        full_ft_gpu = gpu_ids[1] if len(gpu_ids) > 1 else primary_gpu
        preparation_tasks.append(StageTask(
            "full_ft_reference_diagnostic",
            [
                "sft", "--model_path", args.model_path, "--dtype", plan["dtype"],
                "--parameterization", "full", "--train_data", str(train_file),
                "--val_data", str(val_file), "--output_dir", str(full_ft_dir),
                "--epochs", str(preset["sft_epochs"]),
                "--min_epochs", str(preset["sft_min_epochs"]),
                "--early_stop_patience", str(preset["sft_patience"]),
                "--micro_batch", str(plan["micro_batch"]),
                "--grad_accum", str(preset["sft_accum"]),
                "--lr", str(preset["full_ft_lr"]),
                "--eval_batch", str(plan["eval_batch"]),
                "--eval_examples", str(preset["eval_examples"]),
                "--pass_k", str(preset["pass_k"]),
                "--eval_seed", str(args.seed + 5000),
                "--seed", str(args.seed), "--result_status", run_status,
            ],
            logs / "05b_full_ft_reference_diagnostic.log",
            full_ft_gpu,
        ))
    _run_stage_group(preparation_tasks)
    if not offline_file.exists():
        raise RuntimeError("Balanced offline builder did not produce offline_6000.jsonl")
    offline_rows = len(read_jsonl(offline_file))
    if offline_rows != preset["offline"]:
        raise RuntimeError(
            f"Expected {preset['offline']} balanced matched rows, found {offline_rows}"
        )

    full_ft_val_json = full_ft_dir / "val_metrics_best.json"
    if not (full_ft_best / "config.json").exists():
        raise RuntimeError("Full-FT reference diagnostic did not produce best_model/config.json")
    full_ft_flags = ["--model_path", str(full_ft_best), "--dtype", plan["dtype"]]
    _run_stage(
        _evaluation_argv(
            full_ft_flags, val_file, train_file, full_ft_val_json,
            plan["eval_batch"], preset["pass_k"], args.seed + 5000,
        ),
        logs / "05c_full_ft_reference_val.log",
        gpu_id=gpu_ids[1] if len(gpu_ids) > 1 else primary_gpu,
    )
    full_ft_val = json.loads(full_ft_val_json.read_text())

    mechanism_json = root / "mechanism_probe.json"
    mechanism_csv = root / "mechanism_probe_pairs.csv"
    probe_tasks = [StageTask(
        "mechanism_probe",
        [
            "mechanism_probe", *model_flags, "--reference_adapter", str(reference_dir),
            "--offline_data", str(offline_file), "--output_json", str(mechanism_json),
            "--output_csv", str(mechanism_csv),
            "--max_examples", str(preset["probe_examples"]),
            "--dynamics_examples", str(preset["dynamics_examples"]),
            "--min_matched_pairs", str(args.min_matched_pairs),
            "--seed", str(args.seed + 50),
        ],
        logs / "06_mechanism_probe.log",
        gpu_ids[0],
    )]
    if focused_dynamic_pilot_gate_passed and any(method != "positive_only" for method in methods):
        calibration_gpu = gpu_ids[1] if len(gpu_ids) > 1 else gpu_ids[0]
        if args.force or not calibration_json.exists():
            probe_tasks.append(StageTask(
                "negative_budget_calibration",
                [
                    "calibrate_global", *model_flags, "--reference_adapter", str(reference_dir),
                    "--offline_data", str(offline_file), "--output_json", str(calibration_json),
                    "--batch_size", str(plan["micro_batch"]),
                    "--calibration_batches", str(preset["calibration_batches"]),
                    "--seed", str(args.seed + 75),
                ],
                logs / "06b_negative_budget_calibration.log",
                calibration_gpu,
            ))
    _run_stage_group(probe_tasks)
    _record_decision(
        root,
        "mechanism_calibration_schedule",
        "parallel" if len({task.gpu_id for task in probe_tasks}) > 1 else "sequential",
        [task.name for task in probe_tasks],
    )

    methods_to_run = methods if focused_dynamic_pilot_gate_passed else []
    _record_decision(
        root,
        "focused_dynamic_control_pilot",
        "run" if focused_dynamic_pilot_gate_passed else "skipped_by_capability_gate",
        {
            "formal_method_ranking_eligible": method_ranking_gate_passed,
            "methods": methods_to_run,
        },
    )
    if not method_ranking_gate_passed:
        _record_decision(
            root,
            "formal_method_ranking",
            "prohibited_by_floor_effect_gate",
            "focused single-seed dynamic-control results remain pilot evidence only",
        )
    effective_batch = plan["micro_batch"] * preset["method_accum"]
    updates_per_epoch = math.ceil(offline_rows / effective_batch)
    method_steps = updates_per_epoch * preset["method_epochs"]
    method_min_steps = updates_per_epoch * preset["method_min_epochs"]
    eval_every = updates_per_epoch

    shared_method_seed = args.seed + 100
    concurrent_tasks: list[StageTask] = []
    method_gpu_pool = gpu_ids[: min(4, len(gpu_ids))]
    for index, method in enumerate(methods_to_run):
        output = methods_dir / method
        if args.force and output.exists():
            shutil.rmtree(output)
        complete_checkpoint = (
            (output / "terminal_adapter" / "adapter_config.json").exists()
            or (output / "last_finite_adapter" / "adapter_config.json").exists()
        )
        if args.force or not complete_checkpoint:
            command = [
                "train_method", *model_flags, "--reference_adapter", str(reference_dir),
                "--offline_data", str(offline_file), "--val_data", str(val_file),
                "--structure_reference_data", str(train_file), "--output_dir", str(output),
                "--method", method, "--steps", str(method_steps),
                "--micro_batch", str(plan["micro_batch"]),
                "--grad_accum", str(preset["method_accum"]),
                "--min_steps", str(method_min_steps),
                "--early_stop_patience", str(preset["patience"]),
                "--eval_examples", str(preset["eval_examples"]),
                "--eval_batch", str(plan["eval_batch"]),
                "--eval_every", str(eval_every), "--pass_k", str(preset["pass_k"]),
                "--diagnostic_examples", str(preset["diagnostic_examples"]),
                "--diagnostic_gradient_examples", str(preset["diagnostic_gradient_examples"]),
                "--diagnostic_batch", str(preset["diagnostic_batch"]),
                "--eval_seed", str(args.seed + 6000), "--seed", str(shared_method_seed),
                "--result_status", run_status,
            ]
            if method != "positive_only":
                command.extend(["--negative_calibration_json", str(calibration_json)])
            concurrent_tasks.append(StageTask(
                f"train_{method}", command, logs / f"07_train_{method}.log",
                method_gpu_pool[index % len(method_gpu_pool)],
            ))

    base_test_json = root / "base_test_metrics.json"
    reference_test_json = root / "reference_test_metrics.json"
    full_ft_test_json = full_ft_dir / "test_metrics_best.json"
    spare = gpu_ids[4:] if len(gpu_ids) > 4 else gpu_ids
    concurrent_tasks.extend([
        StageTask(
            "test_raw_base",
            _evaluation_argv(
                model_flags, test_file, train_file, base_test_json,
                plan["eval_batch"], preset["pass_k"], args.seed + 7000,
            ),
            logs / "08_test_raw_base.log",
            spare[0],
        ),
        StageTask(
            "test_lora_reference",
            _evaluation_argv(
                model_flags, test_file, train_file, reference_test_json,
                plan["eval_batch"], preset["pass_k"], args.seed + 7000, reference_dir,
            ),
            logs / "08b_test_lora_reference.log",
            spare[1 % len(spare)],
        ),
        StageTask(
            "test_full_ft_reference_diagnostic",
            _evaluation_argv(
                full_ft_flags, test_file, train_file, full_ft_test_json,
                plan["eval_batch"], preset["pass_k"], args.seed + 7000,
            ),
            logs / "08c_test_full_ft_reference.log",
            spare[2 % len(spare)],
        ),
    ])
    _run_stage_group(concurrent_tasks)
    _record_decision(
        root,
        "method_training_schedule",
        {method: method_gpu_pool[index % len(method_gpu_pool)]
         for index, method in enumerate(methods_to_run)},
        {
            "shared_initialization_data_seed": True,
            "updates_per_effective_epoch": updates_per_epoch,
            "max_steps": method_steps,
            "min_steps": method_min_steps,
            "eval_every": eval_every,
        },
    )

    summary_rows: list[dict[str, Any]] = [
        {"method": "raw_base_no_training", **json.loads(base_test_json.read_text())},
        {
            "method": "shared_lora_reference",
            "initialization_mode": initialization_mode,
            "checkpoint_kind": "step_0",
            "eligible_for_method_ranking": method_ranking_gate_passed,
            **json.loads(reference_test_json.read_text()),
        },
        {
            "method": "full_ft_reference_diagnostic",
            "checkpoint_kind": "best",
            "eligible_for_method_ranking": False,
            "diagnostic_role": "parameterization_capacity_check_only",
            **json.loads(full_ft_test_json.read_text()),
        },
    ]
    eval_tasks: list[StageTask] = []
    eval_records: list[tuple[str, Path, dict[str, Any]]] = []
    eval_index = 0
    for method in methods_to_run:
        output = methods_dir / method
        manifest = json.loads((output / "manifest.json").read_text())
        for checkpoint_kind in ("best", "terminal", "last_finite"):
            adapter = output / f"{checkpoint_kind}_adapter"
            if not (adapter / "adapter_config.json").exists():
                continue
            result_json = output / f"test_metrics_{checkpoint_kind}.json"
            extra = {
                "checkpoint_kind": checkpoint_kind,
                "best_step": manifest.get("best_step"),
                "terminal_step": manifest.get("terminal_step"),
                "last_finite_step": manifest.get("last_finite_step"),
                "failure_detected_at_step": manifest.get("failure_detected_at_step"),
                "best_val": manifest.get("best_value"),
                "stop_reason": manifest.get("stop_reason"),
                "numerical_failure": manifest.get("numerical_failure"),
                "result_status": manifest.get("result_status"),
                "negative_scale": manifest.get("negative_scale"),
                "global_matched_gamma": manifest.get("global_matched_gamma"),
                "updates_per_effective_epoch": manifest.get("updates_per_effective_epoch"),
                "completed_effective_epochs": manifest.get("completed_effective_epochs"),
            }
            if args.force or not result_json.exists():
                gpu = gpu_ids[eval_index % len(gpu_ids)]
                eval_tasks.append(StageTask(
                    f"test_{method}_{checkpoint_kind}",
                    _evaluation_argv(
                        model_flags, test_file, train_file, result_json,
                        plan["eval_batch"], preset["pass_k"], args.seed + 7000, adapter,
                    ),
                    logs / f"09_test_{method}_{checkpoint_kind}.log",
                    gpu,
                ))
                eval_index += 1
            eval_records.append((method, result_json, extra))
    _run_stage_group(eval_tasks)
    _record_decision(
        root,
        "checkpoint_evaluation_schedule",
        "all_visible_gpu_fifo",
        {"jobs": len(eval_records), "gpus": gpu_ids},
    )
    for method, result_json, extra in eval_records:
        summary_rows.append({"method": method, **extra, **json.loads(result_json.read_text())})

    summary_path = root / "arena_summary.csv"
    fields = sorted({key for row in summary_rows for key in row})
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(csv_safe_row(row) for row in summary_rows)

    method_audits: dict[str, Any] = {}
    for method in methods_to_run:
        manifest = json.loads((methods_dir / method / "manifest.json").read_text())
        terminal_rows = [
            row for row in summary_rows
            if row["method"] == method
            and row.get("checkpoint_kind") in {"terminal", "last_finite"}
        ]
        method_audits[method] = {
            "task_performance": terminal_rows[0] if terminal_rows else None,
            "support_structure": ({
                "heldout_pattern_family_coverage": terminal_rows[0].get("heldout_pattern_family_coverage"),
                "heldout_pattern_family_precision_micro": terminal_rows[0].get("heldout_pattern_family_precision_micro"),
                "heldout_pattern_family_precision_macro": terminal_rows[0].get("heldout_pattern_family_precision_macro"),
                "valid_rate": terminal_rows[0].get("valid_rate"),
            } if terminal_rows else None),
            "numerical": {
                "numerical_failure": manifest.get("numerical_failure"),
                "failure_detected_at_step": manifest.get("failure_detected_at_step"),
                "last_finite_step": manifest.get("last_finite_step"),
                "stop_reason": manifest.get("stop_reason"),
            },
            "dynamic_diagnostics": manifest.get("dynamic_diagnostics_path"),
        }
    base_commit = run_spec["source_provenance"].get("git_commit")
    terminal_audit = {
        "version": VERSION,
        "experiment_id": EXPERIMENT_ID,
        "base_commit": base_commit,
        "task_performance_support_and_numerical_events_reported_separately": True,
        "reference_gates": {
            "mechanism_gate_passed": mechanism_gate_passed,
            "method_effect_gate_passed": method_ranking_gate_passed,
            "focused_dynamic_pilot_gate_passed": focused_dynamic_pilot_gate_passed,
            "lora_reference_validation": reference_val,
        },
        "full_ft_reference_diagnostic": {
            "eligible_for_method_ranking": False,
            "validation": full_ft_val,
            "test": json.loads(full_ft_test_json.read_text()),
        },
        "method_comparison": {
            "status": (
                "completed_focused_pilot"
                if focused_dynamic_pilot_gate_passed
                else "skipped_by_capability_gate"
            ),
            "formal_ranking_eligible": method_ranking_gate_passed,
            "methods_requested": methods,
            "methods_run": methods_to_run,
        },
        "methods": method_audits,
    }
    _atomic_json(root / "terminal_audit.json", terminal_audit)
    complete = {
        "version": VERSION,
        "experiment_id": EXPERIMENT_ID,
        "base_commit": base_commit,
        "source_provenance": source_provenance(),
        "plan": plan,
        "initialization_mode": initialization_mode,
        "base_validation": base_val,
        "reference_validation": reference_val,
        "reference_gates": terminal_audit["reference_gates"],
        "full_ft_reference_diagnostic": terminal_audit["full_ft_reference_diagnostic"],
        "offline_dataset": {
            "path": str(offline_file),
            "rows": offline_rows,
            "nested_subsets": [1500, 3000, 6000],
            "balanced_by_oracle_pattern": True,
        },
        "mechanism_probe": json.loads(mechanism_json.read_text()),
        "negative_budget_calibration": (
            json.loads(calibration_json.read_text()) if calibration_json.exists() else None
        ),
        "summary": summary_rows,
        "result_status": run_status,
        "terminal_audit_present": all(
            method_audits[method]["task_performance"] is not None
            for method in methods_to_run
        ),
        "full_finetune_confirmation": (
            "isolated reference-capacity diagnostic only; not a full-FT method confirmation"
        ),
        "note": "A completed single-seed run remains a pilot, never a formal multi-seed result.",
    }
    _atomic_json(root / "run_complete.json", complete)
    _atomic_json(root / "RUN_COMPLETE.json", complete)
    _atomic_json(root / "pipeline_status.json", {
        "version": VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": "terminal_audited",
        "completed_unix": time.time(),
        "summary": str(summary_path),
        "method_comparison": terminal_audit["method_comparison"]["status"],
    })
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
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--min_epochs", type=int, default=3)
    ap.add_argument("--early_stop_patience", type=int, default=2)
    ap.add_argument("--parameterization", choices=["lora", "full"], default="lora")
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
    ap.add_argument(
        "--result_status",
        choices=["pilot", "engineering_smoke", "standalone_unclassified"],
        default="standalone_unclassified",
    )
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
    ap.add_argument("--pair_resample_rounds", type=int, default=8)
    ap.add_argument("--min_negative_candidates", type=int, default=8)
    ap.add_argument("--synthetic_rescue_candidates", type=int, default=64)
    ap.add_argument("--score_batch_size", type=int, default=16)
    ap.add_argument("--max_examples", type=int, default=6000, help="0 means all")
    ap.add_argument("--balance_by_oracle_pattern", action="store_true")
    ap.add_argument("--nested_sizes", default="1500,3000,6000")
    ap.add_argument("--nested_output_dir", default=None)
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
            "dynamic_controlled_negative",
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
    ap.add_argument(
        "--negative_scale", type=float, default=None,
        help=(
            "Explicit shared negative scale for a separately registered run. "
            "The V4.3 focused pilot normally reads the automatically calibrated value."
        ),
    )
    ap.add_argument("--near_mix", type=float, default=0.5)
    ap.add_argument("--far_mix", type=float, default=0.5)
    ap.add_argument("--global_gamma", type=float, default=0.55)
    ap.add_argument("--negative_calibration_json", default=None)
    ap.add_argument("--exp_lambda", type=float, default=0.7)
    ap.add_argument("--surprisal_threshold", type=float, default=2.0)
    ap.add_argument("--entropy_coef", type=float, default=0.02)
    ap.add_argument("--target_entropy", type=float, default=1.8)
    ap.add_argument("--target_entropy_coef", type=float, default=0.05)
    ap.add_argument("--sbrc_kappa", type=float, default=0.92)
    ap.add_argument("--entropy_floor", type=float, default=1.0)
    ap.add_argument("--eval_every", type=int, default=100)
    ap.add_argument("--eval_seed", type=int, default=6000)
    ap.add_argument("--diagnostic_examples", type=int, default=32)
    ap.add_argument("--diagnostic_gradient_examples", type=int, default=8)
    ap.add_argument("--diagnostic_batch", type=int, default=8)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--result_status",
        choices=["pilot", "engineering_smoke", "standalone_unclassified"],
        default="standalone_unclassified",
    )
    ap.set_defaults(func=cmd_train_method)

    ap = sub.add_parser(
        "calibrate_global",
        help="Calibrate shared negative scale and fixed global-matched budget",
    )
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
    ap.add_argument("--gpus", default="auto", help="Visible GPU ids, comma-separated, or auto (default)")
    ap.add_argument("--gpu", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--preset", choices=["auto", "0.5b", "small", "3b", "7b"], default="auto")
    ap.add_argument("--memory_mode", choices=["auto", "bf16", "qlora"], default="bf16")
    ap.add_argument(
        "--methods",
        default=(
            "positive_only,controlled_negative,dynamic_controlled_negative,"
            "uncontrolled_negative"
        ),
    )
    ap.add_argument("--min_base_success", type=float, default=0.15)
    ap.add_argument("--min_base_valid", type=float, default=0.80)
    ap.add_argument("--min_sft_success", type=float, default=0.15)
    ap.add_argument("--min_sft_valid", type=float, default=0.95)
    ap.add_argument("--min_mechanism_success", type=float, default=0.08)
    ap.add_argument("--min_mechanism_valid", type=float, default=0.95)
    ap.add_argument("--min_matched_pairs", type=int, default=16)
    ap.add_argument("--pair_resample_rounds", type=int, default=8)
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
    try:
        args.func(args)
    except BaseException as exc:
        work_dir = getattr(args, "work_dir", None)
        if work_dir:
            root = Path(work_dir).resolve()
            root.mkdir(parents=True, exist_ok=True)
            failure = {
                "version": VERSION,
                "experiment_id": EXPERIMENT_ID,
                "status": "failed",
                "exception_type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
                "timestamp_unix": time.time(),
                "rerun_instruction": "Rerun the same one-click command with a new empty work_dir after correcting the reported preflight/stage error.",
            }
            if isinstance(exc, StageExecutionError):
                failure.update({
                    "failed_stage": exc.stage,
                    "exit_code": exc.code,
                    "log_path": str(exc.log_path),
                    "command": exc.command,
                })
            _atomic_json(root / "RUN_FAILED.json", failure)
            _atomic_json(root / "pipeline_status.json", failure)
        raise


if __name__ == "__main__":
    main()
