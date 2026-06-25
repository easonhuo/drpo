#!/usr/bin/env python3
"""Countdown base-first external-validity arena for local Qwen Instruct models (v4).

One-command run
---------------
python3 src/drpo/countdown_qwen_arena_onefile.py run \
  --model_path /ABS/PATH/TO/QWEN-0.5B-INSTRUCT \
  --work_dir /ABS/PATH/TO/COUNTDOWN_RUN \
  --gpu 0 --preset auto --memory_mode auto

The v4 protocol first evaluates the untouched base model. If the base checkpoint
passes the registered verifier/format gate, all compared methods start from one
shared untrained LoRA adapter and no Countdown SFT is performed. A minimal SFT
fallback is used only when the base gate fails.

The run has two responsibilities:
  1. a fixed-negative-advantage near/far mechanism probe on matched legal wrong
     expressions;
  2. a minimal effect comparison: positive-only, controlled-negative, and
     uncontrolled-negative, all paired by data, initialization, and seed.

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
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
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

VERSION = "4.0.0-base-first"


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


def _structure_from_ast(node: ast.AST) -> str:
    if isinstance(node, ast.Expression):
        return _structure_from_ast(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
        return "N"
    if isinstance(node, ast.BinOp) and type(node.op) in _AST_OP:
        left = _structure_from_ast(node.left)
        right = _structure_from_ast(node.right)
        op = _AST_OP[type(node.op)]
        if op in {"+", "*"} and right < left:
            left, right = right, left
        return f"({op} {left} {right})"
    raise ValueError(f"unsupported structure node: {type(node).__name__}")


def expression_structure(text: str) -> str:
    """Canonical operator-tree signature, ignoring literal values."""
    expression = clean_expression(text)
    if not expression:
        raise ValueError("empty expression")
    return _structure_from_ast(ast.parse(expression, mode="eval"))


def _tree_depth(node: ast.AST) -> int:
    if isinstance(node, ast.Expression):
        return _tree_depth(node.body)
    if isinstance(node, ast.Constant):
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


def _structure_sort_key(signature: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{signature}".encode("utf-8")).hexdigest()


def generate_structural_splits(
    train_count: int,
    val_count: int,
    test_count: int,
    seed: int,
    n_numbers: int = 4,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Generate splits whose oracle structure is deterministically assigned once."""
    requested = {"train": train_count, "val": val_count, "test": test_count}
    rows: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    total = train_count + val_count + test_count
    train_cut = train_count / total
    val_cut = (train_count + val_count) / total
    rng = np.random.default_rng(seed)
    seen: set[tuple[tuple[int, ...], int]] = set()
    attempts = 0
    max_attempts = total * 800

    def split_for(signature: str) -> str:
        digest = int(_structure_sort_key(signature, seed), 16)
        value = digest / float(2**256 - 1)
        if value < train_cut:
            return "train"
        if value < val_cut:
            return "val"
        return "test"

    while any(len(rows[name]) < count for name, count in requested.items()):
        attempts += 1
        if attempts > max_attempts:
            counts = {name: len(value) for name, value in rows.items()}
            raise RuntimeError(f"Could not fill structural splits: {counts}")
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
        signature = expression_structure(expression)
        split = split_for(signature)
        # Reserve the numerical problem globally even when its structure bucket is full.
        seen.add(key)
        if len(rows[split]) >= requested[split]:
            continue
        rows[split].append({
            "id": f"cd_{seed}_{split}_{len(rows[split]):07d}",
            "numbers": numbers,
            "target": target,
            "prompt": make_prompt(numbers, target),
            "oracle": expression,
            "oracle_structure": signature,
            "split": split,
        })

    train_set = {r["oracle_structure"] for r in rows["train"]}
    val_set = {r["oracle_structure"] for r in rows["val"]}
    test_set = {r["oracle_structure"] for r in rows["test"]}
    if train_set & val_set or train_set & test_set or val_set & test_set:
        raise AssertionError("structural split overlap")
    manifest = {
        "seed": seed,
        "n_numbers": n_numbers,
        "train_examples": len(rows["train"]),
        "val_examples": len(rows["val"]),
        "test_examples": len(rows["test"]),
        "train_structures": len(train_set),
        "val_structures": len(val_set),
        "test_structures": len(test_set),
        "structure_sets_disjoint": True,
        "problem_keys_disjoint": True,
        "generation_attempts": attempts,
    }
    return rows["train"], rows["val"], rows["test"], manifest

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
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        prompts = [r["prompt"] for r in chunk]
        greedy = generate_outputs(
            model, tokenizer, prompts, max_new_tokens, False, 1.0, 1.0, 1
        )
        if pass_k > 1:
            samples = generate_outputs(
                model, tokenizer, prompts, max_new_tokens, True, 0.8, 0.95, pass_k
            )
        else:
            samples = greedy
        for row, gout, souts in zip(chunk, greedy, samples):
            g = verify_expression(gout[0], row["numbers"], row["target"])
            successes.append(float(g["correct"]))
            valid.append(float(g["valid_format"] and g["uses_numbers"]))
            greedy_unseen = False
            if g["correct"]:
                try:
                    signature = expression_structure(g["expression"])
                    observed_correct_structures.add(signature)
                    greedy_unseen = known_structures is not None and signature not in known_structures
                except Exception:
                    greedy_unseen = False
            greedy_unseen_successes.append(float(greedy_unseen))

            any_correct = False
            any_unseen_correct = False
            for text in souts:
                check = verify_expression(text, row["numbers"], row["target"])
                if not check["correct"]:
                    continue
                any_correct = True
                try:
                    signature = expression_structure(check["expression"])
                    observed_correct_structures.add(signature)
                    if known_structures is not None and signature not in known_structures:
                        any_unseen_correct = True
                except Exception:
                    pass
            sampled_successes.append(float(any_correct))
            pass_unseen_successes.append(float(any_unseen_correct))
    metrics = {
        "greedy_success": float(np.mean(successes)),
        "pass_at_k": float(np.mean(sampled_successes)),
        "valid_rate": float(np.mean(valid)),
        "greedy_unseen_structure_success": float(np.mean(greedy_unseen_successes)),
        "pass_at_k_unseen_structure": float(np.mean(pass_unseen_successes)),
        "unique_correct_structures": float(len(observed_correct_structures)),
        "n_eval": float(len(rows)),
    }
    if was_training:
        model.train()
    return metrics

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
    """LoRA/QLoRA SFT with fixed-seed validation and best-epoch checkpointing."""
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
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    updates_per_epoch = math.ceil(len(loader) / args.grad_accum)
    total_updates = max(1, updates_per_epoch * args.epochs)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, max(1, int(total_updates * args.warmup_ratio)), total_updates
    )
    out_dir = Path(args.output_dir)
    best_dir = out_dir / "best_adapter"
    out_dir.mkdir(parents=True, exist_ok=True)
    if best_dir.exists():
        shutil.rmtree(best_dir)
    best_value = -float("inf")
    best_epoch = -1
    eval_rows: list[dict[str, Any]] = []
    model.train()
    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        running_loss = 0.0
        micro_count = 0
        for step, batch in enumerate(loader):
            batch = move_to_device(batch, device)
            out = model(**batch, use_cache=False)
            raw_loss = out.loss
            (raw_loss / args.grad_accum).backward()
            running_loss += float(raw_loss.detach())
            micro_count += 1
            if (step + 1) % args.grad_accum == 0 or step + 1 == len(loader):
                torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if global_step % args.log_every == 0:
                    print(json.dumps({
                        "stage": "sft", "epoch": epoch + 1, "update": global_step,
                        "loss": running_loss / max(micro_count, 1),
                        "lr": scheduler.get_last_lr()[0],
                    }))
                    running_loss = 0.0
                    micro_count = 0
        # Fixed validation seed makes epochs directly comparable.
        metrics = evaluate_rows(
            model, tokenizer, val_rows[: args.eval_examples], args.eval_batch,
            args.max_new_tokens, args.pass_k, args.eval_seed,
        )
        row = {"epoch": epoch + 1, "update": global_step, **metrics}
        eval_rows.append(row)
        print("SFT_EVAL", json.dumps(row))
        value = float(metrics[args.selection_metric])
        if value > best_value + args.selection_delta:
            best_value = value
            best_epoch = epoch + 1
            if best_dir.exists():
                shutil.rmtree(best_dir)
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
        model.train()
    if best_epoch < 0:
        raise RuntimeError("SFT did not produce a valid checkpoint")
    # Publish only the best validation adapter at output_dir root.
    for child in list(out_dir.iterdir()):
        if child.name == "best_adapter":
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
    for child in best_dir.iterdir():
        target = out_dir / child.name
        if child.is_dir():
            shutil.copytree(child, target)
        else:
            shutil.copy2(child, target)
    with (out_dir / "sft_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(eval_rows[0].keys()))
        writer.writeheader(); writer.writerows(eval_rows)
    (out_dir / "sft_manifest.json").write_text(json.dumps({
        **vars(args), "best_epoch": best_epoch, "best_value": best_value,
    }, indent=2))


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
    indices = [i for i, c in enumerate(expression) if c in OPS]
    if not indices:
        return expression + " + 1"
    idx = rng.choice(indices)
    choices = [x for x in OPS if x != expression[idx]]
    return expression[:idx] + rng.choice(choices) + expression[idx + 1 :]


def make_valid_wrong_expression(
    row: dict[str, Any],
    rng: random.Random,
    avoid: set[str] | None = None,
    max_attempts: int = 200,
) -> str:
    """Create a syntactically valid, number-preserving, verifier-negative expression."""
    avoid = avoid or set()
    candidates = [mutate_expression(row["oracle"], rng)]
    np_rng = np.random.default_rng(rng.randrange(2**31))
    for _ in range(max_attempts):
        if candidates:
            expression = candidates.pop()
        else:
            expression, _ = random_expression(np_rng, list(row["numbers"]))
        check = verify_expression(expression, row["numbers"], row["target"])
        if (
            check["valid_format"]
            and check["uses_numbers"]
            and not check["correct"]
            and check["expression"] not in avoid
        ):
            return check["expression"]
    raise RuntimeError(f"Could not construct a valid wrong expression for {row['id']}")


def candidate_metadata(
    item: dict[str, Any], tokenizer: Any
) -> dict[str, Any]:
    expression = item["expression"]
    value = item.get("value")
    return {
        **item,
        "text": expression,
        "token_length": len(tokenizer(expression, add_special_tokens=False)["input_ids"]),
        "tree_depth": expression_tree_depth(expression),
        "value_error": abs(float(value) - float(item["target"])) if value is not None else float("inf"),
    }


def select_matched_negative_pair(
    candidates: list[dict[str, Any]],
    min_surprisal_gap: float,
    max_token_length_diff: int,
    max_tree_depth_diff: int,
    max_value_error_ratio: float,
) -> tuple[dict[str, Any], dict[str, Any], bool]:
    """Pick the largest-surprisal-gap pair under simple task-distance matching."""
    ordered = sorted(candidates, key=lambda x: float(x["surprisal"]))
    best: tuple[float, dict[str, Any], dict[str, Any]] | None = None
    for i, near in enumerate(ordered):
        for far in ordered[i + 1 :]:
            gap = float(far["surprisal"]) - float(near["surprisal"])
            if gap < min_surprisal_gap:
                continue
            if abs(int(far["token_length"]) - int(near["token_length"])) > max_token_length_diff:
                continue
            if abs(int(far["tree_depth"]) - int(near["tree_depth"])) > max_tree_depth_diff:
                continue
            near_err = max(float(near["value_error"]), 1e-8)
            far_err = max(float(far["value_error"]), 1e-8)
            if max(near_err, far_err) / min(near_err, far_err) > max_value_error_ratio:
                continue
            if best is None or gap > best[0]:
                best = (gap, near, far)
    if best is not None:
        return best[1], best[2], True
    if len(ordered) < 2:
        raise ValueError("At least two valid negative candidates are required")
    return ordered[0], ordered[-1], False


def cmd_build_offline(args: argparse.Namespace) -> None:
    seed_all(args.seed)
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
    if args.max_examples > 0:
        rows = rows[: args.max_examples]
    rng = random.Random(args.seed)
    output_rows: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {
        "sampled_positive": 0,
        "fallback_negative": 0,
        "matched_pair": 0,
        "unmatched_pair": 0,
        "valid_wrong_candidates": [],
    }
    for start in range(0, len(rows), args.batch_size):
        chunk = rows[start : start + args.batch_size]
        generated = generate_outputs(
            model,
            tokenizer,
            [r["prompt"] for r in chunk],
            args.max_new_tokens,
            True,
            args.temperature,
            args.top_p,
            args.rollouts,
        )
        candidate_groups: list[list[dict[str, Any]]] = []
        score_pairs: list[tuple[str, str]] = []
        score_refs: list[tuple[int, int]] = []
        for row_idx, (row, candidates) in enumerate(zip(chunk, generated)):
            evaluated: list[dict[str, Any]] = []
            seen_expr: set[str] = set()
            for text in candidates:
                check = verify_expression(text, row["numbers"], row["target"])
                expr = check["expression"]
                if not expr or expr in seen_expr:
                    continue
                seen_expr.add(expr)
                item = {**check, "target": row["target"]}
                evaluated.append(item)
                score_pairs.append((row["prompt"], expr))
                score_refs.append((row_idx, len(evaluated) - 1))
            candidate_groups.append(evaluated)
        if score_pairs:
            scores = score_completions_batch(
                model, tokenizer, score_pairs, args.max_length, args.score_batch_size
            )
            for (row_idx, cand_idx), value in zip(score_refs, scores):
                candidate_groups[row_idx][cand_idx]["surprisal"] = value

        for row, evaluated in zip(chunk, candidate_groups):
            correct = [x for x in evaluated if x["correct"]]
            oracle_structure = row.get("oracle_structure") or expression_structure(row["oracle"])
            same_structure_correct = []
            for item in correct:
                try:
                    if expression_structure(item["expression"]) == oracle_structure:
                        same_structure_correct.append(item)
                except Exception:
                    pass
            if same_structure_correct:
                positive_item = min(same_structure_correct, key=lambda x: float(x["surprisal"]))
                positive = positive_item["expression"]
                positive_s = float(positive_item["surprisal"])
                diagnostics["sampled_positive"] += 1
            else:
                # Keep positive training inside the registered train-structure support.
                positive = row["oracle"]
                positive_s = score_completions_batch(
                    model, tokenizer, [(row["prompt"], positive)], args.max_length, 1
                )[0]

            valid_wrong = [
                x for x in evaluated
                if x["valid_format"] and x["uses_numbers"] and not x["correct"]
            ]
            seen = {x["expression"] for x in valid_wrong}
            while len(valid_wrong) < 2:
                expression = make_valid_wrong_expression(row, rng, avoid=seen)
                seen.add(expression)
                check = verify_expression(expression, row["numbers"], row["target"])
                score = score_completions_batch(
                    model, tokenizer, [(row["prompt"], expression)], args.max_length, 1
                )[0]
                valid_wrong.append({
                    **check,
                    "target": row["target"],
                    "surprisal": score,
                })
                diagnostics["fallback_negative"] += 1

            detailed = [candidate_metadata(x, tokenizer) for x in valid_wrong]
            diagnostics["valid_wrong_candidates"].append(len(detailed))
            near_item, far_item, matched = select_matched_negative_pair(
                detailed,
                args.min_surprisal_gap,
                args.max_token_length_diff,
                args.max_tree_depth_diff,
                args.max_value_error_ratio,
            )
            diagnostics["matched_pair" if matched else "unmatched_pair"] += 1
            output_rows.append({
                **row,
                "positive": positive,
                "positive_base_surprisal": positive_s,
                "near_negative": near_item["text"],
                "far_negative": far_item["text"],
                "near_base_surprisal": float(near_item["surprisal"]),
                "far_base_surprisal": float(far_item["surprisal"]),
                "surprisal_gap": float(far_item["surprisal"] - near_item["surprisal"]),
                "near_token_length": int(near_item["token_length"]),
                "far_token_length": int(far_item["token_length"]),
                "near_tree_depth": int(near_item["tree_depth"]),
                "far_tree_depth": int(far_item["tree_depth"]),
                "near_value_error": float(near_item["value_error"]),
                "far_value_error": float(far_item["value_error"]),
                "pair_matched": bool(matched),
            })
        print(f"built {len(output_rows)}/{len(rows)}", flush=True)
    write_jsonl(args.output_data, output_rows)
    candidate_counts = diagnostics.pop("valid_wrong_candidates")
    manifest = {
        **vars(args),
        **diagnostics,
        "reference_adapter": reference_adapter,
        "examples": len(output_rows),
        "matched_pair_rate": diagnostics["matched_pair"] / max(len(output_rows), 1),
        "mean_valid_wrong_candidates": float(np.mean(candidate_counts)) if candidate_counts else 0.0,
        "mean_surprisal_gap": float(np.mean([r["surprisal_gap"] for r in output_rows])) if output_rows else 0.0,
    }
    Path(str(args.output_data) + ".manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))

def cmd_train_method(args: argparse.Namespace) -> None:
    seed_all(args.seed)
    tokenizer = load_tokenizer(args.model_path)
    train_rows = read_jsonl(args.offline_data)
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
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, max(1, int(args.steps * args.warmup_ratio)), args.steps
    )
    model.train()
    iterator = iter(loader)
    metrics_rows: list[dict[str, Any]] = []
    out_dir = Path(args.output_dir)
    best_dir = out_dir / "best_adapter"
    last_gamma, last_weight = 1.0, 1.0
    best_value = -float("inf")
    best_step = 0
    stale_checks = 0
    numerical_failure: str | None = None
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
    metrics_rows.append(
        {"step": 0, "method": args.method, "gamma": 1.0, "weight": 1.0, **initial_eval}
    )
    best_value = float(initial_eval[args.selection_metric])
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(best_dir)
    tokenizer.save_pretrained(best_dir)

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
                near_seq_w = torch.ones_like(near["seq_lp"])
                far_seq_w = torch.ones_like(far["seq_lp"])
                far_token_w = torch.ones_like(far["token_lp"])
                if args.method == "controlled_negative":
                    far_token_surprisal = -far["token_lp"].detach()
                    far_token_w = torch.exp(
                        -args.exp_lambda
                        * F.relu(far_token_surprisal - args.surprisal_threshold)
                    )
                elif args.method in {"exp", "hybrid"}:
                    near_s = -near["seq_lp"].detach()
                    far_s = -far["seq_lp"].detach()
                    near_seq_w = torch.exp(
                        -args.exp_lambda * F.relu(near_s - args.surprisal_threshold)
                    )
                    far_seq_w = torch.exp(
                        -args.exp_lambda * F.relu(far_s - args.surprisal_threshold)
                    )

                gamma = 1.0
                if args.method == "global":
                    gamma = args.global_gamma
                elif args.method in {"sbrc", "hybrid"}:
                    pos_budget = pos["score"].detach().mean()
                    neg_budget = args.alpha * (
                        args.near_mix * (near_seq_w * near["score"].detach()).mean()
                        + args.far_mix * (far_seq_w * far["score"].detach()).mean()
                    )
                    g_score = min(
                        1.0, float(args.sbrc_kappa * pos_budget / (neg_budget + 1e-8))
                    )
                    current_h = float(pos["entropy"].detach().mean())
                    floor = args.entropy_floor
                    g_entropy = 1.0 if current_h >= floor else max(0.0, current_h / max(floor, 1e-8))
                    gamma = min(g_score, g_entropy)

                if args.method == "controlled_negative":
                    near_lp = near["seq_lp"]
                    far_lp = weighted_sequence_logprob(far, far_token_w)
                else:
                    near_lp = near_seq_w * near["seq_lp"]
                    far_lp = far_seq_w * far["seq_lp"]
                negative_lp = (
                    args.near_mix * near_lp.mean()
                    + args.far_mix * far_lp.mean()
                )
                objective = positive_lp - args.alpha * gamma * negative_lp
                raw_loss = -objective
                if args.method == "entropy_bonus":
                    raw_loss = raw_loss - args.entropy_coef * pos["entropy"].mean()
                elif args.method == "target_entropy":
                    entropy_gap = F.relu(args.target_entropy - pos["entropy"].mean())
                    raw_loss = raw_loss + args.target_entropy_coef * entropy_gap.square()

                near_weight_value = float(near_seq_w.detach().mean())
                far_weight_value = (
                    float((far_token_w.detach() * far["token_mask"]).sum() / far["token_mask"].sum().clamp_min(1))
                    if args.method == "controlled_negative"
                    else float(far_seq_w.detach().mean())
                )
                mean_weight = args.near_mix * near_weight_value + args.far_mix * far_weight_value

            if not bool(torch.isfinite(raw_loss)):
                numerical_failure = f"nonfinite_loss_at_step_{update_step}"
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
            break
        optimizer.step()
        scheduler.step()
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
            value = float(metrics[args.selection_metric])
            if value > best_value + args.early_stop_delta:
                best_value = value
                best_step = update_step
                stale_checks = 0
                if best_dir.exists():
                    shutil.rmtree(best_dir)
                model.save_pretrained(best_dir)
                tokenizer.save_pretrained(best_dir)
            else:
                stale_checks += 1
            model.train()
            if update_step >= args.min_steps and stale_checks >= args.early_stop_patience:
                print(json.dumps({"early_stop": True, "step": update_step,
                                  "best_step": best_step, "best_value": best_value}))
                stop_training = True
        if stop_training:
            break

    final_adapter = out_dir / "adapter"
    if final_adapter.exists():
        shutil.rmtree(final_adapter)
    shutil.copytree(best_dir, final_adapter)
    with (out_dir / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metrics_rows)
    (out_dir / "manifest.json").write_text(json.dumps({
        **vars(args),
        "best_step": best_step,
        "best_value": best_value,
        "numerical_failure": numerical_failure,
        "result_status": "pilot",
    }, indent=2))


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
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output)
    tokenizer.save_pretrained(output)
    (output / "initialization_manifest.json").write_text(json.dumps({
        "version": VERSION,
        "seed": args.seed,
        "model_path": str(Path(args.model_path).resolve()),
        "training_updates": 0,
        "purpose": "shared zero-effect LoRA initialization for paired base-first methods",
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
    """One-command, resumable, base-first 0.5B Countdown experiment."""
    if args.gpu != "auto":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    requested_memory_mode = args.memory_mode
    plan = resolve_execution_plan(args.model_path, args.preset, args.memory_mode, args.gpu)
    root = Path(args.work_dir).resolve()
    data_dir = root / "data"
    logs = root / "logs"
    reference_dir = root / "reference_adapter"
    sft_dir = root / "sft_adapter"
    offline_file = data_dir / "offline.jsonl"
    methods_dir = root / "methods"
    root.mkdir(parents=True, exist_ok=True)

    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    allowed = {
        "positive_only",
        "controlled_negative",
        "uncontrolled_negative",
        "uncontrolled",
        "global",
        "exp",
        "entropy_bonus",
        "target_entropy",
        "sbrc",
        "hybrid",
    }
    unknown = set(methods) - allowed
    if unknown:
        raise ValueError(f"Unknown methods: {sorted(unknown)}")

    run_spec = {
        "version": VERSION,
        "model_path": str(Path(args.model_path).resolve()),
        "preset": plan["preset"],
        "memory_mode": plan["memory_mode"],
        "methods": methods,
        "seed": args.seed,
        "min_base_success": args.min_base_success,
        "min_base_valid": args.min_base_valid,
        "min_sft_success": args.min_sft_success,
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
        "0.5b": dict(train=6000, val=500, test=1000, offline=1500, rollouts=12,
                     sft_epochs=3, sft_accum=16, method_steps=1200, method_accum=8,
                     method_min_steps=400, patience=6, eval_examples=500,
                     eval_every=100, pass_k=8, probe_examples=32, dynamics_examples=16),
        "small": dict(train=6000, val=500, test=1000, offline=1500, rollouts=12,
                      sft_epochs=3, sft_accum=16, method_steps=1200, method_accum=8,
                      method_min_steps=400, patience=6, eval_examples=500,
                      eval_every=100, pass_k=8, probe_examples=32, dynamics_examples=16),
        "3b": dict(train=20000, val=1000, test=2000, offline=4000, rollouts=12,
                   sft_epochs=3, sft_accum=32, method_steps=3000, method_accum=16,
                   method_min_steps=1000, patience=8, eval_examples=1000,
                   eval_every=100, pass_k=8, probe_examples=32, dynamics_examples=16),
        "7b": dict(train=20000, val=1000, test=2000, offline=4000, rollouts=12,
                   sft_epochs=2, sft_accum=32, method_steps=2500, method_accum=16,
                   method_min_steps=800, patience=8, eval_examples=1000,
                   eval_every=100, pass_k=8, probe_examples=24, dynamics_examples=12),
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
        try:
            _run_stage(["preflight", *model_flags, "--seed", str(args.seed)], logs / "00_preflight.log")
        except RuntimeError:
            if requested_memory_mode == "auto" and not plan["load_in_4bit"]:
                print("BF16 preflight failed; retrying with 4-bit QLoRA.")
                plan["memory_mode"] = "qlora"
                plan["load_in_4bit"] = True
                model_flags = model_flags_from_plan(plan)
                (root / "execution_plan.json").write_text(json.dumps(plan, indent=2))
                _run_stage(["preflight", *model_flags, "--seed", str(args.seed)],
                           logs / "00_preflight_qlora.log")
            else:
                raise

    if args.force or not (train_file.exists() and val_file.exists() and test_file.exists()):
        _run_stage([
            "generate",
            "--train", str(preset["train"]),
            "--val", str(preset["val"]),
            "--test", str(preset["test"]),
            "--train_out", str(train_file),
            "--val_out", str(val_file),
            "--test_out", str(test_file),
            "--manifest_out", str(data_dir / "split_manifest.json"),
            "--seed", str(args.seed),
        ], logs / "01_generate_structural_split.log")

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
        if args.force and reference_dir.exists():
            shutil.rmtree(reference_dir)
        if args.force or not (reference_dir / "adapter_config.json").exists():
            _run_stage([
                "init_adapter", *model_flags,
                "--output_dir", str(reference_dir),
                "--seed", str(args.seed),
            ], logs / "03_init_shared_adapter.log")
    else:
        if args.no_sft_fallback:
            raise RuntimeError(
                f"Base checkpoint failed gate: greedy={base_val['greedy_success']:.3f}, "
                f"valid={base_val['valid_rate']:.3f}. SFT fallback disabled."
            )
        initialization_mode = "minimal_sft_fallback"
        if args.force and sft_dir.exists():
            shutil.rmtree(sft_dir)
        if args.force or not (sft_dir / "adapter_config.json").exists():
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
        reference_dir = sft_dir
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
            "--output_data", str(offline_file),
            "--max_examples", str(preset["offline"]),
            "--rollouts", str(preset["rollouts"]),
            "--batch_size", str(plan["rollout_batch"]),
            "--score_batch_size", str(plan["score_batch"]),
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

    shared_method_seed = args.seed + 100
    for method in methods:
        out = methods_dir / method
        if args.force and out.exists():
            shutil.rmtree(out)
        if args.force or not (out / "adapter" / "adapter_config.json").exists():
            _run_stage([
                "train_method", *model_flags,
                "--reference_adapter", str(reference_dir),
                "--offline_data", str(offline_file),
                "--val_data", str(val_file),
                "--structure_reference_data", str(train_file),
                "--output_dir", str(out),
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
            ], logs / f"07_train_{method}.log")

    summary_rows: list[dict[str, Any]] = []
    base_test_json = root / "base_test_metrics.json"
    _run_stage([
        "evaluate", *model_flags,
        "--data", str(test_file),
        "--structure_reference_data", str(train_file),
        "--batch_size", str(plan["eval_batch"]),
        "--pass_k", str(preset["pass_k"]),
        "--output_json", str(base_test_json),
        "--seed", str(args.seed + 7000),
    ], logs / "08_test_raw_base.log")
    summary_rows.append({"method": "raw_base_no_training", **json.loads(base_test_json.read_text())})

    reference_test_json = root / "reference_test_metrics.json"
    _run_stage([
        "evaluate", *model_flags,
        "--adapter", str(reference_dir),
        "--data", str(test_file),
        "--structure_reference_data", str(train_file),
        "--batch_size", str(plan["eval_batch"]),
        "--pass_k", str(preset["pass_k"]),
        "--output_json", str(reference_test_json),
        "--seed", str(args.seed + 7000),
    ], logs / "08b_test_reference.log")
    summary_rows.append({
        "method": "shared_initial_checkpoint",
        "initialization_mode": initialization_mode,
        **json.loads(reference_test_json.read_text()),
    })

    for method in methods:
        out = methods_dir / method
        adapter = out / "adapter"
        result_json = out / "test_metrics.json"
        _run_stage([
            "evaluate", *model_flags,
            "--adapter", str(adapter),
            "--data", str(test_file),
            "--structure_reference_data", str(train_file),
            "--batch_size", str(plan["eval_batch"]),
            "--pass_k", str(preset["pass_k"]),
            "--output_json", str(result_json),
            "--seed", str(args.seed + 7000),
        ], logs / f"09_test_{method}.log")
        row: dict[str, Any] = {"method": method, **json.loads(result_json.read_text())}
        manifest_path = out / "manifest.json"
        metrics_path = out / "metrics.csv"
        if manifest_path.exists() and metrics_path.exists():
            manifest = json.loads(manifest_path.read_text())
            best_step = int(manifest["best_step"])
            with metrics_path.open() as f:
                records = list(csv.DictReader(f))
            best_record = next((r for r in records if int(r["step"]) == best_step), records[-1])
            row["best_step"] = best_step
            row["best_val"] = manifest["best_value"]
            row["numerical_failure"] = manifest.get("numerical_failure")
            row.update({f"val_{k}": v for k, v in best_record.items() if k != "method"})
        summary_rows.append(row)

    summary_path = root / "arena_summary.csv"
    fields = sorted({k for row in summary_rows for k in row.keys()})
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)
    complete = {
        "version": VERSION,
        "plan": plan,
        "initialization_mode": initialization_mode,
        "base_validation": base_val,
        "reference_validation": reference_val,
        "mechanism_probe": json.loads(mechanism_json.read_text()),
        "summary": summary_rows,
        "formal_result_status": "pilot",
        "note": "A completed single-seed run is a pilot, not a formal multi-seed result.",
    }
    (root / "run_complete.json").write_text(json.dumps(complete, indent=2))
    print("\nDONE. Summary:", summary_path)
    for row in summary_rows:
        print(json.dumps(row, ensure_ascii=False))


def cmd_selftest(args: argparse.Namespace) -> None:
    oracle = "((1 + 2) * (3 + 4))"
    check = verify_expression(oracle, [1, 2, 3, 4], 21)
    assert check["correct"]
    assert expression_structure("(1 + 2) * (3 + 4)") == expression_structure("(2 + 1) * (4 + 3)")
    row = {
        "id": "selftest",
        "numbers": [1, 2, 3, 4],
        "target": 21,
        "oracle": oracle,
    }
    wrong = make_valid_wrong_expression(row, random.Random(7))
    wrong_check = verify_expression(wrong, row["numbers"], row["target"])
    assert wrong_check["valid_format"] and wrong_check["uses_numbers"] and not wrong_check["correct"]
    near, far, matched = select_matched_negative_pair([
        {"surprisal": 1.0, "token_length": 7, "tree_depth": 2, "value_error": 4.0},
        {"surprisal": 2.0, "token_length": 8, "tree_depth": 2, "value_error": 5.0},
        {"surprisal": 5.0, "token_length": 8, "tree_depth": 3, "value_error": 6.0},
    ], 0.5, 2, 1, 4.0)
    assert matched and near["surprisal"] < far["surprisal"]
    train, val, test, manifest = generate_structural_splits(40, 12, 12, seed=17)
    train_s = {x["oracle_structure"] for x in train}
    val_s = {x["oracle_structure"] for x in val}
    test_s = {x["oracle_structure"] for x in test}
    assert not (train_s & val_s or train_s & test_s or val_s & test_s)
    assert manifest["structure_sets_disjoint"]
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
    ap.add_argument("--output_data", required=True)
    ap.add_argument("--rollouts", type=int, default=12)
    ap.add_argument("--batch_size", type=int, default=4)
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
    ap.add_argument("--memory_mode", choices=["auto", "bf16", "qlora"], default="auto")
    ap.add_argument("--methods", default="positive_only,controlled_negative,uncontrolled_negative")
    ap.add_argument("--min_base_success", type=float, default=0.15)
    ap.add_argument("--min_base_valid", type=float, default=0.80)
    ap.add_argument("--min_sft_success", type=float, default=0.15)
    ap.add_argument("--min_matched_pairs", type=int, default=16)
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
