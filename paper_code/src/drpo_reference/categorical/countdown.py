"""Unified reviewer-facing Structured Generation task and objective core.

All nine tasks use one data path: task adapter -> verified oracle/negative bank
-> common completion likelihoods -> method objective -> verifier evaluation.
Task-specific code is restricted to instance construction, canonicalization,
verification, and model-independent negative mutations.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import random
import re
import sys
import types
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import torch
import torch.nn.functional as F

REASONING_GYM_COMMIT = "49b07130b3fcd12f2d064bba7c43869543a0e7e7"
WIKISQL_COMMIT = "7857cfd8aefcc9823245c370f9e39ecd55745ea6"
TASK_NAMES = (
    "countdown",
    "word_sorting",
    "spiral_matrix",
    "mini_sudoku",
    "maze",
    "word_ladder",
    "knights_knaves",
    "graph_color",
    "wikisql",
)
REASONING_GYM_TASKS = TASK_NAMES[1:-1]


OPS = ("+", "-", "*", "/")
MAX_EXPRESSION_LENGTH = 200


def clean_expression(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", str(text), flags=re.DOTALL | re.IGNORECASE)
    answer = re.search(r"<answer>(.*?)</answer>", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if answer:
        cleaned = answer.group(1)
    cleaned = cleaned.replace("```python", "").replace("```", "").strip()
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    cleaned = lines[-1] if lines else ""
    cleaned = re.sub(r"^(answer|expression)\s*[:=]\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.rstrip(". \t")
    return cleaned.split("=", 1)[0].strip() if "=" in cleaned else cleaned


class ExpressionVerifier(ast.NodeVisitor):
    def __init__(self) -> None:
        self.numbers: list[int] = []

    def visit_Expression(self, node: ast.Expression) -> Fraction:
        return self.visit(node.body)

    def visit_Constant(self, node: ast.Constant) -> Fraction:
        if isinstance(node.value, bool) or not isinstance(node.value, int):
            raise TypeError("only integer literals are allowed")
        self.numbers.append(int(node.value))
        return Fraction(int(node.value), 1)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Fraction:
        raise ValueError("unary operators are not allowed")

    def visit_BinOp(self, node: ast.BinOp) -> Fraction:
        left, right = self.visit(node.left), self.visit(node.right)
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
        raise TypeError(f"unsupported syntax: {type(node).__name__}")


def verify_expression(text: str, numbers: Sequence[int], target: int) -> dict[str, Any]:
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
        result["uses_numbers"] = Counter(visitor.numbers) == Counter(int(x) for x in numbers)
        result["value"] = float(value)
        result["correct"] = bool(result["uses_numbers"] and value == Fraction(int(target), 1))
    except (SyntaxError, TypeError, ValueError, ZeroDivisionError):
        pass
    return result


def _random_expression(rng: np.random.Generator, numbers: list[int]) -> tuple[str, Fraction]:
    pool: list[tuple[str, Fraction]] = [(str(n), Fraction(n, 1)) for n in numbers]
    rng.shuffle(pool)
    while len(pool) > 1:
        i, j = rng.choice(len(pool), size=2, replace=False)
        left, right = pool[max(i, j)], pool[min(i, j)]
        del pool[max(i, j)]
        del pool[min(i, j)]
        op = str(rng.choice(OPS))
        if op == "/" and right[1] == 0:
            op = "+"
        if op == "+":
            value = left[1] + right[1]
        elif op == "-":
            value = left[1] - right[1]
        elif op == "*":
            value = left[1] * right[1]
        else:
            value = left[1] / right[1]
        pool.append((f"({left[0]} {op} {right[0]})", value))
    return pool[0]


def _countdown_prompt(numbers: Sequence[int], target: int) -> str:
    return (
        f"Numbers: {', '.join(map(str, numbers))}\n"
        f"Target: {target}\n"
        "Return only a valid expression using every number exactly once."
    )


def _generate_countdown_examples(count: int, seed: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    seen: set[tuple[tuple[int, ...], int]] = set()
    rows: list[dict[str, Any]] = []
    attempts = 0
    while len(rows) < count:
        attempts += 1
        if attempts > count * 500:
            raise RuntimeError("Could not generate enough unique Countdown problems")
        numbers = rng.integers(1, 10, size=4).tolist()
        expression, value = _random_expression(rng, numbers.copy())
        if value.denominator != 1:
            continue
        target = int(value)
        key = (tuple(sorted(numbers)), target)
        if not 5 <= target <= 100 or key in seen:
            continue
        if not verify_expression(expression, numbers, target)["correct"]:
            continue
        seen.add(key)
        rows.append(
            {
                "id": f"cd_{seed}_{len(rows):07d}",
                "numbers": numbers,
                "target": target,
                "prompt": _countdown_prompt(numbers, target),
                "oracle": expression,
            }
        )
    return rows


def _mutate_operator_candidates(expression: str) -> list[tuple[str, str]]:
    candidates: list[tuple[str, str]] = []
    for index, char in enumerate(expression):
        if char in OPS:
            for replacement in OPS:
                if replacement != char:
                    candidates.append(
                        (
                            expression[:index] + replacement + expression[index + 1 :],
                            "operator_flip",
                        )
                    )
    return candidates


def _random_expression_candidates(
    numbers: Sequence[int], rng: random.Random, max_candidates: int
) -> list[tuple[str, str]]:
    candidates: list[tuple[str, str]] = []
    for _ in range(max_candidates):
        np_rng = np.random.default_rng(rng.randrange(0, 2**32 - 1))
        expression, _ = _random_expression(np_rng, list(numbers))
        candidates.append((expression, "random_tree"))
    return candidates


def countdown_modules() -> tuple[Any, Any]:
    countdown = types.SimpleNamespace(
        generate_examples=_generate_countdown_examples,
        verify_expression=verify_expression,
    )
    bank = types.SimpleNamespace(
        _mutate_operator_candidates=_mutate_operator_candidates,
        _random_expression_candidates=_random_expression_candidates,
    )
    return countdown, bank


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def string_edit_distance(left: str, right: str) -> int:
    """Return deterministic Levenshtein distance without an optional dependency."""
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for row, char_left in enumerate(left, start=1):
        current = [row]
        for column, char_right in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (char_left != char_right),
                )
            )
        previous = current
    return previous[-1]


def strip_answer_wrapper(text: str) -> str:
    text = str(text).strip()
    answer = re.findall(
        r"<answer>\s*(.*?)\s*</answer>",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if answer:
        text = answer[-1].strip()
    text = re.sub(
        r"^```(?:json|sql|python)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


@dataclass(frozen=True)
class VerificationResult:
    score: float
    correct: bool
    format_valid: bool
    error_class: str
    canonical_completion: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TaskInstance:
    task: str
    prompt_id: str
    prompt: str
    oracle_completion: str
    metadata: dict[str, Any]
    source_entry: dict[str, Any] = field(repr=False)


@dataclass(frozen=True)
class Mutation:
    completion: str
    mutation_class: str


class TaskAdapter(ABC):
    name: str
    source_kind: str
    source_revision: str
    output_structure: str

    @abstractmethod
    def generate_instances(self, count: int, seed: int) -> Iterable[TaskInstance]:
        """Generate deterministic source instances."""

    @abstractmethod
    def verify(
        self,
        instance: TaskInstance,
        completion: str,
        *,
        mutation_class: str | None = None,
    ) -> VerificationResult:
        """Verify one completion against the frozen instance."""

    @abstractmethod
    def mutation_candidates(self, instance: TaskInstance, rng: random.Random) -> Iterable[Mutation]:
        """Yield deterministic candidate negatives."""

    def accept_negative(self, result: VerificationResult) -> bool:
        return result.format_valid and not result.correct

    def build_bank_row(
        self,
        instance: TaskInstance,
        *,
        negative_count: int,
        seed: int,
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        oracle_check = self.verify(instance, instance.oracle_completion)
        if not oracle_check.correct:
            return None, {
                "prompt_id": instance.prompt_id,
                "reason": "oracle_verification_failed",
                "oracle_verification": oracle_check.to_dict(),
            }

        rng = random.Random(
            int(
                stable_hash({"task": self.name, "prompt_id": instance.prompt_id, "seed": seed})[
                    :16
                ],
                16,
            )
        )
        by_class: dict[str, list[tuple[str, VerificationResult]]] = defaultdict(list)
        seen: set[str] = set()
        for mutation in self.mutation_candidates(instance, rng):
            result = self.verify(
                instance,
                mutation.completion,
                mutation_class=mutation.mutation_class,
            )
            canonical = result.canonical_completion
            if canonical in seen or not self.accept_negative(result):
                continue
            seen.add(canonical)
            by_class[result.error_class].append((mutation.completion, result))

        for error_class, values in by_class.items():
            values.sort(
                key=lambda item: stable_hash(
                    {
                        "task": self.name,
                        "prompt_id": instance.prompt_id,
                        "error_class": error_class,
                        "completion": item[1].canonical_completion,
                        "seed": seed,
                    }
                )
            )

        selected: list[tuple[str, VerificationResult]] = []
        class_names = sorted(by_class)
        cursor = 0
        while len(selected) < negative_count and class_names:
            error_class = class_names[cursor % len(class_names)]
            bucket = by_class[error_class]
            if bucket:
                selected.append(bucket.pop(0))
            if not bucket:
                class_names.remove(error_class)
                cursor = 0
            else:
                cursor += 1

        if len(selected) < negative_count:
            return None, {
                "prompt_id": instance.prompt_id,
                "reason": "insufficient_unique_verified_negatives",
                "available": len(selected),
                "required": negative_count,
                "available_error_classes": sorted(by_class),
            }

        negatives: list[dict[str, Any]] = []
        for index, (completion, result) in enumerate(selected):
            negatives.append(
                {
                    "negative_id": f"{instance.prompt_id}_neg_{index:03d}",
                    "completion": completion,
                    "canonical_completion": result.canonical_completion,
                    "verifier_score": result.score,
                    "binary_correct": result.correct,
                    "format_valid": result.format_valid,
                    "error_class": result.error_class,
                    "verification_details": result.details,
                    "string_edit_distance_to_oracle": string_edit_distance(
                        result.canonical_completion,
                        oracle_check.canonical_completion,
                    ),
                    "response_chars": len(completion),
                }
            )

        row = {
            "schema_version": 1,
            "task": self.name,
            "source_kind": self.source_kind,
            "source_revision": self.source_revision,
            "output_structure": self.output_structure,
            "prompt_id": instance.prompt_id,
            "prompt": instance.prompt,
            "oracle_completion": instance.oracle_completion,
            "oracle_verification": oracle_check.to_dict(),
            "metadata": instance.metadata,
            "negatives": negatives,
        }
        return row, {
            "prompt_id": instance.prompt_id,
            "reason": "accepted",
            "negative_count": len(negatives),
            "error_classes": sorted({item["error_class"] for item in negatives}),
        }


class CountdownAdapter(TaskAdapter):
    name = "countdown"
    source_kind = "drpo_countdown_generator"
    source_revision = "reviewer_unified_countdown_generator"
    output_structure = "arithmetic_expression"

    def generate_instances(self, count: int, seed: int) -> Iterable[TaskInstance]:
        countdown, _ = countdown_modules()
        for row in countdown.generate_examples(count, seed):
            prompt_id = str(row["id"])
            yield TaskInstance(
                task=self.name,
                prompt_id=prompt_id,
                prompt=str(row["prompt"]),
                oracle_completion=str(row["oracle"]),
                metadata={
                    "numbers": [int(value) for value in row["numbers"]],
                    "target": int(row["target"]),
                },
                source_entry=dict(row),
            )

    def verify(
        self,
        instance: TaskInstance,
        completion: str,
        *,
        mutation_class: str | None = None,
    ) -> VerificationResult:
        countdown, _ = countdown_modules()
        check = countdown.verify_expression(
            completion,
            instance.metadata["numbers"],
            int(instance.metadata["target"]),
        )
        correct = bool(check["correct"])
        if correct:
            error_class = "correct"
        elif not check["valid_format"]:
            error_class = "invalid_format"
        elif not check["uses_numbers"]:
            error_class = "number_mismatch"
        else:
            error_class = mutation_class or "arithmetic_wrong"
        return VerificationResult(
            score=float(correct),
            correct=correct,
            format_valid=bool(check["valid_format"]),
            error_class=error_class,
            canonical_completion=str(check["expression"]),
            details={
                "uses_numbers": bool(check["uses_numbers"]),
                "value": check["value"],
            },
        )

    def accept_negative(self, result: VerificationResult) -> bool:
        return (
            result.format_valid and not result.correct and bool(result.details.get("uses_numbers"))
        )

    def mutation_candidates(self, instance: TaskInstance, rng: random.Random) -> Iterable[Mutation]:
        _, countdown_bank = countdown_modules()
        oracle = instance.oracle_completion
        for expression, _ in countdown_bank._mutate_operator_candidates(oracle):
            yield Mutation(expression, "operator_flip")
        numbers = [int(value) for value in instance.metadata["numbers"]]
        for expression, _ in countdown_bank._random_expression_candidates(
            numbers,
            rng,
            max_candidates=256,
        ):
            yield Mutation(expression, "random_tree")
        for index, number in enumerate(numbers):
            altered = list(numbers)
            altered[index] = 1 + (number % 9)
            yield Mutation(" + ".join(map(str, altered)), "number_mismatch")
        yield Mutation("not an expression", "invalid_format")


class ReasoningGymRuntime:
    """Selectively load the pinned official modules without importing all extras."""

    _factory_by_root: ClassVar[dict[Path, types.ModuleType]] = {}

    @staticmethod
    def _load_module(name: str, path: Path, *, package: bool = False) -> types.ModuleType:
        locations = [str(path.parent)] if package else None
        spec = importlib.util.spec_from_file_location(
            name,
            path,
            submodule_search_locations=locations,
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not create import spec for {name} from {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    @classmethod
    def load(cls, checkout: str | Path) -> types.ModuleType:
        checkout = Path(checkout).resolve()
        package_root = checkout / "reasoning_gym"
        if not (package_root / "factory.py").is_file():
            raise FileNotFoundError(
                f"Reasoning Gym checkout is incomplete: {package_root / 'factory.py'}"
            )
        if checkout in cls._factory_by_root:
            return cls._factory_by_root[checkout]

        for name in tuple(sys.modules):
            if name == "reasoning_gym" or name.startswith("reasoning_gym."):
                del sys.modules[name]

        root = types.ModuleType("reasoning_gym")
        root.__path__ = [str(package_root)]  # type: ignore[attr-defined]
        root.__package__ = "reasoning_gym"
        sys.modules["reasoning_gym"] = root

        cls._load_module("reasoning_gym.utils", package_root / "utils.py")

        coaching_root = package_root / "coaching"
        coaching = types.ModuleType("reasoning_gym.coaching")
        coaching.__path__ = [str(coaching_root)]  # type: ignore[attr-defined]
        coaching.__package__ = "reasoning_gym.coaching"
        sys.modules["reasoning_gym.coaching"] = coaching
        attributes = cls._load_module(
            "reasoning_gym.coaching.attributes",
            coaching_root / "attributes.py",
        )
        base_curriculum = cls._load_module(
            "reasoning_gym.coaching.base_curriculum",
            coaching_root / "base_curriculum.py",
        )
        for attribute in (
            "AttributeDefinition",
            "RangeAttributeDefinition",
            "ScalarAttributeDefinition",
        ):
            setattr(coaching, attribute, getattr(attributes, attribute))
        coaching.BaseCurriculum = base_curriculum.BaseCurriculum

        cls._load_module("reasoning_gym.dataset", package_root / "dataset.py")
        factory = cls._load_module("reasoning_gym.factory", package_root / "factory.py")
        cls._load_module(
            "reasoning_gym.data",
            package_root / "data" / "__init__.py",
            package=True,
        )

        modules = {
            "word_sorting": ("algorithmic", "word_sorting.py"),
            "spiral_matrix": ("algorithmic", "spiral_matrix.py"),
            "word_ladder": ("algorithmic", "word_ladder.py"),
            "graph_color": ("algorithmic", "graph_color.py"),
            "mini_sudoku": ("games", "mini_sudoku.py"),
            "maze": ("games", "maze.py"),
            "knights_knaves": ("logic", "knights_knaves.py"),
        }
        for category in sorted({category for category, _ in modules.values()}):
            category_module = types.ModuleType(f"reasoning_gym.{category}")
            category_module.__path__ = [  # type: ignore[attr-defined]
                str(package_root / category)
            ]
            category_module.__package__ = f"reasoning_gym.{category}"
            sys.modules[f"reasoning_gym.{category}"] = category_module
        for task_name, (category, filename) in modules.items():
            cls._load_module(
                f"reasoning_gym.{category}.{task_name}",
                package_root / category / filename,
            )

        cls._factory_by_root[checkout] = factory
        return factory


class ReasoningGymAdapter(TaskAdapter):
    source_kind = "reasoning_gym"
    source_revision = REASONING_GYM_COMMIT

    _OUTPUT_STRUCTURES: ClassVar[dict[str, str]] = {
        "word_sorting": "comma_separated_word_permutation",
        "spiral_matrix": "space_separated_integer_sequence",
        "mini_sudoku": "four_by_four_integer_grid",
        "maze": "single_integer_shortest_path_length",
        "word_ladder": "comma_separated_word_path",
        "knights_knaves": "named_role_assignments",
        "graph_color": "json_vertex_color_map",
    }

    def __init__(
        self,
        name: str,
        checkout: str | Path,
        dataset_kwargs: Mapping[str, Any],
    ) -> None:
        if name not in REASONING_GYM_TASKS:
            raise ValueError(f"Unsupported Reasoning Gym task: {name}")
        self.name = name
        self.output_structure = self._OUTPUT_STRUCTURES[name]
        self.checkout = Path(checkout).resolve()
        self.dataset_kwargs = dict(dataset_kwargs)
        self.factory = ReasoningGymRuntime.load(self.checkout)
        self.dataset: Any | None = None

    def _make_dataset(self, count: int, seed: int) -> Any:
        kwargs = {**self.dataset_kwargs, "size": int(count), "seed": int(seed)}
        return self.factory.create_dataset(self.name, **kwargs)

    def generate_instances(self, count: int, seed: int) -> Iterable[TaskInstance]:
        self.dataset = self._make_dataset(count, seed)
        for index in range(count):
            entry = dict(self.dataset[index])
            oracle = entry.get("answer")
            if self.name == "graph_color":
                oracle = json.dumps(
                    entry["metadata"]["possible_answer"],
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            if not isinstance(oracle, str) or not oracle.strip():
                raise RuntimeError(f"{self.name} produced no serializable oracle at {index}")
            prompt_id = f"rg_{self.name}_{seed}_{index:07d}"
            yield TaskInstance(
                task=self.name,
                prompt_id=prompt_id,
                prompt=str(entry["question"]),
                oracle_completion=oracle,
                metadata=dict(entry.get("metadata", {})),
                source_entry=entry,
            )

    def _require_dataset(self) -> Any:
        if self.dataset is None:
            raise RuntimeError("generate_instances must be called before verify")
        return self.dataset

    def _canonicalize(self, completion: str) -> tuple[str, bool]:
        text = strip_answer_wrapper(completion)
        try:
            if self.name == "word_sorting":
                words = [word.strip() for word in text.split(",") if word.strip()]
                return ", ".join(words), len(words) >= 2
            if self.name == "spiral_matrix":
                values = [int(value) for value in re.findall(r"-?\d+", text)]
                return " ".join(map(str, values)), bool(values)
            if self.name == "mini_sudoku":
                values = [int(value) for value in re.findall(r"[1-4]", text)]
                canonical = "\n".join(
                    " ".join(map(str, values[index : index + 4]))
                    for index in range(0, len(values), 4)
                )
                return canonical, len(values) == 16
            if self.name == "maze":
                if not re.fullmatch(r"[+-]?\d+", text):
                    return text, False
                return str(int(text)), True
            if self.name == "word_ladder":
                words = [word.strip().upper() for word in text.split(",")]
                valid = len(words) >= 2 and all(re.fullmatch(r"[A-Z]+", word) for word in words)
                return ",".join(words), bool(valid)
            if self.name == "knights_knaves":
                assignments = self._require_dataset()._normalize_answer(text)
                canonical = json.dumps(sorted(assignments), ensure_ascii=False)
                return canonical, bool(assignments)
            if self.name == "graph_color":
                value = json.loads(text)
                if not isinstance(value, dict):
                    return text, False
                canonical_map = {str(key): int(color) for key, color in value.items()}
                return json.dumps(canonical_map, sort_keys=True, separators=(",", ":")), True
        except (TypeError, ValueError, json.JSONDecodeError):
            return text, False
        raise ValueError(self.name)

    def verify(
        self,
        instance: TaskInstance,
        completion: str,
        *,
        mutation_class: str | None = None,
    ) -> VerificationResult:
        canonical, format_valid = self._canonicalize(completion)
        score = 0.0
        if format_valid:
            try:
                score = float(
                    self._require_dataset().score_answer(completion, instance.source_entry)
                )
            except (TypeError, ValueError, KeyError):
                format_valid = False
                score = 0.0
        correct = score >= 1.0 - 1.0e-12
        details: dict[str, Any] = {"official_scorer": self.name}
        if self.name == "maze" and format_valid:
            predicted = int(canonical)
            oracle = int(instance.metadata["shortest_path_length"])
            signed_residual = predicted - oracle
            details.update(
                {
                    "predicted_path_length": predicted,
                    "oracle_path_length": oracle,
                    "signed_residual": signed_residual,
                    "absolute_residual": abs(signed_residual),
                }
            )
            if correct:
                error_class = "correct"
            elif signed_residual < 0:
                error_class = "path_length_underestimate"
            elif signed_residual > 0:
                error_class = "path_length_overestimate"
            else:
                error_class = "official_scorer_disagreement"
        else:
            error_class = (
                "correct"
                if correct
                else ("invalid_format" if not format_valid else mutation_class or "wrong_answer")
            )
        return VerificationResult(
            score=score,
            correct=correct,
            format_valid=format_valid,
            error_class=error_class,
            canonical_completion=canonical,
            details=details,
        )

    def mutation_candidates(self, instance: TaskInstance, rng: random.Random) -> Iterable[Mutation]:
        method = getattr(self, f"_mutate_{self.name}")
        yield from method(instance, rng)

    def _mutate_word_sorting(
        self, instance: TaskInstance, rng: random.Random
    ) -> Iterable[Mutation]:
        del rng
        words = list(instance.metadata["sorted_words"])
        for index in range(len(words) - 1):
            candidate = list(words)
            candidate[index], candidate[index + 1] = candidate[index + 1], candidate[index]
            yield Mutation(", ".join(candidate), "adjacent_order_error")
        for shift in range(1, len(words)):
            yield Mutation(", ".join(words[shift:] + words[:shift]), "cyclic_order_error")
        for index in range(len(words)):
            yield Mutation(", ".join(words[:index] + words[index + 1 :]), "missing_word")
            duplicate = list(words)
            duplicate[index] = words[(index + 1) % len(words)]
            yield Mutation(", ".join(duplicate), "duplicate_word")
        yield Mutation(" ".join(words), "invalid_format")

    def _mutate_spiral_matrix(
        self, instance: TaskInstance, rng: random.Random
    ) -> Iterable[Mutation]:
        del rng
        values = [int(value) for value in instance.metadata["solution"]]
        for index in range(len(values) - 1):
            candidate = list(values)
            candidate[index], candidate[index + 1] = candidate[index + 1], candidate[index]
            yield Mutation(" ".join(map(str, candidate)), "local_order_error")
        for shift in range(1, min(len(values), 24)):
            yield Mutation(
                " ".join(map(str, values[shift:] + values[:shift])),
                "cyclic_order_error",
            )
        for index in range(min(len(values), 24)):
            candidate = list(values)
            candidate[index] = (candidate[index] + 1) % 10
            yield Mutation(" ".join(map(str, candidate)), "value_substitution")
        yield Mutation("[" + ", ".join(map(str, values)) + "]", "noncanonical_format")

    def _mutate_mini_sudoku(self, instance: TaskInstance, rng: random.Random) -> Iterable[Mutation]:
        del rng
        solution = [list(map(int, row)) for row in instance.metadata["solution"]]

        def render(board: Sequence[Sequence[int]]) -> str:
            return "\n".join(" ".join(map(str, row)) for row in board)

        for row in range(4):
            for column in range(4):
                board = [values[:] for values in solution]
                board[row][column] = 1 + (board[row][column] % 4)
                yield Mutation(render(board), "cell_value_error")
        for row in range(3):
            board = [values[:] for values in solution]
            board[row], board[row + 1] = board[row + 1], board[row]
            yield Mutation(render(board), "row_permutation_error")
        for column in range(3):
            board = [values[:] for values in solution]
            for row in range(4):
                board[row][column], board[row][column + 1] = (
                    board[row][column + 1],
                    board[row][column],
                )
            yield Mutation(render(board), "column_permutation_error")
        yield Mutation(render(solution[:-1]), "invalid_format")

    def _mutate_maze(self, instance: TaskInstance, rng: random.Random) -> Iterable[Mutation]:
        del rng
        answer = int(instance.metadata["shortest_path_length"])
        for offset in range(1, 33):
            yield Mutation(str(answer + offset), "path_length_overestimate")
            if answer - offset >= 0:
                yield Mutation(str(answer - offset), "path_length_underestimate")
        yield Mutation(f"{answer} steps", "invalid_format")

    def _mutate_word_ladder(self, instance: TaskInstance, rng: random.Random) -> Iterable[Mutation]:
        path = [word.strip().upper() for word in instance.oracle_completion.split(",")]
        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        for index, word in enumerate(path):
            for position in range(len(word)):
                replacement = alphabet[(alphabet.index(word[position]) + 1) % 26]
                candidate = list(path)
                candidate[index] = word[:position] + replacement + word[position + 1 :]
                mutation_class = (
                    "wrong_endpoint" if index in {0, len(path) - 1} else "invalid_internal_word"
                )
                yield Mutation(",".join(candidate), mutation_class)
        for index in range(1, max(1, len(path) - 1)):
            yield Mutation(",".join(path[:index] + path[index + 1 :]), "missing_step")
            yield Mutation(",".join(path[:index] + [path[index]] + path[index:]), "duplicate_step")
        for _ in range(32):
            candidate = list(path)
            index = rng.randrange(len(candidate))
            candidate[index] = "".join(rng.choice(alphabet) for _ in candidate[index])
            yield Mutation(",".join(candidate), "random_word_substitution")
        yield Mutation(" -> ".join(path), "invalid_format")

    def _mutate_knights_knaves(
        self, instance: TaskInstance, rng: random.Random
    ) -> Iterable[Mutation]:
        del rng
        names = list(instance.metadata["names"])
        solution = [bool(value) for value in instance.metadata["solution"]]
        terms = instance.metadata["knight_knave_terms"]
        true_role = str(terms["a_knight"])
        false_role = str(terms["a_knave"])

        def render(assignments: Sequence[tuple[str, bool]]) -> str:
            return ", ".join(
                f"{name} is {true_role if role else false_role}" for name, role in assignments
            )

        original = list(zip(names, solution))
        for mask in range(1, 1 << len(original)):
            candidate = [
                (name, (not role) if mask & (1 << index) else role)
                for index, (name, role) in enumerate(original)
            ]
            yield Mutation(render(candidate), "role_flip")
        for index in range(len(original)):
            yield Mutation(
                render(original[:index] + original[index + 1 :]),
                "missing_assignment",
            )
        yield Mutation("unknown", "invalid_format")

    def _mutate_graph_color(self, instance: TaskInstance, rng: random.Random) -> Iterable[Mutation]:
        del rng
        solution = {
            str(vertex): int(color)
            for vertex, color in instance.metadata["possible_answer"].items()
        }
        puzzle = instance.metadata["puzzle"]
        allowed = [int(value) for value in puzzle["color_options"]]
        for vertex in sorted(solution, key=int):
            candidate = dict(solution)
            del candidate[vertex]
            yield Mutation(
                json.dumps(candidate, sort_keys=True),
                "missing_vertex",
            )
            candidate = dict(solution)
            candidate[vertex] = max(allowed) + 1
            yield Mutation(
                json.dumps(candidate, sort_keys=True),
                "invalid_color",
            )
        for left, right in puzzle["edges"]:
            candidate = dict(solution)
            candidate[str(right)] = candidate[str(left)]
            yield Mutation(
                json.dumps(candidate, sort_keys=True),
                "edge_conflict",
            )
        yield Mutation("{not-json}", "invalid_format")


class WikiSQLAdapter(TaskAdapter):
    name = "wikisql"
    source_kind = "wikisql_official_archive"
    source_revision = WIKISQL_COMMIT
    output_structure = "json_wikisql_logical_form"

    def __init__(self, checkout: str | Path, split: str = "train") -> None:
        self.checkout = Path(checkout).resolve()
        self.split = split
        data_root = self.checkout / "data"
        self.examples_path = data_root / f"{split}.jsonl"
        self.tables_path = data_root / f"{split}.tables.jsonl"
        if not self.examples_path.is_file() or not self.tables_path.is_file():
            raise FileNotFoundError(
                "WikiSQL data is not extracted; expected "
                f"{self.examples_path} and {self.tables_path}"
            )
        self.tables = {str(row["id"]): row for row in self._read_jsonl(self.tables_path)}

    @staticmethod
    def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)

    @staticmethod
    def _canonical_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
        conds = [
            [int(column), int(operator), str(value).lower()]
            for column, operator, value in plan["conds"]
        ]
        conds.sort(key=lambda item: (item[0], item[1], item[2]))
        return {
            "sel": int(plan["sel"]),
            "agg": int(plan["agg"]),
            "conds": conds,
        }

    @staticmethod
    def _render_plan(plan: Mapping[str, Any]) -> str:
        return json.dumps(
            WikiSQLAdapter._canonical_plan(plan),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    def _parse_plan(
        self, completion: str, table: Mapping[str, Any]
    ) -> tuple[dict[str, Any] | None, str]:
        text = strip_answer_wrapper(completion)
        try:
            value = json.loads(text)
            if not isinstance(value, dict) or set(value) != {"sel", "agg", "conds"}:
                return None, text
            plan = self._canonical_plan(value)
            columns = len(table["header"])
            if not (0 <= plan["sel"] < columns and 0 <= plan["agg"] < 6):
                return None, text
            for column, operator, _ in plan["conds"]:
                if not (0 <= column < columns and 0 <= operator < 4):
                    return None, text
            return plan, self._render_plan(plan)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None, text

    def generate_instances(self, count: int, seed: int) -> Iterable[TaskInstance]:
        del seed
        for index, example in enumerate(self._read_jsonl(self.examples_path)):
            if index >= count:
                break
            table_id = str(example["table_id"])
            table = self.tables[table_id]
            schema = ", ".join(
                f"col{column}: {header} ({table['types'][column]})"
                for column, header in enumerate(table["header"])
            )
            prompt = (
                "Translate the question into the official WikiSQL JSON logical form. "
                'Return exactly {"sel":<column index>,"agg":<0..5>,'
                '"conds":[[<column index>,<0..3>,<value>],...]}. '
                "Aggregation IDs are 0=none, 1=MAX, 2=MIN, 3=COUNT, 4=SUM, "
                "5=AVG; condition IDs are 0='=', 1='>', 2='<', 3='OP'.\n"
                f"Schema: {schema}\nQuestion: {example['question']}"
            )
            oracle = self._render_plan(example["sql"])
            yield TaskInstance(
                task=self.name,
                prompt_id=f"wikisql_{self.split}_{index:07d}",
                prompt=prompt,
                oracle_completion=oracle,
                metadata={
                    "split": self.split,
                    "source_index": index,
                    "table_id": table_id,
                    "header": table["header"],
                    "types": table["types"],
                },
                source_entry={"example": example, "table": table},
            )

    def verify(
        self,
        instance: TaskInstance,
        completion: str,
        *,
        mutation_class: str | None = None,
    ) -> VerificationResult:
        table = instance.source_entry["table"]
        plan, canonical = self._parse_plan(completion, table)
        oracle = self._canonical_plan(instance.source_entry["example"]["sql"])
        correct = plan == oracle if plan is not None else False
        format_valid = plan is not None
        error_class = (
            "correct"
            if correct
            else ("invalid_format" if not format_valid else mutation_class or "logical_form_error")
        )
        return VerificationResult(
            score=float(correct),
            correct=correct,
            format_valid=format_valid,
            error_class=error_class,
            canonical_completion=canonical,
            details={"verifier": "official_logical_form_equivalence_unordered_conditions"},
        )

    def mutation_candidates(self, instance: TaskInstance, rng: random.Random) -> Iterable[Mutation]:
        del rng
        oracle = self._canonical_plan(instance.source_entry["example"]["sql"])
        table = instance.source_entry["table"]
        columns = len(table["header"])
        for column in range(columns):
            if column != oracle["sel"]:
                candidate = {**oracle, "sel": column}
                yield Mutation(self._render_plan(candidate), "selection_column_error")
        for aggregate in range(6):
            if aggregate != oracle["agg"]:
                candidate = {**oracle, "agg": aggregate}
                yield Mutation(self._render_plan(candidate), "aggregation_error")
        for index, condition in enumerate(oracle["conds"]):
            for column in range(columns):
                if column != condition[0]:
                    candidate = json.loads(json.dumps(oracle))
                    candidate["conds"][index][0] = column
                    yield Mutation(self._render_plan(candidate), "condition_column_error")
            for operator in range(4):
                if operator != condition[1]:
                    candidate = json.loads(json.dumps(oracle))
                    candidate["conds"][index][1] = operator
                    yield Mutation(self._render_plan(candidate), "condition_operator_error")
            for row in table["rows"][:16]:
                value = row[condition[0]]
                if str(value).lower() != str(condition[2]).lower():
                    candidate = json.loads(json.dumps(oracle))
                    candidate["conds"][index][2] = value
                    yield Mutation(self._render_plan(candidate), "condition_value_error")
            candidate = json.loads(json.dumps(oracle))
            del candidate["conds"][index]
            yield Mutation(self._render_plan(candidate), "missing_condition")
        for column in range(min(columns, 4)):
            if table["rows"]:
                candidate = json.loads(json.dumps(oracle))
                candidate["conds"].append([column, 0, table["rows"][0][column]])
                yield Mutation(self._render_plan(candidate), "spurious_condition")
        yield Mutation("{not-json}", "invalid_format")


def build_adapters(config: Mapping[str, Any], sources_root: str | Path) -> dict[str, TaskAdapter]:
    sources_root = Path(sources_root)
    requested = tuple(config["tasks"]["names"])
    unknown = sorted(set(requested) - set(TASK_NAMES))
    if unknown:
        raise ValueError(f"Unknown tasks: {unknown}")
    adapters: dict[str, TaskAdapter] = {}
    if "countdown" in requested:
        adapters["countdown"] = CountdownAdapter()
    rg_checkout = sources_root / "reasoning-gym"
    task_configs = config["tasks"].get("reasoning_gym", {})
    for name in REASONING_GYM_TASKS:
        if name in requested:
            adapters[name] = ReasoningGymAdapter(
                name,
                rg_checkout,
                task_configs.get(name, {}),
            )
    if "wikisql" in requested:
        adapters["wikisql"] = WikiSQLAdapter(
            sources_root / "wikisql",
            split=str(config["tasks"].get("wikisql", {}).get("split", "train")),
        )
    return {name: adapters[name] for name in requested}


STRUCTURED_GENERATION_SYSTEM_PROMPT = (
    "Answer with only the requested final output and no explanation."
)
STRUCTURED_GENERATION_METHODS = (
    "positive_only",
    "drpo",
    "asymre",
    "joint_fitted_reference_topr",
    "dpo",
)
IGNORE_INDEX = -100


@dataclass(frozen=True)
class EncodedCompletion:
    input_ids: list[int]
    labels: list[int]

    def __post_init__(self) -> None:
        if not self.input_ids or len(self.input_ids) != len(self.labels):
            raise ValueError("input_ids and labels must be aligned and non-empty")
        if all(label == IGNORE_INDEX for label in self.labels):
            raise ValueError("encoded sequence contains no completion token")


@dataclass(frozen=True)
class StructuredTrainingItem:
    positive: EncodedCompletion
    negatives: tuple[EncodedCompletion, ...]

    def __post_init__(self) -> None:
        if not self.negatives:
            raise ValueError("training item has no negative completion")


def format_chat_prompt(tokenizer: Any, prompt: str) -> str:
    messages = [
        {"role": "system", "content": STRUCTURED_GENERATION_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
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
    prefix = format_chat_prompt(tokenizer, prompt)
    eos = getattr(tokenizer, "eos_token", None)
    if not isinstance(eos, str) or not eos:
        raise ValueError("tokenizer must provide eos_token")
    prefix_ids = list(tokenizer(prefix, add_special_tokens=False)["input_ids"])
    full_ids = list(
        tokenizer(prefix + str(completion).strip() + eos, add_special_tokens=False)["input_ids"]
    )[:max_length]
    prefix_length = min(len(prefix_ids), len(full_ids))
    return EncodedCompletion(
        input_ids=full_ids,
        labels=[IGNORE_INDEX] * prefix_length + full_ids[prefix_length:],
    )


def pad_encoded(items: Sequence[EncodedCompletion], pad_id: int) -> dict[str, torch.Tensor]:
    width = max(len(item.input_ids) for item in items)
    ids, labels, masks = [], [], []
    for item in items:
        padding = width - len(item.input_ids)
        ids.append(item.input_ids + [int(pad_id)] * padding)
        labels.append(item.labels + [IGNORE_INDEX] * padding)
        masks.append([1] * len(item.input_ids) + [0] * padding)
    return {
        "input_ids": torch.tensor(ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(masks, dtype=torch.long),
    }


def encode_training_row(
    row: Mapping[str, Any],
    tokenizer: Any,
    max_length: int,
) -> StructuredTrainingItem:
    prompt = str(row["prompt"])
    oracle_verification = row.get("oracle_verification", {})
    positive_text = str(oracle_verification.get("canonical_completion", row["oracle_completion"]))
    positive = encode_prompt_completion(tokenizer, prompt, positive_text, max_length)
    negatives = tuple(
        encode_prompt_completion(
            tokenizer,
            prompt,
            str(item.get("canonical_completion", item["completion"])),
            max_length,
        )
        for item in row["negatives"]
    )
    return StructuredTrainingItem(positive=positive, negatives=negatives)


def collate_training_items(
    items: Sequence[StructuredTrainingItem],
    pad_id: int,
) -> dict[str, Any]:
    flattened = [negative for item in items for negative in item.negatives]
    row_index = [row for row, item in enumerate(items) for _ in range(len(item.negatives))]
    counts = [len(item.negatives) for item in items]
    return {
        "positive": pad_encoded([item.positive for item in items], pad_id),
        "negative": pad_encoded(flattened, pad_id),
        "negative_row_index": torch.tensor(row_index, dtype=torch.long),
        "negative_counts": torch.tensor(counts, dtype=torch.long),
    }


def move_tensor_batch_to_device(
    batch: Mapping[str, torch.Tensor], device: torch.device | str
) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def completion_statistics_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, torch.Tensor]:
    shifted_logits = logits[:, :-1, :].float()
    shifted_labels = labels[:, 1:]
    mask = shifted_labels.ne(IGNORE_INDEX)
    lengths = mask.sum(dim=-1)
    if bool((lengths <= 0).any()):
        raise ValueError("every sequence must contain a completion token")
    safe = shifted_labels.masked_fill(~mask, 0)
    token_lp = F.log_softmax(shifted_logits, dim=-1).gather(-1, safe.unsqueeze(-1)).squeeze(-1)
    summed = (token_lp * mask).sum(dim=-1)
    return {
        "mean_logprob": summed / lengths,
        "sum_logprob": summed,
        "lengths": lengths,
        "token_mask": mask,
    }


def completion_stats(model: Any, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    output = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        use_cache=False,
    )
    return completion_statistics_from_logits(output.logits, batch["labels"])


def prompt_balanced_mean(
    values: torch.Tensor,
    row_index: torch.Tensor,
    counts: torch.Tensor,
) -> torch.Tensor:
    if values.ndim != 1 or row_index.shape != values.shape:
        raise ValueError("values and row_index must be aligned vectors")
    result = torch.zeros(counts.numel(), dtype=values.dtype, device=values.device)
    result.index_add_(0, row_index.to(values.device), values)
    return (result / counts.to(values.device, values.dtype)).mean()


def drpo_weights(
    negative_mean_logprob: torch.Tensor,
    *,
    threshold: float,
    scale: float,
    coefficient: float,
) -> torch.Tensor:
    if scale <= 0.0 or coefficient < 0.0:
        raise ValueError("scale must be positive and coefficient non-negative")
    remoteness = -negative_mean_logprob.detach()
    normalized_far = torch.relu((remoteness - float(threshold)) / float(scale))
    return torch.exp(-float(coefficient) * normalized_far).detach()


def positive_only_objective(positive_mean_logprob: torch.Tensor) -> torch.Tensor:
    return -positive_mean_logprob.mean()


def drpo_objective(
    positive_mean_logprob: torch.Tensor,
    negative_mean_logprob: torch.Tensor,
    row_index: torch.Tensor,
    counts: torch.Tensor,
    *,
    threshold: float,
    scale: float,
    coefficient: float,
) -> torch.Tensor:
    weights = drpo_weights(
        negative_mean_logprob,
        threshold=threshold,
        scale=scale,
        coefficient=coefficient,
    )
    negative = prompt_balanced_mean(weights * negative_mean_logprob, row_index, counts)
    return -(positive_mean_logprob.mean() - negative)


def asymre_objective(
    positive_mean_logprob: torch.Tensor,
    negative_mean_logprob: torch.Tensor,
    row_index: torch.Tensor,
    counts: torch.Tensor,
    *,
    delta_v: float,
) -> torch.Tensor:
    positive = positive_mean_logprob.mean()
    negative = prompt_balanced_mean(negative_mean_logprob, row_index, counts)
    objective = 0.5 * ((1.0 - float(delta_v)) * positive + (-1.0 - float(delta_v)) * negative)
    return -objective


def topr_policy_objective(
    positive_mean_logprob: torch.Tensor,
    negative_mean_logprob: torch.Tensor,
    negative_sum_logprob: torch.Tensor,
    reference_negative_sum_logprob: torch.Tensor,
    row_index: torch.Tensor,
    counts: torch.Tensor,
    *,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    log_ratio = negative_sum_logprob - reference_negative_sum_logprob
    weights = torch.exp(float(beta) * torch.clamp(log_ratio, max=0.0)).detach()
    negative = prompt_balanced_mean(weights * negative_mean_logprob, row_index, counts)
    return -(positive_mean_logprob.mean() - negative), weights


def topr_reference_objective(
    positive_mean_logprob: torch.Tensor,
    negative_mean_logprob: torch.Tensor,
    row_index: torch.Tensor,
    counts: torch.Tensor,
) -> torch.Tensor:
    negative = prompt_balanced_mean(negative_mean_logprob, row_index, counts)
    return -(0.5 * positive_mean_logprob.mean() + 0.5 * negative)


def dpo_objective(
    policy_positive_sum_logprob: torch.Tensor,
    policy_negative_sum_logprob: torch.Tensor,
    reference_positive_sum_logprob: torch.Tensor,
    reference_negative_sum_logprob: torch.Tensor,
    row_index: torch.Tensor,
    counts: torch.Tensor,
    *,
    beta: float,
) -> torch.Tensor:
    index = row_index.to(policy_positive_sum_logprob.device)
    logits = float(beta) * (
        policy_positive_sum_logprob[index]
        - policy_negative_sum_logprob
        - reference_positive_sum_logprob[index]
        + reference_negative_sum_logprob
    )
    pair_losses = -F.logsigmoid(logits)
    return prompt_balanced_mean(pair_losses, row_index, counts)


def build_task_bank(
    adapter: TaskAdapter,
    *,
    candidate_rows: int,
    accepted_rows: int,
    negatives_per_prompt: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, TaskInstance]]:
    rows: list[dict[str, Any]] = []
    instances: dict[str, TaskInstance] = {}
    for instance in adapter.generate_instances(candidate_rows, seed):
        row, _ = adapter.build_bank_row(
            instance,
            negative_count=negatives_per_prompt,
            seed=seed,
        )
        if row is None:
            continue
        rows.append(row)
        instances[instance.prompt_id] = instance
        if len(rows) == accepted_rows:
            break
    if len(rows) != accepted_rows:
        raise RuntimeError(
            f"{adapter.name} produced {len(rows)} accepted rows; expected {accepted_rows}"
        )
    return rows, instances


def split_bank(
    rows: Sequence[Mapping[str, Any]],
    *,
    train_rows: int,
    validation_rows: int,
    test_rows: int,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    if len(rows) != train_rows + validation_rows + test_rows:
        raise ValueError("bank size must equal train+validation+test")
    task = str(rows[0]["task"])
    ordered = sorted(
        (dict(row) for row in rows),
        key=lambda row: stable_hash(
            {"task": task, "prompt_id": str(row["prompt_id"]), "split_seed": int(seed)}
        ),
    )
    validation_end = train_rows + validation_rows
    return {
        "train": ordered[:train_rows],
        "validation": ordered[train_rows:validation_end],
        "test": ordered[validation_end:],
    }


def evaluate_outputs(
    adapter: TaskAdapter,
    instances: Mapping[str, TaskInstance],
    rows: Sequence[Mapping[str, Any]],
    greedy_outputs: Sequence[str],
    sampled_outputs: Sequence[Sequence[str]],
) -> dict[str, Any]:
    if not rows or len(rows) != len(greedy_outputs) or len(rows) != len(sampled_outputs):
        raise ValueError("evaluation rows and generated outputs must align")
    greedy_correct = []
    greedy_valid = []
    pass_k = []
    sampled_valid = []
    k_values = {len(samples) for samples in sampled_outputs}
    if len(k_values) != 1 or 0 in k_values:
        raise ValueError("sample count must be one positive constant")
    for row, greedy, samples in zip(rows, greedy_outputs, sampled_outputs, strict=True):
        instance = instances[str(row["prompt_id"])]
        greedy_result = adapter.verify(instance, greedy)
        sample_results = [adapter.verify(instance, value) for value in samples]
        greedy_correct.append(greedy_result.correct)
        greedy_valid.append(greedy_result.format_valid)
        pass_k.append(any(value.correct for value in sample_results))
        sampled_valid.extend(value.format_valid for value in sample_results)
    return {
        "examples": len(rows),
        "pass_k": k_values.pop(),
        "greedy_success": float(np.mean(greedy_correct)),
        "greedy_valid_rate": float(np.mean(greedy_valid)),
        "pass_at_k": float(np.mean(pass_k)),
        "sampled_valid_rate": float(np.mean(sampled_valid)),
    }


__all__ = [
    "REASONING_GYM_COMMIT",
    "REASONING_GYM_TASKS",
    "STRUCTURED_GENERATION_METHODS",
    "STRUCTURED_GENERATION_SYSTEM_PROMPT",
    "TASK_NAMES",
    "WIKISQL_COMMIT",
    "CountdownAdapter",
    "EncodedCompletion",
    "ReasoningGymAdapter",
    "StructuredTrainingItem",
    "TaskAdapter",
    "TaskInstance",
    "VerificationResult",
    "WikiSQLAdapter",
    "asymre_objective",
    "build_adapters",
    "build_task_bank",
    "clean_expression",
    "collate_training_items",
    "completion_statistics_from_logits",
    "completion_stats",
    "dpo_objective",
    "drpo_objective",
    "drpo_weights",
    "encode_prompt_completion",
    "encode_training_row",
    "evaluate_outputs",
    "move_tensor_batch_to_device",
    "positive_only_objective",
    "prompt_balanced_mean",
    "split_bank",
    "topr_policy_objective",
    "topr_reference_objective",
    "verify_expression",
]
