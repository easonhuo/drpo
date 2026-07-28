"""Task adapters for the model-independent E8 multitask P0 replay bank.

The adapters deliberately separate frozen task construction from policy-relative
diagnostics.  A bank row contains an oracle, verifier outcomes, structured error
classes, and model-independent mutations.  It never contains near/far labels,
surprisal, policy probabilities, or taper weights.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import random
import re
import sys
import types
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, ClassVar

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


@lru_cache(maxsize=1)
def countdown_modules() -> tuple[Any, Any]:
    """Load the existing Countdown implementation only when that adapter is used."""
    from drpo import countdown_e8_oracle_bank_v2 as countdown_bank
    from drpo import countdown_qwen_arena_onefile as countdown

    return countdown, countdown_bank


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
    source_revision = "countdown_qwen_arena_onefile"
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
            details={"official_scorer": self.name},
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
            yield Mutation(str(answer + offset), "numeric_offset")
            if answer - offset >= 0:
                yield Mutation(str(answer - offset), "numeric_offset")
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
