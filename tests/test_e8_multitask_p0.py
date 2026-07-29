from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

try:
    import torch
except ImportError:
    torch = None

from drpo import e8_multitask_p0 as p0
from drpo.e8_multitask_tasks import (
    REASONING_GYM_COMMIT,
    WIKISQL_COMMIT,
    CountdownAdapter,
    ReasoningGymAdapter,
    TaskInstance,
    WikiSQLAdapter,
)


def small_config() -> dict:
    config = p0.with_smoke_overrides(
        p0.load_config(Path("configs/e8_multitask_p0.yaml")),
        rows=2,
        negatives=4,
    )
    config["aggregation"]["quantile_bins"] = 2
    config["aggregation"]["bootstrap_replicates"] = 20
    config["diagnostics"]["points_per_task"] = 4
    return config


def qualified_row(task: str, prompt_id: str, *, one_class: bool = False) -> dict:
    classes = (
        ["path_length_overestimate"] * 4
        if one_class
        else ["path_length_underestimate", "path_length_overestimate"] * 2
    )
    return {
        "task": task,
        "prompt_id": prompt_id,
        "oracle_verification": {"correct": True},
        "negatives": [
            {
                "negative_id": f"{prompt_id}_{index}",
                "canonical_completion": f"wrong-{index}",
                "format_valid": True,
                "binary_correct": False,
                "error_class": error_class,
                "response_chars": 7,
            }
            for index, error_class in enumerate(classes)
        ],
    }


def test_config_pins_and_frozen_task_list() -> None:
    config = p0.load_config(Path("configs/e8_multitask_p0.yaml"))
    assert config["sources"]["reasoning_gym"]["commit"] == REASONING_GYM_COMMIT
    assert config["sources"]["wikisql"]["commit"] == WIKISQL_COMMIT
    assert config["tasks"]["names"] == [
        "word_sorting",
        "spiral_matrix",
        "mini_sudoku",
        "maze",
        "word_ladder",
        "knights_knaves",
        "graph_color",
        "wikisql",
    ]

    assert config["positive_warmstart"]["optimizer_updates"] == 100
    assert config["positive_warmstart"]["checkpoint_kind"] == ("task_positive_warmstart_100")


@pytest.mark.skipif(torch is None, reason="Countdown dependency stack requires Torch")
def test_countdown_bank_is_model_independent_and_diverse() -> None:
    adapter = CountdownAdapter()
    instance = next(iter(adapter.generate_instances(1, seed=13)))
    row, audit = adapter.build_bank_row(instance, negative_count=4, seed=13)
    assert row is not None, audit
    assert row["oracle_verification"]["correct"]
    assert len(row["negatives"]) == 4
    assert len({item["canonical_completion"] for item in row["negatives"]}) == 4
    assert len({item["error_class"] for item in row["negatives"]}) >= 2
    serialized = json.dumps(row, sort_keys=True)
    for forbidden in ("reference_surprisal", "near_negative", "far_negative", "taper_weight"):
        assert forbidden not in serialized


class FakeMazeDataset:
    def score_answer(self, completion, source_entry):
        return float(int(completion) == int(source_entry["answer"]))


def fake_maze_adapter_and_instance() -> tuple[ReasoningGymAdapter, TaskInstance]:
    adapter = object.__new__(ReasoningGymAdapter)
    adapter.name = "maze"
    adapter.output_structure = "single_integer_shortest_path_length"
    adapter.dataset = FakeMazeDataset()
    instance = TaskInstance(
        task="maze",
        prompt_id="maze-0",
        prompt="Return the shortest path length.",
        oracle_completion="12",
        metadata={"shortest_path_length": 12},
        source_entry={"answer": "12"},
    )
    return adapter, instance


def test_maze_verifier_uses_signed_residual_error_taxonomy() -> None:
    adapter, instance = fake_maze_adapter_and_instance()
    underestimate = adapter.verify(instance, "10", mutation_class="numeric_offset")
    overestimate = adapter.verify(instance, "15", mutation_class="numeric_offset")
    assert underestimate.error_class == "path_length_underestimate"
    assert underestimate.details["signed_residual"] == -2
    assert overestimate.error_class == "path_length_overestimate"
    assert overestimate.details["signed_residual"] == 3


def test_maze_bank_contains_two_scalar_error_classes() -> None:
    adapter, instance = fake_maze_adapter_and_instance()
    row, audit = adapter.build_bank_row(instance, negative_count=16, seed=19)
    assert row is not None, audit
    assert {item["error_class"] for item in row["negatives"]} == {
        "path_length_underestimate",
        "path_length_overestimate",
    }
    assert len({item["canonical_completion"] for item in row["negatives"]}) == 16


def test_qualification_rejects_official_scalar_shape_with_one_error_class() -> None:
    config = small_config()
    rows = [
        qualified_row("maze", "maze-0", one_class=True),
        qualified_row("maze", "maze-1", one_class=True),
    ]
    audit = p0.qualify_task("maze", rows, config)
    assert not audit["passed"]
    assert not audit["gates"]["prompt_pass_fraction"]
    assert not audit["gates"]["task_error_class_count"]


def test_qualification_accepts_two_sided_scalar_residual_classes() -> None:
    config = small_config()
    rows = [
        qualified_row("maze", "maze-0"),
        qualified_row("maze", "maze-1"),
    ]
    audit = p0.qualify_task("maze", rows, config)
    assert audit["passed"]
    assert all(audit["gates"].values())


def test_oracle_nll_audit_selection_is_deterministic() -> None:
    rows = [{"prompt_id": f"p{index}"} for index in range(20)]
    first = p0.select_oracle_nll_audit_rows(rows, task="maze", limit=5, seed=11)
    second = p0.select_oracle_nll_audit_rows(rows, task="maze", limit=5, seed=11)
    assert first == second


def test_qualification_accepts_structured_verified_bank() -> None:
    config = small_config()
    rows = [
        qualified_row("word_sorting", "word-0"),
        qualified_row("word_sorting", "word-1"),
    ]
    audit = p0.qualify_task("word_sorting", rows, config)
    assert audit["passed"]
    assert all(audit["gates"].values())
    assert audit["metrics"]["oracle_verification_rate"] == 1.0


def test_diagnostic_selection_is_prompt_balanced() -> None:
    rows = []
    for prompt in range(3):
        rows.append(
            {
                "prompt_id": f"p{prompt}",
                "prompt": f"prompt {prompt}",
                "negatives": [
                    {
                        "negative_id": f"p{prompt}-n{negative}",
                        "completion": f"negative {negative}",
                        "error_class": "wrong",
                        "verifier_score": 0.0,
                        "response_chars": 10,
                    }
                    for negative in range(3)
                ],
            }
        )
    points = p0.select_diagnostic_points(rows, task="task", limit=6, seed=9)
    counts: dict[str, int] = {}
    for point in points:
        counts[str(point["prompt_id"])] = counts.get(str(point["prompt_id"]), 0) + 1
    assert counts == {"p0": 2, "p1": 2, "p2": 2}


class TinyTokenizer:
    eos_token = "~"
    eos_token_id = 63
    pad_token_id = 0

    def apply_chat_template(self, messages, **kwargs):
        del kwargs
        return "|".join(message["content"] for message in messages) + "|"

    def __call__(self, text, add_special_tokens=False):
        del add_special_tokens
        return {"input_ids": [1 + (ord(character) % 62) for character in text]}


if torch is not None:

    class TinyLM(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = torch.nn.Embedding(64, 8)
            self.projection = torch.nn.Linear(8, 64)
            self.config = SimpleNamespace(use_cache=True)

        def forward(self, input_ids, attention_mask, use_cache=False):
            del attention_mask, use_cache
            return SimpleNamespace(logits=self.projection(self.embedding(input_ids)))

else:

    class TinyLM:
        pass


@pytest.mark.skipif(torch is None, reason="Torch is unavailable in the test runtime")
def test_diagnose_point_records_full_parameter_gradient() -> None:
    model = TinyLM()
    point = {
        "task": "tiny",
        "prompt_id": "p0",
        "prompt": "sort the values",
        "negative": {
            "negative_id": "n0",
            "completion": "3,2,1",
            "error_class": "order_error",
            "verifier_score": 0.0,
            "response_chars": 5,
        },
    }
    result = p0.diagnose_point(
        model,
        TinyTokenizer(),
        point,
        max_length=128,
        matched_absolute_advantage=1.0,
    )
    assert result["mean_token_surprisal"] > 0
    assert result["raw_full_parameter_gradient_norm"] > 0
    assert result["implemented_actor_gradient_norm"] == result["raw_full_parameter_gradient_norm"]
    assert result["gradient_parameter_count"] == result["parameter_count"]


def test_diagnose_refuses_unqualified_task_before_model_load(tmp_path: Path) -> None:
    config = small_config()
    config["tasks"]["names"] = ["maze"]
    p0.atomic_json(
        tmp_path / "qualification_audit.json",
        {"tasks": {"maze": {"passed": False}}},
    )
    with pytest.raises(RuntimeError, match="unqualified"):
        p0.cmd_diagnose(
            config,
            tmp_path,
            model_path="not-loaded",
            adapter_path=None,
        )


def test_diagnostic_resume_identity_is_fail_closed(tmp_path: Path) -> None:
    config = small_config()
    config["tasks"]["names"] = ["word_sorting"]
    p0.atomic_json(
        tmp_path / "qualification_audit.json",
        {"tasks": {"word_sorting": {"passed": True}}},
    )
    p0.atomic_jsonl(
        p0.bank_path(tmp_path, "word_sorting"),
        [
            {
                "prompt_id": "p0",
                "prompt": "sort",
                "negatives": [
                    {
                        "negative_id": f"n{index}",
                        "completion": str(index),
                        "error_class": "wrong",
                        "verifier_score": 0.0,
                        "response_chars": 1,
                    }
                    for index in range(4)
                ],
            }
        ],
    )
    p0.atomic_json(
        tmp_path / "diagnostics" / "diagnostic_identity.json",
        {"identity_hash": "wrong"},
    )
    with pytest.raises(RuntimeError, match="identity mismatch"):
        p0.cmd_diagnose(
            config,
            tmp_path,
            model_path="tiny",
            adapter_path="tiny-adapter",
            model_and_tokenizer=None,
        )


def test_task_equal_aggregate_does_not_weight_large_task_more(tmp_path: Path) -> None:
    config = small_config()
    config["tasks"]["names"] = ["small", "large"]
    config["aggregation"]["quantile_bins"] = 2
    config["aggregation"]["bootstrap_replicates"] = 10
    for task, scale, count in (("small", 1.0, 4), ("large", 10.0, 20)):
        rows = []
        for index in range(count):
            high = index >= count // 2
            rows.append(
                {
                    "prompt_id": f"{task}-p{index}",
                    "negative_id": f"{task}-n{index}",
                    "mean_token_surprisal": float(index),
                    "implemented_actor_gradient_norm": scale * (3.0 if high else 1.0),
                    "matched_absolute_advantage": 1.0,
                    "mean_direct_logit_score_norm": 1.0,
                    "response_tokens": 4,
                }
            )
        p0.atomic_jsonl(tmp_path / "diagnostics" / f"{task}.jsonl", rows)
    p0.cmd_aggregate(config, tmp_path)
    aggregate = list(csv.DictReader((tmp_path / "aggregate" / "task_equal_aggregate.csv").open()))
    assert float(aggregate[0]["relative_implemented_actor_gradient"]) == pytest.approx(1.0)
    assert float(aggregate[1]["relative_implemented_actor_gradient"]) == pytest.approx(3.0)


def test_all_stops_before_model_when_any_task_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = small_config()
    monkeypatch.setattr(p0, "cmd_prepare", lambda *args, **kwargs: {"complete": True})
    monkeypatch.setattr(
        p0,
        "cmd_qualify",
        lambda *args, **kwargs: {
            "passed": False,
            "failed_tasks": ["maze"],
        },
    )
    called = False

    def forbidden_loader(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("model loader must not run")

    monkeypatch.setattr(p0, "load_diagnostic_model", forbidden_loader)
    with pytest.raises(RuntimeError, match="before model loading"):
        p0.cmd_all(
            config,
            tmp_path,
            force=False,
            skip_download=True,
            model_path="unused",
            adapter_path=None,
        )
    assert not called


def test_all_runs_warmstart_before_diagnose(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = small_config()
    order: list[str] = []
    monkeypatch.setattr(
        p0,
        "cmd_prepare",
        lambda *args, **kwargs: order.append("prepare") or {"complete": True},
    )
    monkeypatch.setattr(
        p0,
        "cmd_qualify",
        lambda *args, **kwargs: order.append("qualify") or {"passed": True, "failed_tasks": []},
    )
    monkeypatch.setattr(
        p0,
        "cmd_warmstart",
        lambda *args, **kwargs: order.append("warmstart") or {"complete": True},
    )
    monkeypatch.setattr(
        p0,
        "cmd_diagnose",
        lambda *args, **kwargs: order.append("diagnose") or {"complete": True},
    )
    monkeypatch.setattr(
        p0,
        "cmd_aggregate",
        lambda *args, **kwargs: order.append("aggregate") or {"complete": True},
    )
    result = p0.cmd_all(
        config,
        tmp_path,
        force=False,
        skip_download=True,
        model_path="base-model",
        adapter_path=None,
    )
    assert order == ["prepare", "qualify", "warmstart", "diagnose", "aggregate"]
    assert result["warmstart"]["complete"]


def test_all_rejects_supplied_adapter_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = small_config()
    monkeypatch.setattr(p0, "cmd_prepare", lambda *args, **kwargs: {"complete": True})
    monkeypatch.setattr(
        p0,
        "cmd_qualify",
        lambda *args, **kwargs: {"passed": True, "failed_tasks": []},
    )
    with pytest.raises(ValueError, match="diagnose-only"):
        p0.cmd_all(
            config,
            tmp_path,
            force=False,
            skip_download=True,
            model_path="base-model",
            adapter_path="shared-adapter",
        )


def test_wikisql_official_logical_form_verifier_and_mutations(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    table = {
        "id": "t1",
        "header": ["year", "maker"],
        "types": ["real", "text"],
        "rows": [[1998, "A"], [1999, "B"]],
    }
    example = {
        "question": "Who made the 1998 item?",
        "table_id": "t1",
        "sql": {"sel": 1, "agg": 0, "conds": [[0, 0, "1998"]]},
    }
    p0.atomic_jsonl(data / "train.tables.jsonl", [table])
    p0.atomic_jsonl(data / "train.jsonl", [example])
    adapter = WikiSQLAdapter(tmp_path)
    instance = next(iter(adapter.generate_instances(1, seed=0)))
    assert adapter.verify(instance, instance.oracle_completion).correct
    numeric_value_variant = json.dumps({"sel": 1, "agg": 0, "conds": [[0, 0, 1998]]})
    assert adapter.verify(instance, numeric_value_variant).correct
    row, audit = adapter.build_bank_row(instance, negative_count=4, seed=0)
    assert row is not None, audit
    assert len({item["error_class"] for item in row["negatives"]}) >= 2
