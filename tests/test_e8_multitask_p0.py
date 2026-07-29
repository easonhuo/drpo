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


def test_exp_tuning_matrix_has_one_positive_and_seven_exp_per_task() -> None:
    from drpo import e8_multitask_exp_tuning as exp_tuning

    config = exp_tuning.load_config(Path("configs/e8_multitask_exp_tuning.yaml"))
    cells = exp_tuning.build_cells(config)
    waves = exp_tuning.build_waves(config)

    assert len(cells) == 72
    assert len({cell.key for cell in cells}) == 72
    assert sum(cell.method == exp_tuning.METHOD_POSITIVE_ONLY for cell in cells) == 9
    assert sum(cell.method == exp_tuning.METHOD_EXPONENTIAL for cell in cells) == 63
    assert sum(cell.stage == "coarse" for cell in cells) == 45
    assert sum(cell.stage == "refinement" for cell in cells) == 27
    assert [len(wave) for wave in waves] == [16, 16, 13, 16, 11]
    assert all(cell.stage == "coarse" for wave in waves[:3] for cell in wave)
    assert all(cell.stage == "refinement" for wave in waves[3:] for cell in wave)

    for task in config["suite"]["tasks"]:
        task_cells = [cell for cell in cells if cell.task == task]
        assert sum(cell.method == exp_tuning.METHOD_POSITIVE_ONLY for cell in task_cells) == 1
        assert {
            cell.rho for cell in task_cells if cell.method == exp_tuning.METHOD_EXPONENTIAL
        } == {0.9, 0.75, 0.6, 0.5, 0.35, 0.25, 0.125}


def test_exp_tuning_config_rejects_matrix_or_budget_drift() -> None:
    from drpo import e8_multitask_exp_tuning as exp_tuning

    config = exp_tuning.load_config(Path("configs/e8_multitask_exp_tuning.yaml"))
    changed = json.loads(json.dumps(config))
    changed["sweep"]["refinement_rho"] = [0.8, 0.5, 0.25]
    with pytest.raises(ValueError, match="refinement"):
        exp_tuning.validate_config(changed)

    changed = json.loads(json.dumps(config))
    changed["remoteness_calibration"]["target_negative_to_positive_gradient_ratio"] = 0.1
    with pytest.raises(ValueError, match="1/32"):
        exp_tuning.validate_config(changed)

    changed = json.loads(json.dumps(config))
    changed["execution"]["slots_per_gpu"] = 1
    with pytest.raises(ValueError, match="two slots"):
        exp_tuning.validate_config(changed)


def test_exp_tuning_p0_split_is_deterministic_and_disjoint() -> None:
    from drpo import e8_multitask_exp_tuning as exp_tuning

    config = exp_tuning.load_config(Path("configs/e8_multitask_exp_tuning.yaml"))
    config = json.loads(json.dumps(config))
    config["split"].update({"p0_train_rows": 5, "p0_validation_rows": 2, "p0_test_rows": 1})
    rows = [
        {
            "prompt_id": f"p{index}",
            "prompt": f"prompt {index}",
            "oracle_completion": f"answer {index}",
            "negatives": [
                {
                    "negative_id": f"p{index}-n{negative}",
                    "completion": f"wrong {index} {negative}",
                    "binary_correct": False,
                }
                for negative in range(16)
            ],
        }
        for index in range(8)
    ]

    first = exp_tuning.split_p0_rows(rows, task="word_sorting", config=config)
    second = exp_tuning.split_p0_rows(rows, task="word_sorting", config=config)
    assert first == second
    assert {name: len(values) for name, values in first.items()} == {
        "train": 5,
        "validation": 2,
        "test": 1,
    }
    prompt_sets = {name: {row["prompt_id"] for row in values} for name, values in first.items()}
    assert not prompt_sets["train"] & prompt_sets["validation"]
    assert not prompt_sets["train"] & prompt_sets["test"]
    assert not prompt_sets["validation"] & prompt_sets["test"]


def test_exp_tuning_countdown_normalization_preserves_frozen_split() -> None:
    from drpo import e8_multitask_exp_tuning as exp_tuning

    config = exp_tuning.load_config(Path("configs/e8_multitask_exp_tuning.yaml"))
    config = json.loads(json.dumps(config))
    config["split"].update({"countdown_train_rows": 2, "countdown_validation_rows": 1})
    train_rows = [
        {
            "row_id": f"train-{index}",
            "prompt": f"use numbers {index}",
            "oracle_positive": "1 + 2",
            "numbers": [1, 2, 3, 4],
            "target": 3,
            "negatives": [
                {
                    "expression": f"1 + 2 + {negative}",
                    "valid_format": True,
                    "correct": False,
                    "negative_bin": "near_value_wrong",
                }
                for negative in range(16)
            ],
        }
        for index in range(3)
    ]
    validation_rows = [
        {
            "id": "validation-0",
            "prompt": "use validation numbers",
            "oracle": "1 + 2",
            "numbers": [1, 2, 3, 4],
            "target": 3,
        }
    ]
    partitions = exp_tuning.split_countdown_rows(
        train_rows,
        validation_rows,
        config=config,
    )
    assert len(partitions["train"]) == 2
    assert len(partitions["validation"]) == 1
    assert partitions["validation"][0]["source_schema"] == ("countdown_structural_validation")
    assert all(len(row["negatives"]) == 16 for row in partitions["train"])
    assert not {row["prompt_id"] for row in partitions["train"]} & {
        row["prompt_id"] for row in partitions["validation"]
    }


@pytest.mark.skipif(torch is None, reason="Torch is unavailable in the test runtime")
def test_exp_tuning_distance_and_rho_parameterization() -> None:
    from drpo import e8_multitask_exp_tuning as exp_tuning

    sequence_lp = torch.tensor([-1.0, -2.0, -5.0], requires_grad=True)
    distance = exp_tuning.normalized_distance(sequence_lp, tau=1.0, scale=4.0)
    assert distance.tolist() == pytest.approx([0.0, 0.5, 1.0])
    weights = exp_tuning.taper_weight(distance, rho=0.25)
    assert weights.tolist() == pytest.approx([1.0, 0.5, 0.25])
    assert not distance.requires_grad


def test_exp_tuning_aggregate_selects_declared_late_window_winner(tmp_path: Path) -> None:
    from drpo import e8_multitask_exp_tuning as exp_tuning

    config = exp_tuning.load_config(Path("configs/e8_multitask_exp_tuning.yaml"))
    for cell in exp_tuning.build_cells(config):
        if cell.method == exp_tuning.METHOD_POSITIVE_ONLY:
            late = 0.20
        else:
            late = 0.30 - abs(float(cell.rho) - 0.5)
        manifest = {
            "complete": True,
            "evaluation_status": "complete",
            "validation_late_window_pass8_mean": late,
            "validation_terminal_pass8": late - 0.01,
            "validation_late_window_greedy_mean": late - 0.02,
            "validation_terminal_greedy": late - 0.03,
            "validation_terminal_greedy_valid_rate": 0.99,
            "nan_inf_failure": False,
        }
        p0.atomic_json(
            tmp_path / "cells" / cell.key / "cell_manifest.json",
            manifest,
        )

    summary = exp_tuning.cmd_aggregate(config, tmp_path)
    assert summary["cell_count"] == 72
    for task_summary in summary["tasks"].values():
        assert task_summary["selected_exp"]["rho"] == pytest.approx(0.5)
        assert not task_summary["strong_taper_boundary_unclosed"]
        assert not task_summary["all_exp_below_positive_only"]


def test_multitask_baseline_tuning_has_exact_80_cell_five_wave_matrix() -> None:
    from drpo import e8_multitask_baseline_tuning as baseline

    config, exp_config = baseline.load_config(
        Path("configs/e8_multitask_topr_asymre_tuning.yaml"),
        Path("configs/e8_multitask_exp_tuning.yaml"),
    )
    cells = baseline.build_cells(config, exp_config)
    waves = baseline.build_waves(config, exp_config)

    assert len(cells) == 80
    assert len({cell.key for cell in cells}) == 80
    assert [len(wave) for wave in waves] == [16, 16, 16, 16, 16]
    assert sum(cell.method == baseline.METHOD_TOPR for cell in cells) == 40
    assert sum(cell.method == baseline.METHOD_ASYMRE for cell in cells) == 40
    assert "countdown" not in {cell.task for cell in cells}
    for task in config["suite"]["tasks"]:
        task_cells = [cell for cell in cells if cell.task == task]
        assert {cell.parameter for cell in task_cells if cell.method == baseline.METHOD_TOPR} == {
            0.0,
            0.04,
            0.08,
            0.25,
            0.5,
        }
        assert {cell.parameter for cell in task_cells if cell.method == baseline.METHOD_ASYMRE} == {
            -1.0,
            -0.9,
            -0.7,
            -0.5,
            0.0,
        }


def test_multitask_baseline_tuning_rejects_grid_or_reference_drift() -> None:
    from drpo import e8_multitask_baseline_tuning as baseline

    config, exp_config = baseline.load_config(
        Path("configs/e8_multitask_topr_asymre_tuning.yaml"),
        Path("configs/e8_multitask_exp_tuning.yaml"),
    )
    changed = json.loads(json.dumps(config))
    changed["sweep"]["topr_beta"] = [0.0, 0.04, 0.08, 0.25, 0.75]
    with pytest.raises(ValueError, match="TOPR beta"):
        baseline.validate_config(changed, exp_config)

    changed = json.loads(json.dumps(config))
    changed["shared_reference"]["validation_rows_seen"] = 1
    with pytest.raises(ValueError, match="validation or test"):
        baseline.validate_config(changed, exp_config)

    changed = json.loads(json.dumps(config))
    changed["suite"]["tasks"][0] = "countdown"
    with pytest.raises(ValueError, match="exact ordered eight"):
        baseline.validate_config(changed, exp_config)


def test_multitask_baseline_aggregate_excludes_boundary_controls(tmp_path: Path) -> None:
    from drpo import e8_multitask_baseline_tuning as baseline

    config, exp_config = baseline.load_config(
        Path("configs/e8_multitask_topr_asymre_tuning.yaml"),
        Path("configs/e8_multitask_exp_tuning.yaml"),
    )
    for cell in baseline.build_cells(config, exp_config):
        if cell.method == baseline.METHOD_TOPR:
            late = 0.90 if cell.parameter == 0.0 else 0.50 - abs(cell.parameter - 0.25)
        else:
            late = 0.90 if cell.parameter == -1.0 else 0.50 - abs(cell.parameter + 0.5)
        p0.atomic_json(
            tmp_path / "cells" / cell.key / "cell_manifest.json",
            {
                "complete": True,
                "evaluation_status": "complete",
                "validation_late_window_pass8_mean": late,
                "validation_terminal_pass8": late - 0.01,
                "validation_late_window_greedy_mean": late - 0.02,
                "validation_terminal_greedy": late - 0.03,
                "validation_terminal_greedy_valid_rate": 0.99,
                "nan_inf_failure": False,
            },
        )

    summary = baseline.cmd_aggregate(config, exp_config, tmp_path)
    assert summary["cell_count"] == 80
    for methods in summary["tasks"].values():
        topr = methods[baseline.METHOD_TOPR]
        asymre = methods[baseline.METHOD_ASYMRE]
        assert topr["selected_active"]["parameter"] == pytest.approx(0.25)
        assert asymre["selected_active"]["parameter"] == pytest.approx(-0.5)
        assert topr["boundary"]["parameter"] == pytest.approx(0.0)
        assert asymre["boundary"]["parameter"] == pytest.approx(-1.0)
        assert topr["all_active_below_boundary"]
        assert asymre["all_active_below_boundary"]


@pytest.mark.skipif(torch is None, reason="Torch is unavailable in the test runtime")
def test_multitask_baseline_rng_pairing_and_joint_step_rollback() -> None:
    from drpo import e8_multitask_baseline_tuning as baseline

    torch.manual_seed(123)
    state = baseline._capture_rng_state()
    first = torch.rand(4)
    baseline._restore_rng_state(state)
    second = torch.rand(4)
    assert torch.equal(first, second)

    policy = torch.nn.Parameter(torch.tensor([1.0]))
    reference = torch.nn.Parameter(torch.tensor([2.0]))
    policy_optimizer = torch.optim.SGD([policy], lr=1.0)
    reference_optimizer = torch.optim.SGD([reference], lr=1.0)
    policy.grad = torch.tensor([float("inf")])
    reference.grad = torch.tensor([1.0])
    passed = baseline._joint_optimizer_step_with_finite_guard(
        policy_optimizer,
        reference_optimizer,
        [policy],
        [reference],
    )
    assert not passed
    assert policy.item() == pytest.approx(1.0)
    assert reference.item() == pytest.approx(2.0)
