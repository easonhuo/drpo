from __future__ import annotations

import csv
import json
import math
import subprocess
import threading
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

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


def test_exp_dense_matrix_is_seven_task_local_sixteen_cell_waves() -> None:
    from drpo import e8_multitask_exp_tuning as exp_tuning

    config = exp_tuning.load_config(Path("configs/e8_multitask_exp_lambda_dense.yaml"))
    cells = exp_tuning.build_cells(config)
    waves = exp_tuning.build_waves(config)

    assert len(cells) == 112
    assert len({cell.key for cell in cells}) == 112
    assert {cell.task for cell in cells} == set(exp_tuning.TASK_NAMES) - {
        "countdown",
        "spiral_matrix",
    }
    assert all(cell.method == exp_tuning.METHOD_EXPONENTIAL for cell in cells)
    assert all(cell.lambda_value is not None for cell in cells)
    assert [len(wave) for wave in waves] == [16] * 7
    assert all(len({cell.task for cell in wave}) == 1 for wave in waves)
    predecessor_lambdas = {-math.log(rho) for rho in (0.9, 0.75, 0.6, 0.5, 0.35, 0.25, 0.125)}
    for task in config["suite"]["tasks"]:
        task_cells = [cell for cell in cells if cell.task == task]
        lambdas = {float(cell.lambda_value) for cell in task_cells}
        bridge = float(config["sweep"]["bridge_lambda"][task])
        assert len(lambdas) == 16
        assert bridge in lambdas
        assert lambdas & predecessor_lambdas == {bridge}


def test_exp_dense_config_rejects_task_grid_or_bridge_drift() -> None:
    from drpo import e8_multitask_exp_tuning as exp_tuning

    config = exp_tuning.load_config(Path("configs/e8_multitask_exp_lambda_dense.yaml"))
    changed = json.loads(json.dumps(config))
    changed["sweep"]["task_lambda"]["maze"][-1] = changed["sweep"]["task_lambda"]["maze"][-2]
    with pytest.raises(ValueError, match="16 unique"):
        exp_tuning.validate_config(changed)

    changed = json.loads(json.dumps(config))
    changed["sweep"]["bridge_lambda"]["wikisql"] = 9.0
    with pytest.raises(ValueError, match="bridge lambda"):
        exp_tuning.validate_config(changed)

    changed = json.loads(json.dumps(config))
    changed["execution"]["expected_waves"] = 6
    with pytest.raises(ValueError, match="7 waves"):
        exp_tuning.validate_config(changed)


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

    assert config["training"]["micro_batch"] == 1
    assert config["training"]["gradient_accumulation"] == 8
    reference_config = exp_tuning._reference_warmstart_config(
        config,
        Path("configs/e8_multitask_p0.yaml"),
    )
    assert reference_config["micro_batch"] == 2
    assert reference_config["gradient_accumulation"] == 32

    changed = json.loads(json.dumps(config))
    changed["training"].update({"micro_batch": 2, "gradient_accumulation": 32})
    with pytest.raises(ValueError, match="method-training effective prompt batch"):
        exp_tuning.validate_config(changed)


def test_exp_tuning_rejects_qualification_from_another_p0_config(tmp_path: Path) -> None:
    from drpo import e8_multitask_exp_tuning as exp_tuning

    config = exp_tuning.load_config(Path("configs/e8_multitask_exp_tuning.yaml"))
    p0.atomic_json(
        tmp_path / "qualification_audit.json",
        {
            "experiment_id": exp_tuning.PARENT_EXPERIMENT_ID,
            "config_hash": "stale-p0-config",
            "passed": True,
            "tasks": {task: {"passed": True} for task in config["suite"]["p0_tasks"]},
        },
    )
    with pytest.raises(RuntimeError, match="qualification identity"):
        exp_tuning.resolve_task_inputs(
            config,
            p0_work_dir=tmp_path,
            p0_config=Path("configs/e8_multitask_p0.yaml").resolve(),
            countdown_bank=tmp_path / "countdown-bank.jsonl",
            countdown_validation=tmp_path / "countdown-validation.jsonl",
            countdown_adapter=tmp_path / "countdown-adapter",
        )


def test_exp_tuning_builds_only_train_split_references_and_rejects_leakage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from drpo import e8_multitask_exp_tuning as exp_tuning

    config = exp_tuning.load_config(Path("configs/e8_multitask_exp_tuning.yaml"))
    config = json.loads(json.dumps(config))
    config["split"]["p0_train_rows"] = 2
    p0_config = Path("configs/e8_multitask_p0.yaml").resolve()
    p0_tasks = tuple(config["suite"]["p0_tasks"])
    splits: dict[str, object] = {"tasks": {}}
    inputs: dict[str, exp_tuning.TaskInputs] = {}
    for task in p0_tasks:
        train_path = tmp_path / "splits" / task / "train.jsonl"
        p0.atomic_jsonl(
            train_path,
            [
                {
                    "prompt_id": f"{task}-train-{index}",
                    "prompt": f"{task} train prompt {index}",
                    "oracle_completion": f"{task} answer {index}",
                    "negatives": [],
                }
                for index in range(2)
            ],
        )
        splits["tasks"][task] = {
            "paths": {
                "train": str(train_path),
                "validation": str(tmp_path / "must-not-read" / task / "validation.jsonl"),
                "test": str(tmp_path / "must-not-read" / task / "test.jsonl"),
            },
            "prompt_id_hashes": {"train": f"{task}-train-hash"},
            "p0_config_sha256": exp_tuning.sha256_file(p0_config),
        }
        inputs[task] = exp_tuning.TaskInputs(
            task=task,
            bank=tmp_path / "unused-bank.jsonl",
            reference_adapter=None,
            sources_root=tmp_path,
            p0_config=p0_config,
        )
    countdown_adapter = tmp_path / "countdown-adapter"
    countdown_adapter.mkdir()
    (countdown_adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    inputs["countdown"] = exp_tuning.TaskInputs(
        task="countdown",
        bank=tmp_path / "unused-countdown-bank.jsonl",
        reference_adapter=countdown_adapter,
        sources_root=tmp_path,
        p0_config=p0_config,
    )

    monkeypatch.setattr(
        exp_tuning,
        "_load_prepared",
        lambda *args, **kwargs: (splits, inputs),
    )

    def fake_model_identity(model_path: str, adapter_path: str | None) -> dict:
        return {
            "model": {"path": model_path},
            "adapter": {"path": adapter_path},
        }

    observed: dict[str, dict[str, object]] = {}

    def fake_train_task_positive_warmstart(**kwargs: object) -> dict:
        task = str(kwargs["task"])
        rows = list(kwargs["rows"])
        output_dir = Path(str(kwargs["output_dir"]))
        adapter = output_dir / "adapter"
        adapter.mkdir()
        (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
        (adapter / "adapter_model.safetensors").write_bytes(task.encode("utf-8"))
        observed[task] = {
            "prompt_ids": [str(row["prompt_id"]) for row in rows],
            "seed": int(kwargs["seed"]),
            "warmstart_config": dict(kwargs["warmstart_config"]),
        }
        return {
            "task": task,
            "checkpoint_kind": kwargs["warmstart_config"]["checkpoint_kind"],
            "adapter_path": str(adapter),
            "adapter_identity": fake_model_identity(
                str(kwargs["model_path"]),
                str(adapter),
            )["adapter"],
            "complete": True,
            "scientific_status": "not_run",
        }

    monkeypatch.setattr(exp_tuning, "model_identity", fake_model_identity)
    monkeypatch.setattr(
        exp_tuning,
        "train_task_positive_warmstart",
        fake_train_task_positive_warmstart,
    )

    manifest = exp_tuning.cmd_reference(
        config,
        tmp_path,
        base_model_path="base-model",
        tasks=None,
        force=False,
    )
    assert manifest["complete"]
    assert set(observed) == set(p0_tasks)
    base_seed = int(p0.load_config(p0_config)["positive_warmstart"]["seed"])
    for index, task in enumerate(p0_tasks):
        assert observed[task]["prompt_ids"] == [
            f"{task}-train-0",
            f"{task}-train-1",
        ]
        assert observed[task]["seed"] == base_seed + index * 100_003
        assert observed[task]["warmstart_config"]["micro_batch"] == 2
        assert observed[task]["warmstart_config"]["gradient_accumulation"] == 32
        assert manifest["tasks"][task]["train_rows_seen"] == 2
        assert manifest["tasks"][task]["validation_rows_seen"] == 0
        assert manifest["tasks"][task]["test_rows_seen"] == 0

    attached = exp_tuning._attach_references(
        tmp_path,
        config,
        splits,
        inputs,
        base_model_path="base-model",
    )
    assert attached["countdown"].reference_adapter == countdown_adapter
    assert all(attached[task].reference_adapter is not None for task in p0_tasks)

    victim = p0_tasks[0]
    victim_manifest_path = tmp_path / "references" / victim / "task_manifest.json"
    victim_manifest = json.loads(victim_manifest_path.read_text(encoding="utf-8"))
    victim_manifest["validation_rows_seen"] = 1
    p0.atomic_json(victim_manifest_path, victim_manifest)
    top_manifest_path = exp_tuning.reference_manifest_path(tmp_path)
    top_manifest = json.loads(top_manifest_path.read_text(encoding="utf-8"))
    top_manifest["tasks"][victim] = victim_manifest
    p0.atomic_json(top_manifest_path, top_manifest)
    with pytest.raises(RuntimeError, match="leakage audit mismatch"):
        exp_tuning.cmd_reference(
            config,
            tmp_path,
            base_model_path="base-model",
            tasks=[victim],
            force=False,
        )


def test_exp_tuning_partial_calibration_preserves_prior_task_results(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from drpo import e8_multitask_exp_tuning as exp_tuning

    config = exp_tuning.load_config(Path("configs/e8_multitask_exp_tuning.yaml"))
    inputs = {
        task: SimpleNamespace(reference_adapter=tmp_path / task)
        for task in config["suite"]["tasks"]
    }
    monkeypatch.setattr(
        exp_tuning,
        "_load_ready_inputs",
        lambda *args, **kwargs: ({"tasks": {}}, inputs),
    )

    def fake_identity(task: str, **kwargs: object) -> dict:
        del kwargs
        return {"identity_hash": f"identity-{task}"}

    def fake_calibrate(task: str, **kwargs: object) -> dict:
        del kwargs
        result = {
            "experiment_id": exp_tuning.EXPERIMENT_ID,
            "config_hash": exp_tuning.stable_config_hash(config),
            "task": task,
            "identity_hash": f"identity-{task}",
            "complete": True,
        }
        p0.atomic_json(tmp_path / "calibration" / f"{task}.json", result)
        return result

    monkeypatch.setattr(exp_tuning, "_calibration_identity", fake_identity)
    monkeypatch.setattr(exp_tuning, "calibrate_task", fake_calibrate)
    first = exp_tuning.cmd_calibrate(
        config,
        tmp_path,
        base_model_path="base-model",
        tasks=["word_sorting"],
        force=False,
    )
    assert not first["complete"]
    assert set(first["tasks"]) == {"word_sorting"}

    second = exp_tuning.cmd_calibrate(
        config,
        tmp_path,
        base_model_path="base-model",
        tasks=["spiral_matrix"],
        force=False,
    )
    assert not second["complete"]
    assert set(second["tasks"]) == {"word_sorting", "spiral_matrix"}


def test_exp_tuning_wave_requires_calibration_and_liveness_gates(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from drpo import e8_multitask_exp_tuning as exp_tuning

    config = exp_tuning.load_config(Path("configs/e8_multitask_exp_tuning.yaml"))
    with pytest.raises(RuntimeError, match="calibrations"):
        exp_tuning.cmd_run_wave(
            config,
            Path("configs/e8_multitask_exp_tuning.yaml"),
            tmp_path,
            wave_index=1,
            base_model_path="base-model",
            force=False,
        )

    calibration_tasks = {
        task: {"identity_hash": f"calibration-{task}", "complete": True}
        for task in config["suite"]["tasks"]
    }
    p0.atomic_json(
        tmp_path / "calibration" / "calibration_manifest.json",
        {
            "experiment_id": exp_tuning.EXPERIMENT_ID,
            "config_hash": exp_tuning.stable_config_hash(config),
            "tasks": calibration_tasks,
            "complete": True,
        },
    )
    monkeypatch.setattr(
        exp_tuning,
        "_load_ready_inputs",
        lambda *args, **kwargs: (
            {"tasks": {}},
            {task: SimpleNamespace() for task in config["suite"]["tasks"]},
        ),
    )
    monkeypatch.setattr(
        exp_tuning,
        "_calibration_identity",
        lambda task, **kwargs: {"identity_hash": f"calibration-{task}"},
    )
    with pytest.raises(RuntimeError, match="liveness"):
        exp_tuning.cmd_run_wave(
            config,
            Path("configs/e8_multitask_exp_tuning.yaml"),
            tmp_path,
            wave_index=1,
            base_model_path="base-model",
            force=False,
        )


def test_exp_tuning_liveness_uses_fresh_process_reload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from drpo import e8_multitask_exp_tuning as exp_tuning

    config = exp_tuning.load_config(Path("configs/e8_multitask_exp_tuning.yaml"))
    task = "word_sorting"
    reference_adapter = tmp_path / "reference-adapter"
    terminal_adapter = tmp_path / "terminal-adapter"
    for adapter, payload in (
        (reference_adapter, b"reference"),
        (terminal_adapter, b"terminal"),
    ):
        adapter.mkdir()
        (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
        (adapter / "adapter_model.safetensors").write_bytes(payload)
    training_metrics = tmp_path / "training_metrics.jsonl"
    p0.atomic_jsonl(
        training_metrics,
        [
            {
                "positive_loss": 1.0,
                "negative_scalar": -0.25,
                "raw_gradient_norm_before_clip": 0.5,
            }
        ],
    )

    inputs = {
        task: exp_tuning.TaskInputs(
            task=task,
            bank=tmp_path / "bank.jsonl",
            reference_adapter=reference_adapter,
            sources_root=tmp_path,
            p0_config=Path("configs/e8_multitask_p0.yaml"),
        )
    }
    monkeypatch.setattr(
        exp_tuning,
        "_load_ready_inputs",
        lambda *args, **kwargs: ({"tasks": {}}, inputs),
    )

    def fake_model_identity(model_path: str, adapter_path: str | None) -> dict:
        return {
            "model": {"path": model_path},
            "adapter": {"path": adapter_path},
        }

    monkeypatch.setattr(exp_tuning, "model_identity", fake_model_identity)
    monkeypatch.setattr(
        exp_tuning,
        "train_cell",
        lambda *args, **kwargs: {
            "terminal_adapter": str(terminal_adapter),
            "terminal_adapter_identity": fake_model_identity(
                "base-model",
                str(terminal_adapter),
            )["adapter"],
            "training_metrics": str(training_metrics),
            "complete": True,
            "nan_inf_failure": False,
        },
    )
    observed_command: list[str] = []

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        del kwargs
        observed_command.extend(command)
        return SimpleNamespace(
            returncode=0,
            stderr="",
            stdout=json.dumps(
                {
                    "complete": True,
                    "finite": True,
                    "process_id": 999_999,
                    "base_model_identity": fake_model_identity(
                        "base-model",
                        None,
                    )["model"],
                    "adapter_identity": fake_model_identity(
                        "base-model",
                        str(terminal_adapter),
                    )["adapter"],
                }
            ),
        )

    monkeypatch.setattr(exp_tuning.subprocess, "run", fake_run)
    result = exp_tuning.cmd_liveness(
        config,
        Path("configs/e8_multitask_exp_tuning.yaml"),
        tmp_path,
        task=task,
        rho=0.5,
        base_model_path="base-model",
        force=False,
    )
    assert "reload-adapter" in observed_command
    assert result["fresh_process_reload_passed"]
    assert result["reload_process_id"] == 999_999
    assert result["adapter_weight_changed"]


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

    duplicate_rows = json.loads(json.dumps(rows))
    duplicate_rows[-1]["prompt_id"] = duplicate_rows[0]["prompt_id"]
    with pytest.raises(RuntimeError, match="duplicate prompt IDs|overlaps"):
        exp_tuning.split_p0_rows(
            duplicate_rows,
            task="word_sorting",
            config=config,
        )


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


def test_exp_tuning_duplicate_negative_allowance_is_countdown_only() -> None:
    from drpo import e8_multitask_exp_tuning as exp_tuning

    negatives = [
        {
            "negative_id": f"negative-{index}",
            "completion": "cycle-padded completion",
            "binary_correct": False,
        }
        for index in range(16)
    ]
    row = {"prompt_id": "prompt-0", "negatives": negatives}
    exp_tuning._audit_training_rows("countdown", [row], 1)
    with pytest.raises(RuntimeError, match="duplicate negative completions"):
        exp_tuning._audit_training_rows("word_sorting", [row], 1)


def test_exp_tuning_liveness_does_not_open_validation_split(tmp_path: Path) -> None:
    from drpo import e8_multitask_exp_tuning as exp_tuning

    train_path = tmp_path / "train.jsonl"
    p0.atomic_jsonl(
        train_path,
        [{"prompt_id": "train-0", "prompt": "train", "oracle_completion": "answer"}],
    )
    validation_path = tmp_path / "validation-must-not-be-opened.jsonl"
    split_manifest = {
        "tasks": {
            "word_sorting": {
                "paths": {
                    "train": str(train_path),
                    "validation": str(validation_path),
                }
            }
        }
    }
    train_rows, validation_rows = exp_tuning._load_cell_splits(
        split_manifest,
        "word_sorting",
        engineering_liveness=True,
    )
    assert [row["prompt_id"] for row in train_rows] == ["train-0"]
    assert validation_rows == []
    with pytest.raises(FileNotFoundError):
        exp_tuning._load_cell_splits(
            split_manifest,
            "word_sorting",
            engineering_liveness=False,
        )


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


def test_exp_dense_aggregate_combines_parent_anchors_and_bridge_reruns(
    tmp_path: Path,
) -> None:
    from drpo import e8_multitask_exp_tuning as exp_tuning

    config = exp_tuning.load_config(Path("configs/e8_multitask_exp_lambda_dense.yaml"))
    parent_rows: list[dict] = []
    parent_rhos = (None, 0.9, 0.75, 0.6, 0.5, 0.35, 0.25, 0.125)
    for task in config["suite"]["tasks"]:
        for index, rho in enumerate(parent_rhos):
            method = (
                exp_tuning.METHOD_POSITIVE_ONLY if rho is None else exp_tuning.METHOD_EXPONENTIAL
            )
            parent_rows.append(
                {
                    "source": "predecessor",
                    "task": task,
                    "method": method,
                    "rho": rho,
                    "lambda": None if rho is None else -math.log(rho),
                    "seed": 2026072904,
                    "cell_key": f"{task}-parent-{index}",
                    "late_window_pass8_mean": 0.40 if rho is None else 0.42,
                    "terminal_pass8": 0.41,
                    "late_window_greedy_mean": 0.39,
                    "terminal_greedy": 0.38,
                    "terminal_greedy_valid_rate": 0.99,
                    "nan_inf_failure": False,
                }
            )
    p0.atomic_json(
        tmp_path / "inherited" / "parent_response.json",
        {
            "experiment_id": exp_tuning.DENSE_EXPERIMENT_ID,
            "config_hash": exp_tuning.stable_config_hash(config),
            "rows": parent_rows,
            "complete": True,
        },
    )

    expected_selected: dict[str, float] = {}
    for task in config["suite"]["tasks"]:
        task_lambdas = [float(value) for value in config["sweep"]["task_lambda"][task]]
        target = task_lambdas[len(task_lambdas) // 2]
        expected_selected[task] = target
        for cell in [value for value in exp_tuning.build_cells(config) if value.task == task]:
            late = 0.55 - abs(float(cell.lambda_value) - target) * 0.01
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

    summary = exp_tuning.cmd_aggregate(config, tmp_path)
    assert summary["cell_count"] == 112
    assert summary["combined_response_point_count"] == 168
    assert summary["positive_only_source"] == "inherited_parent"
    for task, task_summary in summary["tasks"].items():
        assert task_summary["selected_dense_exp"]["lambda"] == pytest.approx(
            expected_selected[task]
        )
        assert not task_summary["selected_on_grid_edge"]
        assert task_summary["bridge"]["report_only"]


def test_exp_dense_inherit_pins_parent_and_rebinds_train_only_references(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from drpo import e8_multitask_exp_tuning as exp_tuning

    config = exp_tuning.load_config(Path("configs/e8_multitask_exp_lambda_dense.yaml"))
    config = json.loads(json.dumps(config))
    parent_config = exp_tuning.load_config(Path("configs/e8_multitask_exp_tuning.yaml"))
    parent_root = tmp_path / "parent"
    child_root = tmp_path / "child"
    sources_root = tmp_path / "sources"
    sources_root.mkdir()
    parent_hash = exp_tuning.stable_config_hash(parent_config)
    tasks = tuple(str(task) for task in config["suite"]["tasks"])

    parent_splits: dict[str, object] = {
        "experiment_id": exp_tuning.EXPERIMENT_ID,
        "config_hash": parent_hash,
        "tasks": {},
        "complete": True,
    }
    parent_inputs: dict[str, exp_tuning.TaskInputs] = {}
    reference_tasks: dict[str, dict] = {}
    for task in tasks:
        train_path = parent_root / "splits" / task / "train.jsonl"
        p0.atomic_jsonl(train_path, [{"prompt_id": f"{task}-train-0"}])
        bank_path = tmp_path / f"{task}-bank.jsonl"
        p0.atomic_jsonl(bank_path, [{"prompt_id": f"{task}-bank-0"}])
        p0_config_path = Path("configs/e8_multitask_p0.yaml").resolve()
        parent_splits["tasks"][task] = {
            "paths": {"train": str(train_path)},
            "prompt_id_hashes": {"train": f"{task}-train-hash"},
            "p0_config_sha256": exp_tuning.sha256_file(p0_config_path),
            "bank_sha256": exp_tuning.sha256_file(bank_path),
        }
        parent_inputs[task] = exp_tuning.TaskInputs(
            task=task,
            bank=bank_path,
            reference_adapter=None,
            sources_root=sources_root,
            p0_config=p0_config_path,
        )
        adapter = parent_root / "references" / task / "adapter"
        adapter.mkdir(parents=True)
        (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
        (adapter / "adapter_model.safetensors").write_bytes(task.encode("utf-8"))
        task_manifest = {
            "experiment_id": exp_tuning.EXPERIMENT_ID,
            "task": task,
            "identity_hash": f"parent-{task}",
            "checkpoint_kind": config["reference"]["checkpoint_kind"],
            "adapter_path": str(adapter),
            "adapter_identity": {"path": str(adapter)},
            "complete": True,
            "train_only_reference": True,
            "train_rows_seen": 5000,
            "validation_rows_seen": 0,
            "test_rows_seen": 0,
        }
        p0.atomic_json(
            parent_root / "references" / task / "task_manifest.json",
            task_manifest,
        )
        reference_tasks[task] = task_manifest

    p0.atomic_json(
        parent_root / "plan.json",
        {
            "experiment_id": exp_tuning.EXPERIMENT_ID,
            "config_hash": parent_hash,
            "cell_count": 72,
        },
    )
    p0.atomic_json(
        parent_root / "aggregate" / "aggregate_summary.json",
        {
            "experiment_id": exp_tuning.EXPERIMENT_ID,
            "cell_count": 72,
            "test_partition_accessed": False,
        },
    )
    p0.atomic_json(
        exp_tuning.reference_manifest_path(parent_root),
        {
            "experiment_id": exp_tuning.EXPERIMENT_ID,
            "config_hash": parent_hash,
            "checkpoint_kind": config["reference"]["checkpoint_kind"],
            "tasks": reference_tasks,
            "complete": True,
            "validation_rows_seen": 0,
            "test_rows_seen": 0,
        },
    )
    for cell in exp_tuning.build_cells(parent_config):
        if cell.task not in tasks:
            continue
        p0.atomic_json(
            parent_root / "cells" / cell.key / "cell_manifest.json",
            {
                "experiment_id": exp_tuning.EXPERIMENT_ID,
                "config_hash": parent_hash,
                "complete": True,
                "evaluation_status": "complete",
                "validation_late_window_pass8_mean": 0.5,
                "validation_terminal_pass8": 0.49,
                "validation_late_window_greedy_mean": 0.48,
                "validation_terminal_greedy": 0.47,
                "validation_terminal_greedy_valid_rate": 0.99,
                "nan_inf_failure": False,
            },
        )

    config["parent"]["config_hash"] = parent_hash
    artifact_paths = {
        "plan": parent_root / "plan.json",
        "split_manifest": parent_root / "split_manifest.json",
        "reference_manifest": exp_tuning.reference_manifest_path(parent_root),
        "aggregate_summary": parent_root / "aggregate" / "aggregate_summary.json",
    }
    p0.atomic_json(parent_root / "split_manifest.json", parent_splits)
    config["parent"]["artifact_sha256"] = {
        name: exp_tuning.sha256_file(path) for name, path in artifact_paths.items()
    }

    calls: list[Path] = []

    def fake_load_ready_inputs(
        output_root: Path,
        loaded_config: dict,
        **kwargs: object,
    ) -> tuple[dict, dict]:
        del kwargs
        calls.append(output_root)
        if output_root == parent_root:
            assert loaded_config == parent_config
            return parent_splits, parent_inputs
        assert output_root == child_root
        return json.loads((child_root / "split_manifest.json").read_text()), parent_inputs

    monkeypatch.setattr(exp_tuning, "_load_ready_inputs", fake_load_ready_inputs)
    monkeypatch.setattr(
        exp_tuning,
        "model_identity",
        lambda model_path, adapter_path: {
            "model": {"path": model_path},
            "adapter": {"path": adapter_path},
        },
    )
    monkeypatch.setattr(
        exp_tuning,
        "_reference_warmstart_config",
        lambda *args, **kwargs: {
            "checkpoint_kind": config["reference"]["checkpoint_kind"],
            "micro_batch": 2,
            "gradient_accumulation": 32,
        },
    )

    snapshot = exp_tuning.cmd_inherit(
        config,
        child_root,
        parent_output_root=parent_root,
        parent_config_path=Path("configs/e8_multitask_exp_tuning.yaml"),
        base_model_path="base-model",
    )
    assert snapshot["complete"]
    assert calls == [parent_root, child_root]
    assert json.loads((child_root / "plan.json").read_text())["cell_count"] == 112
    parent_response = json.loads((child_root / "inherited" / "parent_response.json").read_text())
    assert len(parent_response["rows"]) == 56
    child_reference = json.loads(exp_tuning.reference_manifest_path(child_root).read_text())
    for task in tasks:
        assert (
            child_reference["tasks"][task]["adapter_path"] == reference_tasks[task]["adapter_path"]
        )
        assert child_reference["tasks"][task]["inherited_from"]["parent_identity_hash"] == (
            f"parent-{task}"
        )



def test_exp_coldstart_matrix_is_208_cells_in_13_hard_waves() -> None:
    from drpo import e8_multitask_exp_tuning as exp_tuning

    config = exp_tuning.load_config(Path("configs/e8_multitask_exp_coldstart.yaml"))
    cells = exp_tuning.build_cells(config)
    waves = exp_tuning.build_waves(config)
    assert len(cells) == 208
    assert len({cell.key for cell in cells}) == 208
    assert len(waves) == 13
    assert [len(wave) for wave in waves] == [16] * 13
    assert config["execution"]["max_concurrent_cells"] == 16
    assert config["execution"]["slots_per_gpu"] == 2
    assert config["execution"]["wave_barriers"] is True
    assert set(config["suite"]["tasks"]) == set(exp_tuning.TASK_NAMES)
    assert set(config["suite"]["p0_tasks"]) == set(exp_tuning.TASK_NAMES) - {"countdown"}

    countdown = [cell for cell in cells if cell.task == "countdown"]
    assert len(countdown) == 16
    assert {cell.seed for cell in countdown} == {4000, 5000}
    assert sum(cell.method == exp_tuning.METHOD_POSITIVE_ONLY for cell in countdown) == 2
    assert sum(cell.method == exp_tuning.METHOD_GLOBAL for cell in countdown) == 2
    assert sum(cell.method == exp_tuning.METHOD_EXPONENTIAL for cell in countdown) == 12
    assert {float(cell.lambda_value) for cell in countdown if cell.method == exp_tuning.METHOD_EXPONENTIAL} == set(
        config["sweep"]["countdown_sentinel_coefficients"]
    )

    for task in config["suite"]["p0_tasks"]:
        task_cells = [cell for cell in cells if cell.task == task]
        assert len(task_cells) == 24
        positives = [cell for cell in task_cells if cell.method == exp_tuning.METHOD_POSITIVE_ONLY]
        exp_cells = [cell for cell in task_cells if cell.method == exp_tuning.METHOD_EXPONENTIAL]
        assert {cell.seed for cell in positives} == {4000, 5000, 6000, 7000}
        assert len(exp_cells) == 20
        assert {cell.seed for cell in exp_cells} == {4000}
        assert exp_tuning.stable_hash(list(exp_tuning._task_lambdas(config, task))) == config["sweep"]["task_grid_hashes"][task]
        assert config["sweep"]["task_grid_provenance"][task]


def test_exp_coldstart_has_no_stochastic_result_gate_or_valid_rate_eligibility() -> None:
    from drpo import e8_multitask_exp_tuning as exp_tuning

    config = exp_tuning.load_config(Path("configs/e8_multitask_exp_coldstart.yaml"))
    serialized = json.dumps(config, sort_keys=True)
    assert "countdown_reproduction_gate" not in serialized
    assert "require_peak_above_positive_only" not in serialized
    assert "late_window_pass8_absolute_tolerance" not in serialized
    assert "terminal_valid_rate_minimum" not in config["selection"]
    assert config["selection"]["terminal_valid_rate_role"] == (
        "diagnostic_only_not_selection_eligibility"
    )
    assert config["reporting"]["countdown_role"] == (
        "diagnostic_regression_sentinel_not_result_gate"
    )


def test_exp_coldstart_reference_remoteness_contract_is_static_selection_dynamic_weighting() -> None:
    from drpo import e8_multitask_exp_tuning as exp_tuning

    config = exp_tuning.load_config(Path("configs/e8_multitask_exp_coldstart.yaml"))
    contract = config["negative_sampling"]["reference_remoteness_bank"]
    assert contract["source_candidates"] == "all_deterministic_verified_wrong_mutations"
    assert contract["selected_negatives_per_prompt"] == 16
    assert contract["coverage_threshold"] is None
    assert contract["reference_rank_role"] == "provenance_and_diagnostic_only"
    assert contract["static_reference_rank_enters_training_weight"] is False
    assert contract["current_policy_surprisal_recomputed_each_update"] is True
    assert contract["original_p0_bank_preserved"] is True
    assert exp_tuning._evenly_spaced_rank_indices(16) == tuple(range(16))
    indices = exp_tuning._evenly_spaced_rank_indices(41)
    assert len(indices) == 16
    assert len(set(indices)) == 16
    assert indices[0] == 0 and indices[-1] == 40


def test_verified_wrong_candidate_reconstruction_uses_full_deterministic_universe() -> None:
    from drpo import e8_multitask_exp_tuning as exp_tuning

    class FakeResult:
        def __init__(self, value: str) -> None:
            self.score = 0.0
            self.correct = False
            self.format_valid = True
            self.error_class = "wrong"
            self.canonical_completion = value
            self.details = {"fake": True}

    class FakeAdapter:
        def mutation_candidates(self, instance, rng):
            del instance
            order = list(range(25))
            rng.shuffle(order)
            for value in order:
                yield SimpleNamespace(completion=f"wrong-{value}", mutation_class="wrong")
            yield SimpleNamespace(completion="wrong-0", mutation_class="wrong")

        def verify(self, instance, completion, mutation_class=None):
            del instance, mutation_class
            return FakeResult(completion)

        def accept_negative(self, result):
            return result.format_valid and not result.correct

    instance = TaskInstance("fake", "p0", "prompt", "oracle", {}, {})
    source_row = {
        "task": "fake",
        "prompt_id": "p0",
        "generation_seed": 17,
        "negatives": [
            {"completion": f"wrong-{value}", "canonical_completion": f"wrong-{value}"}
            for value in range(16)
        ],
    }
    first = exp_tuning._verified_wrong_candidates(FakeAdapter(), instance, source_row)
    second = exp_tuning._verified_wrong_candidates(FakeAdapter(), instance, source_row)
    assert first == second
    assert len(first) == 25
    assert len({row["canonical_completion"] for row in first}) == 25


def test_exp_coldstart_rejects_adapter_runtime_or_grid_drift() -> None:
    from drpo import e8_multitask_exp_tuning as exp_tuning

    config = exp_tuning.load_config(Path("configs/e8_multitask_exp_coldstart.yaml"))
    changed = json.loads(json.dumps(config))
    changed["initialization"]["external_adapter_allowed"] = True
    with pytest.raises(ValueError, match="zero-update"):
        exp_tuning.validate_config(changed)
    changed = json.loads(json.dumps(config))
    changed["execution"]["slots_per_gpu"] = 1
    with pytest.raises(ValueError, match="two slots"):
        exp_tuning.validate_config(changed)
    changed = json.loads(json.dumps(config))
    changed["task_runtime"]["word_sorting"]["evaluation_batch_size"] = 8
    with pytest.raises(ValueError, match="word_sorting"):
        exp_tuning.validate_config(changed)
    changed = json.loads(json.dumps(config))
    changed["sweep"]["task_lambda"]["maze"][0] = 0.11
    with pytest.raises(ValueError, match="maze"):
        exp_tuning.validate_config(changed)


def test_exp_coldstart_imports_locked_kernel_and_forbids_multitask_loader() -> None:
    from drpo import e8_multitask_exp_tuning as exp_tuning

    config = exp_tuning.load_config(Path("configs/e8_multitask_exp_coldstart.yaml"))
    assert config["canonical_coldstart"]["countdown_entry"] == (
        "countdown_e8_alpha1_highc_scan_runtime.worker"
    )
    assert config["canonical_coldstart"]["transfer_entry"] == (
        "countdown_e8_alpha1_c_scan_trainer.train_cell"
    )
    audit = exp_tuning.audit_canonical_coldstart_sources(config)
    assert audit["verified"]
    assert audit["git_blob_shas"] == config["canonical_coldstart"]["expected_git_blob_shas"]
    with pytest.raises(RuntimeError, match="old canonical"):
        exp_tuning._load_reference_model("base-model", None, config, train_mode=True)


def test_task_base_config_transfer_has_batch16_and_pass8_only(tmp_path: Path) -> None:
    from drpo import e8_multitask_exp_tuning as exp_tuning

    config = exp_tuning.load_config(Path("configs/e8_multitask_exp_coldstart.yaml"))
    task_root = tmp_path / "graph_color"
    task_root.mkdir()
    path, changed = exp_tuning._task_base_config(
        config,
        task="graph_color",
        canonical_paths=exp_tuning._canonical_paths(config),
        task_root=task_root,
    )
    assert changed == [
        "evaluation.batch_size",
        "evaluation.pass_ks",
        "model.max_length",
        "model.max_new_tokens",
    ]
    value = yaml.safe_load(path.read_text())
    assert value["model"]["max_length"] == 512
    assert value["model"]["max_new_tokens"] == 128
    assert value["evaluation"]["batch_size"] == 16
    assert value["evaluation"]["pass_ks"] == [8]


def test_exp_coldstart_scheduler_enforces_wave_barriers_and_two_slots_per_gpu(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from drpo import e8_multitask_exp_tuning as exp_tuning

    config = exp_tuning.load_config(Path("configs/e8_multitask_exp_coldstart.yaml"))
    cells = exp_tuning.build_cells(config)
    waves = exp_tuning.build_waves(config)
    monkeypatch.setattr(exp_tuning, "_require_calibration_gate", lambda *args, **kwargs: None)
    monkeypatch.setattr(exp_tuning, "_require_liveness_gate", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        exp_tuning,
        "_countdown_protocol_diagnostic",
        lambda *args, **kwargs: {
            "status": "FAIL",
            "result_gate": False,
            "controls_task_transfer_release": False,
        },
    )
    lock = threading.Lock()
    starts: dict[str, float] = {}
    finishes: dict[str, float] = {}
    active: dict[int, int] = {}
    maximum: dict[int, int] = {}

    def fake_run(**kwargs: object) -> dict[str, object]:
        cell = kwargs["cell"]
        gpu_id = int(kwargs["gpu_id"])
        with lock:
            starts[cell.key] = time.monotonic()
            active[gpu_id] = active.get(gpu_id, 0) + 1
            maximum[gpu_id] = max(maximum.get(gpu_id, 0), active[gpu_id])
        time.sleep(0.002)
        with lock:
            active[gpu_id] -= 1
            finishes[cell.key] = time.monotonic()
        return {
            "cell_key": cell.key,
            "gpu_id": gpu_id,
            "returncode": 0,
            "log": "mock.log",
            "started_unix": 0.0,
            "finished_unix": 1.0,
        }

    monkeypatch.setattr(exp_tuning, "_run_subprocess_cell", fake_run)
    result = exp_tuning.cmd_run_dynamic(
        config,
        Path("configs/e8_multitask_exp_coldstart.yaml"),
        tmp_path,
        base_model_path="base-model",
        force=False,
        retry_incomplete=True,
    )
    assert result["complete"]
    assert result["completed_cells"] == 208
    assert result["wave_barriers"] is True
    assert len(result["waves"]) == 13
    assert result["countdown_protocol_diagnostic"]["status"] == "FAIL"
    assert result["countdown_result_controls_transfer_release"] is False
    assert set(maximum) == set(range(8))
    assert max(maximum.values()) <= 2
    for index in range(1, len(waves)):
        assert min(starts[cell.key] for cell in waves[index]) >= max(
            finishes[cell.key] for cell in waves[index - 1]
        )
    assert {cell.task for cell in cells[16:32]} != {"countdown"}


def test_exp_coldstart_aggregate_does_not_filter_by_terminal_valid_rate(tmp_path: Path) -> None:
    from drpo import e8_multitask_exp_tuning as exp_tuning

    config = exp_tuning._engineering_self_test_config(
        exp_tuning.load_config(Path("configs/e8_multitask_exp_coldstart.yaml"))
    )
    p0.atomic_json(
        tmp_path / "source_provenance.json",
        {"run_id": "cold-test", "source_commit": "a" * 40},
    )
    target_task = "word_sorting"
    target_lambda = config["sweep"]["task_lambda"][target_task][0]
    for cell in exp_tuning.build_cells(config):
        late = 0.10 if cell.method == exp_tuning.METHOD_POSITIVE_ONLY else 0.20
        valid = 1.0
        if (
            cell.task == target_task
            and cell.method == exp_tuning.METHOD_EXPONENTIAL
            and float(cell.lambda_value) == float(target_lambda)
        ):
            late = 0.90
            valid = 0.10
        p0.atomic_json(
            tmp_path / "cells" / cell.key / "cell_manifest.json",
            {
                "complete": True,
                "evaluation_status": "complete",
                "validation_best_pass8": late,
                "validation_late_window_pass8_mean": late,
                "validation_terminal_pass8": late,
                "validation_best_greedy": late / 2,
                "validation_late_window_greedy_mean": late / 2,
                "validation_terminal_greedy": late / 2,
                "validation_best_greedy_valid_rate": valid,
                "validation_terminal_greedy_valid_rate": valid,
                "best_step": 900,
                "terminal_step": 1200,
                "stop_reason": "max_steps",
                "nan_inf_failure": False,
            },
        )
    summary = exp_tuning.cmd_aggregate(config, tmp_path)
    selected = summary["tasks"][target_task]["selected_exp"]
    assert selected["lambda"] == pytest.approx(float(target_lambda))
    assert selected["late_window_pass8_mean"] == pytest.approx(0.90)
    assert selected["terminal_greedy_valid_rate_mean"] == pytest.approx(0.10)
    assert summary["terminal_valid_rate_role"] == "diagnostic_only_not_selection_eligibility"
    assert summary["countdown_result_gate"] is False


def test_coldstart_engineering_self_test_exercises_208_cell_recovery_and_barriers(
    tmp_path: Path,
) -> None:
    from drpo import e8_multitask_exp_tuning as exp_tuning

    config = exp_tuning.load_config(Path("configs/e8_multitask_exp_coldstart.yaml"))
    source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    result = exp_tuning.cmd_engineering_self_test(
        config,
        tmp_path / "engineering-self-test",
        source_commit=source_commit,
    )
    assert result["complete"]
    assert result["scientific_status"] == "not_run"
    assert result["resume_completed_cells"] == 208
    assert result["aggregate_cell_count"] == 208
    assert result["queue_audit"]["wave_barriers_respected"]
    assert result["queue_audit"]["wave_count"] == 13
    assert result["queue_audit"]["slots_per_gpu"] == 2
    assert result["repeat_run_preserved_cell_hashes"]
    assert result["tampered_package_rejected"]
    assert set(result["analysis_ready_tasks"]) == set(config["suite"]["tasks"])
    assert result["task_result_count"] == 9
    output_root = tmp_path / "engineering-self-test"
    with zipfile.ZipFile(result["full_results_zip"]) as archive:
        packaged_names = set(archive.namelist())
    for task in config["suite"]["tasks"]:
        root = output_root / "task_results" / str(task)
        marker = json.loads((root / "TASK_COMPLETE.json").read_text(encoding="utf-8"))
        expected_cells = 16 if task == "countdown" else 24
        assert marker["complete"]
        assert marker["analysis_ready"]
        assert marker["final_aggregate_authority"] is False
        assert marker["cell_count"] == expected_cells
        with (root / "all_cells.csv").open(encoding="utf-8", newline="") as handle:
            assert len(list(csv.DictReader(handle))) == expected_cells
        with (root / "plot_curve_points.csv").open(encoding="utf-8", newline="") as handle:
            assert len(list(csv.DictReader(handle))) == expected_cells
        assert f"task_results/{task}/TASK_COMPLETE.json" in packaged_names



def test_coldstart_task_results_publish_before_global_aggregate(tmp_path: Path) -> None:
    from drpo import e8_multitask_exp_tuning as exp_tuning

    config = exp_tuning._engineering_self_test_config(
        exp_tuning.load_config(Path("configs/e8_multitask_exp_coldstart.yaml"))
    )
    p0.atomic_json(
        tmp_path / "source_provenance.json",
        {"run_id": "early-task-test", "source_commit": "a" * 40},
    )
    countdown_cells = [
        cell for cell in exp_tuning.build_cells(config) if cell.task == "countdown"
    ]
    assert len(countdown_cells) == 16
    for index, cell in enumerate(countdown_cells):
        score = 0.1 + index * 0.01
        p0.atomic_json(
            tmp_path / "cells" / cell.key / "cell_manifest.json",
            {
                "experiment_id": exp_tuning.experiment_id(config),
                "config_hash": exp_tuning.stable_config_hash(config),
                "validation_best_pass8": score,
                "validation_terminal_pass8": score,
                "validation_best_greedy": score / 2,
                "validation_terminal_greedy": score / 2,
                "validation_best_greedy_valid_rate": 1.0,
                "validation_terminal_greedy_valid_rate": 1.0,
                "best_step": 2,
                "terminal_step": 2,
                "stop_reason": "engineering_placeholder_complete",
                "nan_inf_failure": False,
                "evaluation_status": "complete",
                "complete": True,
            },
        )
    ready = exp_tuning._materialize_completed_coldstart_task_results(config, tmp_path)
    assert set(ready) == {"countdown"}
    assert ready["countdown"]["analysis_ready"]
    assert ready["countdown"]["cell_count"] == 16
    assert not (tmp_path / "aggregate" / "aggregate_summary.json").exists()
    assert (tmp_path / "task_results" / "countdown" / "TASK_COMPLETE.json").is_file()
    assert "task_results" in exp_tuning.RECOVERY_TRANSIENT_TOP_LEVEL

def test_coldstart_runbook_embeds_bootstrap_and_current_protocol() -> None:
    runbook = Path("docs/experiments/EXT-C-E8-MULTITASK-EXP-COLDSTART-01_RUNBOOK.md").read_text(
        encoding="utf-8"
    )
    bootstrap = Path("scripts/bootstrap_e8_multitask_exp_coldstart.sh").read_text(
        encoding="utf-8"
    )
    embedded = runbook.split("<!-- ONE_CLICK_BOOTSTRAP_START -->", 1)[1].split(
        "<!-- ONE_CLICK_BOOTSTRAP_END -->", 1
    )[0]
    assert embedded == f"\n```bash\n{bootstrap.rstrip()}\n```\n"
    assert "208 cells" in runbook
    assert "13" in runbook and "16-cell" in runbook
    assert "Spiral Matrix" in runbook
    assert "Pass@64" in runbook
    assert "结果门禁" in runbook
    assert "reference-remoteness" in runbook
    assert "task_results/<task>" in runbook
    assert "TASK_COMPLETE.json" in runbook
    assert "terminal valid rate" in runbook.lower()
    assert "0.002" not in runbook
    assert "峰值必须高于" not in runbook
    assert "run_experiment_guard_hardened.py" in Path(
        "scripts/run_e8_multitask_exp_coldstart.sh"
    ).read_text(encoding="utf-8")


def test_lambda_completion_matrix_is_config_driven_and_lambda_only() -> None:
    from drpo import e8_multitask_exp_tuning as exp_tuning

    config = exp_tuning.load_config(Path("configs/e8_multitask_exp_lambda_completion.yaml"))
    cells = exp_tuning.build_cells(config)
    waves = exp_tuning.build_waves(config)
    assert len(cells) == config["sweep"]["expected_cells"] == 199
    assert [len(wave) for wave in waves] == [16] * 12 + [7]
    assert not any(cell.task == "countdown" for cell in cells)
    for task in config["suite"]["p0_tasks"]:
        task_cells = [cell for cell in cells if cell.task == task]
        positives = [cell for cell in task_cells if cell.method == exp_tuning.METHOD_POSITIVE_ONLY]
        exponentials = [cell for cell in task_cells if cell.method == exp_tuning.METHOD_EXPONENTIAL]
        assert [cell.seed for cell in positives] == config["sweep"]["transfer_positive_only_seed_offsets"]
        assert {cell.seed for cell in exponentials} == {config["sweep"]["task_transfer_seed_offset"]}
        assert [cell.lambda_value for cell in exponentials] == config["sweep"]["task_lambda"][task]
        assert all(cell.rho is None for cell in exponentials)



@pytest.mark.skipif(torch is None, reason="Torch is unavailable in the test runtime")
def test_lambda_only_canonical_transport_is_exactly_equivalent() -> None:
    from drpo import countdown_e8_alpha1_highc_scan_common as paper_common
    from drpo import e8_multitask_exp_tuning as exp_tuning

    configs = [
        exp_tuning.load_config(Path("configs/e8_multitask_exp_coldstart.yaml")),
        exp_tuning.load_config(Path("configs/e8_multitask_exp_lambda_completion.yaml")),
    ]
    lambdas = sorted(
        {
            float(value)
            for config in configs
            for task in config["suite"]["tasks"]
            for value in config["sweep"]["task_lambda"][task]
        }
    )
    row_index = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    unique_counts = torch.tensor([2, 2], dtype=torch.long)

    for lambda_value in lambdas:
        historical = exp_tuning.Cell(
            "maze",
            exp_tuning.METHOD_EXPONENTIAL,
            math.exp(-lambda_value),
            4000,
            "historical",
            lambda_value,
        )
        successor = exp_tuning.Cell(
            "maze",
            exp_tuning.METHOD_EXPONENTIAL,
            None,
            4000,
            "successor",
            lambda_value,
        )
        assert historical.key == successor.key
        assert historical.lambda_value is not None
        assert successor.lambda_value is not None
        historical_coefficient = float(historical.lambda_value)
        successor_coefficient = float(successor.lambda_value)
        assert historical_coefficient.hex() == successor_coefficient.hex()
        assert successor_coefficient.hex() == float(lambda_value).hex()

        historical_lp = torch.tensor(
            [-0.25, -1.0, -4.0, -9.0], dtype=torch.float64, requires_grad=True
        )
        successor_lp = historical_lp.detach().clone().requires_grad_(True)
        historical_weights = paper_common.continuous_exp_weights(
            historical_lp,
            alpha=1.0,
            c=historical_coefficient,
        )
        successor_weights = paper_common.continuous_exp_weights(
            successor_lp,
            alpha=1.0,
            c=successor_coefficient,
        )
        assert torch.equal(historical_weights, successor_weights)

        historical_loss = paper_common.mean_unique_negative_term(
            historical_lp,
            historical_weights,
            row_index,
            unique_counts,
        )
        successor_loss = paper_common.mean_unique_negative_term(
            successor_lp,
            successor_weights,
            row_index,
            unique_counts,
        )
        assert torch.equal(historical_loss, successor_loss)

        historical_gradient = torch.autograd.grad(historical_loss, historical_lp)[0]
        successor_gradient = torch.autograd.grad(successor_loss, successor_lp)[0]
        assert torch.equal(historical_gradient, successor_gradient)
