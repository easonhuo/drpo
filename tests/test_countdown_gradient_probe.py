from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "drpo"
    / "countdown_qwen_arena_onefile.py"
)
SPEC = importlib.util.spec_from_file_location("countdown_gradient_probe_arena", MODULE_PATH)
assert SPEC and SPEC.loader
arena = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = arena
SPEC.loader.exec_module(arena)


class TinyTokenizer:
    pad_token_id = 0


class TinyLanguageModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.adapter_scale = torch.nn.Parameter(torch.tensor(0.2))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        use_cache: bool = False,
    ) -> SimpleNamespace:
        del attention_mask, use_cache
        batch, length = input_ids.shape
        logits = torch.zeros(batch, length, 5, dtype=torch.float32)
        logits[..., 1] = self.adapter_scale
        logits[..., 2] = -0.25 * self.adapter_scale
        return SimpleNamespace(logits=logits)


def fixed_encoded() -> arena.EncodedExample:
    return arena.EncodedExample(
        input_ids=[0, 1, 1],
        labels=[-100, 1, 1],
    )


def test_verifier_categories_are_mutually_exclusive() -> None:
    assert arena.verifier_category(
        arena.verify_expression("(1 + 2) * (3 + 4)", [1, 2, 3, 4], 21)
    ) == "correct"
    assert arena.verifier_category(
        arena.verify_expression("1 + 2 + 3 + 4", [1, 2, 3, 4], 24)
    ) == "arithmetic_wrong"
    assert arena.verifier_category(
        arena.verify_expression("1 + 2 + 3", [1, 2, 3, 4], 6)
    ) == "number_mismatch"
    assert arena.verifier_category(
        arena.verify_expression("not arithmetic", [1, 2, 3, 4], 24)
    ) == "invalid_format"


def test_single_response_gradient_is_finite_and_does_not_mutate_parameters() -> None:
    model = TinyLanguageModel()
    batch = arena.pad_encoded([fixed_encoded()], pad_id=0)
    before = model.adapter_scale.detach().clone()
    metrics = arena._single_response_gradient_metrics(
        model,
        batch,
        [model.adapter_scale],
    )
    assert metrics["token_count"] == 2
    assert metrics["mean_token_surprisal"] > 0.0
    assert metrics["direct_logit_score"] > 0.0
    assert metrics["trainable_parameter_gradient_norm"] > 0.0
    assert torch.equal(model.adapter_scale.detach(), before)
    assert model.adapter_scale.grad is None

    double_batch = {
        key: torch.cat([value, value], dim=0)
        for key, value in batch.items()
    }
    with pytest.raises(ValueError, match="exactly one response"):
        arena._single_response_gradient_metrics(
            model,
            double_batch,
            [model.adapter_scale],
        )


def test_probe_gradients_exports_four_rows_without_updates_or_checkpoints(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = TinyLanguageModel()
    tokenizer = TinyTokenizer()
    encoded = fixed_encoded()
    base_batch = arena.pad_encoded([encoded], pad_id=0)
    stored_surprisal = float(
        -arena.completion_stats(model, base_batch)["seq_lp"].detach().item()
    )

    adapter = tmp_path / "sft_adapter"
    adapter.mkdir()
    adapter_file = adapter / "adapter_model.safetensors"
    adapter_file.write_bytes(b"unchanged adapter bytes")
    adapter_hash_before = arena._directory_tree_digest(adapter)

    offline = tmp_path / "offline_data.jsonl"
    rows = [
        {
            "id": "puzzle-1",
            "prompt": "p1",
            "numbers": [1, 2, 3, 4],
            "target": 24,
            "near_negative": "1 + 2 + 3 + 4",
            "far_negative": "1 + 2 + 3",
            "near_base_surprisal": stored_surprisal,
            "far_base_surprisal": stored_surprisal,
        },
        {
            "id": "puzzle-2",
            "prompt": "p2",
            "numbers": [1, 2, 3, 4],
            "target": 21,
            "near_negative": "not arithmetic",
            "far_negative": "(1 + 2) * (3 + 4)",
            "near_base_surprisal": stored_surprisal,
            "far_base_surprisal": stored_surprisal,
        },
    ]
    with offline.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    monkeypatch.setattr(arena, "load_tokenizer", lambda _: tokenizer)
    monkeypatch.setattr(arena, "load_model", lambda *args, **kwargs: model)
    monkeypatch.setattr(
        arena,
        "encode_prompt_completion",
        lambda tokenizer, prompt, completion, max_length: encoded,
    )

    output_csv = tmp_path / "gradient_samples.csv"
    parser = arena.build_parser()
    args = parser.parse_args(
        [
            "probe_gradients",
            "--model_path",
            str(tmp_path / "model"),
            "--sft_adapter",
            str(adapter),
            "--offline_data",
            str(offline),
            "--output_csv",
            str(output_csv),
            "--max_examples",
            "2",
            "--max_stored_surprisal_delta",
            "0.000001",
            "--seed",
            "100",
        ]
    )
    parameter_before = model.adapter_scale.detach().clone()
    args.func(args)

    with output_csv.open(newline="") as handle:
        exported = list(csv.DictReader(handle))
    assert len(exported) == 4
    assert list(exported[0]) == list(arena.GRADIENT_PROBE_FIELDS)
    assert {row["response_role"] for row in exported} == {"near", "far"}
    assert {row["verifier_category"] for row in exported} == {
        "correct",
        "arithmetic_wrong",
        "number_mismatch",
        "invalid_format",
    }
    assert all(float(row["negative_coefficient_abs"]) == 1.0 for row in exported)
    assert all(float(row["mean_token_surprisal"]) > 0.0 for row in exported)
    assert all(
        float(row["trainable_parameter_gradient_norm"]) > 0.0
        for row in exported
    )

    manifest_path = Path(str(output_csv) + ".manifest.json")
    manifest = json.loads(manifest_path.read_text())
    assert manifest["puzzle_count"] == 2
    assert manifest["response_count"] == 4
    assert manifest["optimizer_created"] is False
    assert manifest["parameter_updates_executed"] == 0
    assert manifest["parameter_grad_buffers_populated"] is False
    assert manifest["checkpoint_written"] is False
    assert manifest["plot_generated"] is False
    assert manifest["trainable_parameter_digest_before"] == manifest[
        "trainable_parameter_digest_after"
    ]
    assert manifest["adapter_tree_digest_before"] == manifest[
        "adapter_tree_digest_after"
    ]
    assert manifest["max_abs_stored_surprisal_delta"] <= 1e-6
    assert torch.equal(model.adapter_scale.detach(), parameter_before)
    assert arena._directory_tree_digest(adapter) == adapter_hash_before
    assert adapter_file.read_bytes() == b"unchanged adapter bytes"
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "gradient_samples.csv",
        "gradient_samples.csv.manifest.json",
        "offline_data.jsonl",
        "sft_adapter",
    ]


def test_probe_rejects_outputs_inside_adapter_directory(tmp_path: Path) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
    offline = tmp_path / "offline.jsonl"
    offline.write_text("{}\n")
    parser = arena.build_parser()
    args = parser.parse_args(
        [
            "probe_gradients",
            "--model_path",
            str(tmp_path / "model"),
            "--sft_adapter",
            str(adapter),
            "--offline_data",
            str(offline),
            "--output_csv",
            str(adapter / "gradient_samples.csv"),
        ]
    )
    with pytest.raises(ValueError, match="outside the SFT adapter directory"):
        args.func(args)
