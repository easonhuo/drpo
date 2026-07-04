import csv
import json
from pathlib import Path

import pytest

from scripts.run_countdown_gradient_probe_full import (
    REQUIRED_PROBE_FIELDS,
    build_parser,
    build_offline_command,
    inspect_offline_data,
    probe_command,
    resolve_adapter,
    verify_probe_csv,
)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_inspect_offline_rejects_raw_train_and_short_smoke(tmp_path: Path) -> None:
    raw = tmp_path / "train.jsonl"
    write_jsonl(raw, [{"id": "p0", "prompt": "x", "numbers": [1, 2, 3, 4], "target": 10}])
    status = inspect_offline_data(raw, target_examples=6000)
    assert not status.usable
    assert status.reason == "missing_key:near_negative"

    smoke = tmp_path / "offline_6000.jsonl"
    write_jsonl(
        smoke,
        [
            {
                "id": f"p{i}",
                "prompt": "x",
                "numbers": [1, 2, 3, 4],
                "target": 10,
                "near_negative": "1+2+3+4",
                "far_negative": "1*2*3*4",
                "near_base_surprisal": 1.0,
                "far_base_surprisal": 2.0,
            }
            for i in range(10)
        ],
    )
    status = inspect_offline_data(smoke, target_examples=6000)
    assert not status.usable
    assert status.reason == "too_few_rows:10<6000"


def test_resolve_adapter_prefers_best_adapter(tmp_path: Path) -> None:
    work_dir = tmp_path / "run"
    best = work_dir / "sft_adapter" / "best_adapter"
    best.mkdir(parents=True)
    (best / "adapter_config.json").write_text("{}")
    resolved = resolve_adapter(work_dir, None)
    assert resolved == best.resolve()


def test_commands_are_full_pipeline_defaults(tmp_path: Path) -> None:
    work_dir = tmp_path / "run"
    runner = tmp_path / "src" / "drpo" / "countdown_qwen_arena_onefile.py"
    runner.parent.mkdir(parents=True)
    runner.write_text("# runner\n")
    adapter = work_dir / "sft_adapter" / "best_adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text("{}")
    model = tmp_path / "Qwen"
    model.mkdir()
    input_data = work_dir / "data" / "train.jsonl"
    split_manifest = work_dir / "data" / "split_manifest.json"
    input_data.parent.mkdir(parents=True)
    input_data.write_text("{}\n")
    split_manifest.write_text("{}")
    offline = work_dir / "data" / "offline_6000_full.jsonl"
    output_csv = work_dir / "countdown_gradient_samples_seed100_full.csv"

    args = build_parser().parse_args([
        "--model_path", str(model),
        "--work_dir", str(work_dir),
    ])
    args.input_data = str(input_data)
    args.split_manifest = str(split_manifest)

    build_cmd = build_offline_command(args, runner, adapter, offline)
    assert "build_offline" in build_cmd
    assert "--reference_adapter" in build_cmd
    assert str(adapter) in build_cmd
    assert "--max_examples" in build_cmd
    assert build_cmd[build_cmd.index("--max_examples") + 1] == "6000"
    assert "--balance_by_oracle_pattern" in build_cmd

    probe_cmd = probe_command(args, runner, adapter, offline, output_csv)
    assert "probe_gradients" in probe_cmd
    assert "--max_examples" not in probe_cmd
    assert str(offline) in probe_cmd
    assert str(output_csv) in probe_cmd


def test_verify_probe_csv_requires_full_near_far_rows(tmp_path: Path) -> None:
    csv_path = tmp_path / "probe.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(REQUIRED_PROBE_FIELDS))
        writer.writeheader()
        for puzzle in range(2):
            for role in ("near", "far"):
                writer.writerow({
                    "seed": 100,
                    "puzzle_id": f"p{puzzle}",
                    "response_role": role,
                    "response": "1+2+3+4",
                    "token_count": 9,
                    "valid_format": True,
                    "uses_numbers": True,
                    "correct": False,
                    "verifier_category": "arithmetic_wrong",
                    "mean_token_surprisal": 1.0,
                    "stored_base_surprisal": 1.0,
                    "direct_logit_score": 0.5,
                    "negative_coefficient_abs": 1.0,
                    "trainable_parameter_gradient_norm": 2.0,
                })
    summary = verify_probe_csv(csv_path, expected_puzzles=2)
    assert summary["response_count"] == 4
    assert summary["roles"] == {"near": 2, "far": 2}

    with pytest.raises(RuntimeError, match="expected 6"):
        verify_probe_csv(csv_path, expected_puzzles=3)
