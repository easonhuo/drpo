#!/usr/bin/env python3
"""Freeze tau and scale_c at the shared fresh-LoRA initialization."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from drpo import countdown_e8_paper_aligned_lambda_minimal_common as paper


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--model_path", required=True)
    result.add_argument("--bank", required=True)
    result.add_argument("--base_config", required=True)
    result.add_argument("--grid_config", required=True)
    result.add_argument("--work_dir", required=True)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    repo = Path(__file__).resolve().parents[1]
    model_path = Path(args.model_path).resolve()
    bank_path = Path(args.bank).resolve()
    base_config_path = Path(args.base_config).resolve()
    grid_config_path = Path(args.grid_config).resolve()
    work_dir = Path(args.work_dir).resolve()
    for path in (model_path, bank_path, base_config_path, grid_config_path):
        if not path.exists():
            raise SystemExit(f"Missing required input: {path}")
    config = paper.load_yaml(grid_config_path)
    paper.validate_grid_config(config)
    base_config = paper.load_yaml(base_config_path)
    calibration_cfg = config["calibration"]
    output = work_dir / "TAPER_CALIBRATION.json"
    identity = {
        "experiment_id": paper.EXPERIMENT_ID,
        "source": paper.git_state(repo),
        "model_path": str(model_path),
        "bank_sha256": paper.sha256_file(bank_path),
        "base_config_sha256": paper.sha256_file(base_config_path),
        "grid_config_sha256": paper.sha256_file(grid_config_path),
        "seed": int(calibration_cfg["seed"]),
    }
    if output.is_file():
        existing = json.loads(output.read_text(encoding="utf-8"))
        if existing.get("identity") == identity:
            print(json.dumps({"calibration": str(output), "reused": True}), flush=True)
            return 0
        raise RuntimeError("Stale calibration identity; use a new work_dir")

    paper.arena.seed_all(int(calibration_cfg["seed"]))
    tokenizer = paper.arena.load_tokenizer(str(model_path))
    model = paper.arena.load_model(
        str(model_path),
        adapter_path=None,
        trainable_adapter=True,
        load_in_4bit=bool(base_config["model"].get("load_in_4bit", False)),
        dtype=str(base_config["model"].get("dtype", "auto")),
        gradient_checkpointing=False,
        parameterization="lora",
    )
    model.eval()
    device = next(model.parameters()).device
    rows = paper.arena.read_jsonl(bank_path)
    selected = list(
        paper.arena.balanced_diagnostic_rows(rows, int(calibration_cfg["prompt_rows"]))
    )
    if len(selected) != int(calibration_cfg["prompt_rows"]):
        raise RuntimeError("Calibration row count is incomplete")
    dataset = paper.ContinuousUniqueBankDataset(
        selected, tokenizer, int(base_config["model"]["max_length"])
    )
    collate = paper.make_continuous_unique_bank_collator(tokenizer.pad_token_id)
    surprisals: list[float] = []
    with torch.no_grad():
        for index in range(len(dataset)):
            packed = collate([dataset[index]])
            stats = paper.arena.completion_stats(
                model, paper.arena.move_to_device(packed["bank"], device)
            )
            surprisals.extend(float(value) for value in (-stats["seq_lp"]).float().cpu().tolist())
    summary = paper.calibration_from_surprisals(
        surprisals,
        minimum_scale=float(calibration_cfg["minimum_surprisal_scale"]),
        minimum_active_fraction=float(calibration_cfg["minimum_active_fraction"]),
    )
    payload = {
        "schema_version": 1,
        "experiment_id": paper.EXPERIMENT_ID,
        "identity": identity,
        "formula": "alpha*exp(-lambda*relu((D-tau)/scale_c))",
        "D_definition": "negative_mean_completion_token_log_probability_with_eos",
        "tau_rule": calibration_cfg["tau_rule"],
        "scale_rule": calibration_cfg["scale_rule"],
        "prompt_rows": len(selected),
        "task_metrics_used": False,
        "test_data_used": False,
        **summary,
    }
    paper.atomic_json(output, payload)
    print(json.dumps({"calibration": str(output), **summary}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
