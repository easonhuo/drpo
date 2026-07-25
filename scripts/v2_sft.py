#!/usr/bin/env python3
"""Train one shared V2 oracle-SFT adapter and reuse it for warm-started DPO.

The historical no-argument invocation is preserved: it trains the original
six-epoch V2 SFT diagnostic.  The governed warm-start pipeline uses the
``run-sft`` and ``dpo-runtime`` subcommands instead:

* ``run-sft`` trains exactly one oracle-SFT epoch, persists ``epoch_1_adapter``
  once, and writes identity-bound completion/gate JSON files;
* ``dpo-runtime`` loads that read-only adapter into every DPO worker, copies it
  to the frozen reference adapter before update 1, and delegates to the existing
  canonical-DPO runtime without introducing another Python training stack.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

MODEL = "/root/models/Qwen2.5-0.5B-Instruct"
V2_BANK = "/root/experiment_output/e8_oracle_bank_v2/data/offline_bank_v2.jsonl"
V2_VAL = "/root/experiment_output/e8_oracle_bank_v2/data/val.jsonl"
OUT = Path("/root/experiment_output/e8_v2_sft")

WARM_DPO_EXPERIMENT_ID = (
    "EXT-C-E8-ORACLE-OFFLINE-V2-SHARED-SFT-CANONICAL-DPO-BETA-SCAN-0.5B-01"
)
COLD_DPO_EXPERIMENT_ID = (
    "EXT-C-E8-ORACLE-OFFLINE-V2-CANONICAL-DPO-BETA-SCAN-0.5B-01"
)
WARM_DPO_POINTS = tuple(
    ("canonical_dpo", 1.0, beta)
    for beta in (0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0)
)
WARM_DPO_SEED_OFFSETS = (4000, 5000)
SFT_SEED = 2026070700
SFT_EVAL_SEED = 2026070790


def _repo_modules() -> tuple[Any, Any]:
    from drpo import countdown_e8_alpha1_highc_scan_common as highc
    from drpo import countdown_qwen_arena_onefile as arena

    return arena, highc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _model_identity(model_path: Path) -> dict[str, Any]:
    payload: dict[str, Any] = {"path": str(model_path.resolve())}
    config = model_path / "config.json"
    if config.is_file():
        payload["config_sha256"] = _sha256(config)
    index_files = sorted(model_path.glob("*.index.json"))
    payload["index_sha256"] = {
        item.name: _sha256(item) for item in index_files if item.is_file()
    }
    return payload


def _adapter_identity(adapter_path: Path) -> dict[str, Any]:
    adapter_path = adapter_path.resolve()
    config = adapter_path / "adapter_config.json"
    weights = [
        item
        for item in sorted(adapter_path.glob("adapter_model.*"))
        if item.is_file() and item.suffix in {".safetensors", ".bin"}
    ]
    if not config.is_file():
        raise FileNotFoundError(f"missing shared SFT adapter config: {config}")
    if len(weights) != 1:
        raise RuntimeError(
            f"expected exactly one shared SFT adapter weight file in {adapter_path}; "
            f"found {[item.name for item in weights]}"
        )
    return {
        "path": str(adapter_path),
        "adapter_config_sha256": _sha256(config),
        "weight_file": weights[0].name,
        "weight_sha256": _sha256(weights[0]),
    }


def _sft_run_identity(args: argparse.Namespace) -> dict[str, Any]:
    model_path = Path(args.model_path).resolve()
    train_data = Path(args.train_data).resolve()
    val_data = Path(args.val_data).resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(f"model directory is missing: {model_path}")
    if not train_data.is_file():
        raise FileNotFoundError(f"V2 training bank is missing: {train_data}")
    if not val_data.is_file():
        raise FileNotFoundError(f"V2 held-out validation split is missing: {val_data}")
    return {
        "experiment_id": WARM_DPO_EXPERIMENT_ID,
        "stage": "shared_v2_oracle_sft",
        "model": _model_identity(model_path),
        "train_data": {"path": str(train_data), "sha256": _sha256(train_data)},
        "validation_data": {"path": str(val_data), "sha256": _sha256(val_data)},
        "sft": {
            "epochs": 1,
            "min_epochs": 1,
            "seed": SFT_SEED,
            "eval_seed": SFT_EVAL_SEED,
            "learning_rate": 2.0e-4,
            "micro_batch": 2,
            "gradient_accumulation": 32,
            "warmup_ratio": 0.05,
            "maximum_gradient_norm": 1.0,
            "parameterization": "lora",
            "dtype": "bf16",
            "save_every_epoch": True,
        },
    }


def _read_metrics(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def run_shared_sft(args: argparse.Namespace) -> int:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)
    arena, _ = _repo_modules()
    output_dir = Path(args.output_dir).resolve()
    identity = _sft_run_identity(args)
    gate_path = output_dir / "SFT_WARMSTART_GATE.json"
    adapter_path = output_dir / "epoch_1_adapter"

    if gate_path.is_file():
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        if gate.get("status") == "PASS" and gate.get("run_identity") == identity:
            current = _adapter_identity(adapter_path)
            if current == gate.get("shared_adapter"):
                print(
                    f"[v2_sft] reusing identity-matched shared adapter: {adapter_path}",
                    flush=True,
                )
                return 0
        raise RuntimeError(
            f"stale or mismatched SFT gate at {gate_path}; use a new output directory"
        )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(
            f"refusing to mix a new shared SFT run into non-empty directory: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    sft_args = argparse.Namespace(
        model_path=str(Path(args.model_path).resolve()),
        train_data=str(Path(args.train_data).resolve()),
        val_data=str(Path(args.val_data).resolve()),
        output_dir=str(output_dir),
        seed=SFT_SEED,
        max_length=256,
        max_new_tokens=80,
        epochs=1,
        min_epochs=1,
        early_stop_patience=1,
        parameterization="lora",
        micro_batch=2,
        grad_accum=32,
        lr=2.0e-4,
        warmup_ratio=0.05,
        max_grad_norm=1.0,
        num_workers=0,
        eval_examples=500,
        eval_batch=8,
        pass_k=8,
        eval_seed=SFT_EVAL_SEED,
        selection_metric="pass_at_k",
        selection_delta=0.0,
        log_every=20,
        load_in_4bit=False,
        dtype="bf16",
        result_status="pilot",
        save_every_epoch=True,
    )
    print(f"[v2_sft] one shared oracle-SFT epoch -> {output_dir}", flush=True)
    started = time.time()
    arena.cmd_sft(sft_args)
    elapsed = time.time() - started

    shared_adapter = _adapter_identity(adapter_path)
    metrics_path = output_dir / "sft_metrics.csv"
    metrics = _read_metrics(metrics_path)
    complete = {
        "schema_version": 1,
        "experiment_id": WARM_DPO_EXPERIMENT_ID,
        "stage": "shared_v2_oracle_sft",
        "status": "COMPLETE",
        "scientific_evidence": False,
        "run_identity": identity,
        "shared_adapter": shared_adapter,
        "metrics_file": str(metrics_path),
        "metrics_rows": metrics,
        "elapsed_seconds": elapsed,
        "checkpoint_policy": "persist epoch_1_adapter once; all 16 DPO cells load it read-only",
        "full_base_model_duplicated": False,
    }
    _atomic_json(output_dir / "SFT_COMPLETE.json", complete)
    gate = {
        "schema_version": 1,
        "experiment_id": WARM_DPO_EXPERIMENT_ID,
        "stage": "shared_v2_oracle_sft",
        "status": "PASS",
        "scientific_evidence": False,
        "run_identity": identity,
        "shared_adapter": shared_adapter,
        "exact_sft_epochs": 1,
        "adaptive_metric_stopping_used": False,
        "test_data_used": False,
    }
    _atomic_json(gate_path, gate)
    print(json.dumps(gate, indent=2, sort_keys=True), flush=True)
    return 0


def _validate_warm_dpo_config(config: Mapping[str, Any]) -> None:
    if config.get("experiment_id") != WARM_DPO_EXPERIMENT_ID:
        raise ValueError("warm-started DPO experiment_id mismatch")
    if config.get("result_status") != "pilot":
        raise ValueError("warm-started DPO must remain a pilot")
    if config.get("registration_state") != "dev_code_first_unregistered":
        raise ValueError("warm-started DPO must remain code-first unregistered")
    if config.get("method_identity") != (
        "canonical_sigmoid_dpo_shared_v2_oracle_sft_frozen_reference"
    ):
        raise ValueError("warm-started DPO method identity changed")

    model = config.get("model", {})
    expected_model = {
        "initialization": "shared_v2_oracle_sft_epoch1_adapter",
        "parameterization": "lora",
        "shared_frozen_backbone": True,
        "policy_adapter": "default",
        "reference_adapter": "reference",
        "copy_policy_adapter_to_reference_at_initialization": True,
        "reference_trainable": False,
        "disable_dropout": True,
        "merge_sft_into_base_model": False,
    }
    for key, expected in expected_model.items():
        if model.get(key) != expected:
            raise ValueError(f"warm-started DPO model field changed: {key}")

    warm = config.get("shared_warmstart", {})
    expected_warm = {
        "training_runs": 1,
        "epochs": 1,
        "seed": SFT_SEED,
        "checkpoint_role": "shared_epoch_1_adapter",
        "reuse_across_all_dpo_cells": True,
        "read_only_during_dpo": True,
        "adaptive_metric_stopping": False,
        "test_split_access": False,
    }
    for key, expected in expected_warm.items():
        if warm.get(key) != expected:
            raise ValueError(f"shared warmstart field changed: {key}")

    points = tuple(
        (str(item["family"]), float(item["alpha"]), float(item["coefficient"]))
        for item in config.get("sweep", {}).get("parameter_points", ())
    )
    if points != WARM_DPO_POINTS:
        raise ValueError("warm-started DPO beta points changed")
    seeds = tuple(int(value) for value in config["sweep"].get("seed_offsets", ()))
    if seeds != WARM_DPO_SEED_OFFSETS:
        raise ValueError("warm-started DPO development seed offsets changed")
    if int(config["sweep"].get("cells", -1)) != 16:
        raise ValueError("warm-started DPO requires sixteen cells")

    training = config.get("training", {})
    if int(training.get("steps", -1)) != 1200 or training.get("early_stop") is not False:
        raise ValueError("warm-started DPO requires a fixed 1200-step horizon")
    execution = config.get("execution", {})
    if execution.get("default_gpus") != [0, 1]:
        raise ValueError("warm-started DPO requires GPU 0-1")
    if int(execution.get("parallel_cells_per_gpu", -1)) != 2:
        raise ValueError("warm-started DPO requires two cells per GPU")
    if int(execution.get("expected_full_waves", -1)) != 4:
        raise ValueError("warm-started DPO requires four full waves")
    if config.get("evaluation", {}).get("separate_test_split_access") is not False:
        raise ValueError("warm-started DPO test access must remain disabled")


def _install_warm_dpo_profile(highc: Any, grid_config: Path) -> None:
    config = highc.load_yaml(grid_config)
    if config.get("experiment_id") != WARM_DPO_EXPERIMENT_ID:
        raise ValueError("grid config is not the shared-SFT warm-started DPO profile")
    if WARM_DPO_EXPERIMENT_ID in highc._PROFILES:
        return
    highc._PROFILES[WARM_DPO_EXPERIMENT_ID] = {
        "experiment_id": WARM_DPO_EXPERIMENT_ID,
        "version": "0.4.0-dev-shared-sft-warmstarted-dpo-8beta-2slot",
        "default_grid_config": str(grid_config),
        "parameter_points": WARM_DPO_POINTS,
        "seed_offsets": WARM_DPO_SEED_OFFSETS,
        "expected_points": 8,
        "expected_cells": 16,
        "requires_positive_only": False,
        "kind": "canonical_dpo",
    }
    original_validate = highc.validate_grid_config
    original_points = highc.parameter_points
    original_cells = highc.build_cells

    def validate(value: Mapping[str, Any]) -> None:
        if value.get("experiment_id") == WARM_DPO_EXPERIMENT_ID:
            _validate_warm_dpo_config(value)
            return
        original_validate(value)

    def parameter_points(value: Mapping[str, Any]) -> tuple[Any, ...]:
        if value.get("experiment_id") == WARM_DPO_EXPERIMENT_ID:
            _validate_warm_dpo_config(value)
            return WARM_DPO_POINTS
        return original_points(value)

    def build_cells(value: Mapping[str, Any]) -> tuple[Any, ...]:
        if value.get("experiment_id") != WARM_DPO_EXPERIMENT_ID:
            return original_cells(value)
        cells = tuple(
            highc.Cell(
                alpha=alpha,
                coefficient=beta,
                seed_offset=seed_offset,
                family=family,
            )
            for family, alpha, beta in WARM_DPO_POINTS
            for seed_offset in WARM_DPO_SEED_OFFSETS
        )
        if len(cells) != 16 or len({cell.name for cell in cells}) != 16:
            raise AssertionError("warm-started DPO must produce sixteen unique cells")
        return cells

    highc.validate_grid_config = validate
    highc.parameter_points = parameter_points
    highc.build_cells = build_cells


def validate_profile(args: argparse.Namespace) -> int:
    _, highc = _repo_modules()
    grid = Path(args.grid_config).resolve()
    _install_warm_dpo_profile(highc, grid)
    config = highc.load_yaml(grid)
    highc.validate_grid_config(config)
    cells = highc.build_cells(config)
    print(
        json.dumps(
            {
                "status": "PASS",
                "experiment_id": WARM_DPO_EXPERIMENT_ID,
                "parameter_points": 8,
                "cells": len(cells),
                "shared_sft_runs": 1,
                "shared_sft_epochs": 1,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _argument_value(arguments: Sequence[str], name: str) -> str | None:
    try:
        index = list(arguments).index(name)
    except ValueError:
        return None
    return arguments[index + 1] if index + 1 < len(arguments) else None


def _rewrite_output_json(root: Path, warmstart: Mapping[str, Any]) -> None:
    if not root.exists():
        return
    for path in root.rglob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        changed = False
        if payload.get("experiment_id") == COLD_DPO_EXPERIMENT_ID:
            payload["experiment_id"] = WARM_DPO_EXPERIMENT_ID
            changed = True
        if payload.get("method") == "canonical_dpo" or path.name in {
            "SWEEP_PLAN.json",
            "SWEEP_COMPLETE.json",
            "SMOKE_GATE.json",
            "terminal_audit.json",
        }:
            payload["method_identity"] = (
                "canonical_sigmoid_dpo_shared_v2_oracle_sft_frozen_reference"
            )
            payload["model_initialization"] = "shared_v2_oracle_sft_epoch1_adapter"
            payload["shared_sft_warmstart"] = dict(warmstart)
            changed = True
        if path.name == "SWEEP_PLAN.json":
            payload["dpo_beta_points"] = [point[2] for point in WARM_DPO_POINTS]
            payload["paired_seed_offsets"] = list(WARM_DPO_SEED_OFFSETS)
            payload["matrix_shape"] = {
                "parameter_points": 8,
                "seeds_per_parameter": 2,
                "cells": 16,
            }
            changed = True
        if path.name == "terminal_audit.json":
            payload["matrix_shape"] = {
                "parameter_points": 8,
                "seeds_per_parameter": 2,
                "cells": 16,
            }
            changed = True
        if changed:
            _atomic_json(path, payload)


def run_dpo_runtime(args: argparse.Namespace) -> int:
    runtime_args = list(args.runtime_args)
    if runtime_args and runtime_args[0] == "--":
        runtime_args = runtime_args[1:]
    if not runtime_args:
        raise ValueError("dpo-runtime requires the delegated runtime command")

    shared_adapter = Path(args.shared_adapter).resolve()
    warmstart = _adapter_identity(shared_adapter)
    grid = Path(args.grid_config).resolve()
    arena, highc = _repo_modules()
    _install_warm_dpo_profile(highc, grid)
    highc.activate_for_grid_config(grid)

    original_load_model = arena.load_model

    def load_shared_sft_model(
        model_path: str,
        *positional: Any,
        adapter_path: str | None = None,
        trainable_adapter: bool = False,
        **keywords: Any,
    ) -> Any:
        if trainable_adapter and adapter_path is None:
            adapter_path = str(shared_adapter)
        return original_load_model(
            model_path,
            *positional,
            adapter_path=adapter_path,
            trainable_adapter=trainable_adapter,
            **keywords,
        )

    arena.load_model = load_shared_sft_model
    original_identity = highc._identity

    def warm_identity(*identity_args: Any, **identity_kwargs: Any) -> dict[str, Any]:
        payload = original_identity(*identity_args, **identity_kwargs)
        payload["model_initialization"] = "shared_v2_oracle_sft_epoch1_adapter"
        payload["shared_sft_warmstart"] = dict(warmstart)
        return payload

    highc._identity = warm_identity

    runtime_path = REPO_ROOT / "src" / "drpo" / "countdown_e8_alpha1_highc_scan_runtime.py"
    spec = importlib.util.spec_from_file_location("_e8_warmstarted_dpo_runtime", runtime_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to import existing DPO runtime: {runtime_path}")
    module = importlib.util.module_from_spec(spec)
    saved_argv = sys.argv
    try:
        sys.argv = [str(runtime_path), *runtime_args]
        spec.loader.exec_module(module)
    finally:
        sys.argv = saved_argv

    wrapper_path = Path(__file__).resolve()

    def worker_command(run_args: Any, cell: Any, output_dir: Path) -> list[str]:
        return [
            sys.executable,
            str(wrapper_path),
            "dpo-runtime",
            "--shared-adapter",
            str(shared_adapter),
            "--grid-config",
            str(grid),
            "--",
            "worker",
            "--model_path",
            run_args.model_path,
            "--bank",
            run_args.bank,
            "--val",
            run_args.val,
            "--base_config",
            run_args.base_config,
            "--grid_config",
            run_args.grid_config,
            "--output_dir",
            str(output_dir),
            "--family",
            str(cell.family),
            "--alpha",
            str(cell.alpha),
            "--c",
            str(float(cell.c)),
            "--seed_offset",
            str(cell.seed_offset),
        ]

    module._worker_command = worker_command
    module._base_runtime._worker_command = worker_command
    result = int(module.main(runtime_args))

    output_dir = _argument_value(runtime_args, "--output_dir")
    work_dir = _argument_value(runtime_args, "--work_dir")
    if output_dir is not None:
        _rewrite_output_json(Path(output_dir).resolve(), warmstart)
    if work_dir is not None:
        _rewrite_output_json(Path(work_dir).resolve(), warmstart)
    return result


def historical_main() -> int:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "6")
    arena, _ = _repo_modules()
    OUT.mkdir(parents=True, exist_ok=True)
    args = argparse.Namespace(
        model_path=MODEL,
        train_data=V2_BANK,
        val_data=V2_VAL,
        output_dir=str(OUT),
        seed=SFT_SEED,
        max_length=256,
        max_new_tokens=80,
        epochs=6,
        min_epochs=3,
        early_stop_patience=2,
        parameterization="lora",
        micro_batch=2,
        grad_accum=32,
        lr=2.0e-4,
        warmup_ratio=0.05,
        max_grad_norm=1.0,
        num_workers=0,
        eval_examples=500,
        eval_batch=8,
        pass_k=8,
        eval_seed=SFT_EVAL_SEED,
        selection_metric="pass_at_k",
        selection_delta=0.0,
        log_every=20,
        load_in_4bit=False,
        dtype="bf16",
        result_status="pilot",
        save_every_epoch=True,
    )
    arena.cmd_sft(args)
    print(f"[v2_sft] B init = {OUT}/epoch_1_adapter", flush=True)
    print(f"[v2_sft] C init = {OUT}/best_adapter", flush=True)
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    subparsers = root.add_subparsers(dest="command")

    sft = subparsers.add_parser("run-sft")
    sft.add_argument("--model-path", default=MODEL)
    sft.add_argument("--train-data", default=V2_BANK)
    sft.add_argument("--val-data", default=V2_VAL)
    sft.add_argument("--output-dir", required=True)
    sft.add_argument("--cuda-visible-devices", default="0")

    validate = subparsers.add_parser("validate")
    validate.add_argument("--grid-config", required=True)

    delegated = subparsers.add_parser("dpo-runtime")
    delegated.add_argument("--shared-adapter", required=True)
    delegated.add_argument("--grid-config", required=True)
    delegated.add_argument("runtime_args", nargs=argparse.REMAINDER)
    return root


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments:
        return historical_main()
    args = parser().parse_args(arguments)
    if args.command == "run-sft":
        return run_shared_sft(args)
    if args.command == "validate":
        return validate_profile(args)
    if args.command == "dpo-runtime":
        return run_dpo_runtime(args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
