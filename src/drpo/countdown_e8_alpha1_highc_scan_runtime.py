#!/usr/bin/env python3
"""Runtime adapter for the E8 paper-aligned taper-family scans."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Sequence

from drpo import countdown_e8_alpha1_highc_scan_common as highc


EVALUATION_SEMANTICS: dict[str, Any] = {
    "evaluation_split_file": "val.jsonl",
    "evaluation_split_role": "structurally_disjoint_held_out_evaluation",
    "evaluation_enters_training_loss": False,
    "training_structure_overlap": "none",
    "training_problem_key_overlap": "none",
    "paper_facing_checkpoint_policy": "late_window_and_terminal",
    "paper_facing_summary": ["late_window_pass_at_8", "terminal_pass_at_8"],
    "best_checkpoint_role": "supplementary_only",
    "separate_test_jsonl_used": False,
    "separate_test_jsonl_required_for_existing_curve_validity": False,
}


def _grid_config_from_argv(argv: list[str]) -> str | None:
    for index, token in enumerate(argv):
        if token == "--grid_config" and index + 1 < len(argv):
            return argv[index + 1]
    return None


_grid_config = _grid_config_from_argv(sys.argv[1:])
if _grid_config is None:
    highc.activate()
else:
    highc.activate_for_grid_config(_grid_config)

from drpo import countdown_e8_alpha1_c_scan_runtime as _base_runtime  # noqa: E402


_ORIGINAL_TRAIN_CELL = _base_runtime.train_cell
_ORIGINAL_PLAN = _base_runtime.plan
_ORIGINAL_AGGREGATE = _base_runtime._aggregate
_ORIGINAL_RUN = _base_runtime.run


def _semantic_payload(config: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = dict(EVALUATION_SEMANTICS)
    if config is not None:
        evaluation = config.get("evaluation", {})
        payload.update(
            {
                "evaluation_split_file": str(
                    evaluation.get("split_file", payload["evaluation_split_file"])
                ),
                "evaluation_split_role": str(
                    evaluation.get("split_role", payload["evaluation_split_role"])
                ),
                "evaluation_enters_training_loss": bool(
                    evaluation.get(
                        "enters_training_loss",
                        payload["evaluation_enters_training_loss"],
                    )
                ),
                "paper_facing_checkpoint_policy": str(
                    evaluation.get(
                        "paper_facing_checkpoint_policy",
                        payload["paper_facing_checkpoint_policy"],
                    )
                ),
                "best_checkpoint_role": str(
                    evaluation.get(
                        "best_checkpoint_role", payload["best_checkpoint_role"]
                    )
                ),
            }
        )
    return payload


def _augment_json(path: Path, payload: dict[str, Any]) -> None:
    if not path.is_file():
        return
    current = json.loads(path.read_text(encoding="utf-8"))
    current.update(payload)
    highc.atomic_json(path, current)


def _train_cell_with_evaluation_semantics(*args: Any, **kwargs: Any) -> dict[str, Any]:
    summary = _ORIGINAL_TRAIN_CELL(*args, **kwargs)
    output_dir = Path(kwargs["output_dir"]).resolve()
    config = highc.load_yaml(kwargs["grid_config_path"])
    summary.update(_semantic_payload(config))
    summary["best_checkpoint_saved"] = True
    reporting = dict(summary.get("reporting_separation", {}))
    reporting["task_performance"] = (
        "predeclared late-window and terminal held-out metrics; "
        "metric-specific best values are supplementary only"
    )
    summary["reporting_separation"] = reporting
    highc.atomic_json(output_dir / "manifest.json", summary)
    highc.atomic_json(output_dir / "summary.json", summary)
    return summary


_base_runtime.train_cell = _train_cell_with_evaluation_semantics


def _worker_command(args, cell, output_dir: Path) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "worker",
        "--model_path",
        args.model_path,
        "--bank",
        args.bank,
        "--val",
        args.val,
        "--base_config",
        args.base_config,
        "--grid_config",
        args.grid_config,
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


_base_runtime._worker_command = _worker_command


def plan(args: argparse.Namespace) -> int:
    result = _ORIGINAL_PLAN(args)
    config = highc.load_yaml(args.grid_config)
    path = Path(args.work_dir).resolve() / "SWEEP_PLAN.json"
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload.update(_semantic_payload(config))
        for row in payload.get("cells", []):
            if row.get("method") == "asymre":
                row["delta_v"] = round(float(row["alpha"]) - 1.0, 12)
        highc.atomic_json(path, payload)
    return result


_base_runtime.plan = plan


def _augment_aggregate_csv(path: Path) -> None:
    if not path.is_file():
        return
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return
    for row in rows:
        row.update(
            {
                "evaluation_split_role": str(
                    EVALUATION_SEMANTICS["evaluation_split_role"]
                ),
                "evaluation_enters_training_loss": "false",
                "paper_facing_checkpoint_policy": str(
                    EVALUATION_SEMANTICS["paper_facing_checkpoint_policy"]
                ),
                "best_checkpoint_role": str(
                    EVALUATION_SEMANTICS["best_checkpoint_role"]
                ),
                "separate_test_jsonl_used": "false",
            }
        )
        if row.get("method") == "asymre":
            row["delta_v"] = str(round(float(row["alpha"]) - 1.0, 12))
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _aggregate(
    work_dir: Path,
    cells: Sequence[highc.Cell],
    *,
    registration_state: str,
) -> dict[str, Any]:
    audit = _ORIGINAL_AGGREGATE(
        work_dir, cells, registration_state=registration_state
    )
    audit.update(_semantic_payload())
    audit["task_performance_status"] = (
        "late_window_and_terminal_reported_not_adjudicated"
    )
    aggregate_dir = work_dir / "aggregate"
    _augment_aggregate_csv(aggregate_dir / "per_cell_summary.csv")
    highc.atomic_json(aggregate_dir / "terminal_audit.json", audit)
    return audit


_base_runtime._aggregate = _aggregate


def run(args: argparse.Namespace) -> int:
    result = _ORIGINAL_RUN(args)
    config = highc.load_yaml(args.grid_config)
    _augment_json(
        Path(args.work_dir).resolve() / "SWEEP_COMPLETE.json",
        _semantic_payload(config),
    )
    return result


_base_runtime.run = run


def smoke(args: argparse.Namespace) -> int:
    repo = Path(__file__).resolve().parents[2]
    model, bank, val, base_config, grid_config = _base_runtime._required_inputs(args)
    config = highc.load_yaml(grid_config)
    highc.validate_grid_config(config)
    liveness = config["execution"]["liveness"]
    family = str(liveness.get("representative_family", "exponential"))
    cell = highc.Cell(
        alpha=float(liveness["representative_alpha"]),
        coefficient=float(liveness["representative_c"]),
        seed_offset=int(highc.SEED_OFFSETS[0]),
        family=family,
    )
    output_dir = Path(args.work_dir).resolve() / "_liveness" / cell.name
    if output_dir.exists() and not (output_dir / "summary.json").exists():
        shutil.rmtree(output_dir)
    try:
        summary = _base_runtime.train_cell(
            cell=cell,
            model_path=model,
            bank=bank,
            val=val,
            base_config_path=base_config,
            grid_config_path=grid_config,
            output_dir=output_dir,
            repo=repo,
            smoke=True,
        )
        passed = summary.get("numerical_failure") is None and int(
            summary.get("terminal_step", -1)
        ) == int(liveness["steps"])
        gate = {
            "schema_version": 1,
            "experiment_id": highc.EXPERIMENT_ID,
            "registration_state": str(config["registration_state"]),
            "status": "PASS" if passed else "FAIL",
            "scientific_evidence": False,
            "cell": cell.name,
            "summary": str(output_dir / "summary.json"),
            "run_identity": summary.get("run_identity"),
            "test_data_used": False,
            **_semantic_payload(config),
        }
    except BaseException as error:
        gate = {
            "schema_version": 1,
            "experiment_id": highc.EXPERIMENT_ID,
            "registration_state": str(config["registration_state"]),
            "status": "FAIL",
            "scientific_evidence": False,
            "error_type": type(error).__name__,
            "error": str(error),
            "test_data_used": False,
            **_semantic_payload(config),
        }
        highc.atomic_json(Path(args.work_dir).resolve() / "SMOKE_GATE.json", gate)
        raise
    highc.atomic_json(Path(args.work_dir).resolve() / "SMOKE_GATE.json", gate)
    return 0 if gate["status"] == "PASS" else 1


def worker(args: argparse.Namespace) -> int:
    cell = highc.Cell(
        alpha=float(args.alpha),
        coefficient=float(args.c),
        seed_offset=int(args.seed_offset),
        family=str(args.family),
    )
    config = highc.load_yaml(args.grid_config)
    expected = highc.build_cells(config)
    if cell not in expected:
        raise ValueError(f"Worker cell is outside the frozen grid: {cell}")
    summary = _base_runtime.train_cell(
        cell=cell,
        model_path=Path(args.model_path).resolve(),
        bank=Path(args.bank).resolve(),
        val=Path(args.val).resolve(),
        base_config_path=Path(args.base_config).resolve(),
        grid_config_path=Path(args.grid_config).resolve(),
        output_dir=Path(args.output_dir).resolve(),
        repo=Path(__file__).resolve().parents[2],
        smoke=False,
    )
    return 0 if summary.get("numerical_failure") is None else 1


def parser() -> argparse.ArgumentParser:
    command_parser = argparse.ArgumentParser(description=__doc__)
    command_parser.add_argument("--version", action="version", version=highc.VERSION)
    subparsers = command_parser.add_subparsers(dest="command", required=True)

    def common(subparser: argparse.ArgumentParser, *, include_work_dir: bool = True) -> None:
        subparser.add_argument("--model_path", required=True)
        subparser.add_argument("--bank", required=True)
        subparser.add_argument("--val", required=True)
        subparser.add_argument("--base_config", required=True)
        subparser.add_argument("--grid_config", required=True)
        if include_work_dir:
            subparser.add_argument("--work_dir", required=True)

    common(subparsers.add_parser("plan"))
    common(subparsers.add_parser("smoke"))
    run_parser = subparsers.add_parser("run")
    common(run_parser)
    run_parser.add_argument("--gpus", required=True)
    run_parser.add_argument("--runtime-slots-per-gpu", type=int, default=2)
    worker_parser = subparsers.add_parser("worker")
    common(worker_parser, include_work_dir=False)
    worker_parser.add_argument("--output_dir", required=True)
    worker_parser.add_argument("--family", required=True)
    worker_parser.add_argument("--alpha", type=float, required=True)
    worker_parser.add_argument("--c", type=float, required=True)
    worker_parser.add_argument("--seed_offset", type=int, required=True)
    return command_parser


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "plan":
        return plan(args)
    if args.command == "smoke":
        return smoke(args)
    if args.command == "run":
        return run(args)
    if args.command == "worker":
        return worker(args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
