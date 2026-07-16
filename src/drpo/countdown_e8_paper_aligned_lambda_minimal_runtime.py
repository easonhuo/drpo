#!/usr/bin/env python3
"""Thin runtime adapter over the canonical 8-GPU x 2-slot E8 scheduler."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

from drpo import countdown_e8_paper_aligned_lambda_minimal_common as paper

paper.activate()

from drpo import countdown_e8_alpha1_c_scan_runtime as _base_runtime  # noqa: E402

_ORIGINAL_PLAN = _base_runtime.plan
_ORIGINAL_RUN = _base_runtime.run
_ORIGINAL_WORKER = _base_runtime.worker


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
        "--alpha",
        str(cell.alpha),
        "--c",
        str(cell.lambda_value),
        "--seed_offset",
        str(cell.seed_offset),
    ]


def _annotate_json(path: Path) -> None:
    if not path.is_file():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "c" in payload:
        payload["lambda"] = payload["c"]
        payload["c_compatibility_role"] = "predecessor_call_signature_only"
    payload.setdefault("formula", "alpha*exp(-lambda*relu((D-tau)/scale_c))")
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _annotate_jsonl(path: Path) -> None:
    if not path.is_file():
        return
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if "c" in row:
            row["lambda"] = row["c"]
        row["legacy_u_fields_equal_normalized_excess_z"] = True
        rows.append(row)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def plan(args) -> int:
    code = _ORIGINAL_PLAN(args)
    path = Path(args.work_dir).resolve() / "SWEEP_PLAN.json"
    if code == 0 and path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["predecessor_commit"] = "929142930a3e2efaa7cafc8e4afe3866600027a5"
        payload["formula"] = "alpha*exp(-lambda*relu((D-tau)/scale_c))"
        payload["c_field_role"] = "internal predecessor lambda alias"
        for cell in payload.get("cells", []):
            if "c" in cell:
                cell["lambda"] = cell["c"]
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return code


def worker(args) -> int:
    code = _ORIGINAL_WORKER(args)
    output = Path(args.output_dir).resolve()
    _annotate_json(output / "summary.json")
    _annotate_jsonl(output / "validation_diagnostics.jsonl")
    _annotate_jsonl(output / "training_diagnostics.jsonl")
    return code


def run(args) -> int:
    code = _ORIGINAL_RUN(args)
    aggregate = Path(args.work_dir).resolve() / "aggregate"
    csv_path = aggregate / "per_cell_summary.csv"
    if csv_path.is_file():
        with csv_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if rows:
            fieldnames = list(rows[0])
            if "lambda" not in fieldnames:
                fieldnames.append("lambda")
            for row in rows:
                row["lambda"] = row.get("c", "")
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
    _annotate_json(aggregate / "terminal_audit.json")
    _annotate_json(Path(args.work_dir).resolve() / "SWEEP_COMPLETE.json")
    return code


_base_runtime._worker_command = _worker_command
_base_runtime.plan = plan
_base_runtime.worker = worker
_base_runtime.run = run

parser = _base_runtime.parser
smoke = _base_runtime.smoke


def main(argv: list[str] | None = None) -> int:
    return _base_runtime.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
