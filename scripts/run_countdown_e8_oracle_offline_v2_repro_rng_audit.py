#!/usr/bin/env python3
"""Run the frozen legacy-vs-isolated E8 RNG reproducibility audit."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

PROTOCOLS = (
    ("legacy_contaminated_v1", "drpo.countdown_e8_repro_legacy_runtime"),
    ("rng_isolated_v2", "drpo.countdown_e8_repro_rng_isolated_runtime"),
)
LATE_STEPS = {800, 900, 1000, 1100, 1200}


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args],
        text=True,
        stderr=subprocess.STDOUT,
    ).strip()


def require_clean_checkout(repo: Path) -> str:
    head = _git(repo, "rev-parse", "HEAD")
    if _git(repo, "status", "--porcelain"):
        raise RuntimeError("Repro RNG audit requires a clean checkout")
    return head


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temp.replace(path)


def run_command(command: list[str], *, repo: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("COMMAND=" + " ".join(command) + "\n")
        log.flush()
        completed = subprocess.run(
            command,
            cwd=repo,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed with return code {completed.returncode}: "
            + " ".join(command)
        )


def common_args(args: argparse.Namespace, phase_dir: Path) -> list[str]:
    return [
        "--model_path",
        str(Path(args.model_path).resolve()),
        "--bank",
        str(Path(args.bank).resolve()),
        "--val",
        str(Path(args.val).resolve()),
        "--base_config",
        str(Path(args.base_config).resolve()),
        "--grid_config",
        str(Path(args.grid_config).resolve()),
        "--work_dir",
        str(phase_dir),
    ]


def read_late_metrics(cell_dir: Path) -> dict[str, float | None]:
    rows = [
        json.loads(line)
        for line in (cell_dir / "validation_diagnostics.jsonl")
        .read_text()
        .splitlines()
        if line.strip()
    ]
    late = [row for row in rows if int(row["step"]) in LATE_STEPS]
    if {int(row["step"]) for row in late} != LATE_STEPS:
        raise RuntimeError(f"Incomplete late window at {cell_dir}")
    terminal = next(row for row in rows if int(row["step"]) == 1200)
    pass64 = [row for row in late if row.get("val_pass_at_64") is not None]
    return {
        "late_pass_at_8": sum(float(row["val_pass_at_8"]) for row in late)
        / len(late),
        "terminal_pass_at_8": float(terminal["val_pass_at_8"]),
        "late_pass_at_64": (
            sum(float(row["val_pass_at_64"]) for row in pass64) / len(pass64)
            if pass64
            else None
        ),
        "late_greedy": sum(float(row["val_greedy"]) for row in late) / len(late),
        "late_valid_rate": sum(float(row["val_valid_rate"]) for row in late)
        / len(late),
    }


def aggregate(work_dir: Path, source_commit: str) -> None:
    phase_rows: dict[str, dict[str, dict[str, Any]]] = {}
    for protocol, _module in PROTOCOLS:
        cells: dict[str, dict[str, Any]] = {}
        methods = work_dir / protocol / "methods"
        for summary_path in sorted(methods.glob("*/summary.json")):
            summary = json.loads(summary_path.read_text())
            cell_name = str(summary["cell"])
            cells[cell_name] = {
                "alpha": float(summary["alpha"]),
                "c": float(summary["c"]),
                "seed_offset": int(summary["seed_offset"]),
                "numerical_failure": summary.get("numerical_failure"),
                **read_late_metrics(summary_path.parent),
            }
        phase_rows[protocol] = cells

    legacy = phase_rows[PROTOCOLS[0][0]]
    isolated = phase_rows[PROTOCOLS[1][0]]
    if set(legacy) != set(isolated) or len(legacy) != 6:
        raise RuntimeError("Both protocols must contain the same six completed cells")

    paired: list[dict[str, Any]] = []
    for cell in sorted(legacy):
        left, right = legacy[cell], isolated[cell]
        paired.append(
            {
                "cell": cell,
                "alpha": left["alpha"],
                "c": left["c"],
                "seed_offset": left["seed_offset"],
                "legacy_late_pass_at_8": left["late_pass_at_8"],
                "isolated_late_pass_at_8": right["late_pass_at_8"],
                "isolated_minus_legacy_late_pass_at_8": (
                    right["late_pass_at_8"] - left["late_pass_at_8"]
                ),
                "legacy_terminal_pass_at_8": left["terminal_pass_at_8"],
                "isolated_terminal_pass_at_8": right["terminal_pass_at_8"],
                "isolated_minus_legacy_terminal_pass_at_8": (
                    right["terminal_pass_at_8"] - left["terminal_pass_at_8"]
                ),
                "legacy_late_pass_at_64": left["late_pass_at_64"],
                "isolated_late_pass_at_64": right["late_pass_at_64"],
                "legacy_late_greedy": left["late_greedy"],
                "isolated_late_greedy": right["late_greedy"],
                "legacy_late_valid_rate": left["late_valid_rate"],
                "isolated_late_valid_rate": right["late_valid_rate"],
            }
        )

    aggregate_dir = work_dir / "aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    with (aggregate_dir / "paired_rng_protocol_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired[0]))
        writer.writeheader()
        writer.writerows(paired)

    atomic_json(
        work_dir / "RNG_AUDIT_COMPLETE.json",
        {
            "schema_version": 1,
            "experiment_id": (
                "EXT-C-E8-ORACLE-OFFLINE-V2-REPRO-RNG-AUDIT-0.5B-01"
            ),
            "source_commit": source_commit,
            "protocol_order": [name for name, _module in PROTOCOLS],
            "cells_per_protocol": 6,
            "total_cells": 12,
            "same_gpu_pool": True,
            "sequential_protocol_phases": True,
            "test_data_used": False,
            "task_performance_status": "reported_not_adjudicated",
            "support_or_structure_boundary_status": "valid_rate_proxy_only",
            "nan_inf_status": (
                "observed"
                if any(
                    row.get("numerical_failure")
                    for cells in phase_rows.values()
                    for row in cells.values()
                )
                else "not_observed"
            ),
            "fixed_1200_steps_is_convergence": False,
            "method_ranking_claim_allowed": False,
            "paired_comparison": str(
                aggregate_dir / "paired_rng_protocol_comparison.csv"
            ),
        },
    )


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--bank", required=True)
    p.add_argument("--val", required=True)
    p.add_argument(
        "--base_config",
        default="configs/countdown_e8_base_rl_replay_0p5b.yaml",
    )
    p.add_argument(
        "--grid_config",
        default=(
            "configs/"
            "countdown_e8_oracle_offline_v2_repro_rng_audit_0p5b.yaml"
        ),
    )
    p.add_argument("--work_dir", required=True)
    p.add_argument("--gpus", required=True)
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    repo = Path(__file__).resolve().parents[1]
    source_commit = require_clean_checkout(repo)
    work_dir = Path(args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    gpu_ids = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if not gpu_ids or len(gpu_ids) != len(set(gpu_ids)):
        raise ValueError("--gpus must contain at least one unique GPU id")

    for protocol, module in PROTOCOLS:
        phase_dir = work_dir / protocol
        log_path = phase_dir / "controller.log"
        base = [sys.executable, "-m", module]
        shared = common_args(args, phase_dir)
        run_command(base + ["plan", *shared], repo=repo, log_path=log_path)
        run_command(base + ["smoke", *shared], repo=repo, log_path=log_path)
        run_command(
            base
            + [
                "run",
                *shared,
                "--gpus",
                ",".join(gpu_ids),
                "--runtime-slots-per-gpu",
                "1",
            ],
            repo=repo,
            log_path=log_path,
        )

    aggregate(work_dir, source_commit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
