"""Reviewer-facing D-U1 revision-4 execution and artifact writer."""

from __future__ import annotations

import concurrent.futures
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from drpo_reference.common import atomic_json, write_csv

from .du1_protocol import (
    FORMAL_METHODS,
    DU1Protocol,
    smoke_protocol,
)
from .du1_reports import mechanism_report, taper_report
from .du1_suite import (
    aggregate,
    assign_task_collapse,
    build_terminal_audit,
    run_seed_bundle,
)
from .du1_training import DU1TerminalProtocol


def smoke_terminal_protocol() -> DU1TerminalProtocol:
    return DU1TerminalProtocol(
        window_1_steps=(0, 2),
        window_2_steps=(2, 4),
    )


def _write_jsonl(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    dict(row),
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )


def _require_empty_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise RuntimeError(
            f"output directory must be new or empty: {path}"
        )
    path.mkdir(parents=True, exist_ok=True)


def _run_seed_payload(
    protocol: DU1Protocol,
    terminal: DU1TerminalProtocol,
    seed: int,
    device_name: str,
) -> dict[str, Any]:
    if device_name == "cpu":
        torch.set_num_threads(1)
    bundle = run_seed_bundle(
        protocol=protocol,
        terminal=terminal,
        seed=seed,
        device=device_name,
    )
    return {
        "seed": seed,
        "audit": bundle.shared_start.audit,
        "calibration": bundle.shared_start.calibration,
        "trajectories": [
            row
            for run in bundle.runs
            for row in run.trajectory
        ],
        "summaries": [
            run.summary for run in bundle.runs
        ],
    }


def _selected_seeds(
    protocol: DU1Protocol,
    seeds: Sequence[int] | None,
) -> tuple[int, ...]:
    if seeds is None:
        return protocol.formal_seeds
    selected = tuple(int(seed) for seed in seeds)
    if not selected:
        raise ValueError("at least one D-U1 seed is required")
    if len(set(selected)) != len(selected):
        raise ValueError("D-U1 seed list contains duplicates")
    return selected


def run_du1(
    *,
    output_root: Path,
    seeds: Sequence[int] | None = None,
    smoke: bool = False,
    device: str = "cpu",
    workers: int | None = None,
) -> dict[str, Any]:
    """Run the complete six-method revision-4 D-U1 matrix."""

    protocol = smoke_protocol() if smoke else DU1Protocol()
    terminal = (
        smoke_terminal_protocol()
        if smoke
        else DU1TerminalProtocol()
    )
    selected = _selected_seeds(protocol, seeds)
    target = torch.device(
        "cuda"
        if device == "auto" and torch.cuda.is_available()
        else "cpu"
        if device == "auto"
        else device
    )
    if not smoke and target.type != "cpu":
        raise RuntimeError(
            "the frozen D-U1 revision-4 formal protocol requires CPU"
        )
    worker_count = (
        (1 if smoke else min(8, len(selected)))
        if workers is None
        else int(workers)
    )
    if worker_count <= 0:
        raise ValueError("workers must be positive")
    worker_count = min(worker_count, len(selected))

    root = Path(output_root)
    _require_empty_output(root)
    manifest = {
        "experiment_id": "D-U1-E6-CARTESIAN-TAPER-01",
        "protocol_revision": 4,
        "terminology": (
            "same-distribution held-out-context generalization"
        ),
        "registered_scientific_status": "not_run",
        "execution_identity": (
            "reviewer_reproduction_smoke"
            if smoke
            else "reviewer_reproduction"
        ),
        "smoke": smoke,
        "device": str(target),
        "workers": worker_count,
        "selected_seeds": list(selected),
        "registered_formal_seeds": list(
            DU1Protocol().formal_seeds
        ),
        "methods": list(FORMAL_METHODS),
        "quartic_active": False,
        "no_method_winner_assumed": True,
        "protocol": asdict(protocol),
        "terminal_protocol": asdict(terminal),
        "formal_evidence_allowed": False,
    }
    atomic_json(root / "run_manifest.json", manifest)

    payloads: list[dict[str, Any]] = []
    if worker_count == 1:
        payloads = [
            _run_seed_payload(
                protocol,
                terminal,
                seed,
                str(target),
            )
            for seed in selected
        ]
    else:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=worker_count
        ) as pool:
            future_by_seed = {
                pool.submit(
                    _run_seed_payload,
                    protocol,
                    terminal,
                    seed,
                    str(target),
                ): seed
                for seed in selected
            }
            for future in concurrent.futures.as_completed(
                future_by_seed
            ):
                payloads.append(future.result())
    payloads.sort(key=lambda payload: int(payload["seed"]))

    audits = [
        dict(payload["audit"]) for payload in payloads
    ]
    calibrations = {
        str(payload["seed"]): dict(payload["calibration"])
        for payload in payloads
    }
    trajectories = [
        dict(row)
        for payload in payloads
        for row in payload["trajectories"]
    ]
    summaries = [
        dict(row)
        for payload in payloads
        for row in payload["summaries"]
    ]
    trajectories.sort(
        key=lambda row: (
            int(row["seed"]),
            str(row["method"]),
            int(row["step"]),
        )
    )
    summaries.sort(
        key=lambda row: (
            int(row["seed"]),
            str(row["method"]),
        )
    )
    assign_task_collapse(summaries, protocol)

    for payload in payloads:
        seed = int(payload["seed"])
        seed_root = root / "checkpoints" / f"seed_{seed}"
        seed_rows = [
            row for row in trajectories
            if int(row["seed"]) == seed
        ]
        seed_summaries = [
            row for row in summaries
            if int(row["seed"]) == seed
        ]
        _write_jsonl(
            seed_root / "trajectories.jsonl",
            seed_rows,
        )
        atomic_json(
            seed_root / "per_run_summary.json",
            seed_summaries,
        )
        atomic_json(
            seed_root / "environment_audit.json",
            payload["audit"],
        )
        atomic_json(
            seed_root / "coordinate_calibration.json",
            payload["calibration"],
        )
        atomic_json(
            seed_root / "CHECKPOINT_COMPLETE.json",
            {
                "experiment_id": (
                    "D-U1-E6-CARTESIAN-TAPER-01"
                ),
                "protocol_revision": 4,
                "seed": seed,
                "methods_completed": [
                    row["method"] for row in seed_summaries
                ],
                "run_count": len(seed_summaries),
                "scientific_status": (
                    "pilot" if smoke else "not_run"
                ),
                "payload_files": [
                    "trajectories.jsonl",
                    "per_run_summary.json",
                    "environment_audit.json",
                    "coordinate_calibration.json",
                ],
            },
        )

    aggregate_summary = aggregate(summaries)
    mechanism_summary = mechanism_report(
        audits,
        summaries,
    )
    taper_summary = taper_report(
        summaries,
        aggregate_summary,
    )
    terminal_audit = build_terminal_audit(
        protocol=protocol,
        summaries=summaries,
        selected_seeds=selected,
        smoke=smoke,
    )

    atomic_json(root / "environment_audits.json", audits)
    atomic_json(
        root / "coordinate_calibration.json",
        calibrations,
    )
    _write_jsonl(root / "trajectories.jsonl", trajectories)
    atomic_json(root / "per_run_summary.json", summaries)
    write_csv(
        root / "per_run_summary.csv",
        [
            {
                key: value
                for key, value in row.items()
                if not isinstance(value, (dict, list))
            }
            for row in summaries
        ],
    )
    atomic_json(
        root / "aggregate_summary.json",
        aggregate_summary,
    )
    atomic_json(
        root / "mechanism_summary.json",
        mechanism_summary,
    )
    atomic_json(
        root / "taper_summary.json",
        taper_summary,
    )
    atomic_json(
        root / "terminal_audit.json",
        terminal_audit,
    )
    manifest["formal_evidence_allowed"] = bool(
        terminal_audit["formal_evidence_allowed"]
    )
    manifest["terminal_audit"] = "terminal_audit.json"
    manifest["aggregate_summary"] = "aggregate_summary.json"
    atomic_json(root / "run_manifest.json", manifest)
    atomic_json(
        root / "RUN_COMPLETE.json",
        {
            "experiment_id": (
                "D-U1-E6-CARTESIAN-TAPER-01"
            ),
            "protocol_revision": 4,
            "completed": True,
            "smoke": smoke,
            "expected_runs": len(selected)
            * len(FORMAL_METHODS),
            "actual_runs": len(summaries),
            "terminal_audit_all_checks_passed": bool(
                terminal_audit[
                    "formal_scientific_acceptance"
                ]
            ),
            "formal_evidence_allowed": bool(
                terminal_audit["formal_evidence_allowed"]
            ),
            "environment_validity_failures": (
                terminal_audit[
                    "environment_validity_failures"
                ]
            ),
            "task_performance_collapse_events": (
                terminal_audit[
                    "task_performance_collapse_events"
                ]
            ),
            "support_boundary_events": terminal_audit[
                "support_boundary_events"
            ],
            "nan_inf_numerical_failures": terminal_audit[
                "nan_inf_numerical_failures"
            ],
        },
    )
    return manifest
