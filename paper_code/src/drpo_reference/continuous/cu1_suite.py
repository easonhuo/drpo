"""Paper-facing orchestration, aggregation, and terminal audit for C-U1.

Public stage names follow the paper's evidence roles rather than the historical
E-numbered implementation history: ``source``, ``causal``, ``phase``, and
``taper``. Formal defaults are frozen in component protocol dataclasses.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

from drpo_reference.common import atomic_json

from .cu1 import audit_environment, make_environment
from .cu1_phase_taper import run_phase_rows, run_taper_rows
from .cu1_public_audit import aggregate_and_audit
from .cu1_public_protocol import (
    CU1Protocols,
    STAGES,
    resolve_device,
    select_seeds,
    smoke_protocols,
)
from .cu1_source_causal import run_causal_rows, run_source_rows


def run_cu1_stage(
    *,
    stage: str,
    output_root: Path,
    seeds: Sequence[int] | None = None,
    smoke: bool = False,
    device: str = "cpu",
) -> dict[str, Any]:
    """Run one paper-facing C-U1 stage and write auditable artifacts."""

    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}")
    protocols = smoke_protocols() if smoke else CU1Protocols()
    requested = None if seeds is None else tuple(int(seed) for seed in seeds)
    selected = select_seeds(stage, protocols, requested)
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    target_device = resolve_device(device)

    environment_audit = audit_environment(
        make_environment(
            selected[0],
            protocols.core,
            device=target_device,
        ),
        protocols.core,
    )
    if not environment_audit["passed"]:
        raise RuntimeError("C-U1 environment audit failed")

    control_rows: list[dict[str, Any]] | None = None
    if stage == "source":
        rows = run_source_rows(
            seeds=selected,
            root=root,
            protocols=protocols,
            device=target_device,
        )
    elif stage == "causal":
        rows = run_causal_rows(
            seeds=selected,
            root=root,
            protocols=protocols,
            device=target_device,
        )
    elif stage == "phase":
        rows, control_rows = run_phase_rows(
            seeds=selected,
            root=root,
            protocols=protocols,
            device=target_device,
        )
    else:
        rows = run_taper_rows(
            seeds=selected,
            root=root,
            protocols=protocols,
            device=target_device,
        )

    audit = aggregate_and_audit(
        stage=stage,
        rows=rows,
        seeds=selected,
        root=root,
        protocols=protocols,
        smoke=smoke,
        control_rows=control_rows,
    )
    manifest: dict[str, Any] = {
        "experiment": "C-U1",
        "paper_stage": stage,
        "terminology": "same-distribution held-out-context generalization",
        "smoke": smoke,
        "device": str(target_device),
        "selected_seeds": list(selected),
        "formal_evidence_allowed": audit["formal_evidence_allowed"],
        "protocols": {
            "core": asdict(protocols.core),
            "positive": asdict(protocols.positive),
            stage: asdict(getattr(protocols, stage)),
        },
        "environment_audit": environment_audit,
        "per_seed_environment_audits": str(
            Path("preparation") / "environment_audits" / "seed_<seed>.json"
        ),
        "terminal_audit": str(Path("terminal_audit") / f"{stage}.json"),
        "aggregate": str(Path("aggregate") / f"{stage}.csv"),
    }
    if stage == "phase":
        manifest["protocols"]["control"] = asdict(protocols.control)
        manifest["control_aggregate"] = str(
            Path("aggregate") / "phase_controls.csv"
        )
    atomic_json(root / f"manifest_{stage}.json", manifest)
    return manifest


def run_cu1_all(
    *,
    output_root: Path,
    smoke: bool = False,
    device: str = "cpu",
) -> dict[str, Any]:
    """Run all four paper stages using each stage's registered seed set."""

    manifests = {
        stage: run_cu1_stage(
            stage=stage,
            output_root=Path(output_root) / stage,
            smoke=smoke,
            device=device,
        )
        for stage in STAGES
    }
    summary = {
        "experiment": "C-U1",
        "stages": manifests,
        "smoke": smoke,
        "formal_evidence_allowed": bool(
            not smoke
            and all(
                manifest["formal_evidence_allowed"]
                for manifest in manifests.values()
            )
        ),
    }
    atomic_json(Path(output_root) / "manifest_all.json", summary)
    return summary
