"""Frozen public-runner protocol bundles for C-U1."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from .cu1 import CU1Protocol
from .cu1_control import CU1ControlProtocol
from .cu1_mechanism import CU1CausalProtocol, CU1SourceProtocol
from .cu1_phase import CU1PhaseProtocol
from .cu1_taper import CU1TaperProtocol
from .cu1_training import CU1PositiveProtocol

STAGES = ("source", "causal", "phase", "taper")
EVENT_FIELDS = (
    "task_performance_collapse_event",
    "support_or_variance_boundary_event",
    "nan_inf_numerical_event",
    "environment_invalid_event",
)


@dataclass(frozen=True)
class CU1Protocols:
    """Complete frozen protocol bundle for the C-U1 public runner."""

    core: CU1Protocol = CU1Protocol()
    positive: CU1PositiveProtocol = CU1PositiveProtocol()
    source: CU1SourceProtocol = CU1SourceProtocol()
    causal: CU1CausalProtocol = CU1CausalProtocol()
    phase: CU1PhaseProtocol = CU1PhaseProtocol()
    control: CU1ControlProtocol = CU1ControlProtocol()
    taper: CU1TaperProtocol = CU1TaperProtocol()


def formal_seeds(stage: str, protocols: CU1Protocols) -> tuple[int, ...]:
    if stage == "source":
        return protocols.source.formal_seeds
    if stage == "causal":
        return protocols.causal.formal_seeds
    if stage == "phase":
        return protocols.phase.formal_seeds
    if stage == "taper":
        return protocols.taper.formal_seeds
    raise ValueError(f"unknown C-U1 stage: {stage}")


def select_seeds(
    stage: str,
    protocols: CU1Protocols,
    requested: tuple[int, ...] | None,
) -> tuple[int, ...]:
    if requested is None:
        return formal_seeds(stage, protocols)
    selected = tuple(int(seed) for seed in requested)
    if not selected:
        raise ValueError("at least one seed is required")
    if len(set(selected)) != len(selected):
        raise ValueError("seed list contains duplicates")
    return selected


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target = torch.device(value)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return target


def smoke_protocols() -> CU1Protocols:
    """Return a visibly non-formal, fast protocol for integration tests."""

    return CU1Protocols(
        core=replace(
            CU1Protocol(),
            n_train_states=32,
            n_test_states=24,
            hidden_dim=16,
        ),
        positive=replace(
            CU1PositiveProtocol(),
            positive_batch_states=8,
            positive_steps=3,
            positive_continuation_steps=1,
            lbfgs_max_iter=1,
            positive_polish_min_steps=1,
            positive_polish_max_steps=1,
            positive_polish_check_every=1,
            eval_every=1,
            probe_states=4,
            formal_seeds=(10,),
        ),
        source=replace(
            CU1SourceProtocol(),
            probe_states=4,
            formal_seeds=(10,),
        ),
        causal=replace(
            CU1CausalProtocol(),
            fixed_steps=2,
            learnable_steps=2,
            evaluation_interval=1,
            formal_seeds=(30,),
        ),
        phase=replace(
            CU1PhaseProtocol(),
            fixed_alphas=(0.0, 1.0),
            learnable_alphas=(0.0, 0.38),
            warm_steps=2,
            continuation_steps=1,
            runaway_steps=3,
            evaluation_interval=1,
            formal_seeds=(50,),
        ),
        control=replace(
            CU1ControlProtocol(),
            steps=3,
            evaluation_interval=1,
            formal_seeds=(50,),
        ),
        taper=replace(
            CU1TaperProtocol(),
            formal_seeds=(70,),
            sensitivity_retentions=(),
            batch_states=8,
            evaluation_interval=1,
            minimum_steps=1,
            maximum_steps=3,
            stable_windows=2,
            probe_states=4,
        ),
    )
