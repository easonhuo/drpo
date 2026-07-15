#!/usr/bin/env python3
"""RNG-state isolation helpers for Countdown evaluation.

Historical Countdown evaluation reseeds Python, NumPy, Torch CPU, and Torch CUDA
RNGs in-process. This module provides an opt-in save/restore boundary so new
protocols can keep deterministic evaluation without mutating the training RNG
stream. Historical runners remain unchanged.
"""
from __future__ import annotations

import random
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

import numpy as np
import torch


@dataclass(frozen=True)
class GlobalRngState:
    """Complete process-global RNG state used by the Countdown trainer."""

    python_state: object
    numpy_state: tuple[object, ...]
    torch_cpu_state: torch.Tensor
    torch_cuda_states: tuple[torch.Tensor, ...]


def capture_global_rng_state() -> GlobalRngState:
    """Capture Python, NumPy, Torch CPU, and all visible CUDA RNG states."""
    cuda_states: tuple[torch.Tensor, ...] = ()
    if torch.cuda.is_available():
        cuda_states = tuple(state.clone() for state in torch.cuda.get_rng_state_all())
    return GlobalRngState(
        python_state=random.getstate(),
        numpy_state=np.random.get_state(),
        torch_cpu_state=torch.random.get_rng_state().clone(),
        torch_cuda_states=cuda_states,
    )


def restore_global_rng_state(state: GlobalRngState) -> None:
    """Restore a state captured by :func:`capture_global_rng_state`."""
    random.setstate(state.python_state)
    np.random.set_state(state.numpy_state)
    torch.random.set_rng_state(state.torch_cpu_state)
    if state.torch_cuda_states:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA RNG state was captured but CUDA is no longer available")
        visible = torch.cuda.device_count()
        if visible != len(state.torch_cuda_states):
            raise RuntimeError(
                "Visible CUDA device count changed across the RNG isolation boundary: "
                f"captured={len(state.torch_cuda_states)} current={visible}"
            )
        torch.cuda.set_rng_state_all(list(state.torch_cuda_states))


@contextmanager
def preserve_global_rng_state() -> Iterator[None]:
    """Restore all process-global RNGs after the enclosed evaluation call.

    Restoration occurs on success and on exception. DataLoader generators that
    are explicitly owned by the trainer are not touched by this boundary.
    """
    state = capture_global_rng_state()
    try:
        yield
    finally:
        restore_global_rng_state(state)
