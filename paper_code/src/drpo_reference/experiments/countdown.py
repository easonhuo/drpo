"""Public Countdown runtime facade with deterministic device-resource release.

The implementation remains in :mod:`drpo_reference.experiments.countdown_runtime`.
This facade installs the reviewer-runtime lifecycle guard before exposing that
module under the stable public import path.
"""

from __future__ import annotations

import gc
import sys
from typing import Any

import torch

from . import countdown_runtime as _runtime


def _release_model(model: Any) -> None:
    """Release completed-model tensors before the next runtime model load."""

    move = getattr(model, "to", None)
    if callable(move):
        try:
            move("cpu")
        except Exception:
            # Some quantized model wrappers do not implement ``to('cpu')``.
            # Clearing the completed top-level module registries below still
            # drops their tensor graph before the next model is loaded.
            pass
    for attribute in ("_parameters", "_buffers", "_modules"):
        registry = getattr(model, attribute, None)
        clear = getattr(registry, "clear", None)
        if callable(clear):
            clear()
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


_runtime._release_model = _release_model
sys.modules[__name__] = _runtime
