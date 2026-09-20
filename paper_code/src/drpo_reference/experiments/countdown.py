"""Public facade for the unified reviewer-facing Structured Generation runtime."""

from __future__ import annotations

from .countdown_runtime import (
    evaluate_model,
    load_structured_generation_config,
    run_structured_generation,
)

# Backward-compatible Python alias only; both names execute the same nine-task path.
run_countdown = run_structured_generation

__all__ = [
    "evaluate_model",
    "load_structured_generation_config",
    "run_structured_generation",
    "run_countdown",
]
