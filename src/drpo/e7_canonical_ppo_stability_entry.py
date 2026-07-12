"""Run the E7 PPO pilot against the validated source RunSpec.

The validated nine-task RunSpec predates explicit OPENBLAS_NUM_THREADS recording.
The historical PPO adapter requires all BLAS thread variables to be one. This
entrypoint adds only the missing runtime thread setting before delegating; it
does not alter the scientific matrix or trainer arguments.
"""

from __future__ import annotations

import copy
from typing import Any

from drpo import e7_canonical_ppo_stability as pilot


_ORIGINAL_SOURCE_LOADER = pilot._BASE_LOAD_RUN_SPEC  # noqa: SLF001


def _load_source_run_spec(path: str) -> tuple[dict[str, Any], str]:
    raw, digest = _ORIGINAL_SOURCE_LOADER(path)
    updated = copy.deepcopy(raw)
    environment = updated.setdefault("environment", {})
    existing = environment.get("OPENBLAS_NUM_THREADS")
    if existing not in (None, "1", 1):
        raise ValueError(
            "source RunSpec OPENBLAS_NUM_THREADS must be absent or equal to 1"
        )
    environment["OPENBLAS_NUM_THREADS"] = "1"
    return updated, digest


def main(argv: list[str] | None = None) -> int:
    previous = pilot._BASE_LOAD_RUN_SPEC  # noqa: SLF001
    pilot._BASE_LOAD_RUN_SPEC = _load_source_run_spec  # noqa: SLF001
    try:
        return pilot.main(argv)
    finally:
        pilot._BASE_LOAD_RUN_SPEC = previous  # noqa: SLF001


if __name__ == "__main__":
    raise SystemExit(main())
