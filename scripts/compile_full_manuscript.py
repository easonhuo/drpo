#!/usr/bin/env python3
"""Compatibility entrypoint for the repository's manuscript release manifest.

The orchestration engine is domain agnostic. Project-specific figures, tables,
proofs, terminology, and output names are declared in the release manifest and
its asset-builder plugin.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any


def _release_module() -> ModuleType:
    path = Path(__file__).with_name("manuscript_release_pipeline.py")
    spec = importlib.util.spec_from_file_location("manuscript_release_pipeline_core", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load release pipeline: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate(root: Path, config: dict[str, Any]) -> None:
    _release_module().validate_release(root, config)


def main(argv: list[str] | None = None) -> int:
    module = _release_module()
    return module.main(
        argv,
        default_config=Path("docs/manuscript/full_paper_assets.yaml"),
        default_output=Path("paper/releases/DRPO_FULL_REVIEW_V1.pdf"),
    )


if __name__ == "__main__":
    raise SystemExit(main())
