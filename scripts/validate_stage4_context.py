#!/usr/bin/env python3
"""Validate the Stage 4 minimal context core and its generated shadow outputs."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BUILDER_PATH = ROOT / "scripts" / "build_stage4_context.py"


def load_builder():
    spec = importlib.util.spec_from_file_location("stage4_context_builder", BUILDER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {BUILDER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BUILDER = load_builder()


def validate(repo_root: Path) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    plan = BUILDER.build_plan(repo_root)
    output_root = repo_root / BUILDER.DEFAULT_OUTPUT
    problems = BUILDER.check_generated(output_root, plan)
    if problems:
        raise BUILDER.ContextBuildError("; ".join(problems))
    handoff = repo_root / "docs/handoff.md"
    registry = repo_root / "experiments/registry.yaml"
    if not handoff.is_file() or not registry.is_file():
        raise BUILDER.ContextBuildError("authoritative handoff or registry is missing")
    return {
        "status": "PASS",
        "policy_id": "GOV-HANDOFF-INDEX-01",
        "authority": "non_authoritative_stage4_minimal_context_shadow",
        "manual_handoff_remains_authoritative": True,
        "authority_cutover_allowed": False,
        "module_count": len(plan.module_order),
        "edge_count": sum(len(values) for values in plan.dependencies.values()),
        "graph_hash": plan.graph_hash,
        "suggestion_count": len(plan.suggestions),
        "semantic_contract_module_count": sum(
            1 for snapshot in plan.snapshots.values() if snapshot.contract_topics
        ),
        "semantic_contract_topic_count": sum(
            len(snapshot.contract_topics) for snapshot in plan.snapshots.values()
        ),
        "semantic_contract_evidence_count": sum(
            len(snapshot.contract_evidence) for snapshot in plan.snapshots.values()
        ),
        "deduplicated_source_chunk_count": sum(
            len(snapshot.deduplicated_source_labels)
            for snapshot in plan.snapshots.values()
        ),
        "acceptance_targets": list(plan.acceptance_results),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = validate(args.repo_root)
    except (BUILDER.ContextBuildError, OSError, RuntimeError) as exc:
        if args.json:
            print(
                json.dumps(
                    {"status": "FAIL", "error": str(exc)},
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
            )
        else:
            print(f"Stage 4 minimal context validation: FAIL: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print("Stage 4 minimal context validation: PASS")
        for key, value in report.items():
            if key != "status":
                print(f"- {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
