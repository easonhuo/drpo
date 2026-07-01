#!/usr/bin/env python3
"""Validate the Stage 4B lossless module-source shadow candidate."""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Any
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import build_stage4b_candidate as builder  # noqa: E402


class ValidationError(ValueError):
    pass


def load_json(path: Path) -> dict[str, Any]:
    try:
        v = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise ValidationError(f"cannot read JSON {path}: {e}") from e
    if not isinstance(v, dict):
        raise ValidationError(f"JSON root must be an object: {path}")
    return v


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        v = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise ValidationError(f"cannot read YAML {path}: {e}") from e
    if not isinstance(v, dict):
        raise ValidationError(f"YAML root must be a mapping: {path}")
    return v


def validate(repo_root: Path) -> dict[str, Any]:
    repo = repo_root.resolve()
    plan = builder.build_plan(repo)
    root = repo / builder.EXPECTED_OUTPUT
    issues = builder.check_generated(root, plan)
    if issues:
        raise ValidationError("; ".join(issues))
    if builder.reconstruct_from_generated(root) != plan.source_bytes:
        raise ValidationError("reconstructed handoff is not byte-identical to docs/handoff.md")
    if (root / "generated/handoff_compat.md").read_bytes() != plan.source_bytes:
        raise ValidationError("compatibility handoff is not byte-identical to docs/handoff.md")
    cov = load_json(root / "manifests/COVERAGE.json")
    for k in (
        "unmapped_count",
        "multi_owner_conflict_count",
        "unresolved_overlap_count",
        "missing_history_count",
    ):
        if cov.get(k) != 0:
            raise ValidationError(f"coverage hard blocker {k}={cov.get(k)!r}")
    for k in ("exact_partition", "exact_reconstruction", "compatibility_handoff_exact_match"):
        if cov.get(k) is not True:
            raise ValidationError(f"coverage exactness flag is not true: {k}")
    if cov.get("block_count") != len(plan.ownership) or cov.get("owned_bytes") != len(
        plan.source_bytes
    ):
        raise ValidationError("coverage count drift")
    own = load_yaml(root / "manifests/OWNERSHIP.yaml")
    rows = own.get("owners")
    if own.get("single_owner_required") is not True or not isinstance(rows, list):
        raise ValidationError("ownership manifest does not enforce a single owner")
    ids = [x.get("block_id") for x in rows if isinstance(x, dict)]
    if (
        len(ids) != len(plan.ownership)
        or len(ids) != len(set(ids))
        or ids != [x.block.block_id for x in plan.ownership]
    ):
        raise ValidationError("ownership IDs/order drift")
    if own.get("source_sha256") != plan.source_hash:
        raise ValidationError("ownership source hash drift")
    reg = load_yaml(root / "manifests/REGISTRY_REFERENCES.yaml")
    if (
        reg.get("authority") != "registry_references_only"
        or reg.get("registry_sha256") != plan.registry_hash
    ):
        raise ValidationError("registry reference authority/hash drift")
    mods = reg.get("modules")
    if not isinstance(mods, list) or [x.get("module_id") for x in mods] != list(plan.module_order):
        raise ValidationError("registry module order drift")
    lin = load_json(root / "manifests/LINEAGE.json")
    if (
        lin.get("inherits_stage4a_module_ids") != list(plan.module_order)
        or lin.get("taxonomy_changes") != []
    ):
        raise ValidationError("Stage 4A taxonomy drift")
    if (
        lin.get("dependency_hash") != plan.dependency_hash
        or lin.get("semantic_graph_hash") != plan.semantic_graph_hash
    ):
        raise ValidationError("lineage hash drift")
    rec = load_json(root / "manifests/RECONSTRUCTION.json")
    if (
        rec.get("source_sha256") != plan.source_hash
        or rec.get("output_sha256") != plan.source_hash
        or rec.get("byte_identical_required") is not True
    ):
        raise ValidationError("reconstruction contract drift")
    gen = load_json(root / "manifests/GENERATED_FILES.json")
    entries = gen.get("files_excluding_this_manifest")
    expected = [
        {"path": p.as_posix(), "sha256": builder.sha256_bytes(data), "bytes": len(data)}
        for p, data in sorted(plan.outputs.items(), key=lambda x: x[0].as_posix())
        if p != Path("manifests/GENERATED_FILES.json")
    ]
    if entries != expected or gen.get("tree_hash") != builder.canonical_hash(expected):
        raise ValidationError("generated-file inventory/tree drift")
    c = load_yaml(repo / builder.DEFAULT_CONFIG)
    if (
        c.get("manual_handoff_remains_authoritative") is not True
        or c.get("authority_cutover_allowed") is not False
    ):
        raise ValidationError("authority boundary changed")
    if c.get("single_write_source") != {
        "current": "docs/handoff.md",
        "candidate_editable": False,
        "generated_compatibility_handoff_editable": False,
    }:
        raise ValidationError("single-write-source boundary changed")
    return {
        "status": "PASS",
        "policy_id": "GOV-HANDOFF-INDEX-01",
        "authority": "non_authoritative_stage4b_shadow_candidate",
        "manual_handoff_remains_authoritative": True,
        "authority_cutover_allowed": False,
        "module_count": len(plan.module_order),
        "block_count": len(plan.ownership),
        "generated_file_count": len(plan.outputs),
        "source_sha256": plan.source_hash,
        "registry_sha256": plan.registry_hash,
        "dependency_hash": plan.dependency_hash,
        "semantic_graph_hash": plan.semantic_graph_hash,
        "unmapped_count": 0,
        "multi_owner_conflict_count": 0,
        "missing_history_count": 0,
        "exact_reconstruction": True,
        "compatibility_handoff_exact_match": True,
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, default=Path("."))
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    try:
        r = validate(args.repo_root)
    except (ValidationError, builder.Stage4BError, OSError, RuntimeError) as e:
        if args.json:
            print(
                json.dumps(
                    {"status": "FAIL", "error": str(e)},
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
            )
        else:
            print(f"Stage 4B candidate validation: FAIL: {e}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(r, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print("Stage 4B candidate validation: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
