#!/usr/bin/env python3
"""Build the deterministic Stage 4B lossless module-source shadow candidate."""

from __future__ import annotations
import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import build_stage4_context as stage4a  # noqa: E402

DEFAULT_CONFIG = Path("docs/handoff_shadow/stage4/candidate/STAGE4B_CONFIG.yaml")
EXPECTED_OUTPUT = Path("docs/handoff_shadow/stage4/candidate/generated")
HEADING_RE = re.compile(r"^#{1,6}\s+.+")
MARKER_START_RE = re.compile(r"^<!-- HANDOFF-DELTA-BLOCK:(.+):START -->$")
SOURCE_START_RE = re.compile(rb"^<!-- STAGE4B-SOURCE-BLOCK:(B\d{6}):START -->\n", re.M)


class Stage4BError(ValueError):
    pass


@dataclass(frozen=True)
class Block:
    block_id: str
    ordinal: int
    start: int
    end: int
    start_line: int
    end_line_exclusive: int
    payload: bytes
    kind: str
    marker_id: str | None
    first_nonblank: str
    heading_titles: tuple[str, ...]


@dataclass(frozen=True)
class OwnedBlock:
    block: Block
    owner_type: str
    owner: str
    candidate_path: str
    basis: str
    candidates: tuple[str, ...]
    references: tuple[str, ...]


@dataclass(frozen=True)
class Plan:
    outputs: dict[Path, bytes]
    source_bytes: bytes
    source_hash: str
    registry_hash: str
    ownership: tuple[OwnedBlock, ...]
    module_order: tuple[str, ...]
    dependency_hash: str
    semantic_graph_hash: str
    coverage: dict[str, Any]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_hash(value: Any) -> str:
    return sha256_bytes(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    )


def json_bytes(v: Any) -> bytes:
    return (json.dumps(v, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()


def yaml_bytes(v: Any) -> bytes:
    return yaml.safe_dump(v, sort_keys=False, allow_unicode=True, width=120).encode()


def load_yaml(path: Path, label: str) -> dict[str, Any]:
    try:
        v = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise Stage4BError(f"cannot read {label} {path}: {e}") from e
    if not isinstance(v, dict):
        raise Stage4BError(f"{label} must be a mapping")
    return v


def load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        v = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise Stage4BError(f"cannot read {label} {path}: {e}") from e
    if not isinstance(v, dict):
        raise Stage4BError(f"{label} must be an object")
    return v


def safe_file(repo: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise Stage4BError(f"{label} must be a non-empty repository-relative path")
    rel = Path(value)
    if rel.is_absolute() or ".." in rel.parts:
        raise Stage4BError(f"unsafe {label}: {value!r}")
    path = (repo / rel).resolve()
    try:
        path.relative_to(repo.resolve())
    except ValueError as e:
        raise Stage4BError(f"{label} escapes repository") from e
    if not path.is_file() or path.is_symlink():
        raise Stage4BError(f"missing or unsafe {label}: {value}")
    return path


def validate_config(repo: Path, path: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    c = load_yaml(path, "Stage 4B config")
    req = {
        "schema_version": 1,
        "policy_id": "GOV-HANDOFF-INDEX-01",
        "authority": "non_authoritative_stage4b_shadow_candidate",
        "research_master": "docs/handoff.md",
        "registry": "experiments/registry.yaml",
        "output_root": EXPECTED_OUTPUT.as_posix(),
        "manual_handoff_remains_authoritative": True,
        "authority_cutover_allowed": False,
    }
    for k, v in req.items():
        if c.get(k) != v:
            raise Stage4BError(f"Stage 4B config {k} must remain {v!r}")
    if c.get("single_write_source") != {
        "current": "docs/handoff.md",
        "candidate_editable": False,
        "generated_compatibility_handoff_editable": False,
    }:
        raise Stage4BError("single-write-source contract changed")
    files = {
        "handoff": safe_file(repo, c["research_master"], "research master"),
        "registry": safe_file(repo, c["registry"], "registry"),
        "modules": safe_file(repo, c.get("stage4a_modules"), "Stage 4A modules"),
        "dependencies": safe_file(repo, c.get("stage4a_dependencies"), "Stage 4A dependencies"),
        "module_index": safe_file(repo, c.get("stage4a_module_index"), "Stage 4A module index"),
        "semantic_manifest": safe_file(
            repo, c.get("stage4a_semantic_manifest"), "Stage 4A semantic manifest"
        ),
    }
    p = c.get("owner_priority")
    if not isinstance(p, list) or not p or len(p) != len(set(p)):
        raise Stage4BError("owner_priority must be a non-empty unique list")
    h = c.get("history_owner")
    if (
        not isinstance(h, dict)
        or h.get("owner_type") != "history_record"
        or not str(h.get("path", "")).startswith("history/archive/")
    ):
        raise Stage4BError("history_owner must define a history_record below history/archive")
    return c, files


def parse_marker_ranges(lines: list[str]) -> dict[int, tuple[str, int]]:
    out = {}
    seen = set()
    i = 0
    while i < len(lines):
        m = MARKER_START_RE.match(lines[i].rstrip("\r\n"))
        if not m:
            i += 1
            continue
        marker = m.group(1)
        if marker in seen:
            raise Stage4BError(f"duplicate HANDOFF-DELTA-BLOCK: {marker}")
        seen.add(marker)
        end = f"<!-- HANDOFF-DELTA-BLOCK:{marker}:END -->"
        j = i + 1
        while j < len(lines) and lines[j].rstrip("\r\n") != end:
            if MARKER_START_RE.match(lines[j].rstrip("\r\n")):
                raise Stage4BError(f"nested marker block: {marker}")
            j += 1
        if j >= len(lines):
            raise Stage4BError(f"unterminated marker block: {marker}")
        out[i] = (marker, j + 1)
        i = j + 1
    return out


def partition(source: bytes) -> tuple[Block, ...]:
    try:
        text = source.decode("utf-8")
    except UnicodeDecodeError as e:
        raise Stage4BError(f"handoff must be UTF-8: {e}") from e
    lines = text.splitlines(keepends=True)
    raw = source.splitlines(keepends=True)
    markers = parse_marker_ranges(lines)
    inside = [False] * len(lines)
    for s, (_, e) in markers.items():
        for i in range(s, e):
            inside[i] = True
    bounds = {0, len(lines)}
    for s, (_, e) in markers.items():
        bounds.update((s, e))
    for i, line in enumerate(lines):
        if not inside[i] and HEADING_RE.match(line.rstrip("\r\n")):
            bounds.add(i)
    ordered = sorted(bounds)
    offs = [0]
    for x in raw:
        offs.append(offs[-1] + len(x))
    blocks = []
    for ordinal, (s, e) in enumerate(zip(ordered, ordered[1:]), 1):
        if s == e:
            continue
        payload = source[offs[s] : offs[e]]
        marker = markers.get(s, (None, None))[0]
        heads = tuple(x.rstrip("\r\n") for x in lines[s:e] if HEADING_RE.match(x.rstrip("\r\n")))
        first = next((x.strip() for x in lines[s:e] if x.strip()), "")
        blocks.append(
            Block(
                f"B{ordinal:06d}",
                ordinal,
                offs[s],
                offs[e],
                s + 1,
                e + 1,
                payload,
                "marker_block" if marker else ("heading_block" if heads else "preamble"),
                marker,
                first,
                heads,
            )
        )
    if b"".join(x.payload for x in blocks) != source:
        raise Stage4BError("source partition is not byte exact")
    return tuple(blocks)


def stage4a_metadata(repo: Path, files: dict[str, Path]):
    modules, _ = stage4a.normalize_modules(stage4a.load_yaml(files["modules"], "Stage 4A modules"))
    ids = [m["module_id"] for m in modules]
    deps, _ = stage4a.normalize_dependencies(
        stage4a.load_yaml(files["dependencies"], "Stage 4A dependencies"), ids
    )
    coverage = {mid: [] for mid in ids}
    refs = {mid: {"experiments": [], "development_experiment_registrations": []} for mid in ids}
    registry_owner = {}
    for m in modules:
        mid = m["module_id"]
        for source in m["sources"]:
            if source.get("kind") == "registry_entries":
                col = source.get("collection", "experiments")
                for eid in source.get("experiment_ids", []):
                    key = (col, eid)
                    if key in registry_owner:
                        raise Stage4BError(f"registry entry has multiple owners: {col}:{eid}")
                    registry_owner[key] = mid
                    refs[mid].setdefault(col, []).append(eid)
            else:
                for chunk in stage4a.extract_source(repo, source):
                    coverage[mid].extend(chunk.spans)
    index = load_json(files["module_index"], "Stage 4A module index")
    if [x.get("module_id") for x in index.get("modules", [])] != ids:
        raise Stage4BError("Stage 4A module IDs/order drifted")
    sem = load_json(files["semantic_manifest"], "Stage 4A semantic manifest")
    sem_hash = sem.get("graph_hash")
    if not isinstance(sem_hash, str) or len(sem_hash) != 64:
        raise Stage4BError("Stage 4A semantic graph hash missing")
    dep_hash = canonical_hash({mid: list(deps[mid]) for mid in ids})
    return modules, deps, coverage, refs, dep_hash, sem_hash


def assign(config, source, blocks, modules, coverage):
    ids = [m["module_id"] for m in modules]
    priority = config["owner_priority"]
    if set(priority) != set(ids):
        raise Stage4BError("owner_priority must exactly cover accepted Stage 4A modules")
    rank = {x: i for i, x in enumerate(priority)}
    fallback = config.get("current_fallback_owner")
    if fallback not in ids:
        raise Stage4BError("current_fallback_owner is unknown")
    lines = source.decode().splitlines()
    matches = [i + 1 for i, x in enumerate(lines) if x == config.get("history_boundary")]
    if len(matches) != 1:
        raise Stage4BError("history_boundary must match exactly once")
    history_line = matches[0]
    h = config["history_owner"]
    out = []
    for b in blocks:
        span = stage4a.SourceSpan("docs/handoff.md", b.start_line - 1, b.end_line_exclusive - 1)
        cand = tuple(mid for mid in ids if any(x.contains(span) for x in coverage[mid]))
        if cand:
            owner = min(cand, key=lambda x: rank[x])
            otype = "canonical_module"
            path = f"canonical/{owner}.md"
            basis = (
                "unique_stage4a_source_owner"
                if len(cand) == 1
                else "explicit_owner_priority_resolution"
            )
        elif b.start_line >= history_line:
            owner = h["owner"]
            otype = "history_record"
            path = h["path"]
            basis = "explicit_post_boundary_history_fallback"
        else:
            owner = fallback
            otype = "canonical_module"
            path = f"canonical/{owner}.md"
            basis = "explicit_current_fallback_owner"
        out.append(
            OwnedBlock(b, otype, owner, path, basis, cand, tuple(x for x in cand if x != owner))
        )
    if b"".join(x.block.payload for x in out) != source:
        raise Stage4BError("ownership does not cover source exactly")
    return tuple(out)


def render_owner(title, otype, owner, responsibility, deps, refs, topics, items):
    if not items:
        raise Stage4BError(f"owner has no content: {owner}")
    lines = [
        f"# {title}",
        "",
        "> Stage 4B lossless source-promotion shadow candidate.",
        "> Generated from `docs/handoff.md`; do not edit while the manual handoff remains authoritative.",
        "",
        f"- Owner type: `{otype}`",
        f"- Owner ID: `{owner}`",
        f"- Responsibility: {responsibility}",
        f"- Dependencies: {', '.join(f'`{x}`' for x in deps) or 'none'}",
        f"- Content-contract topics: {', '.join(f'`{x}`' for x in topics) or 'none'}",
        f"- Owned source blocks: {len(items)}",
        "- Registry references are pointers only; `experiments/registry.yaml` remains the sole editable registry source.",
        "",
        "## Registry references",
        "",
        f"- `experiments`: {', '.join(f'`{x}`' for x in refs.get('experiments', [])) or 'none'}",
        f"- `development_experiment_registrations`: {', '.join(f'`{x}`' for x in refs.get('development_experiment_registrations', [])) or 'none'}",
        "",
        "## Owned source blocks",
        "",
    ]
    chunks = [("\n".join(lines) + "\n").encode()]
    for x in sorted(items, key=lambda z: z.block.ordinal):
        bid = x.block.block_id
        chunks.append(f"<!-- STAGE4B-SOURCE-BLOCK:{bid}:START -->\n".encode())
        chunks.append(x.block.payload)
        if not x.block.payload.endswith(b"\n"):
            raise Stage4BError(f"source block lacks terminal newline: {bid}")
        chunks.append(f"<!-- STAGE4B-SOURCE-BLOCK:{bid}:END -->\n".encode())
    return b"".join(chunks)


def extract_blocks(payload: bytes, label: str) -> dict[str, bytes]:
    out = {}
    pos = 0
    while True:
        m = SOURCE_START_RE.search(payload, pos)
        if not m:
            break
        bid = m.group(1).decode()
        endm = f"<!-- STAGE4B-SOURCE-BLOCK:{bid}:END -->\n".encode()
        end = payload.find(endm, m.end())
        if end < 0 or bid in out:
            raise Stage4BError(f"invalid source block {bid} in {label}")
        out[bid] = payload[m.end() : end]
        pos = end + len(endm)
    return out


def build_plan(repo: Path, config_path: Path = DEFAULT_CONFIG) -> Plan:
    repo = repo.resolve()
    config_path = config_path if config_path.is_absolute() else repo / config_path
    c, files = validate_config(repo, config_path)
    source = files["handoff"].read_bytes()
    reg = files["registry"].read_bytes()
    blocks = partition(source)
    modules, deps, coverage, refs, dh, sh = stage4a_metadata(repo, files)
    owned = assign(c, source, blocks, modules, coverage)
    ids = tuple(m["module_id"] for m in modules)
    by = {}
    for x in owned:
        by.setdefault(x.candidate_path, []).append(x)
    outputs = {}
    for m in modules:
        mid = m["module_id"]
        outputs[Path(f"canonical/{mid}.md")] = render_owner(
            m["title"],
            "canonical_module",
            mid,
            m["responsibility"],
            deps[mid],
            refs[mid],
            [t["topic_id"] for t in m["content_contract"]],
            by.get(f"canonical/{mid}.md", []),
        )
    h = c["history_owner"]
    outputs[Path(h["path"])] = render_owner(
        "Immutable legacy handoff history",
        "history_record",
        h["owner"],
        "Preserve byte-exact historical source payloads outside promoted module ownership.",
        (),
        {},
        [],
        by.get(h["path"], []),
    )
    records = []
    for x in owned:
        b = x.block
        records.append(
            {
                "block_id": b.block_id,
                "ordinal": b.ordinal,
                "source": {
                    "path": "docs/handoff.md",
                    "start_line": b.start_line,
                    "end_line_exclusive": b.end_line_exclusive,
                    "bytes": len(b.payload),
                    "sha256": sha256_bytes(b.payload),
                    "kind": b.kind,
                    "marker_id": b.marker_id,
                    "heading_titles": list(b.heading_titles),
                    "first_nonblank": b.first_nonblank,
                },
                "owner_type": x.owner_type,
                "owner": x.owner,
                "candidate_path": x.candidate_path,
                "ownership_basis": x.basis,
                "stage4a_candidate_modules": list(x.candidates),
                "references": list(x.references),
            }
        )
    outputs[Path("manifests/OWNERSHIP.yaml")] = yaml_bytes(
        {
            "schema_version": 1,
            "policy_id": "GOV-HANDOFF-INDEX-01",
            "authority": "shadow_only",
            "source_path": "docs/handoff.md",
            "source_sha256": sha256_bytes(source),
            "single_owner_required": True,
            "owners": records,
        }
    )
    canon = [x for x in owned if x.owner_type == "canonical_module"]
    hist = [x for x in owned if x.owner_type == "history_record"]
    cov = {
        "schema_version": 1,
        "policy_id": "GOV-HANDOFF-INDEX-01",
        "authority": "shadow_only",
        "source": {
            "path": "docs/handoff.md",
            "sha256": sha256_bytes(source),
            "bytes": len(source),
            "lines": len(source.decode().splitlines()),
        },
        "block_count": len(owned),
        "owned_bytes": len(source),
        "canonical_block_count": len(canon),
        "canonical_bytes": sum(len(x.block.payload) for x in canon),
        "history_block_count": len(hist),
        "history_bytes": sum(len(x.block.payload) for x in hist),
        "module_block_counts": {mid: sum(x.owner == mid for x in canon) for mid in ids},
        "unmapped_count": 0,
        "multi_owner_conflict_count": 0,
        "unresolved_overlap_count": 0,
        "missing_history_count": 0,
        "exact_partition": True,
        "exact_reconstruction": True,
        "compatibility_handoff_exact_match": True,
    }
    outputs[Path("manifests/COVERAGE.json")] = json_bytes(cov)
    outputs[Path("manifests/REGISTRY_REFERENCES.yaml")] = yaml_bytes(
        {
            "schema_version": 1,
            "policy_id": "GOV-HANDOFF-INDEX-01",
            "authority": "registry_references_only",
            "registry_path": "experiments/registry.yaml",
            "registry_sha256": sha256_bytes(reg),
            "modules": [{"module_id": mid, **refs[mid]} for mid in ids],
        }
    )
    outputs[Path("manifests/LINEAGE.json")] = json_bytes(
        {
            "schema_version": 1,
            "policy_id": "GOV-HANDOFF-INDEX-01",
            "authority": "shadow_only",
            "inherits_stage4a_module_ids": list(ids),
            "depends_on": {mid: list(deps[mid]) for mid in ids},
            "dependency_hash": dh,
            "semantic_graph_hash": sh,
            "ownership_overlap_resolutions": [
                {
                    "block_id": x.block.block_id,
                    "owner": x.owner,
                    "references": list(x.references),
                    "basis": x.basis,
                }
                for x in owned
                if x.references
            ],
            "taxonomy_changes": [],
        }
    )
    recon = [
        {
            "block_id": x.block.block_id,
            "ordinal": x.block.ordinal,
            "candidate_path": x.candidate_path,
            "payload_sha256": sha256_bytes(x.block.payload),
            "bytes": len(x.block.payload),
        }
        for x in owned
    ]
    outputs[Path("manifests/RECONSTRUCTION.json")] = json_bytes(
        {
            "schema_version": 1,
            "policy_id": "GOV-HANDOFF-INDEX-01",
            "source_path": "docs/handoff.md",
            "source_sha256": sha256_bytes(source),
            "output_path": "generated/handoff_compat.md",
            "output_sha256": sha256_bytes(source),
            "byte_identical_required": True,
            "blocks": recon,
        }
    )
    outputs[Path("generated/handoff_compat.md")] = source
    extracted = {}
    for p, data in outputs.items():
        if p.parts[0] not in {"canonical", "history"}:
            continue
        for bid, payload in extract_blocks(data, p.as_posix()).items():
            if bid in extracted:
                raise Stage4BError(f"block has multiple owner files: {bid}")
            extracted[bid] = payload
    bids = [x.block.block_id for x in owned]
    if set(extracted) != set(bids) or b"".join(extracted[x] for x in bids) != source:
        raise Stage4BError("owner files cannot reconstruct handoff byte-for-byte")
    entries = [
        {"path": p.as_posix(), "sha256": sha256_bytes(data), "bytes": len(data)}
        for p, data in sorted(outputs.items(), key=lambda x: x[0].as_posix())
    ]
    outputs[Path("manifests/GENERATED_FILES.json")] = json_bytes(
        {
            "schema_version": 1,
            "policy_id": "GOV-HANDOFF-INDEX-01",
            "authority": "shadow_only_generated_files",
            "source_sha256": sha256_bytes(source),
            "registry_sha256": sha256_bytes(reg),
            "files_excluding_this_manifest": entries,
            "tree_hash": canonical_hash(entries),
        }
    )
    return Plan(outputs, source, sha256_bytes(source), sha256_bytes(reg), owned, ids, dh, sh, cov)


def existing_files(root: Path) -> set[Path]:
    if not root.exists():
        return set()
    out = set()
    for p in root.rglob("*"):
        if p.is_symlink():
            raise Stage4BError(f"generated output may not contain symlinks: {p}")
        if p.is_file():
            out.add(p.relative_to(root))
    return out


def check_generated(root: Path, plan: Plan) -> list[str]:
    exp, act = set(plan.outputs), existing_files(root)
    issues = [f"missing generated file: {p}" for p in sorted(exp - act)] + [
        f"unexpected generated file: {p}" for p in sorted(act - exp)
    ]
    for p in sorted(exp & act):
        if (root / p).read_bytes() != plan.outputs[p]:
            issues.append(f"stale or tampered generated file: {p}")
    return issues


def write_generated(root: Path, plan: Plan) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    expected = set(plan.outputs)
    written = []
    reused = []
    removed = []
    for rel in sorted(existing_files(root) - expected):
        (root / rel).unlink()
        removed.append(rel.as_posix())
    for rel, data in sorted(plan.outputs.items(), key=lambda x: x[0].as_posix()):
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.is_file() and p.read_bytes() == data:
            reused.append(rel.as_posix())
        else:
            tmp = p.with_name(p.name + ".stage4b-write")
            tmp.write_bytes(data)
            tmp.replace(p)
            written.append(rel.as_posix())
    return {
        "status": "PASS",
        "written": written,
        "reused": reused,
        "removed": removed,
        "file_count": len(plan.outputs),
        "source_sha256": plan.source_hash,
    }


def reconstruct_from_generated(root: Path) -> bytes:
    m = load_json(root / "manifests/RECONSTRUCTION.json", "reconstruction manifest")
    chunks = []
    cache = {}
    seen = set()
    expected = 1
    for e in m.get("blocks", []):
        bid, ordv, cand = e.get("block_id"), e.get("ordinal"), e.get("candidate_path")
        if bid in seen or ordv != expected or not isinstance(cand, str):
            raise Stage4BError("invalid reconstruction order or duplicate")
        seen.add(bid)
        expected += 1
        if cand not in cache:
            p = root / cand
            if not p.is_file() or p.is_symlink():
                raise Stage4BError(f"missing or unsafe owner file: {cand}")
            cache[cand] = extract_blocks(p.read_bytes(), cand)
        payload = cache[cand].get(bid)
        if (
            payload is None
            or sha256_bytes(payload) != e.get("payload_sha256")
            or len(payload) != e.get("bytes")
        ):
            raise Stage4BError(f"reconstruction payload mismatch: {bid}")
        chunks.append(payload)
    result = b"".join(chunks)
    if sha256_bytes(result) != m.get("output_sha256"):
        raise Stage4BError("reconstruction hash mismatch")
    return result


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, default=Path.cwd())
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--reconstruct-output", type=Path)
    args = ap.parse_args(argv)
    repo = args.repo_root.resolve()
    try:
        cp = args.config if args.config.is_absolute() else repo / args.config
        c = load_yaml(cp, "Stage 4B config")
        if c.get("output_root") != EXPECTED_OUTPUT.as_posix():
            raise Stage4BError("output_root escaped the frozen shadow directory")
        root = repo / EXPECTED_OUTPUT
        plan = build_plan(repo, cp)
        if args.check:
            issues = check_generated(root, plan)
            if issues:
                raise Stage4BError("; ".join(issues))
            if reconstruct_from_generated(root) != plan.source_bytes:
                raise Stage4BError("generated reconstruction differs from handoff")
            print(
                f"Stage 4B candidate: PASS (blocks={len(plan.ownership)}, files={len(plan.outputs)}, source_sha256={plan.source_hash})"
            )
        else:
            print(
                json.dumps(
                    write_generated(root, plan), ensure_ascii=False, indent=2, sort_keys=True
                )
            )
        if args.reconstruct_output:
            target = (
                args.reconstruct_output
                if args.reconstruct_output.is_absolute()
                else repo / args.reconstruct_output
            )
            if target.resolve() in {
                (repo / "docs/handoff.md").resolve(),
                (repo / "experiments/registry.yaml").resolve(),
            }:
                raise Stage4BError("reconstruction output may not overwrite authoritative inputs")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(plan.source_bytes)
    except (Stage4BError, stage4a.ContextBuildError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
