#!/usr/bin/env python3
"""Blind manuscript replay harness for label-to-outline evaluation.

The replay protocol separates three phases:

* label phase: a label paper may be read to extract or improve an outline;
* blind generation phase: once the outline is frozen, generation may read only
  files copied into a manifest-bound workspace;
* evaluation phase: the label may be read again only after generation.

This script implements the auditable blind-generation boundary.  It does not
judge manuscript quality and it does not make scientific-result claims.  Its
job is to prove which files were available to blueprint/prose generation and to
fail closed if label papers, old prose/blueprints, or old paper drafts leak into
that phase.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

SCHEMA_VERSION = 1
MANIFEST_NAME = "BLIND_INPUT_MANIFEST.json"
AUDIT_NAME = "BLIND_REPLAY_AUDIT.json"
DEFAULT_GENERATED_DIR = "generated"

FORBIDDEN_GENERATION_PREFIXES = (
    "paper/releases/",
    "paper/core_review_v2_core/",
    "paper/publication_quality_v1/",
    "paper/overleaf/sections/",
    "paper/overleaf/generated/",
    "paper/overleaf/main.tex",
    "paper/overleaf/main_replacement.tex",
    "paper/overleaf/main.pdf",
    "docs/manuscript/paper_graph.yaml",
    "docs/paper_rewrite_blueprint",
    "docs/paper_rewrite_intro_blueprint",
    "docs/paper_rewrite_prose",
)

DEFAULT_SAFE_INPUTS = (
    "docs/handoff.md",
    "experiments/registry.yaml",
    "docs/manuscript/claim_evidence_matrix.yaml",
    "docs/manuscript/RL_PAPER_WRITING_GUIDANCE.md",
    "docs/manuscript/RL_PAPER_WRITING_PLAYBOOK.md",
    "paper/overleaf/references.bib",
)

OUTLINE_BEGIN_RE = re.compile(r"^<!--\s*MANUSCRIPT:BEGIN\s+([A-Z0-9-]+)\s*-->\s*$")
OUTLINE_END_RE = re.compile(r"^<!--\s*MANUSCRIPT:END\s+([A-Z0-9-]+)\s*-->\s*$")
OUTLINE_TITLE_RE = re.compile(r"^##\s+\[([A-Z0-9-]+)\]\s+(.+?)\s*$")
OUTLINE_FIELD_RE = re.compile(
    r"^\*\*(Claim|Reader question|Role|Required evidence|Must include|Must avoid):\*\*\s*(.*)$"
)
TEXT_SUFFIXES = {
    ".bib",
    ".cls",
    ".csv",
    ".json",
    ".md",
    ".py",
    ".sty",
    ".tex",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}


class BlindReplayError(RuntimeError):
    """Expected fail-closed blind replay error."""


@dataclass(frozen=True)
class InputFile:
    role: str
    repo_path: str
    workspace_path: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class OutlineNode:
    node_id: str
    section: str
    title: str
    claim: str
    reader_question: str
    role: str
    required_evidence: list[str]
    must_include: list[str]
    must_avoid: list[str]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_relative(value: str, *, label: str = "path") -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise BlindReplayError(f"unsafe {label}: {value!r}")
    return path


def resolve_repo_path(repo: Path, relative: str) -> Path:
    rel = safe_relative(relative).as_posix()
    target = (repo / rel).resolve()
    try:
        target.relative_to(repo.resolve())
    except ValueError as exc:
        raise BlindReplayError(f"path escapes repository: {relative}") from exc
    return target


def repo_relative(repo: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(repo.resolve()).as_posix()
    except ValueError as exc:
        raise BlindReplayError(f"path is not under repository: {path}") from exc


def is_forbidden_generation_path(relative: str, extra_forbidden: Iterable[str] = ()) -> bool:
    normalized = safe_relative(relative).as_posix()
    for token in (*FORBIDDEN_GENERATION_PREFIXES, *tuple(extra_forbidden)):
        cleaned = str(token).strip().replace("\\", "/")
        if not cleaned:
            continue
        if cleaned.endswith("/") and normalized.startswith(cleaned):
            return True
        if normalized == cleaned or normalized.startswith(cleaned):
            return True
    return False


def ensure_clean_workspace(workspace: Path) -> None:
    if workspace.exists() and any(workspace.iterdir()):
        raise BlindReplayError(f"workspace already exists and is not empty: {workspace}")
    workspace.mkdir(parents=True, exist_ok=True)


def iter_input_files(repo: Path, relative: str) -> list[Path]:
    source = resolve_repo_path(repo, relative)
    if source.is_symlink():
        raise BlindReplayError(f"symlink inputs are forbidden: {relative}")
    if source.is_file():
        return [source]
    if source.is_dir():
        files: list[Path] = []
        for path in sorted(source.rglob("*")):
            if path.is_symlink():
                raise BlindReplayError(f"symlink inputs are forbidden: {repo_relative(repo, path)}")
            if path.is_file():
                files.append(path)
        if not files:
            raise BlindReplayError(f"input directory is empty: {relative}")
        return files
    raise BlindReplayError(f"input path does not exist: {relative}")


def git_head(repo: Path) -> str | None:
    proc = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    value = proc.stdout.strip()
    if proc.returncode == 0 and re.fullmatch(r"[0-9a-f]{40}", value):
        return value
    return None


def copy_allowed_inputs(
    *,
    repo: Path,
    workspace: Path,
    optimized_outline: str,
    allow_inputs: list[str],
    extra_forbidden: list[str],
) -> list[InputFile]:
    inputs_root = workspace / "inputs"
    seen: set[str] = set()
    rows: list[InputFile] = []
    requested = [optimized_outline, *allow_inputs]
    for requested_relative in requested:
        for source in iter_input_files(repo, requested_relative):
            relative = repo_relative(repo, source)
            if relative in seen:
                continue
            seen.add(relative)
            if is_forbidden_generation_path(relative, extra_forbidden):
                raise BlindReplayError(
                    "blind generation input is forbidden after outline freeze: " + relative
                )
            destination = inputs_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            role = "optimized_outline" if relative == repo_relative(
                repo, resolve_repo_path(repo, optimized_outline)
            ) else "context"
            rows.append(
                InputFile(
                    role=role,
                    repo_path=relative,
                    workspace_path=destination.relative_to(workspace).as_posix(),
                    sha256=sha256_file(destination),
                    size_bytes=destination.stat().st_size,
                )
            )
    if not any(row.role == "optimized_outline" for row in rows):
        raise BlindReplayError("optimized outline was not copied into the blind workspace")
    return rows


def write_manifest(
    *,
    repo: Path,
    workspace: Path,
    inputs: list[InputFile],
    label_source: str | None,
    sentinels: list[str],
    extra_forbidden: list[str],
) -> dict[str, Any]:
    if label_source is not None:
        label_relative = repo_relative(repo, resolve_repo_path(repo, label_source))
    else:
        label_relative = None
    payload = {
        "schema_version": SCHEMA_VERSION,
        "protocol": "paper_label_outline_blind_replay",
        "phase": "generation_only_after_outline_freeze",
        "repo_head": git_head(repo),
        "workspace": str(workspace),
        "label_source": {
            "path": label_relative,
            "allowed_phase": "outline_extraction_and_post_generation_evaluation_only",
            "copied_into_generation_workspace": False,
        },
        "input_policy": {
            "rule": "blueprint/prose/tex generation may read only listed workspace inputs",
            "forbidden_generation_prefixes": [
                *FORBIDDEN_GENERATION_PREFIXES,
                *extra_forbidden,
            ],
            "forbidden_reason": (
                "label papers, existing prose, existing blueprints, release PDFs, "
                "and Overleaf section drafts would contaminate a replay score"
            ),
        },
        "sentinels": sentinels,
        "allowed_inputs": [row.__dict__ for row in inputs],
        "generated_root": DEFAULT_GENERATED_DIR,
    }
    (workspace / MANIFEST_NAME).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def load_manifest(workspace: Path) -> dict[str, Any]:
    manifest_path = workspace / MANIFEST_NAME
    if not manifest_path.is_file():
        raise BlindReplayError(f"missing blind input manifest: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        raise BlindReplayError("unsupported blind input manifest schema")
    return payload


def verify_manifest_inputs(workspace: Path, manifest: dict[str, Any]) -> None:
    rows = manifest.get("allowed_inputs")
    if not isinstance(rows, list) or not rows:
        raise BlindReplayError("manifest allowed_inputs must be a non-empty list")
    optimized_count = 0
    forbidden = manifest.get("input_policy", {}).get("forbidden_generation_prefixes", [])
    if not isinstance(forbidden, list):
        raise BlindReplayError("manifest forbidden_generation_prefixes must be a list")
    for item in rows:
        if not isinstance(item, dict):
            raise BlindReplayError("invalid manifest input entry")
        repo_rel = str(item.get("repo_path", ""))
        workspace_rel = str(item.get("workspace_path", ""))
        if is_forbidden_generation_path(repo_rel, forbidden):
            raise BlindReplayError(f"manifest contains forbidden input: {repo_rel}")
        path = workspace / safe_relative(workspace_rel, label="workspace_path")
        if not path.is_file():
            raise BlindReplayError(f"manifest input is missing from workspace: {workspace_rel}")
        if sha256_file(path) != item.get("sha256") or path.stat().st_size != item.get(
            "size_bytes"
        ):
            raise BlindReplayError(f"manifest input checksum/size mismatch: {workspace_rel}")
        if item.get("role") == "optimized_outline":
            optimized_count += 1
    if optimized_count != 1:
        raise BlindReplayError("manifest must contain exactly one optimized_outline input")


def manifest_outline_path(workspace: Path, manifest: dict[str, Any]) -> Path:
    for item in manifest.get("allowed_inputs", []):
        if isinstance(item, dict) and item.get("role") == "optimized_outline":
            return workspace / safe_relative(str(item["workspace_path"]), label="workspace_path")
    raise BlindReplayError("manifest does not identify the optimized outline")


def parse_list_field(lines: list[str], start: int) -> tuple[list[str], int]:
    values: list[str] = []
    index = start
    while index < len(lines):
        line = lines[index]
        if line.startswith("- "):
            values.append(line[2:].strip())
            index += 1
            continue
        if not line.strip():
            index += 1
            continue
        break
    return values, index


def parse_outline(path: Path) -> list[OutlineNode]:
    lines = path.read_text(encoding="utf-8").splitlines()
    section = ""
    nodes: list[OutlineNode] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if line.startswith("# "):
            section = line[2:].strip()
            index += 1
            continue
        begin = OUTLINE_BEGIN_RE.match(line)
        if begin is None:
            index += 1
            continue
        node_id = begin.group(1)
        block: list[str] = []
        index += 1
        while index < len(lines):
            end = OUTLINE_END_RE.match(lines[index])
            if end is not None:
                if end.group(1) != node_id:
                    raise BlindReplayError(f"outline end marker mismatch for {node_id}")
                break
            block.append(lines[index])
            index += 1
        else:
            raise BlindReplayError(f"outline node has no end marker: {node_id}")
        nodes.append(parse_outline_block(node_id=node_id, section=section, block=block))
        index += 1
    if not nodes:
        raise BlindReplayError(f"outline contains no stable-ID nodes: {path}")
    return nodes


def parse_outline_block(*, node_id: str, section: str, block: list[str]) -> OutlineNode:
    if not block:
        raise BlindReplayError(f"empty outline node: {node_id}")
    title = OUTLINE_TITLE_RE.match(block[0])
    if title is None or title.group(1) != node_id:
        raise BlindReplayError(f"outline node lacks canonical title: {node_id}")
    fields: dict[str, Any] = {}
    current_key: str | None = None
    buffer: list[str] = []

    def flush() -> None:
        nonlocal current_key, buffer
        if current_key is None:
            return
        if current_key in {"required_evidence", "must_include", "must_avoid"}:
            fields[current_key] = [line[2:].strip() for line in buffer if line.startswith("- ")]
        else:
            fields[current_key] = " ".join(line.strip() for line in buffer if line.strip())
        current_key = None
        buffer = []

    for line in block[1:]:
        match = OUTLINE_FIELD_RE.match(line)
        if match is not None:
            flush()
            current_key = match.group(1).lower().replace(" ", "_")
            initial = match.group(2).strip()
            buffer = [initial] if initial else []
            continue
        if current_key is not None:
            buffer.append(line)
    flush()
    missing = [
        key
        for key in (
            "claim",
            "reader_question",
            "role",
            "required_evidence",
            "must_include",
            "must_avoid",
        )
        if key not in fields
    ]
    if missing:
        raise BlindReplayError(f"outline node {node_id} is missing fields: {missing}")
    return OutlineNode(
        node_id=node_id,
        section=section,
        title=title.group(2).strip(),
        claim=str(fields["claim"]).strip(),
        reader_question=str(fields["reader_question"]).strip(),
        role=str(fields["role"]).strip(),
        required_evidence=list(fields["required_evidence"]),
        must_include=list(fields["must_include"]),
        must_avoid=list(fields["must_avoid"]),
    )


def render_blind_scaffold(workspace: Path, manifest: dict[str, Any]) -> list[str]:
    verify_manifest_inputs(workspace, manifest)
    outline_path = manifest_outline_path(workspace, manifest)
    nodes = parse_outline(outline_path)
    generated = workspace / str(manifest.get("generated_root", DEFAULT_GENERATED_DIR))
    generated.mkdir(parents=True, exist_ok=True)

    blueprint_lines = [
        "# Blind replay blueprint scaffold",
        "",
        "This file is generated only from BLIND_INPUT_MANIFEST allowed inputs.",
        "It is a scaffold for replay evaluation, not a paper-quality final draft.",
        "",
    ]
    prose_lines = [
        "# Blind replay prose scaffold",
        "",
        "This prose is intentionally generated without reading label papers or old drafts.",
        "",
    ]
    tex_lines = [
        r"\documentclass{article}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage[utf8]{inputenc}",
        r"\title{Blind Replay Scaffold}",
        r"\date{}",
        r"\begin{document}",
        r"\maketitle",
        "",
    ]
    for node in nodes:
        blueprint_lines.extend(
            [
                f"## {node.node_id} - {node.title}",
                "",
                f"- Section: {node.section}",
                f"- Reader question: {node.reader_question}",
                f"- Claim to realize: {node.claim}",
                f"- Role: {node.role}",
                "- Required evidence:",
                *[f"  - {item}" for item in node.required_evidence],
                "- Must include:",
                *[f"  - {item}" for item in node.must_include],
                "- Must avoid:",
                *[f"  - {item}" for item in node.must_avoid],
                "- Replay instruction: write from this outline node and manifest inputs only.",
                "",
            ]
        )
        prose_lines.extend(
            [
                f"## [{node.node_id}] {node.title}",
                "",
                f"{node.claim} This paragraph should answer: {node.reader_question}.",
                "",
            ]
        )
        tex_lines.extend(
            [
                f"\\section*{{{tex_escape(node.title)}}}",
                tex_escape(node.claim),
                "",
            ]
        )
    tex_lines.append(r"\end{document}")

    (generated / "replay_blueprint.md").write_text(
        "\n".join(blueprint_lines).rstrip() + "\n",
        encoding="utf-8",
    )
    (generated / "replay_prose.md").write_text(
        "\n".join(prose_lines).rstrip() + "\n",
        encoding="utf-8",
    )
    (generated / "replay_main.tex").write_text(
        "\n".join(tex_lines).rstrip() + "\n",
        encoding="utf-8",
    )
    return [
        (generated / "replay_blueprint.md").relative_to(workspace).as_posix(),
        (generated / "replay_prose.md").relative_to(workspace).as_posix(),
        (generated / "replay_main.tex").relative_to(workspace).as_posix(),
    ]


def tex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def text_files_under(root: Path) -> list[Path]:
    if not root.exists():
        return []
    if root.is_file():
        return [root] if root.suffix.lower() in TEXT_SUFFIXES else []
    files: list[Path] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES:
            files.append(path)
    return files


def audit_workspace(workspace: Path, generated_root: Path | None = None) -> dict[str, Any]:
    manifest = load_manifest(workspace)
    verify_manifest_inputs(workspace, manifest)
    relative_root = str(manifest.get("generated_root", DEFAULT_GENERATED_DIR))
    generated = generated_root or (workspace / relative_root)
    if not generated.exists():
        raise BlindReplayError(f"generated root does not exist: {generated}")
    if generated.is_symlink():
        raise BlindReplayError(f"generated root must not be a symlink: {generated}")

    forbidden = manifest.get("input_policy", {}).get("forbidden_generation_prefixes", [])
    sentinels = manifest.get("sentinels", [])
    if not isinstance(forbidden, list) or not isinstance(sentinels, list):
        raise BlindReplayError("manifest forbidden prefixes and sentinels must be lists")
    leakages: list[dict[str, str]] = []
    text_files = text_files_under(generated)
    for path in text_files:
        text = path.read_text(encoding="utf-8", errors="replace")
        relative = path.relative_to(workspace).as_posix() if path.is_relative_to(workspace) else str(path)
        for token in forbidden:
            cleaned = str(token).strip()
            if cleaned and cleaned in text:
                leakages.append({"file": relative, "kind": "forbidden_path_token", "token": cleaned})
        for sentinel in sentinels:
            cleaned = str(sentinel).strip()
            if cleaned and cleaned in text:
                leakages.append({"file": relative, "kind": "sentinel", "token": cleaned})
    status = "PASS" if not leakages else "FAIL"
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "checked_text_files": [
            path.relative_to(workspace).as_posix() if path.is_relative_to(workspace) else str(path)
            for path in text_files
        ],
        "leakages": leakages,
    }
    (workspace / AUDIT_NAME).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if leakages:
        raise BlindReplayError(f"blind replay leakage detected: {leakages}")
    return payload


def cmd_init(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo_root.resolve()
    workspace = args.workspace.resolve()
    ensure_clean_workspace(workspace)
    allow_inputs = list(args.allow_input or [])
    if args.include_default_drpo_context:
        allow_inputs.extend(path for path in DEFAULT_SAFE_INPUTS if (repo / path).exists())
    inputs = copy_allowed_inputs(
        repo=repo,
        workspace=workspace,
        optimized_outline=args.optimized_outline,
        allow_inputs=allow_inputs,
        extra_forbidden=list(args.forbid_input or []),
    )
    manifest = write_manifest(
        repo=repo,
        workspace=workspace,
        inputs=inputs,
        label_source=args.label_source,
        sentinels=list(args.sentinel or []),
        extra_forbidden=list(args.forbid_input or []),
    )
    return {"status": "PASS", "workspace": str(workspace), "allowed_inputs": len(inputs), **manifest}


def cmd_scaffold(args: argparse.Namespace) -> dict[str, Any]:
    workspace = args.workspace.resolve()
    manifest = load_manifest(workspace)
    generated = render_blind_scaffold(workspace, manifest)
    return {"status": "PASS", "workspace": str(workspace), "generated": generated}


def cmd_audit(args: argparse.Namespace) -> dict[str, Any]:
    workspace = args.workspace.resolve()
    root = args.generated_root.resolve() if args.generated_root is not None else None
    return audit_workspace(workspace, root)


def cmd_all(args: argparse.Namespace) -> dict[str, Any]:
    init_payload = cmd_init(args)
    scaffold_payload = cmd_scaffold(args)
    audit_payload = audit_workspace(args.workspace.resolve())
    return {
        "status": "PASS",
        "workspace": str(args.workspace.resolve()),
        "init_allowed_inputs": init_payload["allowed_inputs"],
        "generated": scaffold_payload["generated"],
        "audit": audit_payload,
    }


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--optimized-outline", required=True)
    parser.add_argument("--label-source")
    parser.add_argument("--allow-input", action="append", default=[])
    parser.add_argument("--forbid-input", action="append", default=[])
    parser.add_argument("--sentinel", action="append", default=[])
    parser.add_argument("--include-default-drpo-context", action="store_true")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    init_parser = sub.add_parser("init", help="create a manifest-bound blind workspace")
    add_common_arguments(init_parser)
    scaffold_parser = sub.add_parser(
        "scaffold", help="render deterministic blueprint/prose/tex from the frozen outline"
    )
    scaffold_parser.add_argument("--workspace", type=Path, required=True)
    audit_parser = sub.add_parser("audit", help="audit generated outputs for leakage")
    audit_parser.add_argument("--workspace", type=Path, required=True)
    audit_parser.add_argument("--generated-root", type=Path)
    all_parser = sub.add_parser("all", help="init, scaffold, and audit a blind replay workspace")
    add_common_arguments(all_parser)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "init":
            payload = cmd_init(args)
        elif args.command == "scaffold":
            payload = cmd_scaffold(args)
        elif args.command == "audit":
            payload = cmd_audit(args)
        elif args.command == "all":
            payload = cmd_all(args)
        else:
            raise BlindReplayError(f"unknown command: {args.command}")
    except (BlindReplayError, OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
