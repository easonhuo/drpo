#!/usr/bin/env python3
"""Domain-agnostic manuscript release orchestrator.

A release manifest supplies project-specific renderers, asset builders, template
settings, proof obligations, and output names. The core contains no scientific
claims or project terminology.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


class ReleaseBuildError(RuntimeError):
    pass


_TEX_INCLUDE_RE = re.compile(r"\\(?:input|include)\{([^{}]+)\}")


def read_yaml(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ReleaseBuildError(f"cannot read release manifest {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReleaseBuildError("release manifest root must be a mapping")
    return payload


def run(command: list[str], cwd: Path) -> str:
    proc = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if proc.returncode:
        raise ReleaseBuildError(f"command failed ({' '.join(command)}):\n{proc.stdout}")
    return proc.stdout


def resolve_command(raw: Any, root: Path) -> list[str]:
    if not isinstance(raw, list) or not raw or not all(isinstance(x, str) for x in raw):
        raise ReleaseBuildError("release command must be a non-empty string list")
    replacements = {"{python}": sys.executable, "{repo_root}": str(root)}
    return [replacements.get(item, item.replace("{repo_root}", str(root))) for item in raw]


def resolve_active_template_source(main_path: Path, template_root: Path) -> str:
    """Collect literal TeX source reachable from the configured entrypoint.

    Repository entrypoints may be thin wrappers around a replacement source.
    Follow only existing literal ``\\input`` and ``\\include`` targets that stay
    inside the configured template root. Macro paths, missing optional inputs,
    cycles, and paths outside the root are ignored.
    """

    root = template_root.resolve()
    if not main_path.is_file():
        raise ReleaseBuildError(f"configured main TeX file is missing: {main_path}")
    visited: set[Path] = set()
    chunks: list[str] = []

    def visit(path: Path) -> None:
        resolved = path.resolve()
        try:
            resolved.relative_to(root)
        except ValueError:
            return
        if resolved in visited or not resolved.is_file():
            return
        visited.add(resolved)
        source = resolved.read_text(encoding="utf-8")
        chunks.append(source)
        for raw_target in _TEX_INCLUDE_RE.findall(source):
            target = raw_target.strip()
            if not target or "\\" in target or "#" in target:
                continue
            candidate = resolved.parent / target
            if not candidate.suffix:
                candidate = candidate.with_suffix(".tex")
            visit(candidate)

    visit(main_path)
    return "\n".join(chunks)


def build_assets(root: Path, manifest: dict[str, Any]) -> None:
    render_command = manifest.get("graph_render_command")
    if render_command:
        run(resolve_command(render_command, root), root)
    commands = manifest.get("asset_build_commands", [])
    if not isinstance(commands, list):
        raise ReleaseBuildError("asset_build_commands must be a list")
    for command in commands:
        run(resolve_command(command, root), root)


def inject_assets(root: Path, manifest: dict[str, Any]) -> None:
    template = manifest["paper_template"]
    overleaf = root / template["root"]
    mapping = manifest.get("assets", {})
    if not isinstance(mapping, dict):
        raise ReleaseBuildError("assets must be a node-to-file mapping")
    files = list((overleaf / "sections").glob("*.tex")) + list(
        (overleaf / "appendix").glob("*.tex")
    )
    for path in files:
        text = path.read_text(encoding="utf-8")
        for node, assets in mapping.items():
            marker = f"% END-MANUSCRIPT-NODE: {node}"
            if marker not in text:
                continue
            if not isinstance(assets, list) or not all(isinstance(x, str) for x in assets):
                raise ReleaseBuildError(f"assets for {node} must be a string list")
            addition = "\n".join(f"\\input{{{asset}}}" for asset in assets)
            text = text.replace(marker, marker + "\n" + addition)
        path.write_text(text, encoding="utf-8")


def run_quality_gate(root: Path, manifest: dict[str, Any]) -> None:
    gate = manifest.get("quality_gate")
    if not isinstance(gate, dict):
        raise ReleaseBuildError("quality_gate must be configured")
    command = [
        sys.executable,
        str(gate["script"]),
        "all",
        "--repo-root",
        str(root),
        "--contract",
        str(gate["contract"]),
    ]
    optional_flags = {
        "graph": "--graph",
        "quality_profile": "--quality-profile",
        "project_profile": "--project-profile",
        "output": "--output",
    }
    for key, flag in optional_flags.items():
        if gate.get(key):
            command.extend([flag, str(gate[key])])
    run(command, root)


def validate_release(root: Path, manifest: dict[str, Any]) -> None:
    template = manifest["paper_template"]
    overleaf = root / template["root"]
    main_path = root / template["main_tex"]
    active_source = resolve_active_template_source(main_path, overleaf)
    if (
        int(template.get("columns", 1)) == 2
        and "\\twocolumn" not in active_source
    ):
        raise ReleaseBuildError("configured two-column template is not active")
    family = str(template.get("family", "")).strip()
    if family and family not in active_source:
        raise ReleaseBuildError("configured template family is not active")
    all_tex = "\n".join(path.read_text(encoding="utf-8") for path in overleaf.rglob("*.tex"))
    for obligation in manifest.get("proof_obligations", []):
        if not isinstance(obligation, dict):
            raise ReleaseBuildError("proof obligations must be mappings")
        for label in obligation.values():
            if f"\\label{{{label}}}" not in all_tex:
                raise ReleaseBuildError(f"missing proof/statement label: {label}")
    bib_paths = [(root / template["bibliography"]).resolve()]
    bibliography_root = overleaf.resolve()
    for group in re.findall(r"\\bibliography\{([^}]+)\}", active_source):
        for raw_name in group.split(","):
            name = raw_name.strip()
            if not name or "\\" in name or "#" in name:
                continue
            candidate = (overleaf / name).resolve()
            if candidate.suffix != ".bib":
                candidate = candidate.with_suffix(".bib")
            try:
                candidate.relative_to(bibliography_root)
            except ValueError as exc:
                raise ReleaseBuildError(
                    f"bibliography escapes template root: {name}"
                ) from exc
            if candidate not in bib_paths:
                bib_paths.append(candidate)
    bib_chunks: list[str] = []
    for bib_path in bib_paths:
        if not bib_path.is_file():
            raise ReleaseBuildError(f"missing bibliography file: {bib_path}")
        bib_chunks.append(bib_path.read_text(encoding="utf-8"))
    bib = "\n".join(bib_chunks)
    keys = set(re.findall(r"@[A-Za-z]+\s*\{\s*([^,]+),", bib))
    used: set[str] = set()
    for group in re.findall(r"\\cite[a-zA-Z]*\{([^}]+)\}", all_tex):
        used.update(key.strip() for key in group.split(",") if key.strip())
    missing = sorted(used - keys)
    if missing:
        raise ReleaseBuildError("missing bibliography keys: " + ", ".join(missing))
    for key in manifest.get("required_citation_keys", []):
        if key not in used:
            raise ReleaseBuildError(f"required citation not used: {key}")
    tex_files = list(overleaf.rglob("*.tex"))
    for node, assets in manifest.get("assets", {}).items():
        for asset in assets:
            token = f"\\input{{{asset}}}"
            if not any(token in path.read_text(encoding="utf-8") for path in tex_files):
                raise ReleaseBuildError(f"asset mapping not injected for {node}: {asset}")
            if not (overleaf / asset).exists():
                raise ReleaseBuildError(f"missing asset {asset}")


def compile_pdf(root: Path, manifest: dict[str, Any], output: Path) -> Path:
    template = manifest["paper_template"]
    overleaf = root / template["root"]
    main_name = Path(template["main_tex"]).name
    latexmk = shutil.which("latexmk")
    bibtex = shutil.which("bibtex")
    if bibtex is None:
        fallback = Path("/usr/bin/bibtex.original")
        if fallback.is_file():
            bibtex = str(fallback)
    if latexmk is None:
        raise ReleaseBuildError("latexmk is unavailable; rerun with --skip-compile")
    if bibtex is None:
        raise ReleaseBuildError("bibtex is unavailable; rerun with --skip-compile")
    run([latexmk, "-C", main_name], overleaf)
    command = [
        latexmk,
        "-e",
        f'$bibtex="{bibtex} %O %B"',
        "-pdf",
        "-interaction=nonstopmode",
        "-halt-on-error",
        main_name,
    ]
    run(command, overleaf)
    run(command, overleaf)
    log_path = overleaf / (Path(main_name).stem + ".log")
    if log_path.exists():
        log = log_path.read_text(encoding="utf-8", errors="replace")
        failures = [
            token
            for token in (
                "There were undefined references",
                "There were undefined citations",
                "Overfull \\hbox",
                "Overfull \\vbox",
            )
            if token in log
        ]
        if failures:
            raise ReleaseBuildError("LaTeX quality audit failed: " + ", ".join(failures))
    source = overleaf / (Path(main_name).stem + ".pdf")
    if not source.is_file():
        raise ReleaseBuildError("TeX build completed without a PDF")
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, output)
    return output


def execute(
    root: Path,
    manifest: dict[str, Any],
    *,
    output: Path,
    skip_compile: bool,
) -> dict[str, Any]:
    build_assets(root, manifest)
    inject_assets(root, manifest)
    run_quality_gate(root, manifest)
    validate_release(root, manifest)
    result: dict[str, Any] = {
        "status": "PASS",
        "project_id": manifest.get("project_id"),
        "quality_gate": "PASS",
        "release_validation": "PASS",
    }
    if not skip_compile:
        result["pdf"] = str(compile_pdf(root, manifest, output))
    return result


def parser(
    default_config: Path = Path("docs/manuscript/full_paper_assets.yaml"),
    default_output: Path = Path("paper/releases/manuscript_review.pdf"),
) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, default=Path.cwd())
    ap.add_argument("--config", type=Path, default=default_config)
    ap.add_argument("--output", type=Path, default=default_output)
    ap.add_argument("--skip-compile", action="store_true")
    return ap


def main(
    argv: list[str] | None = None,
    *,
    default_config: Path = Path("docs/manuscript/full_paper_assets.yaml"),
    default_output: Path = Path("paper/releases/manuscript_review.pdf"),
) -> int:
    args = parser(default_config, default_output).parse_args(argv)
    root = args.repo_root.resolve()
    manifest = read_yaml((root / args.config).resolve())
    try:
        result = execute(
            root,
            manifest,
            output=(root / args.output).resolve(),
            skip_compile=args.skip_compile,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    except ReleaseBuildError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
