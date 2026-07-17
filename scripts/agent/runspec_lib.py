#!/usr/bin/env python3
"""Shared helpers for DRPO server-side RunSpec execution.

RunSpec is intentionally small: it is a file-based contract that tells a local
AI executor exactly which lane it may serve, which existing script to run, and
which artifacts may be packaged.  The helpers here are conservative by design:
invalid or ambiguous specs fail closed instead of inviting an executor to guess.
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

try:
    import yaml
except Exception as exc:  # pragma: no cover - PyYAML is expected in DRPO envs.
    raise SystemExit("PyYAML is required for RunSpec tooling") from exc

RUNSPEC_VERSION = 1
STATE_DIRNAME = ".runspec_state"
ARTIFACT_DIRNAME = "runspec_artifacts"
DEFAULT_LANE_FILE = ".agent_lane.yaml"
READY_DIR = Path("runspecs") / "ready"
CLAIMED_DIR = Path(STATE_DIRNAME) / "claimed"
RUNNING_DIR = Path(STATE_DIRNAME) / "running"
DONE_DIR = Path(STATE_DIRNAME) / "done"
FAILED_DIR = Path(STATE_DIRNAME) / "failed"

# The allow-list does the real work; this deny-list is a second safety rail.
DEFAULT_EXCLUDE_PATTERNS = [
    "**/*.pt",
    "**/*.pth",
    "**/*.ckpt",
    "**/*.safetensors",
    "**/*.bin",
    "**/checkpoint*",
    "**/checkpoints/**",
    "**/model*",
    "**/models/**",
    "**/optimizer*",
    "**/wandb/**",
    "**/__pycache__/**",
]
MODEL_LIKE_SUFFIXES = {
    ".pt",
    ".pth",
    ".ckpt",
    ".safetensors",
    ".bin",
}
RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
LANE_RE = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")
EXPERIMENT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
ENV_ASSIGNMENT_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$")


class RunSpecError(RuntimeError):
    """Expected RunSpec validation or execution failure."""


@dataclass(frozen=True)
class RepoContext:
    root: Path

    @classmethod
    def from_path(cls, root: Path | str | None = None) -> "RepoContext":
        candidate = Path(root or ".").resolve()
        return cls(root=candidate)

    def repo_path(self, value: str | Path) -> Path:
        relative = safe_relpath(value)
        return self.root / Path(relative.as_posix())


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_yaml(path: Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load(path.read_text())
    except FileNotFoundError as exc:
        raise RunSpecError(f"missing YAML file: {path}") from exc
    except Exception as exc:
        raise RunSpecError(f"invalid YAML file {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise RunSpecError(f"YAML file must contain a mapping: {path}")
    return data


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def safe_relpath(value: str | Path) -> PurePosixPath:
    text = str(value).strip()
    if not text:
        raise RunSpecError("empty repository-relative path")
    rel = PurePosixPath(text)
    if rel.is_absolute() or ".." in rel.parts:
        raise RunSpecError(f"unsafe repository-relative path: {value}")
    return rel


def git_text(repo: Path, *args: str, check: bool = True) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if check and proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip()
        raise RunSpecError(f"git command failed: git {' '.join(args)}\n{detail}")
    return proc.stdout.strip()


def current_commit(repo: Path) -> str:
    return git_text(repo, "rev-parse", "HEAD")


def is_ancestor(repo: Path, maybe_ancestor: str, commit: str = "HEAD") -> bool:
    proc = subprocess.run(
        ["git", "-C", str(repo), "merge-base", "--is-ancestor", maybe_ancestor, commit],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return proc.returncode == 0


def load_lane_config(repo: Path, lane: str | None = None, lane_file: str = DEFAULT_LANE_FILE) -> dict[str, Any]:
    config_path = repo / lane_file
    config: dict[str, Any] = {}
    if config_path.is_file():
        config = read_yaml(config_path)
        if not isinstance(config, dict):
            raise RunSpecError(f"lane config must be a mapping: {config_path}")
    if lane:
        if config.get("lane") and config["lane"] != lane and config.get("forbid_cross_lane", True):
            raise RunSpecError(
                f"requested lane={lane} conflicts with workspace lane={config['lane']}"
            )
        config["lane"] = lane
    if not config.get("lane"):
        raise RunSpecError(
            f"no lane supplied and {lane_file} is missing or does not define lane"
        )
    lane_value = str(config["lane"])
    if not LANE_RE.match(lane_value):
        raise RunSpecError(f"invalid lane: {lane_value}")
    config["lane"] = lane_value
    config.setdefault("forbid_cross_lane", True)
    config.setdefault("allowed_experiment_prefixes", [])
    config.setdefault("forbidden_experiment_prefixes", [])
    return config


def ensure_experiment_allowed(experiment_id: str, lane_config: dict[str, Any]) -> None:
    allowed = list(lane_config.get("allowed_experiment_prefixes") or [])
    forbidden = list(lane_config.get("forbidden_experiment_prefixes") or [])
    for prefix in forbidden:
        if experiment_id.startswith(str(prefix)):
            raise RunSpecError(
                f"experiment_id {experiment_id} is forbidden by prefix {prefix}"
            )
    if allowed and not any(experiment_id.startswith(str(prefix)) for prefix in allowed):
        raise RunSpecError(
            f"experiment_id {experiment_id} does not match allowed prefixes {allowed}"
        )


def registry_contains_experiment(repo: Path, experiment_id: str) -> bool:
    registry = repo / "experiments" / "registry.yaml"
    if not registry.is_file():
        return False
    return experiment_id in registry.read_text(errors="replace")


def resolve_cwd(repo: Path, cwd_value: str | None) -> Path:
    if cwd_value in {None, "", "repo_root"}:
        return repo
    return repo / Path(safe_relpath(cwd_value).as_posix())


def command_tokens(command: str) -> list[str]:
    try:
        tokens = shlex.split(command)
    except ValueError as exc:
        raise RunSpecError(f"invalid shell command: {exc}") from exc
    if not tokens:
        raise RunSpecError("entrypoint command is empty")
    return tokens


def split_command_env(command: str) -> tuple[dict[str, str], list[str]]:
    """Split leading shell-style NAME=value tokens without invoking a shell."""
    tokens = command_tokens(command)
    environment: dict[str, str] = {}
    index = 0
    while index < len(tokens):
        match = ENV_ASSIGNMENT_RE.fullmatch(tokens[index])
        if match is None:
            break
        environment[match.group(1)] = match.group(2)
        index += 1
    argv = tokens[index:]
    if not argv:
        raise RunSpecError(
            "entrypoint command must include an executable after environment assignments"
        )
    return environment, argv


def command_script_path(command: str) -> str | None:
    _, tokens = split_command_env(command)
    first = tokens[0]
    if first in {"bash", "sh", "python", "python3"} and len(tokens) >= 2:
        candidate = tokens[1]
    else:
        candidate = first
    # Ignore flags and module execution; those are not existing-script contracts.
    if candidate.startswith("-"):
        return None
    if candidate == "-m":
        return None
    if "/" not in candidate and not candidate.endswith((".sh", ".py")):
        return None
    return candidate


def validate_existing_script(repo: Path, command: str, *, field: str) -> None:
    script = command_script_path(command)
    if not script:
        raise RunSpecError(
            f"{field} must point to an existing script; got command={command!r}"
        )
    path = repo / Path(safe_relpath(script).as_posix())
    if not path.is_file():
        raise RunSpecError(f"{field} script does not exist: {script}")


def normalize_spec(raw: dict[str, Any]) -> dict[str, Any]:
    spec = dict(raw)
    if "version" not in spec:
        spec["version"] = RUNSPEC_VERSION
    return spec


def validate_runspec(
    repo: Path,
    spec_path: Path,
    *,
    lane_config: dict[str, Any] | None = None,
    require_registry: bool = True,
) -> dict[str, Any]:
    spec = normalize_spec(read_yaml(spec_path))
    version = spec.get("version")
    if version != RUNSPEC_VERSION:
        raise RunSpecError(f"unsupported RunSpec version: {version}")

    run_id = str(spec.get("run_id") or "")
    lane = str(spec.get("lane") or "")
    experiment_id = str(spec.get("experiment_id") or "")
    if not RUN_ID_RE.match(run_id):
        raise RunSpecError(f"invalid run_id: {run_id}")
    if not LANE_RE.match(lane):
        raise RunSpecError(f"invalid lane: {lane}")
    if not EXPERIMENT_ID_RE.match(experiment_id):
        raise RunSpecError(f"invalid experiment_id: {experiment_id}")

    if lane_config:
        expected_lane = lane_config["lane"]
        if lane != expected_lane and lane_config.get("forbid_cross_lane", True):
            raise RunSpecError(f"RunSpec lane={lane} does not match workspace lane={expected_lane}")
        ensure_experiment_allowed(experiment_id, lane_config)

    if require_registry and not registry_contains_experiment(repo, experiment_id):
        raise RunSpecError(f"experiment_id not found in experiments/registry.yaml: {experiment_id}")

    repo_commit = str(spec.get("repo_commit") or "").strip()
    if repo_commit:
        head = current_commit(repo)
        if repo_commit != head and not is_ancestor(repo, repo_commit, "HEAD"):
            raise RunSpecError(
                f"repo_commit {repo_commit} is not current HEAD and is not an ancestor of HEAD {head}"
            )

    entrypoint = spec.get("entrypoint")
    if not isinstance(entrypoint, dict):
        raise RunSpecError("entrypoint must be a mapping")
    command = str(entrypoint.get("command") or "").strip()
    if not command:
        raise RunSpecError("entrypoint.command is required")
    resolve_cwd(repo, entrypoint.get("cwd") or "repo_root")

    policy = spec.get("policy") or {}
    if not isinstance(policy, dict):
        raise RunSpecError("policy must be a mapping")
    for key in [
        "existing_script_required",
        "forbid_new_launcher",
        "forbid_hparam_change",
        "forbid_cross_lane",
    ]:
        if policy.get(key) is not True:
            raise RunSpecError(f"policy.{key} must be true")
    if policy.get("existing_script_required", True):
        validate_existing_script(repo, command, field="entrypoint.command")

    outputs = spec.get("outputs") or {}
    if not isinstance(outputs, dict):
        raise RunSpecError("outputs must be a mapping")
    for key, value in outputs.items():
        if value in {None, ""}:
            continue
        if isinstance(value, str):
            safe_relpath(value)

    artifacts = spec.get("artifacts") or {}
    if not isinstance(artifacts, dict):
        raise RunSpecError("artifacts must be a mapping")
    package_policy = artifacts.get("package_policy", "manifest_only")
    if package_policy != "manifest_only":
        raise RunSpecError("only artifacts.package_policy=manifest_only is supported in v1")
    includes = artifacts.get("include") or []
    excludes = artifacts.get("exclude") or []
    if not isinstance(includes, list) or not all(isinstance(x, str) for x in includes):
        raise RunSpecError("artifacts.include must be a list of paths/globs")
    if not includes:
        raise RunSpecError("artifacts.include must be non-empty")
    if not isinstance(excludes, list) or not all(isinstance(x, str) for x in excludes):
        raise RunSpecError("artifacts.exclude must be a list of paths/globs")
    for pattern in includes + excludes:
        # Glob patterns are still repo-relative and must not escape the repo.
        safe_relpath(pattern.replace("**/", "").replace("*", "x") or "x")
    max_size = int(artifacts.get("max_package_size_mb", 100))
    if max_size <= 0:
        raise RunSpecError("artifacts.max_package_size_mb must be positive")

    return spec


def spec_filename(run_id: str) -> str:
    return f"{run_id}.yaml"


def state_path(repo: Path, state_dir: Path, run_id: str) -> Path:
    return repo / state_dir / spec_filename(run_id)


def iter_ready_specs(repo: Path) -> Iterable[Path]:
    ready = repo / READY_DIR
    if not ready.is_dir():
        return []
    return sorted(path for path in ready.glob("*.yaml") if path.is_file())


def claim_next_runspec(
    repo: Path,
    *,
    lane_config: dict[str, Any],
    run_id: str | None = None,
) -> Path:
    candidates: list[tuple[int, str, str, Path, dict[str, Any]]] = []
    errors: list[str] = []
    for path in iter_ready_specs(repo):
        try:
            spec = validate_runspec(repo, path, lane_config=lane_config)
        except RunSpecError as exc:
            errors.append(f"{path}: {exc}")
            continue
        if spec["lane"] != lane_config["lane"]:
            continue
        if run_id and spec["run_id"] != run_id:
            continue
        priority = int(spec.get("priority", 0) or 0)
        created_at = str(spec.get("created_at") or "")
        candidates.append((-priority, created_at, spec["run_id"], path, spec))
    if not candidates:
        detail = ""
        if errors:
            detail = "\nRejected candidates:\n" + "\n".join(errors[:10])
        target = f" run_id={run_id}" if run_id else ""
        raise RunSpecError(f"no READY RunSpec for lane={lane_config['lane']}{target}{detail}")
    _, _, selected_run_id, selected_path, spec = sorted(candidates)[0]
    claimed = state_path(repo, CLAIMED_DIR, selected_run_id)
    if claimed.exists():
        raise RunSpecError(f"RunSpec already claimed locally: {claimed}")
    claimed.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(spec)
    payload.setdefault("claim", {})
    payload["claim"] = {
        "claimed_at": now_utc(),
        "source_path": selected_path.relative_to(repo).as_posix(),
        "lane": lane_config["lane"],
    }
    write_yaml(claimed, payload)
    return claimed


def move_state(repo: Path, current: Path, target_dir: Path, status: dict[str, Any]) -> Path:
    target = state_path(repo, target_dir, status["run_id"])
    target.parent.mkdir(parents=True, exist_ok=True)
    data = read_yaml(current)
    data.setdefault("status", {})
    data["status"].update(status)
    write_yaml(target, data)
    if current.resolve() != target.resolve() and current.exists():
        current.unlink()
    return target


def match_any(path: str, patterns: Iterable[str]) -> bool:
    return any(fnmatch.fnmatch(path, pattern) for pattern in patterns)


def is_model_like(path: str) -> bool:
    pure = PurePosixPath(path)
    lowered = path.lower()
    if pure.suffix.lower() in MODEL_LIKE_SUFFIXES:
        return True
    parts = [part.lower() for part in pure.parts]
    return any(
        part.startswith("checkpoint")
        or part.startswith("model")
        or part.startswith("optimizer")
        for part in parts
    )


def expand_include(repo: Path, patterns: list[str]) -> list[Path]:
    matched: list[Path] = []
    seen: set[str] = set()
    for pattern in patterns:
        rel = safe_relpath(pattern.replace("*", "x") or "x")
        _ = rel  # validates the pattern shape while preserving the original glob.
        for path in sorted(repo.glob(pattern)):
            if path.is_file():
                key = path.resolve().as_posix()
                if key not in seen:
                    matched.append(path)
                    seen.add(key)
    return matched


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_artifacts(repo: Path, spec_path: Path, *, output_dir: Path | None = None) -> dict[str, Any]:
    spec = validate_runspec(repo, spec_path, require_registry=False)
    artifacts = spec["artifacts"]
    include_patterns = list(artifacts.get("include") or [])
    exclude_patterns = list(DEFAULT_EXCLUDE_PATTERNS) + list(artifacts.get("exclude") or [])
    fail_if_excluded = bool(artifacts.get("fail_if_excluded_matched", True))
    fail_if_too_large = bool(artifacts.get("fail_if_package_too_large", True))
    max_size_bytes = int(artifacts.get("max_package_size_mb", 100)) * 1024 * 1024

    candidates = expand_include(repo, include_patterns)
    if not candidates:
        raise RunSpecError("artifacts.include matched no files")

    included: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    total = 0
    for path in candidates:
        rel = path.relative_to(repo).as_posix()
        blocked = match_any(rel, exclude_patterns) or is_model_like(rel)
        row = {
            "path": rel,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        if blocked:
            excluded.append(row)
        else:
            included.append(row)
            total += row["size_bytes"]

    if excluded and fail_if_excluded:
        raise RunSpecError(
            "artifact include matched excluded/model-like files: "
            + ", ".join(row["path"] for row in excluded[:20])
        )
    if total > max_size_bytes and fail_if_too_large:
        raise RunSpecError(
            f"artifact package would be too large: {total} bytes > {max_size_bytes} bytes"
        )
    if not included:
        raise RunSpecError("no artifact files remain after exclude/model filters")

    output_root = output_dir or (repo / ARTIFACT_DIRNAME)
    output_root.mkdir(parents=True, exist_ok=True)
    run_id = spec["run_id"]
    zip_path = output_root / f"{run_id}_results.zip"
    manifest_path = output_root / f"{run_id}_manifest.json"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for row in included:
            archive.write(repo / row["path"], row["path"])
    manifest = {
        "schema_version": 1,
        "created_at": now_utc(),
        "run_id": run_id,
        "lane": spec["lane"],
        "experiment_id": spec["experiment_id"],
        "repo_commit": current_commit(repo),
        "package_policy": artifacts.get("package_policy", "manifest_only"),
        "zip_path": zip_path.relative_to(repo).as_posix()
        if zip_path.is_relative_to(repo)
        else str(zip_path),
        "zip_sha256": sha256_file(zip_path),
        "total_included_size_bytes": total,
        "included": included,
        "excluded": excluded,
        "include_patterns": include_patterns,
        "exclude_patterns": exclude_patterns,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def check_success_criteria(repo: Path, spec: dict[str, Any]) -> list[str]:
    missing: list[str] = []
    outputs = spec.get("outputs") or {}
    for key in ["summary_file", "audit_file"]:
        value = outputs.get(key)
        if value and not (repo / safe_relpath(value)).exists():
            missing.append(f"outputs.{key} missing: {value}")
    for item in spec.get("success_criteria") or []:
        if not isinstance(item, str):
            continue
        text = item.strip()
        if text.endswith(" exists"):
            path_text = text[: -len(" exists")].strip()
            if path_text and not (repo / safe_relpath(path_text)).exists():
                missing.append(f"success criterion missing: {text}")
    return missing


def run_entrypoint(repo: Path, spec_path: Path, *, log_dir: Path | None = None) -> dict[str, Any]:
    spec = validate_runspec(repo, spec_path, require_registry=False)
    entrypoint = spec["entrypoint"]
    command = entrypoint["command"]
    cwd = resolve_cwd(repo, entrypoint.get("cwd") or "repo_root")
    command_env, tokens = split_command_env(command)
    process_env = os.environ.copy()
    process_env.update(command_env)
    state_log_dir = log_dir or (repo / STATE_DIRNAME / "logs" / spec["run_id"])
    state_log_dir.mkdir(parents=True, exist_ok=True)
    command_path = state_log_dir / "COMMAND.txt"
    stdout_path = state_log_dir / "STDOUT.log"
    stderr_path = state_log_dir / "STDERR.log"
    command_path.write_text(command + "\n", encoding="utf-8")
    started = time.time()
    with stdout_path.open("w", encoding="utf-8") as out, stderr_path.open("w", encoding="utf-8") as err:
        proc = subprocess.run(
            tokens,
            cwd=cwd,
            env=process_env,
            stdout=out,
            stderr=err,
            check=False,
        )
    elapsed = time.time() - started
    result = {
        "run_id": spec["run_id"],
        "returncode": proc.returncode,
        "elapsed_seconds": round(elapsed, 3),
        "command": command,
        "stdout": stdout_path.relative_to(repo).as_posix()
        if stdout_path.is_relative_to(repo)
        else str(stdout_path),
        "stderr": stderr_path.relative_to(repo).as_posix()
        if stderr_path.is_relative_to(repo)
        else str(stderr_path),
    }
    status_file = state_log_dir / "RUN_STATUS.json"
    status_file.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if proc.returncode != 0:
        raise RunSpecError(f"entrypoint failed with returncode={proc.returncode}")
    missing = check_success_criteria(repo, spec)
    if missing:
        raise RunSpecError("success criteria failed: " + "; ".join(missing))
    return result


def json_main(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def handle_cli_error(exc: Exception, *, json_output: bool = False) -> int:
    if json_output:
        json_main({"status": "FAIL", "error": str(exc)})
    else:
        print(f"ERROR: {exc}", file=sys.stderr)
    return 1


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo-root", default=".", help="Repository root (default: .)")
    parser.add_argument("--lane", default=None, help="Executor lane override")
    parser.add_argument(
        "--lane-file",
        default=DEFAULT_LANE_FILE,
        help="Workspace lane config file (default: .agent_lane.yaml)",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON")
