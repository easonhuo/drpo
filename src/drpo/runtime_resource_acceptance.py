"""Shared contracts for the DRPO runtime-resource acceptance harness."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import shutil
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from drpo.runtime_resource_autotune import atomic_write_json
from drpo.runtime_resource_pool import ResourcePoolError, parse_cpu_pool, parse_gpu_pool

PROFILE_SCHEMA_VERSION = 1
STATES = frozenset({"PASS", "FAIL", "BLOCKED", "INCONCLUSIVE", "NOT_RUN"})
PLACEHOLDER = "REPLACE_WITH_ABSOLUTE_PATH"
ALLOWED_SUFFIXES = frozenset(
    {".json", ".jsonl", ".csv", ".md", ".txt", ".log", ".sha256", ".yaml", ".yml"}
)


class AcceptanceError(RuntimeError):
    """Raised when the acceptance contract cannot be executed safely."""


@dataclasses.dataclass(frozen=True)
class StageResult:
    name: str
    status: str
    started_utc: str
    finished_utc: str
    details: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.status not in STATES:
            raise ValueError(f"unsupported stage status: {self.status}")

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def compact_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: object, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AcceptanceError(f"{context} must be an object")
    return dict(value)


def _closed(
    value: Mapping[str, Any],
    *,
    required: Iterable[str],
    optional: Iterable[str] = (),
    context: str,
) -> None:
    required_set = set(required)
    missing = sorted(required_set - set(value))
    unknown = sorted(set(value) - required_set - set(optional))
    if missing:
        raise AcceptanceError(f"{context} missing fields: {missing}")
    if unknown:
        raise AcceptanceError(f"{context} unknown fields: {unknown}")


def _absolute(value: object, context: str) -> str:
    if not isinstance(value, str) or not value.strip() or PLACEHOLDER in value:
        raise AcceptanceError(f"{context} must be a resolved absolute path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise AcceptanceError(f"{context} must be absolute")
    return str(path.resolve())


def _number(value: object, context: str, *, integer: bool = False) -> float | int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AcceptanceError(f"{context} must be positive")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise AcceptanceError(f"{context} must be finite and positive")
    if integer:
        if not isinstance(value, int):
            raise AcceptanceError(f"{context} must be an integer")
        return int(value)
    return number


def _fraction(value: object, context: str, *, allow_one: bool) -> float:
    number = float(_number(value, context))
    if number > 1 or (number == 1 and not allow_one):
        raise AcceptanceError(f"{context} has an invalid fraction")
    return number


def _full_sha(value: object, context: str) -> str:
    if not isinstance(value, str) or len(value) != 40:
        raise AcceptanceError(f"{context} must be a full Git SHA")
    if any(character not in "0123456789abcdef" for character in value.lower()):
        raise AcceptanceError(f"{context} must be a full Git SHA")
    return value


def _outside_repo(path: Path, repo: Path) -> bool:
    try:
        path.relative_to(repo)
        return False
    except ValueError:
        pass
    try:
        repo.relative_to(path)
        return False
    except ValueError:
        return True


def load_profile(path: str | Path, *, repo_root: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    try:
        profile = _mapping(json.loads(source.read_text(encoding="utf-8")), "profile")
    except (OSError, json.JSONDecodeError) as exc:
        raise AcceptanceError(f"cannot read profile: {source}") from exc
    _closed(
        profile,
        required=(
            "schema_version",
            "output_parent",
            "gpu_selection_commit",
            "gpu_selection_ref",
            "continue_after_failure",
            "conflict_process_patterns",
            "resource_pools",
            "e7",
            "e8",
            "concurrent",
        ),
        optional=("expected_harness_commit",),
        context="profile",
    )
    if profile["schema_version"] != PROFILE_SCHEMA_VERSION:
        raise AcceptanceError("unsupported profile schema")
    expected = profile.get("expected_harness_commit")
    if expected is not None:
        profile["expected_harness_commit"] = _full_sha(expected, "expected_harness_commit")
    profile["gpu_selection_commit"] = _full_sha(
        profile["gpu_selection_commit"], "gpu_selection_commit"
    )
    if not isinstance(profile["gpu_selection_ref"], str) or not profile[
        "gpu_selection_ref"
    ].strip():
        raise AcceptanceError("gpu_selection_ref must be non-empty")
    if not isinstance(profile["continue_after_failure"], bool):
        raise AcceptanceError("continue_after_failure must be boolean")
    patterns = profile["conflict_process_patterns"]
    if not isinstance(patterns, list) or any(not isinstance(item, str) for item in patterns):
        raise AcceptanceError("conflict_process_patterns must be a list of strings")
    output_parent = Path(_absolute(profile["output_parent"], "output_parent"))
    if not _outside_repo(output_parent, Path(repo_root).resolve()):
        raise AcceptanceError("output_parent must be outside the repository")
    profile["output_parent"] = str(output_parent)

    pools = _mapping(profile["resource_pools"], "resource_pools")
    _closed(
        pools,
        required=("e7_cpu_pool", "e8_cpu_pool", "e8_gpu_ids"),
        context="resource_pools",
    )
    try:
        e7_ids = parse_cpu_pool(str(pools["e7_cpu_pool"]))
        e8_ids = parse_cpu_pool(str(pools["e8_cpu_pool"]))
        gpu_text = (
            ",".join(str(item) for item in pools["e8_gpu_ids"])
            if isinstance(pools["e8_gpu_ids"], list)
            else str(pools["e8_gpu_ids"])
        )
        gpu_ids = parse_gpu_pool(gpu_text)
    except ResourcePoolError as exc:
        raise AcceptanceError(str(exc)) from exc
    overlap = sorted(set(e7_ids) & set(e8_ids))
    if overlap:
        raise AcceptanceError(f"E7/E8 CPU pools overlap: {overlap}")
    profile["resource_pools"] = {
        "e7_cpu_pool": str(pools["e7_cpu_pool"]),
        "e7_cpu_ids": list(e7_ids),
        "e8_cpu_pool": str(pools["e8_cpu_pool"]),
        "e8_cpu_ids": list(e8_ids),
        "e8_gpu_ids": list(gpu_ids),
    }

    e7 = _mapping(profile["e7"], "e7")
    e7_required = (
        "enabled", "contract", "run_spec", "grid", "fallback_workers", "probe_steps",
        "probe_seed", "probe_seconds", "plan_timeout_seconds",
        "throughput_retention_fraction", "cpu_fraction", "memory_headroom_fraction",
        "per_worker_safety_factor", "per_worker_cpu_safety_factor",
        "minimum_cpu_cores_per_worker", "max_workers", "max_growth_factor",
        "minimum_branches_for_probe", "revalidation_samples",
        "revalidation_sample_seconds", "liveness_steps", "liveness_seed",
        "liveness_timeout_seconds",
    )
    _closed(e7, required=e7_required, context="e7")
    if not isinstance(e7["enabled"], bool):
        raise AcceptanceError("e7.enabled must be boolean")
    for key in ("contract", "run_spec", "grid"):
        e7[key] = _absolute(e7[key], f"e7.{key}")
    for key in (
        "fallback_workers", "probe_steps", "probe_seed", "minimum_branches_for_probe",
        "revalidation_samples", "liveness_steps", "liveness_seed",
    ):
        e7[key] = _number(e7[key], f"e7.{key}", integer=True)
    if e7["max_workers"] is not None:
        e7["max_workers"] = _number(e7["max_workers"], "e7.max_workers", integer=True)
    for key in (
        "probe_seconds", "plan_timeout_seconds", "per_worker_safety_factor",
        "per_worker_cpu_safety_factor", "minimum_cpu_cores_per_worker",
        "max_growth_factor", "revalidation_sample_seconds", "liveness_timeout_seconds",
    ):
        e7[key] = _number(e7[key], f"e7.{key}")
    e7["throughput_retention_fraction"] = _fraction(
        e7["throughput_retention_fraction"], "e7.throughput_retention_fraction", allow_one=True
    )
    e7["cpu_fraction"] = _fraction(e7["cpu_fraction"], "e7.cpu_fraction", allow_one=True)
    e7["memory_headroom_fraction"] = _fraction(
        e7["memory_headroom_fraction"], "e7.memory_headroom_fraction", allow_one=False
    )
    profile["e7"] = e7

    e8 = _mapping(profile["e8"], "e8")
    e8_required = (
        "enabled", "model_path", "bank", "val", "test", "global_calibration",
        "base_config", "sweep_config", "required_free_gpu_memory_gib",
        "required_host_memory_gib_per_worker", "gpu_memory_headroom_fraction",
        "host_memory_headroom_fraction", "per_worker_host_memory_safety_factor",
        "per_worker_vram_safety_factor", "cpu_fraction",
        "per_worker_cpu_safety_factor", "minimum_cpu_cores_per_worker",
        "maximum_gpu_utilization_percent", "max_devices", "max_slots_per_gpu",
        "single_probe_seconds", "validation_probe_seconds", "probe_budget_seconds",
        "probe_free_floor_gib", "selection_timeout_seconds", "thread_candidates",
    )
    _closed(e8, required=e8_required, context="e8")
    if not isinstance(e8["enabled"], bool):
        raise AcceptanceError("e8.enabled must be boolean")
    for key in (
        "model_path", "bank", "val", "test", "global_calibration", "base_config",
        "sweep_config",
    ):
        e8[key] = _absolute(e8[key], f"e8.{key}")
    for key in ("max_devices", "max_slots_per_gpu"):
        e8[key] = _number(e8[key], f"e8.{key}", integer=True)
    for key in (
        "required_free_gpu_memory_gib", "required_host_memory_gib_per_worker",
        "per_worker_host_memory_safety_factor", "per_worker_vram_safety_factor",
        "per_worker_cpu_safety_factor", "minimum_cpu_cores_per_worker",
        "maximum_gpu_utilization_percent", "single_probe_seconds",
        "validation_probe_seconds", "probe_budget_seconds", "probe_free_floor_gib",
        "selection_timeout_seconds",
    ):
        e8[key] = _number(e8[key], f"e8.{key}")
    e8["gpu_memory_headroom_fraction"] = _fraction(
        e8["gpu_memory_headroom_fraction"],
        "e8.gpu_memory_headroom_fraction",
        allow_one=False,
    )
    e8["host_memory_headroom_fraction"] = _fraction(
        e8["host_memory_headroom_fraction"], "e8.host_memory_headroom_fraction", allow_one=False
    )
    e8["cpu_fraction"] = _fraction(e8["cpu_fraction"], "e8.cpu_fraction", allow_one=True)
    candidates = e8["thread_candidates"]
    if not isinstance(candidates, list) or not candidates:
        raise AcceptanceError("e8.thread_candidates must be non-empty")
    normalized: list[int | None] = []
    for item in candidates:
        normalized.append(
            None
            if item is None
            else int(_number(item, "thread candidate", integer=True))
        )
    if len(normalized) != len(set(normalized)):
        raise AcceptanceError("thread candidates must be unique")
    e8["thread_candidates"] = normalized
    profile["e8"] = e8

    concurrent = _mapping(profile["concurrent"], "concurrent")
    _closed(
        concurrent,
        required=("enabled", "timeout_seconds", "sample_interval_seconds"),
        context="concurrent",
    )
    if not isinstance(concurrent["enabled"], bool):
        raise AcceptanceError("concurrent.enabled must be boolean")
    concurrent["timeout_seconds"] = _number(
        concurrent["timeout_seconds"], "concurrent.timeout_seconds"
    )
    concurrent["sample_interval_seconds"] = _number(
        concurrent["sample_interval_seconds"], "concurrent.sample_interval_seconds"
    )
    profile["concurrent"] = concurrent
    profile["profile_path"] = str(source)
    profile["profile_sha256"] = sha256_file(source)
    return profile


def git_output(repo: str | Path, *args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(Path(repo).resolve()), *args],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=30,
        ).strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise AcceptanceError(f"git failed: {' '.join(args)}") from exc


def verify_checkout(repo: str | Path, expected: str | None = None) -> dict[str, Any]:
    commit = git_output(repo, "rev-parse", "HEAD")
    dirty = bool(git_output(repo, "status", "--porcelain"))
    if dirty:
        raise AcceptanceError("acceptance requires a clean checkout")
    if expected is not None and commit != expected:
        raise AcceptanceError(f"checkout mismatch: {commit} != {expected}")
    return {
        "repo_root": str(Path(repo).resolve()),
        "commit": commit,
        "branch": git_output(repo, "rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": False,
    }


def ensure_commit(repo: Path, commit: str, remote_ref: str) -> None:
    check = subprocess.run(
        ["git", "-C", str(repo), "cat-file", "-e", f"{commit}^{{commit}}"],
        check=False,
    )
    if check.returncode != 0:
        fetch = subprocess.run(
            ["git", "-C", str(repo), "fetch", "--no-tags", "origin", remote_ref],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        if fetch.returncode != 0:
            raise AcceptanceError(f"cannot fetch GPU ref: {fetch.stdout.strip()}")
    if git_output(repo, "rev-parse", commit) != commit:
        raise AcceptanceError("GPU selection commit is unavailable")


def add_worktree(repo: Path, target: Path, commit: str) -> None:
    if target.exists():
        raise AcceptanceError(f"worktree already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["git", "-C", str(repo), "worktree", "add", "--detach", str(target), commit],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.returncode != 0:
        raise AcceptanceError(f"cannot create GPU worktree: {result.stdout.strip()}")
    if git_output(target, "rev-parse", "HEAD") != commit:
        raise AcceptanceError("GPU worktree commit mismatch")
    if git_output(target, "status", "--porcelain"):
        raise AcceptanceError("GPU worktree is dirty")


def remove_worktree(repo: Path, target: Path) -> None:
    subprocess.run(
        ["git", "-C", str(repo), "worktree", "remove", "--force", str(target)],
        check=False,
    )
    subprocess.run(["git", "-C", str(repo), "worktree", "prune"], check=False)
    if target.exists():
        shutil.rmtree(target, ignore_errors=True)


def overall_status(results: Sequence[StageResult]) -> str:
    statuses = [result.status for result in results]
    for status in ("FAIL", "BLOCKED", "INCONCLUSIVE"):
        if status in statuses:
            return status
    if statuses and all(item in {"PASS", "NOT_RUN"} for item in statuses):
        return "PASS"
    return "BLOCKED"


def stage_result(
    root: Path, name: str, status: str, started: str, details: Mapping[str, Any]
) -> StageResult:
    result = StageResult(name, status, started, utc_now(), dict(details))
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    atomic_write_json(directory / "STAGE_RESULT.json", result.as_dict())
    return result


def package_acceptance(root: Path) -> dict[str, Any]:
    manifest = root / "FILE_MANIFEST.sha256"
    paths: list[Path] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise AcceptanceError(f"package forbids symlink: {path}")
        if not path.is_file() or "worktrees" in path.relative_to(root).parts:
            continue
        if path.suffix.lower() not in ALLOWED_SUFFIXES:
            raise AcceptanceError(f"package forbids file type: {path}")
        if path.stat().st_size > 50 * 1024 * 1024:
            raise AcceptanceError(f"evidence file exceeds 50 MiB: {path}")
        paths.append(path)
    lines = [
        f"{sha256_file(path)}  {path.relative_to(root).as_posix()}"
        for path in paths
        if path != manifest
    ]
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if manifest not in paths:
        paths.append(manifest)
    total = sum(path.stat().st_size for path in paths)
    if total > 200 * 1024 * 1024:
        raise AcceptanceError("acceptance evidence exceeds 200 MiB")
    archive = root.parent / f"{root.name}.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        for path in paths:
            handle.add(path, arcname=f"{root.name}/{path.relative_to(root)}")
    return {
        "path": str(archive),
        "size_bytes": archive.stat().st_size,
        "sha256": sha256_file(archive),
        "file_count": len(paths),
    }
