#!/usr/bin/env python3
"""Resume the interrupted E8 V2 taper pilot without changing its science."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

EXPERIMENT_ID = "EXT-C-E8-ORACLE-OFFLINE-V2-TAPER-SWEEP-0.5B-01"
VERSION = "0.2.0"
RUNTIME_REL = Path("src/drpo/countdown_e8_oracle_offline_v2_taper_runtime.py")
CORE_REL = Path("src/drpo/countdown_e8_oracle_offline_v2_taper_sweep.py")
BASE_CONFIG = "configs/countdown_e8_base_rl_replay_0p5b.yaml"
SWEEP_CONFIG = "configs/countdown_e8_oracle_offline_v2_taper_sweep_0p5b.yaml"


class ResumeError(RuntimeError):
    pass


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def rid() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{os.getpid()}"


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ResumeError(f"{label} is not valid JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ResumeError(f"{label} must be a JSON object: {path}")
    return value


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ResumeError(f"Cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise ResumeError((result.stderr or result.stdout).strip())
    return result.stdout.strip()


def git_state(repo: Path) -> dict[str, Any]:
    status = [
        line
        for line in git(repo, "status", "--porcelain", "--untracked-files=all").splitlines()
        if line.strip()
    ]
    return {
        "commit": git(repo, "rev-parse", "HEAD"),
        "branch": git(repo, "rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status),
        "status": status,
    }


def repo_path(repo: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repo / path).resolve()


def import_runner(repo: Path) -> tuple[Any, Any]:
    sys.path.insert(0, str((repo / "src").resolve()))
    try:
        from drpo import countdown_e8_oracle_offline_v2_taper_runtime as runtime
        from drpo import countdown_e8_oracle_offline_v2_taper_sweep as core
    except Exception as exc:  # noqa: BLE001
        raise ResumeError(f"Cannot import the E8 taper runner: {exc}") from exc
    return runtime, core


def conflicts(work_dir: Path) -> list[int]:
    if not Path("/proc").is_dir():
        return []
    needle = str(work_dir.resolve())
    tokens = (
        "run_countdown_e8_oracle_offline_v2_taper_sweep.py",
        "countdown_e8_oracle_offline_v2_taper_runtime.py",
        Path(__file__).name,
    )
    excluded = {os.getpid(), os.getppid()}
    found: list[int] = []
    for item in Path("/proc").iterdir():
        if not item.name.isdigit() or int(item.name) in excluded:
            continue
        try:
            command = (
                (item / "cmdline")
                .read_bytes()
                .replace(b"\0", b" ")
                .decode(errors="replace")
            )
        except OSError:
            continue
        if needle in command and any(token in command for token in tokens):
            found.append(int(item.name))
    return sorted(found)


def resolve_paths(
    args: argparse.Namespace, repo: Path, work_dir: Path
) -> dict[str, Path]:
    result = {
        "model": Path(args.model_path).resolve(),
        "bank": Path(args.bank).resolve(),
        "val": Path(args.val).resolve(),
        "test": Path(args.test).resolve(),
        "global_calibration": Path(args.global_calibration).resolve(),
        "base_config": repo_path(repo, args.base_config),
        "sweep_config": repo_path(repo, args.sweep_config),
        "calibration": work_dir / "calibration" / "taper_budget_calibration.json",
    }
    for name, path in result.items():
        if not path.exists():
            raise ResumeError(f"Missing {name}: {path}")
    return result


def preflight(args: argparse.Namespace) -> tuple[dict[str, Any], list[str]]:
    repo = Path(args.repo_root).resolve()
    work_dir = Path(args.work_dir).resolve()
    if not (repo / ".git").is_dir() or not work_dir.is_dir():
        raise ResumeError("repo_root and existing work_dir are required")

    runtime, core = import_runner(repo)
    paths = resolve_paths(args, repo, work_dir)
    state = git_state(repo)
    original = read_json(work_dir / "run_manifest.json", "run_manifest")
    if original.get("experiment_id") != EXPERIMENT_ID:
        raise ResumeError("run_manifest experiment_id mismatch")
    if original.get("base_commit") != state["commit"]:
        raise ResumeError("HEAD changed since launch; do not git pull before resume")
    if original.get("git_status_at_launch") != state["status"]:
        raise ResumeError("worktree status changed since launch")

    calibration = read_json(paths["calibration"], "taper calibration")
    identity = calibration.get("identity")
    source = identity.get("source") if isinstance(identity, dict) else None
    if not isinstance(identity, dict) or not isinstance(source, dict):
        raise ResumeError("calibration identity is missing")
    if (
        calibration.get("experiment_id") != EXPERIMENT_ID
        or calibration.get("runtime_version") != runtime.VERSION
        or source.get("commit") != state["commit"]
        or bool(source.get("dirty")) != state["dirty"]
        or source.get("runtime_source_sha256") != sha(repo / RUNTIME_REL)
        or source.get("core_source_sha256") != sha(repo / CORE_REL)
        or identity.get("model_path") != str(paths["model"])
    ):
        raise ResumeError("calibration source identity changed")
    expected_hashes = {
        "bank_sha256": sha(paths["bank"]),
        "global_calibration_sha256": sha(paths["global_calibration"]),
        "base_config_sha256": sha(paths["base_config"]),
        "sweep_config_sha256": sha(paths["sweep_config"]),
    }
    if any(identity.get(key) != value for key, value in expected_hashes.items()):
        raise ResumeError("calibration input hash changed")

    config = core.load_yaml(paths["sweep_config"])
    core.validate_sweep_config(config)
    cells = list(core.build_cells(config))
    complete: list[Any] = []
    pending: list[Any] = []
    deferred: list[Any] = []
    for cell in cells:
        summary_path = work_dir / "methods" / cell.name / "summary.json"
        if not summary_path.exists():
            pending.append(cell)
            continue
        summary = read_json(summary_path, cell.name)
        expected = core._run_identity(
            repo=repo,
            model_path=paths["model"],
            bank=paths["bank"],
            val=paths["val"],
            test=paths["test"],
            base_config=paths["base_config"],
            sweep_config=paths["sweep_config"],
            calibration=paths["calibration"],
            cell=cell,
        )
        if not core._identity_equal(summary.get("run_identity", {}), expected):
            raise ResumeError(f"summary identity changed: {cell.name}")
        is_deferred = any(
            isinstance(summary.get(name), dict)
            and summary[name].get("deferred_posthoc_evaluation")
            for name in ("best_evaluation", "terminal_evaluation")
        )
        if is_deferred:
            deferred.append(cell)
            pending.append(cell)
        else:
            complete.append(cell)
    active = conflicts(work_dir)
    if active:
        raise ResumeError(f"existing taper processes target this work_dir: {active}")

    audit = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "resume_version": VERSION,
        "validated_utc": now(),
        "source": {
            "commit": state["commit"],
            "branch": state["branch"],
            "dirty": state["dirty"],
        },
        "git_status": state["status"],
        "work_dir": str(work_dir),
        "completed_cells": len(complete),
        "pending_cells": len(pending),
        "deferred_cells": len(deferred),
        "family_completed": {
            method: sum(cell.method == method for cell in complete)
            for method in core.METHODS
        },
        "pending_cell_names": [cell.name for cell in pending],
        "original_failure_preserved": (work_dir / "RUN_FAILED.json").is_file(),
        "runtime_source_sha256": sha(repo / RUNTIME_REL),
        "core_source_sha256": sha(repo / CORE_REL),
        "calibration_sha256": sha(paths["calibration"]),
    }
    command = [
        sys.executable,
        str((repo / RUNTIME_REL).resolve()),
        "run",
        "--model_path",
        str(paths["model"]),
        "--work_dir",
        str(work_dir),
        "--bank",
        str(paths["bank"]),
        "--val",
        str(paths["val"]),
        "--test",
        str(paths["test"]),
        "--global_calibration",
        str(paths["global_calibration"]),
        "--base_config",
        str(paths["base_config"]),
        "--sweep_config",
        str(paths["sweep_config"]),
        "--calibration_gpu",
        args.calibration_gpu,
        "--gpus",
        args.gpus,
    ]
    return audit, command


def final_audit(work_dir: Path) -> dict[str, Any]:
    value = read_json(work_dir / "SWEEP_COMPLETE.json", "SWEEP_COMPLETE")
    if (
        value.get("expected_cells") != 72
        or value.get("summary_count") != 72
        or value.get("failed_cells") != []
        or value.get("all_expected_cells_present") is not True
    ):
        raise ResumeError(f"resume did not reach clean 72/72: {value}")
    return value


def run(args: argparse.Namespace) -> int:
    audit, command = preflight(args)
    work_dir = Path(args.work_dir).resolve()
    resume_name = args.resume_id or rid()
    resume_dir = work_dir / "resume_runs" / resume_name
    resume_dir.mkdir(parents=True, exist_ok=True)
    log = resume_dir / "runtime.log"
    atomic_json(
        resume_dir / "RESUME_MANIFEST.json",
        {
            **audit,
            "resume_id": resume_name,
            "start_utc": now(),
            "controller_pid": os.getpid(),
            "runtime_log": str(log),
            "command": command,
        },
    )
    if audit["pending_cells"] == 0:
        complete = final_audit(work_dir)
        atomic_json(resume_dir / "RESUME_COMPLETE.json", {"sweep_complete": complete})
        return 0

    with log.open("a", buffering=1) as handle:
        result = subprocess.run(
            command,
            cwd=Path(args.repo_root).resolve(),
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "resume_id": resume_name,
        "resume_version": VERSION,
        "end_utc": now(),
        "runtime_returncode": result.returncode,
        "runtime_log": str(log),
    }
    if result.returncode:
        atomic_json(resume_dir / "RESUME_FAILED.json", payload)
        return result.returncode if result.returncode > 0 else 1
    try:
        payload["sweep_complete"] = final_audit(work_dir)
    except ResumeError as exc:
        payload["terminal_audit_error"] = str(exc)
        atomic_json(resume_dir / "RESUME_FAILED.json", payload)
        return 1
    atomic_json(resume_dir / "RESUME_COMPLETE.json", payload)
    return 0


def child_args(argv: Sequence[str], resume_name: str) -> list[str]:
    rows: list[str] = []
    skip = False
    for item in argv:
        if skip:
            skip = False
            continue
        if item in {"--detach", "--dry-run"}:
            continue
        if item == "--resume-id":
            skip = True
            continue
        if item.startswith("--resume-id="):
            continue
        rows.append(item)
    return [*rows, "--resume-id", resume_name]


def detach(args: argparse.Namespace, argv: Sequence[str]) -> int:
    audit, _command = preflight(args)
    resume_name = args.resume_id or rid()
    resume_dir = Path(args.work_dir).resolve() / "resume_runs" / resume_name
    resume_dir.mkdir(parents=True, exist_ok=False)
    log = resume_dir / "controller.log"
    command = [sys.executable, str(Path(__file__).resolve()), *child_args(argv, resume_name)]
    with log.open("a") as handle:
        process = subprocess.Popen(
            command,
            cwd=Path(args.repo_root).resolve(),
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
    atomic_json(
        resume_dir / "RESUME_LAUNCH.json",
        {
            **audit,
            "resume_id": resume_name,
            "launched_utc": now(),
            "controller_pid": process.pid,
            "controller_log": str(log),
            "command": command,
        },
    )
    print(f"resume started pid={process.pid} status_dir={resume_dir}")
    return 0


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Resume the existing E8 V2 taper pilot")
    value.add_argument("--repo-root", default=str(Path.cwd()))
    value.add_argument("--model_path", default="/root/models/Qwen2.5-0.5B-Instruct")
    value.add_argument("--work_dir", default="/root/experiment_output/e8_v2_taper_sweep")
    value.add_argument(
        "--bank",
        default="/root/experiment_output/e8_oracle_bank_v2/data/offline_bank_v2.jsonl",
    )
    value.add_argument(
        "--val", default="/root/experiment_output/e8_oracle_bank_v2/data/val.jsonl"
    )
    value.add_argument(
        "--test", default="/root/experiment_output/e8_oracle_bank_v2/data/test.jsonl"
    )
    value.add_argument(
        "--global_calibration",
        default="/root/experiment_output/e8_v2_matrix/calibration/base/calibration.json",
    )
    value.add_argument("--base_config", default=BASE_CONFIG)
    value.add_argument("--sweep_config", default=SWEEP_CONFIG)
    value.add_argument("--calibration_gpu", default="0")
    value.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    value.add_argument("--resume-id")
    value.add_argument("--dry-run", action="store_true")
    value.add_argument("--detach", action="store_true")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    args = parser().parse_args(raw)
    try:
        if args.dry_run:
            audit, _command = preflight(args)
            print(json.dumps(audit, indent=2, ensure_ascii=False))
            return 0
        return detach(args, raw) if args.detach else run(args)
    except ResumeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
