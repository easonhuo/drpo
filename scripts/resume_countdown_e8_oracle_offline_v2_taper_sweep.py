#!/usr/bin/env python3
"""Resume the interrupted 72-cell E8 V2 taper pilot in the original work_dir.

The scientific runner is not changed.  This controller verifies that the checkout,
dirty state, calibration, configs, data, and completed-cell identities still match
the first launch, then calls the existing resumable runtime.  Resume supervision is
written under ``<work_dir>/resume_runs`` so first-run failure evidence is preserved.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

EXPERIMENT_ID = "EXT-C-E8-ORACLE-OFFLINE-V2-TAPER-SWEEP-0.5B-01"
VERSION = "0.1.0"
RUNTIME_REL = Path("src/drpo/countdown_e8_oracle_offline_v2_taper_runtime.py")
CORE_REL = Path("src/drpo/countdown_e8_oracle_offline_v2_taper_sweep.py")
BASE_CONFIG_REL = Path("configs/countdown_e8_base_rl_replay_0p5b.yaml")
SWEEP_CONFIG_REL = Path("configs/countdown_e8_oracle_offline_v2_taper_sweep_0p5b.yaml")
SCRIPT_NAME = "resume_countdown_e8_oracle_offline_v2_taper_sweep.py"


class ResumeError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def resume_id() -> str:
    return f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{os.getpid()}"


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ResumeError(f"{label} is not readable JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ResumeError(f"{label} must contain a JSON object: {path}")
    return value


def sha256(path: Path) -> str:
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
        ["git", "-C", str(repo), *args], text=True, capture_output=True, check=False
    )
    if result.returncode:
        raise ResumeError(
            f"git {' '.join(args)} failed: {(result.stderr or result.stdout).strip()}"
        )
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
    src = str((repo / "src").resolve())
    if src not in sys.path:
        sys.path.insert(0, src)
    try:
        from drpo import countdown_e8_oracle_offline_v2_taper_runtime as runtime
        from drpo import countdown_e8_oracle_offline_v2_taper_sweep as core
    except Exception as exc:  # noqa: BLE001
        raise ResumeError(f"Cannot import the frozen E8 runner: {exc}") from exc
    return runtime, core


def live_conflicts(work_dir: Path) -> list[dict[str, Any]]:
    if not Path("/proc").is_dir():
        return []
    needle = str(work_dir.resolve())
    tokens = (
        "run_countdown_e8_oracle_offline_v2_taper_sweep.py",
        "countdown_e8_oracle_offline_v2_taper_runtime.py",
        SCRIPT_NAME,
    )
    excluded = {os.getpid(), os.getppid()}
    rows: list[dict[str, Any]] = []
    for item in Path("/proc").iterdir():
        if not item.name.isdigit() or int(item.name) in excluded:
            continue
        try:
            command = (
                (item / "cmdline")
                .read_bytes()
                .replace(b"\0", b" ")
                .decode(errors="replace")
                .strip()
            )
        except OSError:
            continue
        if needle in command and any(token in command for token in tokens):
            rows.append({"pid": int(item.name), "command": command})
    return sorted(rows, key=lambda row: row["pid"])


def resolved_inputs(args: argparse.Namespace, repo: Path, work_dir: Path) -> dict[str, Path]:
    paths = {
        "model_path": Path(args.model_path).resolve(),
        "bank": Path(args.bank).resolve(),
        "val": Path(args.val).resolve(),
        "test": Path(args.test).resolve(),
        "global_calibration": Path(args.global_calibration).resolve(),
        "base_config": repo_path(repo, args.base_config),
        "sweep_config": repo_path(repo, args.sweep_config),
        "calibration": work_dir / "calibration" / "taper_budget_calibration.json",
    }
    for name, path in paths.items():
        if not path.exists():
            raise ResumeError(f"Missing required {name}: {path}")
    return paths


def validate_summaries(
    *,
    core: Any,
    runtime: Any,
    cells: Sequence[Any],
    paths: dict[str, Path],
    methods_dir: Path,
    repo: Path,
) -> tuple[list[Any], list[Any], list[Any]]:
    expected_names = {cell.name for cell in cells}
    unexpected = sorted(
        path.parent.name
        for path in methods_dir.glob("*/summary.json")
        if path.parent.name not in expected_names
    )
    if unexpected:
        raise ResumeError(f"Unexpected summary directories: {unexpected}")

    complete: list[Any] = []
    pending: list[Any] = []
    deferred: list[Any] = []
    for cell in cells:
        summary_path = methods_dir / cell.name / "summary.json"
        if not summary_path.exists():
            pending.append(cell)
            continue
        summary = read_json(summary_path, f"summary {cell.name}")
        try:
            metadata_matches = (
                summary.get("experiment_id") == EXPERIMENT_ID
                and summary.get("cell") == cell.name
                and summary.get("method") == cell.method
                and float(summary.get("rho")) == float(cell.rho)
                and int(summary.get("seed_offset")) == int(cell.seed_offset)
            )
        except (TypeError, ValueError):
            metadata_matches = False
        if not metadata_matches:
            raise ResumeError(f"Cell metadata mismatch in {summary_path}")
        expected_identity = core._run_identity(
            repo=repo,
            model_path=paths["model_path"],
            bank=paths["bank"],
            val=paths["val"],
            test=paths["test"],
            base_config=paths["base_config"],
            sweep_config=paths["sweep_config"],
            calibration=paths["calibration"],
            cell=cell,
        )
        if not core._identity_equal(summary.get("run_identity", {}), expected_identity):
            raise ResumeError(f"Run identity mismatch in {summary_path}")
        if (
            summary.get("runtime_version") != runtime.VERSION
            or summary.get("runtime_source_sha256") != sha256(repo / RUNTIME_REL)
            or summary.get("best_terminal_same_generation_seed") is not True
        ):
            raise ResumeError(f"Runtime audit mismatch in {summary_path}")
        best = summary.get("best_evaluation", {})
        terminal = summary.get("terminal_evaluation", {})
        is_deferred = bool(
            isinstance(best, dict)
            and best.get("deferred_posthoc_evaluation")
            or isinstance(terminal, dict)
            and terminal.get("deferred_posthoc_evaluation")
        )
        if is_deferred:
            deferred.append(cell)
            pending.append(cell)
        else:
            complete.append(cell)
    return complete, pending, deferred


def preflight(args: argparse.Namespace, *, check_processes: bool = True) -> dict[str, Any]:
    repo = Path(args.repo_root).resolve()
    work_dir = Path(args.work_dir).resolve()
    if not (repo / ".git").is_dir():
        raise ResumeError(f"Not a Git checkout: {repo}")
    if not work_dir.is_dir():
        raise ResumeError(f"Existing work_dir is required: {work_dir}")

    runtime, core = import_runner(repo)
    paths = resolved_inputs(args, repo, work_dir)
    config = core.load_yaml(paths["sweep_config"])
    core.validate_sweep_config(config)
    cells = list(core.build_cells(config))
    if len(cells) != 72:
        raise ResumeError(f"Frozen sweep must contain 72 cells, got {len(cells)}")

    state = git_state(repo)
    original = read_json(work_dir / "run_manifest.json", "original run manifest")
    if original.get("experiment_id") != EXPERIMENT_ID:
        raise ResumeError("Original experiment_id mismatch")
    if original.get("base_commit") != state["commit"]:
        raise ResumeError(
            "Local HEAD differs from the first launch. Do not git pull before resume."
        )
    if original.get("git_status_at_launch") != state["status"]:
        raise ResumeError("Current worktree differs from the first launch snapshot")

    calibration = read_json(paths["calibration"], "taper calibration")
    identity = calibration.get("identity")
    if not isinstance(identity, dict):
        raise ResumeError("Calibration identity is missing")
    source = identity.get("source")
    if not isinstance(source, dict):
        raise ResumeError("Calibration source identity is missing")
    if (
        calibration.get("experiment_id") != EXPERIMENT_ID
        or calibration.get("runtime_version") != runtime.VERSION
        or source.get("commit") != state["commit"]
        or bool(source.get("dirty")) != state["dirty"]
    ):
        raise ResumeError("Calibration source identity differs from the first launch")
    if source.get("runtime_source_sha256") != sha256(repo / RUNTIME_REL):
        raise ResumeError("Runtime source differs from the frozen calibration")
    if source.get("core_source_sha256") != sha256(repo / CORE_REL):
        raise ResumeError("Core source differs from the frozen calibration")
    if identity.get("model_path") != str(paths["model_path"]):
        raise ResumeError("Model path differs from the frozen calibration")
    frozen_hashes = {
        "bank_sha256": sha256(paths["bank"]),
        "global_calibration_sha256": sha256(paths["global_calibration"]),
        "base_config_sha256": sha256(paths["base_config"]),
        "sweep_config_sha256": sha256(paths["sweep_config"]),
    }
    for key, value in frozen_hashes.items():
        if identity.get(key) != value:
            raise ResumeError(f"Frozen input mismatch: {key}")

    complete, pending, deferred = validate_summaries(
        core=core,
        runtime=runtime,
        cells=cells,
        paths=paths,
        methods_dir=work_dir / "methods",
        repo=repo,
    )
    conflicts = live_conflicts(work_dir) if check_processes else []
    if conflicts:
        raise ResumeError(f"Another process already targets this work_dir: {conflicts}")

    family = {
        method: sum(cell.method == method for cell in complete) for method in core.METHODS
    }
    audit = {
        "schema_version": 1,
        "resume_version": VERSION,
        "experiment_id": EXPERIMENT_ID,
        "validated_utc": utc_now(),
        "repo_root": str(repo),
        "work_dir": str(work_dir),
        "source": {
            "commit": state["commit"],
            "branch": state["branch"],
            "dirty": state["dirty"],
        },
        "git_status": state["status"],
        "original_failure_preserved": (work_dir / "RUN_FAILED.json").is_file(),
        "runtime_version": runtime.VERSION,
        "runtime_source_sha256": sha256(repo / RUNTIME_REL),
        "core_source_sha256": sha256(repo / CORE_REL),
        "calibration_sha256": sha256(paths["calibration"]),
        "completed_cells": len(complete),
        "pending_cells": len(pending),
        "deferred_cells": len(deferred),
        "family_completed": family,
        "pending_cell_names": [cell.name for cell in pending],
        "paths": {name: str(path) for name, path in paths.items()},
    }
    return {"audit": audit, "paths": paths}


def runtime_command(
    args: argparse.Namespace, repo: Path, paths: dict[str, Path]
) -> list[str]:
    return [
        sys.executable,
        str((repo / RUNTIME_REL).resolve()),
        "run",
        "--model_path",
        str(paths["model_path"]),
        "--work_dir",
        str(Path(args.work_dir).resolve()),
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


def final_audit(work_dir: Path) -> dict[str, Any]:
    complete = read_json(work_dir / "SWEEP_COMPLETE.json", "SWEEP_COMPLETE")
    if (
        complete.get("expected_cells") != 72
        or complete.get("summary_count") != 72
        or complete.get("failed_cells") != []
        or complete.get("all_expected_cells_present") is not True
    ):
        raise ResumeError(f"Not a clean 72/72 terminal state: {complete}")
    return complete


def safe_print(text: str) -> None:
    try:
        print(text, flush=True)
    except BrokenPipeError:
        pass


def supervise(args: argparse.Namespace) -> int:
    prepared = preflight(args)
    audit, paths = prepared["audit"], prepared["paths"]
    repo = Path(args.repo_root).resolve()
    work_dir = Path(args.work_dir).resolve()
    rid = args.resume_id or resume_id()
    resume_dir = work_dir / "resume_runs" / rid
    resume_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(resume_dir / "RESUME_AUDIT.json", audit)

    lock = (work_dir / ".e8_v2_taper_resume.lock").open("a+")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise ResumeError("Another resume controller holds the work_dir lock") from exc

    if audit["pending_cells"] == 0:
        complete = final_audit(work_dir)
        atomic_json(
            resume_dir / "RESUME_COMPLETE.json",
            {
                "experiment_id": EXPERIMENT_ID,
                "resume_id": rid,
                "resume_version": VERSION,
                "completed_utc": utc_now(),
                "runtime_returncode": 0,
                "launched_runtime": False,
                "sweep_complete": complete,
            },
        )
        safe_print("E8 taper sweep already complete: 72/72.")
        return 0

    command = runtime_command(args, repo, paths)
    runtime_log = resume_dir / "runtime.log"
    manifest = {
        **audit,
        "resume_id": rid,
        "start_utc": utc_now(),
        "controller_pid": os.getpid(),
        "runtime_log": str(runtime_log),
        "command": command,
    }
    atomic_json(resume_dir / "RESUME_MANIFEST.json", manifest)
    with runtime_log.open("a", buffering=1) as handle:
        process = subprocess.Popen(
            command,
            cwd=repo,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        manifest["runtime_pid"] = process.pid
        atomic_json(resume_dir / "RESUME_MANIFEST.json", manifest)
        forwarded: int | None = None

        def relay(signum: int, _frame: Any) -> None:
            nonlocal forwarded
            forwarded = signum
            try:
                os.killpg(process.pid, signum)
            except ProcessLookupError:
                pass

        previous = {
            sig: signal.signal(sig, relay) for sig in (signal.SIGINT, signal.SIGTERM)
        }
        try:
            while process.poll() is None:
                count = sum(
                    path.is_file()
                    for path in (work_dir / "methods").glob("*/summary.json")
                )
                atomic_json(
                    resume_dir / "heartbeat.json",
                    {
                        "experiment_id": EXPERIMENT_ID,
                        "resume_id": rid,
                        "utc": utc_now(),
                        "controller_pid": os.getpid(),
                        "runtime_pid": process.pid,
                        "completed_summaries": count,
                        "expected_summaries": 72,
                        "pending_estimate": max(0, 72 - count),
                        "runtime_log_size_bytes": runtime_log.stat().st_size,
                    },
                )
                time.sleep(args.poll_seconds)
            returncode = int(process.wait())
        finally:
            for sig, old in previous.items():
                signal.signal(sig, old)

    result = {
        "experiment_id": EXPERIMENT_ID,
        "resume_id": rid,
        "resume_version": VERSION,
        "end_utc": utc_now(),
        "runtime_returncode": returncode,
        "forwarded_signal": forwarded,
        "runtime_log": str(runtime_log),
        "source": audit["source"],
        "controller_source_at_end": git_state(repo),
    }
    if returncode:
        atomic_json(resume_dir / "RESUME_FAILED.json", result)
        safe_print(f"Resume failed with returncode={returncode}; see {runtime_log}")
        return returncode if returncode > 0 else 1
    try:
        result["sweep_complete"] = final_audit(work_dir)
    except ResumeError as exc:
        result["terminal_audit_error"] = str(exc)
        atomic_json(resume_dir / "RESUME_FAILED.json", result)
        safe_print(str(exc))
        return 1

    atomic_json(resume_dir / "RESUME_COMPLETE.json", result)
    atomic_json(
        work_dir / "RUN_RAW_COMPLETE.json",
        {
            "schema_version": 3,
            "experiment_id": EXPERIMENT_ID,
            "base_commit": audit["source"]["commit"],
            "execution_state": "raw_complete",
            "result_status": "pilot",
            "completed_utc": utc_now(),
            "resume_id": rid,
            "resume_version": VERSION,
            "summary_count": 72,
            "expected_cells": 72,
            "initial_failure_preserved": (work_dir / "RUN_FAILED.json").is_file(),
            "scientific_acceptance_pending": True,
        },
    )
    safe_print(f"E8 taper resume complete: 72/72. {resume_dir}")
    return 0


def child_argv(argv: Sequence[str], rid: str) -> list[str]:
    result: list[str] = []
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
        result.append(item)
    return [*result, "--resume-id", rid]


def detach(args: argparse.Namespace, argv: Sequence[str]) -> int:
    prepared = preflight(args)
    rid = args.resume_id or resume_id()
    resume_dir = Path(args.work_dir).resolve() / "resume_runs" / rid
    resume_dir.mkdir(parents=True, exist_ok=False)
    log = resume_dir / "controller.log"
    command = [sys.executable, str(Path(__file__).resolve()), *child_argv(argv, rid)]
    with log.open("a", buffering=1) as handle:
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
            "experiment_id": EXPERIMENT_ID,
            "resume_id": rid,
            "resume_version": VERSION,
            "launched_utc": utc_now(),
            "controller_pid": process.pid,
            "controller_log": str(log),
            "command": command,
            "preflight": prepared["audit"],
        },
    )
    safe_print(f"Resume started: pid={process.pid} status_dir={resume_dir}")
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
    value.add_argument("--base_config", default=str(BASE_CONFIG_REL))
    value.add_argument("--sweep_config", default=str(SWEEP_CONFIG_REL))
    value.add_argument("--calibration_gpu", default="0")
    value.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    value.add_argument("--poll-seconds", type=float, default=30.0)
    value.add_argument("--resume-id")
    value.add_argument("--dry-run", action="store_true")
    value.add_argument("--detach", action="store_true")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    args = parser().parse_args(raw)
    if args.poll_seconds <= 0:
        safe_print("ERROR: --poll-seconds must be positive")
        return 2
    try:
        if args.dry_run:
            safe_print(json.dumps(preflight(args)["audit"], indent=2, ensure_ascii=False))
            return 0
        if args.detach:
            return detach(args, raw)
        return supervise(args)
    except ResumeError as exc:
        safe_print(f"ERROR: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
