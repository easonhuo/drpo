#!/usr/bin/env python3
"""Run a formal experiment under an attached supervisor.

The guard records provenance and heartbeats, streams logs, preserves failures, and
immediately builds a raw-complete or failed recovery artifact. It intentionally
does not label a successful process as a scientifically completed experiment;
terminal audit and final packaging remain separate gates.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import queue
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    temp.replace(path)


def git_value(repo: Path, *args: str) -> str | None:
    result = subprocess.run(
        ["git", *args], cwd=repo, text=True, capture_output=True
    )
    return result.stdout.strip() if result.returncode == 0 else None


def latest_mtime(root: Path) -> float | None:
    latest: float | None = None
    for path in root.rglob("*"):
        try:
            if path.is_file():
                value = path.stat().st_mtime
                latest = value if latest is None else max(latest, value)
        except FileNotFoundError:
            continue
    return latest


def progress_counts(root: Path, patterns: list[str]) -> dict[str, int]:
    return {pattern: sum(1 for p in root.glob(pattern) if p.is_file()) for pattern in patterns}


def stream_reader(pipe: Any, events: queue.Queue[tuple[float, str]]) -> None:
    try:
        for line in iter(pipe.readline, ""):
            events.put((time.time(), line))
    finally:
        pipe.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--artifact-output", type=Path, required=True)
    parser.add_argument("--heartbeat-seconds", type=float, default=30.0)
    parser.add_argument("--stale-seconds", type=float, default=600.0)
    parser.add_argument("--fail-on-stale", action="store_true")
    parser.add_argument("--progress-glob", action="append", default=[])
    parser.add_argument("--required-output", action="append", default=[])
    parser.add_argument("--source-file", action="append", default=[])
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("Supply the experiment command after --")
    if args.heartbeat_seconds <= 0 or args.stale_seconds <= 0:
        parser.error("Heartbeat and stale intervals must be positive")
    return args


def package_recovery(
    repo: Path,
    experiment_id: str,
    output_root: Path,
    artifact_output: Path,
    package_kind: str,
    source_files: list[str],
) -> subprocess.CompletedProcess[str]:
    script = repo / "scripts" / "package_experiment.py"
    cmd = [
        sys.executable,
        str(script),
        "--repo-root",
        str(repo),
        "--experiment-id",
        experiment_id,
        "--package-kind",
        package_kind,
        "--result-dir",
        str(output_root),
        "--output",
        str(artifact_output),
        "--test-command",
        "python3 -m pytest -q tests/test_experiment_artifact_protocol.py",
    ]
    for source in source_files:
        cmd.extend(["--source-file", source])
    return subprocess.run(cmd, cwd=repo, text=True, capture_output=True)


def main() -> int:
    args = parse_args()
    repo = args.repo_root.resolve()
    output_root = args.output_root.resolve()
    artifact_output = args.artifact_output.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    logs = output_root / "logs"
    logs.mkdir(exist_ok=True)
    log_path = logs / "supervised_run.log"
    heartbeat_path = output_root / "heartbeat.json"
    base_commit = git_value(repo, "rev-parse", "HEAD")
    branch = git_value(repo, "branch", "--show-current")
    dirty = bool(git_value(repo, "status", "--porcelain"))
    start_wall = time.time()
    start_utc = utc_now()

    run_manifest = {
        "schema_version": 1,
        "experiment_id": args.experiment_id,
        "execution_state": "running",
        "start_utc": start_utc,
        "repo_root": str(repo),
        "branch": branch,
        "base_commit": base_commit,
        "git_dirty_at_launch": dirty,
        "command": args.command,
        "cwd": str(repo),
        "python": sys.version,
        "platform": platform.platform(),
        "pid": None,
        "heartbeat_seconds": args.heartbeat_seconds,
        "stale_seconds": args.stale_seconds,
        "progress_globs": args.progress_glob,
        "required_outputs": args.required_output,
        "artifact_output": str(artifact_output),
    }
    atomic_json(output_root / "run_manifest.json", run_manifest)

    process = subprocess.Popen(
        args.command,
        cwd=repo,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=os.environ.copy(),
        start_new_session=True,
    )
    run_manifest["pid"] = process.pid
    atomic_json(output_root / "run_manifest.json", run_manifest)

    forwarded_signal: int | None = None

    def handle_signal(signum: int, _frame: Any) -> None:
        nonlocal forwarded_signal
        forwarded_signal = signum
        try:
            os.killpg(process.pid, signum)
        except ProcessLookupError:
            pass

    previous_handlers = {
        sig: signal.signal(sig, handle_signal) for sig in (signal.SIGINT, signal.SIGTERM)
    }

    events: queue.Queue[tuple[float, str]] = queue.Queue()
    assert process.stdout is not None
    reader = threading.Thread(target=stream_reader, args=(process.stdout, events), daemon=True)
    reader.start()
    last_console_activity = time.time()
    last_fs_activity = latest_mtime(output_root) or start_wall
    last_heartbeat = 0.0
    stale_detected = False

    try:
        with log_path.open("a", buffering=1) as log:
            log.write(f"[{start_utc}] START pid={process.pid} command={args.command!r}\n")
            while True:
                now = time.time()
                try:
                    while True:
                        event_time, line = events.get_nowait()
                        last_console_activity = event_time
                        sys.stdout.write(line)
                        sys.stdout.flush()
                        log.write(line)
                except queue.Empty:
                    pass

                current_mtime = latest_mtime(output_root)
                if current_mtime is not None:
                    last_fs_activity = max(last_fs_activity, current_mtime)
                last_activity = max(last_console_activity, last_fs_activity)
                stale_for = max(0.0, now - last_activity)
                if stale_for >= args.stale_seconds:
                    stale_detected = True
                    if args.fail_on_stale and process.poll() is None:
                        log.write(f"[{utc_now()}] STALE timeout={stale_for:.1f}s; terminating\n")
                        os.killpg(process.pid, signal.SIGTERM)

                if now - last_heartbeat >= args.heartbeat_seconds:
                    heartbeat = {
                        "experiment_id": args.experiment_id,
                        "execution_state": "running" if process.poll() is None else "exited",
                        "utc": utc_now(),
                        "pid": process.pid,
                        "elapsed_seconds": round(now - start_wall, 3),
                        "process_returncode": process.poll(),
                        "seconds_since_activity": round(stale_for, 3),
                        "stale_detected": stale_detected,
                        "progress": progress_counts(output_root, args.progress_glob),
                        "latest_output_mtime_utc": datetime.fromtimestamp(
                            last_fs_activity, timezone.utc
                        ).isoformat(),
                    }
                    atomic_json(heartbeat_path, heartbeat)
                    last_heartbeat = now

                if process.poll() is not None and events.empty() and not reader.is_alive():
                    break
                time.sleep(min(1.0, args.heartbeat_seconds / 4.0))

            reader.join(timeout=2)
            try:
                while True:
                    _event_time, line = events.get_nowait()
                    sys.stdout.write(line)
                    log.write(line)
            except queue.Empty:
                pass
            returncode = process.wait()
            log.write(f"[{utc_now()}] EXIT returncode={returncode}\n")
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)

    missing_outputs = [name for name in args.required_output if not (output_root / name).exists()]
    success = returncode == 0 and not missing_outputs and not (
        args.fail_on_stale and stale_detected
    )
    common = {
        "schema_version": 1,
        "experiment_id": args.experiment_id,
        "start_utc": start_utc,
        "end_utc": utc_now(),
        "elapsed_seconds": round(time.time() - start_wall, 3),
        "pid": process.pid,
        "returncode": returncode,
        "forwarded_signal": forwarded_signal,
        "stale_detected": stale_detected,
        "missing_required_outputs": missing_outputs,
        "base_commit": base_commit,
        "branch": branch,
        "command": args.command,
        "scientific_acceptance_pending": True,
    }
    if success:
        marker = output_root / "RUN_RAW_COMPLETE.json"
        atomic_json(marker, {**common, "execution_state": "raw_complete"})
        package_kind = "experiment-raw-complete"
    else:
        marker = output_root / "RUN_FAILED.json"
        atomic_json(marker, {**common, "execution_state": "failed"})
        package_kind = "experiment-failed"

    manifest = json.loads((output_root / "run_manifest.json").read_text())
    manifest.update(
        {
            "execution_state": "raw_complete" if success else "failed",
            "end_utc": common["end_utc"],
            "returncode": returncode,
            "missing_required_outputs": missing_outputs,
            "stale_detected": stale_detected,
        }
    )
    atomic_json(output_root / "run_manifest.json", manifest)

    packaged = package_recovery(
        repo,
        args.experiment_id,
        output_root,
        artifact_output,
        package_kind,
        args.source_file,
    )
    package_record = {
        "command": packaged.args,
        "returncode": packaged.returncode,
        "stdout": packaged.stdout,
        "stderr": packaged.stderr,
        "artifact_output": str(artifact_output),
        "artifact_exists": artifact_output.is_file(),
    }
    atomic_json(output_root / "recovery_package_status.json", package_record)
    if packaged.returncode != 0:
        print(packaged.stdout, end="")
        print(packaged.stderr, file=sys.stderr, end="")
        return 3

    print(packaged.stdout, end="")
    if success:
        print(
            "Raw computation completed and a recovery artifact was created. "
            "Terminal audit and final scientific packaging are still required."
        )
        return 0
    print("The run failed; partial outputs were preserved in a failed-run artifact.")
    return returncode if returncode != 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
