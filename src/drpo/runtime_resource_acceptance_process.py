"""Owned-process supervision for runtime-resource acceptance."""
from __future__ import annotations

import dataclasses
import json
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from drpo.runtime_resource_acceptance import AcceptanceError, utc_now


@dataclasses.dataclass(frozen=True)
class CommandResult:
    command: tuple[str, ...]
    cwd: str
    started_utc: str
    finished_utc: str
    elapsed_seconds: float
    returncode: int
    timed_out: bool
    controller_intervened: bool
    process_group_alive_after_cleanup: bool
    log_path: str
    samples_path: str | None
    peak_rss_bytes: int
    peak_process_count: int

    @property
    def ok(self) -> bool:
        return (
            self.returncode == 0
            and not self.timed_out
            and not self.controller_intervened
            and not self.process_group_alive_after_cleanup
        )

    def as_dict(self) -> dict[str, Any]:
        payload = dataclasses.asdict(self)
        payload["ok"] = self.ok
        return payload


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), sort_keys=True) + "\n")


def _proc_stat(path: Path) -> tuple[int, int] | None:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    close = text.rfind(")")
    if close < 0:
        return None
    fields = text[close + 2 :].split()
    if len(fields) < 3:
        return None
    try:
        return int(fields[1]), int(fields[2])
    except ValueError:
        return None


def process_group_snapshot(pgid: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        entries = list(Path("/proc").iterdir())
    except OSError:
        return rows
    for entry in entries:
        if not entry.name.isdigit():
            continue
        parsed = _proc_stat(entry / "stat")
        if parsed is None or parsed[1] != int(pgid):
            continue
        pid = int(entry.name)
        status: dict[str, str] = {}
        try:
            for line in (entry / "status").read_text(encoding="utf-8").splitlines():
                if ":" in line:
                    key, raw = line.split(":", 1)
                    status[key] = raw.strip()
        except OSError:
            pass
        try:
            command = (
                (entry / "cmdline")
                .read_bytes()
                .replace(b"\x00", b" ")
                .decode("utf-8", errors="replace")
                .strip()
            )
        except OSError:
            command = ""
        try:
            affinity = sorted(int(value) for value in os.sched_getaffinity(pid))
        except (OSError, PermissionError, ProcessLookupError):
            affinity = []
        try:
            rss_bytes = int(status.get("VmRSS", "0 kB").split()[0]) * 1024
        except (IndexError, ValueError):
            rss_bytes = 0
        try:
            threads = int(status.get("Threads", "0"))
        except ValueError:
            threads = 0
        rows.append(
            {
                "pid": pid,
                "ppid": parsed[0],
                "pgid": parsed[1],
                "threads": threads,
                "rss_bytes": rss_bytes,
                "affinity_cpu_ids": affinity,
                "command": command,
            }
        )
    return sorted(rows, key=lambda row: int(row["pid"]))


def group_alive(pgid: int) -> bool:
    try:
        os.killpg(int(pgid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def terminate_group(pgid: int, grace_seconds: float = 5.0) -> bool:
    if not group_alive(pgid):
        return False
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return True
    deadline = time.monotonic() + grace_seconds
    while group_alive(pgid) and time.monotonic() < deadline:
        time.sleep(0.05)
    if group_alive(pgid):
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    return True


def run_command(
    command: Sequence[str],
    *,
    cwd: Path,
    environment: Mapping[str, str],
    timeout_seconds: float,
    log_path: Path,
    samples_path: Path | None,
    command_ledger: Path,
    sample_interval_seconds: float = 1.0,
) -> CommandResult:
    if timeout_seconds <= 0 or sample_interval_seconds <= 0:
        raise AcceptanceError("command timeouts must be positive")
    argv = tuple(str(item) for item in command)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if samples_path is not None:
        samples_path.parent.mkdir(parents=True, exist_ok=True)
    started_utc = utc_now()
    started = time.monotonic()
    _append_jsonl(
        command_ledger,
        {
            "started_utc": started_utc,
            "command": list(argv),
            "cwd": str(cwd),
            "timeout_seconds": timeout_seconds,
        },
    )
    peak_rss = 0
    peak_count = 0
    timed_out = False
    intervened = False
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("COMMAND=" + json.dumps(list(argv)) + "\n")
        handle.flush()
        process = subprocess.Popen(
            list(argv),
            cwd=str(cwd),
            env=dict(environment),
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        deadline = started + timeout_seconds
        while True:
            rows = process_group_snapshot(process.pid)
            peak_count = max(peak_count, len(rows))
            peak_rss = max(peak_rss, sum(int(row["rss_bytes"]) for row in rows))
            if samples_path is not None:
                _append_jsonl(
                    samples_path,
                    {
                        "captured_utc": utc_now(),
                        "root_pid": process.pid,
                        "returncode": process.poll(),
                        "processes": rows,
                    },
                )
            returncode = process.poll()
            if returncode is not None:
                break
            if time.monotonic() >= deadline:
                timed_out = True
                intervened = terminate_group(process.pid)
                break
            time.sleep(sample_interval_seconds)
        if process.poll() is None or group_alive(process.pid):
            intervened = terminate_group(process.pid) or intervened
        returncode = int(process.wait())
    result = CommandResult(
        command=argv,
        cwd=str(cwd),
        started_utc=started_utc,
        finished_utc=utc_now(),
        elapsed_seconds=time.monotonic() - started,
        returncode=returncode,
        timed_out=timed_out,
        controller_intervened=intervened,
        process_group_alive_after_cleanup=group_alive(process.pid),
        log_path=str(log_path),
        samples_path=None if samples_path is None else str(samples_path),
        peak_rss_bytes=peak_rss,
        peak_process_count=peak_count,
    )
    _append_jsonl(command_ledger, {"finished_utc": result.finished_utc, "result": result.as_dict()})
    return result


def run_concurrent(
    commands: Mapping[str, Sequence[str]],
    *,
    cwd_by_name: Mapping[str, Path],
    environment_by_name: Mapping[str, Mapping[str, str]],
    timeout_seconds: float,
    log_dir: Path,
    samples_path: Path,
    sample_interval_seconds: float,
    command_ledger: Path,
) -> dict[str, CommandResult]:
    log_dir.mkdir(parents=True, exist_ok=True)
    processes: dict[str, subprocess.Popen[str]] = {}
    handles: dict[str, Any] = {}
    meta: dict[str, dict[str, Any]] = {}
    started = time.monotonic()
    try:
        for name, command in commands.items():
            argv = tuple(str(item) for item in command)
            log = log_dir / f"{name}.log"
            handle = log.open("w", encoding="utf-8")
            process = subprocess.Popen(
                list(argv),
                cwd=str(cwd_by_name[name]),
                env=dict(environment_by_name[name]),
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            processes[name] = process
            handles[name] = handle
            meta[name] = {
                "command": argv,
                "started_utc": utc_now(),
                "log": log,
                "peak_rss": 0,
                "peak_count": 0,
                "timed_out": False,
                "intervened": False,
            }
            _append_jsonl(
                command_ledger,
                {"started_utc": meta[name]["started_utc"], "name": name, "command": list(argv)},
            )
        deadline = started + timeout_seconds
        while True:
            all_done = True
            sample: dict[str, Any] = {"captured_utc": utc_now(), "commands": {}}
            for name, process in processes.items():
                rows = process_group_snapshot(process.pid)
                meta[name]["peak_count"] = max(int(meta[name]["peak_count"]), len(rows))
                meta[name]["peak_rss"] = max(
                    int(meta[name]["peak_rss"]),
                    sum(int(row["rss_bytes"]) for row in rows),
                )
                sample["commands"][name] = {
                    "root_pid": process.pid,
                    "returncode": process.poll(),
                    "processes": rows,
                }
                if process.poll() is None:
                    all_done = False
            _append_jsonl(samples_path, sample)
            if all_done:
                break
            if time.monotonic() >= deadline:
                for name, process in processes.items():
                    if process.poll() is None:
                        meta[name]["timed_out"] = True
                        meta[name]["intervened"] = terminate_group(process.pid)
                break
            time.sleep(sample_interval_seconds)
    finally:
        for name, process in processes.items():
            if process.poll() is None or group_alive(process.pid):
                meta[name]["intervened"] = terminate_group(process.pid) or bool(
                    meta[name]["intervened"]
                )
            process.wait()
            handles[name].close()
    results: dict[str, CommandResult] = {}
    for name, process in processes.items():
        item = meta[name]
        result = CommandResult(
            command=tuple(item["command"]),
            cwd=str(cwd_by_name[name]),
            started_utc=str(item["started_utc"]),
            finished_utc=utc_now(),
            elapsed_seconds=time.monotonic() - started,
            returncode=int(process.returncode),
            timed_out=bool(item["timed_out"]),
            controller_intervened=bool(item["intervened"]),
            process_group_alive_after_cleanup=group_alive(process.pid),
            log_path=str(item["log"]),
            samples_path=str(samples_path),
            peak_rss_bytes=int(item["peak_rss"]),
            peak_process_count=int(item["peak_count"]),
        )
        results[name] = result
        _append_jsonl(
            command_ledger,
            {"finished_utc": result.finished_utc, "name": name, "result": result.as_dict()},
        )
    return results
