from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from drpo import runtime_resource_acceptance_process as process


def test_run_command_cleans_owned_group_when_polling_is_interrupted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_sleep = process.time.sleep
    sleep_calls = 0
    terminated: list[int] = []

    def interrupted_sleep(seconds: float) -> None:
        nonlocal sleep_calls
        sleep_calls += 1
        if sleep_calls == 1:
            raise KeyboardInterrupt
        original_sleep(seconds)

    def fast_terminate(pgid: int, grace_seconds: float = 5.0) -> bool:
        del grace_seconds
        terminated.append(pgid)
        try:
            os.killpg(pgid, process.signal.SIGKILL)
        except ProcessLookupError:
            return False
        return True

    monkeypatch.setattr(process.time, "sleep", interrupted_sleep)
    monkeypatch.setattr(process, "terminate_group", fast_terminate)
    with pytest.raises(KeyboardInterrupt):
        process.run_command(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            cwd=tmp_path,
            environment=os.environ.copy(),
            timeout_seconds=30,
            log_path=tmp_path / "interrupt.log",
            samples_path=tmp_path / "samples.jsonl",
            sample_interval_seconds=0.01,
            command_ledger=tmp_path / "commands.jsonl",
        )
    assert len(terminated) >= 1
    assert process.group_alive(terminated[0]) is False
    assert "owned_process_group_cleaned" in (tmp_path / "commands.jsonl").read_text()
