from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest

from drpo import runtime_resource_pool as pool


SCRIPT = Path("scripts/run_with_resource_pool.py")
SPEC = importlib.util.spec_from_file_location("run_with_resource_pool", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
wrapper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(wrapper)


def test_cpu_pool_parse_and_format() -> None:
    assert pool.parse_cpu_pool("0-2,5,7-8") == (0, 1, 2, 5, 7, 8)
    assert pool.format_cpu_pool((0, 1, 2, 5, 7, 8)) == "0-2,5,7-8"


@pytest.mark.parametrize(
    "value",
    ["", ",", "1,", "-1", "3-1", "a", "1-2-3", "1,1", "1-3,2"],
)
def test_malformed_or_duplicate_cpu_pool_is_rejected(value: str) -> None:
    with pytest.raises(pool.ResourcePoolError):
        pool.parse_cpu_pool(value)


def test_gpu_pool_is_ordered_and_unique() -> None:
    assert pool.parse_gpu_pool("3,1,GPU-abcd") == ("3", "1", "GPU-abcd")
    with pytest.raises(pool.ResourcePoolError, match="duplicate"):
        pool.parse_gpu_pool("1,1")


def test_explicit_pool_must_be_inherited_subset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: {0, 1, 2})
    with pytest.raises(pool.ResourcePoolError, match="outside inherited"):
        pool.activate_resource_pool(cpu_pool="1,3", gpu_enforcement="none")


def test_explicit_pool_is_applied_and_exported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = {"affinity": {0, 1, 2, 3}}
    monkeypatch.setattr(
        os, "sched_getaffinity", lambda _pid: set(state["affinity"])
    )
    monkeypatch.setattr(
        os,
        "sched_setaffinity",
        lambda _pid, values: state.__setitem__("affinity", set(values)),
    )
    environment: dict[str, str] = {}
    selected = pool.activate_resource_pool(
        cpu_pool="1-2",
        gpu_pool="4,5",
        gpu_enforcement="cuda_visible",
        environ=environment,
    )
    assert selected.source == "explicit_cli"
    assert selected.effective_cpu_ids == (1, 2)
    assert environment["DRPO_CPU_POOL"] == "1-2"
    assert environment["CUDA_VISIBLE_DEVICES"] == "4,5"
    assert environment["DRPO_RESOURCE_POOL_DIGEST"] == selected.pool_digest


def test_inherited_pool_is_explicit_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: {2, 4})
    environment: dict[str, str] = {}
    selected = pool.activate_resource_pool(
        cpu_pool=None,
        gpu_enforcement="none",
        environ=environment,
    )
    assert selected.source == "inherited_affinity"
    assert selected.requested_cpu_ids == (2, 4)
    assert selected.effective_cpu_ids == (2, 4)


def test_pool_digest_is_stable_and_source_sensitive() -> None:
    explicit = pool.ResourcePool(
        "explicit_cli", (0, 1), (0,), (0,), (), "none"
    )
    same = pool.ResourcePool(
        "explicit_cli", (0, 1), (0,), (0,), (), "none"
    )
    inherited = pool.ResourcePool(
        "inherited_affinity", (0,), (0,), (0,), (), "none"
    )
    assert explicit.pool_digest == same.pool_digest
    assert explicit.pool_digest != inherited.pool_digest


def test_immutable_pool_identity_rejects_mismatch(tmp_path: Path) -> None:
    first = pool.ResourcePool(
        "explicit_cli", (0, 1), (0,), (0,), (), "none"
    )
    second = pool.ResourcePool(
        "explicit_cli", (0, 1), (1,), (1,), (), "none"
    )
    target = tmp_path / "RESOURCE_POOL.json"
    pool.write_pool_identity(target, first)
    pool.write_pool_identity(target, first)
    assert json.loads(target.read_text())["pool_digest"] == first.pool_digest
    with pytest.raises(pool.ResourcePoolError, match="does not match"):
        pool.write_pool_identity(target, second)


def test_launcher_argument_gpu_pool_must_match_command() -> None:
    selected = pool.ResourcePool(
        "explicit_cli", (0,), (0,), (0,), ("4", "5"), "launcher_argument"
    )
    pool.validate_delegated_gpu_pool(["python", "x.py", "--gpus", "4,5"], selected)
    with pytest.raises(pool.ResourcePoolError, match="does not match"):
        pool.validate_delegated_gpu_pool(
            ["python", "x.py", "--gpus=5,4"], selected
        )
    with pytest.raises(pool.ResourcePoolError, match="requires"):
        pool.validate_delegated_gpu_pool(["python", "x.py"], selected)


def test_wrapper_dry_run_writes_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    state = {"affinity": {0, 1, 2}}
    monkeypatch.setattr(
        os, "sched_getaffinity", lambda _pid: set(state["affinity"])
    )
    monkeypatch.setattr(
        os,
        "sched_setaffinity",
        lambda _pid, values: state.__setitem__("affinity", set(values)),
    )
    target = tmp_path / "RESOURCE_POOL.json"
    result = wrapper.main(
        [
            "--cpu-pool",
            "0-1",
            "--gpu-pool",
            "4,5",
            "--pool-identity",
            str(target),
            "--dry-run",
            "--",
            "python",
            "launcher.py",
            "--gpus",
            "4,5",
        ]
    )
    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["resource_pool"]["effective_cpu_ids"] == [0, 1]
    assert payload["resource_pool"]["requested_gpu_ids"] == ["4", "5"]
    assert target.is_file()


def test_wrapper_exec_preserves_exact_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: {0})
    observed: dict[str, object] = {}

    def fake_exec(program: str, command: list[str], environment: dict[str, str]) -> None:
        observed["program"] = program
        observed["command"] = command
        observed["digest"] = environment["DRPO_RESOURCE_POOL_DIGEST"]
        raise RuntimeError("stop")

    monkeypatch.setattr(os, "execvpe", fake_exec)
    with pytest.raises(RuntimeError, match="stop"):
        wrapper.main(
            [
                "--pool-identity",
                str(tmp_path / "RESOURCE_POOL.json"),
                "--",
                "python",
                "-c",
                "print('ok')",
            ]
        )
    assert observed["program"] == "python"
    assert observed["command"] == ["python", "-c", "print('ok')"]
