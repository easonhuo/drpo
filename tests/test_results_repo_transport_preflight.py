from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
AGENT_DIR = ROOT / "scripts" / "agent"
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

import diagnose_results_repo_transport as probe  # noqa: E402


def test_redact_hides_tokens_and_url_userinfo(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GH_TOKEN", "ghp_supersecret")
    text = "https://user:pass@github.com/x/y ghp_supersecret github_pat_ABC123"
    redacted = probe.redact(text)
    assert "ghp_supersecret" not in redacted
    assert "github_pat_ABC123" not in redacted
    assert "user:pass" not in redacted
    assert "REDACTED" in redacted


def test_parse_ssh_g_extracts_nonstandard_port_and_proxy() -> None:
    parsed = probe.parse_ssh_g(
        "\n".join(
            [
                "host github.com",
                "hostname github.com",
                "user git",
                "port 36000",
                "proxyjump bastion.example",
                "identityfile ~/.ssh/id_ed25519",
                "identityfile ~/.ssh/id_rsa",
            ]
        )
    )
    assert parsed["hostname"] == "github.com"
    assert parsed["port"] == 36000
    assert parsed["proxyjump"] == "bastion.example"
    assert parsed["identityfile"] == ["~/.ssh/id_ed25519", "~/.ssh/id_rsa"]


def test_derive_remote_target_honors_effective_ssh_config() -> None:
    target = probe.derive_remote_target(
        "git@github.com:easonhuo/drpo-results.git",
        {"hostname": "proxy.internal", "port": 36000},
    )
    assert target == {
        "transport": "ssh",
        "configured_host": "github.com",
        "host": "proxy.internal",
        "port": 36000,
    }


@pytest.mark.parametrize(
    ("result", "target_tcp_ok", "expected"),
    [
        (
            {"returncode": 0, "timed_out": False, "stderr": "", "stdout": ""},
            True,
            probe.STATUS_READY,
        ),
        (
            {"returncode": None, "timed_out": True, "stderr": "", "stdout": ""},
            False,
            probe.STATUS_NETWORK,
        ),
        (
            {
                "returncode": 128,
                "timed_out": False,
                "stderr": "Permission denied (publickey).",
                "stdout": "",
            },
            True,
            probe.STATUS_CREDENTIAL,
        ),
        (
            {
                "returncode": 128,
                "timed_out": False,
                "stderr": "Host key verification failed.",
                "stdout": "",
            },
            True,
            probe.STATUS_CONFIGURATION,
        ),
        (
            {
                "returncode": 128,
                "timed_out": False,
                "stderr": "unknown failure",
                "stdout": "",
            },
            True,
            probe.STATUS_CREDENTIAL,
        ),
        (
            {
                "returncode": 128,
                "timed_out": False,
                "stderr": "unknown failure",
                "stdout": "",
            },
            False,
            probe.STATUS_NETWORK,
        ),
    ],
)
def test_classify_ls_remote(
    result: dict[str, object], target_tcp_ok: bool, expected: str
) -> None:
    assert probe.classify_ls_remote(result, target_tcp_ok=target_tcp_ok) == expected


def test_build_report_is_read_only_and_warns_on_nonstandard_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[list[str]] = []

    def fake_which(name: str) -> str | None:
        if name in {"git", "ssh"}:
            return f"/usr/bin/{name}"
        return None

    def fake_run_command(
        command: list[str], *, timeout: float, env: dict[str, str] | None = None
    ) -> dict[str, object]:
        del timeout, env
        commands.append(command)
        if command[:2] == ["ssh", "-G"]:
            return {
                "returncode": 0,
                "timed_out": False,
                "stdout": "hostname github.com\nport 36000\nuser git\n",
                "stderr": "",
            }
        if command[:3] == ["git", "ls-remote", "--heads"]:
            return {
                "returncode": 128,
                "timed_out": True,
                "stdout": "",
                "stderr": "Connection timed out",
            }
        return {
            "returncode": 1,
            "timed_out": False,
            "stdout": "",
            "stderr": "",
        }

    monkeypatch.setattr(probe.shutil, "which", fake_which)
    monkeypatch.setattr(probe, "run_command", fake_run_command)
    monkeypatch.setattr(
        probe,
        "probe_tcp",
        lambda host, port, timeout, label: {
            "label": label,
            "host": host,
            "port": port,
            "ok": False,
        },
    )
    monkeypatch.setattr(
        probe,
        "resolve_host",
        lambda host, port: {
            "host": host,
            "port": port,
            "ok": True,
            "addresses": ["127.0.0.1"],
        },
    )
    monkeypatch.setattr(
        probe,
        "credential_inventory",
        lambda timeout: {"environment_presence": {}, "gh_available": False},
    )

    report = probe.build_report(probe.DEFAULT_REMOTE, 1.0)

    assert report["status"] == probe.STATUS_NETWORK
    assert report["read_only"] is True
    assert report["write_capability_verified"] is False
    assert any("non-standard port 36000" in warning for warning in report["warnings"])
    flattened = [" ".join(command) for command in commands]
    assert not any(" clone " in f" {command} " for command in flattened)
    assert not any(" push " in f" {command} " for command in flattened)
    assert not any(" commit " in f" {command} " for command in flattened)
    assert any(command.startswith("git ls-remote") for command in flattened)
