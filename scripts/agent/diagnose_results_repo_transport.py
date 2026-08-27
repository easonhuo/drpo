#!/usr/bin/env python3
"""Read-only preflight for DRPO results-repository transport.

The probe never claims a RunSpec, clones a repository, writes a commit, or performs a
push. It inspects local configuration and performs DNS, TCP, and ``git ls-remote``
checks so network, configuration, and authentication blockers can be separated before
promoting a results-delivery shadow RunSpec.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

DEFAULT_REPOSITORY = "easonhuo/drpo-results"
DEFAULT_REMOTE = f"git@github.com:{DEFAULT_REPOSITORY}.git"
STATUS_READY = "READY_FOR_SHADOW_READ_PREFLIGHT"
STATUS_NETWORK = "BLOCKED_BY_NETWORK"
STATUS_CREDENTIAL = "BLOCKED_BY_CREDENTIAL"
STATUS_CONFIGURATION = "BLOCKED_BY_CONFIGURATION"

NETWORK_PATTERNS = (
    "connection timed out",
    "operation timed out",
    "could not resolve hostname",
    "could not resolve host",
    "name or service not known",
    "temporary failure in name resolution",
    "network is unreachable",
    "no route to host",
    "connection refused",
    "failed to connect",
)
AUTH_PATTERNS = (
    "permission denied",
    "authentication failed",
    "could not read username",
    "repository not found",
    "access denied",
    "publickey",
)
CONFIG_PATTERNS = (
    "host key verification failed",
    "bad configuration option",
    "bad owner or permissions",
    "identity file",
    "proxycommand",
    "proxyjump",
    "unknown port",
    "unsupported protocol",
    "invalid url",
)
TOKEN_PATTERNS = (
    re.compile(r"github_pat_[A-Za-z0-9_]+"),
    re.compile(r"gh[pousr]_[A-Za-z0-9]+"),
)
SCP_REMOTE_RE = re.compile(r"^(?:(?P<user>[^/@:]+)@)?(?P<host>[^/:]+):(?P<path>.+)$")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _known_secrets() -> list[str]:
    values: list[str] = []
    for key in ("GH_TOKEN", "GITHUB_TOKEN"):
        value = os.environ.get(key, "")
        if value:
            values.append(value)
    override = os.environ.get("DRPO_RESULTS_REMOTE_URL", "")
    if override:
        parsed = urlsplit(override)
        if parsed.password:
            values.append(parsed.password)
        if parsed.username and parsed.scheme in {"http", "https"}:
            values.append(parsed.username)
    return values


def redact(text: str, extra_secrets: list[str] | None = None) -> str:
    """Remove known credentials and URL userinfo from diagnostic text."""

    result = text
    for secret in [*_known_secrets(), *(extra_secrets or [])]:
        if secret:
            result = result.replace(secret, "***REDACTED***")
    for pattern in TOKEN_PATTERNS:
        result = pattern.sub("***REDACTED_TOKEN***", result)
    result = re.sub(
        r"(https?://)([^/@\s]+)@",
        r"\1***REDACTED_USERINFO***@",
        result,
        flags=re.IGNORECASE,
    )
    return result


def sanitize_remote(remote: str) -> str:
    try:
        parsed = urlsplit(remote)
    except ValueError:
        return redact(remote)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return redact(remote)
    host = parsed.hostname or ""
    if parsed.port:
        host = f"{host}:{parsed.port}"
    netloc = host
    if parsed.username or parsed.password:
        netloc = f"***REDACTED_USERINFO***@{host}"
    return urlunsplit((parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment))


def run_command(
    command: list[str],
    *,
    timeout: float,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    started = _utc_now()
    try:
        proc = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        return {
            "command": [redact(part) for part in command],
            "started_at": started,
            "returncode": None,
            "timed_out": True,
            "stdout": redact(stdout.strip()),
            "stderr": redact(stderr.strip()),
        }
    except OSError as exc:
        return {
            "command": [redact(part) for part in command],
            "started_at": started,
            "returncode": None,
            "timed_out": False,
            "stdout": "",
            "stderr": redact(str(exc)),
            "os_error": True,
        }
    return {
        "command": [redact(part) for part in command],
        "started_at": started,
        "returncode": proc.returncode,
        "timed_out": False,
        "stdout": redact(proc.stdout.strip()),
        "stderr": redact(proc.stderr.strip()),
    }


def parse_ssh_g(output: str) -> dict[str, Any]:
    """Parse the subset of ``ssh -G`` output relevant to transport diagnosis."""

    parsed: dict[str, Any] = {"identityfile": []}
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line or " " not in line:
            continue
        key, value = line.split(None, 1)
        key = key.lower()
        if key == "identityfile":
            parsed["identityfile"].append(value)
        elif key in {
            "hostname",
            "port",
            "user",
            "proxycommand",
            "proxyjump",
            "canonicalizehostname",
        }:
            parsed[key] = value
    if "port" in parsed:
        try:
            parsed["port"] = int(parsed["port"])
        except ValueError:
            pass
    return parsed


def resolve_host(host: str, port: int) -> dict[str, Any]:
    try:
        rows = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        return {"host": host, "port": port, "ok": False, "error": redact(str(exc))}
    addresses: list[str] = []
    for row in rows:
        address = str(row[4][0])
        if address not in addresses:
            addresses.append(address)
    return {"host": host, "port": port, "ok": True, "addresses": addresses[:8]}


def probe_tcp(host: str, port: int, timeout: float, label: str) -> dict[str, Any]:
    result: dict[str, Any] = {"label": label, "host": host, "port": port}
    try:
        with socket.create_connection((host, port), timeout=timeout):
            result["ok"] = True
    except OSError as exc:
        result["ok"] = False
        result["error"] = redact(str(exc))
    return result


def derive_remote_target(
    remote: str,
    ssh_effective: dict[str, Any],
) -> dict[str, Any]:
    scp_match = SCP_REMOTE_RE.match(remote)
    if scp_match and "://" not in remote:
        configured_host = scp_match.group("host")
        host = str(ssh_effective.get("hostname") or configured_host)
        port = ssh_effective.get("port", 22)
        try:
            port = int(port)
        except (TypeError, ValueError):
            port = 22
        return {
            "transport": "ssh",
            "configured_host": configured_host,
            "host": host,
            "port": port,
        }
    parsed = urlsplit(remote)
    scheme = parsed.scheme.lower()
    if scheme in {"ssh", "git+ssh"}:
        return {
            "transport": "ssh",
            "configured_host": parsed.hostname,
            "host": parsed.hostname,
            "port": parsed.port or 22,
        }
    if scheme in {"http", "https"}:
        return {
            "transport": scheme,
            "configured_host": parsed.hostname,
            "host": parsed.hostname,
            "port": parsed.port or (443 if scheme == "https" else 80),
        }
    return {
        "transport": "unknown",
        "configured_host": parsed.hostname,
        "host": parsed.hostname,
        "port": parsed.port,
    }


def classify_ls_remote(result: dict[str, Any], *, target_tcp_ok: bool) -> str:
    if result.get("returncode") == 0 and not result.get("timed_out"):
        return STATUS_READY
    detail = f"{result.get('stderr', '')}\n{result.get('stdout', '')}".lower()
    if any(pattern in detail for pattern in CONFIG_PATTERNS):
        return STATUS_CONFIGURATION
    if any(pattern in detail for pattern in AUTH_PATTERNS):
        return STATUS_CREDENTIAL
    if result.get("timed_out") or any(pattern in detail for pattern in NETWORK_PATTERNS):
        return STATUS_NETWORK
    return STATUS_CREDENTIAL if target_tcp_ok else STATUS_NETWORK


def credential_inventory(timeout: float) -> dict[str, Any]:
    git_helper = run_command(
        ["git", "config", "--global", "--get-all", "credential.helper"],
        timeout=timeout,
    )
    ssh_agent = run_command(["ssh-add", "-l"], timeout=timeout)
    return {
        "environment_presence": {
            key: bool(os.environ.get(key))
            for key in (
                "GH_TOKEN",
                "GITHUB_TOKEN",
                "DRPO_RESULTS_REMOTE_URL",
                "GIT_SSH_COMMAND",
                "SSH_AUTH_SOCK",
            )
        },
        "gh_available": shutil.which("gh") is not None,
        "git_credential_helpers": {
            "configured": bool(git_helper.get("stdout", "").strip()),
            "count": len(
                [line for line in git_helper.get("stdout", "").splitlines() if line]
            ),
        },
        "netrc_present": (Path.home() / ".netrc").is_file(),
        "ssh_config_present": (Path.home() / ".ssh" / "config").is_file(),
        "ssh_agent": {
            "returncode": ssh_agent.get("returncode"),
            "timed_out": ssh_agent.get("timed_out"),
            "identity_count": len(
                [line for line in ssh_agent.get("stdout", "").splitlines() if line]
            ),
            "error_class": (
                "none" if ssh_agent.get("returncode") == 0 else "unavailable_or_empty"
            ),
        },
    }


def build_report(remote: str, timeout: float) -> dict[str, Any]:
    ssh_available = shutil.which("ssh") is not None
    git_available = shutil.which("git") is not None
    ssh_probe: dict[str, Any]
    ssh_effective: dict[str, Any] = {}
    if ssh_available:
        ssh_probe = run_command(["ssh", "-G", "github.com"], timeout=timeout)
        if ssh_probe.get("returncode") == 0:
            ssh_effective = parse_ssh_g(ssh_probe.get("stdout", ""))
    else:
        ssh_probe = {"available": False}

    target = derive_remote_target(remote, ssh_effective)
    tcp_probes: list[dict[str, Any]] = []
    target_host = target.get("host")
    target_port = target.get("port")
    if target_host and target_port:
        tcp_probes.append(
            probe_tcp(str(target_host), int(target_port), timeout, "configured_remote")
        )
    fallback_rows = [
        ("github_ssh_default", "github.com", 22),
        ("github_ssh_443", "ssh.github.com", 443),
        ("github_https", "github.com", 443),
    ]
    existing = {(row["host"], row["port"]) for row in tcp_probes}
    for label, host, port in fallback_rows:
        if (host, port) not in existing:
            tcp_probes.append(probe_tcp(host, port, timeout, label))

    dns: list[dict[str, Any]] = []
    seen_dns: set[tuple[str, int]] = set()
    for row in tcp_probes:
        key = (str(row["host"]), int(row["port"]))
        if key not in seen_dns:
            dns.append(resolve_host(*key))
            seen_dns.add(key)

    if git_available:
        git_env = os.environ.copy()
        git_env["GIT_TERMINAL_PROMPT"] = "0"
        if target.get("transport") == "ssh":
            git_env["GIT_SSH_COMMAND"] = (
                f"ssh -o BatchMode=yes -o ConnectTimeout={max(1, int(timeout))} "
                "-o ConnectionAttempts=1"
            )
        ls_remote = run_command(
            ["git", "ls-remote", "--heads", "--tags", remote],
            timeout=max(timeout * 2, timeout + 2),
            env=git_env,
        )
    else:
        ls_remote = {
            "returncode": None,
            "timed_out": False,
            "stderr": "git executable is unavailable",
            "stdout": "",
        }

    target_tcp_ok = any(
        row.get("label") == "configured_remote" and row.get("ok") is True
        for row in tcp_probes
    )
    if not git_available or target.get("transport") == "unknown":
        status = STATUS_CONFIGURATION
    else:
        status = classify_ls_remote(ls_remote, target_tcp_ok=target_tcp_ok)

    warnings: list[str] = []
    effective_port = ssh_effective.get("port")
    if target.get("transport") == "ssh" and effective_port not in (None, 22, 443):
        warnings.append(
            f"ssh -G maps github.com to non-standard port {effective_port}; inspect "
            "~/.ssh/config, Include files, ProxyCommand, and ProxyJump before adding keys"
        )
    if os.environ.get("GIT_SSH_COMMAND"):
        warnings.append(
            "GIT_SSH_COMMAND is set in the shell; the probe reports its presence but uses "
            "a non-interactive timeout-bounded SSH command for git ls-remote"
        )
    if status == STATUS_READY:
        next_action = (
            "Read authentication works. Provision or confirm repository-scoped write "
            "authority, then run the preserved no-training shadow exactly once; write "
            "permission remains unverified until that push."
        )
    elif status == STATUS_NETWORK:
        next_action = (
            "Repair routing/proxy/port reachability first. Prefer a reachable SSH endpoint "
            "or an HTTPS remote; do not provision credentials into a transport that cannot "
            "establish TCP connectivity."
        )
    elif status == STATUS_CONFIGURATION:
        next_action = (
            "Correct SSH/remote configuration before credential work. Re-run this probe "
            "until git ls-remote reaches an authentication decision or succeeds."
        )
    else:
        next_action = (
            "Network transport is reachable but repository authentication failed. Install "
            "a repository-scoped deploy key or fine-grained token without storing it in "
            "the RunSpec, then re-run this probe."
        )

    return {
        "schema_version": 1,
        "probe": "SERVER-RESULTS-TRANSPORT-PREFLIGHT-01",
        "generated_at": _utc_now(),
        "status": status,
        "read_only": True,
        "write_capability_verified": False,
        "remote": sanitize_remote(remote),
        "remote_source": (
            "argument"
            if remote != os.environ.get("DRPO_RESULTS_REMOTE_URL", "").strip()
            and remote != DEFAULT_REMOTE
            else (
                "DRPO_RESULTS_REMOTE_URL"
                if os.environ.get("DRPO_RESULTS_REMOTE_URL", "").strip()
                else "default_ssh"
            )
        ),
        "target": target,
        "executables": {
            "git": shutil.which("git"),
            "ssh": shutil.which("ssh"),
            "gh": shutil.which("gh"),
        },
        "credential_inventory": credential_inventory(timeout),
        "ssh_effective": ssh_effective,
        "ssh_g": {
            "returncode": ssh_probe.get("returncode"),
            "timed_out": ssh_probe.get("timed_out"),
            "stderr": ssh_probe.get("stderr", ""),
        },
        "dns": dns,
        "tcp": tcp_probes,
        "git_ls_remote": ls_remote,
        "warnings": warnings,
        "next_action": next_action,
        "limitations": [
            "git ls-remote is read-only and cannot prove push permission",
            "the probe does not claim or execute a RunSpec",
            "the probe does not clone, commit, or push either repository",
        ],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--remote",
        default=os.environ.get("DRPO_RESULTS_REMOTE_URL", "").strip() or DEFAULT_REMOTE,
        help="Results repository remote; defaults to DRPO_RESULTS_REMOTE_URL or canonical SSH",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=5.0,
        help="Per DNS/TCP/command timeout budget where applicable (default: 5)",
    )
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    parser.add_argument("--json", action="store_true", help="Print full JSON report")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return exit 2 when the status is not READY_FOR_SHADOW_READ_PREFLIGHT",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.timeout_seconds <= 0:
        raise SystemExit("--timeout-seconds must be positive")
    report = build_report(args.remote, args.timeout_seconds)
    encoded = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    if args.json:
        sys.stdout.write(encoded)
    else:
        print(f"status={report['status']}")
        print(f"remote={report['remote']}")
        print(f"write_capability_verified={str(report['write_capability_verified']).lower()}")
        for warning in report["warnings"]:
            print(f"warning={warning}")
        print(f"next_action={report['next_action']}")
        if args.output:
            print(f"report={args.output}")
    if args.strict and report["status"] != STATUS_READY:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
