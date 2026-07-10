from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OBSERVER = REPO_ROOT / "scripts" / "run_pr_gate_observer.py"


def git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip()


def commit_all(repo: Path, message: str) -> str:
    git(repo, "add", "-A")
    git(
        repo,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-m",
        message,
    )
    return git(repo, "rev-parse", "HEAD")


def make_repo(tmp_path: Path) -> tuple[Path, str, str]:
    repo = tmp_path / "repo"
    subprocess.run(["git", "init", str(repo)], check=True, stdout=subprocess.PIPE)
    (repo / "tracked.txt").write_text("base\n", encoding="utf-8")
    base = commit_all(repo, "base")
    (repo / "tracked.txt").write_text("head\n", encoding="utf-8")
    head = commit_all(repo, "head")
    return repo, base, head


def write_map(path: Path, commands: list[list[str]]) -> Path:
    payload = {
        "schema_version": 1,
        "unknown_path_policy": "full",
        "control_plane_patterns": [],
        "groups": [
            {
                "id": "tracked",
                "risk": "low",
                "patterns": ["tracked.txt"],
                "pytest_targets": [],
                "validators": [],
            }
        ],
        "full_commands": commands,
    }
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    return path


def run_observer(
    repo: Path,
    base: str,
    head: str,
    impact_map: Path,
    output_dir: Path,
    *extra: str,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(OBSERVER),
            "--repo",
            str(repo),
            "--base",
            base,
            "--head",
            head,
            "--mode",
            "full",
            "--map",
            str(impact_map),
            "--output-dir",
            str(output_dir),
            *extra,
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def test_observer_records_all_gate_results_without_blocking(tmp_path: Path):
    repo, base, head = make_repo(tmp_path)
    impact_map = write_map(
        tmp_path / "map.json",
        [
            ["{python}", "-c", "print('pass')"],
            ["{python}", "-c", "raise SystemExit(7)"],
        ],
    )
    output_dir = tmp_path / "observer"

    proc = run_observer(repo, base, head, impact_map, output_dir)

    assert proc.returncode == 0, proc.stderr
    report = json.loads((output_dir / "gate_report.json").read_text(encoding="utf-8"))
    assert report["enforcement"] == "observe_only"
    assert report["counts"] == {"total": 2, "pass": 1, "fail": 1, "unavailable": 0}
    assert [gate["status"] for gate in report["gates"]] == ["pass", "fail"]
    assert report["gates"][1]["classification"] == "actionable"
    assert report["gates"][1]["merge_blocking"] is False
    assert all(gate["duration_seconds"] >= 0 for gate in report["gates"])
    assert len(list((output_dir / "logs").glob("*.log"))) == 2
    assert (output_dir / "gate_summary.md").is_file()


def test_observer_classifies_missing_executable_as_environment(tmp_path: Path):
    repo, base, head = make_repo(tmp_path)
    impact_map = write_map(
        tmp_path / "map.json",
        [["drpo-command-that-does-not-exist-7f719d"]],
    )
    output_dir = tmp_path / "observer"

    proc = run_observer(repo, base, head, impact_map, output_dir)

    assert proc.returncode == 0, proc.stderr
    report = json.loads((output_dir / "gate_report.json").read_text(encoding="utf-8"))
    gate = report["gates"][0]
    assert gate["status"] == "unavailable"
    assert gate["classification"] == "environment"
    assert gate["would_block_without_observer"] is True


def test_reviewer_annotation_overrides_default_classification(tmp_path: Path):
    repo, base, head = make_repo(tmp_path)
    command = ["{python}", "-c", "raise SystemExit(3)"]
    impact_map = write_map(tmp_path / "map.json", [command])
    label = " ".join(command) + " [full]"
    annotations = tmp_path / "annotations.json"
    annotations.write_text(
        json.dumps(
            {
                label: {
                    "classification": "false_positive",
                    "merge_blocking": False,
                    "follow_up_commit": "abc123",
                    "note": "known observer-only mismatch",
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "observer"

    proc = run_observer(
        repo,
        base,
        head,
        impact_map,
        output_dir,
        "--classification-json",
        str(annotations),
    )

    assert proc.returncode == 0, proc.stderr
    gate = json.loads((output_dir / "gate_report.json").read_text(encoding="utf-8"))["gates"][0]
    assert gate["classification"] == "false_positive"
    assert gate["follow_up_commit"] == "abc123"
    assert gate["note"] == "known observer-only mismatch"


def test_observer_infrastructure_error_is_nonzero_and_persisted(tmp_path: Path):
    repo, base, _ = make_repo(tmp_path)
    impact_map = write_map(tmp_path / "map.json", [["{python}", "-c", "print('unused')"]])
    output_dir = tmp_path / "observer"

    proc = run_observer(repo, base, "missing-head", impact_map, output_dir)

    assert proc.returncode == 2
    report = json.loads((output_dir / "gate_report.json").read_text(encoding="utf-8"))
    assert report["observer_outcome"] == "infrastructure_error"
    assert report["observer_error"]
