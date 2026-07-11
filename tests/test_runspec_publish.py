from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import textwrap
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
RUN_LANE = ROOT / "scripts" / "agent" / "run_lane.py"
PUBLISH = ROOT / "scripts" / "agent" / "publish_runspec_result.py"


def run(
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    check: bool = True,
):
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        check=False,
    )
    if check and proc.returncode != 0:
        raise AssertionError(
            f"command failed: {' '.join(cmd)}\nstdout={proc.stdout}\nstderr={proc.stderr}"
        )
    return proc


def make_fake_gh(tmp_path: Path, *, existing_pr: bool = False) -> tuple[Path, Path]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log_path = tmp_path / "gh.log"
    script = bin_dir / "gh"
    rows = (
        '[{"number": 77, "url": "https://github.example/pr/77", "isDraft": true}]'
        if existing_pr
        else "[]"
    )
    script.write_text(
        textwrap.dedent(
            f"""
            #!/usr/bin/env python3
            import json
            import os
            import sys
            from pathlib import Path

            log = Path(os.environ["FAKE_GH_LOG"])
            with log.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(sys.argv[1:]) + "\\n")
            args = sys.argv[1:]
            if args[:2] == ["auth", "status"]:
                raise SystemExit(0)
            if args[:3] == ["pr", "list", "--state"]:
                print({rows!r})
                raise SystemExit(0)
            if args[:2] == ["pr", "create"]:
                print("https://github.example/pr/88")
                raise SystemExit(0)
            if args[:2] == ["pr", "comment"]:
                raise SystemExit(0)
            print("unsupported fake gh invocation", args, file=sys.stderr)
            raise SystemExit(2)
            """
        ).lstrip(),
        encoding="utf-8",
    )
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return bin_dir, log_path


def make_repo(tmp_path: Path, *, auto_publish: bool = False) -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    origin = tmp_path / "origin.git"
    repo.mkdir()
    run(["git", "init", "-q"], cwd=repo)
    run(["git", "checkout", "-q", "-b", "dev/e8-demo"], cwd=repo)
    run(["git", "config", "user.name", "RunSpec Test"], cwd=repo)
    run(["git", "config", "user.email", "runspec@example.invalid"], cwd=repo)

    (repo / ".gitignore").write_text(
        "outputs/\nrunspec_artifacts/\n.runspec_state/\n",
        encoding="utf-8",
    )
    (repo / ".agent_lane.yaml").write_text(
        "lane: e8\n"
        "executor_mode: strict\n"
        "forbid_cross_lane: true\n"
        "allowed_experiment_prefixes:\n"
        "  - EXT-C-E8-\n",
        encoding="utf-8",
    )
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    (repo / "scripts" / "demo").mkdir(parents=True)
    runner = repo / "scripts" / "demo" / "run_demo.sh"
    runner.write_text(
        textwrap.dedent(
            """
            #!/usr/bin/env bash
            set -euo pipefail
            mkdir -p outputs/e8/demo/metrics outputs/e8/demo/logs
            printf '{"ok": true}\n' > outputs/e8/demo/summary.json
            printf '{"audit": true}\n' > outputs/e8/demo/audit.json
            printf '{"metric": 1}\n' > outputs/e8/demo/metrics/metric.json
            printf 'done\n' > outputs/e8/demo/logs/stdout.txt
            printf 'must-not-publish\n' > outputs/e8/demo/model.pt
            """
        ).lstrip(),
        encoding="utf-8",
    )
    runner.chmod(runner.stat().st_mode | stat.S_IXUSR)
    (repo / "experiments").mkdir()
    (repo / "experiments" / "registry.yaml").write_text(
        "schema_version: 2\nexperiments:\n  - experiment_id: EXT-C-E8-DEMO-01\n",
        encoding="utf-8",
    )
    (repo / "runspecs" / "ready").mkdir(parents=True)
    spec = {
        "version": 1,
        "run_id": "E8_DEMO_PUBLISH_20260711",
        "lane": "e8",
        "priority": 10,
        "created_at": "2026-07-11T00:00:00Z",
        "experiment_id": "EXT-C-E8-DEMO-01",
        "repo_commit": "",
        "entrypoint": {
            "cwd": "repo_root",
            "command": "bash scripts/demo/run_demo.sh",
        },
        "policy": {
            "existing_script_required": True,
            "forbid_new_launcher": True,
            "forbid_hparam_change": True,
            "forbid_cross_lane": True,
        },
        "outputs": {
            "run_dir": "outputs/e8/demo",
            "summary_file": "outputs/e8/demo/summary.json",
            "audit_file": "outputs/e8/demo/audit.json",
        },
        "success_criteria": [
            "outputs/e8/demo/summary.json exists",
            "outputs/e8/demo/audit.json exists",
        ],
        "artifacts": {
            "package_policy": "manifest_only",
            "include": [
                "outputs/e8/demo/summary.json",
                "outputs/e8/demo/audit.json",
                "outputs/e8/demo/metrics/*.json",
                "outputs/e8/demo/logs/*.txt",
            ],
            "exclude": ["**/*.pt", "**/*.safetensors", "**/*.bin"],
            "max_package_size_mb": 10,
            "fail_if_excluded_matched": True,
            "fail_if_package_too_large": True,
        },
        "publish": {
            "enabled": True,
            "auto": auto_publish,
            "dev_branch": "dev/e8-demo",
            "base_branch": "main",
            "remote": "origin",
            "create_draft_pr": True,
            "commit_message": "Publish E8 demo results",
            "pr_title": "[E8] demo result delivery",
            "commit_paths": [
                "outputs/e8/demo/summary.json",
                "outputs/e8/demo/audit.json",
                "outputs/e8/demo/metrics/metric.json",
            ],
            "max_commit_file_size_mb": 10,
            "max_commit_total_size_mb": 25,
        },
    }
    (repo / "runspecs" / "ready" / "E8_DEMO_PUBLISH_20260711.yaml").write_text(
        yaml.safe_dump(spec, sort_keys=False),
        encoding="utf-8",
    )
    run(["git", "add", "."], cwd=repo)
    run(["git", "commit", "-q", "-m", "base"], cwd=repo)
    run(["git", "init", "--bare", "-q", str(origin)], cwd=tmp_path)
    run(["git", "remote", "add", "origin", str(origin)], cwd=repo)
    run(["git", "push", "-q", "-u", "origin", "dev/e8-demo"], cwd=repo)
    return repo, origin


def gh_env(bin_dir: Path, log_path: Path) -> dict[str, str]:
    return {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}",
        "FAKE_GH_LOG": str(log_path),
    }


def test_publish_commits_only_declared_evidence_and_opens_draft_pr(tmp_path: Path):
    repo, origin = make_repo(tmp_path)
    run(
        [
            sys.executable,
            str(RUN_LANE),
            "--repo-root",
            str(repo),
            "--run-id",
            "E8_DEMO_PUBLISH_20260711",
            "--json",
        ],
        cwd=repo,
    )
    bin_dir, log_path = make_fake_gh(tmp_path)
    published = run(
        [
            sys.executable,
            str(PUBLISH),
            "--repo-root",
            str(repo),
            "--run-id",
            "E8_DEMO_PUBLISH_20260711",
            "--json",
        ],
        cwd=repo,
        env=gh_env(bin_dir, log_path),
    )
    payload = json.loads(published.stdout)
    assert payload["status"] == "PASS"
    assert payload["pushed"] is True
    assert payload["pr_action"] == "created"
    assert payload["automatic_merge"] is False

    files = run(
        [
            "git",
            "--git-dir",
            str(origin),
            "ls-tree",
            "-r",
            "--name-only",
            "refs/heads/dev/e8-demo",
        ],
        cwd=tmp_path,
    ).stdout.splitlines()
    assert "outputs/e8/demo/summary.json" in files
    assert "outputs/e8/demo/audit.json" in files
    assert "outputs/e8/demo/metrics/metric.json" in files
    assert "runspec_deliveries/E8_DEMO_PUBLISH_20260711/DELIVERY_MANIFEST.json" in files
    assert "outputs/e8/demo/model.pt" not in files
    assert not any(path.endswith("_results.zip") for path in files)

    delivery = json.loads(
        run(
            [
                "git",
                "--git-dir",
                str(origin),
                "show",
                "refs/heads/dev/e8-demo:runspec_deliveries/"
                "E8_DEMO_PUBLISH_20260711/DELIVERY_MANIFEST.json",
            ],
            cwd=tmp_path,
        ).stdout
    )
    assert delivery["review_contract"]["automatic_merge_allowed"] is False
    assert delivery["artifact_zip"]["committed_to_git"] is False
    assert delivery["artifact_zip"]["persistence"] == "training_server_local"

    calls = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert ["auth", "status"] in calls
    assert any(call[:2] == ["pr", "create"] and "--draft" in call for call in calls)

    again = run(
        [
            sys.executable,
            str(PUBLISH),
            "--repo-root",
            str(repo),
            "--run-id",
            "E8_DEMO_PUBLISH_20260711",
            "--json",
        ],
        cwd=repo,
        env=gh_env(bin_dir, log_path),
    )
    assert json.loads(again.stdout)["idempotent"] is True


def test_publish_rejects_unrelated_tracked_change(tmp_path: Path):
    repo, _ = make_repo(tmp_path)
    run(
        [
            sys.executable,
            str(RUN_LANE),
            "--repo-root",
            str(repo),
            "--run-id",
            "E8_DEMO_PUBLISH_20260711",
            "--json",
        ],
        cwd=repo,
    )
    (repo / "README.md").write_text("unexpected local edit\n", encoding="utf-8")
    bin_dir, log_path = make_fake_gh(tmp_path)
    proc = run(
        [
            sys.executable,
            str(PUBLISH),
            "--repo-root",
            str(repo),
            "--run-id",
            "E8_DEMO_PUBLISH_20260711",
            "--json",
        ],
        cwd=repo,
        env=gh_env(bin_dir, log_path),
        check=False,
    )
    assert proc.returncode != 0
    assert "outside publish.commit_paths" in json.loads(proc.stdout)["error"]


def test_run_lane_auto_publish_reports_partial_without_reclassifying_run(tmp_path: Path):
    repo, _ = make_repo(tmp_path, auto_publish=True)
    # No gh in PATH: execution and packaging pass, publication fails closed.
    env = {**os.environ, "PATH": "/opt/pyvenv/bin:/usr/bin:/bin"}
    proc = run(
        [
            sys.executable,
            str(RUN_LANE),
            "--repo-root",
            str(repo),
            "--run-id",
            "E8_DEMO_PUBLISH_20260711",
            "--json",
        ],
        cwd=repo,
        env=env,
        check=False,
    )
    assert proc.returncode == 2
    payload = json.loads(proc.stdout)
    assert payload["status"] == "PARTIAL"
    assert payload["publish_status"] == "FAIL"
    assert (repo / ".runspec_state" / "done" / "E8_DEMO_PUBLISH_20260711.yaml").is_file()
    assert not (repo / ".runspec_state" / "failed" / "E8_DEMO_PUBLISH_20260711.yaml").exists()
