#!/usr/bin/env python3
"""Git and GitHub publication transaction for completed RunSpecs."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from runspec_lib import (
    DEFAULT_LANE_FILE,
    DONE_DIR,
    RunSpecError,
    current_commit,
    load_lane_config,
    now_utc,
    read_yaml,
    state_path,
    validate_runspec,
    write_yaml,
)
from runspec_publish_contract import (
    artifact_manifest,
    validate_commit_files,
    validate_publish_block,
    write_delivery_manifest,
)

PUBLISHED_DIR = Path(".runspec_state") / "published"


def run(
    cmd: list[str],
    *,
    cwd: Path,
    check: bool = True,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env=env,
    )
    if check and proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip()
        raise RunSpecError(f"command failed ({proc.returncode}): {' '.join(cmd)}\n{detail}")
    return proc


def git(repo: Path, *args: str, check: bool = True) -> str:
    return run(["git", "-C", str(repo), *args], cwd=repo, check=check).stdout.strip()


def tracked_dirty_paths(repo: Path) -> set[str]:
    paths: set[str] = set()
    for args in [
        ("diff", "--name-only", "--"),
        ("diff", "--cached", "--name-only", "--"),
    ]:
        output = git(repo, *args)
        paths.update(line for line in output.splitlines() if line.strip())
    return paths


def gh_available(repo: Path) -> None:
    if shutil.which("gh") is None:
        raise RunSpecError("GitHub CLI `gh` is required for publish; install it first")
    run(["gh", "auth", "status"], cwd=repo)


def find_or_create_pr(
    repo: Path,
    publish: dict[str, Any],
    body_path: Path,
) -> tuple[str, int | None, str]:
    if not publish.get("create_draft_pr", True):
        return "disabled", None, ""
    gh_available(repo)
    listing = run(
        [
            "gh",
            "pr",
            "list",
            "--state",
            "open",
            "--head",
            publish["dev_branch"],
            "--json",
            "number,url,isDraft",
        ],
        cwd=repo,
    )
    try:
        rows = json.loads(listing.stdout or "[]")
    except json.JSONDecodeError as exc:
        raise RunSpecError("gh pr list returned invalid JSON") from exc
    if rows:
        row = rows[0]
        if row.get("isDraft") is not True:
            raise RunSpecError("existing PR for dev branch is not draft; refusing executor update")
        run(
            ["gh", "pr", "comment", str(row["number"]), "--body-file", str(body_path)],
            cwd=repo,
        )
        return "updated", int(row["number"]), str(row["url"])
    created = run(
        [
            "gh",
            "pr",
            "create",
            "--draft",
            "--base",
            publish["base_branch"],
            "--head",
            publish["dev_branch"],
            "--title",
            str(publish["pr_title"]),
            "--body-file",
            str(body_path),
        ],
        cwd=repo,
    )
    url = created.stdout.strip().splitlines()[-1] if created.stdout.strip() else ""
    return "created", None, url


def publish_completed_run(repo: Path, run_id: str, *, lane: str | None = None) -> dict[str, Any]:
    lane_config = load_lane_config(repo, lane, DEFAULT_LANE_FILE)
    done_path = state_path(repo, DONE_DIR, run_id)
    if not done_path.is_file():
        raise RunSpecError(f"completed RunSpec state is missing: {done_path.relative_to(repo)}")
    spec = validate_runspec(repo, done_path, lane_config=lane_config, require_registry=True)
    publish = validate_publish_block(spec, lane_config["lane"])

    publish_state_dir = repo / ".runspec_state" / "publish" / run_id
    report_path = publish_state_dir / "PUBLISH_REPORT.json"
    if report_path.is_file():
        try:
            prior = json.loads(report_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RunSpecError(f"invalid prior publish report: {exc}") from exc
        if prior.get("status") == "PASS":
            prior["idempotent"] = True
            return prior
        if prior.get("status") in {"COMMITTED_PUSH_PENDING", "PUSHED_PR_PENDING"}:
            expected = str(prior.get("published_commit") or "")
            if not expected or current_commit(repo) != expected:
                raise RunSpecError(
                    "cannot resume publish: current HEAD does not match prior published_commit"
                )
            remote = str(publish.get("remote") or "origin")
            if prior["status"] == "COMMITTED_PUSH_PENDING":
                git(repo, "push", "-u", remote, f"HEAD:refs/heads/{publish['dev_branch']}")
                prior["status"] = "PUSHED_PR_PENDING"
                prior["pushed"] = True
                report_path.write_text(
                    json.dumps(prior, indent=2, sort_keys=True) + "\n", encoding="utf-8"
                )
            body_path = publish_state_dir / "PR_BODY.md"
            if not body_path.is_file():
                raise RunSpecError("cannot resume publish: PR_BODY.md is missing")
            pr_action, pr_number, pr_url = find_or_create_pr(repo, publish, body_path)
            prior["status"] = "PASS"
            prior["published_at"] = now_utc()
            prior["pr_action"] = pr_action
            prior["pr_number"] = pr_number
            prior["pr_url"] = pr_url
            report_path.write_text(
                json.dumps(prior, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            return prior
        raise RunSpecError(f"unsupported prior publish status: {prior.get('status')}")

    current_branch = git(repo, "branch", "--show-current")
    if current_branch != publish["dev_branch"]:
        raise RunSpecError(
            f"current branch={current_branch or '<detached>'} does not match "
            f"publish.dev_branch={publish['dev_branch']}"
        )
    remote = str(publish.get("remote") or "origin")
    git(repo, "remote", "get-url", remote)

    dirty_before = tracked_dirty_paths(repo)
    allowed = set(publish["commit_paths"])
    unexpected = sorted(dirty_before - allowed)
    if unexpected:
        raise RunSpecError(
            "tracked worktree changes outside publish.commit_paths: " + ", ".join(unexpected)
        )
    if git(repo, "diff", "--cached", "--name-only", "--"):
        raise RunSpecError("index is not clean before publish")

    _, artifact = artifact_manifest(repo, run_id)
    commit_files = validate_commit_files(repo, publish["commit_paths"], publish)
    parent = current_commit(repo)
    delivery_path = write_delivery_manifest(
        repo,
        spec=spec,
        publish=publish,
        commit_files=commit_files,
        artifact=artifact,
        parent_commit=parent,
    )
    delivery_rel = delivery_path.relative_to(repo).as_posix()

    stage_paths = list(publish["commit_paths"]) + [delivery_rel]
    validate_commit_files(repo, stage_paths, publish)
    git(repo, "add", "-f", "--", *stage_paths)
    staged = set(git(repo, "diff", "--cached", "--name-only", "--").splitlines())
    unexpected_staged = staged - set(stage_paths)
    if unexpected_staged:
        raise RunSpecError(f"unexpected staged paths: {sorted(unexpected_staged)}")
    if delivery_rel not in staged:
        raise RunSpecError("delivery manifest was not staged")
    git(repo, "commit", "-m", str(publish["commit_message"]))
    published_commit = current_commit(repo)

    publish_state_dir = repo / ".runspec_state" / "publish" / run_id
    publish_state_dir.mkdir(parents=True, exist_ok=True)
    body_path = publish_state_dir / "PR_BODY.md"
    body_path.write_text(
        "\n".join(
            [
                f"## RunSpec result delivery: `{run_id}`",
                "",
                f"- Lane: `{spec['lane']}`",
                f"- Experiment: `{spec['experiment_id']}`",
                f"- Result commit: `{published_commit}`",
                f"- Artifact ZIP SHA-256: `{artifact['zip_sha256']}`",
                f"- Delivery manifest: `{delivery_rel}`",
                "",
                "This is a Draft PR for reviewer-controlled selective integration.",
                "The server executor does not authorize scientific acceptance or automatic merge.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    report_path = publish_state_dir / "PUBLISH_REPORT.json"
    report = {
        "schema_version": 1,
        "status": "COMMITTED_PUSH_PENDING",
        "published_at": None,
        "run_id": run_id,
        "lane": spec["lane"],
        "experiment_id": spec["experiment_id"],
        "dev_branch": publish["dev_branch"],
        "base_branch": publish["base_branch"],
        "parent_commit": parent,
        "published_commit": published_commit,
        "pushed": False,
        "delivery_manifest": delivery_rel,
        "artifact_zip": artifact["zip_path"],
        "artifact_zip_sha256": artifact["zip_sha256"],
        "pr_action": "pending",
        "pr_number": None,
        "pr_url": "",
        "automatic_merge": False,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    git(repo, "push", "-u", remote, f"HEAD:refs/heads/{publish['dev_branch']}")
    report["status"] = "PUSHED_PR_PENDING"
    report["pushed"] = True
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    pr_action, pr_number, pr_url = find_or_create_pr(repo, publish, body_path)

    report["status"] = "PASS"
    report["published_at"] = now_utc()
    report["pr_action"] = pr_action
    report["pr_number"] = pr_number
    report["pr_url"] = pr_url
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    published_state = state_path(repo, PUBLISHED_DIR, run_id)
    published_payload = dict(spec)
    published_payload["status"] = {
        "run_id": run_id,
        "state": "published",
        "published_commit": published_commit,
        "pr_url": pr_url,
        "published_at": report["published_at"],
    }
    write_yaml(published_state, published_payload)
    return report
