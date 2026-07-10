#!/usr/bin/env python3
"""Run the existing DRPO test selector in observe-only mode and persist gate telemetry."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL_DIR = REPO_ROOT / "tools" / "drpo-update"
sys.path.insert(0, str(TOOL_DIR))

from test_selection import (  # noqa: E402
    CommandOutcome,
    TestExecutionError,
    TestSelectionError,
    execute_test_plan,
    select_test_plan,
)

POLICY_ID = "GOV-PR-GATE-OBSERVER-01"
ALLOWED_CLASSIFICATIONS = {
    "actionable",
    "preexisting",
    "environment",
    "flaky",
    "false_positive",
}


@dataclass(frozen=True)
class ReviewerAnnotation:
    classification: str
    merge_blocking: bool
    follow_up_commit: str | None = None
    note: str | None = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _git_paths(repo: Path, base: str, head: str) -> list[str]:
    proc = subprocess.run(
        ["git", "-C", str(repo), "diff", "--name-only", f"{base}..{head}"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise TestSelectionError(proc.stderr.strip() or "git diff --name-only failed")
    return [line for line in proc.stdout.splitlines() if line.strip()]


def _load_annotations(path: Path | None) -> dict[str, ReviewerAnnotation]:
    if path is None:
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TestSelectionError(f"cannot read classification JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise TestSelectionError("classification JSON must be an object keyed by gate label")

    annotations: dict[str, ReviewerAnnotation] = {}
    for label, raw in payload.items():
        if not isinstance(label, str) or not label or not isinstance(raw, dict):
            raise TestSelectionError("classification entries require non-empty string labels and objects")
        classification = raw.get("classification")
        if classification not in ALLOWED_CLASSIFICATIONS:
            raise TestSelectionError(
                f"classification for {label!r} must be one of "
                + ", ".join(sorted(ALLOWED_CLASSIFICATIONS))
            )
        merge_blocking = raw.get("merge_blocking", False)
        if not isinstance(merge_blocking, bool):
            raise TestSelectionError(f"merge_blocking for {label!r} must be boolean")
        follow_up_commit = raw.get("follow_up_commit")
        note = raw.get("note")
        if follow_up_commit is not None and not isinstance(follow_up_commit, str):
            raise TestSelectionError(f"follow_up_commit for {label!r} must be string or null")
        if note is not None and not isinstance(note, str):
            raise TestSelectionError(f"note for {label!r} must be string or null")
        annotations[label] = ReviewerAnnotation(
            classification=classification,
            merge_blocking=merge_blocking,
            follow_up_commit=follow_up_commit,
            note=note,
        )
    return annotations


def _default_classification(outcome: CommandOutcome) -> str | None:
    if outcome.passed:
        return None
    if outcome.returncode == 127 or outcome.error:
        return "environment"
    return "actionable"


def _status(outcome: CommandOutcome) -> str:
    if outcome.passed:
        return "pass"
    if outcome.returncode == 127 or outcome.error:
        return "unavailable"
    return "fail"


def _relative_log(path: str | None, output_dir: Path) -> str | None:
    if path is None:
        return None
    candidate = Path(path)
    try:
        return candidate.resolve().relative_to(output_dir.resolve()).as_posix()
    except ValueError:
        return str(candidate)


def _gate_record(
    outcome: CommandOutcome,
    *,
    duration_seconds: float,
    plan_reason: str,
    changed_paths: Sequence[str],
    output_dir: Path,
    annotation: ReviewerAnnotation | None,
) -> dict[str, Any]:
    default = _default_classification(outcome)
    return {
        "label": outcome.label,
        "command": list(outcome.command),
        "trigger_reason": plan_reason,
        "changed_paths": list(changed_paths),
        "status": _status(outcome),
        "classification": annotation.classification if annotation else default,
        "duration_seconds": round(duration_seconds, 6),
        "returncode": outcome.returncode,
        "log_file": _relative_log(outcome.log_file, output_dir),
        "error": outcome.error,
        "merge_blocking": annotation.merge_blocking if annotation else False,
        "follow_up_commit": annotation.follow_up_commit if annotation else None,
        "note": annotation.note if annotation else None,
        "would_block_without_observer": not outcome.passed,
    }


def _write_summary(path: Path, report: Mapping[str, Any]) -> None:
    plan = report.get("plan", {})
    lines = [
        "# PR Gate Observer Summary",
        "",
        f"- Policy: `{report.get('policy_id')}`",
        f"- Enforcement: `{report.get('enforcement')}`",
        f"- Base: `{report.get('base')}`",
        f"- Head: `{report.get('head')}`",
        f"- Selected mode: `{plan.get('selected_mode', 'unavailable')}`",
        f"- Trigger reason: {plan.get('reason', report.get('observer_error', 'unavailable'))}",
        "",
        "| Gate | Status | Classification | Duration (s) | Merge blocking | Follow-up commit |",
        "|---|---|---|---:|---|---|",
    ]
    for gate in report.get("gates", []):
        lines.append(
            "| {label} | {status} | {classification} | {duration:.6f} | {blocking} | {follow_up} |".format(
                label=str(gate["label"]).replace("|", "\\|"),
                status=gate["status"],
                classification=gate["classification"] or "not_applicable",
                duration=gate["duration_seconds"],
                blocking="yes" if gate["merge_blocking"] else "no",
                follow_up=gate["follow_up_commit"] or "-",
            )
        )
    lines.extend(
        [
            "",
            "> This report is observe-only. Gate failures are recorded but do not change the process exit code.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_report(output_dir: Path, report: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "gate_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_summary(output_dir / "gate_summary.md", report)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=REPO_ROOT)
    parser.add_argument("--base", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--mode", choices=("auto", "fast", "full"), default="auto")
    parser.add_argument("--map", dest="impact_map", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--classification-json", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = args.repo.resolve()
    output_dir = args.output_dir.resolve()
    impact_map = (
        args.impact_map or repo / "tools" / "drpo-update" / "test_impact_map.json"
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = output_dir / "logs"

    started_at = _utc_now()
    start = time.monotonic()
    timed_outcomes: list[tuple[CommandOutcome, float]] = []
    last_callback = start

    def capture(outcome: CommandOutcome) -> None:
        nonlocal last_callback
        now = time.monotonic()
        timed_outcomes.append((outcome, now - last_callback))
        last_callback = now

    report: dict[str, Any] = {
        "schema_version": 1,
        "policy_id": POLICY_ID,
        "enforcement": "observe_only",
        "generated_at": started_at,
        "repository": str(repo),
        "base": args.base,
        "head": args.head,
        "gates": [],
    }

    try:
        annotations = _load_annotations(args.classification_json)
        changed_paths = _git_paths(repo, args.base, args.head)
        plan = select_test_plan(changed_paths, impact_map, requested_mode=args.mode)
        try:
            execute_test_plan(
                plan,
                worktree=repo,
                log_dir=logs_dir,
                outcome_callback=capture,
            )
        except TestExecutionError as exc:
            if not timed_outcomes:
                # Defensive fallback for a selector implementation that raises before callbacks.
                timed_outcomes.extend((outcome, 0.0) for outcome in exc.outcomes)

        report["changed_paths"] = changed_paths
        report["plan"] = plan.to_dict()
        report["gates"] = [
            _gate_record(
                outcome,
                duration_seconds=duration,
                plan_reason=plan.reason,
                changed_paths=changed_paths,
                output_dir=output_dir,
                annotation=annotations.get(outcome.label),
            )
            for outcome, duration in timed_outcomes
        ]
        statuses = [gate["status"] for gate in report["gates"]]
        report["counts"] = {
            "total": len(statuses),
            "pass": statuses.count("pass"),
            "fail": statuses.count("fail"),
            "unavailable": statuses.count("unavailable"),
        }
        report["observer_outcome"] = "completed"
    except TestSelectionError as exc:
        report["observer_outcome"] = "infrastructure_error"
        report["observer_error"] = str(exc)
        report["counts"] = {"total": 0, "pass": 0, "fail": 0, "unavailable": 0}
        report["total_duration_seconds"] = round(time.monotonic() - start, 6)
        _write_report(output_dir, report)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    report["total_duration_seconds"] = round(time.monotonic() - start, 6)
    _write_report(output_dir, report)
    print(
        "PR gate observer: "
        f"mode={report['plan']['selected_mode']}, "
        f"pass={report['counts']['pass']}, "
        f"fail={report['counts']['fail']}, "
        f"unavailable={report['counts']['unavailable']}, "
        f"report={output_dir / 'gate_report.json'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
