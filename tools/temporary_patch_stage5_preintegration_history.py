from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

AUTH_ID = "GOV-STAGE5-PREINTEGRATION-REPORT-HISTORY-BUGFIX-2026-07-20"
BASE_COMMIT = "cfe43571cd8c6d0909c61d36c4f6e4d07c2d2362"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


authority_path = Path("scripts/handoff_authority.py")
authority_text = authority_path.read_text(encoding="utf-8")
insert_anchor = '''def _legacy_inert_materialization_report(
    repo_root: Path,
'''
helper = '''def _validated_integrated_path_history(
    repo_root: Path,
    path: Path,
    *,
    positions: dict[str, int],
    label: str,
    expected_integration_commit: str | None = None,
) -> tuple[str, str]:
    """Validate one immutable path across pre-integration branch revisions.

    A source branch may revise a newly added delta or materialization report before
    the branch is merged.  Every such touch must already be an ancestor of the
    single first-parent integration commit, and the integrated bytes must still
    equal the current bytes.  Any post-integration touch remains fail-closed.
    """

    relative = _repo_relative(repo_root, path, label)
    head = _git_text(repo_root, "rev-parse", "HEAD")
    touches = _git_text(
        repo_root, "log", "--format=%H", "--", relative
    ).splitlines()
    adds = _git_text(
        repo_root,
        "log",
        "--diff-filter=A",
        "--format=%H",
        "--",
        relative,
    ).splitlines()
    if len(adds) != 1 or not touches:
        raise HandoffAuthorityError(
            f"{label} must have one addition and non-empty history: {relative}"
        )
    first_add = adds[0]
    if first_add in positions:
        integration_commit = first_add
    else:
        integration_commit = _first_parent_integration_commit(
            repo_root,
            positions=positions,
            first_add=first_add,
            relative=relative,
        )
    if (
        expected_integration_commit is not None
        and integration_commit != expected_integration_commit
    ):
        raise HandoffAuthorityError(
            f"{label} was not integrated with its authoritative delta: {relative}"
        )
    for touch in touches:
        if (
            _git(
                repo_root,
                "merge-base",
                "--is-ancestor",
                touch,
                integration_commit,
                check=False,
            ).returncode
            != 0
        ):
            raise HandoffAuthorityError(
                f"{label} is not immutable after integration: {relative}"
            )
    integrated_text = _git_show(repo_root, integration_commit, Path(relative))
    if path.read_text(encoding="utf-8") != integrated_text:
        raise HandoffAuthorityError(
            f"{label} bytes differ from the integration commit: {relative}"
        )
    first_parent_touches = _git_text(
        repo_root,
        "rev-list",
        "--first-parent",
        "--reverse",
        f"{integration_commit}..{head}",
        "--",
        relative,
    ).splitlines()
    if first_parent_touches:
        raise HandoffAuthorityError(
            f"{label} changed after first-parent integration: {relative}"
        )
    return first_add, integration_commit


'''
authority_text = replace_once(
    authority_text,
    insert_anchor,
    helper + insert_anchor,
    "insert integrated-path validator",
)
old_discovery = '''        relative = _repo_relative(repo_root, path, "authoritative delta")
        touches = _git_text(repo_root, "log", "--format=%H", "--", relative).splitlines()
        adds = _git_text(
            repo_root, "log", "--diff-filter=A", "--format=%H", "--", relative
        ).splitlines()
        if len(adds) != 1 or len(touches) != 1:
            raise HandoffAuthorityError(f"authoritative delta is not immutable: {relative}")
        first_add = adds[0]
        if first_add in positions:
            integration_commit = first_add
        else:
            integration_commit = _first_parent_integration_commit(
                repo_root,
                positions=positions,
                first_add=first_add,
                relative=relative,
            )
'''
new_discovery = '''        relative = _repo_relative(repo_root, path, "authoritative delta")
        first_add, integration_commit = _validated_integrated_path_history(
            repo_root,
            path,
            positions=positions,
            label="authoritative delta",
        )
'''
authority_text = replace_once(
    authority_text,
    old_discovery,
    new_discovery,
    "replace delta history validation",
)
old_report = '''        report_touches = _git_text(
            repo_root,
            "log",
            "--format=%H",
            "--",
            _repo_relative(repo_root, report, "materialization report"),
        ).splitlines()
        if report_touches != [first_add]:
            raise HandoffAuthorityError(f"materialization report is not immutable: {report}")
'''
new_report = '''        _validated_integrated_path_history(
            repo_root,
            report,
            positions=positions,
            label="materialization report",
            expected_integration_commit=integration_commit,
        )
'''
authority_text = replace_once(
    authority_text,
    old_report,
    new_report,
    "replace report history validation",
)
authority_path.write_text(authority_text, encoding="utf-8")


test_path = Path("tests/stage5_candidate_integration.py")
test_text = test_path.read_text(encoding="utf-8")
test_anchor = '''def run_code_only_noop_and_rollback(tmp_path: Path) -> None:
'''
test_function = '''def run_preintegration_report_revision_and_postintegration_tamper(
    tmp_path: Path,
) -> None:
    central = copy_repository(tmp_path / "preintegration-report-revision")
    _, cutover = activate_delta_mode(central)
    source = make_source_delta(
        central,
        branch="source-preintegration-report",
        base_commit=cutover,
        update_id="STAGE5-TEST-PREINTEGRATION-REPORT",
        block_id="stage5-test-preintegration-report",
        content="Pre-integration materialization-report history probe.",
        target_path=heading_path(central),
    )
    normalized, _, _ = normalize_source(
        central,
        current=cutover,
        source_base=cutover,
        source_commit=source,
        name="preintegration-report",
        tmp_path=tmp_path,
    )

    run(central, "checkout", "-q", "-B", "revised-report", normalized)
    relative = (
        "docs/handoff_deltas/STAGE5-TEST-PREINTEGRATION-REPORT/"
        "MATERIALIZATION_REPORT.json"
    )
    report_path = central / relative
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    report_path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\\n",
        encoding="utf-8",
    )
    run(central, "add", relative)
    run(central, "commit", "-q", "-m", "revise report before integration")
    revised = git_text(central, "rev-parse", "HEAD")

    run(central, "checkout", "-q", "-B", "main", cutover)
    run(
        central,
        "merge",
        "--no-ff",
        "-q",
        revised,
        "-m",
        "integrate report revision",
    )
    accepted = authority.verify_current_state(central)
    assert accepted["status"] == "PASS"
    assert "STAGE5-TEST-PREINTEGRATION-REPORT" in accepted[
        "authoritative_update_ids"
    ]

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    report_path.write_text(
        json.dumps(payload, indent=4, sort_keys=False) + "\\n",
        encoding="utf-8",
    )
    run(central, "add", relative)
    run(central, "commit", "-q", "-m", "tamper report after integration")
    try:
        authority.verify_current_state(central)
    except authority.HandoffAuthorityError as exc:
        assert "not immutable after integration" in str(exc)
    else:  # pragma: no cover - fail closed assertion
        raise AssertionError("post-integration report tamper was accepted")


'''
test_text = replace_once(
    test_text,
    test_anchor,
    test_function + test_anchor,
    "insert Stage 5 history regression",
)
callback_anchor = '''            ("noop", run_code_only_noop_and_rollback),
'''
callback_new = '''            (
                "preintegration-report",
                run_preintegration_report_revision_and_postintegration_tamper,
            ),
            ("noop", run_code_only_noop_and_rollback),
'''
test_text = replace_once(
    test_text,
    callback_anchor,
    callback_new,
    "register Stage 5 history regression",
)
test_path.write_text(test_text, encoding="utf-8")


authorization_path = Path(
    "docs/governance_stage_authorizations/"
    "GOV-STAGE5-PREINTEGRATION-REPORT-HISTORY-BUGFIX-2026-07-20.yaml"
)
authorization = {
    "schema_version": 1,
    "authorization_id": AUTH_ID,
    "kind": "maintenance",
    "change_class": "bugfix",
    "claim_id": "GOV-STAGE5-PREINTEGRATION-REPORT-HISTORY-BUGFIX-01",
    "approval_record": (
        "user_directed_immediate_resolution_2026_07_20_"
        "continue_and_fix_completed_e8_closure_blocker"
    ),
    "base_commit": BASE_COMMIT,
    "stage_ids": ["stage_5"],
    "authorized_stage_statuses": {"stage_5": "closed_maintenance_only"},
    "authorized_file_hashes": {
        authority_path.as_posix(): sha256(authority_path),
        test_path.as_posix(): sha256(test_path),
    },
    "scope": [
        "accept_multiple_preintegration_touches_to_one_new_schema_v3_delta_or_materialization_report",
        "bind_delta_and_report_to_the_same_first_parent_integration_commit",
        "require_every_preintegration_touch_to_be_ancestor_of_that_integration_commit",
        "require_current_bytes_to_equal_the_integrated_tree",
        "reject_every_postintegration_touch_or_byte_drift",
        "add_a_focused_preintegration_revision_and_postintegration_tamper_regression",
        "update_only_the_exact_stage_5_hash_bindings",
    ],
    "excluded_scope": [
        "change_schema_v3_shapes_or_registry_authority",
        "allow_direct_handoff_edits_or_multiple_authoritative_deltas_per_update",
        "weaken_delta_or_materialization_report_immutability",
        "modify_scientific_code_configuration_results_or_claims",
    ],
    "rollback_plan": [
        "revert_the_history_validator_regression_authorization_and_stage_ledger_hashes_as_one_maintenance_change",
        "preserve_all_existing_handoff_delta_materialization_and_scientific_history",
    ],
    "remaining_uncertainties": [
        "octopus_merges_remain_out_of_scope",
        "the_regression_targets_the_current_two_parent_github_merge_route",
    ],
}
authorization_path.parent.mkdir(parents=True, exist_ok=True)
authorization_path.write_text(
    yaml.safe_dump(authorization, sort_keys=False, allow_unicode=True, width=120),
    encoding="utf-8",
)


ledger_path = Path("docs/governance_pipeline_stage_status.yaml")
ledger = yaml.safe_load(ledger_path.read_text(encoding="utf-8"))
protected = ledger["stages"]["stage_5"]["protected_files"]
expected = {
    authority_path.as_posix(): sha256(authority_path),
    test_path.as_posix(): sha256(test_path),
}
seen: set[str] = set()
for record in protected:
    path = record.get("path")
    if path in expected:
        record["sha256"] = expected[path]
        record["authorized_by"] = AUTH_ID
        seen.add(path)
if seen != set(expected):
    raise SystemExit(f"missing protected Stage 5 paths: {sorted(set(expected) - seen)}")
ledger_path.write_text(
    yaml.safe_dump(ledger, sort_keys=False, allow_unicode=True, width=120),
    encoding="utf-8",
)

summary_path = Path("docs/scopes/GOV-STAGE5-PREINTEGRATION-REPORT-HISTORY-BUGFIX-01.md")
summary_path.write_text(
    "# Stage 5 pre-integration report-history bugfix\n\n"
    f"Base: `main@{BASE_COMMIT}`\n\n"
    "The reciprocal closure exposed a fail-closed false rejection: an immutable "
    "materialization report had more than one source-branch revision before its "
    "single GitHub merge integration. The prior validator counted all path touches "
    "as post-acceptance mutations.\n\n"
    "This maintenance change permits multiple revisions only when every touch is "
    "already an ancestor of one first-parent integration commit, the delta and report "
    "map to that same integration, and current bytes equal the integrated tree. Any "
    "post-integration touch or byte drift still fails closed. No scientific file, "
    "schema, registry authority, experiment status, or claim is changed.\n",
    encoding="utf-8",
)
