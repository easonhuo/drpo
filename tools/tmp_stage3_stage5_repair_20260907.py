from __future__ import annotations

from pathlib import Path
import textwrap


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "handoff_delta_shadow.py"
TESTS = ROOT / "tests" / "test_handoff_delta_shadow.py"


def replace_authoritative_metadata() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    start = text.index("def authoritative_report_metadata(")
    end = text.index("\ndef classify_observation(", start)
    replacement = textwrap.dedent(
        '''\
        def first_parent_repository_commit_for_path(repo_root: Path, path: Path) -> str:
            """Map an authoritative path to its unique first-parent integration commit."""

            relative = path.resolve().relative_to(repo_root.resolve()).as_posix()
            touches = git_text(repo_root, "log", "--format=%H", "--", relative).splitlines()
            adds = git_text(
                repo_root,
                "log",
                "--diff-filter=A",
                "--format=%H",
                "--",
                relative,
            ).splitlines()
            if not touches:
                raise HandoffDeltaError(
                    "authoritative path must have non-empty history: " + relative
                )
            if len(adds) == 1:
                first_add = adds[0]
            elif not adds:
                first_add = touches[-1]
                parents = git_text(
                    repo_root, "rev-list", "--parents", "-n", "1", first_add
                ).split()
                if len(parents) != 3:
                    raise HandoffDeltaError(
                        "authoritative path lacks a bounded two-parent merge origin: "
                        + relative
                    )
                if git(
                    repo_root,
                    "cat-file",
                    "-e",
                    f"{parents[1]}:{relative}",
                    check=False,
                ).returncode == 0:
                    raise HandoffDeltaError(
                        "authoritative path predates its merge-introduced origin: "
                        + relative
                    )
            else:
                raise HandoffDeltaError(
                    "authoritative path has multiple addition commits: " + relative
                )

            candidates: list[str] = []
            for commit in git_text(
                repo_root, "rev-list", "--first-parent", "--reverse", "HEAD"
            ).splitlines():
                if git(
                    repo_root,
                    "merge-base",
                    "--is-ancestor",
                    first_add,
                    commit,
                    check=False,
                ).returncode != 0:
                    continue
                if git(
                    repo_root,
                    "cat-file",
                    "-e",
                    f"{commit}:{relative}",
                    check=False,
                ).returncode != 0:
                    continue
                parents = git_text(
                    repo_root, "rev-list", "--parents", "-n", "1", commit
                ).split()
                if len(parents) < 2:
                    continue
                if git(
                    repo_root,
                    "cat-file",
                    "-e",
                    f"{parents[1]}:{relative}",
                    check=False,
                ).returncode == 0:
                    continue
                candidates.append(commit)
            if len(candidates) != 1:
                raise HandoffDeltaError(
                    "authoritative path must have exactly one first-parent integration: "
                    + relative
                )
            return candidates[0]


        def verify_authoritative_path_unchanged_after_integration(
            repo_root: Path, path: Path, integration_commit: str
        ) -> None:
            """Reject every path touch that is not already integrated."""

            relative = path.resolve().relative_to(repo_root.resolve()).as_posix()
            touches = git_text(repo_root, "log", "--format=%H", "--", relative).splitlines()
            for touch in touches:
                if git(
                    repo_root,
                    "merge-base",
                    "--is-ancestor",
                    touch,
                    integration_commit,
                    check=False,
                ).returncode != 0:
                    raise HandoffDeltaError(
                        "authoritative delta/report changed after first-parent integration"
                    )
            if git_show_text(repo_root, integration_commit, relative) != path.read_text(
                encoding="utf-8"
            ):
                raise HandoffDeltaError(
                    "authoritative delta/report bytes differ from integration commit"
                )


        def authoritative_report_metadata(repo_root: Path, delta_path: Path) -> dict[str, Any]:
            report_path = delta_path.parent / AUTHORITY_REPORT_FILENAME
            report = load_json(report_path, "materialization report")
            if report.get("report_schema_version") != 1 or report.get("status") != "PASS":
                raise HandoffDeltaError("materialization report schema/status mismatch")
            if report.get("mode") != "authoritative":
                raise HandoffDeltaError("materialization report must be authoritative")
            if report.get("update_id") != delta_path.parent.name:
                raise HandoffDeltaError("materialization report update_id mismatch")
            repository_commit = first_parent_repository_commit_for_path(repo_root, delta_path)
            report_commit = first_parent_repository_commit_for_path(repo_root, report_path)
            if report_commit != repository_commit:
                raise HandoffDeltaError(
                    "authoritative delta/report must share one first-parent integration commit"
                )
            verify_authoritative_path_unchanged_after_integration(
                repo_root, delta_path, repository_commit
            )
            verify_authoritative_path_unchanged_after_integration(
                repo_root, report_path, repository_commit
            )
            if report.get("delta_sha256") != sha256_file(delta_path):
                raise HandoffDeltaError("materialization report delta hash mismatch")
            return {
                "path": report_path.relative_to(repo_root).as_posix(),
                "sha256": sha256_file(report_path),
                "repository_commit": repository_commit,
                "validation_worktree_head": repository_commit,
                "legacy_head_commit_field": None,
                "performance_total_ms": None,
            }
        '''
    )
    SCRIPT.write_text(text[:start] + replacement + text[end:], encoding="utf-8")


def replace_observation_enumeration() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    observation_start = text.index("def observation_records(")
    start = text.index("    records: list[dict[str, Any]] = []", observation_start)
    end = text.index("        else:\n            validate_delta_shape", start)
    replacement = '''    records: list[dict[str, Any]] = []
    authoritative_verified = False
    authority_payload: dict[str, Any] | None = None
    legacy_inert_update_ids: set[str] = set()
    for delta_path in sorted(root.glob(f"*/{DELTA_FILENAME}")):
        delta = load_yaml(delta_path, "handoff delta")
        if delta.get("schema_version") == AUTHORITATIVE_DELTA_SCHEMA_VERSION:
            if authority_mode(repo_root) != "delta":
                raise HandoffDeltaError(
                    "schema-v3 authoritative delta exists while authority mode is not delta"
                )
            report_path = delta_path.parent / AUTHORITY_REPORT_FILENAME
            if (replay or not report_path.is_file()) and authority_payload is None:
                authority_payload = verify_authoritative_state(repo_root)
                authoritative_verified = True
                raw_legacy_ids = authority_payload.get("legacy_inert_update_ids", [])
                if not isinstance(raw_legacy_ids, list) or not all(
                    isinstance(value, str) for value in raw_legacy_ids
                ):
                    raise HandoffDeltaError(
                        "Stage 5 authority verifier returned invalid legacy_inert_update_ids"
                    )
                legacy_inert_update_ids = set(raw_legacy_ids)
            if delta.get("update_id") in legacy_inert_update_ids:
                continue
            if replay and not authoritative_verified:
                authority_payload = verify_authoritative_state(repo_root)
                authoritative_verified = True
            report_meta = authoritative_report_metadata(repo_root, delta_path)
            repository_commit = report_meta["repository_commit"]
'''
    SCRIPT.write_text(text[:start] + replacement + text[end:], encoding="utf-8")


def append_regression_tests() -> None:
    text = TESTS.read_text(encoding="utf-8")
    if "def test_authoritative_metadata_accepts_side_branch_preintegration_history" not in text:
        text += '''


def test_authoritative_metadata_accepts_side_branch_preintegration_history(
    tmp_path: Path,
) -> None:
    repo, _base = make_repo(tmp_path)
    main_branch = git(repo, "branch", "--show-current")
    git(repo, "checkout", "-b", "topic")

    update_id = "TEST-AUTHORITATIVE-MERGE-HISTORY-01"
    delta_dir = repo / "docs/handoff_deltas" / update_id
    delta_dir.mkdir(parents=True)
    delta = delta_dir / MODULE.DELTA_FILENAME
    delta.write_text(
        "schema_version: 3\\nupdate_id: TEST-AUTHORITATIVE-MERGE-HISTORY-01\\n"
    )
    git(repo, "add", delta.relative_to(repo).as_posix())
    git(repo, "commit", "-m", "add authoritative delta on topic")

    report = delta_dir / MODULE.AUTHORITY_REPORT_FILENAME
    report.write_text(
        json.dumps(
            {
                "report_schema_version": 1,
                "status": "PASS",
                "mode": "authoritative",
                "update_id": update_id,
                "delta_sha256": MODULE.sha256_file(delta),
            },
            indent=2,
            sort_keys=True,
        )
        + "\\n"
    )
    git(repo, "add", report.relative_to(repo).as_posix())
    git(repo, "commit", "-m", "add materialization report on topic")

    git(repo, "checkout", main_branch)
    git(repo, "merge", "--no-ff", "topic", "-m", "integrate authoritative update")
    integration = git(repo, "rev-parse", "HEAD")

    metadata = MODULE.authoritative_report_metadata(repo, delta)
    assert metadata["repository_commit"] == integration


def test_authoritative_metadata_rejects_postintegration_mutation(tmp_path: Path) -> None:
    repo, _base = make_repo(tmp_path)
    main_branch = git(repo, "branch", "--show-current")
    git(repo, "checkout", "-b", "topic")

    update_id = "TEST-AUTHORITATIVE-MERGE-HISTORY-02"
    delta_dir = repo / "docs/handoff_deltas" / update_id
    delta_dir.mkdir(parents=True)
    delta = delta_dir / MODULE.DELTA_FILENAME
    delta.write_text(
        "schema_version: 3\\nupdate_id: TEST-AUTHORITATIVE-MERGE-HISTORY-02\\n"
    )
    report = delta_dir / MODULE.AUTHORITY_REPORT_FILENAME
    report.write_text(
        json.dumps(
            {
                "report_schema_version": 1,
                "status": "PASS",
                "mode": "authoritative",
                "update_id": update_id,
                "delta_sha256": MODULE.sha256_file(delta),
            },
            indent=2,
            sort_keys=True,
        )
        + "\\n"
    )
    git(repo, "add", delta_dir.relative_to(repo).as_posix())
    git(repo, "commit", "-m", "add authoritative pair on topic")

    git(repo, "checkout", main_branch)
    git(repo, "merge", "--no-ff", "topic", "-m", "integrate authoritative pair")
    report.write_text(report.read_text() + "\\n")
    git(repo, "add", report.relative_to(repo).as_posix())
    git(repo, "commit", "-m", "tamper after integration")

    with pytest.raises(
        MODULE.HandoffDeltaError, match="changed after first-parent integration"
    ):
        MODULE.authoritative_report_metadata(repo, delta)
'''
    if "def test_observation_records_skips_stage5_legacy_inert_without_report" not in text:
        text += '''


def test_observation_records_skips_stage5_legacy_inert_without_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, _base = make_repo(tmp_path)
    update_id = "TEST-LEGACY-INERT-V3"
    delta_dir = repo / "docs/handoff_deltas" / update_id
    delta_dir.mkdir(parents=True)
    delta = delta_dir / MODULE.DELTA_FILENAME
    delta.write_text(
        "schema_version: 3\\nupdate_id: TEST-LEGACY-INERT-V3\\n",
        encoding="utf-8",
    )
    git(repo, "add", delta.relative_to(repo).as_posix())
    git(repo, "commit", "-m", "add inert v3 delta")

    monkeypatch.setattr(MODULE, "authority_mode", lambda _repo: "delta")
    monkeypatch.setattr(
        MODULE,
        "verify_authoritative_state",
        lambda _repo: {
            "status": "PASS",
            "legacy_inert_update_ids": [update_id],
        },
    )

    assert MODULE.observation_records(repo, replay=False) == []


def test_observation_records_missing_unlisted_v3_report_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, _base = make_repo(tmp_path)
    update_id = "TEST-MISSING-V3-REPORT"
    delta_dir = repo / "docs/handoff_deltas" / update_id
    delta_dir.mkdir(parents=True)
    delta = delta_dir / MODULE.DELTA_FILENAME
    delta.write_text(
        "schema_version: 3\\nupdate_id: TEST-MISSING-V3-REPORT\\n",
        encoding="utf-8",
    )
    git(repo, "add", delta.relative_to(repo).as_posix())
    git(repo, "commit", "-m", "add unlisted v3 delta")

    monkeypatch.setattr(MODULE, "authority_mode", lambda _repo: "delta")
    monkeypatch.setattr(
        MODULE,
        "verify_authoritative_state",
        lambda _repo: {"status": "PASS", "legacy_inert_update_ids": []},
    )

    with pytest.raises(MODULE.HandoffDeltaError, match="materialization report"):
        MODULE.observation_records(repo, replay=False)
'''
    TESTS.write_text(text, encoding="utf-8")


def main() -> None:
    replace_authoritative_metadata()
    replace_observation_enumeration()
    append_regression_tests()


if __name__ == "__main__":
    main()
