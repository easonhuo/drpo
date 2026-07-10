#!/usr/bin/env python3
"""Build the exact disposable Stage 1 path-normalization maintenance candidate."""
from __future__ import annotations

import hashlib
import json
import textwrap
from pathlib import Path

BASE_COMMIT = "30b6a9cfd9da2feca1e5be22f5c002d1459298ff"
AUTHORIZATION_ID = "GOV-STAGE1-PATH-NORMALIZATION-BUGFIX-2026-07-10"
CLAIM_ID = "GOV-STAGE1-PATH-NORMALIZATION-BUGFIX-01"
OLD_SELECTOR_DIGEST = "c7b305d73a6649c8012492f66e40edf0f234663fa72146bdb5ae59e495e5ad37"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label} expected exactly one match, found {count}")
    return text.replace(old, new, 1)


def write_selector(root: Path) -> str:
    path = root / "tools/drpo-update/test_selection.py"
    text = path.read_text(encoding="utf-8")
    helper = textwrap.dedent(
        '''\
        def _normalize_changed_path(path: str) -> str:
            """Normalize separators without stripping leading dots from hidden paths."""
            normalized = path.replace(os.sep, "/").replace("\\\\", "/")
            return normalized.removeprefix("./")


        '''
    )
    text = replace_once(
        text,
        "def select_test_plan(\n",
        helper + "def select_test_plan(\n",
        "selector helper insertion",
    )
    text = replace_once(
        text,
        '        tuple(path.replace(os.sep, "/").lstrip("./") for path in changed_paths if path)\n',
        "        tuple(_normalize_changed_path(path) for path in changed_paths if path)\n",
        "selector normalization replacement",
    )
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_ledger(root: Path, selector_digest: str) -> None:
    path = root / "docs/governance_pipeline_stage_status.yaml"
    text = path.read_text(encoding="utf-8")
    old = (
        "    - path: tools/drpo-update/test_selection.py\n"
        f"      sha256: {OLD_SELECTOR_DIGEST}\n"
        "      authorized_by: GOV-PIPELINE-STAGE12-CLOSURE-2026-06-27"
    )
    new = (
        "    - path: tools/drpo-update/test_selection.py\n"
        f"      sha256: {selector_digest}\n"
        f"      authorized_by: {AUTHORIZATION_ID}"
    )
    path.write_text(replace_once(text, old, new, "stage ledger selector binding"), encoding="utf-8")


def write_authorization(root: Path, selector_digest: str) -> None:
    path = root / f"docs/governance_stage_authorizations/{AUTHORIZATION_ID}.yaml"
    path.write_text(
        textwrap.dedent(
            f'''\
            schema_version: 1
            authorization_id: {AUTHORIZATION_ID}
            kind: maintenance
            change_class: bugfix
            claim_id: {CLAIM_ID}
            approval_record: user_approved_2026_07_10_continue_after_explicit_stage1_selector_bugfix_confirmation
            base_commit: {BASE_COMMIT}
            stage_ids:
            - stage_1
            authorized_stage_statuses:
              stage_1: closed_maintenance_only
            authorized_file_hashes:
              tools/drpo-update/test_selection.py: {selector_digest}
            scope:
            - preserve_leading_dots_in_repository_hidden_paths_during_changed_path_normalization
            - remove_only_one_explicit_dot_slash_repository_relative_prefix
            - normalize_windows_and_posix_path_separators_to_forward_slashes
            - restore_dot_github_control_plane_matching_and_high_risk_full_suite_classification
            - add_focused_regression_tests_and_update_only_the_exact_stage_1_hash_binding
            excluded_scope:
            - change_test_impact_groups_commands_risk_levels_or_unknown_path_fail_closed_policy
            - modify_branch_protection_required_checks_or_github_actions_enforcement
            - modify_update_packaging_authority_or_formal_experiment_execution
            - modify_docs_handoff_md_experiments_registry_yaml_or_any_scientific_code_config_result_or_claim
            rollback_plan:
            - revert_the_selector_helper_tests_authorization_and_stage_ledger_binding_as_one_maintenance_change
            - preserve_all_existing_stage_closure_history_and_authorization_records
            - keep_unknown_paths_fail_closed_to_the_full_suite
            remaining_uncertainties:
            - phase_1_and_phase_2a_prs_remain_unmerged_and_require_independent_review
            - branch_protection_required_check_configuration_remains_external_to_this_bugfix
            '''
        ),
        encoding="utf-8",
    )


def write_scope(root: Path) -> None:
    path = root / "docs/scopes/GOV-STAGE1-PATH-NORMALIZATION-BUGFIX-01.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        textwrap.dedent(
            f'''\
            # {CLAIM_ID} Scope Contract

            ## Identity

            - Claim: `{CLAIM_ID}`
            - User authorization: approved on 2026-07-10 after the Phase 2A real PR run exposed the selector defect.
            - Authoritative base: `{BASE_COMMIT}`.
            - Classification: Red maintenance bugfix because one frozen Stage 1 protected file and its ledger binding change.

            ## Root cause

            `str.lstrip("./")` removes any leading dot or slash character rather than one exact `./` prefix. It therefore rewrites `.github/workflows/x.yml` to `github/workflows/x.yml`, preventing the registered `.github/**` control-plane pattern from matching. Unknown-path fail-closed behavior still selected the full suite, but risk classification and audit provenance were wrong.

            ## Allowed files

            - `tools/drpo-update/test_selection.py` — exact normalization repair.
            - `tests/test_update_test_selector_hidden_paths.py` — focused regression tests.
            - `docs/governance_stage_authorizations/{AUTHORIZATION_ID}.yaml` — closed-stage maintenance authorization.
            - `docs/governance_pipeline_stage_status.yaml` — update only the protected selector hash and authorization binding.
            - `docs/scopes/{CLAIM_ID}.md` — this scope contract.

            ## Forbidden changes

            - No impact-map patterns, commands, risk levels, selector modes, or fail-closed policy changes.
            - No GitHub Actions, branch-protection, updater authority, handoff authority, formal execution, scientific source, config, data, seed, threshold, budget, formula, result, or experiment-order changes.

            ## Required behavior

            - Preserve `.github/**`, `.env`, and other leading-dot repository paths.
            - Remove one exact leading `./` prefix only.
            - Normalize Windows and POSIX separators to `/`.
            - Make `.github/workflows/*.yml` match `test_control_plane`, classify as high risk, and select the full suite with no unknown path.
            - Keep unmatched paths fail-closed to the full suite.

            ## Validation

            ```bash
            python -m py_compile tools/drpo-update/test_selection.py tests/test_update_test_selector_hidden_paths.py
            python -m pytest -q tests/test_update_test_selector.py tests/test_update_test_selector_hidden_paths.py tests/test_governance_pipeline_stage_status.py
            python scripts/validate_governance_pipeline_stage_status.py --repo-root .
            python -m compileall -q src scripts tools tests
            bash -n tools/drpo-update/drpo-update tools/drpo-update/install.sh
            python scripts/validate_formal_execution_channel.py --repo-root .
            python scripts/validate_governance_rule_inventory.py --repo-root .
            python -m pytest -q
            ruff check .
            ```

            ## Rollback

            Revert the selector change, focused test, authorization record, scope file, and exact ledger binding together. No historical authorization or scientific material is deleted.
            '''
        ),
        encoding="utf-8",
    )


def write_tests(root: Path) -> None:
    path = root / "tests/test_update_test_selector_hidden_paths.py"
    path.write_text(
        textwrap.dedent(
            '''\
            from __future__ import annotations

            import json
            import sys
            from pathlib import Path

            REPO_ROOT = Path(__file__).resolve().parents[1]
            TOOL_DIR = REPO_ROOT / "tools" / "drpo-update"
            IMPACT_MAP = TOOL_DIR / "test_impact_map.json"
            sys.path.insert(0, str(TOOL_DIR))

            from test_selection import select_test_plan  # noqa: E402


            def _write_docs_map(path: Path) -> Path:
                payload = {
                    "schema_version": 1,
                    "unknown_path_policy": "full",
                    "full_commands": [["{python}", "-c", "print('full')"]],
                    "control_plane_patterns": [],
                    "groups": [
                        {
                            "id": "docs",
                            "risk": "low",
                            "patterns": ["docs/**"],
                            "pytest_targets": [],
                            "validators": [],
                        }
                    ],
                }
                path.write_text(json.dumps(payload) + "\\n", encoding="utf-8")
                return path


            def test_dot_github_path_matches_registered_control_plane_pattern() -> None:
                plan = select_test_plan([".github/workflows/check.yml"], IMPACT_MAP)

                assert plan.changed_paths == (".github/workflows/check.yml",)
                assert plan.selected_mode == "full"
                assert plan.risk == "high"
                assert plan.matched_groups == ("test_control_plane",)
                assert plan.unknown_paths == ()
                assert plan.reason == "high-risk path requires full suite"


            def test_exact_dot_slash_prefix_is_removed_without_stripping_hidden_dot(tmp_path: Path) -> None:
                impact_map = _write_docs_map(tmp_path / "map.json")
                plan = select_test_plan(["./docs/plan.md"], impact_map)

                assert plan.changed_paths == ("docs/plan.md",)
                assert plan.selected_mode == "fast"
                assert plan.unknown_paths == ()


            def test_windows_separator_is_normalized_independently_of_host_os(tmp_path: Path) -> None:
                impact_map = _write_docs_map(tmp_path / "map.json")
                plan = select_test_plan([r"docs\\plan.md"], impact_map)

                assert plan.changed_paths == ("docs/plan.md",)
                assert plan.selected_mode == "fast"
                assert plan.unknown_paths == ()


            def test_unknown_hidden_path_preserves_its_leading_dot(tmp_path: Path) -> None:
                impact_map = _write_docs_map(tmp_path / "map.json")
                plan = select_test_plan([".config/settings.toml"], impact_map)

                assert plan.changed_paths == (".config/settings.toml",)
                assert plan.unknown_paths == (".config/settings.toml",)
                assert plan.selected_mode == "full"
                assert plan.reason == "unknown paths require full suite"
            '''
        ),
        encoding="utf-8",
    )


def main() -> int:
    root = Path.cwd()
    selector_digest = write_selector(root)
    write_ledger(root, selector_digest)
    write_authorization(root, selector_digest)
    write_scope(root)
    write_tests(root)
    metadata = {
        "base_commit": BASE_COMMIT,
        "claim_id": CLAIM_ID,
        "selector_sha256": selector_digest,
    }
    (root / "stage1_path_normalization_candidate.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
