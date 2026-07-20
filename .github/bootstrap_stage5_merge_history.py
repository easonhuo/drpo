from pathlib import Path
import hashlib

script = Path("scripts/handoff_authority.py")
text = script.read_text(encoding="utf-8")
old = '''    if len(adds) != 1 or not touches:
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
'''
new = '''    if not touches:
        raise HandoffAuthorityError(
            f"{label} must have non-empty history: {relative}"
        )
    merge_introduced = False
    if len(adds) == 1:
        first_add = adds[0]
    elif not adds:
        first_add = touches[-1]
        parents = _git_text(
            repo_root,
            "rev-list",
            "--parents",
            "-n",
            "1",
            first_add,
        ).split()
        if len(parents) != 3:
            raise HandoffAuthorityError(
                f"{label} lacks a bounded two-parent merge origin: {relative}"
            )
        if _path_exists_at_commit(repo_root, parents[1], relative):
            raise HandoffAuthorityError(
                f"{label} predates its merge-introduced origin: {relative}"
            )
        # The path may come from the merged parent or be created directly
        # in the merge result. Both shapes are bounded by the two-parent
        # origin tree and the later first-parent integration commit.
        merge_introduced = True
    else:
        raise HandoffAuthorityError(
            f"{label} has multiple addition commits: {relative}"
        )
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
if text.count(old) != 1:
    raise SystemExit("Stage 5 history block drift")
text = text.replace(old, new)
old = '''    integrated_text = _git_show(repo_root, integration_commit, Path(relative))
    if path.read_text(encoding="utf-8") != integrated_text:
'''
new = '''    integrated_text = _git_show(repo_root, integration_commit, Path(relative))
    if merge_introduced:
        origin_text = _git_show(repo_root, first_add, Path(relative))
        if origin_text != integrated_text:
            raise HandoffAuthorityError(
                f"{label} merge-origin bytes differ from integration: {relative}"
            )
    if path.read_text(encoding="utf-8") != integrated_text:
'''
if text.count(old) != 1:
    raise SystemExit("Stage 5 integrated-byte block drift")
script.write_text(text.replace(old, new), encoding="utf-8")
script_sha = hashlib.sha256(script.read_bytes()).hexdigest()

ledger = Path("docs/governance_pipeline_stage_status.yaml")
ledger_text = ledger.read_text(encoding="utf-8")
old = '''    - path: scripts/handoff_authority.py
      sha256: c375a4c6c74f604048269b1344ca35a03eb8b6e166a27063153efe64ea5888ff
      authorized_by: GOV-STAGE5-PREINTEGRATION-REPORT-HISTORY-BUGFIX-2026-07-20'''
new = f'''    - path: scripts/handoff_authority.py
      sha256: {script_sha}
      authorized_by: GOV-STAGE5-MERGE-INTRODUCED-PATH-HISTORY-BUGFIX-2026-07-20'''
if ledger_text.count(old) != 1:
    raise SystemExit("Stage 5 protected hash binding drift")
ledger.write_text(ledger_text.replace(old, new), encoding="utf-8")

Path(
    "docs/governance_stage_authorizations/"
    "GOV-STAGE5-MERGE-INTRODUCED-PATH-HISTORY-BUGFIX-2026-07-20.yaml"
).write_text(
    f'''schema_version: 1
authorization_id: GOV-STAGE5-MERGE-INTRODUCED-PATH-HISTORY-BUGFIX-2026-07-20
kind: maintenance
change_class: bugfix
claim_id: GOV-STAGE5-MERGE-INTRODUCED-PATH-HISTORY-BUGFIX-01
approval_record: user_approved_repository_health_repairs_2026_07_20
base_commit: 3e00ca8c2802724d509124541b7ca5a4de1eb90c
stage_ids:
- stage_5
authorized_stage_statuses:
  stage_5: closed_maintenance_only
authorized_file_hashes:
  scripts/handoff_authority.py: {script_sha}
scope:
- accept_a_path_with_no_diff_filter_add_commit_only_when_its_oldest_touch_is_a_two_parent_merge
- require_the_source_merge_first_parent_to_lack_the_path
- allow_the_path_to_come_from_the_merged_parent_or_the_merge_result_tree
- bind_the_origin_tree_to_one_first_parent_integration_commit
- require_origin_integration_and_current_bytes_to_match
- preserve_rejection_of_multiple_additions_and_every_postintegration_touch
excluded_scope:
- change_schema_v3_shapes_registry_authority_or_handoff_semantics
- allow_direct_handoff_edits_octopus_origins_or_unbounded_history_shapes
- weaken_delta_or_materialization_report_immutability
- modify_scientific_code_configuration_results_or_claims
rollback_plan:
- revert_the_merge_origin_support_authorization_and_stage_ledger_hash_together
- preserve_all_existing_handoff_delta_materialization_and_scientific_history
remaining_uncertainties:
- octopus_source_merges_remain_out_of_scope
- the_fix_is_bounded_to_the_current_two_parent_merge_route
''',
    encoding="utf-8",
)

Path(
    "docs/scopes/GOV-STAGE5-MERGE-INTRODUCED-PATH-HISTORY-BUGFIX-01.md"
).write_text(
    '''# Stage 5 merge-introduced path-history bugfix

Base: `main@3e00ca8c2802724d509124541b7ca5a4de1eb90c`

The completed E8 backlog transaction introduced its materialization report in a two-parent source-merge result and then integrated that tree into first-parent `main`. Git records the source merge as the oldest path touch but reports no standalone `--diff-filter=A` commit because the path is absent from both source parents.

This maintenance fix accepts only that bounded history shape: exactly one oldest two-parent merge touch, no path on its first parent, one later first-parent integration, identical origin/integration/current bytes, and no post-integration touch. Multiple additions, non-merge origins, octopus merges, paths already present on the source first parent, byte drift, and post-integration mutation remain rejected. No handoff content, registry state, scientific variable, result, or claim changes.
''',
    encoding="utf-8",
)
print(script_sha)
