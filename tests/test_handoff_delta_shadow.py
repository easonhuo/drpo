from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
import subprocess
import sys
import time
from datetime import timedelta
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "handoff_delta_shadow.py"
SPEC = importlib.util.spec_from_file_location("handoff_delta_shadow_tested", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
sys.modules.setdefault("handoff_delta_shadow", MODULE)

RUNNER_PATH = ROOT / "scripts" / "run_handoff_delta_acceptance.py"
RUNNER_SPEC = importlib.util.spec_from_file_location(
    "run_handoff_delta_acceptance_tested", RUNNER_PATH
)
assert RUNNER_SPEC is not None and RUNNER_SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(RUNNER_SPEC)
RUNNER_SPEC.loader.exec_module(RUNNER)


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()


def make_repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init")
    git(repo, "config", "user.name", "Test User")
    git(repo, "config", "user.email", "test@example.com")
    (repo / "docs/handoff_deltas").mkdir(parents=True)
    (repo / "experiments").mkdir()
    (repo / "scripts").mkdir()
    (repo / "docs/handoff.md").write_text(
        "# Master v1\n\n## A\n\nA body EXP-A-01 GOV-CLAIM-01.\n\n## B\n\nB body.\n"
    )
    (repo / "experiments/registry.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": 2,
                "experiments": [
                    {
                        "id": "EXP-A-01",
                        "status": "not_run",
                        "execution_gate": {"state": "blocked"},
                    }
                ],
            },
            sort_keys=False,
        )
    )
    # Copy the real policy/state-machine contracts so the temporary repository
    # exercises the production parser rather than a test-only substitute.
    for name in ("handoff_delta_policy.yaml", "handoff_delta_state_machines.yaml"):
        (repo / "docs" / name).write_text((ROOT / "docs" / name).read_text())
    git(repo, "add", ".")
    git(repo, "commit", "-m", "base")
    return repo, git(repo, "rev-parse", "HEAD")


def write_delta(
    repo: Path,
    base: str,
    update_id: str,
    operations: list[dict[str, object]],
    *,
    registry: dict[str, object] | None = None,
    manual_text: str | None = None,
) -> Path:
    base_handoff = git(repo, "show", f"{base}:docs/handoff.md") + "\n"
    base_registry = git(repo, "show", f"{base}:experiments/registry.yaml") + "\n"
    rendered = MODULE.render(base_handoff, operations).text
    if manual_text is None:
        manual_text = rendered
    (repo / "docs/handoff.md").write_text(manual_text)
    delta_dir = repo / "docs/handoff_deltas" / update_id
    delta_dir.mkdir(parents=True)
    payload = {
        "schema_version": 2,
        "update_id": update_id,
        "mode": "shadow",
        "base": {
            "commit": base,
            "handoff_sha256": digest(base_handoff),
            "registry_sha256": digest(base_registry),
        },
        "renderer_version": 1,
        "operations": operations,
        "registry": registry
        or {"mode": "unchanged", "expected_after_sha256": None, "changes": []},
        "expected": {
            "candidate_sha256": digest(rendered),
            "manual_sha256": digest(manual_text),
        },
    }
    path = delta_dir / "HANDOFF_DELTA.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True))
    return path


def write_report(repo: Path, delta: Path) -> dict[str, object]:
    result = MODULE.check_delta(repo, delta)
    path = delta.parent / MODULE.REPORT_FILENAME
    path.write_text(json.dumps(result.report, indent=2, sort_keys=True) + "\n")
    return result.report


def commit_all(repo: Path, message: str) -> str:
    git(repo, "add", ".")
    git(repo, "commit", "-m", message)
    return git(repo, "rev-parse", "HEAD")


def write_full_report(repo: Path, path: Path, covered_ids: list[str]) -> None:
    observations = MODULE.observation_records(repo, replay=False)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "report_schema_version": 2,
                "policy_id": "GOV-HANDOFF-INDEX-01",
                "tier": "full",
                "status": "PASS",
                "validation_worktree_head": git(repo, "rev-parse", "HEAD"),
                "reasons": ["test"],
                "elapsed_seconds": 0.1,
                "target_seconds": 900.0,
                "outcomes": [
                    {
                        "command": ["python3", "-m", "pytest"],
                        "returncode": 0,
                        "timed_out": False,
                        "elapsed_seconds": 0.1,
                        "stdout": "passed",
                        "stderr": "",
                    }
                ],
                "coverage": {
                    "bootstrap_observation_count": sum(
                        row["kind"] == "bootstrap" for row in observations
                    ),
                    "successful_real_observation_count": len(covered_ids),
                    "covered_update_ids": sorted(covered_ids),
                    "observation_fingerprint": MODULE.observation_fingerprint(covered_ids),
                },
                "corpus_audit": {
                    "observation_count": len(observations),
                    "all_stored_reports_revalidated": True,
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def test_historical_bootstrap_delta_matches_golden_candidate() -> None:
    delta_dir = ROOT / "docs/handoff_deltas/GOV-STAGE3-SHADOW-BOOTSTRAP-2026-06-27"
    payload = yaml.safe_load((delta_dir / "HANDOFF_DELTA.yaml").read_text())
    base_text = git(ROOT, "show", f"{payload['base']['commit']}:docs/handoff.md") + "\n"
    rendered = MODULE.render(base_text, payload["operations"]).text
    assert digest(rendered) == payload["expected"]["candidate_sha256"]
    golden = gzip.decompress((delta_dir / "candidate.golden.md.gz").read_bytes()).decode()
    assert rendered == golden
    assert MODULE.render(rendered, payload["operations"]).text == rendered


def test_current_repository_semantic_gap_delta_matches_manual_handoff() -> None:
    delta = ROOT / "docs/handoff_deltas/DU1-E6-SEMANTIC-GAP-FORMAL-2026-06-27/HANDOFF_DELTA.yaml"
    result = MODULE.check_delta(
        ROOT, delta, target_commit="0907c3c0e76fc836c2bf2b752abf554c17f79f22"
    )
    assert result.report["status"] == "PASS"
    assert result.report["exact_manual_candidate_match"] is True
    assert result.report["idempotence_passed"] is True
    assert result.report["candidate_replaced_authority"] is False


def test_historical_countdown_v43_delta_matches_repository_after_image() -> None:
    delta = ROOT / "docs/handoff_deltas/EXT-C-E8-V4.3-DYNAMIC-CONTROL-2026-06-27/HANDOFF_DELTA.yaml"
    result = MODULE.check_delta(
        ROOT, delta, target_commit="f203d86032366eb134207f3fd7ab26a31804c8bc"
    )
    assert result.report["status"] == "PASS"
    assert result.report["exact_manual_candidate_match"] is True
    assert result.report["idempotence_passed"] is True
    assert result.report["candidate_replaced_authority"] is False


def test_historical_countdown_v43_stored_report_revalidates_after_later_commits() -> None:
    delta = (
        ROOT
        / "docs/handoff_deltas/EXT-C-E8-V4.3-DYNAMIC-CONTROL-2026-06-27/HANDOFF_DELTA.yaml"
    )
    fresh = MODULE.check_delta(
        ROOT, delta, target_commit="f203d86032366eb134207f3fd7ab26a31804c8bc"
    )
    metadata = MODULE.validate_stored_report(ROOT, delta, fresh.report)
    assert metadata["report_schema_version"] == 2


def test_standard_report_projection_ignores_evidence_digest_but_not_evidence_path() -> None:
    report_path = (
        ROOT
        / "docs/handoff_deltas/EXT-C-E8-V4.3-DYNAMIC-CONTROL-2026-06-27/SHADOW_REPORT.json"
    )
    stored = json.loads(report_path.read_text())
    changed_digest = json.loads(json.dumps(stored))
    changed_digest["registry_change_assertions"][0]["evidence"][0]["sha256"] = "0" * 64
    changed_digest["registry_transition_assertions"][0]["evidence"][0]["sha256"] = "0" * 64
    assert MODULE.report_projection(stored) == MODULE.report_projection(changed_digest)

    changed_path = json.loads(json.dumps(stored))
    changed_path["registry_change_assertions"][0]["evidence"][0]["path"] = "other.md"
    assert MODULE.report_projection(stored) != MODULE.report_projection(changed_path)


def test_historical_stage3_automation_delta_matches_repository_after_image() -> None:
    delta = ROOT / "docs/handoff_deltas/GOV-STAGE3-OBSERVATION-AUTOMATION-2026-06-27/HANDOFF_DELTA.yaml"
    result = MODULE.check_delta(
        ROOT, delta, target_commit="bfa01c28b8dadfd4a191d92e0a239cf9cf69d45d"
    )
    assert result.report["status"] == "PASS"
    assert result.report["exact_manual_candidate_match"] is True
    assert result.report["idempotence_passed"] is True
    assert result.report["candidate_replaced_authority"] is False


def test_historical_e7_canonical_critic_rollout_delta_matches_repository_after_image() -> None:
    delta = (
        ROOT
        / "docs/handoff_deltas/EXT-H-E7-Q2-CANONICAL-CRITIC-ROLLOUT-2026-06-28/HANDOFF_DELTA.yaml"
    )
    result = MODULE.check_delta(
        ROOT, delta, target_commit="fa225510e3e3e4616f36d8f586611aa6af79bf6e"
    )
    assert result.report["status"] == "PASS"
    assert result.report["exact_manual_candidate_match"] is True
    assert result.report["idempotence_passed"] is True
    assert result.report["candidate_replaced_authority"] is False


def test_historical_semantic_gap_closure_delta_matches_repository_after_image() -> None:
    delta = (
        ROOT
        / "docs/handoff_deltas/DU1-E6-SEMANTIC-GAP-CLOSURE-2026-06-28/HANDOFF_DELTA.yaml"
    )
    result = MODULE.check_delta(
        ROOT, delta, target_commit="e70f0d84256cdeb6ebbf80b0495a043582787bf6"
    )
    assert result.report["status"] == "PASS"
    assert result.report["exact_manual_candidate_match"] is True
    assert result.report["idempotence_passed"] is True
    assert result.report["candidate_replaced_authority"] is False
    assert result.report["registry_change_coverage"]["fully_declared"] is True


def test_historical_e6_parent_closure_delta_matches_repository_after_image() -> None:
    delta = (
        ROOT
        / "docs/handoff_deltas/DU1-E6-PARENT-CLOSURE-2026-06-28/HANDOFF_DELTA.yaml"
    )
    result = MODULE.check_delta(
        ROOT, delta, target_commit="5afe6cdee7d3a530cd73a3043a06c28a2c175a32"
    )
    assert result.report["status"] == "PASS"
    assert result.report["exact_manual_candidate_match"] is True
    assert result.report["idempotence_passed"] is True
    assert result.report["candidate_replaced_authority"] is False
    assert result.report["registry_change_coverage"]["fully_declared"] is True


def test_historical_countdown_v44_offline_bank_delta_matches_repository_after_image() -> None:
    delta = (
        ROOT
        / "docs/handoff_deltas/EXT-C-E8-V4.4-OFFLINE-BANK-2026-06-28/HANDOFF_DELTA.yaml"
    )
    result = MODULE.check_delta(
        ROOT, delta, target_commit="194bd6fa77c029420244ae8ba143fdcf7abacf40"
    )
    assert result.report["status"] == "PASS"
    assert result.report["exact_manual_candidate_match"] is True
    assert result.report["idempotence_passed"] is True
    assert result.report["candidate_replaced_authority"] is False
    assert result.report["registry_change_coverage"]["fully_declared"] is True


def test_historical_e7_gymnasium_v4_rollout_delta_matches_repository_after_image() -> None:
    delta = (
        ROOT
        / "docs/handoff_deltas/EXT-H-E7-Q2-GYMNASIUM-V4-ROLLOUT-2026-06-28/HANDOFF_DELTA.yaml"
    )
    result = MODULE.check_delta(
        ROOT, delta, target_commit="58342ae7809354ef8af0e90a1d9938aa51f3a97d"
    )
    assert result.report["status"] == "PASS"
    assert result.report["exact_manual_candidate_match"] is True
    assert result.report["idempotence_passed"] is True
    assert result.report["candidate_replaced_authority"] is False
    assert result.report["registry_change_coverage"]["fully_declared"] is True


def test_historical_countdown_v45_tuning_delta_matches_repository_after_image() -> None:
    delta = (
        ROOT
        / "docs/handoff_deltas/EXT-C-E8-V4.5-OFFLINE-BANK-TUNING-2026-06-28/HANDOFF_DELTA.yaml"
    )
    result = MODULE.check_delta(
        ROOT, delta, target_commit="22161a91c0863278765b0d604ea82401d481b5aa"
    )
    assert result.report["status"] == "PASS"
    assert result.report["exact_manual_candidate_match"] is True
    assert result.report["idempotence_passed"] is True
    assert result.report["candidate_replaced_authority"] is False
    assert result.report["registry_change_coverage"]["fully_declared"] is True


def test_historical_cu1_e4_taper_utility_fairness_delta_matches_repository_after_image() -> None:
    delta = (
        ROOT
        / "docs/handoff_deltas/CU1-E4-TAPER-UTILITY-FAIRNESS-REGISTRATION-2026-06-28/HANDOFF_DELTA.yaml"
    )
    result = MODULE.check_delta(
        ROOT, delta, target_commit="ce5964a0c16b12626ceb81fa9813fff14893c612"
    )
    assert result.report["status"] == "PASS"
    assert result.report["exact_manual_candidate_match"] is True
    assert result.report["idempotence_passed"] is True
    assert result.report["candidate_replaced_authority"] is False
    assert result.report["registry_change_coverage"]["fully_declared"] is True


def test_historical_countdown_v46_online_replay_delta_matches_repository_after_image() -> None:
    delta = (
        ROOT
        / "docs/handoff_deltas/EXT-C-E8-V4.6-ONLINE-OFFPOLICY-REPLAY-2026-06-29/HANDOFF_DELTA.yaml"
    )
    result = MODULE.check_delta(
        ROOT, delta, target_commit="b503b1245d9de150e034d85ad96405c0be5c2b01"
    )
    assert result.report["status"] == "PASS"
    assert result.report["exact_manual_candidate_match"] is True
    assert result.report["idempotence_passed"] is True
    assert result.report["candidate_replaced_authority"] is False
    assert result.report["registry_change_coverage"]["fully_declared"] is True


def test_historical_cu1_e4_taper_near_retention_delta_matches_repository_after_image() -> None:
    delta = (
        ROOT
        / "docs/handoff_deltas/CU1-E4-TAPER-NEAR-RETENTION-2026-06-28/HANDOFF_DELTA.yaml"
    )
    result = MODULE.check_delta(
        ROOT, delta, target_commit="bb06d5ec107c7630adf7100bb9bfb5b95ff7303f"
    )
    assert result.report["status"] == "PASS"
    assert result.report["exact_manual_candidate_match"] is True
    assert result.report["idempotence_passed"] is True
    assert result.report["candidate_replaced_authority"] is False
    assert result.report["registry_change_coverage"]["fully_declared"] is True


def test_historical_cu1_e4_taper_closure_delta_matches_repository_after_image() -> None:
    delta = (
        ROOT
        / "docs/handoff_deltas/C-U1-E4-TAPER-CLOSURE-V63-2026-06-29/HANDOFF_DELTA.yaml"
    )
    result = MODULE.check_delta(
        ROOT, delta, target_commit="bf1437699c2a67c8e6aff725b8658a909353d2d2"
    )
    assert result.report["status"] == "PASS"
    assert result.report["exact_manual_candidate_match"] is True
    assert result.report["idempotence_passed"] is True
    assert result.report["candidate_replaced_authority"] is False
    assert result.report["registry_change_coverage"]["fully_declared"] is True


def test_current_e7_q2_acceptance_delta_matches_manual_handoff() -> None:
    delta = (
        ROOT
        / "docs/handoff_deltas/EXT-H-E7-Q2-ACCEPTANCE-PIPELINE-2026-06-29/HANDOFF_DELTA.yaml"
    )
    result = MODULE.check_delta(ROOT, delta)
    assert result.report["status"] == "PASS"
    assert result.report["exact_manual_candidate_match"] is True
    assert result.report["idempotence_passed"] is True
    assert result.report["candidate_replaced_authority"] is False
    assert result.report["registry_change_coverage"]["fully_declared"] is True


def test_standard_replay_is_idempotent(tmp_path: Path) -> None:
    repo, base = make_repo(tmp_path)
    operations = [
        {
            "operation_id": "rename-master",
            "op": "replace_heading",
            "heading_path": ["Master v1"],
            "new_heading": "Master v2",
            "reason": "version update",
        },
        {
            "operation_id": "insert-a-result",
            "op": "insert_after_heading",
            "heading_path": ["Master v2", "A"],
            "block_id": "a-result",
            "content": "New result EXP-A-02.",
        },
    ]
    delta = write_delta(repo, base, "TEST-IDEMPOTENCE", operations)
    result = MODULE.check_delta(repo, delta)
    assert MODULE.render(result.candidate, operations).text == result.candidate


def test_standard_independent_section_operations_commute(tmp_path: Path) -> None:
    repo, base = make_repo(tmp_path)
    base_text = git(repo, "show", f"{base}:docs/handoff.md") + "\n"
    op_a = {
        "operation_id": "append-a",
        "op": "append_to_section",
        "heading_path": ["Master v1", "A"],
        "block_id": "block-a",
        "content": "A extension.",
    }
    op_b = {
        "operation_id": "append-b",
        "op": "append_to_section",
        "heading_path": ["Master v1", "B"],
        "block_id": "block-b",
        "content": "B extension.",
    }
    ab = MODULE.render(MODULE.render(base_text, [op_a]).text, [op_b]).text
    ba = MODULE.render(MODULE.render(base_text, [op_b]).text, [op_a]).text
    assert ab == ba


def test_standard_same_block_id_with_different_content_conflicts(tmp_path: Path) -> None:
    repo, base = make_repo(tmp_path)
    base_text = git(repo, "show", f"{base}:docs/handoff.md") + "\n"
    first = {
        "operation_id": "append-a",
        "op": "append_to_section",
        "heading_path": ["Master v1", "A"],
        "block_id": "shared-block",
        "content": "first",
    }
    second = {**first, "operation_id": "append-b", "content": "second"}
    once = MODULE.render(base_text, [first]).text
    with pytest.raises(MODULE.HandoffDeltaError, match="different content"):
        MODULE.render(once, [second])


def test_standard_manual_undeclared_change_is_rejected(tmp_path: Path) -> None:
    repo, base = make_repo(tmp_path)
    operations = [
        {
            "operation_id": "append-a",
            "op": "append_to_section",
            "heading_path": ["Master v1", "A"],
            "block_id": "block-a",
            "content": "A extension.",
        }
    ]
    base_text = git(repo, "show", f"{base}:docs/handoff.md") + "\n"
    rendered = MODULE.render(base_text, operations).text
    delta = write_delta(
        repo,
        base,
        "TEST-UNDECLARED",
        operations,
        manual_text=rendered.replace("B body.", "B body changed outside the delta."),
    )
    with pytest.raises(MODULE.HandoffDeltaError, match="candidate does not exactly match"):
        MODULE.check_delta(repo, delta)


def test_standard_base_hash_mismatch_is_rejected(tmp_path: Path) -> None:
    repo, base = make_repo(tmp_path)
    operations = [
        {
            "operation_id": "append-a",
            "op": "append_to_section",
            "heading_path": ["Master v1", "A"],
            "block_id": "block-a",
            "content": "A extension.",
        }
    ]
    delta = write_delta(repo, base, "TEST-BASE-HASH", operations)
    payload = yaml.safe_load(delta.read_text())
    payload["base"]["handoff_sha256"] = "0" * 64
    delta.write_text(yaml.safe_dump(payload, sort_keys=False))
    with pytest.raises(MODULE.HandoffDeltaError, match="Base handoff SHA-256"):
        MODULE.check_delta(repo, delta)


def test_standard_allowed_registry_transition_is_verified(tmp_path: Path) -> None:
    repo, base = make_repo(tmp_path)
    current = yaml.safe_load((repo / "experiments/registry.yaml").read_text())
    current["experiments"][0]["status"] = "pilot"
    (repo / "experiments/registry.yaml").write_text(yaml.safe_dump(current, sort_keys=False))
    evidence = repo / "evidence.json"
    evidence.write_text(json.dumps({"status": "pilot"}))
    operations = [
        {
            "operation_id": "append-a",
            "op": "append_to_section",
            "heading_path": ["Master v1", "A"],
            "block_id": "block-a",
            "content": "Pilot result.",
        }
    ]
    registry = {
        "mode": "expected_after",
        "expected_after_sha256": hashlib.sha256(
            (repo / "experiments/registry.yaml").read_bytes()
        ).hexdigest(),
        "changes": [
            {
                "change_id": "pilot-status",
                "kind": "transition",
                "entity_id": "EXP-A-01",
                "field_path": ["status"],
                "machine": "scientific_status",
                "from": "not_run",
                "to": "pilot",
                "evidence": ["evidence.json"],
            }
        ],
    }
    delta = write_delta(repo, base, "TEST-REGISTRY-ALLOWED", operations, registry=registry)
    result = MODULE.check_delta(repo, delta)
    assert result.report["registry_transition_assertions"][0]["to"] == "pilot"


def test_standard_illegal_registry_transition_is_rejected(tmp_path: Path) -> None:
    repo, base = make_repo(tmp_path)
    current = yaml.safe_load((repo / "experiments/registry.yaml").read_text())
    current["experiments"][0]["status"] = "analytically_proven"
    (repo / "experiments/registry.yaml").write_text(yaml.safe_dump(current, sort_keys=False))
    (repo / "evidence.json").write_text("{}")
    operations = [
        {
            "operation_id": "append-a",
            "op": "append_to_section",
            "heading_path": ["Master v1", "A"],
            "block_id": "block-a",
            "content": "Claim.",
        }
    ]
    registry = {
        "mode": "expected_after",
        "expected_after_sha256": hashlib.sha256(
            (repo / "experiments/registry.yaml").read_bytes()
        ).hexdigest(),
        "changes": [
            {
                "change_id": "invalid-proof",
                "kind": "transition",
                "entity_id": "EXP-A-01",
                "field_path": ["status"],
                "machine": "scientific_status",
                "from": "not_run",
                "to": "analytically_proven",
                "evidence": ["evidence.json"],
            }
        ],
    }
    delta = write_delta(repo, base, "TEST-REGISTRY-ILLEGAL", operations, registry=registry)
    with pytest.raises(MODULE.HandoffDeltaError, match="Illegal scientific_status transition"):
        MODULE.check_delta(repo, delta)


def test_full_acceptance_mutations_are_rejected(tmp_path: Path) -> None:
    repo, base = make_repo(tmp_path)
    base_text = git(repo, "show", f"{base}:docs/handoff.md") + "\n"
    malicious = base_text.replace("EXP-A-01", "")
    with pytest.raises(MODULE.HandoffDeltaError, match="Historical identifiers were removed"):
        MODULE.verify_history_preservation(base_text, malicious, [])


def test_historical_full_acceptance_fast_gate_stays_below_hard_limit() -> None:
    delta = (
        ROOT
        / "docs/handoff_deltas/DU1-E6-SEMANTIC-GAP-CLOSURE-2026-06-28/HANDOFF_DELTA.yaml"
    )
    samples = []
    for _ in range(3):
        started = time.perf_counter()
        MODULE.check_delta(
            ROOT,
            delta,
            target_commit="e70f0d84256cdeb6ebbf80b0495a043582787bf6",
        )
        samples.append(time.perf_counter() - started)
    assert max(samples) < 15.0


def test_standard_after_heading_and_section_end_blocks_keep_distinct_clusters(tmp_path: Path) -> None:
    repo, base = make_repo(tmp_path)
    base_text = git(repo, "show", f"{base}:docs/handoff.md") + "\n"
    after = {
        "operation_id": "after-a",
        "op": "insert_after_heading",
        "heading_path": ["Master v1", "A"],
        "block_id": "after-block",
        "content": "Immediately after A.",
    }
    end = {
        "operation_id": "end-a",
        "op": "append_to_section",
        "heading_path": ["Master v1", "A"],
        "block_id": "end-block",
        "content": "At the end of A.",
    }
    rendered = MODULE.render(base_text, [after, end]).text
    assert rendered.index("Immediately after A.") < rendered.index("A body EXP-A-01")
    assert rendered.index("A body EXP-A-01") < rendered.index("At the end of A.")
    assert MODULE.render(rendered, [after, end]).text == rendered


def test_standard_new_schema_requires_complete_registry_change_coverage(tmp_path: Path) -> None:
    repo, base = make_repo(tmp_path)
    current = yaml.safe_load((repo / "experiments/registry.yaml").read_text())
    current["experiments"].append(
        {"id": "EXP-B-01", "status": "not_run", "execution_gate": {"state": "blocked"}}
    )
    (repo / "experiments/registry.yaml").write_text(yaml.safe_dump(current, sort_keys=False))
    (repo / "evidence.json").write_text("{}")
    operations = [
        {
            "operation_id": "append-a",
            "op": "append_to_section",
            "heading_path": ["Master v1", "A"],
            "block_id": "block-a",
            "content": "Register EXP-B-01.",
        }
    ]
    registry = {
        "mode": "expected_after",
        "expected_after_sha256": hashlib.sha256(
            (repo / "experiments/registry.yaml").read_bytes()
        ).hexdigest(),
        "changes": [],
    }
    delta = write_delta(repo, base, "TEST-MISSING-REGISTRY-COVERAGE", operations, registry=registry)
    payload = yaml.safe_load(delta.read_text())
    payload["registry"]["changes"] = [
        {
            "change_id": "unrelated-field",
            "kind": "update_field",
            "entity_id": "EXP-A-01",
            "field_path": ["status"],
            "from": "not_run",
            "to": "pilot",
            "reason": "intentionally wrong",
            "evidence": ["evidence.json"],
        }
    ]
    delta.write_text(yaml.safe_dump(payload, sort_keys=False))
    with pytest.raises(MODULE.HandoffDeltaError, match="does not match an actual changed field"):
        MODULE.check_delta(repo, delta)


def test_standard_schema2_add_entity_is_fully_declared(tmp_path: Path) -> None:
    repo, base = make_repo(tmp_path)
    current = yaml.safe_load((repo / "experiments/registry.yaml").read_text())
    current["experiments"].append(
        {"id": "EXP-B-01", "status": "not_run", "execution_gate": {"state": "blocked"}}
    )
    (repo / "experiments/registry.yaml").write_text(yaml.safe_dump(current, sort_keys=False))
    (repo / "evidence.json").write_text("{}")
    operations = [
        {
            "operation_id": "append-a",
            "op": "append_to_section",
            "heading_path": ["Master v1", "A"],
            "block_id": "block-a",
            "content": "Register EXP-B-01.",
        }
    ]
    registry = {
        "mode": "expected_after",
        "expected_after_sha256": hashlib.sha256(
            (repo / "experiments/registry.yaml").read_bytes()
        ).hexdigest(),
        "changes": [
            {
                "change_id": "add-exp-b",
                "kind": "add_entity",
                "entity_id": "EXP-B-01",
                "evidence": ["evidence.json"],
            }
        ],
    }
    delta = write_delta(repo, base, "TEST-ADD-ENTITY", operations, registry=registry)
    result = MODULE.check_delta(repo, delta)
    assert result.report["registry_change_coverage"]["fully_declared"] is True
    assert result.report["registry_change_coverage"]["added_entities"] == ["EXP-B-01"]


def test_standard_schema2_registry_entity_removal_is_rejected(tmp_path: Path) -> None:
    repo, base = make_repo(tmp_path)
    current = yaml.safe_load((repo / "experiments/registry.yaml").read_text())
    current["experiments"] = []
    (repo / "experiments/registry.yaml").write_text(yaml.safe_dump(current, sort_keys=False))
    (repo / "evidence.json").write_text("{}")
    operations = [
        {
            "operation_id": "append-a",
            "op": "append_to_section",
            "heading_path": ["Master v1", "A"],
            "block_id": "block-a",
            "content": "Attempt removal.",
        }
    ]
    registry = {
        "mode": "expected_after",
        "expected_after_sha256": hashlib.sha256(
            (repo / "experiments/registry.yaml").read_bytes()
        ).hexdigest(),
        "changes": [
            {
                "change_id": "fake-update",
                "kind": "update_field",
                "entity_id": "__registry__",
                "field_path": ["schema_version"],
                "from": 2,
                "to": 3,
                "reason": "placeholder to satisfy shape",
                "evidence": ["evidence.json"],
            }
        ],
    }
    delta = write_delta(repo, base, "TEST-REMOVE-ENTITY", operations, registry=registry)
    with pytest.raises(MODULE.HandoffDeltaError, match="forbids destructive registry entity removal"):
        MODULE.check_delta(repo, delta)


def test_standard_stored_report_is_revalidated_and_stale_report_is_rejected(tmp_path: Path) -> None:
    repo, base = make_repo(tmp_path)
    operations = [
        {
            "operation_id": "append-a",
            "op": "append_to_section",
            "heading_path": ["Master v1", "A"],
            "block_id": "block-a",
            "content": "A extension.",
        }
    ]
    delta = write_delta(repo, base, "TEST-STORED-REPORT", operations)
    fresh = MODULE.check_delta(repo, delta)
    report_path = delta.parent / MODULE.REPORT_FILENAME
    report_path.write_text(json.dumps(fresh.report, indent=2, sort_keys=True) + "\n")
    metadata = MODULE.validate_stored_report(repo, delta, fresh.report)
    assert metadata["report_schema_version"] == 2
    stored = json.loads(report_path.read_text())
    stored["candidate_sha256"] = "0" * 64
    report_path.write_text(json.dumps(stored, indent=2, sort_keys=True) + "\n")
    with pytest.raises(MODULE.HandoffDeltaError, match="stale or does not match"):
        MODULE.validate_stored_report(repo, delta, fresh.report)


def test_standard_stored_report_allows_runtime_drift_but_checks_hard_limit(
    tmp_path: Path,
) -> None:
    repo, base = make_repo(tmp_path)
    operations = [
        {
            "operation_id": "append-a",
            "op": "append_to_section",
            "heading_path": ["Master v1", "A"],
            "block_id": "block-a",
            "content": "A extension.",
        }
    ]
    delta = write_delta(repo, base, "TEST-RUNTIME-DRIFT", operations)
    fresh = MODULE.check_delta(repo, delta)
    stored = json.loads(json.dumps(fresh.report))
    stored["validation_worktree_head"] = base
    stored["comparison_target"] = "repository_commit"
    stored["repository_commit"] = base
    stored["performance"].update(
        {"total_ms": 1000.0, "within_target": True, "within_hard_limit": True}
    )
    report_path = delta.parent / MODULE.REPORT_FILENAME
    report_path.write_text(json.dumps(stored, indent=2, sort_keys=True) + "\n")
    assert MODULE.validate_stored_report(repo, delta, fresh.report)["performance_total_ms"] == 1000.0

    stored["performance"].update(
        {"total_ms": 16000.0, "within_target": False, "within_hard_limit": False}
    )
    report_path.write_text(json.dumps(stored, indent=2, sort_keys=True) + "\n")
    with pytest.raises(MODULE.HandoffDeltaError, match="exceeded its recorded hard"):
        MODULE.validate_stored_report(repo, delta, fresh.report)


def test_standard_auto_check_requires_sibling_report(tmp_path: Path) -> None:
    repo, base = make_repo(tmp_path)
    operations = [
        {
            "operation_id": "append-a",
            "op": "append_to_section",
            "heading_path": ["Master v1", "A"],
            "block_id": "block-a",
            "content": "A extension.",
        }
    ]
    write_delta(repo, base, "TEST-AUTO-REPORT", operations)
    with pytest.raises(MODULE.HandoffDeltaError, match="missing required SHADOW_REPORT.json"):
        MODULE.auto_check(repo, allow_full_due=True)


def test_standard_pair_check_rejects_registry_semantic_conflict(tmp_path: Path) -> None:
    repo, base = make_repo(tmp_path)
    base_handoff = git(repo, "show", f"{base}:docs/handoff.md") + "\n"
    base_registry = git(repo, "show", f"{base}:experiments/registry.yaml") + "\n"

    def make_delta(update_id: str, section: str, to_value: str) -> Path:
        operation = {
            "operation_id": f"append-{section.lower()}",
            "op": "append_to_section",
            "heading_path": ["Master v1", section],
            "block_id": f"block-{section.lower()}",
            "content": f"{section} update.",
        }
        rendered = MODULE.render(base_handoff, [operation]).text
        directory = repo / "docs/handoff_deltas" / update_id
        directory.mkdir(parents=True)
        payload = {
            "schema_version": 2,
            "update_id": update_id,
            "mode": "shadow",
            "base": {
                "commit": base,
                "handoff_sha256": digest(base_handoff),
                "registry_sha256": digest(base_registry),
            },
            "renderer_version": 1,
            "operations": [operation],
            "registry": {
                "mode": "expected_after",
                "expected_after_sha256": "0" * 64,
                "changes": [
                    {
                        "change_id": f"status-{to_value}",
                        "kind": "transition",
                        "entity_id": "EXP-A-01",
                        "field_path": ["status"],
                        "machine": "scientific_status",
                        "from": "not_run",
                        "to": to_value,
                        "evidence": ["docs/handoff.md"],
                    }
                ],
            },
            "expected": {
                "candidate_sha256": digest(rendered),
                "manual_sha256": digest(rendered),
            },
        }
        path = directory / "HANDOFF_DELTA.yaml"
        path.write_text(yaml.safe_dump(payload, sort_keys=False))
        return path

    delta_a = make_delta("TEST-PAIR-A", "A", "pilot")
    delta_b = make_delta("TEST-PAIR-B", "B", "finite_step_validated")
    with pytest.raises(MODULE.HandoffDeltaError, match="Semantic registry conflict"):
        MODULE.pair_check(repo, delta_a, delta_b)



def test_standard_report_backfill_does_not_count_as_second_new_delta(tmp_path: Path) -> None:
    repo, base = make_repo(tmp_path)
    first_ops = [
        {
            "operation_id": "insert-first",
            "op": "insert_after_heading",
            "heading_path": ["Master v1", "A"],
            "block_id": "first-block",
            "content": "First observation.",
        }
    ]
    first_delta = write_delta(repo, base, "TEST-FIRST-OBS", first_ops)
    first_result = MODULE.check_delta(repo, first_delta)
    (first_delta.parent / MODULE.REPORT_FILENAME).write_text(
        json.dumps(first_result.report, indent=2, sort_keys=True) + "\n"
    )
    git(repo, "add", ".")
    git(repo, "commit", "-m", "first observation")
    first_commit = git(repo, "rev-parse", "HEAD")

    # Start a second authority update while touching only the historical report.
    second_ops = [
        {
            "operation_id": "insert-second",
            "op": "insert_after_heading",
            "heading_path": ["Master v1", "B"],
            "block_id": "second-block",
            "content": "Second observation.",
        }
    ]
    second_delta = write_delta(repo, first_commit, "TEST-SECOND-OBS", second_ops)
    second_result = MODULE.check_delta(repo, second_delta)
    (second_delta.parent / MODULE.REPORT_FILENAME).write_text(
        json.dumps(second_result.report, indent=2, sort_keys=True) + "\n"
    )
    # Re-materialize the historical report with repository-after-image provenance.
    historical = MODULE.check_delta(repo, first_delta, target_commit=first_commit)
    (first_delta.parent / MODULE.REPORT_FILENAME).write_text(
        json.dumps(historical.report, indent=2, sort_keys=True) + "\n"
    )

    result = MODULE.auto_check(repo, allow_full_due=True)
    assert result["status"] == "PASS"
    assert result["changed_delta_file_count"] == 1
    assert result["delta_count"] == 2

def test_full_acceptance_observation_audit_derives_repository_commit_and_coverage(tmp_path: Path) -> None:
    repo, base = make_repo(tmp_path)
    operations = [
        {
            "operation_id": "append-a",
            "op": "append_to_section",
            "heading_path": ["Master v1", "A"],
            "block_id": "block-a",
            "content": "A extension.",
        }
    ]
    delta = write_delta(repo, base, "TEST-REAL-OBSERVATION", operations)
    write_report(repo, delta)
    observation_commit = commit_all(repo, "real observation")
    records = MODULE.observation_records(repo)
    assert len(records) == 1
    assert records[0]["repository_commit"] == observation_commit
    assert records[0]["validation_worktree_head"] == base

    status = MODULE.acceptance_status(repo)
    assert status["full_acceptance_due"] is True
    assert status["full_acceptance_due_reasons"] == ["no_successful_full_acceptance_report"]

    full_report = delta.parent / MODULE.FULL_REPORT_FILENAME
    write_full_report(repo, full_report, ["TEST-REAL-OBSERVATION"])
    full_commit = commit_all(repo, "full acceptance")
    current = MODULE.acceptance_status(repo)
    assert current["full_acceptance_due"] is False
    assert current["latest_full_acceptance"]["repository_commit"] == full_commit

    full_time = MODULE.commit_datetime(repo, full_commit)
    after_eight_days = (full_time + timedelta(days=8)).isoformat()
    still_current = MODULE.acceptance_status(repo, as_of=after_eight_days)
    assert still_current["full_acceptance_due"] is False


def test_full_acceptance_time_trigger_requires_an_uncovered_relevant_update(tmp_path: Path) -> None:
    repo, base = make_repo(tmp_path)
    first_operations = [
        {
            "operation_id": "append-a",
            "op": "append_to_section",
            "heading_path": ["Master v1", "A"],
            "block_id": "block-a",
            "content": "A extension.",
        }
    ]
    first = write_delta(repo, base, "TEST-FIRST-OBSERVATION", first_operations)
    write_report(repo, first)
    commit_all(repo, "first observation")
    full_report = first.parent / MODULE.FULL_REPORT_FILENAME
    write_full_report(repo, full_report, ["TEST-FIRST-OBSERVATION"])
    full_commit = commit_all(repo, "full acceptance")

    second_base = git(repo, "rev-parse", "HEAD")
    second_operations = [
        {
            "operation_id": "append-b",
            "op": "append_to_section",
            "heading_path": ["Master v1", "B"],
            "block_id": "block-b",
            "content": "B extension.",
        }
    ]
    second = write_delta(repo, second_base, "TEST-SECOND-OBSERVATION", second_operations)
    write_report(repo, second)
    commit_all(repo, "second observation")

    as_of = (MODULE.commit_datetime(repo, full_commit) + timedelta(days=8)).isoformat()
    status = MODULE.acceptance_status(repo, as_of=as_of)
    assert status["full_acceptance_due"] is True
    assert "calendar_interval_with_relevant_update_reached" in status[
        "full_acceptance_due_reasons"
    ]
    assert status["uncovered_update_ids"] == ["TEST-SECOND-OBSERVATION"]


def test_full_acceptance_count_trigger_is_machine_enforced(tmp_path: Path) -> None:
    repo, base = make_repo(tmp_path)
    policy_path = repo / "docs/handoff_delta_policy.yaml"
    policy = yaml.safe_load(policy_path.read_text())
    policy["full_acceptance"]["successful_update_interval"] = 1
    policy_path.write_text(yaml.safe_dump(policy, sort_keys=False))
    git(repo, "add", ".")
    git(repo, "commit", "-m", "test policy interval")
    base = git(repo, "rev-parse", "HEAD")
    operations = [
        {
            "operation_id": "append-a",
            "op": "append_to_section",
            "heading_path": ["Master v1", "A"],
            "block_id": "block-a",
            "content": "A extension.",
        }
    ]
    delta = write_delta(repo, base, "TEST-COUNT-TRIGGER", operations)
    write_report(repo, delta)
    commit_all(repo, "count observation")
    status = MODULE.acceptance_status(repo)
    assert status["full_acceptance_due"] is True
    assert "successful_relevant_update_interval_reached" in status[
        "full_acceptance_due_reasons"
    ]


def test_standard_schema2_registry_field_removal_is_rejected(tmp_path: Path) -> None:
    repo, base = make_repo(tmp_path)
    current = yaml.safe_load((repo / "experiments/registry.yaml").read_text())
    del current["experiments"][0]["execution_gate"]
    (repo / "experiments/registry.yaml").write_text(yaml.safe_dump(current, sort_keys=False))
    (repo / "evidence.json").write_text("{}")
    operations = [
        {
            "operation_id": "append-a",
            "op": "append_to_section",
            "heading_path": ["Master v1", "A"],
            "block_id": "block-a",
            "content": "Attempt field removal.",
        }
    ]
    registry = {
        "mode": "expected_after",
        "expected_after_sha256": hashlib.sha256(
            (repo / "experiments/registry.yaml").read_bytes()
        ).hexdigest(),
        "changes": [
            {
                "change_id": "remove-gate",
                "kind": "update_field",
                "entity_id": "EXP-A-01",
                "field_path": ["execution_gate"],
                "from": {"state": "blocked"},
                "to": None,
                "reason": "intentionally destructive",
                "evidence": ["evidence.json"],
            }
        ],
    }
    delta = write_delta(repo, base, "TEST-REMOVE-FIELD", operations, registry=registry)
    with pytest.raises(MODULE.HandoffDeltaError, match="forbids destructive registry field removal"):
        MODULE.check_delta(repo, delta)


def test_standard_changed_paths_do_not_double_count_previous_committed_delta(tmp_path: Path) -> None:
    repo, base = make_repo(tmp_path)
    first_ops = [
        {
            "operation_id": "append-a",
            "op": "append_to_section",
            "heading_path": ["Master v1", "A"],
            "block_id": "block-a",
            "content": "First update.",
        }
    ]
    first = write_delta(repo, base, "TEST-PREVIOUS-DELTA", first_ops)
    write_report(repo, first)
    commit_all(repo, "previous delta")

    second_base = git(repo, "rev-parse", "HEAD")
    second_ops = [
        {
            "operation_id": "append-b",
            "op": "append_to_section",
            "heading_path": ["Master v1", "B"],
            "block_id": "block-b",
            "content": "Second update.",
        }
    ]
    second = write_delta(repo, second_base, "TEST-CURRENT-DELTA", second_ops)
    write_report(repo, second)
    paths = MODULE.changed_paths(repo)
    deltas = MODULE.discover_changed_deltas(repo, paths)
    assert [path.parent.name for path in deltas] == ["TEST-CURRENT-DELTA"]


def test_standard_v2_report_projection_detects_registry_coverage_tamper(tmp_path: Path) -> None:
    repo, base = make_repo(tmp_path)
    operations = [
        {
            "operation_id": "append-a",
            "op": "append_to_section",
            "heading_path": ["Master v1", "A"],
            "block_id": "block-a",
            "content": "A extension.",
        }
    ]
    delta = write_delta(repo, base, "TEST-REPORT-COVERAGE", operations)
    fresh = MODULE.check_delta(repo, delta)
    report_path = delta.parent / MODULE.REPORT_FILENAME
    report_path.write_text(json.dumps(fresh.report, indent=2, sort_keys=True) + "\n")
    stored = json.loads(report_path.read_text())
    stored["registry_change_coverage"]["fully_declared"] = False
    report_path.write_text(json.dumps(stored, indent=2, sort_keys=True) + "\n")
    with pytest.raises(MODULE.HandoffDeltaError, match="stale or does not match"):
        MODULE.validate_stored_report(repo, delta, fresh.report)


def test_standard_acceptance_status_uses_lightweight_observation_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, base = make_repo(tmp_path)
    operations = [
        {
            "operation_id": "append-a",
            "op": "append_to_section",
            "heading_path": ["Master v1", "A"],
            "block_id": "block-a",
            "content": "A extension.",
        }
    ]
    delta = write_delta(repo, base, "TEST-LIGHTWEIGHT-STATUS", operations)
    write_report(repo, delta)
    commit_all(repo, "observation")

    def fail_replay(*args: object, **kwargs: object) -> object:
        raise AssertionError("acceptance_status must not replay the historical corpus")

    monkeypatch.setattr(MODULE, "check_delta", fail_replay)
    status = MODULE.acceptance_status(repo)
    assert status["successful_real_observation_count"] == 1


def test_standard_changed_report_maps_back_to_sibling_delta(tmp_path: Path) -> None:
    repo, base = make_repo(tmp_path)
    operations = [
        {
            "operation_id": "append-a",
            "op": "append_to_section",
            "heading_path": ["Master v1", "A"],
            "block_id": "block-a",
            "content": "A extension.",
        }
    ]
    delta = write_delta(repo, base, "TEST-REPORT-TO-DELTA", operations)
    write_report(repo, delta)
    commit_all(repo, "observation")
    report = delta.parent / MODULE.REPORT_FILENAME
    payload = json.loads(report.read_text())
    payload["performance"]["total_ms"] = 1.0
    report.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    deltas = MODULE.discover_changed_deltas(repo, MODULE.changed_paths(repo))
    assert deltas == [delta.resolve()]


def test_full_acceptance_report_rejects_invalid_fingerprint(tmp_path: Path) -> None:
    repo, base = make_repo(tmp_path)
    operations = [
        {
            "operation_id": "append-a",
            "op": "append_to_section",
            "heading_path": ["Master v1", "A"],
            "block_id": "block-a",
            "content": "A extension.",
        }
    ]
    delta = write_delta(repo, base, "TEST-FULL-FINGERPRINT", operations)
    write_report(repo, delta)
    commit_all(repo, "observation")
    full_report = delta.parent / MODULE.FULL_REPORT_FILENAME
    write_full_report(repo, full_report, ["TEST-FULL-FINGERPRINT"])
    payload = json.loads(full_report.read_text())
    payload["coverage"]["observation_fingerprint"] = "0" * 64
    full_report.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    commit_all(repo, "bad full report")
    observations = MODULE.observation_records(repo, replay=False)
    with pytest.raises(MODULE.HandoffDeltaError, match="fingerprint is invalid"):
        MODULE.full_acceptance_records(repo, observations)

def test_full_acceptance_runner_canonicalizes_coverage_order() -> None:
    observations = [
        {"update_id": "Z-LAST", "kind": "real"},
        {"update_id": "BOOTSTRAP", "kind": "bootstrap"},
        {"update_id": "A-FIRST", "kind": "real"},
    ]
    assert RUNNER.real_observation_ids(observations) == ["A-FIRST", "Z-LAST"]
