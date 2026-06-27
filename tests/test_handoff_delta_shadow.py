from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
import subprocess
import sys
import time
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
        "schema_version": 1,
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
        or {"mode": "unchanged", "expected_after_sha256": None, "transitions": []},
        "expected": {
            "candidate_sha256": digest(rendered),
            "manual_sha256": digest(manual_text),
        },
    }
    path = delta_dir / "HANDOFF_DELTA.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True))
    return path


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
    result = MODULE.check_delta(ROOT, delta)
    assert result.report["status"] == "PASS"
    assert result.report["exact_manual_candidate_match"] is True
    assert result.report["idempotence_passed"] is True
    assert result.report["candidate_replaced_authority"] is False


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
        "transitions": [
            {
                "assertion_id": "pilot-status",
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
        "transitions": [
            {
                "assertion_id": "invalid-proof",
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


def test_full_acceptance_fast_gate_stays_below_hard_limit() -> None:
    delta = ROOT / "docs/handoff_deltas/DU1-E6-SEMANTIC-GAP-FORMAL-2026-06-27/HANDOFF_DELTA.yaml"
    samples = []
    for _ in range(3):
        started = time.perf_counter()
        MODULE.check_delta(ROOT, delta)
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
