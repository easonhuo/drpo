from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "pr-gate-log.yml"
AUTHORITY_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "handoff-authority.yml"
AUTHORITY_RUNNER = REPO_ROOT / "scripts" / "run_handoff_authority_gate.sh"


def _workflow_text() -> str:
    return WORKFLOW.read_text()


def test_tiered_plan_is_shadow_only_and_uses_exact_pr_shas() -> None:
    text = _workflow_text()
    start = text.index("- name: Tiered test plan (shadow only)")
    end = text.index("- name: Python compile", start)
    block = text[start:end]

    assert "continue-on-error: true" in block
    assert "github.event.pull_request.base.sha" in block
    assert "github.event.pull_request.head.sha" in block
    assert "scripts/select_update_tests.py" in block
    assert '--base "$BASE_SHA"' in block
    assert '--head "$HEAD_SHA"' in block
    assert "--mode auto" in block
    assert "--json" in block
    assert "--execute" not in block
    assert "GITHUB_STEP_SUMMARY" in block


def test_shadow_phase_preserves_read_only_checkout_and_legacy_full_gates() -> None:
    text = _workflow_text()

    assert "permissions:\n  contents: read" in text
    assert "fetch-depth: 0" in text
    assert "python -m compileall -q src scripts tools tests" in text
    assert "bash -n tools/drpo-update/drpo-update tools/drpo-update/install.sh" in text
    assert "python scripts/handoff_authority.py verify --repo-root ." in text
    assert "python scripts/validate_formal_execution_channel.py --repo-root ." in text
    assert "python scripts/validate_governance_rule_inventory.py --repo-root ." in text
    assert "python scripts/validate_governance_pipeline_stage_status.py --repo-root ." in text
    assert "python -m pytest -q" in text
    assert "ruff check ." in text


def test_handoff_authority_workflow_is_path_scoped_and_read_only() -> None:
    text = AUTHORITY_WORKFLOW.read_text()

    assert "pull_request:" in text
    assert "paths:" in text
    assert '"docs/handoff_deltas/**"' in text
    assert '"experiments/registry.yaml"' in text
    assert '"scripts/handoff_authority.py"' in text
    assert "permissions:\n  contents: read" in text
    assert "contents: write" not in text
    assert "git push" not in text
    assert "workflow_dispatch" not in text
    assert "fetch-depth: 0" in text
    assert "python -m pip install -e '.[dev]'" in text
    assert "bash -n scripts/run_handoff_authority_gate.sh" in text
    assert "bash scripts/run_handoff_authority_gate.sh" in text
    assert "github.event.pull_request.base.sha" in text
    assert "github.event.pull_request.head.sha" in text


def test_handoff_authority_runner_is_exact_diff_read_only_gate() -> None:
    text = AUTHORITY_RUNNER.read_text()

    assert 'git diff --name-only "$base_sha" "$head_sha"' in text
    assert "docs/handoff_deltas/*/HANDOFF_DELTA.yaml" in text
    assert "scripts/handoff_authority.py verify --repo-root . --json" in text
    assert "scripts/build_stage4_context.py --repo-root . --json check" in text
    assert "scripts/validate_governance_pipeline_stage_status.py --repo-root ." in text
    assert "normalize" not in text
    assert "git push" not in text
