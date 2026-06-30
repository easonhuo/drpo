from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "manuscript_cascade.py"


def load_module():
    spec = importlib.util.spec_from_file_location("manuscript_cascade", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


mc = load_module()


def write_block(paragraph_id: str, title: str, body: str, parent_line: str | None = None) -> str:
    rows = [
        f"<!-- MANUSCRIPT:BEGIN {paragraph_id} -->",
        f"## [{paragraph_id}] {title}",
    ]
    if parent_line is not None:
        rows.append(parent_line)
    rows.extend(["", body, f"<!-- MANUSCRIPT:END {paragraph_id} -->", ""])
    return "\n".join(rows)


def block_hash(text: str, layer: str) -> str:
    return mc.parse_markdown_blocks(text, layer=layer, source="fixture")[0].sha256


def create_artifacts(tmp_path: Path, *, with_prose: bool = True):
    outline_text = write_block("INTRO-P01", "Background", "Outline contract.")
    outline_hash = block_hash(outline_text, "outline")
    blueprint_text = write_block(
        "INTRO-P01",
        "Background",
        "Topic sentence and evidence plan.",
        f"Parent-Outline-SHA256: `{outline_hash}`",
    )
    blueprint_hash = block_hash(blueprint_text, "blueprint")
    prose_text = write_block(
        "INTRO-P01",
        "Background",
        "Finished manuscript paragraph.",
        f"Parent-Blueprint-SHA256: `{blueprint_hash}`",
    )
    (tmp_path / "outline.md").write_text(outline_text)
    (tmp_path / "blueprint.md").write_text(blueprint_text)
    if with_prose:
        (tmp_path / "prose.md").write_text(prose_text)
    config = {
        "schema_version": 1,
        "manuscript_id": "test-paper",
        "sections": [
            {
                "id": "introduction",
                "layers": {
                    "outline": "outline.md",
                    "blueprint": "blueprint.md",
                    "prose": "prose.md" if with_prose else None,
                },
            }
        ],
    }
    config_path = tmp_path / "hierarchy.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    return config_path


def issue_data(
    root: str,
    *,
    state: str = "planned",
    with_prose: bool = True,
    kind: str | None = None,
    reported_layer: str | None = None,
    outline_change_authorized: bool | None = None,
):
    statuses = {}
    blocked = False
    for layer in mc.LAYER_ORDER:
        if layer == "prose" and not with_prose:
            statuses[layer] = "not_present"
        elif blocked:
            statuses[layer] = "blocked"
        elif layer == root:
            statuses[layer] = "fail"
            blocked = True
        else:
            statuses[layer] = "pass"
    root_index = mc.LAYER_ORDER.index(root)
    required = [
        layer
        for layer in mc.LAYER_ORDER[root_index:]
        if not (layer == "prose" and not with_prose)
    ]
    if kind is None:
        kind = "alignment_repair" if root != "outline" else "content_revision"
    if reported_layer is None:
        reported_layer = root
    if outline_change_authorized is None:
        outline_change_authorized = root == "outline"
    return {
        "schema_version": 2,
        "issue_id": "ISSUE-01",
        "section_id": "introduction",
        "paragraph_ids": ["INTRO-P01"],
        "problem": "Reported manuscript problem.",
        "change_control": {
            "kind": kind,
            "reported_layer": reported_layer,
            "outline_change_authorized": outline_change_authorized,
            "authorization_evidence": "Explicit test authorization or verified outline pass.",
        },
        "checks": [
            {
                "layer": layer,
                "status": statuses[layer],
                "evidence": f"Evidence for {layer}.",
            }
            for layer in mc.LAYER_ORDER
        ],
        "resolution": {
            "state": state,
            "required_layers": required,
            "changed_layers": required if state == "completed" else [],
        },
    }



def test_alignment_mismatch_cannot_be_reclassified_as_outline_failure(tmp_path: Path):
    config_path = create_artifacts(tmp_path)
    config = mc.load_config(config_path)
    issue = issue_data(
        "outline",
        kind="alignment_repair",
        reported_layer="blueprint",
        outline_change_authorized=False,
    )
    with pytest.raises(mc.ManuscriptCascadeError, match="alignment mismatch is not evidence"):
        mc.validate_issue(issue, config)


def test_blueprint_alignment_repair_preserves_outline(tmp_path: Path):
    config_path = create_artifacts(tmp_path)
    config = mc.load_config(config_path)
    issue = issue_data(
        "blueprint",
        kind="alignment_repair",
        reported_layer="blueprint",
        outline_change_authorized=False,
    )
    root, required, summary = mc.validate_issue(issue, config)
    assert root == "blueprint"
    assert required == ["blueprint", "prose"]
    assert summary["outline_change_authorized"] is False


def test_outline_content_revision_requires_explicit_authorization(tmp_path: Path):
    config_path = create_artifacts(tmp_path)
    config = mc.load_config(config_path)
    issue = issue_data(
        "outline",
        kind="content_revision",
        reported_layer="blueprint",
        outline_change_authorized=False,
    )
    with pytest.raises(mc.ManuscriptCascadeError, match="explicit outline-change authorization"):
        mc.validate_issue(issue, config)


def test_outline_content_revision_can_cascade_when_authorized(tmp_path: Path):
    config_path = create_artifacts(tmp_path)
    config = mc.load_config(config_path)
    issue = issue_data(
        "outline",
        kind="content_revision",
        reported_layer="blueprint",
        outline_change_authorized=True,
    )
    root, required, summary = mc.validate_issue(issue, config)
    assert root == "outline"
    assert required == ["outline", "blueprint", "prose"]
    assert summary["outline_change_authorized"] is True

def git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    return proc.stdout.strip()


def test_valid_artifact_hierarchy_passes(tmp_path: Path):
    config_path = create_artifacts(tmp_path)
    config = mc.load_config(config_path)
    result = mc.validate_artifacts(config, repo_root=tmp_path)
    assert result["sections"]["introduction"]["status"] == "pass"
    assert result["sections"]["introduction"]["paragraph_ids"] == ["INTRO-P01"]


def test_child_id_or_order_must_match_parent(tmp_path: Path):
    config_path = create_artifacts(tmp_path)
    blueprint = (tmp_path / "blueprint.md").read_text().replace("INTRO-P01", "INTRO-P02")
    (tmp_path / "blueprint.md").write_text(blueprint)
    config = mc.load_config(config_path)
    with pytest.raises(mc.ManuscriptCascadeError, match="paragraph ids/order"):
        mc.validate_artifacts(config, repo_root=tmp_path)


def test_stale_parent_hash_is_rejected(tmp_path: Path):
    config_path = create_artifacts(tmp_path)
    outline = (tmp_path / "outline.md").read_text().replace(
        "Outline contract.", "Changed outline contract."
    )
    (tmp_path / "outline.md").write_text(outline)
    config = mc.load_config(config_path)
    with pytest.raises(mc.ManuscriptCascadeError, match="stale blueprint parent hash"):
        mc.validate_artifacts(config, repo_root=tmp_path)


@pytest.mark.parametrize(
    ("root", "expected"),
    [
        ("outline", ["outline", "blueprint", "prose"]),
        ("blueprint", ["blueprint", "prose"]),
        ("prose", ["prose"]),
    ],
)
def test_first_failing_layer_determines_exact_cascade(
    tmp_path: Path, root: str, expected: list[str]
):
    config_path = create_artifacts(tmp_path)
    config = mc.load_config(config_path)
    issue = issue_data(root)
    actual_root, required, summary = mc.validate_issue(issue, config)
    assert actual_root == root
    assert required == expected
    assert summary["required_layers"] == expected


def test_issue_checks_must_be_top_down(tmp_path: Path):
    config_path = create_artifacts(tmp_path)
    config = mc.load_config(config_path)
    issue = issue_data("blueprint")
    issue["checks"][0], issue["checks"][1] = issue["checks"][1], issue["checks"][0]
    with pytest.raises(mc.ManuscriptCascadeError, match="top-down"):
        mc.validate_issue(issue, config)


def test_downstream_check_must_be_blocked_after_parent_failure(tmp_path: Path):
    config_path = create_artifacts(tmp_path)
    config = mc.load_config(config_path)
    issue = issue_data("outline")
    issue["checks"][1]["status"] = "pass"
    with pytest.raises(mc.ManuscriptCascadeError, match="must be blocked"):
        mc.validate_issue(issue, config)


def test_completed_issue_cannot_skip_downstream_layer(tmp_path: Path):
    config_path = create_artifacts(tmp_path)
    config = mc.load_config(config_path)
    issue = issue_data("outline", state="completed")
    issue["resolution"]["changed_layers"] = ["outline", "blueprint"]
    with pytest.raises(mc.ManuscriptCascadeError, match="exactly the required cascade"):
        mc.validate_issue(issue, config)


def test_absent_prose_is_excluded_from_required_cascade(tmp_path: Path):
    config_path = create_artifacts(tmp_path, with_prose=False)
    config = mc.load_config(config_path)
    issue = issue_data("outline", with_prose=False)
    root, required, _ = mc.validate_issue(issue, config)
    assert root == "outline"
    assert required == ["outline", "blueprint"]


def test_git_cascade_requires_all_downstream_files_and_preserves_upstream(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    git(repo, "config", "user.name", "Test")
    git(repo, "config", "user.email", "test@example.invalid")
    config_path = create_artifacts(repo)
    git(repo, "add", ".")
    git(repo, "commit", "-m", "base")
    base = git(repo, "rev-parse", "HEAD")

    # Blueprint is the first failing layer: leave outline unchanged, update blueprint and prose.
    outline_hash = block_hash((repo / "outline.md").read_text(), "outline")
    blueprint_text = write_block(
        "INTRO-P01",
        "Background",
        "Revised blueprint plan.",
        f"Parent-Outline-SHA256: `{outline_hash}`",
    )
    (repo / "blueprint.md").write_text(blueprint_text)
    blueprint_hash = block_hash(blueprint_text, "blueprint")
    prose_text = write_block(
        "INTRO-P01",
        "Background",
        "Revised manuscript paragraph.",
        f"Parent-Blueprint-SHA256: `{blueprint_hash}`",
    )
    (repo / "prose.md").write_text(prose_text)
    git(repo, "add", "blueprint.md", "prose.md")
    git(repo, "commit", "-m", "cascade")
    head = git(repo, "rev-parse", "HEAD")

    config = mc.load_config(config_path)
    issue = issue_data("blueprint", state="completed")
    _, required, _ = mc.validate_issue(issue, config)
    result = mc.validate_git_cascade(
        issue=issue,
        config=config,
        required_layers=required,
        repo_root=repo,
        base=base,
        head=head,
    )
    assert result["required_paths"] == ["blueprint.md", "prose.md"]


def test_git_cascade_rejects_rewriting_verified_upstream(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    git(repo, "config", "user.name", "Test")
    git(repo, "config", "user.email", "test@example.invalid")
    config_path = create_artifacts(repo)
    git(repo, "add", ".")
    git(repo, "commit", "-m", "base")
    base = git(repo, "rev-parse", "HEAD")
    (repo / "outline.md").write_text((repo / "outline.md").read_text() + "\nUnexpected.\n")
    (repo / "blueprint.md").write_text((repo / "blueprint.md").read_text() + "\nChanged.\n")
    (repo / "prose.md").write_text((repo / "prose.md").read_text() + "\nChanged.\n")
    git(repo, "add", ".")
    git(repo, "commit", "-m", "bad cascade")
    head = git(repo, "rev-parse", "HEAD")

    config = mc.load_config(config_path)
    issue = issue_data("blueprint", state="completed")
    _, required, _ = mc.validate_issue(issue, config)
    with pytest.raises(mc.ManuscriptCascadeError, match="must not be rewritten"):
        mc.validate_git_cascade(
            issue=issue,
            config=config,
            required_layers=required,
            repo_root=repo,
            base=base,
            head=head,
        )
