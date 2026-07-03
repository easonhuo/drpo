from __future__ import annotations

import subprocess
import sys

import yaml
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import handoff_authority as authority  # noqa: E402


def test_manual_mode_is_safe_noop(tmp_path: Path) -> None:
    repo = tmp_path / "manual-repo"
    config = repo / authority.AUTHORITY_PATH
    config.parent.mkdir(parents=True)
    payload = yaml.safe_load(
        (REPO_ROOT / authority.AUTHORITY_PATH).read_text(encoding="utf-8")
    )
    payload["mode"] = "manual"
    payload["delta_authority"]["checkpoint_manifest"] = None
    payload["delta_authority"]["activation_parent_commit"] = None
    payload["generated_views"]["stage4a_minimal_refresh"] = False
    payload["safety"]["direct_handoff_edit_forbidden"] = False
    config.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    verified = authority.verify_current_state(repo)
    assert verified == {
        "status": "PASS",
        "mode": "manual",
        "manual_handoff_authoritative": True,
        "authority_cutover_allowed": False,
    }


def test_current_repository_authority_phase_verifies() -> None:
    config = authority.load_authority(REPO_ROOT)
    if config["mode"] == "manual":
        verified = authority.verify_current_state(REPO_ROOT)
        assert verified["mode"] == "manual"
        return

    head = authority._git_text(REPO_ROOT, "rev-parse", "HEAD")
    activation_parent = config["delta_authority"]["activation_parent_commit"]
    if head == activation_parent:
        verified = authority.verify_prepared_cutover(REPO_ROOT)
        assert verified["mode"] == "cutover_prepared"
    else:
        verified = authority.verify_current_state(REPO_ROOT)
        assert verified["mode"] == "delta"



def test_historical_e8_taper_v73_delta_matches_its_repository_after_image() -> None:
    delta = (
        REPO_ROOT
        / "docs/handoff_deltas/EXT-C-E8-TAPER-0.5B-CORRECTED-V73-2026-07-03/HANDOFF_DELTA.yaml"
    )
    intent = authority.validate_exact_base_intent(
        REPO_ROOT,
        delta,
        source_patch_commit="c2175257140de31d09753d09d7bc2c62aee96219",
    )
    assert intent.delta["update_id"] == (
        "EXT-C-E8-TAPER-0.5B-CORRECTED-V73-2026-07-03"
    )
    historical_handoff = subprocess.run(
        [
            "git",
            "-C",
            str(REPO_ROOT),
            "show",
            "c2175257140de31d09753d09d7bc2c62aee96219:docs/handoff.md",
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout
    assert historical_handoff == intent.candidate
    assert intent.registry_report["coverage"]["fully_declared"] is True

def test_schema_v3_rejects_reserved_marker_injection(tmp_path: Path) -> None:
    path = tmp_path / "BAD" / "HANDOFF_DELTA.yaml"
    path.parent.mkdir()
    delta: dict[str, Any] = {
        "schema_version": 3,
        "update_id": "BAD",
        "mode": "authoritative",
        "base": {
            "commit": "0" * 40,
            "handoff_sha256": "0" * 64,
            "registry_sha256": "0" * 64,
        },
        "renderer_version": 1,
        "operations": [
            {
                "operation_id": "bad-op",
                "op": "append_to_section",
                "heading_path": ["Root"],
                "block_id": "bad-block",
                "content": "<!-- HANDOFF-DELTA-BLOCK location=x id=y -->",
            }
        ],
        "registry": {
            "mode": "unchanged",
            "exact_base_after_sha256": None,
            "changes": [],
        },
        "expected": {"exact_base_candidate_sha256": "0" * 64},
    }
    with pytest.raises(authority.HandoffAuthorityError, match="may not forge"):
        authority.validate_v3_delta(delta, path)


def test_delta_mode_rejects_new_legacy_schema(tmp_path: Path) -> None:
    path = tmp_path / "docs/handoff_deltas/LEGACY/HANDOFF_DELTA.yaml"
    path.parent.mkdir(parents=True)
    path.write_text("schema_version: 2\nupdate_id: LEGACY\n", encoding="utf-8")
    with pytest.raises(authority.HandoffAuthorityError, match="only newly added schema-v3"):
        authority._find_new_v3_delta(
            tmp_path,
            {"docs/handoff_deltas/LEGACY/HANDOFF_DELTA.yaml": "A"},
        )


def test_repo_relative_accepts_symlinked_repo_root(tmp_path: Path) -> None:
    real_root = tmp_path / "real-repo"
    real_root.mkdir()
    protected = real_root / "docs" / "asset.json"
    protected.parent.mkdir()
    protected.write_text("{}\n", encoding="utf-8")
    linked_root = tmp_path / "repo-link"
    try:
        linked_root.symlink_to(real_root, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("directory symlinks are unavailable")
    assert (
        authority._repo_relative(linked_root, protected.resolve(), "protected asset")
        == "docs/asset.json"
    )
