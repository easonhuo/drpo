from __future__ import annotations

import json
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
    """Exercise the current repository through the production CLI boundary.

    The full suite imports the authority module in several test modules. This
    current-repository integration check must not depend on in-process module
    state left by earlier unit tests, so it runs the production command in a
    fresh interpreter.
    """

    config = authority.load_authority(REPO_ROOT)
    head = authority._git_text(REPO_ROOT, "rev-parse", "HEAD")
    activation_parent = config["delta_authority"]["activation_parent_commit"]
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "handoff_authority.py"),
        "verify",
        "--repo-root",
        str(REPO_ROOT),
        "--json",
    ]
    if config["mode"] != "manual" and head == activation_parent:
        command.append("--prepared")
    completed = subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
    verified = json.loads(completed.stdout)

    if config["mode"] == "manual":
        assert verified["mode"] == "manual"
    elif head == activation_parent:
        assert verified["mode"] == "cutover_prepared"
    else:
        assert verified["mode"] == "delta"
        assert verified["legacy_inert_delta_count"] == 1
        assert verified["legacy_inert_update_ids"] == [
            "EXT-H-E7-SQEXP-GAE-FROZEN-DIAGNOSTIC-2026-07-18"
        ]


def test_legacy_inert_delta_compatibility_is_exactly_bound() -> None:
    path = (
        REPO_ROOT
        / "docs/handoff_deltas/EXT-H-E7-SQEXP-GAE-FROZEN-DIAGNOSTIC-2026-07-18/"
        "HANDOFF_DELTA.yaml"
    )
    delta = authority._load_yaml(path, "legacy inert delta")
    first_add = "11992ca5de7f2c4a3837cf32aa4e23696ec18ef3"
    integration = "cd770f47b89f8971923945c19caec49720c0e139"
    report = authority._legacy_inert_materialization_report(
        REPO_ROOT,
        path,
        delta,
        first_add=first_add,
        integration_commit=integration,
    )
    assert report == {
        "current_handoff_before_sha256": (
            "f8ff67ab71c0f53b21fc96967a13aa3e5b8500e42d25464e378df23e1f62c4e8"
        ),
        "materialized_handoff_after_sha256": (
            "f8ff67ab71c0f53b21fc96967a13aa3e5b8500e42d25464e378df23e1f62c4e8"
        ),
        "update_id": "EXT-H-E7-SQEXP-GAE-FROZEN-DIAGNOSTIC-2026-07-18",
    }

    tampered = dict(delta)
    tampered["update_id"] = "EXT-H-E7-SQEXP-GAE-FROZEN-DIAGNOSTIC-TAMPERED"
    with pytest.raises(authority.HandoffAuthorityError, match="provenance mismatch"):
        authority._legacy_inert_materialization_report(
            REPO_ROOT,
            path,
            tampered,
            first_add=first_add,
            integration_commit=integration,
        )

    with pytest.raises(authority.HandoffAuthorityError, match="provenance mismatch"):
        authority._legacy_inert_materialization_report(
            REPO_ROOT,
            path,
            delta,
            first_add=first_add,
            integration_commit=first_add,
        )

    ordinary = (
        REPO_ROOT
        / "docs/handoff_deltas/EXT-C-E8-PAPER-ALIGNED-LAMBDA-ROUND1-2026-07-16/"
        "HANDOFF_DELTA.yaml"
    )
    ordinary_delta = authority._load_yaml(ordinary, "ordinary delta")
    assert (
        authority._legacy_inert_materialization_report(
            REPO_ROOT,
            ordinary,
            ordinary_delta,
            first_add="0" * 40,
            integration_commit="0" * 40,
        )
        is None
    )


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
