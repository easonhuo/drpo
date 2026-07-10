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
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
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
    plan = select_test_plan([r"docs\plan.md"], impact_map)

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
