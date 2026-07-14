from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
AGENT_DIR = ROOT / "scripts" / "agent"
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

from prepare_results_repo_delivery_shadow import prepare_shadow  # noqa: E402
from runspec_lib import RunSpecError  # noqa: E402


def _source(repo: Path) -> Path:
    source = repo / "experiments" / "results" / "completed"
    source.mkdir(parents=True)
    (source / "README.md").write_text("# Completed\n", encoding="utf-8")
    (source / "RESULT_SUMMARY.json").write_text(
        json.dumps(
            {
                "experiment_id": "EXT-C-E8-COMPLETED-01",
                "status": "pilot",
                "package": {"sha256": "abc", "size_bytes": 123},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return source


def test_prepare_shadow_hashes_completed_text_evidence_without_modification(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    source = _source(repo)
    before = {
        path.name: path.read_bytes()
        for path in source.iterdir()
        if path.is_file()
    }
    output = "outputs/e8/shadow/SHADOW_SOURCE_MANIFEST.json"
    payload = prepare_shadow(
        repo,
        "experiments/results/completed",
        output,
    )
    assert payload["status"] == "PASS"
    assert payload["training_executed"] is False
    assert payload["source_modified"] is False
    assert payload["source_file_count"] == 2
    assert (repo / output).is_file()
    after = {
        path.name: path.read_bytes()
        for path in source.iterdir()
        if path.is_file()
    }
    assert after == before


def test_prepare_shadow_rejects_model_like_source(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    source = _source(repo)
    (source / "checkpoint.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(RunSpecError, match="model-like"):
        prepare_shadow(
            repo,
            "experiments/results/completed",
            "outputs/e8/shadow/SHADOW_SOURCE_MANIFEST.json",
        )


def test_prepare_shadow_rejects_symlink(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    source = _source(repo)
    outside = tmp_path / "outside.json"
    outside.write_text("{}\n", encoding="utf-8")
    (source / "linked.json").symlink_to(outside)
    with pytest.raises(RunSpecError, match="symlink"):
        prepare_shadow(
            repo,
            "experiments/results/completed",
            "outputs/e8/shadow/SHADOW_SOURCE_MANIFEST.json",
        )
