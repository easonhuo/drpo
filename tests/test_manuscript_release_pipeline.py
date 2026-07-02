from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/manuscript_release_pipeline.py"
FIXTURE = ROOT / "tests/fixtures/generic_manuscript"


def load_module():
    spec = importlib.util.spec_from_file_location("generic_release_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def prepare_root(tmp_path: Path) -> Path:
    for rel in (
        "scripts/manuscript_publication_pipeline.py",
        "docs/manuscript/generic_publication_quality_profile.yaml",
    ):
        dst = tmp_path / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, dst)
    dst_fixture = tmp_path / "tests/fixtures/generic_manuscript"
    shutil.copytree(FIXTURE, dst_fixture)
    return tmp_path


def test_unrelated_project_uses_same_release_engine_without_domain_leakage(tmp_path: Path) -> None:
    module = load_module()
    root = prepare_root(tmp_path)
    manifest_path = root / "tests/fixtures/generic_manuscript/release_manifest.yaml"
    manifest = yaml.safe_load(manifest_path.read_text())
    quality_output = root / "build/publication-quality"
    manifest["quality_gate"]["output"] = str(quality_output)

    result = module.execute(
        root,
        manifest,
        output=root / "build/review.pdf",
        skip_compile=True,
    )

    assert result["status"] == "PASS"
    assert result["project_id"] == "sensor-scheduling-fixture"
    report = json.loads((quality_output / "quality_report.json").read_text())
    assert report["architecture_separation"] == "PASS"
    payload = json.dumps(report, ensure_ascii=False).lower()
    assert "distributed" in payload or report["node_count"] == 2
    for forbidden in ("drpo", "c-u1", "d-u1", "hopper", "countdown", "far-field"):
        assert forbidden not in payload
