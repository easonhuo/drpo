from __future__ import annotations
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    p = ROOT / "scripts" / f"{name}.py"
    s = importlib.util.spec_from_file_location(f"test_{name}", p)
    assert s and s.loader
    m = importlib.util.module_from_spec(s)
    sys.modules[s.name] = m
    s.loader.exec_module(m)
    return m


B = load("build_stage4b_candidate")
V = load("validate_stage4b_candidate")


def link(src, dst):
    try:
        os.link(src, dst)
        return dst
    except OSError:
        return shutil.copy2(src, dst)


def copy_repo(tmp):
    return Path(
        shutil.copytree(
            ROOT,
            tmp / "repo",
            copy_function=link,
            ignore=shutil.ignore_patterns(
                ".git", "__pycache__", ".pytest_cache", "*.pyc", "outputs", "wandb"
            ),
        )
    )


def replace(p, data):
    q = p.with_name(p.name + ".tmp")
    q.write_bytes(data)
    q.replace(p)


def test_current_candidate_validates():
    r = V.validate(ROOT)
    assert r["status"] == "PASS" and r["module_count"] == 13 and r["unmapped_count"] == 0


def test_partition_and_reconstruction_are_byte_exact():
    src = (ROOT / "docs/handoff.md").read_bytes()
    assert b"".join(x.payload for x in B.partition(src)) == src
    assert B.reconstruct_from_generated(ROOT / B.EXPECTED_OUTPUT) == src


def test_unchanged_write_is_full_noop():
    p = B.build_plan(ROOT)
    r = B.write_generated(ROOT / B.EXPECTED_OUTPUT, p)
    assert not r["written"] and not r["removed"] and len(r["reused"]) == len(p.outputs)


def test_generated_tamper_is_rejected(tmp_path):
    repo = copy_repo(tmp_path)
    p = repo / B.EXPECTED_OUTPUT / "canonical/countdown_e8.md"
    replace(p, p.read_bytes() + b"\nTAMPER\n")
    with pytest.raises(Exception, match="tampered|stale"):
        V.validate(repo)


def test_authority_cutover_config_is_rejected(tmp_path):
    repo = copy_repo(tmp_path)
    p = repo / B.DEFAULT_CONFIG
    d = yaml.safe_load(p.read_text())
    d["authority_cutover_allowed"] = True
    replace(p, yaml.safe_dump(d, sort_keys=False).encode())
    with pytest.raises(Exception, match="authority_cutover_allowed"):
        B.build_plan(repo)


def test_owner_priority_must_exactly_cover_stage4a_modules(tmp_path):
    repo = copy_repo(tmp_path)
    p = repo / B.DEFAULT_CONFIG
    d = yaml.safe_load(p.read_text())
    d["owner_priority"].pop()
    replace(p, yaml.safe_dump(d, sort_keys=False).encode())
    with pytest.raises(Exception, match="owner_priority"):
        B.build_plan(repo)


def test_reconstruction_cannot_overwrite_authoritative_handoff():
    p = subprocess.run(
        [
            sys.executable,
            "scripts/build_stage4b_candidate.py",
            "--repo-root",
            ".",
            "--check",
            "--reconstruct-output",
            "docs/handoff.md",
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert p.returncode == 2 and "may not overwrite authoritative inputs" in p.stderr


def test_registry_is_reference_only():
    d = yaml.safe_load(
        (ROOT / B.EXPECTED_OUTPUT / "manifests/REGISTRY_REFERENCES.yaml").read_text()
    )
    assert d["authority"] == "registry_references_only"
