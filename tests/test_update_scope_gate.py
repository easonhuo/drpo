from __future__ import annotations

import json
import stat
import zipfile
from pathlib import Path

import importlib.util
import sys

_SPEC = importlib.util.spec_from_file_location("validate_update_scope", Path(__file__).resolve().parents[1] / "scripts" / "validate_update_scope.py")
assert _SPEC is not None and _SPEC.loader is not None
validate_update_scope = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = validate_update_scope
_SPEC.loader.exec_module(validate_update_scope)

BASE = "0" * 40


def _patch_for(paths: list[tuple[str, str]]) -> str:
    chunks: list[str] = []
    for path, status in paths:
        if status == "D":
            chunks.append(
                f"diff --git a/{path} b/{path}\n"
                "deleted file mode 100644\n"
                "index 1111111..0000000\n"
                f"--- a/{path}\n"
                "+++ /dev/null\n"
                "@@ -1 +0,0 @@\n"
                "-old\n"
            )
        else:
            chunks.append(
                f"diff --git a/{path} b/{path}\n"
                + ("new file mode 100644\nindex 0000000..1111111\n" if status == "A" else "index 1111111..2222222 100644\n")
                + ("--- /dev/null\n" if status == "A" else f"--- a/{path}\n")
                + f"+++ b/{path}\n"
                + "@@ -0,0 +1 @@\n"
                + "+new\n"
            )
    return "".join(chunks)


def _summary(
    *,
    task_type: str,
    claim: str,
    files: list[str],
    control_plane: str = "no",
    classification: str = "not_applicable",
) -> str:
    bullets = "\n".join(f"- `{path}`" for path in files)
    return f"""# Test Update

Task type: {task_type}
Claim or experiment ID: {claim}
User-requested scope: test scope only
First-failure classification: {classification}
Control-plane touched: {control_plane}

## Modified files

{bullets}
"""


def _manifest(files: list[tuple[str, str]]) -> dict[str, object]:
    changed_files = []
    for path, status in files:
        row: dict[str, object] = {"path": path, "status": status}
        if status != "D":
            row.update({"sha256": _hash_text("new\n"), "size_bytes": 4, "git_mode": "100644"})
        changed_files.append(row)
    return {
        "schema_version": 1,
        "package_format": "bundle-backed-v1",
        "base_commit": BASE,
        "patch_commit": "1" * 40,
        "changed_files": changed_files,
        "files": [],
    }


def _hash_text(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode()).hexdigest()


def _make_root(
    tmp_path: Path,
    files: list[tuple[str, str]],
    *,
    task_type: str = "bug_fix",
    claim: str = "GOV-CODE-SCOPE-GATE-01",
    summary_files: list[str] | None = None,
    control_plane: str = "no",
    classification: str = "not_applicable",
    include_modified: bool = True,
) -> Path:
    root = tmp_path / "package"
    root.mkdir()
    (root / "BASE_COMMIT.txt").write_text(BASE + "\n")
    (root / "update.patch").write_text(_patch_for(files))
    summary_paths = summary_files if summary_files is not None else [path for path, _ in files]
    (root / "CHANGE_SUMMARY.md").write_text(
        _summary(
            task_type=task_type,
            claim=claim,
            files=summary_paths,
            control_plane=control_plane,
            classification=classification,
        )
    )
    (root / "UPDATE_PACKAGE_MANIFEST.json").write_text(json.dumps(_manifest(files)))
    modified = root / "modified_files"
    modified.mkdir()
    if include_modified:
        for path, status in files:
            if status == "D":
                continue
            target = modified / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("new\n")
    return root


def test_doc_update_only_docs_passes(tmp_path: Path) -> None:
    root = _make_root(
        tmp_path,
        [("docs/example.md", "A")],
        task_type="doc_update",
        claim="GOV-CODE-SCOPE-GATE-01",
    )
    report = validate_update_scope.validate_package(root)
    assert report.status == "PASS"
    assert report.errors == []


def test_doc_update_touching_updater_fails(tmp_path: Path) -> None:
    root = _make_root(
        tmp_path,
        [("tools/drpo-update/drpo_update.py", "M")],
        task_type="doc_update",
        claim="GOV-CODE-SCOPE-GATE-01",
    )
    report = validate_update_scope.validate_package(root)
    assert report.status == "FAIL"
    assert any("control-plane" in error or "outside allowed" in error for error in report.errors)


def test_bug_fix_code_and_test_passes(tmp_path: Path) -> None:
    root = _make_root(
        tmp_path,
        [("scripts/example_bug.py", "M"), ("tests/test_example_bug.py", "A")],
        task_type="bug_fix",
        claim="EXT-H-E7-BENCH-01",
        classification="direct_root_cause",
    )
    report = validate_update_scope.validate_package(root)
    assert report.status == "PASS"


def test_bug_fix_touching_handoff_delta_fails_without_authorization(tmp_path: Path) -> None:
    root = _make_root(
        tmp_path,
        [("docs/handoff_deltas/X/HANDOFF_DELTA.yaml", "A")],
        task_type="bug_fix",
        claim="EXT-H-E7-BENCH-01",
        classification="direct_root_cause",
    )
    report = validate_update_scope.validate_package(root)
    assert report.status == "FAIL"
    assert any("control-plane" in error for error in report.errors)


def test_summary_modified_files_mismatch_fails(tmp_path: Path) -> None:
    root = _make_root(
        tmp_path,
        [("scripts/right.py", "M")],
        task_type="bug_fix",
        claim="GOV-CODE-SCOPE-GATE-01",
        summary_files=["scripts/wrong.py"],
        classification="direct_root_cause",
    )
    report = validate_update_scope.validate_package(root)
    assert report.status == "FAIL"
    assert any("CHANGE_SUMMARY.md modified-file list" in error for error in report.errors)


def test_missing_modified_files_after_image_fails(tmp_path: Path) -> None:
    root = _make_root(
        tmp_path,
        [("scripts/missing.py", "A")],
        task_type="bug_fix",
        claim="GOV-CODE-SCOPE-GATE-01",
        classification="direct_root_cause",
        include_modified=False,
    )
    report = validate_update_scope.validate_package(root)
    assert report.status == "FAIL"
    assert any("modified_files inventory mismatch" in error for error in report.errors)


def test_experiment_code_requires_claim_or_experiment_id(tmp_path: Path) -> None:
    root = _make_root(
        tmp_path,
        [("experiments/example.py", "A")],
        task_type="experiment_code",
        claim="",
    )
    report = validate_update_scope.validate_package(root)
    assert report.status == "FAIL"
    assert any("claim_or_experiment_id" in error for error in report.errors)


def test_governance_control_plane_allows_explicit_control_path(tmp_path: Path) -> None:
    root = _make_root(
        tmp_path,
        [("docs/handoff_deltas/GOV-EXAMPLE/HANDOFF_DELTA.yaml", "A")],
        task_type="governance_control_plane",
        claim="GOV-CODE-SCOPE-GATE-01",
        control_plane="yes",
        classification="governance_control_plane_authorized",
    )
    report = validate_update_scope.validate_package(root)
    assert report.status == "PASS"


def test_zip_package_input_is_supported(tmp_path: Path) -> None:
    root = _make_root(
        tmp_path,
        [("docs/example.md", "A")],
        task_type="doc_update",
        claim="GOV-CODE-SCOPE-GATE-01",
    )
    package = tmp_path / "package.zip"
    with zipfile.ZipFile(package, "w") as archive:
        for path in root.rglob("*"):
            if path.is_file():
                info = zipfile.ZipInfo(path.relative_to(root).as_posix())
                info.create_system = 3
                mode = stat.S_IMODE(path.stat().st_mode)
                info.external_attr = (stat.S_IFREG | mode) << 16
                archive.writestr(info, path.read_bytes())
    temp = validate_update_scope.extract_zip_safe(package)
    try:
        report = validate_update_scope.validate_package(temp)
    finally:
        import shutil

        shutil.rmtree(temp, ignore_errors=True)
    assert report.status == "PASS"
