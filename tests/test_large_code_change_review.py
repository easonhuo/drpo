import importlib.util
import json
import subprocess
import sys
from pathlib import Path

MODULE = Path(__file__).resolve().parents[1] / "scripts/validate_large_code_change_review.py"
sys.path.insert(0, str(MODULE.parent))
SPEC = importlib.util.spec_from_file_location("large_review", MODULE)
review = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(review)


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, text=True, capture_output=True, check=True
    ).stdout.strip()


def make_repo(tmp_path: Path) -> tuple[Path, str]:
    git(tmp_path, "init")
    git(tmp_path, "config", "user.email", "test@example.com")
    git(tmp_path, "config", "user.name", "Test")
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "src/base.py").write_text("def helper(x):\n    return x + 1\n")
    (tmp_path / "tests/test_base.py").write_text("def test_base():\n    assert True\n")
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-m", "base")
    return tmp_path, git(tmp_path, "rev-parse", "HEAD")


def body(changed: dict, new_files: dict) -> str:
    data = {
        "why_over_100": "The complete requested behavior needs implementation and regression evidence while preserving all existing public behavior.",
        "smaller_alternative": "A smaller patch was considered by changing only configuration, but it cannot provide the requested executable behavior and checks.",
        "why_smaller_fails": "That smaller alternative omits the required implementation path and therefore would pass the size limit by dropping core functionality.",
        "changed_python_files": changed,
        "nearest_existing": ["src/base.py"],
        "new_python_files": new_files,
        "core_requirements": [{"id": "R1", "statement": "Preserve helper and add behavior", "tests": ["tests/test_base.py::test_base"]}],
        "reuse_evidence": [{"path": "src/base.py", "symbol": "helper", "used_by": "src/base.py"}],
    }
    return review.START + "\n" + json.dumps(data) + "\n" + review.END


def test_valid_review_passes(tmp_path: Path) -> None:
    repo, base = make_repo(tmp_path)
    (repo / "src/base.py").write_text("def helper(x):\n    return x + 2\n")
    (repo / "tests/test_base.py").write_text("def test_base():\n    assert True\n\ndef test_new():\n    assert True\n")
    git(repo, "add", ".")
    git(repo, "commit", "-m", "change")
    changed = {"src/base.py": "Implement behavior while retaining helper.", "tests/test_base.py": "Cover preserved and new behavior."}
    assert review.validate(repo, base, git(repo, "rev-parse", "HEAD"), body(changed, {})) == []


def test_missing_review_is_rejected(tmp_path: Path) -> None:
    repo, base = make_repo(tmp_path)
    assert review.validate(repo, base, base, "")


def test_public_symbol_removal_is_rejected(tmp_path: Path) -> None:
    repo, base = make_repo(tmp_path)
    (repo / "src/base.py").write_text("VALUE = 1\n")
    (repo / "tests/test_base.py").write_text("def test_base():\n    assert True\n\ndef test_new():\n    assert True\n")
    git(repo, "add", ".")
    git(repo, "commit", "-m", "remove")
    changed = {"src/base.py": "Replace implementation.", "tests/test_base.py": "Cover behavior."}
    errors = review.validate(repo, base, git(repo, "rev-parse", "HEAD"), body(changed, {}))
    assert any("public symbols removed" in error for error in errors)


def test_fake_reuse_evidence_is_rejected(tmp_path: Path) -> None:
    repo, base = make_repo(tmp_path)
    (repo / "src/base.py").write_text("def helper(x):\n    return x + 2\n")
    (repo / "tests/test_base.py").write_text("def test_base():\n    assert True\n\ndef test_new():\n    assert True\n")
    git(repo, "add", ".")
    git(repo, "commit", "-m", "change")
    changed = {"src/base.py": "Implement behavior while retaining helper.", "tests/test_base.py": "Cover behavior."}
    errors = review.validate(repo, base, git(repo, "rev-parse", "HEAD"), body(changed, {}).replace('"helper"', '"invented"'))
    assert any("reused symbol is absent" in error for error in errors)


def test_workflow_preserves_small_change_path() -> None:
    text = (MODULE.parents[1] / ".github/workflows/code-change-budget.yml").read_text()
    assert "churn > 100 || structural > 0" in text
    assert "validate_large_code_change_review.py" in text
    assert "environment: large-code-change-approval" in text
