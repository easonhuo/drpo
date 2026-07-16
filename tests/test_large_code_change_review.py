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
    (tmp_path / "src/base.py").write_text(
        "def helper(x):\n    return x + 1\n\n"
        "def parameter_points():\n    return tuple(range(8))\n\n"
        "def weight(u):\n    return u * u\n"
    )
    (tmp_path / "tests/test_base.py").write_text(
        "from src.base import helper, parameter_points, weight\n\n"
        "def test_base():\n    assert helper(1) == 2\n\n"
        "def test_matrix_has_8_points():\n    assert len(parameter_points()) == 8\n\n"
        "def test_weight_is_squared():\n    assert weight(3) == 9\n"
    )
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-m", "base")
    return tmp_path, git(tmp_path, "rev-parse", "HEAD")


def body(changed: dict, new_files: dict, *, tests=None, replacements=None) -> str:
    data = {
        "why_over_100": (
            "The complete requested behavior needs implementation and regression evidence "
            "while preserving all existing public behavior."
        ),
        "smaller_alternative": (
            "A smaller patch was considered by changing only configuration, but it cannot "
            "provide the requested executable behavior and checks."
        ),
        "why_smaller_fails": (
            "That smaller alternative omits the required implementation path and therefore "
            "would pass the size limit by dropping core functionality."
        ),
        "changed_python_files": changed,
        "nearest_existing": ["src/base.py"],
        "new_python_files": new_files,
        "core_requirements": [
            {
                "id": "R1",
                "statement": "Preserve helper and add behavior",
                "tests": tests or ["tests/test_base.py::test_base"],
            }
        ],
        "reuse_evidence": [
            {"path": "src/base.py", "symbol": "helper", "used_by": "src/base.py"}
        ],
        "test_replacements": replacements or {},
    }
    return review.START + "\n" + json.dumps(data) + "\n" + review.END


def commit_change(repo: Path, message: str = "change") -> str:
    git(repo, "add", ".")
    git(repo, "commit", "-m", message)
    return git(repo, "rev-parse", "HEAD")


def test_valid_additive_review_passes(tmp_path: Path) -> None:
    repo, base = make_repo(tmp_path)
    (repo / "src/base.py").write_text(
        "def helper(x):\n    return x + 2\n\n"
        "def parameter_points():\n    return tuple(range(8))\n\n"
        "def weight(u):\n    return u * u\n"
    )
    with (repo / "tests/test_base.py").open("a") as handle:
        handle.write("\ndef test_new():\n    assert helper(2) == 4\n")
    head = commit_change(repo)
    changed = {
        "src/base.py": "Implement behavior while retaining helper.",
        "tests/test_base.py": "Cover preserved and new behavior.",
    }
    assert review.validate(repo, base, head, body(changed, {})) == []


def test_missing_review_is_rejected(tmp_path: Path) -> None:
    repo, base = make_repo(tmp_path)
    assert review.validate(repo, base, base, "")


def test_public_symbol_removal_is_rejected(tmp_path: Path) -> None:
    repo, base = make_repo(tmp_path)
    (repo / "src/base.py").write_text("VALUE = 1\n")
    with (repo / "tests/test_base.py").open("a") as handle:
        handle.write("\ndef test_new():\n    assert VALUE == 1\n")
    head = commit_change(repo, "remove")
    changed = {
        "src/base.py": "Replace implementation.",
        "tests/test_base.py": "Cover behavior.",
    }
    errors = review.validate(repo, base, head, body(changed, {}))
    assert any("public symbols removed" in error for error in errors)


def test_fake_reuse_evidence_is_rejected(tmp_path: Path) -> None:
    repo, base = make_repo(tmp_path)
    (repo / "src/base.py").write_text(
        (repo / "src/base.py").read_text().replace("x + 1", "x + 2")
    )
    with (repo / "tests/test_base.py").open("a") as handle:
        handle.write("\ndef test_new():\n    assert helper(2) == 4\n")
    head = commit_change(repo)
    changed = {
        "src/base.py": "Implement behavior while retaining helper.",
        "tests/test_base.py": "Cover behavior.",
    }
    errors = review.validate(
        repo, base, head, body(changed, {}).replace('"helper"', '"invented"')
    )
    assert any("reused symbol is not defined" in error for error in errors)


def test_legitimate_same_name_rewrites_pass(tmp_path: Path) -> None:
    repo, base = make_repo(tmp_path)
    source = (
        (repo / "src/base.py")
        .read_text()
        .replace("range(8)", "range(16)")
        .replace("u * u", "u")
    )
    (repo / "src/base.py").write_text(source)
    (repo / "tests/test_base.py").write_text(
        "from src.base import helper, parameter_points, weight\n\n"
        "def test_base():\n    assert helper(1) == 2\n\n"
        "def test_matrix_has_8_points():\n    assert len(parameter_points()) == 16\n\n"
        "def test_weight_is_squared():\n    assert weight(3) == 3\n"
    )
    head = commit_change(repo, "repair")
    changed = {
        "src/base.py": "Modify the existing implementation in place.",
        "tests/test_base.py": "Update matrix and formula regression coverage.",
    }
    replacements = {
        "tests/test_base.py::test_matrix_has_8_points": {
            "reason": (
                "The approved matrix expands from eight to sixteen points, so the old "
                "expected cardinality is obsolete."
            ),
            "preserved_behavior": (
                "The test still verifies the exact parameter-point cardinality produced "
                "by the existing builder."
            ),
            "replacements": ["tests/test_base.py::test_matrix_has_8_points"],
            "anchors": ["parameter_points"],
        },
        "tests/test_base.py::test_weight_is_squared": {
            "reason": (
                "The approved formula removes the extra square, so the prior squared "
                "numerical expectation must change."
            ),
            "preserved_behavior": (
                "The test still directly verifies the production weight function at a "
                "fixed deterministic input."
            ),
            "replacements": ["tests/test_base.py::test_weight_is_squared"],
            "anchors": ["weight"],
        },
    }
    tests = [
        "tests/test_base.py::test_base",
        "tests/test_base.py::test_matrix_has_8_points",
        "tests/test_base.py::test_weight_is_squared",
    ]
    assert (
        review.validate(
            repo,
            base,
            head,
            body(changed, {}, tests=tests, replacements=replacements),
        )
        == []
    )


def test_legitimate_renamed_replacement_passes(tmp_path: Path) -> None:
    repo, base = make_repo(tmp_path)
    (repo / "src/base.py").write_text(
        (repo / "src/base.py").read_text().replace("range(8)", "range(16)")
    )
    (repo / "tests/test_base.py").write_text(
        "from src.base import helper, parameter_points, weight\n\n"
        "def test_base():\n    assert helper(1) == 2\n\n"
        "def test_linear_matrix_has_16_points():\n"
        "    assert len(parameter_points()) == 16\n\n"
        "def test_weight_is_squared():\n    assert weight(3) == 9\n"
    )
    head = commit_change(repo, "rename")
    changed = {
        "src/base.py": "Modify the existing implementation in place.",
        "tests/test_base.py": "Replace obsolete matrix coverage.",
    }
    replacements = {
        "tests/test_base.py::test_matrix_has_8_points": {
            "reason": (
                "The approved matrix expands and the old test name states an obsolete "
                "fixed cardinality."
            ),
            "preserved_behavior": (
                "The replacement retains direct cardinality coverage over the same "
                "parameter-point builder."
            ),
            "replacements": ["tests/test_base.py::test_linear_matrix_has_16_points"],
            "anchors": ["parameter_points"],
        }
    }
    tests = [
        "tests/test_base.py::test_base",
        "tests/test_base.py::test_linear_matrix_has_16_points",
    ]
    assert (
        review.validate(
            repo,
            base,
            head,
            body(changed, {}, tests=tests, replacements=replacements),
        )
        == []
    )


def test_declared_trivial_rewrite_is_rejected(tmp_path: Path) -> None:
    repo, base = make_repo(tmp_path)
    (repo / "src/base.py").write_text(
        (repo / "src/base.py").read_text().replace("x + 1", "x + 2")
    )
    (repo / "tests/test_base.py").write_text(
        "from src.base import helper, parameter_points, weight\n\n"
        "def test_base():\n    assert True\n\n"
        "def test_matrix_has_8_points():\n    assert len(parameter_points()) == 8\n\n"
        "def test_weight_is_squared():\n    assert weight(3) == 9\n"
    )
    head = commit_change(repo, "weaken")
    changed = {
        "src/base.py": "Change helper behavior.",
        "tests/test_base.py": "Claim replacement coverage.",
    }
    replacements = {
        "tests/test_base.py::test_base": {
            "reason": (
                "The helper changed and this declaration attempts to authorize "
                "corresponding test maintenance."
            ),
            "preserved_behavior": (
                "The replacement is claimed to keep direct deterministic helper coverage."
            ),
            "replacements": ["tests/test_base.py::test_base"],
            "anchors": ["helper"],
        }
    }
    errors = review.validate(
        repo, base, head, body(changed, {}, replacements=replacements)
    )
    assert any(
        "trivial assertion" in error or "anchor continuity" in error
        for error in errors
    )


def test_rewrite_without_exact_replacement_evidence_is_rejected(tmp_path: Path) -> None:
    repo, base = make_repo(tmp_path)
    (repo / "src/base.py").write_text(
        (repo / "src/base.py").read_text().replace("range(8)", "range(16)")
    )
    (repo / "tests/test_base.py").write_text(
        (repo / "tests/test_base.py").read_text().replace("== 8", "== 16")
    )
    head = commit_change(repo)
    changed = {
        "src/base.py": "Modify matrix.",
        "tests/test_base.py": "Update matrix test.",
    }
    errors = review.validate(repo, base, head, body(changed, {}))
    assert any("lack replacement evidence" in error for error in errors)


def test_unchanged_existing_test_cannot_be_reused_as_replacement(
    tmp_path: Path,
) -> None:
    repo, base = make_repo(tmp_path)
    (repo / "src/base.py").write_text(
        (repo / "src/base.py").read_text().replace("range(8)", "range(16)")
    )
    (repo / "tests/test_base.py").write_text(
        (repo / "tests/test_base.py")
        .read_text()
        .replace(
            "def test_matrix_has_8_points():\n"
            "    assert len(parameter_points()) == 8\n\n",
            "",
        )
    )
    head = commit_change(repo)
    changed = {
        "src/base.py": "Modify matrix.",
        "tests/test_base.py": "Remove obsolete matrix test.",
    }
    replacements = {
        "tests/test_base.py::test_matrix_has_8_points": {
            "reason": (
                "The obsolete matrix test is declared as replaced by another existing test."
            ),
            "preserved_behavior": (
                "The declaration claims that unchanged coverage is sufficient for the "
                "new matrix."
            ),
            "replacements": ["tests/test_base.py::test_base"],
            "anchors": [{"before": "parameter_points", "after": "helper"}],
        }
    }
    tests = ["tests/test_base.py::test_base"]
    errors = review.validate(
        repo,
        base,
        head,
        body(changed, {}, tests=tests, replacements=replacements),
    )
    assert any("not new or changed" in error for error in errors)


def test_copy_heavy_attempt_is_rejected(tmp_path: Path) -> None:
    repo, base = make_repo(tmp_path)
    (repo / "src/copied.py").write_text((repo / "src/base.py").read_text())
    (repo / "tests/test_copied.py").write_text(
        "from src.copied import helper\n\n"
        "def test_copy():\n    assert helper(1) == 2\n"
    )
    head = commit_change(repo, "copy stack")
    changed = {
        "src/copied.py": "Parallel copied implementation.",
        "tests/test_copied.py": "Coverage for copied implementation.",
    }
    new_files = {
        "src/copied.py": {
            "closest_existing": "src/base.py",
            "why_not_extend": (
                "A separate module is claimed necessary even though it duplicates the "
                "existing implementation and should be rejected."
            ),
        },
        "tests/test_copied.py": {
            "closest_existing": "tests/test_base.py",
            "why_not_extend": (
                "A separate test file is claimed necessary for the duplicated "
                "implementation and should be rejected."
            ),
        },
    }
    data = body(changed, new_files, tests=["tests/test_copied.py::test_copy"])
    data = data.replace(
        '"reuse_evidence": [{"path": "src/base.py", "symbol": "helper", '
        '"used_by": "src/base.py"}]',
        '"reuse_evidence": [{"path": "src/base.py", "symbol": "helper", '
        '"used_by": "src/copied.py"}]',
    )
    errors = review.validate(repo, base, head, data)
    assert any("copy detected" in error for error in errors)


def test_declared_test_must_exist(tmp_path: Path) -> None:
    repo, base = make_repo(tmp_path)
    (repo / "src/base.py").write_text(
        (repo / "src/base.py").read_text().replace("x + 1", "x + 2")
    )
    with (repo / "tests/test_base.py").open("a") as handle:
        handle.write("\ndef test_new():\n    assert helper(2) == 4\n")
    head = commit_change(repo)
    changed = {
        "src/base.py": "Implement behavior while retaining helper.",
        "tests/test_base.py": "Cover behavior.",
    }
    invalid = body(changed, {}, tests=["tests/test_base.py::test_missing"])
    errors = review.validate(repo, base, head, invalid)
    assert any("declared core-function test does not exist" in error for error in errors)


def test_replacement_with_fewer_checks_is_rejected(tmp_path: Path) -> None:
    repo, base = make_repo(tmp_path)
    original = (repo / "tests/test_base.py").read_text()
    original = original.replace(
        "def test_base():\n    assert helper(1) == 2\n",
        "def test_base():\n"
        "    assert helper(1) == 2\n"
        "    assert helper(2) == 3\n",
    )
    (repo / "tests/test_base.py").write_text(original)
    git(repo, "add", ".")
    git(repo, "commit", "--amend", "--no-edit")
    base = git(repo, "rev-parse", "HEAD")
    (repo / "src/base.py").write_text(
        (repo / "src/base.py").read_text().replace("x + 1", "x + 2")
    )
    (repo / "tests/test_base.py").write_text(
        (repo / "tests/test_base.py")
        .read_text()
        .replace(
            "    assert helper(1) == 2\n    assert helper(2) == 3\n",
            "    assert helper(1) == 3\n",
        )
    )
    head = commit_change(repo, "reduce checks")
    changed = {
        "src/base.py": "Change helper behavior.",
        "tests/test_base.py": "Update helper regression.",
    }
    replacements = {
        "tests/test_base.py::test_base": {
            "reason": (
                "The helper behavior changed, so the deterministic expectations require "
                "an approved update."
            ),
            "preserved_behavior": (
                "The test must retain direct deterministic checks of the same helper function."
            ),
            "replacements": ["tests/test_base.py::test_base"],
            "anchors": ["helper"],
        }
    }
    errors = review.validate(
        repo, base, head, body(changed, {}, replacements=replacements)
    )
    assert any("weakens executable checks" in error for error in errors)


def test_replacement_must_be_declared_as_core_evidence(tmp_path: Path) -> None:
    repo, base = make_repo(tmp_path)
    (repo / "src/base.py").write_text(
        (repo / "src/base.py").read_text().replace("range(8)", "range(16)")
    )
    (repo / "tests/test_base.py").write_text(
        (repo / "tests/test_base.py").read_text().replace("== 8", "== 16")
    )
    head = commit_change(repo, "matrix")
    changed = {
        "src/base.py": "Modify matrix.",
        "tests/test_base.py": "Update matrix test.",
    }
    replacements = {
        "tests/test_base.py::test_matrix_has_8_points": {
            "reason": (
                "The approved matrix expands from eight to sixteen points, changing the "
                "exact expectation."
            ),
            "preserved_behavior": (
                "The test retains direct cardinality coverage over the same "
                "parameter-point builder."
            ),
            "replacements": ["tests/test_base.py::test_matrix_has_8_points"],
            "anchors": ["parameter_points"],
        }
    }
    errors = review.validate(
        repo, base, head, body(changed, {}, replacements=replacements)
    )
    assert any("not core evidence" in error for error in errors)


def test_workflow_preserves_small_change_path() -> None:
    text = (MODULE.parents[1] / ".github/workflows/code-change-budget.yml").read_text()
    assert "churn > 100 || structural > 0" in text
    assert "validate_large_code_change_review.py" in text
    assert "environment: large-code-change-approval" in text
