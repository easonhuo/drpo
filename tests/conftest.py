"""Ensure pytest imports DRPO modules from the checkout under test.

The local ``drpo-update`` integration gate runs tests in an isolated worktree
while reusing the repository virtual environment.  An editable installation in
that environment can still point at the untouched main worktree.  Put the
candidate worktree's ``src/drpo`` first in the package search path before test
collection imports project modules.
"""

from __future__ import annotations

import atexit
import importlib
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
PACKAGE_ROOT = (SRC_ROOT / "drpo").resolve()
src_root = str(SRC_ROOT)
package_root = str(PACKAGE_ROOT)
if sys.path[0] != src_root:
    if src_root in sys.path:
        sys.path.remove(src_root)
    sys.path.insert(0, src_root)

# A regular ``drpo`` package from a stale editable installation wins over the
# checkout's implicit namespace package even when ``src`` is first on sys.path.
# Import the parent package, then explicitly prepend the candidate submodule
# directory.  This also works when the parent package was imported by a plugin.
package = importlib.import_module("drpo")
old_paths = [str(item) for item in getattr(package, "__path__", [])]
package.__path__ = [package_root, *[item for item in old_paths if item != package_root]]
if package.__spec__ is not None:
    package.__spec__.submodule_search_locations = package.__path__

# Drop project submodules already loaded from outside the candidate worktree so
# subsequent collection imports resolve through the path above.
for module_name, module in list(sys.modules.items()):
    if not module_name.startswith("drpo."):
        continue
    module_file = getattr(module, "__file__", None)
    if module_file is None:
        continue
    try:
        Path(module_file).resolve().relative_to(PACKAGE_ROOT)
    except ValueError:
        del sys.modules[module_name]

importlib.invalidate_caches()

# Stage 4A/4B acceptance artifacts were frozen before Stage 5 switched the
# research handoff to schema-v3 delta authority. Their historical tests must
# continue to validate the accepted snapshot rather than reinterpret the
# frozen inventories/semantic graph/lossless candidate against each new
# materialized handoff. Build one isolated historical fixture for those tests
# while all current-source tests continue to run against the integration tree.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_STAGE4_FROZEN_REGISTRY_COMMIT = "828b7db5fcdf8ee7ad9b0d87693955081e39c27e"
_STAGE4_HISTORICAL_TEST_FILES = {
    "tests/test_stage4a_inventory.py",
    "tests/test_stage4_semantic_graph.py",
    "tests/test_stage4a_acceptance.py",
    "tests/test_stage4b_candidate.py",
}
_stage4_fixture_root: Path | None = None
_stage4_fixture_temp: Path | None = None


def _delta_authority_active() -> bool:
    authority = _REPO_ROOT / "docs/handoff_versions/AUTHORITY.yaml"
    if not authority.is_file():
        return False
    for line in authority.read_text(encoding="utf-8").splitlines():
        if line.strip() == "mode: delta":
            return True
    return False


def _prepare_stage4_historical_fixture() -> Path:
    global _stage4_fixture_root, _stage4_fixture_temp
    if _stage4_fixture_root is not None:
        return _stage4_fixture_root

    _stage4_fixture_temp = Path(tempfile.mkdtemp(prefix="drpo-stage4-frozen-"))
    fixture = _stage4_fixture_temp / "repo"
    shutil.copytree(
        _REPO_ROOT,
        fixture,
        ignore=shutil.ignore_patterns(
            ".git",
            "__pycache__",
            ".pytest_cache",
            ".ruff_cache",
            "*.pyc",
            "outputs",
            "wandb",
            ".venv",
            "venv",
        ),
    )

    compat = (
        fixture
        / "docs/handoff_shadow/stage4/candidate/generated/generated/handoff_compat.md"
    )
    shutil.copy2(compat, fixture / "docs/handoff.md")

    registry = subprocess.run(
        [
            "git",
            "-C",
            str(_REPO_ROOT),
            "show",
            f"{_STAGE4_FROZEN_REGISTRY_COMMIT}:experiments/registry.yaml",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if registry.returncode != 0:
        raise RuntimeError(
            "cannot reconstruct the accepted Stage 4 registry fixture: "
            + registry.stderr.decode(errors="replace")
        )
    (fixture / "experiments/registry.yaml").write_bytes(registry.stdout)

    refresh = subprocess.run(
        [
            sys.executable,
            str(fixture / "scripts/build_stage4_context.py"),
            "--repo-root",
            str(fixture),
            "build",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if refresh.returncode != 0:
        raise RuntimeError(
            "cannot refresh the accepted Stage 4 minimal-context fixture: "
            + (refresh.stderr or refresh.stdout)
        )

    _stage4_fixture_root = fixture
    return fixture


def _cleanup_stage4_historical_fixture() -> None:
    if _stage4_fixture_temp is not None:
        shutil.rmtree(_stage4_fixture_temp, ignore_errors=True)


atexit.register(_cleanup_stage4_historical_fixture)


def pytest_configure(config) -> None:  # type: ignore[no-untyped-def]
    config.addinivalue_line(
        "markers",
        "paper_pipeline: paper pipeline opt-in tests; set "
        "DRPO_RUN_PAPER_PIPELINE_TESTS=1 to execute them explicitly",
    )


def pytest_collection_modifyitems(config, items) -> None:  # type: ignore[no-untyped-def]
    if not _delta_authority_active():
        return
    relevant = [
        item for item in items if item.nodeid.split("::", 1)[0] in _STAGE4_HISTORICAL_TEST_FILES
    ]
    if not relevant:
        return
    fixture = _prepare_stage4_historical_fixture()
    for item in relevant:
        module = item.module
        file_name = item.nodeid.split("::", 1)[0]
        if file_name == "tests/test_stage4a_inventory.py":
            module.ROOT = fixture
        elif file_name == "tests/test_stage4_semantic_graph.py":
            module.ROOT = fixture
            module.DYNAMIC_ROOT = fixture / "docs/handoff_shadow/stage4/dynamic"
        elif file_name == "tests/test_stage4a_acceptance.py":
            module.REPO_ROOT = fixture
        elif file_name == "tests/test_stage4b_candidate.py":
            module.ROOT = fixture
