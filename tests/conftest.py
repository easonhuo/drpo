"""Ensure pytest imports DRPO modules from the checkout under test.

The local ``drpo-update`` integration gate runs tests in an isolated worktree
while reusing the repository virtual environment.  An editable installation in
that environment can still point at the untouched main worktree.  Put the
candidate worktree's ``src/drpo`` first in the package search path before test
collection imports project modules.
"""

from __future__ import annotations

import importlib
import sys
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
