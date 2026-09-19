from __future__ import annotations

import os
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
for source in (ROOT / "paper_code" / "src", ROOT / "src"):
    location = str(source)
    if location not in sys.path:
        sys.path.insert(0, location)

# Match the repository's deterministic CPU policy and avoid pathological thread
# oversubscription in clean-checkout test environments.
torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass
