"""Single source of truth for D4RL reference scores.

Imports REF_MIN_SCORE / REF_MAX_SCORE directly from /root/d4rl/refs/d4rl_infos.py
(vendored copy of the upstream d4rl/infos.py), so they are guaranteed to stay
aligned with the official code base at all times.

Tuple convention in this package is (min, max) — matches official naming
(REF_MIN_SCORE, REF_MAX_SCORE) and matches the IQL / BPPO paper convention
    normalized = (raw - min) / (max - min) * 100
"""
import os
import sys
import importlib.util

# --- Load the official refs file as a private module. The `refs/` dir is
# vendored inside this repo (see exp_log 2026-05-08 D4RL_REF alignment),
# committed alongside this file. We import by path to avoid polluting top-level
# sys.path.
_REFS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          'refs', 'd4rl_infos.py')
if not os.path.isfile(_REFS_PATH):
    raise FileNotFoundError(
        f"Official D4RL infos.py not found at {_REFS_PATH}. "
        "This file is required as the single source of truth — do NOT replace "
        "it with hand-edited numbers. Re-download from "
        "https://github.com/Farama-Foundation/D4RL/blob/master/d4rl/infos.py"
    )

_spec = importlib.util.spec_from_file_location("_d4rl_infos_official", _REFS_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

REF_MIN_SCORE = dict(_mod.REF_MIN_SCORE)
REF_MAX_SCORE = dict(_mod.REF_MAX_SCORE)


def get_ref(dataset_name):
    """Return (min, max) reference scores for `dataset_name`.

    Raises KeyError for unknown datasets — callers that want graceful NaN
    behaviour should wrap in try/except or use normalize_score() which returns
    NaN for unknown keys.
    """
    return (REF_MIN_SCORE[dataset_name], REF_MAX_SCORE[dataset_name])


def normalize_score(dataset_name, raw):
    """D4RL-normalized score in [0, 100]-ish (can exceed 100 for super-expert).

    Formula (official IQL / BPPO / D4RL paper):
        100 * (raw - min) / (max - min)

    Returns NaN for unknown `dataset_name`, consistent with
    eval_d4rl_mp_greedy.py's historical behaviour.
    """
    if dataset_name not in REF_MIN_SCORE or dataset_name not in REF_MAX_SCORE:
        return float('nan')
    lo = REF_MIN_SCORE[dataset_name]
    hi = REF_MAX_SCORE[dataset_name]
    return (raw - lo) / (hi - lo) * 100.0


# Convenience dict (min, max) for code that still wants inline lookup style.
# NOTE: order is (min, max) — officially named (REF_MIN_SCORE, REF_MAX_SCORE).
# Don't reverse this; callers rely on rand/lo first, expert/hi second.
D4RL_REF = {k: (REF_MIN_SCORE[k], REF_MAX_SCORE[k]) for k in REF_MIN_SCORE
            if k in REF_MAX_SCORE}
