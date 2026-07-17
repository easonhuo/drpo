#!/usr/bin/env python3
"""EXT-H-E7-SQEXP-GAE-01 first complete development implementation."""
from __future__ import annotations

from drpo.e7_sqexp_gae_actor_runtime import aggregate_results, train_actor_branch
from drpo.e7_sqexp_gae_contract import *  # noqa: F403
from drpo.e7_sqexp_gae_coordinator import main
from drpo.e7_sqexp_gae_models import *  # noqa: F403
from drpo.e7_sqexp_gae_preparation import train_frozen_critic_and_prepare

__all__ = [name for name in globals() if not name.startswith("_")]

if __name__ == "__main__":
    raise SystemExit(main())
