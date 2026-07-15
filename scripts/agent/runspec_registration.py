#!/usr/bin/env python3
"""Registration-timing policy for RunSpec execution."""
from __future__ import annotations

import re
from typing import Any

from runspec_lib import RunSpecError

PRE_REGISTERED = "pre_registered"
DEFERRED = "deferred"
REGISTRATION_MODES = {PRE_REGISTERED, DEFERRED}
_FULL_COMMIT_RE = re.compile(r"^[0-9a-fA-F]{40}$")


def validate_registration_block(spec: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize the optional RunSpec registration block.

    Missing blocks preserve the historical pre-registered behavior. Deferred mode
    changes only registration timing: it does not alter the RunSpec's scientific or
    execution class, and it requires an immutable full commit SHA for provenance.
    """

    raw = spec.get("registration")
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise RunSpecError("registration must be a mapping")
    mode = str(raw.get("mode") or PRE_REGISTERED)
    if mode not in REGISTRATION_MODES:
        raise RunSpecError(
            "registration.mode must be pre_registered or deferred"
        )
    closure_required = raw.get("closure_required", mode == DEFERRED)
    if not isinstance(closure_required, bool):
        raise RunSpecError("registration.closure_required must be a boolean")
    if mode == DEFERRED and closure_required is not True:
        raise RunSpecError(
            "registration.closure_required must be true for deferred registration"
        )
    if mode == DEFERRED:
        repo_commit = str(spec.get("repo_commit") or "").strip()
        if not _FULL_COMMIT_RE.fullmatch(repo_commit):
            raise RunSpecError(
                "deferred registration requires repo_commit to be a full 40-character Git SHA"
            )
    normalized = dict(raw)
    normalized["mode"] = mode
    normalized["closure_required"] = closure_required
    return normalized


def registration_requires_registry(spec: dict[str, Any]) -> bool:
    """Return whether normal RunSpec validation must find the experiment in registry."""

    return validate_registration_block(spec)["mode"] == PRE_REGISTERED
