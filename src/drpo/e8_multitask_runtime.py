"""Method-agnostic recovery primitives for E8 multitask experiments."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol


class CellLike(Protocol):
    @property
    def key(self) -> str: ...

    task: str
    method: str
    seed: int
    stage: str


@dataclass(frozen=True)
class MethodAuditResult:
    passed: bool
    failure_bucket: str = "terminal_contract_failures"
    failures: tuple[str, ...] = ()


def stable_json_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def recovery_identity(
    cell: CellLike,
    *,
    experiment_id: str,
    config_hash: str,
    cell_identity_fields: Mapping[str, Any],
    common_identity_fields: Mapping[str, Any],
    schema_version: int = 1,
) -> dict[str, Any]:
    cell_identity: dict[str, Any] = {
        "task": cell.task,
        "method": cell.method,
    }
    collisions = sorted(set(cell_identity).intersection(cell_identity_fields))
    if collisions:
        raise ValueError(
            f"Method cell identity attempted to overwrite: {collisions}"
        )
    cell_identity.update(cell_identity_fields)
    for reserved, expected in (
        ("seed", int(cell.seed)),
        ("stage", cell.stage),
    ):
        if reserved in cell_identity:
            raise ValueError(
                f"Method cell identity attempted to overwrite: ['{reserved}']"
            )
        cell_identity[reserved] = expected

    identity: dict[str, Any] = {
        "schema_version": int(schema_version),
        "experiment_id": experiment_id,
        "config_hash": config_hash,
        "cell": cell_identity,
    }
    collisions = sorted(set(identity).intersection(common_identity_fields))
    if collisions:
        raise ValueError(
            f"Common recovery identity attempted to overwrite: {collisions}"
        )
    identity.update(common_identity_fields)
    identity["identity_hash"] = stable_json_hash(identity)
    return identity
