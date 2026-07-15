"""Explicit per-experiment CPU/GPU resource-pool contracts.

The module is intentionally small: it parses and applies Linux CPU affinity, records
the GPU pool already owned by a workload launcher, and produces a stable identity.
It does not schedule, reserve, preempt, or tune scientific/thread parameters.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence


SCHEMA_VERSION = 1


class ResourcePoolError(RuntimeError):
    """Raised when an explicit resource pool cannot be applied safely."""


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


@dataclasses.dataclass(frozen=True)
class ResourcePool:
    source: str
    inherited_cpu_ids: tuple[int, ...]
    requested_cpu_ids: tuple[int, ...]
    effective_cpu_ids: tuple[int, ...]
    requested_gpu_ids: tuple[str, ...]
    gpu_enforcement: str

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "source": self.source,
            "inherited_cpu_ids": list(self.inherited_cpu_ids),
            "requested_cpu_ids": list(self.requested_cpu_ids),
            "effective_cpu_ids": list(self.effective_cpu_ids),
            "cpu_count": len(self.effective_cpu_ids),
            "requested_gpu_ids": list(self.requested_gpu_ids),
            "gpu_enforcement": self.gpu_enforcement,
        }

    @property
    def pool_digest(self) -> str:
        return hashlib.sha256(
            _canonical_json(self.identity_payload()).encode("utf-8")
        ).hexdigest()

    def as_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["pool_digest"] = self.pool_digest
        return payload


def parse_cpu_pool(value: str) -> tuple[int, ...]:
    """Parse Linux CPU-list syntax such as ``0-3,8,10-11``."""

    text = value.strip()
    if not text:
        raise ResourcePoolError("CPU pool must not be empty")
    result: list[int] = []
    seen: set[int] = set()
    for raw_token in text.split(","):
        token = raw_token.strip()
        if not token:
            raise ResourcePoolError("CPU pool contains an empty token")
        if "-" in token:
            if token.count("-") != 1:
                raise ResourcePoolError(f"malformed CPU range: {token!r}")
            first_raw, last_raw = token.split("-", 1)
            if not first_raw.isdigit() or not last_raw.isdigit():
                raise ResourcePoolError(f"malformed CPU range: {token!r}")
            first = int(first_raw)
            last = int(last_raw)
            if last < first:
                raise ResourcePoolError(f"descending CPU range: {token!r}")
            values = range(first, last + 1)
        else:
            if not token.isdigit():
                raise ResourcePoolError(f"malformed CPU id: {token!r}")
            values = (int(token),)
        for cpu_id in values:
            if cpu_id in seen:
                raise ResourcePoolError(f"duplicate CPU id: {cpu_id}")
            seen.add(cpu_id)
            result.append(cpu_id)
    if not result:
        raise ResourcePoolError("CPU pool must contain at least one CPU")
    return tuple(sorted(result))


def format_cpu_pool(cpu_ids: Sequence[int]) -> str:
    values = tuple(sorted(int(value) for value in cpu_ids))
    if not values:
        raise ResourcePoolError("CPU pool must contain at least one CPU")
    if len(values) != len(set(values)) or values[0] < 0:
        raise ResourcePoolError("CPU ids must be unique non-negative integers")
    ranges: list[str] = []
    start = previous = values[0]
    for value in values[1:]:
        if value == previous + 1:
            previous = value
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = value
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(ranges)


def parse_gpu_pool(value: str) -> tuple[str, ...]:
    text = value.strip()
    if not text:
        raise ResourcePoolError("GPU pool must not be empty")
    result: list[str] = []
    seen: set[str] = set()
    for raw_token in text.split(","):
        token = raw_token.strip()
        negative_integer = token.startswith("-") and token[1:].isdigit()
        if (
            not token
            or negative_integer
            or any(character.isspace() for character in token)
        ):
            raise ResourcePoolError(f"malformed GPU id: {raw_token!r}")
        if token in seen:
            raise ResourcePoolError(f"duplicate GPU id: {token}")
        seen.add(token)
        result.append(token)
    return tuple(result)


def _current_affinity() -> tuple[int, ...]:
    if not hasattr(os, "sched_getaffinity"):
        raise ResourcePoolError("explicit CPU pools require os.sched_getaffinity")
    values = tuple(sorted(int(value) for value in os.sched_getaffinity(0)))
    if not values:
        raise ResourcePoolError("current CPU affinity is empty")
    return values


def activate_resource_pool(
    *,
    cpu_pool: str | None,
    gpu_pool: str | None = None,
    gpu_enforcement: str = "launcher_argument",
    environ: MutableMapping[str, str] | None = None,
) -> ResourcePool:
    """Apply an optional CPU pool and export a stable pool identity.

    ``gpu_enforcement`` is provenance, not a scheduler. ``launcher_argument`` means
    the delegated launcher must enforce its existing GPU argument. ``cuda_visible``
    sets ``CUDA_VISIBLE_DEVICES`` here.
    """

    if gpu_enforcement not in {"none", "launcher_argument", "cuda_visible"}:
        raise ResourcePoolError(f"unsupported GPU enforcement: {gpu_enforcement}")
    inherited = _current_affinity()
    if cpu_pool is None:
        requested = inherited
        source = "inherited_affinity"
    else:
        requested = parse_cpu_pool(cpu_pool)
        unavailable = sorted(set(requested) - set(inherited))
        if unavailable:
            raise ResourcePoolError(
                f"requested CPU ids are outside inherited affinity: {unavailable}"
            )
        if not hasattr(os, "sched_setaffinity"):
            raise ResourcePoolError("explicit CPU pools require os.sched_setaffinity")
        os.sched_setaffinity(0, set(requested))
        source = "explicit_cli"
    effective = _current_affinity()
    if effective != requested:
        raise ResourcePoolError(
            "effective CPU affinity does not exactly match the requested pool"
        )

    gpu_ids = () if gpu_pool is None else parse_gpu_pool(gpu_pool)
    if gpu_ids and gpu_enforcement == "none":
        raise ResourcePoolError("a declared GPU pool requires an enforcement mode")
    if not gpu_ids and gpu_enforcement != "none":
        gpu_enforcement = "none"
    environment = os.environ if environ is None else environ
    if gpu_ids and gpu_enforcement == "cuda_visible":
        environment["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)

    pool = ResourcePool(
        source=source,
        inherited_cpu_ids=inherited,
        requested_cpu_ids=requested,
        effective_cpu_ids=effective,
        requested_gpu_ids=gpu_ids,
        gpu_enforcement=gpu_enforcement,
    )
    environment["DRPO_RESOURCE_POOL_DIGEST"] = pool.pool_digest
    environment["DRPO_CPU_POOL"] = format_cpu_pool(pool.effective_cpu_ids)
    if gpu_ids:
        environment["DRPO_GPU_POOL"] = ",".join(gpu_ids)
    else:
        environment.pop("DRPO_GPU_POOL", None)
    return pool


def command_gpu_pool(command: Sequence[str]) -> tuple[str, ...] | None:
    """Extract one ``--gpus`` value from a delegated command."""

    values: list[str] = []
    index = 0
    while index < len(command):
        token = str(command[index])
        if token == "--gpus":
            if index + 1 >= len(command):
                raise ResourcePoolError("delegated --gpus is missing its value")
            values.append(str(command[index + 1]))
            index += 2
            continue
        if token.startswith("--gpus="):
            values.append(token.split("=", 1)[1])
        index += 1
    if not values:
        return None
    if len(values) != 1:
        raise ResourcePoolError("delegated command contains multiple --gpus values")
    return parse_gpu_pool(values[0])


def validate_delegated_gpu_pool(
    command: Sequence[str],
    pool: ResourcePool,
) -> None:
    if not pool.requested_gpu_ids:
        return
    if pool.gpu_enforcement == "cuda_visible":
        return
    if pool.gpu_enforcement != "launcher_argument":
        raise ResourcePoolError("declared GPU pool is not enforced")
    delegated = command_gpu_pool(command)
    if delegated is None:
        raise ResourcePoolError(
            "declared GPU pool requires an identical delegated --gpus argument"
        )
    if delegated != pool.requested_gpu_ids:
        raise ResourcePoolError(
            "delegated --gpus does not match the declared GPU pool"
        )


def _validate_existing_identity(target: Path, payload: Mapping[str, Any]) -> Path:
    try:
        existing = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        raise ResourcePoolError(
            f"cannot read existing pool identity: {target}"
        ) from exc
    if existing != payload:
        raise ResourcePoolError(
            f"existing resource-pool identity does not match: {target}"
        )
    return target


def write_pool_identity(path: str | Path, pool: ResourcePool) -> Path:
    """Create one immutable resource-pool identity, or validate exact reuse."""

    target = Path(path).resolve()
    payload = pool.as_dict()
    if target.exists():
        return _validate_existing_identity(target, payload)
    target.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    try:
        descriptor = os.open(
            target,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o644,
        )
    except FileExistsError:
        return _validate_existing_identity(target, payload)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            target.unlink()
        except OSError:
            pass
        raise
    return target
