"""Paper-facing D4RL-9 task and environment contracts.

This module owns task metadata, dataset identity, rollout identity, and D4RL
reference-score constants. It intentionally contains no actor, critic, trainer,
optimizer, or result-ranking implementation.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterable


ENVIRONMENTS = ("halfcheetah", "hopper", "walker2d")
DATASET_TIERS = ("medium", "medium-replay", "medium-expert")
D4RL_ROLLOUT_BACKEND = "gymnasium_mujoco"

_REFERENCE_SCORES = {
    "halfcheetah": (-280.178953, 12135.0),
    "hopper": (-20.272305, 3234.3),
    "walker2d": (1.629008, 4592.3),
}
_ENV_IDS = {
    "halfcheetah": "HalfCheetah-v4",
    "hopper": "Hopper-v4",
    "walker2d": "Walker2d-v4",
}

_HOPPER_MEDIUM_REPLAY_SHA256 = (
    "e121c5f7c9857a307baa9edc6a2c3b48"
    "e85fedb9ac316ecddd0f48ca7ef4e39b"
)


@dataclass(frozen=True)
class D4RLTaskSpec:
    """One manuscript D4RL locomotion coordinate.

    ``dataset_sha256`` stays ``None`` until the exact source artifact has been
    verified. An unresolved hash is a deliberate fail-closed state, not a
    wildcard.
    """

    task_id: str
    environment: str
    dataset_tier: str
    dataset_id: str
    dataset_basename: str
    env_id: str
    normalized_score_reference_min: float
    normalized_score_reference_max: float
    dataset_sha256: str | None
    provenance_status: str

    def __post_init__(self) -> None:
        if self.environment not in ENVIRONMENTS:
            raise ValueError(
                f"unsupported D4RL environment: {self.environment}"
            )
        if self.dataset_tier not in DATASET_TIERS:
            raise ValueError(
                f"unsupported D4RL dataset tier: {self.dataset_tier}"
            )
        expected_id = f"{self.environment}-{self.dataset_tier}-v2"
        if self.dataset_id != expected_id or self.task_id != expected_id:
            raise ValueError("D4RL task and dataset identities disagree")
        expected_basename = (
            f"{self.environment}_"
            f"{self.dataset_tier.replace('-', '_')}-v2.hdf5"
        )
        if self.dataset_basename != expected_basename:
            raise ValueError(
                "D4RL dataset basename does not match task identity"
            )
        if self.env_id != _ENV_IDS[self.environment]:
            raise ValueError(
                "D4RL Gymnasium environment identity is inconsistent"
            )
        if (
            self.normalized_score_reference_max
            <= self.normalized_score_reference_min
        ):
            raise ValueError(
                "D4RL reference maximum must exceed reference minimum"
            )
        if self.dataset_sha256 is not None:
            if len(self.dataset_sha256) != 64:
                raise ValueError(
                    "dataset SHA-256 must contain 64 hexadecimal digits"
                )
            try:
                int(self.dataset_sha256, 16)
            except ValueError as exc:
                raise ValueError(
                    "dataset SHA-256 must be hexadecimal"
                ) from exc
        expected_status = (
            "verified"
            if self.dataset_sha256 is not None
            else "unresolved"
        )
        if self.provenance_status != expected_status:
            raise ValueError(
                "D4RL provenance status does not match dataset SHA state"
            )

    @property
    def dataset_identity_verified(self) -> bool:
        return self.dataset_sha256 is not None

    @property
    def rollout_backend(self) -> str:
        return D4RL_ROLLOUT_BACKEND

    def normalization_kwargs(self) -> dict[str, float | bool]:
        return {
            "normalized_score_percent": True,
            "reference_min_score": self.normalized_score_reference_min,
            "reference_max_score": self.normalized_score_reference_max,
        }

    def rollout_identity(self) -> dict[str, str]:
        return {
            "backend": self.rollout_backend,
            "dataset_id": self.dataset_id,
            "env_id": self.env_id,
        }

    def validate_rollout_identity(
        self,
        *,
        backend: str,
        dataset_id: str,
        env_id: str,
    ) -> dict[str, str]:
        """Validate task/environment identity without selecting a trainer."""

        expected = self.rollout_identity()
        actual = {
            "backend": str(backend),
            "dataset_id": str(dataset_id),
            "env_id": str(env_id),
        }
        mismatches = {
            key: {
                "expected": expected[key],
                "actual": actual[key],
            }
            for key in expected
            if actual[key] != expected[key]
        }
        if mismatches:
            raise ValueError(
                f"D4RL rollout identity mismatch for {self.task_id}: "
                f"{mismatches}"
            )
        return expected


def _make_task(environment: str, dataset_tier: str) -> D4RLTaskSpec:
    dataset_id = f"{environment}-{dataset_tier}-v2"
    minimum, maximum = _REFERENCE_SCORES[environment]
    digest = (
        _HOPPER_MEDIUM_REPLAY_SHA256
        if dataset_id == "hopper-medium-replay-v2"
        else None
    )
    return D4RLTaskSpec(
        task_id=dataset_id,
        environment=environment,
        dataset_tier=dataset_tier,
        dataset_id=dataset_id,
        dataset_basename=(
            f"{environment}_{dataset_tier.replace('-', '_')}-v2.hdf5"
        ),
        env_id=_ENV_IDS[environment],
        normalized_score_reference_min=minimum,
        normalized_score_reference_max=maximum,
        dataset_sha256=digest,
        provenance_status=(
            "verified" if digest is not None else "unresolved"
        ),
    )


D4RL9_TASKS = tuple(
    _make_task(environment, dataset_tier)
    for dataset_tier in DATASET_TIERS
    for environment in ENVIRONMENTS
)
D4RL9_BY_ID = {task.task_id: task for task in D4RL9_TASKS}


def resolve_d4rl_task(task_id: str) -> D4RLTaskSpec:
    try:
        return D4RL9_BY_ID[task_id]
    except KeyError as exc:
        raise ValueError(
            f"unknown manuscript D4RL task: {task_id}"
        ) from exc


def validate_d4rl9_matrix(
    tasks: Iterable[D4RLTaskSpec],
) -> tuple[D4RLTaskSpec, ...]:
    resolved = tuple(tasks)
    if resolved != D4RL9_TASKS:
        raise ValueError(
            "D4RL-9 tasks must contain the exact manuscript matrix "
            "in registered order"
        )
    if len({task.task_id for task in resolved}) != len(resolved):
        raise ValueError("D4RL-9 task matrix contains duplicates")
    return resolved


def _update_digest(handle: BinaryIO, digest: "hashlib._Hash") -> None:
    for block in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(block)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        _update_digest(handle, digest)
    return digest.hexdigest()


def validate_dataset_path(
    path: str | Path,
    task: D4RLTaskSpec,
    *,
    require_verified_sha: bool,
) -> dict[str, object]:
    """Validate one dataset without treating an unresolved SHA as a wildcard."""

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(
            f"D4RL dataset does not exist: {resolved}"
        )
    if resolved.name != task.dataset_basename:
        raise ValueError(
            "D4RL dataset basename mismatch: "
            f"expected {task.dataset_basename}, got {resolved.name}"
        )
    digest = _sha256_file(resolved)
    if task.dataset_sha256 is None:
        if require_verified_sha:
            raise RuntimeError(
                f"dataset SHA-256 is unresolved for {task.task_id}; "
                "formal execution is blocked"
            )
        identity_verified = False
    else:
        if digest != task.dataset_sha256:
            raise ValueError(
                "D4RL dataset SHA-256 mismatch: "
                f"expected {task.dataset_sha256}, got {digest}"
            )
        identity_verified = True
    return {
        "task_id": task.task_id,
        "path": str(resolved),
        "basename": resolved.name,
        "sha256": digest,
        "registered_sha256": task.dataset_sha256,
        "identity_verified": identity_verified,
        "size_bytes": resolved.stat().st_size,
    }
