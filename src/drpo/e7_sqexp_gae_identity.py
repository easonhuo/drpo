"""Frozen critic and prepared-advantage identity binding."""
from __future__ import annotations

from drpo.e7_sqexp_gae_artifacts import canonical_hash
from drpo.e7_sqexp_gae_protocol import (
    ESTIMATORS, EXPERIMENT_ID, RUNNER_VERSION, DatasetSpec, FrozenProtocol,
)

def critic_identity(
    *,
    dataset: DatasetSpec,
    seed: int,
    protocol: FrozenProtocol,
    source_run_spec_sha256: str,
) -> str:
    return canonical_hash(
        {
            "experiment_id": EXPERIMENT_ID,
            "artifact": "shared_frozen_critic",
            "runner_version": RUNNER_VERSION,
            "dataset_id": dataset.id,
            "dataset_sha256": dataset.sha256,
            "source_run_spec_sha256": source_run_spec_sha256,
            "seed": int(seed),
            "gamma": protocol.gamma,
            "expectile_tau": protocol.expectile_tau,
            "critic_steps": protocol.critic_steps,
            "batch_size": protocol.batch_size,
            "learning_rate": protocol.learning_rate,
            "network": "separate_relu_2x256_orthogonal_global_diagonal_actor",
        }
    )


def prepared_advantage_identity(
    *,
    critic_identity_sha256: str,
    critic_checkpoint_sha256: str,
    estimator: str,
    protocol: FrozenProtocol,
    arrays_sha256: str,
) -> str:
    if estimator not in ESTIMATORS:
        raise ValueError(f"unknown estimator: {estimator}")
    return canonical_hash(
        {
            "experiment_id": EXPERIMENT_ID,
            "artifact": "prepared_advantage",
            "critic_identity_sha256": critic_identity_sha256,
            "critic_checkpoint_sha256": critic_checkpoint_sha256,
            "estimator": estimator,
            "gamma": protocol.gamma,
            "gae_lambda": 0.0 if estimator == "one_step_td" else protocol.gae_lambda,
            "normalization": "none",
            "clipping": "none",
            "actor_dtype": "float32",
            "arrays_sha256": arrays_sha256,
        }
    )


