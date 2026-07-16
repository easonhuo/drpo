from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from drpo.e7_offline_gae import sha256_file
from drpo.e7_sqexp_gae import _verify_prepared


DATASET_ID = "hopper-medium-expert-v2"
DATASET_SHA256 = "1" * 64
SEED = 200


def _write_prepared_artifact(
    root: Path,
    *,
    preparer_version: str = "1.0.1-e7-sqexp-gae",
    gae_dtype: np.dtype[np.floating] = np.dtype(np.float32),
    stored_matches: bool = True,
) -> Path:
    root.mkdir(parents=True)
    critic_path = root / "critic_final.pt"
    critic_path.write_bytes(b"critic-checkpoint-placeholder")
    advantage_path = root / "advantages.npz"
    np.savez_compressed(
        advantage_path,
        td=np.array([1.0, -1.0], dtype=np.float32),
        gae=np.array([1.5, -0.5], dtype=gae_dtype),
        values=np.array([0.1, 0.2], dtype=np.float32),
        next_values=np.array([0.2, 0.3], dtype=np.float32),
    )
    diagnostics = {
        "lambda_zero_max_abs_error": 0.0,
        "numpy_torch_max_abs_error": 0.0,
        "numpy_torch_float64_max_abs_error": 0.0,
        "gae_float32_storage_quantization_max_abs_error": 3.5e-6,
        "stored_gae_vs_float64_cast_max_abs_error": 0.0,
        "stored_gae_dtype": "float32",
        "stored_gae_matches_float64_reference_cast": stored_matches,
    }
    manifest = {
        "status": "complete",
        "preparer_version": preparer_version,
        "dataset_id": DATASET_ID,
        "dataset_sha256": DATASET_SHA256,
        "seed": SEED,
        "critic": {
            "path": str(critic_path.resolve()),
            "sha256": sha256_file(critic_path),
        },
        "advantages": {
            "path": str(advantage_path.resolve()),
            "sha256": sha256_file(advantage_path),
            "diagnostics": diagnostics,
        },
        "trajectory_audit": {"status": "PASS"},
    }
    (root / "ADVANTAGE_MANIFEST.json").write_text(json.dumps(manifest))
    return root


def test_corrected_prepared_artifact_passes_gate(tmp_path: Path) -> None:
    artifact = _write_prepared_artifact(tmp_path / "prepared")

    manifest = _verify_prepared(
        artifact,
        dataset_id=DATASET_ID,
        dataset_sha256=DATASET_SHA256,
        seed=SEED,
    )

    assert manifest["preparer_version"] == "1.0.1-e7-sqexp-gae"


def test_gate_rejects_artifact_from_buggy_preparer(tmp_path: Path) -> None:
    artifact = _write_prepared_artifact(
        tmp_path / "prepared",
        preparer_version="1.0.0-e7-sqexp-gae",
    )

    with pytest.raises(RuntimeError, match="predates the corrected precision gate"):
        _verify_prepared(
            artifact,
            dataset_id=DATASET_ID,
            dataset_sha256=DATASET_SHA256,
            seed=SEED,
        )


def test_gate_rejects_non_float32_actor_advantage(tmp_path: Path) -> None:
    artifact = _write_prepared_artifact(
        tmp_path / "prepared",
        gae_dtype=np.dtype(np.float64),
    )

    with pytest.raises(RuntimeError, match="must remain float32"):
        _verify_prepared(
            artifact,
            dataset_id=DATASET_ID,
            dataset_sha256=DATASET_SHA256,
            seed=SEED,
        )


def test_gate_rejects_failed_reference_cast_check(tmp_path: Path) -> None:
    artifact = _write_prepared_artifact(
        tmp_path / "prepared",
        stored_matches=False,
    )

    with pytest.raises(RuntimeError, match="reference-cast verification failed"):
        _verify_prepared(
            artifact,
            dataset_id=DATASET_ID,
            dataset_sha256=DATASET_SHA256,
            seed=SEED,
        )
