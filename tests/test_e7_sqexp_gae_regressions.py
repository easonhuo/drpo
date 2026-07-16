from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from drpo import e7_sqexp_gae_contract as protocol
from drpo.e7_offline_gae import compute_gae_numpy, compute_gae_torch
from drpo.e7_sqexp_gae_prepare import validate_gae_precision


def _source_run_spec(
    *,
    variant_token: str = "{variant}",
    injected_variant: str | None = "iqlv_exp_rank",
) -> dict[str, object]:
    injected: dict[str, str] = {}
    if injected_variant is not None:
        injected["variant"] = injected_variant
    return {
        "experiment_id": "EXT-H-E7-BENCH-01",
        "run_kind": "pilot",
        "datasets": [
            {
                "id": dataset,
                "path": f"/tmp/{dataset}.hdf5",
                "sha256": "0" * 64,
            }
            for dataset in protocol.EXPECTED_SOURCE_DATASETS
        ],
        "seeds": list(protocol.EXPECTED_SOURCE_SEEDS),
        "environment": {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        },
        "trainer_argv_template": [
            "--dataset",
            "{dataset_id}",
            "--hdf5",
            "{dataset_path}",
            "--variant",
            variant_token,
            "--alpha",
            "0.11",
            "--tau",
            "0.5",
            "--temp",
            "5.0",
            "--steps",
            "1000000",
            "--batch",
            "256",
            "--lr",
            "0.0003",
            "--eval_interval",
            "50000",
            "--eval_episodes",
            "10",
            "--seed",
            "200",
            "--device",
            "cpu",
            "--out_dir",
            "{output_dir}",
        ],
        "injected_template_values": injected,
        "passthrough_variants": [],
    }


def _write_run_spec(tmp_path: Path, payload: dict[str, object]) -> Path:
    path = tmp_path / "source_run_spec.json"
    path.write_text(json.dumps(payload))
    return path


def test_run_spec_resolves_canonical_variant_placeholder(tmp_path: Path) -> None:
    path = _write_run_spec(tmp_path, _source_run_spec())

    resolved, _ = protocol.load_run_spec(path)

    argv = resolved["trainer_argv_template"]
    variant_index = argv.index("--variant") + 1
    assert argv[variant_index] == "iqlv_exp_rank"
    assert resolved["injected_template_values"] == {}
    assert tuple(resolved["seeds"]) == protocol.EXPECTED_SEEDS
    assert tuple(item["id"] for item in resolved["datasets"]) == (
        protocol.EXPECTED_DATASETS
    )


def test_run_spec_accepts_matching_literal_variant(tmp_path: Path) -> None:
    path = _write_run_spec(
        tmp_path,
        _source_run_spec(variant_token="iqlv_exp_rank"),
    )

    resolved, _ = protocol.load_run_spec(path)

    argv = resolved["trainer_argv_template"]
    assert argv[argv.index("--variant") + 1] == "iqlv_exp_rank"


@pytest.mark.parametrize(
    ("variant_token", "injected_variant"),
    [
        ("{variant}", None),
        ("{variant}", "sna2c"),
        ("sna2c", "iqlv_exp_rank"),
        ("iqlv_exp_rank", "sna2c"),
    ],
)
def test_run_spec_rejects_variant_drift(
    tmp_path: Path,
    variant_token: str,
    injected_variant: str | None,
) -> None:
    path = _write_run_spec(
        tmp_path,
        _source_run_spec(
            variant_token=variant_token,
            injected_variant=injected_variant,
        ),
    )

    with pytest.raises(ValueError, match="variant"):
        protocol.load_run_spec(path)


def test_long_gae_crosscheck_separates_float64_parity_from_storage_rounding() -> None:
    transition_count = 4096
    rewards = np.linspace(10.0, 100.0, transition_count, dtype=np.float32)
    values = np.linspace(-5.0, 5.0, transition_count, dtype=np.float32)
    next_values = np.linspace(-4.5, 5.5, transition_count, dtype=np.float32)
    terminals = np.zeros(transition_count, dtype=np.bool_)
    timeouts = np.zeros(transition_count, dtype=np.bool_)
    timeouts[-1] = True

    stored_gae = compute_gae_numpy(
        rewards,
        values,
        next_values,
        terminals,
        timeouts,
        gamma=0.99,
        gae_lambda=0.95,
    )
    torch_float64 = compute_gae_torch(
        torch.from_numpy(rewards.astype(np.float64)),
        torch.from_numpy(values.astype(np.float64)),
        torch.from_numpy(next_values.astype(np.float64)),
        torch.from_numpy(terminals),
        torch.from_numpy(timeouts),
        gamma=0.99,
        gae_lambda=0.95,
    ).numpy()
    old_mixed_precision_error = float(
        np.max(np.abs(torch_float64 - stored_gae.astype(np.float64)))
    )
    assert old_mixed_precision_error > 1e-6

    diagnostics = validate_gae_precision(
        rewards,
        values,
        next_values,
        terminals,
        timeouts,
        stored_gae,
        gamma=0.99,
        gae_lambda=0.95,
    )

    assert diagnostics["numpy_torch_float64_max_abs_error"] <= 1e-6
    assert diagnostics["gae_float32_storage_quantization_max_abs_error"] > 1e-6
    assert diagnostics["stored_gae_vs_float64_cast_max_abs_error"] == 0.0
    assert diagnostics["stored_gae_dtype"] == "float32"
    assert diagnostics["stored_gae_matches_float64_reference_cast"] is True


def test_precision_gate_rejects_modified_stored_advantage() -> None:
    rewards = np.array([1.0, 2.0], dtype=np.float32)
    values = np.array([0.0, 0.0], dtype=np.float32)
    next_values = np.array([0.0, 0.0], dtype=np.float32)
    terminals = np.array([False, False])
    timeouts = np.array([False, True])
    stored = compute_gae_numpy(
        rewards,
        values,
        next_values,
        terminals,
        timeouts,
        gamma=0.99,
        gae_lambda=0.95,
    )
    stored[0] += np.float32(1e-3)

    with pytest.raises(RuntimeError, match="stored float32 GAE"):
        validate_gae_precision(
            rewards,
            values,
            next_values,
            terminals,
            timeouts,
            stored,
            gamma=0.99,
            gae_lambda=0.95,
        )
