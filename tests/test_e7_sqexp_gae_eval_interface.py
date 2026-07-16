from __future__ import annotations

import json
from pathlib import Path

import pytest

from drpo import e7_sqexp_gae_contract as protocol
from drpo import e7_sqexp_gae_trainer as trainer


def _source_run_spec(eval_tokens: list[str]) -> dict[str, object]:
    trainer_argv = [
        "--dataset",
        "{dataset_id}",
        "--hdf5",
        "{dataset_path}",
        "--variant",
        "{variant}",
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
        *eval_tokens,
        "--seed",
        "200",
        "--device",
        "cpu",
        "--out_dir",
        "{output_dir}",
    ]
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
        "trainer_argv_template": trainer_argv,
        "injected_template_values": {"variant": "iqlv_exp_rank"},
        "passthrough_variants": [],
    }


def _write(tmp_path: Path, eval_tokens: list[str]) -> Path:
    path = tmp_path / "source_run_spec.json"
    path.write_text(json.dumps(_source_run_spec(eval_tokens)))
    return path


def _materialize_trainer_args(template: list[str]) -> list[str]:
    replacements = {
        "{dataset_id}": "hopper-medium-expert-v2",
        "{dataset_path}": "/tmp/hopper-medium-expert-v2.hdf5",
        "{seed}": "200",
        "{steps}": "1000000",
        "{output_dir}": "/tmp/gae-output",
    }
    return [replacements.get(token, token) for token in template]


def test_canonical_eval_max_steps_is_validated_then_removed(tmp_path: Path) -> None:
    resolved, _ = protocol.load_run_spec(
        _write(tmp_path, ["--eval_max_steps", "1000"])
    )
    template = resolved["trainer_argv_template"]

    assert "--eval_max_steps" not in template
    parsed = trainer.build_parser().parse_args(
        [
            *_materialize_trainer_args(template),
            "--advantage-manifest",
            "/tmp/ADVANTAGE_MANIFEST.json",
            "--advantage-estimator",
            "td",
        ]
    )
    assert parsed.eval_episodes == 10
    assert parsed.steps == 1_000_000


@pytest.mark.parametrize(
    "eval_tokens",
    [
        ["--eval_max_steps", "999"],
        ["--eval_max_steps", "1000", "--eval_max_steps", "1000"],
        ["--eval_max_steps"],
    ],
)
def test_eval_max_steps_drift_or_malformed_input_fails_closed(
    tmp_path: Path,
    eval_tokens: list[str],
) -> None:
    with pytest.raises(ValueError, match="eval_max_steps"):
        protocol.load_run_spec(_write(tmp_path, eval_tokens))


def test_unknown_source_trainer_flag_fails_before_branch_launch(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsupported GAE flags.*canonical_only"):
        protocol.load_run_spec(
            _write(
                tmp_path,
                [
                    "--eval_max_steps",
                    "1000",
                    "--canonical_only",
                    "1",
                ],
            )
        )
