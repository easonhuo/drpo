from __future__ import annotations

import json
from pathlib import Path

from drpo import countdown_e8_oracle_offline_v2_taper_resource_probe as probe


def base_config() -> dict:
    return {
        "model": {"max_length": 256, "max_new_tokens": 80},
        "offline_training": {
            "micro_batch": 1,
            "gradient_accumulation": 8,
        },
        "evaluation": {"batch_size": 8, "pass_ks": [8, 64]},
    }


def test_evaluation_envelope_preserves_registered_inner_shape() -> None:
    rows = [{"id": index} for index in range(20)]
    selected = probe.evaluation_envelope_rows(rows, base_config())
    assert selected == rows[:8]
    assert len(selected) == 8
    assert probe.maximum_pass_k(base_config()) == 64


def test_phase_state_records_worker_reported_peak(tmp_path: Path) -> None:
    path = tmp_path / "PROBE_PHASES.json"
    path.write_text(
        json.dumps(
            {
                "maximum_reported_worker_vram_bytes": 17_000_000_000,
            }
        ),
        encoding="utf-8",
    )
    assert probe.reported_peak_from_state(path) == 17_000_000_000


def test_resource_probe_contract_covers_train_and_evaluation() -> None:
    assert probe.VERSION == "0.2.0-phase-envelope"
    assert probe.DEFAULT_REQUIRED_PHASES == (
        "model_loaded",
        "training_peak_completed",
        "evaluation_peak_completed",
        "probe_complete",
    )
