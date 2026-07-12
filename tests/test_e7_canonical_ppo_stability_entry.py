from __future__ import annotations

import pytest

from drpo import e7_canonical_ppo_stability_entry as entry


def test_source_loader_adds_only_missing_openblas_thread_setting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = {
        "environment": {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
        "trainer_argv_template": ["--steps", "1000000"],
    }
    monkeypatch.setattr(
        entry,
        "_ORIGINAL_SOURCE_LOADER",
        lambda path: (source, "a" * 64),
    )
    loaded, digest = entry._load_source_run_spec("ignored.json")
    assert digest == "a" * 64
    assert loaded["environment"] == {
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    }
    assert source["environment"] == {
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }


def test_source_loader_rejects_non_unit_openblas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        entry,
        "_ORIGINAL_SOURCE_LOADER",
        lambda path: (
            {"environment": {"OPENBLAS_NUM_THREADS": "2"}},
            "b" * 64,
        ),
    )
    with pytest.raises(ValueError, match="OPENBLAS_NUM_THREADS"):
        entry._load_source_run_spec("ignored.json")
