from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from drpo.countdown_e8_rng_isolation import preserve_global_rng_state


def _draw() -> tuple[float, float, torch.Tensor]:
    return random.random(), float(np.random.rand()), torch.rand(4)


def test_rng_isolation_restores_python_numpy_and_torch_cpu() -> None:
    random.seed(11)
    np.random.seed(12)
    torch.manual_seed(13)
    initial_python = random.getstate()
    initial_numpy = np.random.get_state()
    initial_torch = torch.random.get_rng_state().clone()

    expected = _draw()
    random.setstate(initial_python)
    np.random.set_state(initial_numpy)
    torch.random.set_rng_state(initial_torch)

    with preserve_global_rng_state():
        random.seed(101)
        np.random.seed(102)
        torch.manual_seed(103)
        _draw()

    actual = _draw()
    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])


def test_rng_isolation_restores_after_exception() -> None:
    torch.manual_seed(77)
    state = torch.random.get_rng_state().clone()
    with pytest.raises(RuntimeError, match="boom"):
        with preserve_global_rng_state():
            torch.manual_seed(99)
            raise RuntimeError("boom")
    assert torch.equal(torch.random.get_rng_state(), state)


def test_frozen_audit_grid_and_protocols() -> None:
    from drpo import countdown_e8_repro_rng_audit_common as audit

    repo = Path(__file__).resolve().parents[1]
    config = yaml.safe_load(
        (repo / audit.DEFAULT_GRID_CONFIG).read_text(encoding="utf-8")
    )
    audit.validate_grid_config(config)
    cells = audit.build_cells(config)
    assert len(cells) == 6
    assert [(cell.alpha, cell.c, cell.seed_offset) for cell in cells] == list(
        audit.AUDIT_CELL_SPECS
    )
    assert config["sweep"]["protocols"] == list(audit.PROTOCOLS)
    assert config["sweep"]["total_cells"] == 12


def test_grid_rejects_cell_or_runtime_drift() -> None:
    from drpo import countdown_e8_repro_rng_audit_common as audit

    repo = Path(__file__).resolve().parents[1]
    config = yaml.safe_load(
        (repo / audit.DEFAULT_GRID_CONFIG).read_text(encoding="utf-8")
    )
    config["sweep"]["audit_cells"][0]["seed_offset"] = 3001
    with pytest.raises(ValueError, match="Audit cells changed"):
        audit.validate_grid_config(config)

    config = yaml.safe_load(
        (repo / audit.DEFAULT_GRID_CONFIG).read_text(encoding="utf-8")
    )
    config["execution"]["parallel_cells_per_gpu"] = 2
    with pytest.raises(ValueError, match="one cell per GPU"):
        audit.validate_grid_config(config)
