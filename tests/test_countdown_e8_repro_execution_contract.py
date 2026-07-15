from __future__ import annotations

from argparse import Namespace

import pytest

from drpo.countdown_e8_repro_contract import validate_worker_cell


def test_worker_contract_accepts_only_registered_triples() -> None:
    validate_worker_cell(Namespace(alpha=0.5, c=1.0, seed_offset=3000))
    validate_worker_cell(Namespace(alpha=1.0, c=8.0, seed_offset=16000))

    with pytest.raises(ValueError, match="outside the frozen six-cell"):
        validate_worker_cell(Namespace(alpha=1.0, c=8.0, seed_offset=3000))

    with pytest.raises(ValueError, match="outside the frozen six-cell"):
        validate_worker_cell(Namespace(alpha=0.5, c=1.0, seed_offset=5000))
