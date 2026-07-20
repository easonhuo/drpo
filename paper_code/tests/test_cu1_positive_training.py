from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import torch

from drpo import drpo_cu1_e1_e4_oneclick as legacy
from drpo_reference.continuous.cu1 import CU1Protocol
from drpo_reference.continuous.cu1_training import CU1PositiveProtocol, train_positive


def small_reference_protocol() -> CU1Protocol:
    return CU1Protocol(
        n_train_states=48,
        n_test_states=40,
        hidden_dim=12,
    )


def small_training_protocol() -> CU1PositiveProtocol:
    return CU1PositiveProtocol(
        positive_batch_states=16,
        positive_steps=4,
        positive_continuation_steps=3,
        lbfgs_max_iter=2,
        positive_polish_min_steps=1,
        positive_polish_max_steps=3,
        positive_polish_check_every=1,
        eval_every=2,
        probe_states=8,
        absolute_residual_threshold_alpha_zero=0.0,
    )


def small_legacy_protocol() -> legacy.Protocol:
    return replace(
        legacy.Protocol(),
        n_train_states=48,
        n_test_states=40,
        hidden_dim=12,
        positive_batch_states=16,
        positive_steps=4,
        positive_continuation_steps=3,
        lbfgs_max_iter=2,
        positive_polish_min_steps=1,
        positive_polish_max_steps=3,
        positive_polish_check_every=1,
        eval_every=2,
        probe_states=8,
        absolute_residual_threshold_alpha_zero=0.0,
    )


def load_state(path: Path) -> dict[str, torch.Tensor]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def test_full_small_positive_training_matches_legacy(
    monkeypatch,
    tmp_path: Path,
) -> None:
    protocol = small_reference_protocol()
    monkeypatch.setattr(legacy, "P", small_legacy_protocol())
    monkeypatch.setattr(legacy, "ROOT", tmp_path / "legacy")
    monkeypatch.setattr(legacy, "DEVICE", torch.device("cpu"))
    monkeypatch.setattr(legacy, "DTYPE", torch.float32)

    old_actor, old_environment, old_trajectory, old_summary = legacy.train_positive(19)
    old_initialization = load_state(legacy.positive_initialization_checkpoint_path(19))
    new = train_positive(
        seed=19,
        protocol=protocol,
        training=small_training_protocol(),
    )

    for split_name in ("train", "test"):
        old_split = getattr(old_environment, split_name)
        new_split = getattr(new.environment, split_name)
        for field_name in old_split.__dataclass_fields__:
            torch.testing.assert_close(
                getattr(new_split, field_name),
                getattr(old_split, field_name),
                rtol=0.0,
                atol=0.0,
            )

    assert new.initialization_state.keys() == old_initialization.keys()
    for key, value in old_initialization.items():
        torch.testing.assert_close(
            new.initialization_state[key],
            value,
            rtol=0.0,
            atol=0.0,
        )
    for key, value in old_actor.state_dict().items():
        torch.testing.assert_close(
            new.actor.state_dict()[key],
            value,
            rtol=0.0,
            atol=0.0,
        )
    assert new.trajectory == old_trajectory
    assert new.summary == old_summary
