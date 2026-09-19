from __future__ import annotations

import importlib
import weakref

import torch


def test_countdown_public_facade_releases_registered_tensor_graph(monkeypatch) -> None:
    runtime = importlib.import_module("drpo_reference.experiments.countdown")
    cache_calls: list[bool] = []

    class QuantizedLikeModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layer = torch.nn.Linear(2, 2)

        def to(self, *_args, **_kwargs):
            raise RuntimeError("quantized wrapper cannot move devices")

    model = QuantizedLikeModel()
    parameter_reference = weakref.ref(next(model.parameters()))
    monkeypatch.setattr(runtime.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(runtime.torch.cuda, "empty_cache", lambda: cache_calls.append(True))

    runtime._release_model(model)

    assert parameter_reference() is None
    assert not model._modules
    assert cache_calls == [True]


def test_countdown_release_guard_offloads_regular_models(monkeypatch) -> None:
    runtime = importlib.import_module("drpo_reference.experiments.countdown")
    moved: list[str] = []

    class FakeModel:
        def to(self, device: str):
            moved.append(device)
            return self

    monkeypatch.setattr(runtime.torch.cuda, "is_available", lambda: False)
    runtime._release_model(FakeModel())

    assert moved == ["cpu"]
