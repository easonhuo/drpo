"""Prepared-artifact verification for the E7 trajectory-GAE actor adapter."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from drpo.e7_canonical_injection import sha256_file
from drpo.e7_trajectory_gae import compute_gae_from_td

_ALLOWED_ESTIMATORS = {"td", "gae"}


def state_digest(state: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def load_prepared(
    branch: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any], str, dict[str, Any]]:
    manifest_path = Path(branch["advantage_manifest"]).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text())
    estimator = str(branch["template_values"]["advantage_estimator"])
    if estimator not in _ALLOWED_ESTIMATORS or manifest.get("status") != "complete":
        raise RuntimeError("unsupported estimator or incomplete prepared manifest")
    identity = (
        manifest.get("dataset_id"),
        int(manifest.get("seed", -1)),
        manifest.get("dataset_sha256"),
    )
    expected = branch["dataset_id"], int(branch["seed"]), branch["dataset_sha256"]
    if identity != expected:
        raise RuntimeError(f"prepared identity mismatch: {identity} != {expected}")
    arrays = Path(manifest["advantages"]["path"]).expanduser().resolve()
    critic = Path(manifest["critic"]["path"]).expanduser().resolve()
    if sha256_file(arrays) != manifest["advantages"]["sha256"]:
        raise RuntimeError("prepared advantage hash mismatch")
    if sha256_file(critic) != manifest["critic"]["sha256"]:
        raise RuntimeError("prepared critic hash mismatch")
    with np.load(arrays, allow_pickle=False) as payload:
        td = payload["td"].astype(np.float32, copy=True)
        stored_gae = payload["gae"].astype(np.float32, copy=True)
    dataset = Path(branch.get("dataset_path", manifest["dataset_path"])).expanduser().resolve()
    if sha256_file(dataset) != branch["dataset_sha256"]:
        raise RuntimeError("ordered dataset hash mismatch")
    with h5py.File(dataset, "r") as source:
        terminals = source["terminals"][:].astype(np.bool_)
        timeouts = source["timeouts"][:].astype(np.bool_)
    gae = compute_gae_from_td(
        td,
        terminals,
        timeouts,
        gamma=float(manifest.get("gamma", 0.99)),
        gae_lambda=float(manifest.get("gae_lambda", 0.95)),
    )
    if not np.allclose(gae, stored_gae, atol=1e-6, rtol=1e-6):
        raise RuntimeError("current GAE implementation disagrees with prepared artifact")
    advantage = td if estimator == "td" else gae
    if advantage.ndim != 1 or not np.isfinite(advantage).all():
        raise RuntimeError("prepared advantage must be one finite vector")
    try:
        checkpoint = torch.load(critic, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(critic, map_location="cpu")
    provenance = {
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "gae_recomputed_from_td_and_boundaries": True,
        "gae_matches_prepared_artifact": True,
    }
    return advantage, checkpoint["state_dict"], estimator, provenance
