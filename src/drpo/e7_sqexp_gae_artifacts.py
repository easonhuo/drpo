"""Artifact hashing and atomic serialization helpers."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer): return int(value)
    if isinstance(value, np.floating): return float(value)
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, Path): return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")

def atomic_json(path: str | Path, payload: Any) -> None:
    destination = Path(path); destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n")
    os.replace(temporary, destination)

def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()

def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""): digest.update(block)
    return digest.hexdigest()

def canonical_hash(payload: Any) -> str:
    return sha256_bytes(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=_json_default).encode())

def array_sha256(array: np.ndarray) -> str:
    value=np.ascontiguousarray(array); digest=hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii")); digest.update(b"\0")
    digest.update(json.dumps(list(value.shape)).encode("ascii")); digest.update(b"\0")
    digest.update(value.tobytes(order="C")); return digest.hexdigest()

def write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    destination=Path(path); destination.parent.mkdir(parents=True, exist_ok=True)
    if not rows: destination.write_text(""); return
    fields=[]
    for row in rows:
        for key in row:
            if key not in fields: fields.append(key)
    with destination.open("w", newline="") as handle:
        writer=csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
