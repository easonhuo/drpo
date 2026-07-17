"""Shared records and deterministic utilities used by paper experiments."""

from .aggregate import aggregate_rows, audit_run_matrix, bootstrap_mean_ci
from .events import EventFlags, EventKind, EventRecord
from .io import atomic_json, read_csv, write_csv
from .seeding import cpu_generator, seed_all

__all__ = [
    "EventFlags",
    "EventKind",
    "EventRecord",
    "aggregate_rows",
    "atomic_json",
    "audit_run_matrix",
    "bootstrap_mean_ci",
    "cpu_generator",
    "read_csv",
    "seed_all",
    "write_csv",
]
