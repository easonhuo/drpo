"""Non-overwriting event taxonomy for paper-facing terminal audits."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


class EventKind(str, Enum):
    """Event classes that must remain separately reportable."""

    TASK_PERFORMANCE_COLLAPSE = "task_performance_collapse"
    SUPPORT_OR_PROBABILITY_BOUNDARY = "support_or_probability_boundary"
    NAN_INF_NUMERICAL_FAILURE = "nan_inf_numerical_failure"
    ENVIRONMENT_INVALID = "environment_invalid"


@dataclass(frozen=True)
class EventFlags:
    """Boolean terminal flags without precedence-based collapsing.

    Several flags may be true at the same step. Callers must preserve every
    active class instead of replacing task or boundary evidence with a generic
    failure label.
    """

    task_performance_collapse: bool = False
    support_or_probability_boundary: bool = False
    nan_inf_numerical_failure: bool = False
    environment_invalid: bool = False

    def active_kinds(self) -> tuple[EventKind, ...]:
        active: list[EventKind] = []
        if self.task_performance_collapse:
            active.append(EventKind.TASK_PERFORMANCE_COLLAPSE)
        if self.support_or_probability_boundary:
            active.append(EventKind.SUPPORT_OR_PROBABILITY_BOUNDARY)
        if self.nan_inf_numerical_failure:
            active.append(EventKind.NAN_INF_NUMERICAL_FAILURE)
        if self.environment_invalid:
            active.append(EventKind.ENVIRONMENT_INVALID)
        return tuple(active)

    def as_dict(self) -> dict[str, bool]:
        return {
            EventKind.TASK_PERFORMANCE_COLLAPSE.value: self.task_performance_collapse,
            EventKind.SUPPORT_OR_PROBABILITY_BOUNDARY.value: self.support_or_probability_boundary,
            EventKind.NAN_INF_NUMERICAL_FAILURE.value: self.nan_inf_numerical_failure,
            EventKind.ENVIRONMENT_INVALID.value: self.environment_invalid,
        }


@dataclass(frozen=True)
class EventRecord:
    """One auditable event observation."""

    step: int
    flags: EventFlags
    metrics: Mapping[str, float] = field(default_factory=dict)
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.step < 0:
            raise ValueError("event step must be non-negative")
