from __future__ import annotations

import pytest

from drpo_reference.common import EventFlags, EventKind, EventRecord


def test_event_classes_remain_independently_reportable() -> None:
    flags = EventFlags(
        task_performance_collapse=True,
        support_or_probability_boundary=True,
        nan_inf_numerical_failure=True,
    )
    assert flags.active_kinds() == (
        EventKind.TASK_PERFORMANCE_COLLAPSE,
        EventKind.SUPPORT_OR_PROBABILITY_BOUNDARY,
        EventKind.NAN_INF_NUMERICAL_FAILURE,
    )
    assert flags.as_dict() == {
        "task_performance_collapse": True,
        "support_or_probability_boundary": True,
        "nan_inf_numerical_failure": True,
        "environment_invalid": False,
    }


def test_event_record_rejects_negative_step() -> None:
    with pytest.raises(ValueError):
        EventRecord(step=-1, flags=EventFlags())
